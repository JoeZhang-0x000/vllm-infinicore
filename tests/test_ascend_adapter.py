from __future__ import annotations

import ctypes
from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from vllm_infinicore.ops import ascend_backend as backend
from vllm_infinicore import platform_support
from vllm_infinicore.ops import infinicore_backend as counters


class AscendAdapterTests(unittest.TestCase):
    def tearDown(self):
        backend.library.cache_clear()
        counters.reset_backend_call_counts()

    def test_library_revision_mismatch_rejected(self):
        lib = SimpleNamespace(
            vllmInfinicoreRevision=mock.Mock(return_value=b"wrong"),
            vllmInfinicoreBridgeABI=mock.Mock(return_value=1),
        )
        with (
            mock.patch.dict(os.environ, {backend.LIBRARY_ENV: "/fake.so"}),
            mock.patch.object(ctypes, "CDLL", return_value=lib),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match lock"):
                backend.library()

    def test_library_abi_mismatch_rejected(self):
        revision = json.loads(backend._LOCK.read_text())["revision"]
        lib = SimpleNamespace(
            vllmInfinicoreRevision=mock.Mock(return_value=revision.encode()),
            vllmInfinicoreBridgeABI=mock.Mock(return_value=99),
        )
        with (
            mock.patch.dict(os.environ, {backend.LIBRARY_ENV: "/fake.so"}),
            mock.patch.object(ctypes, "CDLL", return_value=lib),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match lock"):
                backend.library()

    def test_known_create_status_can_fallback_but_launch_status_cannot(self):
        with self.assertRaises(backend.Unsupported):
            backend._check(2, "create", creating=True)
        with self.assertRaises(RuntimeError) as ctx:
            backend._check(2, "launch")
        self.assertNotIsInstance(ctx.exception, backend.Unsupported)
        with self.assertRaises(RuntimeError) as ctx:
            backend._check(1, "create", creating=True)
        self.assertNotIsInstance(ctx.exception, backend.Unsupported)

    def test_disabling_backend_counts_fallback_without_launch(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
        operation = mock.Mock(side_effect=AssertionError("must not launch"))
        with mock.patch.dict(os.environ, {"VLLM_INFINICORE_DISABLE_REAL_BACKEND": "1"}):
            self.assertEqual(backend.execute("linear", tensor, operation, lambda: 7), 7)
        self.assertEqual(counters.backend_fallback_counts(), {"linear": 1})
        self.assertEqual(counters.backend_call_counts(), {})

    def test_cpu_preserves_native_path(self):
        operation = mock.Mock(side_effect=AssertionError("must not launch"))
        self.assertEqual(
            backend.execute("linear", torch.ones(1), operation, lambda: 4), 4
        )
        self.assertEqual(counters.backend_call_counts(), {})

    def test_attention_stays_native_when_operator_library_configured(self):
        with mock.patch.dict(os.environ, {backend.LIBRARY_ENV: "/fake.so"}):
            self.assertEqual(
                set(platform_support.ascend_native_fallback_reasons()),
                {"StoreKVCache", "PagedAttentionPrefill", "PagedAttentionDecode"},
            )

    def test_route_installation_preserves_class_and_restores_original(self):
        from vllm_infinicore.ops import ascend_routes

        class Native:
            def forward_oot(self, x):
                return x + 1

        original = Native.forward_oot
        module = SimpleNamespace(AscendSiluAndMul=Native)
        with (
            mock.patch.object(backend, "library"),
            mock.patch.object(
                ascend_routes.importlib, "import_module", return_value=module
            ),
        ):
            self.assertTrue(ascend_routes.install("SiluAndMul").installed)
            self.assertIs(module.AscendSiluAndMul, Native)
            self.assertEqual(Native().forward_oot(torch.tensor(2)).item(), 3)
            self.assertTrue(ascend_routes.install("SiluAndMul").installed)
            self.assertTrue(ascend_routes.uninstall("SiluAndMul").uninstalled)
            self.assertIs(Native.forward_oot, original)

    def test_uninstall_does_not_overwrite_later_patch(self):
        from vllm_infinicore.ops import ascend_routes

        class Native:
            def forward_oot(self, x):
                return x

        module = SimpleNamespace(AscendSiluAndMul=Native)
        with (
            mock.patch.object(backend, "library"),
            mock.patch.object(
                ascend_routes.importlib, "import_module", return_value=module
            ),
        ):
            ascend_routes.install("SiluAndMul")
            wrapper = Native.forward_oot
            later = lambda self, x: x
            Native.forward_oot = later
            self.assertFalse(ascend_routes.uninstall("SiluAndMul").uninstalled)
            self.assertIs(Native.forward_oot, later)
            Native.forward_oot = wrapper
            ascend_routes.uninstall("SiluAndMul")

    def test_runtime_launch_failure_is_not_retried(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
        api = SimpleNamespace(
            device=lambda _: nullcontext(), is_current_stream_capturing=lambda: False
        )
        native = mock.Mock()
        with (
            mock.patch.object(torch, "npu", api, create=True),
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_INFINICORE_DISABLE_REAL_BACKEND": "0",
                    "VLLM_INFINICORE_STRICT_BACKEND": "0",
                },
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                backend.execute(
                    "linear",
                    tensor,
                    mock.Mock(side_effect=RuntimeError("launch failed")),
                    native,
                )
        native.assert_not_called()
        self.assertEqual(counters.backend_call_counts(), {})

    def test_graph_capture_launches_infinicore(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
        api = SimpleNamespace(
            device=lambda _: nullcontext(), is_current_stream_capturing=lambda: True
        )
        operation = mock.Mock(return_value=5)
        with (
            mock.patch.object(torch, "npu", api, create=True),
            mock.patch.dict(
                os.environ,
                {"VLLM_INFINICORE_DISABLE_REAL_BACKEND": "0"},
                clear=False,
            ),
        ):
            os.environ.pop("VLLM_INFINICORE_ASCEND_GRAPH", None)
            self.assertEqual(backend.execute("linear", tensor, operation, lambda: 8), 5)
        operation.assert_called_once()
        self.assertEqual(counters.backend_call_counts()["linear"], 1)
        self.assertNotIn("linear", counters.backend_fallback_reasons())

    def test_graph_capture_falls_back_when_disabled(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
        api = SimpleNamespace(
            device=lambda _: nullcontext(), is_current_stream_capturing=lambda: True
        )
        operation = mock.Mock()
        with (
            mock.patch.object(torch, "npu", api, create=True),
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_INFINICORE_DISABLE_REAL_BACKEND": "0",
                    "VLLM_INFINICORE_ASCEND_GRAPH": "0",
                },
            ),
        ):
            self.assertEqual(backend.execute("linear", tensor, operation, lambda: 8), 8)
        operation.assert_not_called()
        self.assertIn("capture", counters.backend_fallback_reasons()["linear"])

    def test_untraceable_execute_stays_native_without_python_side_effects(self):
        # Routes emit a registered operator while tracing; reaching execute()
        # under Dynamo means no traceable operator covers the call, so it must
        # keep the native program and leave no counter side effects behind.
        tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))

        def forbidden():
            raise AssertionError("untraceable launch must not enter compiled graph")

        def run(x):
            y = backend.execute("linear", tensor, forbidden, lambda: x * 2)
            return backend.fallback("fused_add_rms_norm", "unsupported", lambda: y + 1)

        compiled = torch.compile(run, backend="eager", fullgraph=True)
        for value in (1.0, 3.0):
            x = torch.full((2,), value)
            torch.testing.assert_close(compiled(x), x * 2 + 1)
        self.assertEqual(counters.backend_call_counts(), {})
        self.assertEqual(counters.backend_fallback_counts(), {})

    def test_fp32_gemm_falls_back_before_pointer_access(self):
        with self.assertRaisesRegex(backend.Unsupported, "reduced-precision"):
            backend.linear(torch.ones(2, 4), torch.ones(3, 4))

    def test_swiglu_tail_tiles_rejected(self):
        with mock.patch.object(backend, "nd", side_effect=lambda t: t):
            with self.assertRaisesRegex(backend.Unsupported, "aligned tiles"):
                backend.silu_and_mul(torch.ones(2, 130, dtype=torch.bfloat16))

    def test_wrong_or_dirty_source_rejected(self):
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts/build_ascend.py"
        spec = importlib.util.spec_from_file_location("build_ascend", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(module, "run", return_value="bad"):
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                module.verify_source(Path("/fake"), "expected")
        with mock.patch.object(module, "run", side_effect=["expected", " M src/op.cc"]):
            with self.assertRaisesRegex(RuntimeError, "tracked modifications"):
                module.verify_source(Path("/fake"), "expected")


if __name__ == "__main__":
    unittest.main()


class AscendGraphOperatorTests(unittest.TestCase):
    """The adapters are only in a compiled graph if a real operator carries them."""

    def tearDown(self):
        counters.reset_backend_call_counts()

    def test_operators_registered_with_shape_propagation(self):
        from vllm_infinicore.ops import ascend_graph_ops  # noqa: F401

        ops = torch.ops.vllm_infinicore_ascend
        with torch._subclasses.FakeTensorMode():
            x = torch.empty(4, 8, dtype=torch.bfloat16)
            weight = torch.empty(6, 8, dtype=torch.bfloat16)
            self.assertEqual(tuple(ops.linear(x, weight, None, "linear").shape), (4, 6))
            ids = torch.empty(3, dtype=torch.int64)
            table = torch.empty(10, 5, dtype=torch.bfloat16)
            self.assertEqual(tuple(ops.embedding(ids, table).shape), (3, 5))
            gate_up = torch.empty(2, 16, dtype=torch.bfloat16)
            self.assertEqual(tuple(ops.silu_and_mul(gate_up).shape), (2, 8))
            self.assertEqual(
                tuple(ops.rms_norm(x, torch.empty(8, dtype=torch.bfloat16), 1e-6).shape),
                (4, 8),
            )

    def test_capability_predicates_match_eager_rejections(self):
        supported, reason = backend.supports_linear(torch.ones(2, 4))
        self.assertFalse(supported)
        self.assertIn("reduced-precision", reason)
        supported, reason = backend.supports_silu_and_mul(
            torch.ones(2, 130, dtype=torch.bfloat16)
        )
        self.assertFalse(supported)
        self.assertIn("aligned tiles", reason)
        # A hidden size past the kernel's limit is the 27B case.
        supported, _ = backend.supports_silu_and_mul(
            torch.ones(2, 34816, dtype=torch.bfloat16)
        )
        self.assertFalse(supported)
        supported, _ = backend.supports_tensor(torch.ones(2, 2, dtype=torch.bfloat16))
        self.assertFalse(supported, "a CPU tensor has no InfiniCore Ascend route")

    def test_capability_predicates_answer_instead_of_raising(self):
        # Every check is evaluated before the call is known to be eligible, so
        # a predicate that raises on an odd tensor takes down an unrelated route.
        for tensor in (torch.tensor(2), torch.ones(0), torch.ones(3)):
            for predicate in (backend.supports_tensor, backend.supports_linear,
                              backend.supports_silu_and_mul):
                supported, reason = predicate(tensor)
                self.assertIsInstance(supported, bool)
                self.assertIsInstance(reason, str)

    def test_unsupported_call_selects_native_at_trace_time(self):
        from vllm_infinicore.ops import ascend_routes

        wrapper = ascend_routes._wrapper(
            "MatMul", lambda self, layer, x, bias=None: "native"
        )
        layer = SimpleNamespace(weight=torch.ones(6, 8, dtype=torch.bfloat16))
        with mock.patch.object(torch.compiler, "is_compiling", lambda: True):
            result = wrapper(SimpleNamespace(), layer, torch.ones(4, 8))
        self.assertEqual(result, "native")
        self.assertEqual(counters.backend_call_counts(), {})

    def test_captured_descriptors_are_never_evicted(self):
        from collections import OrderedDict

        closed = []

        def descriptor(pinned):
            return SimpleNamespace(pinned=pinned, close=lambda: closed.append(pinned))

        cache = OrderedDict(
            (index, descriptor(index < 2)) for index in range(4)
        )
        backend._evict(cache)
        self.assertEqual(closed, [False])
        self.assertEqual(list(cache), [0, 1, 3])

    def test_eviction_keeps_cache_when_every_descriptor_is_captured(self):
        from collections import OrderedDict

        cache = OrderedDict(
            (index, SimpleNamespace(pinned=True, close=lambda: None))
            for index in range(2)
        )
        backend._evict(cache)
        self.assertEqual(list(cache), [0, 1])

    def test_workspace_is_shared_and_sized_to_the_high_water_mark(self):
        backend.clear_cache()
        small = backend._Descriptor.__new__(backend._Descriptor)
        small._workspace, small.device = None, torch.device("cpu")
        small.workspace_size = ctypes.c_size_t(64)
        large = backend._Descriptor.__new__(backend._Descriptor)
        large._workspace, large.device = None, torch.device("cpu")
        large.workspace_size = ctypes.c_size_t(4096)
        first = small.workspace(False)
        grown = large.workspace(False)
        # The larger requirement replaces the shared buffer, and the smaller
        # descriptor then reuses it rather than holding one of its own.
        self.assertGreaterEqual(grown.numel(), 4096)
        self.assertIsNot(grown, first)
        self.assertIs(small.workspace(False), grown)
        self.assertIsNone(small._workspace)
        backend.clear_cache()

    def test_capture_never_reallocates_a_recorded_workspace(self):
        backend.clear_cache()

        def descriptor(size):
            desc = backend._Descriptor.__new__(backend._Descriptor)
            desc._workspace, desc.device = None, torch.device("cpu")
            desc.workspace_size = ctypes.c_size_t(size)
            return desc

        warm = descriptor(64)
        shared = warm.workspace(False)
        recorded = descriptor(64)
        # The capture records this pointer, which locks the buffer in place.
        self.assertIs(recorded.workspace(True), shared)
        bigger = descriptor(8192)
        private = bigger.workspace(False)
        self.assertIsNot(private, shared)
        self.assertIs(bigger._workspace, private)
        self.assertIs(warm.workspace(False), shared)
        backend.clear_cache()

    def test_zero_sized_workspace_still_allocates_a_pointer(self):
        backend.clear_cache()
        desc = backend._Descriptor.__new__(backend._Descriptor)
        desc._workspace, desc.device = None, torch.device("cpu")
        desc.workspace_size = ctypes.c_size_t(0)
        self.assertEqual(desc.workspace(False).numel(), 1)
        backend.clear_cache()

    def test_launch_does_not_record_stream_on_the_current_stream(self):
        # record_stream defers allocator block reuse until a stream event is
        # observed, which serializes the per-call output allocation against
        # device progress. It protects nothing for a same-stream launch.
        tensors = [
            SimpleNamespace(
                device=torch.device("npu:0"),
                shape=(2, 2),
                stride=lambda: (2, 1),
                dtype=torch.bfloat16,
                ndim=2,
                data_ptr=lambda: 1,
                record_stream=mock.Mock(side_effect=AssertionError("record_stream")),
            )
        ]
        stream = SimpleNamespace(npu_stream=7)
        api = SimpleNamespace(
            current_stream=lambda _: stream,
            is_current_stream_capturing=lambda: False,
        )
        descriptor = SimpleNamespace(
            ptr=1,
            workspace_size=SimpleNamespace(value=0),
            workspace=lambda capturing: SimpleNamespace(data_ptr=lambda: 2),
            pinned=False,
            close=lambda: None,
        )
        lib = SimpleNamespace(infiniopGemm=mock.Mock(return_value=0))
        with (
            mock.patch.object(torch, "npu", api, create=True),
            mock.patch.object(backend, "library", return_value=lib),
            mock.patch.object(backend, "_Descriptor", return_value=descriptor),
        ):
            backend._LOCAL.descriptors = __import__("collections").OrderedDict()
            try:
                backend.launch("Gemm", tensors)
            finally:
                backend._LOCAL.descriptors.clear()
        tensors[0].record_stream.assert_not_called()
