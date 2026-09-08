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

    def test_graph_capture_uses_native_before_launch(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
        api = SimpleNamespace(
            device=lambda _: nullcontext(), is_current_stream_capturing=lambda: True
        )
        operation = mock.Mock()
        with (
            mock.patch.object(torch, "npu", api, create=True),
            mock.patch.dict(os.environ, {"VLLM_INFINICORE_DISABLE_REAL_BACKEND": "0"}),
        ):
            self.assertEqual(backend.execute("linear", tensor, operation, lambda: 8), 8)
        operation.assert_not_called()
        self.assertIn("capture", counters.backend_fallback_reasons()["linear"])

    def test_fullgraph_compilation_preserves_native_without_python_side_effects(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))

        def forbidden():
            raise AssertionError("InfiniCore launch must not enter compiled graph")

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
