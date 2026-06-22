from __future__ import annotations

import importlib
import os
import sys
import unittest
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local macOS dev env may not ship torch.
    torch = None

if torch is not None:
    from vllm_infinicore.ops import cpp_bridge, infinicore_backend
else:
    cpp_bridge = None
    infinicore_backend = None


@unittest.skipIf(torch is None, "torch is not installed")
class InfiniCoreBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        infinicore_backend.clear_tensor_wrapper_cache()
        infinicore_backend.clear_stream_cache()
        infinicore_backend.reset_backend_call_counts()

    def tearDown(self) -> None:
        infinicore_backend.clear_tensor_wrapper_cache()
        infinicore_backend.clear_stream_cache()
        infinicore_backend.reset_backend_call_counts()

    def test_tensor_wrapper_cache_reuses_matching_views(self) -> None:
        base = torch.arange(32, dtype=torch.float32).view(4, 8)
        view = base[:, ::2]
        wrapped = SimpleNamespace()

        with mock.patch.object(
            infinicore_backend,
            "_as_infini_strided",
            return_value=wrapped,
        ) as create:
            first = infinicore_backend._as_infini_strided_cached(view)
            second = infinicore_backend._as_infini_strided_cached(view)

        self.assertIs(first, wrapped)
        self.assertIs(second, wrapped)
        create.assert_called_once_with(view)

    def test_tensor_wrapper_cache_distinguishes_shape_and_stride(self) -> None:
        base = torch.arange(32, dtype=torch.float32).view(4, 8)
        row_major = base[:, :4]
        strided = base[:, ::2]

        with mock.patch.object(
            infinicore_backend,
            "_as_infini_strided",
            side_effect=[SimpleNamespace(name="row-major"), SimpleNamespace(name="strided")],
        ) as create:
            first = infinicore_backend._as_infini_strided_cached(row_major)
            second = infinicore_backend._as_infini_strided_cached(strided)

        self.assertEqual(first.name, "row-major")
        self.assertEqual(second.name, "strided")
        self.assertEqual(create.call_count, 2)

    def test_ray_store_kv_cache_is_enabled_by_default(self) -> None:
        tensor = mock.Mock()
        tensor.is_cuda = True
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_RAY_BACKEND": "1",
            },
            clear=True,
        ):
            self.assertTrue(infinicore_backend.store_kv_cache_backend_enabled(tensor))

    def test_ray_store_kv_cache_can_be_explicitly_disabled(self) -> None:
        tensor = mock.Mock()
        tensor.is_cuda = True
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_RAY_BACKEND": "1",
                "VLLM_INFINICORE_DISABLE_RAY_STORE_KV_CACHE": "1",
            },
            clear=True,
        ):
            self.assertFalse(infinicore_backend.store_kv_cache_backend_enabled(tensor))

    def test_ray_backend_enabled_uses_ray_marker(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_INFINICORE_RAY_BACKEND": "1"}, clear=True):
            self.assertTrue(infinicore_backend.ray_backend_enabled())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(infinicore_backend.ray_backend_enabled())

    def test_musa_tensor_enables_real_backend(self) -> None:
        tensor = mock.Mock()
        tensor.is_cuda = False
        tensor.is_musa = True
        tensor.device = SimpleNamespace(type="musa", index=0)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(infinicore_backend.real_backend_enabled(tensor))
            self.assertTrue(infinicore_backend.store_kv_cache_backend_enabled(tensor))

    def test_privateuseone_tensor_maps_to_musa_device_type(self) -> None:
        tensor = mock.Mock()
        tensor.is_cuda = False
        tensor.is_musa = False
        tensor.device = SimpleNamespace(type="privateuseone", index=0)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(infinicore_backend.real_backend_enabled(tensor))
        self.assertEqual(infinicore_backend._infinicore_device_type(tensor), "musa")

    def test_backend_disable_env_still_disables_musa_tensor(self) -> None:
        tensor = mock.Mock()
        tensor.is_cuda = False
        tensor.is_musa = True
        tensor.device = SimpleNamespace(type="musa", index=0)

        with mock.patch.dict(
            os.environ,
            {"VLLM_INFINICORE_DISABLE_REAL_BACKEND": "1"},
            clear=True,
        ):
            self.assertFalse(infinicore_backend.real_backend_enabled(tensor))

    def test_prefill_cpp_bridge_uses_bhsd_cache_layout(self) -> None:
        captured = {}

        def fake_prefill(
            query,
            key_cache,
            value_cache,
            block_table,
            total_lens,
            query_start_loc,
            alibi_slopes,
            scale,
            output,
        ) -> None:
            captured["query"] = query
            captured["key_cache"] = key_cache
            captured["value_cache"] = value_cache
            captured["block_table"] = block_table
            captured["total_lens"] = total_lens
            captured["query_start_loc"] = query_start_loc
            captured["output"] = output

        module = SimpleNamespace(paged_attention_prefill_current_stream=fake_prefill)
        attn_impl = SimpleNamespace(alibi_slopes=None, scale=0.125)
        metadata = SimpleNamespace(
            num_decode_tokens=0,
            num_actual_tokens=2,
            cu_prefix_kv_lens=torch.tensor([0, 2], dtype=torch.int32),
            prefill_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            prefill_block_table=torch.tensor([[0]], dtype=torch.int32),
        )
        query = torch.empty(2, 4, 6)
        key = torch.empty(2, 4, 6)
        kv_cache = torch.empty(2, 3, 4, 5, 6)
        output = torch.empty_like(query)

        with mock.patch.object(cpp_bridge, "module", return_value=module):
            infinicore_backend._paged_attention_prefill_cpp_bridge(
                attn_impl,
                query,
                key,
                kv_cache,
                metadata,
                output,
            )

        self.assertEqual(tuple(captured["key_cache"].shape), (3, 4, 5, 6))
        self.assertEqual(tuple(captured["value_cache"].shape), (3, 4, 5, 6))
        self.assertEqual(tuple(captured["key_cache"].stride()), (120, 30, 6, 1))
        self.assertEqual(captured["output"].data_ptr(), output.data_ptr())

    def test_decode_as_prefill_cpp_bridge_records_prefill_route(self) -> None:
        captured = {}

        def fake_prefill(
            query,
            key_cache,
            value_cache,
            block_table,
            total_lens,
            query_start_loc,
            alibi_slopes,
            scale,
            output,
        ) -> None:
            captured["query"] = query
            captured["key_cache"] = key_cache
            captured["value_cache"] = value_cache
            captured["block_table"] = block_table
            captured["total_lens"] = total_lens
            captured["query_start_loc"] = query_start_loc
            captured["scale"] = scale
            captured["output"] = output

        module = SimpleNamespace(paged_attention_prefill_current_stream=fake_prefill)
        attn_impl = SimpleNamespace(alibi_slopes=None, scale=0.25)
        metadata = SimpleNamespace(
            num_actual_tokens=2,
            decode_block_table=torch.tensor([[7], [8]], dtype=torch.int32),
            decode_seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        )
        query = torch.empty(3, 4, 6)
        key = torch.empty(3, 4, 6)
        kv_cache = torch.empty(2, 3, 4, 5, 6)
        output = torch.empty_like(query)

        with mock.patch.object(cpp_bridge, "module", return_value=module):
            infinicore_backend._paged_attention_decode_as_prefill_cpp_bridge(
                attn_impl,
                query,
                key,
                kv_cache,
                metadata,
                output,
            )

        self.assertEqual(tuple(captured["query"].shape), (2, 4, 6))
        self.assertEqual(tuple(captured["key_cache"].shape), (3, 4, 5, 6))
        self.assertEqual(captured["block_table"].tolist(), [[7]])
        self.assertEqual(captured["total_lens"].tolist(), [5])
        self.assertEqual(captured["query_start_loc"].tolist(), [0, 2])
        self.assertEqual(captured["scale"], 0.25)
        self.assertEqual(captured["output"].data_ptr(), output[:2].data_ptr())
        self.assertEqual(
            cpp_bridge.bridge_call_counts().get(cpp_bridge.PREFILL_ROUTE),
            1,
        )

    def test_musa_flash_decode_uses_flash_cache_layout(self) -> None:
        captured = {}

        def fake_flash_attn_varlen_func(**kwargs):
            captured.update(kwargs)
            kwargs["out"].fill_(3)

        flash_module = ModuleType("flash_attn")
        vllm_interface = ModuleType("flash_attn.vllm_interface")
        vllm_interface.flash_attn_varlen_func = fake_flash_attn_varlen_func

        attn_impl = SimpleNamespace(
            alibi_slopes=None,
            scale=0.125,
            sliding_window=(-1, -1),
            logits_soft_cap=0.0,
            vllm_flash_attn_version=3,
            sinks=None,
        )
        metadata = SimpleNamespace(
            max_seq_len=8,
            decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            decode_seq_lens=torch.tensor([5, 6], dtype=torch.int32),
            decode_block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        )
        query = torch.empty(2, 4, 6)
        key = torch.empty(2, 4, 6)
        kv_cache = torch.empty(2, 3, 5, 4, 6)
        output = torch.empty(2, 24)

        with mock.patch.dict(
            sys.modules,
            {
                "flash_attn": flash_module,
                "flash_attn.vllm_interface": vllm_interface,
            },
        ):
            infinicore_backend._paged_attention_decode_flash_musa(
                attn_impl,
                query,
                key,
                kv_cache,
                metadata,
                output,
                num_decode_tokens=2,
            )

        self.assertEqual(tuple(captured["q"].shape), (2, 4, 6))
        self.assertEqual(tuple(captured["k"].shape), (3, 5, 4, 6))
        self.assertEqual(tuple(captured["v"].shape), (3, 5, 4, 6))
        self.assertEqual(captured["out"].data_ptr(), output.view(2, 4, 6).data_ptr())
        self.assertEqual(captured["max_seqlen_q"], 1)
        self.assertEqual(captured["max_seqlen_k"], 8)
        self.assertEqual(captured["block_table"].tolist(), [[0], [1]])
        self.assertEqual(captured["seqused_k"].tolist(), [5, 6])
        self.assertEqual(captured["fa_version"], 3)
        self.assertEqual(
            cpp_bridge.bridge_call_counts().get(cpp_bridge.FLASH_DECODE_ROUTE),
            1,
        )
        self.assertTrue(torch.equal(output, torch.full_like(output, 3)))

    def test_silu_cpp_bridge_uses_direct_contiguous_input(self) -> None:
        captured = {}

        def fake_silu_and_mul(input_arg):
            captured["input"] = input_arg
            d = input_arg.shape[-1] // 2
            return input_arg[..., :d] + input_arg[..., d:]

        module = SimpleNamespace(silu_and_mul_current_stream=fake_silu_and_mul)
        input_tensor = torch.arange(32, dtype=torch.float32).view(2, 16)[:, ::2]
        self.assertFalse(input_tensor.is_contiguous())

        with mock.patch.object(cpp_bridge, "module", return_value=module):
            result = infinicore_backend._silu_and_mul_cpp_bridge(input_tensor)

        self.assertTrue(captured["input"].is_contiguous())
        self.assertEqual(captured["input"].tolist(), input_tensor.tolist())
        self.assertTrue(
            torch.equal(
                result,
                captured["input"][..., :4] + captured["input"][..., 4:],
            )
        )
        self.assertEqual(
            cpp_bridge.bridge_call_counts().get(cpp_bridge.SILU_AND_MUL_ROUTE),
            1,
        )

    def test_rotary_cpp_bridge_reshapes_flat_heads_with_head_size(self) -> None:
        captured_shapes = []

        def fake_rope(input_tensor, positions, sin_table, cos_table, is_neox_style):
            captured_shapes.append(tuple(input_tensor.shape))
            self.assertEqual(tuple(sin_table.shape), (3, 2))
            self.assertEqual(tuple(cos_table.shape), (3, 2))
            self.assertTrue(is_neox_style)
            return input_tensor + 1

        module = SimpleNamespace(rope_current_stream=fake_rope)
        positions = torch.tensor([0, 1], dtype=torch.int64)
        angles = torch.arange(6, dtype=torch.float32).view(3, 2) * 0.1
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        query = torch.arange(16, dtype=torch.float32).view(2, 8)
        key = query + 10

        with mock.patch.object(cpp_bridge, "module", return_value=module):
            out_query, out_key = infinicore_backend._rotary_embedding_cpp_bridge(
                positions,
                query,
                key,
                4,
                cos_sin_cache,
                True,
            )

        self.assertEqual(captured_shapes, [(2, 2, 4), (2, 2, 4)])
        self.assertEqual(tuple(out_query.shape), tuple(query.shape))
        self.assertEqual(tuple(out_key.shape), tuple(key.shape))
        self.assertTrue(torch.allclose(out_query, query + 1))
        self.assertTrue(torch.allclose(out_key, key + 1))

    def test_lm_head_route_patches_logits_processor_for_tied_embeddings(self) -> None:
        module_name = "vllm_infinicore.ops.vllm_linear"

        class FakeUnquantizedLinearMethod:
            def apply(self, layer, x, bias=None):
                return "native-linear"

        class FakeUnquantizedEmbeddingMethod:
            def apply(self, layer, x, bias=None):
                return "native-embedding"

        class FakeParallelLMHead:
            pass

        class FakeLogitsProcessor:
            org_vocab_size = 3

            def _gather_logits(self, logits):
                return logits + 10

            def _get_logits(self, hidden_states, lm_head, embedding_bias):
                return torch.full((1, 3), -1.0)

        fake_modules = _fake_vllm_linear_modules(
            FakeUnquantizedLinearMethod,
            FakeUnquantizedEmbeddingMethod,
            FakeParallelLMHead,
            FakeLogitsProcessor,
        )
        old_module = sys.modules.pop(module_name, None)
        calls = {}

        def fake_lm_head(hidden_states, weight, bias):
            calls["hidden_states"] = hidden_states
            calls["weight"] = weight
            calls["bias"] = bias
            return torch.arange(5, dtype=torch.float32).view(1, 5)

        try:
            with mock.patch.dict(sys.modules, fake_modules):
                vllm_linear = importlib.import_module(module_name)
                with (
                    mock.patch.object(
                        vllm_linear,
                        "load_custom_ops",
                        return_value=SimpleNamespace(available=True, reason="ok"),
                    ),
                    mock.patch.object(
                        torch.ops,
                        "vllm_infinicore",
                        SimpleNamespace(lm_head=fake_lm_head),
                        create=True,
                    ),
                ):
                    status = vllm_linear.install_vllm_unquantized_linear_route("LMHead")
                    self.assertTrue(status.installed)

                    processor = FakeLogitsProcessor()
                    hidden_states = torch.ones(1, 2)
                    weight = torch.ones(5, 2)
                    bias = torch.ones(5)
                    logits = processor._get_logits(
                        hidden_states,
                        SimpleNamespace(weight=weight),
                        bias,
                    )

                    self.assertTrue(
                        torch.equal(logits, torch.tensor([[10.0, 11.0, 12.0]]))
                    )
                    self.assertIs(calls["hidden_states"], hidden_states)
                    self.assertIs(calls["weight"], weight)
                    self.assertIs(calls["bias"], bias)

                    vllm_linear.uninstall_vllm_unquantized_linear_route("LMHead")
                    self.assertEqual(
                        FakeLogitsProcessor()._get_logits(None, None, None).tolist(),
                        [[-1.0, -1.0, -1.0]],
                    )
        finally:
            sys.modules.pop(module_name, None)
            if old_module is not None:
                sys.modules[module_name] = old_module


def _fake_vllm_linear_modules(
    unquantized_linear_method,
    unquantized_embedding_method,
    parallel_lm_head,
    logits_processor,
) -> dict[str, ModuleType]:
    modules = {
        "vllm": _package_module("vllm"),
        "vllm.model_executor": _package_module("vllm.model_executor"),
        "vllm.model_executor.layers": _package_module("vllm.model_executor.layers"),
        "vllm.model_executor.layers.linear": ModuleType(
            "vllm.model_executor.layers.linear"
        ),
        "vllm.model_executor.layers.logits_processor": ModuleType(
            "vllm.model_executor.layers.logits_processor"
        ),
        "vllm.model_executor.layers.vocab_parallel_embedding": ModuleType(
            "vllm.model_executor.layers.vocab_parallel_embedding"
        ),
    }
    modules["vllm.model_executor.layers.linear"].UnquantizedLinearMethod = (
        unquantized_linear_method
    )
    modules["vllm.model_executor.layers.logits_processor"].LogitsProcessor = (
        logits_processor
    )
    modules[
        "vllm.model_executor.layers.vocab_parallel_embedding"
    ].ParallelLMHead = parallel_lm_head
    modules[
        "vllm.model_executor.layers.vocab_parallel_embedding"
    ].UnquantizedEmbeddingMethod = unquantized_embedding_method
    return modules


def _package_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module

if __name__ == "__main__":
    unittest.main()
