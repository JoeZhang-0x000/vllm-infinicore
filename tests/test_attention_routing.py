"""CPU tests for metadata adaptation and fail-closed routing semantics."""
from enum import Enum
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from vllm_infinicore.operators import attention as route
from vllm_infinicore.operators import attention_ops as ops
from vllm_infinicore.routing.patching import get_default_registry, _parse_route_names


class AttentionRoutingTests(unittest.TestCase):
    def setUp(self):
        self.previous = set(route._ACTIVE)
        route._ACTIVE.clear()
        route._ACTIVE.update(route.ROUTES)
        self.impl = SimpleNamespace(num_heads=4, num_kv_heads=2, head_size=8,
                                    scale=8 ** -0.5, attn_type="decoder")
        self.q = torch.randn(5, 4, 8)
        self.k = torch.randn(5, 2, 8)
        self.cache = torch.zeros(2, 3, 16, 2, 8)
        self.meta = SimpleNamespace(
            num_actual_tokens=5, num_decode_tokens=2, num_decodes=2, num_prefills=1,
            query_start_loc=torch.tensor([0, 1, 2, 5], dtype=torch.int32),
            seq_lens=torch.tensor([9, 7, 3], dtype=torch.int32),
            block_table=torch.tensor([[2], [1], [0]], dtype=torch.int32),
            slot_mapping=torch.arange(5),
        )

    def tearDown(self):
        route._ACTIVE.clear()
        route._ACTIVE.update(self.previous)

    def test_mixed_batch_splits_device_metadata_without_changing_offsets(self):
        native = Mock(side_effect=AssertionError("native attention called"))
        out = torch.empty_like(self.q)
        with patch.object(ops, "selected_backend", return_value="metax"), \
             patch.object(ops, "compute") as compute:
            actual = route._forward(native, "metax")(
                self.impl, None, self.q, self.k, self.k, self.cache, self.meta, out)
        self.assertIs(actual, out)
        self.assertEqual(compute.call_count, 2)
        decode, prefill = compute.call_args_list
        self.assertTrue(decode.kwargs["decode"])
        self.assertFalse(prefill.kwargs["decode"])
        self.assertEqual(decode.args[0].shape[0], 2)
        self.assertEqual(prefill.args[0].shape[0], 3)
        self.assertEqual(prefill.args[5].tolist(), [0, 3])
        self.assertEqual(prefill.args[4].tolist(), [3])
        self.assertEqual(prefill.args[3].tolist(), [[0]])
        native.assert_not_called()

    def test_cache_views_preserve_storage_for_both_layouts(self):
        for backend, cache in (("metax", self.cache),
                               ("ascend", self.cache),
                               ("kunlun", self.cache.transpose(2, 3))):
            with self.subTest(backend=backend), patch.object(ops, "selected_backend", return_value=backend):
                k, v = ops.cache_views(cache, 2)
                self.assertEqual(k.shape, (3, 2, 16, 8))
                self.assertEqual(k.data_ptr(), cache[0].data_ptr())
                self.assertEqual(v.data_ptr(), cache[1].data_ptr())

    def test_vendor_enum_and_unsupported_attention(self):
        class AttentionType(Enum):
            DECODER = "decoder"
        self.impl.attn_type = AttentionType.DECODER
        self.assertIsNone(route._unsupported(self.impl, self.meta, None, None))
        native = Mock()
        self.impl.sliding_window = (128, 0)
        with self.assertRaisesRegex(NotImplementedError, "sliding window"):
            route._forward(native, "metax")(
                self.impl, None, self.q, self.k, self.k, self.cache, self.meta)
        native.assert_not_called()

    def test_failed_launch_is_never_retried_natively(self):
        native = Mock()
        with patch.object(ops, "selected_backend", return_value="metax"), \
             patch.object(ops, "compute", side_effect=RuntimeError("device launch failed")):
            with self.assertRaisesRegex(RuntimeError, "device launch failed"):
                route._forward(native, "metax")(
                    self.impl, None, self.q, self.k, self.k, self.cache, self.meta)
        native.assert_not_called()

    def test_kunlun_store_only_uses_infinicore_and_restores_native(self):
        native = SimpleNamespace(reshape_and_cache_flash=Mock())
        proxy = route._KunlunOps(native)
        with patch.object(ops, "store") as store:
            route._ACTIVE.clear()
            route._ACTIVE.add("StoreKVCache")
            proxy.reshape_and_cache_flash(self.k, self.k, self.cache[0], self.cache[1], torch.arange(5))
            store.assert_called_once()
            native.reshape_and_cache_flash.assert_not_called()
            route._ACTIVE.clear()
            proxy.reshape_and_cache_flash(self.k, self.k, self.cache[0], self.cache[1], torch.arange(5))
            native.reshape_and_cache_flash.assert_called_once()

    def test_store_only_keeps_mixed_attention_native(self):
        route._ACTIVE.clear()
        route._ACTIVE.add("StoreKVCache")
        native = Mock(return_value=self.q)
        with patch.object(ops, "compute") as compute:
            result = route._forward(native, "metax")(
                self.impl, None, self.q, self.k, self.k, self.cache, self.meta)
        self.assertIs(result, self.q)
        native.assert_called_once()
        compute.assert_not_called()

    def test_ascend_unsupported_fused_norm_records_native_without_launch(self):
        from vllm_infinicore.operators.ascend import backend as ascend
        from vllm_infinicore.operators import backend
        native = Mock(return_value="native result")
        with patch.dict(backend._FALLBACK_COUNTS, {}, clear=True), \
             patch.dict(backend._FALLBACK_REASONS, {}, clear=True):
            result = ascend.fallback("fused_add_rms_norm", "unsupported fused kernel", native)
            self.assertEqual(result, "native result")
            self.assertEqual(backend.backend_fallback_counts(), {"fused_add_rms_norm": 1})
            self.assertEqual(backend.backend_fallback_reasons()["fused_add_rms_norm"], "unsupported fused kernel")
        native.assert_called_once()

    def test_recommended_store_selects_before_launch_and_preserves_arguments(self):
        with patch.dict("os.environ", {"VLLM_INFINICORE_ROUTES": "recommended"}), \
             patch.object(ops, "selected_backend", return_value="metax"), \
             patch.object(ops, "store") as store:
            native = Mock()
            update = route._update(native)
            slots = torch.arange(33)
            update(self.impl, None, self.k, self.k, self.cache, slots)
            native.assert_called_once_with(self.impl, None, self.k, self.k, self.cache, slots)
            store.assert_not_called()
            update(self.impl, None, self.k, self.k, self.cache, slots[:32])
            store.assert_called_once()
            self.assertEqual(native.call_count, 1)
            # An explicitly selected store must not acquire an implicit limit.
            with patch.dict("os.environ", {"VLLM_INFINICORE_ROUTES": "StoreKVCache"}):
                update(self.impl, None, self.k, self.k, self.cache, slots)
            self.assertEqual(store.call_count, 2)

    def test_recommended_profiles_keep_attention_native_and_match_bridge(self):
        from vllm_infinicore.operators import cpp_bridge
        for backend in ("ascend", "metax", "kunlun"):
            with self.subTest(backend=backend), patch.dict("os.environ", {
                "VLLM_INFINICORE_OPERATOR_BACKEND": backend,
                "VLLM_INFINICORE_CPP_BRIDGE_ROUTES": "recommended",
                "VLLM_INFINICORE_ENABLE_CPP_BRIDGE": "1",
                "VLLM_INFINICORE_DISABLE_CPP_BRIDGE": "0",
            }):
                selected = _parse_route_names("recommended")
                self.assertIn("StoreKVCache", selected)
                self.assertNotIn("PagedAttentionPrefill", selected)
                self.assertNotIn("PagedAttentionDecode", selected)
                if backend != "ascend":
                    self.assertEqual(selected, cpp_bridge.selected_routes())
        self.assertEqual(_parse_route_names("all", available_routes=("PagedAttentionPrefill",)),
                         ("PagedAttentionPrefill",))

    def test_registry_exposes_attention_for_each_supported_vendor(self):
        for backend in ("metax", "kunlun", "ascend"):
            with self.subTest(backend=backend), patch.dict("os.environ", {
                "VLLM_INFINICORE_OPERATOR_BACKEND": backend,
                "VLLM_INFINICORE_ASCEND_LIBRARY": "/test/library.so",
            }):
                registry = get_default_registry()
                for name in route.ROUTES:
                    self.assertEqual(registry.routes[name].implementation, f"{backend}_operator_adapter")

    def test_metax_rope_routes_compiler_native_entry_and_restores_inheritance(self):
        from vllm_infinicore.operators.metax import routes as metax
        from vllm_infinicore.operators import custom_ops

        class Base:
            def forward_native(self, *args):
                raise AssertionError("compiler bypassed InfiniCore")

        class VendorRoPE(Base):
            head_size, rotary_dim, is_neox_style = 8, 8, True

            def forward_oot(self, *args):
                raise AssertionError("native OOT RoPE called")

            def _match_cos_sin_cache_dtype(self, query):
                return torch.zeros(8, 8)

        original = VendorRoPE.forward_oot
        module_name = "vllm_metax.customized.ops.rotary_embedding"
        with patch.dict(sys.modules, {module_name: SimpleNamespace(MacaRotaryEmbedding=VendorRoPE)}), \
             patch.object(custom_ops, "load_custom_ops", return_value=SimpleNamespace(available=True)), \
             patch.object(torch.ops.vllm_infinicore, "rotary_embedding", create=True) as kernel:
            try:
                self.assertTrue(metax._install_rope().installed)
                op = VendorRoPE()
                op.forward_native(torch.arange(5), self.q, self.k)
                op.forward_oot(torch.arange(5), self.q, self.k)
                self.assertEqual(kernel.call_count, 2)
                self.assertTrue(metax._uninstall_rope().uninstalled)
                self.assertIs(VendorRoPE.forward_oot, original)
                self.assertNotIn("forward_native", vars(VendorRoPE))
            finally:
                if metax._ROPE_PATCH is not None:
                    metax._uninstall_rope()

    def test_patch_restore_does_not_overwrite_another_owner(self):
        class Target:
            def method(self):
                return "original"
        original = Target.method
        before = len(route._PATCHES)
        try:
            route._patch(Target, "method", lambda _: lambda self: "ours")
            ours = Target.method
            Target.method = lambda self: "third party"
            route._ACTIVE.clear()
            route._ACTIVE.add("StoreKVCache")
            result = route.uninstall("StoreKVCache", "metax")
            self.assertFalse(result.uninstalled)
            self.assertEqual(Target().method(), "third party")
            Target.method = ours
            self.assertTrue(route.uninstall("StoreKVCache", "metax").uninstalled)
            self.assertIs(Target.method, original)
        finally:
            del route._PATCHES[before:]

    def test_metax_rms_routes_plain_and_fused_but_preserves_variants_and_uninstall(self):
        from vllm_infinicore.operators.metax import routes as metax
        from vllm_infinicore.operators import custom_ops

        native = Mock(return_value="native residual")

        class Base:
            forward_native = native

        class VendorRMS(Base):
            has_weight, variance_size_override = True, None
            variance_epsilon, weight = 1e-6, torch.ones(8)

            def forward_oot(self, x, residual=None):
                return self.forward_native(x, residual)

        original = VendorRMS.forward_oot
        with patch.dict(sys.modules, {"vllm_metax.customized.ops.layernorm": SimpleNamespace(MacaRMSNorm=VendorRMS)}), \
             patch.object(custom_ops, "load_custom_ops", return_value=SimpleNamespace(available=True)), \
             patch.object(torch.ops.vllm_infinicore, "rms_norm", create=True) as kernel, \
             patch.object(torch.ops.vllm_infinicore, "fused_add_rms_norm", create=True) as fused:
            try:
                self.assertTrue(metax._install_rms_norm().installed)
                op = VendorRMS()
                op.forward_native(self.q)
                op.forward_oot(self.q)
                self.assertEqual(kernel.call_count, 2)
                op.forward_oot(self.q, self.q)
                fused.assert_called_once()
                native.assert_not_called()
                op.pass_weight_add = False
                self.assertEqual(op.forward_oot(self.q, self.q), "native residual")
                native.assert_called_once_with(op, self.q, self.q)
                replacement = VendorRMS.forward_native
                VendorRMS.forward_native = lambda *a: "another owner"
                self.assertFalse(metax._uninstall_rms_norm().uninstalled)
                VendorRMS.forward_native = replacement
                self.assertTrue(metax._uninstall_rms_norm().uninstalled)
                self.assertIs(VendorRMS.forward_oot, original)
                self.assertNotIn("forward_native", vars(VendorRMS))
            finally:
                if metax._RMS_PATCH is not None:
                    metax._uninstall_rms_norm()


if __name__ == "__main__":
    unittest.main()
