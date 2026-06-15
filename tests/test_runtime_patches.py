from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from vllm_infinicore import runtime_patches


class RuntimePatchTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime_patches._SAFE_FAKE_PATCHED = False
        runtime_patches._SAFE_FAKE_PATCH_REASON = "not applied"
        runtime_patches._ACCELERATOR_MEMORY_PATCHED = False
        runtime_patches._ACCELERATOR_MEMORY_PATCH_REASON = "not applied"
        runtime_patches._FUSED_MOE_MONOLITHIC_PATCHED = False
        runtime_patches._FUSED_MOE_MONOLITHIC_PATCH_REASON = "not applied"
        runtime_patches._FUSED_MOE_INT4_PATCHED = False
        runtime_patches._FUSED_MOE_INT4_PATCH_REASON = "not applied"
        runtime_patches._METAX_OOT_MOE_PATCHED = False
        runtime_patches._METAX_OOT_MOE_PATCH_REASON = "not applied"
        runtime_patches._FUNCTORCH_AUTOGRAD_CACHE_PATCHED = False
        runtime_patches._FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON = "not applied"

    def test_dummy_run_source_uses_real_request_count(self) -> None:
        source = """
            @torch.inference_mode()
            def _dummy_run(self):
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                )
        """

        patched = runtime_patches._dummy_run_real_reqs_source(source)

        self.assertIsNotNone(patched)
        assert patched is not None
        self.assertIn("num_reqs=num_reqs,", patched)
        self.assertNotIn("num_reqs=num_reqs_padded,", patched)

    def test_dummy_run_source_returns_none_when_pattern_missing(self) -> None:
        source = """
            def _dummy_run(self):
                return self._build_attention_metadata(num_reqs=num_reqs)
        """

        self.assertIsNone(runtime_patches._dummy_run_real_reqs_source(source))

    def test_accelerator_memory_api_is_mirrored_from_cuda(self) -> None:
        fake_torch = types.SimpleNamespace(
            accelerator=types.SimpleNamespace(),
            cuda=types.SimpleNamespace(
                empty_cache=lambda: "empty",
                memory_stats=lambda: "stats",
                memory_reserved=lambda: "reserved",
                reset_peak_memory_stats=lambda: "reset_peak",
                max_memory_allocated=lambda: "max",
                memory_allocated=lambda: "allocated",
                reset_accumulated_memory_stats=lambda: "reset_accumulated",
            ),
        )

        status = runtime_patches.patch_torch_accelerator_memory_api(fake_torch)

        self.assertTrue(status.applied)
        self.assertEqual(fake_torch.accelerator.memory_reserved(), "reserved")
        self.assertEqual(fake_torch.accelerator.max_memory_allocated(), "max")

    def test_safe_fake_registration_skips_missing_ops(self) -> None:
        def missing_register_fake(*_args, **_kwargs):
            raise RuntimeError("operator vllm::_C::scaled_fp4_quant.out does not exist")

        fake_library = types.SimpleNamespace(register_fake=missing_register_fake)
        fake_torch = types.SimpleNamespace(library=fake_library)

        status = runtime_patches.patch_torch_library_register_fake_missing_ops(
            fake_torch
        )
        decorated = fake_torch.library.register_fake("missing::op")(lambda x: x)
        direct = fake_torch.library.register_fake("missing::op", lambda x: x)

        self.assertTrue(status.applied)
        self.assertEqual(decorated("ok"), "ok")
        self.assertEqual(direct("ok"), "ok")

    def test_functorch_autograd_cache_config_is_added_when_missing(self) -> None:
        try:
            from torch.utils._config_module import _Config, _ConfigEntry
        except Exception as exc:  # pragma: no cover - torch test dependency guard
            self.skipTest(f"torch config helpers unavailable: {exc}")

        module = types.ModuleType("torch._functorch.config")
        module._config = {"enable_autograd_cache": _ConfigEntry(_Config(default=False))}

        with mock.patch.dict(sys.modules, {"torch._functorch.config": module}):
            status = runtime_patches.patch_functorch_autograd_cache_normalize_inputs(
                types.SimpleNamespace()
            )

        self.assertTrue(status.applied)
        self.assertIn("autograd_cache_normalize_inputs", module._config)
        self.assertFalse(module._config["autograd_cache_normalize_inputs"].default)

    def test_fused_moe_is_monolithic_handles_none_experts_cls(self) -> None:
        module = types.ModuleType(
            "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
        )

        class FusedMoEMethodBase:
            moe_kernel = None
            experts_cls = None

            @property
            def is_monolithic(self):
                if self.moe_kernel is None:
                    if hasattr(self, "experts_cls"):
                        return self.experts_cls.is_monolithic()
                    return False
                return self.moe_kernel.is_monolithic

        module.FusedMoEMethodBase = FusedMoEMethodBase

        with mock.patch.dict(sys.modules, {module.__name__: module}):
            status = runtime_patches.patch_fused_moe_method_is_monolithic()

        self.assertTrue(status.applied)
        self.assertFalse(FusedMoEMethodBase().is_monolithic)

    def test_fused_moe_quant_config_gets_int4_w4a8_property(self) -> None:
        module = types.ModuleType("vllm.model_executor.layers.fused_moe.config")

        class FusedMoEQuantConfig:
            def __init__(self) -> None:
                self._a1 = types.SimpleNamespace(dtype="int8")
                self._w1 = types.SimpleNamespace(dtype="int4")

        module.FusedMoEQuantConfig = FusedMoEQuantConfig

        with mock.patch.dict(sys.modules, {module.__name__: module}):
            status = runtime_patches.patch_fused_moe_quant_config_use_int4_w4a8()

        self.assertTrue(status.applied)
        self.assertTrue(FusedMoEQuantConfig().use_int4_w4a8)

    def test_metax_oot_unquantized_moe_backend_uses_triton_experts(self) -> None:
        module = types.ModuleType(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized"
        )
        module.current_platform = types.SimpleNamespace(is_out_of_tree=lambda: True)
        module.UnquantizedMoeBackend = types.SimpleNamespace(TRITON="TRITON")

        def original(_moe_config):
            return "OOT", None

        module.select_unquantized_moe_backend = original
        metax_fused_moe = types.ModuleType("vllm_metax.utils.fused_moe")
        metax_fused_moe.get_triton_experts_cls = lambda: "MetaXExperts"

        with (
            mock.patch.dict(
                sys.modules,
                {
                    module.__name__: module,
                    "vllm_metax.utils.fused_moe": metax_fused_moe,
                },
            ),
            mock.patch.dict(os.environ, {"VLLM_PLUGINS": "metax,vllm_infinicore"}),
        ):
            status = runtime_patches.patch_metax_oot_unquantized_moe_backend()
            selected = module.select_unquantized_moe_backend(object())

        self.assertTrue(status.applied)
        self.assertEqual(selected, ("TRITON", "MetaXExperts"))


if __name__ == "__main__":
    unittest.main()
