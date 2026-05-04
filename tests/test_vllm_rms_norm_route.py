from __future__ import annotations

import os
import unittest


def _prepare_vllm_env() -> None:
    site_packages = "/opt/conda/lib/python3.12/site-packages"
    torch_lib = f"{site_packages}/torch/lib"
    maca_path = "/opt/maca-3.5.3"
    os.environ.setdefault("MACA_PATH", maca_path)
    os.environ.setdefault("MACA_HOME", maca_path)
    os.environ.setdefault("MACA_ROOT", maca_path)
    os.environ.setdefault("INFINI_ROOT", os.path.expanduser("~/.infini"))
    os.environ.setdefault("PYTHON_SITE_PACKAGES", site_packages)
    os.environ.setdefault("TORCH_LIB", torch_lib)
    os.environ.setdefault(
        "FLASH_ATTN_2_CUDA_SO",
        f"{site_packages}/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so",
    )
    os.environ.setdefault("XMAKE_ROOT", "y")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_INFINICORE_ENABLE_PATCHES", "0")


class VllmRMSNormRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _prepare_vllm_env()
        try:
            import torch
            from vllm.config import CompilationConfig, VllmConfig
            from vllm.config.compilation import CompilationMode
            from vllm.config.vllm import set_current_vllm_config
            from vllm.model_executor.layers.layernorm import RMSNorm as NativeRMSNorm
            from vllm_infinicore.ops import custom_ops
            from vllm_infinicore.ops.vllm_rms_norm import (
                InfiniCoreRMSNorm,
                install_vllm_rms_norm_oot,
                uninstall_vllm_rms_norm_oot,
            )
        except Exception as exc:  # pragma: no cover - depends on local vLLM
            raise unittest.SkipTest(f"vLLM RMSNorm route unavailable: {exc}") from exc

        status = custom_ops.load_custom_ops(
            force=True,
            required_ops=(custom_ops.RMS_NORM_OP,),
        )
        if not status.available:
            raise unittest.SkipTest(status.reason)

        cls.torch = torch
        cls.NativeRMSNorm = NativeRMSNorm
        cls.InfiniCoreRMSNorm = InfiniCoreRMSNorm
        cls.install_vllm_rms_norm_oot = staticmethod(install_vllm_rms_norm_oot)
        cls.uninstall_vllm_rms_norm_oot = staticmethod(uninstall_vllm_rms_norm_oot)
        cls.set_current_vllm_config = staticmethod(set_current_vllm_config)
        cls.vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE,
                backend="eager",
                custom_ops=["all"],
            )
        )

    def test_infinicore_rms_norm_matches_vllm_native(self) -> None:
        torch = self.torch
        hidden_size = 16
        eps = 1e-6
        dtypes = [torch.float32, torch.float16]
        if hasattr(torch, "bfloat16"):
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                try:
                    x = torch.randn((4, hidden_size), dtype=torch.float32).to(dtype)
                    weight = (
                        torch.randn((hidden_size,), dtype=torch.float32).abs() + 0.25
                    ).to(dtype)
                except Exception as exc:
                    self.skipTest(f"{dtype} unavailable: {exc}")

                with self.set_current_vllm_config(
                    self.vllm_config, check_compile=False
                ):
                    native = self.NativeRMSNorm(hidden_size, eps=eps, dtype=dtype)
                    candidate = self.InfiniCoreRMSNorm(
                        hidden_size, eps=eps, dtype=dtype
                    )
                    native.weight.data.copy_(weight)
                    candidate.weight.data.copy_(weight)

                    expected = native.forward_native(x.clone())
                    actual = candidate.forward_oot(x.clone())
                torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)

    def test_residual_path_falls_back_to_native(self) -> None:
        torch = self.torch
        hidden_size = 8
        dtype = torch.float32
        x = torch.randn((2, hidden_size), dtype=dtype)
        residual = torch.randn((2, hidden_size), dtype=dtype)
        weight = torch.randn((hidden_size,), dtype=dtype).abs() + 0.25

        with self.set_current_vllm_config(self.vllm_config, check_compile=False):
            native = self.NativeRMSNorm(hidden_size, eps=1e-6, dtype=dtype)
            candidate = self.InfiniCoreRMSNorm(hidden_size, eps=1e-6, dtype=dtype)
            native.weight.data.copy_(weight)
            candidate.weight.data.copy_(weight)

            expected = native.forward_native(x.clone(), residual.clone())
            actual = candidate.forward_oot(x.clone(), residual.clone())

        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 2)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        self.assertFalse(candidate._should_use_infinicore(x, residual))

    def test_variance_override_and_no_weight_do_not_route(self) -> None:
        torch = self.torch
        hidden_size = 8
        x = torch.randn((2, 3, hidden_size), dtype=torch.float32)

        with self.set_current_vllm_config(self.vllm_config, check_compile=False):
            with_override = self.InfiniCoreRMSNorm(
                hidden_size,
                eps=1e-6,
                var_hidden_size=4,
                dtype=torch.float32,
            )
            without_weight = self.InfiniCoreRMSNorm(
                hidden_size,
                eps=1e-6,
                has_weight=False,
                dtype=torch.float32,
            )

            self.assertFalse(with_override._should_use_infinicore(x, None))
            self.assertFalse(without_weight._should_use_infinicore(x, None))
            torch.testing.assert_close(
                with_override.forward_oot(x.clone()),
                with_override.forward_native(x.clone()),
            )
            torch.testing.assert_close(
                without_weight.forward_oot(x.clone()),
                without_weight.forward_native(x.clone()),
            )

    def test_oot_registration_is_idempotent(self) -> None:
        first = self.install_vllm_rms_norm_oot()
        second = self.install_vllm_rms_norm_oot()

        self.assertTrue(first.installed, first.reason)
        self.assertTrue(second.installed, second.reason)

    def test_oot_unregistration_is_idempotent(self) -> None:
        installed = self.install_vllm_rms_norm_oot()
        self.assertTrue(installed.installed, installed.reason)

        first = self.uninstall_vllm_rms_norm_oot()
        second = self.uninstall_vllm_rms_norm_oot()

        self.assertTrue(first.uninstalled, first.reason)
        self.assertFalse(second.uninstalled)
        self.assertIn("not installed", second.reason)

    def test_oot_registration_replaces_vllm_instantiation(self) -> None:
        status = self.install_vllm_rms_norm_oot()
        self.assertTrue(status.installed, status.reason)

        with self.set_current_vllm_config(self.vllm_config, check_compile=False):
            layer = self.NativeRMSNorm(8, eps=1e-6, dtype=self.torch.float32)

        self.assertIsInstance(layer, self.InfiniCoreRMSNorm)


if __name__ == "__main__":
    unittest.main()
