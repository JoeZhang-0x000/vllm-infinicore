from __future__ import annotations

import builtins
from pathlib import Path
import os
import subprocess
import sys
import types
import unittest
from unittest import mock

from vllm_infinicore import platform

ROOT = Path(__file__).resolve().parents[1]


class PlatformPluginTests(unittest.TestCase):
    def test_register_platform_returns_infinicore_platform_class_path(self) -> None:
        self.assertEqual(
            platform.register_platform(),
            "vllm_infinicore.platform.InfiniCorePlatform",
        )

    def test_platform_entry_point_does_not_import_vllm_or_torch(self) -> None:
        code = """
import sys
from vllm_infinicore.platform import register_platform
print(register_platform())
print("torch" in sys.modules)
print("vllm" in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.stdout.strip().splitlines(),
            [
                "vllm_infinicore.platform.InfiniCorePlatform",
                "False",
                "False",
            ],
        )

    def test_platform_manual_seed_all_uses_cuda_compatible_api(self) -> None:
        class DeviceCapability:
            def __init__(self, major: int, minor: int) -> None:
                self.major = major
                self.minor = minor

        class Platform:
            pass

        class PlatformEnum:
            OOT = "oot"

        validated_capabilities = []

        class BackendClass:
            @staticmethod
            def validate_configuration(**kwargs: object) -> list[str]:
                validated_capabilities.append(kwargs["device_capability"])
                return []

        class Backend:
            def __init__(self, name: str, path: str) -> None:
                self.name = name
                self._path = path

            def get_class(self) -> type[BackendClass]:
                return BackendClass

            def get_path(self) -> str:
                return self._path

        class AttentionBackendEnum:
            FLASH_ATTN = Backend("FLASH_ATTN", "native.flash")
            TORCH_SDPA = Backend("TORCH_SDPA", "torch.sdpa")

        vllm_pkg = types.ModuleType("vllm")
        vllm_pkg.__path__ = []
        platforms_pkg = types.ModuleType("vllm.platforms")
        platforms_pkg.__path__ = []
        interface_mod = types.ModuleType("vllm.platforms.interface")
        interface_mod.DeviceCapability = DeviceCapability
        interface_mod.Platform = Platform
        interface_mod.PlatformEnum = PlatformEnum
        v1_pkg = types.ModuleType("vllm.v1")
        v1_pkg.__path__ = []
        attention_pkg = types.ModuleType("vllm.v1.attention")
        attention_pkg.__path__ = []
        backends_pkg = types.ModuleType("vllm.v1.attention.backends")
        backends_pkg.__path__ = []
        registry_mod = types.ModuleType("vllm.v1.attention.backends.registry")
        registry_mod.AttentionBackendEnum = AttentionBackendEnum

        platform._build_platform_class.cache_clear()
        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "vllm": vllm_pkg,
                    "vllm.platforms": platforms_pkg,
                    "vllm.platforms.interface": interface_mod,
                    "vllm.v1": v1_pkg,
                    "vllm.v1.attention": attention_pkg,
                    "vllm.v1.attention.backends": backends_pkg,
                    "vllm.v1.attention.backends.registry": registry_mod,
                },
            ):
                platform_cls = platform._build_platform_class()

            import torch

            device = "musa:0" if hasattr(torch, "musa") else "cuda:0"
            with (
                mock.patch("torch.cuda.set_device") as set_device,
                mock.patch("torch.zeros") as zeros,
            ):
                platform_cls.set_device(device)

            set_device.assert_called_once_with(device)
            if hasattr(torch, "musa"):
                zeros.assert_called_once_with(1, device=device)
            else:
                zeros.assert_not_called()

            with mock.patch("torch.cuda.manual_seed_all") as manual_seed_all:
                platform_cls.manual_seed_all(123)

            manual_seed_all.assert_called_once_with(123)

            properties = types.SimpleNamespace(multi_processor_count=56)
            with mock.patch(
                "torch.cuda.get_device_properties",
                return_value=properties,
            ) as get_device_properties:
                self.assertEqual(platform_cls.num_compute_units(2), 56)

            get_device_properties.assert_called_once_with(2)

            utils_pkg = types.ModuleType("vllm.utils")
            utils_pkg.__path__ = []
            torch_utils_mod = types.ModuleType("vllm.utils.torch_utils")
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "vllm": vllm_pkg,
                        "vllm.utils": utils_pkg,
                        "vllm.utils.torch_utils": torch_utils_mod,
                    },
                ),
                mock.patch("torch.cuda.device_count", return_value=4) as device_count,
            ):
                self.assertEqual(platform_cls.device_count(), 4)

            device_count.assert_called_once_with()

            class SelectorConfig:
                use_mla = False

                def _replace(self, **kwargs: object) -> "SelectorConfig":
                    return self

                def _asdict(self) -> dict[str, object]:
                    return {}

            with (
                mock.patch.object(platform, "register_attention_backends") as register,
                mock.patch.object(
                    platform_cls,
                    "get_device_capability",
                    return_value=DeviceCapability(3, 1),
                ),
            ):
                self.assertEqual(
                    platform_cls.get_attn_backend_cls(
                        AttentionBackendEnum.FLASH_ATTN,
                        SelectorConfig(),
                        num_heads=16,
                    ),
                    platform.ATTENTION_BACKEND_CLASS_PATH,
                )
                self.assertEqual(
                    platform_cls.get_attn_backend_cls(
                        None,
                        SelectorConfig(),
                        num_heads=16,
                    ),
                    platform.ATTENTION_BACKEND_CLASS_PATH,
                )
            self.assertEqual(register.call_count, 2)
            self.assertEqual(
                [
                    (capability.major, capability.minor)
                    for capability in validated_capabilities
                ],
                [(8, 0), (8, 0)],
            )

            imported_modules = []

            def fake_import(name: str, *args: object, **kwargs: object) -> object:
                imported_modules.append(name)
                raise ImportError(name)

            with mock.patch.object(builtins, "__import__", side_effect=fake_import):
                platform_cls.import_kernels()

            self.assertEqual(
                imported_modules,
                ["vllm._C", "vllm._moe_C", "mcoplib._C", "mcoplib._moe_C"],
            )
            expected_dispatch_key = "MUSA" if hasattr(torch, "musa") else "CUDA"
            expected_dist_backend = "mccl" if hasattr(torch, "musa") else "nccl"
            self.assertTrue(platform_cls.opaque_attention_op())
            self.assertFalse(platform_cls.use_custom_allreduce())
            self.assertEqual(platform_cls.dispatch_key, expected_dispatch_key)
            self.assertEqual(platform_cls.dist_backend, expected_dist_backend)
            expected_communicator = (
                "vllm_infinicore.communicator.InfiniCoreMusaCommunicator"
                if hasattr(torch, "musa")
                else "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
            )
            self.assertEqual(
                platform_cls.get_device_communicator_cls(),
                expected_communicator,
            )

            class GraphMode:
                def has_piecewise_cudagraphs(self) -> bool:
                    return True

            fake_vllm_config = types.SimpleNamespace(
                parallel_config=types.SimpleNamespace(worker_cls="auto"),
                cache_config=types.SimpleNamespace(block_size=None),
                model_config=types.SimpleNamespace(
                    enforce_eager=False,
                    disable_cascade_attn=False,
                ),
                compilation_config=types.SimpleNamespace(cudagraph_mode=GraphMode()),
            )
            with (
                mock.patch.object(torch, "musa", types.SimpleNamespace(), create=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "vllm": vllm_pkg,
                        "vllm.platforms": platforms_pkg,
                        "vllm.platforms.interface": interface_mod,
                        "vllm.v1": v1_pkg,
                        "vllm.v1.attention": attention_pkg,
                        "vllm.v1.attention.backends": backends_pkg,
                        "vllm.v1.attention.backends.registry": registry_mod,
                    },
                ),
                mock.patch.dict(
                    os.environ,
                    {},
                    clear=True,
                ),
                mock.patch(
                    "vllm_infinicore.runtime_patches.apply_vllm_020_compat_patches"
                ),
                mock.patch(
                    "vllm_infinicore.runtime_patches."
                    "patch_gpu_model_runner_dummy_run_real_reqs"
                ),
            ):
                os.environ.pop("TORCH_COMPILE_DISABLE", None)
                os.environ.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
                platform._build_platform_class.cache_clear()
                musa_platform_cls = platform._build_platform_class()
                self.assertEqual(
                    musa_platform_cls.get_device_communicator_cls(),
                    "vllm_infinicore.communicator.InfiniCoreMusaCommunicator",
                )
                musa_platform_cls.check_and_update_config(fake_vllm_config)
                self.assertEqual(os.environ.get("TORCH_COMPILE_DISABLE"), "1")
                self.assertEqual(
                    os.environ.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES"),
                    "1",
                )

            self.assertEqual(
                fake_vllm_config.parallel_config.worker_cls,
                "vllm.v1.worker.gpu_worker.Worker",
            )
            self.assertEqual(fake_vllm_config.cache_config.block_size, 16)
            self.assertTrue(fake_vllm_config.model_config.disable_cascade_attn)
        finally:
            platform._build_platform_class.cache_clear()

    def test_musa_communicator_satisfies_cuda_graph_type_check(self) -> None:
        try:
            from vllm.distributed.device_communicators.cuda_communicator import (
                CudaCommunicator,
            )
            from vllm.distributed.device_communicators.xpu_communicator import (
                XpuCommunicator,
            )
            from vllm_infinicore.communicator import InfiniCoreMusaCommunicator
        except ModuleNotFoundError as exc:
            self.skipTest(f"vLLM communicator dependencies unavailable: {exc}")

        self.assertTrue(issubclass(InfiniCoreMusaCommunicator, CudaCommunicator))
        self.assertTrue(issubclass(InfiniCoreMusaCommunicator, XpuCommunicator))

    def test_platform_attention_registration_respects_explicit_routes(self) -> None:
        try:
            from vllm_infinicore.ops import vllm_attention_backend
        except ModuleNotFoundError as exc:
            self.skipTest(f"attention backend dependencies unavailable: {exc}")

        vllm_attention_backend._ACTIVE_ROUTES.clear()
        with (
            mock.patch.object(vllm_attention_backend, "_register_backend_once"),
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                    "VLLM_INFINICORE_ROUTES": "throughput",
                },
            ),
        ):
            vllm_attention_backend.install_platform_attention_backend()

        self.assertEqual(vllm_attention_backend._ACTIVE_ROUTES, set())

    def test_platform_only_attention_registration_keeps_no_metax_default(self) -> None:
        try:
            from vllm_infinicore.ops import vllm_attention_backend
        except ModuleNotFoundError as exc:
            self.skipTest(f"attention backend dependencies unavailable: {exc}")

        vllm_attention_backend._ACTIVE_ROUTES.clear()
        with (
            mock.patch.object(vllm_attention_backend, "_register_backend_once"),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            vllm_attention_backend.install_platform_attention_backend()

        self.assertEqual(
            vllm_attention_backend._ACTIVE_ROUTES,
            set(vllm_attention_backend.INFINICORE_ATTENTION_BACKEND_ROUTES),
        )
        vllm_attention_backend._ACTIVE_ROUTES.clear()


if __name__ == "__main__":
    unittest.main()
