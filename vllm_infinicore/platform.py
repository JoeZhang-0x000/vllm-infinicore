"""InfiniCore platform plugin entry point for vLLM.

The platform class is built lazily so the entry point can be discovered without
importing vLLM or torch. vLLM imports the returned class path only after it has
selected this platform plugin.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser


PLATFORM_CLASS_PATH = "vllm_infinicore.platform.InfiniCorePlatform"
ATTENTION_BACKEND_CLASS_PATH = (
    "vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend"
)


def register_platform() -> str:
    """Return the vLLM platform class path for the InfiniCore platform."""

    return PLATFORM_CLASS_PATH


def register_attention_backends() -> None:
    """Register InfiniCore attention backends with vLLM."""

    from vllm_infinicore.ops.vllm_attention_backend import (
        install_platform_attention_backend,
    )

    install_platform_attention_backend()


def __getattr__(name: str) -> Any:
    if name == "InfiniCorePlatform":
        return _build_platform_class()
    raise AttributeError(name)


@cache
def _build_platform_class() -> type:
    import torch

    from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    @cache
    def backend_priorities(
        use_mla: bool,
        device_capability: DeviceCapability,
    ) -> tuple[AttentionBackendEnum, ...]:
        del use_mla, device_capability
        return (AttentionBackendEnum.FLASH_ATTN,)

    def float8_dtypes() -> tuple[torch.dtype, ...]:
        return tuple(
            dtype
            for name in ("float8_e4m3fn", "float8_e5m2")
            if (dtype := getattr(torch, name, None)) is not None
        )

    class InfiniCorePlatform(Platform):
        _enum = PlatformEnum.OOT
        device_name: str = "infinicore"
        device_type: str = "cuda"
        dispatch_key: str = "CUDA"
        ray_device_key: str = "GPU"
        dist_backend: str = "nccl"
        device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

        supported_quantization: list[str] = [
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
            "gguf",
        ]

        @classmethod
        def set_device(cls, device: torch.device) -> None:
            torch.cuda.set_device(device)

        @classmethod
        @cache
        def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
            major, minor = torch.cuda.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:
            try:
                return torch.cuda.get_device_name(device_id)
            except Exception:
                return f"InfiniCore device {device_id}"

        @classmethod
        def get_device_total_memory(cls, device_id: int = 0) -> int:
            return torch.cuda.get_device_properties(device_id).total_memory

        @classmethod
        def get_current_memory_usage(
            cls,
            device: torch.types.Device | None = None,
        ) -> float:
            return torch.cuda.memory_allocated(device)

        @classmethod
        def device_count(cls) -> int:
            from vllm.utils.torch_utils import cuda_device_count_stateless

            return cuda_device_count_stateless()

        @classmethod
        def get_device_uuid(cls, device_id: int = 0) -> str:
            try:
                return torch.cuda.get_device_properties(device_id).uuid
            except Exception:
                return f"infinicore-{device_id}"

        @classmethod
        def is_fully_connected(cls, device_ids: list[int]) -> bool:
            del device_ids
            return True

        @classmethod
        def log_warnings(cls) -> None:
            return None

        @classmethod
        def is_cuda_alike(cls) -> bool:
            return True

        @classmethod
        def supports_fp8(cls) -> bool:
            return False

        @classmethod
        def check_if_supports_dtype(cls, torch_dtype: torch.dtype) -> None:
            if torch_dtype in float8_dtypes():
                raise ValueError("FP8 is not supported by InfiniCore platform")

        @classmethod
        def opaque_attention_op(cls) -> bool:
            return True

        @classmethod
        def get_attn_backend_cls(
            cls,
            selected_backend: AttentionBackendEnum,
            attn_selector_config: "AttentionSelectorConfig",
        ) -> str:
            device_capability = cls.get_device_capability()
            config = attn_selector_config._replace(block_size=None)

            if selected_backend is not None:
                _validate_attention_backend(selected_backend, device_capability, config)
                return selected_backend.get_path()

            valid_backends = cls._valid_backends(device_capability, config)
            if not valid_backends:
                raise ValueError(f"No valid attention backend found for {cls.device_name}.")
            return valid_backends[0].get_path()

        @classmethod
        def _valid_backends(
            cls,
            device_capability: DeviceCapability,
            config: "AttentionSelectorConfig",
        ) -> list[AttentionBackendEnum]:
            valid = []
            for backend in backend_priorities(config.use_mla, device_capability):
                try:
                    _validate_attention_backend(backend, device_capability, config)
                except ValueError:
                    continue
                valid.append(backend)
            return valid

        @classmethod
        def get_supported_vit_attn_backends(cls) -> list[AttentionBackendEnum]:
            return [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA]

        @classmethod
        def get_vit_attn_backend(
            cls,
            head_size: int,
            dtype: torch.dtype,
            backend: AttentionBackendEnum | None = None,
        ) -> AttentionBackendEnum:
            if backend is not None:
                if backend not in cls.get_supported_vit_attn_backends():
                    raise ValueError(f"Unsupported ViT attention backend: {backend}")
                return backend

            flash_attn = AttentionBackendEnum.FLASH_ATTN.get_class()
            if flash_attn.supports_head_size(head_size) and flash_attn.supports_dtype(dtype):
                return AttentionBackendEnum.FLASH_ATTN
            return AttentionBackendEnum.TORCH_SDPA

        @classmethod
        def support_static_graph_mode(cls) -> bool:
            return True

        @classmethod
        def get_static_graph_wrapper_cls(cls) -> str:
            return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

        @classmethod
        def support_hybrid_kv_cache(cls) -> bool:
            return True

        @classmethod
        def get_device_communicator_cls(cls) -> str:
            return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"

        @classmethod
        def use_custom_allreduce(cls) -> bool:
            return False

        @classmethod
        def get_punica_wrapper(cls) -> str:
            return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

        @classmethod
        def import_kernels(cls) -> None:
            for module_name in ("mcoplib._C", "mcoplib._moe_C"):
                try:
                    __import__(module_name)
                except ImportError:
                    continue

        @classmethod
        def pre_register_and_update(
            cls,
            parser: "FlexibleArgumentParser | None" = None,
        ) -> None:
            del parser
            cls.import_kernels()
            register_attention_backends()

        @classmethod
        def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
            from vllm_infinicore.runtime_patches import (
                patch_gpu_model_runner_dummy_run_real_reqs,
            )

            patch_gpu_model_runner_dummy_run_real_reqs()

            parallel_config = vllm_config.parallel_config
            cache_config = vllm_config.cache_config
            model_config = vllm_config.model_config

            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            if cache_config is not None and cache_config.block_size is None:
                cache_config.block_size = 16
            if model_config is not None:
                model_config.disable_cascade_attn = True

    def _validate_attention_backend(
        backend: AttentionBackendEnum,
        device_capability: DeviceCapability,
        config: "AttentionSelectorConfig",
    ) -> None:
        try:
            reasons = backend.get_class().validate_configuration(
                device_capability=device_capability,
                **config._asdict(),
            )
        except ImportError as exc:
            raise ValueError(f"Attention backend {backend.name} is not importable") from exc
        if reasons:
            raise ValueError(
                f"Attention backend {backend.name} is not valid: {reasons}"
            )

    return InfiniCorePlatform
