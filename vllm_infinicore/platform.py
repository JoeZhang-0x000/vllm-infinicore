"""InfiniCore platform plugin for vLLM.

A single unified out-of-tree ``Platform`` implementation that delegates all
device queries to ``torch.cuda``.  No dependency on ``vllm_metax``.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING, Optional

import torch

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser

# ---------------------------------------------------------------------------
# Attention backend registration
# ---------------------------------------------------------------------------


def register_attention_backends() -> None:
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_infinicore.ops.infinicore_attention.InfiniCoreFlashAttentionBackend",
    )


@cache
def _backend_priorities(
    use_mla: bool,
    _device_capability: DeviceCapability,
) -> list[AttentionBackendEnum]:
    return [AttentionBackendEnum.FLASH_ATTN]


# ---------------------------------------------------------------------------
# InfiniCore platform
# ---------------------------------------------------------------------------


class InfiniCorePlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "infinicore"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    supported_quantization: list[str] = [
        "awq", "gptq", "compressed-tensors", "compressed_tensors",
        "moe_wna16", "gguf",
    ]

    # ---- device basics ---------------------------------------------------

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
            return "InfiniCore Device"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def device_count(cls) -> int:
        from vllm.utils.torch_utils import cuda_device_count_stateless
        return cuda_device_count_stateless()

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"infinicore-{device_id}"

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        return True

    @classmethod
    def log_warnings(cls) -> None:
        pass

    # ---- capability queries ----------------------------------------------

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return True

    @classmethod
    def supports_fp8(cls) -> bool:
        return False

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype) -> None:
        if torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("FP8 is not supported on GPUs")

    # ---- attention -------------------------------------------------------

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
        assert device_capability is not None
        config = attn_selector_config._replace(block_size=None)

        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                reasons = backend_class.validate_configuration(
                    device_capability=device_capability, **config._asdict())
            except ImportError:
                reasons = ["ImportError"]
            if reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid: {reasons}")
            logger.info("Using %s backend.", selected_backend)
            return selected_backend.get_path()

        valid, _invalid = cls._valid_backends(device_capability, config)
        if not valid:
            raise ValueError(f"No valid attention backend found for {cls.device_name}.")
        selected = sorted(valid, key=lambda x: x[1])[0][0]
        logger.info_once("Using %s attention backend.", selected.name)
        return selected.get_path()

    @classmethod
    def _valid_backends(
        cls,
        device_capability: DeviceCapability,
        config: "AttentionSelectorConfig",
    ) -> tuple[list[tuple[AttentionBackendEnum, int]], dict[AttentionBackendEnum, list[str]]]:
        valid: list[tuple[AttentionBackendEnum, int]] = []
        invalid: dict[AttentionBackendEnum, list[str]] = {}
        for priority, backend in enumerate(
            _backend_priorities(config.use_mla, device_capability)
        ):
            try:
                reasons = backend.get_class().validate_configuration(
                    device_capability=device_capability, **config._asdict())
            except ImportError:
                reasons = ["ImportError"]
            if reasons:
                invalid[backend] = reasons
            else:
                valid.append((backend, priority))
        return valid, invalid

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list[AttentionBackendEnum]:
        return [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: Optional[AttentionBackendEnum] = None,
    ) -> AttentionBackendEnum:
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends()
            return backend
        be_cls = AttentionBackendEnum.FLASH_ATTN.get_class()
        if be_cls.supports_head_size(head_size) and be_cls.supports_dtype(dtype):
            return AttentionBackendEnum.FLASH_ATTN
        return AttentionBackendEnum.TORCH_SDPA

    # ---- graph & compilation ---------------------------------------------

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    # ---- memory & kv cache -----------------------------------------------

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    # ---- distributed -----------------------------------------------------

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return False

    # ---- lora ------------------------------------------------------------

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    # ---- initialisation --------------------------------------------------

    @classmethod
    def import_kernels(cls) -> None:
        for modname in ("mcoplib._C", "mcoplib._moe_C"):
            try:
                __import__(modname)
            except ImportError:
                pass

    @classmethod
    def pre_register_and_update(
        cls, parser: "FlexibleArgumentParser | None" = None
    ) -> None:
        cls.import_kernels()
        register_attention_backends()

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        if model_config is not None:
            model_config.disable_cascade_attn = True

        if compilation_config is not None:
            compilation_config._attention_ops.append("vllm::mx_sparse_attn_indexer")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_platform() -> str:
    return "vllm_infinicore.platform.InfiniCorePlatform"


InfiniCorePlatform = InfiniCorePlatform
