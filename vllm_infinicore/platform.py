"""InfiniCore platform plugin for vLLM.

An out-of-tree ``Platform`` implementation that replaces ``vllm_metax`` for
MACA device support.  vLLM discovers MACA hardware through this platform
without requiring the ``metax`` platform plugin.

Architecture
------------

The platform auto-detects hardware at import time:

* **MACA** (pymxsml available) → ``InfiniCoreMacaPlatform``
* **CUDA** (torch.cuda.is_available) → ``InfiniCoreCudaPlatform``
* **CPU / unknown** → ``InfiniCoreFallbackPlatform``

``register_platform()`` returns the class qualname for vLLM's
``resolve_current_platform_cls_qualname()``.
"""

from __future__ import annotations

import logging
import os
from functools import cache, wraps
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

import torch
from typing_extensions import ParamSpec

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser

# ---------------------------------------------------------------------------
# MXSML auto-detection
# ---------------------------------------------------------------------------

_pymxsml = None
_pymxsml_available = False

try:
    from vllm_metax.utils import import_pymxsml as _import_pymxsml

    _pymxsml = _import_pymxsml()
except Exception:
    pass

if _pymxsml is not None:
    try:
        _pymxsml.nvmlInit()
        _pymxsml_available = True
    except Exception:
        pass
    finally:
        if _pymxsml_available:
            try:
                _pymxsml.nvmlShutdown()
            except Exception:
                pass


def _with_mxsml(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        _pymxsml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            _pymxsml.nvmlShutdown()

    return wrapper


# ---------------------------------------------------------------------------
# Attention backend registration
# ---------------------------------------------------------------------------


@cache
def _backend_priorities(
    use_mla: bool,
    _device_capability: DeviceCapability,
) -> list[AttentionBackendEnum]:
    if use_mla:
        return [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.TRITON_MLA,
            AttentionBackendEnum.FLASHMLA_SPARSE,
        ]
    return [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.TREE_ATTN,
        AttentionBackendEnum.FLEX_ATTENTION,
    ]


def register_attention_backends() -> None:
    """Register fallback attention backends into vLLM's registry.

    FLASH_ATTN is auto-registered by vllm_metax; this function only
    registers the secondary backends (MLA, FlashInfer, Triton, etc.)
    that serve as fallback for configurations InfiniCore does not cover.
    """

    for backend, path in [
        (AttentionBackendEnum.FLASHMLA,
         "vllm_metax.v1.attention.backends.mla.flashmla.MacaFlashMLABackend"),
        (AttentionBackendEnum.FLASHMLA_SPARSE,
         "vllm_metax.v1.attention.backends.mla.flashmla_sparse.MacaFlashMLASparseBackend"),
        (AttentionBackendEnum.TRITON_MLA,
         "vllm_metax.v1.attention.backends.mla.triton_mla.MacaTritonMLABackend"),
        (AttentionBackendEnum.FLASHINFER,
         "vllm_metax.v1.attention.backends.flashinfer.MacaFlashInferBackend"),
        (AttentionBackendEnum.TRITON_ATTN,
         "vllm_metax.v1.attention.backends.triton_attn.MacaTritonAttentionBackend"),
        (AttentionBackendEnum.TREE_ATTN,
         "vllm_metax.v1.attention.backends.tree_attn.MacaTreeAttentionBackend"),
        (AttentionBackendEnum.FLEX_ATTENTION,
         "vllm_metax.v1.attention.backends.flex_attention.MacaFlexAttentionBackend"),
    ]:
        register_backend(backend, class_path=path)


# ---------------------------------------------------------------------------
# InfiniCore MACA platform
# ---------------------------------------------------------------------------


class _InfiniCoreMacaPlatform(Platform):
    """MACA platform using pymxsml (NVML-compatible) for device queries.

    Falls back to torch.cuda when pymxsml is unavailable.
    """

    _enum = PlatformEnum.OOT
    device_name: str = "maca"
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
        _ = torch.zeros(1, device=device)

    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        if _pymxsml_available:
            return cls._mxsml_device_capability(device_id)
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        if _pymxsml_available:
            return cls._mxsml_device_name(device_id)
        return "Device 4000"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        if _pymxsml_available:
            return cls._mxsml_device_memory(device_id)
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def device_count(cls) -> int:
        from vllm.utils.torch_utils import cuda_device_count_stateless
        return cuda_device_count_stateless()

    # ---- capability queries ----------------------------------------------

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return True

    @classmethod
    def is_sleep_mode_available(cls) -> bool:
        return True

    @classmethod
    def is_device_capability_family(cls, capability: int, device_id: int = 0) -> bool:
        return False

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        if torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("FP8 is not supported on GPUs")

    @classmethod
    def supports_fp8(cls) -> bool:
        return False

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

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        dst_cache[:, dst_block_indices] = src_cache[:, src_block_indices].to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        dst_cache[:, dst_block_indices] = src_cache[:, src_block_indices].cpu()

    # ---- distributed -----------------------------------------------------

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_metax.distributed.device_communicators.cuda_communicator.MacaCommunicator"

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
        for modname in ("mcoplib._C", "mcoplib._moe_C",
                         "vllm_metax._C", "vllm_metax._moe_C"):
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

        if attention_config := vllm_config.attention_config:
            attention_config.use_cudnn_prefill = False
            attention_config.use_trtllm_ragged_deepseek_prefill = False
            attention_config.use_trtllm_attention = False
            attention_config.disable_flashinfer_prefill = True

        if compilation_config is not None:
            compilation_config._attention_ops.append("vllm::mx_sparse_attn_indexer")

    # ---- MXSML helpers (private) -----------------------------------------

    @classmethod
    @_with_mxsml
    def _mxsml_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            pid = cls.device_id_to_physical_device_id(device_id)
            handle = _pymxsml.nvmlDeviceGetHandleByIndex(pid)
            major, minor = _pymxsml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @_with_mxsml
    def _mxsml_device_name(cls, device_id: int = 0) -> str:
        pid = cls.device_id_to_physical_device_id(device_id)
        handle = _pymxsml.nvmlDeviceGetHandleByIndex(pid)
        return _pymxsml.nvmlDeviceGetName(handle)

    @classmethod
    @_with_mxsml
    def _mxsml_device_memory(cls, device_id: int = 0) -> int:
        pid = cls.device_id_to_physical_device_id(device_id)
        handle = _pymxsml.nvmlDeviceGetHandleByIndex(pid)
        return int(_pymxsml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @_with_mxsml
    def get_device_uuid(cls, device_id: int = 0) -> str:
        pid = cls.device_id_to_physical_device_id(device_id)
        handle = _pymxsml.nvmlDeviceGetHandleByIndex(pid)
        return _pymxsml.nvmlDeviceGetUUID(handle)

    @classmethod
    @_with_mxsml
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        handles = [_pymxsml.nvmlDeviceGetHandleByIndex(i) for i in device_ids]
        for i, h1 in enumerate(handles):
            for j, h2 in enumerate(handles):
                if i < j:
                    try:
                        status = _pymxsml.nvmlDeviceGetP2PStatus(
                            h1, h2, _pymxsml.NVML_P2P_CAPS_INDEX_NVLINK)
                        if status != _pymxsml.NVML_P2P_STATUS_OK:
                            return False
                    except _pymxsml.NVMLError:
                        return False
        return True

    @classmethod
    @_with_mxsml
    def log_warnings(cls):
        count = _pymxsml.nvmlDeviceGetCount()
        if count > 1:
            names = []
            for i in range(count):
                h = _pymxsml.nvmlDeviceGetHandleByIndex(i)
                names.append(_pymxsml.nvmlDeviceGetName(h))
            if len(set(names)) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                logger.warning(
                    "Detected different devices: %s. "
                    "Set CUDA_DEVICE_ORDER=PCI_BUS_ID.", ", ".join(names))


# ---------------------------------------------------------------------------
# CUDA platform (non-MACA NVIDIA GPUs)
# ---------------------------------------------------------------------------


class _InfiniCoreCudaPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.cuda.set_device(device)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"


# ---------------------------------------------------------------------------
# Fallback platform
# ---------------------------------------------------------------------------


class _InfiniCoreFallbackPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return False


# ---------------------------------------------------------------------------
# Auto-detect and select
# ---------------------------------------------------------------------------


if _pymxsml_available:
    _InfiniCorePlatform: type[Platform] = _InfiniCoreMacaPlatform
    _InfiniCoreMacaPlatform.log_warnings()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "3600")

    try:
        import vllm.utils.import_utils as iu
        iu.has_triton_kernels = lambda: False
    except Exception:
        pass

    try:
        import torchvision
        torchvision.disable_beta_transforms_warning()
    except Exception:
        pass

elif torch.cuda.is_available():
    _InfiniCorePlatform = _InfiniCoreCudaPlatform
else:
    _InfiniCorePlatform = _InfiniCoreFallbackPlatform

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_platform() -> str:
    """Return the fully-qualified class name of the detected platform.

    Called by vLLM's ``platform_plugins`` entry-point machinery.
    """
    return f"{_InfiniCorePlatform.__module__}.{_InfiniCorePlatform.__qualname__}"


InfiniCorePlatform = _InfiniCorePlatform
