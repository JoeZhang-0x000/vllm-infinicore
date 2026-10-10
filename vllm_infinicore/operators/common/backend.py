"""Shared InfiniCore Python helpers for CUDA-like operator implementations.

The modular stack uses the public InfiniOps C++ bridge on PyTorch's current
stream. Legacy installations can also use the ``infinicore`` Python package.
CPU tensors intentionally use PyTorch fallbacks.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from .devices import is_accelerator_tensor as _is_accelerator_tensor
from .devices import torch_device_api as _torch_device_api
from .torch_ops import fused_add_rms_norm as _fused_add_rms_norm_torch
from .torch_ops import rms_norm as _rms_norm_torch
from .torch_ops import rotary_embedding as _rotary_embedding_torch
from .torch_ops import silu_and_mul as _silu_and_mul_torch

REAL_BACKEND_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_REAL_BACKEND"
STRICT_BACKEND_ENV = "VLLM_INFINICORE_STRICT_BACKEND"
logger = logging.getLogger(__name__)
_CALL_COUNTS: dict[str, int] = {}
_FALLBACK_COUNTS: dict[str, int] = {}
_FALLBACK_REASONS: dict[str, str] = {}
_FUSED_ADD_RMS_NORM_SUPPORTED: bool | None = None
_DEFAULT_DEVICE_INDEX_SET: int | None = None


def rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _route_or_fallback(
        "rms_norm",
        input_tensor,
        lambda: _rms_norm_infinicore(input_tensor, weight, eps),
        lambda: _rms_norm_torch(input_tensor, weight, eps),
    )


def fused_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _should_use_infinicore(input_tensor) and not fused_add_rms_norm_supported(
        input_tensor, residual, weight, eps
    ):
        return _fused_add_rms_norm_torch(input_tensor, residual, weight, eps)
    return _route_or_fallback(
        "fused_add_rms_norm",
        input_tensor,
        lambda: _fused_add_rms_norm_infinicore(input_tensor, residual, weight, eps),
        lambda: _fused_add_rms_norm_torch(input_tensor, residual, weight, eps),
    )


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    return _route_or_fallback(
        "silu_and_mul",
        input_tensor,
        lambda: _silu_and_mul_infinicore(input_tensor),
        lambda: _silu_and_mul_torch(input_tensor),
    )


def fused_add_rms_norm_inplace(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """vLLM's mutable residual contract, using the modular InfiniOps kernel."""
    from . import cpp_bridge

    def run():
        if not cpp_bridge.uses_modular_api() or not cpp_bridge.enabled_for(
            cpp_bridge.RMS_NORM_ROUTE
        ):
            raise cpp_bridge.CppBridgeError("In-place fused RMSNorm requires its modular C++ route")
        cpp_bridge.module().add_rms_norm_inplace_current_stream(
            input_tensor, residual, weight, float(eps)
        )
        cpp_bridge.record_call(cpp_bridge.RMS_NORM_ROUTE)

    def cpu_or_fallback():
        out, merged = _fused_add_rms_norm_torch(input_tensor, residual, weight, eps)
        input_tensor.copy_(out)
        residual.copy_(merged)

    _route_or_fallback("fused_add_rms_norm", input_tensor, run, cpu_or_fallback)


def linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _route_or_fallback(
        "linear",
        input_tensor,
        lambda: _linear_infinicore(input_tensor, weight, bias),
        lambda: F.linear(input_tensor, weight, bias),
    )


def lm_head(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _route_or_fallback(
        "lm_head",
        input_tensor,
        lambda: _lm_head_infinicore(input_tensor, weight, bias),
        lambda: F.linear(input_tensor, weight, bias),
    )


def embedding(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _route_or_fallback(
        "embedding",
        input_tensor,
        lambda: _embedding_infinicore(input_tensor, weight),
        lambda: F.embedding(input_tensor.long(), weight),
    )


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    args = (
        positions,
        query,
        key,
        head_size,
        rotary_dim,
        cos_sin_cache,
        is_neox_style,
    )
    return _route_or_fallback(
        "rotary_embedding",
        query,
        lambda: _rotary_embedding_infinicore(*args),
        lambda: _rotary_embedding_torch(*args),
    )


def real_backend_enabled(reference_tensor: torch.Tensor) -> bool:
    return _should_use_infinicore(reference_tensor)


def rotary_embedding_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> None:
    """Mutate the Q/K views supplied by vLLM; do not copy their storage."""
    from . import cpp_bridge

    def run():
        if not cpp_bridge.uses_modular_api() or not cpp_bridge.enabled_for(cpp_bridge.ROPE_ROUTE):
            raise cpp_bridge.CppBridgeError("In-place RoPE requires its modular C++ route")
        if cos_sin_cache.shape[-1] != rotary_dim:
            raise ValueError("RoPE cache dimension differs from rotary dimension")
        cpp_bridge.module().rotary_embedding_inplace_current_stream(
            positions, query, key, int(head_size), cos_sin_cache, bool(is_neox_style)
        )
        cpp_bridge.record_call(cpp_bridge.ROPE_ROUTE)

    def cpu_or_fallback():
        q_out, k_out = _rotary_embedding_torch(
            positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style
        )
        query.copy_(q_out)
        if key is not None:
            key.copy_(k_out)

    _route_or_fallback("rotary_embedding", query, run, cpu_or_fallback)


def backend_call_counts() -> dict[str, int]:
    return dict(_CALL_COUNTS)


def backend_fallback_counts() -> dict[str, int]:
    return dict(_FALLBACK_COUNTS)


def backend_fallback_reasons() -> dict[str, str]:
    return dict(_FALLBACK_REASONS)


def reset_backend_call_counts() -> None:
    from . import cpp_bridge

    _CALL_COUNTS.clear()
    _FALLBACK_COUNTS.clear()
    _FALLBACK_REASONS.clear()
    cpp_bridge.reset_bridge_call_counts()


def clear_tensor_wrapper_cache() -> None:
    from . import legacy

    legacy.clear_tensor_wrapper_cache()


def clear_stream_cache() -> None:
    from . import legacy

    legacy.clear_stream_cache()


def _route_or_fallback(
    op_name: str,
    reference_tensor: torch.Tensor,
    call_infinicore: Callable[[], Any],
    call_torch: Callable[[], Any],
) -> Any:
    if not _should_use_infinicore(reference_tensor):
        return call_torch()

    _set_default_device_index(reference_tensor)

    # InfiniCore resets the accelerator's current device to 0 while it
    # dispatches. On a tensor-parallel rank above 0 that leaves every later
    # launch pointed at the wrong device, and MACA reports it as "Pointer
    # argument (at 0) cannot be accessed from Triton" out of vLLM's Triton
    # sampler. A rank already on device 0 has nothing to restore, and
    # ``index`` is None only for a device that carries no index at all, so
    # both cases correctly skip the guard.
    device_index = reference_tensor.device.index
    device_api = _torch_device_api(reference_tensor) if device_index else None
    try:
        result = call_infinicore()
        _record_call(op_name)
        return result
    except Exception as exc:
        if _is_non_fallback_error(exc):
            raise
        if strict_backend_enabled():
            raise RuntimeError(f"InfiniCore {op_name} failed") from exc
        logger.warning(
            "InfiniCore %s failed; falling back to PyTorch/vLLM native path: %s",
            op_name,
            exc,
        )
        return call_torch()
    finally:
        if device_api is not None and device_api.current_device() != device_index:
            device_api.set_device(device_index)


def _record_call(op_name: str) -> None:
    _CALL_COUNTS[op_name] = _CALL_COUNTS.get(op_name, 0) + 1


def _set_default_device_index(tensor: torch.Tensor) -> None:
    """Point InfiniCore's lazy runtime at the vendor worker's selected card."""

    global _DEFAULT_DEVICE_INDEX_SET
    index = tensor.device.index
    if index is not None and index != _DEFAULT_DEVICE_INDEX_SET:
        os.environ["INFINICORE_DEFAULT_DEVICE_INDEX"] = str(index)
        _DEFAULT_DEVICE_INDEX_SET = index


def _should_use_infinicore(tensor: torch.Tensor) -> bool:
    from ..selection import selected_backend

    return (
        selected_backend() in {"cuda", "metax", "kunlun"}
        and _is_accelerator_tensor(tensor)
        and not _env_truthy(REAL_BACKEND_DISABLE_ENV)
    )


def strict_backend_enabled() -> bool:
    return _env_truthy(STRICT_BACKEND_ENV)


def _is_non_fallback_error(exc: Exception) -> bool:
    from .cpp_bridge import CppBridgeError

    return isinstance(exc, CppBridgeError)


def _on_reference_device(
    tensor: torch.Tensor | None, reference: torch.Tensor
) -> torch.Tensor | None:
    if tensor is None or tensor.device == reference.device:
        return tensor
    return tensor.to(device=reference.device)


def _rms_norm_infinicore(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.RMS_NORM_ROUTE):
        return _rms_norm_cpp_bridge(input_tensor, weight, eps)

    from . import legacy

    return legacy.rms_norm(input_tensor, weight, eps)


def _rms_norm_cpp_bridge(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from . import cpp_bridge

    module = cpp_bridge.module()
    result = module.rms_norm_current_stream(input_tensor, weight, float(eps))
    cpp_bridge.record_call(cpp_bridge.RMS_NORM_ROUTE)
    return result


def fused_add_rms_norm_supported(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> bool:
    """Whether this device has a fused add + RMSNorm kernel, probed once.

    ``infiniopAddRMSNorm`` is not registered for every backend that registers
    plain ``infiniopRMSNorm``. A device with no kernel is a capability fact,
    so residual calls use the native fallback for the life of the process.

    Probed from inside the custom op, where the tensors are real. Probing from
    ``_should_use_infinicore`` would run under torch.compile tracing on fake
    tensors, and the traced branch is baked in before any device call happens.
    """

    global _FUSED_ADD_RMS_NORM_SUPPORTED

    if _FUSED_ADD_RMS_NORM_SUPPORTED is not None:
        return _FUSED_ADD_RMS_NORM_SUPPORTED

    from . import cpp_bridge

    supported = True
    try:
        if cpp_bridge.enabled_for(cpp_bridge.RMS_NORM_ROUTE):
            supported = bool(
                cpp_bridge.module().add_rms_norm_supported(
                    input_tensor, residual, weight, float(eps)
                )
            )
        else:
            from . import legacy

            legacy.fused_add_rms_norm(input_tensor, residual, weight, eps)
    except Exception as exc:
        if cpp_bridge.uses_modular_api():
            raise
        # The InfiniCore stream path reports no status code, so treat any probe
        # failure there as "unsupported" and keep the run alive on the fallback.
        supported = False
        logger.warning(
            "InfiniCore fused add+RMSNorm probe failed; using the native "
            "residual path for this process: %s",
            exc,
        )

    if not supported:
        logger.warning(
            "InfiniCore fused add+RMSNorm is unavailable on this device; "
            "the RMSNorm route keeps its non-residual path and falls back "
            "for residual calls."
        )
    _FUSED_ADD_RMS_NORM_SUPPORTED = supported
    return supported


def reset_fused_add_rms_norm_support() -> None:
    """Forget the probed capability."""

    global _FUSED_ADD_RMS_NORM_SUPPORTED

    _FUSED_ADD_RMS_NORM_SUPPORTED = None


def _fused_add_rms_norm_infinicore(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.RMS_NORM_ROUTE):
        module = cpp_bridge.module()
        out, residual_out = module.add_rms_norm_current_stream(
            input_tensor, residual, weight, float(eps)
        )
        cpp_bridge.record_call(cpp_bridge.RMS_NORM_ROUTE)
        return out, residual_out

    from . import legacy

    return legacy.fused_add_rms_norm(input_tensor, residual, weight, eps)


def _silu_and_mul_infinicore(input_tensor: torch.Tensor) -> torch.Tensor:
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.SILU_AND_MUL_ROUTE):
        return _silu_and_mul_cpp_bridge(input_tensor)

    from . import legacy

    return legacy.silu_and_mul(input_tensor)


def _silu_and_mul_cpp_bridge(input_tensor: torch.Tensor) -> torch.Tensor:
    from . import cpp_bridge

    module = cpp_bridge.module()
    input_arg = input_tensor if input_tensor.is_contiguous() else input_tensor.contiguous()
    result = module.silu_and_mul_current_stream(input_arg)
    cpp_bridge.record_call(cpp_bridge.SILU_AND_MUL_ROUTE)
    return result


def _linear_infinicore(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.MATMUL_ROUTE) and (
        bias is None or cpp_bridge.uses_modular_api()
    ):
        return _linear_cpp_bridge(input_tensor, weight, bias)

    from . import legacy

    return legacy.linear(input_tensor, weight, bias)


def _linear_cpp_bridge(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from . import cpp_bridge

    module = cpp_bridge.module()
    result = module.linear_current_stream(input_tensor, weight, bias)
    cpp_bridge.record_call(cpp_bridge.MATMUL_ROUTE)
    return result


def _lm_head_infinicore(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.LM_HEAD_ROUTE):
        return _lm_head_cpp_bridge(input_tensor, weight, bias)
    return _linear_infinicore(input_tensor, weight, bias)


def _lm_head_cpp_bridge(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from . import cpp_bridge

    module = cpp_bridge.module()
    result = module.lm_head(input_tensor, weight, bias)
    cpp_bridge.record_call(cpp_bridge.LM_HEAD_ROUTE)
    return result


def _embedding_infinicore(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.EMBEDDING_ROUTE):
        return _embedding_cpp_bridge(input_tensor, weight)

    from . import legacy

    return legacy.embedding(input_tensor, weight)


def _embedding_cpp_bridge(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    from . import cpp_bridge

    module = cpp_bridge.module()
    input_tensor = _on_reference_device(input_tensor.long(), weight)
    result = module.embedding_current_stream(input_tensor, weight)
    cpp_bridge.record_call(cpp_bridge.EMBEDDING_ROUTE)
    return result


def _rotary_embedding_infinicore(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from . import cpp_bridge

    if cpp_bridge.uses_modular_api() and cpp_bridge.enabled_for(cpp_bridge.ROPE_ROUTE):
        if rotary_dim != cos_sin_cache.shape[-1] or rotary_dim > head_size:
            raise ValueError("RoPE rotary_dim must match the cache width and fit head_size")
        positions = _on_reference_device(positions.flatten().to(torch.int64).contiguous(), query)
        cache = _on_reference_device(cos_sin_cache, query)
        q_out, k_out = cpp_bridge.module().rotary_embedding_current_stream(
            positions, query, key, head_size, cache, is_neox_style
        )
        cpp_bridge.record_call(cpp_bridge.ROPE_ROUTE)
        return q_out, k_out if key is not None else None

    if (
        rotary_dim == head_size
        and cos_sin_cache.shape[-1] == head_size
        and cpp_bridge.enabled_for(cpp_bridge.ROPE_ROUTE)
    ):
        return _rotary_embedding_cpp_bridge(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
        )

    from . import legacy

    return legacy.rotary_embedding(
        positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox_style
    )


def _rotary_embedding_cpp_bridge(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from . import legacy

    return legacy.rotary_embedding_cpp_bridge(
        positions, query, key, head_size, cos_sin_cache, is_neox_style
    )


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
