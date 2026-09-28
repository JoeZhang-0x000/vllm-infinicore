"""Shared InfiniCore Python helpers for CUDA-like operator implementations.

These helpers bridge torch tensors to the installed ``infinicore`` Python
package, whose public functional APIs call the underlying ``_infinicore``
extension. CPU tensors intentionally use PyTorch fallbacks because the local
InfiniCore build is device-oriented and can crash on CPU ``from_torch`` paths.
"""

from __future__ import annotations

from collections import OrderedDict
import ctypes
import logging
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

REAL_BACKEND_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_REAL_BACKEND"
STRICT_BACKEND_ENV = "VLLM_INFINICORE_STRICT_BACKEND"
logger = logging.getLogger(__name__)
_CALL_COUNTS: dict[str, int] = {}
_FUSED_ADD_RMS_NORM_SUPPORTED: bool | None = None
_DEFAULT_DEVICE_INDEX_SET: int | None = None
_PY_CAPSULE_GET_POINTER: Any | None = None
_INFINICORE_STREAM_PTRS: dict[tuple[str, int], int] = {}
_EXTERNAL_STREAMS: dict[tuple[str, int, int], Any] = {}
_INFINI_TENSOR_CACHE_MAX = 4096
_INFINI_TENSOR_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_ROPE_TABLE_CACHE_MAX = 16
_ROPE_TABLE_CACHE: OrderedDict[
    tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]
] = OrderedDict()


def rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _route_or_fallback(
        "rms_norm", input_tensor,
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
        "fused_add_rms_norm", input_tensor,
        lambda: _fused_add_rms_norm_infinicore(input_tensor, residual, weight, eps),
        lambda: _fused_add_rms_norm_torch(input_tensor, residual, weight, eps),
    )


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    return _route_or_fallback(
        "silu_and_mul", input_tensor,
        lambda: _silu_and_mul_infinicore(input_tensor),
        lambda: _silu_and_mul_torch(input_tensor),
    )


def linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _route_or_fallback(
        "linear", input_tensor,
        lambda: _linear_infinicore(input_tensor, weight, bias),
        lambda: F.linear(input_tensor, weight, bias),
    )


def lm_head(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _route_or_fallback(
        "lm_head", input_tensor,
        lambda: _lm_head_infinicore(input_tensor, weight, bias),
        lambda: F.linear(input_tensor, weight, bias),
    )


def embedding(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _route_or_fallback(
        "embedding", input_tensor,
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
        positions, query, key, head_size, rotary_dim, cos_sin_cache,
        is_neox_style,
    )
    return _route_or_fallback(
        "rotary_embedding", query,
        lambda: _rotary_embedding_infinicore(*args),
        lambda: _rotary_embedding_torch(*args),
    )


def real_backend_enabled(reference_tensor: torch.Tensor) -> bool:
    return _should_use_infinicore(reference_tensor)


def backend_call_counts() -> dict[str, int]:
    return dict(_CALL_COUNTS)


def reset_backend_call_counts() -> None:
    _CALL_COUNTS.clear()
    try:
        from . import cpp_bridge

        cpp_bridge.reset_bridge_call_counts()
    except Exception:
        pass


def clear_tensor_wrapper_cache() -> None:
    _INFINI_TENSOR_CACHE.clear()
    _ROPE_TABLE_CACHE.clear()


def clear_stream_cache() -> None:
    _INFINICORE_STREAM_PTRS.clear()
    _EXTERNAL_STREAMS.clear()


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
    from .selection import selected_backend

    return (
        selected_backend() in {"cuda", "metax", "kunlun"}
        and _is_accelerator_tensor(tensor)
        and not _env_truthy(REAL_BACKEND_DISABLE_ENV)
    )


def strict_backend_enabled() -> bool:
    return _env_truthy(STRICT_BACKEND_ENV)


def _is_non_fallback_error(exc: Exception) -> bool:
    try:
        from .cpp_bridge import CppBridgeError
    except Exception:
        return False
    return isinstance(exc, CppBridgeError)


def _as_infini(tensor: torch.Tensor) -> Any:
    import infinicore

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if _is_accelerator_tensor(tensor):
        wrapped = _as_infini_strided(tensor)
        wrapped._torch_ref = tensor
        return wrapped
    _set_infinicore_device(tensor)
    return infinicore.from_torch(tensor)


def _as_infini_cached(tensor: torch.Tensor) -> Any:
    if not tensor.is_contiguous():
        return _as_infini(tensor)
    return _cached_infini_tensor(("contiguous",) + _tensor_cache_key(tensor), tensor)


def _as_infini_contiguous_copy_cached(tensor: torch.Tensor) -> Any:
    if tensor.is_contiguous():
        return _as_infini_cached(tensor)
    cache_key = ("contiguous_copy",) + _tensor_cache_key(tensor)
    cached = _INFINI_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        _INFINI_TENSOR_CACHE.move_to_end(cache_key)
        return cached

    contiguous = tensor.contiguous()
    wrapped = _as_infini(contiguous)
    wrapped._torch_ref = contiguous
    _INFINI_TENSOR_CACHE[cache_key] = wrapped
    if len(_INFINI_TENSOR_CACHE) > _INFINI_TENSOR_CACHE_MAX:
        _INFINI_TENSOR_CACHE.popitem(last=False)
    return wrapped


def _as_infini_strided(tensor: torch.Tensor) -> Any:
    import infinicore
    from infinicore.tensor import to_infinicore_dtype

    device_index = tensor.device.index if tensor.device.index is not None else 0
    _set_infinicore_device(tensor)
    return infinicore.strided_from_blob(
        tensor.data_ptr(),
        list(tensor.shape),
        list(tensor.stride()),
        dtype=to_infinicore_dtype(tensor.dtype),
        device=infinicore.device(_torch_device_type(tensor), device_index),
    )


def _as_infini_strided_cached(tensor: torch.Tensor) -> Any:
    return _cached_infini_tensor(("strided",) + _tensor_cache_key(tensor), tensor)


def _cached_infini_tensor(cache_key: tuple[Any, ...], tensor: torch.Tensor) -> Any:
    cached = _INFINI_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        _INFINI_TENSOR_CACHE.move_to_end(cache_key)
        return cached

    wrapped = _as_infini_strided(tensor)
    # Keep the torch tensor/view alive for wrappers created from raw data_ptr.
    wrapped._torch_ref = tensor
    _INFINI_TENSOR_CACHE[cache_key] = wrapped
    if len(_INFINI_TENSOR_CACHE) > _INFINI_TENSOR_CACHE_MAX:
        _INFINI_TENSOR_CACHE.popitem(last=False)
    return wrapped


def _tensor_cache_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    device_index = tensor.device.index if tensor.device.index is not None else 0
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        tensor.device.type,
        device_index,
    )


def _contiguous_rope_table_cached(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor
    cache_key = _tensor_cache_key(tensor) + (tensor._version,)
    cached = _ROPE_TABLE_CACHE.get(cache_key)
    if cached is not None:
        _ROPE_TABLE_CACHE.move_to_end(cache_key)
        return cached[1]

    contiguous = tensor.contiguous()
    _ROPE_TABLE_CACHE[cache_key] = (tensor, contiguous)
    if len(_ROPE_TABLE_CACHE) > _ROPE_TABLE_CACHE_MAX:
        _ROPE_TABLE_CACHE.popitem(last=False)
    return contiguous


def _run_on_infinicore_stream(
    reference_tensor: torch.Tensor,
    launch: Callable[[], Any],
) -> Any:
    """Launch InfiniCore work on its stream while joining PyTorch stream order."""

    if not _is_accelerator_tensor(reference_tensor):
        return launch()

    stream = _infinicore_external_stream(reference_tensor)
    if stream is None:
        if _is_graph_capturing(reference_tensor):
            raise RuntimeError(
                "InfiniCore stream is unavailable during accelerator graph capture"
            )
        return launch()

    device_api = _torch_device_api(reference_tensor)
    if device_api is None:
        if _is_graph_capturing(reference_tensor):
            raise RuntimeError(
                "InfiniCore stream bridge is unavailable during accelerator graph capture"
            )
        return launch()

    original_stream = device_api.current_stream(reference_tensor.device)
    if (
        reference_tensor.device.type == "cuda"
        and original_stream.cuda_stream == stream.cuda_stream
    ):
        return launch()
    stream.wait_stream(original_stream)
    with device_api.stream(stream):
        result = launch()
    original_stream.wait_stream(stream)
    return result


def _infinicore_external_stream(
    reference_tensor: torch.Tensor,
) -> Any | None:
    device_api = _torch_device_api(reference_tensor)
    if device_api is None or not hasattr(device_api, "ExternalStream"):
        return None
    device_index = reference_tensor.device.index if reference_tensor.device.index is not None else 0
    device_key = (reference_tensor.device.type, device_index)
    ptr = _INFINICORE_STREAM_PTRS.get(device_key)
    if ptr is None:
        try:
            import infinicore

            with device_api.device(reference_tensor.device):
                _set_infinicore_device(reference_tensor)
                ptr = _capsule_pointer(infinicore.get_stream())
        except Exception:
            return None
        _INFINICORE_STREAM_PTRS[device_key] = ptr

    if not ptr:
        return None

    stream_key = device_key + (ptr,)
    stream = _EXTERNAL_STREAMS.get(stream_key)
    if stream is None:
        with device_api.device(reference_tensor.device):
            stream = device_api.ExternalStream(ptr)
        _EXTERNAL_STREAMS[stream_key] = stream
    return stream


def _capsule_pointer(capsule: Any) -> int:
    global _PY_CAPSULE_GET_POINTER

    if _PY_CAPSULE_GET_POINTER is None:
        getter = ctypes.pythonapi.PyCapsule_GetPointer
        getter.restype = ctypes.c_void_p
        getter.argtypes = [ctypes.py_object, ctypes.c_char_p]
        _PY_CAPSULE_GET_POINTER = getter
    return int(_PY_CAPSULE_GET_POINTER(capsule, None) or 0)


def _set_infinicore_device(tensor: torch.Tensor) -> None:
    if not _is_accelerator_tensor(tensor):
        return
    try:
        import infinicore

        device_index = tensor.device.index if tensor.device.index is not None else 0
        infinicore.set_device(
            infinicore.device(_torch_device_type(tensor), device_index)
        )
    except Exception:
        return


def _is_graph_capturing(reference_tensor: torch.Tensor) -> bool:
    if not _is_accelerator_tensor(reference_tensor):
        return False
    device_api = _torch_device_api(reference_tensor)
    if device_api is None or not hasattr(device_api, "is_current_stream_capturing"):
        return False
    try:
        return bool(device_api.is_current_stream_capturing())
    except Exception:
        return False


def _is_accelerator_tensor(tensor: torch.Tensor) -> bool:
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", "")
    return bool(getattr(tensor, "is_cuda", False)) or device_type == "cuda"


def _torch_device_api(tensor: torch.Tensor) -> Any | None:
    device_type = getattr(getattr(tensor, "device", None), "type", "")
    if bool(getattr(tensor, "is_cuda", False)) or device_type == "cuda":
        return getattr(torch, "cuda", None)
    return None


def _torch_device_type(tensor: torch.Tensor) -> str:
    # InfiniCore's Python tensor adaptor uses torch device names. The C++
    # bridge selects NVIDIA, METAX, or KUNLUN independently of this name.
    return getattr(getattr(tensor, "device", None), "type", "")


def _on_reference_device(tensor: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
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

    import infinicore.nn.functional as IF

    out = torch.empty_like(input_tensor)
    _run_on_infinicore_stream(
        input_tensor,
        lambda: IF.rms_norm(
            _as_infini(input_tensor),
            list(weight.shape),
            _as_infini(weight),
            float(eps),
            out=_as_infini(out),
        ),
    )
    return out


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
            _fused_add_rms_norm_stream(input_tensor, residual, weight, eps)
    except Exception as exc:
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


def _fused_add_rms_norm_stream(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from infinicore.ops.add_rms_norm import add_rms_norm as infini_add_rms_norm

    out = torch.empty_like(input_tensor)
    residual_out = torch.empty_like(input_tensor)
    _run_on_infinicore_stream(
        input_tensor,
        lambda: infini_add_rms_norm(
            _as_infini(input_tensor),
            _as_infini(residual),
            _as_infini(weight),
            float(eps),
            out=_as_infini(out),
            residual=_as_infini(residual_out),
        ),
    )
    return out, residual_out


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

    return _fused_add_rms_norm_stream(input_tensor, residual, weight, eps)


def _fused_add_rms_norm_torch(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged = input_tensor + residual
    return _rms_norm_torch(merged, weight, eps), merged


def _rms_norm_torch(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_float = input_tensor.float()
    variance = input_float.pow(2).mean(dim=-1, keepdim=True)
    output = input_float * torch.rsqrt(variance + float(eps))
    return output.to(dtype=input_tensor.dtype) * weight


def _silu_and_mul_infinicore(input_tensor: torch.Tensor) -> torch.Tensor:
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.SILU_AND_MUL_ROUTE):
        return _silu_and_mul_cpp_bridge(input_tensor)

    import infinicore.nn.functional as IF

    d = input_tensor.shape[-1] // 2
    output_shape = input_tensor.shape[:-1] + (input_tensor.shape[-1] // 2,)
    out = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    gate = input_tensor[..., :d].contiguous()
    up = input_tensor[..., d:].contiguous()
    # InfiniCore swiglu(a, b) computes a * silu(b).
    _run_on_infinicore_stream(
        input_tensor,
        lambda: IF.swiglu(_as_infini(up), _as_infini(gate), out=_as_infini(out)),
    )
    return out


def _silu_and_mul_cpp_bridge(input_tensor: torch.Tensor) -> torch.Tensor:
    from . import cpp_bridge

    module = cpp_bridge.module()
    input_arg = input_tensor if input_tensor.is_contiguous() else input_tensor.contiguous()
    result = module.silu_and_mul_current_stream(input_arg)
    cpp_bridge.record_call(cpp_bridge.SILU_AND_MUL_ROUTE)
    return result


def _silu_and_mul_torch(input_tensor: torch.Tensor) -> torch.Tensor:
    d = input_tensor.shape[-1] // 2
    return F.silu(input_tensor[..., :d]) * input_tensor[..., d:]


def _linear_infinicore(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from . import cpp_bridge

    if bias is None and cpp_bridge.enabled_for(cpp_bridge.MATMUL_ROUTE):
        return _linear_cpp_bridge(input_tensor, weight)

    import infinicore.nn.functional as IF

    out = torch.empty(
        input_tensor.shape[:-1] + (weight.shape[0],),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    _run_on_infinicore_stream(
        input_tensor,
        lambda: IF.linear(
            _as_infini(input_tensor),
            _as_infini(weight),
            None if bias is None else _as_infini(bias),
            out=_as_infini(out),
        ),
    )
    return out


def _linear_cpp_bridge(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from . import cpp_bridge

    module = cpp_bridge.module()
    result = module.linear_current_stream(input_tensor, weight, None)
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

    import infinicore.nn.functional as IF

    out = torch.empty(
        input_tensor.shape + (weight.shape[-1],),
        dtype=weight.dtype,
        device=weight.device,
    )
    _run_on_infinicore_stream(
        weight,
        lambda: IF.embedding(
            _as_infini(input_tensor.long()),
            _as_infini(weight),
            out=_as_infini(out),
        ),
    )
    return out


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

    import infinicore.nn.functional as IF

    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    positions = positions.flatten().to(torch.int32)
    max_position = int(cos_sin_cache.shape[0]) - 1
    if max_position >= 0:
        positions = positions.clamp(0, max_position)
    sin_infini = _as_infini_contiguous_copy_cached(sin)
    cos_infini = _as_infini_contiguous_copy_cached(cos)
    algo = IF.RopeAlgo.GPT_NEOX if is_neox_style else IF.RopeAlgo.GPT_J

    def apply_one(tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tensor.shape
        view = tensor.view(positions.shape[0], -1, head_size)
        rot = view[..., :rotary_dim]
        out_rot = torch.empty_like(rot)
        _run_on_infinicore_stream(
            tensor,
            lambda: IF.rope(
                _as_infini(rot),
                _as_infini(positions),
                sin_infini,
                cos_infini,
                algo,
                out=_as_infini(out_rot),
            ),
        )
        if rotary_dim == head_size:
            return out_rot.reshape(original_shape)
        out = torch.empty_like(view)
        out[..., :rotary_dim].copy_(out_rot)
        out[..., rotary_dim:].copy_(view[..., rotary_dim:])
        return out.reshape(original_shape)

    return apply_one(query), apply_one(key) if key is not None else None


def _rotary_embedding_cpp_bridge(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from . import cpp_bridge

    module = cpp_bridge.module()
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    positions = positions.flatten().to(torch.int32)
    max_position = int(cos_sin_cache.shape[0]) - 1
    if max_position >= 0:
        positions = positions.clamp(0, max_position)
    if cpp_bridge.bridge_target() == cpp_bridge.KUNLUN_TARGET:
        sin = _contiguous_rope_table_cached(sin)
        cos = _contiguous_rope_table_cached(cos)
    else:
        sin = sin.contiguous()
        cos = cos.contiguous()

    def apply_one(tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tensor.shape
        view = tensor.view(positions.shape[0], -1, head_size)
        result = module.rope_current_stream(view, positions, sin, cos, is_neox_style)
        cpp_bridge.record_call(cpp_bridge.ROPE_ROUTE)
        return result.reshape(original_shape)

    return apply_one(query), apply_one(key) if key is not None else None


def _rotary_embedding_torch(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    positions = positions.flatten()
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    def apply_one(tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tensor.shape
        view = tensor.view(positions.shape[0], -1, head_size)
        rot = view[..., :rotary_dim]
        passthrough = view[..., rotary_dim:]
        cos_view = cos.unsqueeze(-2).to(rot.dtype)
        sin_view = sin.unsqueeze(-2).to(rot.dtype)
        if is_neox_style:
            first, second = torch.chunk(rot, 2, dim=-1)
            out_rot = torch.cat(
                (first * cos_view - second * sin_view, second * cos_view + first * sin_view),
                dim=-1,
            )
        else:
            first = rot[..., ::2]
            second = rot[..., 1::2]
            out_rot = torch.stack(
                (first * cos_view - second * sin_view, second * cos_view + first * sin_view),
                dim=-1,
            ).flatten(-2)
        return torch.cat((out_rot, passthrough), dim=-1).reshape(original_shape)

    return apply_one(query), apply_one(key) if key is not None else None


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
