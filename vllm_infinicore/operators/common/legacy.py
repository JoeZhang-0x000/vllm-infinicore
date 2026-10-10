"""Compatibility adapters for the pre-modular InfiniCore Python API.

Keep tensor wrappers and stream synchronization out of the modern C++ path.
"""

from __future__ import annotations

import ctypes
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch

from .devices import is_accelerator_tensor as _is_accelerator_tensor
from .devices import torch_device_api as _torch_device_api

_PY_CAPSULE_GET_POINTER: Any | None = None
_INFINICORE_STREAM_PTRS: dict[tuple[str, int], int] = {}
_EXTERNAL_STREAMS: dict[tuple[str, int, int], Any] = {}
_INFINI_TENSOR_CACHE_MAX = 4096
_INFINI_TENSOR_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_ROPE_TABLE_CACHE_MAX = 16
_ROPE_TABLE_CACHE: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()


def clear_tensor_wrapper_cache() -> None:
    _INFINI_TENSOR_CACHE.clear()
    _ROPE_TABLE_CACHE.clear()


def clear_stream_cache() -> None:
    _INFINICORE_STREAM_PTRS.clear()
    _EXTERNAL_STREAMS.clear()


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
            raise RuntimeError("InfiniCore stream is unavailable during accelerator graph capture")
        return launch()

    device_api = _torch_device_api(reference_tensor)
    if device_api is None:
        if _is_graph_capturing(reference_tensor):
            raise RuntimeError(
                "InfiniCore stream bridge is unavailable during accelerator graph capture"
            )
        return launch()

    original_stream = device_api.current_stream(reference_tensor.device)
    if reference_tensor.device.type == "cuda" and original_stream.cuda_stream == stream.cuda_stream:
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
        infinicore.set_device(infinicore.device(_torch_device_type(tensor), device_index))
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


def _torch_device_type(tensor: torch.Tensor) -> str:
    # InfiniCore's Python tensor adaptor uses torch device names. The C++
    # bridge selects NVIDIA, METAX, or KUNLUN independently of this name.
    return getattr(getattr(tensor, "device", None), "type", "")


def rms_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from . import cpp_bridge

    cpp_bridge.require_legacy_python_api(cpp_bridge.RMS_NORM_ROUTE)
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


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    from . import cpp_bridge

    cpp_bridge.require_legacy_python_api(cpp_bridge.SILU_AND_MUL_ROUTE)
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


def linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from . import cpp_bridge

    cpp_bridge.require_legacy_python_api(cpp_bridge.MATMUL_ROUTE)
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


def embedding(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    from . import cpp_bridge

    cpp_bridge.require_legacy_python_api(cpp_bridge.EMBEDDING_ROUTE)
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


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from . import cpp_bridge

    cpp_bridge.require_legacy_python_api(cpp_bridge.ROPE_ROUTE)
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


def fused_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from . import cpp_bridge

    cpp_bridge.require_legacy_python_api(cpp_bridge.RMS_NORM_ROUTE)
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


@lru_cache(maxsize=8)
def _native_rope_positions_supported(module: Any) -> bool:
    capability = getattr(module, "rope_supports_native_positions", None)
    return capability is not None and bool(capability())


def rotary_embedding_cpp_bridge(
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
    kunlun = cpp_bridge.bridge_target() == cpp_bridge.KUNLUN_TARGET
    positions = positions.flatten()
    if not (
        kunlun
        and positions.dtype in (torch.int32, torch.int64)
        and _native_rope_positions_supported(module)
    ):
        positions = positions.to(torch.int32)
        max_position = int(cos_sin_cache.shape[0]) - 1
        if max_position >= 0:
            positions = positions.clamp(0, max_position)
    if kunlun:
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
