"""InfiniCore Python API adapters for torch-backed vLLM routes.

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
RAY_BACKEND_ENV = "VLLM_INFINICORE_RAY_BACKEND"
RAY_STORE_KV_CACHE_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_RAY_STORE_KV_CACHE"

logger = logging.getLogger(__name__)
_CALL_COUNTS: dict[str, int] = {}
_PY_CAPSULE_GET_POINTER: Any | None = None
_INFINICORE_STREAM_PTRS: dict[tuple[str, int], int] = {}
_EXTERNAL_STREAMS: dict[tuple[str, int, int], torch.cuda.ExternalStream] = {}
_INFINI_TENSOR_CACHE_MAX = 4096
_INFINI_TENSOR_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()


def rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _route_or_fallback(
        "rms_norm",
        input_tensor,
        lambda: _rms_norm_infinicore(input_tensor, weight, eps),
        lambda: _rms_norm_torch(input_tensor, weight, eps),
    )


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    return _route_or_fallback(
        "silu_and_mul",
        input_tensor,
        lambda: _silu_and_mul_infinicore(input_tensor),
        lambda: _silu_and_mul_torch(input_tensor),
    )


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
    return _route_or_fallback(
        "rotary_embedding",
        query,
        lambda: _rotary_embedding_infinicore(
            positions,
            query,
            key,
            head_size,
            rotary_dim,
            cos_sin_cache,
            is_neox_style,
        ),
        lambda: _rotary_embedding_torch(
            positions,
            query,
            key,
            head_size,
            rotary_dim,
            cos_sin_cache,
            is_neox_style,
        ),
    )


def store_kv_cache(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    _route_or_fallback(
        "store_kv_cache",
        key,
        lambda: _store_kv_cache_infinicore(kv_cache, key, value, slot_mapping),
        lambda: _store_kv_cache_torch(kv_cache, key, value, slot_mapping),
    )


def paged_attention_prefill(
    attn_layer: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    _route_or_fallback(
        "paged_attention_prefill",
        query,
        lambda: _paged_attention_prefill_infinicore(
            attn_layer.impl, query, key, kv_cache, attn_metadata, output
        ),
        lambda: attn_layer.impl.forward(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
        ),
    )


def paged_attention_decode(
    attn_layer: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    _route_or_fallback(
        "paged_attention_decode",
        query,
        lambda: _paged_attention_decode_infinicore(
            attn_layer.impl, query, key, kv_cache, attn_metadata, output
        ),
        lambda: attn_layer.impl.forward(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
        ),
    )


def real_backend_enabled(reference_tensor: torch.Tensor) -> bool:
    return _should_use_infinicore(reference_tensor)


def store_kv_cache_backend_enabled(reference_tensor: torch.Tensor) -> bool:
    if not _should_use_infinicore(reference_tensor):
        return False
    return not _env_truthy(RAY_STORE_KV_CACHE_DISABLE_ENV)


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


def clear_stream_cache() -> None:
    _INFINICORE_STREAM_PTRS.clear()
    _EXTERNAL_STREAMS.clear()


def paged_attention_prefill_infinicore_only(
    attn_impl: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    if not _should_use_infinicore(query):
        raise RuntimeError("InfiniCore paged attention prefill backend is disabled")
    _paged_attention_prefill_infinicore(
        attn_impl,
        query,
        key,
        kv_cache,
        attn_metadata,
        output,
    )
    _record_call("paged_attention_prefill")


def paged_attention_decode_infinicore_only(
    attn_impl: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    if not _should_use_infinicore(query):
        raise RuntimeError("InfiniCore paged attention decode backend is disabled")
    _paged_attention_decode_infinicore(
        attn_impl,
        query,
        key,
        kv_cache,
        attn_metadata,
        output,
    )
    _record_call("paged_attention_decode")


def paged_attention_decode_as_prefill_infinicore_only(
    attn_impl: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    if not _should_use_infinicore(query):
        raise RuntimeError("InfiniCore paged attention prefill backend is disabled")
    _paged_attention_decode_as_prefill_infinicore(
        attn_impl,
        query,
        key,
        kv_cache,
        attn_metadata,
        output,
    )
    _record_call("paged_attention_prefill")


def _route_or_fallback(
    op_name: str,
    reference_tensor: torch.Tensor,
    call_infinicore: Callable[[], Any],
    call_torch: Callable[[], Any],
) -> Any:
    if not _should_use_infinicore(reference_tensor):
        return call_torch()

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


def _record_call(op_name: str) -> None:
    _CALL_COUNTS[op_name] = _CALL_COUNTS.get(op_name, 0) + 1


def _should_use_infinicore(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and not _env_truthy(REAL_BACKEND_DISABLE_ENV)


def strict_backend_enabled() -> bool:
    return _env_truthy(STRICT_BACKEND_ENV)


def ray_backend_enabled() -> bool:
    return _env_truthy(RAY_BACKEND_ENV)


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
    if tensor.is_cuda:
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
        device=infinicore.device(tensor.device.type, device_index),
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


def _run_on_infinicore_stream(
    reference_tensor: torch.Tensor,
    launch: Callable[[], Any],
) -> Any:
    """Launch InfiniCore work on its stream while joining PyTorch stream order."""

    if not reference_tensor.is_cuda:
        return launch()

    stream = _infinicore_external_stream(reference_tensor)
    if stream is None:
        if _is_cuda_graph_capturing(reference_tensor):
            raise RuntimeError("InfiniCore stream is unavailable during CUDA graph capture")
        return launch()

    original_stream = torch.cuda.current_stream(reference_tensor.device)
    stream.wait_stream(original_stream)
    with torch.cuda.stream(stream):
        result = launch()
    original_stream.wait_stream(stream)
    return result


def _infinicore_external_stream(
    reference_tensor: torch.Tensor,
) -> torch.cuda.ExternalStream | None:
    device_index = reference_tensor.device.index if reference_tensor.device.index is not None else 0
    device_key = (reference_tensor.device.type, device_index)
    ptr = _INFINICORE_STREAM_PTRS.get(device_key)
    if ptr is None:
        try:
            import infinicore

            with torch.cuda.device(reference_tensor.device):
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
        with torch.cuda.device(reference_tensor.device):
            stream = torch.cuda.ExternalStream(ptr)
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
    if not tensor.is_cuda:
        return
    try:
        import infinicore

        device_index = tensor.device.index if tensor.device.index is not None else 0
        infinicore.set_device(infinicore.device(tensor.device.type, device_index))
    except Exception:
        return


def _is_cuda_graph_capturing(reference_tensor: torch.Tensor) -> bool:
    if not reference_tensor.is_cuda:
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _on_reference_device(tensor: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    if tensor is None or tensor.device == reference.device:
        return tensor
    return tensor.to(device=reference.device)


def _rms_norm_infinicore(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
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


def _silu_and_mul_torch(input_tensor: torch.Tensor) -> torch.Tensor:
    d = input_tensor.shape[-1] // 2
    return F.silu(input_tensor[..., :d]) * input_tensor[..., d:]


def _linear_infinicore(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
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


def _lm_head_infinicore(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.LM_HEAD_ROUTE):
        return _run_on_infinicore_stream(
            input_tensor,
            lambda: _lm_head_cpp_bridge(input_tensor, weight, bias),
        )
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


def _rotary_embedding_infinicore(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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


def _cache_views(kv_cache: torch.Tensor, key: torch.Tensor) -> tuple[Any, Any]:
    key_cache, value_cache = _split_kv_cache(kv_cache)
    num_kv_heads = key.shape[1]
    if key_cache.shape[1] == num_kv_heads:
        return _as_infini_strided_cached(key_cache), _as_infini_strided_cached(value_cache)
    if key_cache.shape[2] == num_kv_heads:
        return _as_infini_strided_cached(
            key_cache.permute(0, 2, 1, 3)
        ), _as_infini_strided_cached(value_cache.permute(0, 2, 1, 3))
    raise RuntimeError(f"cannot infer KV cache layout from key={key.shape}, cache={kv_cache.shape}")


def _cache_views_bshd(kv_cache: torch.Tensor, key: torch.Tensor) -> tuple[Any, Any]:
    key_cache, value_cache = _split_kv_cache(kv_cache)
    num_kv_heads = key.shape[1]
    if key_cache.shape[1] == num_kv_heads:
        return _as_infini_strided_cached(
            key_cache.permute(0, 2, 1, 3)
        ), _as_infini_strided_cached(value_cache.permute(0, 2, 1, 3))
    if key_cache.shape[2] == num_kv_heads:
        return _as_infini_strided_cached(key_cache), _as_infini_strided_cached(value_cache)
    raise RuntimeError(f"cannot infer KV cache layout from key={key.shape}, cache={kv_cache.shape}")


def _split_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(kv_cache, torch.Tensor):
        raise RuntimeError(f"expected tensor KV cache, got {type(kv_cache)!r}")
    if kv_cache.numel() == 0 or kv_cache.ndim < 5:
        raise RuntimeError(f"expected non-empty 5D KV cache, got {kv_cache.shape}")
    if kv_cache.shape[0] == 2:
        return kv_cache.unbind(0)
    if kv_cache.shape[1] == 2:
        return kv_cache.unbind(1)
    raise RuntimeError(f"cannot infer KV cache split axis from cache={kv_cache.shape}")


def _store_kv_cache_infinicore(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    import infinicore

    key_cache, value_cache = _cache_views(kv_cache, key)
    slot_mapping = _on_reference_device(slot_mapping, key)
    _run_on_infinicore_stream(
        key,
        lambda: infinicore.paged_caching(
            key_cache,
            value_cache,
            _as_infini(key),
            _as_infini(value),
            _as_infini(slot_mapping.flatten()),
        ),
    )


def _store_kv_cache_torch(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    key_cache, value_cache = _split_kv_cache(kv_cache)
    flat_slots = slot_mapping.flatten().long()
    num_kv_heads = key.shape[1]
    if key_cache.shape[1] == num_kv_heads:
        block_size = key_cache.shape[2]
        cache_layout = "hnd"
    else:
        block_size = key_cache.shape[1]
        cache_layout = "nhd"
    for token_idx, slot in enumerate(flat_slots.tolist()):
        block = slot // block_size
        offset = slot % block_size
        if cache_layout == "hnd":
            key_cache[block, :, offset, :].copy_(key[token_idx])
            value_cache[block, :, offset, :].copy_(value[token_idx])
        else:
            key_cache[block, offset].copy_(key[token_idx])
            value_cache[block, offset].copy_(value[token_idx])


def _prefill_total_lens(attn_metadata: Any) -> torch.Tensor:
    cu_prefix_kv_lens = getattr(attn_metadata, "cu_prefix_kv_lens", None)
    if cu_prefix_kv_lens is not None:
        return (cu_prefix_kv_lens[1:] - cu_prefix_kv_lens[:-1]).to(torch.int64)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None:
        raise RuntimeError("missing seq_lens for paged attention prefill")
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
    if seq_lens.shape[0] >= num_decodes + num_prefills:
        return seq_lens[num_decodes : num_decodes + num_prefills]
    if seq_lens.shape[0] == num_prefills:
        return seq_lens
    raise RuntimeError("cannot derive prefill total lengths")


def _paged_attention_prefill_infinicore(
    attn_impl: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    import infinicore

    key_cache, value_cache = _cache_views_bshd(kv_cache, key)
    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    q = query[num_decode_tokens:num_actual_tokens]
    if q.numel() == 0:
        return
    out = output[num_decode_tokens:num_actual_tokens].view(q.shape)
    total_lens = getattr(attn_metadata, "cu_prefix_kv_lens", None)
    if total_lens is None:
        total_lens = _prefill_total_lens(attn_metadata)
        total_lens = torch.nn.functional.pad(total_lens, (1, 0), value=0).cumsum(
            dim=0, dtype=torch.int32
        )
    total_lens = _on_reference_device(total_lens, query)
    query_start_loc = _on_reference_device(attn_metadata.prefill_query_start_loc, query)
    block_table = _on_reference_device(attn_metadata.prefill_block_table, query)
    alibi_slopes = _on_reference_device(attn_impl.alibi_slopes, query)
    max_query_len = int(getattr(attn_metadata, "max_query_len", q.shape[0]))
    max_seq_len = int(getattr(attn_metadata, "prefill_max_seq_len", max_query_len))
    _run_on_infinicore_stream(
        query,
        lambda: infinicore.mha_varlen(
            _as_infini(q),
            key_cache,
            value_cache,
            _as_infini_cached(query_start_loc),
            _as_infini_cached(total_lens),
            _as_infini_cached(block_table),
            max_query_len,
            max_seq_len,
            _as_infini_cached(alibi_slopes) if alibi_slopes is not None else None,
            attn_impl.scale,
            out=_as_infini(out),
        ),
    )


def _paged_attention_decode_infinicore(
    attn_impl: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    import infinicore

    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    if num_decode_tokens == 0:
        return
    if num_decode_tokens != num_decodes:
        raise RuntimeError("InfiniCore paged attention wrapper does not support speculative decode")
    from . import cpp_bridge

    if cpp_bridge.enabled_for(cpp_bridge.DECODE_ROUTE):
        _run_on_infinicore_stream(
            query,
            lambda: _paged_attention_decode_cpp_bridge(
                query,
                key,
                kv_cache,
                attn_metadata,
                output,
                attn_impl.alibi_slopes,
                attn_impl.scale,
                num_decode_tokens,
                num_decodes,
            ),
        )
        return
    key_cache, value_cache = _cache_views(kv_cache, key)
    q = query[:num_decode_tokens]
    out = output[:num_decode_tokens].view(q.shape)
    decode_block_table = _on_reference_device(attn_metadata.decode_block_table, query)
    decode_seq_lens = _on_reference_device(attn_metadata.decode_seq_lens, query)
    alibi_slopes = _on_reference_device(attn_impl.alibi_slopes, query)
    _run_on_infinicore_stream(
        query,
        lambda: infinicore.paged_attention(
            _as_infini(q),
            key_cache,
            value_cache,
            _as_infini_cached(decode_block_table),
            _as_infini_cached(decode_seq_lens),
            _as_infini_cached(alibi_slopes) if alibi_slopes is not None else None,
            attn_impl.scale,
            out=_as_infini(out),
        ),
    )


def _paged_attention_decode_cpp_bridge(
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
    scale: float,
    num_decode_tokens: int,
    num_decodes: int,
) -> None:
    from . import cpp_bridge

    module = cpp_bridge.module()
    module.paged_attention_decode_out(
        query,
        key,
        kv_cache,
        attn_metadata.decode_seq_lens,
        attn_metadata.decode_block_table,
        alibi_slopes,
        float(scale),
        int(num_decode_tokens),
        int(num_decodes),
        output,
    )
    cpp_bridge.record_call(cpp_bridge.DECODE_ROUTE)


def _paged_attention_decode_as_prefill_infinicore(
    attn_impl: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> None:
    import infinicore

    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    q = query[:num_actual_tokens]
    if q.numel() == 0:
        return
    key_cache, value_cache = _cache_views(kv_cache, key)
    query_start_loc = torch.tensor(
        [0, num_actual_tokens],
        dtype=torch.int32,
        device=query.device,
    )
    decode_block_table = _on_reference_device(attn_metadata.decode_block_table, query)
    decode_seq_lens = _on_reference_device(attn_metadata.decode_seq_lens, query)
    total_lens = decode_seq_lens[:1].to(torch.int64)
    out = output[:num_actual_tokens].view(q.shape)
    alibi_slopes = _on_reference_device(attn_impl.alibi_slopes, query)
    _run_on_infinicore_stream(
        query,
        lambda: infinicore.paged_attention_prefill(
            _as_infini(q),
            key_cache,
            value_cache,
            _as_infini_cached(decode_block_table[:1]),
            _as_infini_cached(total_lens),
            _as_infini_cached(query_start_loc),
            _as_infini_cached(alibi_slopes) if alibi_slopes is not None else None,
            attn_impl.scale,
            out=_as_infini(out),
        ),
    )


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
