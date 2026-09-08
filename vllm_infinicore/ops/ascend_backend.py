"""Pinned InfiniCore C API on torch's current NPU stream (eager only).

Descriptors are cached per device/stream. Tensor storage is owned by torch;
record_stream protects every raw pointer until the external launch completes.
No device, worker, communication or KV-cache runtime is implemented here.
"""

from __future__ import annotations

from collections import OrderedDict
import ctypes as C
from functools import lru_cache
import json
import os
from pathlib import Path
import threading

import torch

LIBRARY_ENV = "VLLM_INFINICORE_ASCEND_LIBRARY"
_LOCK = Path(__file__).resolve().parents[1] / "infinicore.lock.json"
_DTYPES = {
    torch.float16: 12,
    torch.float32: 13,
    torch.bfloat16: 19,
    torch.int32: 5,
    torch.int64: 6,
}
_LOCAL = threading.local()


class Unsupported(RuntimeError):
    """A known unsupported case detected before any InfiniCore launch."""


def _check(status, operation, *, creating=False):
    if status:
        message = f"InfiniCore Ascend {operation}: status {status}"
        if creating and status in {2, 5, 8, 10, 11, 12}:
            raise Unsupported(message)
        raise RuntimeError(message)


@lru_cache(maxsize=1)
def library():
    path = os.environ.get(LIBRARY_ENV)
    if not path:
        raise Unsupported(f"{LIBRARY_ENV} is unset")
    lib = C.CDLL(path)
    lock = json.loads(_LOCK.read_text())
    lib.vllmInfinicoreRevision.restype = C.c_char_p
    lib.vllmInfinicoreBridgeABI.restype = C.c_int
    revision = lib.vllmInfinicoreRevision().decode()
    if (
        revision != lock["revision"]
        or lib.vllmInfinicoreBridgeABI() != lock["ascend_bridge_abi"]
    ):
        raise RuntimeError(f"InfiniCore Ascend library does not match lock: {revision}")
    P, S, I = C.c_void_p, C.c_size_t, C.c_int
    signatures = {
        "vllmInfinicoreCreateAscendHandle": [C.POINTER(P), I],
        "vllmInfinicoreDestroyAscendHandle": [P],
        "vllmInfinicoreDestroyEmbeddingDescriptor": [P],
        "infiniopCreateTensorDescriptor": [
            C.POINTER(P),
            S,
            C.POINTER(S),
            C.POINTER(C.c_ssize_t),
            I,
        ],
        "infiniopDestroyTensorDescriptor": [P],
    }
    for op, count in (
        ("RMSNorm", 3),
        ("SwiGLU", 3),
        ("Gemm", 3),
        ("Embedding", 3),
        ("RoPE", 5),
    ):
        extra = [C.c_float] if op == "RMSNorm" else [I] if op == "RoPE" else []
        signatures[f"infiniopCreate{op}Descriptor"] = (
            [P, C.POINTER(P)] + [P] * count + extra
        )
        signatures[f"infiniopDestroy{op}Descriptor"] = [P]
        if op != "Embedding":
            signatures[f"infiniopGet{op}WorkspaceSize"] = [P, C.POINTER(S)]
        signatures[f"infiniop{op}"] = (
            [P]
            + ([] if op == "Embedding" else [P, S])
            + [P] * count
            + ([C.c_float, C.c_float] if op == "Gemm" else [])
            + [P]
        )
    for name, args in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = None if name == "vllmInfinicoreDestroyAscendHandle" else I
    return lib


def fallback(name, reason, native):
    # Trace the original tensor program without Python counter side effects.
    # These calls are not runtime InfiniCore launches and must not be counted.
    if torch.compiler.is_compiling():
        return native()
    from . import infinicore_backend as counters

    counters._FALLBACK_COUNTS[name] = counters._FALLBACK_COUNTS.get(name, 0) + 1
    counters._FALLBACK_REASONS[name] = reason
    return native()


def execute(name, tensor, operation, native):
    # Raising/converting Unsupported inside Dynamo fullgraph tracing breaks
    # compilation. Keep the eager-only adapter out of the compiled program.
    if torch.compiler.is_compiling():
        return native()
    if tensor.device.type != "npu":
        return native()
    if os.environ.get("VLLM_INFINICORE_DISABLE_REAL_BACKEND") == "1":
        return fallback(name, "real backend explicitly disabled", native)
    try:
        with torch.npu.device(tensor.device):
            if torch.npu.is_current_stream_capturing():
                raise Unsupported("Ascend graph capture has not been validated")
            result = operation()
    except Unsupported as exc:
        return fallback(name, str(exc), native)
    # Never retry a failed device launch: a runtime failure is not a capability miss.
    from . import infinicore_backend as counters

    counters._CALL_COUNTS[name] = counters._CALL_COUNTS.get(name, 0) + 1
    return result


def nd(tensor):
    import torch_npu

    if tensor.device.type != "npu" or tensor.dtype not in _DTYPES:
        raise Unsupported(
            f"unsupported tensor device/dtype: {tensor.device}/{tensor.dtype}"
        )
    if not tensor.numel():
        raise Unsupported("empty tensor")
    if torch_npu.get_npu_format(tensor) != 2:
        tensor = torch_npu.npu_format_cast(tensor, 2)
    return tensor


class _Descriptor:
    def __init__(self, lib, op, tensors, scalar, stream):
        self.lib, self.op, self.stream = lib, op, stream
        self.ptr = C.c_void_p()
        self.handle = C.c_void_p()
        self.workspace_size = C.c_size_t()
        _check(
            lib.vllmInfinicoreCreateAscendHandle(
                C.byref(self.handle), tensors[0].device.index
            ),
            "handle",
        )
        descs = []
        try:
            for tensor in tensors:
                ptr = C.c_void_p()
                shape = (C.c_size_t * tensor.ndim)(*tensor.shape)
                stride = (C.c_ssize_t * tensor.ndim)(*tensor.stride())
                _check(
                    lib.infiniopCreateTensorDescriptor(
                        C.byref(ptr), tensor.ndim, shape, stride, _DTYPES[tensor.dtype]
                    ),
                    "tensor",
                    creating=True,
                )
                descs.append(ptr)
            _check(
                getattr(lib, f"infiniopCreate{op}Descriptor")(
                    self.handle, C.byref(self.ptr), *descs, *scalar
                ),
                op,
                creating=True,
            )
            if op != "Embedding":
                _check(
                    getattr(lib, f"infiniopGet{op}WorkspaceSize")(
                        self.ptr, C.byref(self.workspace_size)
                    ),
                    "workspace",
                )
        except BaseException:
            self.close()
            raise
        finally:
            for ptr in descs:
                lib.infiniopDestroyTensorDescriptor(ptr)

    def close(self):
        # Embedding owns an ACL workspace; executors also must outlive launches.
        with torch.npu.device(self.stream.device):
            if self.ptr.value:
                self.stream.synchronize()
                _check(
                    (
                        self.lib.vllmInfinicoreDestroyEmbeddingDescriptor
                        if self.op == "Embedding"
                        else getattr(self.lib, f"infiniopDestroy{self.op}Descriptor")
                    )(self.ptr),
                    "destroy",
                )
                self.ptr = C.c_void_p()
            if self.handle.value:
                self.lib.vllmInfinicoreDestroyAscendHandle(self.handle)
                self.handle = C.c_void_p()


def clear_cache():
    cache = getattr(_LOCAL, "descriptors", {})
    for desc in cache.values():
        desc.close()
    cache.clear()


def launch(op, tensors, scalar=()):
    lib = library()
    if any(t.device != tensors[0].device for t in tensors):
        raise Unsupported("mixed-device tensors")
    stream = torch.npu.current_stream(tensors[0].device)
    key = (
        op,
        tensors[0].device,
        stream.npu_stream,
        tuple((tuple(t.shape), t.stride(), t.dtype) for t in tensors),
        scalar,
    )
    if not hasattr(_LOCAL, "descriptors"):
        _LOCAL.descriptors = OrderedDict()
    cache = _LOCAL.descriptors
    if key not in cache:
        if len(cache) >= 128:
            cache.popitem(last=False)[1].close()
        cache[key] = _Descriptor(lib, op, tensors, scalar, stream)
    cache.move_to_end(key)
    desc = cache[key]
    args = [desc.ptr]
    if op != "Embedding":
        workspace = torch.empty(
            desc.workspace_size.value, dtype=torch.uint8, device=tensors[0].device
        )
        workspace.record_stream(stream)
        args += [workspace.data_ptr(), desc.workspace_size.value]
    for tensor in tensors:
        tensor.record_stream(stream)
    args += [t.data_ptr() for t in tensors]
    if op == "Gemm":
        args += [1.0, 0.0]
    args += [stream.npu_stream]
    _check(getattr(lib, f"infiniop{op}")(*args), op)
    return tensors[0]


def rms_norm(x, weight, eps):
    x, weight = nd(x).contiguous(), nd(weight).contiguous()
    return launch("RMSNorm", [torch.empty_like(x), x, weight], (float(eps),))


def silu_and_mul(x):
    x = nd(x).contiguous()
    if x.shape[-1] % 2:
        raise Unsupported("SwiGLU requires an even hidden size")
    hidden = x.shape[-1] // 2
    # Upstream uses eight blocks with aligned, unmasked input loads. Reject
    # uneven/tail tiles rather than exposing storage beyond the logical tensor.
    if hidden % (8 * 32 // x.element_size()) or hidden > 8192:
        raise Unsupported("SwiGLU requires eight aligned tiles and hidden size <= 8192")
    original = x.shape[:-1] + (x.shape[-1] // 2,)
    gate, up = x.reshape(-1, x.shape[-1]).chunk(2, dim=-1)
    out = torch.empty(gate.shape, device=x.device, dtype=x.dtype)
    return launch("SwiGLU", [out, up, gate]).reshape(original)


def linear(x, weight, bias=None):
    if x.dtype == torch.float32:
        raise Unsupported(
            "pinned Ascend GEMM uses reduced-precision FP32 math; retain native FP32 linear"
        )
    x, weight = nd(x).contiguous(), nd(weight).contiguous()
    shape = x.shape[:-1] + (weight.shape[0],)
    out = torch.empty(
        (x.numel() // x.shape[-1], weight.shape[0]), dtype=x.dtype, device=x.device
    )
    launch("Gemm", [out, x.reshape(-1, x.shape[-1]), weight.t()])
    out = out.reshape(shape)
    return out if bias is None else out + bias


def embedding(ids, weight):
    ids, weight = nd(ids).contiguous(), nd(weight).contiguous()
    out = torch.empty(
        (*ids.shape, weight.shape[1]), device=weight.device, dtype=weight.dtype
    )
    return launch("Embedding", [out, ids, weight])


def rotary_embedding(positions, query, key, head_size, rotary_dim, cache, neox):
    if rotary_dim != head_size or head_size != 128 or positions.ndim != 1:
        raise Unsupported(
            "only 128-dimensional full-head RoPE with 1D positions is supported"
        )
    positions = nd(positions).contiguous()
    cache = nd(cache.to(device=query.device, dtype=query.dtype))
    cos, sin = (t.contiguous() for t in cache.chunk(2, dim=-1))

    def apply(x):
        if x is None:
            return None
        shaped = nd(x).contiguous().reshape(positions.numel(), -1, head_size)
        out = torch.empty_like(shaped)
        launch("RoPE", [out, shaped, positions, sin, cos], (int(neox),))
        return out.reshape(x.shape)

    return apply(query), apply(key)
