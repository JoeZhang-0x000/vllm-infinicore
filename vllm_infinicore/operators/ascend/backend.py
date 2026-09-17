"""Pinned InfiniCore C API on torch's current NPU stream.

Descriptors are cached per device and shape, deliberately not per stream, so a
descriptor warmed on the default stream is reused during graph capture. Tensor
storage is owned by torch, and every launch is issued on the tensors' own
current stream, which is what keeps the raw pointers valid for the duration of
the launch; see `launch` for why record_stream must not be added back. Launches
are traceable through the operators in `ascend_graph_ops`. No device, worker,
communication or KV-cache runtime is implemented here.
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
_LOCK = Path(__file__).resolve().parents[2] / "infinicore.lock.json"
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
    from .. import backend as counters

    counters._FALLBACK_COUNTS[name] = counters._FALLBACK_COUNTS.get(name, 0) + 1
    counters._FALLBACK_REASONS[name] = reason
    return native()


def graph_enabled():
    """Whether InfiniCore may run inside a captured Ascend graph.

    Set `VLLM_INFINICORE_ASCEND_GRAPH=0` to restore the eager-only behaviour and
    let capture fall back to the native operator.
    """
    return os.environ.get("VLLM_INFINICORE_ASCEND_GRAPH", "1") != "0"


def execute(name, tensor, operation, native):
    # Routes emit a registered custom op while tracing, so reaching here under
    # Dynamo means no traceable operator exists for this call.
    if torch.compiler.is_compiling():
        return native()
    if tensor.device.type != "npu":
        return native()
    if os.environ.get("VLLM_INFINICORE_DISABLE_REAL_BACKEND") == "1":
        return fallback(name, "real backend explicitly disabled", native)
    try:
        with torch.npu.device(tensor.device):
            if torch.npu.is_current_stream_capturing() and not graph_enabled():
                raise Unsupported("Ascend graph capture disabled by environment")
            result = operation()
    except Unsupported as exc:
        return fallback(name, str(exc), native)
    # Never retry a failed device launch: a runtime failure is not a capability miss.
    from .. import backend as counters

    counters._CALL_COUNTS[name] = counters._CALL_COUNTS.get(name, 0) + 1
    return result


def supports_tensor(tensor):
    """The device and dtype half of `nd`, answerable from a traced tensor.

    Format casting stays inside `nd`: it inspects real device memory, so it can
    only run in the operator body, never at trace time.
    """
    if tensor.device.type != "npu" or tensor.dtype not in _DTYPES:
        return (
            False,
            f"unsupported tensor device/dtype: {tensor.device}/{tensor.dtype}",
        )
    return True, ""


def nd(tensor):
    import torch_npu

    supported, reason = supports_tensor(tensor)
    if not supported:
        raise Unsupported(reason)
    if not tensor.numel():
        raise Unsupported("empty tensor")
    if torch_npu.get_npu_format(tensor) != 2:
        tensor = torch_npu.npu_format_cast(tensor, 2)
    return tensor


class _Descriptor:
    def __init__(self, lib, op, tensors, scalar, stream):
        self.lib, self.op, self.stream = lib, op, stream
        # A descriptor created during capture records the capture stream, which
        # is not the stream to synchronize against when it is later destroyed.
        self.device = tensors[0].device
        self.ptr = C.c_void_p()
        self.handle = C.c_void_p()
        self.workspace_size = C.c_size_t()
        self._workspace = None
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

    def workspace(self, capturing):
        """Scratch for one launch, shared by every descriptor on the device.

        Allocating per launch churns the caching allocator tens of thousands of
        times per batch and puts an allocation inside every capture, but a
        buffer per descriptor is worse: sizes are skewed (88 MiB for a large
        prefill GEMM against a 0.16 MiB median), so summing them exhausts the
        headroom left by `gpu_memory_utilization` and the engine fails to start.
        One buffer at the high-water mark costs the maximum instead of the sum.

        Launches on a device are serialized on its stream, so sharing scratch is
        safe. Growth is not: a captured graph records the pointer, so once a
        capture has used the shared buffer it must never be reallocated, and any
        later launch needing more takes a private buffer instead.
        """
        need = max(self.workspace_size.value, 1)
        if self._workspace is not None and self._workspace.numel() >= need:
            return self._workspace
        shared = getattr(_LOCAL, "workspace", None)
        if shared is not None and shared.numel() >= need:
            if capturing:
                _LOCAL.workspace_locked = True
            return shared
        if capturing or getattr(_LOCAL, "workspace_locked", False):
            self._workspace = torch.empty(need, dtype=torch.uint8, device=self.device)
            return self._workspace
        shared = torch.empty(need, dtype=torch.uint8, device=self.device)
        _LOCAL.workspace = shared
        return shared

    def close(self):
        # Embedding owns an ACL workspace; executors also must outlive launches.
        with torch.npu.device(self.device):
            if self.ptr.value:
                torch.npu.current_stream(self.device).synchronize()
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



# The descriptor key includes the token count, so chunked prefill and every
# decode batch width add entries. Evicting one costs a stream synchronize, so
# the limit sits well above the working set a run actually reaches (~90).
_DESCRIPTOR_CACHE_LIMIT = 4096


def _evict(cache):
    """Drop the oldest descriptor that no captured graph depends on."""
    for key, desc in cache.items():
        if not getattr(desc, "pinned", False):
            cache.pop(key).close()
            return


def clear_cache():
    _LOCAL.workspace = None
    _LOCAL.workspace_locked = False
    cache = getattr(_LOCAL, "descriptors", {})
    for desc in cache.values():
        desc.close()
    cache.clear()


def launch(op, tensors, scalar=()):
    lib = library()
    if any(t.device != tensors[0].device for t in tensors):
        raise Unsupported("mixed-device tensors")
    stream = torch.npu.current_stream(tensors[0].device)
    # The stream is a launch argument, not part of the descriptor, so it stays
    # out of the key. Keying on it would force a fresh descriptor during graph
    # capture, whose creation and eviction both touch the host mid-capture.
    key = (
        op,
        tensors[0].device,
        tuple((tuple(t.shape), t.stride(), t.dtype) for t in tensors),
        scalar,
    )
    if not hasattr(_LOCAL, "descriptors"):
        _LOCAL.descriptors = OrderedDict()
    cache = _LOCAL.descriptors
    capturing = torch.npu.is_current_stream_capturing()
    if key not in cache:
        # close() synchronizes, which is illegal mid-capture. A capture adds at
        # most the shapes it records, so letting the cache grow is bounded.
        if len(cache) >= _DESCRIPTOR_CACHE_LIMIT and not capturing:
            _evict(cache)
        cache[key] = _Descriptor(lib, op, tensors, scalar, stream)
        # A captured graph replays the recorded launch without consulting this
        # cache, so destroying its descriptor would leave the graph pointing at
        # freed state. Descriptors recorded into a graph are never evicted.
        cache[key].pinned = capturing
    cache.move_to_end(key)
    desc = cache[key]
    if capturing:
        desc.pinned = True
    args = [desc.ptr]
    if op != "Embedding":
        # The descriptor owns this buffer for its whole lifetime, so it needs no
        # record_stream and its address is stable across a graph replay.
        args += [desc.workspace(capturing).data_ptr(), desc.workspace_size.value]
    # No record_stream here. Every launch goes on the tensors' own current
    # stream, which the caching allocator already orders allocations against, so
    # it would protect nothing. It is far from free: it defers block reuse until
    # the allocator observes a stream event, so with a fresh output allocated per
    # call and little spare memory each allocation blocks on device progress
    # instead of pipelining. Removing it cut a 4-sequence prefill from 12.1 s to
    # 2.0 s against native's 1.8 s, and removed a 16x spread in allocation cost
    # between ranks. Reintroduce it only alongside a launch on some other stream,
    # and never during capture, where the graph pool owns the addresses.
    args += [t.data_ptr() for t in tensors]
    if op == "Gemm":
        args += [1.0, 0.0]
    args += [stream.npu_stream]
    _check(getattr(lib, f"infiniop{op}")(*args), op)
    return tensors[0]


def rms_norm(x, weight, eps):
    x, weight = nd(x).contiguous(), nd(weight).contiguous()
    return launch("RMSNorm", [torch.empty_like(x), x, weight], (float(eps),))


def supports_silu_and_mul(x):
    """Report SwiGLU capability from shape and dtype alone.

    A compiled graph fixes its operators at trace time, so the same predicate
    has to answer before the node is emitted and before an eager launch. Like
    every `supports_*`, it must answer for any tensor rather than raise: callers
    evaluate it before knowing whether the call is eligible at all.
    """
    if x.ndim == 0 or x.shape[-1] % 2:
        return False, "SwiGLU requires an even hidden size"
    hidden = x.shape[-1] // 2
    # Upstream uses eight blocks with aligned, unmasked input loads. Reject
    # uneven/tail tiles rather than exposing storage beyond the logical tensor.
    if hidden % (8 * 32 // x.element_size()) or hidden > 8192:
        return False, "SwiGLU requires eight aligned tiles and hidden size <= 8192"
    return True, ""


def silu_and_mul(x):
    x = nd(x).contiguous()
    supported, reason = supports_silu_and_mul(x)
    if not supported:
        raise Unsupported(reason)
    original = x.shape[:-1] + (x.shape[-1] // 2,)
    gate, up = x.reshape(-1, x.shape[-1]).chunk(2, dim=-1)
    out = torch.empty(gate.shape, device=x.device, dtype=x.dtype)
    return launch("SwiGLU", [out, up, gate]).reshape(original)


def supports_linear(x):
    if x.dtype == torch.float32:
        return (
            False,
            "pinned Ascend GEMM uses reduced-precision FP32 math; retain native FP32 linear",
        )
    return True, ""


def linear(x, weight, bias=None):
    supported, reason = supports_linear(x)
    if not supported:
        raise Unsupported(reason)
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


def supports_rotary_embedding(positions, head_size, rotary_dim):
    if rotary_dim != head_size or head_size != 128 or positions.ndim != 1:
        return (
            False,
            "only 128-dimensional full-head RoPE with 1D positions is supported",
        )
    return True, ""


def rotary_embedding(positions, query, key, head_size, rotary_dim, cache, neox):
    supported, reason = supports_rotary_embedding(positions, head_size, rotary_dim)
    if not supported:
        raise Unsupported(reason)
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
