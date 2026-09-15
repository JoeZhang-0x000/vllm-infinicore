"""Custom operators that let the Ascend adapters survive Dynamo tracing.

`ascend_backend` reaches InfiniCore through ctypes and raw device pointers, which
Dynamo cannot trace. Without a registered operator the tracer drops the adapter
and the compiled program silently keeps the native implementation, which is what
made every earlier graph measurement a native-versus-native comparison. Each
entry here is an opaque node with a fake implementation for shape propagation, so
a compiled graph — and an ACL graph captured from it — keeps calling InfiniCore.

A compiled graph cannot choose an implementation per call, so capability is
decided before the node is emitted, using the same `supports_*` predicates that
guard the eager path. An unsupported case selects the native operator at trace
time instead of raising inside the traced program.
"""

from __future__ import annotations

import torch

from . import ascend_backend as backend


def _count(name: str) -> None:
    """Record a real InfiniCore launch.

    Only bodies that actually execute increment this. An ACL graph replay runs
    the recorded kernels without re-entering Python, so replay-only steps do not
    add counts; capture and every eager or prefill step still do.
    """
    from . import infinicore_backend as counters

    counters._CALL_COUNTS[name] = counters._CALL_COUNTS.get(name, 0) + 1


@torch.library.custom_op("vllm_infinicore_ascend::linear", mutates_args=())
def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, name: str
) -> torch.Tensor:
    out = backend.linear(x, weight, bias)
    _count(name)
    return out


@linear.register_fake
def _(x, weight, bias, name):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op("vllm_infinicore_ascend::embedding", mutates_args=())
def embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = backend.embedding(ids, weight)
    _count("embedding")
    return out


@embedding.register_fake
def _(ids, weight):
    return weight.new_empty((*ids.shape, weight.shape[1]))


@torch.library.custom_op("vllm_infinicore_ascend::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = backend.rms_norm(x, weight, eps)
    _count("rms_norm")
    return out


@rms_norm.register_fake
def _(x, weight, eps):
    return torch.empty_like(x)


@torch.library.custom_op("vllm_infinicore_ascend::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    out = backend.silu_and_mul(x)
    _count("silu_and_mul")
    return out


@silu_and_mul.register_fake
def _(x):
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


@torch.library.custom_op("vllm_infinicore_ascend::rotary_embedding", mutates_args=())
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    cache: torch.Tensor,
    neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_query, out_key = backend.rotary_embedding(
        positions, query, key, head_size, rotary_dim, cache, neox
    )
    _count("rotary_embedding")
    return out_query, out_key


@rotary_embedding.register_fake
def _(positions, query, key, head_size, rotary_dim, cache, neox):
    return torch.empty_like(query), torch.empty_like(key)
