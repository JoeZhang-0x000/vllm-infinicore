"""Default-off InfiniCore-backed PyTorch custom op wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

CUSTOM_OP_ENABLE_ENV = "VLLM_INFINICORE_ENABLE_CUSTOM_OPS"
RMS_NORM_OP = "vllm_infinicore::rms_norm"
SILU_AND_MUL_OP = "vllm_infinicore::silu_and_mul"
LINEAR_OP = "vllm_infinicore::linear"
LM_HEAD_OP = "vllm_infinicore::lm_head"
EMBEDDING_OP = "vllm_infinicore::embedding"
ROTARY_EMBEDDING_OP = "vllm_infinicore::rotary_embedding"

ALL_CUSTOM_OPS = (
    RMS_NORM_OP,
    SILU_AND_MUL_OP,
    LINEAR_OP,
    LM_HEAD_OP,
    EMBEDDING_OP,
    ROTARY_EMBEDDING_OP,
)

_TORCH_LIBRARY: Any | None = None
_REGISTERED_OPS: tuple[str, ...] = ()


@dataclass(frozen=True)
class CustomOpStatus:
    available: bool
    reason: str
    registered_ops: tuple[str, ...] = ()


def load_custom_ops(
    *,
    force: bool = False,
    required_ops: tuple[str, ...] | None = None,
) -> CustomOpStatus:
    """Register custom op prototypes only when explicitly requested.

    The default path intentionally avoids importing torch so dry plugin
    registration remains cheap and graph-conservative.
    """

    requested_ops = required_ops or ALL_CUSTOM_OPS
    unknown_ops = tuple(op for op in requested_ops if op not in ALL_CUSTOM_OPS)
    if unknown_ops:
        return CustomOpStatus(
            available=False,
            reason=f"unsupported custom op request: {', '.join(unknown_ops)}",
            registered_ops=_REGISTERED_OPS,
        )

    if not force and not _env_truthy(CUSTOM_OP_ENABLE_ENV):
        return CustomOpStatus(
            available=False,
            reason=f"{CUSTOM_OP_ENABLE_ENV} is unset or false; custom ops disabled",
            registered_ops=_REGISTERED_OPS,
        )

    if all(op in _REGISTERED_OPS for op in requested_ops):
        return CustomOpStatus(
            available=True,
            reason="requested custom op prototypes already registered",
            registered_ops=_REGISTERED_OPS,
        )

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local install
        return CustomOpStatus(
            available=False,
            reason=f"torch import failed while loading custom ops: {exc}",
        )

    try:
        for op in requested_ops:
            _REGISTERERS[op](torch)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return CustomOpStatus(
            available=False,
            reason=f"custom op registration failed: {exc}",
            registered_ops=_REGISTERED_OPS,
        )

    return CustomOpStatus(
        available=True,
        reason="InfiniCore custom op wrappers registered; vLLM patches remain disabled",
        registered_ops=_REGISTERED_OPS,
    )


def is_available() -> bool:
    return load_custom_ops().available


def rms_norm(input_tensor: Any, weight: Any, eps: float = 1e-6) -> Any:
    """Run the default-off RMSNorm custom op wrapper."""

    _ensure_direct_api_enabled((RMS_NORM_OP,))
    import torch

    return torch.ops.vllm_infinicore.rms_norm(input_tensor, weight, float(eps))


def silu_and_mul(input_tensor: Any) -> Any:
    """Run the default-off SiluAndMul custom op wrapper."""

    _ensure_direct_api_enabled((SILU_AND_MUL_OP,))
    import torch

    return torch.ops.vllm_infinicore.silu_and_mul(input_tensor)


def linear(input_tensor: Any, weight: Any, bias: Any | None = None) -> Any:
    """Run the default-off Linear/MatMul custom op wrapper."""

    _ensure_direct_api_enabled((LINEAR_OP,))
    import torch

    return torch.ops.vllm_infinicore.linear(input_tensor, weight, bias)


def lm_head(input_tensor: Any, weight: Any, bias: Any | None = None) -> Any:
    """Run the default-off LMHead custom op wrapper."""

    _ensure_direct_api_enabled((LM_HEAD_OP,))
    import torch

    return torch.ops.vllm_infinicore.lm_head(input_tensor, weight, bias)


def embedding(input_tensor: Any, weight: Any) -> Any:
    """Run the default-off Embedding custom op wrapper."""

    _ensure_direct_api_enabled((EMBEDDING_OP,))
    import torch

    return torch.ops.vllm_infinicore.embedding(input_tensor, weight)


def rotary_embedding(
    positions: Any,
    query: Any,
    key: Any | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: Any,
    is_neox_style: bool,
) -> tuple[Any, Any | None]:
    """Run the default-off RoPE custom op wrapper."""

    _ensure_direct_api_enabled((ROTARY_EMBEDDING_OP,))
    import torch

    return torch.ops.vllm_infinicore.rotary_embedding(
        positions,
        query,
        key,
        int(head_size),
        int(rotary_dim),
        cos_sin_cache,
        bool(is_neox_style),
    )


def _register_rms_norm(torch: Any) -> None:
    global _REGISTERED_OPS

    if RMS_NORM_OP in _REGISTERED_OPS:
        return

    library = _library(torch)
    library.define("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor")

    def _rms_norm_impl(input_tensor: Any, weight: Any, eps: float) -> Any:
        from . import infinicore_backend

        return infinicore_backend.rms_norm(input_tensor, weight, float(eps))

    library.impl("rms_norm", _rms_norm_impl, "CompositeExplicitAutograd")

    _REGISTERED_OPS = (*_REGISTERED_OPS, RMS_NORM_OP)


def _register_silu_and_mul(torch: Any) -> None:
    global _REGISTERED_OPS

    if SILU_AND_MUL_OP in _REGISTERED_OPS:
        return

    library = _library(torch)
    library.define("silu_and_mul(Tensor input) -> Tensor")

    def _silu_and_mul_impl(input_tensor: Any) -> Any:
        from . import infinicore_backend

        return infinicore_backend.silu_and_mul(input_tensor)

    library.impl("silu_and_mul", _silu_and_mul_impl, "CompositeExplicitAutograd")
    _REGISTERED_OPS = (*_REGISTERED_OPS, SILU_AND_MUL_OP)


def _register_linear(torch: Any) -> None:
    global _REGISTERED_OPS

    if LINEAR_OP in _REGISTERED_OPS:
        return

    library = _library(torch)
    library.define("linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor")

    def _linear_impl(input_tensor: Any, weight: Any, bias: Any | None = None) -> Any:
        from . import infinicore_backend

        return infinicore_backend.linear(input_tensor, weight, bias)

    library.impl("linear", _linear_impl, "CompositeExplicitAutograd")
    _REGISTERED_OPS = (*_REGISTERED_OPS, LINEAR_OP)


def _register_lm_head(torch: Any) -> None:
    global _REGISTERED_OPS

    if LM_HEAD_OP in _REGISTERED_OPS:
        return

    library = _library(torch)
    library.define("lm_head(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor")

    def _lm_head_impl(input_tensor: Any, weight: Any, bias: Any | None = None) -> Any:
        from . import infinicore_backend

        return infinicore_backend.lm_head(input_tensor, weight, bias)

    library.impl("lm_head", _lm_head_impl, "CompositeExplicitAutograd")
    _REGISTERED_OPS = (*_REGISTERED_OPS, LM_HEAD_OP)


def _register_embedding(torch: Any) -> None:
    global _REGISTERED_OPS

    if EMBEDDING_OP in _REGISTERED_OPS:
        return

    library = _library(torch)
    library.define("embedding(Tensor input, Tensor weight) -> Tensor")

    def _embedding_impl(input_tensor: Any, weight: Any) -> Any:
        from . import infinicore_backend

        return infinicore_backend.embedding(input_tensor, weight)

    library.impl("embedding", _embedding_impl, "CompositeExplicitAutograd")
    _REGISTERED_OPS = (*_REGISTERED_OPS, EMBEDDING_OP)


def _register_rotary_embedding(torch: Any) -> None:
    global _REGISTERED_OPS

    if ROTARY_EMBEDDING_OP in _REGISTERED_OPS:
        return

    library = _library(torch)
    library.define(
        "rotary_embedding(Tensor positions, Tensor query, Tensor? key, "
        "int head_size, int rotary_dim, Tensor cos_sin_cache, "
        "bool is_neox_style) -> (Tensor, Tensor?)"
    )

    def _rotary_embedding_impl(
        positions: Any,
        query: Any,
        key: Any | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: Any,
        is_neox_style: bool,
    ) -> tuple[Any, Any | None]:
        from . import infinicore_backend

        return infinicore_backend.rotary_embedding(
            positions,
            query,
            key,
            int(head_size),
            int(rotary_dim),
            cos_sin_cache,
            bool(is_neox_style),
        )

    library.impl(
        "rotary_embedding",
        _rotary_embedding_impl,
        "CompositeExplicitAutograd",
    )
    _REGISTERED_OPS = (*_REGISTERED_OPS, ROTARY_EMBEDDING_OP)


def _library(torch: Any) -> Any:
    global _TORCH_LIBRARY

    if _TORCH_LIBRARY is None:
        _TORCH_LIBRARY = torch.library.Library("vllm_infinicore", "FRAGMENT")
    return _TORCH_LIBRARY


def _ensure_direct_api_enabled(required_ops: tuple[str, ...]) -> None:
    if not _env_truthy(CUSTOM_OP_ENABLE_ENV):
        raise RuntimeError(
            f"{CUSTOM_OP_ENABLE_ENV} is unset or false; custom ops disabled"
        )

    status = load_custom_ops(required_ops=required_ops)
    if not status.available:
        raise RuntimeError(status.reason)


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


_REGISTERERS = {
    RMS_NORM_OP: _register_rms_norm,
    SILU_AND_MUL_OP: _register_silu_and_mul,
    LINEAR_OP: _register_linear,
    LM_HEAD_OP: _register_lm_head,
    EMBEDDING_OP: _register_embedding,
    ROTARY_EMBEDDING_OP: _register_rotary_embedding,
}
