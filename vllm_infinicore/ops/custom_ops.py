"""Default-off prototypes for future InfiniCore-backed PyTorch custom ops."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

CUSTOM_OP_ENABLE_ENV = "VLLM_INFINICORE_ENABLE_CUSTOM_OPS"
RMS_NORM_OP = "vllm_infinicore::rms_norm"

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

    requested_ops = required_ops or (RMS_NORM_OP,)
    unknown_ops = tuple(op for op in requested_ops if op != RMS_NORM_OP)
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
        if RMS_NORM_OP in requested_ops:
            _register_rms_norm(torch)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return CustomOpStatus(
            available=False,
            reason=f"custom op registration failed: {exc}",
            registered_ops=_REGISTERED_OPS,
        )

    return CustomOpStatus(
        available=True,
        reason="RMSNorm custom op prototype registered; vLLM patches remain disabled",
        registered_ops=_REGISTERED_OPS,
    )


def is_available() -> bool:
    return load_custom_ops().available


def rms_norm(input_tensor: Any, weight: Any, eps: float = 1e-6) -> Any:
    """Run the default-off RMSNorm prototype custom op."""

    if not _env_truthy(CUSTOM_OP_ENABLE_ENV):
        raise RuntimeError(
            f"{CUSTOM_OP_ENABLE_ENV} is unset or false; custom ops disabled"
        )

    status = load_custom_ops(required_ops=(RMS_NORM_OP,))
    if not status.available:
        raise RuntimeError(status.reason)

    import torch

    return torch.ops.vllm_infinicore.rms_norm(input_tensor, weight, float(eps))


def _register_rms_norm(torch: Any) -> None:
    global _REGISTERED_OPS, _TORCH_LIBRARY

    if RMS_NORM_OP in _REGISTERED_OPS:
        return

    library = torch.library.Library("vllm_infinicore", "FRAGMENT")
    library.define("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor")

    def _rms_norm_impl(input_tensor: Any, weight: Any, eps: float) -> Any:
        input_float = input_tensor.float()
        variance = input_float.pow(2).mean(dim=-1, keepdim=True)
        output = input_float * torch.rsqrt(variance + float(eps))
        output = output.to(dtype=input_tensor.dtype)
        return output * weight

    library.impl("rms_norm", _rms_norm_impl, "CompositeExplicitAutograd")

    _TORCH_LIBRARY = library
    _REGISTERED_OPS = (*_REGISTERED_OPS, RMS_NORM_OP)


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
