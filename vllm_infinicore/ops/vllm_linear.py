"""vLLM unquantized linear route for InfiniCore wrapper ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from .custom_ops import LINEAR_OP, LM_HEAD_OP, load_custom_ops

VLLM_LINEAR_ROUTE_NAMES = ("MatMul", "LMHead")

_ACTIVE_ROUTES: set[str] = set()
_ORIGINAL_LINEAR_APPLY: Callable[..., torch.Tensor] | None = None
_ORIGINAL_LM_HEAD_APPLY: Callable[..., torch.Tensor] | None = None


@dataclass(frozen=True)
class VllmLinearInstallStatus:
    installed: bool
    reason: str
    route_name: str


@dataclass(frozen=True)
class VllmLinearUninstallStatus:
    uninstalled: bool
    reason: str
    route_name: str


def install_vllm_unquantized_linear_route(
    route_name: str,
) -> VllmLinearInstallStatus:
    """Patch vLLM's unquantized linear method for MatMul/LMHead routes."""

    if route_name not in VLLM_LINEAR_ROUTE_NAMES:
        return VllmLinearInstallStatus(
            installed=False,
            reason=f"unsupported linear route {route_name}",
            route_name=route_name,
        )

    required_op = LINEAR_OP if route_name == "MatMul" else LM_HEAD_OP
    custom_op_status = load_custom_ops(force=True, required_ops=(required_op,))
    if not custom_op_status.available:
        return VllmLinearInstallStatus(
            installed=False,
            reason=custom_op_status.reason,
            route_name=route_name,
        )

    global _ORIGINAL_LINEAR_APPLY, _ORIGINAL_LM_HEAD_APPLY
    if route_name == "MatMul" and _ORIGINAL_LINEAR_APPLY is None:
        _ORIGINAL_LINEAR_APPLY = UnquantizedLinearMethod.apply
        UnquantizedLinearMethod.apply = _patched_linear_apply
    if route_name == "LMHead" and _ORIGINAL_LM_HEAD_APPLY is None:
        _ORIGINAL_LM_HEAD_APPLY = UnquantizedEmbeddingMethod.apply
        UnquantizedEmbeddingMethod.apply = _patched_lm_head_apply

    _ACTIVE_ROUTES.add(route_name)
    return VllmLinearInstallStatus(
        installed=True,
        reason=f"InfiniCore unquantized linear patch active for {route_name}",
        route_name=route_name,
    )


def uninstall_vllm_unquantized_linear_route(
    route_name: str,
) -> VllmLinearUninstallStatus:
    """Remove a MatMul/LMHead route and restore vLLM when none remain."""

    if route_name not in _ACTIVE_ROUTES:
        return VllmLinearUninstallStatus(
            uninstalled=False,
            reason=f"InfiniCore unquantized linear route {route_name} not installed",
            route_name=route_name,
        )

    _ACTIVE_ROUTES.remove(route_name)
    global _ORIGINAL_LINEAR_APPLY, _ORIGINAL_LM_HEAD_APPLY
    if route_name == "MatMul" and _ORIGINAL_LINEAR_APPLY is not None:
        UnquantizedLinearMethod.apply = _ORIGINAL_LINEAR_APPLY
        _ORIGINAL_LINEAR_APPLY = None
    if route_name == "LMHead" and _ORIGINAL_LM_HEAD_APPLY is not None:
        UnquantizedEmbeddingMethod.apply = _ORIGINAL_LM_HEAD_APPLY
        _ORIGINAL_LM_HEAD_APPLY = None

    return VllmLinearUninstallStatus(
        uninstalled=True,
        reason=f"InfiniCore unquantized linear route {route_name} uninstalled",
        route_name=route_name,
    )


def _patched_linear_apply(
    self: UnquantizedLinearMethod,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    try:
        return torch.ops.vllm_infinicore.linear(x, layer.weight, bias)
    except Exception:
        from .infinicore_backend import strict_backend_enabled

        if strict_backend_enabled():
            raise
        if _ORIGINAL_LINEAR_APPLY is None:
            raise
        return _ORIGINAL_LINEAR_APPLY(self, layer, x, bias)


def _patched_lm_head_apply(
    self: UnquantizedEmbeddingMethod,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not isinstance(layer, ParallelLMHead):
        if _ORIGINAL_LM_HEAD_APPLY is None:
            raise RuntimeError("original vLLM LMHead apply is unavailable")
        return _ORIGINAL_LM_HEAD_APPLY(self, layer, x, bias)
    try:
        return torch.ops.vllm_infinicore.lm_head(x, layer.weight, bias)
    except Exception:
        from .infinicore_backend import strict_backend_enabled

        if strict_backend_enabled():
            raise
        if _ORIGINAL_LM_HEAD_APPLY is None:
            raise
        return _ORIGINAL_LM_HEAD_APPLY(self, layer, x, bias)


__all__ = [
    "VLLM_LINEAR_ROUTE_NAMES",
    "VllmLinearInstallStatus",
    "VllmLinearUninstallStatus",
    "install_vllm_unquantized_linear_route",
    "uninstall_vllm_unquantized_linear_route",
]
