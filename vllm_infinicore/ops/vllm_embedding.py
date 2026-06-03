"""vLLM unquantized embedding route for the InfiniCore wrapper op."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from .custom_ops import EMBEDDING_OP, load_custom_ops

VLLM_EMBEDDING_ROUTE_NAME = "Embedding"

_INSTALLED = False
_ORIGINAL_EMBEDDING: Callable[..., torch.Tensor] | None = None


@dataclass(frozen=True)
class VllmEmbeddingInstallStatus:
    installed: bool
    reason: str
    route_name: str = VLLM_EMBEDDING_ROUTE_NAME


@dataclass(frozen=True)
class VllmEmbeddingUninstallStatus:
    uninstalled: bool
    reason: str
    route_name: str = VLLM_EMBEDDING_ROUTE_NAME


def install_vllm_unquantized_embedding_route() -> VllmEmbeddingInstallStatus:
    """Patch vLLM's unquantized embedding lookup method idempotently."""

    custom_op_status = load_custom_ops(force=True, required_ops=(EMBEDDING_OP,))
    if not custom_op_status.available:
        return VllmEmbeddingInstallStatus(
            installed=False,
            reason=custom_op_status.reason,
        )

    global _INSTALLED, _ORIGINAL_EMBEDDING
    if _INSTALLED:
        return VllmEmbeddingInstallStatus(
            installed=True,
            reason="InfiniCore unquantized embedding patch already active",
        )

    _ORIGINAL_EMBEDDING = UnquantizedEmbeddingMethod.embedding
    UnquantizedEmbeddingMethod.embedding = _patched_embedding
    _INSTALLED = True
    return VllmEmbeddingInstallStatus(
        installed=True,
        reason="InfiniCore unquantized embedding patch active",
    )


def uninstall_vllm_unquantized_embedding_route() -> VllmEmbeddingUninstallStatus:
    """Restore vLLM's unquantized embedding lookup method."""

    global _INSTALLED, _ORIGINAL_EMBEDDING
    if not _INSTALLED:
        return VllmEmbeddingUninstallStatus(
            uninstalled=False,
            reason="InfiniCore unquantized embedding route not installed",
        )

    if _ORIGINAL_EMBEDDING is not None:
        UnquantizedEmbeddingMethod.embedding = _ORIGINAL_EMBEDDING
    _ORIGINAL_EMBEDDING = None
    _INSTALLED = False
    return VllmEmbeddingUninstallStatus(
        uninstalled=True,
        reason="InfiniCore unquantized embedding route uninstalled",
    )


def _patched_embedding(
    self: UnquantizedEmbeddingMethod,
    layer: torch.nn.Module,
    input_: torch.Tensor,
) -> torch.Tensor:
    if _should_use_native_embedding(layer):
        if _ORIGINAL_EMBEDDING is None:
            raise RuntimeError("native vLLM embedding route is unavailable")
        return _ORIGINAL_EMBEDDING(self, layer, input_)
    try:
        return torch.ops.vllm_infinicore.embedding(input_, layer.weight)
    except Exception:
        from .infinicore_backend import strict_backend_enabled

        if strict_backend_enabled():
            raise
        if _ORIGINAL_EMBEDDING is None:
            raise
        return _ORIGINAL_EMBEDDING(self, layer, input_)


def _should_use_native_embedding(layer: torch.nn.Module) -> bool:
    return int(getattr(layer, "tp_size", 1) or 1) > 1


__all__ = [
    "VLLM_EMBEDDING_ROUTE_NAME",
    "VllmEmbeddingInstallStatus",
    "VllmEmbeddingUninstallStatus",
    "install_vllm_unquantized_embedding_route",
    "uninstall_vllm_unquantized_embedding_route",
]
