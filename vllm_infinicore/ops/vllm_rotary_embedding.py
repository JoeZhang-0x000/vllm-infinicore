"""vLLM OOT RoPE route for the InfiniCore wrapper op."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding as VllmRotaryEmbedding,
)

from .custom_ops import ROTARY_EMBEDDING_OP, load_custom_ops

VLLM_ROTARY_EMBEDDING_CLASS = "RotaryEmbedding"


@dataclass(frozen=True)
class VllmRotaryEmbeddingInstallStatus:
    installed: bool
    reason: str
    route_name: str = VLLM_ROTARY_EMBEDDING_CLASS


@dataclass(frozen=True)
class VllmRotaryEmbeddingUninstallStatus:
    uninstalled: bool
    reason: str
    route_name: str = VLLM_ROTARY_EMBEDDING_CLASS


class InfiniCoreRotaryEmbedding(VllmRotaryEmbedding):
    """Default Qwen3 RoPE replacement backed by ``vllm_infinicore`` ops."""

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._forward_infinicore_or_native(positions, query, key)

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._forward_infinicore_or_native(positions, query, key)

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._forward_infinicore_or_native(positions, query, key)

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._forward_infinicore_or_native(positions, query, key)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._forward_infinicore_or_native(positions, query, key)

    def _forward_infinicore_or_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        try:
            self._match_cos_sin_cache_dtype(query)
            return torch.ops.vllm_infinicore.rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.rotary_dim,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        except Exception:
            from .infinicore_backend import strict_backend_enabled

            if strict_backend_enabled():
                raise
            return super().forward_native(positions, query, key)


def install_vllm_rotary_embedding_oot() -> VllmRotaryEmbeddingInstallStatus:
    """Install the vLLM OOT default RoPE replacement idempotently."""

    custom_op_status = load_custom_ops(
        force=True,
        required_ops=(ROTARY_EMBEDDING_OP,),
    )
    if not custom_op_status.available:
        return VllmRotaryEmbeddingInstallStatus(
            installed=False,
            reason=custom_op_status.reason,
        )

    existing = op_registry_oot.get(VLLM_ROTARY_EMBEDDING_CLASS)
    if existing is not None:
        if _is_owned(existing):
            return VllmRotaryEmbeddingInstallStatus(
                installed=True,
                reason="InfiniCore RoPE OOT route already registered",
            )
        return VllmRotaryEmbeddingInstallStatus(
            installed=False,
            reason=(
                f"{VLLM_ROTARY_EMBEDDING_CLASS} OOT route already registered by "
                f"{existing.__module__}.{existing.__name__}"
            ),
        )

    CustomOp.register_oot(name=VLLM_ROTARY_EMBEDDING_CLASS)(
        InfiniCoreRotaryEmbedding
    )
    return VllmRotaryEmbeddingInstallStatus(
        installed=True,
        reason="InfiniCore RoPE OOT route registered",
    )


def uninstall_vllm_rotary_embedding_oot() -> VllmRotaryEmbeddingUninstallStatus:
    """Remove this plugin's vLLM OOT RoPE route if it owns the entry."""

    existing = op_registry_oot.get(VLLM_ROTARY_EMBEDDING_CLASS)
    if existing is None:
        return VllmRotaryEmbeddingUninstallStatus(
            uninstalled=False,
            reason="InfiniCore RoPE OOT route not installed",
        )

    if _is_owned(existing):
        op_registry_oot.pop(VLLM_ROTARY_EMBEDDING_CLASS, None)
        return VllmRotaryEmbeddingUninstallStatus(
            uninstalled=True,
            reason="InfiniCore RoPE OOT route unregistered",
        )

    return VllmRotaryEmbeddingUninstallStatus(
        uninstalled=False,
        reason=(
            f"{VLLM_ROTARY_EMBEDDING_CLASS} OOT route is owned by "
            f"{existing.__module__}.{existing.__name__}"
        ),
    )


def _is_owned(existing: type) -> bool:
    return (
        existing is InfiniCoreRotaryEmbedding
        or existing.__module__ == InfiniCoreRotaryEmbedding.__module__
        and existing.__name__ == InfiniCoreRotaryEmbedding.__name__
    )


__all__ = [
    "InfiniCoreRotaryEmbedding",
    "VLLM_ROTARY_EMBEDDING_CLASS",
    "VllmRotaryEmbeddingInstallStatus",
    "VllmRotaryEmbeddingUninstallStatus",
    "install_vllm_rotary_embedding_oot",
    "uninstall_vllm_rotary_embedding_oot",
]
