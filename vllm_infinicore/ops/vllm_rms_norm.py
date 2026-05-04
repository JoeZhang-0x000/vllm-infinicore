"""vLLM OOT RMSNorm route for the InfiniCore prototype op."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

from .custom_ops import RMS_NORM_OP, load_custom_ops

VLLM_RMS_NORM_CLASS = "RMSNorm"


@dataclass(frozen=True)
class VllmRMSNormInstallStatus:
    installed: bool
    reason: str
    route_name: str = VLLM_RMS_NORM_CLASS


@dataclass(frozen=True)
class VllmRMSNormUninstallStatus:
    uninstalled: bool
    reason: str
    route_name: str = VLLM_RMS_NORM_CLASS


class InfiniCoreRMSNorm(VllmRMSNorm):
    """Narrow vLLM RMSNorm replacement backed by the prototype custom op.

    Only the non-residual, weighted, no variance override path is routed. All
    fused-add/residual and variant paths intentionally use vLLM's native
    PyTorch implementation.
    """

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._forward_infinicore_or_native(x, residual)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._forward_infinicore_or_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._forward_infinicore_or_native(x, residual)

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._forward_infinicore_or_native(x, residual)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._forward_infinicore_or_native(x, residual)

    def _forward_infinicore_or_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._should_use_infinicore(x, residual):
            return torch.ops.vllm_infinicore.rms_norm(
                x, self.weight.data, self.variance_epsilon
            )
        return super().forward_native(x, residual)

    def _should_use_infinicore(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> bool:
        return (
            residual is None
            and self.variance_size_override is None
            and self.has_weight
            and x.shape[-1] == self.hidden_size
        )


def install_vllm_rms_norm_oot() -> VllmRMSNormInstallStatus:
    """Install the vLLM OOT RMSNorm class replacement idempotently."""

    custom_op_status = load_custom_ops(force=True, required_ops=(RMS_NORM_OP,))
    if not custom_op_status.available:
        return VllmRMSNormInstallStatus(
            installed=False,
            reason=custom_op_status.reason,
        )

    existing = op_registry_oot.get(VLLM_RMS_NORM_CLASS)
    if existing is not None:
        if (
            existing is InfiniCoreRMSNorm
            or existing.__module__ == InfiniCoreRMSNorm.__module__
            and existing.__name__ == InfiniCoreRMSNorm.__name__
        ):
            return VllmRMSNormInstallStatus(
                installed=True,
                reason="InfiniCore RMSNorm OOT route already registered",
            )
        return VllmRMSNormInstallStatus(
            installed=False,
            reason=(
                f"{VLLM_RMS_NORM_CLASS} OOT route already registered by "
                f"{existing.__module__}.{existing.__name__}"
            ),
        )

    CustomOp.register_oot(name=VLLM_RMS_NORM_CLASS)(InfiniCoreRMSNorm)
    return VllmRMSNormInstallStatus(
        installed=True,
        reason="InfiniCore RMSNorm OOT route registered",
    )


def uninstall_vllm_rms_norm_oot() -> VllmRMSNormUninstallStatus:
    """Remove this plugin's vLLM OOT RMSNorm route if it owns the entry."""

    existing = op_registry_oot.get(VLLM_RMS_NORM_CLASS)
    if existing is None:
        return VllmRMSNormUninstallStatus(
            uninstalled=False,
            reason="InfiniCore RMSNorm OOT route not installed",
        )

    if (
        existing is InfiniCoreRMSNorm
        or existing.__module__ == InfiniCoreRMSNorm.__module__
        and existing.__name__ == InfiniCoreRMSNorm.__name__
    ):
        op_registry_oot.pop(VLLM_RMS_NORM_CLASS, None)
        return VllmRMSNormUninstallStatus(
            uninstalled=True,
            reason="InfiniCore RMSNorm OOT route unregistered",
        )

    return VllmRMSNormUninstallStatus(
        uninstalled=False,
        reason=(
            f"{VLLM_RMS_NORM_CLASS} OOT route is owned by "
            f"{existing.__module__}.{existing.__name__}"
        ),
    )


__all__ = [
    "InfiniCoreRMSNorm",
    "VLLM_RMS_NORM_CLASS",
    "VllmRMSNormInstallStatus",
    "VllmRMSNormUninstallStatus",
    "install_vllm_rms_norm_oot",
    "uninstall_vllm_rms_norm_oot",
]
