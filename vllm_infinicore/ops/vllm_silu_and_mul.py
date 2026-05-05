"""vLLM OOT SiluAndMul route for the InfiniCore wrapper op."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.activation import SiluAndMul as VllmSiluAndMul

from .custom_ops import SILU_AND_MUL_OP, load_custom_ops

VLLM_SILU_AND_MUL_CLASS = "SiluAndMul"


@dataclass(frozen=True)
class VllmSiluAndMulInstallStatus:
    installed: bool
    reason: str
    route_name: str = VLLM_SILU_AND_MUL_CLASS


@dataclass(frozen=True)
class VllmSiluAndMulUninstallStatus:
    uninstalled: bool
    reason: str
    route_name: str = VLLM_SILU_AND_MUL_CLASS


class InfiniCoreSiluAndMul(VllmSiluAndMul):
    """vLLM SiluAndMul replacement backed by ``vllm_infinicore`` ops."""

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_infinicore_or_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_infinicore_or_native(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_infinicore_or_native(x)

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_infinicore_or_native(x)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_infinicore_or_native(x)

    def _forward_infinicore_or_native(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return torch.ops.vllm_infinicore.silu_and_mul(x)
        except Exception:
            from .infinicore_backend import strict_backend_enabled

            if strict_backend_enabled():
                raise
            return super().forward_native(x)


def install_vllm_silu_and_mul_oot() -> VllmSiluAndMulInstallStatus:
    """Install the vLLM OOT SiluAndMul class replacement idempotently."""

    custom_op_status = load_custom_ops(
        force=True,
        required_ops=(SILU_AND_MUL_OP,),
    )
    if not custom_op_status.available:
        return VllmSiluAndMulInstallStatus(
            installed=False,
            reason=custom_op_status.reason,
        )

    existing = op_registry_oot.get(VLLM_SILU_AND_MUL_CLASS)
    if existing is not None:
        if _is_owned(existing):
            return VllmSiluAndMulInstallStatus(
                installed=True,
                reason="InfiniCore SiluAndMul OOT route already registered",
            )
        return VllmSiluAndMulInstallStatus(
            installed=False,
            reason=(
                f"{VLLM_SILU_AND_MUL_CLASS} OOT route already registered by "
                f"{existing.__module__}.{existing.__name__}"
            ),
        )

    CustomOp.register_oot(name=VLLM_SILU_AND_MUL_CLASS)(InfiniCoreSiluAndMul)
    return VllmSiluAndMulInstallStatus(
        installed=True,
        reason="InfiniCore SiluAndMul OOT route registered",
    )


def uninstall_vllm_silu_and_mul_oot() -> VllmSiluAndMulUninstallStatus:
    """Remove this plugin's vLLM OOT SiluAndMul route if it owns the entry."""

    existing = op_registry_oot.get(VLLM_SILU_AND_MUL_CLASS)
    if existing is None:
        return VllmSiluAndMulUninstallStatus(
            uninstalled=False,
            reason="InfiniCore SiluAndMul OOT route not installed",
        )

    if _is_owned(existing):
        op_registry_oot.pop(VLLM_SILU_AND_MUL_CLASS, None)
        return VllmSiluAndMulUninstallStatus(
            uninstalled=True,
            reason="InfiniCore SiluAndMul OOT route unregistered",
        )

    return VllmSiluAndMulUninstallStatus(
        uninstalled=False,
        reason=(
            f"{VLLM_SILU_AND_MUL_CLASS} OOT route is owned by "
            f"{existing.__module__}.{existing.__name__}"
        ),
    )


def _is_owned(existing: type) -> bool:
    return (
        existing is InfiniCoreSiluAndMul
        or existing.__module__ == InfiniCoreSiluAndMul.__module__
        and existing.__name__ == InfiniCoreSiluAndMul.__name__
    )


__all__ = [
    "InfiniCoreSiluAndMul",
    "VLLM_SILU_AND_MUL_CLASS",
    "VllmSiluAndMulInstallStatus",
    "VllmSiluAndMulUninstallStatus",
    "install_vllm_silu_and_mul_oot",
    "uninstall_vllm_silu_and_mul_oot",
]
