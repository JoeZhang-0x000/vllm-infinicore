"""MetaX vLLM operator routes."""

from collections.abc import Callable
from dataclasses import dataclass

from ...routing.patching import PatchInstallResult, PatchUninstallResult
from ..routes import install_shared_route, uninstall_shared_route
from . import SUPPORTED_ROUTES

_METHODS = ("forward_oot", "forward_native")
_MISSING = object()


@dataclass
class _MethodPatch:
    cls: type
    originals: dict[str, object]
    replacement: Callable


_PATCHES: dict[str, _MethodPatch] = {}


def _replace_methods(name: str, cls: type, forward: Callable) -> None:
    originals = {method: vars(cls).get(method, _MISSING) for method in _METHODS}
    for method in _METHODS:
        setattr(cls, method, forward)
    _PATCHES[name] = _MethodPatch(cls, originals, forward)


def _uninstall_methods(name: str) -> PatchUninstallResult:
    patch = _PATCHES.get(name)
    if patch is None:
        return PatchUninstallResult(False, f"MetaX {name} adapter not installed")
    if any(getattr(patch.cls, method, None) is not patch.replacement for method in _METHODS):
        return PatchUninstallResult(False, f"{name} method changed by another patch")
    for method, original in patch.originals.items():
        if original is _MISSING:
            delattr(patch.cls, method)
        else:
            setattr(patch.cls, method, original)
    del _PATCHES[name]
    return PatchUninstallResult(True, f"original MetaX {name} restored")


def _install_silu_and_mul():
    """Route the vendor-owned class through the modular native SiLU kernel."""
    import torch
    from vllm_metax.customized.ops.activation import MacaSiluAndMul

    from ..custom_ops import SILU_AND_MUL_OP, load_custom_ops

    if "SiluAndMul" in _PATCHES:
        return PatchInstallResult(True, "MetaX SiLU adapter already installed")
    status = load_custom_ops(force=True, required_ops=(SILU_AND_MUL_OP,))
    if not status.available:
        return PatchInstallResult(False, status.reason)

    def forward(self, x):
        return torch.ops.vllm_infinicore.silu_and_mul(x)

    _replace_methods("SiluAndMul", MacaSiluAndMul, forward)
    return PatchInstallResult(True, "InfiniOps SiLU on MetaX-owned OOT class")


def _install_rms_norm():
    """Route RMSNorm on the vendor-owned class without an OOT name collision."""
    import torch
    from vllm_metax.customized.ops.layernorm import MacaRMSNorm

    from ..cpp_bridge import uses_modular_api
    from ..custom_ops import (
        FUSED_ADD_RMS_NORM_INPLACE_OP,
        FUSED_ADD_RMS_NORM_OP,
        RMS_NORM_OP,
        load_custom_ops,
    )

    if "RMSNorm" in _PATCHES:
        return PatchInstallResult(True, "MetaX RMSNorm adapter already installed")
    mutable = uses_modular_api()
    fused_op = FUSED_ADD_RMS_NORM_INPLACE_OP if mutable else FUSED_ADD_RMS_NORM_OP
    status = load_custom_ops(force=True, required_ops=(RMS_NORM_OP, fused_op))
    if not status.available:
        return PatchInstallResult(False, status.reason)
    native = MacaRMSNorm.forward_native

    def forward(self, x, residual=None):
        if (
            not self.has_weight
            or self.variance_size_override is not None
            or (residual is not None and not getattr(self, "pass_weight_add", True))
        ):
            # Call the captured implementation directly: forward_oot delegates
            # to forward_cuda, which calls forward_native again on this version.
            return native(self, x, residual)
        if residual is not None:
            if mutable:
                torch.ops.vllm_infinicore.fused_add_rms_norm_(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.vllm_infinicore.fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        return torch.ops.vllm_infinicore.rms_norm(x, self.weight.data, self.variance_epsilon)

    _replace_methods("RMSNorm", MacaRMSNorm, forward)
    return PatchInstallResult(True, "InfiniCore plain and supported fused MetaX RMSNorm")


def _install_rope():
    """Wrap the vendor-owned OOT class without competing for its registry name."""
    from functools import wraps

    import torch
    from vllm_metax.customized.ops.rotary_embedding import MacaRotaryEmbedding

    from ..cpp_bridge import uses_modular_api
    from ..custom_ops import ROTARY_EMBEDDING_INPLACE_OP, ROTARY_EMBEDDING_OP, load_custom_ops

    if "RoPE" in _PATCHES:
        return PatchInstallResult(True, "MetaX RoPE adapter already installed")
    mutable = uses_modular_api()
    rope_op = ROTARY_EMBEDDING_INPLACE_OP if mutable else ROTARY_EMBEDDING_OP
    status = load_custom_ops(force=True, required_ops=(rope_op,))
    if not status.available:
        return PatchInstallResult(False, status.reason)
    original = MacaRotaryEmbedding.forward_oot

    @wraps(original)
    def forward(self, positions, query, key=None):
        cache = self._match_cos_sin_cache_dtype(query)
        if mutable:
            torch.ops.vllm_infinicore.rotary_embedding_(
                positions, query, key, self.head_size, self.rotary_dim, cache, self.is_neox_style
            )
            return query, key
        return torch.ops.vllm_infinicore.rotary_embedding(
            positions, query, key, self.head_size, self.rotary_dim, cache, self.is_neox_style
        )

    # Inductor defaults custom_ops to "none", which dispatches directly to
    # forward_native and otherwise bypasses the registered OOT method.
    _replace_methods("RoPE", MacaRotaryEmbedding, forward)
    return PatchInstallResult(True, "InfiniCore RoPE on MetaX-owned OOT class")


def install(name: str):
    from .. import attention
    from ..cpp_bridge import uses_modular_api

    if name in attention.ROUTES:
        return attention.install(name, "metax")
    if name == "RoPE":
        return _install_rope()
    if name == "RMSNorm":
        return _install_rms_norm()
    if name == "SiluAndMul" and uses_modular_api():
        return _install_silu_and_mul()
    if name not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported MetaX operator route: {name}")
    return install_shared_route(name)


def uninstall(name: str):
    from .. import attention

    if name in attention.ROUTES:
        return attention.uninstall(name, "metax")
    if name == "RoPE":
        return _uninstall_methods("RoPE")
    if name == "RMSNorm":
        return _uninstall_methods("RMSNorm")
    if name == "SiluAndMul" and "SiluAndMul" in _PATCHES:
        return _uninstall_methods("SiluAndMul")
    if name not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported MetaX operator route: {name}")
    return uninstall_shared_route(name)
