"""MetaX vLLM operator routes."""

from ..routes import install_shared_route, uninstall_shared_route
from . import SUPPORTED_ROUTES

_ROPE_PATCH = None
_RMS_PATCH = None


def _install_rms_norm():
    """Route RMSNorm on the vendor-owned class without an OOT name collision."""
    global _RMS_PATCH
    import torch
    from vllm_metax.customized.ops.layernorm import MacaRMSNorm
    from ..custom_ops import RMS_NORM_OP, FUSED_ADD_RMS_NORM_OP, load_custom_ops
    from ...routing.patching import PatchInstallResult

    if _RMS_PATCH is not None:
        return PatchInstallResult(True, "MetaX RMSNorm adapter already installed")
    status = load_custom_ops(force=True, required_ops=(RMS_NORM_OP, FUSED_ADD_RMS_NORM_OP))
    if not status.available:
        return PatchInstallResult(False, status.reason)
    native = MacaRMSNorm.forward_native
    originals = {name: vars(MacaRMSNorm).get(name)
                 for name in ("forward_oot", "forward_native")}

    def forward(self, x, residual=None):
        if (not self.has_weight or self.variance_size_override is not None
                or (residual is not None and not getattr(self, "pass_weight_add", True))):
            # Call the captured implementation directly: forward_oot delegates
            # to forward_cuda, which calls forward_native again on this version.
            return native(self, x, residual)
        if residual is not None:
            return torch.ops.vllm_infinicore.fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon)
        return torch.ops.vllm_infinicore.rms_norm(
            x, self.weight.data, self.variance_epsilon)

    for name in originals:
        setattr(MacaRMSNorm, name, forward)
    _RMS_PATCH = (MacaRMSNorm, originals, forward)
    return PatchInstallResult(True, "InfiniCore plain and supported fused MetaX RMSNorm")


def _uninstall_rms_norm():
    global _RMS_PATCH
    from ...routing.patching import PatchUninstallResult

    if _RMS_PATCH is None:
        return PatchUninstallResult(False, "MetaX RMSNorm adapter not installed")
    cls, originals, forward = _RMS_PATCH
    if any(getattr(cls, name) is not forward for name in originals):
        return PatchUninstallResult(False, "RMSNorm method changed by another patch")
    for name, original in originals.items():
        if original is None:
            delattr(cls, name)
        else:
            setattr(cls, name, original)
    _RMS_PATCH = None
    return PatchUninstallResult(True, "original MetaX RMSNorm restored")


def _install_rope():
    """Wrap the vendor-owned OOT class; never compete for its registry name."""
    global _ROPE_PATCH
    from functools import wraps
    import torch
    from vllm_metax.customized.ops.rotary_embedding import MacaRotaryEmbedding
    from ..custom_ops import ROTARY_EMBEDDING_OP, load_custom_ops
    from ...routing.patching import PatchInstallResult

    if _ROPE_PATCH is not None:
        return PatchInstallResult(True, "MetaX RoPE adapter already installed")
    status = load_custom_ops(force=True, required_ops=(ROTARY_EMBEDDING_OP,))
    if not status.available:
        return PatchInstallResult(False, status.reason)
    original = MacaRotaryEmbedding.forward_oot
    originals = {name: vars(MacaRotaryEmbedding).get(name)
                 for name in ("forward_oot", "forward_native")}

    @wraps(original)
    def forward(self, positions, query, key=None):
        cache = self._match_cos_sin_cache_dtype(query)
        return torch.ops.vllm_infinicore.rotary_embedding(
            positions, query, key, self.head_size, self.rotary_dim,
            cache, self.is_neox_style)

    # Inductor defaults custom_ops to "none", which dispatches directly to
    # forward_native and otherwise bypasses the registered OOT method.
    for name in originals:
        setattr(MacaRotaryEmbedding, name, forward)
    _ROPE_PATCH = (MacaRotaryEmbedding, originals, forward)
    return PatchInstallResult(True, "InfiniCore RoPE on MetaX-owned OOT class")


def _uninstall_rope():
    global _ROPE_PATCH
    from ...routing.patching import PatchUninstallResult

    if _ROPE_PATCH is None:
        return PatchUninstallResult(False, "MetaX RoPE adapter not installed")
    cls, originals, forward = _ROPE_PATCH
    if any(getattr(cls, name) is not forward for name in originals):
        return PatchUninstallResult(False, "RoPE method changed by another patch")
    for name, original in originals.items():
        if original is None:
            delattr(cls, name)
        else:
            setattr(cls, name, original)
    _ROPE_PATCH = None
    return PatchUninstallResult(True, "original MetaX RoPE restored")


def install(name: str):
    from .. import attention

    if name in attention.ROUTES:
        return attention.install(name, "metax")
    if name == "RoPE":
        return _install_rope()
    if name == "RMSNorm":
        return _install_rms_norm()
    if name not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported MetaX operator route: {name}")
    return install_shared_route(name)


def uninstall(name: str):
    from .. import attention

    if name in attention.ROUTES:
        return attention.uninstall(name, "metax")
    if name == "RoPE":
        return _uninstall_rope()
    if name == "RMSNorm":
        return _uninstall_rms_norm()
    if name not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported MetaX operator route: {name}")
    return uninstall_shared_route(name)
