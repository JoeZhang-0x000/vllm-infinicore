"""Adapt Ascend-owned classes without registering competing OOT classes."""

from __future__ import annotations

from functools import wraps
import importlib

from ..patching import PatchInstallResult, PatchUninstallResult
from . import ascend_backend as backend

_TARGETS = {
    "RMSNorm": ("vllm_ascend.ops.layernorm", "AscendRMSNorm", "forward_oot"),
    "SiluAndMul": ("vllm_ascend.ops.activation", "AscendSiluAndMul", "forward_oot"),
    "RoPE": (
        "vllm_ascend.ops.rotary_embedding",
        "AscendRotaryEmbedding",
        "forward_oot",
    ),
    "Embedding": (
        "vllm.model_executor.layers.vocab_parallel_embedding",
        "UnquantizedEmbeddingMethod",
        "embedding",
    ),
    "MatMul": ("vllm_ascend.ops.linear", "AscendUnquantizedLinearMethod", "apply"),
    "LMHead": (
        "vllm.model_executor.layers.vocab_parallel_embedding",
        "UnquantizedEmbeddingMethod",
        "apply",
    ),
}
_PATCHES = {}


def _wrapper(route, original):
    if route == "RMSNorm":

        @wraps(original)
        def rms(self, x, residual=None):
            native = lambda: original(self, x, residual)
            if residual is not None:
                return backend.fallback(
                    "fused_add_rms_norm",
                    "InfiniCore has no Ascend fused Add+RMSNorm kernel",
                    native,
                )

            def run():
                if getattr(self, "variance_size_override", None) not in (
                    None,
                    x.shape[-1],
                ):
                    raise backend.Unsupported("partial RMSNorm variance")
                y = backend.rms_norm(x, self.weight, self.variance_epsilon)
                if self.bias_loaded:
                    y.add_(self.bias)
                from vllm_ascend.utils import get_weight_prefetch_method

                get_weight_prefetch_method().maybe_prefetch_mlp_weight_postprocess(y)
                return y

            return backend.execute("rms_norm", x, run, native)

        return rms
    if route == "SiluAndMul":

        @wraps(original)
        def silu(self, x):
            def run():
                from vllm_ascend.utils import get_weight_prefetch_method

                prefetch = get_weight_prefetch_method()
                prefetch.maybe_prefetch_mlp_weight_preprocess(prefetch.MLP_DOWN, x)
                y = backend.silu_and_mul(x)
                prefetch.maybe_prefetch_mlp_weight_postprocess(y)
                return y

            return backend.execute("silu_and_mul", x, run, lambda: original(self, x))

        return silu
    if route == "RoPE":

        @wraps(original)
        def rope(
            self, positions, query, key, offsets=None, is_neox_style_override=None
        ):
            def run():
                if offsets is not None or getattr(self, "use_mtp", False):
                    raise backend.Unsupported(
                        "offset/MTP RoPE retains Ascend orchestration"
                    )
                neox = (
                    self.is_neox_style
                    if is_neox_style_override is None
                    else is_neox_style_override
                )
                return backend.rotary_embedding(
                    positions,
                    query,
                    key,
                    self.head_size,
                    self.rotary_dim,
                    self.cos_sin_cache,
                    neox,
                )

            return backend.execute(
                "rotary_embedding",
                query,
                run,
                lambda: original(
                    self, positions, query, key, offsets, is_neox_style_override
                ),
            )

        return rope
    if route == "Embedding":

        @wraps(original)
        def embedding(self, layer, input_):
            return backend.execute(
                "embedding",
                input_,
                lambda: backend.embedding(input_, layer.weight),
                lambda: original(self, layer, input_),
            )

        return embedding

    @wraps(original)
    def linear(self, layer, x, bias=None):
        name = "linear" if route == "MatMul" else "lm_head"
        return backend.execute(
            name,
            x,
            lambda: backend.linear(x, layer.weight, bias),
            lambda: original(self, layer, x, bias),
        )

    return linear


def install(route):
    backend.library()  # Reject a mismatched lock/ABI before modifying any class.
    if route in _PATCHES:
        return PatchInstallResult(True, "Ascend adapter already installed")
    module, name, method = _TARGETS[route]
    cls = getattr(importlib.import_module(module), name)
    original = getattr(cls, method)
    wrapper = _wrapper(route, original)
    inherited = method not in vars(cls)
    setattr(cls, method, wrapper)
    _PATCHES[route] = (cls, method, original, wrapper, inherited)
    return PatchInstallResult(
        True,
        f"InfiniCore Ascend {route} adapter; original Ascend method retained for unsupported cases",
    )


def uninstall(route):
    patch = _PATCHES.get(route)
    if patch is None:
        return PatchUninstallResult(False, "Ascend adapter not installed")
    cls, method, original, wrapper, inherited = patch
    if getattr(cls, method) is not wrapper:
        return PatchUninstallResult(
            False, "method changed by another patch; refusing to overwrite it"
        )
    if inherited:
        delattr(cls, method)
    else:
        setattr(cls, method, original)
    del _PATCHES[route]
    return PatchUninstallResult(True, "original Ascend method restored")
