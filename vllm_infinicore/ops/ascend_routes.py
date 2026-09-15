"""Adapt Ascend-owned classes without registering competing OOT classes.

Each route wraps one Ascend or vLLM method and picks between three
implementations per call: the traceable operator from `ascend_graph_ops`, an
eager InfiniCore launch, or the original native method. `_dispatch` holds that
choice in one place so every route makes it the same way.
"""

from __future__ import annotations

from functools import wraps
import importlib

import torch

from ..patching import PatchInstallResult, PatchUninstallResult
from . import ascend_backend as backend
from . import ascend_graph_ops as graph_ops

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


def _prefetch():
    from vllm_ascend.utils import get_weight_prefetch_method

    return get_weight_prefetch_method()


def _dispatch(name, tensor, checks, traced, eager, native):
    """Route one call to the traced operator, the eager launch, or native.

    `checks` are `(supported, reason)` pairs. They are all evaluated before the
    call is known to be eligible, so every predicate must answer for any tensor
    rather than raise. They mean
    different things on each path. While tracing, an unsupported call must
    select the native operator here: a compiled graph fixes its operators and
    cannot choose per call, so raising inside it is not an option. In eager
    execution the same failure becomes an `Unsupported`, which `execute`
    records as a fallback with its reason.
    """
    unsupported = next((reason for supported, reason in checks if not supported), None)
    if torch.compiler.is_compiling():
        return native() if unsupported else traced()

    def run():
        if unsupported:
            raise backend.Unsupported(unsupported)
        return eager()

    return backend.execute(name, tensor, run, native)


def _rms_norm(original):
    @wraps(original)
    def rms(self, x, residual=None):
        native = lambda: original(self, x, residual)
        if residual is not None:
            return backend.fallback(
                "fused_add_rms_norm",
                "InfiniCore has no Ascend fused Add+RMSNorm kernel",
                native,
            )

        def finish(y):
            if self.bias_loaded:
                y = y + self.bias
            _prefetch().maybe_prefetch_mlp_weight_postprocess(y)
            return y

        return _dispatch(
            "rms_norm",
            x,
            [
                (
                    getattr(self, "variance_size_override", None)
                    in (None, x.shape[-1]),
                    "partial RMSNorm variance",
                ),
                backend.supports_tensor(x),
                backend.supports_tensor(self.weight),
            ],
            traced=lambda: finish(
                graph_ops.rms_norm(x, self.weight, self.variance_epsilon)
            ),
            eager=lambda: finish(
                backend.rms_norm(x, self.weight, self.variance_epsilon)
            ),
            native=native,
        )

    return rms


def _silu_and_mul(original):
    @wraps(original)
    def silu(self, x):
        native = lambda: original(self, x)

        def prefetched(launch):
            prefetch = _prefetch()
            prefetch.maybe_prefetch_mlp_weight_preprocess(prefetch.MLP_DOWN, x)
            y = launch()
            prefetch.maybe_prefetch_mlp_weight_postprocess(y)
            return y

        return _dispatch(
            "silu_and_mul",
            x,
            [backend.supports_tensor(x), backend.supports_silu_and_mul(x)],
            traced=lambda: prefetched(lambda: graph_ops.silu_and_mul(x)),
            eager=lambda: prefetched(lambda: backend.silu_and_mul(x)),
            native=native,
        )

    return silu


def _rotary_embedding(original):
    @wraps(original)
    def rope(self, positions, query, key, offsets=None, is_neox_style_override=None):
        native = lambda: original(
            self, positions, query, key, offsets, is_neox_style_override
        )
        if torch.compiler.is_compiling() and key is None:
            # The operator returns two tensors, so a missing key cannot be
            # expressed in its schema and stays on the native path.
            return native()
        neox = (
            self.is_neox_style
            if is_neox_style_override is None
            else is_neox_style_override
        )
        args = (
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            self.cos_sin_cache,
            neox,
        )
        checks = [
            (
                offsets is None and not getattr(self, "use_mtp", False),
                "offset/MTP RoPE retains Ascend orchestration",
            ),
            backend.supports_tensor(positions),
            backend.supports_tensor(query),
            backend.supports_rotary_embedding(
                positions, self.head_size, self.rotary_dim
            ),
        ]
        if key is not None:
            checks.append(backend.supports_tensor(key))
        return _dispatch(
            "rotary_embedding",
            query,
            checks,
            traced=lambda: graph_ops.rotary_embedding(*args),
            eager=lambda: backend.rotary_embedding(*args),
            native=native,
        )

    return rope


def _embedding(original):
    @wraps(original)
    def embedding(self, layer, input_):
        native = lambda: original(self, layer, input_)
        return _dispatch(
            "embedding",
            input_,
            [backend.supports_tensor(input_), backend.supports_tensor(layer.weight)],
            traced=lambda: graph_ops.embedding(input_, layer.weight),
            eager=lambda: backend.embedding(input_, layer.weight),
            native=native,
        )

    return embedding


def _linear(original, name):
    @wraps(original)
    def linear(self, layer, x, bias=None):
        native = lambda: original(self, layer, x, bias)
        return _dispatch(
            name,
            x,
            [
                backend.supports_tensor(x),
                backend.supports_tensor(layer.weight),
                backend.supports_linear(x),
            ],
            traced=lambda: graph_ops.linear(x, layer.weight, bias, name),
            eager=lambda: backend.linear(x, layer.weight, bias),
            native=native,
        )

    return linear


_WRAPPERS = {
    "RMSNorm": _rms_norm,
    "SiluAndMul": _silu_and_mul,
    "RoPE": _rotary_embedding,
    "Embedding": _embedding,
    "MatMul": lambda original: _linear(original, "linear"),
    "LMHead": lambda original: _linear(original, "lm_head"),
}


def _wrapper(route, original):
    return _WRAPPERS[route](original)


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
