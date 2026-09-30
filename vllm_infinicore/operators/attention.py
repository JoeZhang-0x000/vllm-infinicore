"""Vendor-version adapters behind vLLM's opaque attention boundary.

vLLM retains allocation, scheduling, block tables and graph orchestration.
The selected PA/KV operations dispatch to InfiniCore, without retrying a
failed launch through a native operator. Explicitly requested attention routes
fail on unsupported configurations; they never silently select native kernels.
"""

from __future__ import annotations

from functools import wraps
import importlib

import torch

from ..routing.patching import PatchInstallResult, PatchUninstallResult
from ..routing.policy import store_token_limit
from . import attention_ops as ops

ROUTES = frozenset({"StoreKVCache", "PagedAttentionPrefill", "PagedAttentionDecode"})
_TARGETS = {
    "ascend": ("vllm_ascend.attention.attention_v1", "AscendAttentionBackendImpl"),
    "metax": ("vllm_metax.v1.attention.backends.flash_attn", "FlashAttentionImpl"),
    "kunlun": ("vllm_kunlun.v1.attention.backends.kunlun_attn", "KunlunAttentionImpl"),
}
_ACTIVE: set[str] = set()
_PATCHES: list[tuple] = []
_FALLBACKS: dict[str, int] = {}
_NATIVE_CALLS: dict[str, int] = {}
_BACKEND = None


def fallback_counts():
    return dict(_FALLBACKS)


def native_call_counts():
    """Planned native dispatches, excluding graph replay and failed launches."""
    return dict(_NATIVE_CALLS)


def _record_native(name):
    _NATIVE_CALLS[name] = _NATIVE_CALLS.get(name, 0) + 1


def _native_store(backend, num_tokens):
    limit = store_token_limit(backend)
    native = "StoreKVCache" not in _ACTIVE or (limit is not None and num_tokens > limit)
    if native:
        _record_native("StoreKVCache")
    return native


def _unsupported(impl, metadata, output_scale, output_block_scale):
    if output_scale is not None or output_block_scale is not None:
        return "output quantization"
    attn_type = getattr(impl, "attn_type", "decoder")
    if str(getattr(attn_type, "value", attn_type)).lower() != "decoder":
        return "only decoder self attention is supported"
    if getattr(impl, "kv_cache_dtype", "auto") not in ("auto", "float16", "bfloat16"):
        return "quantized KV cache"
    if getattr(impl, "alibi_slopes", None) is not None:
        return "ALiBi"
    if getattr(impl, "sliding_window", None) not in (None, (-1, -1)):
        return "sliding window"
    if getattr(impl, "logits_soft_cap", None) not in (None, 0, 0.0):
        return "logits soft cap"
    if getattr(impl, "sinks", None) is not None:
        return "attention sinks"
    if getattr(impl, "kv_sharing_target_layer_name", None):
        return "shared KV cache"
    if getattr(metadata, "use_cascade", False) or not getattr(metadata, "causal", True):
        return "cascade/noncausal attention"
    if getattr(impl, "dcp_world_size", 1) != 1 or getattr(impl, "pcp_size", 1) != 1:
        return "context parallel attention"
    if getattr(metadata, "kvcomp_metadata", None) is not None:
        return "compressed KV cache"
    return None


def _fallback(reason, native):
    raise NotImplementedError(f"InfiniCore attention: {reason}")


def _forward(original, backend):
    @wraps(original)
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        native = lambda: original(self, layer, query, key, value, kv_cache,
                                  attn_metadata, output, output_scale, output_block_scale)
        if attn_metadata is None:
            return native()  # vLLM's memory profiling has no attention work.
        m = attn_metadata
        n = m.num_actual_tokens
        nd, nt = m.num_decodes, m.num_decode_tokens
        needs = set()
        if n > nt:
            needs.add("PagedAttentionPrefill")
        if nt:
            needs.add("PagedAttentionDecode")
        if not needs.intersection(_ACTIVE):
            for name in needs:
                _record_native(name)
            return native()
        if not needs.issubset(_ACTIVE):
            return _fallback("mixed batch requires both attention routes", native)
        reason = _unsupported(self, m, output_scale, output_block_scale)
        if nt and nt != nd:
            reason = "multi-token speculative decode"
        if reason:
            return _fallback(reason, native)
        q = query.view(-1, self.num_heads, self.head_size)
        if output is None:
            output = torch.empty_like(q)
        out = output.view_as(q)
        k, v = ops.cache_views(kv_cache, self.num_kv_heads)
        # v0.11 Kunlun and Ascend store inside forward. MetaX v0.22 uses
        # do_kv_cache_update through a separate opaque vLLM custom operation.
        if backend != "metax" and key is not None and value is not None:
            if backend == "ascend":
                self.reshape_and_cache(query, key, value, kv_cache, m, output)
            else:
                module = importlib.import_module(_TARGETS[backend][0])
                module.kunlun_ops.reshape_and_cache_flash(
                    key.view(-1, self.num_kv_heads, self.head_size)[:n],
                    value.view(-1, self.num_kv_heads, self.head_size)[:n],
                    k, v, m.slot_mapping[:n], BLHD_LAYOUT=False)
        if backend == "kunlun":
            lengths, blocks = m.seq_lens_tensor, m.block_tables
        elif backend == "ascend":
            lengths, blocks = m._infinicore_seq_lens, m.block_tables
        else:
            lengths, blocks = m.seq_lens, m.block_table
        if nt:
            ops.compute(q[:nt], k, v, blocks[:nd], lengths[:nd], None,
                        self.scale, out[:nt], decode=True)
        if n > nt:
            nr = nd + m.num_prefills
            starts = m.query_start_loc[nd:nr + 1] - nt
            ops.compute(q[nt:n], k, v, blocks[nd:nr], lengths[nd:nr], starts,
                        self.scale, out[nt:n], decode=False)
        return output.view(-1, self.num_heads * self.head_size) if backend == "kunlun" else output
    return forward


def _update(original):
    @wraps(original)
    def update(self, layer, key, value, kv_cache, slot_mapping):
        if _native_store("metax", slot_mapping.numel()):
            return original(self, layer, key, value, kv_cache, slot_mapping)
        reason = _unsupported(self, None, None, None)
        if reason:
            return _fallback(reason, lambda: original(self, layer, key, value, kv_cache, slot_mapping))
        k, v = ops.cache_views(kv_cache, self.num_kv_heads)
        ops.store(k, v, key.view(-1, self.num_kv_heads, self.head_size),
                  value.view(-1, self.num_kv_heads, self.head_size), slot_mapping)
    return update


def _ascend_store(original):
    @wraps(original)
    def store(self, query, key, value, kv_cache, metadata, output):
        if _native_store("ascend", metadata.num_actual_tokens):
            return original(self, query, key, value, kv_cache, metadata, output)
        reason = _unsupported(self, metadata, None, None)
        if reason:
            raise NotImplementedError(reason)
        k, v = ops.cache_views(kv_cache, self.num_kv_heads)
        n = metadata.num_actual_tokens
        ops.store(k, v, key.view(-1, self.num_kv_heads, self.head_size),
                  value.view(-1, self.num_kv_heads, self.head_size), metadata.slot_mapping[:n])
        self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        from vllm_ascend.attention.attention_v1 import notify_kv_cache_written
        notify_kv_cache_written()
        return query, key, value, output
    return store


class _KunlunOps:
    """Replace only the module-local cache call, including store-only routing."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def reshape_and_cache_flash(self, key, value, key_cache, value_cache,
                                slot_mapping, BLHD_LAYOUT=False):
        if _native_store("kunlun", slot_mapping.numel()):
            return self.original.reshape_and_cache_flash(
                key, value, key_cache, value_cache, slot_mapping, BLHD_LAYOUT=BLHD_LAYOUT)
        if BLHD_LAYOUT:
            key_cache, value_cache = key_cache.transpose(1, 2), value_cache.transpose(1, 2)
        return ops.store(key_cache, value_cache, key, value, slot_mapping)


def _ascend_build(original):
    @wraps(original)
    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        m = original(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        # Native Ascend attention uses host lengths with graph task updates.
        # InfiniCore kernels read the model runner's persistent device buffers.
        nr = common_attn_metadata.num_reqs
        m._infinicore_seq_lens = common_attn_metadata.seq_lens[:nr]
        m.query_start_loc = common_attn_metadata.query_start_loc[:nr + 1]
        return m
    return build


def _ascend_graph_update(original):
    @wraps(original)
    def update(*args, **kwargs):
        if {"PagedAttentionPrefill", "PagedAttentionDecode"}.issubset(_ACTIVE):
            return  # InfiniCore consumes device metadata, no FIA task handles.
        return original(*args, **kwargs)
    return staticmethod(update)


def _patch(cls, name, wrapper):
    original_descriptor = vars(cls).get(name)
    original = getattr(cls, name)
    replacement = wrapper(original)
    setattr(cls, name, replacement)
    _PATCHES.append((cls, name, original_descriptor, replacement))


def install(name, backend):
    global _BACKEND
    if name not in ROUTES:
        raise ValueError(name)
    if _BACKEND is not None and _BACKEND != backend:
        raise RuntimeError("cannot change attention backend while routes are installed")
    ops.initialize()  # Fail before modifying vLLM on missing symbols/builds.
    try:
        if not _PATCHES:
            if backend == "ascend":
                # Ascend 0.23's attention imports DeviceOperator, whose package
                # initialization also imports MoE. Initialize the ops package first
                # to avoid entering it through a partially initialized MoE module.
                importlib.import_module("vllm_ascend.ops")
            module_name, class_name = _TARGETS[backend]
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            _patch(cls, "forward", lambda original: _forward(original, backend))
            if hasattr(cls, "do_kv_cache_update"):
                _patch(cls, "do_kv_cache_update", _update)
            if backend == "ascend":
                _patch(cls, "reshape_and_cache", _ascend_store)
                _patch(module.AscendAttentionMetadataBuilder, "build", _ascend_build)
                _patch(cls, "update_graph_params", _ascend_graph_update)
            elif backend == "kunlun":
                _patch(module, "kunlun_ops", _KunlunOps)
            _BACKEND = backend
    except Exception:
        for cls, method, original, replacement in reversed(_PATCHES):
            if vars(cls).get(method) is replacement:
                if original is None:
                    delattr(cls, method)
                else:
                    setattr(cls, method, original)
        _PATCHES.clear()
        _BACKEND = None
        raise
    _ACTIVE.add(name)
    return PatchInstallResult(True, f"InfiniCore {backend} {name} C API on current stream")


def uninstall(name, backend):
    global _BACKEND
    if name not in _ACTIVE:
        return PatchUninstallResult(False, "attention route not installed")
    if len(_ACTIVE) == 1:
        if any(vars(cls).get(method) is not replacement for cls, method, _, replacement in _PATCHES):
            return PatchUninstallResult(False, "attention method changed by another patch")
        for cls, method, original, _ in reversed(_PATCHES):
            if original is None:
                delattr(cls, method)
            else:
                setattr(cls, method, original)
        _PATCHES.clear()
        _BACKEND = None
    _ACTIVE.remove(name)
    return PatchUninstallResult(True, "attention route removed")
