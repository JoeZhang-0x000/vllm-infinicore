"""InfiniCore PA/KV launches; caches use [block, head, token, dimension].

No vendor attention operators or PyTorch attention fallbacks live here. Device
metadata is consumed on the current stream, including during graph replay.
"""

from __future__ import annotations

import torch

from .selection import selected_backend

_COUNTS: dict[str, int] = {}


def call_counts():
    return dict(_COUNTS)


def _record(name):
    _COUNTS[name] = _COUNTS.get(name, 0) + 1


def initialize():
    if selected_backend() == "ascend":
        from .ascend.backend import attention_library
        attention_library()
    else:
        from .cpp_bridge import module
        bridge = module()
        for name in ("store_kv_cache_current_stream",
                     "paged_attention_prefill_current_stream", "paged_attention_decode_out"):
            getattr(bridge, name)


def cache_views(kv_cache, num_kv_heads):
    k, v = kv_cache[0], kv_cache[1]
    if k.ndim != 4 or v.ndim != 4:
        raise NotImplementedError("InfiniCore attention requires unquantized 4D K/V caches")
    if selected_backend() != "kunlun":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    if k.shape[1] != num_kv_heads or v.shape[1] != num_kv_heads:
        raise ValueError("KV cache head count/layout does not match attention")
    return k, v


def store(k, v, key, value, slots):
    slots = slots.reshape(-1).to(dtype=torch.int64)
    n = slots.numel()
    key, value = key[:n], value[:n]
    if key.shape[0] != n or value.shape[0] != n:
        raise ValueError("slot_mapping exceeds key/value token count")
    if not n:
        return
    if selected_backend() == "ascend":
        from .ascend.backend import launch
        launch("PagedCaching", (k, v, key, value, slots))
    else:
        from .cpp_bridge import module
        module().store_kv_cache_current_stream(k, v, key, value, slots)
    _record("StoreKVCache")


def compute(query, k, v, blocks, lengths, starts, scale, output, *, decode):
    # All three metadata arrays must share the index dtype. int32 is supported
    # by these InfiniCore revisions and avoids redundant copies in vLLM.
    blocks = blocks.to(dtype=torch.int32)
    lengths = lengths.to(device=query.device, dtype=torch.int32, non_blocking=True)
    starts = starts.to(dtype=torch.int32) if starts is not None else None
    if selected_backend() == "ascend":
        from .ascend.backend import launch
        if decode:
            launch("PagedAttention", (output, query, k, v, blocks, lengths, None), (scale,))
        else:
            launch("PagedAttentionPrefill", (output, query, k, v, blocks, lengths, starts, None), (scale,))
    else:
        from .cpp_bridge import module
        if decode:
            n = query.shape[0]
            module().paged_attention_decode_out(query, k, v, lengths, blocks,
                                                None, scale, n, n, output)
        else:
            module().paged_attention_prefill_current_stream(
                query, k, v, blocks, lengths, starts, None, scale, output)
    _record("PagedAttentionDecode" if decode else "PagedAttentionPrefill")
