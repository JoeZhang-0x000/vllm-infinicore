"""Lightweight platform capabilities, without importing torch or vLLM.

Ascend owns its OOT classes and attention layout. An InfiniCore kernel existing
upstream does not imply that this plugin has a compatible adapter for it.
Unsupported adapters must keep the platform implementation intact.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import MappingProxyType


ASCEND_TENSOR_BRIDGE_UNAVAILABLE = "InfiniCore NPU adapter is not supported without VLLM_INFINICORE_ASCEND_LIBRARY; using native NPU ops"
_ASCEND_OOT_UNAVAILABLE = "InfiniCore Ascend adapter is not supported without VLLM_INFINICORE_ASCEND_LIBRARY; keeping vLLM-Ascend's class"
_ASCEND_ATTENTION_UNAVAILABLE = (
    "InfiniCore Ascend attention/KV-cache adapter is not supported; "
    "keeping vLLM-Ascend's attention backend"
)

ASCEND_NATIVE_FALLBACK_REASONS = MappingProxyType(
    {
        "RMSNorm": _ASCEND_OOT_UNAVAILABLE,
        "SiluAndMul": _ASCEND_OOT_UNAVAILABLE,
        "RoPE": _ASCEND_OOT_UNAVAILABLE,
        "Embedding": ASCEND_TENSOR_BRIDGE_UNAVAILABLE,
        "MatMul": ASCEND_TENSOR_BRIDGE_UNAVAILABLE,
        "LMHead": ASCEND_TENSOR_BRIDGE_UNAVAILABLE,
        "StoreKVCache": _ASCEND_ATTENTION_UNAVAILABLE,
        "PagedAttentionPrefill": _ASCEND_ATTENTION_UNAVAILABLE,
        "PagedAttentionDecode": _ASCEND_ATTENTION_UNAVAILABLE,
    }
)


def ascend_platform_selected() -> bool:
    """Inspect selection without triggering vLLM's lazy platform discovery.

    An already resolved platform wins. Before resolution, honor an explicit
    plugin allowlist; only use package discovery when vLLM is auto-discovering
    plugins.
    """

    platforms = sys.modules.get("vllm.platforms")
    current = (
        vars(platforms).get("_current_platform") if platforms is not None else None
    )
    if current is not None:
        return getattr(current, "device_type", None) == "npu"
    plugins = os.environ.get("VLLM_PLUGINS")
    if plugins is not None:
        return "ascend" in {name.strip() for name in plugins.split(",")}
    return _package_available("vllm_ascend") and _package_available("torch_npu")


def _package_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def ascend_native_fallback_reasons():
    """Library presence is checked without importing frameworks during discovery."""
    if not os.environ.get("VLLM_INFINICORE_ASCEND_LIBRARY"):
        return dict(ASCEND_NATIVE_FALLBACK_REASONS)
    return {
        name: reason
        for name, reason in ASCEND_NATIVE_FALLBACK_REASONS.items()
        if name in {"StoreKVCache", "PagedAttentionPrefill", "PagedAttentionDecode"}
    }
