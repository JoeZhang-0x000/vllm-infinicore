"""Identify the vendor platform that owns the vLLM runtime.

The InfiniCore plugin is a general plugin. It never competes with a vendor
platform plugin during platform discovery, and this module does not import
torch, vLLM, or vendor packages.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys


_PLUGIN_NAMES = {
    "ascend": "ascend",
    "metax": "metax",
    "kunlun": "kunlun",
}
_MODULE_PREFIXES = {
    "vllm_ascend": "ascend",
    "vllm_metax": "metax",
    "vllm_kunlun": "kunlun",
}


@dataclass(frozen=True)
class PlatformSelection:
    name: str
    source: str


def selected_platform() -> PlatformSelection:
    """Prefer vLLM's resolved platform, then its explicit plugin allowlist."""

    platforms = sys.modules.get("vllm.platforms")
    current = vars(platforms).get("_current_platform") if platforms else None
    unresolved = current is not None and (
        type(current).__name__ == "UnspecifiedPlatform"
        or getattr(current, "__name__", "") == "UnspecifiedPlatform"
    )
    if current is not None and not unresolved:
        module = getattr(current, "__module__", type(current).__module__)
        for prefix, name in _MODULE_PREFIXES.items():
            if module == prefix or module.startswith(prefix + "."):
                return PlatformSelection(name, "resolved vLLM platform")
        device_name = str(getattr(current, "device_name", "")).lower()
        device_type = str(getattr(current, "device_type", "")).lower()
        if device_type == "npu":
            return PlatformSelection("ascend", "resolved NPU platform")
        if device_name in ("maca", "metax"):
            return PlatformSelection("metax", "resolved MetaX platform")
        if device_name in ("kunlun", "xpu"):
            return PlatformSelection("kunlun", "resolved Kunlun platform")
        return PlatformSelection("unknown", "resolved non-vendor platform")

    allowlist = os.environ.get("VLLM_PLUGINS", "")
    names = {item.strip() for item in allowlist.split(",") if item.strip()}
    vendors = {vendor for plugin, vendor in _PLUGIN_NAMES.items() if plugin in names}
    if len(vendors) == 1:
        return PlatformSelection(vendors.pop(), "VLLM_PLUGINS")
    if vendors:
        return PlatformSelection("unknown", "ambiguous VLLM_PLUGINS")
    return PlatformSelection("unknown", "no vendor platform selected")


def ascend_platform_selected() -> bool:
    return selected_platform().name == "ascend"


__all__ = ["PlatformSelection", "ascend_platform_selected", "selected_platform"]
