"""InfiniCore attention backend route manager.

This thin module manages the route lifecycle for InfiniCore attention
operators.  The actual backend implementation lives in
``vllm_infinicore.ops.infinicore_attention``, which registers itself as
vLLM's ``FLASH_ATTN`` backend on import via the ``@register_backend``
decorator.  No dependence on ``vllm_metax`` is required.
"""

from __future__ import annotations

from dataclasses import dataclass

INFINICORE_ATTENTION_BACKEND_ROUTES = (
    "StoreKVCache",
    "PagedAttentionPrefill",
    "PagedAttentionDecode",
)

_ACTIVE_ROUTES: set[str] = set()
_REGISTERED = False
_ATTENTION_COUNTS: dict[str, int] = {}


@dataclass(frozen=True)
class InfiniCoreAttentionBackendStatus:
    installed: bool
    reason: str
    route_name: str


@dataclass(frozen=True)
class InfiniCoreAttentionBackendUninstallStatus:
    uninstalled: bool
    reason: str
    route_name: str


def install_infinicore_attention_backend(
    route_name: str,
) -> InfiniCoreAttentionBackendStatus:
    if route_name not in INFINICORE_ATTENTION_BACKEND_ROUTES:
        return InfiniCoreAttentionBackendStatus(
            installed=False,
            reason=f"unsupported attention backend route {route_name}",
            route_name=route_name,
        )

    _ACTIVE_ROUTES.add(route_name)
    _register_backend_once()
    return InfiniCoreAttentionBackendStatus(
        installed=True,
        reason=(
            "InfiniCore FLASH_ATTN backend active; "
            f"route {route_name} enabled"
        ),
        route_name=route_name,
    )


def uninstall_infinicore_attention_backend(
    route_name: str,
) -> InfiniCoreAttentionBackendUninstallStatus:
    if route_name not in _ACTIVE_ROUTES:
        return InfiniCoreAttentionBackendUninstallStatus(
            uninstalled=False,
            reason=f"InfiniCore attention backend route {route_name} not installed",
            route_name=route_name,
        )
    _ACTIVE_ROUTES.remove(route_name)
    return InfiniCoreAttentionBackendUninstallStatus(
        uninstalled=True,
        reason=f"InfiniCore attention backend route {route_name} disabled",
        route_name=route_name,
    )


def attention_backend_route_counts() -> dict[str, int]:
    return dict(_ATTENTION_COUNTS)


def reset_attention_backend_route_counts() -> None:
    _ATTENTION_COUNTS.clear()


def active_routes() -> frozenset[str]:
    return frozenset(_ACTIVE_ROUTES)


def _register_backend_once() -> None:
    """Register the InfiniCore FLASH_ATTN backend override.

    First ensures the platform-level MetaX fallback backends are registered
    (via :func:`vllm_infinicore.platform.register_attention_backends`), then
    overwrites ``FLASH_ATTN`` with the InfiniCore implementation.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from vllm_infinicore.platform import register_attention_backends
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_attention_backends()
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "vllm_infinicore.ops.infinicore_attention.InfiniCoreFlashAttentionBackend",
    )
    _REGISTERED = True


__all__ = [
    "INFINICORE_ATTENTION_BACKEND_ROUTES",
    "InfiniCoreAttentionBackendStatus",
    "InfiniCoreAttentionBackendUninstallStatus",
    "active_routes",
    "attention_backend_route_counts",
    "install_infinicore_attention_backend",
    "reset_attention_backend_route_counts",
    "uninstall_infinicore_attention_backend",
]
