"""vLLM attention backend override for InfiniCore PA/KV kernels.

This module registers an AttentionBackend-level implementation instead of
monkey-patching an existing backend class. The first implementation deliberately
reuses the platform FlashAttention metadata builder and KV cache layout, then
routes supported decoder KV update/paged-attention calls through InfiniCore.
Unsupported paths fall back to the platform backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import torch

from . import infinicore_backend

INFINICORE_ATTENTION_BACKEND_ROUTES = (
    "StoreKVCache",
    "PagedAttentionPrefill",
    "PagedAttentionDecode",
)

_ACTIVE_ROUTES: set[str] = set()
_REGISTERED = False
_PATCHED_METAX_REGISTER = False
_ATTENTION_COUNTS: dict[str, int] = {}


def _load_platform_flash_attention() -> tuple[type, type, type, type]:
    """Return platform FlashAttention backend classes.

    MetaX overrides vLLM's FLASH_ATTN backend in its platform package. Prefer
    that implementation when available so metadata and KV-cache layout remain
    compatible with the local vLLM runtime.
    """

    try:
        from vllm_metax.v1.attention.backends.flash_attn import (
            FlashAttentionImpl,
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
            MacaFlashAttentionBackend,
        )

        return (
            MacaFlashAttentionBackend,
            FlashAttentionImpl,
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
        )
    except Exception:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
            FlashAttentionImpl,
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
        )

        return (
            FlashAttentionBackend,
            FlashAttentionImpl,
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
        )


(
    _BaseFlashAttentionBackend,
    _BaseFlashAttentionImpl,
    FlashAttentionMetadata,
    _FlashAttentionMetadataBuilder,
) = _load_platform_flash_attention()


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


class InfiniCoreFlashAttentionBackend(_BaseFlashAttentionBackend):
    """FLASH_ATTN backend override using InfiniCore for supported PA/KV paths."""

    @staticmethod
    def get_impl_cls() -> type["InfiniCoreFlashAttentionImpl"]:
        return InfiniCoreFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type:
        return _FlashAttentionMetadataBuilder


class InfiniCoreFlashAttentionImpl(_BaseFlashAttentionImpl):
    """Attention impl with backend-level InfiniCore dispatch."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not _should_use_infinicore_forward(
            self,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        ):
            _record_attention("backend_forward_fallback")
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        assert output is not None
        try:
            num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
            num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
            num_actual_tokens = int(
                getattr(attn_metadata, "num_actual_tokens", query.shape[0])
            )
            num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
            decode_as_prefill = _decode_as_prefill(
                num_decode_tokens,
                num_decodes,
                num_prefills,
                num_actual_tokens,
            )
            needs_prefill = (
                decode_as_prefill
                or num_prefills > 0
                or num_actual_tokens > num_decode_tokens
            )
            needs_decode = (not decode_as_prefill) and (
                num_decodes > 0 or num_decode_tokens > 0
            )

            if needs_prefill:
                _record_attention("backend_prefill_infinicore")
                if decode_as_prefill:
                    infinicore_backend.paged_attention_decode_as_prefill_infinicore_only(
                        self,
                        query,
                        key,
                        kv_cache,
                        attn_metadata,
                        output,
                    )
                else:
                    infinicore_backend.paged_attention_prefill_infinicore_only(
                        self,
                        query,
                        key,
                        kv_cache,
                        attn_metadata,
                        output,
                    )
            if needs_decode:
                _record_attention("backend_decode_infinicore")
                infinicore_backend.paged_attention_decode_infinicore_only(
                    self,
                    query,
                    key,
                    kv_cache,
                    attn_metadata,
                    output,
                )
            _record_attention("backend_forward_infinicore")
            return output
        except Exception:
            if infinicore_backend.strict_backend_enabled():
                raise
            _record_attention("backend_forward_fallback.exception")
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if not _should_use_infinicore_kv_update(self, key, value, kv_cache, slot_mapping):
            _record_attention("backend_kv_update_fallback")
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        try:
            infinicore_backend.store_kv_cache(kv_cache, key, value, slot_mapping)
            _record_attention("backend_kv_update_infinicore")
        except Exception:
            if infinicore_backend.strict_backend_enabled():
                raise
            _record_attention("backend_kv_update_fallback.exception")
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)


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
            "InfiniCore FLASH_ATTN backend override active; "
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


def _register_backend_once() -> None:
    global _REGISTERED
    _patch_metax_backend_registration()
    _register_infinicore_backend_path()
    _REGISTERED = True


def _register_infinicore_backend_path() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend",
    )


def _patch_metax_backend_registration() -> None:
    global _PATCHED_METAX_REGISTER
    if _PATCHED_METAX_REGISTER:
        return
    # vllm_metax registers its FLASH_ATTN override from a module-level
    # decorator and from platform.register_attention_backends(). Ensure both
    # registration paths run before our override, then re-apply the InfiniCore
    # override after vllm_metax refreshes its backend table.
    try:
        importlib.import_module("vllm_metax.v1.attention.backends.flash_attn")
        metax_platform = importlib.import_module("vllm_metax.platform")
    except Exception:
        _PATCHED_METAX_REGISTER = True
        return

    original = getattr(metax_platform, "register_attention_backends", None)
    if original is None or getattr(original, "_vllm_infinicore_wrapped", False):
        _PATCHED_METAX_REGISTER = True
        return

    def register_attention_backends_with_infinicore() -> None:
        original()
        _register_infinicore_backend_path()

    register_attention_backends_with_infinicore._vllm_infinicore_wrapped = True  # type: ignore[attr-defined]
    metax_platform.register_attention_backends = register_attention_backends_with_infinicore
    _PATCHED_METAX_REGISTER = True


def _should_use_infinicore_forward(
    attn_impl: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    output: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    output_block_scale: torch.Tensor | None,
) -> bool:
    if output is None or attn_metadata is None:
        return False
    if output_scale is not None or output_block_scale is not None:
        return False
    if getattr(attn_metadata, "use_cascade", False):
        return False
    if getattr(attn_impl, "kv_sharing_target_layer_name", None) is not None:
        return False
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        return False
    if not _valid_kv_cache(kv_cache):
        return False
    if not infinicore_backend.real_backend_enabled(query):
        return False

    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]))
    num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
    decode_as_prefill = _decode_as_prefill(
        num_decode_tokens,
        num_decodes,
        num_prefills,
        num_actual_tokens,
    )
    needs_prefill = (
        decode_as_prefill
        or num_prefills > 0
        or num_actual_tokens > num_decode_tokens
    )
    needs_decode = (not decode_as_prefill) and (
        num_decodes > 0 or num_decode_tokens > 0
    )
    if needs_prefill and "PagedAttentionPrefill" not in _ACTIVE_ROUTES:
        return False
    if needs_decode and "PagedAttentionDecode" not in _ACTIVE_ROUTES:
        return False
    if needs_decode and num_decode_tokens != num_decodes:
        return False
    return needs_prefill or needs_decode


def _should_use_infinicore_kv_update(
    attn_impl: object,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor | None,
) -> bool:
    return (
        "StoreKVCache" in _ACTIVE_ROUTES
        and getattr(attn_impl, "kv_sharing_target_layer_name", None) is None
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
        and isinstance(slot_mapping, torch.Tensor)
        and _valid_kv_cache(kv_cache)
        and infinicore_backend.store_kv_cache_backend_enabled(key)
    )


def _valid_kv_cache(kv_cache: torch.Tensor) -> bool:
    return (
        isinstance(kv_cache, torch.Tensor)
        and kv_cache.ndim >= 5
        and kv_cache.numel() > 0
        and (kv_cache.shape[0] == 2 or kv_cache.shape[1] == 2)
    )


def _decode_as_prefill(
    num_decode_tokens: int,
    num_decodes: int,
    num_prefills: int,
    num_actual_tokens: int,
) -> bool:
    return (
        num_prefills == 0
        and num_decodes == 1
        and num_actual_tokens > 1
        and num_decode_tokens == num_actual_tokens
    )


def _record_attention(name: str) -> None:
    _ATTENTION_COUNTS[name] = _ATTENTION_COUNTS.get(name, 0) + 1


__all__ = [
    "INFINICORE_ATTENTION_BACKEND_ROUTES",
    "InfiniCoreAttentionBackendStatus",
    "InfiniCoreAttentionBackendUninstallStatus",
    "InfiniCoreFlashAttentionBackend",
    "InfiniCoreFlashAttentionImpl",
    "attention_backend_route_counts",
    "install_infinicore_attention_backend",
    "reset_attention_backend_route_counts",
    "uninstall_infinicore_attention_backend",
]
