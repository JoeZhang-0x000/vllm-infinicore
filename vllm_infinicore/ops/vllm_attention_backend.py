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
import os
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

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

    if _metax_platform_plugin_enabled():
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
            pass

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


def _metax_platform_plugin_enabled() -> bool:
    plugins = os.environ.get("VLLM_PLUGINS")
    if plugins is None:
        return True
    return "metax" in {plugin.strip() for plugin in plugins.split(",") if plugin.strip()}


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
        if output is None:
            output = torch.empty(
                (query.shape[0], query.shape[1] * query.shape[2]),
                dtype=query.dtype,
                device=query.device,
            )
        if attn_metadata is None:
            _record_attention("backend_forward_profile_zero")
            return output.fill_(0)
        attn_metadata = _metadata_with_infinicore_fields(attn_metadata, query)
        if _should_return_profile_output(query, kv_cache, attn_metadata, output):
            _record_attention("backend_forward_profile_zero")
            return output.fill_(0)
        skip_reason = _infinicore_forward_skip_reason(
            self,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        if skip_reason is not None:
            _record_attention("backend_forward_fallback")
            if (
                not _metax_platform_plugin_enabled()
                and infinicore_backend.strict_backend_enabled()
            ):
                raise RuntimeError(f"InfiniCore attention skipped: {skip_reason}")
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


def install_platform_attention_backend() -> None:
    """Install the InfiniCore attention backend for the platform plugin path."""

    _ACTIVE_ROUTES.update(INFINICORE_ATTENTION_BACKEND_ROUTES)
    _register_backend_once()


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
    if not _metax_platform_plugin_enabled():
        _PATCHED_METAX_REGISTER = True
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
    return (
        _infinicore_forward_skip_reason(
            attn_impl,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        is None
    )


def _infinicore_forward_skip_reason(
    attn_impl: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    output: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    output_block_scale: torch.Tensor | None,
) -> str | None:
    if output is None or attn_metadata is None:
        return "missing_output_or_metadata"
    if output_scale is not None or output_block_scale is not None:
        return "output_quantization"
    if getattr(attn_metadata, "use_cascade", False):
        return "cascade"
    if getattr(attn_impl, "kv_sharing_target_layer_name", None) is not None:
        return "kv_sharing"
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        return "missing_key_value"
    if not _valid_kv_cache(kv_cache):
        shape = getattr(kv_cache, "shape", None)
        return f"invalid_kv_cache:{shape}"
    if not infinicore_backend.real_backend_enabled(query):
        return f"real_backend_disabled:query_is_cuda={getattr(query, 'is_cuda', None)}"

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
        return "prefill_route_inactive"
    if needs_decode and "PagedAttentionDecode" not in _ACTIVE_ROUTES:
        return "decode_route_inactive"
    if needs_decode and num_decode_tokens != num_decodes:
        return f"unsupported_spec_decode:{num_decode_tokens}!={num_decodes}"
    if not (needs_prefill or needs_decode):
        return "no_prefill_or_decode"
    return None


def _should_return_profile_output(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    output: torch.Tensor | None,
) -> bool:
    return (
        output is not None
        and attn_metadata is not None
        and not _metax_platform_plugin_enabled()
        and infinicore_backend.real_backend_enabled(query)
        and not _valid_kv_cache(kv_cache)
    )


def _metadata_with_infinicore_fields(
    attn_metadata: object,
    query: torch.Tensor,
) -> object:
    if hasattr(attn_metadata, "num_decode_tokens") and hasattr(
        attn_metadata, "prefill_query_start_loc"
    ):
        return attn_metadata

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    if not (
        isinstance(query_start_loc, torch.Tensor)
        and isinstance(seq_lens, torch.Tensor)
        and isinstance(block_table, torch.Tensor)
        and query_start_loc.numel() >= 2
    ):
        return attn_metadata

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_reqs = int(query_lens.numel())
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]))

    num_decodes = 0
    for query_len in query_lens.tolist():
        if int(query_len) > 1:
            break
        num_decodes += 1
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = (
        int(query_start_loc[num_decodes].item()) if num_decodes > 0 else 0
    )
    num_prefill_tokens = num_actual_tokens - num_decode_tokens

    if num_decodes > 0:
        decode_query_start_loc = query_start_loc[: num_decodes + 1]
        decode_seq_lens = seq_lens[:num_decodes]
        decode_block_table = block_table[:num_decodes]
    else:
        decode_query_start_loc = None
        decode_seq_lens = None
        decode_block_table = None

    if num_prefills > 0:
        prefill_query_start_loc = (
            query_start_loc[num_decodes : num_reqs + 1] - query_start_loc[num_decodes]
        )
        prefill_seq_lens = seq_lens[num_decodes:num_reqs]
        prefill_max_seq_len = int(prefill_seq_lens.max().item())
        prefill_block_table = block_table[num_decodes:num_reqs]
        cu_prefix_kv_lens = F.pad(prefill_seq_lens, (1, 0), value=0).cumsum(
            dim=0,
            dtype=torch.int32,
        )
    else:
        prefill_query_start_loc = None
        prefill_max_seq_len = 0
        prefill_block_table = None
        cu_prefix_kv_lens = None

    values = {
        name: getattr(attn_metadata, name)
        for name in dir(attn_metadata)
        if not name.startswith("_")
    }
    values.update(
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        decode_query_start_loc=decode_query_start_loc,
        decode_seq_lens=decode_seq_lens,
        decode_block_table=decode_block_table,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill_query_start_loc=prefill_query_start_loc,
        prefill_max_seq_len=prefill_max_seq_len,
        prefill_block_table=prefill_block_table,
        cu_prefix_kv_lens=cu_prefix_kv_lens,
        cu_seqlens_k=None,
    )
    return SimpleNamespace(**values)


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
    "install_platform_attention_backend",
    "reset_attention_backend_route_counts",
    "uninstall_infinicore_attention_backend",
]
