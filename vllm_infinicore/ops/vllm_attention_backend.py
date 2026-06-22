"""vLLM attention backend override for InfiniCore PA/KV kernels.

This module registers an AttentionBackend-level implementation instead of
monkey-patching an existing backend class. The first implementation deliberately
reuses the platform FlashAttention metadata builder and KV cache layout, then
routes supported decoder KV update/paged-attention calls through InfiniCore.
Unsupported paths fall back to the platform backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
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
METAX_COMPAT_FA_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA"

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

    flash_attn_module = importlib.import_module("vllm.v1.attention.backends.flash_attn")
    _patch_native_flash_attention_module(flash_attn_module)
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


def _patch_native_flash_attention_module(module: Any) -> None:
    if getattr(module, "_vllm_infinicore_fa2_compat_patched", False):
        return

    def get_flash_attn_version(*args: Any, **kwargs: Any) -> int:
        requires_alibi = kwargs.get("requires_alibi", args[0] if args else False)
        del requires_alibi
        return 2

    def flash_attn_supports_fp8(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return False

    def flash_attn_supports_sinks(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True

    module.get_flash_attn_version = get_flash_attn_version
    module.flash_attn_supports_fp8 = flash_attn_supports_fp8
    module.flash_attn_supports_sinks = flash_attn_supports_sinks
    module._vllm_infinicore_fa2_compat_patched = True


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
        return InfiniCoreFlashAttentionMetadataBuilder


class InfiniCoreFlashAttentionMetadataBuilder(_FlashAttentionMetadataBuilder):
    """FlashAttention metadata builder with MetaX-compatible decode split fields."""

    reorder_batch_threshold: int = 128

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        init_threshold = getattr(self, "_init_reorder_batch_threshold", None)
        if callable(init_threshold):
            init_threshold(self.reorder_batch_threshold, True)

    def build(self, *args: Any, **kwargs: Any) -> Any:
        metadata = super().build(*args, **kwargs)
        return _add_infinicore_metadata_fields(
            metadata,
            decode_threshold=int(getattr(self, "reorder_batch_threshold", 1) or 1),
        )

    def update_block_table(
        self,
        metadata: Any,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Any:
        update = getattr(super(), "update_block_table", None)
        if callable(update):
            try:
                updated = update(metadata, blk_table, slot_mapping)
            except NotImplementedError:
                updated = copy.copy(metadata)
                updated.block_table = blk_table
                updated.slot_mapping = slot_mapping
        else:
            updated = copy.copy(metadata)
            updated.block_table = blk_table
            updated.slot_mapping = slot_mapping
        return _add_infinicore_metadata_fields(
            updated,
            decode_threshold=int(getattr(self, "reorder_batch_threshold", 1) or 1),
        )


class InfiniCoreFlashAttentionImpl(_BaseFlashAttentionImpl):
    """Attention impl with backend-level InfiniCore dispatch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vllm_flash_attn_version = 2
        self.supports_quant_query_input = False
        self.supports_per_head_quant_scales = False

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

            if _metax_compatible_fa_enabled():
                if _forward_metax_compatible_fa(
                    self,
                    query,
                    kv_cache,
                    attn_metadata,
                    output,
                    num_decode_tokens,
                    num_actual_tokens,
                    num_prefills,
                    num_decodes,
                ):
                    _record_attention("backend_forward_infinicore")
                    return output

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

    if _platform_should_activate_attention_routes():
        _ACTIVE_ROUTES.update(INFINICORE_ATTENTION_BACKEND_ROUTES)
    _register_backend_once()


def _platform_should_activate_attention_routes() -> bool:
    """Return whether platform registration should enable attention routes.

    Platform-only no-MetaX smoke runs rely on the InfiniCore attention backend
    because vLLM's native FlashAttention probe is not usable on the current MACA
    stack. When explicit route patching is enabled, route activation must stay
    with the route registry so throughput/isolation profiles can actually
    disable the attention bridge.
    """

    patches_enabled = os.environ.get("VLLM_INFINICORE_ENABLE_PATCHES", "")
    selected_routes = os.environ.get("VLLM_INFINICORE_ROUTES", "")
    if (
        patches_enabled.strip().lower() in {"1", "true", "yes", "on"}
        and selected_routes.strip()
    ):
        return False
    return True


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


def _add_infinicore_metadata_fields(
    attn_metadata: object,
    *,
    decode_threshold: int = 1,
) -> object:
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
    num_actual_tokens = int(
        getattr(attn_metadata, "num_actual_tokens", int(query_start_loc[-1].item()))
    )

    num_decodes = 0
    for query_len in query_lens.tolist():
        if int(query_len) > decode_threshold:
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

    setattr(attn_metadata, "num_decodes", num_decodes)
    setattr(attn_metadata, "num_decode_tokens", num_decode_tokens)
    setattr(attn_metadata, "decode_query_start_loc", decode_query_start_loc)
    setattr(attn_metadata, "decode_seq_lens", decode_seq_lens)
    setattr(attn_metadata, "decode_block_table", decode_block_table)
    setattr(attn_metadata, "num_prefills", num_prefills)
    setattr(attn_metadata, "num_prefill_tokens", num_prefill_tokens)
    setattr(attn_metadata, "prefill_query_start_loc", prefill_query_start_loc)
    setattr(attn_metadata, "prefill_max_seq_len", prefill_max_seq_len)
    setattr(attn_metadata, "prefill_block_table", prefill_block_table)
    setattr(attn_metadata, "cu_prefix_kv_lens", cu_prefix_kv_lens)
    if not hasattr(attn_metadata, "cu_seqlens_k"):
        setattr(attn_metadata, "cu_seqlens_k", None)
    return attn_metadata


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


def _metax_compatible_fa_enabled() -> bool:
    return (
        not _metax_platform_plugin_enabled()
        and not _env_truthy(METAX_COMPAT_FA_DISABLE_ENV)
    )


def _forward_metax_compatible_fa(
    attn_impl: object,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    output: torch.Tensor,
    num_decode_tokens: int,
    num_actual_tokens: int,
    num_prefills: int,
    num_decodes: int,
) -> bool:
    if getattr(attn_metadata, "use_cascade", False):
        return False
    if getattr(attn_impl, "kv_cache_dtype", "").startswith("fp8"):
        return False

    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    except Exception:
        return False

    key_cache, value_cache = _split_kv_cache_for_flash(kv_cache)
    handled = False

    if num_prefills > 0:
        prefill_query = query[num_decode_tokens:num_actual_tokens]
        if prefill_query.numel() > 0:
            prefill_output = flash_attn_varlen_func(
                q=prefill_query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                cu_seqlens_k=attn_metadata.cu_prefix_kv_lens,
                max_seqlen_q=getattr(attn_metadata, "max_query_len", prefill_query.shape[0]),
                max_seqlen_k=getattr(
                    attn_metadata,
                    "prefill_max_seq_len",
                    getattr(attn_metadata, "max_seq_len", prefill_query.shape[0]),
                ),
                softmax_scale=float(getattr(attn_impl, "scale")),
                causal=bool(getattr(attn_metadata, "causal", True)),
                alibi_slopes=_on_query_device(getattr(attn_impl, "alibi_slopes", None), query),
                window_size=getattr(attn_impl, "sliding_window", (-1, -1)),
                block_table=attn_metadata.prefill_block_table,
                softcap=float(getattr(attn_impl, "logits_soft_cap", 0.0) or 0.0),
                s_aux=getattr(attn_impl, "sinks", None),
            )
            _copy_attention_output(
                output[num_decode_tokens:num_actual_tokens],
                prefill_output,
                prefill_query.shape,
            )
            infinicore_backend.record_backend_call("paged_attention_prefill")
            _record_attention("backend_prefill_infinicore")
            _record_attention("backend_prefill_metax_compatible")
            handled = True

    if num_decodes > 0:
        decode_query = query[:num_decode_tokens]
        decode_query = _reshape_query_for_spec_decode(decode_query, num_decodes)
        decode_output = flash_attn_with_kvcache(
            q=decode_query,
            k_cache=key_cache,
            v_cache=value_cache,
            block_table=attn_metadata.decode_block_table,
            cache_seqlens=attn_metadata.decode_seq_lens,
            softmax_scale=float(getattr(attn_impl, "scale")),
            causal=True,
            window_size=getattr(attn_impl, "sliding_window", (-1, -1)),
            alibi_slopes=_on_query_device(getattr(attn_impl, "alibi_slopes", None), query),
            num_splits=_flash_decode_num_splits(),
            softcap=float(getattr(attn_impl, "logits_soft_cap", 0.0) or 0.0),
            s_aux=getattr(attn_impl, "sinks", None),
        )
        decoded = _reshape_attn_output_for_spec_decode(decode_output)
        _copy_attention_output(output[:num_decode_tokens], decoded, decoded.shape)
        infinicore_backend.record_backend_call("paged_attention_decode")
        _record_attention("backend_decode_infinicore")
        _record_attention("backend_decode_metax_compatible")
        handled = True

    return handled


def _split_kv_cache_for_flash(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.shape[0] == 2:
        return kv_cache.unbind(0)
    if kv_cache.shape[1] == 2:
        return kv_cache.unbind(1)
    raise RuntimeError(f"cannot infer KV cache split axis from cache={kv_cache.shape}")


def _reshape_query_for_spec_decode(
    query: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    total_tokens, num_heads, head_dim = query.shape
    if total_tokens % batch_size != 0:
        raise RuntimeError(
            f"decode query tokens {total_tokens} are not divisible by {batch_size}"
        )
    return query.view(batch_size, total_tokens // batch_size, num_heads, head_dim)


def _reshape_attn_output_for_spec_decode(attn_output: torch.Tensor) -> torch.Tensor:
    if attn_output.dim() == 3:
        return attn_output
    if attn_output.dim() != 4:
        raise RuntimeError(f"expected 3D/4D attention output, got {attn_output.dim()}D")
    return attn_output.view(
        attn_output.shape[0] * attn_output.shape[1],
        attn_output.shape[2],
        attn_output.shape[3],
    )


def _copy_attention_output(
    output_slice: torch.Tensor,
    attention_output: torch.Tensor,
    view_shape: torch.Size | tuple[int, ...],
) -> None:
    output_slice.view(view_shape).copy_(attention_output.view(view_shape))


def _on_query_device(
    tensor: torch.Tensor | None,
    query: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None or tensor.device == query.device:
        return tensor
    return tensor.to(device=query.device)


def _flash_decode_num_splits() -> int:
    try:
        from . import cpp_bridge

        return cpp_bridge.flash_decode_num_splits()
    except Exception:
        return 0


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


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
