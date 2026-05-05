"""vLLM attention backend routes for InfiniCore PA/KV wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Callable

import torch

VLLM_ATTENTION_ROUTE_NAMES = (
    "StoreKVCache",
    "PagedAttentionPrefill",
    "PagedAttentionDecode",
)

_ATTENTION_IMPL_CLASS_PATHS = (
    "vllm_metax.v1.attention.backends.flash_attn.FlashAttentionImpl",
    "vllm_metax.v1.attention.backends.triton_attn.TritonAttentionImpl",
    "vllm_metax.v1.attention.backends.flashinfer.FlashInferImpl",
    "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl",
    "vllm.v1.attention.backends.triton_attn.TritonAttentionImpl",
    "vllm.v1.attention.backends.flashinfer.FlashInferImpl",
    "vllm.v1.attention.backends.rocm_attn.RocmAttentionImpl",
)

_ACTIVE_ROUTES: set[str] = set()
_ORIGINAL_FORWARDS: dict[type, Callable[..., torch.Tensor]] = {}
_ORIGINAL_KV_UPDATES: dict[type, Callable[..., None]] = {}
_ATTENTION_COUNTS: dict[str, int] = {}


@dataclass(frozen=True)
class VllmAttentionInstallStatus:
    installed: bool
    reason: str
    route_name: str


@dataclass(frozen=True)
class VllmAttentionUninstallStatus:
    uninstalled: bool
    reason: str
    route_name: str


def install_vllm_attention_route(route_name: str) -> VllmAttentionInstallStatus:
    """Patch attention backend impls without replacing ``Attention.forward``.

    vLLM keeps CUDA/MACA graph paths stable by calling opaque attention custom
    ops from ``Attention.forward``. The Python backend impl runs behind that
    boundary, so replacing impl methods is less graph-hostile than replacing
    ``Attention.forward`` itself.
    """

    if route_name not in VLLM_ATTENTION_ROUTE_NAMES:
        return VllmAttentionInstallStatus(
            installed=False,
            reason=f"unsupported attention route {route_name}",
            route_name=route_name,
        )

    patched = _patch_attention_impl_classes()
    if not patched:
        return VllmAttentionInstallStatus(
            installed=False,
            reason="no supported vLLM attention backend classes were importable",
            route_name=route_name,
        )

    _ACTIVE_ROUTES.add(route_name)
    return VllmAttentionInstallStatus(
        installed=True,
        reason=(
            f"InfiniCore attention backend patch active for {route_name} "
            f"({patched} class(es))"
        ),
        route_name=route_name,
    )


def uninstall_vllm_attention_route(route_name: str) -> VllmAttentionUninstallStatus:
    """Remove a PA/KV route and restore backend methods when none remain."""

    if route_name not in _ACTIVE_ROUTES:
        return VllmAttentionUninstallStatus(
            uninstalled=False,
            reason=f"InfiniCore attention route {route_name} not installed",
            route_name=route_name,
        )

    _ACTIVE_ROUTES.remove(route_name)
    if not _ACTIVE_ROUTES:
        for impl_cls, original in tuple(_ORIGINAL_FORWARDS.items()):
            impl_cls.forward = original
        for impl_cls, original in tuple(_ORIGINAL_KV_UPDATES.items()):
            impl_cls.do_kv_cache_update = original
        _ORIGINAL_FORWARDS.clear()
        _ORIGINAL_KV_UPDATES.clear()

    return VllmAttentionUninstallStatus(
        uninstalled=True,
        reason=f"InfiniCore attention route {route_name} uninstalled",
        route_name=route_name,
    )


def attention_route_counts() -> dict[str, int]:
    return dict(_ATTENTION_COUNTS)


def reset_attention_route_counts() -> None:
    _ATTENTION_COUNTS.clear()


def _patch_attention_impl_classes() -> int:
    patched = 0
    for class_path in _ATTENTION_IMPL_CLASS_PATHS:
        impl_cls = _import_class(class_path)
        if impl_cls is None:
            continue
        if impl_cls not in _ORIGINAL_FORWARDS:
            original_forward = impl_cls.forward
            _ORIGINAL_FORWARDS[impl_cls] = original_forward
            impl_cls.forward = _make_forward_wrapper(impl_cls, original_forward)
            patched += 1
        if hasattr(impl_cls, "do_kv_cache_update") and impl_cls not in _ORIGINAL_KV_UPDATES:
            original_update = impl_cls.do_kv_cache_update
            _ORIGINAL_KV_UPDATES[impl_cls] = original_update
            impl_cls.do_kv_cache_update = _make_kv_update_wrapper(original_update)
    return patched or len(_ORIGINAL_FORWARDS)


def _import_class(class_path: str) -> type | None:
    module_name, _, class_name = class_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    impl_cls = getattr(module, class_name, None)
    return impl_cls if isinstance(impl_cls, type) else None


def _make_forward_wrapper(
    impl_cls: type,
    original_forward: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    def forward(
        self: object,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: object,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def call_original() -> torch.Tensor:
            return original_forward(
                self,
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

        try:
            skip_reason = _infinicore_attention_skip_reason(
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
                _record_attention(f"forward_skip.{skip_reason}")
                return call_original()
            _record_attention("forward_infinicore")
            return _forward_infinicore_impl(
                self,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
        except Exception:
            from .infinicore_backend import strict_backend_enabled

            if strict_backend_enabled():
                raise
            return call_original()

    forward.__name__ = getattr(original_forward, "__name__", "forward")
    forward.__qualname__ = f"{impl_cls.__qualname__}.forward"
    return forward


def _make_kv_update_wrapper(
    original_update: Callable[..., None],
) -> Callable[..., None]:
    def do_kv_cache_update(
        self: object,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        def call_original() -> None:
            return original_update(self, layer, key, value, kv_cache, slot_mapping)

        try:
            if "StoreKVCache" not in _ACTIVE_ROUTES:
                return call_original()
            if not _valid_store_inputs(self, key, value, kv_cache, slot_mapping):
                _record_attention("kv_update_skip.invalid_inputs")
                return call_original()
            from . import infinicore_backend

            if not infinicore_backend.real_backend_enabled(key):
                _record_attention("kv_update_skip.real_backend_disabled")
                return call_original()
            infinicore_backend.store_kv_cache(kv_cache, key, value, slot_mapping)
            _record_attention("kv_update_infinicore")
            return None
        except Exception:
            from .infinicore_backend import strict_backend_enabled

            if strict_backend_enabled():
                raise
            return call_original()

    do_kv_cache_update.__name__ = getattr(
        original_update, "__name__", "do_kv_cache_update"
    )
    return do_kv_cache_update


def _infinicore_attention_skip_reason(
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
        return "no_output_or_metadata"
    if output_scale is not None or output_block_scale is not None:
        return "output_quant"
    if getattr(attn_metadata, "use_cascade", False):
        return "cascade"
    if getattr(attn_metadata, "local_attn_metadata", None) is not None:
        return "local_attn"
    if getattr(attn_impl, "kv_sharing_target_layer_name", None) is not None:
        return "kv_sharing"
    if not _valid_kv_cache(kv_cache):
        return "invalid_kv_cache"
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        return "missing_key_value"
    from . import infinicore_backend

    if not infinicore_backend.real_backend_enabled(query):
        return "real_backend_disabled"

    num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]))
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
    if (needs_prefill or needs_decode) and "StoreKVCache" not in _ACTIVE_ROUTES:
        return "store_route_inactive"
    if needs_decode and num_decode_tokens != num_decodes:
        return (
            "speculative_decode"
            f".tokens{num_decode_tokens}"
            f".decodes{num_decodes}"
            f".prefills{num_prefills}"
            f".actual{num_actual_tokens}"
        )
    if not (needs_prefill or needs_decode):
        return "no_work"
    return None


def _forward_infinicore_impl(
    attn_impl: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    output: torch.Tensor | None,
) -> torch.Tensor:
    assert output is not None
    from . import infinicore_backend

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
    if not hasattr(attn_impl, "do_kv_cache_update"):
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if _valid_store_inputs(attn_impl, key, value, kv_cache, slot_mapping):
            infinicore_backend.store_kv_cache(
                kv_cache,
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                slot_mapping,
            )

    if needs_prefill:
        _record_attention("prefill_infinicore")
        if decode_as_prefill:
            infinicore_backend.paged_attention_decode_as_prefill_infinicore_only(
                attn_impl,
                query,
                key,
                kv_cache,
                attn_metadata,
                output,
            )
        else:
            infinicore_backend.paged_attention_prefill_infinicore_only(
                attn_impl,
                query,
                key,
                kv_cache,
                attn_metadata,
                output,
            )
    if needs_decode:
        _record_attention("decode_infinicore")
        infinicore_backend.paged_attention_decode_infinicore_only(
            attn_impl,
            query,
            key,
            kv_cache,
            attn_metadata,
            output,
        )
    return output


def _valid_store_inputs(
    attn_impl: object,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor | None,
) -> bool:
    if getattr(attn_impl, "kv_sharing_target_layer_name", None) is not None:
        return False
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        return False
    if slot_mapping is None or not isinstance(slot_mapping, torch.Tensor):
        return False
    return _valid_kv_cache(kv_cache)


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
    "VLLM_ATTENTION_ROUTE_NAMES",
    "VllmAttentionInstallStatus",
    "VllmAttentionUninstallStatus",
    "attention_route_counts",
    "install_vllm_attention_route",
    "reset_attention_route_counts",
    "uninstall_vllm_attention_route",
]
