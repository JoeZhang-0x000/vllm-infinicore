"""Fully independent InfiniCore attention backend for vLLM.

This module implements a self-contained ``AttentionBackend`` that routes
paged-attention and KV-cache operations through InfiniCore kernels
(``infinicore.mha_varlen``, ``infinicore.mha_kvcache``,
``infinicore.paged_caching``) **without** inheriting from or depending on
``vllm_metax``.

Architecture
------------

* ``InfiniCoreFlashAttentionMetadata`` — frozen dataclass carrying all fields
  needed by InfiniCore kernels (prefill/decode split, block tables, sequence
  lengths, etc.).
* ``InfiniCoreFlashAttentionMetadataBuilder`` — produces the metadata from
  vLLM's ``CommonAttentionMetadata``.  Handles the decode/prefill split
  internally.
* ``InfiniCoreFlashAttentionBackend`` — concrete ``AttentionBackend``
  subclass; provides KV-cache shape, builder class, and impl class.
* ``InfiniCoreFlashAttentionImpl`` — concrete ``AttentionImpl`` subclass;
  dispatches ``forward`` and ``do_kv_cache_update`` to InfiniCore.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    split_decodes_and_prefills,
)

from . import infinicore_backend

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType as CacheDType_t
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------


@dataclass
class InfiniCoreFlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # decode split
    num_decodes: int
    num_decode_tokens: int
    decode_query_start_loc: torch.Tensor | None
    decode_seq_lens: torch.Tensor | None
    decode_block_table: torch.Tensor | None

    # prefill split
    num_prefills: int
    num_prefill_tokens: int
    prefill_query_start_loc: torch.Tensor | None
    prefill_max_seq_len: int
    prefill_block_table: torch.Tensor | None

    cu_prefix_kv_lens: torch.Tensor | None

    # cascade (currently unsupported – fallback to torch)
    use_cascade: bool = False
    common_prefix_len: int = 0
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None

    # DCP (currently unsupported – fallback to torch)
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0
    causal: bool = True


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class InfiniCoreFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        if vllm_config is not None:
            block_size = vllm_config.cache_config.block_size
            assert block_size is not None
            return [16, 32, block_size]
        return [16, 32, 64, 128]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER)

    @staticmethod
    def get_impl_cls() -> type[InfiniCoreFlashAttentionImpl]:
        return InfiniCoreFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[InfiniCoreFlashAttentionMetadataBuilder]:
        return InfiniCoreFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 3, 2, 4, 5)
        return (1, 3, 0, 2, 4)

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype == "fp8_e4m3":
            return torch.float8_e4m3fn
        if kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        raise ValueError(f"Unsupported fp8 kv_cache_dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (32, 64, 80, 96, 128, 160, 192, 224, 256)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None or kv_cache_dtype == "auto":
            return True
        if kv_cache_dtype in ("fp8_e4m3", "fp8_e5m2"):
            return True
        return kv_cache_dtype == "bfloat16" or kv_cache_dtype.startswith("fp8")

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        kv_cache_dtype: CacheDType_t | None,
        head_size: int,
        block_size: int | None,
        attn_type: str,
        dtype: torch.dtype,
        enable_prefix_caching: bool,
        capacity: int | None = None,
    ) -> bool:
        if not cls.supports_dtype(dtype):
            return False
        if not cls.supports_attn_type(attn_type):
            return False
        if not cls.supports_head_size(head_size):
            return False
        if not cls.supports_block_size(block_size):
            return False
        return cls.supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def validate_configuration(
        cls,
        device_capability: DeviceCapability,
        kv_cache_dtype: CacheDType_t | None,
        head_size: int,
        block_size: int | None,
        attn_type: str,
        dtype: torch.dtype,
        enable_prefix_caching: bool = False,
        capacity: int | None = None,
        **kwargs: object,
    ) -> list[str]:
        reasons: list[str] = []
        if not cls.supports_dtype(dtype):
            reasons.append(f"dtype {dtype} not supported")
        if not cls.supports_attn_type(attn_type):
            reasons.append(f"attn_type {attn_type} not supported")
        if not cls.supports_head_size(head_size):
            reasons.append(f"head_size {head_size} not supported")
        if not cls.supports_block_size(block_size):
            reasons.append(f"block_size {block_size} not supported")
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            reasons.append(f"kv_cache_dtype {kv_cache_dtype} not supported")
        return reasons

    @classmethod
    def get_required_kv_cache_layout(cls) -> str | None:
        return get_kv_cache_layout()


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


class InfiniCoreFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[InfiniCoreFlashAttentionMetadata]
):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 128
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if (
            vllm_config.model_config is not None
            and vllm_config.model_config.is_encoder_decoder
        ):
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InfiniCoreFlashAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        if num_decodes > 0:
            decode_query_start_loc = common_attn_metadata.query_start_loc[
                : num_decodes + 1
            ]
            decode_seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            decode_block_table_tensor = common_attn_metadata.block_table_tensor[
                :num_decodes
            ]
        else:
            decode_query_start_loc = None
            decode_seq_lens = None
            decode_block_table_tensor = None

        if num_prefills > 0:
            prefill_query_start_loc = (
                common_attn_metadata.query_start_loc[num_decodes : num_reqs + 1]
                - common_attn_metadata.query_start_loc[num_decodes]
            )
            prefill_seq_lens = common_attn_metadata.seq_lens[num_decodes:num_reqs]
            prefill_max_seq_len = int(prefill_seq_lens.max().item())
            prefill_block_table_tensor = common_attn_metadata.block_table_tensor[
                num_decodes:num_reqs
            ]
            cu_prefix_kv_lens = F.pad(
                prefill_seq_lens,
                (1, 0),
                value=0,
            ).cumsum(dim=0, dtype=torch.int32)
        else:
            prefill_query_start_loc = None
            prefill_seq_lens = None
            prefill_max_seq_len = 0
            prefill_block_table_tensor = None
            cu_prefix_kv_lens = None

        use_cascade = common_prefix_len > 0

        return InfiniCoreFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            decode_query_start_loc=decode_query_start_loc,
            decode_seq_lens=decode_seq_lens,
            decode_block_table=decode_block_table_tensor,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_max_seq_len=prefill_max_seq_len,
            prefill_block_table=prefill_block_table_tensor,
            cu_prefix_kv_lens=cu_prefix_kv_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            causal=causal,
        )

    def update_block_table(
        self,
        metadata: InfiniCoreFlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> InfiniCoreFlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args: Any, **kwargs: Any) -> bool:
        return False


# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------


class InfiniCoreFlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: InfiniCoreFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Unsupported paths → fallback eagerly
        if attn_metadata.use_cascade:
            return self._metax_fallback_forward(
                layer, query, key, value, kv_cache, attn_metadata, output
            )

        if self.attn_type not in (AttentionType.DECODER,):
            return self._metax_fallback_forward(
                layer, query, key, value, kv_cache, attn_metadata, output
            )

        if output_scale is not None or output_block_scale is not None:
            return self._metax_fallback_forward(
                layer, query, key, value, kv_cache, attn_metadata, output
            )

        if not infinicore_backend.real_backend_enabled(query):
            return self._metax_fallback_forward(
                layer, query, key, value, kv_cache, attn_metadata, output
            )

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills

        decode_as_prefill = (
            num_prefills == 0
            and num_decodes == 1
            and num_actual_tokens > 1
            and num_decode_tokens == num_actual_tokens
        )

        needs_prefill = (
            decode_as_prefill
            or num_prefills > 0
            or num_actual_tokens > num_decode_tokens
        )
        needs_decode = (not decode_as_prefill) and (
            num_decodes > 0 or num_decode_tokens > 0
        )

        try:
            if needs_prefill:
                if decode_as_prefill:
                    infinicore_backend.paged_attention_decode_as_prefill_infinicore_only(
                        self, query, key, kv_cache, attn_metadata, output
                    )
                else:
                    infinicore_backend.paged_attention_prefill_infinicore_only(
                        self, query, key, kv_cache, attn_metadata, output
                    )
            if needs_decode:
                infinicore_backend.paged_attention_decode_infinicore_only(
                    self, query, key, kv_cache, attn_metadata, output
                )
            global _INFINI_CALL_COUNT
            _INFINI_CALL_COUNT += 1
            return output
        except Exception as exc:
            if infinicore_backend.strict_backend_enabled():
                raise
            global _FALLBACK_COUNT
            _FALLBACK_COUNT += 1
            logger.warning(
                "InfiniCore attention kernel failed (fallback #%d): %s",
                _FALLBACK_COUNT, exc,
            )
            return self._metax_fallback_forward(
                layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale, output_block_scale,
            )


_INFINI_CALL_COUNT = 0
_FALLBACK_COUNT = 0


def reset_attention_counts() -> None:
    global _INFINI_CALL_COUNT, _FALLBACK_COUNT
    _INFINI_CALL_COUNT = 0
    _FALLBACK_COUNT = 0


    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> None:
        if self.attn_type not in (AttentionType.DECODER,):
            return
        if self.kv_sharing_target_layer_name is not None:
            return
        if not isinstance(slot_mapping, torch.Tensor):
            return
        if not infinicore_backend.real_backend_enabled(key):
            return
        try:
            infinicore_backend.store_kv_cache(
                kv_cache, key, value, slot_mapping
            )
        except Exception:
            if infinicore_backend.strict_backend_enabled():
                raise

    def _metax_fallback_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: InfiniCoreFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_metax.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
        )
        from vllm.v1.attention.backends.utils import (
            reshape_attn_output_for_spec_decode,
            reshape_query_for_spec_decode,
        )

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        if attn_metadata.num_prefills > 0:
            q = query[attn_metadata.num_decode_tokens:num_actual_tokens]
            output[attn_metadata.num_decode_tokens:num_actual_tokens] = (
                flash_attn_varlen_func(
                    q=q,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                    cu_seqlens_k=attn_metadata.cu_prefix_kv_lens,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.prefill_max_seq_len,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=attn_metadata.prefill_block_table,
                    softcap=self.logits_soft_cap,
                )
            )

        if attn_metadata.num_decodes > 0:
            decode_query = reshape_query_for_spec_decode(
                query[:attn_metadata.num_decode_tokens],
                attn_metadata.num_decodes,
            )
            output_unreshaped = flash_attn_with_kvcache(
                q=decode_query,
                k_cache=key_cache,
                v_cache=value_cache,
                block_table=attn_metadata.decode_block_table,
                cache_seqlens=attn_metadata.decode_seq_lens,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
            )
            output[:attn_metadata.num_decode_tokens] = (
                reshape_attn_output_for_spec_decode(output_unreshaped)
            )

        return output


def attention_counts() -> tuple[int, int]:
    return (_INFINI_CALL_COUNT, _FALLBACK_COUNT)
