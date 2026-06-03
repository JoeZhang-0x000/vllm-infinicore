# Qwen3 Operator Scope

This document defines the first operator scope for single-node Qwen3 inference. All routes are disabled by default, and requested routes without a proven installer use native vLLM fallback.

| Operator | Category | Planned route | Default | Current installer | Native fallback |
|---|---|---|---|---|---|
| `RMSNorm` | Normalization | PyTorch custom op wrapper | Disabled | vLLM OOT class -> `vllm_infinicore::rms_norm` | vLLM RMSNorm `forward_native` |
| `SiluAndMul` | MLP activation | PyTorch custom op wrapper | Disabled | vLLM OOT class -> `vllm_infinicore::silu_and_mul` | vLLM native activation |
| `RoPE` | Attention position encoding | PyTorch custom op wrapper | Disabled | vLLM OOT class -> `vllm_infinicore::rotary_embedding` | vLLM native rotary embedding |
| `Embedding` | Token embedding | PyTorch custom op wrapper | Disabled | `UnquantizedEmbeddingMethod.embedding` -> `vllm_infinicore::embedding` | vLLM native token embedding |
| `MatMul` | Linear projections | PyTorch custom op wrapper | Disabled | `UnquantizedLinearMethod.apply` -> `vllm_infinicore::linear` | vLLM native linear layers |
| `LMHead` | Final projection | PyTorch custom op wrapper | Disabled | `UnquantizedEmbeddingMethod.apply` for `ParallelLMHead` -> `vllm_infinicore::lm_head` | vLLM native logits projection |
| `StoreKVCache` | KV cache update | InfiniCore PA/KV wrapper | Disabled | attention backend `do_kv_cache_update` -> `infinicore.paged_caching` | vLLM native KV cache store path |
| `PagedAttentionPrefill` | Paged attention | InfiniCore PA/KV wrapper | Disabled | attention backend `forward` -> `infinicore.paged_attention_prefill` | vLLM native paged attention prefill backend |
| `PagedAttentionDecode` | Paged attention | InfiniCore PA/KV wrapper | Disabled | attention backend `forward` -> C++ bridge -> InfiniCore `mha_kvcache_` | vLLM native paged attention decode backend |

## Routing And Defaults

Current routing policy:

- Keep patching opt-in and disabled by default.
- Install routes only when `VLLM_INFINICORE_ENABLE_PATCHES=1` and
  `VLLM_INFINICORE_ROUTES=...` request them.
- Allow `VLLM_INFINICORE_ROUTES=all` for the full nine-route Qwen3 scope or a
  comma-separated subset for isolation.
- Allow `VLLM_INFINICORE_DISABLED_ROUTES=...` to remove selected routes from an
  otherwise larger request.
- Allow `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=1` to exercise route-state
  plumbing while keeping native vLLM execution.
- Preserve vLLM native cudagraph correctness as the baseline.

## Current Route-State Smoke

Artifact: `artifacts/qwen3_128_32_smoke.json`

Settings:

- Model: `/mnt/geogpt-doc-new/default/xb/qwen3-8B`
- Prompt IDs: generated once and reused across cases
- Input/output: exact `128 / 32` tokens
- Sampling: `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS ignored,
  `min_tokens=max_tokens=32`
- CUDA Graph: PIECEWISE, capture sizes `[1, 2, 4, 8]`,
  `backend="eager"`, `enforce_eager=False`
- Warmup/repeats: `1 / 2`

Result:

| Case | Route state | Graph evidence | Validation |
|---|---|---|---|
| `native-graph` | no plugin routes requested | `num_cudagraph_captured=148` | `validation_errors=[]` |
| `plugin-fallback-graph` | all nine routes requested, all native fallback | `num_cudagraph_captured=148` | `validation_errors=[]` |

Output health for both cases: exact input count `128`, exact generated count
`32`, readable decoded preview, replacement/control chars `0`, and no
degenerate repetition flagged. Output-only TPS in this quick smoke was
`53.11` native graph and `53.27` plugin fallback graph; this is a sanity
comparison, not a formal throughput conclusion.

## Current All-Routes InfiniCore Smokes

Eager artifact:
`artifacts/qwen3_128_32_all_routes_streamed_strict_eager.json`

Graph artifact:
`artifacts/qwen3_128_32_all_routes_streamed_strict_graph.json`

Settings:

- Model: `/mnt/geogpt-doc-new/default/xb/qwen3-8B`
- Routes: `VLLM_INFINICORE_ROUTES=all`
- Strict backend: `VLLM_INFINICORE_STRICT_BACKEND=1`
- Native fallback forcing: `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0`
- vLLM eager mode: `enforce_eager=True`
- vLLM graph mode: PIECEWISE cudagraph with `backend="eager"` and
  `enforce_eager=False`
- Input/output: exact `128 / 32` tokens
- Sampling: `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS ignored,
  `min_tokens=max_tokens=32`

Result:

| Case | Installed routes | Graph captures | Validation |
|---|---|---:|---|
| `custom-eager` | all nine scoped routes | 0 | `validation_errors=[]` |
| `custom-graph` | all nine scoped routes | 148 | `validation_errors=[]` |

Backend call counts in the all-routes eager measured iteration:

| Backend wrapper | Calls |
|---|---:|
| `embedding` | 32 |
| `linear` | 4608 |
| `lm_head` | 32 |
| `paged_attention_prefill` | 36 |
| `paged_attention_decode` | 1116 |
| `rms_norm` | 2336 |
| `rotary_embedding` | 1152 |
| `silu_and_mul` | 1152 |
| `store_kv_cache` | 1152 |

Backend call counts in the all-routes graph measured iteration:

| Backend wrapper | Calls |
|---|---:|
| `embedding` | 1 |
| `linear` | 144 |
| `lm_head` | 32 |
| `paged_attention_prefill` | 36 |
| `paged_attention_decode` | 1116 |
| `rms_norm` | 73 |
| `rotary_embedding` | 36 |
| `silu_and_mul` | 36 |
| `store_kv_cache` | 36 |

Graph note:

- The previous graph replay failure was caused by launching `_infinicore`
  kernels on InfiniCore's runtime stream without joining that stream to
  PyTorch's captured stream.
- The backend now wraps the InfiniCore stream with `torch.cuda.ExternalStream`
  and adds `wait_stream` dependencies around each launch.
- Python backend call counters in graph mode count graph capture and
  non-captured paths; replay of captured non-attention ops does not re-enter
  Python, so `graph_capture_count` and decoded-output validation remain part of
  the evidence.

## Acceptance Before Enabling A Route

Before any route can be enabled by default, it needs:

- Unit-level numeric comparison against the native vLLM path.
- 128 input / 32 output correctness smoke with exact token accounting.
- Output preview validation and `validation_errors=[]`.
- 2048 input / 512 output warmup and repeated measurement.
- CUDA Graph capture evidence when used in graph mode.
- No reliance on historical TPS tables for the conclusion.

## Throughput Route Policy

For current graph-mode all-operator throughput runs, use the full
vLLM-InfiniCore route profile:

```text
VLLM_INFINICORE_ROUTES=all
```

This expands to all nine scoped Qwen3 routes:
`RMSNorm,SiluAndMul,RoPE,Embedding,MatMul,LMHead,StoreKVCache,`
`PagedAttentionPrefill,PagedAttentionDecode`.

Isolation data remains useful for diagnosis only. It must not be used as the
delivery configuration when the requirement is that every scoped called
operator routes through InfiniCore. The current bottleneck is the
paged-attention kernel pair, not KV cache update. Isolation at `bs=8`,
`input_len=4096`, `output_len=128`, graph mode showed:

| Routes | Output TPS | Artifact |
|---|---:|---|
| `StoreKVCache,PagedAttentionPrefill,PagedAttentionDecode` | 17.90 | `artifacts/attention-wrapper-cache-bs8-in4096-out128-graph-20260505-142100` |
| `StoreKVCache,PagedAttentionPrefill` | 60.84 | `artifacts/attention-no-decode-bs8-in4096-out128-graph-20260505-142722` |
| `PagedAttentionPrefill` | 61.65 | `artifacts/attention-prefill-only-bs8-in4096-out128-graph-20260505-143036` |
| `StoreKVCache` | 168.66 | `artifacts/attention-storekv-only-bs8-in4096-out128-graph-20260505-142918` |

The latest full-route benchmark at `bs=8`, `input_len=4096`,
`output_len=512`, graph mode was:

| Engine/routes | Output TPS | Graph captures | Artifact |
|---|---:|---:|---|
| vLLM native | 283.29 | 148 | `artifacts/all-routes-gap-ablation-bs8-in4096-out512-graph-20260505-165647` |
| vLLM-InfiniCore `all` | 262.41 | 148 | `artifacts/all-routes-after-rope-opt-bs8-in4096-out512-graph-20260505-172113` |

The `all` run installed all nine scoped routes and recorded nonzero InfiniCore
calls for Embedding, RMSNorm, MatMul/Linear, RoPE, StoreKVCache,
PagedAttentionPrefill, SiluAndMul, LMHead, and PagedAttentionDecode.
`PagedAttentionPrefill` now dispatches to InfiniCore's FlashAttention-wrapped
`mha_varlen`, and `PagedAttentionDecode` dispatches to
the plugin C++ bridge to InfiniCore `mha_kvcache_`. This keeps the
all-scoped-operators requirement while avoiding the slower Python
`infinicore.paged_attention` decode wrapper.

Older `bs=8`, `input_len=4096`, `output_len=512` runs reduced the full-route
result from `43.17` to `211.73`, then to `262.41` output tok/s after the RoPE
wrapper optimization. A later 95% follow-up retained per-device InfiniCore
stream pointer caching and measured `262.97` output tok/s for
`VLLM_INFINICORE_ROUTES=all` against `280.93` same-run vLLM native (`93.61%`).

`PagedAttentionDecode` now routes through the plugin C++ bridge by default.
The bridge calls InfiniCore `mha_kvcache_` below the Python wrapper layer and
records bridge counters. It can be disabled for A/B tests with
`VLLM_INFINICORE_DISABLE_CPP_BRIDGE=1`.

`LMHead` remains opt-in through:

```text
VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecode,LMHead
```

On the current Qwen3-4B single-GPU decision shape (`bs=8`, `input_len=1024`,
`output_len=512`), default bridged decode measured `412.44` output tok/s
against `417.31` vLLM native (`98.8%`) with 148 graph captures and no native
fallback. Artifact:
`artifacts/single-gpu-cpp-decode-qwen3-4b-20260603-200530`.

## Attention Backend Status

The attention/KV route group is now installed through a vLLM attention backend
override rather than through per-method monkey patches. When these routes are
requested, `FLASH_ATTN` is re-registered to:

```text
vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend
```

The implementation reuses the MetaX/vLLM FlashAttention metadata builder and
KV-cache layout, then dispatches supported KV update, prefill, and decode paths
to `_infinicore`. Current validation artifacts:

- Eager `128/32` custom attention-backend smoke:
  `artifacts/attention_backend_custom_eager_128_32_v3.json`
- Graph `bs=2`, `128/32` attention-backend smoke:
  `artifacts/attention-backend-smoke-bs2-in128-out32-v3`

Eager validation shows backend-level nonzero calls for StoreKVCache,
PagedAttentionPrefill, and PagedAttentionDecode. Graph validation shows
backend-level nonzero calls for StoreKVCache and PagedAttentionDecode with
`graph_capture_count=148`; the prefill forward in that graph smoke currently
falls back to the platform backend and remains a follow-up item.
