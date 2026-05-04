# Qwen3 Operator Scope

This document defines the first operator scope for single-node Qwen3 inference. All routes are disabled by default, and requested routes without a proven installer use native vLLM fallback.

| Operator | Category | Planned route | Default | Current installer | Native fallback |
|---|---|---|---|---|---|
| `RMSNorm` | Normalization | C++ PyTorch custom op | Disabled | Python prototype via vLLM OOT custom op | vLLM RMSNorm `forward_native` |
| `SiluAndMul` | MLP activation | C++ PyTorch custom op | Disabled | None | vLLM native activation |
| `RoPE` | Attention position encoding | C++ PyTorch custom op | Disabled | None | vLLM native rotary embedding |
| `Embedding` | Token embedding | C++ PyTorch custom op | Disabled | None | vLLM native token embedding |
| `MatMul` | Linear projections | C++ PyTorch custom op | Disabled | None | vLLM native linear layers |
| `LMHead` | Final projection | C++ PyTorch custom op | Disabled | None | vLLM native logits projection |
| `StoreKVCache` | KV cache update | Deferred PA/KV path | Disabled | None | vLLM native KV cache store path |
| `PagedAttentionPrefill` | Paged attention | Deferred PA/KV path | Disabled | None | vLLM native paged attention prefill backend |
| `PagedAttentionDecode` | Paged attention | Deferred PA/KV path | Disabled | None | vLLM native paged attention decode backend |

## First Phase

The first phase establishes interfaces and routing only:

- Declare target operators and their planned implementation class.
- Keep patching opt-in and disabled by default.
- Allow the first RMSNorm vLLM route only when both
  `VLLM_INFINICORE_ENABLE_PATCHES=1` and
  `VLLM_INFINICORE_ROUTES=RMSNorm` are set.
- Allow `VLLM_INFINICORE_ROUTES=all` for route-state and fallback validation.
- Allow `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=1` to exercise the full route
  table while keeping native vLLM execution.
- Keep PA/KV explicit graph path deferred.
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

## Acceptance Before Enabling A Route

Before any route can be enabled by default, it needs:

- Unit-level numeric comparison against the native vLLM path.
- 128 input / 32 output correctness smoke with exact token accounting.
- Output preview validation and `validation_errors=[]`.
- 2048 input / 512 output warmup and repeated measurement.
- CUDA Graph capture evidence when used in graph mode.
- No reliance on historical TPS tables for the conclusion.
