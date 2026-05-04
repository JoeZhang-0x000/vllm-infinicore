# Qwen3 Operator Scope

This document defines the first operator scope for single-node Qwen3 inference. The bootstrap only declares these routes; all routes are disabled by default.

| Operator | Category | Planned route | Default | Graph policy |
|---|---|---|---|---|
| `RMSNorm` | Normalization | C++ PyTorch custom op | Disabled | Requires graph-safety proof |
| `SiluAndMul` | MLP activation | C++ PyTorch custom op | Disabled | Requires graph-safety proof |
| `RoPE` | Attention position encoding | C++ PyTorch custom op | Disabled | Requires graph-safety proof |
| `Embedding` | Token embedding | C++ PyTorch custom op | Disabled | Avoid `from_blob` or `from_torch` inside capture |
| `MatMul` | Linear projections | C++ PyTorch custom op | Disabled | Requires graph-safety proof |
| `LMHead` | Final projection | C++ PyTorch custom op | Disabled | Requires graph-safety proof |
| `StoreKVCache` | KV cache update | Deferred PA/KV path | Disabled | Dedicated descriptor and graph phase |
| `PagedAttentionPrefill` | Paged attention | Deferred PA/KV path | Disabled | Keep native vLLM cudagraph baseline intact |
| `PagedAttentionDecode` | Paged attention | Deferred PA/KV path | Disabled | Keep native vLLM cudagraph baseline intact |

## First Phase

The first phase establishes interfaces and routing only:

- Declare target operators and their planned implementation class.
- Keep patching opt-in and disabled by default.
- Keep PA/KV explicit graph path deferred.
- Preserve vLLM native cudagraph correctness as the baseline.

## Acceptance Before Enabling A Route

Before any route can be enabled by default, it needs:

- Unit-level numeric comparison against the native vLLM path.
- 128 input / 32 output correctness smoke with exact token accounting.
- Output preview validation and `validation_errors=[]`.
- 2048 input / 512 output warmup and repeated measurement.
- CUDA Graph capture evidence when used in graph mode.
- No reliance on historical TPS tables for the conclusion.
