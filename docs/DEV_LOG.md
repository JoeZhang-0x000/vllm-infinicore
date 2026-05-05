# Development Log

## 2026-05-04 Bootstrap

Created a clean `vllm-infinicore` project skeleton at `/root/vllm-infinicore`.

Initial goals:

- Build an independent vLLM general plugin package.
- Target single-node Qwen3 inference on MetaX C550 with MACA 3.5.3.
- Start with operator route declarations and a dry registration chain only.
- Keep CUDA Graph behavior conservative until patched paths are proven safe.

Trusted facts imported from the benchmark audit:

- The current audit model is `/mnt/geogpt-doc-new/default/xb/qwen3-8B`.
- vLLM native cudagraph works on MetaX when using PIECEWISE cudagraph with `backend="eager"` and `enforce_eager=False`.
- Old TPS tables are historical and must not be used for new performance claims.
- Future benchmarks must use exact prompt token IDs, aligned sampling, output-only TPS, decoded-output validation, warmup, and repeated measurement.

Implemented in the bootstrap:

- `pyproject.toml` declares package `vllm-infinicore`.
- Entry point group is `vllm.general_plugins`.
- Entry point name is `vllm_infinicore`.
- Entry point target is `vllm_infinicore:register`.
- `vllm_infinicore.register()` is idempotent and dry by default.
- `vllm_infinicore.patching` records the initial Qwen3 operator scope.
- `vllm_infinicore.ops` reserves a future C++/PyTorch custom op loader.
- `configs/qwen3_infinicore_graph.yaml` documents conservative route defaults.

Known non-goals for this bootstrap:

- No C++ InfiniCore kernels are implemented.
- No vLLM internals are monkey patched.
- PA/KV explicit graph paths are not enabled.
- No throughput conclusion is made.

Next steps:

1. Add a structured config loader and route validation.
2. Add one minimal non-PA PyTorch custom op prototype behind an explicit env flag.
3. Build a 128/32 correctness smoke that verifies token counts and decoded output health.
4. Add a graph-safety probe before enabling any path during vLLM cudagraph capture.

## 2026-05-04 Foundation Hardening

Implemented the first foundation pass before enabling any vLLM execution path:

- Added a structured YAML config loader and route registry validator.
- Added regression tests for dry registration, entry point metadata, config consistency, and default-off custom op loading.
- Added a minimal RMSNorm PyTorch custom op prototype behind `VLLM_INFINICORE_ENABLE_CUSTOM_OPS`.
- Kept `vllm_infinicore.register()` dry by default with no torch import and no monkey patches.

Still deferred:

- No C++ InfiniCore kernels are implemented.
- No vLLM internals are monkey patched.
- No route is enabled by default.
- No performance conclusion is made.

## 2026-05-04 RMSNorm Opt-In Route

Added the first explicit vLLM integration route while preserving the dry
default:

- `VLLM_INFINICORE_ENABLE_PATCHES=1` plus
  `VLLM_INFINICORE_ROUTES=RMSNorm` installs the RMSNorm route.
- The route uses vLLM's out-of-tree `CustomOp.register_oot(name="RMSNorm")`
  registry and does not edit site-packages.
- `InfiniCoreRMSNorm` routes only weighted RMSNorm calls without residuals or
  variance override to `vllm_infinicore::rms_norm`.
- Residual/fused-add, no-weight, and variance override cases fall back to the
  vLLM PyTorch-native RMSNorm implementation.
- Custom op loading can be forced by the patch installer, while direct
  `vllm_infinicore.ops.rms_norm()` calls remain gated by
  `VLLM_INFINICORE_ENABLE_CUSTOM_OPS`.
- Added `scripts/qwen3_128_32_smoke.py` for the Qwen3-8B 128 input / 32 output
  vLLM baseline correctness smoke.

Still deferred:

- No C++ InfiniCore kernel is implemented; the RMSNorm op is still a Python
  PyTorch custom op prototype.
- No PA/KV, RoPE, MatMul, SiluAndMul, Embedding, or LMHead route is enabled.
- No throughput or graph-safety conclusion is made from this route alone.

## 2026-05-04 Full Route-State And Fallback Framework

Extended the scaffold from a single RMSNorm opt-in route to full Qwen3 operator
coverage:

- Added `RouteState` records for all nine scoped operators.
- Added route selection with `VLLM_INFINICORE_ROUTES=all` or comma-separated
  route subsets.
- Added per-operator disable control through
  `VLLM_INFINICORE_DISABLED_ROUTES`.
- Added `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=1` so the full route table can be
  requested while preserving vLLM native execution.
- Added idempotent uninstall plumbing through `vllm_infinicore.unregister()`.
- Extended YAML config validation with native fallback and validation-path
  fields for every operator.
- Added pure-Python validation utilities for token counts, decoded text health,
  repetition checks, graph evidence, and output-only TPS records.
- Reworked `scripts/qwen3_128_32_smoke.py` into a subprocess-isolated graph
  smoke harness with shared prompt token IDs.

Validation run:

- `python -m compileall vllm_infinicore tests`
- `python -m unittest discover -s tests` (`30` tests passed)
- Dry import/register check with patching disabled
- Full-route native-fallback registration check with
  `VLLM_INFINICORE_ENABLE_PATCHES=1`,
  `VLLM_INFINICORE_ROUTES=all`,
  `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=1`
- Config load and registry consistency check
- Qwen3-8B graph smoke:
  `python scripts/qwen3_128_32_smoke.py --trust-remote-code --warmup 1 --repeats 2 --cases native-graph,plugin-fallback-graph`

Qwen3-8B smoke artifact:

- Summary: `artifacts/qwen3_128_32_smoke.json`
- Native graph: `artifacts/qwen3_128_32_smoke_cases/native-graph.json`
- Plugin fallback graph:
  `artifacts/qwen3_128_32_smoke_cases/plugin-fallback-graph.json`
- Prompt IDs: `artifacts/qwen3_128_32_smoke_cases/prompt-in128.json`

Smoke result:

| Case | Input tokens | Output tokens | Graph captures | Output TPS | Validation |
|---|---:|---:|---:|---:|---|
| `native-graph` | 128 | 32 | 148 | 53.11 | `validation_errors=[]` |
| `plugin-fallback-graph` | 128 | 32 | 148 | 53.27 | `validation_errors=[]` |

The decoded preview was readable for both cases, with replacement/control
characters at `0` and no degenerate repetition flagged. This is a quick graph
and correctness smoke. It is not a formal throughput benchmark or a claim that
the plugin is faster than vLLM native graph; all plugin routes in the fallback
case intentionally used native vLLM execution.

## 2026-05-05 All-Routes InfiniCore Eager Smoke

Implemented actual InfiniCore-backed wrappers for the full Qwen3-8B scoped
operator set:

- `RMSNorm`, `SiluAndMul`, `RoPE`, `Embedding`, `MatMul`, and `LMHead` route
  through `torch.ops.vllm_infinicore.*` wrappers backed by the installed
  `infinicore` Python APIs and underlying `_infinicore` extension.
- `StoreKVCache`, `PagedAttentionPrefill`, and `PagedAttentionDecode` patch the
  vLLM attention backend implementation methods rather than
  `Attention.forward`, preserving vLLM's opaque attention op boundary.
- `LMHead` now patches `UnquantizedEmbeddingMethod.apply` for
  `ParallelLMHead`; the general MatMul path patches
  `UnquantizedLinearMethod.apply`.
- Strict backend mode (`VLLM_INFINICORE_STRICT_BACKEND=1`) raises on wrapper
  failures instead of silently falling back.
- Runtime backend call counters were added to the smoke artifact so route
  installation is not used as the only evidence of InfiniCore execution.

Validation run:

- `python -m compileall vllm_infinicore scripts/qwen3_128_32_smoke.py tests`
- `python -m unittest discover -s tests` (`30` tests passed)
- all-routes install/uninstall check with `VLLM_INFINICORE_ROUTES=all`
- LMHead strict device probe: max diff `0.0`
- attention backend strict StoreKVCache + Decode probe: max diff `0.0`
- Qwen3-8B all-routes strict eager smoke:
  `artifacts/qwen3_128_32_all_routes_strict_eager.json`

Qwen3-8B strict eager smoke result:

| Case | Input tokens | Output tokens | Validation | Output TPS |
|---|---:|---:|---|---:|
| `custom-eager` / `all` routes | 128 | 32 | `validation_errors=[]` | 21.54 |

Measured InfiniCore backend calls in the smoke artifact:

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

Historical graph-safety blocker before the stream bridge:

- all-routes strict graph smoke reached CUDA graph capture but triggered a MACA
  Xnack/ATU fault in a RoPE kernel:
  `_Z23ropeThreadPerItemKernel...`.
- At this point in the log, the all-routes InfiniCore path was validated only
  for eager vLLM inference. The later stream-bridge entry below supersedes this
  status.

## 2026-05-05 InfiniCore Stream Bridge And Graph Smoke

Resolved the graph replay failure by explicitly joining InfiniCore's runtime
stream with PyTorch's current stream around every `_infinicore` launch:

- `infinicore.context::getStream()` is a runtime-owned stream, while vLLM
  cudagraph capture follows PyTorch stream ordering.
- Direct InfiniCore launches during `torch.cuda.CUDAGraph()` produced an empty
  graph in a standalone RMSNorm probe and replayed stale output.
- Wrapping the InfiniCore stream with `torch.cuda.ExternalStream` and adding
  `wait_stream` dependencies before and after the launch made the standalone
  RMSNorm graph replay match the eager result.
- The bridge is now used by RMSNorm, SiluAndMul, MatMul/LMHead, Embedding,
  RoPE, StoreKVCache, PagedAttentionPrefill, and PagedAttentionDecode.
- If the bridge cannot obtain an InfiniCore stream during CUDA graph capture,
  strict mode raises instead of silently taking the old graph-unsafe path.

Validation run after the stream bridge:

- `python -m compileall vllm_infinicore scripts/qwen3_128_32_smoke.py tests`
- `python -m unittest discover -s tests` (`30` tests passed)
- standalone RMSNorm CUDAGraph replay probe: patched max diff `0.03125`
- Qwen3-8B single-route strict graph smokes for `RMSNorm`, `SiluAndMul`,
  `Embedding`, `MatMul`, and `LMHead`: all `validation_errors=[]`
- Qwen3-8B strict graph smoke for non-attention routes except RoPE:
  `artifacts/qwen3_128_32_non_attention_no_rope_streamed_strict_graph.json`
- Qwen3-8B strict graph smoke for `RoPE`:
  `artifacts/qwen3_128_32_route_rope_streamed_strict_graph.json`
- Qwen3-8B all-routes strict eager smoke:
  `artifacts/qwen3_128_32_all_routes_streamed_strict_eager.json`
- Qwen3-8B all-routes strict graph smoke:
  `artifacts/qwen3_128_32_all_routes_streamed_strict_graph.json`

Current Qwen3-8B all-routes strict smoke results:

| Case | Input tokens | Output tokens | Graph captures | Validation |
|---|---:|---:|---:|---|
| `custom-eager` / `all` routes | 128 | 32 | 0 | `validation_errors=[]` |
| `custom-graph` / `all` routes | 128 | 32 | 148 | `validation_errors=[]` |

Measured InfiniCore backend calls in the current all-routes eager artifact:

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

Measured InfiniCore backend calls in the current all-routes graph artifact:

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

In graph mode, Python backend counters are evidence that the wrappers ran
during graph capture or non-captured paths; captured graph replay does not
re-enter Python for non-attention model ops. The output validation and graph
capture count are therefore required alongside these counters.

## 2026-05-05 Qwen3-8B Three-Engine Graph Throughput

Added `scripts/qwen3_three_engine_throughput.py` for fair graph-mode throughput
checks across InfiniLM, vLLM native, and vLLM-InfiniCore:

- Generates one tokenizer prompt ID sequence once and reuses the exact same
  prompt IDs for every engine.
- Uses output-only TPS as the primary metric.
- Records actual per-request input/output token counts, decoded preview, and
  text-health counters.
- Aligns sampling with `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS
  disabled, and vLLM `min_tokens=max_tokens=output_len`.
- Runs vLLM graph with `CUDAGraphMode.PIECEWISE`, capture sizes
  `[1, 2, 4, 8]`, one graph warmup, and `backend="eager"`.
- Runs InfiniLM with `enable_graph_compiling=True`.

Smoke note: InfiniLM batch mode requires paged-cache `num_blocks` to cover all
requests, not just one request. The script therefore uses
`ceil((input_len + output_len) / block_size) * batch_size`.

Formal run:

```bash
python scripts/qwen3_three_engine_throughput.py \
  --engines infinilm,vllm-native,vllm-infinicore \
  --batch-size 8 \
  --input-len 4096 \
  --output-len 512 \
  --warmup 1 \
  --repeats 3 \
  --max-model-len 5120 \
  --run-dir artifacts/qwen3-8b-three-engine-bs8-in4096-out512-graph-20260505-115121
```

Current graph-mode throughput results for this run:

| Engine | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---:|---:|---:|---:|
| InfiniLM | true | 287.95 | 287.95 | n/a |
| vLLM native | true | 286.10 | 286.06 | 148 |
| vLLM-InfiniCore | true | 43.20 | 43.19 | 148 |

All three cases produced `8 * 512 * 3 = 12288` measured output tokens with
`validation_errors=[]`. vLLM-InfiniCore installed all nine Qwen3 scoped routes
with no native fallback routes in this run. Its measured backend route counters
included nonzero calls for embedding, RMSNorm, MatMul/LMHead, RoPE,
StoreKVCache, PagedAttentionPrefill, PagedAttentionDecode, and SiluAndMul.

## 2026-05-05 vLLM-InfiniCore Throughput Bottleneck Isolation

Investigated the vLLM-InfiniCore graph throughput regression from the
three-engine run above.

Key isolation runs at `bs=8`, `input_len=4096`, `output_len=128`, graph mode:

| Route set | Output TPS | Validation |
|---|---:|---|
| vLLM native | 175.38 | valid |
| all InfiniCore routes | 17.32 | valid |
| only StoreKVCache/PagedAttentionPrefill/PagedAttentionDecode | 17.88 | valid |
| all except StoreKVCache/PagedAttentionPrefill/PagedAttentionDecode | 129.56 | valid |
| RMSNorm/SiluAndMul/Embedding | 150.96 | valid |
| RoPE only | 145.75 | valid |

Conclusion: the severe slowdown is dominated by the InfiniCore attention/KV
routes. In this integration those routes still execute through the Python
attention backend wrapper on decode replay (`PagedAttentionDecode` was called
9432 times in the `output_len=128` all-routes run), so graph capture does not
remove the per-token/layer Python wrapper overhead. Disabling those three
routes while keeping non-attention routes restored most throughput.

Follow-up validation at the requested long-output shape (`bs=8`,
`input_len=4096`, `output_len=512`, graph mode, `warmup=1`, `repeats=3`):

| vLLM-InfiniCore route set | Output TPS | Graph captures | Validation |
|---|---:|---:|---|
| all nine scoped routes | 43.20 | 148 | valid |
| all except attention/KV routes | 218.87 | 148 | valid |
| RMSNorm/SiluAndMul/Embedding | 268.21 | 148 | valid |

The route selector now supports `VLLM_INFINICORE_ROUTES=throughput`, which
expands to `RMSNorm,SiluAndMul,Embedding`. The throughput benchmark script uses
that profile by default for vLLM-InfiniCore. Use `--infinicore-routes all`
explicitly when validating full operator coverage. The attention/KV routes
remain available for correctness and coverage probes, but should not be used
for throughput conclusions until they are moved out of the Python replay path
or otherwise proven performant.

Formal three-engine rerun with the throughput profile:

```bash
python scripts/qwen3_three_engine_throughput.py \
  --engines infinilm,vllm-native,vllm-infinicore \
  --batch-size 8 \
  --input-len 4096 \
  --output-len 512 \
  --warmup 1 \
  --repeats 3 \
  --max-model-len 5120 \
  --run-dir artifacts/qwen3-8b-three-engine-bs8-in4096-out512-graph-throughput-profile-20260505-122840
```

| Engine | Route/profile | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---|---:|---:|---:|---:|
| InfiniLM | graph compiling | true | 288.05 | 287.98 | n/a |
| vLLM native | native graph | true | 282.72 | 282.78 | 148 |
| vLLM-InfiniCore | `throughput` | true | 269.14 | 269.17 | 148 |

All three cases produced `12288` measured output tokens with
`validation_errors=[]`. The vLLM-InfiniCore result installed
`RMSNorm,SiluAndMul,Embedding` and no native fallback routes.
