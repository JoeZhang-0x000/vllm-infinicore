# Development Log

## 2026-06-04 Stage Four No-MetaX Remote Smoke Hardening

Hardened `tests/remote/run_qwen_smoke.py` so no-MetaX is a first-class remote
smoke path rather than only a benchmark-harness mode:

- Removed module-import side effects; environment setup and plugin registration
  now happen inside `main()`, so the module is unit-testable.
- Added runtime bootstrap for MACA, InfiniCore, torch, and loader paths.
- Added a one-time `os.execvpe()` re-exec after setting `LD_LIBRARY_PATH` so
  `libinfinicore_cpp_api.so` is visible to the dynamic loader before importing
  InfiniCore wrappers.
- Added `VLLM_SMOKE_FORBID_METAX_LOAD=1`, which selects
  `VLLM_PLUGINS=infinicore,vllm_infinicore` by default and fails the smoke if
  `vllm_metax` is loaded locally or inside Ray workers.
- Added exact output-token validation with `min_tokens=max_tokens`,
  `ignore_eos=True`, `temperature=0.0`, `top_p=1.0`, and `top_k=1`.
- Ray smoke now propagates the runtime environment and checks worker-side
  `vllm_metax` load state through `collective_rpc`.

Remote validation:

- Single-card no-MetaX smoke:
  `MODEL=/mnt/geogpt-doc-new/default/xb/qwen3-8B`,
  `VLLM_SMOKE_FORBID_METAX_LOAD=1`, `VLLM_SMOKE_MAX_MODEL_LEN=128`,
  `VLLM_SMOKE_MAX_TOKENS=1`, `VLLM_SMOKE_ENFORCE_EAGER=1`:
  `VLLM_SMOKE_OK`, `OUTPUT_TOKEN_COUNT 1`, `vllm_metax_loaded False`.
- Two-card Ray no-MetaX smoke with `CUDA_VISIBLE_DEVICES=0,1`,
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`,
  `VLLM_TENSOR_PARALLEL_SIZE=2`, and
  `VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray`:
  `VLLM_SMOKE_OK`, `OUTPUT_TOKEN_COUNT 1`, `vllm_metax_loaded False`.

The smoke still logs vLLM's native FlashAttention/Triton probe errors on this
MACA stack, but the route registration, generation, exact token count, and
no-`vllm_metax` checks all pass.

## 2026-06-04 No-MetaX Qwen3 128/32 Stage Three

Extended `scripts/qwen3_128_32_smoke.py` so no-MetaX validation uses the same
prompt-token and measurement harness as the graph smoke:

- Added `no-metax-eager` and `no-metax-graph` cases.
- Added `--plugins` so the harness can run with
  `VLLM_PLUGINS=infinicore,vllm_infinicore` without being overwritten by the
  historical MetaX default.
- Added `--forbid-metax-load` / case-level validation that fails if any
  `vllm_metax` module is present in `sys.modules`.
- Artifacts now record the effective plugin environment, selected attention
  backend, `vllm_metax_loaded`, and InfiniCore backend/attention/bridge
  counters.

Remote validation on the MetaX C550 machine:

```bash
python scripts/qwen3_128_32_smoke.py \
  --trust-remote-code \
  --warmup 1 \
  --repeats 2 \
  --cases no-metax-eager,no-metax-graph \
  --output-json artifacts/qwen3_128_32_no_metax_stage3.json \
  --output-dir artifacts/qwen3_128_32_no_metax_stage3_cases
```

Result:

| Case | Validation | Graph captures | `vllm_metax_loaded` | Output TPS |
|---|---|---:|---|---:|
| `no-metax-eager` | `validation_errors=[]` | 0 | `False` | 12.46 |
| `no-metax-graph` | `validation_errors=[]` | 148 | `False` | 40.66 |

Both cases used `VLLM_PLUGINS=infinicore,vllm_infinicore`,
`VLLM_INFINICORE_ROUTES=all`, and
`VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0`. Both installed all nine scoped
routes: `RMSNorm`, `SiluAndMul`, `RoPE`, `Embedding`, `MatMul`, `LMHead`,
`StoreKVCache`, `PagedAttentionPrefill`, and `PagedAttentionDecode`.

Stage-three graph counters included nonzero InfiniCore calls for every scoped
route, `backend_prefill_infinicore=72`, `backend_decode_infinicore=2232`, and
`PagedAttentionDecode` C++ bridge calls `2232`. vLLM still logs native
FlashAttention probe failures on this MACA stack (`libcudart.so.12` missing),
but runtime attention used the InfiniCore backend and the strict no-MetaX
module-load check passed.

This closes the current single-card no-`vllm_metax` Qwen3 128/32 eager + graph
smoke target. Multi-card no-MetaX validation and larger throughput benchmarks
remain future work.

### Stage Three Graph-Safe Strict Check

Re-ran the no-MetaX graph case with strict backend validation enabled:

```bash
VLLM_INFINICORE_STRICT_BACKEND=1 \
python scripts/qwen3_128_32_smoke.py \
  --trust-remote-code \
  --warmup 1 \
  --repeats 3 \
  --cases no-metax-graph \
  --output-json artifacts/qwen3_128_32_no_metax_graphsafe_stage3.json \
  --output-dir artifacts/qwen3_128_32_no_metax_graphsafe_stage3_cases
```

Result:

- `valid=True`, `validation_errors=[]`
- `VLLM_PLUGINS=infinicore,vllm_infinicore`
- `vllm_metax_loaded=False`
- PIECEWISE graph with `backend="eager"` and `num_cudagraph_captured=148`
- three measured graph replays, each with exact `128` input tokens and `32`
  output tokens
- all nine scoped routes installed, with `native_fallback_routes=[]` and
  `skipped_routes=[]`
- graph counters included `store_kv_cache=108`,
  `paged_attention_prefill=108`, `paged_attention_decode=3348`,
  `backend_decode_infinicore=3348`, and C++ bridge
  `PagedAttentionDecode=3348`

This is the current graph-safe evidence for the single-card no-`vllm_metax`
Qwen3 128/32 path. It validates capture plus replay correctness for this shape;
it is not yet a multi-card or long-context graph-safety claim.

### Stage Three Coverage Benchmarks

Extended `scripts/qwen3_three_engine_throughput.py` so throughput runs can
exercise the no-MetaX platform path:

- Added `--vllm-infinicore-plugins` and `--vllm-native-plugins` to remove the
  historical hardcoded `VLLM_PLUGINS=metax,vllm_infinicore`.
- Added `--forbid-metax-load` so vLLM benchmark artifacts fail validation when
  `vllm_metax` is present in `sys.modules`.
- Artifacts now record the effective vLLM plugin environment, selected platform,
  `vllm_metax_loaded`, strict backend state, and route counters.
- Ray tensor-parallel runs now aggregate worker cudagraph counters through
  collective RPC; the TP=2 smoke reports `296` total captures, matching two
  workers with `148` captures each.

Remote no-MetaX coverage results:

| Coverage | Shape | TP/backend | Repeats | Graph captures | Output TPS | Validation |
|---|---|---|---:|---:|---:|---|
| Harness smoke | `bs=1,in=128,out=32` | `1` | 1 | 148 | 39.72 | `validation_errors=[]` |
| Multi-card | `bs=1,in=128,out=32` | `2,ray` | 1 | 296 | 37.86 | `validation_errors=[]` |
| Long context | `bs=1,in=4096,out=128` | `1` | 1 | 148 | 36.51 | `validation_errors=[]` |
| Large batch | `bs=8,in=1024,out=128` | `1` | 1 | 148 | 247.79 | `validation_errors=[]` |
| Formal throughput | `bs=8,in=4096,out=512` | `1` | 3 | 148 | 213.20 | `validation_errors=[]` |

All runs used `VLLM_PLUGINS=infinicore,vllm_infinicore`,
`VLLM_INFINICORE_ROUTES=all`, `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0`, and
`VLLM_INFINICORE_STRICT_BACKEND=1`. Every run reported
`vllm_metax_loaded=False`, all nine scoped routes installed,
`native_fallback_routes=[]`, and `skipped_routes=[]`.

Formal throughput artifact:
`artifacts/no-metax-formal-throughput-bs8-in4096-out512-20260604`.
The formal run measured `12288` output tokens across three iterations. Per-run
TPS stats were mean `213.26`, median `213.83`, min `208.78`, max `217.16`, and
stdev `4.22`.

These runs expand the stage-three evidence from the 128/32 graph-safe smoke to
multi-card startup, long context, large batch, and the current formal
throughput shape. They are no-MetaX vLLM-InfiniCore coverage benchmarks, not a
new comparison against vLLM native or InfiniLM.

## 2026-06-04 No-MetaX Platform Attention Smoke

Added an experimental InfiniCore vLLM platform plugin entry point:

- `vllm.platform_plugins`: `infinicore = vllm_infinicore.platform:register_platform`
- `register_platform()` returns `vllm_infinicore.platform.InfiniCorePlatform`
- platform entry-point discovery stays lazy and does not import torch or vLLM
- platform initialization imports `mcoplib._C` / `mcoplib._moe_C` so vLLM native
  custom ops such as `_C.silu_and_mul` are registered without loading
  `vllm_metax`

The attention backend now respects the selected platform plugin:

- `VLLM_PLUGINS=metax,vllm_infinicore` keeps the existing MetaX-compatible path
  and prefers `vllm_metax.v1.attention.backends.flash_attn`.
- `VLLM_PLUGINS=infinicore,vllm_infinicore` skips importing `vllm_metax`.
- The InfiniCore platform path activates `StoreKVCache`,
  `PagedAttentionPrefill`, and `PagedAttentionDecode` from platform
  registration.
- No-MetaX attention normalizes vLLM native metadata into the decode/prefill
  fields required by the InfiniCore PA/KV wrappers.
- Profile/warmup calls that omit an output buffer or use an invalid temporary KV
  cache return zero-filled profile output rather than falling back to native
  FlashAttention.

Remote validation on the MetaX C550 machine:

- `python -m unittest discover -s tests`: `41` tests passed with `3` skipped.
- `VLLM_PLUGINS=infinicore` pre-register selected
  `vllm_infinicore.platform.InfiniCorePlatform`, activated the three attention
  routes, and reported `vllm_metax_loaded=False`.
- `VLLM_PLUGINS=metax,vllm_infinicore` still selected the MetaX
  FlashAttention base backend and reported `vllm_metax_loaded=True`.
- No-MetaX eager LLM smoke with Qwen3-8B, `max_model_len=128`,
  `max_tokens=1`, and `enforce_eager=True` generated one token and reported
  `NO_METAX_WITH_GENERAL_PLUGIN_SMOKE_OK`.

Smoke counters for the no-MetaX generated request:

| Counter group | Counts |
|---|---|
| InfiniCore backend | `store_kv_cache=36`, `paged_attention_decode=36` |
| Attention backend | `backend_kv_update_infinicore=36`, `backend_decode_infinicore=36`, `backend_forward_infinicore=36` |

The smoke also reported `vllm_metax_loaded=False`. This is a correctness and
runtime-independence smoke, not a graph-safety or throughput benchmark. Graph
mode, multi-card, and full all-operator no-MetaX benchmarks remain future
validation work.

## 2026-06-03 Single-GPU Decode Bridge Default

Re-tested the current Qwen3-4B single-GPU production-debug shape with
`batch_size=8`, `input_len=1024`, `output_len=512`, `warmup=1`, `repeats=2`,
PIECEWISE CUDA graph, and `backend="eager"`.

Baseline all-routes InfiniCore still dispatched `PagedAttentionDecode` through
the Python `infinicore.paged_attention` wrapper and measured only `209.42`
output tok/s against `417.31` vLLM native (`50.2%`). The route counters showed
all nine routes installed with no native fallback and `paged_attention_decode`
called `36792` times, making decode the dominant single-GPU gap.

Enabling the plugin C++ bridge for `PagedAttentionDecode` dispatches the same
route through InfiniCore `mha_kvcache_` and measured `412.44` output tok/s on
the same shape (`98.8%` of the native run). The run was valid with 148 graph
captures, no fallback routes, and bridge counter `PagedAttentionDecode=36792`.

`PagedAttentionDecode` now uses the C++ bridge by default. It can be explicitly
disabled with `VLLM_INFINICORE_DISABLE_CPP_BRIDGE=1` or
`VLLM_INFINICORE_ENABLE_CPP_BRIDGE=0` when comparing against the slower Python
wrapper path. `LMHead` remains opt-in through
`VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecode,LMHead`.

Artifacts:

- `artifacts/single-gpu-decision-qwen3-4b-20260603-195957`
- `artifacts/single-gpu-cpp-decode-qwen3-4b-20260603-200530`

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

## 2026-05-05 InfiniLM vs vLLM Native Reason Analysis

Added `docs/INFINILM_VS_VLLM_REASON.md` to document the current evidence for
why InfiniLM can beat vLLM native in long-output Qwen3-8B graph runs.

Output length sweep at `bs=8`, `input_len=4096`, graph mode:

| Output length | InfiniLM TPS | vLLM native TPS | Faster engine |
|---:|---:|---:|---|
| 32 | 55.94 | 69.30 | vLLM native |
| 128 | 157.30 | 175.60 | vLLM native |
| 512 | 288.05 | 282.72 | InfiniLM |
| 1024 | 332.96 | 311.57 | InfiniLM |

Linear fit of average iteration time against output length:

| Engine | Fixed cost / iteration | Decode step cost | Steady decode TPS at bs=8 |
|---|---:|---:|---:|
| InfiniLM | 3.919 s | 20.184 ms/token-step | 396.35 tok/s |
| vLLM native | 2.916 s | 22.784 ms/token-step | 351.12 tok/s |

Conclusion: InfiniLM's long-output advantage comes from lower steady-state
decode-loop cost, not lower fixed/prefill overhead. vLLM native has lower fixed
cost and wins at short outputs, but InfiniLM's per-step decode cost is lower
once output length is high enough to amortize its fixed cost. A vLLM native
control run with `detokenize=False` did not improve throughput (`282.98` tok/s
vs `283.61` tok/s with detokenize), so text decoding is not the cause.

## 2026-05-05 InfiniCore Attention Backend First Cut

Moved the InfiniCore attention/KV integration from the old method-patch layer
to a vLLM attention backend override:

- Added `vllm_infinicore.ops.vllm_attention_backend`.
- Attention routes now register
  `InfiniCoreFlashAttentionBackend` as vLLM's `FLASH_ATTN` backend.
- The installer wraps MetaX `register_attention_backends()` so MetaX can refresh
  its backend table first, then InfiniCore re-applies its `FLASH_ATTN` override.
- The backend reuses the platform FlashAttention metadata builder and KV cache
  layout, while `InfiniCoreFlashAttentionImpl` routes supported KV update,
  prefill, and decode calls through `infinicore_backend`.
- The old `vllm_attention.py` monkey-patch module is no longer used by the
  attention route installer.

Validation:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests` (`26` tests passed, `2` skipped)
- Registration test confirms `AttentionBackendEnum.FLASH_ATTN.get_path()` is
  `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`.
- Runtime introspection confirmed the earlier issue where MetaX re-registered
  `FLASH_ATTN` after plugin registration; the installer now wraps the MetaX
  registration hook to keep the InfiniCore backend selected.
- Eager Qwen3-8B `128/32` attention-backend smoke:
  `artifacts/attention_backend_custom_eager_128_32_v3.json`
  - `vllm_attention_backend`:
    `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
  - backend `_infinicore` calls:
    `store_kv_cache=1152`, `paged_attention_prefill=36`,
    `paged_attention_decode=1116`
  - backend route counters:
    `backend_kv_update_infinicore=1152`,
    `backend_prefill_infinicore=36`,
    `backend_decode_infinicore=1116`
  - `validation_errors=[]`
- Graph Qwen3-8B `bs=2`, `128/32` attention-backend smoke:
  `artifacts/attention-backend-smoke-bs2-in128-out32-v3`
  - `vllm_attention_backend`:
    `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
  - backend `_infinicore` calls:
    `store_kv_cache=36`, `paged_attention_decode=1116`
  - `graph_capture_count=148`
  - `validation_errors=[]`

Current limitation: in the graph smoke, the prefill attention forward still
falls back to the platform backend (`backend_forward_fallback=36`). Eager mode
exercises InfiniCore prefill correctly. The next performance step is to make
the graph prefill metadata path satisfy the InfiniCore backend's supported
descriptor contract and then re-benchmark long decode.

Follow-up throughput check at the requested production shape (`bs=8`,
`input_len=4096`, `output_len=512`, graph mode, `warmup=1`, `repeats=3`):

```bash
python scripts/qwen3_three_engine_throughput.py \
  --engines vllm-native,vllm-infinicore \
  --batch-size 8 \
  --input-len 4096 \
  --output-len 512 \
  --warmup 1 \
  --repeats 3 \
  --max-model-len 5120 \
  --infinicore-routes StoreKVCache,PagedAttentionPrefill,PagedAttentionDecode \
  --run-dir artifacts/attention-backend-vs-native-bs8-in4096-out512-graph-20260505-135727
```

| Engine | Attention backend | Valid | Output TPS | Graph captures |
|---|---|---:|---:|---:|
| vLLM native | `vllm_metax...MacaFlashAttentionBackend` | true | 280.99 | 148 |
| vLLM-InfiniCore | `vllm_infinicore...InfiniCoreFlashAttentionBackend` | true | 44.93 | 148 |

The vLLM-InfiniCore run installed only the three attention/KV routes and
recorded nonzero backend `_infinicore` calls:
`store_kv_cache=540`, `paged_attention_prefill=540`, and
`paged_attention_decode=55620`. This confirms the backend override is active,
but performance is still approximately `6.25x` slower than vLLM native. The
remaining bottleneck is therefore not the old monkey-patch dispatch itself; it
is the per-token/layer InfiniCore attention backend path, especially decode,
still executing too many Python/backend descriptor/stream-bridge calls.

## 2026-05-05 Attention Gap Isolation And Throughput-Safe Profile

Added a small InfiniCore tensor-wrapper LRU cache for stable attention metadata
and KV cache views. Validation:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests` (`28` tests passed, `2` skipped)
- `git diff --check`
- Added unit coverage for wrapper cache reuse and stride-sensitive keys.

The cache did not materially improve the slow attention profile. A broader
q/out wrapper cache was rejected because it caused CUDA OOM at the Qwen3
benchmark shape by retaining activation buffers.

Route isolation at `bs=8`, `input_len=4096`, `output_len=128`, graph mode:

| Routes | Valid | Output TPS | Artifact |
|---|---:|---:|---|
| `StoreKVCache,PagedAttentionPrefill,PagedAttentionDecode` | true | 17.90 | `artifacts/attention-wrapper-cache-bs8-in4096-out128-graph-20260505-142100` |
| `StoreKVCache,PagedAttentionPrefill` | true | 60.84 | `artifacts/attention-no-decode-bs8-in4096-out128-graph-20260505-142722` |
| `PagedAttentionPrefill` | true | 61.65 | `artifacts/attention-prefill-only-bs8-in4096-out128-graph-20260505-143036` |
| `StoreKVCache` | true | 168.66 | `artifacts/attention-storekv-only-bs8-in4096-out128-graph-20260505-142918` |

Conclusion: `StoreKVCache` is acceptable for the current throughput profile;
`PagedAttentionPrefill` and `PagedAttentionDecode` are the performance-risk
routes. They remain available for correctness/operator coverage, but should not
be used for throughput comparisons until the underlying PA kernels or call
granularity are redesigned.

Implemented `VLLM_INFINICORE_ROUTES=attention-safe`, expanding to
`StoreKVCache`, and updated tests for the alias. The current throughput-safe
configuration is:

```text
VLLM_INFINICORE_ROUTES=throughput,attention-safe
```

Formal graph benchmark at `bs=8`, `input_len=4096`, `output_len=512`,
`warmup=1`, `repeats=3`:

| Engine/routes | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---:|---:|---:|---:|
| vLLM native | true | 286.55 | 286.46 | 148 |
| vLLM-InfiniCore `throughput,attention-safe` | true | 267.32 | 267.31 | 148 |

Artifact:
`artifacts/throughput-attention-safe-vs-native-bs8-in4096-out512-graph-20260505-143335`.
The throughput-safe plugin profile is now `93.3%` of vLLM native graph
throughput on this benchmark.

Requirement correction: this isolation profile is not an acceptable fix when
the target is that every scoped called operator routes through InfiniCore. The
benchmark script default was changed back to `VLLM_INFINICORE_ROUTES=all`, and
the `attention-safe` selector was removed. The isolation results above remain
diagnostic evidence only: they show that the remaining work is to improve the
InfiniCore `PagedAttentionPrefill`/`PagedAttentionDecode` paths rather than
bypassing them.

Reran the requested all-scoped-operator benchmark after the correction:

```bash
python scripts/qwen3_three_engine_throughput.py \
  --engines vllm-native,vllm-infinicore \
  --batch-size 8 \
  --input-len 4096 \
  --output-len 512 \
  --warmup 1 \
  --repeats 3 \
  --max-model-len 5120 \
  --infinicore-routes all \
  --run-dir artifacts/all-routes-vs-native-bs8-in4096-out512-graph-20260505-150547
```

| Engine/routes | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---:|---:|---:|---:|
| vLLM native | true | 282.96 | 282.92 | 148 |
| vLLM-InfiniCore `all` | true | 43.17 | 43.18 | 148 |

The vLLM-InfiniCore run installed all nine scoped routes:
`RMSNorm,SiluAndMul,RoPE,Embedding,MatMul,LMHead,StoreKVCache,`
`PagedAttentionPrefill,PagedAttentionDecode`. Runtime counters were nonzero
for every scoped route family: `embedding=15`, `rms_norm=1095`, `linear=2160`,
`rotary_embedding=540`, `store_kv_cache=540`,
`paged_attention_prefill=540`, `silu_and_mul=540`, `lm_head=1548`, and
`paged_attention_decode=55620`. The all-route performance gap remains open.

Follow-up: switched the PA routes to the FlashAttention-wrapped InfiniCore
operators used by InfiniLM's `FlashAttentionImpl`:

- `PagedAttentionPrefill`: `infinicore.paged_attention_prefill` ->
  `infinicore.mha_varlen`
- `PagedAttentionDecode`: `infinicore.paged_attention` ->
  `infinicore.mha_kvcache`
- KV cache views are presented in BSHD layout for these FA wrapper calls.

Validation:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests` (`28` tests passed, `2` skipped)
- Qwen3-8B all-routes graph smoke:
  `artifacts/qwen3_128_32_all_routes_mha_fa_graph.json`
  with `validation_errors=[]`
- Short all-routes throughput at `bs=8`, `input_len=4096`,
  `output_len=128`: `127.50` output tok/s,
  artifact `artifacts/all-routes-mha-fa-bs8-in4096-out128-graph-20260505-155737`

Formal all-routes graph benchmark after the FA wrapper switch:

| Engine/routes | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---:|---:|---:|---:|
| vLLM native | true | 283.00 | 283.04 | 148 |
| vLLM-InfiniCore `all` | true | 211.73 | 211.59 | 148 |

Artifact:
`artifacts/all-routes-mha-fa-vs-native-bs8-in4096-out512-graph-20260505-155903`.
All nine scoped routes were still installed, and runtime counters remained
nonzero for every route family. The full-route profile improved from `43.17`
to `211.73` output tok/s, reaching `74.8%` of vLLM native graph throughput.

## 2026-05-05 All-Route Gap Ablation And RoPE Optimization

Added an ablation-matrix mode to `scripts/qwen3_three_engine_throughput.py`.
The mode generates one prompt ID manifest and reuses it across graph cases at
`bs=8`, `input_len=4096`, `output_len=512`, `warmup=1`, `repeats=3`.

Ablation artifact:
`artifacts/all-routes-gap-ablation-bs8-in4096-out512-graph-20260505-165647`

| Case | Output TPS | Recovered gap |
|---|---:|---:|
| vLLM native | 283.29 | 100.00% |
| vLLM-InfiniCore `all` | 212.49 | 0.00% |
| `all-minus-matmul-lmhead` | 215.84 | 4.74% |
| `all-minus-rope` | 247.66 | 49.67% |
| `attention-only` | 261.62 | 69.39% |
| `light-known-good` | 266.34 | 76.05% |

Conclusion from the deterministic order:

- Disabling `MatMul,LMHead` recovered only `3.35` tok/s, so Linear/LMHead was
  not the first optimization target.
- The attention-only path was `7.65%` below native, below the `15%` threshold.
- Disabling `RoPE` recovered `35.16` tok/s, so RoPE was the first optimization
  target.

Implemented RoPE wrapper optimizations:

- Cache stable contiguous InfiniCore wrappers for the sin/cos RoPE tables.
- Avoid `torch.cat` reconstruction when `rotary_dim == head_size`; Qwen3-8B
  rotates the full head, so the InfiniCore output can be reshaped directly.
- Keep strict InfiniCore routing active; no scoped route is bypassed in the
  final all-routes benchmark.

Validation after the RoPE optimization:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests` (`28` tests passed, `2` skipped)
- `git diff --check`
- Targeted `all` with `RoPE` disabled:
  `artifacts/target-all-minus-rope-after-rope-opt-bs8-in4096-out512-graph-20260505-171740`
  - Output TPS: `254.32`
  - `graph_capture_count=148`
  - `validation_errors=[]`
- Full all-routes production benchmark:
  `artifacts/all-routes-after-rope-opt-bs8-in4096-out512-graph-20260505-172113`
  - Output TPS: `262.41`
  - Native baseline from the same ablation manifest: `283.29`
  - Ratio: `92.62%` of vLLM native
  - `graph_capture_count=148`
  - `validation_errors=[]`
  - Installed all nine scoped routes.
  - Backend counters were nonzero for Embedding, RMSNorm, MatMul/Linear, RoPE,
    StoreKVCache, PagedAttentionPrefill, SiluAndMul, LMHead, and
    PagedAttentionDecode.
- `128/32` all-routes strict graph smoke:
  `artifacts/qwen3_128_32_all_routes_after_rope_opt_graph.json`
  - `graph_capture_count=148`
  - `validation_errors=[]`
  - Installed all nine scoped routes with nonzero route-family counters.

Current status: vLLM-InfiniCore all-routes graph mode now exceeds the
`>=90%` vLLM-native acceptance target at the production benchmark shape.

## 2026-05-05 95% All-Routes Follow-Up

Attempted to close the remaining all-route gap against same-manifest vLLM
native graph throughput at `bs=8`, `input_len=4096`, `output_len=512`,
`warmup=1`, `repeats=3`.

Implemented and kept:

- Cached the InfiniCore runtime stream pointer per device before constructing
  `torch.cuda.ExternalStream`, avoiding repeated capsule lookup on every
  backend launch.
- Added `--ablation-cases` to `scripts/qwen3_three_engine_throughput.py` so
  focused production ablations can reuse one manifest without running the full
  historical matrix.

Focused post-RoPE ablation artifact:
`artifacts/all-routes-decode-opt-ablation-bs8-in4096-out512-graph-20260505`

| Case | Output TPS | Graph captures | Validation |
|---|---:|---:|---|
| `native` | 283.48 | 148 | `validation_errors=[]` |
| `all` | 257.55 | 148 | `validation_errors=[]` |
| `attention-only` | 264.15 | 148 | `validation_errors=[]` |
| `non-attn-only` | 271.03 | 148 | `validation_errors=[]` |
| `all-minus-rope` | 248.22 | 148 | `validation_errors=[]` |
| `light-known-good` | 268.01 | 148 | `validation_errors=[]` |

Best retained native/all production comparison from this pass:
`artifacts/all-routes-stream-cache-vs-native-bs8-in4096-out512-graph-20260505`

| Engine/routes | Output TPS | Graph captures | Validation |
|---|---:|---:|---|
| vLLM native | 280.93 | 148 | `validation_errors=[]` |
| vLLM-InfiniCore `all` | 262.97 | 148 | `validation_errors=[]` |

This is `93.61%` of same-run vLLM native and does not meet the new `>=95%`
target. All nine scoped routes were installed with no native fallbacks, and
runtime counters were nonzero for every route family:
`embedding`, `rms_norm`, `linear`, `rotary_embedding`, `store_kv_cache`,
`paged_attention_prefill`, `silu_and_mul`, `lm_head`, and
`paged_attention_decode`.

Rejected variants from this pass:

- Decode `q/out` uncached raw-stride wrappers regressed the focused all-route
  case to `257.55` tok/s.
- A non-retaining alias cache for decode `q/out` wrappers was graph-correct but
  regressed production throughput to `258.43` tok/s.
- Reusing retained wrappers for stable linear/LMHead/embedding/RMSNorm weights
  was graph-correct but regressed production throughput to `256.89` tok/s.
- Saturating Python route counters kept nonzero evidence but did not improve
  throughput (`256.90` tok/s), so exact counters were preserved.
- The non-in-place `mha_kvcache` API plus copy-back was graph-correct in the
  `128/32` smoke but slower than the in-place decode path and was not retained.

Conclusion: the `>=95%` all-routes target remains open. The current best
evidence still points to the combined attention decode/LMHead Python/backend
boundary rather than simple wrapper cache misses. Future work should avoid the
rejected descriptor/cache variants above and focus on reducing per-token
attention decode and logits projection call overhead without disabling scoped
routes.

## 2026-05-05 Plugin C++ Bridge Probe

Implemented an opt-in plugin-owned C++ bridge for the hottest remaining route
families without changing InfiniCore or InfiniLM:

- `VLLM_INFINICORE_ENABLE_CPP_BRIDGE=1`
- `VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecode,LMHead`
- The bridge is built on demand through `torch.utils.cpp_extension.load`.
- `PagedAttentionDecode` calls `infinicore::op::mha_kvcache_` from C++ with
  torch tensor raw-pointer views.
- `LMHead` calls `infinicore::op::linear_` from C++.
- The existing Python stream bridge still wraps the C++ launch to preserve
  graph ordering.
- Bridge call counters are now recorded in smoke and throughput artifacts.

Validation:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests`
- `git diff --check`
- C++ bridge load probe succeeded:
  `vllm_infinicore_cpp_bridge.so` built under torch extension cache.
- `128/32` all-routes strict graph smoke with `PagedAttentionDecode` bridged:
  `artifacts/qwen3_128_32_all_routes_cpp_decode_graph.json`
  - `graph_capture_count=148`
  - `validation_errors=[]`
  - all nine routes installed; bridge counter `PagedAttentionDecode=1116`
- `128/32` all-routes strict graph smoke with `PagedAttentionDecode,LMHead`
  bridged:
  `artifacts/qwen3_128_32_all_routes_cpp_decode_lmhead_graph.json`
  - `graph_capture_count=148`
  - `validation_errors=[]`
  - all nine routes installed; bridge counters `PagedAttentionDecode=1116`,
    `LMHead=32`

Production benchmark results at `bs=8`, `input_len=4096`, `output_len=512`,
`warmup=1`, `repeats=3`:

| Bridge routes | Native TPS | All-routes TPS | Ratio | Artifact |
|---|---:|---:|---:|---|
| `PagedAttentionDecode` | 286.35 | 262.20 | 91.57% | `artifacts/all-routes-cpp-decode-vs-native-bs8-in4096-out512-graph-20260505` |
| `PagedAttentionDecode,LMHead` | 286.39 | 263.83 | 92.12% | `artifacts/all-routes-cpp-decode-lmhead-vs-native-bs8-in4096-out512-graph-20260505` |

The bridge path is correct and remains available as an opt-in diagnostic path,
but it is not a throughput win over the previous best retained all-routes run
(`262.97 / 280.93 = 93.61%`). It is therefore not enabled by default and does
not close the `>=95%` target.

Focused bridge-enabled ablation:

- Partial matrix artifact:
  `artifacts/all-routes-cpp-bridge-ablation-bs8-in4096-out512-graph-20260505`
- Separate light-known-good artifact:
  `artifacts/all-routes-cpp-bridge-ablation-light-known-good-bs8-in4096-out512-graph-20260505`

| Case | Output TPS | Graph captures | Validation |
|---|---:|---:|---|
| `native` | 281.13 | 148 | `validation_errors=[]` |
| `all` | 259.09 | 148 | `validation_errors=[]` |
| `attention-only` | 261.22 | 148 | `validation_errors=[]` |
| `non-attn-only` | 271.13 | 148 | `validation_errors=[]` |
| `all-minus-rope` | 254.46 | 148 | `validation_errors=[]` |
| `light-known-good` | 267.80 | 148 | `validation_errors=[]` |

Conclusion: moving only the Python descriptor construction for decode/LMHead
into a C++ extension is insufficient. The remaining gap is more likely in the
underlying InfiniCore decode/logits kernel/API behavior, vLLM scheduling around
attention/logits, or stream synchronization granularity. Further work should
profile kernel time versus stream-wait time and compare the exact InfiniLM C++
execution context before adding more bridge code.
