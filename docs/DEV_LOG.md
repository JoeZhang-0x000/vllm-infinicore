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
