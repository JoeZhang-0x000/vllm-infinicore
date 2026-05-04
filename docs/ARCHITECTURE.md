# Architecture

## Goal

`vllm-infinicore` is an out-of-tree vLLM plugin for InfiniCore operator experiments on single-node Qwen3 inference. The project starts as a conservative scaffold: it registers cleanly with vLLM and declares the operator scope, but it does not replace vLLM native execution paths yet.

## Layers

### 1. vLLM Plugin Entry

The package exposes:

```text
vllm_infinicore:register
```

`pyproject.toml` registers this callable under:

```text
vllm.general_plugins
```

The local vLLM loader imports entry points from this group and executes each callable with no arguments. `register()` must therefore stay idempotent and safe to execute in multiple vLLM processes.

### 2. Python Patch And Route Layer

`vllm_infinicore.patching` owns the route declarations and runtime route states for Qwen3 operators. Every scoped operator has a declared implementation family, graph policy, native fallback, and validation path.

Default behavior:

- No monkey patches.
- No model import.
- No torch import.
- No InfiniCore runtime import.
- No CUDA Graph behavior changes.

This keeps dry import safe and avoids disturbing the vLLM native cudagraph baseline.

Patch installation is selected by explicit environment gates:

- `VLLM_INFINICORE_ENABLE_PATCHES=1`
- `VLLM_INFINICORE_ROUTES=RMSNorm`, `all`, or a comma-separated route subset
- `VLLM_INFINICORE_DISABLED_ROUTES=...` to remove selected routes
- `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=1` to request routes but keep vLLM native execution

Unknown route names are rejected. Known routes without an installer are recorded
as `native_fallback`, not as an enabled replacement. The registration result
records requested, installed, skipped, disabled, native-fallback, and per-route
state entries. `vllm_infinicore.unregister()` provides an idempotent uninstall
hook for routes owned by this plugin.

### 3. Custom Op Layer

`vllm_infinicore.ops` reserves the interface for future C++/PyTorch custom ops.

First-stage direction:

- Non-PA operators use PyTorch custom op wrappers backed by InfiniCore kernels.
- `vllm_infinicore::rms_norm` exists as a Python PyTorch custom op prototype only when `VLLM_INFINICORE_ENABLE_CUSTOM_OPS` is explicitly truthy.
- The RMSNorm prototype can also be loaded explicitly by the patch installer.
  Direct `vllm_infinicore.ops.rms_norm()` calls remain gated by
  `VLLM_INFINICORE_ENABLE_CUSTOM_OPS`.
- The RMSNorm vLLM route uses `CustomOp.register_oot(name="RMSNorm")` and
  routes only the weighted, non-residual, no variance override path. Residual
  and variant paths fall back to vLLM's PyTorch-native RMSNorm implementation.
- The RMSNorm prototype is not a performance path.
- PA/KV operators stay deferred for a dedicated descriptor and graph-safety phase.
- The Python side should avoid dynamic object creation inside captured graph paths.

### 4. Config Layer

`configs/qwen3_infinicore_graph.yaml` documents the planned route table, native fallback, validation path, and graph policy. `vllm_infinicore.config.load_config()` parses it with a structured YAML parser and validates it against the in-code route registry.

The vLLM plugin registration path still does not load this config by default. Config loading is an explicit validation and tooling API until there is a proven safe patch installer.

### 5. Validation Layer

`vllm_infinicore.validation` is pure Python and imports neither torch nor vLLM
at module import. It provides:

- exact input and generated-output token count checks,
- decoded text health counters,
- degenerate repetition detection,
- graph evidence records for cudagraph mode, backend, capture sizes, and log/counter evidence,
- benchmark result records with output-only TPS.

### 6. Qwen3 Smoke Harness

`scripts/qwen3_128_32_smoke.py` generates prompt token IDs once and runs
subprocess-isolated graph cases against `/mnt/geogpt-doc-new/default/xb/qwen3-8B`.
The default cases compare `native-graph` with `plugin-fallback-graph`, where all
Qwen3 routes are requested but forced to native fallback.

## CUDA Graph Policy

vLLM native cudagraph is the graph baseline on MetaX. Use PIECEWISE cudagraph with `backend="eager"` and `enforce_eager=False`.

Do not enable patched paths during graph capture until:

- The exact operator output is validated against vLLM native.
- Actual input and output token counts are recorded.
- Decoded output health is checked.
- Graph capture completes in logs.
- The path avoids graph-unsafe InfiniCore wrapper construction.

The current graph smoke artifact is
`artifacts/qwen3_128_32_smoke.json`. It records PIECEWISE cudagraph with
`backend="eager"`, `enforce_eager=False`, and `num_cudagraph_captured=148` for
both native graph and all-routes-native-fallback graph. The script does not
explicitly set `CompilationMode.VLLM_COMPILE`; local vLLM logs may still show an
internal compilation mode while also logging that the eager backend disables AOT
compile.

## Benchmark Policy

All benchmark work must follow the current fairness rules:

- Generate prompt token IDs once.
- Reuse the same prompt IDs across engines.
- Align sampling: `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS disabled.
- Use output-only TPS as the primary metric.
- Warm up before measurement and run repeated iterations.
- Treat old TPS tables as historical until rebenchmarked under these rules.
