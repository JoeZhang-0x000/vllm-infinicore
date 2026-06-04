# Architecture

## Goal

`vllm-infinicore` is an out-of-tree vLLM plugin for InfiniCore operator
experiments on single-node Qwen3 inference. It remains default-off, but when
explicitly enabled it now installs InfiniCore-backed eager routes for the full
Qwen3-8B scoped inference operator set.

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

### 2. vLLM Platform Entry

The package also exposes an experimental platform plugin:

```text
vllm_infinicore.platform:register_platform
```

`pyproject.toml` registers this callable under:

```text
vllm.platform_plugins
```

The platform entry point returns:

```text
vllm_infinicore.platform.InfiniCorePlatform
```

The platform module keeps entry-point discovery lightweight: importing
`register_platform()` does not import torch or vLLM. The actual
`InfiniCorePlatform` class is constructed lazily after vLLM selects this
platform plugin. The no-MetaX single-card path has passed Qwen3-8B 128/32
eager and PIECEWISE graph smoke validation with
`VLLM_PLUGINS=infinicore,vllm_infinicore`, all nine scoped routes installed,
exact input/output token accounting, and `vllm_metax_loaded=False`. Multi-card
no-MetaX validation remains future work.

### 3. Python Patch And Route Layer

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

### 4. Custom Op Layer

`vllm_infinicore.ops` reserves the interface for future C++/PyTorch custom ops.

Current route implementation:

- Non-PA operators use PyTorch custom op wrappers backed by InfiniCore Python
  APIs, which call the installed `_infinicore` extension.
- `vllm_infinicore::rms_norm`, `silu_and_mul`, `linear`, `lm_head`,
  `embedding`, and `rotary_embedding` are registered only when explicitly
  loaded by direct custom-op opt-in or patch installation.
- Direct `vllm_infinicore.ops.*` calls remain gated by
  `VLLM_INFINICORE_ENABLE_CUSTOM_OPS`; vLLM route installers may force-load the
  needed custom op wrappers.
- RMSNorm, SiluAndMul, and RoPE use vLLM OOT `CustomOp` replacement classes.
- Embedding patches `UnquantizedEmbeddingMethod.embedding`.
- MatMul patches `UnquantizedLinearMethod.apply`.
- LMHead patches `UnquantizedEmbeddingMethod.apply` for `ParallelLMHead`.
- StoreKVCache and PagedAttention now use an attention-backend override:
  `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
  is registered as vLLM's `FLASH_ATTN` backend after the MetaX backend table is
  refreshed. This keeps the implementation at vLLM attention backend level
  instead of monkey-patching `FlashAttentionImpl.forward`.
- When `VLLM_PLUGINS` explicitly includes `metax`, the attention backend keeps
  the current validated behavior and prefers MetaX's FlashAttention metadata
  builder and KV-cache layout. When `VLLM_PLUGINS` excludes `metax`, it skips
  importing `vllm_metax` and falls back to vLLM's native FlashAttention backend
  classes for metadata shape compatibility, but the platform plugin activates
  the InfiniCore StoreKV/Prefill/Decode routes so runtime attention calls do not
  fall back to vLLM's native FlashAttention implementation. The native fallback
  does not detect a usable FlashAttention version on the current MACA stack.
- In no-MetaX mode, the backend normalizes vLLM native attention metadata into
  the decode/prefill fields needed by the InfiniCore PA/KV wrappers. It also
  handles vLLM profile/warmup calls that omit an output buffer or use an invalid
  temporary KV cache by returning zero-filled profile output instead of calling
  native FlashAttention.
- CPU tensors intentionally use PyTorch fallbacks because the local InfiniCore
  CPU `from_torch` path can crash. Strict backend validation is done on MACA
  device tensors.
- Device launches are wrapped with an InfiniCore/PyTorch stream bridge.
  InfiniCore owns a runtime stream exposed through `infinicore.get_stream()`,
  while vLLM cudagraph capture is ordered through PyTorch streams. The bridge
  wraps the InfiniCore stream as a `torch.cuda.ExternalStream` and adds
  `wait_stream` dependencies before and after each `_infinicore` launch so
  graph capture and replay see the InfiniCore kernels in the correct order.

### 5. Config Layer

`configs/qwen3_infinicore_graph.yaml` documents the planned route table, native fallback, validation path, and graph policy. `vllm_infinicore.config.load_config()` parses it with a structured YAML parser and validates it against the in-code route registry.

The vLLM plugin registration path still does not load this config by default. Config loading is an explicit validation and tooling API until there is a proven safe patch installer.

### 6. Validation Layer

`vllm_infinicore.validation` is pure Python and imports neither torch nor vLLM
at module import. It provides:

- exact input and generated-output token count checks,
- decoded text health counters,
- degenerate repetition detection,
- graph evidence records for cudagraph mode, backend, capture sizes, and log/counter evidence,
- benchmark result records with output-only TPS.

### 7. Qwen3 Smoke Harness

`scripts/qwen3_128_32_smoke.py` generates prompt token IDs once and runs
subprocess-isolated graph cases against `/mnt/geogpt-doc-new/default/xb/qwen3-8B`.
The default cases compare `native-graph` with `plugin-fallback-graph`, where all
Qwen3 routes are requested but forced to native fallback. The stage-three
no-MetaX cases are `no-metax-eager` and `no-metax-graph`; both use
`VLLM_PLUGINS=infinicore,vllm_infinicore`, request `VLLM_INFINICORE_ROUTES=all`,
force no native fallback, and fail validation if any `vllm_metax` module is
loaded.

## CUDA Graph Policy

vLLM native cudagraph is the graph baseline on MetaX. Use PIECEWISE cudagraph with `backend="eager"` and `enforce_eager=False`.

Do not claim patched paths are graph-safe until:

- The exact operator output is validated against vLLM native.
- Actual input and output token counts are recorded.
- Decoded output health is checked.
- Graph capture completes in logs.
- The path avoids graph-unsafe InfiniCore wrapper construction.

The native/fallback graph smoke artifact is
`artifacts/qwen3_128_32_smoke.json`. It records PIECEWISE cudagraph with
`backend="eager"`, `enforce_eager=False`, and `num_cudagraph_captured=148` for
both native graph and all-routes-native-fallback graph. The script does not
explicitly set `CompilationMode.VLLM_COMPILE`; local vLLM logs may still show an
internal compilation mode while also logging that the eager backend disables AOT
compile.

The current all-routes InfiniCore artifacts are:

- `artifacts/qwen3_128_32_all_routes_streamed_strict_eager.json`
- `artifacts/qwen3_128_32_all_routes_streamed_strict_graph.json`
- `artifacts/qwen3_128_32_no_metax_stage3.json`

The eager artifact records `enforce_eager=True`, exact 128 input / 32 output
token validation, and nonzero InfiniCore backend call counts for every scoped
route. The graph artifact records PIECEWISE cudagraph with `backend="eager"`,
`enforce_eager=False`, `num_cudagraph_captured=148`, exact 128 input / 32
output token validation, and all nine scoped routes installed. In graph mode,
Python counters count capture and non-captured paths; captured non-attention
op replay does not re-enter Python, so decoded-output validation and graph
capture evidence are required together with the counters.

The no-MetaX stage-three artifact records the same exact 128/32 validation for
`no-metax-eager` and `no-metax-graph`, with all nine scoped routes installed,
`vllm_metax_loaded=False`, and `num_cudagraph_captured=148` for the graph case.

## Benchmark Policy

All benchmark work must follow the current fairness rules:

- Generate prompt token IDs once.
- Reuse the same prompt IDs across engines.
- Align sampling: `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS disabled.
- Use output-only TPS as the primary metric.
- Warm up before measurement and run repeated iterations.
- Treat old TPS tables as historical until rebenchmarked under these rules.

Current all-operator vLLM-InfiniCore throughput runs should use
`VLLM_INFINICORE_ROUTES=all`, which expands to the full nine scoped Qwen3
routes. Diagnostic isolation profiles may identify bottlenecks, but they are
not acceptable delivery configurations when the requirement is that every
scoped called operator routes through InfiniCore.

The current attention routes use InfiniCore's available PA/FA operators:
prefill dispatches to `infinicore.mha_varlen`, and decode dispatches through
the plugin C++ bridge to InfiniCore `mha_kvcache_`. The slower Python
`infinicore.paged_attention` decode wrapper remains available only for A/B
tests by disabling the bridge. The older fair graph benchmark at
`bs=8`, `input_len=4096`, `output_len=512`, `warmup=1`, `repeats=3` measured
vLLM native at `283.29` output tok/s and vLLM-InfiniCore `all` routes at
`262.41` output tok/s after the RoPE wrapper optimization, both with
`validation_errors=[]` and `148` graph captures. Artifacts:
`artifacts/all-routes-gap-ablation-bs8-in4096-out512-graph-20260505-165647`
and
`artifacts/all-routes-after-rope-opt-bs8-in4096-out512-graph-20260505-172113`.
The all-route profile was previously `92.62%` to `93.61%` of vLLM native at
the older production benchmark shape. Future improvements must remain inside
the all-route InfiniCore path rather than bypassing scoped operators.

`PagedAttentionDecode` now routes through the plugin C++ bridge by default.
The bridge calls InfiniCore `mha_kvcache_` below the Python wrapper layer and
keeps route/counter accounting inside `PagedAttentionDecode`. On the current
Qwen3-4B single-GPU decision shape (`bs=8`, `input_len=1024`,
`output_len=512`) this measured `412.44` output tok/s against `417.31` vLLM
native (`98.8%`) with 148 graph captures and no native fallback. Artifact:
`artifacts/single-gpu-cpp-decode-qwen3-4b-20260603-200530`.

The bridge can be disabled for A/B tests with
`VLLM_INFINICORE_DISABLE_CPP_BRIDGE=1`. `LMHead` remains opt-in through
`VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecode,LMHead`.
