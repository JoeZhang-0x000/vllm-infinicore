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

`vllm_infinicore.patching` owns the route declarations for Qwen3 operators. In the bootstrap it returns registration metadata only. Future patch installers should be added here behind explicit config and environment gates.

Default behavior:

- No monkey patches.
- No model import.
- No torch import.
- No InfiniCore runtime import.
- No CUDA Graph behavior changes.

This keeps dry import safe and avoids disturbing the vLLM native cudagraph baseline.

### 3. Custom Op Layer

`vllm_infinicore.ops` reserves the interface for future C++/PyTorch custom ops.

First-stage direction:

- Non-PA operators use PyTorch custom op wrappers backed by InfiniCore kernels.
- PA/KV operators stay deferred for a dedicated descriptor and graph-safety phase.
- The Python side should avoid dynamic object creation inside captured graph paths.

### 4. Config Layer

`configs/qwen3_infinicore_graph.yaml` documents the planned route table and graph policy. It is intentionally not consumed by the skeleton plugin. When config loading is added, use a structured YAML parser and validate every route field.

## CUDA Graph Policy

vLLM native cudagraph is the graph baseline on MetaX. Use PIECEWISE cudagraph with `backend="eager"` and `enforce_eager=False`.

Do not enable patched paths during graph capture until:

- The exact operator output is validated against vLLM native.
- Actual input and output token counts are recorded.
- Decoded output health is checked.
- Graph capture completes in logs.
- The path avoids graph-unsafe InfiniCore wrapper construction.

## Benchmark Policy

All benchmark work must follow the current fairness rules:

- Generate prompt token IDs once.
- Reuse the same prompt IDs across engines.
- Align sampling: `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS disabled.
- Use output-only TPS as the primary metric.
- Warm up before measurement and run repeated iterations.
- Treat old TPS tables as historical until rebenchmarked under these rules.
