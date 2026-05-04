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
