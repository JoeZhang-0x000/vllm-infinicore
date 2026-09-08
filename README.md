# vLLM InfiniCore Plugin

This repository contains a first-pass vLLM operator plugin for routing the
covered Qwen decoder operator path through InfiniCore. It supports the existing
MetaX platform-plugin stack and now has an experimental InfiniCore platform
entry point for running without loading `vllm_metax`.

It also declares an experimental InfiniCore vLLM platform plugin entry point:

```bash
export VLLM_PLUGINS=infinicore,vllm_infinicore
```

When `VLLM_PLUGINS` does not include `metax`, the InfiniCore attention backend
skips the MetaX backend import path and enables the InfiniCore
StoreKV/Prefill/Decode routes from the platform plugin.

The current no-MetaX single-card closure smoke uses exact 128 input / 32 output
token validation for both eager and PIECEWISE graph modes:

```bash
export VLLM_PLUGINS=infinicore,vllm_infinicore
export VLLM_ENABLE_V1_MULTIPROCESSING=0
python scripts/qwen3_128_32_smoke.py \
  --trust-remote-code \
  --warmup 1 \
  --repeats 2 \
  --cases no-metax-eager,no-metax-graph \
  --output-json artifacts/qwen3_128_32_no_metax_stage3.json \
  --output-dir artifacts/qwen3_128_32_no_metax_stage3_cases
```

The remote stage-three runs installed all nine scoped routes, reported
`vllm_metax_loaded=False`, captured `148` cudagraphs for single-card graph
cases, and captured `296` cudagraphs for the two-card Ray smoke. The current
formal no-MetaX throughput coverage is `bs=8`, `input_len=4096`,
`output_len=512`.

## Ascend NPU Adapter

Keep `vllm_ascend` as the platform/runtime provider. This plugin supplies
InfiniCore operator adapters; device management, workers, communication,
attention and KV cache remain owned by `vllm_ascend`.

InfiniCore is pinned to official `main` revision
`d3551f37538896056e164abf91b120e38c27007b` (resolved 2026-09-07) in
[`infinicore.lock.json`](vllm_infinicore/infinicore.lock.json). Fetch and build
that exact revision in an initialized CANN development environment:

```bash
python scripts/build_ascend.py --build-dir /workspace/infinicore-build \
  --soc Ascend910B4 --cann "$ASCEND_TOOLKIT_HOME"
export VLLM_INFINICORE_ASCEND_LIBRARY=/workspace/infinicore-build/libvllm_infinicore_ascend.so
pip install --no-deps .
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling,vllm_infinicore
export VLLM_INFINICORE_ENABLE_PATCHES=1
export VLLM_INFINICORE_ROUTES=all
export VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0
export VLLM_INFINICORE_STRICT_BACKEND=1
```

Use the SoC matching the target NPU. `--source /path/to/InfiniCore` accepts an
existing clean checkout at the locked revision. The build compiles the required
upstream operator sources without modifying them; optional InfiniCore Python,
InfiniRT and communication components are not required. A manifest records the
revision, SoC, CANN path and library SHA256. The adapter checks the embedded
revision and ABI before installing routes, including in spawned workers.

Use **eager execution** (`enforce_eager=True`) to exercise the real Ascend adapters.
Graph compilation retains native backbone operations; the tested 27B TP=2/4
graph path invokes InfiniCore only for LMHead outside the graph. See the
[graph throughput report](docs/ASCEND_27B_GRAPH_THROUGHPUT.md) for measurements
and correctness limits. Supported eager calls use the current
torch NPU stream and torch-owned tensor storage. Unsupported cases retain the
original Ascend method, with fallback counts/reasons separate from InfiniCore
call counts. Runtime launch errors propagate rather than being silently retried.
Without `VLLM_INFINICORE_ASCEND_LIBRARY`, all nine routes remain native.
Patches remain default-off. Automatic platform discovery also defers to Ascend,
so no competing OOT classes or platform runtime are registered.

Run the numeric probe and Qwen3-0.6B check after installation:

```bash
python tests/remote/probe_ascend_ops.py --output /tmp/ascend-operators.json
python tests/remote/run_ascend_smoke.py prepare --root artifacts/ascend-smoke
python tests/remote/run_ascend_smoke.py native --root artifacts/ascend-smoke
python tests/remote/run_ascend_smoke.py all --root artifacts/ascend-smoke \
  --ascend-library "$VLLM_INFINICORE_ASCEND_LIBRARY" --allow-native-fallback
python tests/remote/run_ascend_smoke.py autoall --root artifacts/ascend-smoke \
  --ascend-library "$VLLM_INFINICORE_ASCEND_LIBRARY" \
  --allow-native-fallback --auto-discover-plugins
```

The model defaults to `/models/Qwen3-0.6B`; override it with `--model`.
The harness uses NPU 0 and shared prompt IDs, checks output tokens/text, and
reads route states and counters from the worker. An installed route with zero
InfiniCore calls fails validation even when native fallback is allowed.
See [the Ascend report](docs/ASCEND_QWEN3_06B_AVAILABILITY.md) for tested operator
coverage and limitations. No NPU performance or graph-safety claim is made.

## MetaX Remote Setup

```bash
cd /root/vllm-infinicore
source /opt/conda/etc/profile.d/conda.sh
conda activate base

export MACA_PATH=/opt/maca-3.5.3
export MACA_HOME=/opt/maca-3.5.3
export MACA_ROOT=/opt/maca-3.5.3
export INFINI_ROOT=$HOME/.infini
export PYTHON_SITE_PACKAGES=/opt/conda/lib/python3.12/site-packages
export TORCH_LIB=$PYTHON_SITE_PACKAGES/torch/lib
export LD_LIBRARY_PATH=/opt/conda/lib:$TORCH_LIB:$INFINI_ROOT/lib:$MACA_PATH/lib:$MACA_PATH/lib64:${LD_LIBRARY_PATH:-}
export VLLM_PLUGINS=infinicore,vllm_infinicore
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_INFINICORE_ENABLE_PATCHES=1
export VLLM_INFINICORE_ROUTES=all
export VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0
export VLLM_INFINICORE_STRICT_BACKEND=1
export VLLM_SMOKE_FORBID_METAX_LOAD=1

pip install -e .
python tests/remote/run_qwen_smoke.py
```

Use `VLLM_INFINICORE_DISABLED_ROUTES=...` to remove a route from the all-route
profile for isolation runs.

Single-node two-card Ray tensor parallel smoke:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_PLUGINS=infinicore,vllm_infinicore
export VLLM_INFINICORE_ENABLE_PATCHES=1
export VLLM_INFINICORE_ROUTES=all
export VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0
export VLLM_INFINICORE_STRICT_BACKEND=1
export VLLM_SMOKE_FORBID_METAX_LOAD=1
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray
export MODEL=/mnt/geogpt-doc-new/default/xb/qwen3-8B
python tests/remote/run_qwen_smoke.py
```

Ray must not rewrite `CUDA_VISIBLE_DEVICES` for the vLLM workers on this
MetaX stack. Otherwise rank 1 sees only one visible device and fails during
`torch.cuda.set_device(cuda:1)`.

For strict backend validation, set `VLLM_INFINICORE_STRICT_BACKEND=1`. In
non-strict mode, unsupported or failing operator calls fall back to native vLLM
paths where the installed route allows it.
