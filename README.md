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

## Remote Setup

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
