# vLLM InfiniCore Plugin

This repository contains a first-pass vLLM operator plugin that keeps the MetaX
platform plugin for device/runtime integration while routing the covered decoder
operator path through InfiniCore.

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
export VLLM_PLUGINS=metax,vllm_infinicore
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_INFINICORE_ENABLE_PATCHES=1
export VLLM_INFINICORE_ROUTES=all

pip install -e .
python tests/remote/run_qwen_smoke.py
```

Use `VLLM_INFINICORE_DISABLED_ROUTES=...` to remove a route from the all-route
profile for isolation runs.

Single-node two-card Ray tensor parallel smoke:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_PLUGINS=metax,vllm_infinicore
export VLLM_INFINICORE_ENABLE_PATCHES=1
export VLLM_INFINICORE_ROUTES=all
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray
export MODEL=/mnt/geogpt-doc-new/default/infinilm-models/Qwen2.5-0.5B-Instruct
python tests/remote/run_qwen_smoke.py
```

Ray must not rewrite `CUDA_VISIBLE_DEVICES` for the vLLM workers on this
MetaX stack. Otherwise rank 1 sees only one visible device and fails during
`torch.cuda.set_device(cuda:1)`.

For strict backend validation, set `VLLM_INFINICORE_STRICT_BACKEND=1`. In
non-strict mode, unsupported or failing operator calls fall back to native vLLM
paths where the installed route allows it.
