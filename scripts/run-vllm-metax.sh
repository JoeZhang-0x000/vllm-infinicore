#!/usr/bin/env bash
# Native MetaX vLLM launcher for metax-1; no InfiniCore installation required.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: run-vllm-metax.sh [MODEL_DIRECTORY] [vllm serve options...]
       run-vllm-metax.sh --chat [vllm chat options...]

Defaults: Qwen3-8B, GPU 7, http://127.0.0.1:8000, eager inference.
Overrides: METAX_GPU, METAX_PORT, METAX_HOST, METAX_MODEL.
Example: METAX_GPU=0 run-vllm-metax.sh /path/to/model
HELP
    exit 0
fi

export MACA_PATH=/opt/maca-3.8.0
export MACA_HOME="$MACA_PATH" MACA_ROOT="$MACA_PATH"
export PATH="/opt/conda/bin:$MACA_PATH/bin:$MACA_PATH/ompi/bin:$PATH"
export LD_LIBRARY_PATH="/opt/conda/lib:/opt/conda/lib/python3.10/site-packages/torch/lib:$MACA_PATH/lib:$MACA_PATH/lib64:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib"
export CUDA_VISIBLE_DEVICES="${METAX_GPU:-7}"
export LANG=C.utf8 LC_ALL=C.utf8
export VLLM_PLUGINS=metax,metax_enhanced_customized,metax_enhanced_model
export VLLM_INFINICORE_ENABLE_PATCHES=0
export HF_HUB_OFFLINE=1
unset PYTHONPATH

metax_host="${METAX_HOST:-127.0.0.1}"
metax_port="${METAX_PORT:-8000}"
metax_model="${METAX_MODEL:-/mnt/infinilm/infra/models/Qwen3-8B}"
metax_cli=/opt/conda/bin/vllm

if [[ ! -x "$metax_cli" || ! -d "$MACA_PATH" ]]; then
    echo "Expected /opt/conda/bin/vllm and /opt/maca-3.8.0 on this machine." >&2
    exit 1
fi

if [[ "${1:-}" == "--chat" ]]; then
    shift
    exec "$metax_cli" chat --url "http://$metax_host:$metax_port/v1" "$@"
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
    metax_model="$1"
    shift
fi
if [[ ! -f "$metax_model/config.json" ]]; then
    echo "Model config not found: $metax_model/config.json" >&2
    exit 1
fi

echo "Starting native MetaX vLLM: GPU=$CUDA_VISIBLE_DEVICES model=$metax_model"
echo "API: http://$metax_host:$metax_port/v1; chat in another terminal: $0 --chat"
exec "$metax_cli" serve "$metax_model" \
    --host "$metax_host" --port "$metax_port" \
    --dtype bfloat16 --max-model-len 2048 \
    --gpu-memory-utilization 0.55 --enforce-eager "$@"
