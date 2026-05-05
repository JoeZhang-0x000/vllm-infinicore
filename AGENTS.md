# vLLM-InfiniCore Agent Guide

**Updated:** 2026-05-04
**Target machine:** MetaX C550 with MACA 3.5.3. Work locally; no SSH is needed.

## Source Of Truth

Read these first when changing this project:

- `docs/DEV_LOG.md`
- `docs/ARCHITECTURE.md`
- `docs/QWEN3_OP_SCOPE.md`

This project is a clean vLLM plugin scaffold. Older `vLLM-NT` and `InfiniLM` benchmark tables are historical unless a new fair benchmark proves the claim. Do not claim InfiniLM or this plugin is faster than vLLM native cudagraph from old TPS tables.

## Environment

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base

export MACA_PATH=/opt/maca-3.5.3
export MACA_HOME=/opt/maca-3.5.3
export MACA_ROOT=/opt/maca-3.5.3
export INFINI_ROOT=$HOME/.infini
export PYTHON_SITE_PACKAGES=/opt/conda/lib/python3.12/site-packages
export TORCH_LIB=$PYTHON_SITE_PACKAGES/torch/lib
export FLASH_ATTN_2_CUDA_SO=$PYTHON_SITE_PACKAGES/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so

export PATH=/mnt/geogpt-doc-new/default/xmake_env/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$MACA_PATH/bin:$PATH
export LD_LIBRARY_PATH=/opt/conda/lib:$TORCH_LIB:$INFINI_ROOT/lib:$MACA_PATH/lib:$MACA_PATH/lib64
export XMAKE_ROOT=y

# Include both the MetaX platform plugin and this general plugin when testing after installation.
export VLLM_PLUGINS=infinicore,vllm_infinicore
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

Default behavior is conservative. `vllm_infinicore.register()` installs no monkey patches unless a future implementation explicitly enables a safe path. Keep `VLLM_INFINICORE_ENABLE_PATCHES=0` or unset for dry import and baseline runs.

## Current Model

```text
/mnt/geogpt-doc-new/default/xb/qwen3-8B
```

This is Qwen3-8B.

## CUDA Graph Rules

vLLM native cudagraph works on this machine with:

```python
from vllm.config import CUDAGraphMode

comp_config = {
    "cudagraph_mode": CUDAGraphMode.PIECEWISE,
    "cudagraph_capture_sizes": [1, 2, 4, 8],
    "cudagraph_num_of_warmups": 1,
    "backend": "eager",
}
llm = LLM(..., enforce_eager=False, compilation_config=comp_config)
```

Rules:

- Do not use `enforce_eager=True` when measuring cudagraph.
- Do not use `CompilationMode.VLLM_COMPILE` on MetaX unless explicitly testing compile failure modes.
- Use `backend="eager"` to skip torch.compile while keeping cudagraph.
- Keep this plugin graph-conservative until every patched path is proven graph-safe.

## Benchmark Rules

Future benchmark scripts must:

1. Generate prompt token IDs once with the model tokenizer.
2. Use exact same prompt IDs for every engine.
3. Record actual input and generated output token counts.
4. Use output-only TPS as the primary metric.
5. Align sampling: `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS disabled.
6. For vLLM, use `min_tokens=max_tokens=output_len`.
7. Print decoded output preview and text-health counters before trusting TPS.
8. Warm up before measurement and run enough repeats to avoid single-request overhead.

## Validation

Skeleton validation:

```bash
python -m compileall vllm_infinicore
python -c "import vllm_infinicore; vllm_infinicore.register()"
python - <<'PY'
import tomllib
from pathlib import Path
data = tomllib.loads(Path("pyproject.toml").read_text())
print(data["project"]["entry-points"]["vllm.general_plugins"])
PY
```

## Notifications

Use the configured Feishu task update webhook from the parent benchmark project
only for important conclusions or data.

Send a notification for:

- Formal benchmark completion with result data or artifact paths.
- Important correctness, graph-safety, or performance conclusions.
- Blocking failures that need human attention.
- Resolution of a previously reported blocking failure.

Do not send Feishu notifications for routine file edits, ordinary task
completion, small refactors, dry imports, compile checks, intermediate progress,
or non-blocking findings. Prefer a concise final chat update for those cases.
