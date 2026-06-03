# MetaX InfiniCore FA2 Build

This note records the InfiniCore changes and runtime configuration needed to
enable the MetaX/MACA FlashAttention 2 path used by `vllm-infinicore`.

## Source Layout

The patched InfiniCore source snapshot from the MetaX test machine is copied to:

```bash
/Users/zhangxin/Desktop/InfiniCore-metax-fa2
```

The remote source of that snapshot was:

```bash
/root/InfiniCore
branch: metax_fla
head: f749d436dc519c6f796555767b9e0670965c0118
```

The snapshot was copied with build outputs excluded. It includes the FA2 source
patches, but the copied submodule git metadata may not be suitable for local
`git status`; use it as a source snapshot unless submodules are reinitialized.

## Required InfiniCore Patches

Three InfiniCore-side fixes were required on MACA 3.5.3:

1. `xmake.lua`

   Define the MACA 3.x FlashAttention ABI switch early in the MetaX `use-mc`
   branch:

   ```lua
   add_defines("INFINICORE_HPCC_VERSION_MAJOR=3")
   ```

2. `xmake.lua`

   Add MetaX cu-bridge and MACA include directories only to the
   `infinicore_cpp_api` ATen target path. Do not add these include directories
   globally, because global injection can break `.maca` device compilation.

   Required include roots:

   ```text
   $MACA_ROOT/tools/cu-bridge/include
   $MACA_ROOT/include
   $MACA_ROOT/include/common
   $MACA_ROOT/include/mcr
   $MACA_ROOT/include/mcblas
   $MACA_ROOT/include/mcdnn
   $MACA_ROOT/include/mcfft
   $MACA_ROOT/include/mcrand
   $MACA_ROOT/include/mcsolver
   $MACA_ROOT/include/mcsparse
   ```

3. `include/infinicore/adaptor/flash_attention_adaptor.hpp` and
   `src/infinicore/ops/multi_head_attention_varlen/mha_varlen_flashattn.cc`

   The installed MetaX `flash_attn_2_cuda` exports `mha_varlen_fwd` with one
   extra trailing `bool` after the Mars extension tensor. The declaration and
   call must match:

   ```cpp
   std::optional<at::Tensor> &flash_attn_mars_ext_,
   bool flash_attn_mars_use_ext_
   ```

   The current call site passes:

   ```cpp
   flash_attn_mars_ext,
   false
   ```

Without these patches, the build either fails on missing MACA/CUDA compatibility
headers or imports with an unresolved `mha_varlen_fwd` symbol.

## Build Environment

Run on the MetaX machine:

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
export LD_LIBRARY_PATH=/opt/conda/lib:$TORCH_LIB:$INFINI_ROOT/lib:$MACA_PATH/lib:$MACA_PATH/lib64:${LD_LIBRARY_PATH:-}
export CUTLASS_ROOT=/root/InfiniCore/third_party/cutlass
export XMAKE_ROOT=y
```

## Build And Install InfiniCore

From the patched InfiniCore checkout:

```bash
cd /root/InfiniCore

xmake f \
  --metax-gpu=y \
  --use-mc=y \
  --aten=y \
  --flash-attn=/root/InfiniCore/third_party/flash-attention \
  -cv

xmake build
xmake install

xmake build _infinicore
xmake install _infinicore

pip install -e .
```

Use this runtime library path for validation and vLLM runs:

```bash
export LD_LIBRARY_PATH=/root/InfiniCore/python/infinicore/lib:/opt/conda/lib:$TORCH_LIB:$INFINI_ROOT/lib:$MACA_PATH/lib:$MACA_PATH/lib64:${LD_LIBRARY_PATH:-}
```

## Validate FA2 Symbols

Check that `libinfinicore_cpp_api.so` depends on the MetaX FlashAttention
extension and that the undefined symbols match the extension ABI:

```bash
ldd /root/InfiniCore/python/infinicore/lib/libinfinicore_cpp_api.so | grep flash_attn
nm -D /root/InfiniCore/python/infinicore/lib/libinfinicore_cpp_api.so | c++filt | grep -E "mha_varlen_fwd|mha_fwd_kvcache"
nm -D "$FLASH_ATTN_2_CUDA_SO" | c++filt | grep -E "mha_varlen_fwd|mha_fwd_kvcache"
```

The `mha_varlen_fwd` signature should include the trailing `bool` on both sides.

## Validate FA2 Decode

Run the direct decode probe from `/root/vllm-infinicore`:

```bash
cd /root/vllm-infinicore
export CUDA_VISIBLE_DEVICES=0

python - <<'PY'
import torch, infinicore
from vllm_infinicore.ops.infinicore_backend import (
    _as_infini,
    _as_infini_strided_cached,
    _set_infinicore_device,
)

torch.cuda.set_device(0)
q = torch.randn((1, 1, 32, 128), device="cuda", dtype=torch.bfloat16)
key_cache_t = torch.randn((8, 32, 16, 128), device="cuda", dtype=torch.bfloat16)
value_cache_t = torch.randn((8, 32, 16, 128), device="cuda", dtype=torch.bfloat16)
seq_lens = torch.tensor([1], device="cuda", dtype=torch.int32)
block_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
out = torch.empty_like(q)

_set_infinicore_device(q)
infinicore.mha_kvcache(
    _as_infini(q),
    _as_infini_strided_cached(key_cache_t),
    _as_infini_strided_cached(value_cache_t),
    _as_infini(seq_lens),
    _as_infini(block_table),
    None,
    1.0 / (128 ** 0.5),
    out=_as_infini(out),
)
torch.cuda.synchronize()
print("MHA_KVCACHE_OK", float(out.float().abs().sum().item()))
PY
```

Expected result:

```text
MHA_KVCACHE_OK ...
```

Known issue: after this rebuild, a pure `import infinicore` process can abort at
interpreter shutdown with `double free or corruption`. The FA2 decode probe and
vLLM inference runs completed successfully despite this shutdown-time issue.

## vLLM Plugin Runtime Config

Use both the MetaX platform plugin and this plugin:

```bash
export VLLM_PLUGINS=metax,vllm_infinicore
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_INFINICORE_ENABLE_PATCHES=1
```

For vLLM cudagraph on MetaX, use PIECEWISE cudagraph with an eager compilation
backend. Do not use `enforce_eager=True` for throughput measurements that are
intended to include graph execution.

```python
from vllm import LLM
from vllm.config import CUDAGraphMode

compilation_config = {
    "cudagraph_mode": CUDAGraphMode.PIECEWISE,
    "cudagraph_capture_sizes": [1, 2, 4, 8],
    "cudagraph_num_of_warmups": 1,
    "backend": "eager",
}

llm = LLM(
    model="/mnt/geogpt-doc-new/default/xb/qwen3-8B",
    enforce_eager=False,
    compilation_config=compilation_config,
    gpu_memory_utilization=0.3,
    max_model_len=512,
)
```

For single-GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/qwen3_three_engine_throughput.py \
  --engines vllm-native,vllm-infinicore \
  --batch-size 1 \
  --input-len 128 \
  --output-len 32 \
  --warmup 1 \
  --repeats 2 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.3 \
  --run-dir artifacts/throughput-single-gpu-native-vs-infinicore-$(date +%Y%m%d-%H%M%S)
```

For single-node two-GPU Ray tensor parallel:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python scripts/qwen3_three_engine_throughput.py \
  --engines vllm-native,vllm-infinicore \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2 \
  --batch-size 1 \
  --input-len 128 \
  --output-len 32 \
  --warmup 1 \
  --repeats 2 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.3 \
  --run-dir artifacts/throughput-ray-tp2-native-vs-infinicore-$(date +%Y%m%d-%H%M%S)
```

The benchmark harness sets deterministic sampling:

```text
temperature=0.0, top_p=1.0, top_k=1, min_tokens=max_tokens=output_len, EOS disabled
```

## Last Verified Results

On MetaX C550 with MACA 3.5.3:

| Scenario | vLLM native | vLLM InfiniCore | InfiniCore/native |
|---|---:|---:|---:|
| Single GPU | 53.34 output tok/s | 51.62 output tok/s | 96.78% |
| Single node, Ray TP2 | 54.64 output tok/s | 54.94 output tok/s | 100.56% |

Artifacts:

```text
/root/vllm-infinicore/artifacts/throughput-single-gpu-native-vs-infinicore-20260603-143641
/root/vllm-infinicore/artifacts/throughput-ray-tp2-native-vs-infinicore-20260603-143838
```
