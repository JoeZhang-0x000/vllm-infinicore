# vLLM InfiniCore plugin

通过 vLLM general plugin 将选定算子转发到 InfiniCore，设备、调度和集合通信由厂商 vLLM 插件提供。

InfiniCore 固定为 2026-10-08 适配的 `8254d3b29ba3f052e9f7c1f084fb0a324dd7a817`，子模块版本见 [infinicore.lock.json](vllm_infinicore/infinicore.lock.json)。新版桥接使用 InfiniOps TensorView 和 PyTorch 当前 stream，输出及 workspace 由 PyTorch 分配。

## MetaX 构建

进入现有 MACA / PyTorch / vLLM MetaX 环境后执行：

```sh
git clone https://github.com/InfiniTensor/InfiniCore.git /path/to/InfiniCore
git -C /path/to/InfiniCore checkout --detach 8254d3b29ba3f052e9f7c1f084fb0a324dd7a817
git -C /path/to/InfiniCore submodule update --init --recursive

python -m pip install cmake==3.31.10 ninja libclang==18.1.1
export MACA_PATH=/opt/maca-3.8.0
python scripts/build_infinicore.py --source /path/to/InfiniCore \
  --prefix /path/to/infini-modular --platform metax --jobs 8 \
  --metax-patches upstream
python -m pip install -e .
```

构建脚本校验锁定版本，在独立源码副本中应用 MetaX 优化：hidden=4096 的小批次融合 RMSNorm、并行 Embedding，以及小维度 Q/K RMSNorm。大 prefill 保留 InfiniOps 通用融合 kernel。安装目录的 `manifest.json` 记录补丁提交、获取方式与动态库 SHA256。

三项算子优化已分别提交到 InfiniCore 的算子组件 InfiniOps：

| 优化 | 上游 PR | 固定提交 |
| --- | --- | --- |
| 向量化 Embedding | [#997](https://github.com/InfiniTensor/InfiniOps/pull/997) | `ab2a5af5049a62e3a9a4805c9ba148cc01284722` |
| 小批次 Fused Add+RMSNorm | [#998](https://github.com/InfiniTensor/InfiniOps/pull/998) | `465ff90169f59cb0345a4af427f6ff83ab1eb98a` |
| 小维度 Q/K RMSNorm 线程块 | [#999](https://github.com/InfiniTensor/InfiniOps/pull/999) | `10ad53399aaf0dcba1d2f3d1d9164e9c61da7833` |

MetaX 默认按 [补丁清单](scripts/patches/manifest.json) 从 GitHub 固定提交下载标准 Git 补丁，校验 SHA256 后缓存到构建目录。上游 PR 合入前即可使用。`scripts/patches/*.patch` 暂时保留为相同内容的离线副本，使用 `--metax-patches local`；`--metax-patches none` 构建未优化版本。CUDA 默认不应用这些补丁。切换补丁集时使用新的构建目录。

上游合入并更新本项目的组件锁后，可移除对应补丁及离线副本。构建流程不再依赖 Python 字符串插入算子源码。

使用独立安装目录。`--cxx11-abi` 默认 1，须与 `torch._C._GLIBCXX_USE_CXX11_ABI` 一致；CMake 使用 nlohmann_json 3.12.0，离线环境可传入 `--cmake-option=-Dnlohmann_json_DIR=/path/to/cmake-package`。

## MetaX 使用

```sh
export INFINI_ROOT=/path/to/infini-modular
export LD_LIBRARY_PATH="$INFINI_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_INFINICORE_OPERATOR_BACKEND=metax
export VLLM_INFINICORE_STRICT_BACKEND=1
export VLLM_INFINICORE_ENABLE_PATCHES=1
export VLLM_PLUGINS=metax,metax_enhanced_customized,metax_enhanced_model,vllm_infinicore
export VLLM_INFINICORE_ROUTES=RMSNorm,SiluAndMul,RoPE,Embedding,MatMul,LMHead,StoreKVCache
export VLLM_INFINICORE_CPP_BRIDGE_ROUTES="$VLLM_INFINICORE_ROUTES"

CUDA_VISIBLE_DEVICES=6,7 vllm serve /path/to/Qwen3-8B \
  --tensor-parallel-size 2 --dtype bfloat16
```

七条路由使用 InfiniCore，Attention prefill/decode 使用厂商实现。TP Embedding 复用 vLLM 的分片索引、mask 和 all-reduce；融合 RMSNorm / RoPE 原地执行，Q/K RMSNorm 保留 packed QKV 的步长。

单卡设置一个可见设备和 `--tensor-parallel-size 1`。关闭插件时设置 `VLLM_INFINICORE_ENABLE_PATCHES=0`，并从 `VLLM_PLUGINS` 中移除 `vllm_infinicore`。需要 InfiniCore Attention 时，将 `PagedAttentionPrefill,PagedAttentionDecode` 同时加入两个 routes 变量。

## GSM8K 评测

```sh
python -m pip install -e '.[gsm8k]'
CUDA_VISIBLE_DEVICES=7 vllm-infinicore-gsm8k \
  --model /path/to/Qwen3-8B --mode both --output-dir results/gsm8k
```

从 ModelScope 的 `AI-ModelScope/gsm8k` 下载 `main/test` 全量 1319 题，在独立进程中分别关闭、开启插件。评测入口默认启用全部九条路由，采用 0-shot、非 thinking、贪心解码，按显式 `####` 或 `\boxed{}` 最终答案计算数值精确匹配。输出逐题预测、得分和 `comparison.json`。

`--prepare-only` 仅下载数据；`--dataset-file` 复用冻结数据；`--limit` 指定样本数；`--rescore-only` 离线重评分。也可执行 `python -m vllm_infinicore.benchmarks.gsm8k`，完整参数见 `--help`。Ascend 使用 `--platform ascend` 和 `ASCEND_RT_VISIBLE_DEVICES`。

## 其他入口与兼容范围

- [scripts/run-vllm-metax.sh](scripts/run-vllm-metax.sh)：加载 `metax-1` 的 `/opt/conda`、MACA 3.8.0 环境并启动原生 vLLM；`--chat` 进入聊天。支持 `METAX_GPU`、`METAX_PORT`、`METAX_HOST` 和 `METAX_MODEL`。
- Ascend 使用锁文件的 `legacy_ascend` C API / ABI，通过 [scripts/build_ascend.py](scripts/build_ascend.py) 构建并设置 `VLLM_INFINICORE_ASCEND_LIBRARY`；融合 Add+RMSNorm 及不支持尺寸的 SiLU 保留原生实现。
- CUDA 复用新版桥接，尚未在 NVIDIA 硬件验证；Kunlun 保留旧版接口。

## 项目结构

- `vllm_infinicore/routing/`：路由策略、安装和卸载管理。
- `vllm_infinicore/operators/`：统一调用入口、custom ops、C++ 桥接及各平台适配。`legacy.py` 集中旧版 Python API 兼容逻辑，`torch_ops.py` 集中原生回退实现。
- `vllm_infinicore/benchmarks/`：可安装的 GSM8K 评测入口与评分逻辑。
- `scripts/`：组件构建和启动入口；`infinicore_build/` 负责源码校验、补丁获取和应用。
- `docs/`：性能优化记录；[MetaX 优化小结](docs/metax-optimization-summary.md) 区分算子优化与插件桥接收益。

实验临时脚本和历史测试不属于发布包；构建缓存、库文件与评测结果均由 `.gitignore` 排除。Python 源码格式与检查使用 `ruff format vllm_infinicore scripts` 和 `ruff check vllm_infinicore scripts`，C++ 桥接使用项目的 `.clang-format`。
