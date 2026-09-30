# 测试入口

在仓库根目录运行 `python -m tests.benchmarks <suite> --help`。四类测试共用设备初始化、数值校验、graph 计时和结果保存，不再依赖固定机器路径的远程启动脚本。

| suite | 用途 |
| --- | --- |
| `model` | vLLM eager / graph 推理、TPS、输出健康度和实际路由检查 |
| `attention` | prefill / decode 成对单算子测试，含独立 CPU FP32 抽样参考 |
| `operators` | RoPE、Embedding、StoreKV、RMSNorm、MatMul、LMHead 等成对测试 |
| `accuracy` | Kunlun 多种子、完整 CPU FP64 / 精确索引参考及 graph 重放稳定性压测 |

CPU 回归测试需要 PyTorch 和 NumPy，不需要模型或设备：

```sh
VLLM_INFINICORE_ENABLE_PATCHES=0 python -m unittest discover -s tests -v
```

## 设备环境

先激活相应厂商的 Python / 驱动环境，安装本项目并选择空闲设备。`--platform` 不负责安装依赖或切换 Python。Kunlun 使用其配套 XPytorch、kunlun_ops 和 glibc 启动环境；`accuracy` 不依赖 vLLM。

| 平台 | 设备选择 | vLLM 原生插件 `VLLM_PLUGINS` |
| --- | --- | --- |
| Ascend | `ASCEND_RT_VISIBLE_DEVICES` | `ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling` |
| MetaX | `CUDA_VISIBLE_DEVICES` | `metax,metax_enhanced_customized,metax_enhanced_model` |
| Kunlun | `XPU_VISIBLE_DEVICES` | `kunlun,kunlun_model,kunlun_quant,kunlun_tool_parser,kunlun_reasoning_parser` |

在已配置设备库的环境中，以下是 MetaX 单算子测试示例：

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_INFINICORE_OPERATOR_BACKEND=metax
export VLLM_INFINICORE_STRICT_BACKEND=1
export VLLM_INFINICORE_ENABLE_PATCHES=0
export VLLM_INFINICORE_ROUTES=RMSNorm,MatMul,LMHead,RoPE,Embedding,StoreKVCache,PagedAttentionPrefill,PagedAttentionDecode
export VLLM_INFINICORE_CPP_BRIDGE_ROUTES="$VLLM_INFINICORE_ROUTES"

python -m tests.benchmarks attention --platform metax \
  --cases prefill:1:128:128,decode:16:1:3072 --output results/attention.json
python -m tests.benchmarks operators --platform metax \
  --ops RoPE,Embedding,StoreKVCache --tokens 16,2048 --output results/operators.json
```

Ascend 还需设置 `VLLM_INFINICORE_ASCEND_LIBRARY` 指向构建出的适配库；7B 单算子测试使用 `attention --heads 28 --kv-heads 4`。MetaX / Kunlun 默认 32 / 8。Kunlun 完整数值压测：

```sh
python -m tests.benchmarks accuracy --extra-down-4096 --output results/accuracy.json
```

## 模型测速

每个配置使用独立进程。native 设置 `VLLM_INFINICORE_ENABLE_PATCHES=0`，`VLLM_PLUGINS` 仅含厂商插件；InfiniCore 设置为 `1` 并在厂商插件列表后追加 `,vllm_infinicore`，显式选择 `VLLM_INFINICORE_ROUTES` 和同名 C++ bridge 路由。`--mode` 用于验收实际运行路径，不会代替环境设置。

```sh
export HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
# 每次切换代码、路由或缓存消融配置都使用新目录，防止复用旧的编译路由。
export VLLM_CACHE_ROOT="$(mktemp -d /tmp/vllm-infinicore-bench.XXXXXX)"
python -m tests.benchmarks model --platform metax --mode infinicore \
  --model /path/to/Qwen3-8B --input-len 2048 --output-len 2048 \
  --batches 16 --repeats 3 --warmup-output-len 32 \
  --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --capture-sizes 1,2,4,8,16,32 --step-timing --output results/model.json
```

加 `--enforce-eager` 可测试 eager。混合组可加 `--expect-native-attention --expect-infinicore-store-all`，MetaX 另加 `--expect-fused-rms`。`model --diagnostic-cache-rope` 与 `operators --cached-rope` 是进程内缓存消融开关，未启用时不会修改生产 RoPE 路径。

结果均写入 JSON，并记录测试模块的 SHA256。数值校验不通过时不发布单算子速度比；Kunlun 无有效设备事件计时时使用同步 wall time。`model` 的输出健康度告警、`accuracy` 的数值差异均返回非零状态，应检查 JSON，不能把“生成了文件”当作验收通过。

本地测量结论、受测 commit 和限制统一保存在 `docs/test-report-2026-09-29.md`。历史脚本、报告生成器和机器启动命令归档在 `artifacts/archive/workspace-cleanup-20260930.tar.gz`，同名 JSON 保存逐文件 SHA256。`docs/` 和 `artifacts/` 是被 Git 忽略的本地实验资料，不随代码仓库分发；运行上述命令可生成自己的结果。历史 JSON 使用当时的脚本，不应声称由当前整理后的入口重新测得。

## Chitu Triton Attention 对照

MetaX 环境中可单独测试 Chitu 的公开 Attention kernel，无需安装整个 Chitu 推理引擎。`--chitu-source` 指向固定版本的源码根目录；测试直接加载未修改的 `device_type.py`、Triton `utils.py`、`attn/decode.py` 和 `attn/prefill.py`，跳过引擎包初始化。记录源码 SHA256，保留上游版权和许可证。首次 JIT、自动调优及预热不计入 graph 延迟。

```sh
CUDA_VISIBLE_DEVICES=7 MACA_PATH=/opt/maca \
VLLM_PLUGINS=metax VLLM_INFINICORE_ENABLE_PATCHES=0 \
TRITON_CACHE_DIR=/tmp/chitu-attention-cache \
python -m tests.benchmarks.chitu_attention \
  --chitu-source /path/to/chitu --chitu-revision FULL_COMMIT_SHA \
  --cases decode:16:1:3072,prefill:16:2048:2048,prefill:16:128:3072 \
  --output results/chitu-attention.json
```

默认使用 BF16、32 query heads、8 KV heads、head dimension 128、page size 16，并打乱物理页顺序。decode 比较同一分页输入的 Chitu 与厂商 FlashAttention；prefill 分别报告连续 K/V 的两侧计算，以及 Chitu 整理分页 K/V 加计算与厂商直接分页计算。所有输出逐元素对照厂商结果，另做独立 CPU FP32 抽样校验和 graph 重放校验。`--ragged --check-only` 可补测每个请求不同的 Q/KV 长度。此入口不修改生产 Attention 路由，也不提供整模型 TPS。
