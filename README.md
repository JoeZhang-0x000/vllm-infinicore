# vllm-infinicore

把 Qwen3 解码路径上已纳入范围的算子路由到 InfiniCore 的 out-of-tree vLLM 插件。
插件默认全部关闭，只有显式开启时才安装路由。目前支持两条线：

| 平台 | 上游平台插件 | 最新结果 | 文档 |
|---|---|---|---|
| Ascend NPU（910B4） | `vllm_ascend` | 2026-09-15 性能矩阵：0.6B 为原生的 24.5%–98.2%，27B 为 95.6%–99.5% | [`docs/ASCEND.md`](docs/ASCEND.md) |
| MetaX GPU（C550） | `vllm_metax`，或本插件自带的 InfiniCore 平台入口 | 2026-09-07 TP 矩阵：为 vllm-metax 的 69%–86% | [`docs/DEV_LOG.md`](docs/DEV_LOG.md) |
| MUSA（Moore Threads） | 本插件的 InfiniCore 平台入口 | 仅打通启动、图路由与 TP 通信，无正式性能结果 | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |

## 术语与命名约定

全部文档统一使用下列写法，请勿在新文档中引入别名：

| 概念 | 统一写法 |
|---|---|
| 本项目 | `vllm-infinicore` |
| 上游平台插件（项目名） | `vllm-ascend` / `vllm-metax` |
| 上游平台插件（Python 模块） | `vllm_ascend` / `vllm_metax` |
| 27B checkpoint | 按目录名 `/models/Qwen3.8-27B`；其 config 声明 `Qwen3_5ForConditionalGeneration` / `qwen3_5`。**不要写成 “Qwen3.5-27B”。** |
| 吞吐指标 | **输出 TPS** = 实际生成 token 总数 / `LLM.generate()` 墙钟耗时（含 prefill、decode 与生成 API 开销，不含加载、编译、预热） |
| 路由总数 | **九条 scoped 路由** = 六条非 attention 路由 + 三条 attention/KV 路由 |
| 日期 | ISO 格式 `2026-09-15` |
| Ascend 图 | 称 ACL graph（vLLM 的配置项名仍是 `CUDAGraphMode`） |

本表约束的是**文档正文**。代码内的运行时字符串沿用既有拼写（例如 `patching.py` 的
`native_fallback="vLLM-Ascend native ..."`，该字符串被 `tests/test_platform_support.py` 断言），
改动它需要同步改测试，不要顺手统一。

## Ascend NPU

`vllm_ascend` 保留平台、设备管理、worker、通信、attention 与 KV cache 的全部所有权；
本插件只提供 InfiniCore 算子适配。自动平台发现同样让位给 Ascend，不注册竞争的 OOT 类或平台运行时。

InfiniCore 固定在官方 `main` 的 `d3551f37538896056e164abf91b120e38c27007b`（2026-09-07 解析），
记录在 [`infinicore.lock.json`](vllm_infinicore/infinicore.lock.json)。在已初始化的 CANN 开发环境中构建：

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

适配器在 **eager 与编译执行下都可用**。算子以 `torch.library` custom op 的形式注册在
`vllm_infinicore_ascend::` 命名空间，Dynamo 因此能把它们 trace 进编译后的程序，ACL graph 也能捕获它们；
能力判定在 trace 时完成，不支持的调用在那里就选原生算子，而不是在图内抛异常。
`VLLM_INFINICORE_ASCEND_GRAPH=0` 可退回 eager-only 行为。

路由覆盖取决于模型架构和每 rank 的分片形状，**安装六条非 attention 路由不等于六条都被调用**：
`/models/Qwen3.8-27B` 在 TP=4、`routes=all` 下执行 Embedding、MatMul、LMHead 和 SiluAndMul
（分片后 MLP intermediate width 为 4352，在 kernel 的 8192 限制内）；Ascend 把 RMSNorm 钉在原生，
该模型也不调用所包装的 RoPE 入口。该 checkpoint 无 TP=1 配置：51.75 GiB 的 BF16 权重超过单张 910B4
可用的 29.49 GiB。未配置 `VLLM_INFINICORE_ASCEND_LIBRARY` 时九条路由全部保持原生。

安装后跑数值探针与 Qwen3-0.6B 检查：

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

模型默认为 `/models/Qwen3-0.6B`，可用 `--model` 覆盖。harness 使用 NPU 0 和共享 prompt ID，
检查输出 token/文本，并从 worker 读取路由状态与计数器；**已安装但 InfiniCore 调用数为 0 的路由
即使允许原生回退也判定为失败**。完整的构建、实现、性能矩阵、正确性与限制见
[`docs/ASCEND.md`](docs/ASCEND.md)。

## MetaX GPU

当前验证的运行时是 **MACA `3.8.0.23`**（Python `3.10.10`、PyTorch `2.10.0+metax3.8.0.7`、
vLLM `0.22.0` 与配套 vllm-metax、FlashAttention `2.6.3+metax3.8.0.7torch2.10`）。
最初的 MACA 3.5.3 开发目标已不再是当前目标。

上游 InfiniCore 用 `--metax-gpu=y --use-mc=y --aten=y` 配置。**MACA 3.8 的 dense/varlen
FlashAttention 导出多了一个 Mars workspace tensor 和一个尾随 bool，而 KV-cache 导出保留旧签名**，
所以上游检出带有一个以 `INFINICORE_METAX_FLASHATTN_38_ABI` 为门控的三文件兼容补丁，
构建时需加 `--cxxflags=-DINFINICORE_METAX_FLASHATTN_38_ABI`。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# MACA 安装路径随主机而变，取该主机实际的版本目录
export MACA_PATH=/opt/maca-<version>
export MACA_HOME=$MACA_PATH
export MACA_ROOT=$MACA_PATH
export INFINI_ROOT=$HOME/.infini
export PYTHON_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
export TORCH_LIB=$PYTHON_SITE_PACKAGES/torch/lib
export LD_LIBRARY_PATH=/opt/conda/lib:$TORCH_LIB:$INFINI_ROOT/lib:$MACA_PATH/lib:$MACA_PATH/lib64:${LD_LIBRARY_PATH:-}

export VLLM_PLUGINS=infinicore,vllm_infinicore
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_INFINICORE_ENABLE_PATCHES=1
export VLLM_INFINICORE_ROUTES=all
export VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0
export VLLM_INFINICORE_STRICT_BACKEND=1
export VLLM_SMOKE_FORBID_METAX_LOAD=1

# vLLM 0.22 必需：V2 model runner 的预热会发出投机解码形状，被严格 attention 包装器拒绝
export VLLM_USE_V2_MODEL_RUNNER=0
# 强制 attention 走 InfiniCore 路径，而不是 FlashAttention 兼容捷径
export VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA=1

pip install -e .
python tests/remote/run_qwen_smoke.py
```

`VLLM_PLUGINS` 不含 `metax` 时，InfiniCore attention 后端跳过 MetaX 后端导入路径，
并由平台插件启用 InfiniCore 的 StoreKV/Prefill/Decode 路由。

测试主机上用两个 wrapper 固定后端并 `exec "$@"`，避免基线被残留变量污染：
`run-infinicore.sh` 设置 `VLLM_PLUGINS=infinicore,vllm_infinicore` 与严格路由设置（每项都可被调用方覆盖），
`run-metax.sh` 设置 `VLLM_PLUGINS=metax` 并主动 unset 所有 `VLLM_INFINICORE_*` 变量。
注意 `vllm chat` 和 `vllm complete` 是 HTTP 客户端，给它们加前缀没有任何作用——后端在服务端启动时就定了。

单节点多卡 Ray 张量并行 smoke：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray
export MODEL=/root/models/Qwen3-8B
python tests/remote/run_qwen_smoke.py
```

Ray 不能为 vLLM worker 改写 `CUDA_VISIBLE_DEVICES`，否则 rank 1 只看得到一张设备，
会在 `torch.cuda.set_device(cuda:1)` 处失败。

> **TP≥4 需要 InfiniCore 侧的修复。** InfiniCore 的 `ContextImpl` 把默认 `Runtime` 硬编码建在设备 0 上，
> 于是每个 TP worker 都在一张它从不计算的卡上多持有一整套运行时，在 MACA 上会让 TP≥4 无限卡在驱动的
> `mxkwCreateQueueBlock ioctl ... timeout`。插件侧已导出 `INFINICORE_DEFAULT_DEVICE_INDEX`，
> 但只对会读该变量的 InfiniCore 构建生效；该改动属于 InfiniCore 仓库，正在向上游提。
> 详见 [`docs/DEV_LOG.md`](docs/DEV_LOG.md) 的 2026-09-07 条目。

## MUSA（Moore Threads）

平台入口在检测到 `torch.musa` 时把设备名/类型切到 `musa`、dispatch key 切到 `MUSA`、
分布式后端切到 `mccl`，并在使用 cudagraph 时设置一组 MUSA 图环境默认值。
`communicator.py` 提供 MUSA communicator 适配，使 torch.distributed 集合通信满足 vLLM 的图检查。
当前流 C++ bridge 在 MUSA 上默认覆盖全部九条路由（`MUSA_DEFAULT_ROUTES`），
与 MetaX 上只默认三条不同。

这条线目前只验证到"能起来、图路由可用、TP 通信可用"，**没有按基准公平性规则测过的性能结果**，
因此不要引用任何 MUSA 吞吐数字。

## 路由与环境变量

| 变量 | 作用 |
|---|---|
| `VLLM_INFINICORE_ENABLE_PATCHES` | 总开关，默认关闭 |
| `VLLM_INFINICORE_ROUTES` | `all`（展开为九条 scoped 路由）或逗号分隔子集 |
| `VLLM_INFINICORE_DISABLED_ROUTES` | 从上面的请求集合中移除若干条，用于隔离实验 |
| `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK` | 请求路由但保持 vLLM 原生执行，用于验证路由状态管线 |
| `VLLM_INFINICORE_STRICT_BACKEND` | 严格后端校验；非严格模式下不支持或失败的调用在路由允许时回退原生 |
| `VLLM_INFINICORE_ASCEND_LIBRARY` | Ascend C API bridge 动态库路径；未设置时九条路由全部保持原生 |
| `VLLM_INFINICORE_ASCEND_GRAPH` | 设为 `0` 退回 eager-only 行为 |

路由清单、安装器与原生回退对照见 [`docs/QWEN3_OP_SCOPE.md`](docs/QWEN3_OP_SCOPE.md)，
分层设计见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 测试

```bash
python -m unittest discover -s tests
```

跑测试套件需要设置 `VLLM_PLUGINS`；不设时只跑 80 个测试且 4 个平台/注册用例报错。
在 MetaX 主机上请通过后端 wrapper 运行以拿到完整套件。
`tests/remote/` 下的脚本需要真实设备，不在本地套件内。
