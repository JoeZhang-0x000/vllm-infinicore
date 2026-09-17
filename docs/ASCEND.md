# Ascend NPU 集成与性能

本文是 Ascend 侧的唯一权威文档，覆盖适配方式、图集成实现、最新性能矩阵与已知限制。
历史报告（2026-09-07 的 0.6B/27B 可用性与原生骨干图吞吐、2026-09-09 的图集成报告、
2026-09-15 的性能矩阵）已合并至此，原文件已删除。

**最新结果日期：** 2026-09-15 ｜ **主机：** `npu-worker-08` ｜ **容器：** `zx-vllm-ascend-023`

## 1. 分工与定位

`vllm_ascend` 保留平台、设备管理、worker、通信、attention 与 KV cache 的全部所有权。
本插件只提供 InfiniCore 算子适配：包装 Ascend 已有类的方法，不注册竞争的 OOT 类，
不实现自己的设备/worker/通信运行时。自动平台发现同样让位给 Ascend。

未配置 `VLLM_INFINICORE_ASCEND_LIBRARY` 时，九条 scoped 路由全部保持原生。
配置但版本/ABI 不匹配时视为安装失败，而不是降级为可用后端。

## 2. 构建与启用

InfiniCore 固定在官方 `main` 的 `d3551f37538896056e164abf91b120e38c27007b`
（2026-09-07 解析），记录在 [`infinicore.lock.json`](../vllm_infinicore/infinicore.lock.json)。
在已初始化的 CANN 开发环境中构建该精确修订：

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

`--soc` 用目标 NPU 对应的型号；`--source /path/to/InfiniCore` 接受一份处于锁定修订的干净检出。
构建只编译所需的上游算子源码，不做修改，也不需要 InfiniCore Python、InfiniRT 和通信组件。
manifest 记录修订、SoC、CANN 路径与库 SHA256；运行时在安装任何路由前校验内嵌修订与 ABI，
派生的 worker 同样校验。

`VLLM_INFINICORE_ASCEND_GRAPH=0` 可退回到此前的 eager-only 行为。

## 3. 图集成实现

`ascend_backend` 通过 ctypes 和裸设备指针调用 InfiniCore，Dynamo 无法 trace 这种调用。
早期有两个 guard 把编译路径静默降级为纯原生执行：`execute()` 在
`torch.compiler.is_compiling()` 为真时直接返回原生算子，另一个 guard 拒绝 stream capture
期间的任何调用。**因此 2026-09-09 之前发布的全部图测量都是原生对原生。**

移除这两个 guard 需要五项改动：

1. **`operators/ascend/graph_ops.py`** 把每个算子注册为 `vllm_infinicore_ascend::` 命名空间下的
   `torch.library.custom_op` 并提供 fake 实现，使 tracer 能把 ctypes launch 作为不透明节点
   放进图里。`vllm_infinicore::` 命名空间已被默认关闭的 `operators/custom_ops.py` 占用，不能复用。
2. **能力判定前移到 trace time。** `supports_linear`、`supports_silu_and_mul`、
   `supports_rotary_embedding`、`supports_tensor` 仅依据 dtype 和 shape 作答，并与 eager 路径共用，
   两者不可能出现分歧。编译后的图一次性固定算子，所以不支持的调用必须在建节点前就选原生，
   而不是在图内抛异常。
3. **`launch()` 对 capture 安全。** descriptor key 不再包含 stream，capture 期间可复用预热好的
   descriptor；capture 期间跳过 eviction；已录入图的 descriptor 被固定，避免销毁 replay 仍在引用的状态。
4. **workspace 按设备共享到 high-water mark。** 每次 launch 单独分配 workspace 会在 `bs=32` 下
   每批 churn 约 40,000 次 caching allocator，并把一次分配放进每一次 capture。共享一块取最大值
   而不是取总和：27B 形状下共享需 88 MiB，每 descriptor 一块需 342 MiB，后者足以耗尽
   `gpu_memory_utilization` 留出的余量并导致引擎起不来。
5. **`record_stream` 从 launch 路径移除。** 这是单项收益最大的改动，见下节。

## 4. record_stream：收益最大的单项修复

launch 路径曾对每次 launch 的每个 tensor 调用 `tensor.record_stream(stream)`。
只有当 tensor 被用在非其分配所在的 stream 上时这才有意义；本适配器始终在 tensor 自身的当前
stream 上 launch，而 caching allocator 本就对此排序，所以该调用什么也没保护。它并不免费：
`record_stream` 让 allocator 推迟 block 复用直到观察到 stream 事件，而每次调用都新分配一块输出
（2,048-token prefill tile 下为 71 MiB），于是每次分配都在等设备进度而不是流水化。

TP=2、4 序列 prefill 下的移除效果：

| | 耗时 | `output_alloc` rank 0 / rank 1 | host 总计 |
| --- | ---: | ---: | ---: |
| 保留 `record_stream` | 12.15 s / 16.43 s | 166 / 595 µs 每次 | 2.12 s / 5.03 s |
| 移除后 | 2.01 s / 2.02 s | 36.9 / 37.7 µs 每次 | 1.16 s / 1.16 s |

原生同一用例为 1.80 s，即 prefill 从原生的 6.8 倍降到 1.12 倍。

这一处调用同时解释了此前被分别追查的一组现象：TP=2 `bs=4` 上无法解释的 54.2% 单元、
TP=4 `bs=1` 的间歇性停顿（重复测量中 25.7 对 32.3 tok/s）、两个 rank 分配耗时相差 3.6 倍的
不对称、以及"看起来像 allocator 压力但 `expandable_segments:True` 毫无作用"这个矛盾。
它也解释了为什么所有隔离 microbenchmark 都显示持平：空闲设备内存充足，推迟复用永远不阻塞。
影响随内存压力放大，所以 TP=2（约 15,040 cache token）受创严重，而 TP=4（约 200,320）几乎不受影响。

## 5. 算子核对齐

27B TP=2 每 rank 形状下的单次 launch 成本，方法是把 50 次 launch 录进 ACL graph 后计时 replay。
小 `M` 下墙钟循环回答不了这个问题：原生 dispatch 和 ctypes launch 在那里都是 enqueue-bound，
测到的是 launch 开销而完全掩盖了算子核。graph replay 不做 enqueue，这也正是生产 decode 路径的行为。

| 形状 | M | 原生 | InfiniCore | 比值 |
| --- | ---: | ---: | ---: | ---: |
| qkv | 1 | 49.8 µs | 49.5 µs | 0.99x |
| gate_up | 1 | 256.6 µs | 256.1 µs | 1.00x |
| qkv | 4 | 49.4 µs | 49.3 µs | 1.00x |
| gate_up | 4 | 257.1 µs | 257.3 µs | 1.00x |
| down | 16 | 39.2 µs | 39.1 µs | 1.00x |
| gate_up | 32 | 268.4 µs | 268.3 µs | 1.00x |
| qkv | 2048 | 596.7 µs | 602.0 µs | 1.01x |
| gate_up | 2048 | 1841.0 µs | 1848.0 µs | 1.00x |

从单个 decode token 到完整 prefill tile，每个形状上算子核都与原生无法区分，因为两者最终都到达
同一个 CANN aclnn GEMM。图外每次 InfiniCore 调用额外付出约 270 µs 的 Python 开销，
这正是被捕获的步骤省掉、而未捕获的步骤仍要付的部分。

一个值得记录的后果：`torch_npu.npu_format_cast` **不能在 graph capture 期间运行**，会抛
`Cannot run aclop operators during NPU graph capture`。`nd()` 在 tensor 不是 ND 格式时就会调用它，
所以权重被转成 `FRACTAL_NZ` 的模型会 capture 失败而不是变慢。`VLLM_ASCEND_ENABLE_NZ` 默认为 1，
这条路径是可达的；本 checkpoint 的权重已经是 ND，因此未触发。

## 6. 性能矩阵（2026-09-15）

已采集 **18/18** 组原生/插件对比。本轮插件 TPS 均低于原生：0.6B 为原生的 24.5%–98.2%，
27B 为 95.6%–99.5%，没有任何一个测量单元快于原生。最终核对覆盖 108 次测量、900 条输出、
768,000 个生成 token。

### 6.1 配置与指标

- 设备：Ascend 910B4；Qwen3-0.6B 使用 TP=1，`/models/Qwen3.8-27B` 使用 TP=4。
- `/models/Qwen3.8-27B/config.json` 声明 `Qwen3_5ForConditionalGeneration` / `qwen3_5`；
  全文按实际目录名称呼该 checkpoint。
- BF16；batch size 1/8/16；输入/输出 256/256、2048/256、256/2048。
- `enable_chunked_prefill=False`、`enable_prefix_caching=False`、`enforce_eager=False`。
- `CompilationMode.VLLM_COMPILE`、`CUDAGraphMode.FULL_DECODE_ONLY`；capture sizes `[1,2,4,8,16]`。
  Prefill 不在 decode graph 内。
- `max_num_seqs=16`、`max_num_batched_tokens=32768`、`gpu_memory_utilization=0.90`；
  每组 `max_model_len=input_len+output_len`。
- 显式请求 `block_size=128`。Ascend 对 27B hybrid attention 的有效 block size 还会自动调整，见原始日志。
- 每个模型/输入长度只生成一次 token ID，缓存后供两引擎及不同输出长度复用，验证 prompt SHA256 一致。
- temperature=0、top_p=1、top_k=1、ignore_eos=True、min_tokens=max_tokens=输出长度，seed=0。
- 每组完整长度预热 1 次、测量 3 次；表中为输出 TPS 的中位数。
- **输出 TPS = 实际生成 token 总数 / `LLM.generate()` 墙钟耗时**，包含 prefill、decode 和生成 API 开销，
  排除加载、编译、预热和测试脚本的结果分析；不是仅 decode 的 TPS。

### 6.2 Qwen3-0.6B（TP=1）

| 输入/输出 | Batch | 原生 TPS | 插件 TPS | 插件/原生 | 输出完全相同 | 原生/插件重复一致 |
|---|---:|---:|---:|---:|---:|---|
| 256/256 | 1 | 94.93 | 90.85 | 95.7% | 3/3 | 是/是 |
| 256/256 | 8 | 721.41 | 594.39 | 82.4% | 24/24 | 是/是 |
| 256/256 | 16 | 1341.12 | 970.00 | 72.3% | 48/48 | 是/是 |
| 2048/256 | 1 | 95.79 | 77.62 | 81.0% | 3/3 | 是/是 |
| 2048/256 | 8 | 700.91 | 266.11 | 38.0% | 24/24 | 是/是 |
| 2048/256 | 16 | 1231.18 | 301.21 | 24.5% | 48/48 | 是/是 |
| 256/2048 | 1 | 96.10 | 94.39 | 98.2% | 3/3 | 是/是 |
| 256/2048 | 8 | 715.19 | 678.73 | 94.9% | 24/24 | 是/是 |
| 256/2048 | 16 | 1345.39 | 1029.36 | 76.5% | 48/48 | 是/是 |

### 6.3 Qwen3.8-27B（TP=4）

| 输入/输出 | Batch | 原生 TPS | 插件 TPS | 插件/原生 | 输出完全相同 | 原生/插件重复一致 |
|---|---:|---:|---:|---:|---:|---|
| 256/256 | 1 | 32.43 | 31.93 | 98.5% | 1/3 | 否/否 |
| 256/256 | 8 | 222.67 | 217.04 | 97.5% | 11/24 | 否/否 |
| 256/256 | 16 | 396.71 | 388.24 | 97.9% | 21/48 | 否/否 |
| 2048/256 | 1 | 32.52 | 31.45 | 96.7% | 0/3 | 否/否 |
| 2048/256 | 8 | 184.63 | 178.85 | 96.9% | 6/24 | 否/否 |
| 2048/256 | 16 | 289.12 | 276.28 | 95.6% | 15/48 | 否/否 |
| 256/2048 | 1 | 33.90 | 33.69 | 99.4% | 0/3 | 否/否 |
| 256/2048 | 8 | 229.57 | 228.37 | 99.5% | 0/24 | 否/否 |
| 256/2048 | 16 | 413.89 | 409.88 | 99.0% | 0/48 | 否/否 |

两表中，“输出完全相同”按相同 repeat 和请求位置比较完整 token 序列；“重复一致”检查同一引擎
每个请求位置在三次测量中的完整序列。

### 6.4 稳定性

- Qwen3-0.6B：所有测量每 rank `execute_model` 调用数（forward_steps）= 输出长度+1；
  graph replays = 输出长度−1；三次测量 TPS 极差/中位数最大 3.38%。
- Qwen3.8-27B：同样满足上述两项计数；三次测量 TPS 极差/中位数最大 1.18%。

## 7. 路由覆盖

九条 scoped 路由 = 六条非 attention 路由 + 三条 attention/KV 路由。三条 attention/KV 路由
按设计始终保持原生 Ascend。实际覆盖取决于模型架构和每 rank 的分片形状，**安装六条不等于
六条都被调用**。

| 路由 | Qwen3-0.6B TP=1 | Qwen3.8-27B TP=4 |
|---|---|---|
| Embedding | InfiniCore（192 次） | InfiniCore |
| MatMul | InfiniCore（21,504 次） | InfiniCore |
| LMHead | InfiniCore（192 次） | InfiniCore |
| SiluAndMul | InfiniCore（5,376 次） | InfiniCore，分片后 intermediate width=4352 |
| RMSNorm | InfiniCore（10,944 次） | 所包装入口无调用 |
| RoPE | InfiniCore（5,376 次） | 所包装入口无调用 |
| StoreKVCache / PagedAttentionPrefill / PagedAttentionDecode | 原生 Ascend | 原生 Ascend |

0.6B 上只有 `fused_add_rms_norm` 走原生回退（10,752 次）；上游没有 Ascend 的融合 kernel。

27B 上两条路由无调用的原因是架构本身，不是能力不足：

- **RMSNorm**：`vllm_ascend` 把 IR op 优先级钉死在原生
  （`IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])`），所包装入口拿不到调用。
- **RoPE**：64 层中有 48 层是 `linear_attention`，走 gated-delta-rule 路径而非标准 rotary 算子。

> **对早期结论的更正。** 2026-09-09 的报告依据未分片的 `intermediate_size=17408 > 8192`
> 断言 SwiGLU 在 27B 上"完全无法运行"。该推断用错了宽度：kernel 限制作用于**每 rank 分片后**
> 的 intermediate width。TP=2 时分片宽度为 8,704，确实超限并每次回退；TP=4 时为 4,352，
> SwiGLU 正常执行。按未分片宽度做的一刀切排除是不准确的。

Graph smoke 每张卡捕获 5 个 graph；InfiniCore 描述符被标记为已捕获并固定生命周期。
Python 调用计数反映预热、capture 和未捕获路径；graph replay 不再进入 Python，
**不能把这些计数当作全部 replay 算子数**。编译时选中的原生路径也不会必然增加运行时 fallback 计数。

## 8. 正确性

- 54 项 eager 算子数值/非默认流检查通过，覆盖 BF16/FP16/FP32；FP32 GEMM 的三种形状按能力约束回退原生。
- 90 项 BF16 graph 回放数值检查通过，最大绝对误差为 0。覆盖代表性的 27B TP=4 GEMM、Embedding、
  SwiGLU 形状和 1/8/16/256/2048 行；每次回放前更新输入。
- Qwen3-0.6B：eager smoke 12/12、graph smoke 50/50、正式测试 225/225 条输出逐 token 与原生一致，
  且重复稳定。
- Qwen3.8-27B TP=4：**原生与插件各自均不可重复**。正式测试跨引擎一致 54/225，graph smoke 5/50。
  **整模型正确性未通过严格逐 token 验收，第 6 节的 TPS 仅作测量参考。**

以上算子级检查不等于整模型正确性证明。27B 的输出不稳定在无本插件的原生运行中同样存在，
这既不能单独证明插件正确，也不能把全部差异归因于插件。

基线显式只加载 Ascend 插件组；worker 的进程内存映射检查确认未加载 InfiniCore 算子动态库。

## 9. 已知限制

- **27B 无 BF16 单卡配置。** BF16 权重 51.75 GiB，单张 910B4 可用 29.49 GiB，加载即
  `torch.OutOfMemoryError`。量化与 CPU offload 未引入，因为两者都会改变权重或数据路径，
  使原生对比失去意义。
- **27B TP=2 的 `bs=16`/`bs=32` 没有可引用的比值。** 该区间测到 InfiniCore *快于*原生
  （`bs=32` 下 130.6 对 53.2 tok/s，三次全新启动稳定复现）。插桩显示两侧 forward pass 数相同
  （783 对 781），但 ACL graph replay 数差异巨大（745 对 501）：原生留下 280 步在 eager 执行。
  该单元测的是 vllm-ascend 在 KV cache 压力下的批处理行为，不是算子吞吐；调度为何分叉尚未查明。
  已按测量排除：算子核差异（原生的 `torch.ops.vllm.unquantized_gemm` 就是
  `torch.nn.functional.linear`，M=1 到 M=2048 与 InfiniCore 持平）、额外重算、基线不稳、
  descriptor 缓存驱逐、`FRACTAL_NZ` 权重转换。
- **默认 block size 16 在该栈上会让 0.6B 原生 graph 初始化失败**（Ascend attention 不支持），
  配对运行显式请求 128。失败记录保存在 `failed-block16/`，不计入 TPS。
- **重复检测误报。** 7 条输出因合法浮点示例 `0.30000000000000004` 中的连续零被重复检测器标记。
  原始 JSON 未改动；复核 JSON 保存原始错误、上下文和原始文件 SHA256，汇总时验证哈希以防误用过期复核。
  仅屏蔽该已确认数值片段后重新运行检测，其他重复或健康错误仍会失败。复现命令中的
  `--review-numeric-literals` 显式启用这一严格限定的处理。
- 每种输入/输出长度下每个引擎只启动一次，依次测量三个 batch size；三次重复共用该实例，
  未给出跨启动置信区间。

## 10. 环境与复现

```text
vllm: 0.23.0+empty
vllm_ascend: 0.23.0
torch: 2.10.0+cpu
torch_npu: 2.10.0.post4
CANN: 9.1.0
InfiniCore revision: d3551f37538896056e164abf91b120e38c27007b
library SHA256: ab76cfadb6e16c5194d70abb78d22b248ce0cc0636ddae86e641bbd91e52d1a0
```

在容器的 `/workspace/vllm-infinicore` 下运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_INFINICORE_ASCEND_LIBRARY=/workspace/infinicore-build/libvllm_infinicore_ascend.so
python tests/remote/probe_ascend_ops.py --output /tmp/ascend-operators.json
```

**第 6 节的 18 组矩阵由一套一次性 harness 采集（矩阵编排、graph 算子探针、重复输出审计），
该 harness 只存在于当时的工作区，未随仓库保留**；矩阵数字、核对结论与全部原始 JSON 以
`artifacts/ascend-matrix-20260915/` 为准（见第 11 节）。日常工具保留三个：
算子数值探针 `probe_ascend_ops.py`、可用性 smoke `run_ascend_smoke.py`、
吞吐对比 `bench_ascend_throughput.py`（graph 算子计时另见 `bench_ascend_graph.py`）。
复现矩阵需要先按第 6.1 节的配置重建同等 harness。

矩阵脚本把完整 token 序列、文本健康计数、每次 TPS、路由计数、graph capture/replay、异常和启动配置
保存在各子目录 JSON 中；整模型跨引擎一致性另由 `summary.json` 给出，不会因为单引擎
`validation_errors=[]` 就判定整模型逐 token 验收通过。

单卡 0.6B 可用性检查：

```bash
python tests/remote/run_ascend_smoke.py prepare --root artifacts/ascend-smoke
python tests/remote/run_ascend_smoke.py native --root artifacts/ascend-smoke
python tests/remote/run_ascend_smoke.py all --root artifacts/ascend-smoke \
  --ascend-library "$VLLM_INFINICORE_ASCEND_LIBRARY" --allow-native-fallback
python tests/remote/run_ascend_smoke.py autoall --root artifacts/ascend-smoke \
  --ascend-library "$VLLM_INFINICORE_ASCEND_LIBRARY" \
  --allow-native-fallback --auto-discover-plugins
```

模型默认为 `/models/Qwen3-0.6B`，可用 `--model` 覆盖。该 harness 使用 NPU 0 和共享 prompt ID，
检查输出 token/文本，并从 worker 读取路由状态与计数器。**已安装但 InfiniCore 调用数为 0 的路由
即使允许原生回退也判定为失败。**

吞吐对比使用 `tests/remote/bench_ascend_throughput.py`。**必须用三次测量重复，两次不够**：
两个值的中位数就是它们的均值，一次停顿的重复会让结果偏移几十个百分点；早期两次重复的
TP=4 `bs=16` 报出 33.2%，三次重复下为 90.5%。该脚本在启动引擎前会清空所有
`VLLM_INFINICORE_*` 环境变量，因此外部传入的覆盖会被静默抹掉，覆盖必须走 CLI 参数。
`LLM()` 构造失败会遗留 TP worker，每卡占住约 29.5 GiB，使后续运行全部因显存不足而失败；
harness 已在构造失败时回收子进程。

## 11. 原始记录

- 容器：`/workspace/vllm-infinicore/artifacts/ascend-matrix-20260915/`
- 本地：`artifacts/ascend-matrix-20260915/`（`artifacts/` 不纳入版本控制）
- [汇总 JSON](../artifacts/ascend-matrix-20260915/summary.json)
- [最终数据核对](../artifacts/ascend-matrix-20260915/final-audit.json)
- [Graph smoke 汇总](../artifacts/ascend-matrix-20260915/smoke-summary.json)
- [Eager 算子探针](../artifacts/ascend-matrix-20260915/operator-probe.json)
- [Graph 算子探针](../artifacts/ascend-matrix-20260915/operator-graph-probe.json)
- [环境与源码哈希](../artifacts/ascend-matrix-20260915/environment.json)
- [35 项容器回归测试日志](../artifacts/ascend-matrix-20260915/unit-tests-final.log)
