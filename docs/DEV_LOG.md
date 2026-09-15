# 开发日志

约定：

- **倒序排列，最新在前。** 每条目标注对应 commit；commit 落库日期与工作日期不一致时单独注明。
- **只保留 2026-09-01 及之后的条目。** 更早的 MetaX 期（2026-05/06）条目已删除：其 artifacts 已随
  `9f8ba43 chore: remove generated artifacts` 清理，表格无法复现；仍然有效的结论已并入
  [`ARCHITECTURE.md`](ARCHITECTURE.md)、[`QWEN3_OP_SCOPE.md`](QWEN3_OP_SCOPE.md) 和
  [`ASCEND.md`](ASCEND.md)。
- **吞吐统一为输出 TPS** = 实际生成 token 总数 / `LLM.generate()` 墙钟耗时。
- 27B checkpoint 统一按目录名 `/models/Qwen3.8-27B` 称呼；其 config 声明
  `Qwen3_5ForConditionalGeneration` / `qwen3_5`。
- **artifact 路径相对于当次运行所在的主机**，且 `artifacts/` 不纳入版本控制。
  本机只保留 `ascend-matrix-20260915`、`tp-matrix-20260907b`、`tp2-graph-r3` / `tp4-graph-r3` 等近期目录；
  MetaX 侧的 `bench-tp1-*` 等只存在于对应的远端主机上。

---

## 2026-09-15 — Ascend 性能矩阵（工作区，尚未提交）

完整结果见 [`ASCEND.md`](ASCEND.md) 第 6–9 节。**未改动任何算子实现**，新增的只是记录真实
graph/路由证据和可复核验证数据的 harness。

- 在 `npu-worker-08` / `zx-vllm-ascend-023` 完成 **18/18** 组原生/插件对比：Qwen3-0.6B TP=1 与
  `/models/Qwen3.8-27B` TP=4，batch 1/8/16，输入/输出 256/256、2048/256、256/2048。
  关闭 chunked prefill 与 prefix caching，开启 FULL_DECODE_ONLY 图。
- 插件 TPS：0.6B 为原生的 24.5%–98.2%，27B 为 95.6%–99.5%；**没有任何一个单元快于原生**。
- 正确性：0.6B eager smoke 12/12、graph smoke 50/50、正式测试 225/225 逐 token 与原生一致。
  27B TP=4 原生与插件各自均不可重复，跨引擎一致仅 54/225（graph smoke 5/50），
  **整模型正确性不接受，其 TPS 仅作参考**。
- 算子级检查：54 项 eager 数值/非默认流检查通过；另有 90 项 BF16 graph replay 检查，
  每次回放前更新输入，最大绝对误差为 0。两者都不等于整模型正确性证明。
- **更正了 2026-09-09 的路由结论**：27B TP=4 的 SwiGLU 是支持的。此前按未分片
  `intermediate_size=17408 > 8192` 做的一刀切排除用错了宽度——kernel 限制作用于每 rank 分片后的
  宽度，TP=4 为 4,352（TP=2 为 8,704，确实超限）。27B 实际执行 Embedding、MatMul、LMHead、SiluAndMul。
- 最终核对覆盖 108 次测量、900 条输出、768,000 个生成 token：两引擎 LLM 配置与 prompt 哈希一致，
  每 rank graph replays = 输出长度−1、model-runner 调用 = 输出长度+1，基线 worker 未加载 InfiniCore
  动态库，收工后所有 NPU 空闲。
- 该栈上默认 block size 16 会让 0.6B 原生 graph 初始化失败，配对运行显式请求 128；
  失败记录留在 `failed-block16/`，不计入 TPS。
- 7 条输出因合法浮点示例 `0.30000000000000004` 触发重复检测误报。原始 JSON 未改动，
  复核 JSON 保存原始错误、上下文与源文件 SHA256 并在汇总时校验；仅屏蔽该已确认片段，
  其他重复/健康错误仍会失败。对应 `--review-numeric-literals`。
- 35 项容器回归测试通过，含原始 artifact 哈希未改动校验、以及拒绝为无关重复/健康错误开脱的用例。

工作区新增/改动：`tests/remote/run_ascend_matrix.py`、`tests/remote/probe_ascend_graph.py`、
`tests/remote/audit_ascend_repetition.py`、`tests/test_ascend_matrix_harness.py`、
`tests/test_ascend_repetition_audit.py`、`tests/remote/bench_ascend_throughput.py`。
原始记录：`artifacts/ascend-matrix-20260915/`（`artifacts/` 不纳入版本控制）。

## 2026-09-09 — InfiniCore 算子进入编译后的 Ascend 图

commit `ce4b9cf` feat / `6cab6c0` fix / `78ed820` docs / `fa9707a` merge（均于 2026-09-15 落库）。
实现细节见 [`ASCEND.md`](ASCEND.md) 第 3–5 节。

- **此前发布的全部 Ascend 图测量都是原生对原生。** `ascend_backend.execute()` 在
  `torch.compiler.is_compiling()` 为真时直接返回原生算子，另一个 guard 拒绝 stream capture 期间的
  任何调用，于是 trace 出来的程序里根本没有 InfiniCore 节点。
- 五项改动去掉这两个 guard：`ops/ascend_graph_ops.py` 用 `vllm_infinicore_ascend::` 命名空间注册
  `torch.library.custom_op`（`vllm_infinicore::` 已被默认关闭的 `ops/custom_ops.py` 占用）；
  能力判定前移到 trace time 并与 eager 共用；`launch()` 对 capture 安全（descriptor key 去 stream、
  capture 期间不驱逐、入图 descriptor 固定生命周期）；workspace 按设备共享到 high-water mark
  （27B 形状 88 MiB，每 descriptor 一块则需 342 MiB，足以让引擎起不来）；移除 `record_stream`。
- **`record_stream` 是单项收益最大的改动。** 它只在 tensor 被用于非自身 stream 时才有意义，
  而本适配器始终在当前 stream 上 launch，所以什么也没保护，却让 allocator 推迟 block 复用直到
  观察到 stream 事件。每次调用新分配 71 MiB 输出的情况下，每次分配都在等设备进度。
  TP=2、4 序列 prefill：12.15 s → 2.01 s（原生 1.80 s），每次 `output_alloc`
  166/595 µs → 36.9/37.7 µs，两 rank 由 3.6 倍不对称变为对称。
- 它一次解释了此前分头追查的一串现象：TP=2 `bs=4` 的 54.2% 异常单元、TP=4 `bs=1` 的间歇停顿、
  rank 间分配耗时不对称、以及"像 allocator 压力但 `expandable_segments:True` 无效"的矛盾。
  隔离 microbenchmark 从不复现，是因为空闲设备内存充足，推迟复用永不阻塞。影响随内存压力放大：
  TP=2（约 15,040 cache token）受创严重，TP=4（约 200,320）几乎不受影响。
- 该轮结果（1,024 输入 / 256 输出，`routes=Embedding,MatMul,LMHead`，三次重复中位数，占原生比）：
  TP=2 `bs=1` 99.3%、`bs=4` 96.8%；TP=4 `bs=1` 98.8%、`bs=4` 95.3%、`bs=16` 90.9%、`bs=32` 89.9%。
  每格三次重复内稳定在约 1%。后续被 2026-09-15 的矩阵覆盖，此处保留作为 profile 的直接产出。
- **TP=2 `bs=16`/`bs=32` 不可引用**：该区间测到 InfiniCore 快于原生（`bs=32` 下 130.6 对 53.2 tok/s，
  三次全新启动稳定复现）。两侧 forward pass 数相同（783 对 781）但 ACL graph replay 差异巨大
  （745 对 501），原生留下 280 步 eager 执行。该格测的是 vllm-ascend 在 KV cache 压力下的批处理，
  不是算子吞吐；调度为何分叉未查明。
- 算子核与原生无法区分：把 50 次 launch 录进 ACL graph 计时 replay，从 M=1 到 M=2048 比值均为
  0.99x–1.01x，因为两者最终都到达同一个 CANN aclnn GEMM。图外每次 InfiniCore 调用额外约 270 µs Python。
- 方法笔记（仍然适用）：
  - 测算子核时间要把 launch 录进图再计时 replay。小 M 下墙钟循环两侧都是 enqueue-bound，测不到算子核。
  - **用三次测量重复，不要用两次**：两个值的中位数就是均值，一次停顿会让结果偏移几十个百分点。
    早期两次重复的 TP=4 `bs=16` 报 33.2%，三次重复下为 90.5%。
  - `bench_ascend_throughput.py` 在启动引擎前清空所有 `VLLM_INFINICORE_*` 变量，外部传入的覆盖会被
    静默抹掉；覆盖必须走 CLI 参数。此坑曾使两次实验作废。
  - `LLM()` 构造失败会遗留 TP worker，每卡占住约 29.5 GiB，使后续运行全部因显存不足而失败。
    harness 已在构造失败时回收子进程。
- `torch_npu.npu_format_cast` 不能在 graph capture 期间运行；`VLLM_ASCEND_ENABLE_NZ` 默认为 1，
  权重被转为 `FRACTAL_NZ` 的模型会 capture 失败而不是变慢。本 checkpoint 权重已是 ND。

## 2026-09-07 — 固定版本的 InfiniCore Ascend 算子适配器

commit `485f73f` feat / `d97a185` merge / `cb455fa` Merge PR #4（2026-09-08 落库）。

- 新增 RMSNorm、SwiGLU、RoPE、Embedding、MatMul、LMHead 六个 eager Ascend C API 适配器。
  `vllm_ascend` 继续拥有平台、设备、worker、通信、attention 和 KV cache；包装既有 Ascend 类方法，
  不做重复 OOT 注册，保留原生实现供能力回退。
- 把官方 InfiniCore `main` 固定在 `d3551f37538896056e164abf91b120e38c27007b` 并打包进 lock 文件。
  构建脚本拉取/校验该精确干净源码；C API bridge 内嵌修订与 ABI 供运行时校验。
  只编译所需上游算子源码 + CANN，不需要 InfiniRT 与 Python InfiniCore。用全新构建树规避 CANN
  增量预处理失败。上游 Embedding destroy 分派缺 Ascend 分支，由插件侧 destructor shim 补上。
- 支持的调用在 torch 当前 NPU 流上提交；blocked 权重转 ND；descriptor 缓存有界并在销毁前同步。
  Fused Add+RMSNorm、FP32 GEMM、不支持的布局/形状与 attention/KV 路径保持原生。
  设备 launch 的意外失败向上抛出，不静默重试。
- 验证：容器内 59 项单元/回归测试通过；910B4 上 54 项数值检查通过（含 3 项显式原生的 FP32 GEMM）；
  Qwen3-0.6B eager `all` 与自动发现 `autoall` 各 12/12 输出与原生逐 token 一致，六条已安装路由
  计数均非零，另有 10,752 次 fused norm 原生回退。**无图/性能主张。**
- 同期在容器默认解释器 `/usr/local/python3.12.13` 内安装了完整 InfiniCore Python 包
  （同一锁定源码，构建 CPU + Ascend910B4，关闭 CCL/ATen/OpenMP）。该安装不改动插件 bridge，
  也不会补齐缺失的上游 Ascend 算子或移除插件的原生回退。
- 同日的 27B TP=2 可用性 smoke 与"原生骨干 + 图外 LMHead"的图吞吐对比**已被
  [`ASCEND.md`](ASCEND.md) 的矩阵取代**：前者的 `all` 路由覆盖判定基于未分片宽度，后者按 2026-09-09
  条目所述本质是原生对原生。两份报告已删除。

## 2026-09-07 — MetaX 张量并行支持与 TP≥4 驱动卡死

commit `b273b0d` fix / `9a16d8e` fix / `e82e610` docs / `78ceb75` Merge PR #3。
迁到 8 卡主机 `ssh.v5000-prod-gw.nhss.zhejianglab.com:31266`（8× MetaX C550，MACA `3.8.0.23`，
Python `3.10.10`，PyTorch `2.10.0+metax3.8.0.7`，vLLM `0.22.0` 与配套 `vllm-metax`），
模型在 `/root/models`。这是本插件第一次多卡工作，此前 TP>1 完全跑不起来。

**让 TP>1 能跑的三项插件修复：**

1. `platform.py` 的 `set_device` 只在 MUSA 下做 eager 的 `torch.zeros(1, device=device)`。
   在 MACA 上 rank>0 因此保留惰性 context，并在 vLLM 的 Triton sampler 里死于
   `Pointer argument (at 0) cannot be accessed from Triton`。`vllm_metax` 是无条件做的，现在本插件也是。
2. InfiniCore 在 dispatch 过程中把加速器当前设备重置为 0。隔离验证：对 `cuda:1` 上的 tensor 做一次
   `rms_norm` 后，`torch.cuda.current_device()` 读出 0。`_route_or_fallback` 现在在 `finally` 里恢复
   参考 tensor 的设备，rank 0 跳过该 guard。
3. benchmark 只在 ray 下通过集合 RPC 汇总 worker 计数器，导致所有 multiproc TP>1 用例都以
   `graph_capture_count=0` 判失败。现在只要 `tensor_parallel_size > 1` 就汇总。

修复 1 和 2 解锁 TP=1 和 TP=2，但不足以解决 TP≥4。

**游离的 card-0 运行时：** InfiniCore 的 `ContextImpl::ContextImpl()` 把默认 `Runtime` 硬编码建在设备
索引 **0** 上，与进程实际使用哪张卡无关。`Runtime` 构造会分配 primary context、一个 stream
（`infinirtStreamCreate`）、一个 infiniop handle 和两个 allocator——于是每个 TP worker 都在一张它
从不计算的卡上多持有一整套运行时。PyTorch 不会这样，它只在你用到的设备上建 context。

单进程、无框架即可复现（进程 pin 在卡 3，做一次 InfiniCore 分配后看 `mx-smi`）：

| 场景（进程 pin 在卡 3） | 持有的卡 |
|---|---|
| 只有 torch，不碰 InfiniCore | 3 |
| 一次 InfiniCore 分配 | **0 和 3** |
| 同上，完全不加载 torch | **0 和 3** |
| 先调 `infinicore.set_device(3)` | **0 和 3**（无效） |
| `CUDA_VISIBLE_DEVICES=3` | 3（此时卡 0 *就是*卡 3） |

`setDevice()` 在切换前调用 `getCurrentRuntime()`，后者仅为回答一次比较就惰性建出默认 card-0 运行时，
这正是提前调 `infinicore.set_device()` 也躲不掉的原因——试过并测过，无效。

在 MACA 上这会让 TP≥4 在引擎完全初始化成功之后、第一次 forward 时无限卡在
`[MXKW][E] queues.c:844 mxkwCreateQueueBlock ioctl create queue block timeout ... Retrying`。
这些行全部没有 worker 前缀，即由驱动进程而非 worker 发出。运行中用 `mx-smi` 数每卡进程：

| 引擎 | TP | 卡 0 上进程数 | 总数 | 结果 |
|---|---:|---:|---:|---|
| vllm-metax | 4 | 2 | 5 | 正常 |
| vllm-infinicore（修复前） | 4 | 4 | 7 | 卡死 |
| vllm-infinicore（修复后） | 4 | 2 | 5 | 正常 |

**驱动为何卡死没有查清，不应断言。** 提出的三种机制各被自己的实验证伪：每卡 context 数上限
（8 个进程在卡 0 上各持一个 context 无事）、每卡 stream 数上限（单进程 4096 个 stream 无事）、
以及多进程并发建队列的竞争（四个进程各在一张卡上建 32 个 stream 确实复现同样报错，但"卡 0 上一个
重负载进程 + 三个游离运行时"这个忠实的合成模型并不失败）。也不是显存：游离运行时显示 0 MiB，
`--gpu-memory-utilization 0.5` 无帮助。驱动侧的记账属于 MACA，我们没有解释它。

已确立的是因果而非机制：**去掉游离运行时后，TP≥4 从必然卡死变成 16/16 全部通过。** 修复就建立在这一点上。

插件侧的做法是 `set_device` 导出 `INFINICORE_DEFAULT_DEVICE_INDEX` 指向该 worker 自己的卡——
这是仍早于第一次 InfiniCore 调用的、最早的 per-worker 钩子。它只对会读这个变量的 InfiniCore 起作用；
该改动属于 InfiniCore 仓库而不是这里，正在向上游提。变量不设置时 InfiniCore 行为与以前完全一致，
所以对原版构建是惰性的。

另记：在严格平台上无法做路由子集二分——没有 `vllm_metax` 提供 `reshape_and_cache_flash` 回退，
任何部分路由集都会独立地以 `NameError` 失败，路由是全有或全无。ray 也不是绕路，装了 2.58.0 试过，
TP=4 仍然卡死。

**TP 矩阵**（`bs=8`，`input_len=1024`，`output_len=256`，BF16，PIECEWISE cudagraph，
关闭 chunked prefill 并设 `max_num_batched_tokens=16384`，1 次预热 3 次测量，输出 TPS。
基线 `VLLM_PLUGINS=metax`；被测 `VLLM_PLUGINS=infinicore,vllm_infinicore` 且每个 InfiniCore
用例都验证 `vllm_metax_loaded=false`。16 个用例全部 `validation_errors=[]` 且 graph capture 数匹配，
`artifacts/tp-matrix-20260907b`）：

| 模型 | TP | vllm-metax | InfiniCore | 比值 |
|---|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-7B | 1 | 478.19 | 410.46 | 85.8% |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 615.71 | 447.85 | 72.7% |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 784.34 | 540.75 | 68.9% |
| Qwen3-32B | 2 | 196.71 | 157.04 | 79.8% |
| Qwen3-32B | 4 | 270.84 | 207.71 | 76.7% |
| Qwen3-32B | 8 | 333.37 | 246.68 | 74.0% |
| Qwen2.5-72B | 4 | 150.48 | 124.28 | 82.6% |
| Qwen2.5-72B | 8 | 200.03 | 160.17 | 80.1% |

metax 基线在每个 TP（含 8）都健康，卡死是 InfiniCore 路径特有的，不是平台问题。InfiniCore 保持在
vllm-metax 的 69%–86%，且在每个模型上比值随 TP 单调下降，说明差距在多卡扩展而不只是单卡算子速度。
比值随模型变大而收窄，这正是固定的每步开销被摊薄的样子。

**运维注意：** 卡死的 TP 运行会泄漏 worker，`VLLM::Worker` 进程在父进程被 SIGKILL 后存活，
每卡无限期占住约 58 GiB。2026-09-04 曾有四个这样的进程占住卡 0–3 两天，饿死了之后所有用例，
报废一整轮矩阵并产生"TP=2 也失败"的假读数。任何无人值守跑 TP 的 harness 必须用
`pkill -9 -f "VLLM::[W]orker"` 回收（方括号转义，否则 pkill 会匹配到自己的 ssh 命令行），
并轮询 `mx-smi` 直到各卡回落到约 858 MiB 再开始下一个用例。

## 2026-09-04 — MetaX decode gap 定位与两项修复

commit `8f81362` perf / `afc401c` perf / `fc58f21` docs / `5e944e1` refactor / `63d1234` fix /
`c43494c` fix / `264d02d` docs / `4b19335` refactor / `be3242b` docs / `64ee6ac` Merge PR #1。
迁到专用单卡主机 `ssh.v5000-prod-gw.nhss.zhejianglab.com:31919`（1× MetaX C550，MACA `3.8.0.23`）。
之前的主机与无关 GPU 任务共享且四张卡被占满，在那里取的任何吞吐测量都不成立。

**基准形状：** TP=1、BF16、`batch_size=8`、PIECEWISE CUDA Graph（capture sizes `[1,2,4,8]`，
`backend="eager"`），1 次预热 3 次测量，确定性采样并禁用 EOS，`min_tokens == max_tokens`，
同模型两引擎共享同一份 prompt token ID。关闭 chunked prefill
（`max_num_batched_tokens = max(16384, batch_size * input_len)`）使每次 prefill 落在单个调度步内、
绝不与 decode 混合。`VLLM_USE_V2_MODEL_RUNNER=0`、`VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA=1`。

两种流量形状：短输入长输出 `in256/out2048`，长输入短输出 `in4096/out256`。

**修复前基线**（`artifacts/bench-tp1-*-20260903`）：

| 形状 | 模型 | vllm-metax | InfiniCore | 比值 |
|---|---|---:|---:|---:|
| `in256/out2048` | Qwen3-0.6B | 1057.98 | 662.21 | 62.59% |
| `in256/out2048` | Qwen3-4B-Instruct-2507 | 614.89 | 458.89 | 74.63% |
| `in256/out2048` | Meta-Llama-3-8B-Instruct | 466.57 | 375.08 | 80.39% |
| `in4096/out256` | Qwen3-0.6B | 742.43 | 477.59 | 64.33% |
| `in4096/out256` | Qwen3-4B-Instruct-2507 | 367.45 | 248.63 | 67.66% |
| `in4096/out256` | Meta-Llama-3-8B-Instruct | 285.47 | 220.52 | 77.25% |

**差距在哪里。** 新增 `scripts/decode_gap_profile.py` 把算子核时间与 host 停顿分开：先在关闭 profiler
的情况下测干净的稳态 decode 步延迟（用两个输出长度作差消掉 prefill 和采样初始化），再对一段 decode
窗口做 profile 取设备算子时间、GPU busy 比例和 host op 分解。注意其 `self_device_time_total` 求和
必须限制在 `DeviceType.CUDA` 事件上，把 host op 一起求和会重复计数并报出超过 100% 的 GPU busy。

Llama-3-8B，`bs=8`，`in=256`，128 个稳态 decode 步，两项修复之前：

| 指标 | vllm-metax | InfiniCore |
|---|---:|---:|
| Decode 步 | 16.83 ms | 20.47 ms |
| 设备算子时间 | 2605.9 ms | 2870.0 ms |
| GPU busy | 87.64% | 75.55% |
| Host CPU 总计 | 2230.9 ms | 2990.1 ms |
| `unified_attention_with_output` host self | 336.1 ms | 800.9 ms |
| `unified_kv_cache_update` host self | 92.1 ms | 343.7 ms |

**核心数学从来不是问题。** 两条路径跑的是同样的 `mcblas` GEMM 和同样的
`flash_fwd_splitkv_kernel`，成本相当（`1904.11` 对 `1893.99` ms，`86.95` 对 `90.08` ms）。
差距是 host 侧饿死加上额外的胶水 kernel。

按层按 decode 步归一化后，host 额外开销是 attention `+90.8` µs、KV 更新 `+49.1` µs，合计 `+139.9` µs。
Qwen3-0.6B 在层数和 hidden size 都不同的情况下算出同样的两个数（误差 `0.1` µs 以内），
说明这是固定的每次调用成本，与模型规模无关。这也解释了原始的比值趋势：Qwen3-0.6B 在 GPU busy
只有 `45.8%` 时已经是 host-bound，同样的绝对开销对它的代价远大于对 GPU busy `87.6%` 的 Llama-3-8B。

**修复一：缓存 C++ bridge target（`8f81362`）。** `cpp_bridge._bridge_target()` 没有缓存，
每次 decode attention 调用都会进到 `_torch_musa_package_dirs()` 并执行
`importlib.util.find_spec("torch_musa")`，在一台 MetaX 机器上探测 Moore Threads 运行时。
cProfile 记录 `5088` 次调用（恰好每层每 decode 步一次），展开成 `152640` 次 `_path_join` 和
`31008` 次 `posix.stat`。现在按 `VLLM_INFINICORE_CPP_BRIDGE_TARGET` 的取值 memoize，
沿用既有的 `_ROUTES_CACHE_KEY` 模式使环境变量覆盖仍然有效；`_torch_musa_package_dirs()` 按进程缓存
探测结果；`reset_bridge_target_cache()` 供测试使用。

同轮 A/B（`in256/out2048`）：Qwen3-0.6B `662.21 → 793.32`（`+19.8%`，62.59% → 75.09%）；
Meta-Llama-3-8B-Instruct `375.08 → 377.75`（`+0.7%`，80.39% → 81.01%）。metax 基线在同一对运行中
变化小于 `0.5%`，所以这个差值是真的。分裂本身才是重点：**减少 host 工作只在 host 是瓶颈的地方有用。**
在 Llama-3-8B 上腾出的 host 时间直接变成了更多阻塞的 `mcStreamSynchronize` 等待
（`1018 → 1380` ms），decode 步根本没动。

> cProfile 把 `4.09` s 窗口中的 `0.733` s 归给了这条调用链，据此会预测出大得多的收益。
> 其中几乎全部是 cProfile 自己在约 40 万个极小栈帧上的每次调用开销。
> **不要用 cProfile 的 cumtime 去估一条高频调用路径上的优化收益，要做 A/B。**

**修复二：把 fused residual-add RMSNorm 路由到 InfiniCore（`afc401c`）。**
`InfiniCoreRMSNorm._should_use_infinicore()` 要求 `residual is None`，其 docstring 还声称 fused-add
路径是有意走 vLLM 原生实现的。运行时分支计数显示，**一个 decoder layer 的 57 个 RMSNorm 调用点里有
56 个带 residual**，于是 RMSNorm 路由虽然已安装、后端计数非零，却缺席了自己 98% 的调用。
**torch.compile 在 trace 时就解析掉那个 Python 分支，所以 `backend_call_counts` 非零并不能证明一条
路由活在热路径上。**

`infiniop` 早已暴露 `infiniopAddRMSNorm`（`y`、`residual_out`、`a`、`b`、`weight`、`epsilon`），
与 vLLM 的 `fused_add_rms_norm` 契约完全对应，只是没人接上去。新增：
`vllm_infinicore/csrc/infinicore_bridge.cpp` 的 `add_rms_norm_current_stream`、
`ops/custom_ops.py` 的 `vllm_infinicore::fused_add_rms_norm`、
`ops/infinicore_backend.py` 的后端分派与 torch 回退，以及 `ops/vllm_rms_norm.py` 的 residual 路径
（以 `pass_weight_add` 为门控，因为 vLLM 只在该标志置位时才在 fused 路径上应用 weight，
并要求 residual 形状与 dtype 匹配）。

数值在任何吞吐运行之前先对 fp32 参考做了校验：bf16 和 fp16 在所测全部形状上都在 `2` ULP 以内，
`residual_out` 逐位精确；fp32 最大 `5.7` ULP / `3.4e-7` 相对误差，属于求和顺序噪声。
修复后分支计数确认 `56/56` residual 调用点都走 InfiniCore，零回退，所有调用点 `pass_weight_add=True`，
后端 `rms_norm` 计数大致翻倍。

同轮 A/B（`in256/out2048`，叠加在修复一之上）：Meta-Llama-3-8B-Instruct
`377.75 → 403.63`（`+6.9%`，81.01% → 86.62%）；Qwen3-0.6B `793.32 → 826.15`（`+4.1%`，75.09% → 81.31%）。

**能力探测与回退（`63d1234`）。** `infiniopAddRMSNorm` 并非在所有注册了 `infiniopRMSNorm` 的后端上都注册：
`rms_norm` 覆盖 ALI CAMBRICON CPU HYGON ILUVATAR KUNLUN METAX MOORE NVIDIA QY **ASCEND**，
而 `add_rms_norm` 少了 ASCEND。无条件路由 residual 路径会在严格模式下把"缺少 kernel"变成"运行失败"，
而这条路径在此之前压根没碰过 InfiniCore。bridge 因此增加了一个只建/销毁 descriptor 的能力探测——
不分配 workspace、不 launch——对 `NOT_IMPLEMENTED`、`DEVICE_TYPE_NOT_SUPPORTED`、
`DEVICE_ARCHITECTURE_NOT_SUPPORTED` 返回 false，其他状态一律重新抛出，使真实故障不会被误读成缺能力。
结论按进程缓存。

探测放在 custom op 内部而不是 `_should_use_infinicore()` 里：后者在 torch.compile 的 fake tensor 上
tracing 运行，其分支在任何设备调用发生之前就被烘进图里；把判定留在不透明 op 内部，才能让一个迟到的
能力结论改变行为而不改变 traced graph。

**计数器拆分（`5e944e1`）。** fused op 起初记在共享的 `rms_norm` 计数器里，恰好掩盖了这轮要查的东西——
这些计数器是"路由是否到达热路径"的唯一证据，而 RMSNorm 路由正是在计数非零的同时缺席了 98% 的调用。
`fused_add_rms_norm` 现在有独立计数器。两个 op 仍在 `RMSNorm` 路由下，消融与
`VLLM_INFINICORE_DISABLED_ROUTES` 语义不变。Qwen3-0.6B 的 decode 窗口现在报
`rms_norm=1026` 对 `fused_add_rms_norm=1008`，此前是混在一起的单个 `2034`。

**两项修复合并结果**（`artifacts/bench-tp1-*-rmsfused-20260904`，`12/12` 有效。每个 InfiniCore 用例都
安装了全部九条路由、零跳过零原生回退，`vllm_metax_loaded=false`，每个后端路由族计数非零，
输出 token 数精确符合请求。**六组模型/形状的输出 token 预览与改动前基线完全一致**）：

| 形状 | 模型 | 基线 | 两项修复后 | 比值变化 | 提升 |
|---|---|---:|---:|---|---:|
| `in256/out2048` | Qwen3-0.6B | 662.2 | 834.6 | 62.6% → 77.4% | `+26.0%` |
| `in256/out2048` | Qwen3-4B-Instruct-2507 | 458.9 | 531.4 | 74.6% → 85.4% | `+15.8%` |
| `in256/out2048` | Meta-Llama-3-8B-Instruct | 375.1 | 405.7 | 80.4% → 86.0% | `+8.2%` |
| `in4096/out256` | Qwen3-0.6B | 477.6 | 523.1 | 64.3% → 66.9% | `+9.5%` |
| `in4096/out256` | Qwen3-4B-Instruct-2507 | 248.6 | 269.7 | 67.7% → 72.7% | `+8.5%` |
| `in4096/out256` | Meta-Llama-3-8B-Instruct | 220.5 | 240.8 | 77.2% → 83.6% | `+9.2%` |

两项修复互补：bridge target 缓存只在 host 是瓶颈处见效（Qwen3-0.6B `+19.8%`，Llama-3-8B 为零），
且对 prefill 重的形状无帮助；RMSNorm 融合在所有形状上都见效，包括 `in4096/out256`。

测量注意：Qwen3-0.6B `in256/out2048` 是唯一噪声大的单元，其 InfiniCore 三次迭代为
`859.8 / 815.9 / 829.4`（stdev `22.51`），metax 分母在不同运行间在 `1016`–`1085` 漂移，
所以隔离 A/B 把同一构建算成 `81.3%` 而矩阵运行算成 `77.4%`。InfiniCore 绝对值稳定在 `826`–`835`。
其他所有单元 stdev 低于 `11`。

**被否决的变体：** 把 fused RMSNorm 改走 C++ bridge 而非 InfiniCore stream 路径，实测后否决——
Llama-3-8B `401.91` 对 `403.63`，Qwen3-0.6B `825.01`（stdev `19.68`）对 `826.15`（stdev `4.79`）。
`add_rms_norm_current_stream` 每次调用都建/销毁 descriptor，而融合后 RMSNorm 每步只跑 57 次，
descriptor 成本超过了它省下的 stream 交接成本。这与 StoreKV 的结论相反——那里 bridge 胜出是因为该路由
在基准形状下跑 `43008` 次。bridge 入口保留为可选项（`--cpp-bridge-routes ...,RMSNorm`），不设为默认路由。

**单 token decode 阈值（`c43494c`）。** `reorder_batch_threshold` 原为 `128`，任何不超过 128 token 的
prefill 都会被归类为 decode 并被严格包装器拒绝。现改为 `1`，与已安装的 vllm-metax 0.22 builder 一致，
也符合正确的非投机语义：一个 decode 步每个请求恰好贡献一个 query token。

这不只是离线基准的限制，它让插件无法用于服务。对 Qwen3-0.6B 跑 `vllm serve`：

| 用例 | 修复前 | 修复后 |
|---|---|---|
| 1 条短 prompt（9 token） | OK | OK |
| 1 条长 prompt（140 token） | OK | OK |
| 8 条并发短 prompt | 8/8 失败，engine core 挂掉 | 8/8 OK |
| 16 条并发短 prompt | 连接被拒 | 16/16 OK |

修复后服务日志中 `attention skipped` 行为零。离线路径不受影响：路由自检仍安装全部九条路由，无跳过无回退。

> 跑测试套件需要设置 `VLLM_PLUGINS`。不设时只跑 80 个测试且 4 个平台/注册用例报错。
> 用后端 wrapper 跑（`./run-infinicore.sh python -m unittest discover -s tests`）才能拿到完整的 90 个。
> 这是既有行为，已通过还原阈值复现同样的 80/4 结果确认。

**测量噪声与该信哪个指标（`be3242b`）。** `scripts/decode_gap_profile.py` 的短 profiler 窗口在
端到端计时上的运行间噪声远大于在设备时间上的。同一构建在 `bs=8`、`in=256`、64 个测量 decode 步下
连续三次：

| 重复 | Decode 步 | 设备算子 | Host CPU | GPU idle |
|---|---:|---:|---:|---:|
| 1 | `9.864` ms | `463.54` ms | `834.93` ms | `800.05` ms |
| 2 | `10.485` ms | `464.35` ms | `846.70` ms | `817.88` ms |
| 3 | `9.090` ms | `464.21` ms | `772.04` ms | `737.19` ms |
| 极差 | `±7%` | `±0.09%` | `±4.8%` | `±5.2%` |

这不是 GPU 频率。算子时间三次运行保持在 `0.09%` 以内就排除了这一点：同样的 kernel 序列在变化的时钟下
不可能都落在 `464` 的半毫秒内。`mx-smi` 报告唯一活跃的限频原因是 `Idle`，功耗、温度和应用限制均未激活，
且这张卡根本不暴露时钟锁定（只有 `--set-persistence-mode` 和 `--set-power-mode`，没有 `nvidia-smi -lgc`
的等价物）。host CPU governor 是 `performance`，锁在 `3000` MHz。

方差完全落在 host CPU 时间和 GPU idle 时间上，且两者同向移动。单个进程内部计时很紧
（重复 3 测得 `0.952 / 0.959 / 0.953` s），而进程间偏移很大，这是进程放置而非频率：容器在一颗共享的
192 核 Xeon 8558 上有 `16` CPU 配额（`1600000/100000`），每次启动落在哪些物理核和 NUMA 节点上会变。
在这个形状下 GPU busy 只有约 `36%`，decode 循环是 host-bound，任何调度差异都直接体现在墙钟上。

实践结论：

- 判断一项 host 侧优化需要带重复的完整基准 harness，而不是这个短窗口。bridge target 缓存在
  Llama-3-8B 上值 `+0.7%`，这个窗口根本分辨不出来。
- 短窗口对**结构性证据**是可靠的——设备算子时间、GPU busy 比例、设备事件计数、每 op 的 host self 时间。
  本条目正是靠这些定位到差距的；C++ descriptor 重构也正是靠算子时间过关的，因为那恰好是这种噪声
  碰不到的指标。
- 如果将来必须用这个脚本做 host 侧测量，`taskset` 可以削掉进程放置这一分量。

**剩余差距。** 两项修复之后最大的一项是设备侧胶水。在 Llama-3-8B 上，InfiniCore 发出 `210312` 个设备
事件而 MetaX 是 `158503`，多出来的几乎全是 `elementwise_kernel`：`376.50` ms / `138496` 次
对 `208.84` ms / `77886` 次。MetaX 还用 `MACA_CatArrayBatchedCopy`（`65.06` ms）融合了 QKV split，
而 InfiniCore 发出独立的 elementwise copy。减少 InfiniCore tensor view 构造中的 cast 和 contiguity
churn 是下一个目标。`unified_kv_cache_update` 的 host 成本也仍是 MetaX 的约 `3.3` 倍，
且这个成本在 torch C++ dispatch 里而不在任何 Python 栈帧里。

## 2026-09-01 — Qwen3-0.6B 在 MACA 3.8 上的 bring-up 与单路由消融

commit `8e46d8a` perf: enable current-stream StoreKV by default（2026-09-02 落库）。

**已验证运行时**（与最初的 MACA 3.5.3 开发目标不同，后者已不再是当前目标）：
MACA `3.8.0.23`、Python `3.10.10`、PyTorch `2.10.0+metax3.8.0.7`、vLLM `0.22.0` 与配套 `vllm-metax`、
FlashAttention `2.6.3+metax3.8.0.7torch2.10`、InfiniCore 上游 `35b46277bd666772c11bb417ad4231c5be492822`。

- 上游 InfiniCore 用 `--metax-gpu=y --use-mc=y --aten=y` 配置并链接已安装的 MetaX FlashAttention 扩展。
  **MACA 3.8 的 dense/varlen FlashAttention 导出多了一个 Mars workspace tensor 和一个尾随 bool，
  而 KV-cache 导出保留旧签名。** 上游检出因此带有一个以 `INFINICORE_METAX_FLASHATTN_38_ABI` 为门控的
  三文件兼容补丁，构建时加 `--cxxflags=-DINFINICORE_METAX_FLASHATTN_38_ABI`。
  更新 `/root/InfiniCore` 时不要丢掉这些远端改动；补丁副本在
  `/mnt/geogpt-doc-new/default/zx/xfer-20260903/infinicore-metax-fa38-abi.patch`。
- vLLM 0.22 必须同时设 `VLLM_USE_V2_MODEL_RUNNER=0`，否则其 V2 预热会发出投机解码形状
  （256 个请求对应 512 个 query token）而被严格 InfiniCore attention 包装器拒绝。
  配 `VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA=1` 可强制 attention 走 InfiniCore 路径而非
  FlashAttention 兼容捷径。
- 严格无 `vllm_metax` 的 PIECEWISE cudagraph 验证通过：`graph_capture_count=116`、
  `validation_errors=[]`、`vllm_metax_loaded=false`，九个路由族计数均非零，
  graph 与 eager 参考产出完全相同的 32 个输出 token ID。这是图安全性与逐 token 正确性证据，不是性能结论。

**单卡多模型 CUDA Graph 对比**（TP=1、BF16、`bs=8`、`input_len=2048`、`output_len=512`、
1 次预热 3 次测量、capture sizes `[1,2,4,8]`、`backend="eager"`、关闭 chunked prefill 并设
`max_num_batched_tokens=16384`。八个引擎结果全部 `valid=True` / `validation_errors=[]`，
每个请求精确生成 512 个输出 token，同模型两引擎输出预览一致）：

| 模型 | vllm-metax TPS | 严格无 MetaX InfiniCore TPS | 比值 | Graph captures |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 1080.83 | 483.74 | 44.76% | 100 / 100 |
| Qwen3-0.6B | 961.42 | 433.03 | 45.04% | 116 / 116 |
| Qwen3-4B-Instruct-2507 | 549.26 | 329.89 | 60.06% | 148 / 148 |
| DeepSeek-R1-Distill-Qwen-7B | 470.24 | 323.40 | 68.77% | 116 / 116 |

差距随模型算力增长而收窄：两个亚十亿模型上约为 MetaX 的 `45%`，7B 上为 `69%`。
这与固定的每步 Python/后端/stream 交接成本对小模型相对影响更大的解释一致，
但在把剩余差距归给某条具体路由之前仍需 profiling（即 2026-09-04 条目所做的工作）。

**DeepSeek-R1-Distill-Qwen-7B 单路由消融**（同一单卡图形状；严格无 MetaX 矩阵的同轮原生基线
`470.83` 输出 TPS，全路由基线 `317.33`）：

| 禁用的路由 | 输出 TPS | 相对 `all` | 相对变化 |
|---|---:|---:|---:|
| 无（`all`） | 317.33 | 0.00 | 0.00% |
| `RMSNorm` | 314.78 | -2.55 | -0.80% |
| `SiluAndMul` | 326.02 | +8.68 | +2.74% |
| `RoPE` | 320.05 | +2.72 | +0.86% |
| `Embedding` | 321.85 | +4.52 | +1.42% |
| `MatMul` | 328.92 | +11.59 | +3.65% |
| `LMHead` | 324.88 | +7.55 | +2.38% |

三条 attention/KV 路由在无 MetaX 栈上无法被禁用成有效的 base-vLLM 回退（底层后端缺
`reshape_and_cache_flash` / `flash_attn_varlen_func`），因此另用 `VLLM_PLUGINS=metax,vllm_infinicore`
的诊断矩阵测量，此时只有被禁用的那条回退到 MetaX 实现（该矩阵自身的全路由基线为 `351.60`）：

| 禁用的路由 | 输出 TPS | 相对诊断 `all` | 相对变化 |
|---|---:|---:|---:|
| 无（`all`） | 351.60 | 0.00 | 0.00% |
| `StoreKVCache` | 395.19 | +43.59 | +12.40% |
| `PagedAttentionPrefill` | 355.41 | +3.81 | +1.08% |
| `PagedAttentionDecode` | 343.72 | -7.88 | -2.24% |

加载 MetaX 平台/元数据上下文本身就把全路由结果从 `317.33` 抬到 `351.60`（`+10.80%`），
**这个上下文效应必须与严格无 MetaX 的路由归因分开看。**

**根因确认与落地改动：** 保留全部九条 InfiniCore 路由、不加载任何 `vllm_metax` 模块，
把 `StoreKVCache` 加入当时默认的 `PagedAttentionDecodeFlash,MatMul` C++ bridge 集合，
得到 `366.14` 输出 TPS——相对严格无 MetaX 全路由 `+48.81` TPS / `+15.38%`，
达到同轮原生基线 `470.83` 的 `77.77%`，`graph_capture_count=116`、`validation_errors=[]`、
bridge 计数 `StoreKVCache=43008`。

此前的 Python StoreKV 实现通过 InfiniCore 外部 stream 调 `infinicore.paged_caching`，
每次 launch 前后都要 join PyTorch stream，而在该基准形状下 StoreKV 跑 `43008` 次。
当前流 bridge 在保留 InfiniCore `infiniopPagedCaching` 算子核的前提下去掉了这次重复的 stream 交接，
所以这个 A/B 指认主要瓶颈是 **StoreKV 的 Python/外部 stream 边界，而不是 paged-caching 数学本身**。
`MatMul`、`SiluAndMul`、`LMHead` 是次要候选；低于 2% 的差值接近运行波动，不应过度解读。

最小生产改动就是把 `StoreKVCache` 纳入既有的当前流 C++ bridge 默认集合，未新增任何替代 StoreKV 实现。
同步源码后的同轮复测：严格无 MetaX InfiniCore `366.44` 对 vllm-metax `469.60`，即原生的 `78.03%`；
默认 bridge 路由为 `PagedAttentionDecodeFlash,MatMul,StoreKVCache`。
