# NPU 优化前性能差距与 profile

日期：2026 年 10 月 9 日。机器：npu-worker-08，容器：zx-vllm-ascend-023，Ascend 910B4（每卡 32 GB）。本轮按用户调整后的范围测试 Qwen3-8B 的 TP1/2/4，保留 native Attention prefill/decode。

优化前实现未满足“每个已测配置均达到 native 的 90%”的目标：TP1 的有效比值为 53.55%–85.09%，另 5 项在 prefill 后同步卡住；TP2 为 33.87%–81.56%；TP4 为 24.29%–93.29%，仅 128→128、batch 1 达到吞吐门槛。共 25 个可计时配置、165 个正式计时样本，1 个配置达到 90%。超时配置记为 N/A，不能计为 0 TPS。

本报告记录初始测量与诊断阶段，当时未修改生产算子或动态库。诊断指出应先修复 KV 写入的 launch 越界，再处理 RoPE 的 token 串行循环及桥接成本。后续实现与复测见 [NPU 优化记录](optimization-summary.md)。

## 测量口径

- BF16；输入→输出长度为 128→128 和 2048→512；batch 为 1、4、16、32、64。
- 每个配置预热一次，再测三次，取输出 TPS 中位数。TPS = 总输出 token 数 / 完整 generate 耗时，包含 prefill；模型初始化、编译、Graph 捕获与结果检查不计入。
- 同一 TP 的 native 与 InfiniCore 使用相同输入 token、模型与引擎参数；关闭 prefix caching，固定 seed=0、temperature=0、ignore_eos=True，强制生成指定长度。
- max_model_len=2816，max_num_seqs=64，max_num_batched_tokens=8192，block_size=128；TP1/2 显存利用率 0.85，TP4 为 0.70。
- 启用 FULL_DECODE_ONLY Graph，捕获 batch 1/2/4/8/16/32/64；每个有效样本逐 rank 检查实际重放次数和路由注册。
- native 关闭 InfiniCore patches，保留 vllm-ascend；InfiniCore 注册 RMSNorm、SiluAndMul、RoPE、Embedding、MatMul、LMHead、StoreKVCache。Attention prefill/decode、集合通信和残差 Add+RMSNorm 保留厂商实现。TP1 的 SwiGLU intermediate=12288 超出当前 kernel 的 8192 上限，也保留 native；TP2/4 的 6144/3072 已实际调用 InfiniCore。
- TP1 使用卡 0，TP2 使用卡 2/3，TP4 使用卡 0/1/2/3。TP2 测量期间，TP1 的部分诊断使用另一组卡；设备未共享，但未做机器级 CPU/PCIe 隔离。TP4 成对测量及后续多卡 profiles 串行运行。4–7 号卡上的其他用户作业未改动。

TP1 native 的 KV 容量为 66,304 tokens，InfiniCore 为 63,360 tokens；长请求的大 batch 超过单卡容量，native 通过调度分批完成，TPS 包含该成本。原生每项均完成，InfiniCore 的五个失败项在首个大 prefill 后卡住。

## 完整吞吐表

下表单位为输出 token/s；比值使用两种模式各自三次 TPS 的中位数计算。

| TP | 输入→输出 | Batch | native TPS | InfiniCore TPS | 相对 native |
| --- | --- | --- | --- | --- | --- |
| 1 | 128→128 | 1 | 41.16 | 35.02 | 85.09% |
| 1 | 128→128 | 4 | 159.53 | 128.41 | 80.49% |
| 1 | 128→128 | 16 | 567.58 | 367.87 | 64.81% |
| 1 | 128→128 | 32 | 999.91 | 535.49 | 53.55% |
| 1 | 128→128 | 64 | 1614.06 | N/A（超时） | N/A |
| 1 | 2048→512 | 1 | 39.44 | 33.29 | 84.41% |
| 1 | 2048→512 | 4 | 141.58 | N/A（超时） | N/A |
| 1 | 2048→512 | 16 | 401.48 | N/A（超时） | N/A |
| 1 | 2048→512 | 32 | 393.91 | N/A（超时） | N/A |
| 1 | 2048→512 | 64 | 460.52 | N/A（超时） | N/A |
| 2 | 128→128 | 1 | 62.04 | 50.60 | 81.56% |
| 2 | 128→128 | 4 | 240.63 | 182.39 | 75.80% |
| 2 | 128→128 | 16 | 821.47 | 478.76 | 58.28% |
| 2 | 128→128 | 32 | 1360.31 | 655.58 | 48.19% |
| 2 | 128→128 | 64 | 2130.35 | 815.86 | 38.30% |
| 2 | 2048→512 | 1 | 60.78 | 48.42 | 79.68% |
| 2 | 2048→512 | 4 | 216.91 | 140.90 | 64.96% |
| 2 | 2048→512 | 16 | 581.01 | 273.72 | 47.11% |
| 2 | 2048→512 | 32 | 834.22 | 328.56 | 39.39% |
| 2 | 2048→512 | 64 | 1081.62 | 366.33 | 33.87% |
| 4 | 128→128 | 1 | 68.83 | 64.21 | 93.29% |
| 4 | 128→128 | 4 | 276.05 | 240.29 | 87.04% |
| 4 | 128→128 | 16 | 1037.50 | 598.98 | 57.73% |
| 4 | 128→128 | 32 | 1879.94 | 792.70 | 42.17% |
| 4 | 128→128 | 64 | 3189.71 | 969.44 | 30.39% |
| 4 | 2048→512 | 1 | 72.63 | 63.32 | 87.18% |
| 4 | 2048→512 | 4 | 277.59 | 181.11 | 65.24% |
| 4 | 2048→512 | 16 | 904.06 | 334.13 | 36.96% |
| 4 | 2048→512 | 32 | 1297.96 | 388.84 | 29.96% |
| 4 | 2048→512 | 64 | 1775.30 | 431.23 | 24.29% |

## 稳定性：KV 写入 blockDim 越界

当前锁定源码的 PagedCaching 以 `num_tokens * num_kv_heads` 设置 `block_dim`，再直接传入 Ascend kernel launch。Qwen3-8B TP1 的 8192-token prefill 有 8 个 KV heads，产生 65536 个逻辑 block。[CANN 9.1 官方文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/language_extension/SIMD-BuiltIn_keyword.md)规定 numBlocks 合法范围为 1–65535。

独立复现使用与模型相同的 BF16、head_dim=128、block_size=128 和真实 cache/source stride，launch 前先同步输入，排除待提交输入的干扰：

| Tokens | KV heads | 逻辑 blocks | 结果 | K/V 最大绝对误差 |
| --- | --- | --- | --- | --- |
| 4096 | 8 | 32768 | 正常 | 0 / 0 |
| 8191 | 8 | 65528 | 正常 | 0 / 0 |
| 8192 | 8 | 65536 | stream synchronize 超过 30 秒，设备 idle | N/A |
| 8192 | 4 | 32768 | 正常 | 0 / 0 |
| 8192 | 2 | 16384 | 正常 | 0 / 0 |

这与 TP1 的五项失败阈值一致：128×64=8192；2048×4=8192，更大 batch 的首个 chunk 也为 8192。TP2/4 的 head 分片避免同一 launch 越界。原始整模型栈停在 acl_graph.py 的 current_stream.synchronize；CANN 日志反复报告待完成任务，NPU AICore 利用率为 0%。关闭 TASK_QUEUE_ENABLE 的诊断仍超时。

整模型消融进一步确认该问题：仅将 StoreKVCache 交回 native、保留其他六条注册路由，TP1 的 128→128、batch 64 完成一次预热和一次计时，所有请求正常生成 128 tokens，Graph 重放 127 次。该运行是故障定位对照，未替换正式基线中的 N/A，也未按单次计时判断性能达标。

## 性能热点

CPU 与 NPU 同时采集 Level1 trace，每个配置记录前 32 个 model steps。TP1 选择有效配置中差距最大的短/长请求；TP2/4 选择短/长请求 batch 64，每个 rank 均采集。TP2/4 profile 专用运行将输出上限缩为 32，输入与引擎配置保持原样，记录最初 32 步的 prefill、mixed、decode 调度信息；这些运行的 generate 时间和 TPS 不作为正式吞吐数据。

多卡 worker 为 daemon，TorchNPU 在其中禁止启动解析子进程；使用官方 analyse API 在独立进程离线解析已收集的原始数据。部分直接 C ABI GEMM 未得到 profiler 的 Step Id，通过设备时间轴与相邻已归属 step 对齐；原始文件保留，汇总记录这些重归属项。

以下为首个 prefill 的设备 kernel 时间，单位 ms，单元格按 **native / InfiniCore** 排列；多卡取各 rank 的中位数，未将各 rank 相加。总时间含通信 kernel 的持续时间，不能等同于请求关键路径。native 的 QK 一列包含融合 RMSNorm+RoPE，InfiniCore 一列仅为 RoPE。

| 配置 | 首步 tokens | kernel 总时间 | QK 路径 / RoPE | KV 写入 | SwiGLU | GEMM |
| --- | --- | --- | --- | --- | --- | --- |
| TP1，128→128，B32 | 4096 | 317.06 / 1790.56 | 7.96 / 1339.97 | 1.26 / 133.81 | 17.71 / 17.76 | 261.93 / 262.05 |
| TP1，2048→512，B1 | 2048 | 178.82 / 917.00 | 3.79 / 670.26 | 0.83 / 66.19 | 8.32 / 8.30 | 138.51 / 139.13 |
| TP2，128→128，B64 | 8192 | 601.03 / 3502.83 | 6.94 / 2679.26 | 2.01 / 133.42 | 17.98 / 93.10 | 254.98 / 254.94 |
| TP2，2048→512，B64 | 8192 | 624.03 / 3518.22 | 7.02 / 2679.19 | 2.00 / 132.91 | 17.93 / 93.10 | 254.61 / 254.99 |
| TP4，128→128，B64 | 8192 | 331.53 / 3166.84 | 3.62 / 2678.61 | 2.01 / 66.82 | 9.37 / 93.11 | 128.08 / 128.32 |
| TP4，2048→512，B64 | 8192 | 341.06 / 3175.84 | 3.62 / 2678.64 | 1.96 / 66.62 | 9.28 / 93.11 | 128.62 / 128.25 |

以下为同一 trace 中纯 decode steps 的累计设备时间；长请求另有 16 个 mixed steps，未混入本表。单位与多卡汇总方式同上。

| 配置 | Decode steps | 设备时间轴跨度 | QK 路径 / RoPE | KV 写入 | GEMM |
| --- | --- | --- | --- | --- | --- |
| TP1，128→128，B32 | 31 | 968.67 / 1377.29 | 20.02 / 342.03 | 7.69 / 38.57 | 704.48 / 738.41 |
| TP1，2048→512，B1 | 31 | 785.78 / 891.31 | 13.94 / 29.21 | 2.78 / 6.46 | 669.29 / 699.15 |
| TP2，128→128，B64 | 31 | 786.25 / 1579.07 | 19.04 / 660.21 | 9.31 / 38.20 | 385.53 / 409.53 |
| TP2，2048→512，B64 | 15 | 571.07 / 944.32 | 9.80 / 319.61 | 4.75 / 19.19 | 183.95 / 183.68 |
| TP4，128→128，B64 | 31 | 764.47 / 1290.53 | 22.56 / 658.32 | 9.04 / 22.14 | 199.33 / 226.26 |
| TP4，2048→512，B64 | 15 | 377.06 / 733.05 | 10.56 / 318.59 | 4.85 / 10.97 | 95.85 / 95.26 |

TP2 与 TP4 的 8192-token 首步中，InfiniCore RoPE 均约 2679 ms；TP4 的 GEMM 已从约 255 ms 降至 128 ms，RoPE 却没有获得对应缩放。短请求的 31 个 decode steps 中，TP2/4 的 InfiniCore RoPE 也分别约 660/658 ms。当前源码每个 head 对 tokens 串行执行 copyIn/compute/copyOut，缺少 token 维度并行，与该现象一致。

长请求 batch 64 的两种模式均记录 1 个 prefill、16 个 mixed、15 个 decode steps，逐步调度 tokens 和请求数匹配。TP2 的 16 个 mixed steps 设备跨度为 native 9.64 s / InfiniCore 53.25 s，TP4 为 5.24 s / 47.97 s；其中 InfiniCore RoPE 均累计约 40.35 s。SwiGLU 的首步成本在 TP2 为约 17.98/93.10 ms，TP4 为 9.37/93.11 ms，也未随 head/hidden 分片有效缩放。

TP2/4 profile 与 TP4 确定性运行中，逐 rank 读取第一层 QKV、O、gate/up、down 及 LMHead 权重的实际 NPU format 均为 2（ND），shape/stride 已记录。不存在 native 使用 NZ 而 InfiniCore 使用 ND 的权重格式差异证据；主要 prefill GEMM 的设备时间也接近。

TP1、128→128、batch 32 的 31 个 decode steps 中，InfiniCore RoPE 累计 342.03 ms，native 融合 QK RMSNorm+RoPE 为 20.02 ms；KV 写入为 38.57 ms / 7.69 ms。InfiniCore 还出现单独 RMSNorm 30.42 ms 和 layout/copy/cast 44.76 ms。native 的融合路径同时完成更多工作，因此该对照体现整体路径成本，不能解释为纯 RoPE kernel 的等工作量比较。

CPU 上 model_step_1 的大段 self duration 对应等待先前 prefill 的设备工作，不能直接作为可消除的 Python 开销。以设备 kernel、逐 step 时间轴及串行对照为归因依据；各 rank 的时间不能相加为请求延迟，kernel 时间差也不能直接当作模型 TPS 收益。

## 输出与执行路径核对

所有可计时配置的输入/输出 token 数、非空文本、替换字符与空字符检查通过；每个 TP rank 的 Graph 捕获与有效样本重放均通过。TP1 发生重启后，最终 worker snapshot 属于最后一次仅检查失败项的进程，重放次数为 0；有效样本的实际重放证据保存在各自 workers_after 和 graph_replays 字段中。

TP1 的 15 个有效计时样本、TP2 的 30 个样本，逐请求输出 hash 均与 native 对应样本一致。TP4 的 native 与 InfiniCore 在各自三次运行之间都有 hash 变化，10 个配置均不能逐 token 复现；不能仅凭两者 hash 不同判定 InfiniCore 的数值错误。当前默认模式下尚未证明 TP4 的逐 token 一致性。

补充 TP4 确定性诊断同时设置 HCCL_DETERMINISTIC=true 与 torch.use_deterministic_algorithms(True)，串行复测 batch 1 的 128→128 和 2048→512，每项预热一次、计时三次。两种模式的输出各自稳定，且六个对应样本的 token hash 全部一致，所有 rank 的 Graph 重放通过。此结论仅覆盖这两个配置；确定性设置的计时结果单独保存，不替换默认模式的正式吞吐表。

## 后续优化顺序

1. 修复 PagedCaching 的 blockDim 越界，使用有限核数配合 grid-stride/tile 循环覆盖所有 token/head；同时将 head_dim 上的逐元素 GM GetValue/SetValue 改为 UB 与向量搬运。该实现同时存在稳定性问题和明显设备耗时。
2. 改写 RoPE tiling：当前每个 head 对所有 tokens 串行循环，launch 仅为 batch×heads；TP 分片使 heads 更少，token 并行度仍未增加。需要按 token/head 切 tile 并复用 sin/cos 数据。
3. 依据多卡 SwiGLU profile 调整 token 并行、tile 和向量搬运，保持覆盖 TP2/4 已接入的路径。
4. 减少桥接中反复拆分并 contiguous 复制整个 cos/sin cache 的成本，处理 packed QKV stride、Q/K 中间张量和原地 RoPE 契约；后续考虑融合 QK RMSNorm+RoPE，避免 native 融合路径被拆为多个算子。
5. 先完成稳定性与数值核对，再以相同卡组、完全串行的 native/InfiniCore 三次中位数重新验证全部配置。当前 profile 不足以预测改动后的具体 TPS。

## 环境与复现材料

插件基线为 main `95ff1ae9561f067d3688586d8f7844afd8f6d6e3`（PR #7 合并后）。InfiniCore Ascend 使用锁定 legacy ABI 1 提交 `d3551f37538896056e164abf91b120e38c27007b`，源码 checkout 无本地改动。

| 组件 | 版本 / 提交 |
| --- | --- |
| vLLM | 0.23.0+empty，0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 |
| vllm-ascend | 0.23.0，5cb98caaadeff42b5b62b996e34bb2aaa29d20fd |
| PyTorch / TorchNPU | 2.10.0+cpu / 2.10.0.post4 |
| Transformers | 5.5.4 |
| CANN / 驱动 | 9.1.0 / 26.0.rc1 |
| 模型 | /workspace/work/models/Qwen3-8B，36 层，hidden 4096，intermediate 12288，32 Q heads / 8 KV heads |
| Ascend 动态库 SHA256 | 38abca514aa143965e9890bc322f836b80456cd666fecc6082f6cfa6358871c8 |

原始材料保存在容器 `/workspace/work/infinicore-npu-gap-20261009`（host bind 目录 `/root/zx/containers/zx-vllm-ascend-023/workspace/infinicore-npu-gap-20261009`），包含全部 28 份 CPU/NPU trace_view.json 和 profiler 原始二进制数据。本地副本在仓库的 `results/npu-gap-20261009/`，该目录受 .gitignore 保护；已同步全部 kernel_details.csv、逐步调度信息、benchmark 参数与诊断日志，以及 TP1 的 trace_view.json。报告里的正式吞吐取自 `results/comparison-verified.json`，热点取自 `results/profile-summary.json`，补充验证结果为 `results/diagnostic-validation.json`。

复现脚本位于上述材料的 scripts 目录：bench_npu_gap.py、run_matrix.py、run_profiles.py、offline_profiles.py、analyze_results.py、analyze_profiles.py；probe_kv_grid.py 可独立复现 KV 边界。使用新输出目录重跑，保留 source 快照、动态库和设备参数，勿用 profile 的缩短输出 TPS 替代正式结果。所有启动必须将 source 目录前置到原有 PYTHONPATH，并保留 CANN 的 Python 路径。

补充 micro-tp1.json 的 MLP 使用 intermediate=14336，与本轮模型的 12288 不同；该结果未用于报告归因，限制记录在 micro-tp1-notes.json。报告中的 GEMM 比较全部取自真实整模型 trace。

最终核对：30 个成对配置的吞吐表重算通过；12 组 profile、28 个 rank 各有 32 个 steps，所有 trace 均解析完成，多卡 native/InfiniCore 的逐步调度匹配，无未归属 kernel。有效样本逐 rank Graph 重放通过，TP1 native-KV 消融的 64 个请求输出 hash 与 native 一致，TP4 确定性诊断六个样本一致。实验脚本均已退出，0–3 号卡健康状态 OK、AICore 0%，无运行中的 NPU 进程；最终设备状态保存为 final-device-state.txt。生产代码与锁定动态库均未修改。
