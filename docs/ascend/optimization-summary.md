# NPU 算子与插件优化

日期：2026 年 10 月 9–10 日（Asia/Shanghai）。机器：npu-worker-08；容器：zx-vllm-ascend-023；Ascend 910B4，每卡 32 GB。

Qwen3-8B BF16 的 TP1/2/4 共 **30 个配置全部达到 native 的 90%以上**，最低 **92.62%**。TP1 原来的五个 prefill 超时配置全部恢复。Attention prefill/decode、集合通信和残差 Add+RMSNorm 保持厂商实现，未开启 InfiniCore Attention 路由。

| TP | 配置数 | 相对 native 的 TPS 范围 | 达到 90% |
| --- | --- | --- | --- |
| 1 | 10 | 94.95%–104.67% | 10/10 |
| 2 | 10 | 92.62%–98.78% | 10/10 |
| 4 | 10 | 94.55%–103.84% | 10/10 |

范围仅覆盖本次 Qwen3-8B、两组长度和五档 batch；按用户调整后的范围测试到 TP4。优化前测量与完整热点数据见 [性能差距报告](performance-gap.md)。

## 算子修复

- **PagedCaching**：限制 launch blocks，以 grid-stride 覆盖所有 token/head，修复 8192×8=65536 超过 CANN launch 上限的卡住问题。经 UB 按 tile 搬运 K/V，替代逐元素 GM 访问；布局允许时合并同 token 的 heads，显式同步 MTE2/MTE3 后复用 UB。
- **RoPE**：增加 token/head 并行，Neox 使用向量拆分、广播与合并；BF16、128 维且 heads 连续时合并搬运。sin/cos 表支持连续内维和独立行 stride；其他设备保留原有连续表要求。GPT-J 修正半表搬运长度。
- **SwiGLU**：在原来的八个 hidden tiles 上增加 token 行分组，改善 TP2/4 的并行度。原有尺寸检查保持有效；TP1 intermediate=12288 仍使用 native，TP2/4 的 6144/3072 实际调用 InfiniCore。
- **Q/K RMSNorm**：128 维 BF16 的同 token heads 合并搬运、归约和广播，直接读取 packed QKV stride，workspace 为 0；其他 dtype、权重组合与布局继续使用原 ACLNN 实现。平方根再除法与 native QK 融合路径对齐，避免倒平方根估算后的舍入差异。

六个标准补丁及应用后十个文件的 SHA256 位于 [Ascend 补丁清单](../../scripts/patches/ascend/manifest.json)。[构建脚本](../../scripts/build_ascend.py) 校验锁定提交，复制 src/include 后应用并校验补丁，不修改共享 InfiniCore checkout；默认使用本地补丁，`--ascend-patches none` 可构建原版本。切换补丁集必须使用新的构建目录。

对应优化已适配当前 InfiniOps 接口并提交 [InfiniOps #1000](https://github.com/InfiniTensor/InfiniOps/pull/1000)，通过 230 项算子回归和 87 项 Ascend smoke 测试。上游 SwiGLU 根据独立对照，仅在 padded 输出启用新核；本仓库保留已验证的 legacy 临时补丁独立提交，补丁及清单指纹保持不变。本报告整模型数据对应本仓库 legacy 后端与插件的联合优化，上游新接口的验证及算子延迟见该 PR。

## 插件调度与布局修复

- [Ascend backend](../../vllm_infinicore/operators/platforms/ascend/backend.py) 直接传入 cos/sin cache 视图，省去每层每步复制整个表。
- Q/K RMSNorm 与 RoPE 保留 packed QKV 的 token/head stride，减少归一化前复制；输出独立分配，保持非原地 custom-op 契约。
- 动态库通过可选能力查询选择新路径。旧 ABI-1 库没有查询符号时，继续使用连续 RMSNorm 输入和连续 sin/cos 表，桥接 ABI 与锁定提交保持不变。
- [Graph fake 实现](../../vllm_infinicore/operators/platforms/ascend/graph_ops.py) 的 RMSNorm/RoPE 输出布局与真实连续输出一致，避免转置输入时编译器采用错误 stride。

本轮未修改 vLLM 请求调度策略。profile 的逐步请求数与 token 数匹配，优化集中在算子并行、调用布局和额外设备工作。

## 正式吞吐

输入→输出为 128→128、2048→512；batch 为 1/4/16/32/64。每项预热一次，计时三次，取输出 TPS 中位数。TPS 包含完整 generate 的 prefill，排除初始化、编译、Graph 捕获及结果检查。参数与优化前相同：max_model_len=2816、max_num_seqs=64、max_num_batched_tokens=8192、block_size=128；显存利用率 TP1/2=0.85、TP4=0.70；FULL_DECODE_ONLY Graph 捕获 batch 1/2/4/8/16/32/64。

两种模式的 prompts 与 LLM 参数逐项一致。TP1/2 采用刚完成的 validation-v7 native 基线及随后串行运行的 v8 候选；使用卡组 0、0/1。TP4 采用相邻时段重新串行测量的完整 native/插件对照，使用卡组 0/1/2/3。最终表共 180 个计时样本（90 对）；来源与 SHA256 保存于 validation-final/reused-result-provenance.json。正式测量期间没有其他本轮 NPU 作业或编译；4–7 号卡上的作业未改动。

| TP | 输入→输出 | Batch | native TPS | InfiniCore TPS | 相对 native |
| --- | --- | --- | --- | --- | --- |
| 1 | 128→128 | 1 | 41.19 | 39.45 | 95.79% |
| 1 | 128→128 | 4 | 159.83 | 152.44 | 95.38% |
| 1 | 128→128 | 16 | 560.75 | 532.46 | 94.95% |
| 1 | 128→128 | 32 | 957.07 | 974.62 | 101.83% |
| 1 | 128→128 | 64 | 1477.83 | 1546.84 | 104.67% |
| 1 | 2048→512 | 1 | 39.53 | 39.02 | 98.71% |
| 1 | 2048→512 | 4 | 141.57 | 139.18 | 98.31% |
| 1 | 2048→512 | 16 | 402.24 | 389.71 | 96.88% |
| 1 | 2048→512 | 32 | 394.19 | 384.24 | 97.48% |
| 1 | 2048→512 | 64 | 461.19 | 451.84 | 97.97% |
| 2 | 128→128 | 1 | 62.54 | 58.65 | 93.77% |
| 2 | 128→128 | 4 | 241.35 | 223.54 | 92.62% |
| 2 | 128→128 | 16 | 816.58 | 764.04 | 93.57% |
| 2 | 128→128 | 32 | 1336.92 | 1320.66 | 98.78% |
| 2 | 128→128 | 64 | 2070.49 | 2039.32 | 98.49% |
| 2 | 2048→512 | 1 | 60.85 | 60.05 | 98.67% |
| 2 | 2048→512 | 4 | 215.78 | 210.19 | 97.41% |
| 2 | 2048→512 | 16 | 578.11 | 556.47 | 96.26% |
| 2 | 2048→512 | 32 | 826.97 | 810.94 | 98.06% |
| 2 | 2048→512 | 64 | 1076.70 | 1046.49 | 97.19% |
| 4 | 128→128 | 1 | 62.64 | 65.04 | 103.84% |
| 4 | 128→128 | 4 | 261.28 | 258.30 | 98.86% |
| 4 | 128→128 | 16 | 987.16 | 980.08 | 99.28% |
| 4 | 128→128 | 32 | 1809.39 | 1847.30 | 102.10% |
| 4 | 128→128 | 64 | 3105.77 | 2936.56 | 94.55% |
| 4 | 2048→512 | 1 | 69.28 | 70.93 | 102.38% |
| 4 | 2048→512 | 4 | 262.57 | 270.87 | 103.16% |
| 4 | 2048→512 | 16 | 876.55 | 856.69 | 97.73% |
| 4 | 2048→512 | 32 | 1286.84 | 1233.91 | 95.89% |
| 4 | 2048→512 | 64 | 1752.28 | 1670.79 | 95.35% |

## 精度与执行路径

- 80 项 KV/RoPE/SwiGLU、40 项 RMSNorm、10 项 RMSNorm 输入范围检查通过，覆盖 BF16/FP16/FP32、packed stride、负 slot、尾部长度和 8191/8192-token 边界。
- 随机回归中的 68,823,296 个 BF16 RMSNorm 元素与 native 全部相同，最大差异为 0 ULP；KV 检查的最大误差为 0。
- Graph 捕获后改变输入、权重、位置与 cache 的重放检查通过；转置输入 stride=[128,8192,1] 时，fake/实际 Q/K 输出均为 [1024,128,1]，重放与重新计算差异为 0。
- TP1 的 24 个不受 KV 容量限制的计时样本，完整请求 hash 向量与 native 一致。默认 native/插件 KV 容量为 66,304/63,360 tokens，长请求 batch 32/64 会产生不同调度批次；将两种模式固定到 9,342,812,160 bytes（63,360 tokens）后，最终候选的 **288/288 个请求完整 token 数组及 hash 一致**。同容量诊断计时未替换默认容量吞吐。
- TP2 的 **30/30 个计时样本、702/702 个请求 hash 一致**。
- TP4 默认通信下，native 与插件各自三次输出稳定的配置均为 0/10、0/10；不能仅凭跨模式 hash 判断算子精度。设置 HCCL_DETERMINISTIC=true 与 torch.use_deterministic_algorithms(True)，全部四个 worker 的实际标志均已核实；batch 1/64、两种长度各预热一次、计时三次，**12/12 个成对样本、390/390 个请求 hash 一致**，两种模式各自输出稳定。此数值对照未替换默认模式吞吐。
- 所有有效样本的生成长度、文本健康检查、每 rank Graph 捕获与实际重放通过；七类注册路由与实际调用已核对，InfiniCore Attention 调用和 fallback 均为 0。旧 ABI-1 库的动态输入/权重/cache Graph 兼容检查通过。

早期 v7 的 RMSNorm 随机检查仅有少量 1 ULP 差异，但 TP2 模型输出随后出现回归，因此未作为验收版本。最终 v8 使用平方根再除法，完成算子与模型两级核对。

## 归因与控制实验

v3 使用同一修复库比较“仅算子”和“算子＋插件”，每项预热一次、计时三次，相对初始 native 基线；这一消融阶段未重跑 native。

| TP1 输入→输出 / Batch | 仅算子 | 算子＋插件 |
| --- | --- | --- |
| 128→128 / 1 | 85.90% | 93.64% |
| 128→128 / 32 | 81.45% | 82.04% |
| 128→128 / 64 | 77.09% | 74.20% |
| 2048→512 / 1 | 88.35% | 96.64% |
| 2048→512 / 32 | 85.79% | 90.28% |
| 2048→512 / 64 | 86.24% | 89.92% |

进一步的 v4 成对 profile 显示，TP1 短请求 batch 64 的 31 个 decode steps 中，native QK 融合合计 21.46 ms，插件分开的 RMSNorm＋RoPE 为 60.07 ms；布局复制为 1.46/11.37 ms。由此增加同 token heads 的 RMSNorm 向量路径和直接 stride 支持。上述是 profile 的 kernel 时间，不能直接换算为 TPS 收益。

v8 首轮 TP4 候选与较早 native 基线相比，短请求 batch 1/4/16/32 为 80.97%/83.45%/84.61%/88.29%，触发最差配置四 rank 成对 profile。31 个 decode steps 的 GEMM 合计为 native 181.01 ms / 插件 181.21 ms，native QK 融合为 17.59 ms / 插件 RMSNorm＋RoPE 17.11 ms；prefill 通信为 115.56/229.38 ms。主机时间与通信等待需要结合时间轴解释，不能把 HCCL 等待全部当作通信算法或算子退化。

随后完整近时复测中，native 短请求 batch 1 从 68.80 变为 62.64 TPS，batch 32 从 1918.08 变为 1809.39 TPS，候选自身也有变化；这说明按较早基线作比值存在时段偏差。最终表采用紧邻的完整 native/插件对照，两套原始数据及补充 profile 全部保留，未混用两套 TP4 native。

## 环境与复现材料

环境沿用优化前报告：vLLM 0.23.0、vllm-ascend 0.23.0、PyTorch 2.10.0+cpu、TorchNPU 2.10.0.post4、CANN 9.1.0、驱动 26.0.rc1；InfiniCore 锁定 d3551f37538896056e164abf91b120e38c27007b、桥接 ABI 1。共享 checkout 保持原提交且未修改。

最终动态库 SHA256：`1548005e4b6dd9a50987d0bf4ddc8a1623a805b279427050e19e1e660dcfc629`。

容器中的实验根目录为 `/workspace/work/infinicore-npu-optimization-20261009`：

- `source-final`：最终插件源码；`build-operators-v8/libvllm_infinicore_ascend.so` 与 manifest.json：已验证动态库及构建指纹。
- `validation-final/results`：最终正式表、参数、逐样本结果、路由/Graph 验证及日志。
- `validation-v8`：首轮完整候选及八份补充 rank profiles；profile 原始二进制与 trace 保留在容器。
- `equal-kv-v8`、`diagnostic-v8`：同容量与 TP4 确定性输出对照。
- `scripts`：算子、RMSNorm、Graph 布局检查与实验控制脚本；正式计时脚本位于 validation-final/scripts。

本地副本为仓库下 `results/npu-optimization-20261009/`，受 .gitignore 保护。计时与 profile 分开保存，复现时将 `source-final` 前置到原有 PYTHONPATH，保留 CANN Python 路径，设置 VLLM_INFINICORE_ASCEND_LIBRARY 指向上述动态库，并使用新结果目录。

最终核对：动态库、六个补丁及十个应用后文件的 SHA256 校验通过，共享 InfiniCore checkout 无 tracked 修改；Python 全仓库 Ruff 检查与格式检查通过。所有本轮作业已退出，0–3 号卡健康状态 OK、AICore 0%，无运行中的 NPU 进程。最终设备状态与运行源码指纹保存于 final-device-state.txt、final-source-fingerprints.json。
