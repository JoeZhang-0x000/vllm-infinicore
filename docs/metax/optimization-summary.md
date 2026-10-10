# MetaX 平台性能优化小结

日期：2026 年 10 月 9 日。测试机器：MetaX-2，MetaX C550。

本轮保留厂商原生 Attention，将插件已覆盖的其他算子交给重构版 InfiniCore。通过减少桥接中的张量复制，以及优化 Embedding 和 RMSNorm 两类算子，Qwen3-8B 单卡吞吐达到原生的 **90.58%～94.18%**，TP=2 达到 **98.28%～101.81%**。随后在 4 个模型上验证 TP=4/8，80 组成对配置全部达到原生的 90%以上，最低为 **90.95%**。

“原生”指关闭本插件、保留厂商 vLLM 插件的基线。Attention prefill/decode 和集合通信始终使用厂商实现；FusedMoE、mRoPE、GemmaRMSNorm、RMSNormGated、GatedDeltaNet 等尚未覆盖的路径也使用原生实现。KV Cache 写入属于已覆盖路由，不等同于 Attention 计算。

## 优化前后的吞吐

以下范围均取各配置的三次 TPS 测量中位数；“TPS 提升”逐配置计算，不能用范围端点相除。

| 场景与对照阶段 | 优化前占原生 TPS | 优化后占原生 TPS | 插件 TPS 提升 |
| --- | --- | --- | --- |
| Qwen3-8B 单卡，桥接优化前后 | 65.07%～79.85% | 90.58%～94.18% | 16.41%～43.98% |
| Qwen3-8B TP=2，完整适配流程起点至最终实现 | 87.91%～92.66% | 98.28%～101.81% | 6.34%～11.79% |
| Qwen3-8B TP=2，最后一轮融合 RMSNorm 的 kernel 选择优化 | 84.91%～99.42% | 98.28%～101.81% | 0.23%～16.16% |

单卡对照中，InfiniOps 和 InfiniRT 动态库的 SHA256 均相同，改动集中在插件桥接，因此这一阶段的收益可以归因于桥接优化。TP=2 完整流程的起点仍使用原生 TP Embedding，最终实现已接入 InfiniCore Embedding，这一行反映整个适配流程的效果。

TP=2 中途新增的融合 RMSNorm kernel 在大 prefill 上出现性能退化。最后一轮保持插件生产代码、模型、输入、计时脚本和运行参数不变，仅将新 kernel 限制在小批次，大 prefill 使用 InfiniOps 原有通用 kernel，因此最后一行反映算子执行路径选择的收益。

部分可直接核对的配置如下，TPS 单位为输出 token/s。

| 对照阶段 | 输入→输出长度 | Batch | 优化前 TPS | 优化后 TPS | 提升 | 占原生比例变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 单卡桥接 | 128→128 | 64 | 2106.42 | 2848.65 | 35.24% | 67.86%→91.97% |
| 单卡桥接 | 2048→512 | 64 | 1071.35 | 1249.26 | 16.61% | 79.85%→93.04% |
| TP=2 最后一轮 kernel 选择 | 128→128 | 64 | 4057.16 | 4401.78 | 8.49% | 93.74%→101.81% |
| TP=2 最后一轮 kernel 选择 | 2048→512 | 64 | 1818.56 | 2112.39 | 16.16% | 84.91%→98.76% |

最早启用 InfiniCore Attention 的单卡结果为原生的 9.15%～75.54%；改用原生 Attention 后为 65.07%～79.85%。这是路由策略调整，单独于上述桥接和算子优化收益。

## 算子优化

算子侧新增了两套 MetaX 专用 kernel，并调整了小维度 RMSNorm 的 launch 参数。

| 优化项 | 具体改动 | 适用范围与分类 |
| --- | --- | --- |
| Embedding | 将 embedding 向量的搬运改为线程块内并行、向量化读取和写入 | 连续且满足对齐条件、hidden=4096；属于 kernel 实现优化 |
| 含残差的 Fused Add+RMSNorm | 增加向量化融合 kernel，将残差相加、归约和归一化在同一 kernel 内完成 | hidden=4096、token 数≤128 且满足对齐条件；属于 kernel 实现优化 |
| 融合 RMSNorm 的规模选择 | 大 prefill 使用 InfiniOps 通用融合 kernel，避免小批次专用实现拖慢大输入 | 仍然使用 InfiniOps；属于算子执行路径选择优化 |
| Q/K RMSNorm | 对归一化维度≤128 的情况，将线程块大小设为 128，减少空闲归约线程 | 属于 launch 参数优化，未重写计算 kernel |

同一 GPU 上的微基准示例如下。耗时是 Graph 重放下每次算子调用的设备时间，单位为微秒。

| 算子与形状 | 优化前 | 优化后 | 耗时降低 |
| --- | --- | --- | --- |
| Embedding，64 tokens，hidden=4096 | 215.72 | 4.09 | 98.10% |
| Embedding，8192 tokens，hidden=4096 | 656.32 | 96.29 | 85.33% |
| Fused Add+RMSNorm，1 token，hidden=4096 | 40.38 | 27.20 | 32.63% |
| Q/K RMSNorm，8192 tokens，16 heads，head dim=128 | 741.41 | 369.40 | 50.18% |

融合 RMSNorm 微基准包含用于重置输入的两次 clone，优化前后口径相同；该耗时包含这些复制，不能作为纯 kernel 延迟。Q/K 的线程块调优主要改善大批次和 prefill，小批次耗时变化很小。各微基准收益不能直接相加成模型 TPS 收益。

实现分别提交到 InfiniOps 的 [Embedding PR #997](https://github.com/InfiniTensor/InfiniOps/pull/997)、[融合 RMSNorm PR #998](https://github.com/InfiniTensor/InfiniOps/pull/998) 和 [Q/K RMSNorm PR #999](https://github.com/InfiniTensor/InfiniOps/pull/999)。[构建脚本](../../scripts/build_infinicore.py) 按固定提交及 SHA256 拉取标准补丁，在独立源码副本中应用；[本地补丁清单](../../scripts/patches/metax/manifest.json) 暂时保留离线副本。MetaX 专用路径之外仍使用通用实现。

## 插件桥接优化

桥接优化改变张量的传递、原地更新契约和路由方式，减少额外搬运及运行时开销。

| 优化项 | 改动与作用 |
| --- | --- |
| Q/K RMSNorm 保留布局 | 直接传递 packed QKV 中的三维 Q/K view 和真实 stride，避免将其强制 reshape 成二维时分配和复制数据 |
| 残差 RMSNorm 原地执行 | 注册带 mutation 标记的 custom op，直接更新 input 和 residual，去掉适配层原有的两次 clone |
| RoPE 原地执行 | 直接更新 vLLM 提供的 Q/K view，去掉 Q/K clone；这里优化的是调用接口，RoPE 计算 kernel 沿用 InfiniOps |
| 接入 TP Embedding | 将 vLLM 已转换的本地词表分片索引传给 InfiniOps；mask 和 all-reduce 继续由 vLLM 负责 |
| 编译与运行时适配 | 使用 TensorView 包装 PyTorch 张量，并在当前 PyTorch stream 调用 InfiniOps；缓存 API 与路由检测，TP Embedding 的 API 判断移到安装阶段，避免 forward 中的文件系统查询打断 Dynamo 编译；显式选择 InfiniOps 的实现编号 0，避免图捕获时触发在线调优同步 |

实现位于 [C++ 桥接](../../vllm_infinicore/operators/common/csrc/infiniops_bridge.cpp)、[MetaX 算子路由](../../vllm_infinicore/operators/platforms/metax/routes.py)、[TP Embedding 路由](../../vllm_infinicore/routing/routes/embedding.py) 和 [桥接配置缓存](../../vllm_infinicore/operators/common/cpp_bridge.py)。前三项对应单卡前后对照，16.41%～43.98% 是这些桥接改动的整体收益。TP 路由及运行时适配未分别量化 TPS 收益。

## 多模型 TP 验证

最终实现的 TPS 相对原生比例如下。每个表格单元包含两组输入输出长度与五种 batch，共 10 个配置。

| 模型 | TP=4 | TP=8 | 实际覆盖的路由类别 |
| --- | --- | --- | --- |
| Qwen3-8B | 97.90%～101.87% | 97.09%～100.80% | 7 |
| Mistral-7B-Instruct-v0.3 | 94.96%～97.10% | 91.64%～98.16% | 7 |
| Qwen3-30B-A3B-Instruct-2507 | 91.07%～100.26% | 90.95%～103.20% | 6 |
| Qwen3.8-27B | 95.04%～97.08% | 93.61%～97.17% | 5 |

七类路由为 Embedding、RMSNorm（含残差融合）、MatMul、RoPE、SiluAndMul、LMHead、StoreKVCache。MoE 模型的专家计算仍用原生 FusedMoE，未经过独立 SiluAndMul 路由；27B 的 Norm 变体和 mRoPE 尚未接入。上述结果证明已测配置达标，未覆盖的路径不计入 InfiniCore 覆盖率。

TP=4/8 共 80 组成对配置、480 个计时样本，所有 TP rank 的 Graph 重放及已覆盖路由调用均已核对。这一轮验证使用最终实现，没有针对四个模型分别重复优化前的消融测试。

测试采用 BF16，输入→输出长度为 128→128 和 2048→512，batch 为 1、4、16、32、64。每个配置预热后计时三次，TPS 为总输出 token 数除以完整 generate 耗时，包含 prefill。单卡及 TP=2 的显存利用率参数为 0.85，TP=4/8 为 0.70；开启和关闭插件的成对参数一致。

运行环境为 MACA 3.8.0.23、PyTorch 2.10.0+metax3.8.0.7、vLLM 0.22.0。InfiniCore 基于锁定提交 `8254d3b`，InfiniOps 基于 `8c2f70a` 加本地 MetaX 优化补丁，详见 [版本锁文件](../../vllm_infinicore/infinicore.lock.json)。

原始评测、微基准和 TP=4/8 对照记录保存在 MetaX-2 的 `/root/infinicore-mixed-20261008`、`/root/infinicore-optimize-20261008/final1`、`/root/infinicore-tp-optimize-20261009` 和 `/root/infinicore-tp48-20261009`。
