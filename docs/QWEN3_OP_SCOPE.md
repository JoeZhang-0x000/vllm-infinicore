# Qwen3 算子范围

本文定义单节点 Qwen3 推理的算子范围。**所有路由默认关闭**；被请求但没有可用安装器的路由使用
vLLM 原生回退。术语与命名约定见 [README](../README.md#术语与命名约定)。

## 九条 scoped 路由

| 算子 | 类别 | 默认 | 安装器 | 原生回退 |
|---|---|---|---|---|
| `RMSNorm` | 归一化 | 关闭 | vLLM OOT 类 → `vllm_infinicore::rms_norm` / `::fused_add_rms_norm` | vLLM RMSNorm `forward_native` |
| `SiluAndMul` | MLP 激活 | 关闭 | vLLM OOT 类 → `vllm_infinicore::silu_and_mul` | vLLM 原生激活 |
| `RoPE` | 位置编码 | 关闭 | vLLM OOT 类 → `vllm_infinicore::rotary_embedding` | vLLM 原生 rotary embedding |
| `Embedding` | token embedding | 关闭 | `UnquantizedEmbeddingMethod.embedding` → `vllm_infinicore::embedding` | vLLM 原生 token embedding |
| `MatMul` | 线性投影 | 关闭 | `UnquantizedLinearMethod.apply` → `vllm_infinicore::linear` | vLLM 原生 linear |
| `LMHead` | 最终投影 | 关闭 | `ParallelLMHead` 的 `UnquantizedEmbeddingMethod.apply` → `vllm_infinicore::lm_head` | vLLM 原生 logits 投影 |
| `StoreKVCache` | KV cache 更新 | 关闭 | attention 后端 `do_kv_cache_update` → `infinicore.paged_caching` / bridge `StoreKVCache` | vLLM 原生 KV cache 写入 |
| `PagedAttentionPrefill` | paged attention | 关闭 | attention 后端 `forward` → `infinicore.mha_varlen` | vLLM 原生 paged attention prefill |
| `PagedAttentionDecode` | paged attention | 关闭 | attention 后端 `forward` → C++ bridge `PagedAttentionDecodeFlash` | vLLM 原生 paged attention decode |

前六条为非 attention 路由，后三条为 attention/KV 路由。`RMSNorm` 路由同时覆盖普通 RMSNorm 和
fused residual-add RMSNorm 两个 op，它们有各自独立的计数器，但消融与
`VLLM_INFINICORE_DISABLED_ROUTES` 的语义按单条路由处理。

## 路由策略

- 保持 opt-in，默认关闭。
- 只有在 `VLLM_INFINICORE_ENABLE_PATCHES=1` 且 `VLLM_INFINICORE_ROUTES=...` 请求时才安装路由。
- `VLLM_INFINICORE_ROUTES=all` 展开为完整九条；也接受逗号分隔子集用于隔离实验。
- `VLLM_INFINICORE_DISABLED_ROUTES=...` 从更大的请求集合中移除若干条。
- `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=1` 请求路由但保持 vLLM 原生执行，用于验证路由状态管线。
- 以 vLLM 原生 cudagraph 的正确性为基线。

### 平台差异

**MetaX。** 与 `vllm_metax` 共存，或用本插件的平台入口在不加载 `vllm_metax` 的情况下运行。
后一种模式下**路由是全有或全无**：底层后端缺 `reshape_and_cache_flash` / `flash_attn_varlen_func`，
任何部分路由集都会独立地以 `NameError` 失败，因此无法在严格平台上做路由子集二分。
attention/KV 路由的消融必须另用 `VLLM_PLUGINS=metax,vllm_infinicore` 的诊断矩阵，
而**加载 MetaX 上下文本身就会改变全路由基线**（2026-09-01 实测 `+10.80%`），这个效应必须与路由归因分开。

**Ascend。** 通过固定版本的 InfiniCore C API 库适配 `vllm_ascend` 既有的算子方法。
未配置库时全部路由保持原生；配置后安装能力支持的路由，attention/KV 路由保持原生。
每次调用中不支持的情形使用原始 Ascend 实现，并单独上报回退计数与原因。
适配器不注册竞争的 OOT 类，也不实现自己的 Ascend 设备/worker/通信运行时。
支持的路由可通过 `ops/ascend_graph_ops.py` 在编译后的 ACL graph 内执行，不支持的形状在 trace 时选原生；
capture 会固定算子 descriptor，replay 不再增加 Python launch 计数。详见 [`ASCEND.md`](ASCEND.md)。

**MUSA。** 当前流 C++ bridge 在 MUSA 上默认覆盖全部九条路由，与 MetaX 上只默认三条不同。

## 覆盖率不等于安装数

**"安装了 N 条路由"不等于"N 条路由都被调用"**，也不等于"它们在热路径上"。实际覆盖取决于模型架构
和每 rank 的分片形状。已踩过的三个坑：

1. **计数非零不代表在热路径上。** torch.compile 在 trace 时解析 Python 分支。2026-09-04 的 RMSNorm
   路由已安装、后端计数非零，却因为要求 `residual is None` 而缺席了自己 98% 的调用
   （一个 decoder layer 的 57 个调用点里有 56 个带 residual）。需要运行时分支计数才能确认。
2. **能力限制作用于分片后的形状。** InfiniCore Ascend SwiGLU kernel 的宽度限制是 8192，
   作用于**每 rank 分片后**的 intermediate width。`/models/Qwen3.8-27B` 的未分片 `intermediate_size`
   是 17,408，但 TP=2 分片后为 8,704（超限，回退），TP=4 分片后为 4,352（支持，正常执行）。
   按未分片宽度做的一刀切排除是错的。
3. **上游可能把某个 op 钉在原生。** `vllm_ascend` 把 IR op 优先级设为
   `IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])`，所包装的 RMSNorm 入口
   因此拿不到任何调用；这不是能力不足。同样地，`/models/Qwen3.8-27B` 的 64 层中有 48 层是
   `linear_attention`，走 gated-delta-rule 路径而不调用标准 rotary op。

各模型的实测覆盖见 [`ASCEND.md`](ASCEND.md) 第 7 节。

## 启用一条路由之前的验收

在一条路由可以默认启用之前，需要：

- 对照 vLLM 原生路径的单元级数值比较；
- 带精确 token 计数的 128 输入 / 32 输出正确性 smoke；
- 输出预览验证且 `validation_errors=[]`；
- 2048 输入 / 512 输出的预热与重复测量；
- 图模式下使用时的 graph capture 证据；
- 结论不依赖历史 TPS 表。

## 吞吐路由策略

图模式下的全算子吞吐运行必须使用完整路由剖面：

```text
VLLM_INFINICORE_ROUTES=all
```

即 `RMSNorm,SiluAndMul,RoPE,Embedding,MatMul,LMHead,StoreKVCache,PagedAttentionPrefill,PagedAttentionDecode`。

隔离数据只用于诊断。当要求是"每一个纳入范围且被调用的算子都走 InfiniCore"时，
它不能作为交付配置。

当前的 attention 路由使用 InfiniCore 可用的 PA/FA 算子：prefill 分派到 `infinicore.mha_varlen`，
decode 分派到插件 C++ bridge 路由 `PagedAttentionDecodeFlash`（使用 InfiniCore 自带的 FlashAttention
适配器，在 vLLM 当前流上执行）。较慢的 Python `infinicore.paged_attention` decode 包装器仅保留用于
A/B（关闭 bridge 即可），旧的外部 stream `mha_kvcache_` bridge 可用
`VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecode` 显式选择。
`LMHead` 走 bridge 仍是 opt-in：`VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecodeFlash,LMHead`。

**已知的路由级瓶颈排序**（DeepSeek-R1-Distill-Qwen-7B，`bs=8`、`input_len=2048`、`output_len=512`、
严格无 MetaX PIECEWISE 图模式，2026-09-01 实测）：主瓶颈是 StoreKVCache 的 Python/外部 stream 边界，
不是 paged-caching 数学本身——保留 InfiniCore StoreKV 路由而把它从 Python/外部 stream 路径切到当前流
C++ `infiniopPagedCaching` bridge，把全路由吞吐从 `317.33` 抬到 `366.14` 输出 TPS（`+15.38%`）。
`MatMul` 是最大的非 attention 次要路由（禁用时 `+3.65%`）；`PagedAttentionDecode` 相对其 MetaX 回退
是净收益（禁用时 `-2.24%`）。低于 2% 的差值接近运行波动，不应过度解读。
完整消融表与后续的 decode gap profile 见 [`DEV_LOG.md`](DEV_LOG.md)。
