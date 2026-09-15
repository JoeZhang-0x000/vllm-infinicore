# 架构

术语与命名约定见 [README](../README.md#术语与命名约定)。

## 目标

`vllm-infinicore` 是一个用于 InfiniCore 算子实验的 out-of-tree vLLM 插件，面向单节点 Qwen3 推理。
它默认全部关闭；显式开启后，为纳入范围的 Qwen3 算子集安装 InfiniCore 路由。

两条平台线的边界不同：

- **Ascend**：走 C API 适配器（`ops/ascend_backend.py`），由 `infinicore.lock.json` 中的精确上游修订编译。
  `vllm_ascend` 保留平台、设备、worker、通信、attention 与 KV cache 的全部所有权。
- **MetaX**：既可与 `vllm_metax` 共存，也可通过本插件自带的实验性 InfiniCore 平台入口在不加载
  `vllm_metax` 的情况下运行；后者下 attention/KV 也由本插件的 InfiniCore 路由承担。

## 分层

### 1. vLLM 插件入口

包对外暴露 `vllm_infinicore:register`，在 `pyproject.toml` 中注册于 `vllm.general_plugins`。
vLLM loader 导入该组的入口点并无参调用。因此 `register()` 必须幂等，且可在多个 vLLM 进程中安全执行。

### 2. vLLM 平台入口

包同时暴露实验性平台插件 `vllm_infinicore.platform:register_platform`，注册于 `vllm.platform_plugins`，
返回 `vllm_infinicore.platform.InfiniCorePlatform`。

该模块保持入口点发现的轻量：调用 `register_platform()` 不会导入 torch 或 vLLM，
真正的 `InfiniCorePlatform` 类在 vLLM 选中该平台插件之后才惰性构造。
**自动平台发现在 Ascend 上让位给 `vllm_ascend`**，MetaX 兼容性改写也随之跳过，
因此不会注册竞争的 OOT 类或平台运行时。

### 3. Python patch 与路由层

`vllm_infinicore.patching` 拥有 Qwen3 算子的路由声明与运行时路由状态。每条纳入范围的算子都有
声明的实现族、图策略、原生回退和验证路径。

默认行为：不打 monkey patch、不导入模型、不导入 torch、不导入 InfiniCore 运行时、不改变 CUDA Graph 行为。
这保证 dry import 安全，也不会扰动 vLLM 原生 cudagraph 基线。

路由安装由显式环境变量开启（清单见 [README](../README.md#路由与环境变量)）。
未知路由名会被拒绝；已知但没有安装器的路由记为 `native_fallback`，而不是"已启用的替换"。
注册结果记录 requested、installed、skipped、disabled、native-fallback 以及每条路由的状态项。
`vllm_infinicore.unregister()` 提供幂等的卸载钩子，只卸载本插件拥有的路由。

### 4. Custom op 层

`vllm_infinicore.ops` 承载具体实现：

- 非 PA 算子使用 PyTorch custom op 包装，底层调用已安装的 `_infinicore` 扩展。
- `vllm_infinicore::rms_norm`、`fused_add_rms_norm`、`silu_and_mul`、`linear`、`lm_head`、
  `embedding`、`rotary_embedding` 只在直接 custom-op opt-in 或路由安装时才注册加载。
- 直接调用 `vllm_infinicore.ops.*` 仍受 `VLLM_INFINICORE_ENABLE_CUSTOM_OPS` 门控；
  vLLM 路由安装器可以强制加载所需的 custom op 包装。
- RMSNorm、SiluAndMul、RoPE 使用 vLLM OOT `CustomOp` 替换类；
  Embedding 打 `UnquantizedEmbeddingMethod.embedding`；MatMul 打 `UnquantizedLinearMethod.apply`；
  LMHead 打 `ParallelLMHead` 的 `UnquantizedEmbeddingMethod.apply`。
- StoreKVCache 与 PagedAttention 使用 attention 后端覆盖，而不是给 `FlashAttentionImpl.forward`
  打 monkey patch：`vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
  在 MetaX 后端表刷新之后被注册为 vLLM 的 `FLASH_ATTN` 后端。这把实现保持在 vLLM attention 后端层面。
- `VLLM_PLUGINS` 显式包含 `metax` 时，attention 后端优先使用 MetaX 的 FlashAttention 元数据构造器和
  KV-cache 布局。不含 `metax` 时跳过导入 `vllm_metax`，改用 vLLM 原生 FlashAttention 后端类做元数据
  形状兼容，但由平台插件激活 InfiniCore 的 StoreKV/Prefill/Decode 路由，使运行时 attention 调用不会
  落回 vLLM 原生 FlashAttention 实现。
- 无 MetaX 模式下，后端把 vLLM 原生 attention 元数据归一化为 InfiniCore PA/KV 包装器所需的
  decode/prefill 字段，并处理 vLLM 的 profile/warmup 调用——那些调用省略输出缓冲或使用无效的临时 KV
  cache，此时返回零填充的 profile 输出而不是调用原生 FlashAttention。
- CPU tensor 有意使用 PyTorch 回退，因为本地 InfiniCore 的 CPU `from_torch` 路径可能崩溃。
  严格后端校验只在设备 tensor 上进行。
- 设备 launch 用 InfiniCore/PyTorch stream 桥接包装。InfiniCore 通过 `infinicore.get_stream()`
  暴露自己的运行时 stream，而 vLLM cudagraph capture 按 PyTorch stream 排序。
  桥接把 InfiniCore stream 包成 `torch.cuda.ExternalStream`，并在每次 `_infinicore` launch 前后加
  `wait_stream` 依赖，使 capture 和 replay 看到正确顺序的 InfiniCore kernel。
- 部分高频路由另有一条**当前流 C++ bridge**（`ops/cpp_bridge.py` + `csrc/`），直接在 vLLM 当前流上
  launch，省掉重复的 stream 交接。默认 bridge 路由为
  `PagedAttentionDecodeFlash,MatMul,StoreKVCache`，可用 `VLLM_INFINICORE_DISABLE_CPP_BRIDGE=1`
  关闭做 A/B，或用 `VLLM_INFINICORE_CPP_BRIDGE_ROUTES=...` 指定集合。
  **bridge 是否更快取决于调用频率**：StoreKV 在基准形状下跑 43,008 次，bridge 明显胜出；
  融合后的 RMSNorm 每步只跑 57 次，per-call 的 descriptor 成本反而超过省下的交接成本，因此不设为默认。

### 5. Ascend C API 适配层

最小 bridge 为调用方**已经选定**的设备创建算子 handle，从不初始化或拥有设备运行时。
调用使用 torch 当前 NPU 流，在暴露指针前把 blocked 权重转为 ND，并在同一条流上 launch。
算子 descriptor 按线程/设备/形状缓存，stream 是 launch 参数而不是 key 的一部分。
已被 capture 的 descriptor 固定生命周期；驱逐其他 descriptor 时先同步其所属流再销毁 workspace。

`ops/ascend_routes.py` 包装既有 Ascend 类的方法，而不是注册重名的 OOT 类。原生方法及其编排保留下来
供能力回退，卸载时恢复原方法。attention/KV cache 路由保持原生。未配置算子库时全部路由保持原生，
且不导入重型框架。配置了但修订/ABI 不匹配时视为安装失败，而不是可用后端。
已知不支持的情形在 launch 之前回退；运行时 launch 错误向上抛出。

torch.compile 期间，支持的路由发出带 fake 实现的不透明 `vllm_infinicore_ascend` custom op。
这些节点在 capture 期间 launch 真实的 InfiniCore kernel，ACL graph replay 执行已录制的 kernel
而不再进入 Python。不支持的情形在建图之前就选原生实现。
`VLLM_INFINICORE_ASCEND_GRAPH=0` 恢复此前的 eager-only 行为。
实现与观测到的限制见 [`ASCEND.md`](ASCEND.md)。

### 6. 配置层

`configs/qwen3_infinicore_graph.yaml` 记录计划中的路由表、原生回退、验证路径与图策略。
`vllm_infinicore.config.load_config()` 用结构化 YAML 解析器读取它，并对照代码内的路由注册表校验。

vLLM 插件注册路径默认仍不加载该配置。在有被证明安全的 patch 安装器之前，
配置加载只是一个显式的校验与工具 API。

### 7. 验证层

`vllm_infinicore.validation` 是纯 Python，模块导入时既不导入 torch 也不导入 vLLM。它提供：

- 输入与生成输出的精确 token 计数检查；
- 解码文本健康计数；
- 退化重复检测；
- cudagraph 模式、后端、capture sizes 与日志/计数器证据的图证据记录；
- 带输出 TPS 的基准结果记录。

### 8. Smoke harness

- `scripts/qwen3_128_32_smoke.py`：MetaX 侧，一次性生成 prompt token ID，用子进程隔离跑各图用例。
- `tests/remote/run_qwen_smoke.py`：MetaX 远端 smoke。
- `tests/remote/run_ascend_smoke.py`、`run_ascend_matrix.py`、`bench_ascend_throughput.py`：
  Ascend 侧的可用性 smoke、性能矩阵与吞吐对比。
- `scripts/decode_gap_profile.py`：把 decode 步延迟拆成设备算子时间与 host 停顿。
  **注意其 `self_device_time_total` 求和必须限制在 `DeviceType.CUDA` 事件上**，
  把 host op 一起求和会重复计数并报出超过 100% 的 GPU busy。

## 图策略

MetaX 上以 vLLM 原生 cudagraph 为基线：使用 PIECEWISE cudagraph，`backend="eager"`，
`enforce_eager=False`。Ascend 上使用 `CompilationMode.VLLM_COMPILE` 配
`CUDAGraphMode.FULL_DECODE_ONLY`，prefill 不在 decode graph 内。

在满足以下全部条件之前，不得声称某条打过 patch 的路径是图安全的：

- 算子输出已对照 vLLM 原生校验；
- 实际输入与输出 token 数已记录；
- 解码输出健康度已检查；
- 日志中 graph capture 完成；
- 该路径避免了图不安全的 InfiniCore 包装器构造。

**图模式下的计数器有两个陷阱，两者都曾导致错误结论：**

1. **Python 后端计数非零不能证明一条路由活在热路径上。** torch.compile 在 trace 时就解析掉 Python
   分支，一条只在少数调用点命中的路由同样会有非零计数。2026-09-04 的 RMSNorm 就是这样：路由已安装、
   计数非零，却缺席了自己 98% 的调用。需要配合运行时分支计数确认。
2. **图模式下的计数只反映预热、capture 和未捕获路径。** 被捕获的非 attention 算子在 replay 时不再进入
   Python，所以不能把这些计数当作全部 replay 算子数。图证据与解码输出验证必须与计数一起使用。

## 基准公平性规则

所有基准工作必须遵守：

- prompt token ID 只生成一次，并在各引擎间复用；
- 对齐采样：`temperature=0.0`、`top_p=1.0`、`top_k=1`、禁用 EOS、`min_tokens=max_tokens`；
- 以**输出 TPS** 为主指标（定义见 README）；
- 先预热再测量，并做重复测量；**用三次重复，不要用两次**——两个值的中位数就是均值，
  一次停顿的重复会让结果偏移几十个百分点；
- 记录 graph capture 数、路由状态与后端计数器作为证据；
- 旧的 TPS 表在按上述规则重测之前只作历史看待。

当要求是"每一个纳入范围且被调用的算子都走 InfiniCore"时，全路由（`VLLM_INFINICORE_ROUTES=all`）
才是交付配置。隔离剖面可以用来定位瓶颈，但不是可接受的交付配置。

判断一项 **host 侧**优化需要带重复的完整基准 harness，短 profiler 窗口分辨不出来；
短窗口对**结构性证据**（设备算子时间、GPU busy 比例、设备事件计数、每 op 的 host self 时间）是可靠的。
不要用 cProfile 的 cumtime 去估一条高频调用路径上的优化收益，要做 A/B。
