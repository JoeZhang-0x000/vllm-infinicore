# 架构

术语与命名约定见 [README](../README.md#术语与命名约定)。

## 目标与定位

`vllm-infinicore` 是一个 out-of-tree vLLM 插件，面向单节点 Qwen3 推理。**在完成 InfiniCore
算子接入的基础上，针对 vLLM 高频调用中的适配、调度和资源管理开销进行优化。**

算子接入本身已经落地：九条 scoped 路由（清单见 [`QWEN3_OP_SCOPE.md`](QWEN3_OP_SCOPE.md)）
可以按平台把纳入范围的算子交给 InfiniCore 执行，全部默认关闭、显式开启。接入之后的差距
不在算子核——MetaX 上两条路径跑同样的 `mcblas` GEMM 与 `flash_fwd_splitkv_kernel`，
Ascend 上从 M=1 到 M=2048 的 GEMM 算子核与原生比值 0.99x–1.01x——而在逐层、逐 token
重复执行的 host 侧：Python 对象构造与参数编组、重复的平台/能力探测、跨运行时的流交接、
以及编译与图路径上外部调用无法被保留的问题。六个优化方向即围绕这些开销组织。

两条平台线的边界不同：

- **Ascend**：走 C API 适配器（`operators/ascend/backend.py`），由 `infinicore.lock.json` 中的精确上游修订编译。
  `vllm_ascend` 保留平台、设备、worker、通信、attention 与 KV cache 的全部所有权。
- **MetaX**：既可与 `vllm_metax` 共存，也可通过本插件自带的实验性 InfiniCore 平台入口在不加载
  `vllm_metax` 的情况下运行；后者下 attention/KV 也由本插件的 InfiniCore 路由承担。

## 优化方向总览

| 方向 | 要消除的开销 | 现状 |
|---|---|---|
| [Tensor Bridge](#1-tensor-bridge) | Python 对象构造、参数编组、多次跨语言调用 | 已落地两条桥接路径（当前流 C++ bridge、Ascend C API bridge），按路由与平台启用 |
| [缓存与资源复用](#2-缓存与资源复用) | 逐调用的平台探测、能力检测、描述符与工作区重复创建 | 分派信息已前移到初始化期；tensor 包装与 descriptor/workspace 已按执行签名缓存 |
| [Stream 对齐](#3-stream-对齐) | 跨运行时流交接与事件依赖维护 | bridge 路径已在 vLLM 当前流上提交；Python 转发路径保留外部 stream 桥接用于 A/B |
| [编译与设备图适配](#4-编译与设备图适配) | 外部调用无法被编译器/设备图保留，replay 仍走 Python | custom op + fake 已接通两条编译路径；capture/replay 资源生命周期已管理 |
| [能力与性能分派](#5-能力与性能分派) | 无效替换、逐调用能力重判、错过已有融合路径 | 能力判定与性能策略分层，按平台/签名/测量选择实现 |
| [张量并行（TP）](#6-张量并行tp) | 多余运行时、设备归属错误导致的卡死与开销 | 运行时适配已完成（TP≥4 卡死已修复）；扩展效率是后续重点 |

每个方向小节的**约束与边界**是表述红线：对外描述不得越过它们（例如不得写成
"C++ 调用没有开销"或"保证走到最好分支"）。

### 1. Tensor Bridge

**问题。** 简单的 Python 转发虽然能调用算子，但每次调用都要在 Python 里提取 shape、stride、
dtype、设备和数据指针，创建 InfiniCore 包装对象、组织调用参数，多次往返跨语言边界。
Ascend 上实测图外每次 InfiniCore 调用额外付出约 270 µs 的 Python 开销
（[`ASCEND.md`](ASCEND.md) 第 5 节）。

**已落地。** Bridge 把这些工作集中到 C++ 适配层，直接接收 `at::Tensor`，读取底层元数据并转换为
InfiniCore 所需的描述符或视图；布局兼容时复用原有存储，避免不必要的数据复制：

- **当前流 C++ bridge**：`operators/cpp_bridge.py` 按需 JIT 编译 `operators/csrc/infinicore_bridge.cpp`
  （pybind11 + libtorch），C++ 侧经 `strided_from_blob` 以 `data_ptr` 零拷贝建 InfiniCore 视图。
  默认路由 MetaX 为 `PagedAttentionDecodeFlash,MatMul,StoreKVCache`，MUSA 为全部九条
  （`MUSA_DEFAULT_ROUTES`）；`VLLM_INFINICORE_DISABLE_CPP_BRIDGE=1` 关闭做 A/B，
  `VLLM_INFINICORE_CPP_BRIDGE_ROUTES=...` 指定集合。
- **Ascend C API bridge**：`operators/ascend/csrc/bridge.cpp`（C ABI shim）+ `operators/ascend/backend.py`
  （ctypes），把 shape/stride/dtype 编组成 infiniop descriptor，以裸设备指针 launch。
- 未走 bridge 的路由仍由 Python `infinicore` 包转发（`operators/backend.py`），
  作为行为对照与 A/B 基线保留。

**证据。** StoreKV 从 Python/外部 stream 路径切到当前流 C++ bridge，全路由吞吐
`317.33 → 366.14` 输出 TPS（`+15.38%`，该形状下 StoreKV 跑 43,008 次）。反例同样重要：
融合 RMSNorm 每步只跑 57 次，bridge 每次调用建/销毁 descriptor 的成本反而超过省下的
交接成本，实测后不设为默认（见方向 5）。**收益取决于调用频率。**

**约束与边界。** 相对简单 Python 转发，目标是减少 Python 对象构造、参数编组和多次跨语言
调用；不能简单写成"C++ 调用没有开销"，也不能把所有收益都归因于语言变化。C++ bridge 目前
每次调用仍创建并销毁 descriptor，这正是方向 2 的复用缺口。

> 对外表述：构建面向 vLLM 的 Tensor Bridge，以尽可能零拷贝的方式连接 PyTorch Tensor 与
> InfiniCore 算子接口，减少高频调用中的类型包装和元数据转换开销。

### 2. 缓存与资源复用

把稳定信息移出热路径，分两层。

**第一层：分派信息缓存（初始化期前移）。**

- 路由集合在 `register()` 时由环境变量解析一次（`routing/patching.py`）；cpp bridge 的路由集合按
  环境变量键缓存，bridge target 按覆盖值 memoize。修复前 `cpp_bridge._bridge_target()`
  无缓存，每个 decode attention 调用都会 `find_spec("torch_musa")`，一个窗口 5,088 次、
  展开成 152,640 次 `_path_join`；修复后在 host 是瓶颈的 Qwen3-0.6B 上 `+19.8%`
  （DEV_LOG 2026-09-04）。
- 融合 Add+RMSNorm 的能力探测每进程只做一次（只建/销毁 descriptor、不分配 workspace、
  不 launch，见方向 5）。
- 库加载惰性且只做一次：Ascend `.so` 经 `@lru_cache` 的 `library()` 加载并完成修订/ABI
  校验；涉及设备的初始化按 worker/rank 完成（`INFINICORE_DEFAULT_DEVICE_INDEX`，见方向 6）。

**第二层：执行资源缓存（按执行签名复用）。**

- torch↔InfiniCore tensor 包装 LRU（上限 4,096），键为
  `(mode, data_ptr, shape, stride, dtype, device)`；`wrapped._torch_ref` 持有 torch tensor
  保证底层存储有效。
- Ascend descriptor 按线程/设备/dtype/shape/stride 缓存（上限 4,096）；capture 期间不驱逐，
  已入图的 descriptor 固定生命周期，销毁前先同步其所属流。
- workspace 按设备共享一块取 high-water mark 而不是每 descriptor 一块：27B 形状下共享需
  88 MiB，逐 descriptor 分配需 342 MiB，后者足以让引擎起不来（[`ASCEND.md`](ASCEND.md) 第 3 节）。
  capture 之后 workspace 冻结，保证 replay 引用的地址稳定。
- InfiniCore stream 指针按设备缓存。

**约束与边界。** 缓存不能只按算子名字建立：描述符依赖设备、dtype、shape、stride 和算子参数，
动态形状变化时要重新选择或创建对应资源（上述键设计覆盖了这一点）；指针包装必须保证底层
存储有效（`_torch_ref`）。工作区复用同时减少设备内存占用，但必须满足串行执行或其他明确的
并发隔离条件——当前所有 launch 都在当前流上串行提交，满足该条件；若未来引入并行流，
需要重新论证。

> 对外表述：将平台识别和稳定能力检测前移到初始化阶段，并按执行签名复用描述符与临时资源，
> 降低逐层、逐 token 重复执行的 host 管理成本。

### 3. Stream 对齐

**问题。** PyTorch 与 InfiniCore 各自独立管理 stream，**不能假定它们自动使用同一条流**。
简单转发在 InfiniCore 自有流上执行，因此需要在两个运行时的流之间建立前后依赖；对高频
小算子，这些事件与调度工作会累积成可观开销——StoreKV 的 Python/外部 stream 边界曾是
全路由的最大单项瓶颈（DEV_LOG 2026-09-01）。

**已落地。** 适配层取得 vLLM 当前使用的 PyTorch stream，显式传给 InfiniCore 的算子执行接口，
使算子与上下游沿用同一执行顺序；图捕获期间跟随当前捕获流：

- C++ bridge 每次调用经 `current_stream_from_torch` 取 torch 当前流，作为 launch stream
  参数传给 infiniop；Ascend 侧每次取 `torch.npu.current_stream().npu_stream` 传入 launch。
- 不强制使用设备默认流，也不修改 InfiniCore 的全局流状态；优先使用显式的 launch stream 参数。
- descriptor 缓存键不含 stream，同一形状的 capture 与 replay 可复用预热好的 descriptor。
- `record_stream` 已从 launch 路径移除（单项收益最大的修复，prefill 12.15 s → 2.01 s，
  原生 1.80 s，[`ASCEND.md`](ASCEND.md) 第 4 节）：适配器始终在 tensor 自身当前流上 launch，
  该调用不保护任何东西，只让 allocator 推迟 block 复用。**只有重新引入跨流 launch 时才需要
  重新评估它。**
- 旧的外部 stream 桥接（把 InfiniCore 流包成 `ExternalStream` 并在 launch 前后加
  `wait_stream`）仍保留在 Python 转发路径，供 A/B 与未走 bridge 的路由使用。

**约束与边界。** 表述为"两者独立管理 stream，不能假定自动同一条流"，而不是"两个运行时的
stream 必然不同"——是否同一条流取决于运行时状态，插件不做该假设，也不依赖它。

> 对外表述：将 InfiniCore 算子提交到 vLLM 当前执行流，减少跨运行时的流交接和依赖维护开销，
> 同时保持正确的执行顺序。

### 4. 编译与设备图适配

目标：接入后的算子继续享受编译与图执行收益。分两个层次，两者不能混为一谈。

**编译器层面。** 普通 Python/ctypes/native 调用没有自动向 Dynamo 提供完整的算子语义。
问题的根源是**缺少编译器可理解的契约**，不只是"内部用了 C type"。做法是注册 custom op，
声明输入输出、修改和别名契约，并提供 fake 实现，让编译器在不执行真实计算的情况下推导
输出元数据，将外部调用保留为图节点：

- `operators/custom_ops.py` 在 `vllm_infinicore::` 命名空间 define+impl 七个 custom op
  （`CompositeExplicitAutograd`），仅在直接 opt-in 或路由安装时注册。
- `operators/ascend/graph_ops.py` 用 `torch.library.custom_op` 注册五个
  `vllm_infinicore_ascend::` 算子并全部提供 `register_fake` 形状推导，Dynamo 因此能把
  ctypes launch 作为不透明节点放进编译后的程序。能力判定前移到 trace time 并与 eager
  共用同一谓词，不支持的调用在建节点前就选原生。**在 2026-09-09 去掉旧 guard 之前，
  `execute()` 在编译期直接返回原生算子，此前发布的全部 Ascend 图测量都是原生对原生**
  （[`ASCEND.md`](ASCEND.md) 第 3 节）。

**设备图层面。** 保证 launch 可捕获、工作区地址稳定、描述符生命周期覆盖 replay，并避免
捕获期间的不安全分配、同步和释放：descriptor key 去 stream（capture 可复用）、capture 期间
跳过驱逐、入图 descriptor 固定、workspace 共享到 high-water 并在 capture 后冻结
（详见方向 2 与 [`ASCEND.md`](ASCEND.md) 第 3 节）。

**两个层次的分工。** custom op 与 fake 负责接通编译器；已捕获区域的 replay 进一步省去
逐算子的 Python 包装与 host launch——Ascend 图外每次调用约 270 µs 的 Python 开销正是
被捕获步骤省掉、未捕获步骤仍要付的部分。因此图模式下的 Python 计数只反映预热、capture
与未捕获路径（见[图策略](#图策略)）。

> 对外表述：为 InfiniCore 算子提供编译器可识别的算子契约，并管理设备图捕获与回放所需的
> 资源生命周期，使外部算子能够保留在编译及图执行路径中。

### 5. 能力与性能分派

分派由两张表回答不同的问题。

**能力表**：该平台、版本及调用形状是否支持这个实现，数值和输入输出语义是否满足要求。

- Ascend 的 `supports_*` 谓词仅依据 dtype 和 shape 作答，trace time 与 eager 共用，
  两者不可能出现分歧。
- 融合 Add+RMSNorm 的能力探测只建/销毁 descriptor：`NOT_IMPLEMENTED`、
  `DEVICE_TYPE_NOT_SUPPORTED`、`DEVICE_ARCHITECTURE_NOT_SUPPORTED` 判为不支持，
  其他状态一律重新抛出，使真实故障不会被误读成缺能力；结论按进程缓存。探测放在
  custom op 内部而不是 `_should_use_infinicore()` 里——后者在 fake tensor 上 tracing，
  分支在设备调用之前就被烘进图里。
- 能力限制作用于**每 rank 分片后的形状**：SwiGLU 的 8,192 宽度限制下，27B 的
  `intermediate_size=17408` 在 TP=2 分片为 8,704（超限回退）、TP=4 分片为 4,352（支持）。
  按未分片宽度做一刀切排除是错的。
- 上游把某个 op 钉在原生（`vllm_ascend` 的 IR 优先级 `rms_norm=['native']`）不是能力不足；
  路由状态如实记为"所包装入口无调用"。
- Embedding 在 `tp_size > 1` 时保持原生（TP 感知的调用守卫）。

**性能策略表**：在多个正确实现都可用时，针对某类工作负载，哪个实现经过测量更合适。

- bridge 默认路由按平台与调用频率设定：MetaX 三条（StoreKV 43,008 次调用下 bridge 明显
  胜出）、MUSA 九条；RMSNorm 走 bridge 的变体实测否决（每步 57 次，descriptor 成本超过
  省下的交接成本），入口保留为 opt-in。
- 融合 RMSNorm 优先接通 InfiniCore 已有的融合实现：`infiniopAddRMSNorm` 与 vLLM
  `fused_add_rms_norm` 契约完全对应（接入后 Llama-3-8B `+6.9%`）；平台不支持、调用契约
  不匹配或验证发现不适合时，使用经过验证的原生实现。
- 路由级消融数据指导策略（MatMul 禁用 `+3.65%`、PagedAttentionDecode 禁用 `-2.24%`，
  低于 2% 的差值不解读），完整表见 [`QWEN3_OP_SCOPE.md`](QWEN3_OP_SCOPE.md)。
- **两个目标需要区分**：全 InfiniCore 路由（`VLLM_INFINICORE_ROUTES=all`）用于验证接入
  覆盖；追求端到端性能时，可以采用允许原生实现的混合策略（`throughput` 剖面、bridge 路由集）。

**约束与边界。** 不表述为"保证走到最好分支"——准确的说法是**选择经过验证的合适分支**；
分派结论必须有运行时分支计数或 A/B 数据支撑，后端计数非零不能证明一条路由活在热路径上
（见[图策略](#图策略)）。

> 对外表述：按平台能力和实际调用签名选择算子实现，接通已有融合路径，并结合验证与性能
> 反馈制定分派策略，避免无效替换和不必要的算子拆分。

### 6. 张量并行（TP）

**已完成：运行时适配。** 各 rank 的设备归属、运行时初始化、通信接口及图兼容性：

- **设备归属与游离运行时修复**：InfiniCore 的 `ContextImpl` 把默认 `Runtime` 硬编码建在
  设备 0 上，每个 TP worker 都在一张它从不计算的卡上多持有一整套运行时，在 MACA 上让
  TP≥4 无限卡死在驱动 `mxkwCreateQueueBlock ioctl`。`platform.set_device` 现在 eager
  建立设备上下文并导出 `INFINICORE_DEFAULT_DEVICE_INDEX` 指向 worker 自己的卡。
  去掉游离运行时后 TP≥4 从必然卡死变为 16/16 通过（因果已确立；驱动侧记账机制未查明，
  不应断言）。该修复依赖会读该变量的 InfiniCore 构建，正在向上游提。
- **每次调用恢复当前设备**：InfiniCore dispatch 会把加速器当前设备重置为 0，
  `_route_or_fallback` 在 `finally` 里恢复参考 tensor 的设备。
- **通信接口适配**：MUSA communicator 委托 torch.distributed 集合通信并满足 vLLM 图检查；
  `use_custom_allreduce=False`。
- **图兼容与 worker 环境**：路由剖面经 `vllm.envs` 注册进 Ray worker；
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 保证 rank 1 能看到自己的设备。

**待优化：扩展效率。** 现有结果只支持"部分配置接近原生"：MetaX 已测 TP 矩阵为原生的
69%–86%，且每个模型上比值随 TP 单调下降、随模型变大而收窄——固定每步开销被摊薄的形态；
Ascend 27B TP=4 为原生的 95.6%–99.5%（其原生自身不可重复，TPS 仅作参考，见
[`ASCEND.md`](ASCEND.md) 第 6、8 节）。**"单卡已达原生 90% 以上"不宜泛化。**

下一步是分别测量每 rank 的计算时间、host 提交时间、集合通信时间和等待时间，再决定是否
优化通信计算重叠、同步或分片后的算子路径。**目前不能直接把多卡差距归因为通信。**

> 对外表述：已完成多平台张量并行运行时适配。部分配置的吞吐已接近原生，但多卡扩展效率
> 仍有差距，后续将重点分析计算、调度与通信之间的额外开销。

## 模块分层

以下是从入口到实现的静态视图；六个方向描述行为动机，本节描述代码归属。

### 分层原则与目录总览

目录按层划分，**通用实现放在该层的抽象目录下，平台特化各自建子目录**：

```text
vllm_infinicore/
├── __init__.py / plugin.py        # 入口与装配
├── infinicore.lock.json           # 上游 InfiniCore 修订与 ABI 锁定
├── operators/                     # 算子层：对接 vLLM 算子需求，向下调用 InfiniCore
│   ├── backend.py                 #   Python 转发后端：分派/回退、tensor 包装缓存、stream 桥
│   ├── cpp_bridge.py              #   当前流 C++ Tensor Bridge 装载（JIT）
│   ├── custom_ops.py              #   vllm_infinicore:: custom op 注册
│   ├── csrc/infinicore_bridge.cpp #   C++ bridge 源码
│   ├── routes/                    #   通用 vLLM 安装器：rms_norm / silu_and_mul /
│   │                              #   rotary_embedding / embedding / linear / attention
│   └── ascend/                    #   Ascend 特化：C API bridge（backend）、路由包装
│                                  #   （routes）、图算子（graph_ops）、csrc/
├── device/                        # 设备层：替代厂商 vLLM 平台插件的设备抽象
│   ├── platform.py                #   InfiniCorePlatform（设备/图/通信选择）
│   ├── detection.py               #   无 torch 的平台探测与原生回退原因表
│   ├── distributed.py             #   Ray/TP worker 环境传播
│   └── musa/communicator.py       #   MUSA 通信特化
├── routing/                       # 耦合层：与 vLLM 补丁机制的耦合
│   ├── patching.py                #   九条 scoped 路由表 + PatchRegistry
│   ├── runtime_patches.py         #   vLLM 0.20/0.22 版本兼容补丁
│   └── config.py                  #   声明式路由 YAML 的解析与校验
└── common/                        # 通用层：与 vLLM/设备均无关
    └── validation.py              #   基准输出验证（token 计数/文本健康/图证据）
```

设备层的目标是**代替厂商原生 vLLM 平台插件**，提供设备上的抽象（设备与 dtype 选择、图策略、
通信等能力）。该抽象当前只在 MetaX/MUSA 路径上完整工作；Ascend 上平台发现让位给
`vllm_ascend`，本插件只提供算子适配（`operators/ascend/`）而不接管设备运行时——设备层的
 Ascend 化是未完成项，接入新的平台时应优先在 `device/` 下扩展抽象、在子目录放特化。

### 1. vLLM 插件入口

包对外暴露 `vllm_infinicore:register`，在 `pyproject.toml` 中注册于 `vllm.general_plugins`。
vLLM loader 导入该组的入口点并无参调用。因此 `register()` 必须幂等，且可在多个 vLLM 进程中
安全执行。`plugin.py` 只做装配：Ray 环境注册、vLLM 版本兼容补丁、路由注册。

### 2. vLLM 平台入口（设备层）

包同时暴露实验性平台插件 `vllm_infinicore.device.platform:register_platform`，注册于
`vllm.platform_plugins`，返回 `device.platform.InfiniCorePlatform`。

该模块保持入口点发现的轻量：调用 `register_platform()` 不会导入 torch 或 vLLM，
真正的 `InfiniCorePlatform` 类在 vLLM 选中该平台插件之后才惰性构造。
**自动平台发现在 Ascend 上让位给 `vllm_ascend`**，MetaX 兼容性改写也随之跳过，
因此不会注册竞争的 OOT 类或平台运行时。

### 3. 耦合层：路由注册与 vLLM 兼容

`routing/patching.py` 拥有 Qwen3 算子的路由声明（`QWEN3_OPERATOR_ROUTES`）与运行时
路由状态。每条纳入范围的算子都有声明的实现族、图策略、原生回退和验证路径。

默认行为：不打 monkey patch、不导入模型、不导入 torch、不导入 InfiniCore 运行时、不改变
CUDA Graph 行为。这保证 dry import 安全，也不会扰动 vLLM 原生 cudagraph 基线。

路由安装由显式环境变量开启（清单见 [README](../README.md#路由与环境变量)）。
未知路由名会被拒绝；已知但没有安装器的路由记为 `native_fallback`，而不是"已启用的替换"。
注册结果记录 requested、installed、skipped、disabled、native-fallback 以及每条路由的状态项。
`vllm_infinicore.unregister()` 提供幂等的卸载钩子，只卸载本插件拥有的路由。

`routing/runtime_patches.py` 承装对上游 vLLM 的版本兼容 monkey patch（`register_fake` 缺 op
容错、`torch.accelerator` 内存 API 镜像、MetaX OOT MoE 后端、`GPUModelRunner._dummy_run`
源码改写等），在 `register()` 与 `check_and_update_config` 时应用，可用
`VLLM_INFINICORE_DISABLE_VLLM020_COMPAT=1` 关闭。这些补丁只做兼容，不改变路由语义。

`routing/config.py` 读取 `configs/qwen3_infinicore_graph.yaml`（计划中的路由表、原生回退、
验证路径与图策略）并对照代码内路由注册表校验。vLLM 插件注册路径默认仍不加载该配置；
在有被证明安全的 patch 安装器之前，配置加载只是一个显式的校验与工具 API。

### 4. 算子层

`operators/` 的根目录承载与平台无关的后端与 Tensor Bridge，`routes/` 承载通用安装器：

- RMSNorm、SiluAndMul、RoPE 使用 vLLM OOT `CustomOp` 替换类；Embedding 打
  `UnquantizedEmbeddingMethod.embedding`；MatMul 打 `UnquantizedLinearMethod.apply`；
  LMHead 打 `ParallelLMHead` 的 `UnquantizedEmbeddingMethod.apply` 与
  `LogitsProcessor._get_logits`。
- `vllm_infinicore::` 命名空间的 custom op 只在直接 opt-in（`VLLM_INFINICORE_ENABLE_CUSTOM_OPS`）
  或路由安装时注册加载；直接调用 `vllm_infinicore.operators.*` 仍受同一环境变量门控。
- StoreKVCache 与 PagedAttention 使用 attention 后端覆盖，而不是给 `FlashAttentionImpl.forward`
  打 monkey patch：`operators/routes/attention.InfiniCoreFlashAttentionBackend` 在 MetaX 后端表
  刷新之后被注册为 vLLM 的 `FLASH_ATTN` 后端。这把实现保持在 vLLM attention 后端层面。
- `VLLM_PLUGINS` 显式包含 `metax` 时，attention 后端优先使用 MetaX 的 FlashAttention 元数据
  构造器和 KV-cache 布局。不含 `metax` 时跳过导入 `vllm_metax`，改用 vLLM 原生 FlashAttention
  后端类做元数据形状兼容，但由平台插件激活 InfiniCore 的 StoreKV/Prefill/Decode 路由，
  使运行时 attention 调用不会落回 vLLM 原生 FlashAttention 实现。
- 无 MetaX 模式下，后端把 vLLM 原生 attention 元数据归一化为 InfiniCore PA/KV 包装器所需的
  decode/prefill 字段，并处理 vLLM 的 profile/warmup 调用——那些调用省略输出缓冲或使用无效的
  临时 KV cache，此时返回零填充的 profile 输出而不是调用原生 FlashAttention。
- CPU tensor 有意使用 PyTorch 回退，因为本地 InfiniCore 的 CPU `from_torch` 路径可能崩溃。
  严格后端校验只在设备 tensor 上进行。

### 5. Ascend 特化（operators/ascend/）

安装器包装既有 `vllm_ascend` 类的方法而不是注册重名 OOT 类，原生方法保留供
能力回退，卸载时恢复原方法。`backend.py` 走 ctypes C API 与裸设备指针，
`graph_ops.py` 提供带 fake 实现的 `vllm_infinicore_ascend::` custom op（见方向 4）。
实现与限制见 [`ASCEND.md`](ASCEND.md)。

### 6. 验证层（common/）

`common/validation.py` 是纯 Python，模块导入时既不导入 torch 也不导入 vLLM。它提供：
输入与生成输出的精确 token 计数检查、解码文本健康计数、退化重复检测、cudagraph 模式/
后端/capture sizes 与日志/计数器证据的图证据记录、带输出 TPS 的基准结果记录。

### 7. Smoke 与基准 harness

- `scripts/qwen3_128_32_smoke.py`：MetaX 侧，一次性生成 prompt token ID，用子进程隔离跑各图用例。
- `tests/remote/run_qwen_smoke.py`：MetaX 远端 smoke。
- `tests/remote/run_ascend_smoke.py`、`probe_ascend_ops.py`、`bench_ascend_throughput.py`、
  `bench_ascend_graph.py`：Ascend 侧的可用性 smoke、算子数值探针与吞吐对比。
  2026-09-15 性能矩阵使用的一次性 harness（矩阵编排、graph 探针、重复审计）未随仓库保留；
  矩阵结果与 artifacts 记录见 [`ASCEND.md`](ASCEND.md)。
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

1. **Python 后端计数非零不能证明一条路由活在热路径上。** torch.compile 在 trace 时就解析掉
   Python 分支，一条只在少数调用点命中的路由同样会有非零计数。2026-09-04 的 RMSNorm 就是
   这样：路由已安装、计数非零，却缺席了自己 98% 的调用。需要配合运行时分支计数确认。
2. **图模式下的计数只反映预热、capture 和未捕获路径。** 被捕获的非 attention 算子在 replay
   时不再进入 Python，所以不能把这些计数当作全部 replay 算子数。图证据与解码输出验证必须
   与计数一起使用。

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
短窗口对**结构性证据**（设备算子时间、GPU busy 比例、设备事件计数、每 op 的 host self 时间）
是可靠的。不要用 cProfile 的 cumtime 去估一条高频调用路径上的优化收益，要做 A/B。
