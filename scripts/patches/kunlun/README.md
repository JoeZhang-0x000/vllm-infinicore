# Kunlun legacy 构建补丁

基线固定为 InfiniCore `a81b18fe6d88f835b35e34801300966741ff423f`，该版本包含 Kunlun KV 写入及 BF16 单 token GEMM 修复。模块化 InfiniOps 暂无 Kunlun 后端。

`kunlun-legacy-runtime.patch` 处理构建与运行兼容性：

- 移除源码未使用的 Boost 构建依赖。
- 隔离 C++ API 库内部符号，避免与 xpytorch 的 HydraLog 共用不同版本的 spdlog registry。
- 禁用 CPU 构建时跳过 CPU runtime 初始化。
- 在厂商 XBLAS 头文件之后添加 CUDA 辅助类型头文件，保留 `cublas*` 到 `xblas*` 的厂商映射。

`kunlun-rope-clusters-positions.patch` 优化 RoPE：

- 查询设备的 cluster 数，最多使用 12 个；P800 从固定 8 个增加到 12 个。任务编号交错分配到 cluster，计算公式保持相同。
- 直接读取 I32 / I64 位置，在内核中裁剪到 `[0, table_len-1]`，去掉每层独立的位置 cast / clamp kernel。空表返回尺寸错误。
- 导出能力位 `infinicoreKunlunRoPECapabilities`，bit 0 表示支持上述位置契约。插件仅在检测到该能力时省略预处理；旧库继续使用原有 cast / clamp 路径。

补丁及应用后文件的 SHA256 保存在 [manifest.json](manifest.json)。构建脚本在独立副本中应用补丁，保留输入 checkout；切换补丁集需使用新的构建目录。

在已配置 xpytorch、SDK 与 `INFINI_ROOT` 的 Kunlun 环境，可执行 `python tests/check_kunlun_rope.py --output /tmp/kunlun-rope.json`，检查两类位置整数、三种浮点格式、两种 RoPE 布局、packed stride、边界及动态 Graph 重放。
