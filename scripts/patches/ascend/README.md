# Ascend 临时算子补丁

本目录保留针对 InfiniCore legacy 提交 `d3551f37538896056e164abf91b120e38c27007b`、桥接 ABI 1 的六个标准补丁，已独立提交。补丁及应用后源码的 SHA256 见 [manifest.json](manifest.json)；`scripts/build_ascend.py` 在独立源码副本中应用并核验，保持共享 checkout 不变。

对应优化已适配当前算子接口，提交至 [InfiniOps #1000](https://github.com/InfiniTensor/InfiniOps/pull/1000)：packed Q/K RMSNorm、NeoX RoPE、strided KV 写入和 padded 输出的 SwiGLU。现代 InfiniOps 的连续输出 SwiGLU 继续使用现有 ACLNN；legacy 内核保留本次已验证的 token 并行改动。

本目录补丁与锁定版本保持一致，复现本仓库性能结果时继续使用这些本地补丁。整模型优化与测量口径见 [NPU 优化记录](../../../docs/npu-optimization-summary.md)。
