# MetaX 算子补丁

本目录保存针对 InfiniOps 锁定提交 `8c2f70a2eebcdeb2f63b2184d1d2d3b087a8db02` 的三个补丁：Embedding、小批次 Fused Add+RMSNorm、小维度 Q/K RMSNorm。固定上游提交、应用顺序、补丁及应用后源码的 SHA256 见 [manifest.json](manifest.json)。

[build_infinicore.py](../../build_infinicore.py) 在独立 InfiniOps 源码副本中应用补丁：`--platform metax` 默认使用 `--metax-patches upstream`；离线使用 `--metax-patches local`；`--metax-patches none` 构建未优化版本。CUDA 默认不应用这些补丁。切换补丁集时使用新的构建目录。

算子及插件桥接的测量结果见 [MetaX 优化记录](../../../docs/metax/optimization-summary.md)。
