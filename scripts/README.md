# 构建与启动脚本

本目录根部只放可直接执行的入口。共享构建工具放在 `infinicore_build/`，平台补丁放在 `patches/<平台>/`，构建缓存及生成的库保存在用户指定的构建目录。

| 入口 | 平台 | 职责 |
| --- | --- | --- |
| [build_infinicore.py](build_infinicore.py) | MetaX、CUDA | 校验模块化 InfiniCore 及子模块，构建 InfiniRT / InfiniOps；MetaX 可选择上游或本地补丁，CUDA 不应用 MetaX 补丁 |
| [build_ascend.py](build_ascend.py) | Ascend | 校验 legacy InfiniCore，构建 C API 桥接与所需算子，可选择本地补丁 |
| [build_kunlun.py](build_kunlun.py) | Kunlun | 校验 legacy InfiniCore 和 spdlog，使用厂商 SDK 构建 C++ API 与所需算子 |
| [fix_kunlun_vendor_cache.py](fix_kunlun_vendor_cache.py) | Kunlun | 校验或修复实验环境中固定版本的厂商 BHLD KV 写入；原生与插件共用修复 |
| [run-vllm-metax.sh](run-vllm-metax.sh) | MetaX | 加载 `metax-1` 的环境并启动原生 vLLM 或聊天客户端 |

各入口都支持 `--help`，可从任意工作目录通过脚本的完整路径执行。完整构建与使用示例见 [项目 README](../README.md)。

`infinicore_build/sources.py` 负责版本锁和源码指纹校验；`patches.py` 负责补丁获取、SHA256 校验、独立源码副本及缓存复用校验；平台模块只负责各平台的源码位置、复制范围和补丁模式。新增平台时按这一边界添加策略，不把平台特例写入共享补丁处理。

补丁清单与应用顺序见 [补丁索引](patches/README.md)。切换补丁集时使用新的构建目录。
