# 测试说明

无需厂商 SDK 或设备的回归测试：

```sh
python -m unittest discover -s tests -v
ruff check vllm_infinicore scripts tests
ruff format --check vllm_infinicore scripts tests
```

| 文件 | 验证范围 |
| --- | --- |
| `test_registration.py` | 关闭插件时保持延迟导入，支持各平台的注册和卸载 |
| `test_build_sources.py` | 版本和补丁校验、独立源码复制、缓存复用与污染拒绝 |
| `test_benchmark_common.py` | 平台环境切换、共享 worker 校验及评测入口的延迟导入 |
| `test_kunlun_benchmark.py` | 静态协议、完整计时、原生/插件条件匹配，以及逐 rank 路由与 Graph 证据 |

`check_*.py` 是手动设备检查，不由上述 unittest 命令收集。Kunlun RoPE 检查需使用已配置 xpytorch、厂商 SDK 和 `INFINI_ROOT` 的解释器，在项目根目录执行：

```sh
python tests/check_kunlun_rope.py --output /tmp/kunlun-rope.json
```

该检查覆盖 BF16/FP16/FP32、I32/I64 位置、NeoX/GPT-J 布局、位置边界、packed QKV stride 和输入变化后的 Graph 重放。输出路径必须尚不存在。

设备实验结果保存在忽略的 `results/`、`artifacts/` 或 `reports/` 下。整理代码时保留已完成实验的源码快照、原始结果和指纹；历史精度与吞吐结论以相应报告记录的快照为准。
