# 文档索引与目录约定

| 平台 | 文档 |
| --- | --- |
| Ascend | [性能差距报告](ascend/performance-gap.md)、[优化与复测记录](ascend/optimization-summary.md) |
| MetaX | [算子与桥接优化记录](metax/optimization-summary.md) |

## 代码边界

```text
vllm_infinicore/
├── plugin.py                   # vLLM 插件注册入口
├── infinicore.lock.json         # 构建和运行时共同使用的版本锁
├── operators/
│   ├── custom_ops.py            # 对外 PyTorch custom op 接口
│   ├── attention_ops.py         # Attention/KV 算子调用
│   ├── selection.py             # 显式平台选择
│   ├── common/                  # 共享执行、桥接、legacy API 和原生回退
│   │   └── csrc/                # 模块化与 legacy C++ 桥接
│   └── platforms/
│       ├── ascend/              # C API、图算子、专用路由和 csrc/
│       ├── cuda/
│       ├── kunlun/
│       └── metax/
├── routing/
│   ├── patching.py              # 路由注册、状态、安装与卸载
│   ├── policy.py                # 推荐路由与分派策略
│   └── routes/                 # 按算子组织的共享 vLLM 路由补丁
└── benchmarks/                 # GSM8K 评测入口与评分
scripts/
├── build_infinicore.py          # MetaX / CUDA 模块化构建入口
├── build_ascend.py              # Ascend 构建入口
├── run-vllm-metax.sh            # MetaX 启动入口
├── infinicore_build/            # 共享构建工具与独立平台策略
└── patches/
    ├── ascend/                 # manifest.json、补丁和说明
    └── metax/                  # manifest.json、补丁和说明
docs/
├── ascend/
└── metax/
```

平台目录负责厂商能力与差异，共享的 vLLM 类替换和安装机制放在 `routing/routes/`，张量执行与桥接放在 `operators/`。路由注册和平台能力声明保持轻量，只有启用路由时才导入 torch、vLLM 或厂商运行时。

新增平台时，在 `operators/platforms/<平台>/` 声明能力和实现，在 `selection.py` 注册名称；构建策略、补丁和性能记录分别归入对应目录。共享 C++ 源码跟随共享桥接，平台专用 C++ / CMake 源码跟随平台适配。

项目顶层 `register` / `unregister`、算子公开接口、构建脚本参数和 GSM8K 命令保持稳定。内部模块导入使用上述目录，仓库内引用和文档链接必须随文件移动同步更新；实验结果继续保存在 Git 忽略的 `results/`、`artifacts/` 或 `reports/`。
