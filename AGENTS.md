# vllm-infinicore 智能体指南

**更新：** 2026-09-17

## 先读这些

改动本项目前请先读：

- [`README.md`](README.md) — 项目定位与六个优化方向、**术语与命名约定**、环境变量清单
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 六个优化方向（含**对外表述红线**）、分层设计、图策略、基准公平性规则
- [`docs/QWEN3_OP_SCOPE.md`](docs/QWEN3_OP_SCOPE.md) — 九条 scoped 路由与覆盖率陷阱
- [`docs/ASCEND.md`](docs/ASCEND.md) — Ascend 集成、最新性能矩阵与限制
- [`docs/DEV_LOG.md`](docs/DEV_LOG.md) — 2026-09-01 起的开发日志与对应 commit

**所有文档使用中文**，代码与代码注释保持英文。新文档必须遵守 README 中的术语表，不要引入别名。

**不要用旧的 TPS 表下结论。** 更早的 `vLLM-NT` / `InfiniLM` 基准表已作废，其 artifacts 已随
`9f8ba43 chore: remove generated artifacts` 清理。除非有按当前公平性规则重新测出的结果，
否则不要声称 InfiniLM 或本插件快于 vLLM 原生 cudagraph。

## 平台与环境

三条平台线。具体环境变量见 [README](README.md)，此处只列每条线的要害：

| 平台 | 工作方式 | 要害 |
|---|---|---|
| Ascend NPU 910B4 | 远端容器（`npu-worker-08` / `zx-vllm-ascend-023`），模型在 `/models` | 需要 `VLLM_INFINICORE_ASCEND_LIBRARY`；未设置时九条路由全部保持原生 |
| MetaX C550 | 远端主机，MACA `3.8.0.23`，模型在 `/root/models` | vLLM 0.22 必须设 `VLLM_USE_V2_MODEL_RUNNER=0`；用 `run-infinicore.sh` / `run-metax.sh` wrapper 固定后端 |
| MUSA | 仅打通启动/图路由/TP 通信 | 无正式性能结果，不要引用 MUSA 吞吐数字 |

最初的 MACA 3.5.3 本地开发目标已不再是当前目标。

默认行为是保守的：`vllm_infinicore.register()` 不安装任何 monkey patch。
做 dry import 和基线运行时保持 `VLLM_INFINICORE_ENABLE_PATCHES=0` 或不设置。

## 图规则

**MetaX** 上 vLLM 原生 cudagraph 可用：

```python
from vllm.config import CUDAGraphMode

comp_config = {
    "cudagraph_mode": CUDAGraphMode.PIECEWISE,
    "cudagraph_capture_sizes": [1, 2, 4, 8],
    "cudagraph_num_of_warmups": 1,
    "backend": "eager",
}
llm = LLM(..., enforce_eager=False, compilation_config=comp_config)
```

- 测 cudagraph 时不要用 `enforce_eager=True`。
- **MetaX 上不要用 `CompilationMode.VLLM_COMPILE`**，除非就是在测编译失败模式。
- 用 `backend="eager"` 跳过 torch.compile 同时保留 cudagraph。
- 在每条打过 patch 的路径被证明图安全之前，保持本插件图保守。

**Ascend** 上相反：使用 `CompilationMode.VLLM_COMPILE` 配 `CUDAGraphMode.FULL_DECODE_ONLY`，
prefill 不在 decode graph 内。该栈上默认 KV block size 16 会让 graph 初始化失败，需显式请求 128。

## 基准规则

基准脚本必须：

1. 用模型 tokenizer 一次性生成 prompt token ID，并在各引擎间复用同一份。
2. 记录实际的输入与生成输出 token 数。
3. 以**输出 TPS**（定义见 README）为主指标。
4. 对齐采样：`temperature=0.0`、`top_p=1.0`、`top_k=1`、禁用 EOS，vLLM 侧
   `min_tokens=max_tokens=output_len`。
5. 先预热，**再做三次测量重复**——两次不够，两个值的中位数就是均值。
6. 在信任 TPS 之前先打印解码输出预览和文本健康计数。
7. 记录 graph capture 数、路由状态与后端计数器。

几个反复踩到的坑：

- **后端计数非零不能证明一条路由活在热路径上**，torch.compile 在 trace 时就解析掉 Python 分支。
- 图模式下的 Python 计数只反映预热、capture 和未捕获路径，不是全部 replay 算子数。
- **不要用 cProfile 的 cumtime 去估高频路径上的优化收益**，要做 A/B。
- 判断 host 侧优化需要完整基准 harness；短 profiler 窗口只对结构性证据（设备算子时间、
  GPU busy、设备事件计数）可靠。
- 卡死或失败的运行会遗留 worker 并占住显存，使后续用例因显存不足而失败，必须回收后再继续。

## 验证

```bash
python -m compileall vllm_infinicore
python -c "import vllm_infinicore; vllm_infinicore.register()"
python -m unittest discover -s tests
```

跑测试套件需要设置 `VLLM_PLUGINS`；不设时只跑 80 个测试且 4 个平台/注册用例报错。
在 MetaX 主机上通过后端 wrapper 运行以拿到完整套件。

## 通知

只对重要结论或数据使用上级基准项目配置的飞书任务更新 webhook：

- 带结果数据或 artifact 路径的正式基准完成；
- 重要的正确性、图安全性或性能结论；
- 需要人介入的阻塞性失败，以及此类失败的解除。

不要为常规文件编辑、普通任务完成、小重构、dry import、编译检查、中间进度或非阻塞发现发通知，
这些用一条简洁的最终聊天更新即可。
