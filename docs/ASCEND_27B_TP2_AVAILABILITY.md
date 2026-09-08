# Ascend 27B TP=2 Availability — 2026-09-07

The model in `/models/Qwen3.8-27B` runs with the InfiniCore plugin on two
Ascend 910B4 devices. Its actual config architecture is
`Qwen3_5ForConditionalGeneration` (`model_type=qwen3_5`), with hybrid linear
attention, Gemma-style RMSNorm and partial multimodal RoPE.

## Tested configuration

- Host/container: `npu-worker-08` / `zx-vllm-ascend-023`.
- Devices: `0,1`; tensor parallel size: 2; BF16 weights, no quantization.
- vLLM / vLLM-Ascend: `0.23.0+empty` / `0.23.0`.
- InfiniCore SHA: `d3551f37538896056e164abf91b120e38c27007b`.
- Eager mode, maximum context 512, maximum batch 4, batched tokens 512,
  memory utilization 0.90, prefix caching disabled, text-only inputs.
- Passing route selection: `VLLM_INFINICORE_ROUTES=Embedding,MatMul,LMHead`.
- Platform, workers, HCCL communication, attention and caches remain owned
  by vLLM-Ascend.

Each worker loads about 25.17 GiB of weights. The tested engine reports
about 0.87 GiB available for KV cache and 2,304 token capacity. These figures
apply to this short-context configuration, not arbitrary context lengths.

## Results

Prompts are tokenized once and shared across runs: an exact 128-token English
input and a 19-token Chinese chat input. Greedy sampling fixes each output
at 32 tokens with EOS ignored. After warmup, batch 1 and batch 4 English and
batch 1 Chinese cases each run twice, producing 12 checked outputs per run.

| Case | Generation | Exact token matches to native | Route validation |
| --- | --- | --- | --- |
| Native vLLM-Ascend TP=2 | 12/12 healthy outputs | Baseline | Pass |
| InfiniCore `all` TP=2 | 12/12 healthy outputs | 12/12 | Fails full route coverage |
| InfiniCore `projections` TP=2 | 12/12 healthy outputs | 12/12 | Pass, no validation errors |

Both InfiniCore cases record the following **per worker** real backend calls
after warmup, with identical counts on both TP ranks:

| Operation | Calls per worker |
| --- | ---: |
| Embedding | 224 |
| MatMul (`linear`) | 68,096 |
| LMHead | 224 |

The `all` case cannot claim all six adapters are exercised. The model's
normalization and RoPE paths do not invoke the ordinary RMSNorm/RoPE wrappers.
SwiGLU has TP-local intermediate width 8,704, exceeding the current InfiniCore
Ascend kernel limit of 8,192, so it records 14,336 native fallbacks per worker.
The harness correctly reports those missing real-kernel routes as coverage
errors despite correct generation. The passing `projections` configuration
requests only the three exercised adapters and leaves other operations native.

Example Chinese output:

> 人工智能是计算机科学的一个分支，旨在开发能够模拟、延伸和扩展人类智能，从而执行如学习、推理、感知和决策等复杂任务的系统。

This is an availability and correctness smoke test. Graph execution, long
contexts, image/video inputs and throughput are not validated here.

## Reproduction and artifacts

Inside the container:

```bash
bash /workspace/work/infinicore-27b-tp2-20260907/reproduce.sh
```

This runs native and projection cases sequentially and writes fresh results
under `/workspace/work/infinicore-27b-tp2-20260907/recheck` by default. It uses
the staged plugin at `/workspace/work/infinicore-ascend-20260907/site` and the
locked bridge at that directory's `build/libvllm_infinicore_ascend.so`.

Original remote results/logs:
`/workspace/work/infinicore-27b-tp2-20260907`.
Local copies: `artifacts/ascend-27b-tp2-20260907`, including all three cases,
worker counters, exact tokens, text-health checks and reproduction script.

The smoke harness now accepts model/device/TP/memory/text-only options and
checks the number of returned worker states against the requested TP size.
