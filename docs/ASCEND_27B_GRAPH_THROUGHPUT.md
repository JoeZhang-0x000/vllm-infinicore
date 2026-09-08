# 27B Ascend Graph Throughput — 2026-09-07

The tested InfiniCore integration runs at approximately native throughput.
The compiled backbone uses native Ascend implementations; **only LMHead outside
the graph records actual InfiniCore calls**. These ratios measure that mixed
implementation and do not establish performance of InfiniCore kernels inside
an Ascend graph.

## Measured results

Output-only tokens/s, median of three measured runs after one warmup per batch.
Input and output are each exactly 1,024 tokens per request.

| TP | Submitted batch | vLLM-Ascend | vLLM-InfiniCore | InfiniCore / Ascend |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1 | 20.99 | 20.94 | 99.76% |
| 2 | 4 | 74.83 | 74.68 | 99.81% |
| 2 | 16 | 72.80 | 74.07 | 101.75% |
| 2 | 32 | 60.08 | 60.56 | 100.81% |
| 4 | 1 | 34.05 | 34.01 | 99.86% |
| 4 | 4 | 117.47 | 117.24 | 99.81% |
| 4 | 16 | 348.37 | 349.07 | 100.20% |
| 4 | 32 | 534.05 | 533.80 | 99.95% |

TP=1 was not launched: the BF16 checkpoint is 51.75 GiB, and TP=2 loads about
25.17 GiB of text-model weights per rank, exceeding single-card capacity when
combined. CPU offload and quantization were not introduced into this comparison.
TP=8 is unavailable in the specified container: it maps only devices 0–3;
host devices 4–7 were occupied by other processes when inspected. Neither
missing configuration has an inferred throughput or percentage.

## Hardware, workload and timing

- Host: `npu-worker-08`; container: `zx-vllm-ascend-023`.
- Ascend 910B4, 32 GiB physical HBM per card; TP=2 uses 0,1 and TP=4 uses 0–3.
- Model: `/models/Qwen3.8-27B`; config architecture is
  `Qwen3_5ForConditionalGeneration`, model type `qwen3_5`.
- BF16, text-only, no quantization or offload. InfiniCore is locked at
  `d3551f37538896056e164abf91b120e38c27007b`.
- vLLM `0.23.0+empty`, vLLM-Ascend `0.23.0`, torch `2.10.0+cpu`,
  torch_npu `2.10.0.post4`. Exact versions and source/library hashes are in
  the artifact manifest.
- Graph: `enforce_eager=False`, `CompilationMode.VLLM_COMPILE`,
  `CUDAGraphMode.FULL_DECODE_ONLY`, capture sizes `[1,2,4,8,16,32]`.
  Ascend's configured `enable_npugraph_ex` is enabled in both engines.
- Context length 2,048; max sequences 32; max batched tokens 1,024;
  memory utilization 0.95; prefix caching disabled; image/video limits zero.
- One shared 1,024-token English prompt is generated once by the model tokenizer.
  Every request uses those exact IDs; prompt hashes and engine kwargs match
  between engines at each TP size.
- Sampling: temperature 0, top_p 1, top_k 1, EOS ignored,
  min_tokens=max_tokens=1,024. Warmup and all three measured repeats use
  the requested output length.
- Engines run sequentially on the same selected devices. Timing covers the
  synchronous `LLM.generate` call, including prefill, decode and scheduling.
  Loading, compilation, warmup, worker RPCs and text validation are excluded.
- Each engine/TP case measures 159 request outputs across 12 timed batches.
  Each row's throughput is actual generated output tokens divided by elapsed
  wall time; the percentage divides the two medians, with no mixed-TP average.

## Capacity and graph evidence

Both engines have identical reported cache capacity within each TP setting:

| TP | Weights per rank | Cache token capacity | Full 2,048-token concurrency estimate |
| --- | ---: | ---: | ---: |
| 2 | 25.17 GiB | 20,070 | 9.8 |
| 4 | 12.65 GiB | 257,228 | 125.6 |

Batch size is the number of requests submitted together. In TP=2, batches
16 and 32 exceed full-length resident cache capacity; the table measures
completion of the submitted batch under scheduling pressure. It is not a
claim that all full-length requests remain resident concurrently.

Every rank in all four completed cases records six captures and replay activity.
The following counters include warmups as well as measurement:

| Case | Captures per rank | Replays per rank | Actual InfiniCore calls per rank |
| --- | ---: | ---: | --- |
| Native TP=2 | 6 | 28,540 | None |
| InfiniCore TP=2 | 6 | 28,540 | LMHead: 32,865 |
| Native TP=4 | 6 | 16,368 | None |
| InfiniCore TP=4 | 6 | 16,368 | LMHead: 16,593 |

The requested plugin routes are `Embedding,MatMul,LMHead`. Embedding and
MatMul execute as native operations in the compiled backbone. The Python
compile bypass intentionally has no runtime counter side effects; empty
fallback counters do not mean these routes used InfiniCore. Devices, workers,
communication, attention and caches remain managed by vLLM-Ascend.

## Correctness observations and limits

- TP=2: all 159 plugin outputs exactly match native token-for-token. Each engine
  also has 106/106 exact matches when comparing its first repeat with later
  repeats at each batch/index.
- TP=4: both native and plugin have 0/106 exact output matches across their own
  repeats; cross-engine matches are 0/159. The first differing tokens in inspected native
  examples occur after substantial shared prefixes. Output lengths and text
  health are checked, but strict token-level determinism/correctness is not
  established for this TP=4 graph configuration. The source of the variation
  was not isolated in this throughput task.
- Native TP=4, bs=16, repeat=1, output index=7 triggers the repetition detector
  because its coherent computer-architecture explanation contains a
  backtick-delimited binary instruction example with 59 consecutive zero tokens.
  Full-text inspection confirms prose continues normally after the literal.
  `text_health_review.json` records the exact output hash, detector reasons and
  manual classification. Raw warnings remain in the result JSON; the original
  native TP=4 process exits 1 solely because of this detector flag. Other cases
  have no detector/graph/token-count validation errors.
- Small percentage differences around 100% should not be attributed to a general
  InfiniCore graph-kernel speedup: the graph backbone is native in these runs.

## Compatibility fix and reproduction

Both engines use VLLM_COMPILE, which initializes the Ascend graph workspaces.
During compilation the plugin directly returns the native tensor program,
without Python exception or counter side effects. Real InfiniCore capture
remains disabled. Fourteen adapter tests, including a fullgraph regression,
pass against both source and the staged installation.

The latest complete reproduction script is
`artifacts/ascend-27b-graph-20260907/reproduce.sh` in this workspace. Copy it
into the container and run it there with NPU 0–3 free. It uses the staged
source and plugin paths already present in that container.

The script writes fresh data under `recheck` by default, runs all four cases
sequentially and retains a nonzero overall exit code if any raw validation
flag occurs. Inspect such flags before accepting the result; the known binary
literal review does not disable the generic detector.

Remote artifacts: `/workspace/work/infinicore-27b-graph-20260907`.
Local copies: `artifacts/ascend-27b-graph-20260907`, including raw per-request
outputs, elapsed times, worker evidence, failures, review, manifest, script and
`summary.json`. All benchmark workers have exited and devices 0–3 are released.
The last upload of the postprocessed summary/report/revised reproduction script
failed because the SSH jump host closed the connection. The complete final
archive is local; raw case JSON/logs, the original reproduction script and the
validated compile-fallback fix are already in the container. The older remote
script can stop at the native TP=4 raw repetition flag; use the revised local
script to run all cases before reviewing flags.
