# 27B Ascend Throughput With InfiniCore Operators — 2026-09-09

The compiled path now actually calls InfiniCore, and reaches 89.9-99.3% of
native vLLM-Ascend. The earlier
[`ASCEND_27B_GRAPH_THROUGHPUT.md`](ASCEND_27B_GRAPH_THROUGHPUT.md) measured
something else: it compiled a native backbone and, as its own text says, reached
InfiniCore only for LMHead outside the graph. This report supersedes it.

Eager-mode figures taken during this work (39.8-65.1% of native) are not
reproduced here. They were measured before the `record_stream` defect described
below, which throttled every InfiniCore launch, so they describe that defect
rather than the adapter.

## What was blocking it

`ascend_backend` reaches InfiniCore through ctypes and raw device pointers,
which Dynamo cannot trace. Two guards turned that into a silent native run:

- `execute()` returned the native operator whenever `torch.compiler.is_compiling()`
  was true, so the traced program never contained an InfiniCore call.
- A second guard rejected any call made while a stream was capturing.

Five changes remove both, and one more accounts for most of the speed:

1. **`ops/ascend_graph_ops.py`** registers each operator as a
   `torch.library.custom_op` in the `vllm_infinicore_ascend::` namespace with a
   fake implementation, giving the tracer an opaque node it can place in the
   graph. The `vllm_infinicore::` namespace was already taken by the default-off
   `ops/custom_ops.py` layer.
2. **Capability moves ahead of the node.** `supports_linear`,
   `supports_silu_and_mul`, `supports_rotary_embedding` and `supports_tensor`
   answer from dtype and shape alone and are shared by the eager path, so the
   two can never disagree. A compiled graph fixes its operators once, so an
   unsupported call must select the native operator at trace time rather than
   raise from inside the graph.
3. **`launch()` is capture-safe.** The descriptor key no longer includes the
   stream, so a descriptor warmed on the default stream is reused during
   capture instead of being rebuilt mid-capture; `record_stream` and cache
   eviction are skipped while capturing, both being illegal there; and a
   descriptor recorded into a graph is pinned, because destroying it would
   leave the graph replaying against freed state.
4. **Workspaces are shared per device rather than allocated per launch.** A
   per-launch workspace churned the caching allocator roughly 40,000 times per
   batch at `bs=32` and put an allocation inside every capture. One buffer at
   the high-water mark costs the maximum instead of the sum: a buffer per
   descriptor needs 342 MiB for the 27B shapes against 88 MiB shared, and the
   difference is enough to exhaust the headroom `gpu_memory_utilization` leaves
   and stop the engine from starting.
5. **`record_stream` is gone from the launch path.** This was the single largest
   effect, and is described below.

## Measured results

Output-only tokens/s, median of three measured runs after one warmup per batch,
1,024 input and 256 output tokens per request. Requested routes
`Embedding,MatMul,LMHead`; no operator fallback was recorded in any case.

| TP | Submitted batch | vLLM-Ascend | vLLM-InfiniCore | InfiniCore / Ascend |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1 | 20.48 | 20.34 | 99.3% |
| 2 | 4 | 66.98 | 64.86 | 96.8% |
| 4 | 1 | 32.82 | 32.42 | 98.8% |
| 4 | 4 | 100.17 | 95.51 | 95.3% |
| 4 | 16 | 237.13 | 215.52 | 90.9% |
| 4 | 32 | 311.18 | 279.86 | 89.9% |

Every cell is stable to within about 1% across its three repeats, on both
engines.

**TP=2 at batch 16 and 32 is excluded: it measures a regime difference, not
operator throughput.** Those cells report InfiniCore *faster* than native — 130.6
against 53.2 tok/s at `bs=32`, a factor of 2.46 — reproducibly, across three
fresh engine launches each (native 54.06 / 52.92 / 52.51, InfiniCore
131.41 / 130.32 / 129.94). Instrumenting both engines shows why, and it is not
the operators:

| | forward passes | ACL graph replays | not replayed | ms per pass |
| --- | ---: | ---: | ---: | ---: |
| native | 781 | 501 | 280 | 199.6 |
| InfiniCore | 783 | 745 | 38 | 80.1 |

Both engines run the same number of forward passes for the same output, so
neither is recomputing more than the other. The difference is that InfiniCore
gets almost every step onto the captured decode graph while native leaves 280
steps to execute eagerly, and a replayed step is far cheaper. Why the scheduler
produces more capturable pure-decode batches for one engine than the other at
this cache size is not established. Until it is, the ratio says something about
vLLM-Ascend's batching under KV cache pressure, not about InfiniCore, and it
would be wrong to quote it as a speedup.

Ruled out along the way, each by measurement: a kernel difference (native's
`torch.ops.vllm.unquantized_gemm` is literally `torch.nn.functional.linear`, and
that is at parity with InfiniCore at every shape); extra recompute by native
(equal pass counts); an unstable native baseline (steady to 3% across launches);
and KV cache capacity, which cannot matter once the pass counts match.

TP=4 never enters this regime, with ~200,320 cache tokens against TP=2's ~15,040.

## What the remaining gap was: record_stream

The launch path called `tensor.record_stream(stream)` on every tensor of every
launch. That is only meaningful when a tensor is used on a stream other than the
one it was allocated on; this adapter always launches on the tensors' own current
stream, which the caching allocator already orders against, so it protected
nothing. It was not free. `record_stream` makes the allocator defer reuse of a
block until it observes a stream event, so with a fresh output allocated per call
— 71 MiB at a 2,048-token prefill tile — each allocation ended up waiting on
device progress instead of pipelining.

Removing it, measured on a 4-sequence prefill at TP=2:

| | elapsed | `output_alloc` rank 0 / rank 1 | host total |
| --- | ---: | ---: | ---: |
| with `record_stream` | 12.15 s / 16.43 s | 166 / 595 µs per call | 2.12 s / 5.03 s |
| without | 2.01 s / 2.02 s | 36.9 / 37.7 µs per call | 1.16 s / 1.16 s |

Native runs the same case in 1.80 s, so prefill went from 6.8x native to 1.12x.

This one call accounted for a cluster of symptoms that had been chased
separately: an unexplained 54.2% cell at TP=2 `bs=4`, intermittent stalls at TP=4
`bs=1` (25.7 against 32.3 tok/s across repeats), an asymmetry where one rank
spent 3.6x the other in allocation, and the fact that it looked like allocator
pressure yet `expandable_segments:True` changed nothing. It also explains why
every isolated microbenchmark showed parity: an idle device has free memory, so
deferred reuse never blocks. The effect scales with memory pressure, which is why
TP=2 (~15,040 cache tokens) was hit hard and TP=4 (~200,320) barely at all —
TP=4 `bs=32` moved only from 87.8% to 89.9%.

Measurements taken before this fix, including a full matrix with
`enable_chunked_prefill=False`, are superseded and are not reproduced here.

## Kernel parity

Per-launch cost on the 27B TP=2 per-rank shapes, measured by capturing 50
launches into an ACL graph and timing the replay. A wall-clock loop cannot
answer this at small `M`: both native dispatch and a ctypes launch are
enqueue-bound there, so the measurement reports launch overhead and hides the
kernel entirely. A graph replay performs no enqueue, which is also exactly what
the production decode path does.

| Shape | M | native | InfiniCore | ratio |
| --- | ---: | ---: | ---: | ---: |
| qkv | 1 | 49.8 µs | 49.5 µs | 0.99x |
| gate_up | 1 | 256.6 µs | 256.1 µs | 1.00x |
| qkv | 4 | 49.4 µs | 49.3 µs | 1.00x |
| gate_up | 4 | 257.1 µs | 257.3 µs | 1.00x |
| down | 16 | 39.2 µs | 39.1 µs | 1.00x |
| gate_up | 32 | 268.4 µs | 268.3 µs | 1.00x |
| qkv | 2048 | 596.7 µs | 602.0 µs | 1.01x |
| gate_up | 2048 | 1841.0 µs | 1848.0 µs | 1.00x |

The kernels are indistinguishable from native at every shape from a single
decode token to a full prefill tile, because both reach the same CANN aclnn
GEMM. Outside a graph each InfiniCore call additionally pays about 270 µs of
Python, which is what a captured step skips and a non-captured step does not.

One consequence worth recording: `torch_npu.npu_format_cast` **cannot run during
graph capture** — it raises `Cannot run aclop operators during NPU graph
capture`. `nd()` calls it whenever a tensor is not already in ND format, so any
model whose weights are converted to `FRACTAL_NZ` would fail capture rather than
run slowly. `VLLM_ASCEND_ENABLE_NZ` defaults to 1, so this is reachable; it does
not occur for this checkpoint, whose weights are already ND.

## Route coverage on this model

Requested routes are `Embedding,MatMul,LMHead`. That is not an arbitrary
subset: it is every route that InfiniCore can actually serve for this
architecture. Running `VLLM_INFINICORE_ROUTES=all` against the 27B installs six
routes but three of them contribute nothing:

| Route | Qwen3-0.6B (single card) | Qwen3.5-27B |
| --- | --- | --- |
| `Embedding` | executes | executes |
| `MatMul` | executes | executes |
| `LMHead` | executes | executes |
| `RMSNorm` | executes (10,944 calls) | never called |
| `RoPE` | executes (5,376 calls) | never called |
| `SiluAndMul` | executes (5,376 calls) | called 33,088 times, **falls back every time** |
| `StoreKVCache` | native by design | native by design |
| `PagedAttentionPrefill` | native by design | native by design |
| `PagedAttentionDecode` | native by design | native by design |

- `SiluAndMul` reports the fallback reason `SwiGLU requires eight aligned tiles
  and hidden size <= 8192`. This model's `intermediate_size` is 17,408, so the
  InfiniCore SwiGLU kernel cannot run at all here.
- `RMSNorm` receives no calls because vLLM-Ascend pins the IR op priority to
  `IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])`.
- `RoPE` receives no calls because 48 of the 64 layers are `linear_attention`
  and use a gated-delta-rule path rather than the standard rotary op.

Attention, KV cache, device management, workers and communication remain owned
by vLLM-Ascend in both engines. The comparison therefore isolates InfiniCore's
GEMM, embedding and logits projection against their native Ascend equivalents,
with every other operator identical on both sides.

## Single card

TP=1 cannot run this checkpoint. The BF16 weights are 51.75 GiB and a 910B4
exposes 29.49 GiB usable, so weight loading aborts:

```
torch.OutOfMemoryError: NPU out of memory. Tried to allocate 342.00 MiB
(NPU 0; 29.49 GiB total capacity; 28.95 GiB already allocated;
 28.95 GiB current active; 159.95 MiB free; 28.96 GiB reserved in total by PyTorch).
```

No BF16 single-card configuration exists for this model on this hardware.
Quantization and CPU offload were not introduced, because either would change
the weights or the data path and make the native comparison meaningless.

Single-card function was therefore validated on `/models/Qwen3-0.6B`, which is
the harness default. That case passes with `validation_errors: []`, installs
all six non-attention routes, records InfiniCore calls on every one of them
(`embedding` 192, `rms_norm` 10,944, `linear` 21,504, `rotary_embedding` 5,376,
`silu_and_mul` 5,376, `lm_head` 192) and produces **12/12 outputs identical to
native token-for-token**. Only `fused_add_rms_norm` falls back. The build and
the adapter are therefore proven working on one card before any TP run.

## Correctness

Every case completed with `validation_errors: []`, six captures on every rank,
and no operator fallback.

**TP=2.** Both engines reproduce themselves exactly across repeats. Against each
other, InfiniCore is token-identical at `bs=1` (3/3) and `bs=4` (12/12) and
diverges as batch grows, consistent with a different GEMM accumulation order at
larger shapes turning into a different greedy token that then compounds.

**TP=4.** Neither engine reproduces its own repeats, so the cross-engine counts
(0/3 at `bs=1`, 0/12 at `bs=4`, 3/48, 9/96) carry no information about
InfiniCore. This nondeterminism is a property of the vLLM-Ascend TP=4
configuration, is present with the plugin absent, and was recorded independently
in both earlier reports. No TP=4 correctness claim is made in either direction.

## Known limits

- **TP=2 at batch 16 and 32 has no quotable ratio**, for the reason given under
  the results table: the two engines get very different numbers of steps onto
  the captured graph (745 of 783 against 501 of 781), so the cell measures
  vLLM-Ascend batching under KV cache pressure rather than operator throughput.
  The cause of that divergence is open.
- The TP=4 `bs=1` instability reported before the `record_stream` fix (25.7,
  28.5, 32.3 tok/s) is resolved: the same cell now measures 32.4, 32.4 and 32.6
  against native's 32.8. An earlier theory blaming descriptor-cache eviction was
  tested and disproved — the working set is 82-88 descriptors with zero
  evictions — before `record_stream` was identified.
- Routes are unchanged in scope: only `Embedding`, `MatMul` and `LMHead` execute
  on this architecture. `SiluAndMul` still cannot run because the model's
  `intermediate_size` of 17,408 exceeds the kernel's 8,192 limit.
- `VLLM_INFINICORE_ASCEND_GRAPH=0` restores the previous eager-only behaviour
  and lets capture fall back to native.

## Reproduction

```bash
python scripts/build_ascend.py --build-dir /workspace/infinicore-build \
  --soc Ascend910B4 --cann "$ASCEND_TOOLKIT_HOME"
pip install --no-deps .
python tests/remote/bench_ascend_throughput.py prepare --root artifacts/tp2-graph-r3 \
  --input-len 1024 --output-len 256
for case in native infinicore; do
  python tests/remote/bench_ascend_throughput.py $case --root artifacts/tp2-graph-r3 \
    --tp 2 --devices 0,1 --batches 1,4,16,32 --input-len 1024 --output-len 256 \
    --repeats 3 --warmups 1 --graph --routes Embedding,MatMul,LMHead \
    --library /workspace/infinicore-build/libvllm_infinicore_ascend.so
done
```

Use three repeats. Two is not enough: the median of two values is their mean, so
a single stalled repeat moves the reported figure by tens of percent. An earlier
two-repeat pass reported TP=4 `bs=16` at 33.2%, which three repeats put at 90.5%.

Host `npu-worker-08`, container `zx-vllm-ascend-023`, Ascend 910B4, TP=2 on
cards 0-1 and TP=4 on cards 0-3, cards idle before each case. vLLM `0.23.0+empty`,
vLLM-Ascend `0.23.0`, torch `2.10.0+cpu`, torch_npu `2.10.0.post4`, CANN 9.1.0.
InfiniCore pinned at `d3551f37538896056e164abf91b120e38c27007b`, library SHA256
`ab76cfadb6e16c5194d70abb78d22b248ce0cc0636ddae86e641bbd91e52d1a0`.
Graph settings: `CompilationMode.VLLM_COMPILE`, `CUDAGraphMode.FULL_DECODE_ONLY`,
capture sizes `[1,2,4,8,16,32]`. Sampling: temperature 0, top_p 1, top_k 1, EOS
ignored, `min_tokens=max_tokens=256`. One shared 1,024-token prompt, hash
verified equal across engines at each TP.
