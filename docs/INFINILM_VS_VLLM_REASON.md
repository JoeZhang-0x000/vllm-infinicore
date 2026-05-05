# InfiniLM vs vLLM Native Throughput Reason

Date: 2026-05-05

This note explains why InfiniLM can be faster than vLLM native in the current
Qwen3-8B graph benchmark. It uses only current local measurements, not
historical TPS tables.

## Setup

- Model: `/mnt/geogpt-doc-new/default/xb/qwen3-8B`
- Batch size: `8`
- Input length: `4096`
- Graph enabled for both engines
- Sampling aligned: `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS disabled
- Prompt token IDs generated once and reused
- Metric: output-only TPS

## Output Length Sweep

| Output length | InfiniLM TPS | vLLM native TPS | Faster engine |
|---:|---:|---:|---|
| 32 | 55.94 | 69.30 | vLLM native |
| 128 | 157.30 | 175.60 | vLLM native |
| 512 | 288.05 | 282.72 | InfiniLM |
| 1024 | 332.96 | 311.57 | InfiniLM |

Short outputs favor vLLM native, while long outputs favor InfiniLM. This means
InfiniLM's advantage is not lower prefill/fixed overhead. It appears when the
decode loop dominates the measured time.

## Fixed Cost vs Decode Cost

A linear fit of average iteration time against output length gives:

| Engine | Fixed cost / iteration | Decode step cost | Steady decode TPS at bs=8 |
|---|---:|---:|---:|
| InfiniLM | 3.919 s | 20.184 ms/token-step | 396.35 tok/s |
| vLLM native | 2.916 s | 22.784 ms/token-step | 351.12 tok/s |

Interpretation:

- vLLM native has lower fixed cost, which is why it wins at output lengths 32
  and 128.
- InfiniLM has lower per-step decode cost, which is why it wins once output
  length is long enough to amortize its higher fixed cost.
- The crossover occurs between 128 and 512 generated tokens for this exact
  `bs=8`, `input_len=4096` shape.

## Code Path Difference

InfiniLM's direct `InferEngine.generate()` path is a fixed-shape batch decode
loop. In each decode step, Python prepares small metadata tensors, then calls
the C++ `_infinilm.InferEngine.forward()` and receives `output_ids`. The
sampling parameters are passed into that C++ forward call, so model forward and
token selection are handled inside the InfiniLM engine path.

vLLM native uses the general V1 serving engine even in offline `LLM.generate()`:
the log shows chunked prefill, asynchronous scheduling, vLLM compile/eager
backend setup, CUDA graph capture, scheduler output handling, sampling, request
state updates, and final output assembly. This generality is valuable for
serving, mixed workloads, preemption, prefix/cache management, and streaming,
but it leaves more per-token machinery in the decode loop.

## Ruled-Out Explanation

The difference is not caused by vLLM detokenization. A 512-token vLLM native
control run produced:

| vLLM detokenize | Output TPS |
|---|---:|
| true | 283.61 |
| false | 282.98 |

The difference is within run noise and does not explain InfiniLM's 1024-token
advantage.

## Conclusion

The core reason InfiniLM is faster in long-output runs is lower steady-state
decode-loop cost, not lower fixed/prefill overhead. In this benchmark, InfiniLM
is a narrower fixed-batch inference path, while vLLM native carries the general
serving scheduler/request/sampling/output machinery around each decode step.

For short outputs, vLLM native is faster because its fixed cost is lower. For
long outputs, InfiniLM becomes faster because its per-token decode step is about
`11.4%` lower (`20.184 ms` vs `22.784 ms` per batch token-step).
