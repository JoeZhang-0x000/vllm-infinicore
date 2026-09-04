# Development Log

## 2026-09-04 Decode Gap Localization And Two Fixes

Moved to a dedicated single-card host,
`ssh.v5000-prod-gw.nhss.zhejianglab.com:31919` (one MetaX C550, MACA `3.8.0.23`,
Python `3.10.10`, PyTorch `2.10.0+metax3.8.0.7`, vLLM `0.22.0` and matching
`vllm-metax`). The previous host was shared with unrelated GPU jobs that held
all four cards, which invalidated any throughput measurement taken there.
`/root/InfiniCore` was migrated whole from the old host over the shared
`/mnt/geogpt-doc-new` mount, preserving upstream commit `35b4627` and its local
`INFINICORE_METAX_FLASHATTN_38_ABI` compatibility patch; a copy of that patch is
kept at
`/mnt/geogpt-doc-new/default/zx/xfer-20260903/infinicore-metax-fa38-abi.patch`.
The only missing prerequisite on the new host was the OpenMPI runtime
(`libmpi.so.40`); installing `libopenmpi-dev` `4.1.2` matched the old host.

### Benchmark Shape

TP=1, BF16, `batch_size=8`, PIECEWISE CUDA Graph with capture sizes
`[1,2,4,8]` and `backend="eager"`, one warmup and three measured iterations,
deterministic sampling with EOS disabled, `min_tokens == max_tokens`, and the
same prompt token IDs shared by both engines per model. Chunked prefill is
disabled with `max_num_batched_tokens = max(16384, batch_size * input_len)` so
every prefill lands in one scheduler step and never mixes with decode.
`VLLM_USE_V2_MODEL_RUNNER=0` and `VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA=1`.

Two traffic shapes: short-input/long-output `in=256/out=2048` and
long-input/short-output `in=4096/out=256`. `input_len=128` could not be used
while these numbers were taken: the strict metadata builder hardcoded
`reorder_batch_threshold = 128`, so a 128-token prefill was classified as decode
and rejected as `unsupported_spec_decode:1024!=8`. That defect is fixed further
down in this entry; the measurements above predate the fix and are unaffected by
it, because every shape they use has `input_len >= 256`.

Baseline before either fix (`artifacts/bench-tp1-*-20260903`):

| Shape | Model | vLLM-MetaX | InfiniCore | Ratio |
|---|---|---:|---:|---:|
| `in256/out2048` | Qwen3-0.6B | 1057.98 | 662.21 | 62.59% |
| `in256/out2048` | Qwen3-4B-Instruct-2507 | 614.89 | 458.89 | 74.63% |
| `in256/out2048` | Meta-Llama-3-8B-Instruct | 466.57 | 375.08 | 80.39% |
| `in4096/out256` | Qwen3-0.6B | 742.43 | 477.59 | 64.33% |
| `in4096/out256` | Qwen3-4B-Instruct-2507 | 367.45 | 248.63 | 67.66% |
| `in4096/out256` | Meta-Llama-3-8B-Instruct | 285.47 | 220.52 | 77.25% |

### Where The Gap Was

`scripts/decode_gap_profile.py` was added to separate kernel time from host
stalls. It measures a clean steady-state decode step latency with the profiler
off (differencing two output lengths so prefill and sampling setup cancel), then
profiles a decode window for device kernel time, GPU busy fraction, and host op
breakdown. Note that its `self_device_time_total` sum must be restricted to
`DeviceType.CUDA` events; summing host op events too double counts and reports a
GPU busy fraction above 100%.

Llama-3-8B, `bs=8`, `in=256`, 128 steady-state decode steps, before either fix:

| Metric | vLLM-MetaX | InfiniCore |
|---|---:|---:|
| Decode step | 16.83 ms | 20.47 ms |
| Device kernel time | 2605.9 ms | 2870.0 ms |
| GPU busy | 87.64% | 75.55% |
| Host CPU total | 2230.9 ms | 2990.1 ms |
| `unified_attention_with_output` host self | 336.1 ms | 800.9 ms |
| `unified_kv_cache_update` host self | 92.1 ms | 343.7 ms |

The core math was never the problem. The same `mcblas` GEMMs and the same
`flash_fwd_splitkv_kernel` run on both paths at matching cost
(`1904.11` vs `1893.99` ms and `86.95` vs `90.08` ms). The gap was host-side
starvation plus extra glue kernels.

Normalized per layer per decode step, the host overhead was `+90.8` us for
attention and `+49.1` us for the KV update, totalling `+139.9` us. The same two
numbers came out of Qwen3-0.6B to within `0.1` us despite a different layer
count and hidden size, which identified a fixed per-call cost rather than
anything proportional to model size. That also explains the original ratio
trend: at `45.8%` GPU busy Qwen3-0.6B was already host-bound, so the same
absolute overhead cost it far more than it cost Llama-3-8B at `87.6%`.

### Fix One: Cache The C++ Bridge Target

`cpp_bridge._bridge_target()` had no cache. Every decode attention call reached
`_torch_musa_package_dirs()` and ran `importlib.util.find_spec("torch_musa")`,
probing for a Moore Threads runtime on a MetaX machine. cProfile recorded
`5088` calls, exactly one per layer per decode step, expanding into `152640`
`_path_join` calls and `31008` `posix.stat` calls.

`_bridge_target()` now memoizes per distinct `VLLM_INFINICORE_CPP_BRIDGE_TARGET`
value, mirroring the existing `_ROUTES_CACHE_KEY` pattern so the env override
still works, and `_torch_musa_package_dirs()` caches its probe for the process.
`reset_bridge_target_cache()` exists for tests.

Same-run A/B at `in256/out2048`:

| Model | Before | After | Delta | Ratio |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 662.21 | 793.32 | `+19.8%` | 62.59% -> 75.09% |
| Meta-Llama-3-8B-Instruct | 375.08 | 377.75 | `+0.7%` | 80.39% -> 81.01% |

The metax baselines moved by less than `0.5%` across the same pair of runs, so
the delta is real. The split is the point: removing host work only helps where
the host is the limiter. On Llama-3-8B the freed host time converted directly
into more blocking `mcStreamSynchronize` wait (`1018` -> `1380` ms) and the
decode step did not move at all.

cProfile attributed `0.733` s of a `4.09` s window to this call chain, which
would have predicted a far larger win. Almost all of it was cProfile's own
per-call overhead across roughly 400k tiny frames. Do not size an optimization
from cProfile cumtime on a call-heavy path; A/B it.

### Fix Two: Route The Fused Residual-Add RMSNorm

`InfiniCoreRMSNorm._should_use_infinicore()` required `residual is None`, and
its docstring stated the fused-add path intentionally used vLLM's native
implementation. A runtime branch count showed that in a decoder layer **56 of
57 RMSNorm sites carry a residual**, so the RMSNorm route was installed,
reported nonzero backend counts, and was still absent from 98% of its own calls.
torch.compile resolves that Python branch at trace time, so nonzero
`backend_call_counts` is not evidence that a route is live in the hot path.

`infiniop` already exposed `infiniopAddRMSNorm`
(`y`, `residual_out`, `a`, `b`, `weight`, `epsilon`), which matches vLLM's
`fused_add_rms_norm` contract exactly; nothing had been wired to it. Added:

- `add_rms_norm_current_stream` in `vllm_infinicore/csrc/infinicore_bridge.cpp`
- `vllm_infinicore::fused_add_rms_norm` in `vllm_infinicore/ops/custom_ops.py`
- backend dispatch and a torch fallback in `ops/infinicore_backend.py`
- the residual path in `ops/vllm_rms_norm.py`, gated on `pass_weight_add`
  because vLLM only applies the weight on the fused path when that flag is set,
  and on matching residual shape and dtype

Numerics were checked against an fp32 reference before any throughput run:
bf16 and fp16 agree within `2` ULP at every shape tested and `residual_out` is
bit-exact. fp32 shows up to `5.7` ULP at `3.4e-7` relative error, which is
summation-order noise.

Post-fix branch counts confirm `56/56` residual sites route to InfiniCore with
zero fallbacks, `pass_weight_add=True` on all sites, and backend `rms_norm`
counts roughly doubling.

Same-run A/B at `in256/out2048`, on top of fix one:

| Model | Before | After | Delta | Ratio |
|---|---:|---:|---:|---|
| Meta-Llama-3-8B-Instruct | 377.75 | 403.63 | `+6.9%` | 81.01% -> 86.62% |
| Qwen3-0.6B | 793.32 | 826.15 | `+4.1%` | 75.09% -> 81.31% |

`infiniopAddRMSNorm` is not registered for every backend that registers plain
`infiniopRMSNorm`. Comparing the InfiniCore dispatch tables, `add_rms_norm`
covers `ALI CAMBRICON CPU HYGON ILUVATAR KUNLUN METAX MOORE NVIDIA QY` while
`rms_norm` additionally covers `ASCEND`. Routing the residual path
unconditionally would therefore turn a missing kernel into a failed run under
strict mode, on a path that previously never touched InfiniCore at all. MUSA is
covered, so the MOORE work is unaffected.

The bridge gained a descriptor-only capability probe -- no workspace, no launch,
no extra allocation -- returning false for `NOT_IMPLEMENTED`,
`DEVICE_TYPE_NOT_SUPPORTED` and `DEVICE_ARCHITECTURE_NOT_SUPPORTED` and
rethrowing every other status so a genuine failure is never misread as a missing
capability. The verdict is cached for the process and the residual path falls
back when the device has no kernel.

The probe runs inside the custom op, where the tensors are real. Probing from
`_should_use_infinicore` would run under torch.compile tracing on fake tensors,
and that branch is baked into the graph before any device call happens, so
keeping the decision inside the opaque op is what lets a late capability verdict
change behaviour without changing the traced graph. On this MetaX C550 the probe
reports supported and the fused path still takes `1008` of the `2034` RMSNorm
calls.

### Combined Result

Full matrix after both fixes (`artifacts/bench-tp1-*-rmsfused-20260904`),
`12/12` valid. Every InfiniCore case installed all nine routes with zero skips
and zero native fallbacks, reported `vllm_metax_loaded=false`, had nonzero
counters for every backend route family, and produced exactly the requested
output token count. The recorded output token previews are **identical to the
pre-change baseline for all six model/shape pairs**.

| Shape | Model | Baseline | Both fixes | Ratio then -> now | Gain |
|---|---|---:|---:|---|---:|
| `in256/out2048` | Qwen3-0.6B | 662.2 | 834.6 | 62.6% -> 77.4% | `+26.0%` |
| `in256/out2048` | Qwen3-4B-Instruct-2507 | 458.9 | 531.4 | 74.6% -> 85.4% | `+15.8%` |
| `in256/out2048` | Meta-Llama-3-8B-Instruct | 375.1 | 405.7 | 80.4% -> 86.0% | `+8.2%` |
| `in4096/out256` | Qwen3-0.6B | 477.6 | 523.1 | 64.3% -> 66.9% | `+9.5%` |
| `in4096/out256` | Qwen3-4B-Instruct-2507 | 248.6 | 269.7 | 67.7% -> 72.7% | `+8.5%` |
| `in4096/out256` | Meta-Llama-3-8B-Instruct | 220.5 | 240.8 | 77.2% -> 83.6% | `+9.2%` |

The two fixes are complementary. The bridge target cache only pays off where the
host is the limiter, so it is worth `+19.8%` on Qwen3-0.6B and nothing on
Llama-3-8B, and it does not help the prefill-heavy shape. The RMSNorm fusion
pays off everywhere, including `in4096/out256`.

Measurement caveat: the Qwen3-0.6B `in256/out2048` pair is the only noisy cell.
Its InfiniCore iterations were `859.8 / 815.9 / 829.4` (stdev `22.51`) and its
metax denominator drifts between `1016` and `1085` across runs, so the isolated
A/B put the same build at `81.3%` while the matrix run put it at `77.4%`. The
InfiniCore absolute figure is stable at `826-835`. Every other cell has stdev
under `11`.

### Rejected Variant

Routing the fused RMSNorm through the C++ bridge instead of the InfiniCore
stream path was measured and rejected: Llama-3-8B `401.91` versus `403.63`, and
Qwen3-0.6B `825.01` at stdev `19.68` versus `826.15` at stdev `4.79`.
`add_rms_norm_current_stream` creates and destroys its descriptor on every call,
and after fusion RMSNorm runs only 57 times per step, so the descriptor cost
exceeds the stream-handoff cost it removes. This is the opposite of the
StoreKV result, where the bridge won because that route ran `43008` times per
benchmark shape. The bridge entry point is kept as an opt-in
(`--cpp-bridge-routes ...,RMSNorm`) but is not a default route.

### Separate Counters For The Two RMSNorm Ops

The fused op initially reported into the shared `rms_norm` backend counter. That
hid the very thing this pass was about: these counters are the only evidence
that a route reaches the hot path, and the RMSNorm route had been installed with
nonzero counts while missing 98% of its own calls. `fused_add_rms_norm` now has
its own counter. Both ops stay on the `RMSNorm` route, so ablation and
`VLLM_INFINICORE_DISABLED_ROUTES` semantics are unchanged. A Qwen3-0.6B decode
window now reports `rms_norm=1026` against `fused_add_rms_norm=1008`, where
before it reported a single conflated `2034`.

### Falling Back When A Device Has No Fused Kernel

`infiniopAddRMSNorm` is not registered for every backend that registers plain
`infiniopRMSNorm`. Checking the dispatch tables in the InfiniCore source:

| Op | Backends |
|---|---|
| `rms_norm` | ALI CAMBRICON CPU HYGON ILUVATAR KUNLUN METAX MOORE NVIDIA QY **ASCEND** |
| `add_rms_norm` | ALI CAMBRICON CPU HYGON ILUVATAR KUNLUN METAX MOORE NVIDIA QY |

MetaX and Moore are both covered, so this host and the MUSA work are unaffected,
but Ascend would have hit a descriptor-creation failure. Routing the residual
path introduced that failure mode, since before this pass the residual path
never reached InfiniCore at all.

A device with no kernel is a capability fact, not a failure, so it must not fail
a strict run. `add_rms_norm_supported()` in the bridge creates and destroys the
descriptor only -- no workspace, no launch -- and returns false for
`INFINI_STATUS_NOT_IMPLEMENTED`, `DEVICE_TYPE_NOT_SUPPORTED` and
`DEVICE_ARCHITECTURE_NOT_SUPPORTED`, while rethrowing anything else so a genuine
failure is never mistaken for a missing capability. The result is cached for the
process.

The probe lives inside the custom op, not in `_should_use_infinicore()`. The
latter runs under torch.compile tracing on fake tensors, and its branch is baked
into the graph before any device call happens; probing there would be both
impossible and too late. Probing inside the opaque op means an unsupported
device changes only the op body, never the traced graph.

Verified on this host: the probe returns supported, and unit tests cover both a
supported device (probed once, keeps routing) and an unsupported one (probed
once, falls back, does not raise under strict mode).

### Single-Token Decode Threshold

`reorder_batch_threshold` was `128`, so any prefill of 128 tokens or fewer was
classified as decode and the strict wrapper rejected the step. It is now `1`,
matching the installed vLLM-MetaX 0.22 builder and the correct non-speculative
semantics: a decode step contributes exactly one query token per request.

This was not only an offline benchmark limit. It made the plugin unusable for
serving. Against `vllm serve` on Qwen3-0.6B, a single short request happened to
survive, but eight concurrent `hello`-sized requests crashed the engine core
with `unsupported_spec_decode:64!=8` and took the API server down with it --
the most ordinary traffic a deployment sees.

| Case | Before | After |
|---|---|---|
| 1 short prompt (9 tokens) | OK | OK |
| 1 long prompt (140 tokens) | OK | OK |
| 8 concurrent short prompts | 8/8 fail, engine dies | 8/8 OK |
| 16 concurrent short prompts | connection refused | 16/16 OK |

Zero `attention skipped` lines in the server log after the fix. The offline
paths are unaffected: the route self-check still installs all nine routes with
no skips or fallbacks, and the suite is unchanged.

Note on running the suite: it needs `VLLM_PLUGINS` set. With it unset only 80
tests run and four platform/registration cases error out. Run it through a
backend wrapper (`./run-infinicore.sh python -m unittest discover -s tests`) to
get the full 90. This is pre-existing behaviour, confirmed by reproducing the
same 80/4 result with the threshold reverted.

### Deployment Wrappers

`~/infini-vllm` on the single-card host holds `InfiniCore/`, `infini/`
(`INFINI_ROOT`), `vllm-infinicore/`, a shared `env.sh`, and two wrappers that
set the backend and then `exec "$@"`:

- `run-infinicore.sh` -- `VLLM_PLUGINS=infinicore,vllm_infinicore` plus the
  strict route settings, every value overridable from the caller.
- `run-metax.sh` -- `VLLM_PLUGINS=metax`, and it actively unsets every
  `VLLM_INFINICORE_*` variable so a stray value cannot quietly turn a baseline
  into a mixed run.

Verified with one unmodified user script: the InfiniCore wrapper reports
`InfiniCorePlatform` with `vllm_metax` not loaded, the MetaX wrapper reports
`MxsmlMacaPlatform` with it loaded. Note that `vllm chat` and `vllm complete`
are HTTP clients, so prefixing them changes nothing; the backend is fixed when
the server starts.

### Measurement Noise And Which Metric To Trust

The short profiler window in `scripts/decode_gap_profile.py` carries far more
run-to-run noise on end-to-end timing than on device time. Three consecutive
repeats of the same build at `bs=8`, `in=256`, 64 measured decode steps:

| Repeat | Decode step | Device kernel | Host CPU | GPU idle |
|---|---:|---:|---:|---:|
| 1 | `9.864` ms | `463.54` ms | `834.93` ms | `800.05` ms |
| 2 | `10.485` ms | `464.35` ms | `846.70` ms | `817.88` ms |
| 3 | `9.090` ms | `464.21` ms | `772.04` ms | `737.19` ms |
| Spread | `+-7%` | `+-0.09%` | `+-4.8%` | `+-5.2%` |

This is not GPU frequency. Kernel time holding to `0.09%` across the three runs
rules that out: the same kernel sequence at a varying clock could not land
within half a millisecond of `464`. `mx-smi` reports `Idle` as the only active
throttle reason, with power, temperature and application limits all inactive,
and this card exposes no clock lock at all -- only `--set-persistence-mode` and
`--set-power-mode`, with no equivalent of `nvidia-smi -lgc`. The host CPU
governor is `performance`, pinned at `3000` MHz.

The variance sits entirely in host CPU time and in GPU idle time, and the two
move together. Within a single process the timings are tight (repeat 3 measured
`0.952 / 0.959 / 0.953` s) while the offset between processes is large, which is
process placement rather than frequency: the container has a `16` CPU quota
(`1600000/100000`) on a shared 192-core Xeon 8558, so which physical cores and
NUMA node a launch lands on varies. At this shape GPU busy is only about `36%`,
so the decode loop is host-bound and any scheduling difference shows up directly
in wall time.

Practical consequence for future passes:

- Judging a host-side optimization needs the full benchmark harness with
  repeats, not this short window. The bridge target cache was worth `+0.7%` on
  Llama-3-8B, which this window cannot resolve at all.
- The short window is reliable for structural evidence -- device kernel time,
  GPU busy fraction, device event counts, and per-op host self time. Those are
  what localized the gap in this entry, and the C++ descriptor refactor was
  cleared on kernel time precisely because it is the metric this noise does not
  touch.
- `taskset` would cut the placement component if a host-side measurement ever
  has to be made with this script.

### Remaining Gap

After both fixes the largest remaining item is device-side glue. On Llama-3-8B,
InfiniCore issues `210312` device events against MetaX's `158503`, and nearly
all of the excess is `elementwise_kernel` work: `376.50` ms over `138496` calls
against `208.84` ms over `77886`. MetaX also still fuses the QKV split through
`MACA_CatArrayBatchedCopy` (`65.06` ms) where InfiniCore emits separate
elementwise copies. Reducing cast and contiguity churn in InfiniCore tensor view
construction is the next target. `unified_kv_cache_update` also remains around
`3.3x` MetaX's host cost, and that cost is in torch C++ dispatch rather than in
any Python frame.

## 2026-09-01 Qwen3-0.6B Bring-Up On MACA 3.8

Configured the remote MetaX C550 host
`ssh.v5000-prod-gw.nhss.zhejianglab.com:30278` with upstream InfiniCore and
installed this plugin from `/root/vllm-infinicore` in editable mode. The
validated runtime differs from the original MACA 3.5.3 development target:

- MACA `3.8.0.23`, Python `3.10.10`
- PyTorch `2.10.0+metax3.8.0.7`
- vLLM `0.22.0` and matching `vllm-metax`
- FlashAttention `2.6.3+metax3.8.0.7torch2.10`
- InfiniCore upstream commit `35b46277bd666772c11bb417ad4231c5be492822`
- Model `/mnt/geogpt-doc-new/default/infinilm-models/Qwen3-0.6B`

Installed the missing OpenMPI runtime and build prerequisites. Upstream
InfiniCore was configured with `--metax-gpu=y --use-mc=y --aten=y` and linked
to the installed MetaX FlashAttention extension. MACA 3.8's dense and varlen
FlashAttention exports add both a Mars workspace tensor and a trailing boolean,
while its KV-cache export retains the older signature. The upstream checkout
therefore has a local three-file compatibility patch guarded by
`INFINICORE_METAX_FLASHATTN_38_ABI`; it is built with
`--cxxflags=-DINFINICORE_METAX_FLASHATTN_38_ABI`. Do not discard these remote
changes when updating `/root/InfiniCore`.

Validation:

- `import infinicore` succeeds with the configured runtime environment.
- Upstream MetaX RMSNorm suite passed `108/108` cases.
- Native vLLM MetaX baseline generated exactly eight tokens from Qwen3-0.6B.
- Plugin strict-backend eager smoke registered all nine routes with no skips or
  native fallbacks and generated exactly eight tokens, ending in
  `VLLM_SMOKE_OK`.
- A separate two-token execution probe reported nonzero backend calls for every
  scoped route family: `embedding=4`, `rms_norm=228`, `linear=448`,
  `rotary_embedding=112`, `silu_and_mul=112`, `lm_head=4`,
  `store_kv_cache=84`, `paged_attention_prefill=28`, and
  `paged_attention_decode=28`. C++ bridge counts were `MatMul=452` and
  `PagedAttentionDecodeFlash=28`.

The smoke used `VLLM_PLUGINS=metax,vllm_infinicore`, strict backend mode, and
`enforce_eager=True`; it is a correctness bring-up, not a cudagraph or
performance claim. The conservative persistent profile at
`/etc/profile.d/infinicore.sh` leaves plugin patches disabled by default.

The same Qwen3-0.6B two-token smoke also passed without loading `vllm-metax`:
`VLLM_PLUGINS=infinicore,vllm_infinicore`,
`VLLM_SMOKE_FORBID_METAX_LOAD=1`, and `vllm_metax_loaded=false`. vLLM 0.22 must
also use `VLLM_USE_V2_MODEL_RUNNER=0`; otherwise its V2 warmup emits a
speculative-decode shape (`512` query tokens for `256` requests) that the
strict InfiniCore attention wrapper rejects. With V1 and
`VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA=1`, attention is forced through the
InfiniCore path rather than the direct FlashAttention compatibility shortcut.
All route-family counters were nonzero, including
`paged_attention_prefill=28`, `paged_attention_decode=28`, and bridge counter
`PagedAttentionDecodeFlash=28`, and the run ended in `VLLM_SMOKE_OK`.

Strict no-`vllm_metax` PIECEWISE cudagraph validation also passed on this
stack. The run used `backend="eager"`, `enforce_eager=False`, capture sizes
`[1, 2, 4, 8]`, `VLLM_USE_V2_MODEL_RUNNER=0`, and
`VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA=1`. Evidence:

- Graph artifact:
  `/root/vllm-infinicore/artifacts/qwen3-06b-no-metax-infinicore-cudagraph-20260901.json`
- Eager reference artifact:
  `/root/vllm-infinicore/artifacts/qwen3-06b-no-metax-infinicore-eager-reference-20260901.json`
- Shape `input_len=128`, `output_len=32`, one warmup and one measured request.
- `valid=true`, `validation_errors=[]`, `graph_capture_count=116`, and
  `vllm_metax_loaded=false`.
- All nine route families had nonzero measured-call evidence. Attention counts
  included `paged_attention_prefill=28`, `paged_attention_decode=868`, and
  bridge `PagedAttentionDecodeFlash=868`.
- The graph and eager reference produced the exact same 32 output token IDs.
- Text-health counters reported zero control and replacement characters.

This is graph-safety and exact-token correctness evidence only. The single
measured request is not a formal performance comparison.

Formal Qwen3-0.6B CUDA Graph comparison was then run at `batch_size=8`,
`input_len=2048`, `output_len=512`, BF16, TP=1, one warmup, and three measured
iterations. Both cases reused prompt token IDs with SHA-256
`c77a630e4d4f5449b67d49876d9a4cf4d94872a3dc12ede238b41ccc5c12e141`
and used deterministic sampling with EOS disabled. To keep the scheduler shape
inside the current strict InfiniCore attention contract, both cases used
`async_scheduling=False`, `enable_chunked_prefill=False`, and
`max_num_batched_tokens=16384`. The benchmark script now exposes and records
these scheduler controls.

| Plugin set | Output TPS | Median | Min | Max | Graph captures |
|---|---:|---:|---:|---:|---:|
| `VLLM_PLUGINS=metax` | 738.99 | 738.95 | 738.15 | 739.87 | 116 |
| `VLLM_PLUGINS=infinicore,vllm_infinicore` | 433.64 | 439.21 | 415.82 | 447.13 | 116 |

The strict no-MetaX result reached `58.68%` of the same-run vLLM-MetaX
throughput (`41.32%` lower, or vLLM-MetaX was `1.70x` as fast). Both results
were valid: every request generated exactly 512 output tokens, all three
iterations were text-healthy, and the recorded 64-token output previews were
identical across engines and repeats. The no-MetaX worker reported
`vllm_metax_loaded=false`, installed all nine scoped routes with no skips or
native fallbacks, and recorded nonzero backend calls for every route family.

Artifact:
`/root/vllm-infinicore/artifacts/qwen3-06b-metax-vs-no-metax-infinicore-bs8-in2048-out512-graph-synced-scheduler-20260901`.

Known scheduler limitation: with vLLM 0.22 defaults at the same `bs=8/in=2048`
shape, the MetaX case reached `962.42` output tok/s, but strict no-MetaX
InfiniCore rejected the warmup as
`unsupported_spec_decode:11!=8`. This was a mixed scheduled-token step from
chunked prefill under the default `8192` token budget, not configured
speculative decoding. Disabling async scheduling alone reproduced the same
failure, while disabling chunked prefill with `max_num_batched_tokens=16384`
passed with async scheduling enabled. The immediate plugin bug is the strict
metadata builder's stale `reorder_batch_threshold=128`: it classifies the
mixed query lengths `[1,1,1,1,1,1,1,4]` as eight decode requests and then
rejects `11` decode tokens for eight requests. The installed vLLM-MetaX 0.22
builder uses threshold `1` and correctly leaves the four-token tail in the
prefill partition. Do not use the failed attempt as an InfiniCore throughput
result.

Single-GPU multi-model CUDA Graph comparison with chunked prefill disabled:

- One MetaX C550, TP=1, BF16, `batch_size=8`, `input_len=2048`,
  `output_len=512`, one warmup, and three measured iterations.
- PIECEWISE CUDA Graph with capture sizes `[1,2,4,8]`, `backend="eager"`,
  and `enforce_eager=False`.
- `enable_chunked_prefill=False`, `max_num_batched_tokens=16384`; async
  scheduling remained enabled for both engines.
- Every same-model pair reused the exact same prompt token IDs and deterministic
  sampling configuration.

| Model | vLLM-MetaX TPS | strict no-MetaX InfiniCore TPS | InfiniCore / MetaX | Gap | Graph captures |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 1080.83 | 483.74 | 44.76% | -55.24% | 100 / 100 |
| Qwen3-0.6B | 961.42 | 433.03 | 45.04% | -54.96% | 116 / 116 |
| Qwen3-4B-Instruct-2507 | 549.26 | 329.89 | 60.06% | -39.94% | 148 / 148 |
| DeepSeek-R1-Distill-Qwen-7B | 470.24 | 323.40 | 68.77% | -31.23% | 116 / 116 |

All eight engine results reported `valid=True` and `validation_errors=[]`.
Every request generated exactly 512 output tokens, text-health checks passed,
and the recorded output token previews matched across engines and repeats for
each model. Each strict InfiniCore worker reported
`vllm_metax_loaded=false`, installed all nine scoped routes with no skips or
native fallbacks, and recorded nonzero calls for every backend route family.

Artifacts:

- `/root/vllm-infinicore/artifacts/single-gpu-cudagraph-no-chunked-qwen25-05b-bs8-in2048-out512-20260901`
- `/root/vllm-infinicore/artifacts/single-gpu-cudagraph-no-chunked-qwen3-06b-bs8-in2048-out512-20260901`
- `/root/vllm-infinicore/artifacts/single-gpu-cudagraph-no-chunked-qwen3-4b-bs8-in2048-out512-20260901`
- `/root/vllm-infinicore/artifacts/single-gpu-cudagraph-no-chunked-deepseek-qwen7b-bs8-in2048-out512-20260901`

The gap narrows as model compute grows in this tested set: strict InfiniCore is
roughly `45%` of MetaX throughput on the two sub-billion models and `69%` on
the 7B model. This is consistent with fixed per-step Python/backend/stream
handoff costs having a larger relative impact on small models, but profiling
is still required before assigning the remaining gap to a specific route.

### DeepSeek-Qwen-7B Single-Route Ablation

Ran one-at-a-time route ablation on DeepSeek-R1-Distill-Qwen-7B with the same
single-GPU graph shape: TP=1, BF16, `bs=8`, `input_len=2048`,
`output_len=512`, one warmup, three repeats, chunked prefill disabled,
`max_num_batched_tokens=16384`, and async scheduling enabled.

The strict no-MetaX matrix used an in-run native baseline of `470.83` output
tok/s and an all-route baseline of `317.33` output tok/s. The six
non-attention switches were valid:

| Disabled route | Output TPS | Delta vs `all` | Relative delta | Gap recovered |
|---|---:|---:|---:|---:|
| none (`all`) | 317.33 | 0.00 | 0.00% | 0.00% |
| `RMSNorm` | 314.78 | -2.55 | -0.80% | -1.66% |
| `SiluAndMul` | 326.02 | +8.68 | +2.74% | 5.66% |
| `RoPE` | 320.05 | +2.72 | +0.86% | 1.77% |
| `Embedding` | 321.85 | +4.52 | +1.42% | 2.94% |
| `MatMul` | 328.92 | +11.59 | +3.65% | 7.55% |
| `LMHead` | 324.88 | +7.55 | +2.38% | 4.92% |

`StoreKVCache`, `PagedAttentionPrefill`, and `PagedAttentionDecode` cannot be
disabled into a valid base-vLLM fallback on this no-MetaX stack. The installed
base backend lacks `reshape_and_cache_flash` / `flash_attn_varlen_func`, so
these attempts fail before producing throughput. They were not treated as
ablation data.

The three attention/KV routes were therefore measured in a separate diagnostic
matrix with `VLLM_PLUGINS=metax,vllm_infinicore`, where only the disabled route
falls back to the MetaX implementation. Its own all-route baseline was
`351.60` output tok/s:

| Disabled route | Output TPS | Delta vs diagnostic `all` | Relative delta | Diagnostic gap recovered |
|---|---:|---:|---:|---:|
| none (`all`) | 351.60 | 0.00 | 0.00% | 0.00% |
| `StoreKVCache` | 395.19 | +43.59 | +12.40% | 36.56% |
| `PagedAttentionPrefill` | 355.41 | +3.81 | +1.08% | 3.19% |
| `PagedAttentionDecode` | 343.72 | -7.88 | -2.24% | -6.61% |

The diagnostic all-route case still installed and executed all nine InfiniCore
routes. Loading the MetaX platform/metadata context raised the all-route result
from `317.33` to `351.60` output tok/s (`+10.80%`) even before disabling a
route, so that context effect must remain separate from strict no-MetaX route
attribution.

StoreKV root-cause confirmation used the existing plugin-owned current-stream
C++ bridge while retaining all nine InfiniCore routes and no `vllm_metax`
modules. Adding `StoreKVCache` to the then-default
`PagedAttentionDecodeFlash,MatMul` bridge set produced `366.14` output tok/s:

- `+48.81` output tok/s / `+15.38%` versus strict no-MetaX all routes.
- `77.77%` of the same-run `470.83` native baseline.
- `graph_capture_count=116`, `validation_errors=[]`, exact 512 output tokens
  per request, healthy text, and stable token previews.
- C++ bridge count `StoreKVCache=43008`; every InfiniCore route-family counter
  remained nonzero with no route fallback.

The previous Python StoreKV implementation calls `infinicore.paged_caching` through
the InfiniCore external stream and joins the PyTorch stream before and after
every launch. At this benchmark shape StoreKV ran `43008` times. The
current-stream bridge removes that repeated stream handoff while preserving
the InfiniCore `infiniopPagedCaching` kernel, so the A/B identifies the primary
bottleneck as the StoreKV Python/external-stream boundary, not the paged-caching
math alone. `MatMul`, `SiluAndMul`, and `LMHead` are secondary candidates;
sub-2% deltas are close to run variability and should not be overinterpreted.

Artifacts:

- `/root/vllm-infinicore/artifacts/single-gpu-cudagraph-no-chunked-deepseek7b-single-route-ablation-no-metax-20260901`
- `/root/vllm-infinicore/artifacts/single-gpu-cudagraph-no-chunked-deepseek7b-attention-route-ablation-metax-fallback-20260901`
- `/root/vllm-infinicore/artifacts/single-gpu-cudagraph-no-chunked-deepseek7b-storekv-current-stream-bridge-20260901`

The minimal production change makes `StoreKVCache` part of the existing
current-stream C++ bridge default; no alternative StoreKV implementation was
added. A fresh same-run comparison after syncing the local source measured
`366.44` output tok/s for strict no-MetaX InfiniCore versus `469.60` output
tok/s for vLLM-MetaX. InfiniCore reached `78.03%` of native throughput, a
`21.97%` gap. Both results were valid with 116 graph captures and exactly 512
output tokens per request. The InfiniCore worker reported
`vllm_metax_loaded=false`, default bridge routes
`PagedAttentionDecodeFlash,MatMul,StoreKVCache`, and
`StoreKVCache=43008` bridge calls.

Artifact:

- `/root/vllm-infinicore/artifacts/single-gpu-cudagraph-no-chunked-deepseek7b-default-storekv-current-stream-vs-metax-final-20260902`

## 2026-06-05 Strict InfiniCore FA Metadata Builder Port

Root-caused the remaining Qwen2.5 TP=1 throughput gap after the direct FA2
forward port. The issue was not the final FA2 adaptor call and not the
non-attention InfiniCore routes. The strict no-`vllm_metax` backend was still
using vLLM's native FlashAttention metadata builder/base initialization, while
the fast diagnostic MetaX path used MetaX's FA metadata builder.

Target shape: Qwen2.5-0.5B-Instruct, TP=1, `input_len=2048`,
`output_len=1024`, `warmup=1`, `repeats=1`.

| Case | Output TPS | Notes | Artifact |
|---|---:|---|---|
| InfiniCore forward-only MetaX-compatible FA | 109.15 | strict no-`vllm_metax`; old builder | `/root/vllm-infinicore/artifacts/xzh-53-qwen25-metax-compat-20260605-105848` |
| Diagnostic MetaX FA base/builder | 183.59 | InfiniCore platform; `vllm_metax_loaded=true`; diagnostic only | `/root/vllm-infinicore/artifacts/xzh-53-qwen25-diag-metax-fa-base-20260605-1133` |
| Strict InfiniCore builder port | 170.37 | `vllm_metax_loaded=false`, `--forbid-metax-load` | `/root/vllm-infinicore/artifacts/xzh-53-qwen25-infinicore-fa2compat-strict-20260605-1146` |
| Current vLLM MetaX baseline | 146.50 | `VLLM_PLUGINS=metax` | `/root/vllm-infinicore/artifacts/xzh-53-qwen25-metax-baseline-current-20260605-1148` |

The strict InfiniCore result is `116.3%` of the current vLLM MetaX baseline
and `56.1%` faster than the previous strict forward-only port. It reported
`valid=True`, `validation_errors=[]`, `vllm_metax_loaded=false`,
`VLLM_PLUGINS=infinicore,vllm_infinicore`, all nine scoped routes installed,
`backend_decode_metax_compatible=24552`, and C++ bridge counter `MatMul=72`.

Fix:

- Added `InfiniCoreFlashAttentionMetadataBuilder`, which keeps vLLM's native
  builder as the base but ports MetaX's `reorder_batch_threshold=128` decode /
  prefill split metadata into the strict backend.
- Recomputes split metadata after `update_block_table()` so cudagraph replay
  uses the current block table slices.
- Locally patches vLLM native FA version/support probes for this backend to
  MetaX-compatible FA2 semantics (`version=2`, no FP8, sinks supported) without
  importing `vllm_metax`.
- Kept the strict runtime on
  `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`;
  no `MacaFlashAttentionBackend` is registered in the strict path.

## 2026-06-05 MetaX-Compatible InfiniCore FA Backend Port

Ported the MetaX FlashAttention backend's prefill/decode split structure into
`vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
as a no-`vllm_metax` custom backend path:

- The plugin still registers
  `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
  as vLLM's `FLASH_ATTN` backend.
- The runtime still uses `VLLM_PLUGINS=infinicore,vllm_infinicore`; no
  `vllm_metax` module is loaded.
- `StoreKVCache` remains routed through the InfiniCore KV-cache update path.
- Prefill/decode forward now mirrors MetaX's direct
  `flash_attn_varlen_func` / `flash_attn_with_kvcache` split when
  `VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA` is unset.
- `VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA=1` disables the new path for A/B
  fallback to the previous bridge-backed decode path.

Remote Qwen2.5-0.5B-Instruct TP=1 target shape,
`input_len=2048`, `output_len=1024`, `warmup=1`, `repeats=1`:

| Path | Output TPS | Ratio vs MetaX TP=1 baseline | Artifact |
|---|---:|---:|---|
| vLLM MetaX baseline | 145.65 | 100.0% | `artifacts/xzh-53-metax-bs1-in2048-out1024-repeats1-20260604-172756/qwen25-05b-tp1` |
| InfiniCore MetaX-compatible backend | 109.15 | 74.9% | remote `/root/vllm-infinicore/artifacts/xzh-53-qwen25-metax-compat-20260605-105848` |
| InfiniCore bridge fallback (`VLLM_INFINICORE_DISABLE_METAX_COMPAT_FA=1`) | 93.90 | 64.5% | remote `/root/vllm-infinicore/artifacts/xzh-53-qwen25-metax-compat-disabled-20260605-110053` |

The enabled run reported `valid=True`, `validation_errors=[]`,
`vllm_metax_loaded=false`,
`vllm_attention_backend=vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`,
all nine scoped routes installed with no native fallbacks,
`backend_prefill_metax_compatible=24`,
`backend_decode_metax_compatible=24552`, and C++ bridge counter
`MatMul=72` with no decode bridge calls.

This confirms the vLLM-side custom FA backend port improves TP=1 throughput,
but it still does not match MetaX's native backend. The remaining gap is now
inside the direct FA2/Mars backend integration details around the custom
backend and non-attention route mix, not the old C++ decode bridge.

## 2026-06-05 Flash Decode Split Sweep And Compliance Finding

Added a diagnostic `VLLM_INFINICORE_FLASH_DECODE_NUM_SPLITS` knob for the
`PagedAttentionDecodeFlash` C++ bridge. The default remains `0`, matching the
FlashAttention heuristic used by the MetaX Python wrapper.

Remote Qwen2.5-0.5B-Instruct TP=1 target shape,
`input_len=2048`, `output_len=1024`, `warmup=1`, `repeats=1`,
strict no-`vllm_metax`:

| `num_splits` | Output TPS | Validation |
|---:|---:|---|
| 0 | 104.43 | `valid=True` |
| 1 | 74.50 | `valid=True` |
| 2 | 82.41 | `valid=True` |
| 4 | 95.88 | `valid=True` |
| 8 | 92.82 | `valid=True` |

The split sweep did not explain the gap to the MetaX baseline; the heuristic
`0` is still the best tested value. A diagnostic no-`vllm_metax` Python
`flash_attn_with_kvcache` wrapper path measured `97.93` output tok/s, slower
than the C++ bridge, so the remaining TP=1 gap is not caused by the bridge
calling convention alone.

Final same-shape validation after removing the diagnostic Python wrapper path:

| Engine | Output TPS | Ratio vs MetaX TP=1 baseline | Artifact |
|---|---:|---:|---|
| vLLM MetaX baseline | 145.65 | 100.0% | `artifacts/xzh-53-metax-bs1-in2048-out1024-repeats1-20260604-172756/qwen25-05b-tp1` |
| vLLM-InfiniCore strict no-`vllm_metax` | 100.73 | 69.2% | remote `/root/vllm-infinicore/artifacts/xzh-53-qwen25-final-split0-20260605-103953` |

The final run reported `valid=True`, `validation_errors=[]`,
`VLLM_PLUGINS=infinicore,vllm_infinicore`, `vllm_metax_loaded=false`,
`vllm_attention_backend=vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`,
all nine scoped routes installed with no native fallbacks, and C++ bridge
counters `PagedAttentionDecodeFlash=24552`, `MatMul=72`.

Compliance finding:

- The runtime does not load `vllm_metax` and does not register
  `MacaFlashAttentionBackend`.
- However, `libinfinicore_cpp_api.so` on the remote machine has a dynamic
  dependency on
  `/opt/conda/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`,
  and `mha_fwd_kvcache` is resolved from that library. The current fast decode
  path is therefore not yet a pure standalone InfiniCore FlashAttention kernel
  implementation.
- `infinicore::op::flash_attention` / `infiniopFlashAttention` exists, but it
  is the non-paged FlashAttention interface and does not directly replace vLLM
  paged KV-cache decode. `infinicore::op::mha_kvcache_` is the paged decode
  interface, and its current implementation wraps the same FlashAttention
  KV-cache adaptor.

## 2026-06-05 TP=1 MatMul Default And GQA Direct-Out Rejection

Retained the strict no-MetaX default C++ bridge route set as
`PagedAttentionDecodeFlash,MatMul`:

- `MatMul` now uses the C++ current-stream `infiniopGemm` bridge when no bias
  is present.
- C++ bridge route selection is cached by the relevant environment values, so
  high-frequency route checks no longer parse environment strings per call.
- The Flash decode path no longer imports `infinicore` on the hot C++ bridge
  path before checking route selection.

Current remote verification for Qwen2.5-0.5B-Instruct TP=1 at
`input_len=2048`, `output_len=1024`, `warmup=1`, `repeats=1`:

| Engine | Output TPS | Ratio vs MetaX TP=1 baseline | Artifact |
|---|---:|---:|---|
| vLLM MetaX baseline | 145.65 | 100.0% | `artifacts/xzh-53-metax-bs1-in2048-out1024-repeats1-20260604-172756/qwen25-05b-tp1` |
| vLLM-InfiniCore default | 102.14 | 70.1% | `artifacts/xzh-53-qwen25-clean-default-20260605-011125` |

The InfiniCore run reported `valid=True`, `validation_errors=[]`,
`vllm_metax_loaded=false`,
`vllm_attention_backend=vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`,
`cpp_bridge_routes=PagedAttentionDecodeFlash,MatMul`, and bridge counters
`PagedAttentionDecodeFlash=24552`, `MatMul=72`.

Rejected experiment:

- Passing a direct GQA-shaped output buffer to `mha_fwd_kvcache` can satisfy
  the FlashAttention shape check, but the full `2048/1024` generation failed
  text-health validation with garbled output.
- Therefore the default Flash decode route keeps the correct dynamic output
  plus copy-back path for GQA. The remaining TP=1 gap is still in the decode
  attention backend/kernel boundary, not a route-selection bug.
- A pure `infiniopPagedAttention` current-stream decode bridge was also tested
  as an explicit route. It was correct but much slower at the target Qwen2.5
  TP=1 shape: `14.23` output tok/s for `2048/1024` in
  `artifacts/xzh-53-qwen25-infiniop-pa-full-20260605-012004`. It is not
  retained as a route.

## 2026-06-04 Strict No-MetaX TP=1 Flash Decode Bridge

Fixed the strict no-MetaX TP=1 decode bottleneck by making
`PagedAttentionDecode` use an InfiniCore FlashAttention C++ bridge on the
current vLLM stream by default:

- Added `PagedAttentionDecodeFlash` to the C++ bridge route set.
- The new bridge calls the InfiniCore-vendored FlashAttention adaptor directly
  from `vllm_infinicore`, without importing `vllm_metax` or registering
  `MacaFlashAttentionBackend`.
- The previous default bridge route remains available as explicit
  `VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecode` for A/B tests.
- Default C++ bridge routes are now `PagedAttentionDecodeFlash,MatMul`.

Root cause:

- Strict no-MetaX runs were already using
  `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`,
  but decode went through InfiniCore `mha_kvcache_` on the InfiniCore external
  stream.
- Qwen2.5 TP=1 microbench at the 2k decode shape measured old bridged decode
  at roughly `0.12-0.14 ms` per layer call.
- The current-stream Flash decode bridge measured `0.059 ms` per layer call,
  removing most of the per-layer stream handoff overhead while keeping the
  no-`vllm_metax` runtime constraint.
- A global no-wait experiment was rejected: it triggered a device ATU fault in
  RoPE during graph capture, proving stream waits cannot be removed globally.

Strict remote verification used `VLLM_PLUGINS=infinicore,vllm_infinicore`,
`VLLM_INFINICORE_ROUTES=all`, `VLLM_INFINICORE_STRICT_BACKEND=1`, and
`--forbid-metax-load`:

| Model | TP | Previous strict no-MetaX TPS | New strict no-MetaX TPS | Validation |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B-Instruct | 1 | 76.32 | 94.00 | `validation_errors=[]`, `vllm_metax_loaded=false` |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 43.13 | 52.17 | `validation_errors=[]`, `vllm_metax_loaded=false` |

Both verified runs reported
`vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
and `cpp_bridge_routes=PagedAttentionDecodeFlash`. Artifacts:
`artifacts/xzh-53-qwen25-default-flashdecode-tp1-in2048-out1024-20260604-235557`
and
`artifacts/xzh-53-deepseek-default-flashdecode-tp1-in2048-out1024-20260604-235818`.

The earlier `VLLM_PLUGINS=metax,vllm_infinicore` throughput-profile experiment
that retained `MacaFlashAttentionBackend` is diagnostic only and is not an
acceptable solution for strict no-MetaX delivery.

## 2026-06-04 Stage Four No-MetaX Remote Smoke Hardening

Hardened `tests/remote/run_qwen_smoke.py` so no-MetaX is a first-class remote
smoke path rather than only a benchmark-harness mode:

- Removed module-import side effects; environment setup and plugin registration
  now happen inside `main()`, so the module is unit-testable.
- Added runtime bootstrap for MACA, InfiniCore, torch, and loader paths.
- Added a one-time `os.execvpe()` re-exec after setting `LD_LIBRARY_PATH` so
  `libinfinicore_cpp_api.so` is visible to the dynamic loader before importing
  InfiniCore wrappers.
- Added `VLLM_SMOKE_FORBID_METAX_LOAD=1`, which selects
  `VLLM_PLUGINS=infinicore,vllm_infinicore` by default and fails the smoke if
  `vllm_metax` is loaded locally or inside Ray workers.
- Added exact output-token validation with `min_tokens=max_tokens`,
  `ignore_eos=True`, `temperature=0.0`, `top_p=1.0`, and `top_k=1`.
- Ray smoke now propagates the runtime environment and checks worker-side
  `vllm_metax` load state through `collective_rpc`.

Remote validation:

- Single-card no-MetaX smoke:
  `MODEL=/mnt/geogpt-doc-new/default/xb/qwen3-8B`,
  `VLLM_SMOKE_FORBID_METAX_LOAD=1`, `VLLM_SMOKE_MAX_MODEL_LEN=128`,
  `VLLM_SMOKE_MAX_TOKENS=1`, `VLLM_SMOKE_ENFORCE_EAGER=1`:
  `VLLM_SMOKE_OK`, `OUTPUT_TOKEN_COUNT 1`, `vllm_metax_loaded False`.
- Two-card Ray no-MetaX smoke with `CUDA_VISIBLE_DEVICES=0,1`,
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`,
  `VLLM_TENSOR_PARALLEL_SIZE=2`, and
  `VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray`:
  `VLLM_SMOKE_OK`, `OUTPUT_TOKEN_COUNT 1`, `vllm_metax_loaded False`.

The smoke still logs vLLM's native FlashAttention/Triton probe errors on this
MACA stack, but the route registration, generation, exact token count, and
no-`vllm_metax` checks all pass.

## 2026-06-04 No-MetaX Qwen3 128/32 Stage Three

Extended `scripts/qwen3_128_32_smoke.py` so no-MetaX validation uses the same
prompt-token and measurement harness as the graph smoke:

- Added `no-metax-eager` and `no-metax-graph` cases.
- Added `--plugins` so the harness can run with
  `VLLM_PLUGINS=infinicore,vllm_infinicore` without being overwritten by the
  historical MetaX default.
- Added `--forbid-metax-load` / case-level validation that fails if any
  `vllm_metax` module is present in `sys.modules`.
- Artifacts now record the effective plugin environment, selected attention
  backend, `vllm_metax_loaded`, and InfiniCore backend/attention/bridge
  counters.

Remote validation on the MetaX C550 machine:

```bash
python scripts/qwen3_128_32_smoke.py \
  --trust-remote-code \
  --warmup 1 \
  --repeats 2 \
  --cases no-metax-eager,no-metax-graph \
  --output-json artifacts/qwen3_128_32_no_metax_stage3.json \
  --output-dir artifacts/qwen3_128_32_no_metax_stage3_cases
```

Result:

| Case | Validation | Graph captures | `vllm_metax_loaded` | Output TPS |
|---|---|---:|---|---:|
| `no-metax-eager` | `validation_errors=[]` | 0 | `False` | 12.46 |
| `no-metax-graph` | `validation_errors=[]` | 148 | `False` | 40.66 |

Both cases used `VLLM_PLUGINS=infinicore,vllm_infinicore`,
`VLLM_INFINICORE_ROUTES=all`, and
`VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0`. Both installed all nine scoped
routes: `RMSNorm`, `SiluAndMul`, `RoPE`, `Embedding`, `MatMul`, `LMHead`,
`StoreKVCache`, `PagedAttentionPrefill`, and `PagedAttentionDecode`.

Stage-three graph counters included nonzero InfiniCore calls for every scoped
route, `backend_prefill_infinicore=72`, `backend_decode_infinicore=2232`, and
`PagedAttentionDecode` C++ bridge calls `2232`. vLLM still logs native
FlashAttention probe failures on this MACA stack (`libcudart.so.12` missing),
but runtime attention used the InfiniCore backend and the strict no-MetaX
module-load check passed.

This closes the current single-card no-`vllm_metax` Qwen3 128/32 eager + graph
smoke target. Multi-card no-MetaX validation and larger throughput benchmarks
remain future work.

### Stage Three Graph-Safe Strict Check

Re-ran the no-MetaX graph case with strict backend validation enabled:

```bash
VLLM_INFINICORE_STRICT_BACKEND=1 \
python scripts/qwen3_128_32_smoke.py \
  --trust-remote-code \
  --warmup 1 \
  --repeats 3 \
  --cases no-metax-graph \
  --output-json artifacts/qwen3_128_32_no_metax_graphsafe_stage3.json \
  --output-dir artifacts/qwen3_128_32_no_metax_graphsafe_stage3_cases
```

Result:

- `valid=True`, `validation_errors=[]`
- `VLLM_PLUGINS=infinicore,vllm_infinicore`
- `vllm_metax_loaded=False`
- PIECEWISE graph with `backend="eager"` and `num_cudagraph_captured=148`
- three measured graph replays, each with exact `128` input tokens and `32`
  output tokens
- all nine scoped routes installed, with `native_fallback_routes=[]` and
  `skipped_routes=[]`
- graph counters included `store_kv_cache=108`,
  `paged_attention_prefill=108`, `paged_attention_decode=3348`,
  `backend_decode_infinicore=3348`, and C++ bridge
  `PagedAttentionDecode=3348`

This is the current graph-safe evidence for the single-card no-`vllm_metax`
Qwen3 128/32 path. It validates capture plus replay correctness for this shape;
it is not yet a multi-card or long-context graph-safety claim.

### Stage Three Coverage Benchmarks

Extended `scripts/qwen3_three_engine_throughput.py` so throughput runs can
exercise the no-MetaX platform path:

- Added `--vllm-infinicore-plugins` and `--vllm-native-plugins` to remove the
  historical hardcoded `VLLM_PLUGINS=metax,vllm_infinicore`.
- Added `--forbid-metax-load` so vLLM benchmark artifacts fail validation when
  `vllm_metax` is present in `sys.modules`.
- Artifacts now record the effective vLLM plugin environment, selected platform,
  `vllm_metax_loaded`, strict backend state, and route counters.
- Ray tensor-parallel runs now aggregate worker cudagraph counters through
  collective RPC; the TP=2 smoke reports `296` total captures, matching two
  workers with `148` captures each.

Remote no-MetaX coverage results:

| Coverage | Shape | TP/backend | Repeats | Graph captures | Output TPS | Validation |
|---|---|---|---:|---:|---:|---|
| Harness smoke | `bs=1,in=128,out=32` | `1` | 1 | 148 | 39.72 | `validation_errors=[]` |
| Multi-card | `bs=1,in=128,out=32` | `2,ray` | 1 | 296 | 37.86 | `validation_errors=[]` |
| Long context | `bs=1,in=4096,out=128` | `1` | 1 | 148 | 36.51 | `validation_errors=[]` |
| Large batch | `bs=8,in=1024,out=128` | `1` | 1 | 148 | 247.79 | `validation_errors=[]` |
| Formal throughput | `bs=8,in=4096,out=512` | `1` | 3 | 148 | 213.20 | `validation_errors=[]` |

All runs used `VLLM_PLUGINS=infinicore,vllm_infinicore`,
`VLLM_INFINICORE_ROUTES=all`, `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=0`, and
`VLLM_INFINICORE_STRICT_BACKEND=1`. Every run reported
`vllm_metax_loaded=False`, all nine scoped routes installed,
`native_fallback_routes=[]`, and `skipped_routes=[]`.

Formal throughput artifact:
`artifacts/no-metax-formal-throughput-bs8-in4096-out512-20260604`.
The formal run measured `12288` output tokens across three iterations. Per-run
TPS stats were mean `213.26`, median `213.83`, min `208.78`, max `217.16`, and
stdev `4.22`.

These runs expand the stage-three evidence from the 128/32 graph-safe smoke to
multi-card startup, long context, large batch, and the current formal
throughput shape. They are no-MetaX vLLM-InfiniCore coverage benchmarks, not a
new comparison against vLLM native or InfiniLM.

## 2026-06-04 No-MetaX Platform Attention Smoke

Added an experimental InfiniCore vLLM platform plugin entry point:

- `vllm.platform_plugins`: `infinicore = vllm_infinicore.platform:register_platform`
- `register_platform()` returns `vllm_infinicore.platform.InfiniCorePlatform`
- platform entry-point discovery stays lazy and does not import torch or vLLM
- platform initialization imports `mcoplib._C` / `mcoplib._moe_C` so vLLM native
  custom ops such as `_C.silu_and_mul` are registered without loading
  `vllm_metax`

The attention backend now respects the selected platform plugin:

- `VLLM_PLUGINS=metax,vllm_infinicore` keeps the existing MetaX-compatible path
  and prefers `vllm_metax.v1.attention.backends.flash_attn`.
- `VLLM_PLUGINS=infinicore,vllm_infinicore` skips importing `vllm_metax`.
- The InfiniCore platform path activates `StoreKVCache`,
  `PagedAttentionPrefill`, and `PagedAttentionDecode` from platform
  registration.
- No-MetaX attention normalizes vLLM native metadata into the decode/prefill
  fields required by the InfiniCore PA/KV wrappers.
- Profile/warmup calls that omit an output buffer or use an invalid temporary KV
  cache return zero-filled profile output rather than falling back to native
  FlashAttention.

Remote validation on the MetaX C550 machine:

- `python -m unittest discover -s tests`: `41` tests passed with `3` skipped.
- `VLLM_PLUGINS=infinicore` pre-register selected
  `vllm_infinicore.platform.InfiniCorePlatform`, activated the three attention
  routes, and reported `vllm_metax_loaded=False`.
- `VLLM_PLUGINS=metax,vllm_infinicore` still selected the MetaX
  FlashAttention base backend and reported `vllm_metax_loaded=True`.
- No-MetaX eager LLM smoke with Qwen3-8B, `max_model_len=128`,
  `max_tokens=1`, and `enforce_eager=True` generated one token and reported
  `NO_METAX_WITH_GENERAL_PLUGIN_SMOKE_OK`.

Smoke counters for the no-MetaX generated request:

| Counter group | Counts |
|---|---|
| InfiniCore backend | `store_kv_cache=36`, `paged_attention_decode=36` |
| Attention backend | `backend_kv_update_infinicore=36`, `backend_decode_infinicore=36`, `backend_forward_infinicore=36` |

The smoke also reported `vllm_metax_loaded=False`. This is a correctness and
runtime-independence smoke, not a graph-safety or throughput benchmark. Graph
mode, multi-card, and full all-operator no-MetaX benchmarks remain future
validation work.

## 2026-06-03 Single-GPU Decode Bridge Default

Re-tested the current Qwen3-4B single-GPU production-debug shape with
`batch_size=8`, `input_len=1024`, `output_len=512`, `warmup=1`, `repeats=2`,
PIECEWISE CUDA graph, and `backend="eager"`.

Baseline all-routes InfiniCore still dispatched `PagedAttentionDecode` through
the Python `infinicore.paged_attention` wrapper and measured only `209.42`
output tok/s against `417.31` vLLM native (`50.2%`). The route counters showed
all nine routes installed with no native fallback and `paged_attention_decode`
called `36792` times, making decode the dominant single-GPU gap.

Enabling the plugin C++ bridge for `PagedAttentionDecode` dispatches the same
route through InfiniCore `mha_kvcache_` and measured `412.44` output tok/s on
the same shape (`98.8%` of the native run). The run was valid with 148 graph
captures, no fallback routes, and bridge counter `PagedAttentionDecode=36792`.

`PagedAttentionDecode` now uses the C++ bridge by default. It can be explicitly
disabled with `VLLM_INFINICORE_DISABLE_CPP_BRIDGE=1` or
`VLLM_INFINICORE_ENABLE_CPP_BRIDGE=0` when comparing against the slower Python
wrapper path. `LMHead` remains opt-in through
`VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecode,LMHead`.
As of the 2026-06-04 strict no-MetaX TP=1 fix, the current default decode
bridge route is `PagedAttentionDecodeFlash`; the `PagedAttentionDecode` route
is retained as the older `mha_kvcache_` A/B path.

Artifacts:

- `artifacts/single-gpu-decision-qwen3-4b-20260603-195957`
- `artifacts/single-gpu-cpp-decode-qwen3-4b-20260603-200530`

## 2026-05-04 Bootstrap

Created a clean `vllm-infinicore` project skeleton at `/root/vllm-infinicore`.

Initial goals:

- Build an independent vLLM general plugin package.
- Target single-node Qwen3 inference on MetaX C550 with MACA 3.5.3.
- Start with operator route declarations and a dry registration chain only.
- Keep CUDA Graph behavior conservative until patched paths are proven safe.

Trusted facts imported from the benchmark audit:

- The current audit model is `/mnt/geogpt-doc-new/default/xb/qwen3-8B`.
- vLLM native cudagraph works on MetaX when using PIECEWISE cudagraph with `backend="eager"` and `enforce_eager=False`.
- Old TPS tables are historical and must not be used for new performance claims.
- Future benchmarks must use exact prompt token IDs, aligned sampling, output-only TPS, decoded-output validation, warmup, and repeated measurement.

Implemented in the bootstrap:

- `pyproject.toml` declares package `vllm-infinicore`.
- Entry point group is `vllm.general_plugins`.
- Entry point name is `vllm_infinicore`.
- Entry point target is `vllm_infinicore:register`.
- `vllm_infinicore.register()` is idempotent and dry by default.
- `vllm_infinicore.patching` records the initial Qwen3 operator scope.
- `vllm_infinicore.ops` reserves a future C++/PyTorch custom op loader.
- `configs/qwen3_infinicore_graph.yaml` documents conservative route defaults.

Known non-goals for this bootstrap:

- No C++ InfiniCore kernels are implemented.
- No vLLM internals are monkey patched.
- PA/KV explicit graph paths are not enabled.
- No throughput conclusion is made.

Next steps:

1. Add a structured config loader and route validation.
2. Add one minimal non-PA PyTorch custom op prototype behind an explicit env flag.
3. Build a 128/32 correctness smoke that verifies token counts and decoded output health.
4. Add a graph-safety probe before enabling any path during vLLM cudagraph capture.

## 2026-05-04 Foundation Hardening

Implemented the first foundation pass before enabling any vLLM execution path:

- Added a structured YAML config loader and route registry validator.
- Added regression tests for dry registration, entry point metadata, config consistency, and default-off custom op loading.
- Added a minimal RMSNorm PyTorch custom op prototype behind `VLLM_INFINICORE_ENABLE_CUSTOM_OPS`.
- Kept `vllm_infinicore.register()` dry by default with no torch import and no monkey patches.

Still deferred:

- No C++ InfiniCore kernels are implemented.
- No vLLM internals are monkey patched.
- No route is enabled by default.
- No performance conclusion is made.

## 2026-05-04 RMSNorm Opt-In Route

Added the first explicit vLLM integration route while preserving the dry
default:

- `VLLM_INFINICORE_ENABLE_PATCHES=1` plus
  `VLLM_INFINICORE_ROUTES=RMSNorm` installs the RMSNorm route.
- The route uses vLLM's out-of-tree `CustomOp.register_oot(name="RMSNorm")`
  registry and does not edit site-packages.
- `InfiniCoreRMSNorm` routes only weighted RMSNorm calls without residuals or
  variance override to `vllm_infinicore::rms_norm`.
- Residual/fused-add, no-weight, and variance override cases fall back to the
  vLLM PyTorch-native RMSNorm implementation.
- Custom op loading can be forced by the patch installer, while direct
  `vllm_infinicore.ops.rms_norm()` calls remain gated by
  `VLLM_INFINICORE_ENABLE_CUSTOM_OPS`.
- Added `scripts/qwen3_128_32_smoke.py` for the Qwen3-8B 128 input / 32 output
  vLLM baseline correctness smoke.

Still deferred:

- No C++ InfiniCore kernel is implemented; the RMSNorm op is still a Python
  PyTorch custom op prototype.
- No PA/KV, RoPE, MatMul, SiluAndMul, Embedding, or LMHead route is enabled.
- No throughput or graph-safety conclusion is made from this route alone.

## 2026-05-04 Full Route-State And Fallback Framework

Extended the scaffold from a single RMSNorm opt-in route to full Qwen3 operator
coverage:

- Added `RouteState` records for all nine scoped operators.
- Added route selection with `VLLM_INFINICORE_ROUTES=all` or comma-separated
  route subsets.
- Added per-operator disable control through
  `VLLM_INFINICORE_DISABLED_ROUTES`.
- Added `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=1` so the full route table can be
  requested while preserving vLLM native execution.
- Added idempotent uninstall plumbing through `vllm_infinicore.unregister()`.
- Extended YAML config validation with native fallback and validation-path
  fields for every operator.
- Added pure-Python validation utilities for token counts, decoded text health,
  repetition checks, graph evidence, and output-only TPS records.
- Reworked `scripts/qwen3_128_32_smoke.py` into a subprocess-isolated graph
  smoke harness with shared prompt token IDs.

Validation run:

- `python -m compileall vllm_infinicore tests`
- `python -m unittest discover -s tests` (`30` tests passed)
- Dry import/register check with patching disabled
- Full-route native-fallback registration check with
  `VLLM_INFINICORE_ENABLE_PATCHES=1`,
  `VLLM_INFINICORE_ROUTES=all`,
  `VLLM_INFINICORE_FORCE_NATIVE_FALLBACK=1`
- Config load and registry consistency check
- Qwen3-8B graph smoke:
  `python scripts/qwen3_128_32_smoke.py --trust-remote-code --warmup 1 --repeats 2 --cases native-graph,plugin-fallback-graph`

Qwen3-8B smoke artifact:

- Summary: `artifacts/qwen3_128_32_smoke.json`
- Native graph: `artifacts/qwen3_128_32_smoke_cases/native-graph.json`
- Plugin fallback graph:
  `artifacts/qwen3_128_32_smoke_cases/plugin-fallback-graph.json`
- Prompt IDs: `artifacts/qwen3_128_32_smoke_cases/prompt-in128.json`

Smoke result:

| Case | Input tokens | Output tokens | Graph captures | Output TPS | Validation |
|---|---:|---:|---:|---:|---|
| `native-graph` | 128 | 32 | 148 | 53.11 | `validation_errors=[]` |
| `plugin-fallback-graph` | 128 | 32 | 148 | 53.27 | `validation_errors=[]` |

The decoded preview was readable for both cases, with replacement/control
characters at `0` and no degenerate repetition flagged. This is a quick graph
and correctness smoke. It is not a formal throughput benchmark or a claim that
the plugin is faster than vLLM native graph; all plugin routes in the fallback
case intentionally used native vLLM execution.

## 2026-05-05 All-Routes InfiniCore Eager Smoke

Implemented actual InfiniCore-backed wrappers for the full Qwen3-8B scoped
operator set:

- `RMSNorm`, `SiluAndMul`, `RoPE`, `Embedding`, `MatMul`, and `LMHead` route
  through `torch.ops.vllm_infinicore.*` wrappers backed by the installed
  `infinicore` Python APIs and underlying `_infinicore` extension.
- `StoreKVCache`, `PagedAttentionPrefill`, and `PagedAttentionDecode` patch the
  vLLM attention backend implementation methods rather than
  `Attention.forward`, preserving vLLM's opaque attention op boundary.
- `LMHead` now patches `UnquantizedEmbeddingMethod.apply` for
  `ParallelLMHead`; the general MatMul path patches
  `UnquantizedLinearMethod.apply`.
- Strict backend mode (`VLLM_INFINICORE_STRICT_BACKEND=1`) raises on wrapper
  failures instead of silently falling back.
- Runtime backend call counters were added to the smoke artifact so route
  installation is not used as the only evidence of InfiniCore execution.

Validation run:

- `python -m compileall vllm_infinicore scripts/qwen3_128_32_smoke.py tests`
- `python -m unittest discover -s tests` (`30` tests passed)
- all-routes install/uninstall check with `VLLM_INFINICORE_ROUTES=all`
- LMHead strict device probe: max diff `0.0`
- attention backend strict StoreKVCache + Decode probe: max diff `0.0`
- Qwen3-8B all-routes strict eager smoke:
  `artifacts/qwen3_128_32_all_routes_strict_eager.json`

Qwen3-8B strict eager smoke result:

| Case | Input tokens | Output tokens | Validation | Output TPS |
|---|---:|---:|---|---:|
| `custom-eager` / `all` routes | 128 | 32 | `validation_errors=[]` | 21.54 |

Measured InfiniCore backend calls in the smoke artifact:

| Backend wrapper | Calls |
|---|---:|
| `embedding` | 32 |
| `linear` | 4608 |
| `lm_head` | 32 |
| `paged_attention_prefill` | 36 |
| `paged_attention_decode` | 1116 |
| `rms_norm` | 2336 |
| `rotary_embedding` | 1152 |
| `silu_and_mul` | 1152 |
| `store_kv_cache` | 1152 |

Historical graph-safety blocker before the stream bridge:

- all-routes strict graph smoke reached CUDA graph capture but triggered a MACA
  Xnack/ATU fault in a RoPE kernel:
  `_Z23ropeThreadPerItemKernel...`.
- At this point in the log, the all-routes InfiniCore path was validated only
  for eager vLLM inference. The later stream-bridge entry below supersedes this
  status.

## 2026-05-05 InfiniCore Stream Bridge And Graph Smoke

Resolved the graph replay failure by explicitly joining InfiniCore's runtime
stream with PyTorch's current stream around every `_infinicore` launch:

- `infinicore.context::getStream()` is a runtime-owned stream, while vLLM
  cudagraph capture follows PyTorch stream ordering.
- Direct InfiniCore launches during `torch.cuda.CUDAGraph()` produced an empty
  graph in a standalone RMSNorm probe and replayed stale output.
- Wrapping the InfiniCore stream with `torch.cuda.ExternalStream` and adding
  `wait_stream` dependencies before and after the launch made the standalone
  RMSNorm graph replay match the eager result.
- The bridge is now used by RMSNorm, SiluAndMul, MatMul/LMHead, Embedding,
  RoPE, StoreKVCache, PagedAttentionPrefill, and PagedAttentionDecode.
- If the bridge cannot obtain an InfiniCore stream during CUDA graph capture,
  strict mode raises instead of silently taking the old graph-unsafe path.

Validation run after the stream bridge:

- `python -m compileall vllm_infinicore scripts/qwen3_128_32_smoke.py tests`
- `python -m unittest discover -s tests` (`30` tests passed)
- standalone RMSNorm CUDAGraph replay probe: patched max diff `0.03125`
- Qwen3-8B single-route strict graph smokes for `RMSNorm`, `SiluAndMul`,
  `Embedding`, `MatMul`, and `LMHead`: all `validation_errors=[]`
- Qwen3-8B strict graph smoke for non-attention routes except RoPE:
  `artifacts/qwen3_128_32_non_attention_no_rope_streamed_strict_graph.json`
- Qwen3-8B strict graph smoke for `RoPE`:
  `artifacts/qwen3_128_32_route_rope_streamed_strict_graph.json`
- Qwen3-8B all-routes strict eager smoke:
  `artifacts/qwen3_128_32_all_routes_streamed_strict_eager.json`
- Qwen3-8B all-routes strict graph smoke:
  `artifacts/qwen3_128_32_all_routes_streamed_strict_graph.json`

Current Qwen3-8B all-routes strict smoke results:

| Case | Input tokens | Output tokens | Graph captures | Validation |
|---|---:|---:|---:|---|
| `custom-eager` / `all` routes | 128 | 32 | 0 | `validation_errors=[]` |
| `custom-graph` / `all` routes | 128 | 32 | 148 | `validation_errors=[]` |

Measured InfiniCore backend calls in the current all-routes eager artifact:

| Backend wrapper | Calls |
|---|---:|
| `embedding` | 32 |
| `linear` | 4608 |
| `lm_head` | 32 |
| `paged_attention_prefill` | 36 |
| `paged_attention_decode` | 1116 |
| `rms_norm` | 2336 |
| `rotary_embedding` | 1152 |
| `silu_and_mul` | 1152 |
| `store_kv_cache` | 1152 |

Measured InfiniCore backend calls in the current all-routes graph artifact:

| Backend wrapper | Calls |
|---|---:|
| `embedding` | 1 |
| `linear` | 144 |
| `lm_head` | 32 |
| `paged_attention_prefill` | 36 |
| `paged_attention_decode` | 1116 |
| `rms_norm` | 73 |
| `rotary_embedding` | 36 |
| `silu_and_mul` | 36 |
| `store_kv_cache` | 36 |

In graph mode, Python backend counters are evidence that the wrappers ran
during graph capture or non-captured paths; captured graph replay does not
re-enter Python for non-attention model ops. The output validation and graph
capture count are therefore required alongside these counters.

## 2026-05-05 Qwen3-8B Three-Engine Graph Throughput

Added `scripts/qwen3_three_engine_throughput.py` for fair graph-mode throughput
checks across InfiniLM, vLLM native, and vLLM-InfiniCore:

- Generates one tokenizer prompt ID sequence once and reuses the exact same
  prompt IDs for every engine.
- Uses output-only TPS as the primary metric.
- Records actual per-request input/output token counts, decoded preview, and
  text-health counters.
- Aligns sampling with `temperature=0.0`, `top_p=1.0`, `top_k=1`, EOS
  disabled, and vLLM `min_tokens=max_tokens=output_len`.
- Runs vLLM graph with `CUDAGraphMode.PIECEWISE`, capture sizes
  `[1, 2, 4, 8]`, one graph warmup, and `backend="eager"`.
- Runs InfiniLM with `enable_graph_compiling=True`.

Smoke note: InfiniLM batch mode requires paged-cache `num_blocks` to cover all
requests, not just one request. The script therefore uses
`ceil((input_len + output_len) / block_size) * batch_size`.

Formal run:

```bash
python scripts/qwen3_three_engine_throughput.py \
  --engines infinilm,vllm-native,vllm-infinicore \
  --batch-size 8 \
  --input-len 4096 \
  --output-len 512 \
  --warmup 1 \
  --repeats 3 \
  --max-model-len 5120 \
  --run-dir artifacts/qwen3-8b-three-engine-bs8-in4096-out512-graph-20260505-115121
```

Current graph-mode throughput results for this run:

| Engine | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---:|---:|---:|---:|
| InfiniLM | true | 287.95 | 287.95 | n/a |
| vLLM native | true | 286.10 | 286.06 | 148 |
| vLLM-InfiniCore | true | 43.20 | 43.19 | 148 |

All three cases produced `8 * 512 * 3 = 12288` measured output tokens with
`validation_errors=[]`. vLLM-InfiniCore installed all nine Qwen3 scoped routes
with no native fallback routes in this run. Its measured backend route counters
included nonzero calls for embedding, RMSNorm, MatMul/LMHead, RoPE,
StoreKVCache, PagedAttentionPrefill, PagedAttentionDecode, and SiluAndMul.

## 2026-05-05 vLLM-InfiniCore Throughput Bottleneck Isolation

Investigated the vLLM-InfiniCore graph throughput regression from the
three-engine run above.

Key isolation runs at `bs=8`, `input_len=4096`, `output_len=128`, graph mode:

| Route set | Output TPS | Validation |
|---|---:|---|
| vLLM native | 175.38 | valid |
| all InfiniCore routes | 17.32 | valid |
| only StoreKVCache/PagedAttentionPrefill/PagedAttentionDecode | 17.88 | valid |
| all except StoreKVCache/PagedAttentionPrefill/PagedAttentionDecode | 129.56 | valid |
| RMSNorm/SiluAndMul/Embedding | 150.96 | valid |
| RoPE only | 145.75 | valid |

Conclusion: the severe slowdown is dominated by the InfiniCore attention/KV
routes. In this integration those routes still execute through the Python
attention backend wrapper on decode replay (`PagedAttentionDecode` was called
9432 times in the `output_len=128` all-routes run), so graph capture does not
remove the per-token/layer Python wrapper overhead. Disabling those three
routes while keeping non-attention routes restored most throughput.

Follow-up validation at the requested long-output shape (`bs=8`,
`input_len=4096`, `output_len=512`, graph mode, `warmup=1`, `repeats=3`):

| vLLM-InfiniCore route set | Output TPS | Graph captures | Validation |
|---|---:|---:|---|
| all nine scoped routes | 43.20 | 148 | valid |
| all except attention/KV routes | 218.87 | 148 | valid |
| RMSNorm/SiluAndMul/Embedding | 268.21 | 148 | valid |

The route selector now supports `VLLM_INFINICORE_ROUTES=throughput`, which
expands to `RMSNorm,SiluAndMul,Embedding`. The throughput benchmark script uses
that profile by default for vLLM-InfiniCore. Use `--infinicore-routes all`
explicitly when validating full operator coverage. The attention/KV routes
remain available for correctness and coverage probes, but should not be used
for throughput conclusions until they are moved out of the Python replay path
or otherwise proven performant.

Formal three-engine rerun with the throughput profile:

```bash
python scripts/qwen3_three_engine_throughput.py \
  --engines infinilm,vllm-native,vllm-infinicore \
  --batch-size 8 \
  --input-len 4096 \
  --output-len 512 \
  --warmup 1 \
  --repeats 3 \
  --max-model-len 5120 \
  --run-dir artifacts/qwen3-8b-three-engine-bs8-in4096-out512-graph-throughput-profile-20260505-122840
```

| Engine | Route/profile | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---|---:|---:|---:|---:|
| InfiniLM | graph compiling | true | 288.05 | 287.98 | n/a |
| vLLM native | native graph | true | 282.72 | 282.78 | 148 |
| vLLM-InfiniCore | `throughput` | true | 269.14 | 269.17 | 148 |

All three cases produced `12288` measured output tokens with
`validation_errors=[]`. The vLLM-InfiniCore result installed
`RMSNorm,SiluAndMul,Embedding` and no native fallback routes.

## 2026-05-05 InfiniLM vs vLLM Native Reason Analysis

Added `docs/INFINILM_VS_VLLM_REASON.md` to document the current evidence for
why InfiniLM can beat vLLM native in long-output Qwen3-8B graph runs.

Output length sweep at `bs=8`, `input_len=4096`, graph mode:

| Output length | InfiniLM TPS | vLLM native TPS | Faster engine |
|---:|---:|---:|---|
| 32 | 55.94 | 69.30 | vLLM native |
| 128 | 157.30 | 175.60 | vLLM native |
| 512 | 288.05 | 282.72 | InfiniLM |
| 1024 | 332.96 | 311.57 | InfiniLM |

Linear fit of average iteration time against output length:

| Engine | Fixed cost / iteration | Decode step cost | Steady decode TPS at bs=8 |
|---|---:|---:|---:|
| InfiniLM | 3.919 s | 20.184 ms/token-step | 396.35 tok/s |
| vLLM native | 2.916 s | 22.784 ms/token-step | 351.12 tok/s |

Conclusion: InfiniLM's long-output advantage comes from lower steady-state
decode-loop cost, not lower fixed/prefill overhead. vLLM native has lower fixed
cost and wins at short outputs, but InfiniLM's per-step decode cost is lower
once output length is high enough to amortize its fixed cost. A vLLM native
control run with `detokenize=False` did not improve throughput (`282.98` tok/s
vs `283.61` tok/s with detokenize), so text decoding is not the cause.

## 2026-05-05 InfiniCore Attention Backend First Cut

Moved the InfiniCore attention/KV integration from the old method-patch layer
to a vLLM attention backend override:

- Added `vllm_infinicore.ops.vllm_attention_backend`.
- Attention routes now register
  `InfiniCoreFlashAttentionBackend` as vLLM's `FLASH_ATTN` backend.
- The installer wraps MetaX `register_attention_backends()` so MetaX can refresh
  its backend table first, then InfiniCore re-applies its `FLASH_ATTN` override.
- The backend reuses the platform FlashAttention metadata builder and KV cache
  layout, while `InfiniCoreFlashAttentionImpl` routes supported KV update,
  prefill, and decode calls through `infinicore_backend`.
- The old `vllm_attention.py` monkey-patch module is no longer used by the
  attention route installer.

Validation:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests` (`26` tests passed, `2` skipped)
- Registration test confirms `AttentionBackendEnum.FLASH_ATTN.get_path()` is
  `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`.
- Runtime introspection confirmed the earlier issue where MetaX re-registered
  `FLASH_ATTN` after plugin registration; the installer now wraps the MetaX
  registration hook to keep the InfiniCore backend selected.
- Eager Qwen3-8B `128/32` attention-backend smoke:
  `artifacts/attention_backend_custom_eager_128_32_v3.json`
  - `vllm_attention_backend`:
    `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
  - backend `_infinicore` calls:
    `store_kv_cache=1152`, `paged_attention_prefill=36`,
    `paged_attention_decode=1116`
  - backend route counters:
    `backend_kv_update_infinicore=1152`,
    `backend_prefill_infinicore=36`,
    `backend_decode_infinicore=1116`
  - `validation_errors=[]`
- Graph Qwen3-8B `bs=2`, `128/32` attention-backend smoke:
  `artifacts/attention-backend-smoke-bs2-in128-out32-v3`
  - `vllm_attention_backend`:
    `vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend`
  - backend `_infinicore` calls:
    `store_kv_cache=36`, `paged_attention_decode=1116`
  - `graph_capture_count=148`
  - `validation_errors=[]`

Current limitation: in the graph smoke, the prefill attention forward still
falls back to the platform backend (`backend_forward_fallback=36`). Eager mode
exercises InfiniCore prefill correctly. The next performance step is to make
the graph prefill metadata path satisfy the InfiniCore backend's supported
descriptor contract and then re-benchmark long decode.

Follow-up throughput check at the requested production shape (`bs=8`,
`input_len=4096`, `output_len=512`, graph mode, `warmup=1`, `repeats=3`):

```bash
python scripts/qwen3_three_engine_throughput.py \
  --engines vllm-native,vllm-infinicore \
  --batch-size 8 \
  --input-len 4096 \
  --output-len 512 \
  --warmup 1 \
  --repeats 3 \
  --max-model-len 5120 \
  --infinicore-routes StoreKVCache,PagedAttentionPrefill,PagedAttentionDecode \
  --run-dir artifacts/attention-backend-vs-native-bs8-in4096-out512-graph-20260505-135727
```

| Engine | Attention backend | Valid | Output TPS | Graph captures |
|---|---|---:|---:|---:|
| vLLM native | `vllm_metax...MacaFlashAttentionBackend` | true | 280.99 | 148 |
| vLLM-InfiniCore | `vllm_infinicore...InfiniCoreFlashAttentionBackend` | true | 44.93 | 148 |

The vLLM-InfiniCore run installed only the three attention/KV routes and
recorded nonzero backend `_infinicore` calls:
`store_kv_cache=540`, `paged_attention_prefill=540`, and
`paged_attention_decode=55620`. This confirms the backend override is active,
but performance is still approximately `6.25x` slower than vLLM native. The
remaining bottleneck is therefore not the old monkey-patch dispatch itself; it
is the per-token/layer InfiniCore attention backend path, especially decode,
still executing too many Python/backend descriptor/stream-bridge calls.

## 2026-05-05 Attention Gap Isolation And Throughput-Safe Profile

Added a small InfiniCore tensor-wrapper LRU cache for stable attention metadata
and KV cache views. Validation:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests` (`28` tests passed, `2` skipped)
- `git diff --check`
- Added unit coverage for wrapper cache reuse and stride-sensitive keys.

The cache did not materially improve the slow attention profile. A broader
q/out wrapper cache was rejected because it caused CUDA OOM at the Qwen3
benchmark shape by retaining activation buffers.

Route isolation at `bs=8`, `input_len=4096`, `output_len=128`, graph mode:

| Routes | Valid | Output TPS | Artifact |
|---|---:|---:|---|
| `StoreKVCache,PagedAttentionPrefill,PagedAttentionDecode` | true | 17.90 | `artifacts/attention-wrapper-cache-bs8-in4096-out128-graph-20260505-142100` |
| `StoreKVCache,PagedAttentionPrefill` | true | 60.84 | `artifacts/attention-no-decode-bs8-in4096-out128-graph-20260505-142722` |
| `PagedAttentionPrefill` | true | 61.65 | `artifacts/attention-prefill-only-bs8-in4096-out128-graph-20260505-143036` |
| `StoreKVCache` | true | 168.66 | `artifacts/attention-storekv-only-bs8-in4096-out128-graph-20260505-142918` |

Conclusion: `StoreKVCache` is acceptable for the current throughput profile;
`PagedAttentionPrefill` and `PagedAttentionDecode` are the performance-risk
routes. They remain available for correctness/operator coverage, but should not
be used for throughput comparisons until the underlying PA kernels or call
granularity are redesigned.

Implemented `VLLM_INFINICORE_ROUTES=attention-safe`, expanding to
`StoreKVCache`, and updated tests for the alias. The current throughput-safe
configuration is:

```text
VLLM_INFINICORE_ROUTES=throughput,attention-safe
```

Formal graph benchmark at `bs=8`, `input_len=4096`, `output_len=512`,
`warmup=1`, `repeats=3`:

| Engine/routes | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---:|---:|---:|---:|
| vLLM native | true | 286.55 | 286.46 | 148 |
| vLLM-InfiniCore `throughput,attention-safe` | true | 267.32 | 267.31 | 148 |

Artifact:
`artifacts/throughput-attention-safe-vs-native-bs8-in4096-out512-graph-20260505-143335`.
The throughput-safe plugin profile is now `93.3%` of vLLM native graph
throughput on this benchmark.

Requirement correction: this isolation profile is not an acceptable fix when
the target is that every scoped called operator routes through InfiniCore. The
benchmark script default was changed back to `VLLM_INFINICORE_ROUTES=all`, and
the `attention-safe` selector was removed. The isolation results above remain
diagnostic evidence only: they show that the remaining work is to improve the
InfiniCore `PagedAttentionPrefill`/`PagedAttentionDecode` paths rather than
bypassing them.

Reran the requested all-scoped-operator benchmark after the correction:

```bash
python scripts/qwen3_three_engine_throughput.py \
  --engines vllm-native,vllm-infinicore \
  --batch-size 8 \
  --input-len 4096 \
  --output-len 512 \
  --warmup 1 \
  --repeats 3 \
  --max-model-len 5120 \
  --infinicore-routes all \
  --run-dir artifacts/all-routes-vs-native-bs8-in4096-out512-graph-20260505-150547
```

| Engine/routes | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---:|---:|---:|---:|
| vLLM native | true | 282.96 | 282.92 | 148 |
| vLLM-InfiniCore `all` | true | 43.17 | 43.18 | 148 |

The vLLM-InfiniCore run installed all nine scoped routes:
`RMSNorm,SiluAndMul,RoPE,Embedding,MatMul,LMHead,StoreKVCache,`
`PagedAttentionPrefill,PagedAttentionDecode`. Runtime counters were nonzero
for every scoped route family: `embedding=15`, `rms_norm=1095`, `linear=2160`,
`rotary_embedding=540`, `store_kv_cache=540`,
`paged_attention_prefill=540`, `silu_and_mul=540`, `lm_head=1548`, and
`paged_attention_decode=55620`. The all-route performance gap remains open.

Follow-up: switched the PA routes to the FlashAttention-wrapped InfiniCore
operators used by InfiniLM's `FlashAttentionImpl`:

- `PagedAttentionPrefill`: `infinicore.paged_attention_prefill` ->
  `infinicore.mha_varlen`
- `PagedAttentionDecode`: `infinicore.paged_attention` ->
  `infinicore.mha_kvcache`
- KV cache views are presented in BSHD layout for these FA wrapper calls.

Validation:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests` (`28` tests passed, `2` skipped)
- Qwen3-8B all-routes graph smoke:
  `artifacts/qwen3_128_32_all_routes_mha_fa_graph.json`
  with `validation_errors=[]`
- Short all-routes throughput at `bs=8`, `input_len=4096`,
  `output_len=128`: `127.50` output tok/s,
  artifact `artifacts/all-routes-mha-fa-bs8-in4096-out128-graph-20260505-155737`

Formal all-routes graph benchmark after the FA wrapper switch:

| Engine/routes | Valid | Output TPS | Median iter TPS | Graph captures |
|---|---:|---:|---:|---:|
| vLLM native | true | 283.00 | 283.04 | 148 |
| vLLM-InfiniCore `all` | true | 211.73 | 211.59 | 148 |

Artifact:
`artifacts/all-routes-mha-fa-vs-native-bs8-in4096-out512-graph-20260505-155903`.
All nine scoped routes were still installed, and runtime counters remained
nonzero for every route family. The full-route profile improved from `43.17`
to `211.73` output tok/s, reaching `74.8%` of vLLM native graph throughput.

## 2026-05-05 All-Route Gap Ablation And RoPE Optimization

Added an ablation-matrix mode to `scripts/qwen3_three_engine_throughput.py`.
The mode generates one prompt ID manifest and reuses it across graph cases at
`bs=8`, `input_len=4096`, `output_len=512`, `warmup=1`, `repeats=3`.

Ablation artifact:
`artifacts/all-routes-gap-ablation-bs8-in4096-out512-graph-20260505-165647`

| Case | Output TPS | Recovered gap |
|---|---:|---:|
| vLLM native | 283.29 | 100.00% |
| vLLM-InfiniCore `all` | 212.49 | 0.00% |
| `all-minus-matmul-lmhead` | 215.84 | 4.74% |
| `all-minus-rope` | 247.66 | 49.67% |
| `attention-only` | 261.62 | 69.39% |
| `light-known-good` | 266.34 | 76.05% |

Conclusion from the deterministic order:

- Disabling `MatMul,LMHead` recovered only `3.35` tok/s, so Linear/LMHead was
  not the first optimization target.
- The attention-only path was `7.65%` below native, below the `15%` threshold.
- Disabling `RoPE` recovered `35.16` tok/s, so RoPE was the first optimization
  target.

Implemented RoPE wrapper optimizations:

- Cache stable contiguous InfiniCore wrappers for the sin/cos RoPE tables.
- Avoid `torch.cat` reconstruction when `rotary_dim == head_size`; Qwen3-8B
  rotates the full head, so the InfiniCore output can be reshaped directly.
- Keep strict InfiniCore routing active; no scoped route is bypassed in the
  final all-routes benchmark.

Validation after the RoPE optimization:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests` (`28` tests passed, `2` skipped)
- `git diff --check`
- Targeted `all` with `RoPE` disabled:
  `artifacts/target-all-minus-rope-after-rope-opt-bs8-in4096-out512-graph-20260505-171740`
  - Output TPS: `254.32`
  - `graph_capture_count=148`
  - `validation_errors=[]`
- Full all-routes production benchmark:
  `artifacts/all-routes-after-rope-opt-bs8-in4096-out512-graph-20260505-172113`
  - Output TPS: `262.41`
  - Native baseline from the same ablation manifest: `283.29`
  - Ratio: `92.62%` of vLLM native
  - `graph_capture_count=148`
  - `validation_errors=[]`
  - Installed all nine scoped routes.
  - Backend counters were nonzero for Embedding, RMSNorm, MatMul/Linear, RoPE,
    StoreKVCache, PagedAttentionPrefill, SiluAndMul, LMHead, and
    PagedAttentionDecode.
- `128/32` all-routes strict graph smoke:
  `artifacts/qwen3_128_32_all_routes_after_rope_opt_graph.json`
  - `graph_capture_count=148`
  - `validation_errors=[]`
  - Installed all nine scoped routes with nonzero route-family counters.

Current status: vLLM-InfiniCore all-routes graph mode now exceeds the
`>=90%` vLLM-native acceptance target at the production benchmark shape.

## 2026-05-05 95% All-Routes Follow-Up

Attempted to close the remaining all-route gap against same-manifest vLLM
native graph throughput at `bs=8`, `input_len=4096`, `output_len=512`,
`warmup=1`, `repeats=3`.

Implemented and kept:

- Cached the InfiniCore runtime stream pointer per device before constructing
  `torch.cuda.ExternalStream`, avoiding repeated capsule lookup on every
  backend launch.
- Added `--ablation-cases` to `scripts/qwen3_three_engine_throughput.py` so
  focused production ablations can reuse one manifest without running the full
  historical matrix.

Focused post-RoPE ablation artifact:
`artifacts/all-routes-decode-opt-ablation-bs8-in4096-out512-graph-20260505`

| Case | Output TPS | Graph captures | Validation |
|---|---:|---:|---|
| `native` | 283.48 | 148 | `validation_errors=[]` |
| `all` | 257.55 | 148 | `validation_errors=[]` |
| `attention-only` | 264.15 | 148 | `validation_errors=[]` |
| `non-attn-only` | 271.03 | 148 | `validation_errors=[]` |
| `all-minus-rope` | 248.22 | 148 | `validation_errors=[]` |
| `light-known-good` | 268.01 | 148 | `validation_errors=[]` |

Best retained native/all production comparison from this pass:
`artifacts/all-routes-stream-cache-vs-native-bs8-in4096-out512-graph-20260505`

| Engine/routes | Output TPS | Graph captures | Validation |
|---|---:|---:|---|
| vLLM native | 280.93 | 148 | `validation_errors=[]` |
| vLLM-InfiniCore `all` | 262.97 | 148 | `validation_errors=[]` |

This is `93.61%` of same-run vLLM native and does not meet the new `>=95%`
target. All nine scoped routes were installed with no native fallbacks, and
runtime counters were nonzero for every route family:
`embedding`, `rms_norm`, `linear`, `rotary_embedding`, `store_kv_cache`,
`paged_attention_prefill`, `silu_and_mul`, `lm_head`, and
`paged_attention_decode`.

Rejected variants from this pass:

- Decode `q/out` uncached raw-stride wrappers regressed the focused all-route
  case to `257.55` tok/s.
- A non-retaining alias cache for decode `q/out` wrappers was graph-correct but
  regressed production throughput to `258.43` tok/s.
- Reusing retained wrappers for stable linear/LMHead/embedding/RMSNorm weights
  was graph-correct but regressed production throughput to `256.89` tok/s.
- Saturating Python route counters kept nonzero evidence but did not improve
  throughput (`256.90` tok/s), so exact counters were preserved.
- The non-in-place `mha_kvcache` API plus copy-back was graph-correct in the
  `128/32` smoke but slower than the in-place decode path and was not retained.

Conclusion: the `>=95%` all-routes target remains open. The current best
evidence still points to the combined attention decode/LMHead Python/backend
boundary rather than simple wrapper cache misses. Future work should avoid the
rejected descriptor/cache variants above and focus on reducing per-token
attention decode and logits projection call overhead without disabling scoped
routes.

## 2026-05-05 Plugin C++ Bridge Probe

Implemented an opt-in plugin-owned C++ bridge for the hottest remaining route
families without changing InfiniCore or InfiniLM:

- `VLLM_INFINICORE_ENABLE_CPP_BRIDGE=1`
- `VLLM_INFINICORE_CPP_BRIDGE_ROUTES=PagedAttentionDecode,LMHead`
- The bridge is built on demand through `torch.utils.cpp_extension.load`.
- `PagedAttentionDecode` calls `infinicore::op::mha_kvcache_` from C++ with
  torch tensor raw-pointer views.
- `LMHead` calls `infinicore::op::linear_` from C++.
- The existing Python stream bridge still wraps the C++ launch to preserve
  graph ordering.
- Bridge call counters are now recorded in smoke and throughput artifacts.

This probe records the older `mha_kvcache_` bridge. The current default decode
bridge route is `PagedAttentionDecodeFlash`.

Validation:

- `python -m compileall vllm_infinicore scripts tests`
- `python -m unittest discover -s tests`
- `git diff --check`
- C++ bridge load probe succeeded:
  `vllm_infinicore_cpp_bridge.so` built under torch extension cache.
- `128/32` all-routes strict graph smoke with `PagedAttentionDecode` bridged:
  `artifacts/qwen3_128_32_all_routes_cpp_decode_graph.json`
  - `graph_capture_count=148`
  - `validation_errors=[]`
  - all nine routes installed; bridge counter `PagedAttentionDecode=1116`
- `128/32` all-routes strict graph smoke with `PagedAttentionDecode,LMHead`
  bridged:
  `artifacts/qwen3_128_32_all_routes_cpp_decode_lmhead_graph.json`
  - `graph_capture_count=148`
  - `validation_errors=[]`
  - all nine routes installed; bridge counters `PagedAttentionDecode=1116`,
    `LMHead=32`

Production benchmark results at `bs=8`, `input_len=4096`, `output_len=512`,
`warmup=1`, `repeats=3`:

| Bridge routes | Native TPS | All-routes TPS | Ratio | Artifact |
|---|---:|---:|---:|---|
| `PagedAttentionDecode` | 286.35 | 262.20 | 91.57% | `artifacts/all-routes-cpp-decode-vs-native-bs8-in4096-out512-graph-20260505` |
| `PagedAttentionDecode,LMHead` | 286.39 | 263.83 | 92.12% | `artifacts/all-routes-cpp-decode-lmhead-vs-native-bs8-in4096-out512-graph-20260505` |

The bridge path is correct and remains available as an opt-in diagnostic path,
but it is not a throughput win over the previous best retained all-routes run
(`262.97 / 280.93 = 93.61%`). It is therefore not enabled by default and does
not close the `>=95%` target.

Focused bridge-enabled ablation:

- Partial matrix artifact:
  `artifacts/all-routes-cpp-bridge-ablation-bs8-in4096-out512-graph-20260505`
- Separate light-known-good artifact:
  `artifacts/all-routes-cpp-bridge-ablation-light-known-good-bs8-in4096-out512-graph-20260505`

| Case | Output TPS | Graph captures | Validation |
|---|---:|---:|---|
| `native` | 281.13 | 148 | `validation_errors=[]` |
| `all` | 259.09 | 148 | `validation_errors=[]` |
| `attention-only` | 261.22 | 148 | `validation_errors=[]` |
| `non-attn-only` | 271.13 | 148 | `validation_errors=[]` |
| `all-minus-rope` | 254.46 | 148 | `validation_errors=[]` |
| `light-known-good` | 267.80 | 148 | `validation_errors=[]` |

Conclusion: moving only the Python descriptor construction for decode/LMHead
into a C++ extension is insufficient. The remaining gap is more likely in the
underlying InfiniCore decode/logits kernel/API behavior, vLLM scheduling around
attention/logits, or stream synchronization granularity. Further work should
profile kernel time versus stream-wait time and compare the exact InfiniLM C++
execution context before adding more bridge code.
