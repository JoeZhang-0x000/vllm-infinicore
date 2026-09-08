# Ascend Qwen3-0.6B Availability — 2026-09-07

The current worktree includes **real InfiniCore Ascend operator adapters**,
while retaining `vllm_ascend` for the platform, device management, workers,
communication, attention and KV cache.

## Pinned Ascend Operator Integration

InfiniCore official `main` was resolved on 2026-09-07 to
`d3551f37538896056e164abf91b120e38c27007b`. The full revision and bridge ABI are
tracked in `vllm_infinicore/infinicore.lock.json`. `scripts/build_ascend.py`
fetches that exact commit or verifies an existing clean checkout. It builds
only the required unmodified upstream operator sources, records the library
SHA256 in a manifest, and uses a fresh CMake tree to avoid CANN's non-repeatable
incremental preprocessing. The runtime verifies embedded revision and ABI
before any route is installed.

The adapter wraps methods on Ascend-owned classes; OOT registrations still
refer to `vllm_ascend` classes. Tensor pointers use the current torch NPU stream,
with allocator stream recording and ND conversion for blocked weights. The
bridge creates only operator handles; it does not initialize/reset devices or
replace the Ascend runtime. Bounded descriptors are keyed by device, stream,
shape, stride, dtype and scalar parameters, and synchronized before eviction.

The pinned upstream Embedding destroy dispatcher omits Ascend. The plugin's
bridge explicitly destroys the upstream Ascend descriptor, releasing its ACL
workspace without modifying the pinned upstream checkout.

| Route / case | Current behavior |
|---|---|
| RMSNorm without residual | InfiniCore Ascend |
| Fused Add+RMSNorm | Original Ascend method; upstream lacks an Ascend fused kernel |
| SiluAndMul / SwiGLU | InfiniCore for aligned eight-block tiles, hidden size <= 8192; otherwise native |
| RoPE | InfiniCore for full 128-dimensional heads and 1D positions; offsets/MTP and other head sizes remain native |
| Embedding | InfiniCore Ascend, including correct descriptor/workspace destruction |
| MatMul / LMHead, BF16 and FP16 | InfiniCore GEMM; blocked weights converted to ND |
| MatMul / LMHead, FP32 | Native fallback because pinned GEMM uses reduced-precision FP32 math |
| StoreKVCache / PagedAttentionPrefill / PagedAttentionDecode | Unmodified Ascend attention backend |
| No operator library configured | All nine routes remain native; no heavy framework imports required for discovery |

Validation on the same NPU 0 / 910B4 / vLLM 0.23 / Qwen3-0.6B stack:

- 59 targeted unit/regression tests pass in the container.

- 54 operator checks pass across BF16/FP16/FP32, 1/4/128 tokens and non-default
  NPU streams. Three FP32 GEMM cases explicitly use native fallback.
- BF16 RMSNorm, SwiGLU, GEMM and Embedding match native exactly in these probes.
  RoPE matches an independent mathematical reference (BF16/FP32 exact;
  FP16 maximum absolute difference 0.00390625 within the declared tolerance).
- Native, `all` and automatic-discovery `autoall` eager model cases each
  produce 12 valid outputs, using shared prompt IDs, batch sizes 1/4 and
  exactly 32 generated tokens. Both plugin cases match all 12 native outputs
  token-for-token. `autoall` leaves `VLLM_PLUGINS` unset.
- Both plugin cases have the same worker counts shown below and retain
  Ascend's OOT classes. Fused Add+RMSNorm is counted only as native fallback.
- The full route case installs six real routes and keeps three attention/KV
  routes native, with no registration failure. Per-worker counts after warmup:

| Counter | Calls in each of `all` / `autoall` |
|---|---:|
| `embedding` | 192 |
| `rms_norm` | 10,944 |
| `linear` | 21,504 |
| `rotary_embedding` | 5,376 |
| `silu_and_mul` | 5,376 |
| `lm_head` | 192 |
| `fused_add_rms_norm` (native fallback) | 10,752 |

Artifacts live in `artifacts/ascend-adapter-20260907/`, mirrored from
`/workspace/work/infinicore-ascend-20260907/results/` in `zx-vllm-ascend-023`.
The isolated library is
`/workspace/work/infinicore-ascend-20260907/build/libvllm_infinicore_ascend.so`.
See the README's Ascend section for build/install/test commands. This validates
single-NPU eager availability. See [27B TP=2 validation](ASCEND_27B_TP2_AVAILABILITY.md)
and [27B graph throughput](ASCEND_27B_GRAPH_THROUGHPUT.md) for separate model
and graph results and their limitations.
