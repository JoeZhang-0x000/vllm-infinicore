"""Numeric and non-default-stream validation for the locked Ascend bridge."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch_npu
from vllm_infinicore.ops import ascend_backend as backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(41)
    torch.npu.set_device(0)
    results = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for rows in (1, 4, 128):
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                x = torch.randn(rows, 1024, device="npu", dtype=dtype)
                weight = torch.randn(1024, device="npu", dtype=dtype)
                gateup = torch.randn(rows, 6144, device="npu", dtype=dtype)
                w = torch.randn(256, 1024, device="npu", dtype=dtype) / 32
                ids = torch.arange(rows, device="npu", dtype=torch.int64) % 256
                positions = torch.arange(rows, device="npu", dtype=torch.int64)
                q = torch.randn(rows, 16 * 128, device="npu", dtype=dtype)
                k = torch.randn(rows, 8 * 128, device="npu", dtype=dtype)
                angle = torch.randn(512, 64, device="npu", dtype=torch.float32)
                cache = torch.cat((angle.cos(), angle.sin()), -1).to(dtype)
                cos_ref, sin_ref = cache[positions].float().chunk(2, dim=-1)

                def rope_reference(tensor):
                    shaped = tensor.float().reshape(rows, -1, 128)
                    first, second = shaped.chunk(2, dim=-1)
                    cos_r, sin_r = cos_ref[:, None, :], sin_ref[:, None, :]
                    return (
                        torch.cat(
                            (
                                first * cos_r - second * sin_r,
                                first * sin_r + second * cos_r,
                            ),
                            dim=-1,
                        )
                        .reshape(tensor.shape)
                        .to(dtype)
                    )

                expected_q, expected_k = rope_reference(q), rope_reference(k)
                actual_q, actual_k = backend.rotary_embedding(
                    positions, q, k, 128, 128, cache, True
                )
                # The blocked weight path must be converted to ND before exposing raw pointers.
                blocked = (
                    torch_npu.npu_format_cast(w, 29) if dtype != torch.float32 else w
                )
                pairs = {
                    "rms_norm": (
                        backend.rms_norm(x, weight, 1e-6),
                        torch_npu.npu_rms_norm(x, weight, 1e-6)[0],
                    ),
                    "silu_and_mul": (
                        backend.silu_and_mul(gateup),
                        torch_npu.npu_swiglu(gateup),
                    ),
                    "linear_nz": (
                        backend.execute(
                            "linear",
                            x,
                            lambda: backend.linear(x, blocked),
                            lambda: torch.nn.functional.linear(x, w),
                        ),
                        torch.nn.functional.linear(x, w),
                    ),
                    "embedding": (
                        backend.embedding(ids, w),
                        torch.nn.functional.embedding(ids, w),
                    ),
                    "rope_q": (actual_q, expected_q),
                    "rope_k": (actual_k, expected_k),
                }
                # Queue allocator activity immediately after raw-pointer launches.
                torch.empty_like(gateup).fill_(123)
            stream.synchronize()
            for name, (actual, expected) in pairs.items():
                diff = (actual.float() - expected.float()).abs()
                tolerance = (
                    0.04
                    if dtype == torch.bfloat16
                    else 0.004
                    if dtype == torch.float16
                    else 1e-4
                )
                passed = torch.allclose(
                    actual, expected, rtol=tolerance, atol=tolerance
                )
                print(dtype, rows, name, diff.max().item(), passed, flush=True)
                results.append(
                    dict(
                        op=name,
                        dtype=str(dtype),
                        rows=rows,
                        max_abs=diff.max().item(),
                        passed=passed,
                    )
                )
    backend.clear_cache()
    from vllm_infinicore.ops import infinicore_backend as counters

    Path(args.output).write_text(
        json.dumps(
            dict(
                passed=all(r["passed"] for r in results),
                checks=results,
                fallback_counts=counters.backend_fallback_counts(),
                fallback_reasons=counters.backend_fallback_reasons(),
            ),
            indent=2,
        )
    )
    assert all(r["passed"] for r in results), "numeric failures; see output JSON"
    print(f"Passed {len(results)} numeric checks on non-default NPU streams")


if __name__ == "__main__":
    main()
