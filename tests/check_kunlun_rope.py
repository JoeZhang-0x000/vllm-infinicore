"""Manual Kunlun GPU regression: signed positions, bounds, packed strides, Graph replay.

Run in the configured xpytorch/InfiniCore environment:
    python tests/check_kunlun_rope.py --output /tmp/kunlun-rope.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from vllm_infinicore.benchmarks.common import mode_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(mode_environment("infinicore", os.environ, "kunlun", ("RoPE",)))
    import torch

    from vllm_infinicore.operators.common import backend, cpp_bridge

    if not torch.cuda.is_available():
        raise RuntimeError("Run this check with xpytorch on Kunlun")
    torch.set_num_threads(4)
    module = cpp_bridge.module()
    rows = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for pos_dtype in (torch.int32, torch.int64):
            for neox in (True, False):
                torch.manual_seed(2027)
                count, dim, qheads, kvheads, length = 17, 128, 8, 2, 37
                packed = torch.randn(count, (qheads + 2 * kvheads) * dim, dtype=dtype)
                qcpu = packed[:, : qheads * dim]
                kcpu = packed[:, qheads * dim : (qheads + kvheads) * dim]
                packed_gpu = packed.cuda()
                query = packed_gpu[:, : qheads * dim]
                key = packed_gpu[:, qheads * dim : (qheads + kvheads) * dim]
                poscpu = torch.arange(count, dtype=pos_dtype)
                poscpu[:3] = torch.tensor([-13, -1, 0], dtype=pos_dtype)
                poscpu[-3:] = torch.tensor([length - 1, length, length + 123], dtype=pos_dtype)
                positions = poscpu.cuda()
                freq = torch.outer(
                    torch.arange(length).float(),
                    1 / (1e6 ** (torch.arange(0, dim, 2).float() / dim)),
                )
                tablecpu = torch.cat((freq.cos(), freq.sin()), -1).to(dtype)
                table = tablecpu.cuda()

                def reference(x):
                    cosine, sine = tablecpu[poscpu.long().clamp(0, length - 1)].float().chunk(2, -1)
                    value = x.view(count, -1, dim).float()
                    if neox:
                        a, b = value.chunk(2, -1)
                        out = torch.cat(
                            (
                                a * cosine[:, None] - b * sine[:, None],
                                b * cosine[:, None] + a * sine[:, None],
                            ),
                            -1,
                        )
                    else:
                        a, b = value[..., 0::2], value[..., 1::2]
                        out = torch.stack(
                            (
                                a * cosine[:, None] - b * sine[:, None],
                                b * cosine[:, None] + a * sine[:, None],
                            ),
                            -1,
                        ).flatten(-2)
                    return out.to(dtype).view_as(x)

                def calculate():
                    return backend.rotary_embedding(positions, query, key, dim, dim, table, neox)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        calculate()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = calculate()
                for replay in range(2):
                    qcpu.normal_()
                    kcpu.normal_()
                    query.copy_(qcpu)
                    key.copy_(kcpu)
                    poscpu = poscpu.flip(0)
                    positions.copy_(poscpu)
                    graph.replay()
                    torch.cuda.synchronize()
                    eager = calculate()
                    torch.cuda.synchronize()
                    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(captured, eager))
                    assert torch.equal(positions.cpu(), poscpu)
                    tolerance = (
                        0 if dtype == torch.bfloat16 else (0.01 if dtype == torch.float16 else 2e-6)
                    )
                    for output, cpu in zip(captured, (qcpu, kcpu)):
                        assert torch.allclose(
                            output.cpu().float(), reference(cpu).float(), atol=tolerance, rtol=1e-6
                        ), (dtype, pos_dtype, neox)
                    rows.append(
                        dict(
                            dtype=str(dtype),
                            pos_dtype=str(pos_dtype),
                            neox=neox,
                            replay=replay,
                            packed_q_stride=list(query.stride()),
                            packed_k_stride=list(key.stride()),
                            cpu_tolerance=tolerance,
                            dynamic_graph_exact=True,
                            output_hashes=[
                                hashlib.sha256(x.cpu().float().numpy().tobytes()).hexdigest()
                                for x in captured
                            ],
                        )
                    )
                del graph, captured
    print("ALL POSITION/STRIDE/GRAPH CHECKS PASSED", len(rows), flush=True)
    assert not any(backend.backend_fallback_counts().values())
    args.output.write_text(
        json.dumps(
            dict(native_position_capability=module.rope_supports_native_positions(), rows=rows),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
