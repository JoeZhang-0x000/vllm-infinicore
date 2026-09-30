#!/usr/bin/env python3
"""Correctness-gated, paired native/InfiniCore attention kernel benchmarks.

Example: python -m tests.benchmarks attention --platform metax --output results/kernels.json

Cases are phase:batch:query_length:kv_length, separated by commas. Inputs,
metadata and any native layout conversion are prepared before measurement.
There are no model weights, scheduler, RoPE, KV writes, or production patches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .common import compare_outputs, measure_graph, setup_device, source_fingerprints, write_json


def parse_case(value):
    phase, batch, qlen, klen = value.split(":")
    batch, qlen, klen = int(batch), int(qlen), int(klen)
    if phase not in ("prefill", "decode") or not (batch > 0 and 0 < qlen <= klen):
        raise ValueError(value)
    if phase == "decode" and qlen != 1:
        raise ValueError("decode requires one query token per sequence")
    return phase, batch, qlen, klen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("ascend", "metax", "kunlun"))
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--cases", help="phase:batch:query_length:kv_length,...")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-ms", type=float, default=50.)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--canonical-kv", action="store_true",
                        help="prepare contiguous BH(T)D KV for InfiniCore outside timing")
    parser.add_argument("--native-paged-prefill", action="store_true",
                        help="Kunlun control: use paged KV even for full prefill")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    assert args.repeats >= 3 and args.target_ms > 0
    assert args.heads % args.kv_heads == 0
    platform = args.platform
    api, device, dtype = setup_device(platform)
    if platform == "ascend":
        import torch_npu
    from vllm_infinicore.operators import attention_ops as ops
    ops.initialize()
    if args.cases:
        cases = [parse_case(value) for value in args.cases.split(",")]
    else:
        prefill_batches = (1, 2) if platform == "ascend" else (4, 16, 32)
        cases = [("prefill", b, 2048, 2048) for b in prefill_batches]
        cases += [("prefill", 16, 128, 3072)]
        cases += [("decode", b, 1, n) for b in (4, 16, 32) for n in (2048, 3072, 4096)]
    result = {
        "arguments": vars(args) | {"output": str(args.output)},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_sources": source_fingerprints(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__, "device_name": api.get_device_name(0),
        "dtype": str(dtype), "rows": [],
        "environment": {key: value for key, value in os.environ.items()
                        if key in ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "XPU_VISIBLE_DEVICES",
                                   "VLLM_INFINICORE_ROUTES", "VLLM_INFINICORE_CPP_BRIDGE_ROUTES",
                                   "VLLM_INFINICORE_ENABLE_PATCHES")},
        "measurement": "single-layer graph replay; native output copy included where needed",
        "correctness": "all-output finite/allclose plus sampled independent CPU FP32 causal attention",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result["infinicore_calls"] = ops.call_counts()
        write_json(args.output, result)

    def compare(actual, expected):
        return compare_outputs(actual, expected, dtype=dtype)

    def measure(fn, output, validated_output, phase):
        return measure_graph(fn, validated_output, api=api, platform=platform,
                             dtype=dtype, unroll=8 if phase == "decode" else 1,
                             repeats=args.repeats, target_ms=args.target_ms,
                             output=output, nan_sentinel=True)

    def run_case(index, phase, batch, qlen, klen):
        torch.manual_seed(4300 + index)
        h, hk, d = args.heads, args.kv_heads, 128
        block = 16 if platform == "metax" else 128
        pages = (klen+block-1)//block
        row = {"phase": phase, "batch": batch, "query_len": qlen, "kv_len": klen,
               "query_heads": h, "kv_heads": hk, "head_dim": d, "block_size": block,
               "query_tokens": batch*qlen, "status": "running", "seed": 4300+index}
        result["rows"].append(row)
        save()
        print(json.dumps({"starting": row}), flush=True)
        shape = (2, batch*pages, hk, block, d) if platform == "kunlun" else (2, batch*pages, block, hk, d)
        cache = torch.randn(shape, dtype=dtype, device=device)
        k, v = ops.cache_views(cache, hk)
        if args.canonical_kv:
            k, v = k.contiguous(), v.contiguous()
        q = torch.randn(batch*qlen, h, d, dtype=dtype, device=device)
        outputs = {name: torch.full_like(q, float("nan")) for name in ("infinicore", "native")}
        blocks = torch.arange(batch*pages, dtype=torch.int32, device=device).reshape(batch, pages)
        lens_cpu = torch.full((batch,), klen, dtype=torch.int32)
        lens = lens_cpu.to(device)
        starts_cpu = torch.arange(batch+1, dtype=torch.int32)*qlen
        starts = starts_cpu.to(device)
        kv_starts_cpu = torch.arange(batch+1, dtype=torch.int32)*klen
        kv_starts = kv_starts_cpu.to(device)
        row["q_stride"] = list(q.stride())
        row["kv_logical_stride"] = list(k.stride())

        def infini():
            ops.compute(q, k, v, blocks, lens, starts if phase == "prefill" else None,
                        d**-.5, outputs["infinicore"], decode=phase == "decode")

        if platform == "metax":
            from vllm_metax.v1.attention.backends.fa_utils import flash_attn_varlen_func, flash_attn_with_kvcache
            row["native_entry"] = "flash_attn_varlen_func (paged)" if phase == "prefill" else "flash_attn_with_kvcache"
            def native():
                if phase == "prefill":
                    x = flash_attn_varlen_func(q=q, k=cache[0], v=cache[1],
                        cu_seqlens_q=starts, cu_seqlens_k=kv_starts,
                        max_seqlen_q=qlen, max_seqlen_k=klen, softmax_scale=d**-.5,
                        causal=True, block_table=blocks)
                else:
                    x = flash_attn_with_kvcache(q=q[:, None], k_cache=cache[0], v_cache=cache[1],
                        block_table=blocks, cache_seqlens=lens, softmax_scale=d**-.5, causal=True)[:, 0]
                outputs["native"].copy_(x)
        elif platform == "kunlun":
            import kunlun_ops
            prefix = qlen < klen or args.native_paged_prefill
            if phase == "prefill" and not prefix:
                # The native full-prefill path consumes the current dense K/V.
                # Recover those identical values outside the timed region.
                dense_k = k.transpose(1, 2).reshape(batch, pages*block, hk, d)[:, :klen].reshape(-1, hk, d).contiguous()
                dense_v = v.transpose(1, 2).reshape(batch, pages*block, hk, d)[:, :klen].reshape(-1, hk, d).contiguous()
            row["native_entry"] = ("kunlun_ops.speculative_attention" if phase == "decode" else
                                   "kunlun_ops.prefill_attention (paged prefix)" if prefix else
                                   "kunlun_ops.prefill_attention (dense current KV)")
            def native():
                if phase == "decode":
                    kunlun_ops.speculative_attention(out=outputs["native"], q=q.unsqueeze(0),
                        k_cache=k, v_cache=v, context_lens_cpu=lens_cpu, context_lens_xpu=lens,
                        batch_num=batch, qlen=1, max_context_len=131072, head_num=h,
                        head_dim=d, scale=0., kv_head_num=hk, block_size=block,
                        max_num_blocks_per_seq=pages, max_window_size=-1, block_tables=blocks)
                elif prefix:
                    kunlun_ops.prefill_attention(q=q, k=k, v=v, out=outputs["native"],
                        is_causal=True, is_prefix_cache=True, block_table=blocks,
                        context_qlen_lod_cpu=starts_cpu, context_qlen_lod_xpu=starts,
                        context_kvlen_lod_cpu=kv_starts_cpu, context_kvlen_lod_xpu=kv_starts)
                else:
                    kunlun_ops.prefill_attention(q=q, k=dense_k, v=dense_v,
                        out=outputs["native"], is_causal=True,
                        context_qlen_lod_cpu=starts_cpu, context_qlen_lod_xpu=starts)
        else:
            native_k = cache[0].view(batch*pages, block, hk*d)
            native_v = cache[1].view(batch*pages, block, hk*d)
            mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).to(device)
            q_ends, kv_lens = starts_cpu[1:].tolist(), lens_cpu.tolist()
            row["native_entry"] = "torch_npu.npu_fused_infer_attention_score (paged TND)"
            def native():
                x, _ = torch_npu.npu_fused_infer_attention_score(query=q, key=native_k, value=native_v,
                    atten_mask=mask if phase == "prefill" else None, block_table=blocks,
                    input_layout="TND", block_size=block, actual_seq_lengths=q_ends,
                    actual_seq_lengths_kv=kv_lens, num_key_value_heads=hk, num_heads=h,
                    scale=d**-.5, sparse_mode=3 if phase == "prefill" else 0)
                outputs["native"].copy_(x)

        cpu_outputs, checks = {}, {}
        for name, fn in (("native", native), ("infinicore", infini)):
            api.synchronize()
            fn()
            api.synchronize()
            cpu_outputs[name] = outputs[name].cpu()
            finite_heads = torch.isfinite(cpu_outputs[name]).all(dim=-1)
            checks[name] = {"finite": bool(finite_heads.all()),
                            "finite_query_heads": int(finite_heads.sum()),
                            "total_query_heads": batch*qlen*h}
        pair = compare(cpu_outputs["infinicore"], cpu_outputs["native"])
        # The reference does not use either vendor attention implementation.
        # Test first/middle/last sequences, query positions and GQA groups.
        sampled = 0
        ref_max = {"infinicore": 0., "native": 0.}
        ref_pass = {"infinicore": True, "native": True}
        for seq in sorted({0, batch//2, batch-1}):
            for head in sorted({0, h//2, h-1}):
                kh = head//(h//hk)
                kc = k[seq*pages:(seq+1)*pages, kh].reshape(-1, d)[:klen].float().cpu()
                vc = v[seq*pages:(seq+1)*pages, kh].reshape(-1, d)[:klen].float().cpu()
                for pos in sorted({0, qlen//2, qlen-1}):
                    attended = klen-qlen+pos+1
                    qc = q[seq*qlen+pos, head].float().cpu()
                    ref = ((kc[:attended]@qc)*d**-.5).softmax(0)@vc[:attended]
                    for name, actual in cpu_outputs.items():
                        check = compare(actual[seq*qlen+pos, head], ref)
                        ref_pass[name] &= check["passed"]
                        ref_max[name] = max(ref_max[name], check["max_abs_error"])
                    sampled += 1
        for name in checks:
            checks[name].update(cpu_fp32_sampled_query_heads=sampled,
                                cpu_fp32_max_abs_error=ref_max[name],
                                cpu_fp32_passed=ref_pass[name])
        row["checks"], row["pair_check"] = checks, pair
        passed = pair["passed"] and all(c["finite"] and c["cpu_fp32_passed"] for c in checks.values())
        if not passed:
            row["status"] = "correctness_failed"
        elif args.check_only:
            row["status"] = "passed_check_only"
        else:
            row["timings"] = {}
            # Alternate measurement order across cases; both use one device.
            order = (("native", native), ("infinicore", infini))
            if index % 2:
                order = tuple(reversed(order))
            for name, fn in order:
                row["timings"][name] = measure(fn, outputs[name], cpu_outputs[name], phase)
                save()
            if all(t["status"] == "passed" for t in row["timings"].values()):
                metric = ("graph_device_ms" if all(t["graph_device_ms"] is not None
                          for t in row["timings"].values()) else "graph_wall_ms")
                inf_ms, native_ms = (row["timings"][n][metric] for n in ("infinicore", "native"))
                row.update(status="passed", ratio_metric=metric,
                           latency_ratio_infinicore_over_native=inf_ms/native_ms,
                           query_tps_ratio_infinicore_over_native=native_ms/inf_ms,
                           infinicore_query_tokens_per_second=batch*qlen*1000/inf_ms,
                           native_query_tokens_per_second=batch*qlen*1000/native_ms)
            else:
                row["status"] = "graph_correctness_failed"
        save()
        print(json.dumps(row, allow_nan=False), flush=True)

    for index, case in enumerate(cases):
        try:
            run_case(index, *case)
        except Exception:
            error = traceback.format_exc()
            result["rows"][-1].update(status="error", error=error)
            save()
            print(error, flush=True)
            # Device errors may poison the process; preserve evidence and stop.
            result["stopped_after_error"] = True
            break
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    return 0 if len(result["rows"]) == len(cases) and all(
        row["status"] in ("passed", "passed_check_only") for row in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
