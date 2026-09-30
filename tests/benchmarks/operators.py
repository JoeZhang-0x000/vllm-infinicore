#!/usr/bin/env python3
"""Paired native/InfiniCore benchmarks for the operators around attention.

No model weights or vLLM scheduler are used. Disable production patches and
select the supported routes with VLLM_INFINICORE_CPP_BRIDGE_ROUTES.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from .common import (cache_rope_tables, compare_outputs, cpu, measure_graph,
                     microseconds, setup_device, source_fingerprints, write_json)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", required=True, choices=("ascend", "metax", "kunlun"))
    p.add_argument("--tokens", default="4,16,32,2048")
    p.add_argument("--ops", default="StoreKVCache,RoPE,RMSNorm,SiluAndMul,Embedding,MatMul,LMHead")
    p.add_argument("--cached-rope", action="store_true")
    p.add_argument("--dense-kv", action="store_true", help="KV store control: prepare dense K/V before timing")
    p.add_argument("--sequential-slots", action="store_true")
    p.add_argument("--no-padding", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    import torch
    import torch.nn.functional as F

    api, device, dtype = setup_device(args.platform)
    if args.platform == "ascend":
        import torch_npu
        from vllm_infinicore.operators.ascend import backend
        import vllm_ascend.ops
        from vllm_ascend.device.device_op import DeviceOperator
        h, hk, hidden, intermediate, vocab = 28, 4, 3584, 18944, 152064
    else:
        from vllm_infinicore.operators import backend, cpp_bridge
        h, hk, hidden, intermediate, vocab = 32, 8, 4096, 12288, 151936
    from vllm_infinicore.operators import attention_ops as attention
    attention.initialize()
    if args.cached_rope:
        cache_rope_tables(args.platform)
    if args.platform == "metax":
        from vllm_metax.v1.attention.backends.fa_utils import reshape_and_cache_flash
        from vllm import _custom_ops as native_ops
    elif args.platform == "kunlun":
        import kunlun_ops
        import vllm_kunlun.vllm_utils_wrapper
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as native_ops

    result = {"arguments": vars(args) | {"output": str(args.output)}, "rows": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_sources": source_fingerprints(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dtype": str(dtype),
        "geometry": dict(heads=h, kv_heads=hk, head_dim=128, hidden=hidden, intermediate=intermediate, vocab=vocab),
        "environment": {k: v for k, v in os.environ.items() if k in (
            "VLLM_INFINICORE_CPP_BRIDGE_ROUTES", "VLLM_INFINICORE_ROUTES", "VLLM_INFINICORE_ENABLE_PATCHES",
            "ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "XPU_VISIBLE_DEVICES")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        result["attention_calls"] = attention.call_counts()
        if args.platform != "ascend":
            result["bridge_calls"] = cpp_bridge.bridge_call_counts()
            result["backend_calls"] = backend.backend_call_counts()
        write_json(args.output, result)

    def flat_cpu(value):
        parts = value if isinstance(value, (tuple, list)) else (value,)
        return list(cpu(parts))

    def check(a, b, exact=False):
        return compare_outputs(a, b, dtype=dtype, exact=exact)

    def measure(fn, reference, exact=False):
        return microseconds(measure_graph(fn, reference, api=api, platform=args.platform,
                                         dtype=dtype, unroll=4, warmup=2, exact=exact))

    def run(op, n, variant=None):
        row = dict(operator=op, tokens=n, variant=variant, status="running")
        result["rows"].append(row)
        save()
        torch.manual_seed(912+n)
        if args.platform == "kunlun" and op in ("RMSNorm", "SiluAndMul"):
            row.update(status="unsupported", reason="not exposed by the installed Kunlun adapter")
            return
        exact = op in ("StoreKVCache", "Embedding")
        if op == "StoreKVCache":
            block = 16 if args.platform == "metax" else 128
            capacity = math.ceil((n*2+128)/block)*block
            shape = (2, capacity//block, hk, block, 128) if args.platform == "kunlun" else (2, capacity//block, block, hk, 128)
            caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(2)]
            ik, iv = attention.cache_views(caches[0], hk)
            nk, nv = attention.cache_views(caches[1], hk)
            packed = torch.randn(n, h+2*hk, 128, dtype=dtype, device=device)
            key, value = packed[:, h:h+hk], packed[:, h+hk:]
            if args.dense_kv:
                key, value = key.contiguous(), value.contiguous()
            slots_cpu = (torch.arange(n) if args.sequential_slots else torch.randperm(capacity)[:n]).to(torch.int64)
            if n >= 32 and not args.no_padding:
                slots_cpu[::17] = -1
            slots = slots_cpu.to(device)
            slots32 = slots.to(torch.int32)
            scale = torch.ones((), dtype=torch.float32, device=device)
            def infini():
                attention.store(ik, iv, key, value, slots)
                return caches[0]
            def native():
                if args.platform == "ascend":
                    DeviceOperator.reshape_and_cache(key=key, value=value,
                        key_cache=caches[1][0], value_cache=caches[1][1], slot_mapping=slots32)
                elif args.platform == "metax":
                    reshape_and_cache_flash(key, value, caches[1][0], caches[1][1], slots, "auto", scale, scale)
                else:
                    kunlun_ops.reshape_and_cache_flash(key, value, nk, nv, slots, BLHD_LAYOUT=False)
                return caches[1]
            expected = torch.zeros(2, capacity//block, hk, block, 128, dtype=dtype)
            kc, vc = key.cpu(), value.cpu()
            for index, slot in enumerate(slots_cpu.tolist()):
                if slot >= 0:
                    page, offset = divmod(slot, block)
                    expected[0, page, :, offset] = kc[index]
                    expected[1, page, :, offset] = vc[index]
            if args.platform != "kunlun":
                expected = expected.transpose(2, 3)
            row["block_size"], row["key_stride"] = block, list(key.stride())
        elif op == "RoPE":
            packed = torch.randn(n, h+2*hk, 128, dtype=dtype, device=device)
            q, key = packed[:, :h], packed[:, h:h+hk]
            pos = torch.arange(n, dtype=torch.int64, device=device) % 2048 + 1024
            inv = 1/(1e6**(torch.arange(0, 128, 2).float()/128))
            angles = torch.arange(40960).float()[:, None]*inv[None]
            table = torch.cat((angles.cos(), angles.sin()), -1).to(device=device, dtype=dtype)
            # Native kernels may mutate Q/K; include identical fresh copies on
            # both paths so repeated graph calls do not accumulate rotations.
            def infini():
                return backend.rotary_embedding(pos, q.clone(), key.clone(), 128, 128, table, True)
            def native():
                qq, kk = q.clone().reshape(n, -1), key.clone().reshape(n, -1)
                if args.platform == "ascend":
                    torch_npu._npu_rotary_embedding(pos, qq, kk, 128, table, True)
                else:
                    native_ops.rotary_embedding(pos, qq, kk, 128, table, True)
                return qq.view_as(q), kk.view_as(key)
            selected = table.cpu()[pos.cpu()].float()
            cos, sin = selected.chunk(2, -1)
            def rope_ref(x):
                a, b = x.float().cpu().chunk(2, -1)
                return torch.cat((a*cos[:, None]-b*sin[:, None], b*cos[:, None]+a*sin[:, None]), -1)
            expected = (rope_ref(q), rope_ref(key))
            row["symmetric_qk_clone_included"] = True
        elif op == "RMSNorm":
            x = torch.randn(n, hidden, dtype=dtype, device=device)
            weight = torch.ones(hidden, dtype=dtype, device=device)
            native_out = torch.empty_like(x)
            infini = lambda: backend.rms_norm(x, weight, 1e-6)
            def native():
                if args.platform == "ascend":
                    return torch_npu.npu_rms_norm(x, weight, epsilon=1e-6)[0]
                native_ops.rms_norm(native_out, x, weight, 1e-6)
                return native_out
            xx = x.float().cpu()
            expected = xx*torch.rsqrt(xx.square().mean(-1, keepdim=True)+1e-6)
        elif op == "SiluAndMul":
            x = torch.randn(n, 2*intermediate, dtype=dtype, device=device)
            if args.platform == "ascend" and not backend.supports_silu_and_mul(x)[0]:
                row.update(status="unsupported", reason=backend.supports_silu_and_mul(x)[1])
                return
            native_out = torch.empty(n, intermediate, dtype=dtype, device=device)
            infini = lambda: backend.silu_and_mul(x)
            def native():
                if args.platform == "ascend":
                    return torch_npu.npu_swiglu(x)
                torch.ops._C.silu_and_mul(native_out, x)
                return native_out
            gate, up = x.float().cpu().chunk(2, -1)
            expected = F.silu(gate)*up
        else:
            if op in ("Embedding", "LMHead"):
                kdim, ndim = hidden, vocab
            else:
                kdim, ndim = {"qkv": (hidden, (h+2*hk)*128), "gate_up": (hidden, 2*intermediate),
                              "down": (intermediate, hidden), "o_proj": (hidden, hidden)}[variant]
            weight = torch.randn(ndim, kdim, dtype=dtype, device=device)*.02
            if op == "Embedding":
                ids = torch.randint(0, vocab, (n,), dtype=torch.int64, device=device)
                infini = lambda: backend.embedding(ids, weight)
                native = lambda: F.embedding(ids, weight)
                expected = weight[ids].cpu()
            else:
                x = torch.randn(n, kdim, dtype=dtype, device=device)
                inf_fn = backend.linear if args.platform == "ascend" or op == "MatMul" else backend.lm_head
                infini = lambda: inf_fn(x, weight)
                native = lambda: F.linear(x, weight)
                # Independent CPU FP32 sample; all output elements are still
                # compared against native separately below.
                rows = sorted({0, n//2, n-1})
                columns = torch.linspace(0, ndim-1, 17).long()
                expected = x[rows].float().cpu() @ weight[columns].float().cpu().t()
                row["linear_shape"] = [n, ndim, kdim]

        native_value, infini_value = native(), infini()
        api.synchronize()
        nc, ic = flat_cpu(native_value), flat_cpu(infini_value)
        row["pair_check"] = check(ic, nc, exact)
        if op in ("MatMul", "LMHead"):
            samples = [a[0][rows][:, columns] for a in (ic, nc)]
            row["reference_checks"] = [check([a], [expected]) for a in samples]
        else:
            reference = list(expected) if isinstance(expected, tuple) else [expected]
            row["reference_checks"] = [check(a, reference, exact) for a in (ic, nc)]
        if not row["pair_check"]["passed"] or not all(c["passed"] for c in row["reference_checks"]):
            row["status"] = "correctness_failed"
            return
        row["timings"] = {"native": measure(native, nc, exact), "infinicore": measure(infini, ic, exact)}
        if not all(t["status"] == "passed" for t in row["timings"].values()):
            row["status"] = "graph_check_failed"
            return
        metric = ("graph_device_us" if args.platform != "kunlun" and all(
            t["graph_device_us"] is not None for t in row["timings"].values()) else "graph_wall_us")
        it, nt = (row["timings"][name][metric] for name in ("infinicore", "native"))
        row.update(status="passed", metric=metric, latency_ratio=it/nt, extra_us=it-nt)

    for op in args.ops.split(","):
        for n in map(int, args.tokens.split(",")):
            if op == "LMHead" and n > 32:
                continue  # Only last tokens need logits in ordinary prefill.
            for variant in ("qkv", "gate_up", "down", "o_proj") if op == "MatMul" else (None,):
                try:
                    run(op, n, variant)
                except Exception:
                    error = traceback.format_exc()
                    result["rows"][-1].update(status="error", error=error)
                    if any(text in error for text in ("launch failure", "device-side", "illegal memory")):
                        result["stopped_after_device_error"] = True
                        save()
                        print(error, flush=True)
                        return 1
                save()
                print(json.dumps(result["rows"][-1], allow_nan=False), flush=True)
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    return int(any(r["status"] not in ("passed", "unsupported") for r in result["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
