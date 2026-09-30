#!/usr/bin/env python3
"""Kunlun operator accuracy stress test with complete independent CPU references.

Run inside the pinned Kunlun environment, with production patches disabled but
MatMul/LMHead/RoPE/Embedding/StoreKV C++ routes selected. No model is loaded.
Floating checks use |actual-reference| <= 0.005 + 0.02*|reference|;
StoreKV and Embedding require exact equality. Failed checks remain in the JSON.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import time
import traceback

from .common import cpu, setup_device, source_fingerprints, write_json


def as_array(value):
    import numpy as np
    import torch

    if isinstance(value, (tuple, list)):
        return np.concatenate([as_array(part).reshape(-1) for part in value])
    return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def error_stats(actual, reference, exact=False):
    """All elements, bounded temporary memory; relative L2 avoids near-zero bias."""
    import numpy as np

    a, b = as_array(actual).reshape(-1), as_array(reference).reshape(-1)
    assert a.shape == b.shape, (a.shape, b.shape)
    total = a.size
    failed = different = nonfinite = valid = 0
    sum_abs = sum_square = sum_ref_square = max_abs = max_rel = 0.0
    worst = []
    for start in range(0, total, 1 << 20):
        x = a[start:start + (1 << 20)].astype(np.float64)
        y = b[start:start + (1 << 20)].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        delta = np.abs(x - y)
        limit = 0.0 if exact else 0.005 + 0.02 * np.abs(y)
        failed += int(np.count_nonzero(~finite | (delta > limit)))
        different += int(np.count_nonzero(x != y))
        nonfinite += int(np.count_nonzero(~finite))
        valid += int(np.count_nonzero(finite))
        if not finite.any():
            continue
        dif, ref = delta[finite], y[finite]
        max_abs = max(max_abs, float(dif.max()))
        sum_abs += float(dif.sum(dtype=np.float64))
        sum_square += float(np.dot(dif, dif))
        sum_ref_square += float(np.dot(ref, ref))
        mask = finite & (np.abs(y) >= 0.005)
        if mask.any():
            max_rel = max(max_rel, float((delta[mask] / np.abs(y[mask])).max()))
        candidates = np.where(finite, delta, -1.0)
        indices = np.argpartition(candidates, -min(4, candidates.size))[-4:]
        worst.extend({"flat_index": start + int(i), "actual": float(x[i]),
                      "reference": float(y[i]), "abs_error": float(delta[i])}
                     for i in indices if finite[i])
        worst = sorted(worst, key=lambda v: v["abs_error"], reverse=True)[:4]
    return {
        "elements": int(total), "failed_elements": failed,
        "failed_percent": 100.0 * failed / total,
        "different_elements": different, "nonfinite_elements": nonfinite,
        "max_abs_error": max_abs if valid else None,
        "mean_abs_error": sum_abs / valid if valid else None,
        "rmse": math.sqrt(sum_square / valid) if valid else None,
        "relative_l2": math.sqrt(sum_square / sum_ref_square) if sum_ref_square else None,
        "max_relative_error_ref_abs_ge_0_005": max_rel,
        "atol": 0.0 if exact else 0.005, "rtol": 0.0 if exact else 0.02,
        "passed": failed == 0, "worst_elements": worst,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ops", default="MatMul,StoreKVCache,RoPE,Embedding,LMHead")
    parser.add_argument("--tokens", default="16,2048")
    parser.add_argument("--store-tokens", default="16,2048,4096")
    parser.add_argument("--seed-offsets", default="0,1,2")
    parser.add_argument("--graph-replays", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--extra-down-4096", action="store_true")
    args = parser.parse_args(argv)
    if args.graph_replays < 1 or args.threads < 1:
        parser.error("--graph-replays and --threads must be positive")
    import torch
    import torch.nn.functional as F

    api, device, dtype = setup_device("kunlun", threads=args.threads)
    from vllm_infinicore.operators import attention_ops, backend, cpp_bridge
    import kunlun_ops
    attention_ops.initialize()
    # These are the Qwen3-8B single-TP geometries from the model benchmark.
    h, hk, dim, hidden, intermediate, vocab = 32, 8, 128, 4096, 12288, 151936
    dimensions = {"qkv": (hidden, (h + 2 * hk) * dim),
                  "gate_up": (hidden, 2 * intermediate),
                  "down": (intermediate, hidden), "o_proj": (hidden, hidden)}
    versions = {}
    for name in ("torch", "xmlir", "torch-xmlir", "kunlun-ops", "infinicore", "vllm-kunlun"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    result = {
        "arguments": vars(args) | {"output": str(args.output)},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_sources": source_fingerprints(),
        "versions": versions, "device": api.get_device_name(0), "dtype": str(dtype),
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_INFINICORE", "XMLIR", "XPU_VISIBLE"))},
        "reference": "all-output CPU FP64 from the exact same quantized FP16 inputs; exact CPU indices for StoreKV/Embedding",
        "rows": [], "completed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result.update(backend_calls=backend.backend_call_counts(),
                      fallback_calls=backend.backend_fallback_counts(),
                      bridge_calls=cpp_bridge.bridge_call_counts(),
                      attention_calls=attention_ops.call_counts())
        write_json(args.output, result)

    def compare_paths(values, expected, exact):
        return {"infinicore_vs_native": error_stats(values["infinicore"], values["native"], exact),
                "infinicore_vs_reference": error_stats(values["infinicore"], expected, exact),
                "native_vs_reference": error_stats(values["native"], expected, exact)}

    def capture_stress(fn, eager, store=False):
        def invalidate(value):
            if isinstance(value, (tuple, list)):
                for part in value:
                    invalidate(part)
            elif store:
                value.zero_()  # Unwritten cache slots must remain zero.
            else:
                value.fill_(float("nan"))
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        invalidate(out)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        first = cpu(out, clone=True)
        begin = time.perf_counter()
        for _ in range(max(0, args.graph_replays - 2)):
            graph.replay()
        if args.graph_replays > 1:
            torch.cuda.synchronize()
            invalidate(out)
            graph.replay()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        last = cpu(out, clone=True)
        return last, {
            "replays": args.graph_replays,
            "write_verified_with_sentinel": "zeroed cache" if store else "NaN output",
            "timing_includes_last_replay_sentinel_reset": True,
            "replay_wall_us_observed": elapsed * 1e6 / max(1, args.graph_replays - 1),
            "first_vs_eager": error_stats(first, eager, exact=True),
            "last_vs_first": error_stats(last, first, exact=True),
        }

    def run(op, n, variant, seed):
        row = dict(operator=op, tokens=n, variant=variant, seed=seed, status="running")
        result["rows"].append(row)
        save()
        print("START " + json.dumps({k: row[k] for k in ("operator", "tokens", "variant", "seed")}), flush=True)
        torch.manual_seed(seed)
        exact = op in ("StoreKVCache", "Embedding")
        if op in ("MatMul", "LMHead"):
            kdim, ndim = dimensions[variant] if op == "MatMul" else (hidden, vocab)
            weight = torch.randn(ndim, kdim, device=device, dtype=dtype) * .02
            x = torch.randn(n, kdim, device=device, dtype=dtype)
            inf_fn = backend.linear if op == "MatMul" else backend.lm_head
            infini, native = lambda: inf_fn(x, weight), lambda: F.linear(x, weight)
            ref_start = time.perf_counter()
            expected = x.cpu().double() @ weight.cpu().double().t()
            row.update(shape_mnk=[n, ndim, kdim], reference_seconds=time.perf_counter() - ref_start,
                       input_distribution="FP16 N(0,1); weight=FP16 N(0,1)*0.02, rounded on device")
        elif op == "StoreKVCache":
            block = 128
            capacity = math.ceil((2 * n + block) / block) * block
            shape = (2, capacity // block, hk, block, dim)
            caches = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(2)]
            packed = torch.randn(n, h + 2 * hk, dim, device=device, dtype=dtype)
            key, val = packed[:, h:h + hk], packed[:, h + hk:]
            if variant.startswith("dense"):
                key, val = key.contiguous(), val.contiguous()
            elif variant == "vllm_native_v_contiguous_padded":
                # Qwen3 applies K norm before attention, producing a dense K.
                # Its native attention path then makes V dense as well.
                key = key.contiguous()
            slots_cpu = torch.arange(n) if "sequential" in variant else torch.randperm(capacity)[:n]
            if variant.endswith("_padded") and n >= 32:
                slots_cpu[::17] = -1
            slots = slots_cpu.long().to(device)
            native_val = val.contiguous() if variant == "vllm_native_v_contiguous_padded" else val
            expected = torch.zeros(shape, dtype=dtype)
            kc, vc = key.cpu(), val.cpu()
            for i, slot in enumerate(slots_cpu.tolist()):
                if slot >= 0:
                    page, offset = divmod(slot, block)
                    expected[0, page, :, offset] = kc[i]
                    expected[1, page, :, offset] = vc[i]
            def infini():
                attention_ops.store(caches[0][0], caches[0][1], key, val, slots)
                return caches[0]
            def native():
                kunlun_ops.reshape_and_cache_flash(key, native_val, caches[1][0], caches[1][1], slots, BLHD_LAYOUT=False)
                return caches[1]
            row.update(key_stride=list(key.stride()), value_stride=list(val.stride()), native_value_stride=list(native_val.stride()), cache_shape=list(shape),
                       valid_tokens=int((slots_cpu >= 0).sum()), padded_tokens=int((slots_cpu < 0).sum()),
                       compared_region="entire cache, including untouched slots")
        elif op == "Embedding":
            weight = torch.randn(vocab, hidden, device=device, dtype=dtype) * .02
            ids = torch.randint(0, vocab, (n,), device=device, dtype=torch.int64)
            infini, native = lambda: backend.embedding(ids, weight), lambda: F.embedding(ids, weight)
            expected = weight.cpu()[ids.cpu()]
        elif op == "RoPE":
            packed = torch.randn(n, h + 2 * hk, dim, device=device, dtype=dtype)
            q, key = packed[:, :h], packed[:, h:h + hk]
            positions_cpu = torch.arange(n) % 2048 + 1024
            positions = positions_cpu.long().to(device)
            inv = 1 / (1e6 ** (torch.arange(0, dim, 2).float() / dim))
            angles = torch.arange(40960).float()[:, None] * inv[None]
            table = torch.cat((angles.cos(), angles.sin()), -1).to(device=device, dtype=dtype)
            infini = lambda: backend.rotary_embedding(positions, q.clone(), key.clone(), dim, dim, table, True)
            # Same native entry as KunlunOps.rotary_embedding, without importing vLLM.
            def native():
                qq, kk = q.clone().reshape(n, -1), key.clone().reshape(n, -1)
                kunlun_ops.rotary_embedding(positions, qq, kk, dim, table, True)
                return qq.view_as(q), kk.view_as(key)
            cos, sin = table.cpu()[positions_cpu].double().chunk(2, -1)
            def rotate_ref(value):
                a, b = value.cpu().double().chunk(2, -1)
                return torch.cat((a * cos[:, None] - b * sin[:, None], b * cos[:, None] + a * sin[:, None]), -1)
            expected = (rotate_ref(q), rotate_ref(key))
        else:
            raise ValueError(op)
        values = {}
        for name, fn in (("native", native), ("infinicore", infini)):
            values[name] = cpu(fn(), clone=True)
        row["eager"] = compare_paths(values, expected, exact)
        row["graph_stability"] = {}
        graph_values = {}
        for name, fn in (("native", native), ("infinicore", infini)):
            graph_values[name], row["graph_stability"][name] = capture_stress(fn, values[name], store=op == "StoreKVCache")
        row["graph"] = compare_paths(graph_values, expected, exact)
        row["status"] = "passed" if all(
            check["passed"] for stage in ("eager", "graph") for check in row[stage].values()
        ) else "numerical_difference"
        row["graph_bitwise_stable"] = all(
            value["first_vs_eager"]["passed"] and value["last_vs_first"]["passed"]
            for value in row["graph_stability"].values())
        if not row["graph_bitwise_stable"]:
            row["status"] = "graph_changed_output"
        save()
        print("DONE " + json.dumps({"operator": op, "tokens": n, "variant": variant, "seed": seed,
              "status": row["status"], "max_abs": {name: check["max_abs_error"] for name, check in row["graph"].items()},
              "failed": {name: check["failed_elements"] for name, check in row["graph"].items()}}), flush=True)

    jobs = []
    tokens, offsets = [int(v) for v in args.tokens.split(",")], [int(v) for v in args.seed_offsets.split(",")]
    for op in args.ops.split(","):
        sizes = [int(v) for v in args.store_tokens.split(",")] if op == "StoreKVCache" else [16] if op == "LMHead" else tokens
        variants = list(dimensions) if op == "MatMul" else ["strided_random_padded", "vllm_native_v_contiguous_padded", "dense_random_padded", "strided_random_unpadded", "strided_sequential_unpadded"] if op == "StoreKVCache" else [None]
        jobs.extend((op, n, variant, 912 + n + offset) for n in sizes for variant in variants for offset in offsets)
        if op == "MatMul" and args.extra_down_4096 and 4096 not in sizes:
            jobs.extend((op, 4096, "down", 5008 + offset) for offset in offsets)
    result["planned_cases"] = len(jobs)
    for job in jobs:
        try:
            run(*job)
        except Exception:
            error = traceback.format_exc()
            result["rows"][-1].update(status="error", error=error)
            save()
            print(error, flush=True)
            if any(s in error.lower() for s in ("device-side", "illegal memory", "launch failure")):
                break
        backend.clear_tensor_wrapper_cache()
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = len(result["rows"]) == len(jobs) and not any(r["status"] in ("error", "running") for r in result["rows"])
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    return int(not result["completed"] or any(r["status"] != "passed" for r in result["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
