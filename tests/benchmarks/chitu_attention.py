"""Measure unmodified Chitu Triton attention against MetaX FlashAttention.

Point --chitu-source at a pinned Chitu checkout (or the four-file source
snapshot). The engine's package initializers are skipped; device detection,
autotuning helpers and both attention kernels are loaded without modification.
Prefill reports dense computation and paged-to-dense materialization separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

from .attention import parse_case
from .common import compare_outputs, measure_graph, setup_device, source_fingerprints, write_json

SOURCE_FILES = (
    "chitu/device_type.py",
    "chitu/ops/triton_ops/utils.py",
    "chitu/ops/triton_ops/attn/decode.py",
    "chitu/ops/triton_ops/attn/prefill.py",
)


def load_chitu(root):
    root = Path(root).resolve()
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Chitu source is missing {missing}")
    for name in ("chitu", "chitu.ops", "chitu.ops.triton_ops", "chitu.ops.triton_ops.attn"):
        package = types.ModuleType(name)
        package.__path__ = [str(root / name.replace(".", "/"))]
        sys.modules[name] = package
    modules = {}
    for name in SOURCE_FILES:
        module_name = name[:-3].replace("/", ".")
        spec = importlib.util.spec_from_file_location(module_name, root / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chitu-source", type=Path, required=True)
    parser.add_argument("--chitu-revision", required=True)
    parser.add_argument("--cases", default="decode:4:1:2048,decode:16:1:3072,decode:32:1:4096,"
                        "prefill:1:2048:2048,prefill:4:2048:2048,prefill:16:2048:2048,"
                        "prefill:16:128:3072")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=100.)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--ragged", action="store_true", help="also vary per-request Q/KV lengths")
    parser.add_argument("--sequential-pages", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.heads % args.kv_heads or args.repeats < 3 or args.target_ms <= 0:
        parser.error("need divisible head counts, >=3 repeats and positive target-ms")
    import torch
    import triton
    api, device, _ = setup_device("metax")
    dtype = getattr(torch, args.dtype)
    modules = load_chitu(args.chitu_source)
    if not modules[SOURCE_FILES[0]].is_muxi():
        raise RuntimeError("this benchmark requires a MetaX device")
    decode = modules[SOURCE_FILES[2]]
    prefill = modules[SOURCE_FILES[3]]
    from vllm_metax.v1.attention.backends.fa_utils import flash_attn_varlen_func, flash_attn_with_kvcache
    cases = [parse_case(value) for value in args.cases.split(",")]
    result = {
        "arguments": vars(args) | {"chitu_source": str(args.chitu_source), "output": str(args.output)},
        "chitu_repository": "https://github.com/thu-pacman/chitu",
        "chitu_revision": args.chitu_revision,
        "upstream_sources": {name: hashlib.sha256((args.chitu_source / name).read_bytes()).hexdigest()
                             for name in SOURCE_FILES},
        "benchmark_sources": source_fingerprints(),
        "torch_version": torch.__version__, "triton_version": triton.__version__,
        "device_name": api.get_device_name(0),
        "device_properties": str(api.get_device_properties(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "measurement": "single-layer graph device events; JIT/autotuning/warmup excluded; no output copies",
        "correctness": "all-output finite/allclose vs native plus independent CPU FP32 samples; graph replay checked",
        "layout": "physical K/V [page, token, kv_head, dimension]; shuffled physical pages by default",
        "rows": [],
    }

    def save():
        write_json(args.output, result)

    def compare(actual, expected):
        return compare_outputs(actual, expected, dtype=dtype)

    def run_case(index, phase, batch, qlen, klen):
        torch.manual_seed(4300 + index)
        h, hk, d, block = args.heads, args.kv_heads, args.head_dim, args.block_size
        qlens = [qlen] * batch
        klens = [klen] * batch
        if args.ragged:
            qlens = [max(1, qlen - (i % 3) * max(1, qlen // 5)) for i in range(batch)]
            klens = [max(qlens[i], klen - (i % 3) * max(1, klen // 7) - i % 2)
                     for i in range(batch)]
        pages = (klen + block - 1) // block
        row = {"phase": phase, "batch": batch, "query_len": qlen, "kv_len": klen,
               "query_lengths": qlens, "kv_lengths": klens,
               "query_heads": h, "kv_heads": hk, "head_dim": d, "block_size": block,
               "query_tokens": sum(qlens), "status": "running", "seed": 4300 + index}
        result["rows"].append(row)
        save()
        print(json.dumps({"starting": row}), flush=True)
        cache = torch.randn((2, batch * pages, block, hk, d), dtype=dtype, device=device)
        page_ids = torch.arange(batch * pages) if args.sequential_pages else torch.randperm(batch * pages)
        blocks = page_ids.reshape(batch, pages).to(device=device, dtype=torch.int32)
        lens = torch.tensor(klens, device=device, dtype=torch.int32)
        qo_lens = torch.tensor(qlens, device=device, dtype=torch.int32)
        qstarts = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()),
                               device=device, dtype=torch.int32)
        kstarts = torch.tensor([0] + list(torch.tensor(klens).cumsum(0).tolist()),
                               device=device, dtype=torch.int32)
        q = torch.randn(sum(qlens), h, d, device=device, dtype=dtype)
        out = torch.empty_like(q)
        scale = d ** -.5
        token_indices = torch.cat([
            blocks[i, torch.arange(length, device=device) // block].long() * block
            + torch.arange(length, device=device) % block
            for i, length in enumerate(klens)
        ])
        flat_k, flat_v = cache[0].view(-1, hk, d), cache[1].view(-1, hk, d)
        dense_k = torch.index_select(flat_k, 0, token_indices)
        dense_v = torch.index_select(flat_v, 0, token_indices)
        functions = {}
        if phase == "decode":
            # Match Chitu's TritonAttnBackend split-count policy for MetaX.
            splits = 3 if batch > 32 else 8 if batch > 1 else 16
            logits = torch.empty((batch, h, splits, d + 1), dtype=torch.float32, device=device)
            row["num_kv_splits"] = splits

            def chitu_paged():
                decode.decode_paged_kv_triton(q, cache[0], cache[1], out, blocks,
                                              lens, logits, splits, scale, block)
                return out

            def native_paged():
                return flash_attn_with_kvcache(q=q[:, None], k_cache=cache[0], v_cache=cache[1],
                    block_table=blocks, cache_seqlens=lens, softmax_scale=scale, causal=True)[:, 0]

            functions.update(chitu_paged=chitu_paged, native_paged=native_paged)
        else:
            def chitu_dense():
                prefill.prefill_ragged_qkvo_triton(q, dense_k, dense_v, out, qstarts,
                    kstarts, qo_lens, lens, max(qlens), scale, True)
                return out

            def native_dense():
                return flash_attn_varlen_func(q=q, k=dense_k, v=dense_v,
                    cu_seqlens_q=qstarts, cu_seqlens_k=kstarts,
                    max_seqlen_q=max(qlens), max_seqlen_k=max(klens), softmax_scale=scale, causal=True)

            def native_paged():
                return flash_attn_varlen_func(q=q, k=cache[0], v=cache[1],
                    cu_seqlens_q=qstarts, cu_seqlens_k=kstarts,
                    max_seqlen_q=max(qlens), max_seqlen_k=max(klens), softmax_scale=scale,
                    causal=True, block_table=blocks)

            def chitu_materialize():
                torch.index_select(flat_k, 0, token_indices, out=dense_k)
                torch.index_select(flat_v, 0, token_indices, out=dense_v)
                return chitu_dense()

            functions.update(chitu_dense=chitu_dense, native_dense=native_dense,
                             native_paged=native_paged, chitu_materialize=chitu_materialize)
        values, checks = {}, {}
        for name, fn in functions.items():
            out.fill_(float("nan"))
            started = time.perf_counter()
            actual = fn()
            api.synchronize()
            row.setdefault("first_call_wall_seconds", {})[name] = time.perf_counter() - started
            values[name] = actual.detach().cpu()
            checks[name] = {"finite": bool(torch.isfinite(values[name]).all())}
            if name == "chitu_paged" and hk != h:
                row["chitu_autotune_config"] = str(decode._fwd_grouped_kernel_stage1.best_config)
            save()
        row["pair_checks"] = {name: compare(actual, values["native_paged"])
                              for name, actual in values.items() if name != "native_paged"}
        q_cpu, k_cpu, v_cpu = q.float().cpu(), dense_k.float().cpu(), dense_v.float().cpu()
        qoffset, koffset = 0, 0
        maxima = {name: 0. for name in values}
        passes = {name: True for name in values}
        samples = 0
        for seq in range(batch):
            if seq in {0, batch // 2, batch - 1}:
                for head in sorted({0, h // 2, h - 1}):
                    for pos in sorted({0, qlens[seq] // 2, qlens[seq] - 1}):
                        attended = klens[seq] - qlens[seq] + pos + 1
                        kh = head // (h // hk)
                        keys = k_cpu[koffset:koffset + attended, kh]
                        vals = v_cpu[koffset:koffset + attended, kh]
                        ref = ((keys @ q_cpu[qoffset + pos, head]) * scale).softmax(0) @ vals
                        for name, actual in values.items():
                            check = compare(actual[qoffset + pos, head], ref)
                            passes[name] &= check["passed"]
                            maxima[name] = max(maxima[name], check["max_abs_error"])
                        samples += 1
            qoffset += qlens[seq]
            koffset += klens[seq]
        for name in checks:
            checks[name].update(cpu_fp32_samples=samples, cpu_fp32_max_abs_error=maxima[name],
                                cpu_fp32_passed=passes[name])
        row["checks"] = checks
        if not all(c["finite"] and c["cpu_fp32_passed"] for c in checks.values()) or not all(
                c["passed"] for c in row["pair_checks"].values()):
            row["status"] = "correctness_failed"
        elif args.check_only:
            row["status"] = "passed_check_only"
        else:
            row["timings"] = {}
            order = list(functions)
            if index % 2:
                order.reverse()
            for name in order:
                row["timings"][name] = measure_graph(functions[name], values[name], api=api,
                    platform="metax", dtype=dtype, unroll=16 if phase == "decode" else 1,
                    repeats=args.repeats, target_ms=args.target_ms, warmup=3, nan_sentinel=True)
                save()
            row["status"] = "passed" if all(t["status"] == "passed" for t in row["timings"].values()) else "graph_correctness_failed"
            if row["status"] == "passed":
                metric = "graph_device_ms"
                if any(t[metric] is None for t in row["timings"].values()):
                    metric = "graph_wall_ms"
                row["ratio_metric"] = metric
                ms = {name: timing[metric] for name, timing in row["timings"].items()}
                row["speedups"] = ({"chitu_over_native_paged": ms["native_paged"] / ms["chitu_paged"]}
                    if phase == "decode" else {"chitu_dense_over_native_dense": ms["native_dense"] / ms["chitu_dense"],
                    "chitu_materialize_over_native_paged": ms["native_paged"] / ms["chitu_materialize"]})
        save()
        print(json.dumps({"finished": row}), flush=True)

    for index, case in enumerate(cases):
        try:
            run_case(index, *case)
        except Exception:
            error = traceback.format_exc()
            result["rows"][-1].update(status="error", error=error)
            result["stopped_after_error"] = True
            save()
            print(error, flush=True)
            break
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    return 0 if len(result["rows"]) == len(cases) and all(
        row["status"] in ("passed", "passed_check_only") for row in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
