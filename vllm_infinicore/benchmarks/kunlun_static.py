#!/usr/bin/env python3
"""Static Kunlun throughput: fixed lengths, warmup, three complete generate timings."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import time
import traceback
from collections.abc import Iterator
from pathlib import Path

from .common import (
    ROUTES,
    mode_environment,
    sha256,
    verify_workers,
    worker_state,
    write_json,
)

DEFAULT_ROUTES = ("RoPE", "Embedding", "MatMul", "LMHead", "StoreKVCache")
DEFAULT_BATCHES = (1, 4, 16, 32, 64)
DEFAULT_LENGTHS = ((128, 128), (2048, 512))
MAX_MODEL_LEN = 2816
MAX_NUM_SEQS = 64
GRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64)
LIBRARIES = (
    "libinfinicore_cpp_api.so",
    "libinfiniop.so",
    "libinfinirt.so",
    "libinfiniccl.so",
)


def compare_results(native_path: Path, infini_path: Path) -> dict:
    native, infini = [json.loads(path.read_text()) for path in (native_path, infini_path)]
    for expected, data in (("native", native), ("infinicore", infini)):
        if data["arguments"]["mode"] != expected or not data["completed"] or data["errors"]:
            raise ValueError(f"Incomplete or wrong-mode benchmark: {expected}")
    for key in ("llm_kwargs", "model_config_sha256", "prompt_sha256", "versions"):
        if native[key] != infini[key]:
            raise ValueError(f"Benchmark conditions differ: {key}")
    for key in ("devices", "routes", "batches", "lengths", "repeats", "enforce_eager"):
        if native["arguments"][key] != infini["arguments"][key]:
            raise ValueError(f"Benchmark conditions differ: {key}")
    if native["vendor_cache_sha256"] != infini["vendor_cache_sha256"]:
        raise ValueError("Vendor cache implementation differs")
    for mode, data in (("native", native), ("infinicore", infini)):
        states = data["final_workers"]
        tp = data["arguments"]["tp"]
        if len(states) != tp or {state["rank"] for state in states} != set(range(tp)):
            raise ValueError("Missing worker evidence")
        verify_workers(
            mode,
            states,
            data["arguments"]["enforce_eager"],
            True,
            routes=tuple(data["arguments"]["routes"].split(",")),
        )
    rows = []

    def key(row):
        return row["input_len"], row["output_len"], row["batch"]

    off = {key(row): row for row in native["summaries"]}
    on = {key(row): row for row in infini["summaries"]}
    expected = {
        (*map(int, lengths.split(":")), int(batch))
        for lengths in native["arguments"]["lengths"].split(",")
        for batch in native["arguments"]["batches"].split(",")
    }
    if set(off) != expected or set(on) != expected:
        raise ValueError("Missing benchmark cases")
    for case in sorted(expected):
        medians = []
        for data in (native, infini):
            timed = [row for row in data["rows"] if key(row) == case]
            repeats = data["arguments"]["repeats"]
            if len(timed) != repeats or {row["repeat"] for row in timed} != set(range(repeats)):
                raise ValueError(f"Missing timing repetitions: {case}")
            samples = [row["output_tps"] for row in timed]
            if any(not math.isfinite(value) or value <= 0 for value in samples):
                raise ValueError("Invalid throughput sample")
            medians.append(statistics.median(samples))
        if medians != [off[case]["median_tps"], on[case]["median_tps"]]:
            raise ValueError("Summary differs from timing samples")
        ratio = medians[1] / medians[0]
        rows.append(
            dict(
                input_len=case[0],
                output_len=case[1],
                batch=case[2],
                native_tps=medians[0],
                infinicore_tps=medians[1],
                ratio=ratio,
                reached_90_percent=ratio >= 0.90,
            )
        )
    return dict(
        tensor_parallel_size=native["arguments"]["tp"],
        native=str(native_path),
        infinicore=str(infini_path),
        cases=rows,
        minimum_ratio=min(row["ratio"] for row in rows),
        geometric_mean_ratio=statistics.geometric_mean(row["ratio"] for row in rows),
        all_cases_reached_90_percent=all(row["reached_90_percent"] for row in rows),
    )


def prompt_ids(tokenizer, length):
    marker = "__BENCH_NOTES__"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    before, after = text.split(marker)

    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False)

    prefix = encode(before + "Background notes:\n")
    suffix = encode(
        "\nExplain the design of reliable computer systems in detail, "
        "using concrete examples and discussing tradeoffs." + after
    )
    facts = encode(
        "".join(
            f"Case {i}: A team measured memory bandwidth, network latency, "
            f"and recovery time across {i % 29 + 3} workloads for {i % 17 + 2} days. "
            "They compared caching, redundancy, scheduling and capacity planning.\n"
            for i in range(300)
        )
    )
    capacity = length - len(prefix) - len(suffix)
    if not 0 <= capacity <= len(facts):
        raise ValueError(f"Cannot create a prompt with {length} tokens")
    return prefix + facts[:capacity] + suffix


def benchmark_matrix(args: argparse.Namespace) -> tuple[list[int], list[tuple[int, int]]]:
    """Parse the matrix without adding derived fields to the saved arguments."""
    batches = [int(value) for value in args.batches.split(",")]
    lengths = []
    for value in args.lengths.split(","):
        input_len, output_len = map(int, value.split(":"))
        lengths.append((input_len, output_len))
    return batches, lengths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "infinicore"))
    parser.add_argument("--model")
    parser.add_argument("--tp", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--devices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-native", type=Path)
    parser.add_argument("--compare-infinicore", type=Path)
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument(
        "--lengths",
        default=",".join(f"{input_len}:{output_len}" for input_len, output_len in DEFAULT_LENGTHS),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory", type=float)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)
    if args.compare_native or args.compare_infinicore:
        if not (args.compare_native and args.compare_infinicore):
            parser.error("Supply both comparison paths")
        return args
    if not (args.mode and args.model and args.tp and args.devices):
        parser.error("Supply --mode, --model, --tp and --devices")
    routes = tuple(args.routes.split(","))
    allowed = set(ROUTES) - {"PagedAttentionPrefill", "PagedAttentionDecode"}
    if set(routes) - allowed or not routes:
        parser.error("Only non-Attention routes may be requested")
    if len(args.devices.split(",")) != args.tp:
        parser.error("Device count must equal TP")
    try:
        batches, lengths = benchmark_matrix(args)
    except ValueError:
        parser.error("Use integer batches and input:output length pairs")
    if min(batches) < 1 or max(batches) > MAX_NUM_SEQS or args.repeats < 1:
        parser.error("Invalid batches/repeats")
    if any(min(i, o) < 1 or i + o > MAX_MODEL_LEN for i, o in lengths):
        parser.error(f"Lengths must be positive and fit max_model_len={MAX_MODEL_LEN}")
    return args


def llm_kwargs(args: argparse.Namespace) -> dict:
    """Keep the measured generation and Graph protocol in one place."""
    kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        seed=0,
        enforce_eager=args.enforce_eager,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=args.memory or (0.85 if args.tp <= 2 else 0.70),
        enable_prefix_caching=False,
        trust_remote_code=True,
        block_size=128,
        distributed_executor_backend="mp",
    )
    if not args.enforce_eager:
        kwargs["compilation_config"] = dict(
            cudagraph_capture_sizes=list(GRAPH_CAPTURE_SIZES),
            cudagraph_num_of_warmups=1,
        )
    return kwargs


def runtime_metadata(args: argparse.Namespace, vendor_cache: Path, device_name: str) -> dict:
    """Record the installed vendor cache, libraries, versions and model config."""
    prefix = Path(os.environ["INFINI_ROOT"])
    metadata = {
        "vendor_cache_sha256": sha256(vendor_cache),
        "device_name": device_name,
        "infinicore_libraries": {name: sha256(prefix / "lib" / name) for name in LIBRARIES},
    }
    if (prefix / "manifest.json").is_file():
        metadata["infinicore_manifest"] = json.loads((prefix / "manifest.json").read_text())
    metadata["versions"] = {}
    for name in (
        "torch",
        "vllm",
        "vllm-kunlun",
        "xmlir",
        "kunlun-ops",
        "xspeedgate-ops",
        "triton",
    ):
        try:
            metadata["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata["model_config_sha256"] = sha256(Path(args.model) / "config.json")
    return metadata


def timed_rows(llm, requests, sampling, case: dict, repeats: int) -> Iterator[dict]:
    """Warm up once, then time complete generate calls with fixed output lengths."""
    llm.generate(requests, sampling, use_tqdm=False)
    batch, output_len = case["batch"], case["output_len"]
    for repeat in range(repeats):
        started = time.perf_counter()
        outputs = llm.generate(requests, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        if len(outputs) != batch:
            raise RuntimeError("Missing requests")
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        if any(len(ids) != output_len for ids in token_ids):
            raise RuntimeError("Generation did not reach the fixed output length")
        yield dict(
            **case,
            repeat=repeat,
            seconds=elapsed,
            output_tps=batch * output_len / elapsed,
            output_token_hashes=[
                hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in token_ids
            ],
            first_text=outputs[0].outputs[0].text,
        )


def run_mode(args: argparse.Namespace) -> int:
    routes = tuple(args.routes.split(","))
    batches, lengths = benchmark_matrix(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(mode_environment(args.mode, os.environ, "kunlun", routes))
    os.environ.update(CUDA_VISIBLE_DEVICES=args.devices, XPU_VISIBLE_DEVICES=args.devices)
    os.environ["VLLM_CACHE_ROOT"] = str(args.output.parent / f"cache-{args.output.stem}")
    result = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "completed": False,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "errors": [],
        "rows": [],
        "summaries": [],
        "environment": {
            k: os.getenv(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "XPU_VISIBLE_DEVICES",
                "INFINI_ROOT",
                "VLLM_PLUGINS",
                "VLLM_INFINICORE_ROUTES",
                "XMLIR_FORCE_USE_XPU_GRAPH",
                "XMLIR_ENABLE_MOCK_TORCH_COMPILE",
                "USE_ORI_ROPE",
                "ROPE_NATIVE_2D",
            )
        },
    }
    write_json(args.output, result)
    try:
        import kunlun_ops
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        torch.set_num_threads(4)
        result.update(
            runtime_metadata(
                args,
                Path(kunlun_ops.__file__).with_name("_cache.py"),
                torch.cuda.get_device_name(0),
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=True
        )
        inputs = {str(i): prompt_ids(tokenizer, i) for i, _ in lengths}
        result["prompt_token_ids"] = inputs
        result["prompt_sha256"] = hashlib.sha256(json.dumps(inputs).encode()).hexdigest()
        kwargs = llm_kwargs(args)
        result["llm_kwargs"] = kwargs
        write_json(args.output, result)
        llm = LLM(**kwargs)
        states = llm.collective_rpc(worker_state)
        result["startup_workers"] = states
        if len(states) != args.tp or {s["rank"] for s in states} != set(range(args.tp)):
            raise RuntimeError("Missing tensor-parallel workers")
        verify_workers(args.mode, states, args.enforce_eager, finished=False, routes=routes)
        for input_len, output_len in lengths:
            sampling = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                max_tokens=output_len,
                min_tokens=output_len,
                ignore_eos=True,
                seed=0,
            )
            for batch in batches:
                case = dict(input_len=input_len, output_len=output_len, batch=batch)
                result["active_case"] = case
                write_json(args.output, result)
                requests = [{"prompt_token_ids": inputs[str(input_len)]} for _ in range(batch)]
                rows = []
                for row in timed_rows(llm, requests, sampling, case, args.repeats):
                    rows.append(row)
                    result["rows"].append(row)
                    write_json(args.output, result)
                summary = dict(
                    **case, median_tps=statistics.median(row["output_tps"] for row in rows)
                )
                result["summaries"].append(summary)
                print("CASE", args.mode, args.tp, json.dumps(summary), flush=True)
        states = llm.collective_rpc(worker_state)
        result["final_workers"] = states
        verify_workers(args.mode, states, args.enforce_eager, finished=True, routes=routes)
        for state in states:
            if any(
                state.get("attention_calls", {}).get(name, 0)
                for name in ("PagedAttentionPrefill", "PagedAttentionDecode")
            ):
                raise RuntimeError("InfiniCore Attention was called")
        result.pop("active_case", None)
        result["completed"] = True
    except Exception:
        result["errors"].append(traceback.format_exc())
        print(result["errors"][-1], flush=True)
    finally:
        result["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.output, result)
    return 0 if result["completed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Use a fresh result path: {args.output}")
    if args.compare_native:
        comparison = compare_results(args.compare_native, args.compare_infinicore)
        write_json(args.output, comparison)
        print(json.dumps(comparison, indent=2))
        return 0
    return run_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
