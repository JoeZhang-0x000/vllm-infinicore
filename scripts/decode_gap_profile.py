#!/usr/bin/env python
"""Locate the vLLM-InfiniCore vs vLLM-MetaX steady-state decode gap.

Reports, per engine:
  * clean steady-state decode step latency (profiler off, differenced so prefill
    and sampling setup cancel out),
  * GPU busy fraction during a profiled decode window,
  * top kernels by exclusive device time,
  * top host-side ops by exclusive CPU time, and synchronization cost.

A lower GPU busy fraction at equal kernel time means the gap is host-side launch
or stream-handoff stalls. Higher kernel time at equal busy fraction means the
gap is kernel efficiency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=("vllm-metax", "vllm-infinicore"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--short-output-len", type=int, default=32)
    parser.add_argument("--long-output-len", type=int, default=160)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--top-k-rows", type=int, default=25)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--trace-dir", default="")
    parser.add_argument(
        "--python-profile",
        action="store_true",
        help="Also run cProfile over one decode window for function-level attribution.",
    )
    return parser.parse_args()


def build_prompt_ids(tokenizer: Any, input_len: int) -> list[int]:
    seed = (
        "Qwen3 benchmark prompt. Compare graph execution, token accounting, "
        "operator routing, and output health. Keep the answer technical. "
    )
    ids: list[int] = []
    index = 0
    while len(ids) < input_len:
        ids.extend(tokenizer.encode(f"{seed} Segment {index}.\n", add_special_tokens=False))
        index += 1
    return ids[:input_len]


def configure_engine(engine: str) -> dict[str, str] | None:
    if engine != "vllm-infinicore":
        os.environ["VLLM_PLUGINS"] = "metax"
        return None
    os.environ["VLLM_PLUGINS"] = "infinicore,vllm_infinicore"
    os.environ["VLLM_INFINICORE_ENABLE_PATCHES"] = "1"
    os.environ["VLLM_INFINICORE_ROUTES"] = "all"
    os.environ["VLLM_INFINICORE_FORCE_NATIVE_FALLBACK"] = "0"
    os.environ["VLLM_INFINICORE_STRICT_BACKEND"] = "1"
    os.environ["VLLM_INFINICORE_DISABLE_REAL_BACKEND"] = "0"
    import vllm_infinicore

    registration = vllm_infinicore.register()
    return {
        "installed_routes": list(getattr(registration, "installed_routes", []) or []),
        "skipped_routes": list(getattr(registration, "skipped_routes", []) or []),
        "native_fallback_routes": list(getattr(registration, "native_fallback_routes", []) or []),
    }


def out_path_with_suffix(path: str, suffix: str) -> str:
    base = Path(path)
    return str(base.with_suffix(suffix))


def main() -> int:
    args = parse_args()
    registration = configure_engine(args.engine)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config import CUDAGraphMode
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = build_prompt_ids(tokenizer, args.input_len)
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(args.batch_size)]

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.input_len + args.long_output_len + 128,
        enforce_eager=False,
        compilation_config={
            "cudagraph_mode": CUDAGraphMode.PIECEWISE,
            "cudagraph_capture_sizes": [1, 2, 4, 8],
            "cudagraph_num_of_warmups": 1,
            "backend": "eager",
        },
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    def sampling(output_len: int) -> Any:
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            max_tokens=output_len,
            min_tokens=output_len,
            ignore_eos=True,
            detokenize=False,
        )

    def run(output_len: int) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling(output_len), use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        for out in outputs:
            assert len(out.outputs[0].token_ids) == output_len
        return elapsed

    run(args.short_output_len)  # warm up

    short_times = [run(args.short_output_len) for _ in range(args.latency_repeats)]
    long_times = [run(args.long_output_len) for _ in range(args.latency_repeats)]
    decode_steps = args.long_output_len - args.short_output_len
    step_ms = (
        (statistics.median(long_times) - statistics.median(short_times)) / decode_steps * 1000.0
    )

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        profiled_wall = run(args.long_output_len)

    from torch.autograd import DeviceType

    events = prof.key_averages()
    # Only device-side events (kernels, memcpy, memset) carry real GPU occupancy.
    # Host op events also report attributed device time, so summing both double counts.
    kernel_events = [e for e in events if e.device_type == DeviceType.CUDA]
    host_events = [e for e in events if e.device_type != DeviceType.CUDA]
    device_total_us = sum(max(0.0, e.self_device_time_total) for e in kernel_events)
    cpu_total_us = sum(max(0.0, e.self_cpu_time_total) for e in host_events)

    def rows(source: list[Any], attr: str, denom: float, limit: int) -> list[dict[str, Any]]:
        ranked = sorted(source, key=lambda e: getattr(e, attr), reverse=True)
        out = []
        for e in ranked[:limit]:
            value = getattr(e, attr)
            if value <= 0:
                break
            out.append(
                {
                    "name": e.key[:110],
                    "count": int(e.count),
                    "self_ms": round(value / 1000.0, 3),
                    "share_pct": round(value / denom * 100.0, 2) if denom > 0 else None,
                }
            )
        return out

    def _host_us(tokens: tuple[str, ...]) -> float:
        return sum(
            max(0.0, e.self_cpu_time_total)
            for e in host_events
            if any(token in e.key.lower() for token in tokens)
        )

    sync_us = _host_us(("synchronize", "streamwait", "eventquery"))
    launch_us = _host_us(("launchkernel", "graphlaunch", "mclaunch"))
    kernel_launches = sum(
        int(e.count) for e in host_events if "launchkernel" in e.key.lower()
    )

    payload = {
        "engine": args.engine,
        "model": args.model,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "short_output_len": args.short_output_len,
        "long_output_len": args.long_output_len,
        "decode_steps_measured": decode_steps,
        "registration": registration,
        "vllm_plugins": os.environ.get("VLLM_PLUGINS", ""),
        "latency": {
            "short_times_s": short_times,
            "long_times_s": long_times,
            "decode_step_ms": round(step_ms, 4),
            "decode_tps_per_request": round(1000.0 / step_ms, 2) if step_ms > 0 else None,
            "decode_tps_batch": round(args.batch_size * 1000.0 / step_ms, 2) if step_ms > 0 else None,
        },
        "profiled": {
            "wall_s": round(profiled_wall, 4),
            "device_kernel_ms": round(device_total_us / 1000.0, 2),
            "host_cpu_ms": round(cpu_total_us / 1000.0, 2),
            "gpu_busy_pct": round(device_total_us / 1000.0 / (profiled_wall * 1000.0) * 100.0, 2)
            if profiled_wall > 0
            else None,
            "gpu_idle_ms": round(profiled_wall * 1000.0 - device_total_us / 1000.0, 2),
            "host_sync_ms": round(sync_us / 1000.0, 2),
            "host_launch_ms": round(launch_us / 1000.0, 2),
            "kernel_launch_count": kernel_launches,
            "device_event_count": int(sum(e.count for e in kernel_events)),
            "top_device_kernels": rows(
                kernel_events, "self_device_time_total", device_total_us, args.top_k_rows
            ),
            "top_host_ops": rows(host_events, "self_cpu_time_total", cpu_total_us, args.top_k_rows),
            "top_host_ops_by_device_time": rows(
                host_events, "self_device_time_total", device_total_us, args.top_k_rows
            ),
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.python_profile:
        import cProfile
        import pstats
        import io as _io

        pr = cProfile.Profile()
        pr.enable()
        run(args.long_output_len)
        pr.disable()
        prof_path = out_path_with_suffix(args.output_json, ".prof")
        pr.dump_stats(prof_path)
        payload["python_profile_stats_path"] = prof_path
        stream = _io.StringIO()
        stats = pstats.Stats(pr, stream=stream)
        stats.sort_stats("tottime").print_stats(40)
        stream.write("\n\n===== CALLERS: find_spec / _find_and_load / __import__ =====\n")
        stats.print_callers("find_spec|_find_and_load|_handle_fromlist|__import__")
        stream.write("\n\n===== CALLERS: os.environ __getitem__ =====\n")
        stats.print_callers("_collections_abc.py:821")
        text = stream.getvalue()
        payload["python_profile_tottime_top40"] = text
        py_out = out_path_with_suffix(args.output_json, ".pyprof.txt")
        Path(py_out).write_text(text, encoding="utf-8")
        payload["python_profile_path"] = py_out

    if args.engine == "vllm-infinicore":
        try:
            from vllm_infinicore.ops import cpp_bridge, infinicore_backend

            payload["infinicore_backend_call_counts"] = dict(
                infinicore_backend.backend_call_counts()
            )
            payload["infinicore_cpp_bridge_call_counts"] = dict(
                cpp_bridge.bridge_call_counts()
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            payload["infinicore_backend_call_counts_error"] = repr(exc)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_dir / f"{args.engine}-decode.json"))

    print(
        f"{args.engine}: decode_step={step_ms:.3f} ms  "
        f"batch_decode_tps={payload['latency']['decode_tps_batch']}  "
        f"gpu_busy={payload['profiled']['gpu_busy_pct']}%  "
        f"kernel={payload['profiled']['device_kernel_ms']} ms  "
        f"host_sync={payload['profiled']['host_sync_ms']} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
