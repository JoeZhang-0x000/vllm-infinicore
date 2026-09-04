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
from collections.abc import Callable
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


def measure_decode_window(
    run: Callable[[int], float],
    output_len: int,
    top_k: int,
) -> dict[str, Any]:
    """Profile one decode window and summarize where its wall time went."""

    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        wall = run(output_len)

    events = prof.key_averages()
    # Only device-side events (kernels, memcpy, memset) carry real GPU occupancy.
    # Host op events also report attributed device time, so summing both double
    # counts and yields a GPU busy fraction above 100%.
    kernel_events = [e for e in events if e.device_type == DeviceType.CUDA]
    host_events = [e for e in events if e.device_type != DeviceType.CUDA]
    device_us = sum(max(0.0, e.self_device_time_total) for e in kernel_events)
    host_us = sum(max(0.0, e.self_cpu_time_total) for e in host_events)

    def rows(source: list[Any], attr: str, denom: float) -> list[dict[str, Any]]:
        ranked = sorted(source, key=lambda event: getattr(event, attr), reverse=True)
        out: list[dict[str, Any]] = []
        for event in ranked[:top_k]:
            value = getattr(event, attr)
            if value <= 0:
                break
            out.append(
                {
                    "name": event.key[:110],
                    "count": int(event.count),
                    "self_ms": round(value / 1000.0, 3),
                    "share_pct": round(value / denom * 100.0, 2) if denom > 0 else None,
                }
            )
        return out

    def host_us_matching(tokens: tuple[str, ...]) -> float:
        return sum(
            max(0.0, e.self_cpu_time_total)
            for e in host_events
            if any(token in e.key.lower() for token in tokens)
        )

    sync_us = host_us_matching(("synchronize", "streamwait", "eventquery"))
    launch_us = host_us_matching(("launchkernel", "graphlaunch", "mclaunch"))
    return {
        "prof": prof,
        "wall_s": round(wall, 4),
        "device_kernel_ms": round(device_us / 1000.0, 2),
        "host_cpu_ms": round(host_us / 1000.0, 2),
        "gpu_busy_pct": round(device_us / (wall * 1e6) * 100.0, 2) if wall > 0 else None,
        "gpu_idle_ms": round(wall * 1000.0 - device_us / 1000.0, 2),
        "host_sync_ms": round(sync_us / 1000.0, 2),
        "host_launch_ms": round(launch_us / 1000.0, 2),
        "kernel_launch_count": sum(
            int(e.count) for e in host_events if "launchkernel" in e.key.lower()
        ),
        "device_event_count": int(sum(e.count for e in kernel_events)),
        "top_device_kernels": rows(kernel_events, "self_device_time_total", device_us),
        "top_host_ops": rows(host_events, "self_cpu_time_total", host_us),
    }


def profile_python_frames(
    run: Callable[[int], float],
    output_len: int,
    out_json: Path,
) -> dict[str, Any]:
    """cProfile one decode window for function-level attribution.

    cProfile inflates call-heavy paths badly, so treat its ranking as a lead to
    A/B rather than as an estimate of what a fix is worth.
    """

    import cProfile
    import io
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    run(output_len)
    profiler.disable()

    stats_path = out_json.with_suffix(".prof")
    profiler.dump_stats(str(stats_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("tottime").print_stats(40)
    stream.write("\n\n===== CALLERS: import machinery =====\n")
    stats.print_callers("find_spec|_find_and_load|_handle_fromlist|__import__")
    stream.write("\n\n===== CALLERS: os.environ lookups =====\n")
    stats.print_callers("_collections_abc.py:821")
    text = stream.getvalue()

    text_path = out_json.with_suffix(".pyprof.txt")
    text_path.write_text(text, encoding="utf-8")
    return {
        "stats_path": str(stats_path),
        "text_path": str(text_path),
        "tottime_top40": text,
    }


def infinicore_counters() -> dict[str, Any]:
    try:
        from vllm_infinicore.ops import cpp_bridge, infinicore_backend

        return {
            "infinicore_backend_call_counts": dict(infinicore_backend.backend_call_counts()),
            "infinicore_cpp_bridge_call_counts": dict(cpp_bridge.bridge_call_counts()),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"infinicore_backend_call_counts_error": repr(exc)}


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

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    window = measure_decode_window(run, args.long_output_len, args.top_k_rows)
    prof = window.pop("prof")

    payload: dict[str, Any] = {
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
            "decode_tps_batch": round(args.batch_size * 1000.0 / step_ms, 2)
            if step_ms > 0
            else None,
        },
        "profiled": window,
    }
    if args.python_profile:
        payload["python_profile"] = profile_python_frames(run, args.long_output_len, out_path)
    if args.engine == "vllm-infinicore":
        payload.update(infinicore_counters())

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
