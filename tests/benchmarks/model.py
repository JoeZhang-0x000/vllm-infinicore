#!/usr/bin/env python3
"""Offline vLLM throughput and output-health benchmark for vendor backends.

Run each model/mode in a fresh process. The caller selects vendor plugins and
InfiniCore environment variables, so the same script measures both paths.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from .common import cache_rope_tables, source_fingerprints, write_json


TOPICS = (
    "processor scheduling and latency", "memory bandwidth and caching",
    "persistent storage and data integrity", "network congestion and routing",
    "software testing and observability", "energy efficiency and cooling",
    "security boundaries and access control", "distributed consensus and recovery",
    "database indexing and transactions", "human factors in system design",
    "scientific reproducibility", "capacity planning and queueing",
)


def make_prompt(tokenizer, length: int) -> list[int]:
    marker = "__BENCHMARK_BODY__"
    try:
        template = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before, after = template.split(marker)
    except (AttributeError, TypeError, ValueError):
        before, after = "User: ", "\nAssistant: "
    header = "Reference notes:\n"
    instruction = (
        "\nExplain how a computer cache works, with a concrete example. "
        "Write clear, connected prose.\n"
    )
    facts = "".join(
        f"Case {i:03d}: A team studied {TOPICS[i % len(TOPICS)]}. "
        f"Its measurements covered {8 + i % 41} users, "
        f"{3 + (i * 7) % 29} workloads, and {2 + (i * 11) % 17} days. "
        "The team compared a simple design with a more complex one and "
        "recorded both benefits and failure modes.\n"
        for i in range(1, 300)
    )
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    start = encode(before + header)
    end = encode(instruction + after)
    middle = encode(facts)
    capacity = length - len(start) - len(end)
    if capacity <= 0 or len(middle) < capacity:
        raise ValueError("cannot construct requested prompt length")
    ids = start + middle[:capacity] + end
    assert len(ids) == length
    return ids


def text_health(text: str, token_ids: list[int]) -> dict:
    bad_controls = sum(ord(c) < 32 and c not in "\n\r\t" for c in text)
    tail = text[-500:]
    tail_unique_chars = len(set(tail))
    longest_identical_line_run = 0
    line_run = 0
    previous_line = None
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 6:
            line_run = 0
            previous_line = None
            continue
        line_run = line_run + 1 if line == previous_line else 1
        longest_identical_line_run = max(longest_identical_line_run, line_run)
        previous_line = line
    longest_run = 0
    run = 0
    previous = None
    for token in token_ids:
        run = run + 1 if token == previous else 1
        longest_run = max(longest_run, run)
        previous = token
    repeated_cycle = None
    for span in (4, 8, 16, 32):
        for offset in range(len(token_ids) - 4 * span + 1):
            block = token_ids[offset : offset + span]
            if all(
                token_ids[offset + j * span : offset + (j + 1) * span]
                == block
                for j in (1, 2, 3)
            ):
                repeated_cycle = {"start": offset, "span": span}
                break
        if repeated_cycle:
            break
    issues = []
    if not text.strip():
        issues.append("empty_text")
    if "\ufffd" in text:
        issues.append("unicode_replacement_character")
    if "\x00" in text or bad_controls:
        issues.append("invalid_control_character")
    if longest_run >= 16:
        issues.append("same_token_repeated_16_times")
    if repeated_cycle:
        issues.append("four_consecutive_identical_token_blocks")
    if len(tail) >= 200 and tail_unique_chars <= 12:
        issues.append("low_diversity_tail")
    if longest_identical_line_run >= 5:
        issues.append("five_consecutive_identical_lines")
    return {
        "issues": issues,
        "replacement_chars": text.count("\ufffd"),
        "null_chars": text.count("\x00"),
        "bad_control_chars": bad_controls,
        "longest_same_token_run": longest_run,
        "repeated_cycle": repeated_cycle,
        "tail_unique_chars": tail_unique_chars,
        "longest_identical_line_run": longest_identical_line_run,
    }


def worker_state(worker):
    state = {"pid": os.getpid(), "rank": getattr(worker, "rank", None)}
    import torch
    # Install after startup/capture, before warmup and timed requests. Count
    # actual graph.replay calls, not merely enforce_eager=False in config.
    state["graph_replays"] = {}
    for api_name, class_name in (("cuda", "CUDAGraph"), ("npu", "NPUGraph")):
        api = getattr(torch, api_name, None)
        cls = getattr(api, class_name, None)
        if cls is None:
            continue
        if not hasattr(cls, "_attention_bench_replays"):
            original = cls.replay
            def counted_replay(self, *args, _original=original, _cls=cls, **kwargs):
                result = _original(self, *args, **kwargs)
                _cls._attention_bench_replays += 1
                return result
            cls._attention_bench_replays = 0
            cls.replay = counted_replay
        state["graph_replays"][api_name] = cls._attention_bench_replays
    try:
        from vllm.compilation.counter import compilation_counter

        state["graph_captures"] = compilation_counter.num_cudagraph_captured
    except Exception as exc:
        state["graph_captures_error"] = repr(exc)
    if os.environ.get("VLLM_INFINICORE_ENABLE_PATCHES") == "1":
        try:
            from vllm_infinicore import plugin
            from vllm_infinicore.operators import backend, cpp_bridge

            registration = plugin._REGISTRATION_RESULT
            state["plugin_file"] = plugin.__file__
            state["registration"] = (
                dataclasses.asdict(registration) if registration else None
            )
            state["backend_calls"] = backend.backend_call_counts()
            if hasattr(backend, "backend_fallback_counts"):
                state["fallback_calls"] = backend.backend_fallback_counts()
                state["fallback_reasons"] = backend.backend_fallback_reasons()
            state["bridge_calls"] = cpp_bridge.bridge_call_counts()
            from vllm_infinicore.operators import attention, attention_ops
            state["attention_calls"] = attention_ops.call_counts()
            state["attention_fallbacks"] = attention.fallback_counts()
            state["native_attention_calls"] = attention.native_call_counts()
        except Exception as exc:
            state["plugin_state_error"] = repr(exc)
    return state


def aggregate_delta(before: list[dict], after: list[dict], key: str) -> dict:
    result = {}
    for b, a in zip(before, after):
        for name, value in a.get(key, {}).items():
            result[name] = result.get(name, 0) + value - b.get(key, {}).get(name, 0)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=("ascend", "metax", "kunlun"), required=True)
    p.add_argument("--mode", choices=("native", "infinicore"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--input-len", type=int, default=2048)
    p.add_argument("--output-len", type=int, default=2048)
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--memory", type=float, default=0.85)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--enforce-eager", action="store_true",
                   help="Diagnostic control only; main matrix uses graphs")
    p.add_argument("--capture-sizes", default=None,
                   help="Comma-separated graph sizes; default powers of two up to max batch")
    p.add_argument("--warmup-output-len", type=int, default=None,
                   help="Use a shorter untimed warmup for very slow diagnostic kernels")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="Keep engine capacity identical when measuring one batch separately")
    p.add_argument("--diagnostic-cache-rope", action="store_true",
                   help="Diagnostic only: cache the immutable RoPE table copies before graph capture")
    p.add_argument("--step-timing", action="store_true",
                   help="Diagnostic only: record engine.step wall times to separate steady decode from prefill")
    p.add_argument("--progress-every", type=int, default=0,
                   help="With --step-timing, print progress every N engine steps (0 disables)")
    p.add_argument("--skip-warmup", action="store_true",
                   help="Diagnostic only: measure steady decode after startup without another full prefill")
    p.add_argument("--expect-native-attention", action="store_true",
                   help="Require native prefill/decode launches and zero InfiniCore attention launches")
    p.add_argument("--expect-infinicore-store-all", action="store_true",
                   help="Require InfiniCore StoreKV with no native StoreKV dispatch at any size")
    p.add_argument("--expect-fused-rms", action="store_true",
                   help="Require actual InfiniCore fused Add+RMSNorm launches")
    args = p.parse_args(argv)
    if args.diagnostic_cache_rope:
        cache_rope_tables(args.platform)
        os.environ["VLLM_CACHE_ROOT"] = os.environ.get("VLLM_CACHE_ROOT", ".cache/vllm") + "/diagnostic-cached-rope"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batches = [int(v) for v in args.batches.split(",")]
    capture_sizes = ([int(v) for v in args.capture_sizes.split(",")]
                     if args.capture_sizes else
                     sorted({1, max(batches)} | {2 ** i for i in range(1, 10)
                                                if 2 ** i <= max(batches)}))
    from vllm_infinicore.routing.patching import _parse_route_names, QWEN3_OPERATOR_ROUTES
    from vllm_infinicore.routing.policy import recommended_selected
    requested_routes = set(_parse_route_names(
        os.environ.get("VLLM_INFINICORE_ROUTES", ""),
        available_routes=tuple(r.name for r in QWEN3_OPERATOR_ROUTES)))
    result = {
        "arguments": vars(args) | {"output": str(args.output)},
        "environment": {
            k: v for k, v in os.environ.items()
            if k.startswith(("VLLM_", "ASCEND_RT_", "XPU_", "XMLIR_", "CUDA_VISIBLE_", "INFINI_"))
            and not any(x in k for x in ("TOKEN", "KEY", "PASSWORD"))
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark_sources": source_fingerprints(),
        "completed": False,
        "rows": [],
        "errors": [],
    }

    def save():
        write_json(args.output, result)

    llm = None
    try:
        if args.platform == "kunlun":
            # XPytorch's NVML shim reports zero devices on this P800 runtime.
            import torch

            torch.cuda._device_count_nvml = lambda: -1
            torch.cuda._cached_device_count = None
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        if os.environ.get("VLLM_INFINICORE_REGISTRY_IN_PROCESS") == "1":
            # The packaged Kunlun Python needs an explicit glibc loader. Its
            # registry subprocess relaunches sys.executable without that loader
            # and can segfault before inference begins. This changes only the
            # model-class inspection step, equally for native and plugin runs.
            import vllm.model_executor.models.registry as model_registry

            model_registry._run_in_subprocess = lambda fn: fn()

        result["versions"] = {}
        for package in ("vllm", "vllm-ascend", "vllm-metax", "vllm-kunlun", "torch"):
            try:
                result["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                                   trust_remote_code=True)
        prompt_ids = make_prompt(tokenizer, args.input_len)
        result["prompt_sha256"] = hashlib.sha256(
            json.dumps(prompt_ids).encode()
        ).hexdigest()
        result["prompt_tail"] = tokenizer.decode(prompt_ids[-160:])
        kwargs = dict(
            model=args.model,
            dtype="float16" if args.platform == "kunlun" else "bfloat16",
            tensor_parallel_size=args.tp,
            enforce_eager=args.enforce_eager,
            max_model_len=((args.input_len + args.output_len + 383) // 128) * 128,
            max_num_seqs=args.max_num_seqs or max(batches),
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.memory,
            enable_prefix_caching=False,
            trust_remote_code=True,
            seed=0,
        )
        if args.platform in ("ascend", "kunlun"):
            kwargs["block_size"] = 128
        if args.platform == "ascend" and not args.enforce_eager:
            from vllm.config import CompilationMode, CUDAGraphMode

            kwargs["compilation_config"] = dict(
                mode=CompilationMode.VLLM_COMPILE,
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=capture_sizes,
                cudagraph_num_of_warmups=1,
            )
            kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        if args.platform == "metax" and not args.enforce_eager:
            kwargs["compilation_config"] = dict(
                cudagraph_capture_sizes=capture_sizes,
                cudagraph_num_of_warmups=1,
            )
        if args.platform == "kunlun":
            kwargs["enable_chunked_prefill"] = False
            kwargs["compilation_config"] = dict(
                level=3 if not args.enforce_eager else 0, backend="eager",
                cudagraph_capture_sizes=capture_sizes,
                cudagraph_num_of_warmups=1,
            )
        result["llm_kwargs"] = json.loads(json.dumps(kwargs, default=str))
        save()
        llm = LLM(**kwargs)
        step_times = []
        progress_phase = "startup"
        progress_started = time.perf_counter()
        if args.step_timing:
            original_step = llm.llm_engine.step
            def timed_step(*a, **kw):
                started = time.perf_counter()
                outputs = original_step(*a, **kw)
                step_times.append(time.perf_counter() - started)
                if args.progress_every > 0 and len(step_times) % args.progress_every == 0:
                    print("STEP_PROGRESS", json.dumps({
                        "phase": progress_phase, "steps": len(step_times),
                        "elapsed_s": round(time.perf_counter() - progress_started, 2),
                    }), flush=True)
                return outputs
            llm.llm_engine.step = timed_step
        result["startup_workers"] = llm.collective_rpc(worker_state)
        save()
        if args.mode == "infinicore":
            required = requested_routes
            for state in result["startup_workers"]:
                installed = (state.get("registration") or {}).get("installed_routes", [])
                if not required.issubset(installed):
                    raise RuntimeError(f"Missing requested attention routes: {state}")
        sampling = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1,
            ignore_eos=True, min_tokens=args.output_len,
            max_tokens=args.output_len,
        )
        warmup_sampling = sampling if args.warmup_output_len is None else SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, ignore_eos=True,
            min_tokens=args.warmup_output_len, max_tokens=args.warmup_output_len)
        for batch in batches:
            requests = [{"prompt_token_ids": prompt_ids} for _ in range(batch)]
            if not args.skip_warmup:
                progress_phase = f"warmup:bs{batch}"
                progress_started = time.perf_counter()
                llm.generate(requests, warmup_sampling, use_tqdm=False)  # untimed warmup
            for repeat in range(args.repeats):
                before = llm.collective_rpc(worker_state)
                step_times.clear()
                start = time.perf_counter()
                progress_phase = f"measure:bs{batch}:repeat{repeat}"
                progress_started = start
                outputs = llm.generate(requests, sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
                after = llm.collective_rpc(worker_state)
                records = []
                for output in outputs:
                    completion = output.outputs[0]
                    ids = list(completion.token_ids)
                    health = text_health(completion.text, ids)
                    if len(output.prompt_token_ids) != args.input_len:
                        health["issues"].append("input_length_mismatch")
                    if len(ids) != args.output_len:
                        health["issues"].append("output_length_mismatch")
                    records.append({
                        "input_tokens": len(output.prompt_token_ids),
                        "output_tokens": len(ids),
                        "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        "health": health,
                        "text": completion.text,
                    })
                generated = sum(rec["output_tokens"] for rec in records)
                row = {
                    "batch": batch, "repeat": repeat,
                    "elapsed_s": elapsed, "output_tokens": generated,
                    "output_tps": generated / elapsed,
                    "graph_captures_before": [s.get("graph_captures") for s in before],
                    "graph_captures_after": [s.get("graph_captures") for s in after],
                    "graph_replays": aggregate_delta(before, after, "graph_replays"),
                    "attention_calls": aggregate_delta(before, after, "attention_calls"),
                    "attention_fallbacks": aggregate_delta(before, after, "attention_fallbacks"),
                    "native_attention_calls": aggregate_delta(before, after, "native_attention_calls"),
                    "backend_calls": aggregate_delta(before, after, "backend_calls"),
                    "bridge_calls": aggregate_delta(before, after, "bridge_calls"),
                    "fallback_calls": aggregate_delta(before, after, "fallback_calls"),
                    "outputs": records,
                }
                if args.step_timing:
                    row["step_times_s"] = list(step_times)
                    middle = step_times[len(step_times)//4:3*len(step_times)//4]
                    row["middle_half_step_median_s"] = statistics.median(middle) if middle else None
                result["rows"].append(row)
                print("MEASURE", json.dumps({
                    "batch": batch, "repeat": repeat,
                    "elapsed_s": round(elapsed, 2),
                    "output_tps": round(row["output_tps"], 2),
                    "issues": [r["health"]["issues"] for r in records],
                    "backend_calls": row["backend_calls"],
                    "bridge_calls": row["bridge_calls"],
                }), flush=True)
                save()
        result["final_workers"] = llm.collective_rpc(worker_state)
        result["summary"] = {
            str(batch): statistics.median(
                row["output_tps"] for row in result["rows"]
                if row["batch"] == batch
            ) for batch in batches
        }
        if args.mode == "infinicore":
            for state in result["final_workers"]:
                registration = state.get("registration") or {}
                installed = registration.get("installed_routes") or []
                if not installed:
                    result["errors"].append("no_plugin_routes_installed")
                required = requested_routes & {"StoreKVCache", "PagedAttentionPrefill", "PagedAttentionDecode"}
                if not required.issubset(installed):
                    result["errors"].append("missing_attention_routes")
                calls = state.get("attention_calls", {})
                if any(calls.get(name, 0) == 0 for name in required):
                    result["errors"].append("missing_attention_launches")
                if state.get("attention_fallbacks"):
                    result["errors"].append("native_attention_fallback")
                if recommended_selected() or args.expect_native_attention:
                    for name in ("PagedAttentionPrefill", "PagedAttentionDecode"):
                        if calls.get(name, 0) or not state.get("native_attention_calls", {}).get(name, 0):
                            result["errors"].append(f"expected_attention_not_native:{name}")
                if args.expect_infinicore_store_all and (
                    not calls.get("StoreKVCache", 0)
                    or state.get("native_attention_calls", {}).get("StoreKVCache", 0)
                ):
                    result["errors"].append("expected_all_store_infinicore")
                for name, counter in {
                    "RMSNorm": "rms_norm", "SiluAndMul": "silu_and_mul",
                    "Embedding": "embedding", "MatMul": "linear", "LMHead": "lm_head",
                }.items():
                    if name in requested_routes and not state.get("backend_calls", {}).get(counter, 0):
                        result["errors"].append(f"missing_operator_launches:{name}")
                if args.expect_fused_rms and not state.get("backend_calls", {}).get("fused_add_rms_norm", 0):
                    result["errors"].append("missing_fused_rms_launches")
                if "RoPE" in requested_routes:
                    if "RoPE" not in installed or not (
                        state.get("backend_calls", {}).get("rotary_embedding", 0)
                        or state.get("bridge_calls", {}).get("RoPE", 0)
                    ):
                        result["errors"].append("missing_rope_launches")
        if not args.enforce_eager and not all(
            s.get("graph_captures", 0) > 0 for s in result["final_workers"]
        ):
            result["errors"].append("no_graph_capture")
        if not args.enforce_eager and not all(
            sum(s.get("graph_replays", {}).values()) > 0 for s in result["final_workers"]
        ):
            result["errors"].append("no_graph_replay")
        if any(r["health"]["issues"] for row in result["rows"] for r in row["outputs"]):
            result["errors"].append("output_health_issues")
        result["completed"] = True
    except Exception:
        result["errors"].append(traceback.format_exc())
        print(result["errors"][-1], flush=True)
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        print("RESULT_PATH", args.output, flush=True)
        if llm is not None:
            del llm
    return 0 if result["completed"] and not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
