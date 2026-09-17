"""Eager or graph Ascend throughput case with per-route InfiniCore evidence.

Records what actually executed, not just how long it took: InfiniCore call and
fallback counters per route, graph captures, model forward passes and ACL graph
replays. A case whose installed route records no InfiniCore call fails, so a
silently-native run cannot be reported as an InfiniCore result, and the pass and
replay counts are comparable across both engines when one is unexpectedly faster.

Note that every `VLLM_INFINICORE_*` variable is cleared before the engine starts,
so overrides have to be flags on this script rather than environment variables.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROUTE_COUNTERS = {
    "RMSNorm": ("rms_norm", "fused_add_rms_norm"),
    "SiluAndMul": ("silu_and_mul",),
    "RoPE": ("rotary_embedding",),
    "Embedding": ("embedding",),
    "MatMul": ("linear",),
    "LMHead": ("lm_head",),
    "StoreKVCache": ("store_kv_cache",),
    "PagedAttentionPrefill": ("paged_attention_prefill",),
    "PagedAttentionDecode": ("paged_attention_decode",),
}


def worker_state(worker):
    import dataclasses
    import os
    from vllm.compilation.counter import compilation_counter
    from vllm_infinicore.operators import backend as infinicore_backend
    from vllm_infinicore import plugin

    registration = plugin._REGISTRATION_RESULT
    return {
        "pid": os.getpid(),
        "rank": worker.rank,
        "ascend_library": os.environ.get("VLLM_INFINICORE_ASCEND_LIBRARY"),
        "captures": compilation_counter.num_cudagraph_captured,
        "calls": infinicore_backend.backend_call_counts(),
        "fallbacks": infinicore_backend.backend_fallback_counts(),
        "fallback_reasons": infinicore_backend.backend_fallback_reasons(),
        "registration": dataclasses.asdict(registration)
        if registration is not None
        else None,
    }


def native_worker_state(worker):
    import os
    from vllm.compilation.counter import compilation_counter

    return {
        "pid": os.getpid(),
        "rank": worker.rank,
        "captures": compilation_counter.num_cudagraph_captured,
        "registration": None,
    }


def instrument_steps(worker):
    """Count model forward passes, for either engine.

    InfiniCore call counters only exist when the plugin is loaded, so they
    cannot say whether the native engine is doing more work. A step count is
    comparable across both, and separates "our operators are slower" from "the
    scheduler ran more passes", e.g. by recomputing preempted prefills.
    """
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return -1
    if not hasattr(runner, "_bench_steps"):
        runner._bench_steps = 0
        original = runner.execute_model

        def counted(*args, **kwargs):
            runner._bench_steps += 1
            return original(*args, **kwargs)

        runner.execute_model = counted
    return runner._bench_steps


def instrument_replays(worker):
    """Count ACL graph replays, for either engine.

    Equal step counts with unequal time can mean one engine replays a captured
    decode graph while the other re-executes it, which is invisible to both the
    step count and the InfiniCore call counters.
    """
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation import acl_graph

    if not hasattr(acl_graph, "_bench_replays"):
        acl_graph._bench_replays = 0
        original = acl_graph.ACLGraphWrapper.__call__

        def counted(self, *args, **kwargs):
            context = get_forward_context()
            entry = self.concrete_aclgraph_entries.get(context.batch_descriptor)
            replayed = (
                context.cudagraph_runtime_mode != CUDAGraphMode.NONE
                and context.cudagraph_runtime_mode == self.runtime_mode
                and entry is not None
                and entry.aclgraph is not None
            )
            result = original(self, *args, **kwargs)
            if replayed:
                acl_graph._bench_replays += 1
            return result

        acl_graph.ACLGraphWrapper.__call__ = counted
    return acl_graph._bench_replays


def replay_count(worker):
    from vllm_ascend.compilation import acl_graph

    return getattr(acl_graph, "_bench_replays", -1)


def step_count(worker):
    return getattr(getattr(worker, "model_runner", None), "_bench_steps", -1)


def sync_worker(worker):
    import torch

    torch.npu.synchronize()


def reap_workers():
    """Kill leftover child workers after a failed engine construction.

    An LLM() that raises part-way leaves its spawned tensor-parallel workers
    alive, each still holding its share of device memory. Nothing later can
    allocate on those cards, so every subsequent case fails on free memory
    rather than on its own merits.
    """
    import signal

    me = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        if f"PPid:\t{me}\n" not in status:
            continue
        try:
            os.kill(int(entry.name), signal.SIGKILL)
        except OSError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=["prepare", "native", "infinicore"])
    p.add_argument("--root", required=True)
    p.add_argument("--model", default="/models/Qwen3.8-27B")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--devices", help="ASCEND_RT_VISIBLE_DEVICES; defaults to 0..tp-1")
    p.add_argument("--batches", default="1,4,16,32")
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024)
    p.add_argument("--memory", type=float, default=0.95)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--routes", default="all")
    p.add_argument("--strict-backend", default="0")
    p.add_argument("--graph", action="store_true", help="Compile instead of eager")
    p.add_argument(
        "--no-chunked-prefill",
        action="store_true",
        help="Keep prefill out of decode steps so FULL_DECODE_ONLY can capture them; "
        "needs --max-num-batched-tokens >= input_len + output_len",
    )
    p.add_argument("--library", default="/workspace/infinicore-build/libvllm_infinicore_ascend.so")
    a = p.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    batches = [int(x) for x in a.batches.split(",")]
    for key in list(os.environ):
        if key.startswith("VLLM_INFINICORE_"):
            del os.environ[key]
    devices = a.devices or ",".join(map(str, range(a.tp)))
    os.environ.update(
        ASCEND_RT_VISIBLE_DEVICES=devices,
        HF_HUB_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_PLUGINS="ascend,ascend_kv_connector,ascend_model,ascend_model_loader,"
        "ascend_service_profiling" + (",vllm_infinicore" if a.case == "infinicore" else ""),
        VLLM_INFINICORE_ENABLE_PATCHES="1" if a.case == "infinicore" else "0",
        VLLM_INFINICORE_ROUTES=a.routes,
        VLLM_INFINICORE_ASCEND_LIBRARY=a.library,
        VLLM_INFINICORE_FORCE_NATIVE_FALLBACK="0",
        VLLM_INFINICORE_STRICT_BACKEND=a.strict_backend,
    )
    result = dict(
        case=a.case, args=vars(a), devices=devices, completed=False,
        validation_errors=[], rows=[],
    )
    mode = "graph" if a.graph else "eager"
    dest = root / f"{a.case}-tp{a.tp}-{mode}.json"

    def save():
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    llm = None
    try:
        if a.case == "prepare":
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
            message = (
                "Write a detailed educational chapter on how computers process information. "
                "Explain processors, memory, storage, networking, algorithms, practical examples, "
                "limitations, and future developments in connected prose. "
            )
            seed = tok.encode(message, add_special_tokens=False)
            ids = (seed * (a.input_len // len(seed) + 1))[: a.input_len]
            (root / "prompts.json").write_text(
                json.dumps(dict(token_ids=ids, input_len=a.input_len))
            )
            result["completed"] = True
            return 0
        import importlib.metadata
        from vllm import LLM, SamplingParams
        from vllm.config import CompilationMode, CUDAGraphMode
        from vllm_infinicore.common.validation import compute_text_health, detect_degenerate_repetition

        result["versions"] = {
            n: importlib.metadata.version(n) for n in ["vllm", "vllm_ascend", "torch", "torch_npu"]
        }
        prompt_data = json.loads((root / "prompts.json").read_text())
        ids = prompt_data["token_ids"]
        assert len(ids) == a.input_len
        result["prompt_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        kwargs = dict(
            model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16",
            max_model_len=a.input_len + a.output_len, max_num_seqs=a.max_num_seqs,
            max_num_batched_tokens=a.max_num_batched_tokens,
            gpu_memory_utilization=a.memory, enforce_eager=not a.graph,
            enable_prefix_caching=False, seed=0,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        if a.no_chunked_prefill:
            # A step mixing prefill with decode is not a pure-decode batch, so
            # FULL_DECODE_ONLY cannot capture it and it runs in Python.
            if a.max_num_batched_tokens < a.input_len + a.output_len:
                p.error(
                    "--no-chunked-prefill needs --max-num-batched-tokens >= "
                    f"{a.input_len + a.output_len}"
                )
            kwargs["enable_chunked_prefill"] = False
        if a.graph:
            sizes = [x for x in [1, 2, 4, 8, 16, 32] if x <= a.max_num_seqs]
            kwargs["compilation_config"] = dict(
                mode=CompilationMode.VLLM_COMPILE,
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=sizes, cudagraph_num_of_warmups=1,
            )
        result["llm_kwargs"] = json.loads(json.dumps(kwargs, default=str))
        save()
        probe = worker_state if a.case == "infinicore" else native_worker_state
        llm = LLM(**kwargs)
        result["startup_workers"] = llm.collective_rpc(probe)
        result["steps_instrumented"] = llm.collective_rpc(instrument_steps)
        result["replays_instrumented"] = llm.collective_rpc(instrument_replays)
        assert len(result["startup_workers"]) == a.tp
        params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, ignore_eos=True,
            min_tokens=a.output_len, max_tokens=a.output_len,
        )
        for bs in batches:
            requests = [{"prompt_token_ids": ids} for _ in range(bs)]
            preview = None
            for _ in range(a.warmups):
                preview = llm.generate(requests, params, use_tqdm=False)
            if preview is not None:
                print("PREVIEW", bs, preview[0].outputs[0].text[:300], flush=True)
            for rep in range(a.repeats):
                llm.collective_rpc(sync_worker)
                before = llm.collective_rpc(probe)
                steps_before = llm.collective_rpc(step_count)
                replays_before = llm.collective_rpc(replay_count)
                started = time.perf_counter()
                outputs = llm.generate(requests, params, use_tqdm=False)
                elapsed = time.perf_counter() - started
                after = llm.collective_rpc(probe)
                steps_after = llm.collective_rpc(step_count)
                replays_after = llm.collective_rpc(replay_count)
                records = []
                for output in outputs:
                    completion = output.outputs[0]
                    tokens = list(completion.token_ids)
                    health = compute_text_health(completion.text, tokens)
                    repetition = detect_degenerate_repetition(tokens)
                    errors = health.validation_errors()
                    if repetition.is_degenerate:
                        errors += list(repetition.reasons)
                    if len(tokens) != a.output_len or len(output.prompt_token_ids) != a.input_len:
                        errors.append("token_count_mismatch")
                    result["validation_errors"].extend(errors)
                    records.append(
                        dict(
                            input_tokens=len(output.prompt_token_ids),
                            output_tokens=len(tokens), token_ids=tokens,
                            text=completion.text, text_health=health.as_dict(),
                            repetition=repetition.as_dict(),
                        )
                    )
                generated = sum(x["output_tokens"] for x in records)
                row = dict(
                    batch_size=bs, repeat=rep, elapsed_s=elapsed, output_tokens=generated,
                    output_tps=generated / elapsed, workers_before=before,
                    workers_after=after,
                    forward_steps=[y - x for x, y in zip(steps_before, steps_after)],
                    graph_replays=[y - x for x, y in zip(replays_before, replays_after)],
                    outputs=records,
                )
                result["rows"].append(row)
                delta = {}
                if a.case == "infinicore":
                    for b, c in zip(before, after):
                        for op, n in c["calls"].items():
                            delta[op] = delta.get(op, 0) + n - b["calls"].get(op, 0)
                print(
                    "MEASURE",
                    json.dumps(dict(batch_size=bs, repeat=rep, elapsed_s=round(elapsed, 3),
                                    output_tokens=generated,
                                    output_tps=round(generated / elapsed, 2),
                                    infinicore_calls=delta)),
                    flush=True,
                )
                save()
        result["summary"] = {
            str(bs): dict(
                median_output_tps=statistics.median(
                    x["output_tps"] for x in result["rows"] if x["batch_size"] == bs
                )
            )
            for bs in batches
        }
        result["final_workers"] = llm.collective_rpc(probe)
        if a.case == "infinicore":
            for state in result["final_workers"]:
                registration = state.get("registration") or {}
                for route in registration.get("installed_routes", []):
                    counters = ROUTE_COUNTERS.get(route, ())
                    if not any(state["calls"].get(op, 0) for op in counters):
                        result["validation_errors"].append(
                            f"installed_route_has_no_backend_calls:{route}"
                        )
        result["completed"] = True
    except Exception:
        result["exception"] = traceback.format_exc()
        print(result["exception"], flush=True)
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=10.0)
            except Exception:
                result["shutdown_exception"] = traceback.format_exc()
        elif a.case != "prepare":
            reap_workers()
        save()
        print("RESULT_PATH", dest, flush=True)
    return 0 if result["completed"] and not result["validation_errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
