"""Run one isolated Ascend availability case; this is not a performance benchmark."""

import argparse
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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


def validate_worker_routes(states, *, allow_native_fallback):
    errors = []
    if not states:
        return ["worker_state_missing"]
    for state in states:
        registration = state.get("registration")
        if not registration:
            errors.append("worker_registration_missing")
            continue
        if registration.get("failure_reason"):
            errors.append(registration["failure_reason"])
        installed = set(registration["installed_routes"])
        fallback = {
            route["name"]
            for route in registration["route_states"]
            if route["status"] == "native_fallback"
            and route["reason"]
            and route["native_fallback"]
        }
        covered = installed | fallback if allow_native_fallback else installed
        if covered != set(registration["requested_routes"]):
            errors.append("requested_routes_not_covered")
        counts = state["backend_call_counts"]
        for route in installed:
            if not any(counts.get(op, 0) for op in ROUTE_COUNTERS[route]):
                errors.append(f"installed_route_has_no_backend_calls:{route}")
    return errors


def worker_state(worker):
    import dataclasses
    import os
    from vllm_infinicore.ops import infinicore_backend
    from vllm_infinicore import plugin
    from vllm.model_executor.custom_op import op_registry_oot

    registration = plugin._REGISTRATION_RESULT
    return {
        "pid": os.getpid(),
        "ascend_library": os.environ.get("VLLM_INFINICORE_ASCEND_LIBRARY"),
        "backend_call_counts": infinicore_backend.backend_call_counts(),
        "backend_fallback_counts": infinicore_backend.backend_fallback_counts(),
        "backend_fallback_reasons": infinicore_backend.backend_fallback_reasons(),
        "registration": dataclasses.asdict(registration)
        if registration is not None
        else None,
        "oot_classes": {
            k: v.__module__ + "." + v.__name__ for k, v in op_registry_oot.items()
        },
    }


def reset_worker_counts(worker):
    from vllm_infinicore.ops import infinicore_backend

    infinicore_backend.reset_backend_call_counts()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=[
            "prepare",
            "native",
            "off",
            "fallback",
            "all",
            "autoall",
            "projections",
        ],
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--ascend-library", help="Locked InfiniCore Ascend bridge shared library")
    parser.add_argument("--model", default="/models/Qwen3-0.6B")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--allow-native-fallback", action="store_true")
    parser.add_argument("--auto-discover-plugins", action="store_true")
    args = parser.parse_args()
    if args.auto_discover_plugins and args.case != "autoall":
        parser.error("--auto-discover-plugins requires the autoall case")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for name in list(os.environ):
        if name.startswith("VLLM_INFINICORE_"):
            del os.environ[name]
    if args.ascend_library:
        os.environ["VLLM_INFINICORE_ASCEND_LIBRARY"] = args.ascend_library
    os.environ.update(
        ASCEND_RT_VISIBLE_DEVICES=args.devices,
        HF_HUB_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
    )
    ascend_plugins = "ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling"
    os.environ["VLLM_PLUGINS"] = ascend_plugins + (
        ",vllm_infinicore" if args.case not in {"prepare", "native"} else ""
    )
    if args.auto_discover_plugins:
        os.environ.pop("VLLM_PLUGINS")
    os.environ["VLLM_INFINICORE_ENABLE_PATCHES"] = (
        "1"
        if args.case in {"fallback", "all", "autoall", "projections"}
        else "0"
    )
    os.environ["VLLM_INFINICORE_ROUTES"] = {
        "projections": "Embedding,MatMul,LMHead",
    }.get(args.case, "all")
    os.environ["VLLM_INFINICORE_FORCE_NATIVE_FALLBACK"] = (
        "1" if args.case == "fallback" else "0"
    )
    os.environ["VLLM_INFINICORE_STRICT_BACKEND"] = "1"
    result = {
        "case": args.case,
        "model": args.model,
        "native_fallback_allowed": args.allow_native_fallback,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("VLLM_") or k == "ASCEND_RT_VISIBLE_DEVICES"
        },
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["torch", "torch_npu", "vllm", "vllm_ascend"]
        },
        "validation_errors": [],
    }
    llm = None
    try:
        if args.case == "prepare":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                result["model"], local_files_only=True
            )
            seed = tokenizer.encode(
                "Explain how computers process information and help people solve everyday problems. ",
                add_special_tokens=False,
            )
            ids = (seed * (128 // len(seed) + 1))[:128]
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": "请用一句中文解释什么是人工智能。"}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            if hasattr(chat, "keys"):
                chat = chat["input_ids"]
            (root / "prompts.json").write_text(
                json.dumps({"exact128": ids, "chat": chat})
            )
            result["prompt_lengths"] = {"exact128": len(ids), "chat": len(chat)}
        else:
            from vllm import LLM, SamplingParams
            from vllm.platforms import current_platform

            result["platform"] = (
                type(current_platform).__module__
                + "."
                + type(current_platform).__name__
            )
            if args.case not in {"native", "autoall"}:
                import vllm_infinicore

                result["registration"] = dataclasses.asdict(vllm_infinicore.register())
            prompts = json.loads((root / "prompts.json").read_text())
            result["llm_kwargs"] = dict(
                model=result["model"],
                tensor_parallel_size=args.tensor_parallel_size,
                dtype="bfloat16",
                max_model_len=512,
                max_num_seqs=4,
                max_num_batched_tokens=512,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=True,
                seed=0,
                enable_prefix_caching=False,
            )
            if args.text_only:
                result["llm_kwargs"]["limit_mm_per_prompt"] = {"image": 0, "video": 0}
            llm = LLM(**result["llm_kwargs"])
            params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                ignore_eos=True,
                min_tokens=32,
                max_tokens=32,
            )
            from vllm_infinicore.validation import (
                compute_text_health,
                detect_degenerate_repetition,
            )
            from vllm_infinicore.ops import infinicore_backend as backend

            llm.generate(
                [{"prompt_token_ids": prompts["exact128"]}], params, use_tqdm=False
            )
            backend.reset_backend_call_counts()
            llm.collective_rpc(reset_worker_counts)
            result["outputs"] = []
            for shape, batch_size in [("exact128", 1), ("exact128", 4), ("chat", 1)]:
                for repeat in range(2):
                    outputs = llm.generate(
                        [
                            {"prompt_token_ids": prompts[shape]}
                            for _ in range(batch_size)
                        ],
                        params,
                        use_tqdm=False,
                    )
                    for index, output in enumerate(outputs):
                        completion = output.outputs[0]
                        tokens = list(completion.token_ids)
                        health = compute_text_health(completion.text, tokens)
                        repetition = detect_degenerate_repetition(tokens)
                        row = dict(
                            shape=shape,
                            batch_size=batch_size,
                            repeat=repeat,
                            index=index,
                            input_tokens=len(output.prompt_token_ids),
                            output_tokens=len(tokens),
                            token_ids=tokens,
                            text=completion.text,
                            text_health=health.as_dict(),
                            repetition=repetition.as_dict(),
                        )
                        result["outputs"].append(row)
                        result["validation_errors"].extend(health.validation_errors())
                        if (
                            row["input_tokens"] != len(prompts[shape])
                            or len(tokens) != 32
                        ):
                            result["validation_errors"].append("token_count_mismatch")
                        if repetition.is_degenerate:
                            result["validation_errors"].extend(repetition.reasons)
                        print("OUTPUT", json.dumps(row, ensure_ascii=False), flush=True)
            result["backend_call_counts"] = backend.backend_call_counts()
            result["worker_states"] = llm.collective_rpc(worker_state)
            if len(result["worker_states"]) != args.tensor_parallel_size:
                result["validation_errors"].append("tensor_parallel_worker_count_mismatch")
            if args.case == "autoall":
                result["registration"] = result["worker_states"][0]["registration"]
            if args.case in {"all", "autoall", "projections"}:
                result["validation_errors"].extend(
                    validate_worker_routes(
                        result["worker_states"],
                        allow_native_fallback=args.allow_native_fallback,
                    )
                )
            if args.case != "native" and (root / "native.json").exists():
                baseline = json.loads((root / "native.json").read_text())["outputs"]
                pairs = zip(baseline, result["outputs"])
                result["outputs_matching_native"] = sum(
                    a["token_ids"] == b["token_ids"] for a, b in pairs
                )
                if len(baseline) != len(result["outputs"]) or result[
                    "outputs_matching_native"
                ] != len(baseline):
                    result["validation_errors"].append(
                        "output_tokens_differ_from_native"
                    )
        result["completed"] = True
    except Exception:
        result["completed"] = False
        result["exception"] = traceback.format_exc()
        print(result["exception"], flush=True)
    finally:
        if args.case == "autoall":
            from vllm_infinicore import plugin

            if plugin._REGISTRATION_RESULT is not None:
                result["registration"] = dataclasses.asdict(plugin._REGISTRATION_RESULT)
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=10.0)
            except Exception:
                result["shutdown_exception"] = traceback.format_exc()
                result["validation_errors"].append("shutdown_failed")
        (root / (args.case + ".json")).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["completed"] and not result["validation_errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
