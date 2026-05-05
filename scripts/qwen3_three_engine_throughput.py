#!/usr/bin/env python
"""Qwen3-8B throughput benchmark for InfiniLM, vLLM native, and vLLM-InfiniCore."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "/mnt/geogpt-doc-new/default/xb/qwen3-8B"
DEFAULT_INFINILM_ROOT = "/root/InfiniLM"
ENGINES = ("infinilm", "vllm-native", "vllm-infinicore")
DEFAULT_INFINICORE_THROUGHPUT_ROUTES = "throughput"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=4096)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--infinilm-root", default=DEFAULT_INFINILM_ROOT)
    parser.add_argument("--infini-device", default="cuda")
    parser.add_argument("--infini-attn", default="flash-attn")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--infinicore-routes", default=DEFAULT_INFINICORE_THROUGHPUT_ROUTES)
    parser.add_argument("--infinicore-disabled-routes", default="")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", default="", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def configure_runtime_environment() -> None:
    site_packages = "/opt/conda/lib/python3.12/site-packages"
    maca_path = "/opt/maca-3.5.3"
    infini_root = os.path.expanduser("~/.infini")
    torch_lib = f"{site_packages}/torch/lib"

    os.environ.setdefault("MACA_PATH", maca_path)
    os.environ.setdefault("MACA_HOME", maca_path)
    os.environ.setdefault("MACA_ROOT", maca_path)
    os.environ.setdefault("INFINI_ROOT", infini_root)
    os.environ.setdefault("PYTHON_SITE_PACKAGES", site_packages)
    os.environ.setdefault("TORCH_LIB", torch_lib)
    os.environ.setdefault(
        "FLASH_ATTN_2_CUDA_SO",
        f"{site_packages}/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so",
    )
    os.environ.setdefault("XMAKE_ROOT", "y")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    _prepend_env(
        "PATH",
        [
            "/mnt/geogpt-doc-new/default/xmake_env/bin",
            os.path.expanduser("~/.local/bin"),
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            f"{maca_path}/bin",
        ],
    )
    _prepend_env(
        "LD_LIBRARY_PATH",
        [
            "/opt/conda/lib",
            torch_lib,
            f"{infini_root}/lib",
            f"{maca_path}/lib",
            f"{maca_path}/lib64",
        ],
    )


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_hash(prompt_ids: list[int]) -> str:
    raw = json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def text_health(text: str) -> dict[str, Any]:
    replacement = text.count("\ufffd")
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    return {
        "chars": len(text),
        "replacement_chars": replacement,
        "control_chars": control,
        "looks_garbled": bool(replacement or control),
    }


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "stdev": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def build_manifest(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = build_prompt_ids(tokenizer, args.input_len)
    prompt_payload = {
        "input_len": args.input_len,
        "prompt_token_ids": prompt_ids,
        "prompt_sha256": prompt_hash(prompt_ids),
        "prompt_preview": tokenizer.decode(prompt_ids[:256]),
    }
    prompt_path = run_dir / f"prompt-in{args.input_len}.json"
    write_json(prompt_path, prompt_payload)

    requested = [item.strip() for item in args.engines.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(ENGINES))
    if unknown:
        raise ValueError(f"unknown engine(s): {unknown}")

    cases = []
    for engine in requested:
        case_id = f"{engine}-graph-bs{args.batch_size}-in{args.input_len}-out{args.output_len}"
        cases.append(
            {
                "case_id": case_id,
                "engine": engine,
                "result_path": str(run_dir / "results" / f"{case_id}.json"),
                "log_path": str(run_dir / "logs" / f"{case_id}.log"),
            }
        )

    return {
        "schema_version": 1,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "model": args.model,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "graph_mode": True,
        "prompt_path": str(prompt_path),
        "infinilm_root": args.infinilm_root,
        "infini_device": args.infini_device,
        "infini_attn": args.infini_attn,
        "block_size": args.block_size,
        "infinicore_routes": args.infinicore_routes,
        "infinicore_disabled_routes": args.infinicore_disabled_routes,
        "cases": cases,
    }


def parent_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else Path("artifacts") / f"qwen3-three-engine-bs{args.batch_size}-in{args.input_len}-out{args.output_len}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, run_dir)
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)

    results = []
    for case in manifest["cases"]:
        log_path = Path(case["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--manifest",
            str(manifest_path),
            "--case-id",
            case["case_id"],
        ]
        print(f"running {case['case_id']} ...", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=_worker_env(manifest, case["engine"]),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        result_path = Path(case["result_path"])
        if result_path.exists():
            result = read_json(result_path)
            result["returncode"] = proc.returncode
        else:
            result = {
                "case_id": case["case_id"],
                "engine": case["engine"],
                "valid": False,
                "returncode": proc.returncode,
                "error": f"worker produced no result; see {log_path}",
            }
            write_json(result_path, result)
        results.append(result)
        print(_result_line(result), flush=True)

    write_json(run_dir / "summary.json", {"results": results})
    _write_summary_md(run_dir, results)
    valid = sum(1 for result in results if result.get("valid"))
    print(f"summary: {valid}/{len(results)} valid; run_dir={run_dir}", flush=True)
    return 0 if valid == len(results) else 1


def worker_main(args: argparse.Namespace) -> int:
    configure_runtime_environment()
    manifest = read_json(Path(args.manifest))
    case = next(item for item in manifest["cases"] if item["case_id"] == args.case_id)
    prompt_payload = read_json(Path(manifest["prompt_path"]))
    try:
        if case["engine"] in {"vllm-native", "vllm-infinicore"}:
            result = run_vllm(case, manifest, prompt_payload)
        else:
            result = run_infinilm(case, manifest, prompt_payload)
    except BaseException as exc:
        result = {
            "case_id": case["case_id"],
            "engine": case["engine"],
            "valid": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    write_json(Path(case["result_path"]), result)
    if case["engine"] == "infinilm":
        os._exit(0)
    return 0 if result.get("valid") else 1


def run_vllm(case: dict[str, Any], manifest: dict[str, Any], prompt_payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config import CUDAGraphMode
    from vllm.inputs import TokensPrompt

    if case["engine"] == "vllm-infinicore":
        os.environ["VLLM_INFINICORE_ENABLE_PATCHES"] = "1"
        os.environ["VLLM_INFINICORE_ROUTES"] = manifest["infinicore_routes"]
        os.environ["VLLM_INFINICORE_FORCE_NATIVE_FALLBACK"] = "0"
        os.environ["VLLM_INFINICORE_STRICT_BACKEND"] = "1"
        os.environ["VLLM_INFINICORE_DISABLE_REAL_BACKEND"] = "0"
        disabled_routes = manifest.get("infinicore_disabled_routes") or ""
        if disabled_routes:
            os.environ["VLLM_INFINICORE_DISABLED_ROUTES"] = disabled_routes
        else:
            os.environ.pop("VLLM_INFINICORE_DISABLED_ROUTES", None)
        import vllm_infinicore

        registration = vllm_infinicore.register()
    else:
        registration = None

    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], trust_remote_code=True)
    prompt_ids = prompt_payload["prompt_token_ids"]
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(manifest["batch_size"])]
    comp_config = {
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "cudagraph_num_of_warmups": 1,
        "backend": "eager",
    }
    llm = LLM(
        model=manifest["model"],
        dtype=manifest["dtype"],
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=manifest["gpu_memory_utilization"],
        max_model_len=manifest["max_model_len"],
        enforce_eager=False,
        compilation_config=comp_config,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=manifest["output_len"],
        min_tokens=manifest["output_len"],
        ignore_eos=True,
        detokenize=True,
    )

    def generate_once() -> tuple[int, list[str], list[list[int]]]:
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        texts: list[str] = []
        ids_by_request: list[list[int]] = []
        total_tokens = 0
        for output in outputs:
            completion = output.outputs[0]
            ids = [int(token) for token in completion.token_ids]
            text = completion.text or tokenizer.decode(ids, skip_special_tokens=True)
            ids_by_request.append(ids)
            texts.append(text)
            total_tokens += len(ids)
        return total_tokens, texts, ids_by_request

    for _ in range(manifest["warmup"]):
        generate_once()

    _reset_infinicore_counts()
    iterations = []
    for index in range(1, manifest["repeats"] + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        total_tokens, texts, ids_by_request = generate_once()
        elapsed = time.perf_counter() - start
        iterations.append(
            _iteration_result(
                index,
                elapsed,
                total_tokens,
                texts,
                ids_by_request,
                manifest,
            )
        )

    result = _base_result(case, manifest, prompt_payload)
    result.update(
        {
            "vllm_compilation_config": str(comp_config),
            "registration": _registration_dict(registration),
            "infinicore_backend_call_counts": _infinicore_backend_counts(),
            "infinicore_attention_route_counts": _infinicore_attention_counts(),
            "graph_capture_count": _vllm_graph_count(),
        }
    )
    return _finalize(result, iterations)


def run_infinilm(case: dict[str, Any], manifest: dict[str, Any], prompt_payload: dict[str, Any]) -> dict[str, Any]:
    infini_python = str(Path(manifest["infinilm_root"]) / "python")
    if infini_python not in sys.path:
        sys.path.insert(0, infini_python)

    import infinicore
    import numpy as np
    from transformers import AutoTokenizer
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import GenerationConfig, InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], trust_remote_code=True)
    prompt_ids = prompt_payload["prompt_token_ids"]
    max_blocks_per_request = math.ceil((manifest["input_len"] + manifest["output_len"]) / manifest["block_size"])
    max_blocks = max_blocks_per_request * manifest["batch_size"]
    cache_config = PagedKVCacheConfig(max_blocks, manifest["block_size"])
    engine = InferEngine(
        manifest["model"],
        device=infinicore.device(manifest["infini_device"], 0),
        distributed_config=DistConfig(1),
        cache_config=cache_config,
        enable_graph_compiling=True,
        attention_backend=manifest["infini_attn"],
    )
    load_model_state_dict_by_file(engine, manifest["model"], dtype=engine.dtype)
    gen_config = GenerationConfig(
        max_new_tokens=manifest["output_len"],
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        eos_token_id=[],
        stop_on_eos=False,
    )
    batch_prompt_ids = [prompt_ids for _ in range(manifest["batch_size"])]

    def generate_once() -> tuple[int, list[str], list[list[int]]]:
        infinicore.sync_device()
        start_input = infinicore.from_list(batch_prompt_ids)
        output_tensors = engine.generate(start_input, gen_config, _measure_and_log_time=False)
        infinicore.sync_device()
        token_steps = [np.array(t.to_numpy()).reshape(-1).astype("int64").tolist() for t in output_tensors]
        ids_by_request = [
            [int(step[request_idx]) for step in token_steps]
            for request_idx in range(manifest["batch_size"])
        ]
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in ids_by_request]
        total_tokens = sum(len(ids) for ids in ids_by_request)
        return total_tokens, texts, ids_by_request

    for _ in range(manifest["warmup"]):
        generate_once()
    engine.reset_cache(cache_config)

    iterations = []
    for index in range(1, manifest["repeats"] + 1):
        start = time.perf_counter()
        total_tokens, texts, ids_by_request = generate_once()
        elapsed = time.perf_counter() - start
        iterations.append(
            _iteration_result(
                index,
                elapsed,
                total_tokens,
                texts,
                ids_by_request,
                manifest,
            )
        )

    result = _base_result(case, manifest, prompt_payload)
    result.update(
        {
            "infini_enable_graph_compiling": True,
            "infini_attn": manifest["infini_attn"],
            "block_size": manifest["block_size"],
        }
    )
    return _finalize(result, iterations)


def _iteration_result(
    index: int,
    elapsed: float,
    total_tokens: int,
    texts: list[str],
    ids_by_request: list[list[int]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    expected_total = manifest["batch_size"] * manifest["output_len"]
    errors: list[str] = []
    if total_tokens != expected_total:
        errors.append(f"total_output_tokens={total_tokens}, expected={expected_total}")
    per_request_counts = [len(ids) for ids in ids_by_request]
    bad_counts = [count for count in per_request_counts if count != manifest["output_len"]]
    if bad_counts:
        errors.append(f"bad per-request output lengths: {per_request_counts}")
    health = text_health(texts[0] if texts else "")
    if health["looks_garbled"]:
        errors.append(f"text_health failed: {health}")
    return {
        "iteration": index,
        "elapsed_seconds": elapsed,
        "total_output_tokens": total_tokens,
        "output_tps": total_tokens / elapsed if elapsed > 0 else None,
        "per_request_output_token_counts": per_request_counts,
        "first_output_token_ids_preview": ids_by_request[0][:64] if ids_by_request else [],
        "first_output_preview": (texts[0] if texts else "")[:500],
        "first_text_health": health,
        "validation_errors": errors,
    }


def _base_result(case: dict[str, Any], manifest: dict[str, Any], prompt_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "engine": case["engine"],
        "model": manifest["model"],
        "batch_size": manifest["batch_size"],
        "input_len": manifest["input_len"],
        "output_len": manifest["output_len"],
        "input_token_count_per_request": len(prompt_payload["prompt_token_ids"]),
        "warmup": manifest["warmup"],
        "repeats": manifest["repeats"],
        "graph_mode_requested": True,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "min_tokens": manifest["output_len"],
            "max_tokens": manifest["output_len"],
            "eos_disabled": True,
        },
        "prompt_sha256": prompt_payload["prompt_sha256"],
    }


def _finalize(result: dict[str, Any], iterations: list[dict[str, Any]]) -> dict[str, Any]:
    total_elapsed = sum(item["elapsed_seconds"] for item in iterations)
    total_tokens = sum(item["total_output_tokens"] for item in iterations)
    validation_errors = [
        error
        for item in iterations
        for error in item.get("validation_errors", [])
    ]
    graph_errors = []
    if result["engine"].startswith("vllm") and result.get("graph_capture_count", 0) <= 0:
        graph_errors.append("graph_capture_count=0")
    validation_errors.extend(graph_errors)
    result.update(
        {
            "iterations": iterations,
            "measured_iterations_completed": len(iterations),
            "total_elapsed_seconds": total_elapsed,
            "total_output_tokens": total_tokens,
            "output_tps": total_tokens / total_elapsed if total_elapsed > 0 else None,
            "iteration_tps_stats": stats([item["output_tps"] for item in iterations if item["output_tps"] is not None]),
            "validation_errors": validation_errors,
            "valid": not validation_errors and len(iterations) == result["repeats"],
        }
    )
    return result


def _worker_env(manifest: dict[str, Any], engine: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + ":" + str(Path(manifest["infinilm_root"]) / "python") + ":" + env.get("PYTHONPATH", "")
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_PLUGINS"] = "metax,vllm_infinicore" if engine == "vllm-infinicore" else "metax"
    return env


def _registration_dict(registration: Any) -> dict[str, Any] | None:
    if registration is None:
        return None
    return {
        "installed_routes": list(registration.installed_routes),
        "native_fallback_routes": list(registration.native_fallback_routes),
        "skipped_routes": list(registration.skipped_routes),
        "failure_reason": registration.failure_reason,
    }


def _vllm_graph_count() -> int:
    try:
        from vllm.compilation.counter import compilation_counter

        return int(getattr(compilation_counter, "num_cudagraph_captured", 0))
    except Exception:
        return 0


def _reset_infinicore_counts() -> None:
    try:
        from vllm_infinicore.ops import infinicore_backend, vllm_attention

        infinicore_backend.reset_backend_call_counts()
        vllm_attention.reset_attention_route_counts()
    except Exception:
        return


def _infinicore_backend_counts() -> dict[str, int]:
    try:
        from vllm_infinicore.ops import infinicore_backend

        return infinicore_backend.backend_call_counts()
    except Exception:
        return {}


def _infinicore_attention_counts() -> dict[str, int]:
    try:
        from vllm_infinicore.ops import vllm_attention

        return vllm_attention.attention_route_counts()
    except Exception:
        return {}


def _result_line(result: dict[str, Any]) -> str:
    if not result.get("valid"):
        return f"{result.get('case_id')} failed: {result.get('error') or result.get('validation_errors')}"
    return (
        f"{result['case_id']}: {result['output_tps']:.2f} output tok/s, "
        f"valid={result['valid']}"
    )


def _write_summary_md(run_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Qwen3 Three Engine Throughput",
        "",
        "| Engine | Valid | Output TPS | Median Iter TPS | Graph Captures | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        stats_ = result.get("iteration_tps_stats") or {}
        result_path = Path("results") / f"{result.get('case_id')}.json"
        lines.append(
            f"| {result.get('engine')} | {result.get('valid')} | "
            f"{_fmt(result.get('output_tps'))} | {_fmt(stats_.get('median'))} | "
            f"{result.get('graph_capture_count', '')} | {result_path} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else ""


def _prepend_env(name: str, paths: list[str]) -> None:
    existing = [part for part in os.environ.get(name, "").split(":") if part]
    new_parts = [path for path in paths if path and path not in existing]
    os.environ[name] = ":".join([*new_parts, *existing])


def main() -> int:
    args = parse_args()
    configure_runtime_environment()
    if args.worker:
        return worker_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
