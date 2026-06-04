#!/usr/bin/env python
"""Qwen3 vLLM graph smoke with exact token accounting.

The default parent mode generates prompt token IDs once and reuses them across
subprocess-isolated cases:

- ``native-graph``: vLLM native graph baseline with this plugin present but dry.
- ``plugin-fallback-graph``: all Qwen3 routes requested, forced to native fallback.
- ``no-metax-eager``: all Qwen3 routes requested on the InfiniCore platform
  without loading ``vllm_metax``.
- ``no-metax-graph``: the same no-``vllm_metax`` route set with PIECEWISE
  cudagraph.

The same script can run a single case via ``--single-case`` for debugging or
later route-subset checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm_infinicore.validation import BenchmarkResult, GraphEvidence


DEFAULT_MODEL = "/mnt/geogpt-doc-new/default/xb/qwen3-8B"
DEFAULT_OUTPUT_JSON = "artifacts/qwen3_128_32_smoke.json"
DEFAULT_OUTPUT_DIR = "artifacts/qwen3_128_32_smoke_cases"

CASE_SPECS: dict[str, dict[str, object]] = {
    "native-graph": {
        "description": "vLLM native PIECEWISE graph baseline; plugin dry",
        "patches": "0",
        "routes": "",
        "force_native_fallback": "0",
        "require_cudagraph": True,
        "plugins": None,
        "forbid_metax_load": False,
    },
    "plugin-fallback-graph": {
        "description": "all Qwen3 plugin routes requested with native fallback",
        "patches": "1",
        "routes": "all",
        "force_native_fallback": "1",
        "require_cudagraph": True,
        "plugins": None,
        "forbid_metax_load": False,
    },
    "plugin-rmsnorm-graph": {
        "description": "RMSNorm prototype route requested for route-subset probing",
        "patches": "1",
        "routes": "RMSNorm",
        "force_native_fallback": "0",
        "require_cudagraph": True,
        "plugins": None,
        "forbid_metax_load": False,
    },
    "no-metax-eager": {
        "description": "all Qwen3 routes on InfiniCore platform without vllm_metax",
        "patches": "1",
        "routes": "all",
        "force_native_fallback": "0",
        "require_cudagraph": False,
        "enforce_eager": True,
        "plugins": "infinicore,vllm_infinicore",
        "forbid_metax_load": True,
    },
    "no-metax-graph": {
        "description": "all Qwen3 routes on InfiniCore platform without vllm_metax, PIECEWISE graph",
        "patches": "1",
        "routes": "all",
        "force_native_fallback": "0",
        "require_cudagraph": True,
        "enforce_eager": False,
        "plugins": "infinicore,vllm_infinicore",
        "forbid_metax_load": True,
    },
    "custom-graph": {
        "description": "custom route subset from --custom-routes",
        "patches": "1",
        "routes": None,
        "force_native_fallback": None,
        "require_cudagraph": True,
        "enforce_eager": False,
        "plugins": None,
        "forbid_metax_load": False,
    },
    "custom-eager": {
        "description": "custom route subset from --custom-routes with vLLM eager",
        "patches": "1",
        "routes": None,
        "force_native_fallback": None,
        "require_cudagraph": False,
        "enforce_eager": True,
        "plugins": None,
        "forbid_metax_load": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cases",
        default="native-graph,plugin-fallback-graph",
        help="Comma-separated parent-mode cases.",
    )
    parser.add_argument("--single-case", choices=sorted(CASE_SPECS), default=None)
    parser.add_argument("--custom-routes", default="")
    parser.add_argument("--disabled-routes", default="")
    parser.add_argument("--cpp-bridge-routes", default="")
    parser.add_argument("--force-native-fallback", action="store_true")
    parser.add_argument(
        "--plugins",
        default="",
        help=(
            "Override VLLM_PLUGINS for parent and child cases. "
            "When omitted, the existing environment is respected, falling back "
            "to metax,vllm_infinicore for compatibility."
        ),
    )
    parser.add_argument(
        "--forbid-metax-load",
        action="store_true",
        help="Fail the case if any vllm_metax module is loaded.",
    )
    parser.add_argument("--prompt-json", default="")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def configure_runtime_environment(args: argparse.Namespace) -> None:
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
    if args.plugins:
        os.environ["VLLM_PLUGINS"] = args.plugins
    else:
        os.environ.setdefault("VLLM_PLUGINS", "metax,vllm_infinicore")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    _prepend_env(
        "PATH",
        [
            "/mnt/geogpt-doc-new/default/xmake_env/bin",
            os.path.expanduser("~/.local/bin"),
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


def configure_case_environment(case_name: str, args: argparse.Namespace) -> None:
    spec = CASE_SPECS[case_name]
    plugins = spec.get("plugins") or args.plugins
    if plugins:
        os.environ["VLLM_PLUGINS"] = str(plugins)

    os.environ["VLLM_INFINICORE_ENABLE_PATCHES"] = str(spec["patches"])

    routes = spec["routes"]
    if case_name.startswith("custom-"):
        routes = args.custom_routes
    if routes:
        os.environ["VLLM_INFINICORE_ROUTES"] = str(routes)
    else:
        os.environ.pop("VLLM_INFINICORE_ROUTES", None)

    force_native = spec["force_native_fallback"]
    if case_name.startswith("custom-"):
        force_native = "1" if args.force_native_fallback else "0"
    os.environ["VLLM_INFINICORE_FORCE_NATIVE_FALLBACK"] = str(force_native)

    if args.disabled_routes:
        os.environ["VLLM_INFINICORE_DISABLED_ROUTES"] = args.disabled_routes
    else:
        os.environ.pop("VLLM_INFINICORE_DISABLED_ROUTES", None)
    if args.cpp_bridge_routes:
        os.environ["VLLM_INFINICORE_ENABLE_CPP_BRIDGE"] = "1"
        os.environ["VLLM_INFINICORE_CPP_BRIDGE_ROUTES"] = args.cpp_bridge_routes
    else:
        os.environ.pop("VLLM_INFINICORE_ENABLE_CPP_BRIDGE", None)
        os.environ.pop("VLLM_INFINICORE_CPP_BRIDGE_ROUTES", None)


def build_prompt_ids(tokenizer: Any, input_len: int) -> list[int]:
    if input_len <= 0:
        raise ValueError("--input-len must be positive")

    prompt_ids: list[int] = []
    seed = (
        "Qwen3 smoke validation prompt. "
        "Report concise facts about deterministic token accounting, "
        "operator correctness, and decoded text health. "
    )
    index = 0
    while len(prompt_ids) < input_len:
        text = f"{seed} Segment {index}: keep the response plain and stable.\n"
        prompt_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        index += 1
    return prompt_ids[:input_len]


def load_or_create_prompt_ids(args: argparse.Namespace) -> tuple[list[int], str | None]:
    if args.prompt_json:
        data = json.loads(Path(args.prompt_json).read_text(encoding="utf-8"))
        return [int(token) for token in data["prompt_token_ids"]], args.prompt_json

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    return build_prompt_ids(tokenizer, args.input_len), None


def make_sampling_params(output_len: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        min_tokens=output_len,
        max_tokens=output_len,
        ignore_eos=True,
        skip_special_tokens=False,
    )


def make_compilation_config() -> dict[str, Any]:
    from vllm.config import CUDAGraphMode

    return {
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "cudagraph_num_of_warmups": 1,
        "backend": "eager",
    }


def make_llm(
    args: argparse.Namespace,
    compilation_config: dict[str, Any],
    *,
    enforce_eager: bool,
) -> Any:
    from vllm import LLM

    return LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


def run_generation(
    llm: Any,
    prompt_ids: list[int],
    sampling_params: Any,
) -> tuple[Any, float]:
    from vllm.inputs import TokensPrompt

    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    start = time.perf_counter()
    outputs = llm.generate([prompt], [sampling_params], use_tqdm=False)
    elapsed = time.perf_counter() - start
    return outputs[0], elapsed


def run_parent(args: argparse.Namespace) -> int:
    configure_runtime_environment(args)
    cases = _parse_cases(args.cases)
    output_json = Path(args.output_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    prompt_ids, _ = load_or_create_prompt_ids(args)
    prompt_path = output_dir / f"prompt-in{args.input_len}.json"
    prompt_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "input_len": args.input_len,
                "prompt_token_ids": prompt_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    case_results: list[dict[str, Any]] = []
    for case_name in cases:
        case_json = output_dir / f"{case_name}.json"
        case_log = output_dir / f"{case_name}.log"
        command = _child_command(args, case_name, prompt_path, case_json)
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = _join_pythonpath(ROOT, child_env.get("PYTHONPATH", ""))
        with case_log.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        result: dict[str, Any]
        if case_json.exists():
            result = json.loads(case_json.read_text(encoding="utf-8"))
            result["returncode"] = completed.returncode
        else:
            result = {
                "case_id": case_name,
                "returncode": completed.returncode,
                "valid": False,
                "validation_errors": [f"case subprocess failed with {completed.returncode}"],
            }
        result["artifact"] = str(case_json)
        result["log"] = str(case_log)
        case_results.append(result)
        _print_case_summary(result)

    summary = {
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "warmup": 0 if args.skip_warmup else args.warmup,
        "repeats": args.repeats,
        "prompt_json": str(prompt_path),
        "cases": case_results,
        "validation_errors": [
            f"{case['case_id']}: {error}"
            for case in case_results
            for error in case.get("validation_errors", [])
        ],
    }
    summary["valid"] = not summary["validation_errors"] and all(
        case.get("returncode") == 0 for case in case_results
    )
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={output_json}")
    return 0 if summary["valid"] else 1


def run_single_case(args: argparse.Namespace) -> int:
    assert args.single_case is not None
    configure_runtime_environment(args)
    configure_case_environment(args.single_case, args)

    import vllm_infinicore

    registration = vllm_infinicore.register()
    prompt_ids, prompt_json = load_or_create_prompt_ids(args)
    sampling_params = make_sampling_params(args.output_len)
    enforce_eager = bool(CASE_SPECS[args.single_case].get("enforce_eager", False))
    compilation_config = make_compilation_config()
    llm = make_llm(args, compilation_config, enforce_eager=enforce_eager)

    warmup_iterations = 0 if args.skip_warmup else args.warmup
    for _ in range(warmup_iterations):
        run_generation(llm, prompt_ids, sampling_params)

    _reset_infinicore_backend_counts()
    raw_measurements: list[tuple[Any, float]] = []
    for _ in range(args.repeats):
        raw_measurements.append(run_generation(llm, prompt_ids, sampling_params))

    graph_capture_count = _read_vllm_graph_capture_count()
    graph_evidence = GraphEvidence.from_compilation_config(
        _artifact_compilation_config(enforce_eager=enforce_eager),
        enforce_eager=enforce_eager,
        evidence_strings=(f"num_cudagraph_captured={graph_capture_count}",),
    )
    extra_errors = []
    if CASE_SPECS[args.single_case]["require_cudagraph"] and graph_capture_count <= 0:
        extra_errors.append("num_cudagraph_captured=0")
    forbid_metax_load = bool(
        CASE_SPECS[args.single_case].get("forbid_metax_load")
        or args.forbid_metax_load
    )
    vllm_metax_loaded = _vllm_metax_loaded()
    if forbid_metax_load and vllm_metax_loaded:
        extra_errors.append("vllm_metax_loaded=True")

    measurements = [
        _measurement_result(
            name=f"{args.single_case}-repeat{index + 1}",
            request_output=request_output,
            elapsed_seconds=elapsed_seconds,
            prompt_ids=prompt_ids,
            expected_input_tokens=args.input_len,
            expected_output_tokens=args.output_len,
            graph_evidence=graph_evidence,
            require_cudagraph=bool(CASE_SPECS[args.single_case]["require_cudagraph"]),
            extra_errors=extra_errors,
        )
        for index, (request_output, elapsed_seconds) in enumerate(raw_measurements)
    ]

    total_elapsed = sum(item["elapsed_seconds"] for item in measurements)
    total_output_tokens = sum(item["actual_output_tokens"] for item in measurements)
    validation_errors = _unique_errors(
        error
        for item in measurements
        for error in item["validation_errors"]
    )
    artifact = {
        "case_id": args.single_case,
        "case_description": CASE_SPECS[args.single_case]["description"],
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "prompt_json": prompt_json,
        "warmup_iterations": warmup_iterations,
        "measured_iterations_requested": args.repeats,
        "measured_iterations_completed": len(measurements),
        "total_elapsed_seconds": total_elapsed,
        "total_output_tokens": total_output_tokens,
        "output_tps": total_output_tokens / total_elapsed if total_elapsed > 0 else None,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "min_tokens": args.output_len,
            "max_tokens": args.output_len,
            "ignore_eos": True,
        },
        "compilation_config": _artifact_compilation_config(enforce_eager=enforce_eager),
        "graph_capture_count": graph_capture_count,
        "graph_evidence": graph_evidence.as_dict(),
        "registration": _registration_as_dict(registration),
        "environment": {
            "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS", ""),
            "VLLM_INFINICORE_ENABLE_PATCHES": os.environ.get(
                "VLLM_INFINICORE_ENABLE_PATCHES",
                "",
            ),
            "VLLM_INFINICORE_ROUTES": os.environ.get("VLLM_INFINICORE_ROUTES", ""),
            "VLLM_INFINICORE_DISABLED_ROUTES": os.environ.get(
                "VLLM_INFINICORE_DISABLED_ROUTES",
                "",
            ),
            "VLLM_INFINICORE_FORCE_NATIVE_FALLBACK": os.environ.get(
                "VLLM_INFINICORE_FORCE_NATIVE_FALLBACK",
                "",
            ),
        },
        "vllm_metax_loaded": vllm_metax_loaded,
        "forbid_metax_load": forbid_metax_load,
        "vllm_platform": _vllm_platform_name(),
        "infinicore_backend_call_counts": _infinicore_backend_call_counts(),
        "infinicore_attention_route_counts": _infinicore_attention_route_counts(),
        "infinicore_attention_backend_route_counts": _infinicore_attention_backend_route_counts(),
        "infinicore_cpp_bridge_call_counts": _infinicore_cpp_bridge_call_counts(),
        "cpp_bridge_enabled": bool(_infinicore_cpp_bridge_selected_routes()),
        "cpp_bridge_routes": ",".join(_infinicore_cpp_bridge_selected_routes()),
        "vllm_attention_backend": _vllm_attention_backend_path(),
        "measurements": measurements,
        "first_decoded_preview": (
            measurements[0]["decoded_preview"] if measurements else ""
        ),
        "validation_errors": validation_errors,
    }
    artifact["valid"] = not validation_errors

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    _print_case_summary({**artifact, "artifact": str(output_path)})
    return 0 if artifact["valid"] else 1


def _measurement_result(
    *,
    name: str,
    request_output: Any,
    elapsed_seconds: float,
    prompt_ids: list[int],
    expected_input_tokens: int,
    expected_output_tokens: int,
    graph_evidence: GraphEvidence,
    require_cudagraph: bool,
    extra_errors: list[str],
) -> dict[str, Any]:
    completion = request_output.outputs[0]
    output_token_ids = [int(token) for token in completion.token_ids]
    actual_input_tokens = len(request_output.prompt_token_ids or prompt_ids)
    actual_output_tokens = len(output_token_ids)
    result = BenchmarkResult(
        name=name,
        expected_input_tokens=expected_input_tokens,
        actual_input_tokens=actual_input_tokens,
        expected_output_tokens=expected_output_tokens,
        actual_output_tokens=actual_output_tokens,
        elapsed_seconds=elapsed_seconds,
        output_token_ids=output_token_ids,
        decoded_text=completion.text,
        graph_evidence=graph_evidence,
        require_cudagraph=require_cudagraph,
        extra_validation_errors=extra_errors,
    )
    result_dict = result.as_dict()
    result_dict["input_token_count"] = actual_input_tokens
    result_dict["output_token_count"] = actual_output_tokens
    return result_dict


def _registration_as_dict(registration: Any) -> dict[str, Any]:
    return {
        "route_count": registration.route_count,
        "patching_enabled": registration.patching_enabled,
        "reason": registration.reason,
        "requested_routes": list(registration.requested_routes),
        "installed_routes": list(registration.installed_routes),
        "native_fallback_routes": list(registration.native_fallback_routes),
        "skipped_routes": list(registration.skipped_routes),
        "failure_reason": registration.failure_reason,
        "route_states": [
            {
                "name": state.name,
                "requested": state.requested,
                "disabled_by_env": state.disabled_by_env,
                "status": state.status,
                "implementation": state.implementation,
                "native_fallback": state.native_fallback,
                "validation": state.validation,
                "graph_policy": state.graph_policy,
                "reason": state.reason,
            }
            for state in registration.route_states
        ],
    }


def _artifact_compilation_config(*, enforce_eager: bool = False) -> dict[str, Any]:
    if enforce_eager:
        return {
            "cudagraph_mode": "NONE",
            "cudagraph_capture_sizes": [],
            "cudagraph_num_of_warmups": 0,
            "backend": "eager",
            "enforce_eager": True,
        }
    return {
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "cudagraph_num_of_warmups": 1,
        "backend": "eager",
        "enforce_eager": enforce_eager,
    }


def _read_vllm_graph_capture_count() -> int:
    try:
        from vllm.compilation.counter import compilation_counter
    except Exception:
        return 0
    return int(getattr(compilation_counter, "num_cudagraph_captured", 0))


def _reset_infinicore_backend_counts() -> None:
    try:
        from vllm_infinicore.ops import infinicore_backend
        from vllm_infinicore.ops import cpp_bridge
        from vllm_infinicore.ops import vllm_attention
        from vllm_infinicore.ops import vllm_attention_backend
    except Exception:
        return
    infinicore_backend.reset_backend_call_counts()
    cpp_bridge.reset_bridge_call_counts()
    vllm_attention.reset_attention_route_counts()
    vllm_attention_backend.reset_attention_backend_route_counts()


def _infinicore_backend_call_counts() -> dict[str, int]:
    try:
        from vllm_infinicore.ops import infinicore_backend
    except Exception:
        return {}
    return infinicore_backend.backend_call_counts()


def _infinicore_attention_route_counts() -> dict[str, int]:
    try:
        from vllm_infinicore.ops import vllm_attention
    except Exception:
        return {}
    return vllm_attention.attention_route_counts()


def _infinicore_attention_backend_route_counts() -> dict[str, int]:
    try:
        from vllm_infinicore.ops import vllm_attention_backend
    except Exception:
        return {}
    return vllm_attention_backend.attention_backend_route_counts()


def _infinicore_cpp_bridge_call_counts() -> dict[str, int]:
    try:
        from vllm_infinicore.ops import cpp_bridge
    except Exception:
        return {}
    return cpp_bridge.bridge_call_counts()


def _infinicore_cpp_bridge_selected_routes() -> tuple[str, ...]:
    try:
        from vllm_infinicore.ops import cpp_bridge
    except Exception:
        return ()
    return cpp_bridge.selected_routes()


def _vllm_attention_backend_path() -> str:
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception:
        return ""
    try:
        return AttentionBackendEnum.FLASH_ATTN.get_path()
    except Exception:
        return ""


def _vllm_metax_loaded() -> bool:
    return any(
        name == "vllm_metax" or name.startswith("vllm_metax.")
        for name in sys.modules
    )


def _vllm_platform_name() -> str:
    try:
        from vllm.platforms import current_platform
    except Exception:
        return ""
    platform_cls = (
        current_platform
        if isinstance(current_platform, type)
        else type(current_platform)
    )
    return f"{platform_cls.__module__}.{platform_cls.__qualname__}"


def _child_command(
    args: argparse.Namespace,
    case_name: str,
    prompt_path: Path,
    case_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-case",
        case_name,
        "--model",
        args.model,
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--prompt-json",
        str(prompt_path),
        "--output-json",
        str(case_json),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.skip_warmup:
        command.append("--skip-warmup")
    if args.custom_routes:
        command.extend(["--custom-routes", args.custom_routes])
    if args.disabled_routes:
        command.extend(["--disabled-routes", args.disabled_routes])
    if args.cpp_bridge_routes:
        command.extend(["--cpp-bridge-routes", args.cpp_bridge_routes])
    if args.force_native_fallback:
        command.append("--force-native-fallback")
    if args.plugins:
        command.extend(["--plugins", args.plugins])
    if args.forbid_metax_load:
        command.append("--forbid-metax-load")
    return command


def _parse_cases(value: str) -> list[str]:
    cases = [case.strip() for case in value.split(",") if case.strip()]
    if not cases:
        raise ValueError("--cases must contain at least one case")
    unknown = [case for case in cases if case not in CASE_SPECS]
    if unknown:
        raise ValueError("unknown case(s): " + ", ".join(unknown))
    return cases


def _unique_errors(errors: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for error in errors:
        if error in seen:
            continue
        seen.add(error)
        unique.append(error)
    return unique


def _print_case_summary(result: dict[str, Any]) -> None:
    print(
        "case={case} valid={valid} input={input_tokens} output={output_tokens} "
        "tps={tps} errors={errors} artifact={artifact} log={log}".format(
            case=result.get("case_id"),
            valid=result.get("valid"),
            input_tokens=result.get("input_len")
            or _first_measurement_value(result, "input_token_count"),
            output_tokens=result.get("output_len")
            or _first_measurement_value(result, "output_token_count"),
            tps=result.get("output_tps"),
            errors=json.dumps(result.get("validation_errors", []), ensure_ascii=False),
            artifact=result.get("artifact", ""),
            log=result.get("log", ""),
        )
    )


def _first_measurement_value(result: dict[str, Any], key: str) -> object:
    measurements = result.get("measurements") or []
    if not measurements:
        return None
    return measurements[0].get(key)


def _prepend_env(name: str, paths: list[str]) -> None:
    existing = [part for part in os.environ.get(name, "").split(":") if part]
    new_parts = [path for path in paths if path and path not in existing]
    os.environ[name] = ":".join([*new_parts, *existing])


def _join_pythonpath(root: Path, existing: str) -> str:
    parts = [str(root)]
    parts.extend(part for part in existing.split(":") if part and part != str(root))
    return ":".join(parts)


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.single_case is not None:
        return run_single_case(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
