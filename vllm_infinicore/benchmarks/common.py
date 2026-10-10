"""Shared benchmark environments, worker evidence, and result artifacts.

Inference dependencies are imported only when inspecting a running worker.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from enum import Enum
from pathlib import Path

from ..routing.patching import QWEN3_OPERATOR_ROUTES

ROUTES = tuple(route.name for route in QWEN3_OPERATOR_ROUTES)
VENDOR_PLUGINS = {
    "metax": "metax,metax_enhanced_customized,metax_enhanced_model",
    "ascend": "ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling",
    "kunlun": "kunlun,kunlun_model,kunlun_tool_parser,kunlun_reasoning_parser",
}
ASCEND_FALLBACK_REASONS = {
    "fused_add_rms_norm": "InfiniCore has no Ascend fused Add+RMSNorm kernel",
    "silu_and_mul": "SwiGLU requires eight aligned tiles and hidden size <= 8192",
}
BACKEND_COUNTERS = {
    "Embedding": "embedding",
    "RMSNorm": "rms_norm",
    "MatMul": "linear",
    "RoPE": "rotary_embedding",
    "SiluAndMul": "silu_and_mul",
    "LMHead": "lm_head",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, Enum):
        return value.name
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def mode_environment(
    mode: str,
    environment: dict[str, str],
    platform: str = "metax",
    routes: tuple[str, ...] = ROUTES,
) -> dict[str, str]:
    env = dict(environment)
    vendors = [
        p.strip()
        for p in env.get("VLLM_PLUGINS", VENDOR_PLUGINS[platform]).split(",")
        if p.strip() and p.strip() != "vllm_infinicore"
    ]
    enabled = mode == "infinicore"
    env["VLLM_PLUGINS"] = ",".join(vendors + (["vllm_infinicore"] if enabled else []))
    env["VLLM_INFINICORE_ENABLE_PATCHES"] = "1" if enabled else "0"
    env["VLLM_INFINICORE_OPERATOR_BACKEND"] = platform
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if enabled:
        env.update(
            VLLM_INFINICORE_STRICT_BACKEND="1",
            VLLM_INFINICORE_ENABLE_CPP_BRIDGE="1",
            VLLM_INFINICORE_DISABLE_CPP_BRIDGE="0",
            VLLM_INFINICORE_DISABLED_ROUTES="",
            VLLM_INFINICORE_FORCE_NATIVE_FALLBACK="0",
            VLLM_INFINICORE_DISABLE_REAL_BACKEND="0",
            VLLM_INFINICORE_ROUTES=",".join(routes),
            VLLM_INFINICORE_CPP_BRIDGE_ROUTES=",".join(routes),
        )
        if platform == "ascend":
            env["VLLM_INFINICORE_ASCEND_GRAPH"] = "1"
    return env


def worker_state(worker):
    import torch

    platform = os.environ["VLLM_INFINICORE_OPERATOR_BACKEND"]
    api_name, class_name = ("npu", "NPUGraph") if platform == "ascend" else ("cuda", "CUDAGraph")
    cls = getattr(getattr(torch, api_name), class_name)
    if not hasattr(cls, "_gsm8k_replays"):
        original = cls.replay

        def replay(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            cls._gsm8k_replays += 1
            return result

        cls._gsm8k_replays = 0
        cls.replay = replay
    state = {
        "rank": worker.rank,
        "platform": platform,
        "graph_api": api_name,
        "graph_replays": cls._gsm8k_replays,
    }
    from vllm.compilation.counter import compilation_counter

    state["graph_captures"] = compilation_counter.num_cudagraph_captured
    if os.getenv("VLLM_INFINICORE_ENABLE_PATCHES") == "1":
        from .. import plugin
        from ..operators import attention_ops
        from ..operators.common import backend, cpp_bridge
        from ..routing.routes import attention

        state.update(
            registration=dataclasses.asdict(plugin._REGISTRATION_RESULT),
            backend_calls=backend.backend_call_counts(),
            bridge_calls=cpp_bridge.bridge_call_counts(),
            fallback_calls=backend.backend_fallback_counts(),
            fallback_reasons=backend.backend_fallback_reasons(),
            attention_calls=attention_ops.call_counts(),
            attention_fallbacks=attention.fallback_counts(),
            native_attention_calls=attention.native_call_counts(),
        )
        if platform == "ascend":
            width = worker.vllm_config.model_config.hf_config.intermediate_size
            state["model_intermediate_size"] = width
            state["known_native_paths"] = {
                "fused_add_rms_norm": ASCEND_FALLBACK_REASONS["fused_add_rms_norm"]
            }
            if width > 8192 or width % 128:
                state["known_native_paths"]["silu_and_mul"] = ASCEND_FALLBACK_REASONS[
                    "silu_and_mul"
                ]
    return state


def verify_workers(
    mode: str,
    workers: list[dict],
    eager: bool,
    finished: bool,
    routes: tuple[str, ...] = ROUTES,
) -> None:
    if not workers:
        raise RuntimeError("vLLM returned no worker state")
    for state in workers:
        if not eager and (not state["graph_captures"] or (finished and not state["graph_replays"])):
            raise RuntimeError("Requested Graph mode did not capture/replay")
        if mode == "native":
            if state.get("registration", {}).get("installed_routes"):
                raise RuntimeError("Native evaluation unexpectedly installed plugin routes")
            continue
        installed = state["registration"]["installed_routes"]
        if set(installed) != set(routes):
            raise RuntimeError(f"Not all requested plugin routes installed: {installed}")
        for route in ("PagedAttentionPrefill", "PagedAttentionDecode"):
            if route not in routes and state["attention_calls"].get(route, 0):
                raise RuntimeError(f"Unrequested InfiniCore Attention was called: {route}")
        ascend = state.get("platform") == "ascend"
        allowed_fallbacks = ASCEND_FALLBACK_REASONS if ascend else {}
        for name in ("fallback_calls", "attention_fallbacks", "native_attention_calls"):
            if name == "native_attention_calls":
                planned_native = {
                    "StoreKVCache",
                    "PagedAttentionPrefill",
                    "PagedAttentionDecode",
                } - set(routes)
                unexpected = {
                    route: count
                    for route, count in state[name].items()
                    if count and route not in planned_native
                }
                if unexpected:
                    raise RuntimeError(f"Unexpected plugin fallback: {name}={unexpected}")
                continue
            if name == "fallback_calls" and ascend:
                for counter, count in state[name].items():
                    if count and (
                        counter not in allowed_fallbacks
                        or state.get("fallback_reasons", {}).get(counter)
                        != allowed_fallbacks[counter]
                    ):
                        raise RuntimeError(
                            f"Unexpected plugin fallback: {counter}={state.get('fallback_reasons')}"
                        )
                continue
            if any(state[name].values()):
                raise RuntimeError(f"Unexpected plugin fallback: {name}={state[name]}")
        if finished:
            calls = {**state["bridge_calls"], **state["attention_calls"]}
            if ascend:
                calls.update(
                    {
                        route: state["backend_calls"].get(counter, 0)
                        for route, counter in BACKEND_COUNTERS.items()
                    }
                )
            required = set(routes)
            if ascend and (
                state.get("fallback_calls", {}).get("silu_and_mul", 0)
                or state.get("known_native_paths", {}).get("silu_and_mul")
                == ASCEND_FALLBACK_REASONS["silu_and_mul"]
            ):
                required.discard("SiluAndMul")
            if any(calls.get(route, 0) <= 0 for route in required):
                raise RuntimeError(f"Missing actual plugin route calls: {calls}")
            if (
                "RMSNorm" in routes
                and not ascend
                and state.get("platform") != "kunlun"
                and not state["backend_calls"].get("fused_add_rms_norm", 0)
            ):
                raise RuntimeError("Fused Add+RMSNorm was not called")
