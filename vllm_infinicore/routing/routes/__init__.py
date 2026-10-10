"""Shared vLLM patch mechanics used by explicit operator implementations."""

from __future__ import annotations

from importlib import import_module

_ROUTE_FUNCTIONS = {
    "RMSNorm": ("rms_norm", "vllm_rms_norm_oot"),
    "SiluAndMul": ("silu_and_mul", "vllm_silu_and_mul_oot"),
    "RoPE": ("rotary_embedding", "vllm_rotary_embedding_oot"),
    "Embedding": ("embedding", "vllm_unquantized_embedding_route"),
    "MatMul": ("linear", "vllm_unquantized_linear_route"),
    "LMHead": ("linear", "vllm_unquantized_linear_route"),
}


def _call(name: str, action: str):
    try:
        module_name, suffix = _ROUTE_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported operator route: {name}") from exc
    module = import_module(f".{module_name}", __name__)
    function = getattr(module, f"{action}_{suffix}")
    return function(name) if module_name == "linear" else function()


def install_shared_route(name: str):
    return _call(name, "install")


def uninstall_shared_route(name: str):
    return _call(name, "uninstall")
