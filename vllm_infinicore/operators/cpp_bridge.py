"""On-demand C++ bridge for explicitly selected operator implementations."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
from typing import Any

from .selection import OPERATOR_BACKEND_ENV, selected_backend

CPP_BRIDGE_ENABLE_ENV = "VLLM_INFINICORE_ENABLE_CPP_BRIDGE"
CPP_BRIDGE_ROUTES_ENV = "VLLM_INFINICORE_CPP_BRIDGE_ROUTES"
CPP_BRIDGE_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_CPP_BRIDGE"

CUDA_TARGET = "cuda"
METAX_TARGET = "metax"
KUNLUN_TARGET = "kunlun"
CPP_BRIDGE_TARGETS = frozenset({CUDA_TARGET, METAX_TARGET, KUNLUN_TARGET})

DECODE_ROUTE = "PagedAttentionDecode"
FLASH_DECODE_ROUTE = "PagedAttentionDecodeFlash"
EMBEDDING_ROUTE = "Embedding"
MATMUL_ROUTE = "MatMul"
RMS_NORM_ROUTE = "RMSNorm"
SILU_AND_MUL_ROUTE = "SiluAndMul"
ROPE_ROUTE = "RoPE"
STORE_KV_CACHE_ROUTE = "StoreKVCache"
LM_HEAD_ROUTE = "LMHead"
PREFILL_ROUTE = "PagedAttentionPrefill"
SUPPORTED_ROUTES = frozenset(
    {
        EMBEDDING_ROUTE,
        MATMUL_ROUTE,
        RMS_NORM_ROUTE,
        SILU_AND_MUL_ROUTE,
        ROPE_ROUTE,
        LM_HEAD_ROUTE,
    }
)
NATIVE_ATTENTION_ROUTES = frozenset(
    {DECODE_ROUTE, FLASH_DECODE_ROUTE, STORE_KV_CACHE_ROUTE, PREFILL_ROUTE}
)

_MODULES: dict[str, Any] = {}
_LOAD_ERRORS: dict[str, str] = {}
_CALL_COUNTS: dict[str, int] = {}
_ROUTES_CACHE_KEY: tuple[str | None, str | None, str | None, str | None] | None = None
_ROUTES_CACHE: tuple[str, ...] | None = None
_ROUTES_SET_CACHE: frozenset[str] | None = None


class CppBridgeError(RuntimeError):
    pass


def bridge_target() -> str:
    target = selected_backend()
    if target not in CPP_BRIDGE_TARGETS:
        raise CppBridgeError(
            f"C++ bridge requires {OPERATOR_BACKEND_ENV}=cuda, metax, or kunlun"
        )
    return target


def _backend_config() -> Any:
    return import_module(f".{bridge_target()}.bridge", __package__)


def enabled_for(route_name: str) -> bool:
    return route_name in _selected_route_set()


def selected_routes() -> tuple[str, ...]:
    routes, _ = _cached_routes()
    return routes


def _selected_route_set() -> frozenset[str]:
    _, route_set = _cached_routes()
    return route_set


def _cached_routes() -> tuple[tuple[str, ...], frozenset[str]]:
    global _ROUTES_CACHE_KEY, _ROUTES_CACHE, _ROUTES_SET_CACHE

    cache_key = (
        os.environ.get(CPP_BRIDGE_DISABLE_ENV),
        os.environ.get(CPP_BRIDGE_ENABLE_ENV),
        os.environ.get(CPP_BRIDGE_ROUTES_ENV),
        os.environ.get(OPERATOR_BACKEND_ENV),
    )
    if (
        cache_key == _ROUTES_CACHE_KEY
        and _ROUTES_CACHE is not None
        and _ROUTES_SET_CACHE is not None
    ):
        return _ROUTES_CACHE, _ROUTES_SET_CACHE

    routes = _parse_selected_routes()
    route_set = frozenset(routes)
    _ROUTES_CACHE_KEY = cache_key
    _ROUTES_CACHE = routes
    _ROUTES_SET_CACHE = route_set
    return routes, route_set


def _parse_selected_routes() -> tuple[str, ...]:
    if _env_truthy(CPP_BRIDGE_DISABLE_ENV) or _env_falsey(CPP_BRIDGE_ENABLE_ENV):
        return ()

    config = _backend_config()
    raw = os.environ.get(CPP_BRIDGE_ROUTES_ENV)
    if raw is None or not raw.strip():
        return config.DEFAULT_ROUTES

    routes = tuple(route.strip() for route in raw.split(",") if route.strip())
    if routes == ("all",):
        return tuple(sorted(config.SUPPORTED_ROUTES))
    unknown = tuple(
        route for route in routes
        if route not in SUPPORTED_ROUTES and route not in NATIVE_ATTENTION_ROUTES
    )
    if unknown:
        raise CppBridgeError(f"unsupported C++ bridge route(s): {', '.join(unknown)}")
    routes = tuple(route for route in routes if route in SUPPORTED_ROUTES)
    unsupported = tuple(route for route in routes if route not in config.SUPPORTED_ROUTES)
    if unsupported:
        raise CppBridgeError(
            f"unsupported {bridge_target()} C++ bridge route(s): "
            + ", ".join(unsupported)
        )
    return routes


def module() -> Any:
    target = bridge_target()
    if not selected_routes():
        raise CppBridgeError(
            f"C++ bridge is disabled; unset {CPP_BRIDGE_DISABLE_ENV} and avoid "
            f"setting {CPP_BRIDGE_ENABLE_ENV}=0"
        )
    if target in _MODULES:
        return _MODULES[target]
    if target in _LOAD_ERRORS:
        raise CppBridgeError(_LOAD_ERRORS[target])

    try:
        loaded = _compile_bridge()
    except Exception as exc:
        message = f"{target} C++ bridge load failed: {exc}"
        _LOAD_ERRORS[target] = message
        raise CppBridgeError(message) from exc
    _MODULES[target] = loaded
    return loaded


def bridge_call_counts() -> dict[str, int]:
    return dict(_CALL_COUNTS)


def reset_bridge_call_counts() -> None:
    _CALL_COUNTS.clear()


def record_call(route_name: str) -> None:
    _CALL_COUNTS[route_name] = _CALL_COUNTS.get(route_name, 0) + 1


def _compile_bridge() -> Any:
    from torch.utils.cpp_extension import load

    config = _bridge_build_config()
    return load(
        name=config["name"],
        sources=config["sources"],
        extra_include_paths=config["extra_include_paths"],
        extra_cflags=config["extra_cflags"],
        extra_ldflags=config["extra_ldflags"],
        verbose=_env_truthy("VLLM_INFINICORE_CPP_BRIDGE_VERBOSE"),
    )


def _bridge_build_config() -> dict[str, Any]:
    source = Path(__file__).resolve().parent / "csrc" / "infinicore_bridge.cpp"
    infini_root = Path(os.environ.get("INFINI_ROOT", str(Path.home() / ".infini")))
    infini_lib_dir = Path(os.environ.get("INFINI_LIB_DIR", str(infini_root / "lib")))
    target = bridge_target()
    adapter = _backend_config()

    include_paths = [str(infini_root / "include"), *adapter.extra_include_paths()]
    cflags = [
        "-std=c++17",
        "-DINFINICORE_HPCC_VERSION_MAJOR=3",
        *adapter.EXTRA_CFLAGS,
    ]
    ldflags = [
        f"-L{infini_lib_dir}",
        f"-Wl,-rpath,{infini_lib_dir}",
        "-linfinicore_cpp_api",
        "-linfiniop",
        "-linfinirt",
        "-linfiniccl",
    ]
    return {
        "name": adapter.MODULE_NAME,
        "sources": [str(source)],
        "extra_include_paths": _dedupe(include_paths),
        "extra_cflags": _dedupe(cflags),
        "extra_ldflags": _dedupe(ldflags),
        "target": target,
    }


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_falsey(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"0", "false", "no", "off"}
