"""On-demand C++ bridge for MetaX and Kunlun operator routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

CPP_BRIDGE_ENABLE_ENV = "VLLM_INFINICORE_ENABLE_CPP_BRIDGE"
CPP_BRIDGE_ROUTES_ENV = "VLLM_INFINICORE_CPP_BRIDGE_ROUTES"
CPP_BRIDGE_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_CPP_BRIDGE"
CPP_BRIDGE_TARGET_ENV = "VLLM_INFINICORE_CPP_BRIDGE_TARGET"

METAX_TARGET = "metax"
KUNLUN_TARGET = "kunlun"
BRIDGE_TARGET_ALIASES = {
    METAX_TARGET: frozenset({"cuda", "maca", "metax", "muxi"}),
    KUNLUN_TARGET: frozenset({"kunlun", "xpu"}),
}

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
DEFAULT_ROUTES = (MATMUL_ROUTE,)
KUNLUN_VALIDATED_ROUTES = (
    EMBEDDING_ROUTE,
    LM_HEAD_ROUTE,
    MATMUL_ROUTE,
    ROPE_ROUTE,
)
KUNLUN_SUPPORTED_ROUTES = KUNLUN_VALIDATED_ROUTES
RAY_DEFAULT_ROUTES = (MATMUL_ROUTE,)

_MODULE: Any | None = None
_LOAD_ERROR: str | None = None
_CALL_COUNTS: dict[str, int] = {}
_BRIDGE_TARGET_CACHE: dict[str, str] = {}
_ROUTES_CACHE_KEY: tuple[str | None, str | None, str | None, str | None] | None = None
_ROUTES_CACHE: tuple[str, ...] | None = None
_ROUTES_SET_CACHE: frozenset[str] | None = None


class CppBridgeError(RuntimeError):
    pass


def enabled_for(route_name: str) -> bool:
    return route_name in _selected_route_set()


def selected_routes() -> tuple[str, ...]:
    routes, _ = _cached_routes()
    return routes


def bridge_target() -> str:
    return _bridge_target()


def _selected_route_set() -> frozenset[str]:
    _, route_set = _cached_routes()
    return route_set


def _cached_routes() -> tuple[tuple[str, ...], frozenset[str]]:
    global _ROUTES_CACHE_KEY, _ROUTES_CACHE, _ROUTES_SET_CACHE

    cache_key = (
        os.environ.get(CPP_BRIDGE_DISABLE_ENV),
        os.environ.get(CPP_BRIDGE_ENABLE_ENV),
        os.environ.get(CPP_BRIDGE_ROUTES_ENV),
        os.environ.get(CPP_BRIDGE_TARGET_ENV),
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

    raw = os.environ.get(CPP_BRIDGE_ROUTES_ENV)
    if raw is None or not raw.strip():
        if _bridge_target() == KUNLUN_TARGET:
            return ()
        if _env_truthy("VLLM_INFINICORE_RAY_BACKEND"):
            return RAY_DEFAULT_ROUTES
        return DEFAULT_ROUTES
    routes = tuple(route.strip() for route in raw.split(",") if route.strip())
    target = _bridge_target()
    if routes == ("all",):
        if target == KUNLUN_TARGET:
            return tuple(sorted(KUNLUN_VALIDATED_ROUTES))
        return tuple(sorted(SUPPORTED_ROUTES))
    unknown = tuple(
        route for route in routes
        if route not in SUPPORTED_ROUTES and route not in NATIVE_ATTENTION_ROUTES
    )
    if unknown:
        raise CppBridgeError(f"unsupported C++ bridge route(s): {', '.join(unknown)}")
    routes = tuple(route for route in routes if route in SUPPORTED_ROUTES)
    if target == KUNLUN_TARGET:
        unsupported = tuple(
            route for route in routes if route not in KUNLUN_SUPPORTED_ROUTES
        )
        if unsupported:
            raise CppBridgeError(
                f"unsupported Kunlun C++ bridge route(s): {', '.join(unsupported)}"
            )
    return routes


def module() -> Any:
    global _MODULE, _LOAD_ERROR

    routes = selected_routes()
    if not routes:
        raise CppBridgeError(
            f"C++ bridge is disabled; unset {CPP_BRIDGE_DISABLE_ENV} and avoid "
            f"setting {CPP_BRIDGE_ENABLE_ENV}=0"
        )
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ERROR is not None:
        raise CppBridgeError(_LOAD_ERROR)

    try:
        _MODULE = _compile_bridge()
    except Exception as exc:
        _LOAD_ERROR = f"C++ bridge load failed: {exc}"
        raise CppBridgeError(_LOAD_ERROR) from exc
    assert _MODULE is not None
    return _MODULE


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
    target = _bridge_target()

    include_paths = [str(infini_root / "include")]
    cflags = [
        "-std=c++17",
        "-DINFINICORE_HPCC_VERSION_MAJOR=3",
    ]
    ldflags = [
        f"-L{infini_lib_dir}",
        f"-Wl,-rpath,{infini_lib_dir}",
        "-linfinicore_cpp_api",
        "-linfiniop",
        "-linfinirt",
        "-linfiniccl",
    ]

    if target == KUNLUN_TARGET:
        cflags.extend(["-DENABLE_CUDA_API", "-DENABLE_KUNLUN_API"])
        include_paths.append(
            str(Path(os.environ.get("XPU_HOME", "/usr/local/xpu")) / "include")
        )
    else:
        cflags.append("-DENABLE_METAX_API")

    return {
        "name": (
            "vllm_infinicore_kunlun_cpp_bridge"
            if target == KUNLUN_TARGET
            else "vllm_infinicore_cpp_bridge"
        ),
        "sources": [str(source)],
        "extra_include_paths": _dedupe(include_paths),
        "extra_cflags": _dedupe(cflags),
        "extra_ldflags": _dedupe(ldflags),
        "target": target,
    }


def _bridge_target() -> str:
    # Resolve once per override. With no override, follow the selected vLLM
    # vendor platform instead of guessing from installed Python packages.
    raw = os.environ.get(CPP_BRIDGE_TARGET_ENV)
    key = raw.strip() if raw is not None else ""
    cached = _BRIDGE_TARGET_CACHE.get(key)
    if cached is not None:
        return cached
    if key:
        normalized = _normalize_bridge_target(key)
        if normalized is None:
            valid = sorted(
                alias for aliases in BRIDGE_TARGET_ALIASES.values() for alias in aliases
            )
            raise CppBridgeError(
                f"{CPP_BRIDGE_TARGET_ENV} must be one of {', '.join(valid)}"
            )
        target = normalized
    else:
        from ..device.detection import selected_platform

        target = selected_platform().name
        if target not in BRIDGE_TARGET_ALIASES:
            raise CppBridgeError(
                "C++ bridge requires a selected MetaX or Kunlun vLLM platform; "
                f"set {CPP_BRIDGE_TARGET_ENV} to override"
            )
    _BRIDGE_TARGET_CACHE[key] = target
    return target


def _normalize_bridge_target(value: str) -> str | None:
    normalized = value.strip().lower()
    for target, aliases in BRIDGE_TARGET_ALIASES.items():
        if normalized in aliases:
            return target
    return None


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
