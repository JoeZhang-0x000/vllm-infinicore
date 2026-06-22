"""On-demand C++ bridge for hot InfiniCore routes."""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path
from typing import Any

CPP_BRIDGE_ENABLE_ENV = "VLLM_INFINICORE_ENABLE_CPP_BRIDGE"
CPP_BRIDGE_ROUTES_ENV = "VLLM_INFINICORE_CPP_BRIDGE_ROUTES"
CPP_BRIDGE_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_CPP_BRIDGE"
CPP_BRIDGE_TARGET_ENV = "VLLM_INFINICORE_CPP_BRIDGE_TARGET"
FLASH_DECODE_NUM_SPLITS_ENV = "VLLM_INFINICORE_FLASH_DECODE_NUM_SPLITS"

CUDA_TARGET = "cuda"
MUSA_TARGET = "musa"
BRIDGE_TARGET_ALIASES = {
    CUDA_TARGET: frozenset({"cuda", "metax", "muxi"}),
    MUSA_TARGET: frozenset({"musa", "moore"}),
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
        DECODE_ROUTE,
        FLASH_DECODE_ROUTE,
        EMBEDDING_ROUTE,
        MATMUL_ROUTE,
        RMS_NORM_ROUTE,
        SILU_AND_MUL_ROUTE,
        ROPE_ROUTE,
        STORE_KV_CACHE_ROUTE,
        LM_HEAD_ROUTE,
        PREFILL_ROUTE,
    }
)
DEFAULT_ROUTES = (
    FLASH_DECODE_ROUTE,
    MATMUL_ROUTE,
)
MUSA_DEFAULT_ROUTES = (
    DECODE_ROUTE,
    EMBEDDING_ROUTE,
    LM_HEAD_ROUTE,
    MATMUL_ROUTE,
    PREFILL_ROUTE,
    RMS_NORM_ROUTE,
    ROPE_ROUTE,
    SILU_AND_MUL_ROUTE,
    STORE_KV_CACHE_ROUTE,
)
RAY_DEFAULT_ROUTES = (
    FLASH_DECODE_ROUTE,
    MATMUL_ROUTE,
    STORE_KV_CACHE_ROUTE,
)

_MODULE: Any | None = None
_LOAD_ERROR: str | None = None
_CALL_COUNTS: dict[str, int] = {}
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
        if _bridge_target() == MUSA_TARGET:
            return MUSA_DEFAULT_ROUTES
        if _env_truthy("VLLM_INFINICORE_RAY_BACKEND"):
            return RAY_DEFAULT_ROUTES
        return DEFAULT_ROUTES
    routes = tuple(route.strip() for route in raw.split(",") if route.strip())
    target = _bridge_target()
    if routes == ("all",):
        if target == MUSA_TARGET:
            return tuple(sorted(MUSA_DEFAULT_ROUTES))
        return tuple(sorted(SUPPORTED_ROUTES))
    unknown = tuple(route for route in routes if route not in SUPPORTED_ROUTES)
    if unknown:
        raise CppBridgeError(f"unsupported C++ bridge route(s): {', '.join(unknown)}")
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


def flash_decode_num_splits() -> int:
    raw = os.environ.get(FLASH_DECODE_NUM_SPLITS_ENV)
    if raw is None or not raw.strip():
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise CppBridgeError(
            f"{FLASH_DECODE_NUM_SPLITS_ENV} must be an integer, got {raw!r}"
        ) from exc
    if value < 0:
        raise CppBridgeError(f"{FLASH_DECODE_NUM_SPLITS_ENV} must be >= 0")
    return value


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
    root = Path(__file__).resolve().parents[1]
    source = root / "csrc" / "infinicore_bridge.cpp"
    infini_root = Path(os.environ.get("INFINI_ROOT", str(Path.home() / ".infini")))
    maca_path = Path(os.environ.get("MACA_PATH", "/opt/maca-3.5.3"))
    target = _bridge_target()

    include_paths = [str(infini_root / "include")]
    cflags = [
        "-std=c++17",
        "-DINFINICORE_HPCC_VERSION_MAJOR=3",
    ]
    ldflags = [
        f"-L{infini_root / 'lib'}",
        f"-Wl,-rpath,{infini_root / 'lib'}",
        "-linfinicore_cpp_api",
        "-linfiniop",
        "-linfinirt",
        "-linfiniccl",
    ]

    if target == MUSA_TARGET:
        try:
            import torch_musa  # noqa: F401
        except Exception:
            pass
        cflags.extend(["-DENABLE_MUSA_API", "-DENABLE_MOORE_API"])
        include_paths.extend(str(path) for path in _musa_include_paths())
        ldflags.extend(str(flag) for flag in _musa_link_flags())
    else:
        cflags.extend(["-DENABLE_FLASH_ATTN", "-DENABLE_METAX_API"])
        ldflags.extend(
            [
                os.environ.get(
                    "FLASH_ATTN_2_CUDA_SO",
                    "/opt/conda/lib/python3.12/site-packages/"
                    "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so",
                ),
                f"-Wl,-rpath,{maca_path / 'lib'}",
                f"-Wl,-rpath,{maca_path / 'lib64'}",
            ]
        )

    return {
        "name": "vllm_infinicore_cpp_bridge",
        "sources": [str(source)],
        "extra_include_paths": _dedupe(include_paths),
        "extra_cflags": _dedupe(cflags),
        "extra_ldflags": _dedupe(ldflags),
        "target": target,
    }


def _bridge_target() -> str:
    raw = os.environ.get(CPP_BRIDGE_TARGET_ENV)
    if raw is not None and raw.strip():
        normalized = _normalize_bridge_target(raw)
        if normalized is not None:
            return normalized
        valid = sorted(
            alias for aliases in BRIDGE_TARGET_ALIASES.values() for alias in aliases
        )
        raise CppBridgeError(
            f"{CPP_BRIDGE_TARGET_ENV} must be one of {', '.join(valid)}"
        )
    if _torch_musa_package_dirs():
        return MUSA_TARGET
    return CUDA_TARGET


def _normalize_bridge_target(value: str) -> str | None:
    normalized = value.strip().lower()
    for target, aliases in BRIDGE_TARGET_ALIASES.items():
        if normalized in aliases:
            return target
    return None


def _musa_include_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for package_dir in _torch_musa_package_dirs():
        paths.append(package_dir.parent)
        paths.append(package_dir / "share" / "generated_cuda_compatible")
    for root in _musa_roots():
        paths.append(root / "include")
    return tuple(paths)


def _musa_link_flags() -> tuple[str, ...]:
    flags: list[str] = []
    for package_dir in _torch_musa_package_dirs():
        lib_dir = package_dir / "lib"
        flags.extend(
            [
                f"-L{lib_dir}",
                f"-Wl,-rpath,{lib_dir}",
                "-lmusa_python",
                "-lmusa_kernels",
            ]
        )
    for root in _musa_roots():
        lib_dir = root / "lib"
        flags.extend([f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}", "-lmusart"])
    return tuple(flags)


def _torch_musa_package_dirs() -> tuple[Path, ...]:
    spec = importlib.util.find_spec("torch_musa")
    if spec is None or spec.submodule_search_locations is None:
        return ()
    return tuple(Path(location) for location in spec.submodule_search_locations)


def _musa_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for name in ("MUSA_HOME", "MUSA_PATH", "MUSA_ROOT"):
        value = os.environ.get(name)
        if value:
            roots.append(Path(value))
    roots.append(Path("/usr/local/musa"))
    roots.extend(Path(path) for path in glob.glob("/usr/local/musa-*"))
    return tuple(roots)


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
