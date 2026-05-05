"""On-demand C++ bridge for hot InfiniCore routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

CPP_BRIDGE_ENABLE_ENV = "VLLM_INFINICORE_ENABLE_CPP_BRIDGE"
CPP_BRIDGE_ROUTES_ENV = "VLLM_INFINICORE_CPP_BRIDGE_ROUTES"

DECODE_ROUTE = "PagedAttentionDecode"
LM_HEAD_ROUTE = "LMHead"
SUPPORTED_ROUTES = frozenset({DECODE_ROUTE, LM_HEAD_ROUTE})

_MODULE: Any | None = None
_LOAD_ERROR: str | None = None
_CALL_COUNTS: dict[str, int] = {}


class CppBridgeError(RuntimeError):
    pass


def enabled_for(route_name: str) -> bool:
    return _env_truthy(CPP_BRIDGE_ENABLE_ENV) and route_name in selected_routes()


def selected_routes() -> tuple[str, ...]:
    raw = os.environ.get(CPP_BRIDGE_ROUTES_ENV, "")
    routes = tuple(route.strip() for route in raw.split(",") if route.strip())
    if routes == ("all",):
        return tuple(sorted(SUPPORTED_ROUTES))
    unknown = tuple(route for route in routes if route not in SUPPORTED_ROUTES)
    if unknown:
        raise CppBridgeError(f"unsupported C++ bridge route(s): {', '.join(unknown)}")
    return routes


def module() -> Any:
    global _MODULE, _LOAD_ERROR

    if not _env_truthy(CPP_BRIDGE_ENABLE_ENV):
        raise CppBridgeError(f"{CPP_BRIDGE_ENABLE_ENV} is unset or false")
    selected_routes()
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

    root = Path(__file__).resolve().parents[1]
    source = root / "csrc" / "infinicore_bridge.cpp"
    infini_root = Path(os.environ.get("INFINI_ROOT", str(Path.home() / ".infini")))
    maca_path = Path(os.environ.get("MACA_PATH", "/opt/maca-3.5.3"))

    return load(
        name="vllm_infinicore_cpp_bridge",
        sources=[str(source)],
        extra_include_paths=[str(infini_root / "include")],
        extra_cflags=["-std=c++17"],
        extra_ldflags=[
            f"-L{infini_root / 'lib'}",
            f"-Wl,-rpath,{infini_root / 'lib'}",
            f"-Wl,-rpath,{maca_path / 'lib'}",
            f"-Wl,-rpath,{maca_path / 'lib64'}",
            "-linfinicore_cpp_api",
            "-linfiniop",
            "-linfinirt",
            "-linfiniccl",
        ],
        verbose=_env_truthy("VLLM_INFINICORE_CPP_BRIDGE_VERBOSE"),
    )


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
