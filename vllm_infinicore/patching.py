"""Conservative Qwen3 operator routing scaffold.

The first project stage defines the routing surface only. It intentionally
does not patch vLLM internals or register C++ kernels until each replacement
path has a graph-safety and correctness test.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Mapping

PATCH_ENABLE_ENV = "VLLM_INFINICORE_ENABLE_PATCHES"


@dataclass(frozen=True)
class OperatorRoute:
    """Declared target route for one Qwen3 inference operator."""

    name: str
    category: str
    implementation: str
    default_enabled: bool
    graph_policy: str
    notes: str


@dataclass(frozen=True)
class RegistrationResult:
    """Result returned by the plugin register hook."""

    route_count: int
    patching_enabled: bool
    reason: str
    enabled_routes: tuple[str, ...]


QWEN3_OPERATOR_ROUTES: tuple[OperatorRoute, ...] = (
    OperatorRoute(
        name="RMSNorm",
        category="normalization",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="requires explicit graph-safety proof",
        notes="First non-PA candidate for a C++ PyTorch custom op.",
    ),
    OperatorRoute(
        name="SiluAndMul",
        category="mlp",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="requires explicit graph-safety proof",
        notes="Candidate for a fused activation custom op.",
    ),
    OperatorRoute(
        name="RoPE",
        category="attention",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="requires explicit graph-safety proof",
        notes="Keep tensor views and shape metadata stable before enabling graph.",
    ),
    OperatorRoute(
        name="Embedding",
        category="embedding",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="avoid from_blob/from_torch calls inside graph capture",
        notes="Known graph-sensitive area from prior experiments.",
    ),
    OperatorRoute(
        name="MatMul",
        category="linear",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="requires explicit graph-safety proof",
        notes="Covers QKV, gate/up, down, and output projection candidates.",
    ),
    OperatorRoute(
        name="LMHead",
        category="linear",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="requires explicit graph-safety proof",
        notes="Separate route name for final logits projection accounting.",
    ),
    OperatorRoute(
        name="StoreKVCache",
        category="kv-cache",
        implementation="deferred_pa_kv",
        default_enabled=False,
        graph_policy="deferred; do not enable explicit graph path in skeleton",
        notes="KV descriptor stability needs a dedicated phase.",
    ),
    OperatorRoute(
        name="PagedAttentionPrefill",
        category="paged-attention",
        implementation="deferred_pa_kv",
        default_enabled=False,
        graph_policy="deferred; keep native vLLM cudagraph baseline intact",
        notes="FA2/PA path requires separate correctness and graph validation.",
    ),
    OperatorRoute(
        name="PagedAttentionDecode",
        category="paged-attention",
        implementation="deferred_pa_kv",
        default_enabled=False,
        graph_policy="deferred; keep native vLLM cudagraph baseline intact",
        notes="Decode path is performance critical but not enabled by default.",
    ),
)


class PatchRegistry:
    """In-process registry for planned vLLM operator patches."""

    def __init__(self, routes: tuple[OperatorRoute, ...]) -> None:
        self._routes = {route.name: route for route in routes}

    @property
    def routes(self) -> Mapping[str, OperatorRoute]:
        return MappingProxyType(self._routes)

    def enabled_route_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, route in self._routes.items() if route.default_enabled
        )

    def register_from_environment(self) -> RegistrationResult:
        requested = _env_truthy(PATCH_ENABLE_ENV)
        enabled_routes = self.enabled_route_names() if requested else ()

        if not requested:
            reason = f"{PATCH_ENABLE_ENV} is unset or false; no vLLM patches installed"
        elif not enabled_routes:
            reason = "patching requested, but skeleton has no enabled routes yet"
        else:
            reason = "patch route declarations loaded; patch installers are not implemented"

        return RegistrationResult(
            route_count=len(self._routes),
            patching_enabled=requested and bool(enabled_routes),
            reason=reason,
            enabled_routes=enabled_routes,
        )


def get_default_registry() -> PatchRegistry:
    return PatchRegistry(QWEN3_OPERATOR_ROUTES)


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
