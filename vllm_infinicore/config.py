"""Structured configuration loading for vllm-infinicore."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .patching import (
    FORCE_NATIVE_FALLBACK_ENV,
    PATCH_ENABLE_ENV,
    ROUTE_DISABLE_ENV,
    ROUTE_SELECT_ENV,
    PatchRegistry,
    get_default_registry,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "qwen3_infinicore_graph.yaml"
)

_ROUTE_IMPLEMENTATIONS = {"torch_custom_op", "deferred_pa_kv"}
_GRAPH_POLICIES = {
    "requires_explicit_graph_safety_proof",
    "avoid_from_blob_or_from_torch_inside_capture",
    "deferred_no_explicit_graph_path",
    "deferred_keep_native_vllm_cudagraph_baseline",
}


class ConfigValidationError(ValueError):
    """Raised when the plugin config is malformed or out of sync."""


@dataclass(frozen=True)
class PluginConfig:
    name: str
    entry_point_group: str
    entry_point_name: str
    mode: str
    patch_enable_env: str
    route_select_env: str
    route_disable_env: str
    force_native_fallback_env: str


@dataclass(frozen=True)
class TargetConfig:
    machine: str
    maca: str
    model_family: str
    audit_model_path: str


@dataclass(frozen=True)
class CudaGraphConfig:
    default_policy: str
    vllm_compilation_config: Mapping[str, Any]
    enforce_eager: bool
    avoid_vllm_compile_inductor_by_default: bool


@dataclass(frozen=True)
class RouteConfig:
    name: str
    category: str
    implementation: str
    default_enabled: bool
    graph_policy: str
    native_fallback: str
    validation: str


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int
    eos: str


@dataclass(frozen=True)
class VllmTokensConfig:
    min_tokens_equals_max_tokens: bool


@dataclass(frozen=True)
class ValidationConfig:
    require_output_token_count: bool
    require_decoded_preview_health: bool
    require_warmup_and_repeats: bool


@dataclass(frozen=True)
class BenchmarkRulesConfig:
    prompt_ids: str
    metric: str
    sampling: SamplingConfig
    vllm_tokens: VllmTokensConfig
    validation: ValidationConfig


@dataclass(frozen=True)
class InfiniCoreConfig:
    plugin: PluginConfig
    target: TargetConfig
    cuda_graph: CudaGraphConfig
    routes: tuple[RouteConfig, ...]
    benchmark_rules: BenchmarkRulesConfig
    source: Path | str


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    validate_registry: bool = True,
) -> InfiniCoreConfig:
    """Load and validate a vllm-infinicore YAML config file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    config = parse_config(raw_config, source=config_path)
    if validate_registry:
        validate_config_against_registry(config)
    return config


def parse_config(raw_config: object, *, source: Path | str = "<memory>") -> InfiniCoreConfig:
    """Parse raw YAML data into typed config objects."""

    root = _require_mapping(raw_config, "config root", source)
    plugin = _parse_plugin(_require_mapping(root.get("plugin"), "plugin", source))
    target = _parse_target(_require_mapping(root.get("target"), "target", source))
    cuda_graph = _parse_cuda_graph(
        _require_mapping(root.get("cuda_graph"), "cuda_graph", source)
    )
    routes = _parse_routes(_require_list(root.get("routes"), "routes", source), source)
    benchmark_rules = _parse_benchmark_rules(
        _require_mapping(root.get("benchmark_rules"), "benchmark_rules", source)
    )

    return InfiniCoreConfig(
        plugin=plugin,
        target=target,
        cuda_graph=cuda_graph,
        routes=routes,
        benchmark_rules=benchmark_rules,
        source=source,
    )


def validate_config_against_registry(
    config: InfiniCoreConfig,
    registry: PatchRegistry | None = None,
) -> None:
    """Ensure config route declarations match the in-code registry."""

    if config.plugin.patch_enable_env != PATCH_ENABLE_ENV:
        raise ConfigValidationError(
            "plugin.patch_enable_env must match "
            f"{PATCH_ENABLE_ENV!r}, got {config.plugin.patch_enable_env!r}"
        )
    if config.plugin.route_select_env != ROUTE_SELECT_ENV:
        raise ConfigValidationError(
            "plugin.route_select_env must match "
            f"{ROUTE_SELECT_ENV!r}, got {config.plugin.route_select_env!r}"
        )
    if config.plugin.route_disable_env != ROUTE_DISABLE_ENV:
        raise ConfigValidationError(
            "plugin.route_disable_env must match "
            f"{ROUTE_DISABLE_ENV!r}, got {config.plugin.route_disable_env!r}"
        )
    if config.plugin.force_native_fallback_env != FORCE_NATIVE_FALLBACK_ENV:
        raise ConfigValidationError(
            "plugin.force_native_fallback_env must match "
            f"{FORCE_NATIVE_FALLBACK_ENV!r}, "
            f"got {config.plugin.force_native_fallback_env!r}"
        )

    registry = registry or get_default_registry()
    declared_routes = registry.routes
    configured_names = {route.name for route in config.routes}
    declared_names = set(declared_routes)

    missing = sorted(declared_names - configured_names)
    extra = sorted(configured_names - declared_names)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing routes: {', '.join(missing)}")
        if extra:
            parts.append(f"unknown routes: {', '.join(extra)}")
        raise ConfigValidationError("; ".join(parts))

    mismatches: list[str] = []
    for route in config.routes:
        declared = declared_routes[route.name]
        for field_name in (
            "category",
            "implementation",
            "default_enabled",
            "graph_policy",
            "native_fallback",
            "validation",
        ):
            configured = getattr(route, field_name)
            expected = getattr(declared, field_name)
            if configured != expected:
                mismatches.append(
                    f"{route.name}.{field_name}: config={configured!r} registry={expected!r}"
                )

    if mismatches:
        raise ConfigValidationError("; ".join(mismatches))


def _parse_plugin(data: Mapping[str, Any]) -> PluginConfig:
    return PluginConfig(
        name=_require_str(data, "name"),
        entry_point_group=_require_str(data, "entry_point_group"),
        entry_point_name=_require_str(data, "entry_point_name"),
        mode=_require_str(data, "mode"),
        patch_enable_env=_require_str(data, "patch_enable_env"),
        route_select_env=_require_str(data, "route_select_env"),
        route_disable_env=_require_str(data, "route_disable_env"),
        force_native_fallback_env=_require_str(data, "force_native_fallback_env"),
    )


def _parse_target(data: Mapping[str, Any]) -> TargetConfig:
    return TargetConfig(
        machine=_require_str(data, "machine"),
        maca=str(_require_scalar(data, "maca")),
        model_family=_require_str(data, "model_family"),
        audit_model_path=_require_str(data, "audit_model_path"),
    )


def _parse_cuda_graph(data: Mapping[str, Any]) -> CudaGraphConfig:
    compilation_config = dict(
        _require_mapping(
            data.get("vllm_compilation_config"),
            "cuda_graph.vllm_compilation_config",
            "<memory>",
        )
    )
    return CudaGraphConfig(
        default_policy=_require_str(data, "default_policy"),
        vllm_compilation_config=compilation_config,
        enforce_eager=_require_bool(data, "enforce_eager"),
        avoid_vllm_compile_inductor_by_default=_require_bool(
            data, "avoid_vllm_compile_inductor_by_default"
        ),
    )


def _parse_routes(routes: list[Any], source: Path | str) -> tuple[RouteConfig, ...]:
    parsed_routes: list[RouteConfig] = []
    seen_names: set[str] = set()

    for index, raw_route in enumerate(routes):
        context = f"{source}:routes[{index}]"
        route_data = _require_mapping(raw_route, context, source)
        route = RouteConfig(
            name=_require_str(route_data, "name"),
            category=_require_str(route_data, "category"),
            implementation=_require_str(route_data, "implementation"),
            default_enabled=_require_bool(route_data, "default_enabled"),
            graph_policy=_require_str(route_data, "graph_policy"),
            native_fallback=_require_str(route_data, "native_fallback"),
            validation=_require_str(route_data, "validation"),
        )

        if route.name in seen_names:
            raise ConfigValidationError(f"duplicate route {route.name!r} in {source}")
        seen_names.add(route.name)

        if route.implementation not in _ROUTE_IMPLEMENTATIONS:
            raise ConfigValidationError(
                f"{context}.implementation has unsupported value {route.implementation!r}"
            )
        if route.graph_policy not in _GRAPH_POLICIES:
            raise ConfigValidationError(
                f"{context}.graph_policy has unsupported value {route.graph_policy!r}"
            )

        parsed_routes.append(route)

    return tuple(parsed_routes)


def _parse_benchmark_rules(data: Mapping[str, Any]) -> BenchmarkRulesConfig:
    sampling = _require_mapping(data.get("sampling"), "benchmark_rules.sampling", "<memory>")
    vllm_tokens = _require_mapping(
        data.get("vllm_tokens"), "benchmark_rules.vllm_tokens", "<memory>"
    )
    validation = _require_mapping(
        data.get("validation"), "benchmark_rules.validation", "<memory>"
    )
    return BenchmarkRulesConfig(
        prompt_ids=_require_str(data, "prompt_ids"),
        metric=_require_str(data, "metric"),
        sampling=SamplingConfig(
            temperature=float(_require_scalar(sampling, "temperature")),
            top_p=float(_require_scalar(sampling, "top_p")),
            top_k=int(_require_scalar(sampling, "top_k")),
            eos=_require_str(sampling, "eos"),
        ),
        vllm_tokens=VllmTokensConfig(
            min_tokens_equals_max_tokens=_require_bool(
                vllm_tokens, "min_tokens_equals_max_tokens"
            )
        ),
        validation=ValidationConfig(
            require_output_token_count=_require_bool(
                validation, "require_output_token_count"
            ),
            require_decoded_preview_health=_require_bool(
                validation, "require_decoded_preview_health"
            ),
            require_warmup_and_repeats=_require_bool(
                validation, "require_warmup_and_repeats"
            ),
        ),
    )


def _require_mapping(value: object, name: str, source: Path | str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{source}: {name} must be a mapping")
    return value


def _require_list(value: object, name: str, source: Path | str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigValidationError(f"{source}: {name} must be a list")
    return value


def _require_str(data: Mapping[str, Any], name: str) -> str:
    value = _require_scalar(data, name)
    if not isinstance(value, str):
        raise ConfigValidationError(f"{name} must be a string")
    return value


def _require_bool(data: Mapping[str, Any], name: str) -> bool:
    value = _require_scalar(data, name)
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{name} must be a boolean")
    return value


def _require_scalar(data: Mapping[str, Any], name: str) -> object:
    if name not in data:
        raise ConfigValidationError(f"missing required field {name}")
    value = data[name]
    if isinstance(value, (Mapping, list)):
        raise ConfigValidationError(f"{name} must be a scalar")
    return value
