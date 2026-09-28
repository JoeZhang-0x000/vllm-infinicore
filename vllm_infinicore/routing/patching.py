"""Opt-in Qwen3 operator forwarding with native attention and KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from types import MappingProxyType
from typing import Callable, Mapping

PATCH_ENABLE_ENV = "VLLM_INFINICORE_ENABLE_PATCHES"
ROUTE_SELECT_ENV = "VLLM_INFINICORE_ROUTES"
ROUTE_DISABLE_ENV = "VLLM_INFINICORE_DISABLED_ROUTES"
FORCE_NATIVE_FALLBACK_ENV = "VLLM_INFINICORE_FORCE_NATIVE_FALLBACK"

ALL_ROUTES_TOKEN = "all"
ROUTE_STATE_DISABLED = "disabled"
ROUTE_STATE_INSTALLED = "installed"
ROUTE_STATE_NATIVE_FALLBACK = "native_fallback"


@dataclass(frozen=True)
class OperatorRoute:
    """Declared target route for one Qwen3 inference operator."""

    name: str
    category: str
    implementation: str
    default_enabled: bool
    graph_policy: str
    native_fallback: str


@dataclass(frozen=True)
class RouteState:
    """Runtime route state for one Qwen3 operator."""

    name: str
    requested: bool
    disabled_by_env: bool
    status: str
    implementation: str
    native_fallback: str
    graph_policy: str
    reason: str

    @property
    def installed(self) -> bool:
        return self.status == ROUTE_STATE_INSTALLED

    @property
    def fallback_active(self) -> bool:
        return self.status == ROUTE_STATE_NATIVE_FALLBACK


@dataclass(frozen=True)
class RegistrationResult:
    """Result returned by the plugin register hook."""

    route_count: int
    patching_enabled: bool
    reason: str
    requested_routes: tuple[str, ...]
    installed_routes: tuple[str, ...]
    skipped_routes: tuple[str, ...]
    failure_reason: str | None = None
    route_states: tuple[RouteState, ...] = ()

    @property
    def enabled_routes(self) -> tuple[str, ...]:
        """Backward-compatible alias for installed route names."""

        return self.installed_routes

    @property
    def native_fallback_routes(self) -> tuple[str, ...]:
        return tuple(
            state.name for state in self.route_states if state.fallback_active
        )

    @property
    def disabled_routes(self) -> tuple[str, ...]:
        return tuple(
            state.name
            for state in self.route_states
            if state.status == ROUTE_STATE_DISABLED
        )


@dataclass(frozen=True)
class PatchInstallResult:
    """Result from one route installer."""

    installed: bool
    reason: str


@dataclass(frozen=True)
class PatchUninstallResult:
    """Result from one route uninstaller."""

    uninstalled: bool
    reason: str


@dataclass(frozen=True)
class PatchUninstallSummary:
    """Summary returned by an explicit patch uninstall call."""

    route_count: int
    requested_routes: tuple[str, ...]
    uninstalled_routes: tuple[str, ...]
    skipped_routes: tuple[str, ...]
    failure_reason: str | None = None


PatchInstaller = Callable[[], PatchInstallResult]
PatchUninstaller = Callable[[], PatchUninstallResult]


QWEN3_OPERATOR_ROUTES: tuple[OperatorRoute, ...] = (
    OperatorRoute(
        name="RMSNorm",
        category="normalization",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM RMSNorm forward_native",
    ),
    OperatorRoute(
        name="SiluAndMul",
        category="mlp",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native SiluAndMul activation",
    ),
    OperatorRoute(
        name="RoPE",
        category="attention",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native rotary embedding",
    ),
    OperatorRoute(
        name="Embedding",
        category="embedding",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native token embedding",
    ),
    OperatorRoute(
        name="MatMul",
        category="linear",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native linear layers",
    ),
    OperatorRoute(
        name="LMHead",
        category="linear",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native LMHead/logits projection",
    ),
    OperatorRoute(
        name="StoreKVCache",
        category="kv-cache",
        implementation="native_vllm",
        default_enabled=False,
        graph_policy="native_vllm",
        native_fallback="vLLM native KV cache store path",
    ),
    OperatorRoute(
        name="PagedAttentionPrefill",
        category="paged-attention",
        implementation="native_vllm",
        default_enabled=False,
        graph_policy="native_vllm",
        native_fallback="vLLM native paged attention prefill backend",
    ),
    OperatorRoute(
        name="PagedAttentionDecode",
        category="paged-attention",
        implementation="native_vllm",
        default_enabled=False,
        graph_policy="native_vllm",
        native_fallback="vLLM native paged attention decode backend",
    ),
)

NATIVE_ATTENTION_ROUTES = frozenset(
    {"StoreKVCache", "PagedAttentionPrefill", "PagedAttentionDecode"}
)
NON_ATTENTION_ROUTES = tuple(
    route.name for route in QWEN3_OPERATOR_ROUTES
    if route.name not in NATIVE_ATTENTION_ROUTES
)


class PatchRegistry:
    """In-process registry for planned vLLM operator patches."""

    def __init__(
        self,
        routes: tuple[OperatorRoute, ...],
        *,
        installers: Mapping[str, PatchInstaller] | None = None,
        uninstallers: Mapping[str, PatchUninstaller] | None = None,
        native_fallback_reasons: Mapping[str, str] | None = None,
    ) -> None:
        self._routes = {route.name: route for route in routes}
        self._installers = dict(installers or {})
        self._uninstallers = dict(uninstallers or {})
        self._native_fallback_reasons = dict(native_fallback_reasons or {})

    @property
    def routes(self) -> Mapping[str, OperatorRoute]:
        return MappingProxyType(self._routes)

    def enabled_route_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, route in self._routes.items() if route.default_enabled
        )

    def register_from_environment(self) -> RegistrationResult:
        patching_requested = _env_truthy(PATCH_ENABLE_ENV)
        force_native_fallback = _env_truthy(FORCE_NATIVE_FALLBACK_ENV)
        requested_routes = _parse_route_names(
            os.environ.get(ROUTE_SELECT_ENV, ""),
            available_routes=tuple(self._routes),
        )
        disabled_route_names = _parse_route_names(
            os.environ.get(ROUTE_DISABLE_ENV, ""),
            available_routes=tuple(self._routes),
        )

        if not patching_requested:
            reason = f"{PATCH_ENABLE_ENV} is unset or false; no vLLM patches installed"
            route_states = self._build_disabled_states(
                requested_routes=requested_routes,
                disabled_route_names=disabled_route_names,
                reason=reason,
            )
            return RegistrationResult(
                route_count=len(self._routes),
                patching_enabled=False,
                reason=reason,
                requested_routes=requested_routes,
                installed_routes=(),
                skipped_routes=requested_routes,
                route_states=route_states,
            )

        if not requested_routes:
            reason = (
                f"{PATCH_ENABLE_ENV} is true but {ROUTE_SELECT_ENV} is unset; "
                "no vLLM patches installed"
            )
            route_states = self._build_disabled_states(
                requested_routes=(),
                disabled_route_names=disabled_route_names,
                reason=reason,
            )
            return RegistrationResult(
                route_count=len(self._routes),
                patching_enabled=False,
                reason=reason,
                requested_routes=(),
                installed_routes=(),
                skipped_routes=(),
                route_states=route_states,
            )

        unknown_routes = self._unknown_routes(
            (*requested_routes, *disabled_route_names)
        )
        if unknown_routes:
            failure_reason = (
                f"unknown {ROUTE_SELECT_ENV} route(s): {', '.join(unknown_routes)}"
            )
            route_states = self._build_disabled_states(
                requested_routes=requested_routes,
                disabled_route_names=disabled_route_names,
                reason=f"patching rejected: {failure_reason}",
            )
            return RegistrationResult(
                route_count=len(self._routes),
                patching_enabled=False,
                reason=f"patching rejected: {failure_reason}",
                requested_routes=requested_routes,
                installed_routes=(),
                skipped_routes=requested_routes,
                failure_reason=failure_reason,
                route_states=route_states,
            )

        installed_routes: list[str] = []
        skipped_routes: list[str] = []
        failure_reasons: list[str] = []
        route_states: list[RouteState] = []
        disabled_route_set = set(disabled_route_names)
        requested_route_set = set(requested_routes)

        for route_name, route in self._routes.items():
            requested = route_name in requested_route_set
            disabled_by_env = route_name in disabled_route_set
            if not requested:
                route_states.append(
                    self._route_state(
                        route,
                        requested=False,
                        disabled_by_env=disabled_by_env,
                        status=ROUTE_STATE_DISABLED,
                        reason="route not requested",
                    )
                )
                continue

            if disabled_by_env:
                skipped_routes.append(route_name)
                route_states.append(
                    self._route_state(
                        route,
                        requested=True,
                        disabled_by_env=True,
                        status=ROUTE_STATE_DISABLED,
                        reason=f"route disabled by {ROUTE_DISABLE_ENV}",
                    )
                )
                continue

            if force_native_fallback:
                skipped_routes.append(route_name)
                route_states.append(
                    self._route_state(
                        route,
                        requested=True,
                        disabled_by_env=False,
                        status=ROUTE_STATE_NATIVE_FALLBACK,
                        reason=(
                            f"{FORCE_NATIVE_FALLBACK_ENV} is true; using "
                            f"{route.native_fallback}"
                        ),
                    )
                )
                continue

            unsupported_reason = self._native_fallback_reasons.get(route_name)
            if unsupported_reason is not None:
                # A missing adapter is a supported native-fallback state.
                skipped_routes.append(route_name)
                route_states.append(
                    self._route_state(
                        route,
                        requested=True,
                        disabled_by_env=False,
                        status=ROUTE_STATE_NATIVE_FALLBACK,
                        reason=unsupported_reason,
                    )
                )
                continue

            if route_name in NATIVE_ATTENTION_ROUTES:
                skipped_routes.append(route_name)
                route_states.append(
                    self._route_state(
                        route,
                        requested=True,
                        disabled_by_env=False,
                        status=ROUTE_STATE_NATIVE_FALLBACK,
                        reason="v1 keeps attention and KV cache on native vLLM operators",
                    )
                )
                continue

            installer = self._installers.get(route_name)
            if installer is None:
                skipped_routes.append(route_name)
                route_states.append(
                    self._route_state(
                        route,
                        requested=True,
                        disabled_by_env=False,
                        status=ROUTE_STATE_NATIVE_FALLBACK,
                        reason=f"no installer yet; using {route.native_fallback}",
                    )
                )
                continue

            try:
                install_result = installer()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                install_result = PatchInstallResult(
                    installed=False,
                    reason=f"installer raised {type(exc).__name__}: {exc}",
                )

            if install_result.installed:
                installed_routes.append(route_name)
                route_states.append(
                    self._route_state(
                        route,
                        requested=True,
                        disabled_by_env=False,
                        status=ROUTE_STATE_INSTALLED,
                        reason=install_result.reason,
                    )
                )
            else:
                skipped_routes.append(route_name)
                failure_reasons.append(
                    f"{route_name}: {install_result.reason}; "
                    f"using {route.native_fallback}"
                )
                route_states.append(
                    self._route_state(
                        route,
                        requested=True,
                        disabled_by_env=False,
                        status=ROUTE_STATE_NATIVE_FALLBACK,
                        reason=(
                            f"installer unavailable: {install_result.reason}; "
                            f"using {route.native_fallback}"
                        ),
                    )
                )

        failure_reason = "; ".join(failure_reasons) or None
        if installed_routes:
            reason = "installed vLLM patches for routes: " + ", ".join(
                installed_routes
            )
            if skipped_routes:
                reason += "; skipped routes: " + ", ".join(skipped_routes)
        elif failure_reason:
            reason = f"patch installation fell back to native: {failure_reason}"
        else:
            reason = "requested routes use native fallback: " + ", ".join(skipped_routes)

        return RegistrationResult(
            route_count=len(self._routes),
            patching_enabled=bool(installed_routes),
            reason=reason,
            requested_routes=requested_routes,
            installed_routes=tuple(installed_routes),
            skipped_routes=tuple(skipped_routes),
            failure_reason=failure_reason,
            route_states=tuple(route_states),
        )

    def uninstall_routes(
        self,
        route_names: tuple[str, ...] | None = None,
    ) -> PatchUninstallSummary:
        """Uninstall route patches owned by this plugin.

        Routes without an installed patch are treated as no-op native fallback
        paths, so repeated calls remain safe.
        """

        requested_routes = route_names
        if requested_routes is None:
            requested_routes = tuple(self._routes)

        unknown_routes = self._unknown_routes(requested_routes)
        if unknown_routes:
            failure_reason = f"unknown route(s): {', '.join(unknown_routes)}"
            return PatchUninstallSummary(
                route_count=len(self._routes),
                requested_routes=requested_routes,
                uninstalled_routes=(),
                skipped_routes=requested_routes,
                failure_reason=failure_reason,
            )

        uninstalled_routes: list[str] = []
        skipped_routes: list[str] = []
        failure_reasons: list[str] = []
        for route_name in requested_routes:
            uninstaller = self._uninstallers.get(route_name)
            if uninstaller is None:
                skipped_routes.append(route_name)
                continue

            try:
                uninstall_result = uninstaller()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                uninstall_result = PatchUninstallResult(
                    uninstalled=False,
                    reason=f"uninstaller raised {type(exc).__name__}: {exc}",
                )

            if uninstall_result.uninstalled:
                uninstalled_routes.append(route_name)
            else:
                skipped_routes.append(route_name)
                if "not installed" not in uninstall_result.reason:
                    failure_reasons.append(f"{route_name}: {uninstall_result.reason}")

        return PatchUninstallSummary(
            route_count=len(self._routes),
            requested_routes=requested_routes,
            uninstalled_routes=tuple(uninstalled_routes),
            skipped_routes=tuple(skipped_routes),
            failure_reason="; ".join(failure_reasons) or None,
        )

    def _unknown_routes(self, route_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            route_name for route_name in route_names if route_name not in self._routes
        )

    def _build_disabled_states(
        self,
        *,
        requested_routes: tuple[str, ...],
        disabled_route_names: tuple[str, ...],
        reason: str,
    ) -> tuple[RouteState, ...]:
        requested_route_set = set(requested_routes)
        disabled_route_set = set(disabled_route_names)
        return tuple(
            self._route_state(
                route,
                requested=route.name in requested_route_set,
                disabled_by_env=route.name in disabled_route_set,
                status=ROUTE_STATE_DISABLED,
                reason=reason,
            )
            for route in self._routes.values()
        )

    @staticmethod
    def _route_state(
        route: OperatorRoute,
        *,
        requested: bool,
        disabled_by_env: bool,
        status: str,
        reason: str,
    ) -> RouteState:
        return RouteState(
            name=route.name,
            requested=requested,
            disabled_by_env=disabled_by_env,
            status=status,
            implementation=route.implementation,
            native_fallback=route.native_fallback,
            graph_policy=route.graph_policy,
            reason=reason,
        )


def get_default_registry() -> PatchRegistry:
    from dataclasses import replace

    from ..operators.selection import OPERATOR_BACKEND_ENV, selected_backend

    backend_name = selected_backend()
    adapter = (
        import_module(f"..operators.{backend_name}", __package__)
        if backend_name is not None
        else None
    )
    supported = set(adapter.SUPPORTED_ROUTES) if adapter is not None else set()
    unsupported = supported.difference(NON_ATTENTION_ROUTES)
    if unsupported:
        raise ValueError(
            f"operator backend {backend_name} declares unsupported routes: "
            + ", ".join(sorted(unsupported))
        )
    if backend_name == "ascend" and not os.environ.get(
        "VLLM_INFINICORE_ASCEND_LIBRARY"
    ):
        supported.clear()

    fallback_reasons = {}
    for route in QWEN3_OPERATOR_ROUTES:
        if route.name in supported:
            continue
        if route.name in NATIVE_ATTENTION_ROUTES:
            reason = "v1 keeps attention and KV cache on native vLLM operators"
        elif backend_name is None:
            reason = f"{OPERATOR_BACKEND_ENV} is unset"
        elif backend_name == "ascend" and not os.environ.get(
            "VLLM_INFINICORE_ASCEND_LIBRARY"
        ):
            reason = "VLLM_INFINICORE_ASCEND_LIBRARY is unset"
        else:
            reason = f"{backend_name} operator adapter does not support this route"
        fallback_reasons[route.name] = reason

    def installer(name: str) -> PatchInstaller:
        def install() -> PatchInstallResult:
            routes = import_module(f"..operators.{backend_name}.routes", __package__)
            status = routes.install(name)
            return PatchInstallResult(status.installed, status.reason)

        return install

    def uninstaller(name: str) -> PatchUninstaller:
        def uninstall() -> PatchUninstallResult:
            routes = import_module(f"..operators.{backend_name}.routes", __package__)
            status = routes.uninstall(name)
            return PatchUninstallResult(status.uninstalled, status.reason)

        return uninstall

    routes = tuple(
        replace(
            route,
            implementation=(
                f"{backend_name}_operator_adapter"
                if route.name in supported
                else "native_vllm"
            ),
            graph_policy=(
                adapter.GRAPH_POLICY
                if route.name in supported
                else "native_vllm"
            ),
        )
        for route in QWEN3_OPERATOR_ROUTES
    )
    return PatchRegistry(
        routes,
        installers={name: installer(name) for name in supported},
        uninstallers={name: uninstaller(name) for name in supported},
        native_fallback_reasons=fallback_reasons,
    )


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_route_names(
    value: str,
    *,
    available_routes: tuple[str, ...] = (),
) -> tuple[str, ...]:
    route_names: list[str] = []
    seen: set[str] = set()
    for raw_name in value.split(","):
        route_name = raw_name.strip()
        if not route_name:
            continue
        route_token = route_name.lower()
        if route_token == ALL_ROUTES_TOKEN and available_routes:
            expanded_routes = available_routes
        else:
            expanded_routes = ()
        if expanded_routes:
            for available_route in expanded_routes:
                if available_route not in seen:
                    seen.add(available_route)
                    route_names.append(available_route)
            continue
        if route_name in seen:
            continue
        seen.add(route_name)
        route_names.append(route_name)
    return tuple(route_names)
