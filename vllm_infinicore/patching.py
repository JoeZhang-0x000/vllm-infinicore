"""Conservative Qwen3 operator routing scaffold.

Routes are default-off and graph-conservative. A route is installed only through
explicit environment gates; otherwise requested operators remain on vLLM native
fallback paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Callable, Mapping


@dataclass(frozen=True)
class OperatorRoute:
    """Declared target route for one Qwen3 inference operator."""

    name: str
    category: str
    implementation: str
    default_enabled: bool
    graph_policy: str
    native_fallback: str
    validation: str
    notes: str


@dataclass(frozen=True)
class RouteState:
    """Runtime route state for one Qwen3 operator."""

    name: str
    requested: bool
    disabled_by_env: bool
    status: str
    implementation: str
    native_fallback: str
    validation: str
    graph_policy: str
    reason: str

    @property
    def installed(self) -> bool:
        return self.status == "installed"

    @property
    def fallback_active(self) -> bool:
        return self.status == "native_fallback"


@dataclass(frozen=True)
class RegistrationResult:
    """Result returned by the plugin register hook."""

    route_count: int
    patching_enabled: bool
    reason: str
    requested_routes: tuple[str, ...]
    installed_routes: tuple[str, ...]
    failed_routes: tuple[str, ...]
    skipped_routes: tuple[str, ...] = ()


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
        validation="unit_numeric_compare; qwen3_exact_token_smoke; graph_smoke",
        notes="InfiniCore-backed OOT RMSNorm route.",
    ),
    OperatorRoute(
        name="SiluAndMul",
        category="mlp",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native SiluAndMul activation",
        validation="qwen3_exact_token_smoke and graph_smoke",
        notes="InfiniCore swiglu-backed fused activation route.",
    ),
    OperatorRoute(
        name="RoPE",
        category="attention",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native rotary embedding",
        validation="qwen3_exact_token_smoke and graph evidence",
        notes="InfiniCore RoPE route with stream-bridge graph validation.",
    ),
    OperatorRoute(
        name="Embedding",
        category="embedding",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native token embedding",
        validation="qwen3_exact_token_smoke and graph capture probe",
        notes="InfiniCore embedding route with stream-bridge graph validation.",
    ),
    OperatorRoute(
        name="MatMul",
        category="linear",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native linear layers",
        validation="operator numeric compare; qwen3_exact_token_smoke; graph_smoke",
        notes="Covers QKV, gate/up, down, and output projection candidates.",
    ),
    OperatorRoute(
        name="LMHead",
        category="linear",
        implementation="torch_custom_op",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native LMHead/logits projection",
        validation="logits numeric compare; qwen3_exact_token_smoke; graph_smoke",
        notes="Separate route name for final logits projection accounting.",
    ),
    OperatorRoute(
        name="StoreKVCache",
        category="kv-cache",
        implementation="infinicore_attention_backend",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native KV cache store path",
        validation="attention probe; qwen3_exact_token_smoke; graph_smoke",
        notes="Patches attention backend KV update to InfiniCore paged_caching.",
    ),
    OperatorRoute(
        name="PagedAttentionPrefill",
        category="paged-attention",
        implementation="infinicore_attention_backend",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native paged attention prefill backend",
        validation="attention output compare; qwen3_exact_token_smoke; graph evidence",
        notes="Patches attention backend prefill path to InfiniCore paged_attention_prefill.",
    ),
    OperatorRoute(
        name="PagedAttentionDecode",
        category="paged-attention",
        implementation="infinicore_attention_backend",
        default_enabled=False,
        graph_policy="stream_bridge_graph_validated",
        native_fallback="vLLM native paged attention decode backend",
        validation="decode health; qwen3_exact_token_smoke; graph evidence",
        notes="Patches attention backend decode path to InfiniCore paged_attention.",
    ),
)


def install_all_routes() -> RegistrationResult:
    import os

    if os.environ.get("VLLM_INFINICORE_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return RegistrationResult(
            route_count=len(QWEN3_OPERATOR_ROUTES),
            patching_enabled=False,
            reason="VLLM_INFINICORE_DISABLE is set",
            requested_routes=(),
            installed_routes=(),
            failed_routes=(),
            skipped_routes=(),
        )

    installed: list[str] = []
    failed: list[str] = []
    for route_name in _DEFAULT_INSTALLERS:
        installer = _DEFAULT_INSTALLERS[route_name]
        try:
            result = installer()
            if result.installed:
                installed.append(route_name)
            else:
                failed.append(route_name)
        except Exception:
            failed.append(route_name)

    reason = f"installed {len(installed)} routes" if installed else "installation failed"
    return RegistrationResult(
        route_count=len(QWEN3_OPERATOR_ROUTES),
        patching_enabled=bool(installed),
        reason=reason,
        requested_routes=tuple(_DEFAULT_INSTALLERS.keys()),
        installed_routes=tuple(installed),
        failed_routes=tuple(failed),
        skipped_routes=(),
    )


def _install_rms_norm_route() -> PatchInstallResult:
    from .ops.vllm_rms_norm import install_vllm_rms_norm_oot

    status = install_vllm_rms_norm_oot()
    return PatchInstallResult(installed=status.installed, reason=status.reason)


def _uninstall_rms_norm_route() -> PatchUninstallResult:
    from .ops.vllm_rms_norm import uninstall_vllm_rms_norm_oot

    status = uninstall_vllm_rms_norm_oot()
    return PatchUninstallResult(uninstalled=status.uninstalled, reason=status.reason)


def _install_silu_and_mul_route() -> PatchInstallResult:
    from .ops.vllm_silu_and_mul import install_vllm_silu_and_mul_oot

    status = install_vllm_silu_and_mul_oot()
    return PatchInstallResult(installed=status.installed, reason=status.reason)


def _uninstall_silu_and_mul_route() -> PatchUninstallResult:
    from .ops.vllm_silu_and_mul import uninstall_vllm_silu_and_mul_oot

    status = uninstall_vllm_silu_and_mul_oot()
    return PatchUninstallResult(uninstalled=status.uninstalled, reason=status.reason)


def _install_rotary_embedding_route() -> PatchInstallResult:
    from .ops.vllm_rotary_embedding import install_vllm_rotary_embedding_oot

    status = install_vllm_rotary_embedding_oot()
    return PatchInstallResult(installed=status.installed, reason=status.reason)


def _uninstall_rotary_embedding_route() -> PatchUninstallResult:
    from .ops.vllm_rotary_embedding import uninstall_vllm_rotary_embedding_oot

    status = uninstall_vllm_rotary_embedding_oot()
    return PatchUninstallResult(uninstalled=status.uninstalled, reason=status.reason)


def _install_embedding_route() -> PatchInstallResult:
    from .ops.vllm_embedding import install_vllm_unquantized_embedding_route

    status = install_vllm_unquantized_embedding_route()
    return PatchInstallResult(installed=status.installed, reason=status.reason)


def _uninstall_embedding_route() -> PatchUninstallResult:
    from .ops.vllm_embedding import uninstall_vllm_unquantized_embedding_route

    status = uninstall_vllm_unquantized_embedding_route()
    return PatchUninstallResult(uninstalled=status.uninstalled, reason=status.reason)


def _make_linear_installer(route_name: str) -> PatchInstaller:
    def installer() -> PatchInstallResult:
        from .ops.vllm_linear import install_vllm_unquantized_linear_route

        status = install_vllm_unquantized_linear_route(route_name)
        return PatchInstallResult(installed=status.installed, reason=status.reason)

    return installer


def _make_linear_uninstaller(route_name: str) -> PatchUninstaller:
    def uninstaller() -> PatchUninstallResult:
        from .ops.vllm_linear import uninstall_vllm_unquantized_linear_route

        status = uninstall_vllm_unquantized_linear_route(route_name)
        return PatchUninstallResult(uninstalled=status.uninstalled, reason=status.reason)

    return uninstaller


def _make_attention_installer(route_name: str) -> PatchInstaller:
    def installer() -> PatchInstallResult:
        from .ops.infinicore_attention import InfiniCoreFlashAttentionBackend
        return PatchInstallResult(
            installed=True,
            reason=f"InfiniCore FLASH_ATTN backend registered for {route_name}",
        )

    return installer


def _make_attention_uninstaller(route_name: str) -> PatchUninstaller:
    def uninstaller() -> PatchUninstallResult:
        return PatchUninstallResult(
            uninstalled=True,
            reason=f"InfiniCore attention backend {route_name} unregistered",
        )

    return uninstaller


_DEFAULT_INSTALLERS: Mapping[str, PatchInstaller] = MappingProxyType(
    {
        "RMSNorm": _install_rms_norm_route,
        "SiluAndMul": _install_silu_and_mul_route,
        "RoPE": _install_rotary_embedding_route,
        "Embedding": _install_embedding_route,
        "MatMul": _make_linear_installer("MatMul"),
        "LMHead": _make_linear_installer("LMHead"),
        "StoreKVCache": _make_attention_installer("StoreKVCache"),
        "PagedAttentionPrefill": _make_attention_installer("PagedAttentionPrefill"),
        "PagedAttentionDecode": _make_attention_installer("PagedAttentionDecode"),
    }
)
_DEFAULT_UNINSTALLERS: Mapping[str, PatchUninstaller] = MappingProxyType(
    {
        "RMSNorm": _uninstall_rms_norm_route,
        "SiluAndMul": _uninstall_silu_and_mul_route,
        "RoPE": _uninstall_rotary_embedding_route,
        "Embedding": _uninstall_embedding_route,
        "MatMul": _make_linear_uninstaller("MatMul"),
        "LMHead": _make_linear_uninstaller("LMHead"),
        "StoreKVCache": _make_attention_uninstaller("StoreKVCache"),
        "PagedAttentionPrefill": _make_attention_uninstaller("PagedAttentionPrefill"),
        "PagedAttentionDecode": _make_attention_uninstaller("PagedAttentionDecode"),
    }
)
