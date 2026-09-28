"""vLLM general plugin registration entry point."""

from __future__ import annotations

import logging

from .routing.patching import (
    PatchRegistry,
    PatchUninstallSummary,
    RegistrationResult,
    get_default_registry,
)
from .operators.selection import selected_backend

logger = logging.getLogger(__name__)

_REGISTERED = False
_REGISTRATION_RESULT: RegistrationResult | None = None
_REGISTRY: PatchRegistry | None = None


def register() -> RegistrationResult:
    """Register the plugin with vLLM.

    vLLM calls this function with no arguments from the
    ``vllm.general_plugins`` entry point. Missing operator implementations
    leave native operators intact and are reported as native fallback.
    """

    global _REGISTERED, _REGISTRATION_RESULT, _REGISTRY

    if _REGISTERED and _REGISTRATION_RESULT is not None:
        return _REGISTRATION_RESULT

    registry = get_default_registry()
    result = registry.register_from_environment()

    _REGISTERED = True
    _REGISTRATION_RESULT = result
    _REGISTRY = registry
    logger.info(
        "vllm-infinicore registered: operator_backend=%s routes=%d patching=%s installed=%s reason=%s",
        selected_backend() or "unset",
        result.route_count,
        "enabled" if result.patching_enabled else "disabled",
        ",".join(result.installed_routes) or "-",
        result.reason,
    )
    for state in result.route_states:
        if state.fallback_active:
            logger.info("vllm-infinicore %s: native_fallback (%s)", state.name, state.reason)
    return result


def unregister() -> PatchUninstallSummary:
    """Uninstall patches owned by this plugin and reset registration state."""

    global _REGISTERED, _REGISTRATION_RESULT, _REGISTRY

    installed_routes = (
        _REGISTRATION_RESULT.installed_routes
        if _REGISTRATION_RESULT is not None
        else ()
    )
    registry = _REGISTRY or get_default_registry()
    result = registry.uninstall_routes(installed_routes)
    _REGISTERED = False
    _REGISTRATION_RESULT = None
    _REGISTRY = None
    logger.info(
        "vllm-infinicore unregistered: uninstalled=%s skipped=%s reason=%s",
        ",".join(result.uninstalled_routes) or "-",
        ",".join(result.skipped_routes) or "-",
        result.failure_reason or "ok",
    )
    return result
