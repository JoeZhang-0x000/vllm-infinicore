"""vLLM general plugin registration entry point."""

from __future__ import annotations

import logging

from .patching import PatchUninstallSummary, RegistrationResult, get_default_registry

logger = logging.getLogger(__name__)

_REGISTERED = False
_REGISTRATION_RESULT: RegistrationResult | None = None


def register() -> RegistrationResult:
    """Register the plugin with vLLM.

    vLLM calls this function with no arguments from the
    ``vllm.general_plugins`` entry point. The current skeleton deliberately
    records the Qwen3 routing scope without installing monkey patches.
    """

    global _REGISTERED, _REGISTRATION_RESULT

    if _REGISTERED and _REGISTRATION_RESULT is not None:
        return _REGISTRATION_RESULT

    registry = get_default_registry()
    result = registry.register_from_environment()

    _REGISTERED = True
    _REGISTRATION_RESULT = result
    logger.info(
        "vllm-infinicore registered: routes=%d patching=%s installed=%s reason=%s",
        result.route_count,
        "enabled" if result.patching_enabled else "disabled",
        ",".join(result.installed_routes) or "-",
        result.reason,
    )
    return result


def unregister() -> PatchUninstallSummary:
    """Uninstall patches owned by this plugin and reset registration state."""

    global _REGISTERED, _REGISTRATION_RESULT

    installed_routes = (
        _REGISTRATION_RESULT.installed_routes
        if _REGISTRATION_RESULT is not None
        else ()
    )
    registry = get_default_registry()
    result = registry.uninstall_routes(installed_routes)
    _REGISTERED = False
    _REGISTRATION_RESULT = None
    logger.info(
        "vllm-infinicore unregistered: uninstalled=%s skipped=%s reason=%s",
        ",".join(result.uninstalled_routes) or "-",
        ",".join(result.skipped_routes) or "-",
        result.failure_reason or "ok",
    )
    return result
