"""vLLM general plugin registration entry point."""

from __future__ import annotations

import logging

from .patching import RegistrationResult, get_default_registry

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
        "vllm-infinicore registered: routes=%d patching=%s reason=%s",
        result.route_count,
        "enabled" if result.patching_enabled else "disabled",
        result.reason,
    )
    return result
