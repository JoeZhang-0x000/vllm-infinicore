"""vLLM general plugin registration entry point."""

from __future__ import annotations

import logging

from .patching import RegistrationResult

logger = logging.getLogger(__name__)

_REGISTERED = False
_REGISTRATION_RESULT: RegistrationResult | None = None


def register() -> RegistrationResult:
    """Register the InfiniCore plugin with vLLM.

    Installs ALL operator routes automatically.  Set
    ``VLLM_INFINICORE_DISABLE=1`` to skip registration.
    """
    global _REGISTERED, _REGISTRATION_RESULT

    if _REGISTERED and _REGISTRATION_RESULT is not None:
        return _REGISTRATION_RESULT

    from .patching import install_all_routes

    result = install_all_routes()
    _REGISTERED = True
    _REGISTRATION_RESULT = result
    logger.info(
        "vllm-infinicore registered: installed=%s failed=%s",
        ",".join(result.installed_routes) or "-",
        ",".join(result.failed_routes) or "-",
    )
    return result
