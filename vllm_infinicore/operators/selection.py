"""Explicit InfiniCore operator implementation selection."""

from __future__ import annotations

import os

OPERATOR_BACKEND_ENV = "VLLM_INFINICORE_OPERATOR_BACKEND"
OPERATOR_BACKENDS = frozenset({"ascend", "cuda", "kunlun", "metax"})


def selected_backend() -> str | None:
    """Return the requested operator implementation without probing vLLM."""

    value = os.environ.get(OPERATOR_BACKEND_ENV, "").strip().lower()
    if not value:
        return None
    if value not in OPERATOR_BACKENDS:
        choices = ", ".join(sorted(OPERATOR_BACKENDS))
        raise ValueError(f"{OPERATOR_BACKEND_ENV} must be one of {choices}")
    return value
