"""Small vLLM runtime patches needed by the InfiniCore platform."""

from __future__ import annotations

import inspect
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

DUMMY_RUN_REAL_REQS_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_DUMMY_RUN_REAL_REQS"

_DUMMY_RUN_PATCHED = False
_DUMMY_RUN_PATCH_REASON = "not applied"


@dataclass(frozen=True)
class RuntimePatchStatus:
    applied: bool
    reason: str


def dummy_run_real_reqs_status() -> RuntimePatchStatus:
    return RuntimePatchStatus(
        applied=_DUMMY_RUN_PATCHED,
        reason=_DUMMY_RUN_PATCH_REASON,
    )


def patch_gpu_model_runner_dummy_run_real_reqs() -> RuntimePatchStatus:
    """Use real request count when dummy-run builds graph attention metadata.

    vLLM's upstream dummy graph capture passes ``num_reqs_padded`` into
    ``_build_attention_metadata``. On the MetaX runtime this captures larger
    padded metadata than the real decode shape and can force a much slower
    execution path. The MetaX platform patch changes just this argument back to
    ``num_reqs``; this local patch mirrors that behavior without importing
    ``vllm_metax``.
    """

    global _DUMMY_RUN_PATCHED, _DUMMY_RUN_PATCH_REASON

    if _env_truthy(DUMMY_RUN_REAL_REQS_DISABLE_ENV):
        _DUMMY_RUN_PATCH_REASON = f"{DUMMY_RUN_REAL_REQS_DISABLE_ENV} is true"
        return dummy_run_real_reqs_status()
    if _DUMMY_RUN_PATCHED:
        return dummy_run_real_reqs_status()

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:
        _DUMMY_RUN_PATCH_REASON = (
            f"GPUModelRunner unavailable: {type(exc).__name__}: {exc}"
        )
        return dummy_run_real_reqs_status()

    original = GPUModelRunner._dummy_run
    if getattr(original, "_vllm_infinicore_real_reqs_patch", False):
        _DUMMY_RUN_PATCHED = True
        _DUMMY_RUN_PATCH_REASON = "already patched"
        return dummy_run_real_reqs_status()

    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as exc:
        _DUMMY_RUN_PATCH_REASON = f"source unavailable: {type(exc).__name__}: {exc}"
        return dummy_run_real_reqs_status()

    patched_source = _dummy_run_real_reqs_source(source)
    if patched_source is None:
        if "num_reqs=num_reqs," in source:
            _DUMMY_RUN_PATCHED = True
            _DUMMY_RUN_PATCH_REASON = "source already uses real num_reqs"
        else:
            _DUMMY_RUN_PATCH_REASON = "expected num_reqs_padded call not found"
        return dummy_run_real_reqs_status()

    module = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    globals_dict = module.__dict__ if module is not None else original.__globals__
    namespace: dict[str, Any] = {}
    exec(patched_source, globals_dict, namespace)
    patched = namespace["_dummy_run"]
    patched.__doc__ = original.__doc__
    patched.__module__ = original.__module__
    patched._vllm_infinicore_real_reqs_patch = True
    patched._vllm_infinicore_original = original
    GPUModelRunner._dummy_run = patched

    _DUMMY_RUN_PATCHED = True
    _DUMMY_RUN_PATCH_REASON = "patched num_reqs_padded -> num_reqs"
    return dummy_run_real_reqs_status()


def _dummy_run_real_reqs_source(source: str) -> str | None:
    dedented = textwrap.dedent(source)
    target = "num_reqs=num_reqs_padded,"
    if target not in dedented:
        return None
    return dedented.replace(target, "num_reqs=num_reqs,", 1)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}
