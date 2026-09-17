"""Small vLLM runtime patches needed by the InfiniCore platform."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

DUMMY_RUN_REAL_REQS_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_DUMMY_RUN_REAL_REQS"
VLLM_020_COMPAT_DISABLE_ENV = "VLLM_INFINICORE_DISABLE_VLLM020_COMPAT"

_DUMMY_RUN_PATCHED = False
_DUMMY_RUN_PATCH_REASON = "not applied"
_SAFE_FAKE_PATCHED = False
_SAFE_FAKE_PATCH_REASON = "not applied"
_ACCELERATOR_MEMORY_PATCHED = False
_ACCELERATOR_MEMORY_PATCH_REASON = "not applied"
_FUSED_MOE_MONOLITHIC_PATCHED = False
_FUSED_MOE_MONOLITHIC_PATCH_REASON = "not applied"
_FUSED_MOE_INT4_PATCHED = False
_FUSED_MOE_INT4_PATCH_REASON = "not applied"
_METAX_OOT_MOE_PATCHED = False
_METAX_OOT_MOE_PATCH_REASON = "not applied"
_FUNCTORCH_AUTOGRAD_CACHE_PATCHED = False
_FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON = "not applied"


@dataclass(frozen=True)
class RuntimePatchStatus:
    applied: bool
    reason: str


@dataclass(frozen=True)
class Vllm020CompatStatus:
    """Summary for the vLLM 0.20 compatibility patch group."""

    safe_fake_registration: RuntimePatchStatus
    torch_accelerator_memory_api: RuntimePatchStatus
    fused_moe_is_monolithic: RuntimePatchStatus
    fused_moe_int4_w4a8: RuntimePatchStatus
    metax_oot_unquantized_moe_backend: RuntimePatchStatus
    functorch_autograd_cache_normalize_inputs: RuntimePatchStatus

    @property
    def applied(self) -> bool:
        return any(
            status.applied
            for status in (
                self.safe_fake_registration,
                self.torch_accelerator_memory_api,
                self.fused_moe_is_monolithic,
                self.fused_moe_int4_w4a8,
                self.metax_oot_unquantized_moe_backend,
                self.functorch_autograd_cache_normalize_inputs,
            )
        )


def dummy_run_real_reqs_status() -> RuntimePatchStatus:
    return RuntimePatchStatus(
        applied=_DUMMY_RUN_PATCHED,
        reason=_DUMMY_RUN_PATCH_REASON,
    )


def vllm_020_compat_status() -> Vllm020CompatStatus:
    return Vllm020CompatStatus(
        safe_fake_registration=RuntimePatchStatus(
            applied=_SAFE_FAKE_PATCHED,
            reason=_SAFE_FAKE_PATCH_REASON,
        ),
        torch_accelerator_memory_api=RuntimePatchStatus(
            applied=_ACCELERATOR_MEMORY_PATCHED,
            reason=_ACCELERATOR_MEMORY_PATCH_REASON,
        ),
        fused_moe_is_monolithic=RuntimePatchStatus(
            applied=_FUSED_MOE_MONOLITHIC_PATCHED,
            reason=_FUSED_MOE_MONOLITHIC_PATCH_REASON,
        ),
        fused_moe_int4_w4a8=RuntimePatchStatus(
            applied=_FUSED_MOE_INT4_PATCHED,
            reason=_FUSED_MOE_INT4_PATCH_REASON,
        ),
        metax_oot_unquantized_moe_backend=RuntimePatchStatus(
            applied=_METAX_OOT_MOE_PATCHED,
            reason=_METAX_OOT_MOE_PATCH_REASON,
        ),
        functorch_autograd_cache_normalize_inputs=RuntimePatchStatus(
            applied=_FUNCTORCH_AUTOGRAD_CACHE_PATCHED,
            reason=_FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON,
        ),
    )


def apply_vllm_020_compat_patches() -> Vllm020CompatStatus:
    """Apply narrow compatibility patches for vLLM 0.20 on MetaX/MACA.

    The function only patches modules that are already imported, except for
    vLLM fused-MoE submodules when the vLLM package is already present in the
    process. This keeps plugin dry registration lightweight while still fixing
    the worker/runtime paths reached by vLLM's plugin loader.
    """

    if _env_truthy(VLLM_020_COMPAT_DISABLE_ENV):
        return vllm_020_compat_status()

    patch_torch_library_register_fake_missing_ops()
    patch_torch_accelerator_memory_api()
    patch_functorch_autograd_cache_normalize_inputs()
    if "vllm" in sys.modules:
        patch_fused_moe_method_is_monolithic()
        patch_fused_moe_quant_config_use_int4_w4a8()
        patch_metax_oot_unquantized_moe_backend()
    return vllm_020_compat_status()


def patch_functorch_autograd_cache_normalize_inputs(
    torch_module: Any | None = None,
) -> RuntimePatchStatus:
    """Add a missing functorch config expected by vLLM graph compilation."""

    global _FUNCTORCH_AUTOGRAD_CACHE_PATCHED, _FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON

    torch = torch_module or sys.modules.get("torch")
    if torch is None:
        _FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON = "torch is not loaded"
        return vllm_020_compat_status().functorch_autograd_cache_normalize_inputs

    try:
        functorch_config = importlib.import_module("torch._functorch.config")
    except Exception as exc:
        _FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON = (
            f"torch._functorch.config unavailable: {type(exc).__name__}: {exc}"
        )
        return vllm_020_compat_status().functorch_autograd_cache_normalize_inputs

    config_entries = getattr(functorch_config, "_config", None)
    if not isinstance(config_entries, dict):
        _FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON = "torch._functorch.config._config unavailable"
        return vllm_020_compat_status().functorch_autograd_cache_normalize_inputs
    if "autograd_cache_normalize_inputs" in config_entries:
        _FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON = (
            "autograd_cache_normalize_inputs already present"
        )
        return vllm_020_compat_status().functorch_autograd_cache_normalize_inputs

    try:
        from torch.utils._config_module import _Config, _ConfigEntry
    except Exception as exc:
        _FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON = (
            f"torch config entry helpers unavailable: {type(exc).__name__}: {exc}"
        )
        return vllm_020_compat_status().functorch_autograd_cache_normalize_inputs

    config_entries["autograd_cache_normalize_inputs"] = _ConfigEntry(
        _Config(default=False, value_type=bool)
    )
    _FUNCTORCH_AUTOGRAD_CACHE_PATCHED = True
    _FUNCTORCH_AUTOGRAD_CACHE_PATCH_REASON = (
        "added torch._functorch.config.autograd_cache_normalize_inputs"
    )
    return vllm_020_compat_status().functorch_autograd_cache_normalize_inputs


def patch_torch_accelerator_memory_api(torch_module: Any | None = None) -> RuntimePatchStatus:
    """Mirror missing MetaX ``torch.accelerator`` memory methods from CUDA."""

    global _ACCELERATOR_MEMORY_PATCHED, _ACCELERATOR_MEMORY_PATCH_REASON

    torch = torch_module or sys.modules.get("torch")
    if torch is None:
        _ACCELERATOR_MEMORY_PATCH_REASON = "torch is not loaded"
        return vllm_020_compat_status().torch_accelerator_memory_api

    accelerator = getattr(torch, "accelerator", None)
    cuda = getattr(torch, "cuda", None)
    if accelerator is None or cuda is None:
        _ACCELERATOR_MEMORY_PATCH_REASON = "torch.accelerator or torch.cuda unavailable"
        return vllm_020_compat_status().torch_accelerator_memory_api

    method_names = (
        "empty_cache",
        "memory_stats",
        "memory_reserved",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "memory_allocated",
        "reset_accumulated_memory_stats",
    )
    installed: list[str] = []
    for name in method_names:
        if hasattr(accelerator, name) or not hasattr(cuda, name):
            continue
        setattr(accelerator, name, getattr(cuda, name))
        installed.append(name)

    if installed:
        _ACCELERATOR_MEMORY_PATCHED = True
        _ACCELERATOR_MEMORY_PATCH_REASON = (
            "mapped torch.accelerator memory API from torch.cuda: "
            + ", ".join(installed)
        )
    elif _ACCELERATOR_MEMORY_PATCHED:
        _ACCELERATOR_MEMORY_PATCH_REASON = "already patched"
    else:
        _ACCELERATOR_MEMORY_PATCH_REASON = "torch.accelerator memory API already present"
    return vllm_020_compat_status().torch_accelerator_memory_api


def patch_torch_library_register_fake_missing_ops(
    torch_module: Any | None = None,
) -> RuntimePatchStatus:
    """Skip fake registrations for missing vLLM empty-build C++ ops."""

    global _SAFE_FAKE_PATCHED, _SAFE_FAKE_PATCH_REASON

    torch = torch_module or sys.modules.get("torch")
    if torch is None:
        _SAFE_FAKE_PATCH_REASON = "torch is not loaded"
        return vllm_020_compat_status().safe_fake_registration

    torch_library = getattr(torch, "library", None)
    original = getattr(torch_library, "register_fake", None)
    if original is None:
        _SAFE_FAKE_PATCH_REASON = "torch.library.register_fake unavailable"
        return vllm_020_compat_status().safe_fake_registration
    if getattr(original, "_vllm_infinicore_safe_fake_patch", False):
        _SAFE_FAKE_PATCHED = True
        _SAFE_FAKE_PATCH_REASON = "already patched"
        return vllm_020_compat_status().safe_fake_registration

    def safe_register_fake(op: Any, func: Any | None = None, /, *args: Any, **kwargs: Any) -> Any:
        identity_decorator = _identity_fake_registration(func)
        try:
            if func is None:
                decorator = original(op, *args, **kwargs)
            else:
                return original(op, func, *args, **kwargs)
        except RuntimeError as exc:
            if _is_missing_fake_op_error(exc):
                return identity_decorator
            raise

        def decorate(fake_func: Any) -> Any:
            try:
                return decorator(fake_func)
            except RuntimeError as exc:
                if _is_missing_fake_op_error(exc):
                    return fake_func
                raise

        return decorate

    safe_register_fake._vllm_infinicore_safe_fake_patch = True  # type: ignore[attr-defined]
    safe_register_fake._vllm_infinicore_original = original  # type: ignore[attr-defined]
    torch_library.register_fake = safe_register_fake
    _SAFE_FAKE_PATCHED = True
    _SAFE_FAKE_PATCH_REASON = "wrapped torch.library.register_fake"
    return vllm_020_compat_status().safe_fake_registration


def patch_fused_moe_method_is_monolithic() -> RuntimePatchStatus:
    """Handle vLLM 0.20 MoE methods with ``experts_cls = None``."""

    global _FUSED_MOE_MONOLITHIC_PATCHED, _FUSED_MOE_MONOLITHIC_PATCH_REASON

    if _FUSED_MOE_MONOLITHIC_PATCHED:
        return vllm_020_compat_status().fused_moe_is_monolithic

    try:
        module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
        )
    except Exception as exc:
        _FUSED_MOE_MONOLITHIC_PATCH_REASON = (
            f"fused_moe_method_base unavailable: {type(exc).__name__}: {exc}"
        )
        return vllm_020_compat_status().fused_moe_is_monolithic

    method_base = getattr(module, "FusedMoEMethodBase", None)
    if method_base is None:
        _FUSED_MOE_MONOLITHIC_PATCH_REASON = "FusedMoEMethodBase unavailable"
        return vllm_020_compat_status().fused_moe_is_monolithic

    current = getattr(method_base, "is_monolithic", None)
    current_getter = getattr(current, "fget", None)
    if getattr(current_getter, "_vllm_infinicore_experts_none_patch", False):
        _FUSED_MOE_MONOLITHIC_PATCHED = True
        _FUSED_MOE_MONOLITHIC_PATCH_REASON = "already patched"
        return vllm_020_compat_status().fused_moe_is_monolithic

    def is_monolithic(self: Any) -> bool:
        moe_kernel = getattr(self, "moe_kernel", None)
        if moe_kernel is None:
            experts_cls = getattr(self, "experts_cls", None)
            if experts_cls is not None:
                return bool(experts_cls.is_monolithic())
            return False
        return bool(moe_kernel.is_monolithic)

    is_monolithic._vllm_infinicore_experts_none_patch = True  # type: ignore[attr-defined]
    method_base.is_monolithic = property(is_monolithic)
    _FUSED_MOE_MONOLITHIC_PATCHED = True
    _FUSED_MOE_MONOLITHIC_PATCH_REASON = "patched experts_cls None handling"
    return vllm_020_compat_status().fused_moe_is_monolithic


def patch_fused_moe_quant_config_use_int4_w4a8() -> RuntimePatchStatus:
    """Add the MetaX-expected ``use_int4_w4a8`` property when vLLM lacks it."""

    global _FUSED_MOE_INT4_PATCHED, _FUSED_MOE_INT4_PATCH_REASON

    if _FUSED_MOE_INT4_PATCHED:
        return vllm_020_compat_status().fused_moe_int4_w4a8

    try:
        module = importlib.import_module("vllm.model_executor.layers.fused_moe.config")
    except Exception as exc:
        _FUSED_MOE_INT4_PATCH_REASON = (
            f"fused_moe.config unavailable: {type(exc).__name__}: {exc}"
        )
        return vllm_020_compat_status().fused_moe_int4_w4a8

    config_cls = getattr(module, "FusedMoEQuantConfig", None)
    if config_cls is None:
        _FUSED_MOE_INT4_PATCH_REASON = "FusedMoEQuantConfig unavailable"
        return vllm_020_compat_status().fused_moe_int4_w4a8
    if hasattr(config_cls, "use_int4_w4a8"):
        _FUSED_MOE_INT4_PATCHED = True
        _FUSED_MOE_INT4_PATCH_REASON = "use_int4_w4a8 already present"
        return vllm_020_compat_status().fused_moe_int4_w4a8

    def use_int4_w4a8(self: Any) -> bool:
        return getattr(getattr(self, "_a1", None), "dtype", None) == "int8" and (
            getattr(getattr(self, "_w1", None), "dtype", None) == "int4"
        )

    config_cls.use_int4_w4a8 = property(use_int4_w4a8)
    _FUSED_MOE_INT4_PATCHED = True
    _FUSED_MOE_INT4_PATCH_REASON = "added FusedMoEQuantConfig.use_int4_w4a8"
    return vllm_020_compat_status().fused_moe_int4_w4a8


def patch_metax_oot_unquantized_moe_backend() -> RuntimePatchStatus:
    """Use MetaX Triton experts for vLLM 0.20 unquantized MoE on OOT MetaX."""

    global _METAX_OOT_MOE_PATCHED, _METAX_OOT_MOE_PATCH_REASON

    if _METAX_OOT_MOE_PATCHED:
        return vllm_020_compat_status().metax_oot_unquantized_moe_backend

    try:
        module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized"
        )
    except Exception as exc:
        _METAX_OOT_MOE_PATCH_REASON = (
            f"fused_moe.oracle.unquantized unavailable: {type(exc).__name__}: {exc}"
        )
        return vllm_020_compat_status().metax_oot_unquantized_moe_backend

    original = getattr(module, "select_unquantized_moe_backend", None)
    if original is None:
        _METAX_OOT_MOE_PATCH_REASON = "select_unquantized_moe_backend unavailable"
        return vllm_020_compat_status().metax_oot_unquantized_moe_backend
    if getattr(original, "_vllm_infinicore_metax_oot_moe_patch", False):
        _METAX_OOT_MOE_PATCHED = True
        _METAX_OOT_MOE_PATCH_REASON = "already patched"
        return vllm_020_compat_status().metax_oot_unquantized_moe_backend

    def select_unquantized_moe_backend(moe_config: Any) -> Any:
        current_platform = getattr(module, "current_platform", None)
        if (
            current_platform is not None
            and current_platform.is_out_of_tree()
            and _metax_plugin_available_for_moe()
        ):
            try:
                from vllm_metax.utils.fused_moe import get_triton_experts_cls

                return module.UnquantizedMoeBackend.TRITON, get_triton_experts_cls()
            except Exception:
                return original(moe_config)
        return original(moe_config)

    select_unquantized_moe_backend._vllm_infinicore_metax_oot_moe_patch = True  # type: ignore[attr-defined]
    select_unquantized_moe_backend._vllm_infinicore_original = original  # type: ignore[attr-defined]
    module.select_unquantized_moe_backend = select_unquantized_moe_backend
    _METAX_OOT_MOE_PATCHED = True
    _METAX_OOT_MOE_PATCH_REASON = "patched OOT unquantized MoE backend selection"
    return vllm_020_compat_status().metax_oot_unquantized_moe_backend


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


def _identity_fake_registration(func: Any | None) -> Any:
    if func is not None:
        return func

    def decorate(fake_func: Any) -> Any:
        return fake_func

    return decorate


def _is_missing_fake_op_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "does not exist" in message and (
        "operator" in message or "OpOverload" in message
    )


def _metax_plugin_available_for_moe() -> bool:
    plugins = {
        plugin.strip()
        for plugin in os.environ.get("VLLM_PLUGINS", "").split(",")
        if plugin.strip()
    }
    if "metax" in plugins:
        return True
    return any(
        name == "vllm_metax" or name.startswith("vllm_metax.")
        for name in sys.modules
    )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}
