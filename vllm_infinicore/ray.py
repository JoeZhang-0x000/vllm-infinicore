"""Ray worker environment support for vLLM-InfiniCore."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
RAY_BACKEND_ENV = "VLLM_INFINICORE_RAY_BACKEND"

PLUGIN_ENV_VARS: tuple[str, ...] = (
    "VLLM_INFINICORE_ENABLE_PATCHES",
    "VLLM_INFINICORE_ROUTES",
    "VLLM_INFINICORE_DISABLED_ROUTES",
    "VLLM_INFINICORE_FORCE_NATIVE_FALLBACK",
    "VLLM_INFINICORE_ENABLE_CUSTOM_OPS",
    "VLLM_INFINICORE_STRICT_BACKEND",
    "VLLM_INFINICORE_DISABLE_REAL_BACKEND",
    "VLLM_INFINICORE_ENABLE_CPP_BRIDGE",
    "VLLM_INFINICORE_CPP_BRIDGE_ROUTES",
    "VLLM_INFINICORE_CPP_BRIDGE_VERBOSE",
    "VLLM_INFINICORE_RAY_BACKEND",
    "VLLM_INFINICORE_DISABLE_RAY_STORE_KV_CACHE",
)

RUNTIME_ENV_VARS: tuple[str, ...] = (
    "VLLM_PLUGINS",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "MACA_PATH",
    "MACA_HOME",
    "MACA_ROOT",
    "INFINI_ROOT",
    "PYTHON_SITE_PACKAGES",
    "TORCH_LIB",
    "FLASH_ATTN_2_CUDA_SO",
    "LD_LIBRARY_PATH",
    "PATH",
)


@dataclass(frozen=True)
class RayEnvironmentStatus:
    """Summary of Ray-specific environment setup."""

    enabled: bool
    noset_cuda_visible_devices: str
    registered_env_vars: tuple[str, ...]
    reason: str


def configure_ray_environment(
    *,
    distributed_executor_backend: str | None = None,
    extra_env_vars: Iterable[str] = (),
) -> RayEnvironmentStatus:
    """Propagate plugin environment variables to vLLM Ray workers.

    vLLM captures worker-visible environment variables through
    ``vllm.envs.environment_variables``. Registering the plugin and runtime
    variables here keeps Ray tensor-parallel workers on the same route profile
    as the driver process.
    """

    if distributed_executor_backend != "ray":
        return RayEnvironmentStatus(
            enabled=False,
            noset_cuda_visible_devices=os.environ.get(
                RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV, ""
            ),
            registered_env_vars=(),
            reason="distributed executor backend is not ray",
        )

    os.environ.setdefault(RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV, "1")
    os.environ[RAY_BACKEND_ENV] = "1"
    names = _ordered_unique((*PLUGIN_ENV_VARS, *RUNTIME_ENV_VARS, *extra_env_vars))
    registered = _register_vllm_env_vars(names)
    return RayEnvironmentStatus(
        enabled=True,
        noset_cuda_visible_devices=os.environ.get(
            RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV, ""
        ),
        registered_env_vars=registered,
        reason="ray worker environment configured",
    )


def _register_vllm_env_vars(names: tuple[str, ...]) -> tuple[str, ...]:
    try:
        import vllm.envs as vllm_envs
    except Exception:
        return ()

    environment_variables = getattr(vllm_envs, "environment_variables", None)
    if not isinstance(environment_variables, dict):
        return ()

    for name in names:
        environment_variables.setdefault(name, lambda name=name: os.environ.get(name))
    return names


def _ordered_unique(names: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return tuple(result)


__all__ = [
    "PLUGIN_ENV_VARS",
    "RAY_BACKEND_ENV",
    "RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV",
    "RUNTIME_ENV_VARS",
    "RayEnvironmentStatus",
    "configure_ray_environment",
]
