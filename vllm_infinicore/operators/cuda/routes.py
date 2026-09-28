"""NVIDIA CUDA vLLM operator routes."""

from ..routes import install_shared_route, uninstall_shared_route
from . import SUPPORTED_ROUTES


def install(name: str):
    if name not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported CUDA operator route: {name}")
    return install_shared_route(name)


def uninstall(name: str):
    if name not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported CUDA operator route: {name}")
    return uninstall_shared_route(name)
