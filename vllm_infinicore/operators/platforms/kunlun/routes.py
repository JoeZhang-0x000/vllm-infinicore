"""Kunlun vLLM operator routes."""

from ....routing.routes import install_shared_route, uninstall_shared_route
from . import SUPPORTED_ROUTES


def install(name: str):
    from ....routing.routes import attention

    if name in attention.ROUTES:
        return attention.install(name, "kunlun")
    if name not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported Kunlun operator route: {name}")
    return install_shared_route(name)


def uninstall(name: str):
    from ....routing.routes import attention

    if name in attention.ROUTES:
        return attention.uninstall(name, "kunlun")
    if name not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported Kunlun operator route: {name}")
    return uninstall_shared_route(name)
