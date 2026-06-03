"""vLLM InfiniCore plugin entry package."""

from .plugin import register, unregister
from .ray import configure_ray_environment

__all__ = ["configure_ray_environment", "register", "unregister"]
__version__ = "0.1.0"
