"""vLLM InfiniCore plugin entry package."""

from .platform import register_platform
from .plugin import register, unregister

__all__ = ["register", "register_platform", "unregister"]
__version__ = "0.1.0"
