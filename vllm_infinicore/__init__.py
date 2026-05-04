"""vLLM InfiniCore plugin entry package."""

from .plugin import register, unregister

__all__ = ["register", "unregister"]
__version__ = "0.1.0"
