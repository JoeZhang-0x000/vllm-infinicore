"""vLLM InfiniCore plugin entry package."""
from .platform import InfiniCorePlatform, register_platform
from .plugin import register

__all__ = ["InfiniCorePlatform", "register", "register_platform"]
__version__ = "0.1.0"
