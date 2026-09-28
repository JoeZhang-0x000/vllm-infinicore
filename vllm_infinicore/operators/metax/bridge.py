"""MetaX C++ bridge configuration."""

from . import SUPPORTED_ROUTES

MODULE_NAME = "vllm_infinicore_metax_cpp_bridge"
EXTRA_CFLAGS = ("-DENABLE_METAX_API",)
DEFAULT_ROUTES = ("MatMul",)


def extra_include_paths() -> tuple[str, ...]:
    return ()
