"""NVIDIA CUDA C++ bridge configuration."""

from . import SUPPORTED_ROUTES

MODULE_NAME = "vllm_infinicore_cuda_cpp_bridge"
EXTRA_CFLAGS = ("-DENABLE_NVIDIA_API",)
DEFAULT_ROUTES: tuple[str, ...] = ()


def extra_include_paths() -> tuple[str, ...]:
    return ()
