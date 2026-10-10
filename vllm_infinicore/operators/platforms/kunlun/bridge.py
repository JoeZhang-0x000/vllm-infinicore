"""Kunlun C++ bridge configuration."""

import os
from pathlib import Path

from . import SUPPORTED_ROUTES as SUPPORTED_ROUTES

MODULE_NAME = "vllm_infinicore_kunlun_cpp_bridge"
EXTRA_CFLAGS = ("-DENABLE_CUDA_API", "-DENABLE_KUNLUN_API")
DEFAULT_ROUTES: tuple[str, ...] = ()


def extra_include_paths() -> tuple[str, ...]:
    return (str(Path(os.environ.get("XPU_HOME", "/usr/local/xpu")) / "include"),)
