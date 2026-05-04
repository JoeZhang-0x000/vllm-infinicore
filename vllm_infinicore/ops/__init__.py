"""Interfaces for future InfiniCore-backed PyTorch custom ops."""

from .custom_ops import (
    CUSTOM_OP_ENABLE_ENV,
    RMS_NORM_OP,
    CustomOpStatus,
    is_available,
    load_custom_ops,
    rms_norm,
)

__all__ = [
    "CUSTOM_OP_ENABLE_ENV",
    "RMS_NORM_OP",
    "CustomOpStatus",
    "is_available",
    "load_custom_ops",
    "rms_norm",
]
