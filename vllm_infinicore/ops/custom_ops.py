"""Placeholder loader for future InfiniCore C++/PyTorch extension ops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CustomOpStatus:
    available: bool
    reason: str


def load_custom_ops() -> CustomOpStatus:
    """Return extension availability without importing torch in the skeleton."""

    return CustomOpStatus(
        available=False,
        reason="InfiniCore C++ custom op extension is not implemented yet",
    )


def is_available() -> bool:
    return load_custom_ops().available
