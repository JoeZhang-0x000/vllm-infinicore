"""Device helpers shared by the modern and legacy bridges."""

from typing import Any

import torch


def is_accelerator_tensor(tensor: torch.Tensor) -> bool:
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", "")
    return bool(getattr(tensor, "is_cuda", False)) or device_type == "cuda"


def torch_device_api(tensor: torch.Tensor) -> Any | None:
    device_type = getattr(getattr(tensor, "device", None), "type", "")
    if bool(getattr(tensor, "is_cuda", False)) or device_type == "cuda":
        return getattr(torch, "cuda", None)
    return None
