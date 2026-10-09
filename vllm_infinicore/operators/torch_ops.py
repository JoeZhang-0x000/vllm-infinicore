"""PyTorch implementations used for CPU tensors and native fallbacks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged = input_tensor + residual
    return rms_norm(merged, weight, eps), merged


def rms_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_float = input_tensor.float()
    variance = input_float.pow(2).mean(dim=-1, keepdim=True)
    output = input_float * torch.rsqrt(variance + float(eps))
    return output.to(dtype=input_tensor.dtype) * weight


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    d = input_tensor.shape[-1] // 2
    return F.silu(input_tensor[..., :d]) * input_tensor[..., d:]


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    positions = positions.flatten()
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    def apply_one(tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tensor.shape
        view = tensor.view(positions.shape[0], -1, head_size)
        rot = view[..., :rotary_dim]
        passthrough = view[..., rotary_dim:]
        cos_view = cos.unsqueeze(-2).to(rot.dtype)
        sin_view = sin.unsqueeze(-2).to(rot.dtype)
        if is_neox_style:
            first, second = torch.chunk(rot, 2, dim=-1)
            out_rot = torch.cat(
                (first * cos_view - second * sin_view, second * cos_view + first * sin_view),
                dim=-1,
            )
        else:
            first = rot[..., ::2]
            second = rot[..., 1::2]
            out_rot = torch.stack(
                (first * cos_view - second * sin_view, second * cos_view + first * sin_view),
                dim=-1,
            ).flatten(-2)
        return torch.cat((out_rot, passthrough), dim=-1).reshape(original_shape)

    return apply_one(query), apply_one(key) if key is not None else None
