import torch
from torch import Tensor


def find_grid(
    weight: Tensor,
    num_bits: int,
) -> tuple[Tensor, Tensor]:
    weight_min = weight.amin(dim=-1).clamp(max=0)
    weight_max = weight.amax(dim=-1).clamp(min=0)
    scale = ((weight_max - weight_min) / (2**num_bits - 1)).clamp(min=1e-8)
    zero = torch.round(-weight_min / scale)
    return scale, zero


def round_to_grid(
    weight: Tensor,
    scale: Tensor,
    zero: Tensor,
    num_bits: int,
) -> Tensor:
    level = torch.clamp(torch.round(weight / scale) + zero, 0, 2**num_bits - 1)
    return (level - zero) * scale
