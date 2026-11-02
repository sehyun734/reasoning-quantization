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
    # detach only round, not whole quantized weight
    # so q = round(w/s) * s keeps gradient to scale, and scale can train (lsq)
    level = weight / scale
    level = level + (torch.round(level) - level).detach() + zero
    # torch.clamp gives zero gradient on end levels too (0 and 15 for 4 bits)
    # so weights on end levels can't move back inward
    # is_inside keeps gradient up to end levels, and cuts only out of range
    # e.g. 2 bits (level 0~3), scale 0.2, zero 1
    # weight              [-0.2, 0.0, 0.2, 0.4, 0.6]
    # level               [   0,   1,   2,   3,   4]
    # clamp gradient      [   0,   1,   1,   0,   0]
    # is_inside gradient  [   1,   1,   1,   1,   0]
    is_inside = (level >= 0) & (level <= 2**num_bits - 1)
    level = torch.where(is_inside, level, level.clamp(0, 2**num_bits - 1).detach())
    return (level - zero) * scale
