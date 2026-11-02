from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear

from utils.format import format_duration, format_memory
from utils.hub import save_model
from utils.model import load_model
from utils.paths import make_mxfp4_path

GROUP_SIZE = 32


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"


def find_scale(
    weight: Tensor,
) -> Tensor:
    weight_max = weight.abs().amax(dim=-1).clamp(min=2**-125)
    return 2 ** (torch.floor(torch.log2(weight_max)) - 2)


def round_to_fp4(
    weight: Tensor,
    scale: Tensor,
) -> Tensor:
    magnitude = (weight.abs() / scale).clamp(max=6)
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    return torch.sign(weight) * torch.round(magnitude / step) * step * scale


def quantize_weight(
    weight: Tensor,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, GROUP_SIZE)
    return round_to_fp4(groups, find_scale(groups)[..., None]).reshape(weight.shape)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model, _ = load_model(config.model_name, config.model_name)
    layers = model.model.layers

    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(layers):
        for linear in layer.modules():
            if not isinstance(linear, Linear):
                continue
            linear.weight.copy_(quantize_weight(linear.weight))

        print(f"layer {layer_index + 1}/{len(layers)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_model(make_mxfp4_path(config.model_name), model)


if __name__ == "__main__":
    main()
