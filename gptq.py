from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module

from primitives.capture import capture_layer_inputs
from primitives.dataset import load_numina
from primitives.hub import save_grid, save_quantized
from primitives.model import load_model
from primitives.paths import make_numina_path, make_quantized_path
from primitives.quantization import find_grid, round_to_grid
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    block_size: int = 128
    damp_ratio: float = 0.01
    # which numina data to use as calibration
    num_samples: int = 128
    max_generation_len: int = 2048
    max_sequence_len: int = 2048
    seed: int = 0


def compute_hessian(
    layer: Module,
    linear: Linear,
    inputs: list[tuple[tuple, dict]],
) -> Tensor:
    hessian = torch.zeros(linear.in_features, linear.in_features, device=linear.weight.device)

    def hook(
        module: Module,
        args: tuple[Tensor, ...],
    ) -> None:
        x = args[0].reshape(-1, args[0].shape[-1]).float()
        hessian.add_(x.T @ x)

    handle = linear.register_forward_pre_hook(hook)
    for args, kwargs in inputs:
        layer(*args, **kwargs)
    handle.remove()
    return hessian


def quantize_weight(
    weight: Tensor,
    hessian: Tensor,
    num_bits: int,
    group_size: int,
    block_size: int,
    damp_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    weight = weight.float().clone()
    # zero-input columns make hessian non-invertible, so zero weight and set diagonal to 1
    is_dead = hessian.diagonal() == 0
    weight[:, is_dead] = 0
    hessian.diagonal()[is_dead] = 1
    # add small damping to diagonal so inverse stays stable
    hessian.diagonal().add_(damp_ratio * hessian.diagonal().mean())
    hessian_inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)

    quantized = torch.zeros_like(weight)
    scale = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    zero = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    error = torch.zeros(weight.shape[0], block_size, device=weight.device)
    # quantize column by column, one block at a time
    for start in range(0, weight.shape[1], block_size):
        end = min(start + block_size, weight.shape[1])
        for column in range(start, end):
            group = column // group_size
            # set grid when group starts, so it sees already compensated weights
            if column % group_size == 0:
                scale[:, group], zero[:, group] = find_grid(weight[:, column : column + group_size], num_bits)
            # quantize this column
            quantized[:, column] = round_to_grid(weight[:, column], scale[:, group], zero[:, group], num_bits)
            # push its error to columns on the right inside block
            error[:, column - start] = (weight[:, column] - quantized[:, column]) / hessian_inv[column, column]
            weight[:, column:end] -= torch.outer(error[:, column - start], hessian_inv[column, column:end])
        # push whole block's error to columns after block at once
        weight[:, end:] -= error[:, : end - start] @ hessian_inv[start:end, end:]
    return quantized, scale, zero


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    quantized_path = make_quantized_path("gptq", config.model_name, config.num_bits, config.group_size, config.num_samples)
    model, _ = load_model(config.model_name, config.model_name)
    samples = [input_ids[: config.max_sequence_len] for input_ids in load_numina(make_numina_path(config.model_name, config.num_samples, config.max_generation_len, config.seed))]
    layers = model.model.layers
    grid = {}

    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(layers):
        inputs = capture_layer_inputs(model, layer, samples)
        for name, linear in layer.named_modules():
            if not isinstance(linear, Linear):
                continue
            hessian = compute_hessian(layer, linear, inputs)
            quantized, scale, zero = quantize_weight(linear.weight, hessian, config.num_bits, config.group_size, config.block_size, config.damp_ratio)
            linear.weight.copy_(quantized)
            grid[f"{layer_index}.{name}.scale"] = scale.cpu()
            grid[f"{layer_index}.{name}.zero"] = zero.cpu()

        print(f"layer {layer_index + 1}/{len(layers)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_quantized(quantized_path, model)
    save_grid(quantized_path, grid)


if __name__ == "__main__":
    main()
