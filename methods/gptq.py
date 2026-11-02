from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module

from utils.capture import capture_layer_inputs
from utils.data import load_numina
from utils.format import format_duration, format_memory
from utils.hub import save_grid, save_quantized
from utils.model import load_model
from utils.paths import make_numina_path, make_quantized_path
from utils.quantization import find_grid, round_to_grid


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    block_size: int = 128
    damp_ratio: float = 0.01
    max_sequence_len: int = 2048
    # numina calibration path
    num_samples: int = 128
    max_generation_len: int = 2048
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
    # 1) zero dead columns, so hessian stays invertible
    is_dead = hessian.diagonal() == 0
    weight[:, is_dead] = 0
    hessian.diagonal()[is_dead] = 1
    # 2) damp diagonal, so inverse stays stable
    hessian.diagonal().add_(damp_ratio * hessian.diagonal().mean())
    hessian_inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)

    quantized = torch.zeros_like(weight)
    scale = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    zero = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    error = torch.zeros(weight.shape[0], block_size, device=weight.device)
    # 3) quantize block by block, column by column
    for start in range(0, weight.shape[1], block_size):
        end = min(start + block_size, weight.shape[1])
        for column in range(start, end):
            group = column // group_size
            # set grid when group starts, so it reflects compensation
            if column % group_size == 0:
                scale[:, group], zero[:, group] = find_grid(weight[:, column : column + group_size], num_bits)
            # 4) quantize column and compensate column's error to right columns in block
            quantized[:, column] = round_to_grid(weight[:, column], scale[:, group], zero[:, group], num_bits)
            error[:, column - start] = (weight[:, column] - quantized[:, column]) / hessian_inv[column, column]
            weight[:, column:end] -= torch.outer(error[:, column - start], hessian_inv[column, column:end])
        # 5) compensate block's error to remaining columns after block
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
