from dataclasses import dataclass
import math
import time

import numpy as np
from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module

from utils.capture import capture_layer_inputs
from utils.dataset import load_numina
from utils.format import format_duration, format_memory
from utils.hub import save_model
from utils.model import load_model
from utils.paths import make_numina_path, make_quantized_path
from utils.quantization import find_grid, round_to_grid


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_grid: int = 20
    max_shrink: float = 0.5
    # NOTE: saved awq model used uncut calibration, so results may differ very slightly
    max_sequence_len: int = 2048
    # numina calibration path
    num_samples: int = 128
    max_generation_len: int = 2048
    seed: int = 0


def find_linear_groups(
    layer: Module,
    args: tuple,
    kwargs: dict,
) -> list[list[Linear]]:
    linear_groups = []

    def hook(
        linear: Linear,
        hook_args: tuple[Tensor, ...],
    ) -> None:
        for x, linears in linear_groups:
            if x is hook_args[0]:
                linears.append(linear)
                return
        linear_groups.append((hook_args[0], [linear]))

    handles = [linear.register_forward_pre_hook(hook) for linear in layer.modules() if isinstance(linear, Linear)]
    layer(*args, **kwargs)
    for handle in handles:
        handle.remove()
    return [linears for _, linears in linear_groups]


def compute_statistics(
    layer: Module,
    linear: Linear,
    inputs: list[tuple[tuple, dict]],
) -> tuple[Tensor, Tensor]:
    abs_sum = torch.zeros(linear.in_features, device=linear.weight.device)
    hessian = torch.zeros(linear.in_features, linear.in_features, device=linear.weight.device)
    num_tokens = 0

    def hook(
        module: Module,
        args: tuple[Tensor, ...],
    ) -> None:
        nonlocal num_tokens
        x = args[0].reshape(-1, args[0].shape[-1]).float()
        abs_sum.add_(x.abs().sum(dim=0))
        hessian.add_(x.T @ x)
        num_tokens += x.shape[0]

    handle = linear.register_forward_pre_hook(hook)
    for args, kwargs in inputs:
        layer(*args, **kwargs)
    handle.remove()
    return abs_sum / num_tokens, hessian


def forward_layer(
    layer: Module,
    inputs: list[tuple[tuple, dict]],
) -> list[Tensor]:
    outputs = []
    for args, kwargs in inputs:
        output = layer(*args, **kwargs)
        outputs.append(output[0] if isinstance(output, tuple) else output)
    return outputs


def quantize_weight(
    weight: Tensor,
    smooth: Tensor,
    clip_value: Tensor | None,
    num_bits: int,
    group_size: int,
) -> Tensor:
    groups = (weight.float() * smooth).reshape(weight.shape[0], -1, group_size)
    clipped = groups if clip_value is None else groups.clamp(-clip_value[..., None], clip_value[..., None])
    scale, zero = find_grid(clipped, num_bits)
    return round_to_grid(groups, scale[..., None], zero[..., None], num_bits).reshape(weight.shape) / smooth


def find_smooth(
    layer: Module,
    linears: list[Linear],
    inputs: list[tuple[tuple, dict]],
    abs_mean: Tensor,
    num_bits: int,
    group_size: int,
    num_grid: int,
) -> Tensor:
    # compare whole layer output, not each sub-module, so code stays simple
    outputs = forward_layer(layer, inputs)
    weights = [linear.weight.clone() for linear in linears]

    best_error = math.inf
    for ratio in np.linspace(0.0, 1.0, num_grid, endpoint=False):
        smooth = abs_mean.pow(ratio).clamp(min=1e-4)
        smooth = smooth / (smooth.max() * smooth.min()).sqrt()
        for linear, weight in zip(linears, weights):
            linear.weight.copy_(quantize_weight(weight, smooth, None, num_bits, group_size))
        error = sum((output - quantized_output).float().pow(2).mean().item() for output, quantized_output in zip(outputs, forward_layer(layer, inputs)))
        if error < best_error:
            best_error, best_smooth = error, smooth

    for linear, weight in zip(linears, weights):
        linear.weight.copy_(weight)
    return best_smooth


def find_clip(
    weight: Tensor,
    smooth: Tensor,
    hessian: Tensor,
    num_bits: int,
    group_size: int,
    num_grid: int,
    max_shrink: float,
) -> Tensor:
    # outliers widen grid range and step size
    # so grid search clip for each quantization group
    # 1) apply smooth to weight (w * s) and hessian (x / s)
    groups = (weight.float() * smooth).reshape(weight.shape[0], -1, group_size)
    num_groups = groups.shape[1]
    smoothed_hessian = hessian / smooth[:, None] / smooth[None, :]
    # clip is picked per group, so keep only each group's diagonal block
    hessian_blocks = smoothed_hessian.reshape(num_groups, group_size, num_groups, group_size).diagonal(dim1=0, dim2=2).permute(2, 0, 1)

    # 2) grid search clip
    weight_max = groups.abs().amax(dim=-1)
    best_clip_value = weight_max.clone()
    best_error = torch.full_like(weight_max, math.inf)
    for shrink in np.linspace(1.0, 1.0 - max_shrink, int(max_shrink * num_grid), endpoint=False):
        clip_value = weight_max * shrink
        scale, zero = find_grid(groups.clamp(-clip_value[..., None], clip_value[..., None]), num_bits)
        quantized = round_to_grid(groups, scale[..., None], zero[..., None], num_bits)
        delta = quantized - groups
        # actually, other groups in row o also add to its error
        # but ignore them, so each group's clip can be searched alone
        error = torch.einsum("ogi,gij,ogj->og", delta, hessian_blocks, delta)
        # keep best clip for each quantization group
        is_better = error < best_error
        best_error = torch.where(is_better, error, best_error)
        best_clip_value = torch.where(is_better, clip_value, best_clip_value)
    return best_clip_value


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model, _ = load_model(config.model_name, config.model_name)
    samples = [input_ids[: config.max_sequence_len] for input_ids in load_numina(make_numina_path(config.model_name, config.num_samples, config.max_generation_len, config.seed))]
    layers = model.model.layers

    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(layers):
        inputs = capture_layer_inputs(model, layer, samples)
        for linears in find_linear_groups(layer, *inputs[0]):
            abs_mean, hessian = compute_statistics(layer, linears[0], inputs)
            smooth = find_smooth(layer, linears, inputs, abs_mean, config.num_bits, config.group_size, config.num_grid)
            for linear in linears:
                clip_value = find_clip(linear.weight, smooth, hessian, config.num_bits, config.group_size, config.num_grid, config.max_shrink)
                linear.weight.copy_(quantize_weight(linear.weight, smooth, clip_value, config.num_bits, config.group_size))

        print(f"layer {layer_index + 1}/{len(layers)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_model(make_quantized_path("awq", config.model_name, config.num_bits, config.group_size, config.num_samples), model)


if __name__ == "__main__":
    main()
