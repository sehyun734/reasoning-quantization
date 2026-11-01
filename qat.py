from dataclasses import dataclass
import math
import random
from statistics import mean
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import linear, log_softmax
from torch.optim import AdamW
from transformers import set_seed

from primitives.dataset import load_numina
from primitives.hub import download_model, load_grid, save_quantized
from primitives.model import load_model
from primitives.paths import make_numina_path, make_qat_path, make_quantized_path
from primitives.quantization import round_to_grid
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    # which numina data to train on
    qat_num_samples: int = 2048
    max_generation_len: int = 8192
    # cut prompt + reasoning to this length
    max_sequence_len: int = 8192
    hidden_weight: float = 1.0
    on_policy: bool = False
    scale_learning_rate: float = 0.0
    learning_rate: float = 1e-4
    batch_size: int = 8
    seed: int = 0


class STELinear(Module):
    def __init__(
        self,
        linear: Linear,
        scale: Tensor,
        zero: Tensor,
        num_bits: int,
        group_size: int,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.scale = scale
        self.zero = zero
        self.num_bits = num_bits
        self.group_size = group_size

    def quantize(
        self,
    ) -> Tensor:
        groups = self.weight.reshape(self.weight.shape[0], -1, self.group_size)
        quantized = round_to_grid(groups, self.scale[..., None], self.zero[..., None], self.num_bits)
        return quantized.reshape(self.weight.shape)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return linear(x, self.quantize(), self.bias)


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    quantized_path = make_quantized_path("gptq", config.model_name, config.num_bits, config.group_size, config.num_samples)
    bf16_model, _ = load_model(config.model_name, config.model_name)
    model, _ = load_model(download_model(quantized_path), config.model_name)
    model.float()
    if config.on_policy:
        numina_path = make_numina_path(quantized_path, config.qat_num_samples, config.max_generation_len, config.seed)
    else:
        numina_path = make_numina_path(config.model_name, config.qat_num_samples, config.max_generation_len, config.seed)
    samples = [input_ids[: config.max_sequence_len] for input_ids in load_numina(numina_path)]

    grid = load_grid(quantized_path)
    linears = [(name, module) for name, module in model.model.layers.named_modules() if isinstance(module, Linear)]
    # swap every linear for ste linear
    ste_linears = []
    for name, module in linears:
        ste_linear = STELinear(module, grid[f"{name}.scale"].to(model.device), grid[f"{name}.zero"].to(model.device), config.num_bits, config.group_size)
        ste_linear.weight.requires_grad_(True)
        ste_linear.scale.requires_grad_(config.scale_learning_rate > 0)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model.model.layers.get_submodule(parent_name), child_name, ste_linear)
        ste_linears.append(ste_linear)
    parameters = [{"params": [ste_linear.weight for ste_linear in ste_linears], "lr": config.learning_rate}]
    if config.scale_learning_rate > 0:
        parameters.append({"params": [ste_linear.scale for ste_linear in ste_linears], "lr": config.scale_learning_rate})
    optimizer = AdamW(parameters, weight_decay=0)

    num_batches = math.ceil(len(samples) / config.batch_size)
    kls = []
    hidden_losses = []
    random.shuffle(samples)
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, start in enumerate(range(0, len(samples), config.batch_size)):
        batch = samples[start : start + config.batch_size]
        num_positions = sum(len(input_ids) - 1 for input_ids in batch)
        batch_kl = 0.0
        batch_hidden_loss = 0.0

        for input_ids in batch:
            input_ids = input_ids[None].to(model.device)
            with torch.no_grad():
                bf16_hiddens = bf16_model.model(input_ids=input_ids).last_hidden_state[0, :-1]
                bf16_logprobs = log_softmax(bf16_model.lm_head(bf16_hiddens).float(), dim=-1)
                bf16_probs = bf16_logprobs.exp()
                entropy = -(bf16_probs * bf16_logprobs).sum(dim=-1)
                del bf16_logprobs
                bf16_hiddens = bf16_hiddens.float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hiddens = model.model(input_ids=input_ids).last_hidden_state[0, :-1]
                logprobs = log_softmax(model.lm_head(hiddens).float(), dim=-1)
            # kl to bf16 distribution
            kl = -entropy - (bf16_probs * logprobs).sum(dim=-1)
            # hidden error to bf16 weighted by entropy (higher entropy, more weight)
            hidden_loss = entropy * ((hiddens.float() - bf16_hiddens) ** 2).sum(dim=-1) / (bf16_hiddens**2).sum(dim=-1)
            loss = (kl.sum() + config.hidden_weight * hidden_loss.sum()) / num_positions
            loss.backward()
            batch_kl += kl.sum().item() / num_positions
            batch_hidden_loss += hidden_loss.sum().item() / num_positions

        optimizer.step()
        optimizer.zero_grad()
        kls.append(batch_kl)
        hidden_losses.append(batch_hidden_loss)
        if (step + 1) % 16 == 0:
            print(f"batch {step + 1}/{num_batches} kl {mean(kls):.4f} hidden_loss {mean(hidden_losses):.4f} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
            kls = []
            hidden_losses = []
            log_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for (name, module), ste_linear in zip(linears, ste_linears):
            module.weight.copy_(ste_linear.quantize())
            parent_name, child_name = name.rsplit(".", 1)
            setattr(model.model.layers.get_submodule(parent_name), child_name, module)
    model.to(torch.bfloat16)
    save_quantized(make_qat_path(config.model_name, config.num_bits, config.group_size, config.hidden_weight, config.on_policy, config.scale_learning_rate), model)


if __name__ == "__main__":
    main()
