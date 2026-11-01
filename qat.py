from dataclasses import dataclass
import math
import random
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import linear, log_softmax
from torch.optim import AdamW
from transformers import PreTrainedModel, get_constant_schedule, get_cosine_schedule_with_warmup, set_seed

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
    num_train_samples: int = 2048
    hidden_weight: float = 0.0
    on_policy: bool = False
    scale_learning_rate: float = 0.0
    lr: float = 1e-4
    lr_schedule: Literal["constant", "cosine"] = "constant"
    warmup_ratio: float = 0.05
    num_epochs: int = 1
    batch_size: int = 8
    high_entropy_ratio: float = 0.2
    save: bool = True
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


@torch.no_grad()
def evaluate(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    samples: list[Tensor],
    high_entropy_ratio: float,
) -> tuple[float, float]:
    entropies = []
    kls = []
    for input_ids in samples:
        input_ids = input_ids[None].to(model.device)
        bf16_logprobs = log_softmax(bf16_model(input_ids=input_ids).logits[0, :-1].float(), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logprobs = log_softmax(model(input_ids=input_ids).logits[0, :-1].float(), dim=-1)
        bf16_probs = bf16_logprobs.exp()
        entropies.append(-(bf16_probs * bf16_logprobs).sum(dim=-1).cpu())
        kls.append((bf16_probs * (bf16_logprobs - logprobs)).sum(dim=-1).cpu())
    entropy = torch.cat(entropies)
    kl = torch.cat(kls)
    is_high = entropy > entropy.quantile(1 - high_entropy_ratio)
    return kl[~is_high].mean().item(), kl[is_high].mean().item()


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
    samples = [input_ids[: config.max_sequence_len] for input_ids in load_numina(numina_path)[: config.num_train_samples]]

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
    parameters = [{"params": [ste_linear.weight for ste_linear in ste_linears], "lr": config.lr}]
    if config.scale_learning_rate > 0:
        parameters.append({"params": [ste_linear.scale for ste_linear in ste_linears], "lr": config.scale_learning_rate})
    optimizer = AdamW(parameters, weight_decay=0)
    num_batches = math.ceil(len(samples) / config.batch_size)
    num_steps = config.num_epochs * num_batches
    if config.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, round(config.warmup_ratio * num_steps), num_steps)
    else:
        scheduler = get_constant_schedule(optimizer)

    low_kl, high_kl = evaluate(model, bf16_model, samples, config.high_entropy_ratio)
    print(f"before training low entropy kl {low_kl:.4f} high entropy kl {high_kl:.4f}")
    kls = []
    hidden_losses = []
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config.num_epochs):
        random.shuffle(samples)
        for batch_index, start in enumerate(range(0, len(samples), config.batch_size)):
            step = epoch * num_batches + batch_index + 1
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
            scheduler.step()
            optimizer.zero_grad()
            kls.append(batch_kl)
            hidden_losses.append(batch_hidden_loss)
            if step % 16 == 0:
                print(f"epoch {epoch + 1}/{config.num_epochs} batch {batch_index + 1}/{num_batches} kl {mean(kls):.4f} hidden_loss {mean(hidden_losses):.4f} lr {scheduler.get_last_lr()[0]:.2e} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                kls = []
                hidden_losses = []
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

    low_kl, high_kl = evaluate(model, bf16_model, samples, config.high_entropy_ratio)
    print(f"after training low entropy kl {low_kl:.4f} high entropy kl {high_kl:.4f}")
    if not config.save:
        return
    with torch.no_grad():
        for (name, module), ste_linear in zip(linears, ste_linears):
            module.weight.copy_(ste_linear.quantize())
            parent_name, child_name = name.rsplit(".", 1)
            setattr(model.model.layers.get_submodule(parent_name), child_name, module)
    model.to(torch.bfloat16)
    save_quantized(make_qat_path(config.model_name, config.num_bits, config.group_size, config.hidden_weight, config.on_policy, config.scale_learning_rate), model)


if __name__ == "__main__":
    main()
