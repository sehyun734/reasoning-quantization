from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from transformers import PreTrainedModel

from primitives.dataset import load_math500
from primitives.hub import download_model, load_result, result_exists, save_tensors
from primitives.model import load_model
from primitives.paths import make_qat_path, make_quantized_path
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["gptq", "awq", "mxfp4", "qat"] = "gptq"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    hidden_weight: float = 1.0
    on_policy: bool = False
    scale_learning_rate: float = 0.0
    seed: int = 0
    chunk_len: int = 2048


def get_think_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    return model.model(input_ids=input_ids[None].to(model.device)).last_hidden_state[0, prompt_len - 1 : -1]


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    if config.method == "qat":
        quantized_path = make_qat_path(config.model_name, config.num_bits, config.group_size, config.hidden_weight, config.on_policy, config.scale_learning_rate)
    else:
        quantized_path = make_quantized_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_samples)
    if result_exists("teacher_forcing", quantized_path, file_name, "safetensors"):
        print(f"{quantized_path} already computed")
        return
    bf16_model, tokenizer = load_model(config.model_name, config.model_name)
    model, _ = load_model(download_model(quantized_path), config.model_name)
    result = load_result("math500", f"bf16/{config.model_name}", file_name)

    # use only bf16 reasoning on problems which it answers correctly
    samples = []
    for (prompt, _), think, correct in zip(load_math500(), result["thinks"], result["corrects"]):
        if correct:
            prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt")["input_ids"][0]
            think_ids = tokenizer(think, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            samples.append((prompt_ids, think_ids))
    print(f"samples {len(samples)} positions {sum(len(think_ids) for _, think_ids in samples)}")

    tensors = defaultdict(list)
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, (prompt_ids, think_ids) in enumerate(samples):
        input_ids = torch.cat([prompt_ids, think_ids])
        bf16_hiddens = get_think_hiddens(bf16_model, input_ids, len(prompt_ids))
        hiddens = get_think_hiddens(model, input_ids, len(prompt_ids))
        for start in range(0, len(think_ids), config.chunk_len):
            bf16_logprobs = log_softmax(bf16_model.lm_head(bf16_hiddens[start : start + config.chunk_len]).float(), dim=-1)
            logprobs = log_softmax(model.lm_head(hiddens[start : start + config.chunk_len]).float(), dim=-1)
            bf16_probs = bf16_logprobs.exp()
            bf16_top1 = bf16_logprobs.argmax(dim=-1)
            tensors["entropy"].append(-(bf16_probs * bf16_logprobs).sum(dim=-1).cpu())
            tensors["kl"].append((bf16_probs * (bf16_logprobs - logprobs)).sum(dim=-1).cpu())
            # top1 token id of each model
            tensors["bf16_top1"].append(bf16_top1.cpu())
            tensors["top1"].append(logprobs.argmax(dim=-1).cpu())
            tensors["bf16_prob"].append(bf16_probs.gather(-1, bf16_top1[:, None])[:, 0].cpu())
            # probability quantized model gives to bf16 top1 token, not its own top1
            tensors["prob"].append(logprobs.gather(-1, bf16_top1[:, None])[:, 0].exp().cpu())
        tensors["think_ids"].append(think_ids)

        if (step + 1) % 50 == 0:
            print(f"sample {step + 1}/{len(samples)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
            log_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

    tensors = {name: torch.cat(values) for name, values in tensors.items()}
    print(f"{quantized_path} entropy {tensors['entropy'].mean():.4f} kl {tensors['kl'].mean():.4f} top1 mismatch {(tensors['bf16_top1'] != tensors['top1']).float().mean():.2%}")
    save_tensors("teacher_forcing", quantized_path, file_name, tensors)


if __name__ == "__main__":
    main()
