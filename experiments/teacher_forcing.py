from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer

from utils.dataset import load_math500, load_mmlu_redux
from utils.format import format_duration, format_memory
from utils.hub import download_model, load_result, result_exists, save_tensors
from utils.model import TEMPERATURE, TOP_K, TOP_P, get_prompt_ids, get_think_hiddens, get_think_ids, load_model
from utils.paths import make_quantized_path


@dataclass
class Config:
    seed: int = 0
    dataset: Literal["math500", "mmlu_redux"] = "math500"
    num_per_subject: int = 40
    chunk_len: int = 2048
    # model to compare with bf16
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["gptq", "awq", "mxfp4"] = "gptq"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def get_sampling_probs(
    logits: Tensor,
) -> Tensor:
    # actual sampling probs after temperature, top-k and top-p
    top_logits, top_ids = (logits / TEMPERATURE).topk(TOP_K, dim=-1)
    top_probs = top_logits.softmax(dim=-1)
    top_probs = torch.where(top_probs.cumsum(dim=-1) - top_probs < TOP_P, top_probs, 0)
    return torch.zeros_like(logits).scatter(-1, top_ids, top_probs / top_probs.sum(dim=-1, keepdim=True))


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    entry_name = f"teacher_forcing/{config.dataset}"
    quantized_path = make_quantized_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_samples)
    if result_exists(entry_name, quantized_path, file_name, "safetensors"):
        print(f"{quantized_path} already computed")
        return
    prompts = [prompt for prompt, _ in (load_math500() if config.dataset == "math500" else load_mmlu_redux(config.num_per_subject))]
    result = load_result(config.dataset, f"bf16/{config.model_name}", file_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bf16_model = load_model(config.model_name)
    quantized_model = load_model(download_model(quantized_path))

    # use only bf16 reasoning on correctly solved problems
    samples = []
    for prompt, think, correct in zip(prompts, result["thinks"], result["corrects"]):
        if correct:
            prompt_ids = get_prompt_ids(tokenizer, prompt)
            think_ids = get_think_ids(tokenizer, think)
            samples.append((prompt_ids, think_ids))
    print(f"samples {len(samples)} positions {sum(len(think_ids) for _, think_ids in samples)}")

    tensors = defaultdict(list)
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for sample_index, (prompt_ids, think_ids) in enumerate(samples):
        input_ids = torch.cat([prompt_ids, think_ids])
        bf16_hiddens = get_think_hiddens(bf16_model, input_ids, len(prompt_ids))
        quantized_hiddens = get_think_hiddens(quantized_model, input_ids, len(prompt_ids))
        for start in range(0, len(think_ids), config.chunk_len):
            bf16_logits = bf16_model.lm_head(bf16_hiddens[start : start + config.chunk_len]).float()
            bf16_logprobs = log_softmax(bf16_logits, dim=-1)
            bf16_probs = bf16_logprobs.exp()
            bf16_top1 = bf16_logprobs.argmax(dim=-1)
            quantized_logits = quantized_model.lm_head(quantized_hiddens[start : start + config.chunk_len]).float()
            quantized_logprobs = log_softmax(quantized_logits, dim=-1)

            tensors["bf16_entropy"].append(-(bf16_probs * bf16_logprobs).sum(dim=-1).cpu())
            tensors["kl"].append((bf16_probs * (bf16_logprobs - quantized_logprobs)).sum(dim=-1).cpu())
            tensors["bf16_top1"].append(bf16_top1.cpu())
            tensors["quantized_top1"].append(quantized_logprobs.argmax(dim=-1).cpu())
            tensors["bf16_prob"].append(bf16_probs.gather(-1, bf16_top1[:, None])[:, 0].cpu())
            # quantized model's prob of bf16 top1 token, not its own top1
            tensors["quantized_prob"].append(quantized_logprobs.gather(-1, bf16_top1[:, None])[:, 0].exp().cpu())
            # gap between actual sampling probs, so prob that sampled token differs
            tensors["flip_prob"].append(0.5 * (get_sampling_probs(bf16_logits) - get_sampling_probs(quantized_logits)).abs().sum(dim=-1).cpu())
            # how unevenly quantization shifts each logit, so std
            tensors["logit_std_error"].append((quantized_logits - bf16_logits).gather(-1, bf16_logits.topk(TOP_K, dim=-1).indices).std(dim=-1).cpu())
        tensors["think_ids"].append(think_ids)

        if (sample_index + 1) % 50 == 0:
            print(f"sample {sample_index + 1}/{len(samples)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
            log_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

    tensors = {name: torch.cat(values) for name, values in tensors.items()}
    print(f"{quantized_path} bf16 entropy {tensors['bf16_entropy'].mean():.4f} kl {tensors['kl'].mean():.4f} top1 mismatch {(tensors['bf16_top1'] != tensors['quantized_top1']).float().mean():.2%} flip prob {tensors['flip_prob'].mean():.2%} logit std error {tensors['logit_std_error'].mean():.4f}")
    save_tensors(entry_name, quantized_path, file_name, tensors)


if __name__ == "__main__":
    main()
