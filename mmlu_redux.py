from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse

from primitives.dataset import load_mmlu_redux
from primitives.evaluation import evaluate_mmlu_redux
from primitives.hub import download_model, result_exists, save_result
from primitives.model import THINK_END, generate_vllm, load_vllm
from primitives.paths import make_qat_path, make_target_path
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["bf16", "gptq", "awq", "mxfp4", "qat"] = "bf16"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    qat_num_samples: int = 2048
    on_policy: bool = False
    high_entropy_weight: float = 0.0
    hidden_weight: float = 0.0
    loss: Literal["kl", "ce", "cakld"] = "kl"
    scale_lr: float = 0.0
    num_epochs: int = 1
    lr_schedule: Literal["constant", "cosine"] = "constant"
    # smallest subject (virology) has 43 error-free problems, so take 40 from each
    # and keep dataset small to save generation time
    num_per_subject: int = 40
    max_generation_len: int = 32768


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    if config.method == "qat":
        target = make_qat_path(config.model_name, config.num_bits, config.group_size, config.qat_num_samples, config.loss, config.hidden_weight, config.on_policy, config.scale_lr, config.lr_schedule, config.num_epochs, config.high_entropy_weight)
    else:
        target = make_target_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_samples)
    if result_exists("mmlu_redux", target, file_name):
        print(f"{target} already evaluated")
        return
    samples = load_mmlu_redux(config.num_per_subject, config.seed)

    model_path = config.model_name if config.method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name, config.seed)
    tokenizer = model.get_tokenizer()
    think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)

    log_time = time.perf_counter()
    outputs = generate_vllm(model, [prompt for prompt, _ in samples], config.max_generation_len)
    result = evaluate_mmlu_redux(outputs, [answer for _, answer in samples], tokenizer, think_end_id)
    print(f"{target} accuracy {result['accuracy']:.2%} think_len {mean(result['think_lens']):.0f} truncated {mean(result['truncateds']):.2%} time {format_duration(time.perf_counter() - log_time)}")
    save_result("mmlu_redux", target, file_name, result)


if __name__ == "__main__":
    main()
