from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse

from utils.dataset import load_mmlu_redux
from utils.evaluation import evaluate_mmlu_redux
from utils.format import format_duration
from utils.hub import download_model, result_exists, save_result
from utils.model import generate_vllm, load_vllm
from utils.paths import make_target_path


@dataclass
class Config:
    seed: int = 0
    num_per_subject: int = 40
    max_generation_len: int = 32768
    # model to evaluate
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["bf16", "gptq", "awq", "mxfp4"] = "bf16"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    target = make_target_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_samples)
    if result_exists("mmlu_redux", target, file_name):
        print(f"{target} already evaluated")
        return
    samples = load_mmlu_redux(config.num_per_subject)

    model_path = config.model_name if config.method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name, config.seed)
    tokenizer = model.get_tokenizer()

    log_time = time.perf_counter()
    outputs = generate_vllm(model, [prompt for prompt, _ in samples], config.max_generation_len)
    result = evaluate_mmlu_redux(outputs, [answer for _, answer in samples], tokenizer)
    print(f"{target} accuracy {result['accuracy']:.2%} think_len {mean(result['think_lens']):.0f} truncated {mean(result['truncateds']):.2%} time {format_duration(time.perf_counter() - log_time)}")
    save_result("mmlu_redux", target, file_name, result)


if __name__ == "__main__":
    main()
