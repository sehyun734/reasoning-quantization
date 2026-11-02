from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse

from primitives.dataset import load_math500
from primitives.evaluation import evaluate_math500
from primitives.hub import download_model, result_exists, save_result
from primitives.model import THINK_END, generate_vllm, load_vllm
from primitives.paths import make_penalty_path, make_target_path
from primitives.utils import format_duration

OVERTHINKING_WORDS = [
    "perhaps",
    "maybe",
    "wait",
    "Wait",
    "actually",
    "hold",
    "Hmm",
    "hmm",
    "Alternatively",
    "alternatively",
    "However",
    "however",
    "instead",
    "Instead",
    "But",
    "but",
    "though",
    "although",
    "yet",
    "rather",
    "unless",
    "otherwise",
    "nonetheless",
    "nevertheless",
    "regardless",
    "still",
    "anyway",
    "Or",
    "or",
    "either",
    "whether",
    "uncertain",
    "unsure",
    "possibly",
    "might",
    "could",
    "another",
    "different",
    "reconsider",
    "rethink",
    "backtrack",
    "retry",
    "revisit",
    "doubt",
    "confused",
    "wrong",
    "mistake",
    "error",
    "incorrect",
]


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["bf16", "gptq", "awq", "mxfp4"] = "gptq"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    penalty: float = 1.0
    max_generation_len: int = 32768


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    model_target = make_target_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_samples)
    target = make_penalty_path(model_target, config.penalty)
    if result_exists("math500", target, file_name):
        print(f"{target} already evaluated")
        return
    samples = load_math500()

    model_path = config.model_name if config.method == "bf16" else download_model(model_target)
    model = load_vllm(model_path, config.model_name)
    tokenizer = model.get_tokenizer()
    think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)
    logit_bias = {}
    for word in OVERTHINKING_WORDS:
        (token_id,) = tokenizer.encode(f" {word}", add_special_tokens=False)
        logit_bias[token_id] = -config.penalty
    print(f"penalized tokens {len(logit_bias)}")

    log_time = time.perf_counter()
    outputs = generate_vllm(model, [prompt for prompt, _ in samples], config.max_generation_len, config.seed, logit_bias)
    result = evaluate_math500(outputs, [answer for _, answer in samples], tokenizer, think_end_id)
    print(f"{target} accuracy {result['accuracy']:.2%} think_len {mean(result['think_lens']):.0f} truncated {mean(result['truncateds']):.2%} time {format_duration(time.perf_counter() - log_time)}")
    save_result("math500", target, file_name, result)


if __name__ == "__main__":
    main()
