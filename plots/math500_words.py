from collections import Counter
from dataclasses import dataclass
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_words
from utils.paths import make_bf16_path, make_target_path
from utils.plot import METHOD_COLORS, get_words_figsize, plot_words, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    num_words: int = 15


def count_word_rates(
    think: str,
    think_len: int,
) -> dict[str, float]:
    counts = Counter(get_words(think))
    return {word: count / think_len * 1000 for word, count in counts.items()}


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    bf16_result = load_result("math500", make_bf16_path(config.model_name), file_name)
    results = {}
    for method in ["gptq", "awq", "mxfp4"]:
        target = make_target_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("math500", target, file_name):
            results[method] = load_result("math500", target, file_name)

    for method, result in results.items():
        groups = {"both correct": ([], []), f"{method} wrong": ([], [])}
        for think, think_len, correct, truncated, bf16_think, bf16_think_len, bf16_correct in zip(result["thinks"], result["think_lens"], result["corrects"], result["truncateds"], bf16_result["thinks"], bf16_result["think_lens"], bf16_result["corrects"]):
            if bf16_correct and not truncated:
                bf16_word_rates, quantized_word_rates = groups["both correct" if correct else f"{method} wrong"]
                bf16_word_rates.append(count_word_rates(bf16_think, bf16_think_len))
                quantized_word_rates.append(count_word_rates(think, think_len))

        fig = plt.figure(figsize=get_words_figsize(config.num_words, len(groups)), layout="constrained")
        for subfig, (group, (bf16_word_rates, quantized_word_rates)) in zip(fig.subfigures(len(groups), 1, squeeze=False)[:, 0], groups.items()):
            words = set().union(*bf16_word_rates, *quantized_word_rates)
            bf16_mean_rates = {word: mean(rates.get(word, 0) for rates in bf16_word_rates) for word in words}
            quantized_mean_rates = {word: mean(rates.get(word, 0) for rates in quantized_word_rates) for word in words}
            plot_words(subfig, bf16_mean_rates, quantized_mean_rates, method, config.num_words, f"{len(quantized_word_rates)} {group} problems", "mean count per 1k reasoning tokens")

        handles = [Patch(color=METHOD_COLORS[name], label=name) for name in ["bf16", method]]
        fig.suptitle(f"{config.model_name} math500 words {method} {file_name}, truncated excluded")
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        save_figure(fig, config.model_name, f"math500_words_{method}")


if __name__ == "__main__":
    main()
