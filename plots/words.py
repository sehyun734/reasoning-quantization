from collections import Counter
from dataclasses import dataclass
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.hub import load_result, result_exists
from primitives.paths import make_target_path
from primitives.plot import METHOD_COLORS, get_words, plot_words, save_figure


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
    bf16_result = load_result("math500", make_target_path("bf16", config.model_name, config.num_bits, config.group_size, config.num_samples), file_name)
    results = {}
    for method in ["gptq", "awq"]:
        target = make_target_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("math500", target, file_name):
            results[method] = load_result("math500", target, file_name)

    rows = []
    for method, result in results.items():
        groups = {"both correct": ([], []), f"{method} wrong": ([], [])}
        for think, think_len, correct, truncated, bf16_think, bf16_think_len, bf16_correct in zip(result["thinks"], result["think_lens"], result["corrects"], result["truncateds"], bf16_result["thinks"], bf16_result["think_lens"], bf16_result["corrects"]):
            if bf16_correct and not truncated:
                word_rates, bf16_word_rates = groups["both correct" if correct else f"{method} wrong"]
                word_rates.append(count_word_rates(think, think_len))
                bf16_word_rates.append(count_word_rates(bf16_think, bf16_think_len))
        rows.extend((method, group, word_rates, bf16_word_rates) for group, (word_rates, bf16_word_rates) in groups.items())

    fig, axes = plt.subplots(len(rows), 3, figsize=(21, len(rows) * (1 + 0.4 * config.num_words)), layout="constrained", squeeze=False)
    for row_axes, (method, group, word_rates, bf16_word_rates) in zip(axes, rows):
        words = set().union(*word_rates, *bf16_word_rates)
        mean_rates = {word: mean(rates.get(word, 0) for rates in word_rates) for word in words}
        bf16_mean_rates = {word: mean(rates.get(word, 0) for rates in bf16_word_rates) for word in words}
        plot_words(row_axes, bf16_mean_rates, mean_rates, method, config.num_words, f"{method}, {len(word_rates)} {group} problems", "mean count per 1k reasoning tokens")

    handles = [Patch(color=METHOD_COLORS[method], label=method) for method in ["bf16", *results]]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} math500 words {file_name}, truncated excluded")
    save_figure(fig, config.model_name, "words")


if __name__ == "__main__":
    main()
