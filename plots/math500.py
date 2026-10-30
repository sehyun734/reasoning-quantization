from dataclasses import dataclass
from statistics import mean, median

import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.hub import load_result, result_exists
from primitives.paths import make_target_path
from primitives.plot import METHOD_COLORS, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0


def is_loop(
    think: str,
) -> bool:
    return len(think) > 200 and think[:-200].count(think[-200:-150]) >= 5


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    results = {}
    for method in ["bf16", "gptq", "awq"]:
        target = make_target_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("math500", target, file_name):
            results[method] = load_result("math500", target, file_name)
    methods = list(results)
    bf16_result = results["bf16"]

    accuracies, truncateds, ratios, loops = [], [], [], []
    for method in methods:
        result = results[method]
        accuracies.append(result["accuracy"])
        truncateds.append(mean(result["truncateds"]))
        ratios.append(median(think_len / bf16_think_len for think_len, bf16_think_len, correct, bf16_correct in zip(result["think_lens"], bf16_result["think_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct))
        loops.append(mean(is_loop(think) for think in result["thinks"]))
    panels = [
        ("accuracy", accuracies, [f"{accuracy:.1%}\ntruncated {truncated:.1%}" for accuracy, truncated in zip(accuracies, truncateds)]),
        ("reasoning tokens / bf16 (median)", ratios, [f"{ratio:.2f}x" for ratio in ratios]),
        ("loop ratio", loops, [f"{loop:.1%}" for loop in loops]),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=((1 + len(methods)) * len(panels), 4), layout="constrained")
    for ax, (ylabel, values, texts) in zip(axes, panels):
        ax.bar(range(len(methods)), values, width=0.4, color=[METHOD_COLORS[method] for method in methods])
        for x, (value, text) in enumerate(zip(values, texts)):
            ax.annotate(text, (x, value), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(methods)), methods)
        ax.set_xlim(-0.75, len(methods) - 0.25)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.2)
        ax.set_ylim(bottom=0)
    fig.suptitle(f"{config.model_name} math500 {file_name}")
    save_figure(fig, config.model_name, "math500")


if __name__ == "__main__":
    main()
