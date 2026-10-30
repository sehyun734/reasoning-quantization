from dataclasses import dataclass
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from primitives.hub import load_result, result_exists
from primitives.paths import make_target_path
from primitives.plot import METHOD_COLORS, fit_bar_width, get_length_ratio, get_loop_ratio, get_figsize, get_panel_width, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0


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
        ratios.append(get_length_ratio(result, bf16_result))
        loops.append(get_loop_ratio(result))
    panels = [
        ("accuracy", accuracies, [f"{accuracy:.1%}\ntruncated {truncated:.1%}" for accuracy, truncated in zip(accuracies, truncateds)]),
        ("median reasoning tokens / bf16, both correct", ratios, [f"{ratio:.2f}x" for ratio in ratios]),
        ("looped reasoning ratio", loops, [f"{loop:.1%}" for loop in loops]),
    ]

    panel_widths = [get_panel_width(len(methods), 1)] * len(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=get_figsize(panel_widths, 1), width_ratios=panel_widths, layout="constrained")
    for ax, (ylabel, values, texts) in zip(axes, panels):
        plot_bars(ax, methods, [(values, [METHOD_COLORS[method] for method in methods], texts)])
        ax.set_ylabel(ylabel)
        ax.margins(y=0.2)
        ax.set_ylim(bottom=0)
    for ax in [axes[0], axes[2]]:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.suptitle(f"{config.model_name} math500 {file_name}")
    fit_bar_width(fig, axes)
    save_figure(fig, config.model_name, "math500")


if __name__ == "__main__":
    main()
