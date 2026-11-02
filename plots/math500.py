from dataclasses import dataclass
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.dataset import load_math500
from utils.hub import load_result, result_exists
from utils.metrics import get_intermediate_answer_ratio, get_length_ratio, get_loop_ratio
from utils.paths import make_target_path
from utils.plot import METHOD_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure


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
    for method in ["bf16", "gptq", "awq", "mxfp4"]:
        target = make_target_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("math500", target, file_name):
            results[method] = load_result("math500", target, file_name)
    methods = list(results)
    bf16_result = results["bf16"]
    answers = [answer for _, answer in load_math500()]

    accuracies, truncateds, length_ratios, loop_ratios, intermediate_answer_ratios = [], [], [], [], []
    for method in methods:
        result = results[method]
        accuracies.append(result["accuracy"])
        truncateds.append(mean(result["truncateds"]))
        length_ratios.append(get_length_ratio(result, bf16_result))
        loop_ratios.append(get_loop_ratio(result))
        intermediate_answer_ratios.append(get_intermediate_answer_ratio(result, answers))
    panels = [
        ("accuracy", accuracies, [f"{accuracy:.1%}\ntruncated {truncated:.1%}" for accuracy, truncated in zip(accuracies, truncateds)]),
        ("median reasoning tokens / bf16, both correct", length_ratios, [f"{length_ratio:.2f}x" for length_ratio in length_ratios]),
        ("looped reasoning ratio", loop_ratios, [f"{loop_ratio:.1%}" for loop_ratio in loop_ratios]),
        ("correct intermediate, wrong final answer", intermediate_answer_ratios, [f"{intermediate_answer_ratio:.1%}" for intermediate_answer_ratio in intermediate_answer_ratios]),
    ]

    panel_widths = [get_panel_width(len(methods), 1)] * len(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=get_figsize(panel_widths, 1), width_ratios=panel_widths, layout="constrained")
    for ax, (ylabel, values, texts) in zip(axes, panels):
        plot_bars(ax, methods, [(values, [METHOD_COLORS[method] for method in methods], texts)])
        ax.set_ylabel(ylabel)
        ax.margins(y=0.2)
        ax.set_ylim(bottom=0)
    for ax in [axes[0], axes[2], axes[3]]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(f"{config.model_name} math500 {file_name}")
    fit_bar_width(fig, axes)
    save_figure(fig, config.model_name, "math500")


if __name__ == "__main__":
    main()
