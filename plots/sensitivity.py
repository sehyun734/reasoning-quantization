from dataclasses import dataclass
from statistics import mean, median

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_length_ratio, get_loop_ratio
from utils.paths import make_bf16_path, make_quantized_path, make_sensitivity_path
from utils.plot import METHOD_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, plot_baseline, save_figure, spread_baseline_labels

CONDITION_COLORS = {"original": METHOD_COLORS["bf16"], "swapped": "tab:olive"}
CONDITION_LABELS = {"original": "original token", "swapped": "one token swapped"}


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    num_bins: int = 5
    num_continuations: int = 4


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    result = load_result("sensitivity/math500", make_sensitivity_path(config.model_name, config.num_bins, config.num_continuations), file_name)
    bf16_result = load_result("math500", make_bf16_path(config.model_name), file_name)
    quantized_results = {}
    for method in ["gptq", "awq", "mxfp4"]:
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("math500", quantized_path, file_name):
            quantized_results[method] = load_result("math500", quantized_path, file_name)
    labels = [f"{index * 100 // config.num_bins}-{(index + 1) * 100 // config.num_bins}%" for index in range(config.num_bins)]
    in_bins = [[entropy_bin == index for entropy_bin in result["entropy_bins"]] for index in range(config.num_bins)]

    def get_bin_values(name: str, reduce) -> list[float]:
        return [mean(reduce(value) for value, used in zip(result[name], in_bin) if used) for in_bin in in_bins]

    correct_indices = [index for index, correct in enumerate(bf16_result["corrects"]) if correct]
    quantized_accuracies = {}
    quantized_loop_ratios = {}
    quantized_length_ratios = {}
    for method, quantized_result in quantized_results.items():
        quantized_accuracies[method] = mean(quantized_result["corrects"][index] for index in correct_indices)
        quantized_loop_ratios[method] = get_loop_ratio({"thinks": [quantized_result["thinks"][index] for index in correct_indices]})
        quantized_length_ratios[method] = get_length_ratio(quantized_result, bf16_result)
    length_ratios = []
    for in_bin in in_bins:
        length_ratios.append(median(mean(swapped) / mean(original) for original, swapped, used in zip(result["original_think_lens"], result["swapped_think_lens"], in_bin) if used))
    panels = [
        ("accuracy", [(get_bin_values(f"{condition}_corrects", mean), CONDITION_COLORS[condition]) for condition in CONDITION_COLORS], quantized_accuracies, "{x:.1%}", PercentFormatter(1)),
        ("looped reasoning ratio", [(get_bin_values(f"{condition}_loop_ratios", float), CONDITION_COLORS[condition]) for condition in CONDITION_COLORS], quantized_loop_ratios, "{x:.1%}", PercentFormatter(1)),
        ("median reasoning tokens / bf16", [(length_ratios, CONDITION_COLORS["swapped"])], quantized_length_ratios, "{x:.2f}x", StrMethodFormatter("{x:.2f}x")),
    ]

    panel_widths = [get_panel_width(config.num_bins, len(groups)) for _, groups, _, _, _ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=get_figsize(panel_widths, 1), width_ratios=panel_widths, layout="constrained")
    for ax, (ylabel, groups, quantized_values, text_format, formatter) in zip(axes, panels):
        plot_bars(ax, labels, [(values, color, [text_format.format(x=value) for value in values]) for values, color in groups])
        for method, quantized_value in quantized_values.items():
            plot_baseline(ax, quantized_value, METHOD_COLORS[method], text_format.format(x=quantized_value))
        ax.set_xlabel("bf16 entropy percentile of swapped position")
        ax.set_ylabel(ylabel)
        ax.margins(y=0.2)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(formatter)

    fig.suptitle(f"{config.model_name} math500 sensitivity {file_name}, num_bins={config.num_bins}, num_continuations={config.num_continuations}, bf16 correct only")
    handles = [Patch(color=color, label=CONDITION_LABELS[condition]) for condition, color in CONDITION_COLORS.items()]
    handles += [Line2D([], [], color=METHOD_COLORS[method], linestyle="--", label=method) for method in quantized_results]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fit_bar_width(fig, axes)
    spread_baseline_labels(fig, axes)
    save_figure(fig, config.model_name, "sensitivity")


if __name__ == "__main__":
    main()
