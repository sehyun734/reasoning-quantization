from dataclasses import dataclass
import math

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from primitives.hub import load_result, result_exists
from primitives.paths import make_quantized_path, make_routing_path, make_target_path
from primitives.plot import METHOD_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure

ROUTE_COLORS = {"high": "tab:red", "low": "tab:blue", "random": "tab:purple"}
ROUTE_LABELS = {"high": "high entropy", "low": "low entropy", "random": "random"}
ROUTE_RATIOS = [0.1, 0.2, 0.8]


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
    bf16_accuracy = load_result("math500", make_target_path("bf16", config.model_name, config.num_bits, config.group_size, config.num_samples), file_name)["accuracy"]
    methods = []
    for method in ["gptq", "awq"]:
        if result_exists("math500", make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples), file_name):
            methods.append(method)

    panel_widths = [get_panel_width(len(ROUTE_RATIOS), len(ROUTE_COLORS))] * len(methods)
    fig, axes = plt.subplots(1, len(methods), figsize=get_figsize(panel_widths, 1), width_ratios=panel_widths, layout="constrained", squeeze=False)
    for ax, method in zip(axes[0], methods):
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        quantized_accuracy = load_result("math500", quantized_path, file_name)["accuracy"]
        for line_method, line_accuracy in [("bf16", bf16_accuracy), (method, quantized_accuracy)]:
            ax.axhline(line_accuracy, color=METHOD_COLORS[line_method], linestyle="--")
            ax.annotate(f"{line_accuracy:.1%}", (1, line_accuracy), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=METHOD_COLORS[line_method])
        groups = []
        for route, color in ROUTE_COLORS.items():
            accuracies = []
            for route_ratio in ROUTE_RATIOS:
                target = make_routing_path(quantized_path, route_ratio, route)
                accuracies.append(load_result("math500", target, file_name)["accuracy"] if result_exists("math500", target, file_name) else math.nan)
            groups.append((accuracies, color, ["" if math.isnan(accuracy) else f"{accuracy:.1%}" for accuracy in accuracies]))
        plot_bars(ax, ROUTE_RATIOS, groups)
        all_accuracies = [bf16_accuracy, quantized_accuracy] + [accuracy for accuracies, _, _ in groups for accuracy in accuracies if not math.isnan(accuracy)]
        ax.set_xlabel("ratio of positions routed to bf16")
        ax.set_ylabel("accuracy")
        ax.set_ylim(min(all_accuracies) - 0.03, max(all_accuracies) + 0.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(method)
    fig.suptitle(f"{config.model_name} math500 routing {file_name}")
    handles = [Line2D([], [], color=METHOD_COLORS[method], linestyle="--", label=method) for method in ["bf16", *methods]]
    handles += [Patch(color=color, label=ROUTE_LABELS[route]) for route, color in ROUTE_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fit_bar_width(fig, axes[0])
    save_figure(fig, config.model_name, "routing")


if __name__ == "__main__":
    main()
