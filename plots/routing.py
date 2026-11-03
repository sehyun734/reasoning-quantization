from dataclasses import dataclass
import math

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from simple_parsing import parse

from utils.dataset import load_math500
from utils.hub import load_result, result_exists
from utils.metrics import get_intermediate_answer_ratio, get_length_ratio, get_loop_ratio
from utils.paths import make_bf16_path, make_quantized_path, make_routing_path
from utils.plot import ENTROPY_COLORS, METHOD_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, plot_baseline, save_figure, spread_baseline_labels

ROUTE_POSITION_COLORS = {"high": ENTROPY_COLORS["high entropy"], "low": ENTROPY_COLORS["low entropy"], "random": "tab:purple"}
ROUTE_POSITION_LABELS = {"high": "high entropy", "low": "low entropy", "random": "random"}
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
    bf16_result = load_result("math500", make_bf16_path(config.model_name), file_name)
    answers = [answer for _, answer in load_math500()]
    panels = [
        ("accuracy", lambda result: result["accuracy"], "{x:.1%}"),
        ("median reasoning tokens / bf16, both correct", lambda result: get_length_ratio(result, bf16_result), "{x:.2f}x"),
        ("looped reasoning ratio", get_loop_ratio, "{x:.1%}"),
        ("correct intermediate, wrong final answer", lambda result: get_intermediate_answer_ratio(result, answers), "{x:.1%}"),
    ]

    for method in ["gptq", "awq", "mxfp4"]:
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        route_results = {}
        for route_position in ROUTE_POSITION_COLORS:
            route_results[route_position] = []
            for route_ratio in ROUTE_RATIOS:
                target = make_routing_path(quantized_path, route_ratio, route_position)
                route_results[route_position].append(load_result("routing/math500", target, file_name) if result_exists("routing/math500", target, file_name) else None)
        is_used = [any(route_results[route_position][index] for route_position in ROUTE_POSITION_COLORS) for index in range(len(ROUTE_RATIOS))]
        if not any(is_used):
            continue
        route_ratios = [ratio for ratio, used in zip(ROUTE_RATIOS, is_used) if used]
        for route_position in ROUTE_POSITION_COLORS:
            route_results[route_position] = [result for result, used in zip(route_results[route_position], is_used) if used]
        quantized_result = load_result("math500", quantized_path, file_name)

        panel_widths = [get_panel_width(len(route_ratios), len(ROUTE_POSITION_COLORS))] * len(panels)
        fig, axes = plt.subplots(1, len(panels), figsize=get_figsize(panel_widths, 1), width_ratios=panel_widths, layout="constrained")
        for ax, (ylabel, get_value, text_format) in zip(axes, panels):
            line_values = [get_value(bf16_result), get_value(quantized_result)]
            for line_method, line_value in zip(["bf16", method], line_values):
                plot_baseline(ax, line_value, METHOD_COLORS[line_method], text_format.format(x=line_value))
            groups = []
            for route_position, color in ROUTE_POSITION_COLORS.items():
                values = [get_value(result) if result else math.nan for result in route_results[route_position]]
                groups.append((values, color, ["" if math.isnan(value) else text_format.format(x=value) for value in values]))
            plot_bars(ax, route_ratios, groups)
            all_values = line_values + [value for values, _, _ in groups for value in values if not math.isnan(value)]
            span = max(all_values) - min(all_values)
            ax.set_ylim(max(min(all_values) - 0.2 * span, 0), max(all_values) + 0.2 * span)
            ax.set_xlabel("ratio of positions routed to bf16")
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(StrMethodFormatter(text_format))

        fig.suptitle(f"{config.model_name} math500 routing {method} {file_name}")
        handles = [Line2D([], [], color=METHOD_COLORS[name], linestyle="--", label=name) for name in ["bf16", method]]
        handles += [Patch(color=color, label=ROUTE_POSITION_LABELS[route_position]) for route_position, color in ROUTE_POSITION_COLORS.items()]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        fit_bar_width(fig, axes)
        spread_baseline_labels(fig, axes)
        save_figure(fig, config.model_name, f"routing_{method}")


if __name__ == "__main__":
    main()
