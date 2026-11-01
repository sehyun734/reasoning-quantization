from dataclasses import dataclass
import math

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from simple_parsing import parse

from primitives.dataset import load_math500
from primitives.hub import load_result, result_exists
from primitives.paths import make_quantized_path, make_routing_path, make_target_path
from primitives.plot import METHOD_COLORS, fit_bar_width, get_figsize, get_intermediate_answer_ratio, get_length_ratio, get_loop_ratio, get_panel_width, plot_bars, save_figure

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
    bf16_result = load_result("math500", make_target_path("bf16", config.model_name, config.num_bits, config.group_size, config.num_samples), file_name)
    results = {}
    for method in ["gptq", "awq", "mxfp4"]:
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        route_results = {}
        for route in ROUTE_COLORS:
            route_results[route] = []
            for route_ratio in ROUTE_RATIOS:
                target = make_routing_path(quantized_path, route_ratio, route)
                route_results[route].append(load_result("math500", target, file_name) if result_exists("math500", target, file_name) else None)
        if any(any(ratio_results) for ratio_results in route_results.values()):
            results[method] = (load_result("math500", quantized_path, file_name), route_results)
    answers = [answer for _, answer in load_math500()]
    panels = [
        ("accuracy", lambda result: result["accuracy"], "{x:.1%}"),
        ("median reasoning tokens / bf16, both correct", lambda result: get_length_ratio(result, bf16_result), "{x:.2f}x"),
        ("looped reasoning ratio", get_loop_ratio, "{x:.1%}"),
        ("wrong answers reaching correct answer midway", lambda result: get_intermediate_answer_ratio(result, answers), "{x:.1%}"),
    ]

    panel_widths = [get_panel_width(len(ROUTE_RATIOS), len(ROUTE_COLORS))] * len(panels)
    fig = plt.figure(figsize=get_figsize(panel_widths, len(results)), layout="constrained")
    for subfig, (method, (quantized_result, route_results)) in zip(fig.subfigures(len(results), 1, squeeze=False)[:, 0], results.items()):
        axes = subfig.subplots(1, len(panels), width_ratios=panel_widths)
        for ax, (ylabel, get_value, text_format) in zip(axes, panels):
            line_values = [get_value(bf16_result), get_value(quantized_result)]
            for line_method, line_value in zip(["bf16", method], line_values):
                ax.axhline(line_value, color=METHOD_COLORS[line_method], linestyle="--")
                ax.annotate(text_format.format(x=line_value), (1, line_value), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=METHOD_COLORS[line_method])
            groups = []
            for route, color in ROUTE_COLORS.items():
                values = [get_value(result) if result else math.nan for result in route_results[route]]
                groups.append((values, color, ["" if math.isnan(value) else text_format.format(x=value) for value in values]))
            plot_bars(ax, ROUTE_RATIOS, groups)
            all_values = line_values + [value for values, _, _ in groups for value in values if not math.isnan(value)]
            span = max(all_values) - min(all_values)
            ax.set_ylim(max(min(all_values) - 0.2 * span, 0), max(all_values) + 0.2 * span)
            ax.set_xlabel("ratio of positions routed to bf16")
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(StrMethodFormatter(text_format))
        subfig.suptitle(method)
    fig.suptitle(f"{config.model_name} math500 routing {file_name}")
    handles = [Line2D([], [], color=METHOD_COLORS[method], linestyle="--", label=method) for method in ["bf16", *results]]
    handles += [Patch(color=color, label=ROUTE_LABELS[route]) for route, color in ROUTE_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fit_bar_width(fig, axes)
    save_figure(fig, config.model_name, "routing")


if __name__ == "__main__":
    main()
