from dataclasses import dataclass
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from primitives.dataset import load_math500
from primitives.hub import load_result, load_tensors
from primitives.paths import make_qat_path, make_target_path
from primitives.plot import ENTROPY_COLORS, METHOD_COLORS, fit_bar_width, get_figsize, get_intermediate_answer_ratio, get_length_ratio, get_loop_ratio, get_panel_width, plot_bars, save_figure

HIDDEN_WEIGHTS = [0.0, 1.0]


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    high_entropy_ratio: float = 0.2


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    targets = {method: make_target_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples) for method in ["bf16", "gptq"]}
    for hidden_weight in HIDDEN_WEIGHTS:
        targets[f"qat\nhidden {hidden_weight:g}"] = make_qat_path(config.model_name, config.num_bits, config.group_size, hidden_weight)
    labels = list(targets)
    colors = [METHOD_COLORS[label.split("\n")[0]] for label in labels]
    results = {label: load_result("math500", target, file_name) for label, target in targets.items()}
    bf16_result = results["bf16"]
    answers = [answer for _, answer in load_math500()]

    accuracies, truncateds, ratios, loops, intermediates = [], [], [], [], []
    for result in results.values():
        accuracies.append(result["accuracy"])
        truncateds.append(mean(result["truncateds"]))
        ratios.append(get_length_ratio(result, bf16_result))
        loops.append(get_loop_ratio(result))
        intermediates.append(get_intermediate_answer_ratio(result, answers))
    panels = [
        ("accuracy", accuracies, [f"{accuracy:.1%}\ntruncated {truncated:.1%}" for accuracy, truncated in zip(accuracies, truncateds)]),
        ("median reasoning tokens / bf16, both correct", ratios, [f"{ratio:.2f}x" for ratio in ratios]),
        ("looped reasoning ratio", loops, [f"{loop:.1%}" for loop in loops]),
        ("wrong answers reaching correct answer midway", intermediates, [f"{intermediate:.1%}" for intermediate in intermediates]),
    ]

    kl_groups = {group: [] for group in ENTROPY_COLORS}
    for label in labels[1:]:
        tensors = load_tensors("teacher_forcing", targets[label], file_name)
        is_high = tensors["entropy"] > tensors["entropy"].quantile(1 - config.high_entropy_ratio)
        kl_groups["low entropy"].append(tensors["kl"][~is_high].mean().item())
        kl_groups["high entropy"].append(tensors["kl"][is_high].mean().item())

    panel_widths = [get_panel_width(len(labels), 1)] * len(panels) + [get_panel_width(len(labels) - 1, len(ENTROPY_COLORS))]
    fig, axes = plt.subplots(1, len(panel_widths), figsize=get_figsize(panel_widths, 1), width_ratios=panel_widths, layout="constrained")
    for ax, (ylabel, values, texts) in zip(axes, panels):
        plot_bars(ax, labels, [(values, colors, texts)])
        ax.set_ylabel(ylabel)
    plot_bars(axes[-1], labels[1:], [(kls, ENTROPY_COLORS[group], [f"{kl:.3f}" for kl in kls]) for group, kls in kl_groups.items()])
    axes[-1].set_ylabel("mean kl to bf16")
    for ax in axes:
        ax.margins(y=0.2)
        ax.set_ylim(bottom=0)
    for ax in [axes[0], axes[2], axes[3]]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(f"{config.model_name} math500 qat {file_name}, high_entropy_ratio={config.high_entropy_ratio:g}")
    fig.legend(handles=[Patch(color=color, label=group) for group, color in ENTROPY_COLORS.items()], loc="outside lower center", ncol=len(ENTROPY_COLORS))
    fit_bar_width(fig, axes)
    save_figure(fig, config.model_name, "qat")


if __name__ == "__main__":
    main()
