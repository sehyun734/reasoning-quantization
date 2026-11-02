from dataclasses import dataclass
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from primitives.hub import load_result, load_tensors
from primitives.paths import make_qat_path, make_target_path
from primitives.plot import ENTROPY_COLORS, METHOD_COLORS, PANEL_MARGIN, fit_bar_width, get_figsize, get_length_ratio, get_loop_ratio, get_panel_width, plot_bars, save_figure

QAT_VARIANTS = {
    "3 epochs\n+ hidden loss": (2048, "kl", 1.0, False, 0.0, "cosine", 3, 0.0, "tab:olive"),
    "5 epochs\n+ hidden loss": (2048, "kl", 1.0, False, 0.0, "cosine", 5, 0.0, "darkolivegreen"),
}


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
    for label, (qat_num_samples, loss, hidden_weight, on_policy, scale_learning_rate, lr_schedule, num_epochs, high_entropy_weight, _) in QAT_VARIANTS.items():
        targets[label] = make_qat_path(config.model_name, config.num_bits, config.group_size, qat_num_samples, loss, hidden_weight, on_policy, scale_learning_rate, lr_schedule, num_epochs, high_entropy_weight)
    labels = list(targets)
    colors = [METHOD_COLORS["bf16"], METHOD_COLORS["gptq"]] + [color for *_, color in QAT_VARIANTS.values()]
    results = {label: load_result("mmlu_redux", target, file_name) for label, target in targets.items()}
    bf16_result = results["bf16"]

    accuracies, truncateds, ratios, loops, sames = [], [], [], [], []
    for label, result in results.items():
        accuracies.append(result["accuracy"])
        truncateds.append(mean(result["truncateds"]))
        ratios.append(get_length_ratio(result, bf16_result))
        loops.append(get_loop_ratio(result))
        if label != "bf16":
            sames.append(mean(prediction is not None and prediction == bf16_prediction for prediction, bf16_prediction in zip(result["predictions"], bf16_result["predictions"])))
    panels = [
        ("accuracy", labels, colors, accuracies, [f"{accuracy:.1%}\ntruncated {truncated:.1%}" for accuracy, truncated in zip(accuracies, truncateds)]),
        ("median reasoning tokens / bf16, both correct", labels, colors, ratios, [f"{ratio:.2f}x" for ratio in ratios]),
        ("looped reasoning ratio", labels, colors, loops, [f"{loop:.1%}" for loop in loops]),
        ("same choice as bf16", labels[1:], colors[1:], sames, [f"{same:.1%}" for same in sames]),
    ]

    kl_groups = {group: [] for group in ENTROPY_COLORS}
    for label in labels[1:]:
        tensors = load_tensors("teacher_forcing_mmlu_redux", targets[label], file_name)
        is_high = tensors["entropy"] > tensors["entropy"].quantile(1 - config.high_entropy_ratio)
        kl_groups["low entropy"].append(tensors["kl"][~is_high].mean().item())
        kl_groups["high entropy"].append(tensors["kl"][is_high].mean().item())

    top_widths = [get_panel_width(len(labels), 1)] * 3
    bottom_widths = [get_panel_width(len(labels) - 1, 1), get_panel_width(len(labels) - 1, len(ENTROPY_COLORS))]
    fig = plt.figure(figsize=get_figsize(top_widths, 2), layout="constrained")
    top, bottom = fig.subfigures(2, 1)
    bottom_axes = bottom.subplots(1, len(bottom_widths) + 1, width_ratios=[*bottom_widths, sum(top_widths) + PANEL_MARGIN * (len(top_widths) - len(bottom_widths)) - sum(bottom_widths)])
    bottom_axes[-1].axis("off")
    axes = [*top.subplots(1, len(top_widths), width_ratios=top_widths), *bottom_axes[:-1]]
    for ax, (ylabel, panel_labels, panel_colors, values, texts) in zip(axes, panels):
        plot_bars(ax, panel_labels, [(values, panel_colors, texts)])
        ax.set_ylabel(ylabel)
    plot_bars(axes[-1], labels[1:], [(kls, ENTROPY_COLORS[group], [f"{kl:.3f}" for kl in kls]) for group, kls in kl_groups.items()])
    axes[-1].set_ylabel("mean kl to bf16")
    for ax in axes:
        ax.margins(y=0.2)
        ax.set_ylim(bottom=0)
    for ax in [axes[0], axes[2], axes[3]]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(f"{config.model_name} mmlu_redux {file_name}, high_entropy_ratio={config.high_entropy_ratio:g}")
    fig.legend(handles=[Patch(color=color, label=group) for group, color in ENTROPY_COLORS.items()], loc="outside lower center", ncol=len(ENTROPY_COLORS))
    fit_bar_width(fig, axes[:len(top_widths)])
    save_figure(fig, config.model_name, "mmlu_redux")


if __name__ == "__main__":
    main()
