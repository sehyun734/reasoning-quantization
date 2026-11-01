from dataclasses import dataclass

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse
import torch
from torch import Tensor
from transformers import AutoTokenizer

from primitives.hub import load_tensors, result_exists
from primitives.paths import make_quantized_path
from primitives.plot import ENTROPY_COLORS, METHOD_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure

TOKEN_CLASSES = ["digit", "symbol", "alpha", "other"]


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    high_entropy_ratio: float = 0.2


def get_token_classes(
    tokens: list[str],
) -> Tensor:
    classes = []
    for token in tokens:
        token = token.strip()
        if token.isdigit():
            classes.append(TOKEN_CLASSES.index("digit"))
        elif token and not any(char.isalnum() for char in token):
            classes.append(TOKEN_CLASSES.index("symbol"))
        elif token.isalpha():
            classes.append(TOKEN_CLASSES.index("alpha"))
        else:
            classes.append(TOKEN_CLASSES.index("other"))
    return torch.tensor(classes)


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    results = {}
    for method in ["gptq", "awq", "mxfp4"]:
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("teacher_forcing", quantized_path, file_name, "safetensors"):
            results[method] = load_tensors("teacher_forcing", quantized_path, file_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    token_classes = get_token_classes([tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])

    panel_widths = [get_panel_width(len(ENTROPY_COLORS), 1), get_panel_width(len(TOKEN_CLASSES), 2), get_panel_width(len(TOKEN_CLASSES), 2)]
    fig = plt.figure(figsize=get_figsize(panel_widths, len(results)), layout="constrained")
    for subfig, (method, result) in zip(fig.subfigures(len(results), 1, squeeze=False)[:, 0], results.items()):
        axes = subfig.subplots(1, len(panel_widths), width_ratios=panel_widths)
        kl = result["kl"]
        mismatch = (result["bf16_top1"] != result["top1"]).float()
        is_high = result["entropy"] > result["entropy"].quantile(1 - config.high_entropy_ratio)
        group_masks = {"low entropy": ~is_high, "high entropy": is_high}
        classes = token_classes[result["think_ids"]]

        kl_shares = [(kl[group_mask].sum() / kl.sum()).item() for group_mask in group_masks.values()]
        plot_bars(axes[0], list(group_masks), [(kl_shares, list(ENTROPY_COLORS.values()), [f"{kl_share:.1%}" for kl_share in kl_shares])])
        axes[0].set_ylabel("share of total kl")
        axes[0].set_title("kl share by entropy")

        groups = []
        for group, group_mask in group_masks.items():
            group_mismatches = [mismatch[(classes == index) & group_mask].mean().item() for index in range(len(TOKEN_CLASSES))]
            groups.append((group_mismatches, ENTROPY_COLORS[group], [f"{group_mismatch:.1%}" for group_mismatch in group_mismatches]))
        plot_bars(axes[1], TOKEN_CLASSES, groups)
        axes[1].set_ylabel("top-1 mismatch rate")
        axes[1].set_title("top-1 mismatch by token class")

        groups = []
        for color, probs in [(METHOD_COLORS["bf16"], result["bf16_prob"]), (METHOD_COLORS[method], result["prob"])]:
            tail_masses = [(1 - probs[(classes == index) & ~is_high]).mean().item() for index in range(len(TOKEN_CLASSES))]
            groups.append((tail_masses, color, [f"{tail_mass:.1%}" for tail_mass in tail_masses]))
        plot_bars(axes[2], TOKEN_CLASSES, groups)
        axes[2].set_ylabel("1 - prob of bf16 top-1 token")
        axes[2].set_title("low entropy tail mass by token class")

        for ax in axes:
            ax.margins(y=0.2)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        subfig.suptitle(method)

    handles = [Patch(color=color, label=group) for group, color in ENTROPY_COLORS.items()]
    handles += [Patch(color=METHOD_COLORS[method], label=method) for method in ["bf16", *results]]
    fig.suptitle(f"{config.model_name} math500 teacher forcing {file_name}, high_entropy_ratio={config.high_entropy_ratio:g}")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fit_bar_width(fig, axes)
    save_figure(fig, config.model_name, "teacher_forcing")


if __name__ == "__main__":
    main()
