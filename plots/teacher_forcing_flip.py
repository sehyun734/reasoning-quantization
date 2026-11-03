from dataclasses import dataclass

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from simple_parsing import parse
import torch

from utils.hub import load_tensors, result_exists
from utils.paths import make_quantized_path
from utils.plot import ENTROPY_COLORS, LINE_PANEL_WIDTH, METHOD_COLORS, get_figsize, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    num_bins: int = 20
    high_entropy_ratio: float = 0.2


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    results = {}
    for method in ["gptq", "awq", "mxfp4"]:
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("teacher_forcing/math500", quantized_path, file_name, "safetensors"):
            results[method] = load_tensors("teacher_forcing/math500", quantized_path, file_name)
    panels = [
        ("logit std error over bf16 top-20", lambda result: result["logit_std_error"], None, StrMethodFormatter("{x:.2f}")),
        ("1 - prob of bf16 top-1 token", lambda result: 1 - result["quantized_prob"], lambda result: 1 - result["bf16_prob"], PercentFormatter(1)),
        ("prob of sampling different token", lambda result: result["flip_prob"], None, PercentFormatter(1)),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=get_figsize([LINE_PANEL_WIDTH] * len(panels), 1), layout="constrained")
    bin_centers = (torch.arange(config.num_bins) + 0.5) / config.num_bins
    for method_index, (method, result) in enumerate(results.items()):
        entropy_bins = result["bf16_entropy"].argsort().argsort() * config.num_bins // len(result["bf16_entropy"])
        for ax, (ylabel, get_value, get_bf16_value, _) in zip(axes, panels):
            if get_bf16_value and method_index == 0:
                bf16_values = torch.stack([get_bf16_value(result)[entropy_bins == bin_index].mean() for bin_index in range(config.num_bins)])
                ax.plot(bin_centers, bf16_values, color=METHOD_COLORS["bf16"], marker="o", markersize=3)
            values = torch.stack([get_value(result)[entropy_bins == bin_index].mean() for bin_index in range(config.num_bins)])
            ax.plot(bin_centers, values, color=METHOD_COLORS[method], marker="o", markersize=3)
    for ax, (ylabel, _, _, formatter) in zip(axes, panels):
        ax.axvline(1 - config.high_entropy_ratio, color=ENTROPY_COLORS["high entropy"], linestyle="--")
        ax.set_xlabel("bf16 entropy percentile")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(formatter)

    fig.suptitle(f"{config.model_name} math500 teacher forcing flip {file_name}, num_bins={config.num_bins}, high_entropy_ratio={config.high_entropy_ratio:g}")
    handles = [Line2D([], [], color=METHOD_COLORS[method], marker="o", markersize=3, label=method) for method in ["bf16", *results]]
    handles.append(Line2D([], [], color=ENTROPY_COLORS["high entropy"], linestyle="--", label="high entropy"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save_figure(fig, config.model_name, "teacher_forcing_flip")


if __name__ == "__main__":
    main()
