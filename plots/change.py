from dataclasses import dataclass

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from simple_parsing import parse
import torch

from primitives.hub import load_tensors, result_exists
from primitives.paths import make_quantized_path
from primitives.plot import LINE_PANEL_WIDTH, METHOD_COLORS, get_figsize, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    num_bins: int = 20


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    results = {}
    for method in ["gptq", "awq", "mxfp4"]:
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("teacher_forcing", quantized_path, file_name, "safetensors"):
            results[method] = load_tensors("teacher_forcing", quantized_path, file_name)
    panels = [
        ("logit error std over bf16 top-20", lambda result: result["logit_error"], None, StrMethodFormatter("{x:.2f}")),
        ("tail mass / bf16 tail mass", lambda result: 1 - result["prob"], lambda result: 1 - result["bf16_prob"], StrMethodFormatter("{x:g}x")),
        ("tail mass - bf16 tail mass", lambda result: result["bf16_prob"] - result["prob"], None, PercentFormatter(1)),
        ("prob of sampling different token", lambda result: result["change_prob"], None, PercentFormatter(1)),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=get_figsize([LINE_PANEL_WIDTH] * len(panels), 1), layout="constrained")
    bin_centers = (torch.arange(config.num_bins) + 0.5) / config.num_bins
    for method, result in results.items():
        bins = result["entropy"].argsort().argsort() * config.num_bins // len(result["entropy"])
        for ax, (ylabel, get_value, get_base, _) in zip(axes, panels):
            values = torch.stack([get_value(result)[bins == index].mean() for index in range(config.num_bins)])
            if get_base:
                values /= torch.stack([get_base(result)[bins == index].mean() for index in range(config.num_bins)])
            ax.plot(bin_centers, values, color=METHOD_COLORS[method], marker="o", markersize=3)
    for ax, (ylabel, _, _, formatter) in zip(axes, panels):
        ax.set_xlabel("bf16 entropy percentile")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(formatter)
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(panels[1][3])
    fig.suptitle(f"{config.model_name} math500 change by entropy {file_name}, num_bins={config.num_bins}")
    handles = [Line2D([], [], color=METHOD_COLORS[method], marker="o", markersize=3, label=method) for method in results]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save_figure(fig, config.model_name, "change")


if __name__ == "__main__":
    main()
