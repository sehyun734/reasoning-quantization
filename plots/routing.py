from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.hub import load_result, result_exists
from primitives.paths import make_quantized_path, make_routing_path, make_target_path
from primitives.plot import METHOD_COLORS, save_figure

ROUTE_COLORS = {"high": "tab:red", "low": "tab:blue", "random": "tab:purple"}
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

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), layout="constrained", squeeze=False)
    for ax, method in zip(axes[0], methods):
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        ax.axhline(bf16_accuracy, color=METHOD_COLORS["bf16"], linestyle="--", label=f"bf16 {bf16_accuracy:.1%}")
        quantized_accuracy = load_result("math500", quantized_path, file_name)["accuracy"]
        ax.axhline(quantized_accuracy, color=METHOD_COLORS[method], linestyle="--", label=f"{method} {quantized_accuracy:.1%}")
        all_accuracies = [bf16_accuracy, quantized_accuracy]
        for offset, (route, color) in zip([-0.25, 0, 0.25], ROUTE_COLORS.items()):
            xs, accuracies = [], []
            for x, route_ratio in enumerate(ROUTE_RATIOS):
                target = make_routing_path(quantized_path, route_ratio, route)
                if result_exists("math500", target, file_name):
                    xs.append(x + offset)
                    accuracies.append(load_result("math500", target, file_name)["accuracy"])
            ax.bar(xs, accuracies, width=0.25, color=color, label=route)
            all_accuracies += accuracies
            for x, accuracy in zip(xs, accuracies):
                ax.annotate(f"{accuracy:.1%}", (x, accuracy), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ROUTE_RATIOS)), ROUTE_RATIOS)
        ax.set_xlabel("route ratio")
        ax.set_ylabel("accuracy")
        ax.set_ylim(min(all_accuracies) - 0.03, max(all_accuracies) + 0.03)
        ax.set_title(method)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(f"{config.model_name} math500 routing {file_name}")
    save_figure(fig, config.model_name, "routing")


if __name__ == "__main__":
    main()
