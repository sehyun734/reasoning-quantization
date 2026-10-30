from collections import Counter
from dataclasses import dataclass

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse
from transformers import AutoTokenizer

from primitives.hub import load_tensors, result_exists
from primitives.paths import make_quantized_path
from primitives.plot import METHOD_COLORS, get_words, plot_words, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    high_entropy_ratio: float = 0.2
    num_words: int = 15


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    results = {}
    for method in ["gptq", "awq"]:
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("teacher_forcing", quantized_path, file_name, "safetensors"):
            results[method] = load_tensors("teacher_forcing", quantized_path, file_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    token_words = [words[0] if len(words) == 1 else None for words in map(get_words, [tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])]

    fig, axes = plt.subplots(len(results), 3, figsize=(21, len(results) * (1 + 0.4 * config.num_words)), layout="constrained", squeeze=False)
    for row_axes, (method, result) in zip(axes, results.items()):
        is_high = result["entropy"] > result["entropy"].quantile(1 - config.high_entropy_ratio)
        num_high = is_high.sum().item()
        bf16_counts = Counter(token_words[token_id] for token_id in result["bf16_top1"][is_high].tolist() if token_words[token_id])
        counts = Counter(token_words[token_id] for token_id in result["top1"][is_high].tolist() if token_words[token_id])
        bf16_rates = {word: count / num_high * 1000 for word, count in bf16_counts.items()}
        rates = {word: count / num_high * 1000 for word, count in counts.items()}
        plot_words(row_axes, bf16_rates, rates, method, config.num_words, f"{method}, {num_high} high entropy positions", "top-1 count per 1k high entropy positions")

    handles = [Patch(color=METHOD_COLORS[method], label=method) for method in ["bf16", *results]]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} math500 teacher forcing words {file_name}, high_entropy_ratio={config.high_entropy_ratio:g}")
    save_figure(fig, config.model_name, "teacher_forcing_words")


if __name__ == "__main__":
    main()
