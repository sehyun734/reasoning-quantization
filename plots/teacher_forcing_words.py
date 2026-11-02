from collections import Counter
from dataclasses import dataclass

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse
from transformers import AutoTokenizer

from utils.hub import load_tensors, result_exists
from utils.metrics import get_words
from utils.paths import make_quantized_path
from utils.plot import METHOD_COLORS, get_words_figsize, plot_words, save_figure


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
    for method in ["gptq", "awq", "mxfp4"]:
        quantized_path = make_quantized_path(method, config.model_name, config.num_bits, config.group_size, config.num_samples)
        if result_exists("teacher_forcing/math500", quantized_path, file_name, "safetensors"):
            results[method] = load_tensors("teacher_forcing/math500", quantized_path, file_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    token_words = [words[0] if len(words) == 1 else None for words in map(get_words, [tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])]

    for method, result in results.items():
        is_high_entropy = result["bf16_entropy"] > result["bf16_entropy"].quantile(1 - config.high_entropy_ratio)
        num_high_entropy = is_high_entropy.sum().item()
        bf16_counts = Counter(token_words[token_id] for token_id in result["bf16_top1"][is_high_entropy].tolist() if token_words[token_id])
        quantized_counts = Counter(token_words[token_id] for token_id in result["quantized_top1"][is_high_entropy].tolist() if token_words[token_id])
        bf16_rates = {word: count / num_high_entropy * 1000 for word, count in bf16_counts.items()}
        quantized_rates = {word: count / num_high_entropy * 1000 for word, count in quantized_counts.items()}

        fig = plt.figure(figsize=get_words_figsize(config.num_words, 1), layout="constrained")
        plot_words(fig.subfigures(1, 1), bf16_rates, quantized_rates, method, config.num_words, f"{num_high_entropy} high entropy positions", "top-1 count per 1k high entropy positions")
        handles = [Patch(color=METHOD_COLORS[name], label=name) for name in ["bf16", method]]
        fig.suptitle(f"{config.model_name} math500 teacher forcing words {method} {file_name}, high_entropy_ratio={config.high_entropy_ratio:g}")
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        save_figure(fig, config.model_name, f"teacher_forcing_words_{method}")


if __name__ == "__main__":
    main()
