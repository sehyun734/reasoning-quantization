from pathlib import Path
import re

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

METHOD_COLORS = {"bf16": "tab:gray", "gptq": "tab:orange", "awq": "tab:green"}
BAR_INCHES = 0.5
BAR_GAP = 1
PANEL_HEIGHT = 3.5
PANEL_MARGIN = 0.9
STOPWORDS = {"the", "an", "is", "are", "was", "were", "be", "been", "being", "am", "it", "it's", "its", "this", "that", "that's", "these", "those", "of", "to", "in", "on", "at", "by", "with", "from", "as", "into", "for", "and", "or", "we", "me", "my", "our", "us", "you", "your", "he", "she", "they", "them", "their", "his", "her", "there", "there's", "here", "which", "what", "where", "who", "how", "has", "have", "had", "do", "does", "did", "can", "could", "would", "will"}
MARKERS = {
    "doubt": ["wait", "but", "however", "hmm", "wrong"],
    "branch": ["alternatively", "maybe", "perhaps"],
    "check": ["verify", "confirm", "remember", "sure"],
}

plt.rcParams.update({
    "figure.titlesize": 13,
    "figure.constrained_layout.w_pad": 0.08,
    "figure.constrained_layout.h_pad": 0.08,
    "figure.constrained_layout.wspace": 0.08,
    "figure.constrained_layout.hspace": 0.08,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
})


def get_words(
    text: str,
) -> list[str]:
    words = re.findall(r"(?<![\\a-z])[a-z]+(?:'[a-z]+)?(?:-[a-z]{2,})*", text.lower())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


def get_panel_width(
    num_slots: int,
    num_groups: int,
) -> float:
    return BAR_INCHES * num_slots * (num_groups + BAR_GAP)


def get_figsize(
    panel_widths: list[float],
    num_rows: int,
) -> tuple[float, float]:
    return sum(panel_widths) + PANEL_MARGIN * len(panel_widths), PANEL_HEIGHT * num_rows + 1


def plot_bars(
    ax: Axes,
    labels: list[str],
    groups: list[tuple[list[float], str | list[str], list[str]]],
) -> None:
    slot_width = len(groups) + BAR_GAP
    for index, (values, color, texts) in enumerate(groups):
        xs = [slot * slot_width + index - (len(groups) - 1) / 2 for slot in range(len(labels))]
        ax.bar(xs, values, width=1, color=color)
        for x, value, text in zip(xs, values, texts):
            ax.annotate(text, (x, value), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([slot * slot_width for slot in range(len(labels))], labels)
    ax.set_xlim(-slot_width / 2, (len(labels) - 0.5) * slot_width)


def fit_bar_width(
    fig: Figure,
    axes: list[Axes],
) -> None:
    for _ in range(2):
        fig.canvas.draw()
        axes_width = sum(ax.get_window_extent().width for ax in axes) / fig.dpi
        num_bars = sum(ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes)
        fig.set_figwidth(fig.get_figwidth() + BAR_INCHES * num_bars - axes_width)


def plot_words(
    axes: list[Axes],
    bf16_rates: dict[str, float],
    rates: dict[str, float],
    method: str,
    num_words: int,
    title: str,
    xlabel: str,
) -> None:
    words = sorted(bf16_rates.keys() | rates.keys(), key=lambda word: rates.get(word, 0) - bf16_rates.get(word, 0), reverse=True)
    word_groups = {"increased": words[:num_words], "decreased": words[::-1][:num_words], "markers": [word for marker_words in MARKERS.values() for word in marker_words]}
    for ax, (group, group_words) in zip(axes, word_groups.items()):
        ys = range(len(group_words))
        for offset, color, method_rates in [(-0.2, METHOD_COLORS["bf16"], bf16_rates), (0.2, METHOD_COLORS[method], rates)]:
            values = [method_rates.get(word, 0) for word in group_words]
            ax.barh([y + offset for y in ys], values, height=0.4, color=color)
            for y, value in zip(ys, values):
                ax.annotate(f"{value:.2f}", (value, y + offset), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(ys, group_words)
        ax.invert_yaxis()
        ax.margins(x=0.25)
        ax.set_title(f"{title}\n{group}", fontsize=10)
        ax.set_xlabel(xlabel)


def save_figure(
    fig: Figure,
    model_name: str,
    name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
