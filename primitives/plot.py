from pathlib import Path
import re
from statistics import mean, median

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
import matplotlib.pyplot as plt
import math_verify

METHOD_COLORS = {"bf16": "tab:gray", "gptq": "tab:orange", "awq": "tab:green", "mxfp4": "tab:brown"}
ENTROPY_COLORS = {"low entropy": "tab:blue", "high entropy": "tab:red"}
BAR_INCHES = 0.5
BAR_GAP = 1
PANEL_HEIGHT = 3.5
LINE_PANEL_WIDTH = 3.5
PANEL_MARGIN = 0.9
WORD_BAR_INCHES = 0.12
WORD_PANEL_WIDTH = 4.5
STOPWORDS = {"the", "an", "is", "are", "was", "were", "be", "been", "being", "am", "it", "it's", "its", "this", "that", "that's", "these", "those", "of", "to", "in", "on", "at", "by", "with", "from", "as", "into", "for", "and", "or", "we", "me", "my", "our", "us", "you", "your", "he", "she", "they", "them", "their", "his", "her", "there", "there's", "here", "which", "what", "where", "who", "how", "has", "have", "had", "do", "does", "did", "can", "could", "would", "will"}
MARKERS = {
    "doubt": ["wait", "but", "however", "hmm", "wrong"],
    "branch": ["alternatively", "maybe", "perhaps"],
    "check": ["verify", "confirm", "remember", "sure"],
}
ANSWER_PATTERN = re.compile(r"answer (?:is|should be|would be|must be)\s*:?\s*([^\n]{1,80}?)(?:\.\s|\n|$)", re.IGNORECASE)

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


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(len(think) > 200 and think[:-200].count(think[-200:-150]) >= 5 for think in result["thinks"])


def get_length_ratio(
    result: dict,
    bf16_result: dict,
) -> float:
    return median(think_len / bf16_think_len for think_len, bf16_think_len, correct, bf16_correct in zip(result["think_lens"], bf16_result["think_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct)


def get_intermediate_answer_ratio(
    result: dict,
    answers: list[str],
) -> float:
    ratios = []
    for think, correct, answer in zip(result["thinks"], result["corrects"], answers):
        if correct:
            continue
        gold = math_verify.parse(f"${answer}$")
        ratios.append(any(math_verify.verify(gold, math_verify.parse(match)) for match in ANSWER_PATTERN.findall(think)))
    return mean(ratios)


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


def get_words_figsize(
    num_words: int,
    num_rows: int,
) -> tuple[float, float]:
    num_slots = max(num_words, sum(map(len, MARKERS.values())))
    return WORD_PANEL_WIDTH * 3, num_rows * (WORD_BAR_INCHES * num_slots * (2 + BAR_GAP) + 1.2) + 0.8


def plot_words(
    subfig: SubFigure,
    bf16_rates: dict[str, float],
    rates: dict[str, float],
    method: str,
    num_words: int,
    title: str,
    xlabel: str,
) -> None:
    words = sorted(bf16_rates.keys() | rates.keys(), key=lambda word: rates.get(word, 0) - bf16_rates.get(word, 0), reverse=True)
    word_groups = {"increased": words[:num_words], "decreased": words[::-1][:num_words], "markers": [word for marker_words in MARKERS.values() for word in marker_words]}
    slot_width = 2 + BAR_GAP
    num_slots = max(map(len, word_groups.values()))
    for ax, (group, group_words) in zip(subfig.subplots(1, len(word_groups)), word_groups.items()):
        for index, (color, method_rates) in enumerate([(METHOD_COLORS["bf16"], bf16_rates), (METHOD_COLORS[method], rates)]):
            ys = [slot * slot_width + index - 0.5 for slot in range(len(group_words))]
            ax.barh(ys, [method_rates.get(word, 0) for word in group_words], height=1, color=color)
        for slot, word in enumerate(group_words):
            bf16_rate, rate = bf16_rates.get(word, 0), rates.get(word, 0)
            ax.annotate(f"{rate / bf16_rate:.2f}x" if bf16_rate else "new", (max(bf16_rate, rate), slot * slot_width), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks([slot * slot_width for slot in range(len(group_words))], group_words)
        ax.set_ylim((num_slots - 0.5) * slot_width, -slot_width / 2)
        ax.margins(x=0.15)
        ax.set_title(group)
        ax.set_xlabel(xlabel)
    subfig.suptitle(title)


def save_figure(
    fig: Figure,
    model_name: str,
    name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
