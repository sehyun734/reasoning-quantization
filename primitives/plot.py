from pathlib import Path

from matplotlib.figure import Figure

METHOD_COLORS = {"bf16": "tab:gray", "gptq": "tab:orange", "awq": "tab:green"}


def save_figure(
    fig: Figure,
    model_name: str,
    name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")
