from simple_parsing import parse

from plots.qat import Config, plot_qat

PENALTIES = {
    "gptq\n+ penalty 1.0": (1.0, "sandybrown"),
    "gptq\n+ penalty 2.0": (2.0, "chocolate"),
}
LOSS_VARIANTS = {
    "qat\nkl": (2048, "kl", 0.0, False, 0.0, "cosine", 3, "darkmagenta"),
    "qat\nce": (2048, "ce", 0.0, False, 0.0, "cosine", 3, "teal"),
    "qat\ncakld": (2048, "cakld", 0.0, False, 0.0, "cosine", 3, "darkkhaki"),
    "qat\n+ hidden loss": (2048, "kl", 1.0, False, 0.0, "cosine", 3, "tab:olive"),
}


def main() -> None:
    plot_qat(parse(Config), PENALTIES, LOSS_VARIANTS, "baselines")


if __name__ == "__main__":
    main()
