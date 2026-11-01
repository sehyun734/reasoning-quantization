from simple_parsing import parse

from plots.qat import Config, plot_qat

SCHEDULE_VARIANTS = {
    "constant\n1 epoch": (2048, "kl", 0.0, False, 0.0, "constant", 1, "tab:pink"),
    "cosine\n1 epoch": (2048, "kl", 0.0, False, 0.0, "cosine", 1, "orchid"),
    "cosine\n3 epochs": (2048, "kl", 0.0, False, 0.0, "cosine", 3, "darkmagenta"),
    "3 epochs\n+ hidden loss": (2048, "kl", 1.0, False, 0.0, "cosine", 3, "tab:olive"),
    "5 epochs\n+ hidden loss": (2048, "kl", 1.0, False, 0.0, "cosine", 5, "darkolivegreen"),
}


def main() -> None:
    plot_qat(parse(Config), SCHEDULE_VARIANTS, "qat_schedule")


if __name__ == "__main__":
    main()
