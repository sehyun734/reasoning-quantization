from simple_parsing import parse

from plots.qat import Config, plot_qat

SCHEDULE_VARIANTS = {
    "constant\n1 epoch": (0.0, False, 0.0, "constant", 1, "tab:pink"),
    "cosine\n1 epoch": (0.0, False, 0.0, "cosine", 1, "orchid"),
    "cosine\n3 epochs": (0.0, False, 0.0, "cosine", 3, "darkmagenta"),
    "cosine 3 epochs\n+ hidden loss": (1.0, False, 0.0, "cosine", 3, "tab:olive"),
}


def main() -> None:
    plot_qat(parse(Config), SCHEDULE_VARIANTS, "qat_schedule")


if __name__ == "__main__":
    main()
