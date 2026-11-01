from simple_parsing import parse

from plots.qat import Config, plot_qat

SCHEDULE_VARIANTS = {
    "qat\nconstant\n1 epoch": (0.0, False, 0.0, "constant", 1, "tab:pink"),
    "qat\ncosine\n1 epoch": (0.0, False, 0.0, "cosine", 1, "orchid"),
    "qat\ncosine\n3 epochs": (0.0, False, 0.0, "cosine", 3, "darkmagenta"),
}


def main() -> None:
    plot_qat(parse(Config), SCHEDULE_VARIANTS, "qat_schedule")


if __name__ == "__main__":
    main()
