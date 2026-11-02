import re
from statistics import mean, median

import math_verify

ANSWER_PATTERN = re.compile(r"answer (?:is|should be|would be|must be)\s*:?\s*([^\n]{1,80}?)(?:\.\s|\n|$)", re.IGNORECASE)


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
