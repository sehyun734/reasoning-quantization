import re

import math_verify
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.model import THINK_END

BOXED_CHOICE_PATTERN = re.compile(r"\\boxed\{(?:\\text\{)?([A-D])\b")


def split_outputs(
    outputs: list[CompletionOutput],
    tokenizer: TokenizerLike,
) -> dict:
    think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)
    thinks = []
    responses = []
    think_lens = []
    for output in outputs:
        token_ids = list(output.token_ids)
        if think_end_id in token_ids:
            think_len = token_ids.index(think_end_id)
            response = tokenizer.decode(token_ids[think_len + 1 :], skip_special_tokens=True)
        else:
            # ran out of tokens while thinking, so no response
            think_len = len(token_ids)
            response = None
        thinks.append(tokenizer.decode(token_ids[:think_len], skip_special_tokens=True))
        responses.append(response)
        think_lens.append(think_len)
    return {
        "think_lens": think_lens,
        "truncateds": [output.finish_reason == "length" for output in outputs],
        "thinks": thinks,
        "responses": responses,
    }


def evaluate_math500(
    outputs: list[CompletionOutput],
    answers: list[str],
    tokenizer: TokenizerLike,
) -> dict:
    split = split_outputs(outputs, tokenizer)
    parsed = [None if response is None else math_verify.parse(response) for response in split["responses"]]
    corrects = [prediction is not None and math_verify.verify(math_verify.parse(f"${answer}$"), prediction) for prediction, answer in zip(parsed, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": [str(prediction[-1]) if prediction else None for prediction in parsed],
        **split,
    }


def evaluate_mmlu_redux(
    outputs: list[CompletionOutput],
    answers: list[str],
    tokenizer: TokenizerLike,
) -> dict:
    split = split_outputs(outputs, tokenizer)
    predictions = []
    for response in split["responses"]:
        choices = [] if response is None else BOXED_CHOICE_PATTERN.findall(response)
        predictions.append(choices[-1] if choices else None)
    corrects = [prediction == answer for prediction, answer in zip(predictions, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": predictions,
        **split,
    }
