import math_verify
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore


def evaluate_math500(
    outputs: list[CompletionOutput],
    answers: list[str],
    tokenizer: TokenizerLike,
    think_end_id: int,
) -> dict:
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

    parsed = [None if response is None else math_verify.parse(response) for response in responses]
    corrects = [prediction is not None and math_verify.verify(math_verify.parse(f"${answer}$"), prediction) for prediction, answer in zip(parsed, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": [str(prediction[-1]) if prediction else None for prediction in parsed],
        "think_lens": think_lens,
        "truncateds": [output.finish_reason == "length" for output in outputs],
        "thinks": thinks,
        "responses": responses,
    }
