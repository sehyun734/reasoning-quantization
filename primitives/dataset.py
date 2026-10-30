from datasets import load_dataset
from safetensors.torch import load_file, save
import torch
from torch import Tensor

from primitives.hub import download_file, upload_file

MATH_PROMPT = """\
{problem}
Please reason step by step, and put your final answer within \\boxed{{}}."""


def load_numina_prompts(
    num_samples: int,
    seed: int,
) -> list[str]:
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train").shuffle(seed=seed)
    prompts = []
    for row in dataset:
        # skip synthetic problems to avoid overlap with math500
        # only word problems that have one boxed answer
        # and only valid, complete problems
        if row["synthetic"] or (row["question_type"] != "math-word-problem") or (row["problem_is_valid"] != "Yes"):
            continue
        prompts.append(MATH_PROMPT.format(problem=row["problem"]))
        if len(prompts) == num_samples:
            break
    return prompts


def save_numina(
    path: str,
    input_ids: list[Tensor],
) -> None:
    upload_file(f"{path}/inputs.safetensors", save({"input_ids": torch.cat(input_ids), "input_lens": torch.tensor([len(sample_input_ids) for sample_input_ids in input_ids])}))


def load_numina(
    path: str,
) -> list[Tensor]:
    tensors = load_file(download_file(f"{path}/inputs.safetensors"))
    return list(tensors["input_ids"].split(tensors["input_lens"].tolist()))


def load_math500() -> list[tuple[str, str]]:
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [(MATH_PROMPT.format(problem=row["problem"]), row["answer"]) for row in dataset]
