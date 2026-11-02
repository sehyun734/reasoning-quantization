from datasets import get_dataset_config_names, load_dataset
from safetensors.torch import load_file, save
import torch
from torch import Tensor

from utils.hub import download_file, upload_file

MATH_PROMPT = """\
{problem}
Please reason step by step, and put your final answer within \\boxed{{}}."""

MMLU_PROMPT = """\
{question}
{choices}
Please reason step by step, and put your final answer (A, B, C, or D) within \\boxed{{}}."""

MMLU_MATH_SUBJECTS = {"abstract_algebra", "college_mathematics", "elementary_mathematics", "high_school_mathematics", "high_school_statistics"}


def load_numina_prompts(
    num_samples: int,
    seed: int,
) -> list[str]:
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train").shuffle(seed=seed)
    prompts = []
    for row in dataset:
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


def load_mmlu_redux(
    num_per_subject: int,
) -> list[tuple[str, str]]:
    samples = []
    for subject in get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0"):
        if subject in MMLU_MATH_SUBJECTS:
            continue
        dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test").filter(lambda row: row["error_type"] == "ok")
        for row in dataset.select(range(num_per_subject)):
            choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
            samples.append((MMLU_PROMPT.format(question=row["question"], choices=choices), "ABCD"[row["answer"]]))
    return samples
