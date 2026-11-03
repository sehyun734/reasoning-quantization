from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
import time

from simple_parsing import parse
import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import RequestOutput  # type: ignore

from utils.dataset import load_math500
from utils.evaluation import evaluate_math500
from utils.format import format_duration
from utils.hub import load_result, result_exists, save_result, save_tensors
from utils.metrics import get_loop_ratio
from utils.model import TEMPERATURE, TOP_K, TOP_P, get_prompt_ids, get_think_hiddens, get_think_ids, load_model, load_vllm
from utils.paths import make_bf16_path, make_sensitivity_path


@dataclass
class Config:
    seed: int = 0
    # number of entropy bins (e.g. 5 is 0~20%, 20~40%, ..., 80~100%)
    num_bins: int = 5
    # continuations per condition, so result depends less on luck
    num_continuations: int = 4
    max_generation_len: int = 32768
    chunk_len: int = 2048
    # model to swap tokens in
    model_name: str = "Qwen/Qwen3-0.6B"


def generate_continuations(
    model: LLM,
    input_ids: list[list[int]],
    max_generation_lens: list[int],
    num_continuations: int,
) -> list[RequestOutput]:
    sampling_params = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=max_generation_len, n=num_continuations) for max_generation_len in max_generation_lens]
    return model.generate([{"prompt_token_ids": ids} for ids in input_ids], sampling_params, use_tqdm=True)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    target = make_sensitivity_path(config.model_name, config.num_bins, config.num_continuations)
    if result_exists("sensitivity/math500", target, file_name):
        print(f"{target} already evaluated")
        return
    model = load_model(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    problems = load_math500()
    bf16_result = load_result("math500", make_bf16_path(config.model_name), file_name)

    # 1) use only bf16 reasoning on correctly solved problems
    samples = []
    for problem_index, ((prompt, answer), think, correct) in enumerate(zip(problems, bf16_result["thinks"], bf16_result["corrects"])):
        if correct:
            prompt_ids = get_prompt_ids(tokenizer, prompt)
            think_ids = get_think_ids(tokenizer, think)
            samples.append((problem_index, answer, prompt_ids, think_ids))

    # 2) get bf16 entropy and top-k candidates at each reasoning position
    entropies = []
    top_ids = []
    top_probs = []
    for _, _, prompt_ids, think_ids in samples:
        hiddens = get_think_hiddens(model, torch.cat([prompt_ids, think_ids]), len(prompt_ids))
        sample_entropies = []
        sample_top_ids = []
        sample_top_probs = []
        for start in range(0, len(think_ids), config.chunk_len):
            logprobs = log_softmax(model.lm_head(hiddens[start : start + config.chunk_len]).float(), dim=-1)
            sample_entropies.append(-(logprobs.exp() * logprobs).sum(dim=-1).cpu())
            chunk_top_logprobs, chunk_top_ids = logprobs.topk(TOP_K, dim=-1)
            sample_top_ids.append(chunk_top_ids.cpu())
            sample_top_probs.append(chunk_top_logprobs.exp().cpu())
        entropies.append(torch.cat(sample_entropies))
        top_ids.append(torch.cat(sample_top_ids))
        top_probs.append(torch.cat(sample_top_probs))
    del model
    torch.cuda.empty_cache()

    # 3) entropy boundaries over all positions (e.g. 20%, 40%, 60%, 80%)
    entropy_thresholds = torch.cat(entropies).quantile(torch.linspace(0, 1, config.num_bins + 1)[1:-1])
    print(f"samples {len(samples)} positions {sum(map(len, entropies))} thresholds {[round(threshold, 4) for threshold in entropy_thresholds.tolist()]}")

    # 4) pick position and token to swap
    generator = torch.Generator().manual_seed(config.seed)
    special_ids = torch.tensor(list(tokenizer.added_tokens_decoder))
    swaps = []
    for (problem_index, answer, prompt_ids, think_ids), sample_entropies, sample_top_ids, sample_top_probs in zip(samples, entropies, top_ids, top_probs):
        # 4-1) give each position its entropy bin (e.g. 0~4), and pick one random position per bin
        entropy_bins = torch.bucketize(sample_entropies, entropy_thresholds)
        for bin_index in range(config.num_bins):
            candidates = (entropy_bins == bin_index).nonzero()[:, 0]
            for position in candidates[torch.randperm(len(candidates), generator=generator)[:1]].tolist():
                # 4-2) drop original and special tokens, and sample one from remaining top-k by prob
                original_token_id = think_ids[position].item()
                is_excluded = (sample_top_ids[position] == original_token_id) | torch.isin(sample_top_ids[position], special_ids)
                probs = torch.where(is_excluded, 0, sample_top_probs[position])
                # multinomial takes unnormalized weights, so no renormalizing after dropping
                swapped_token_id = sample_top_ids[position][torch.multinomial(probs, 1, generator=generator)].item()
                swaps.append({"problem_index": problem_index, "answer": answer, "entropy_bin": bin_index, "position": position, "entropy": sample_entropies[position].item(), "original_token_id": original_token_id, "swapped_token_id": swapped_token_id, "prompt_ids": prompt_ids.tolist(), "prefix_ids": think_ids[:position].tolist()})
    print(f"swaps {len(swaps)}")

    # 5) continue from original and swapped token with bf16
    model = load_vllm(config.model_name, config.model_name, config.seed)
    log_time = time.perf_counter()
    input_ids = [swap["prompt_ids"] + swap["prefix_ids"] + [swap[f"{condition}_token_id"]] for swap in swaps for condition in ["original", "swapped"]]
    max_generation_lens = [config.max_generation_len - len(swap["prefix_ids"]) - 1 for swap in swaps for _ in range(2)]
    outputs = generate_continuations(model, input_ids, max_generation_lens, config.num_continuations)

    # 6) score original and swapped
    result = defaultdict(list)
    continuation_ids = []
    for swap_index, swap in enumerate(swaps):
        result["problem_indices"].append(swap["problem_index"])
        result["entropy_bins"].append(swap["entropy_bin"])
        result["positions"].append(swap["position"])
        result["entropies"].append(swap["entropy"])
        result["original_token_ids"].append(swap["original_token_id"])
        result["swapped_token_ids"].append(swap["swapped_token_id"])
        for offset, condition in enumerate(["original", "swapped"]):
            continuations = outputs[2 * swap_index + offset].outputs
            continuation_ids += [torch.tensor(continuation.token_ids, dtype=torch.int32) for continuation in continuations]
            evaluated = evaluate_math500(continuations, [swap["answer"]] * len(continuations), tokenizer)
            prefix_ids = swap["prefix_ids"] + [swap[f"{condition}_token_id"]]
            thinks = [tokenizer.decode(prefix_ids + list(continuation.token_ids)[:think_len], skip_special_tokens=True) for continuation, think_len in zip(continuations, evaluated["think_lens"])]
            result[f"{condition}_corrects"].append(evaluated["corrects"])
            result[f"{condition}_think_lens"].append([len(prefix_ids) + think_len for think_len in evaluated["think_lens"]])
            result[f"{condition}_loop_ratios"].append(get_loop_ratio({"thinks": thinks}))

    for bin_index in range(config.num_bins):
        in_bin = [position_bin == bin_index for position_bin in result["entropy_bins"]]
        accuracies = [mean(mean(corrects) for corrects, used in zip(result[f"{condition}_corrects"], in_bin) if used) for condition in ["original", "swapped"]]
        loop_ratios = [mean(loop_ratio for loop_ratio, used in zip(result[f"{condition}_loop_ratios"], in_bin) if used) for condition in ["original", "swapped"]]
        print(f"bin {bin_index} accuracy {accuracies[0]:.2%} -> {accuracies[1]:.2%} loop {loop_ratios[0]:.2%} -> {loop_ratios[1]:.2%}")
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")
    save_result("sensitivity/math500", target, file_name, result)
    save_tensors("sensitivity/math500", target, file_name, {"continuation_ids": torch.cat(continuation_ids), "continuation_lens": torch.tensor([len(ids) for ids in continuation_ids]).view(len(swaps), 2, config.num_continuations)})


if __name__ == "__main__":
    main()
