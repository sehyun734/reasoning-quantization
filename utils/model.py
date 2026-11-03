import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore

THINK_END = "</think>"
# qwen3 recommended setting for thinking mode
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20


def load_model(
    model_path: str,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model


def load_vllm(
    model_path: str,
    model_name: str,
    seed: int,
    hf_overrides: dict | None = None,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=40960, seed=seed, hf_overrides=hf_overrides)


def generate_vllm(
    model: LLM,
    prompts: list[str],
    max_generation_len: int,
) -> list[CompletionOutput]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=max_generation_len)
    outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]


def get_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> Tensor:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt")["input_ids"][0]


def get_think_ids(
    tokenizer: PreTrainedTokenizerBase,
    think: str,
) -> Tensor:
    return tokenizer(think, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def get_think_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # hidden at t predicts token t+1
    # so keep from last prompt token to second last reasoning token
    # e.g. input [p0, p1, p2, r0, r1, r2]
    #      keep  [        p2, r0, r1    ] (each predicts r0, r1, r2)
    return model.model(input_ids=input_ids[None, :].to(model.device)).last_hidden_state[0, prompt_len - 1 : -1]
