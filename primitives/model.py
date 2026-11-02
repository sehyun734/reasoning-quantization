import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore

THINK_END = "</think>"
# qwen3's recommended settings for thinking mode
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20


def load_model(
    model_path: str,
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model, tokenizer


def load_vllm(
    model_path: str,
    model_name: str,
    hf_overrides: dict | None = None,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=40960, hf_overrides=hf_overrides)


def generate_vllm(
    model: LLM,
    prompts: list[str],
    max_generation_len: int,
    seed: int,
    logit_bias: dict[int, float] | None = None,
) -> list[CompletionOutput]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=max_generation_len, seed=seed, logit_bias=logit_bias)
    outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]
