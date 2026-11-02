from dataclasses import dataclass
from statistics import mean
import time

from simple_parsing import parse
import torch

from utils.data import load_numina_prompts, save_numina
from utils.format import format_duration
from utils.model import generate_vllm, load_vllm
from utils.paths import make_numina_path


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_samples: int = 128
    max_generation_len: int = 2048
    seed: int = 0


def main() -> None:
    # 1) load prompts and model
    config = parse(Config)
    prompts = load_numina_prompts(config.num_samples, config.seed)
    model = load_vllm(config.model_name, config.model_name, config.seed)
    tokenizer = model.get_tokenizer()

    # 2) generate reasoning with bf16 model
    log_time = time.perf_counter()
    outputs = generate_vllm(model, prompts, config.max_generation_len)
    print(f"generated {len(outputs)} output_len {mean(len(output.token_ids) for output in outputs):.0f} time {format_duration(time.perf_counter() - log_time)}")

    # 3) save prompt and reasoning as calibration data
    input_ids = []
    for prompt, output in zip(prompts, outputs):
        prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt")["input_ids"][0]
        input_ids.append(torch.cat([prompt_ids, torch.tensor(output.token_ids)]))
    save_numina(make_numina_path(config.model_name, config.num_samples, config.max_generation_len, config.seed), input_ids)


if __name__ == "__main__":
    main()
