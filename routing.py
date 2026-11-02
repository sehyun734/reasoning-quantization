from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from statistics import mean
from tempfile import TemporaryDirectory
import time
from typing import Literal

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax
from vllm import ModelRegistry  # type: ignore
from vllm.compilation.decorators import support_torch_compile  # type: ignore
from vllm.config import VllmConfig  # type: ignore
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # type: ignore
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # type: ignore
from vllm.model_executor.models.qwen2 import Qwen2Model  # type: ignore
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer  # type: ignore
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix  # type: ignore

from primitives.dataset import load_math500
from primitives.evaluation import evaluate_math500
from primitives.hub import download_model, load_tensors, result_exists, save_result
from primitives.model import THINK_END, generate_vllm, load_vllm
from primitives.paths import make_quantized_path, make_routing_path
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["gptq", "awq", "mxfp4"] = "gptq"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    seed: int = 0
    route_ratio: float = 0.2
    route: Literal["high", "low", "random"] = "high"
    max_generation_len: int = 32768


def merge(
    bf16_path: str,
    quantized_path: str,
    directory: str,
) -> None:
    # vllm loads only one model, so merge bf16 and quantized layers into one
    # forward gives two hiddens, one from each half
    model_config = json.loads((Path(quantized_path) / "config.json").read_text())
    num_layers = model_config["num_hidden_layers"]
    model_config["architectures"] = [RoutedQwen3ForCausalLM.__name__]
    model_config["num_hidden_layers"] = 2 * num_layers
    model_config["max_window_layers"] = 2 * num_layers
    model_config["layer_types"] = model_config["layer_types"] * 2

    weights = {}
    for file in sorted(Path(bf16_path).glob("model*.safetensors")):
        weights.update(load_file(file))
    for file in sorted(Path(quantized_path).glob("model*.safetensors")):
        for name, tensor in load_file(file).items():
            match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
            if match:
                weights[f"model.layers.{int(match[1]) + num_layers}.{match[2]}"] = tensor
            else:
                assert torch.equal(tensor, weights[name]), name

    # save merged model to local directory
    (Path(directory) / "config.json").write_text(json.dumps(model_config, indent=2))
    shutil.copy(Path(bf16_path) / "generation_config.json", Path(directory) / "generation_config.json")
    save_file(weights, Path(directory) / "model.safetensors")
    print(f"merged {len(weights)} tensors num_hidden_layers {model_config['num_hidden_layers']}")


# vllm compiles forward with torch.compile
# all running problems share one row, so token count changes every step
# by default each new count recompiles, so dynamic_arg_dims marks dims that can change
@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "intermediate_tensors": 0, "inputs_embeds": 0})
class MergedQwen3Model(Qwen2Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        # run same input through each half
        # vllm expects one tensor, so concat two hiddens and split them in compute_logits
        num_layers = len(self.layers) // 2
        hiddens = []
        for layers in [self.layers[:num_layers], self.layers[num_layers:]]:
            hidden_states = self.embed_input_ids(input_ids)
            residual = None
            for layer in layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            hiddens.append(hidden_states)
        return torch.cat(hiddens, dim=-1)


class RoutedQwen3ForCausalLM(Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = MergedQwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        bf16_hiddens, quantized_hiddens = hidden_states.chunk(2, dim=-1)
        bf16_logits = self.logits_processor(self.lm_head, bf16_hiddens.contiguous())
        quantized_logits = self.logits_processor(self.lm_head, quantized_hiddens.contiguous())
        # random routes each position with route_ratio probability
        if self.config.route == "random":
            is_routed = torch.rand(len(hidden_states), device=hidden_states.device) < self.config.route_ratio
        else:
            # route by bf16 entropy compared to threshold
            bf16_logprobs = log_softmax(bf16_logits.float(), dim=-1)
            entropy = -(bf16_logprobs.exp() * bf16_logprobs).sum(dim=-1)
            if self.config.route == "high":
                is_routed = entropy > self.config.entropy_threshold
            else:
                is_routed = entropy < self.config.entropy_threshold
        return torch.where(is_routed[:, None], bf16_logits, quantized_logits)

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
    ) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    quantized_path = make_quantized_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_samples)
    target = make_routing_path(quantized_path, config.route_ratio, config.route)
    if result_exists("math500", target, file_name):
        print(f"{target} already evaluated")
        return
    # threshold from bf16 entropy in teacher forcing (seed 0), so it is set before generation
    entropy = load_tensors("teacher_forcing", quantized_path, "seed=0")["entropy"]
    entropy_threshold = None
    if config.route == "high":
        entropy_threshold = entropy.quantile(1 - config.route_ratio).item()
    elif config.route == "low":
        entropy_threshold = entropy.quantile(config.route_ratio).item()
    print(f"entropy threshold {entropy_threshold}")
    samples = load_math500()

    # register_model only works in this process, so run vllm engine here too
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    ModelRegistry.register_model(RoutedQwen3ForCausalLM.__name__, RoutedQwen3ForCausalLM)
    hf_overrides = {"route": config.route, "entropy_threshold": entropy_threshold, "route_ratio": config.route_ratio}
    with TemporaryDirectory() as directory:
        merge(snapshot_download(config.model_name, allow_patterns=["*.json", "*.safetensors"]), download_model(quantized_path), directory)
        # load merged model from directory with vllm
        model = load_vllm(directory, config.model_name, config.seed, hf_overrides=hf_overrides)
    tokenizer = model.get_tokenizer()
    think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)

    log_time = time.perf_counter()
    outputs = generate_vllm(model, [prompt for prompt, _ in samples], config.max_generation_len)
    result = evaluate_math500(outputs, [answer for _, answer in samples], tokenizer, think_end_id)
    print(f"{target} accuracy {result['accuracy']:.2%} think_len {mean(result['think_lens']):.0f} truncated {mean(result['truncateds']):.2%} time {format_duration(time.perf_counter() - log_time)}")
    save_result("math500", target, file_name, result)


if __name__ == "__main__":
    main()
