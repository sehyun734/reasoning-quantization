from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedModel


class StopForward(Exception):
    pass


def capture_layer_inputs(
    model: PreTrainedModel,
    layer: Module,
    samples: list[Tensor],
) -> list[tuple[tuple, dict]]:
    inputs = []

    def hook(
        module: Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        inputs.append((args, kwargs))
        raise StopForward

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    for input_ids in samples:
        try:
            model(input_ids[None, :].to(model.device), use_cache=False)
        except StopForward:
            pass
    handle.remove()
    return inputs
