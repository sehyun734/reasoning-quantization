def make_numina_path(
    target: str,
    num_samples: int,
    max_generation_len: int,
    seed: int,
) -> str:
    return f"numina/{target}/num_samples={num_samples},max_generation_len={max_generation_len},seed={seed}"


def make_mxfp4_path(
    model_name: str,
) -> str:
    return f"mxfp4/{model_name}"


def make_quantized_path(
    method: str,
    model_name: str,
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    if method == "mxfp4":
        return make_mxfp4_path(model_name)
    return f"{method}/{model_name}/num_bits={num_bits},group_size={group_size},num_samples={num_samples}"


def make_target_path(
    method: str,
    model_name: str,
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    if method == "bf16":
        return f"bf16/{model_name}"
    return make_quantized_path(method, model_name, num_bits, group_size, num_samples)


def make_qat_path(
    model_name: str,
    num_bits: int,
    group_size: int,
    hidden_weight: float,
    on_policy: bool,
    scale_learning_rate: float,
) -> str:
    return f"qat/{model_name}/num_bits={num_bits},group_size={group_size},hidden_weight={hidden_weight},on_policy={on_policy},scale_learning_rate={scale_learning_rate}"


def make_routing_path(
    quantized_path: str,
    route_ratio: float,
    route: str,
) -> str:
    return f"routing/{quantized_path}/route_ratio={route_ratio},route={route}"
