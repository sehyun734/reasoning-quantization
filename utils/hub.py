import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import load_file, save
from torch import Tensor
from transformers import PreTrainedModel


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/reasoning-quantization"


def upload_file(
    path: str,
    data: bytes,
) -> None:
    HfApi().upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=get_repo_name(), commit_message=f"save {path}")
    print(f"saved {path}")


def download_file(
    path: str,
) -> str:
    return hf_hub_download(get_repo_name(), path)


def save_model(
    path: str,
    model: PreTrainedModel,
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        HfApi().upload_folder(folder_path=directory, path_in_repo=path, repo_id=get_repo_name(), commit_message=f"save {path}")
    print(f"saved {path}")


def save_grid(
    path: str,
    grid: dict[str, Tensor],
) -> None:
    upload_file(f"{path}/grid.safetensors", save(grid))


def download_model(
    path: str,
) -> str:
    return str(Path(snapshot_download(get_repo_name(), allow_patterns=[f"{path}/*.json", f"{path}/model*.safetensors"])) / path)


def result_exists(
    entry_name: str,
    target: str,
    file_name: str,
    extension: str = "json",
) -> bool:
    return HfApi().file_exists(get_repo_name(), f"{entry_name}/{target}/{file_name}.{extension}")


def save_result(
    entry_name: str,
    target: str,
    file_name: str,
    result: dict,
) -> None:
    upload_file(f"{entry_name}/{target}/{file_name}.json", json.dumps(result, indent=2, ensure_ascii=False).encode())


def load_result(
    entry_name: str,
    target: str,
    file_name: str,
) -> dict:
    return json.loads(Path(download_file(f"{entry_name}/{target}/{file_name}.json")).read_text())


def save_tensors(
    entry_name: str,
    target: str,
    file_name: str,
    tensors: dict[str, Tensor],
) -> None:
    upload_file(f"{entry_name}/{target}/{file_name}.safetensors", save(tensors))


def load_tensors(
    entry_name: str,
    target: str,
    file_name: str,
) -> dict[str, Tensor]:
    return load_file(download_file(f"{entry_name}/{target}/{file_name}.safetensors"))
