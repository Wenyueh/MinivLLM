import os
from collections.abc import Iterable, Iterator

import torch
from safetensors import safe_open
from torch import nn


DEFAULT_PACKED_MODULES_MAPPING = {
    "qkv_projection": [
        ("q_proj", "q"),
        ("k_proj", "k"),
        ("v_proj", "v"),
    ],
    "gate_up": [
        ("gate_proj", 0),
        ("up_proj", 1),
    ],
}


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Copy an unsharded checkpoint tensor into a replicated parameter."""
    if param.shape != loaded_weight.shape:
        raise ValueError(
            f"Shape mismatch: model parameter {tuple(param.shape)} vs "
            f"checkpoint tensor {tuple(loaded_weight.shape)}"
        )
    param.data.copy_(loaded_weight)


def _map_weight_name(
    model: nn.Module, hf_name: str
) -> tuple[str, str | int | None]:
    """Map one Hugging Face parameter to this model's packed parameter."""
    packed_modules_mapping = getattr(
        model, "packed_modules_mapping", DEFAULT_PACKED_MODULES_MAPPING
    )
    for target_module, source_modules in packed_modules_mapping.items():
        for source_module, shard_id in source_modules:
            source = f".{source_module}."
            if source in hf_name:
                target_name = hf_name.replace(
                    source, f".{target_module}.", 1
                )
                return target_name, shard_id
    return hf_name, None


def load_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load Hugging Face tensors, using each parameter's weight loader.

    Packed checkpoint tensors such as q_proj/k_proj/v_proj are passed to the
    same target parameter one at a time. Its custom weight_loader selects the
    current tensor-parallel rank's shard and writes it into the correct packed
    region.
    """
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded_names: set[str] = set()
    loaded_param_ids: set[int] = set()
    unexpected_names: list[str] = []

    for hf_name, loaded_weight in weights:
        target_name, shard_id = _map_weight_name(model, hf_name)
        param = params.get(target_name)
        if param is None:
            unexpected_names.append(hf_name)
            continue

        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        try:
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load '{hf_name}' into '{target_name}'"
            ) from error

        loaded_names.add(target_name)
        loaded_param_ids.add(id(param))

    missing_names = [
        name for name, param in params.items()
        if id(param) not in loaded_param_ids
    ]
    if unexpected_names or missing_names:
        messages = []
        if unexpected_names:
            messages.append(
                "checkpoint tensors without a model parameter: "
                + ", ".join(unexpected_names[:10])
            )
        if missing_names:
            messages.append(
                "model parameters without a checkpoint tensor: "
                + ", ".join(missing_names[:10])
            )
        raise RuntimeError("Weight loading is incomplete; " + "; ".join(messages))

    return loaded_names


def _resolve_checkpoint_path(model_name_or_path: str) -> str:
    path = os.path.expanduser(model_name_or_path)
    if os.path.isdir(path):
        return path

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=model_name_or_path,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
        )
    except Exception as error:
        raise ValueError(
            f"Could not find or download model '{model_name_or_path}'"
        ) from error


def _iter_safetensor_weights(checkpoint_path: str) -> Iterator[tuple[str, torch.Tensor]]:
    safetensor_files = sorted(
        file_name for file_name in os.listdir(checkpoint_path)
        if file_name.endswith(".safetensors")
    )
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    seen_names: set[str] = set()
    for file_name in safetensor_files:
        file_path = os.path.join(checkpoint_path, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as checkpoint:
            for weight_name in checkpoint.keys():
                if weight_name in seen_names:
                    raise ValueError(
                        f"Duplicate checkpoint tensor '{weight_name}'"
                    )
                seen_names.add(weight_name)
                yield weight_name, checkpoint.get_tensor(weight_name)


def load_weights_from_checkpoint(
    model: nn.Module, model_name_or_path: str
) -> set[str]:
    """Load a local or Hugging Face safetensors checkpoint into ``model``."""
    checkpoint_path = _resolve_checkpoint_path(model_name_or_path)
    loaded_names = load_weights(
        model, _iter_safetensor_weights(checkpoint_path)
    )
    print(
        f"Loaded {len(loaded_names)} model parameters from "
        f"{model_name_or_path}"
    )
    return loaded_names
