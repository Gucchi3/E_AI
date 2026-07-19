"""学習済み重みを読み込む。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_model_weight(model: nn.Module, device: torch.device, weight_path: str | Path) -> Path:
    """state_dictをモデルへ読み込む。"""
    path = Path(weight_path)
    if not path.is_file():
        raise FileNotFoundError(f"Weight file was not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Weight file does not contain a state_dict: {path}")

    mapped_state, unused_keys = _map_state_dict(model, state_dict)
    incompatible              = model.load_state_dict(mapped_state, strict=False)
    allowed_missing           = (".scale", ".running_min", ".running_max", ".range_initialized", ".bias_scale", ".bias_integer", ".accumulator_bound", ".multiplier", ".shift")
    invalid_missing           = [name for name in incompatible.missing_keys if not name.endswith(allowed_missing)]
    allowed_unexpected        = (".running_min", ".running_max", ".range_initialized", ".accumulator_bound", ".multiplier", ".shift")
    unexpected                = sorted(name for name in set(unused_keys) | set(incompatible.unexpected_keys) if not name.endswith(allowed_unexpected))
    if invalid_missing or unexpected:
        raise RuntimeError(f"Weight structure does not match the model. Missing: {invalid_missing}, unexpected: {unexpected}")
    return path


def _map_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """FP32のConv・BNキーをfold対応QATモデルへ対応付ける。"""
    mapped_state = {}
    used_keys    = set()
    for target_name in model.state_dict():
        source_name = _fp32_source_name(target_name)
        for candidate in (target_name, source_name):
            if candidate in state_dict:
                mapped_state[target_name] = state_dict[candidate]
                used_keys.add(candidate)
                break
    unused_keys = [name for name in state_dict if name not in used_keys]
    return mapped_state, unused_keys


def _fp32_source_name(target_name: str) -> str:
    """fold対応QATモデルのキーからFP32モデルのキーを作る。"""
    return target_name.replace(".conv.conv.", ".conv.").replace(".conv.norm.", ".norm.")
