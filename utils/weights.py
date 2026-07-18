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

    model.load_state_dict(state_dict)
    return path
