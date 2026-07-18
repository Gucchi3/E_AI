"""モデルのMACsを計測する。"""

from __future__ import annotations

import warnings

import torch
from torch import nn
from thop import profile


def get_macs(model: nn.Module, device: torch.device, image_size: int) -> str:
    """thopでMACsを計測する。"""
    was_training = model.training
    try:
        input_tensor = torch.randn(1, 3, image_size, image_size, device=device)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    finally:
        model.train(was_training)

    return _format_count(macs)


def _format_count(value: float) -> str:
    """計算量にK、M、Gの単位を付ける。"""
    if value >= 1e9:
        return f"{int(value):,} ({value / 1e9:.2f}G)"
    if value >= 1e6:
        return f"{int(value):,} ({value / 1e6:.2f}M)"
    if value >= 1e3:
        return f"{int(value):,} ({value / 1e3:.2f}K)"
    return f"{int(value):,}"
