"""モデルの計算量を計測する。"""

from __future__ import annotations

import warnings

import torch
from torch import nn


def get_macs_and_flops(model: nn.Module, device: torch.device, image_size: int) -> tuple[str, str]:
    """1枚の画像を入力したときのMACsとFLOPsを返す。"""
    was_training = model.training
    try:
        from thop import profile

        input_tensor = torch.randn(1, 3, image_size, image_size, device=device)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        flops = macs * 2
        return _format_count(macs), _format_count(flops)
    except Exception:
        return "N/A", "N/A"
    finally:
        model.train(was_training)


def _format_count(value: float) -> str:
    """計算量にK、M、Gの単位を付ける。"""
    if value >= 1e9:
        return f"{int(value):,} ({value / 1e9:.2f}G)"
    if value >= 1e6:
        return f"{int(value):,} ({value / 1e6:.2f}M)"
    if value >= 1e3:
        return f"{int(value):,} ({value / 1e3:.2f}K)"
    return f"{int(value):,}"
