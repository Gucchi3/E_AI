"""FP32モデルで使用する畳み込みブロック。"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """畳み込み、BatchNorm、ReLUをまとめたブロック。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv       = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm       = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """特徴量を変換する。"""
        return self.activation(self.norm(self.conv(value)))
