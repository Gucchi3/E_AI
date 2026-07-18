"""CIFAR-10用の小さなCNN。"""

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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴量を変換する。"""
        return self.activation(self.norm(self.conv(x)))



class TinyCifarCNN(nn.Module):
    """畳み込みと線形層だけで構成した分類モデル。"""

    def __init__(self, num_classes: int = 10, image_size: int = 32) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4.")
        final_kernel    = image_size // 4
        self.stem       = ConvBlock(3, 16, stride=1)
        self.stage1     = ConvBlock(16, 32, stride=2)
        self.stage2     = ConvBlock(32, 64, stride=2)
        self.head       = ConvBlock(64, 64, stride=1, kernel_size=final_kernel, padding=0)
        self.classifier = nn.Linear(64, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """クラスごとのlogitを返す。"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.head(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
