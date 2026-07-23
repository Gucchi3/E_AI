"""FP32のCIFAR-10 CNN。"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock


class CNN(nn.Module):
    """Conv、BatchNorm、Linearで構成したFP32分類モデル。"""

    def __init__(self, num_classes: int = 10, image_size: int = 32) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        feature_size    = image_size // 8
        self.stem       = ConvBlock(3, 16, stride=1)
        self.stage1     = ConvBlock(16, 32, stride=2)
        self.stage2     = ConvBlock(32, 64, stride=2)
        self.head       = ConvBlock(64, 64, stride=2)
        self.classifier = nn.Linear(64 * feature_size * feature_size, num_classes)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """クラスごとのlogitを返す。"""
        value = self.stem(value)
        value = self.stage1(value)
        value = self.stage2(value)
        value = self.head(value)
        value = torch.flatten(value, start_dim=1)
        return self.classifier(value)
