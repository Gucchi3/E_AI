"""A small CNN for CIFAR-10 that accepts 32x32 and 256x256 inputs."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolution, batch normalization, and ReLU as one readable unit."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv       = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm       = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class TinyCifarCNN(nn.Module):
    """Small CIFAR-10 classifier with adaptive pooling for 32 or 256 pixels."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem       = ConvBlock(3, 16, stride=1)
        self.stage1     = ConvBlock(16, 32, stride=2)
        self.stage2     = ConvBlock(32, 64, stride=2)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
