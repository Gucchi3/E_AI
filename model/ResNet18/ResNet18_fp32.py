"""FP32 ResNet-18 adapted to 32x32 CIFAR-10 images."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Two-convolution residual block used by ResNet-18."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(value)
        value    = self.relu1(self.bn1(self.conv1(value)))
        value    = self.bn2(self.conv2(value))
        return self.relu2(value + identity)


class ResNet18FP32(nn.Module):
    """CIFAR-10-adapted ResNet-18 with a stride-1 stem and no max pooling."""

    def __init__(self, num_classes: int = 10, image_size: int = 32) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("ResNet18FP32 is adapted specifically for 32x32 CIFAR images.")

        self.in_channels = 64
        self.stem       = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1     = self._make_stage(out_channels=64, block_count=2, stride=1)
        self.stage2     = self._make_stage(out_channels=128, block_count=2, stride=2)
        self.stage3     = self._make_stage(out_channels=256, block_count=2, stride=2)
        self.stage4     = self._make_stage(out_channels=512, block_count=2, stride=2)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_stage(self, out_channels: int, block_count: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        blocks.extend(BasicBlock(self.in_channels, out_channels) for _ in range(1, block_count))
        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        value = self.stage1(value)
        value = self.stage2(value)
        value = self.stage3(value)
        value = self.stage4(value)
        value = self.pool(value)
        value = torch.flatten(value, start_dim=1)
        return self.classifier(value)
