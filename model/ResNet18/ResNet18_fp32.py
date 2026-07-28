"""32x32のCIFAR-10入力に合わせたFP32 ResNet-18。"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """2個の3x3 Convと残差接続で構成するResNet-18のBasicBlock。"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(value)

        value = self.conv1(value)
        value = self.bn1(value)
        value = self.relu1(value)

        value = self.conv2(value)
        value = self.bn2(value)

        value = value + identity
        value = self.relu2(value)
        return value


class ResNet18FP32(nn.Module):
    """ImageNet版の7x7 StemとMaxPoolを3x3 stride-1 Stemへ変更したResNet-18。"""

    def __init__(self, num_classes: int = 10, image_size: int = 32) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("ResNet18FP32 is adapted specifically for 32x32 CIFAR images.")

        self.stem       = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1     = nn.Sequential(BasicBlock(64, 64, stride=1), BasicBlock(64, 64, stride=1))
        self.stage2     = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128, stride=1))
        self.stage3     = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256, stride=1))
        self.stage4     = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512, stride=1))
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

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
        value = self.stem[0](value)
        value = self.stem[1](value)
        value = self.stem[2](value)

        value = self.stage1[0](value)
        value = self.stage1[1](value)
        value = self.stage2[0](value)
        value = self.stage2[1](value)
        value = self.stage3[0](value)
        value = self.stage3[1](value)
        value = self.stage4[0](value)
        value = self.stage4[1](value)

        value = self.pool(value)
        value = torch.flatten(value, start_dim=1)
        value = self.classifier(value)
        return value
