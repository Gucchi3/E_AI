"""FP32 MobileNetV2 adapted to 32x32 CIFAR-10 images."""

from __future__ import annotations

import math

import torch
from torch import nn


def _make_divisible(value: float, divisor: int = 8) -> int:
    """Round a channel count while avoiding a reduction larger than 10 percent."""
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class ConvBNReLU6(nn.Sequential):
    """Convolution followed by BatchNorm and ReLU6."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 1) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block with a linear bottleneck."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("InvertedResidual stride must be 1 or 2.")

        hidden_channels = int(round(in_channels * expand_ratio))
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_channels, kernel_size=1))
        layers.extend([ConvBNReLU6(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels), nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels)])
        self.block        = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        transformed = self.block(value)
        if self.use_residual:
            return value + transformed
        return transformed


class MobileNetV2FP32(nn.Module):
    """MobileNetV2 whose stem stride is adapted from 2 to 1 for CIFAR-10."""

    inverted_residual_setting = (
        # expansion, output channels, repeats, first stride
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )

    def __init__(self, num_classes: int = 10, image_size: int = 32, width_multiplier: float = 1.0, dropout: float = 0.2) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("MobileNetV2FP32 is adapted specifically for 32x32 CIFAR images.")
        if width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        input_channels = _make_divisible(32 * width_multiplier)
        last_channels  = _make_divisible(1280 * max(1.0, width_multiplier))
        # The original ImageNet stem uses stride 2. CIFAR-10 uses stride 1.
        features: list[nn.Module] = [ConvBNReLU6(3, input_channels, stride=1)]

        for expand_ratio, channels, repeats, first_stride in self.inverted_residual_setting:
            output_channels = _make_divisible(channels * width_multiplier)
            for block_index in range(repeats):
                stride = first_stride if block_index == 0 else 1
                features.append(InvertedResidual(input_channels, output_channels, stride, expand_ratio))
                input_channels = output_channels

        features.append(ConvBNReLU6(input_channels, last_channels, kernel_size=1))
        self.features   = nn.Sequential(*features)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(last_channels, num_classes))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.features(value)
        value = self.pool(value)
        value = torch.flatten(value, start_dim=1)
        return self.classifier(value)
