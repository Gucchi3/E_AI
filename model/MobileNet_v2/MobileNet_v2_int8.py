"""INT8 QAT MobileNetV2 for 32x32 CIFAR-10 images."""

from __future__ import annotations

import math

import torch
from torch import nn

from utils.quantization import IntegerQuantizer, QuantConv2d, QuantLinear

from .MobileNet_v2_fp32 import _make_divisible


class INT8ConvBNReLU6(nn.Sequential):
    """INT8 convolution followed by ReLU6 and unsigned INT8 fake quantization."""

    def __init__(self, in_channels: int, out_channels: int, rounding: str, activation_range_momentum: float, kernel_size: int = 3, stride: int = 1, groups: int = 1) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(QuantConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True, weight_bits=8, rounding=rounding, quantizer="integer"), nn.Identity(), nn.ReLU6(inplace=False))
        self.activation_quantizer = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, range_momentum=activation_range_momentum)

    @property
    def output_scale(self) -> torch.Tensor:
        return self.activation_quantizer.scale

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        return self.activation_quantizer(self[2](self[0](value, input_scale)))


class INT8InvertedResidual(nn.Module):
    """INT8 inverted residual block with a signed linear bottleneck."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int, rounding: str, activation_range_momentum: float) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("INT8InvertedResidual stride must be 1 or 2.")

        hidden_channels = int(round(in_channels * expand_ratio))
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(INT8ConvBNReLU6(in_channels, hidden_channels, rounding, activation_range_momentum, kernel_size=1))
        layers.extend([INT8ConvBNReLU6(hidden_channels, hidden_channels, rounding, activation_range_momentum, stride=stride, groups=hidden_channels), QuantConv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=8, rounding=rounding, quantizer="integer"), nn.Identity()])
        self.block                = nn.Sequential(*layers)
        self.use_residual         = stride == 1 and in_channels == out_channels
        self.expand_ratio         = expand_ratio
        self.projection_quantizer = IntegerQuantizer(bit_width=8, signed=True, rounding=rounding, range_momentum=activation_range_momentum)
        self.residual_quantizer   = IntegerQuantizer(bit_width=8, signed=True, rounding=rounding, range_momentum=activation_range_momentum) if self.use_residual else None

    @property
    def output_scale(self) -> torch.Tensor:
        return self.residual_quantizer.scale if self.residual_quantizer is not None else self.projection_quantizer.scale

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        identity = value
        if self.expand_ratio == 1:
            hidden      = self.block[0](value, input_scale)
            transformed = self.block[1](hidden, self.block[0].output_scale)
        else:
            hidden      = self.block[0](value, input_scale)
            hidden      = self.block[1](hidden, self.block[0].output_scale)
            transformed = self.block[2](hidden, self.block[1].output_scale)

        transformed = self.projection_quantizer(transformed)
        if not self.use_residual:
            return transformed
        if self.residual_quantizer is None:
            raise RuntimeError("Residual output quantizer is not initialized.")
        # Fake-quantized branches are dequantized float values here; the sum is quantized again immediately below.
        return self.residual_quantizer(identity + transformed)


class MobileNetV2INT8(nn.Module):
    """CIFAR MobileNetV2 with INT8 weights and activations."""

    inverted_residual_setting = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))

    def __init__(self, num_classes: int = 10, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32, width_multiplier: float = 1.0, dropout: float = 0.2) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("MobileNetV2INT8 is adapted specifically for 32x32 CIFAR images.")
        if input_bits != 8:
            raise ValueError("MobileNetV2INT8 requires input_bits=8.")
        if width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.input_quantizer = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, fixed_scale=1.0 / 255.0)
        input_channels       = _make_divisible(32 * width_multiplier)
        last_channels        = _make_divisible(1280 * max(1.0, width_multiplier))
        features: list[nn.Module] = [INT8ConvBNReLU6(3, input_channels, rounding, activation_range_momentum, stride=1)]

        for expand_ratio, channels, repeats, first_stride in self.inverted_residual_setting:
            output_channels = _make_divisible(channels * width_multiplier)
            for block_index in range(repeats):
                stride = first_stride if block_index == 0 else 1
                features.append(INT8InvertedResidual(input_channels, output_channels, stride, expand_ratio, rounding, activation_range_momentum))
                input_channels = output_channels

        features.append(INT8ConvBNReLU6(input_channels, last_channels, rounding, activation_range_momentum, kernel_size=1))
        self.features         = nn.Sequential(*features)
        self.pool             = nn.AvgPool2d(kernel_size=2, stride=2)
        self.output_quantizer = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, range_momentum=activation_range_momentum)
        self.classifier       = nn.Sequential(nn.Dropout(p=dropout), QuantLinear(last_channels, num_classes, weight_bits=8, rounding=rounding, quantizer="integer"))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_quantizer(value)
        stem = self.features[0]
        if not isinstance(stem, INT8ConvBNReLU6):
            raise TypeError("The first MobileNetV2 feature must be INT8ConvBNReLU6.")
        value = stem(value, self.input_quantizer.scale)
        scale = stem.output_scale

        for index in range(1, len(self.features) - 1):
            block = self.features[index]
            if not isinstance(block, INT8InvertedResidual):
                raise TypeError("MobileNetV2 middle features must be INT8InvertedResidual modules.")
            value = block(value, scale)
            scale = block.output_scale

        head = self.features[-1]
        if not isinstance(head, INT8ConvBNReLU6):
            raise TypeError("The final MobileNetV2 feature must be INT8ConvBNReLU6.")
        value = head(value, scale)
        value = self.output_quantizer(self.pool(value))
        value = torch.flatten(value, start_dim=1)
        dropout = self.classifier[0]
        if not isinstance(dropout, nn.Dropout):
            raise TypeError("The first MobileNetV2 classifier module must be Dropout.")
        value = dropout(value)
        classifier_input_scale = self.output_quantizer.scale / (1.0 - dropout.p) if self.training else self.output_quantizer.scale
        return self.classifier[1](value, classifier_input_scale)
