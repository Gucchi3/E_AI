"""INT QAT用の小さなCNN。"""

from __future__ import annotations

import torch
from torch import nn

from utils.quantization import IntegerQuantizer, QuantLinear

from .blocks import BlockQuantization, QuantConvBlock


class TinyQATCNN(nn.Module):
    """BN吸収済み畳み込みと線形層で構成したQATモデル。"""

    def __init__(self, num_classes: int = 10, weight_bits: int = 8, activation_bits: int = 8, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4.")
        final_kernel         = image_size // 4
        quantization         = BlockQuantization("integer", "integer", weight_bits, activation_bits)
        self.input_quantizer = IntegerQuantizer(bit_width=input_bits, signed=False, rounding=rounding, fixed_scale=1.0 / (2**input_bits - 1))
        self.stem            = QuantConvBlock(3, 16, stride=1, quantization=quantization, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage1          = QuantConvBlock(16, 32, stride=2, quantization=quantization, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage2          = QuantConvBlock(32, 64, stride=2, quantization=quantization, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.head            = QuantConvBlock(64, 64, stride=1, quantization=quantization, rounding=rounding, activation_range_momentum=activation_range_momentum, kernel_size=final_kernel, padding=0)
        self.classifier      = QuantLinear(64, num_classes, weight_bits=weight_bits, rounding=rounding)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """uint8画像を模擬してクラスごとのlogitを返す。"""
        value = self.input_quantizer(value)
        value = self.stem(value, self.input_quantizer.scale)
        value = self.stage1(value, self.stem.activation_quantizer.scale)
        value = self.stage2(value, self.stage1.activation_quantizer.scale)
        value = self.head(value, self.stage2.activation_quantizer.scale)
        value = torch.flatten(value, start_dim=1)
        return self.classifier(value, self.head.activation_quantizer.scale)
