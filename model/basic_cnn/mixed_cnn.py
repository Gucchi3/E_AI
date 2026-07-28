"""INT8とFP4を組み合わせたCIFAR-10 CNN。"""

from __future__ import annotations

import torch
from torch import nn

from utils.quantization import IntegerQuantizer, QuantLinear

from ..blocks import BlockQuantization, QuantConvBlock


class MixedCNN(nn.Module):
    """先頭と末尾をINT8、中間をFP4へFake Quantizationする分類モデル。"""

    def __init__(self, num_classes: int = 10, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        if input_bits != 8:
            raise ValueError("MixedCNN input_bits must be 8.")

        feature_size         = image_size // 8
        int8_to_fp4          = BlockQuantization("integer", "fp4", 8, 4)
        fp4                  = BlockQuantization("fp4", "fp4", 4, 4)
        fp4_to_int8          = BlockQuantization("fp4", "integer", 4, 8)
        self.input_quantizer = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, fixed_scale=1.0 / 255.0)
        self.stem            = QuantConvBlock(3, 16, stride=1, quantization=int8_to_fp4, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage1          = QuantConvBlock(16, 32, stride=2, quantization=fp4, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage2          = QuantConvBlock(32, 64, stride=2, quantization=fp4, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.head            = QuantConvBlock(64, 64, stride=2, quantization=fp4_to_int8, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.classifier      = QuantLinear(64 * feature_size * feature_size, num_classes, weight_bits=8, rounding=rounding, quantizer="integer")


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """uint8画像を模擬してクラスごとのlogitを返す。"""
        value = self.input_quantizer(value)
        value = self.stem(value, self.input_quantizer.scale)
        value = self.stage1(value, self.stem.activation_quantizer.scale)
        value = self.stage2(value, self.stage1.activation_quantizer.scale)
        value = self.head(value, self.stage2.activation_quantizer.scale)
        value = torch.flatten(value, start_dim=1)
        return self.classifier(value, self.head.activation_quantizer.scale)
