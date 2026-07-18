"""QAT学習を確認するための小型CNN。"""

from __future__ import annotations

import torch
from torch import nn

from utils.quantization import FixedPointRequantizer, IntegerQuantizer, QuantBNConv2d, QuantLinear


class QuantConvBlock(nn.Module):
    """BN fold対応の量子化畳み込みと活性量子化をまとめたブロック。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, weight_bits: int, activation_bits: int, input_bits: int, rounding: str, activation_range_momentum: float, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv                 = QuantBNConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, weight_bits=weight_bits, rounding=rounding, input_bits=input_bits, input_signed=False)
        self.activation           = nn.ReLU(inplace=False)
        self.activation_quantizer = IntegerQuantizer(bit_width=activation_bits, signed=False, rounding=rounding, range_momentum=activation_range_momentum)
        self.requantizer          = FixedPointRequantizer(channels=out_channels, bit_width=activation_bits, signed=False, rounding=rounding)


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        """Fake Quantization畳み込み後の活性を再量子化する。"""
        value        = self.activation(self.conv(value, input_scale))
        output_scale = self.activation_quantizer.scale_for(value)
        return self.requantizer(value, self.conv.bias_scale, output_scale, self.conv.accumulator_bound)



class TinyQATCNN(nn.Module):
    """BN fold対応畳み込みと線形層で構成したQATモデル。"""

    def __init__(self, num_classes: int = 10, weight_bits: int = 8, activation_bits: int = 8, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4.")
        final_kernel         = image_size // 4
        self.input_quantizer = IntegerQuantizer(bit_width=input_bits, signed=False, rounding=rounding, fixed_scale=1.0 / (2**input_bits - 1))
        self.stem            = QuantConvBlock(3, 16, stride=1, weight_bits=weight_bits, activation_bits=activation_bits, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage1          = QuantConvBlock(16, 32, stride=2, weight_bits=weight_bits, activation_bits=activation_bits, input_bits=activation_bits, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage2          = QuantConvBlock(32, 64, stride=2, weight_bits=weight_bits, activation_bits=activation_bits, input_bits=activation_bits, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.head            = QuantConvBlock(64, 64, stride=1, weight_bits=weight_bits, activation_bits=activation_bits, input_bits=activation_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, kernel_size=final_kernel, padding=0)
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
