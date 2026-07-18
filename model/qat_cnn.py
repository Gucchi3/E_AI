"""QAT学習を確認するための小型CNN。"""

from __future__ import annotations

import torch
from torch import nn

from utils.quantization import IntegerQuantizer, QuantConv2d, QuantLinear


class QuantConvBlock(nn.Module):
    """量子化畳み込みと活性量子化をまとめたブロック。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, weight_bits: int, activation_bits: int, rounding: str) -> None:
        super().__init__()
        self.conv                 = QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, weight_bits=weight_bits, rounding=rounding)
        self.norm                 = nn.BatchNorm2d(out_channels)
        self.activation           = nn.ReLU(inplace=False)
        self.activation_quantizer = IntegerQuantizer(bit_width=activation_bits, signed=False, rounding=rounding)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """畳み込み後の非負活性を量子化する。"""
        value = self.conv(value)
        value = self.norm(value)
        value = self.activation(value)
        return self.activation_quantizer(value)



class TinyQATCNN(nn.Module):
    """CIFAR-10用の小型QATモデル。"""

    def __init__(self, num_classes: int = 10, weight_bits: int = 8, activation_bits: int = 8, input_bits: int = 8, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        self.input_quantizer = IntegerQuantizer(bit_width=input_bits, signed=False, rounding=rounding, fixed_scale=1.0 / (2**input_bits - 1))
        self.stem            = QuantConvBlock(3, 8, stride=1, weight_bits=weight_bits, activation_bits=activation_bits, rounding=rounding)
        self.stage1          = QuantConvBlock(8, 16, stride=2, weight_bits=weight_bits, activation_bits=activation_bits, rounding=rounding)
        self.stage2          = QuantConvBlock(16, 32, stride=2, weight_bits=weight_bits, activation_bits=activation_bits, rounding=rounding)
        self.pool            = nn.AdaptiveAvgPool2d(1)
        self.pool_quantizer  = IntegerQuantizer(bit_width=activation_bits, signed=False, rounding=rounding)
        self.classifier      = QuantLinear(32, num_classes, weight_bits=weight_bits, rounding=rounding)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """uint8画像を模擬してクラスごとのlogitを返す。"""
        value = self.input_quantizer(value)
        value = self.stem(value)
        value = self.stage1(value)
        value = self.stage2(value)
        value = self.pool(value)
        value = self.pool_quantizer(value)
        value = torch.flatten(value, start_dim=1)
        return self.classifier(value)
