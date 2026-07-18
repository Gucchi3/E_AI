"""整数Quantizerを使用する基本レイヤー。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .integer import IntegerQuantizer


class QuantConv2d(nn.Conv2d):
    """重みを出力チャネル単位でFake Quantizationする畳み込み。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", weight_bits: int = 8, rounding: str = "ties_away_from_zero") -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, rounding=rounding)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """量子化した重みで畳み込みを行う。"""
        quantized_weight = self.weight_quantizer(self.weight)
        return F.conv2d(value, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class QuantLinear(nn.Linear):
    """重みを出力単位でFake Quantizationする全結合層。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_bits: int = 8, rounding: str = "ties_away_from_zero") -> None:
        super().__init__(in_features, out_features, bias)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, rounding=rounding)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """量子化した重みで全結合演算を行う。"""
        quantized_weight = self.weight_quantizer(self.weight)
        return F.linear(value, quantized_weight, self.bias)
