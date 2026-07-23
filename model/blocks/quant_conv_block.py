"""QATモデルで使用する量子化畳み込みブロック。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from utils.quantization import FP4Quantizer, IntegerQuantizer, QuantConv2d, QuantizerName


ActivationQuantizer = IntegerQuantizer | FP4Quantizer


@dataclass(frozen=True)
class BlockQuantization:
    """畳み込み重みと出力活性の量子化設定。"""

    weight_quantizer    : QuantizerName
    activation_quantizer: QuantizerName
    weight_bits         : int
    activation_bits     : int


class QuantConvBlock(nn.Module):
    """BN吸収済み重みを使用する量子化畳み込みと活性量子化をまとめる。"""

    activation_quantizer: ActivationQuantizer

    def __init__(self, in_channels: int, out_channels: int, stride: int, quantization: BlockQuantization, rounding: str, activation_range_momentum: float, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv                 = QuantConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, weight_bits=quantization.weight_bits, rounding=rounding, quantizer=quantization.weight_quantizer)
        self.activation           = nn.ReLU(inplace=False)
        self.activation_quantizer = _create_activation_quantizer(quantization, rounding, activation_range_momentum)


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        """畳み込み後の活性をFake Quantizationする。"""
        value = self.activation(self.conv(value, input_scale))
        return self.activation_quantizer(value)


def _create_activation_quantizer(quantization: BlockQuantization, rounding: str, range_momentum: float) -> ActivationQuantizer:
    """ブロック設定から出力活性のQuantizerを生成する。"""
    if quantization.activation_quantizer == "integer":
        return IntegerQuantizer(bit_width=quantization.activation_bits, signed=False, rounding=rounding, range_momentum=range_momentum)
    if quantization.activation_quantizer == "fp4":
        if quantization.activation_bits != 4:
            raise ValueError("FP4 activation_bits must be 4.")
        return FP4Quantizer(rounding=rounding, range_momentum=range_momentum)
    raise ValueError(f"Unsupported activation quantizer: {quantization.activation_quantizer!r}.")
