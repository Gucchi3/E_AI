"""QAT学習を確認するための小型CNN。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from utils.quantization import FP4Quantizer, IntegerQuantizer, QuantBNConv2d, QuantizerName, QuantLinear


ActivationQuantizer = IntegerQuantizer | FP4Quantizer


@dataclass(frozen=True)
class BlockQuantization:
    """畳み込み重みと出力活性の量子化設定。"""

    weight_quantizer    : QuantizerName
    activation_quantizer: QuantizerName
    weight_bits         : int
    activation_bits     : int



class QuantConvBlock(nn.Module):
    """BN fold対応の量子化畳み込みと活性量子化をまとめたブロック。"""

    activation_quantizer: ActivationQuantizer

    def __init__(self, in_channels: int, out_channels: int, stride: int, quantization: BlockQuantization, rounding: str, activation_range_momentum: float, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv                 = QuantBNConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, weight_bits=quantization.weight_bits, rounding=rounding, quantizer=quantization.weight_quantizer)
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



class TinyQATCNN(nn.Module):
    """BN fold対応畳み込みと線形層で構成したQATモデル。"""

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



class TinyMixedQATCNN(nn.Module):
    """先頭と末尾をINT8、中間をFP4にしたQATモデル。"""

    def __init__(self, num_classes: int = 10, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4.")
        if input_bits != 8:
            raise ValueError("TinyMixedQATCNN input_bits must be 8.")

        final_kernel         = image_size // 4
        int8_to_fp4          = BlockQuantization("integer", "fp4", 8, 4)
        fp4                  = BlockQuantization("fp4", "fp4", 4, 4)
        fp4_to_int8          = BlockQuantization("fp4", "integer", 4, 8)
        self.input_quantizer = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, fixed_scale=1.0 / 255.0)
        self.stem            = QuantConvBlock(3, 16, stride=1, quantization=int8_to_fp4, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage1          = QuantConvBlock(16, 32, stride=2, quantization=fp4, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage2          = QuantConvBlock(32, 64, stride=2, quantization=fp4, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.head            = QuantConvBlock(64, 64, stride=1, quantization=fp4_to_int8, rounding=rounding, activation_range_momentum=activation_range_momentum, kernel_size=final_kernel, padding=0)
        self.classifier      = QuantLinear(64, num_classes, weight_bits=8, rounding=rounding, quantizer="integer")


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """混合精度QATでクラスごとのlogitを返す。"""
        value = self.input_quantizer(value)
        value = self.stem(value, self.input_quantizer.scale)
        value = self.stage1(value, self.stem.activation_quantizer.scale)
        value = self.stage2(value, self.stage1.activation_quantizer.scale)
        value = self.head(value, self.stage2.activation_quantizer.scale)
        value = torch.flatten(value, start_dim=1)
        return self.classifier(value, self.head.activation_quantizer.scale)
