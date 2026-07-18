"""整数Quantizerを使用する基本レイヤー。"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .batch_norm import batch_norm_scale, fold_batch_norm
from .integer import IntegerQuantizer
from .rounding import get_rounding_function


DEFAULT_ROUNDING = "ties_away_from_zero"
Size2d           = int | tuple[int, int]


class QuantConv2d(nn.Conv2d):
    """重みとbiasをFake Quantizationする畳み込み。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Size2d, stride: Size2d = 1, padding: Size2d = 0, dilation: Size2d = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", weight_bits: int = 8, rounding: str = DEFAULT_ROUNDING) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, channel_size=out_channels, rounding=rounding)
        self._round           = get_rounding_function(rounding)
        self.register_buffer("bias_scale", torch.ones(out_channels, dtype=torch.float32) if bias else None)
        self.register_buffer("bias_integer", torch.zeros(out_channels, dtype=torch.int32) if bias else None)


    @property
    def weight_scale(self) -> torch.Tensor:
        """出力チャンネル単位のweight scaleを返す。"""
        return self.weight_quantizer.scale


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Fake Quantizationした重みとbiasで畳み込みを行う。"""
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_bias   = self.bias
        if self.bias is not None and input_scale is not None:
            self._update_bias_scale(input_scale)
            quantized_bias = _quantize_int32(self.bias, self.bias_scale, self.bias_integer, self._round)
        return F.conv2d(value, quantized_weight, quantized_bias, self.stride, self.padding, self.dilation, self.groups)


    def _update_bias_scale(self, input_scale: torch.Tensor) -> None:
        """入力scaleとweight scaleからbias scaleを更新する。"""
        scale = _accumulator_scale(input_scale, self.weight_scale, "QuantConv2d")
        with torch.no_grad():
            self.bias_scale.copy_(scale)



class QuantBNConv2d(nn.Module):
    """BatchNorm fold後の重みとbiasを量子化する畳み込み層。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Size2d, stride: Size2d = 1, padding: Size2d = 0, dilation: Size2d = 1, groups: int = 1, bias: bool = False, padding_mode="zeros", weight_bits: int = 8, rounding: str = DEFAULT_ROUNDING, batch_norm_eps=1e-5, batch_norm_momentum=0.1) -> None:
        super().__init__()
        self.conv             = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.norm             = nn.BatchNorm2d(out_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, channel_size=out_channels, rounding=rounding)
        self._round           = get_rounding_function(rounding)
        self.register_buffer("bias_scale", torch.ones(out_channels, dtype=torch.float32))
        self.register_buffer("bias_integer", torch.zeros(out_channels, dtype=torch.int32))


    @property
    def weight_scale(self) -> torch.Tensor:
        """fold後の重みに使用したチャンネル別scaleを返す。"""
        return self.weight_quantizer.scale


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        """Fake Quantizationした重みとbiasで畳み込みを行う。"""
        if self.training:
            quantized_weight = self._training_weight()
            self._update_bias_scale(input_scale)
            output = self.conv._conv_forward(value, quantized_weight, self.conv.bias)
            return self.norm(output)

        folded_weight, folded_bias = self.folded_parameters()
        quantized_weight           = self.weight_quantizer(folded_weight)
        self._update_bias_scale(input_scale)
        quantized_bias             = _quantize_int32(folded_bias, self.bias_scale, self.bias_integer, self._round)
        return self.conv._conv_forward(value, quantized_weight, quantized_bias)


    def folded_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """running統計でBatchNormを統合した重みとbiasを返す。"""
        return fold_batch_norm(self.conv.weight, self.conv.bias, self.norm)


    def _training_weight(self) -> torch.Tensor:
        """fold後に量子化してから学習用の畳み込み重みへ戻す。"""
        with torch.no_grad():
            scale = batch_norm_scale(self.norm).detach().to(dtype=self.conv.weight.dtype)
        weight_shape     = (self.conv.out_channels,) + (1,) * (self.conv.weight.ndim - 1)
        folded_weight    = self.conv.weight * scale.reshape(weight_shape)
        quantized_folded = self.weight_quantizer(folded_weight)
        epsilon          = torch.finfo(self.conv.weight.dtype).eps
        valid_scale      = scale.abs() > epsilon
        safe_scale       = torch.where(valid_scale, scale, torch.ones_like(scale))
        unfolded_weight  = quantized_folded / safe_scale.reshape(weight_shape)
        return torch.where(valid_scale.reshape(weight_shape), unfolded_weight, self.conv.weight)


    def _update_bias_scale(self, input_scale: torch.Tensor) -> None:
        """入力scaleとweight scaleからbias scaleを更新する。"""
        scale = _accumulator_scale(input_scale, self.weight_scale, "QuantBNConv2d")
        with torch.no_grad():
            self.bias_scale.copy_(scale)



class QuantLinear(nn.Linear):
    """重みとbiasをFake Quantizationする全結合層。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_bits: int = 8, rounding: str = DEFAULT_ROUNDING) -> None:
        super().__init__(in_features, out_features, bias)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, channel_size=out_features, rounding=rounding)
        self._round           = get_rounding_function(rounding)
        self.register_buffer("bias_scale", torch.ones(out_features, dtype=torch.float32) if bias else None)
        self.register_buffer("bias_integer", torch.zeros(out_features, dtype=torch.int32) if bias else None)


    @property
    def weight_scale(self) -> torch.Tensor:
        """出力単位のweight scaleを返す。"""
        return self.weight_quantizer.scale


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Fake Quantizationした重みとbiasで全結合演算を行う。"""
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_bias   = self.bias
        if self.bias is not None and input_scale is not None:
            self._update_bias_scale(input_scale)
            quantized_bias = _quantize_int32(self.bias, self.bias_scale, self.bias_integer, self._round)
        return F.linear(value, quantized_weight, quantized_bias)


    def _update_bias_scale(self, input_scale: torch.Tensor) -> None:
        """入力scaleとweight scaleからbias scaleを更新する。"""
        scale = _accumulator_scale(input_scale, self.weight_scale, "QuantLinear")
        with torch.no_grad():
            self.bias_scale.copy_(scale)


def _accumulator_scale(input_scale: torch.Tensor, weight_scale: torch.Tensor, layer_name: str) -> torch.Tensor:
    """入力scaleとweight scaleからaccumulator scaleを求める。"""
    flat_input_scale = input_scale.detach().reshape(-1)
    if flat_input_scale.numel() != 1:
        raise ValueError(f"{layer_name} requires a per-tensor input scale.")
    minimum_scale = torch.finfo(weight_scale.dtype).tiny
    return (flat_input_scale[0].to(device=weight_scale.device, dtype=weight_scale.dtype) * weight_scale.detach()).clamp_min(minimum_scale)


def _quantize_int32(value: torch.Tensor, scale: torch.Tensor, integer_buffer: torch.Tensor, round_function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """値をint32格子へFake Quantizationして整数値を保存する。"""
    value_scale   = scale.to(device=value.device, dtype=value.dtype)
    scaled        = value / value_scale
    integer_value = round_function(scaled).clamp(-(2**31), 2**31 - 1)
    rounded       = scaled + (integer_value - scaled).detach()
    with torch.no_grad():
        integer_buffer.copy_(integer_value.detach().to(dtype=torch.int32))
    return rounded * value_scale
