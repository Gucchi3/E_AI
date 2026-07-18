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
INT32_MIN        = -(2**31)
INT32_MAX        = 2**31 - 1


class QuantConv2d(nn.Conv2d):
    """重みとbiasをFake Quantizationする畳み込み。"""

    weight_quantizer: IntegerQuantizer
    bias_scale      : torch.Tensor | None
    bias_integer    : torch.Tensor | None

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
            if self.bias_scale is None or self.bias_integer is None:
                raise RuntimeError("QuantConv2d bias buffers are not initialized.")
            self._update_bias_scale(input_scale)
            quantized_bias = _quantize_int32(self.bias, self.bias_scale, self.bias_integer, self._round)
        return F.conv2d(value, quantized_weight, quantized_bias, self.stride, self.padding, self.dilation, self.groups)


    def _update_bias_scale(self, input_scale: torch.Tensor) -> None:
        """入力scaleとweight scaleからbias scaleを更新する。"""
        if self.bias_scale is None:
            raise RuntimeError("QuantConv2d bias scale is not initialized.")
        scale = _accumulator_scale(input_scale, self.weight_scale, "QuantConv2d")
        with torch.no_grad():
            self.bias_scale.copy_(scale)



class QuantBNConv2d(nn.Module):
    """BatchNorm fold後の重みとbiasを量子化する畳み込み層。"""

    conv              : nn.Conv2d
    norm              : nn.BatchNorm2d
    weight_quantizer  : IntegerQuantizer
    bias_scale       : torch.Tensor
    bias_integer     : torch.Tensor
    accumulator_bound: torch.Tensor

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Size2d, stride: Size2d = 1, padding: Size2d = 0, dilation: Size2d = 1, groups: int = 1, bias: bool = False, padding_mode="zeros", weight_bits: int = 8, rounding: str = DEFAULT_ROUNDING, batch_norm_eps=1e-5, batch_norm_momentum=0.1, input_bits: int = 8, input_signed: bool = False) -> None:
        super().__init__()
        if input_bits <= 0 or input_bits > 16:
            raise ValueError("input_bits must be in [1, 16].")
        self.conv             = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.norm             = nn.BatchNorm2d(out_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, channel_size=out_channels, rounding=rounding)
        self.input_bits       = input_bits
        self.input_signed     = input_signed
        self._round           = get_rounding_function(rounding)
        self.register_buffer("bias_scale", torch.ones(out_channels, dtype=torch.float32))
        self.register_buffer("bias_integer", torch.zeros(out_channels, dtype=torch.int32))
        self.register_buffer("accumulator_bound", torch.zeros(out_channels, dtype=torch.int32))


    @property
    def weight_scale(self) -> torch.Tensor:
        """fold後の重みに使用したチャンネル別scaleを返す。"""
        return self.weight_quantizer.scale


    @property
    def input_qmin(self) -> int:
        """入力整数の最小値を返す。"""
        return -(2 ** (self.input_bits - 1)) if self.input_signed else 0


    @property
    def input_qmax(self) -> int:
        """入力整数の最大値を返す。"""
        return 2 ** (self.input_bits - 1) - 1 if self.input_signed else 2**self.input_bits - 1


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        """Fake Quantizationした重みとbiasで畳み込みを行う。"""
        folded_weight, folded_bias = self.folded_parameters()
        if self.training:
            with torch.no_grad():
                scale = batch_norm_scale(self.norm).detach().to(dtype=self.conv.weight.dtype)
            weight_shape  = (self.conv.out_channels,) + (1,) * (self.conv.weight.ndim - 1)
            folded_weight = self.conv.weight * scale.reshape(weight_shape)
        quantized_weight           = self.weight_quantizer(folded_weight)
        self._update_bias_scale(input_scale)
        self._update_integer_parameters(quantized_weight, folded_bias)
        if self.training:
            training_weight = self._training_weight(quantized_weight)
            output = self.conv._conv_forward(value, training_weight, self.conv.bias)
            return self.norm(output)

        quantized_bias             = _quantize_int32(folded_bias, self.bias_scale, self.bias_integer, self._round)
        return self.conv._conv_forward(value, quantized_weight, quantized_bias)


    def folded_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """running統計でBatchNormを統合した重みとbiasを返す。"""
        return fold_batch_norm(self.conv.weight, self.conv.bias, self.norm)


    def _training_weight(self, quantized_folded: torch.Tensor) -> torch.Tensor:
        """fold後に量子化してから学習用の畳み込み重みへ戻す。"""
        with torch.no_grad():
            scale = batch_norm_scale(self.norm).detach().to(dtype=self.conv.weight.dtype)
        weight_shape    = (self.conv.out_channels,) + (1,) * (self.conv.weight.ndim - 1)
        epsilon         = torch.finfo(self.conv.weight.dtype).eps
        valid_scale     = scale.abs() > epsilon
        safe_scale      = torch.where(valid_scale, scale, torch.ones_like(scale))
        unfolded_weight = quantized_folded / safe_scale.reshape(weight_shape)
        return torch.where(valid_scale.reshape(weight_shape), unfolded_weight, self.conv.weight)


    def _update_bias_scale(self, input_scale: torch.Tensor) -> None:
        """入力scaleとweight scaleからbias scaleを更新する。"""
        scale = _accumulator_scale(input_scale, self.weight_scale, "QuantBNConv2d")
        with torch.no_grad():
            self.bias_scale.copy_(scale)


    def _update_integer_parameters(self, quantized_weight: torch.Tensor, folded_bias: torch.Tensor) -> None:
        """整数biasとaccumulator上限を更新する。"""
        weight_shape   = (self.conv.out_channels,) + (1,) * (quantized_weight.ndim - 1)
        weight_scale   = self.weight_scale.detach().reshape(weight_shape).to(device=quantized_weight.device, dtype=quantized_weight.dtype)
        integer_weight = self._round(quantized_weight.detach() / weight_scale).clamp(self.weight_quantizer.qmin, self.weight_quantizer.qmax).to(dtype=torch.int64)
        integer_bias   = _integer_int32(folded_bias.detach(), self.bias_scale, self._round)
        lower          = torch.minimum(integer_weight * self.input_qmin, integer_weight * self.input_qmax).sum(dim=tuple(range(1, integer_weight.ndim))) + integer_bias.to(dtype=torch.int64)
        upper          = torch.maximum(integer_weight * self.input_qmin, integer_weight * self.input_qmax).sum(dim=tuple(range(1, integer_weight.ndim))) + integer_bias.to(dtype=torch.int64)
        bound          = torch.maximum(lower.abs(), upper.abs())
        if bool((bound > INT32_MAX).any()):
            raise OverflowError("QuantBNConv2d accumulator exceeded the signed int32 range.")
        with torch.no_grad():
            self.bias_integer.copy_(integer_bias)
            self.accumulator_bound.copy_(bound.to(dtype=torch.int32))



class QuantLinear(nn.Linear):
    """重みとbiasをFake Quantizationする全結合層。"""

    weight_quantizer: IntegerQuantizer
    bias_scale      : torch.Tensor | None
    bias_integer    : torch.Tensor | None

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
            if self.bias_scale is None or self.bias_integer is None:
                raise RuntimeError("QuantLinear bias buffers are not initialized.")
            self._update_bias_scale(input_scale)
            quantized_bias = _quantize_int32(self.bias, self.bias_scale, self.bias_integer, self._round)
        return F.linear(value, quantized_weight, quantized_bias)


    def _update_bias_scale(self, input_scale: torch.Tensor) -> None:
        """入力scaleとweight scaleからbias scaleを更新する。"""
        if self.bias_scale is None:
            raise RuntimeError("QuantLinear bias scale is not initialized.")
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
    integer_value = _integer_int32(value, scale, round_function)
    rounded       = scaled + (integer_value - scaled).detach()
    with torch.no_grad():
        integer_buffer.copy_(integer_value.detach())
    return rounded * value_scale


def _integer_int32(value: torch.Tensor, scale: torch.Tensor, round_function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """値をsigned int32へ量子化する。"""
    value_scale = scale.to(device=value.device, dtype=value.dtype)
    return round_function(value / value_scale).clamp(INT32_MIN, INT32_MAX).to(dtype=torch.int32)
