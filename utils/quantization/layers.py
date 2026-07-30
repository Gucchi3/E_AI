"""整数またはFP4 Quantizerを使用する基本レイヤー。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .fp4 import FP4Quantizer
from .integer import IntegerQuantizer
from .rounding import get_rounding_function


DEFAULT_ROUNDING = "ties_away_from_zero"
Size2d           = int | tuple[int, int]
QuantizerName    = Literal["integer", "fp4"]
WeightQuantizer  = IntegerQuantizer | FP4Quantizer
INT32_MIN        = -(2**31)
INT32_MAX        = 2**31 - 1


class QuantConv2d(nn.Conv2d):
    """重みとbiasをFake Quantizationする畳み込み。"""

    weight_quantizer: WeightQuantizer
    bias_scale      : torch.Tensor | None
    bias_integer    : torch.Tensor | None

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Size2d, stride: Size2d = 1, padding: Size2d = 0, dilation: Size2d = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", weight_bits: int = 8, rounding: str = DEFAULT_ROUNDING, quantizer: QuantizerName = "integer") -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.weight_quantizer = _create_weight_quantizer(quantizer, weight_bits, out_channels, rounding)
        self._round           = get_rounding_function(rounding)
        self.register_buffer("bias_scale", torch.ones(out_channels, dtype=torch.float32) if bias else None)
        self.register_buffer("bias_integer", torch.zeros(out_channels, dtype=torch.int32) if bias else None)


    @property
    def weight_scale(self) -> torch.Tensor:
        """出力チャンネル単位のweight scaleを返す。"""
        return self.weight_quantizer.scale


    @property
    def accumulator_scale(self) -> torch.Tensor:
        """入力と重みの積に対応する出力チャンネル単位のINT32 accumulator scaleを返す。"""
        if self.bias_scale is None:
            raise RuntimeError("QuantConv2d accumulator scale is unavailable when bias=False.")
        return self.bias_scale


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Fake Quantizationした重みとbiasで畳み込みを行う。"""
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_bias   = self.bias
        if self.bias is not None:
            if input_scale is None:
                raise ValueError("input_scale is required when QuantConv2d has bias.")
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



class QuantLinear(nn.Linear):
    """重みとbiasをFake Quantizationする全結合層。"""

    weight_quantizer: WeightQuantizer
    bias_scale      : torch.Tensor | None
    bias_integer    : torch.Tensor | None

    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_bits: int = 8, rounding: str = DEFAULT_ROUNDING, quantizer: QuantizerName = "integer") -> None:
        super().__init__(in_features, out_features, bias)
        self.weight_quantizer = _create_weight_quantizer(quantizer, weight_bits, out_features, rounding)
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
        if self.bias is not None:
            if input_scale is None:
                raise ValueError("input_scale is required when QuantLinear has bias.")
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


def _create_weight_quantizer(name: QuantizerName, bit_width: int, channels: int, rounding: str) -> WeightQuantizer:
    """名前からチャンネル別の重みQuantizerを生成する。"""
    if name == "integer":
        return IntegerQuantizer(bit_width=bit_width, signed=True, channel_axis=0, channel_size=channels, rounding=rounding)
    if name == "fp4":
        if bit_width != 4:
            raise ValueError("FP4 weight_bits must be 4.")
        return FP4Quantizer(channel_axis=0, channel_size=channels, rounding=rounding)
    raise ValueError(f"Unsupported quantizer: {name!r}.")


def _accumulator_scale(input_scale: torch.Tensor, weight_scale: torch.Tensor, layer_name: str) -> torch.Tensor:
    """入力scaleとweight scaleからaccumulator scaleを求める。"""
    flat_input_scale = input_scale.detach().reshape(-1)
    if flat_input_scale.numel() != 1:
        raise ValueError(f"{layer_name} requires a per-tensor input scale.")
    if not bool(torch.isfinite(flat_input_scale).all()) or bool((flat_input_scale <= 0.0).any()):
        raise ValueError(f"{layer_name} input_scale must be finite and positive.")
    if not bool(torch.isfinite(weight_scale).all()) or bool((weight_scale <= 0.0).any()):
        raise ValueError(f"{layer_name} weight_scale must be finite and positive.")
    minimum_scale = torch.finfo(weight_scale.dtype).tiny
    scale         = flat_input_scale[0].to(device=weight_scale.device, dtype=weight_scale.dtype) * weight_scale.detach()
    if not bool(torch.isfinite(scale).all()):
        raise OverflowError(f"{layer_name} bias scale exceeded the floating-point range.")
    return scale.clamp_min(minimum_scale)


def _quantize_int32(value: torch.Tensor, scale: torch.Tensor, integer_buffer: torch.Tensor, round_function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """値をint32格子へFake Quantizationして整数値を保存する。"""
    value_scale     = scale.to(device=value.device, dtype=value.dtype)
    integer_value   = _integer_int32(value, scale, round_function)
    quantized_value = integer_value.to(dtype=value.dtype) * value_scale
    with torch.no_grad():
        integer_buffer.copy_(integer_value.detach())
    return value + (quantized_value - value).detach()


def _integer_int32(value: torch.Tensor, scale: torch.Tensor, round_function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """値をsigned int32へ量子化する。"""
    value_scale = scale.to(device=value.device, dtype=value.dtype)
    return round_function(value / value_scale).clamp(INT32_MIN, INT32_MAX).to(dtype=torch.int32)
