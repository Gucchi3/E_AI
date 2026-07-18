"""整数Quantizerを使用する基本レイヤー。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .batch_norm import batch_norm_scale, fold_batch_norm
from .integer import IntegerQuantizer
from .rounding import get_rounding_function


class QuantConv2d(nn.Conv2d):
    """重みを出力チャネル単位でFake Quantizationする畳み込み。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", weight_bits: int = 8, rounding: str = "ties_away_from_zero") -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, channel_size=out_channels, rounding=rounding)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """量子化した重みで畳み込みを行う。"""
        quantized_weight = self.weight_quantizer(self.weight)
        return F.conv2d(value, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class QuantBNConv2d(nn.Module):
    """BatchNorm fold後の重みとbiasを量子化する畳み込み層。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, groups: int = 1, bias: bool = False, padding_mode: str = "zeros", weight_bits: int = 8, rounding: str = "ties_away_from_zero", batch_norm_eps: float = 1e-5, batch_norm_momentum: float = 0.1) -> None:
        super().__init__()
        self.conv             = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.norm             = nn.BatchNorm2d(out_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, channel_size=out_channels, rounding=rounding)
        self._round           = get_rounding_function(rounding)
        self.register_buffer("bias_scale", torch.ones(out_channels, dtype=torch.float32))


    @property
    def weight_scale(self) -> torch.Tensor:
        """fold後の重みに使用したチャンネル別scaleを返す。"""
        return self.weight_quantizer.scale


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        """学習時はBN統計を更新し、評価時はfold済み畳み込みを行う。"""
        if self.training:
            quantized_weight = self._training_weight()
            self._update_bias_scale(input_scale)
            output           = self.conv._conv_forward(value, quantized_weight, self.conv.bias)
            return self.norm(output)

        folded_weight, folded_bias = self.folded_parameters()
        quantized_weight           = self.weight_quantizer(folded_weight)
        self._update_bias_scale(input_scale)
        quantized_bias             = self._quantize_bias(folded_bias)
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
        """入力scaleと重みscaleからint32 biasのscaleを更新する。"""
        flat_input_scale = input_scale.detach().reshape(-1)
        if flat_input_scale.numel() != 1:
            raise ValueError("QuantBNConv2d requires a per-tensor input scale.")
        scale            = flat_input_scale[0].to(dtype=self.bias_scale.dtype) * self.weight_scale.detach().to(dtype=self.bias_scale.dtype)
        epsilon          = torch.finfo(self.bias_scale.dtype).eps
        with torch.no_grad():
            self.bias_scale.copy_(scale.clamp_min(epsilon))


    def _quantize_bias(self, bias: torch.Tensor) -> torch.Tensor:
        """fold後のbiasを入力scaleと重みscaleに対応するint32へ量子化する。"""
        scale   = self.bias_scale.to(device=bias.device, dtype=bias.dtype)
        scaled  = bias / scale
        rounded = scaled + (self._round(scaled) - scaled).detach()
        clipped = torch.clamp(rounded, -(2**31), 2**31 - 1)
        return clipped * scale



class QuantLinear(nn.Linear):
    """重みを出力単位でFake Quantizationする全結合層。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_bits: int = 8, rounding: str = "ties_away_from_zero") -> None:
        super().__init__(in_features, out_features, bias)
        self.weight_quantizer = IntegerQuantizer(bit_width=weight_bits, signed=True, channel_axis=0, channel_size=out_features, rounding=rounding)


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """量子化した重みで全結合演算を行う。"""
        quantized_weight = self.weight_quantizer(self.weight)
        return F.linear(value, quantized_weight, self.bias)
