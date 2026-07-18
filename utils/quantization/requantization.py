"""PULP式の整数multiplierと右shiftによる再量子化。"""

from __future__ import annotations

import torch
from torch import nn

from .rounding import get_rounding_function


INT32_MAX           = 2**31 - 1
INT32_MIN           = -(2**31)
PULP_MULTIPLIER_MAX = 2**15 - 1
PULP_SHIFT_MAX      = 31


def fixed_point_parameters(input_scale: torch.Tensor, output_scale: torch.Tensor, accumulator_bound: torch.Tensor, rounding: str = "ties_away_from_zero") -> tuple[torch.Tensor, torch.Tensor]:
    """scale比率をPULP式のmultiplierと右shiftへ変換する。"""
    ratio = input_scale.detach().to(dtype=torch.float64) / output_scale.detach().to(dtype=torch.float64)
    bound = accumulator_bound.detach().to(device=ratio.device, dtype=torch.float64)
    if not bool(torch.isfinite(ratio).all()) or bool((ratio <= 0.0).any()):
        raise ValueError("Requantization scale ratio must be finite and positive.")
    if not bool(torch.isfinite(bound).all()) or bool((bound < 0.0).any()):
        raise ValueError("Accumulator bound must be finite and non-negative.")
    if ratio.shape != bound.shape:
        raise ValueError("Input scale and accumulator bound must have the same shape.")

    round_function = get_rounding_function(rounding)
    best_error      = torch.full_like(ratio, torch.inf)
    best_multiplier = torch.zeros_like(ratio)
    best_shift      = torch.zeros_like(ratio, dtype=torch.int32)
    parameter_found = torch.zeros_like(ratio, dtype=torch.bool)
    for shift in range(PULP_SHIFT_MAX + 1):
        denominator = float(1 << shift)
        multiplier  = round_function(ratio * denominator)
        error       = torch.abs(multiplier / denominator - ratio)
        valid       = (multiplier >= 1.0) & (multiplier <= PULP_MULTIPLIER_MAX) & (bound * multiplier <= INT32_MAX)
        better      = valid & (error <= best_error)
        best_error      = torch.where(better, error, best_error)
        best_multiplier = torch.where(better, multiplier, best_multiplier)
        best_shift      = torch.where(better, torch.full_like(best_shift, shift), best_shift)
        parameter_found = parameter_found | valid

    if not bool(parameter_found.all()):
        raise OverflowError("No PULP requantization parameter fits the signed int32 product.")
    return best_multiplier.to(dtype=torch.int32), best_shift



class FixedPointRequantizer(nn.Module):
    """PULP式の再量子化を模擬し、実機用parameterを保存する。"""

    multiplier: torch.Tensor
    shift     : torch.Tensor

    def __init__(self, channels: int, bit_width: int = 8, signed: bool = False, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if bit_width <= 0 or bit_width > 32:
            raise ValueError("bit_width must be in [1, 32].")
        self.channels  = channels
        self.bit_width = bit_width
        self.signed    = signed
        self.rounding  = rounding
        self._round    = get_rounding_function(rounding)
        self.register_buffer("multiplier", torch.ones(channels, dtype=torch.int32))
        self.register_buffer("shift", torch.zeros(channels, dtype=torch.int32))


    @property
    def qmin(self) -> int:
        """出力整数の最小値を返す。"""
        return -(2 ** (self.bit_width - 1)) if self.signed else 0


    @property
    def qmax(self) -> int:
        """出力整数の最大値を返す。"""
        return 2 ** (self.bit_width - 1) - 1 if self.signed else 2**self.bit_width - 1


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor, output_scale: torch.Tensor, accumulator_bound: torch.Tensor) -> torch.Tensor:
        """浮動小数点TensorへPULP式の再量子化誤差を加える。"""
        flat_input_scale       = input_scale.detach().reshape(-1)
        flat_output_scale      = output_scale.detach().reshape(-1)
        flat_accumulator_bound = accumulator_bound.detach().reshape(-1).to(dtype=torch.int64)
        if flat_input_scale.numel() != self.channels:
            raise ValueError(f"input_scale must contain {self.channels} channel values.")
        if flat_output_scale.numel() != 1:
            raise ValueError("output_scale must be per-tensor.")
        if flat_accumulator_bound.numel() != self.channels:
            raise ValueError(f"accumulator_bound must contain {self.channels} channel values.")

        input_scale_view   = _channel_view(flat_input_scale.to(device=value.device, dtype=torch.float64), value.ndim)
        input_integer      = self._round(value.detach().to(dtype=torch.float64) / input_scale_view).to(dtype=torch.int64)
        actual_bound       = _channel_maximum(input_integer.abs()).to(device=flat_accumulator_bound.device)
        effective_bound    = torch.maximum(flat_accumulator_bound, actual_bound)
        multiplier, shift  = fixed_point_parameters(flat_input_scale, flat_output_scale[0], effective_bound, self.rounding)
        with torch.no_grad():
            self.multiplier.copy_(multiplier)
            self.shift.copy_(shift)

        output_scale_view = flat_output_scale[0].to(device=value.device, dtype=torch.float64).reshape((1,) * value.ndim)
        multiplier_view   = _channel_view(self.multiplier.to(device=value.device, dtype=torch.int64), value.ndim)
        shift_view        = _channel_view(self.shift.to(device=value.device, dtype=torch.int64), value.ndim)
        product           = input_integer * multiplier_view
        if bool(((product < INT32_MIN) | (product > INT32_MAX)).any()):
            raise OverflowError("PULP requantization product exceeded the signed int32 range.")
        shifted_integer  = torch.bitwise_right_shift(product, shift_view)
        output_integer   = shifted_integer.clamp(self.qmin, self.qmax)
        requantized      = (output_integer.to(dtype=torch.float64) * output_scale_view).to(dtype=value.dtype)
        inside_range     = (shifted_integer >= self.qmin) & (shifted_integer <= self.qmax)
        straight_through = value + (requantized - value).detach()
        return torch.where(inside_range, straight_through, requantized)


def _channel_view(value: torch.Tensor, ndim: int) -> torch.Tensor:
    """チャンネル別TensorをNCHWまたはNCへbroadcastできる形にする。"""
    if ndim < 2:
        raise ValueError("Requantization input must have at least two dimensions.")
    return value.reshape((1, value.numel()) + (1,) * (ndim - 2))


def _channel_maximum(value: torch.Tensor) -> torch.Tensor:
    """NCHWまたはNC Tensorのチャンネル別最大値を返す。"""
    if value.ndim < 2:
        raise ValueError("Requantization input must have at least two dimensions.")
    dimensions = (0,) + tuple(range(2, value.ndim))
    return value.amax(dim=dimensions)
