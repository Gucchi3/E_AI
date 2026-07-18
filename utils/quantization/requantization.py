"""整数multiplierと右shiftによる再量子化。"""

from __future__ import annotations

import torch
from torch import nn

from .rounding import get_rounding_function


MULTIPLIER_FRACTION_BITS = 31


def fixed_point_parameters(input_scale: torch.Tensor, output_scale: torch.Tensor, rounding: str = "ties_away_from_zero") -> tuple[torch.Tensor, torch.Tensor]:
    """scale比率をsigned int32 multiplierと右shiftへ変換する。"""
    ratio = input_scale.detach().to(dtype=torch.float64) / output_scale.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(ratio).all()) or bool((ratio <= 0.0).any()):
        raise ValueError("Requantization scale ratio must be finite and positive.")

    round_function      = get_rounding_function(rounding)
    mantissa, exponent  = torch.frexp(ratio)
    multiplier          = round_function(mantissa * float(1 << MULTIPLIER_FRACTION_BITS))
    shift               = MULTIPLIER_FRACTION_BITS - exponent.to(dtype=torch.int32)
    multiplier_overflow = multiplier == 1 << MULTIPLIER_FRACTION_BITS
    multiplier          = torch.where(multiplier_overflow, multiplier / 2.0, multiplier)
    shift               = torch.where(multiplier_overflow, shift - 1, shift)
    if bool((shift < 0).any()) or bool((shift > 62).any()):
        raise ValueError("Requantization right shift must be in [0, 62].")
    return multiplier.to(dtype=torch.int32), shift



class FixedPointRequantizer(nn.Module):
    """Q31再量子化を模擬し、実機用parameterを保存する。"""

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


    def forward(self, value: torch.Tensor, input_scale: torch.Tensor, output_scale: torch.Tensor) -> torch.Tensor:
        """浮動小数点TensorへQ31再量子化誤差を加える。"""
        flat_input_scale  = input_scale.detach().reshape(-1)
        flat_output_scale = output_scale.detach().reshape(-1)
        if flat_input_scale.numel() != self.channels:
            raise ValueError(f"input_scale must contain {self.channels} channel values.")
        if flat_output_scale.numel() != 1:
            raise ValueError("output_scale must be per-tensor.")

        multiplier, shift = fixed_point_parameters(flat_input_scale, flat_output_scale[0], self.rounding)
        with torch.no_grad():
            self.multiplier.copy_(multiplier)
            self.shift.copy_(shift)

        input_scale_view  = _channel_view(flat_input_scale.to(device=value.device, dtype=torch.float64), value.ndim)
        output_scale_view = flat_output_scale[0].to(device=value.device, dtype=torch.float64).reshape((1,) * value.ndim)
        multiplier_view   = _channel_view(self.multiplier.to(device=value.device, dtype=torch.float64), value.ndim)
        shift_view        = _channel_view(self.shift.to(device=value.device), value.ndim)
        input_integer     = self._round(value.detach().to(dtype=torch.float64) / input_scale_view)
        scaled_integer    = torch.ldexp(input_integer * multiplier_view, -shift_view)
        rounded_integer   = self._round(scaled_integer)
        output_integer    = rounded_integer.clamp(self.qmin, self.qmax)
        requantized       = (output_integer * output_scale_view).to(dtype=value.dtype)
        inside_range      = (rounded_integer >= self.qmin) & (rounded_integer <= self.qmax)
        straight_through  = value + (requantized - value).detach()
        return torch.where(inside_range, straight_through, requantized)


def _channel_view(value: torch.Tensor, ndim: int) -> torch.Tensor:
    """チャンネル別TensorをNCHWまたはNCへbroadcastできる形にする。"""
    if ndim < 2:
        raise ValueError("Requantization input must have at least two dimensions.")
    return value.reshape((1, value.numel()) + (1,) * (ndim - 2))
