"""整数Fake Quantization。"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .rounding import RoundingFunction, get_rounding_function


SUPPORTED_BIT_WIDTHS = (2, 4, 8, 16)


@dataclass(frozen=True)
class QuantizedTensor:
    """整数値と量子化パラメータをまとめた結果。"""

    values    : torch.Tensor
    scale     : torch.Tensor
    zero_point: int



class IntegerQuantizer(nn.Module):
    """指定したビット幅で整数Fake Quantizationを行う。"""

    def __init__(self, bit_width: int = 8, signed: bool = True, channel_axis: int | None = None, rounding: str = "ties_away_from_zero", fixed_scale: float | None = None) -> None:
        super().__init__()
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}.")
        if fixed_scale is not None and fixed_scale <= 0.0:
            raise ValueError("fixed_scale must be positive.")

        self.bit_width             = bit_width
        self.signed                = signed
        self.channel_axis          = channel_axis
        self.rounding              = rounding
        self.fixed_scale           = fixed_scale
        self._round                = get_rounding_function(rounding)
        self._last_scale           : torch.Tensor | None = None


    @property
    def qmin(self) -> int:
        """表現可能な最小整数を返す。"""
        return -(2 ** (self.bit_width - 1)) if self.signed else 0


    @property
    def qmax(self) -> int:
        """表現可能な最大整数を返す。"""
        return 2 ** (self.bit_width - 1) - 1 if self.signed else 2**self.bit_width - 1


    @property
    def scale(self) -> torch.Tensor:
        """直前の量子化で使用したscaleを返す。"""
        if self._last_scale is None:
            raise RuntimeError("scale is available after the first quantization.")
        return self._last_scale


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """STEを使ってFake Quantizationを行う。"""
        scale   = self._calculate_scale(value)
        scaled  = value / scale
        rounded = scaled + (self._round(scaled) - scaled).detach()
        clipped = torch.clamp(rounded, self.qmin, self.qmax)

        self._last_scale = scale.detach()

        return clipped * scale


    @torch.no_grad()
    def quantize(self, value: torch.Tensor) -> QuantizedTensor:
        """実際の整数値とscaleを返す。"""
        scale          = self._calculate_scale(value)
        integer_values = self._round(value / scale).clamp(self.qmin, self.qmax)

        self._last_scale = scale.detach()

        return QuantizedTensor(values=integer_values.to(self._integer_dtype()), scale=scale.detach(), zero_point=0)


    def _calculate_scale(self, value: torch.Tensor) -> torch.Tensor:
        """Tensorの範囲からscaleを求める。"""
        if self.fixed_scale is not None:
            return value.new_tensor(self.fixed_scale)

        reduce_dimensions = self._reduce_dimensions(value.ndim)
        source            = value.detach().abs() if self.signed else value.detach().clamp_min(0.0)
        maximum           = source.amax(dim=reduce_dimensions, keepdim=True) if reduce_dimensions else source
        denominator       = self.qmax
        epsilon           = torch.finfo(value.dtype).eps

        return (maximum / denominator).clamp_min(epsilon)


    def _reduce_dimensions(self, ndim: int) -> tuple[int, ...]:
        """scale計算で縮約する次元を返す。"""
        if self.channel_axis is None:
            return tuple(range(ndim))

        channel_axis = self.channel_axis % ndim
        return tuple(dimension for dimension in range(ndim) if dimension != channel_axis)


    def _integer_dtype(self) -> torch.dtype:
        """ビット幅に対応する保存用の型を返す。"""
        if self.signed:
            return torch.int8 if self.bit_width <= 8 else torch.int16
        return torch.uint8 if self.bit_width <= 8 else torch.int32
