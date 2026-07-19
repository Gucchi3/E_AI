"""整数Fake Quantization。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .rounding import get_rounding_function


SUPPORTED_BIT_WIDTHS = (2, 4, 8, 16)


@dataclass(frozen=True)
class QuantizedTensor:
    """整数値と量子化パラメータをまとめた結果。"""

    values    : torch.Tensor
    scale     : torch.Tensor
    zero_point: int



class IntegerQuantizer(nn.Module):
    """指定したビット幅で整数Fake Quantizationを行う。"""

    scale: torch.Tensor

    def __init__(self, bit_width: int = 8, signed: bool = True, channel_axis: int | None = None, channel_size: int | None = None, rounding: str = "ties_away_from_zero", fixed_scale: float | None = None, range_momentum: float | None = None) -> None:
        super().__init__()
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}.")
        if fixed_scale is not None and fixed_scale <= 0.0:
            raise ValueError("fixed_scale must be positive.")
        if channel_axis is not None and (channel_size is None or channel_size <= 0):
            raise ValueError("channel_size must be positive when channel_axis is specified.")
        if channel_axis is None and channel_size is not None:
            raise ValueError("channel_size requires channel_axis.")
        if range_momentum is not None and not 0.0 <= range_momentum < 1.0:
            raise ValueError("range_momentum must be in [0.0, 1.0).")
        if fixed_scale is not None and range_momentum is not None:
            raise ValueError("fixed_scale and range_momentum cannot be used together.")

        state_size          = channel_size if channel_size is not None else 1
        initial_scale       = fixed_scale if fixed_scale is not None else torch.nan
        self.bit_width      = bit_width
        self.signed         = signed
        self.channel_axis   = channel_axis
        self.channel_size   = channel_size
        self.rounding       = rounding
        self.fixed_scale    = fixed_scale
        self.range_momentum = range_momentum
        self._round         = get_rounding_function(rounding)

        self.register_buffer("scale", torch.full((state_size,), initial_scale, dtype=torch.float32))


    @property
    def qmin(self) -> int:
        """表現可能な最小整数を返す。"""
        return -(2 ** (self.bit_width - 1)) if self.signed else 0


    @property
    def qmax(self) -> int:
        """表現可能な最大整数を返す。"""
        return 2 ** (self.bit_width - 1) - 1 if self.signed else 2**self.bit_width - 1


    @property
    def uses_running_scale(self) -> bool:
        """scaleの移動平均を使用するか返す。"""
        return self.range_momentum is not None


    @property
    def scale_initialized(self) -> bool:
        """scaleが初期化済みか返す。"""
        return bool(torch.isfinite(self.scale).all())


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """STEを使ってFake Quantizationを行う。"""
        scale   = self.scale_for(value)
        scaled  = value / scale
        rounded = scaled + (self._round(scaled) - scaled).detach()
        clipped = torch.clamp(rounded, self.qmin, self.qmax)
        return clipped * scale


    @torch.no_grad()
    def quantize(self, value: torch.Tensor) -> QuantizedTensor:
        """実際の整数値とscaleを返す。"""
        broadcast_scale = self.scale_for(value)
        integer_values  = self._round(value / broadcast_scale).clamp(self.qmin, self.qmax)
        return QuantizedTensor(values=integer_values.to(self._integer_dtype()), scale=self.scale.detach().clone(), zero_point=0)


    def scale_for(self, value: torch.Tensor) -> torch.Tensor:
        """rangeを更新し、valueへbroadcastできるscaleを返す。"""
        return self._select_scale(value)


    def _select_scale(self, value: torch.Tensor) -> torch.Tensor:
        """量子化に使用するscaleを選ぶ。"""
        if self.fixed_scale is not None:
            return self._broadcast(self.scale.to(dtype=value.dtype), value.ndim)

        current_scale = self._current_scale(value)
        if self.uses_running_scale:
            if self.training:
                self._update_running_scale(current_scale)
            elif not self.scale_initialized:
                raise RuntimeError("Scale is not initialized. Run the quantizer in train mode before evaluation.")
            selected_scale = self.scale
        else:
            with torch.no_grad():
                self.scale.copy_(current_scale)
            selected_scale = current_scale
        return self._broadcast(selected_scale.to(dtype=value.dtype), value.ndim)


    def _current_scale(self, value: torch.Tensor) -> torch.Tensor:
        """現在のTensorからscaleを求める。"""
        detached = value.detach()
        if self.channel_axis is None:
            minimum = detached.amin().reshape(1)
            maximum = detached.amax().reshape(1)
        else:
            channel_axis      = self.channel_axis % value.ndim
            reduce_dimensions = tuple(dimension for dimension in range(value.ndim) if dimension != channel_axis)
            minimum           = detached.amin(dim=reduce_dimensions)
            maximum           = detached.amax(dim=reduce_dimensions)

        minimum   = minimum.to(dtype=self.scale.dtype)
        maximum   = maximum.to(dtype=self.scale.dtype)
        magnitude = torch.maximum(minimum.abs(), maximum.abs()) if self.signed else maximum.clamp_min(0.0)
        epsilon   = torch.finfo(self.scale.dtype).eps
        return (magnitude / self.qmax).clamp_min(epsilon)


    def _update_running_scale(self, current_scale: torch.Tensor) -> None:
        """scaleを移動平均で更新する。"""
        momentum = self.range_momentum
        if momentum is None:
            raise RuntimeError("range_momentum is required to update running scale.")
        with torch.no_grad():
            current_scale = current_scale.to(dtype=self.scale.dtype)
            if not self.scale_initialized:
                self.scale.copy_(current_scale)
                return

            self.scale.mul_(momentum).add_(current_scale, alpha=1.0 - momentum)


    def _broadcast(self, value: torch.Tensor, ndim: int) -> torch.Tensor:
        """scaleを入力Tensorへbroadcastできる形にする。"""
        if self.channel_axis is None:
            return value.reshape((1,) * ndim)

        channel_axis    = self.channel_axis % ndim
        broadcast_shape = [1] * ndim
        broadcast_shape[channel_axis] = self.channel_size
        return value.reshape(broadcast_shape)


    def _integer_dtype(self) -> torch.dtype:
        """ビット幅に対応する保存用の型を返す。"""
        if self.signed:
            return torch.int8 if self.bit_width <= 8 else torch.int16
        return torch.uint8 if self.bit_width <= 8 else torch.int32
