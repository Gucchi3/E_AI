"""UFP4 E2M2 Fake Quantization。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .rounding import get_rounding_function


UFP4_FORMAT    = "E2M2"
UFP4_MAX_VALUE = 7.0
E2M2_VALUES    = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0)


@dataclass(frozen=True)
class UFP4QuantizedTensor:
    """4bit codeとscaleをまとめた結果。"""

    values: torch.Tensor
    scale : torch.Tensor



class UFP4Quantizer(nn.Module):
    """符号なしE2M2形式でUFP4 Fake Quantizationを行う。"""

    scale: torch.Tensor

    def __init__(self, channel_axis: int | None = None, channel_size: int | None = None, rounding: str = "ties_away_from_zero", fixed_scale: float | None = None, range_momentum: float | None = None) -> None:
        super().__init__()
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
        self.channel_axis   = channel_axis
        self.channel_size   = channel_size
        self.rounding       = rounding
        self.fixed_scale    = fixed_scale
        self.range_momentum = range_momentum
        get_rounding_function(rounding)

        self.register_buffer("scale", torch.full((state_size,), initial_scale, dtype=torch.float32))


    @property
    def uses_running_scale(self) -> bool:
        """scaleの移動平均を使用するか返す。"""
        return self.range_momentum is not None


    @property
    def scale_initialized(self) -> bool:
        """scaleが初期化済みか返す。"""
        return bool(torch.isfinite(self.scale).all())


    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """STEを使ってUFP4 Fake Quantizationを行う。"""
        scale      = self.scale_for(value)
        normalized = value / scale
        bounded    = normalized.clamp(0.0, UFP4_MAX_VALUE)
        quantized  = decode_e2m2(encode_e2m2(bounded, self.rounding), dtype=value.dtype)
        return (bounded + (quantized - bounded).detach()) * scale


    @torch.no_grad()
    def quantize(self, value: torch.Tensor) -> UFP4QuantizedTensor:
        """4bit codeとscaleを返す。"""
        scale   = self.scale_for(value)
        encoded = encode_e2m2(value / scale, self.rounding)
        return UFP4QuantizedTensor(values=encoded, scale=self.scale.detach().clone())


    def dequantize(self, value: UFP4QuantizedTensor) -> torch.Tensor:
        """4bit codeとscaleから浮動小数点Tensorを復元する。"""
        decoded = decode_e2m2(value.values)
        scale   = self._broadcast(value.scale.to(device=decoded.device, dtype=decoded.dtype), decoded.ndim)
        return decoded * scale


    def scale_for(self, value: torch.Tensor) -> torch.Tensor:
        """scaleを更新し、valueへbroadcastできる形で返す。"""
        self._validate_value(value)
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


    def _validate_value(self, value: torch.Tensor) -> None:
        """Tensorの型、形、値を検証する。"""
        if not value.is_floating_point():
            raise TypeError("UFP4Quantizer input must be a floating-point Tensor.")
        if value.ndim == 0 and self.channel_axis is not None:
            raise ValueError("Per-channel quantization requires a Tensor with at least one dimension.")
        if not bool(torch.isfinite(value).all()):
            nan_count     = int(torch.isnan(value).sum().item())
            positive_inf = int(torch.isposinf(value).sum().item())
            negative_inf = int(torch.isneginf(value).sum().item())
            raise ValueError(f"UFP4Quantizer input contains non-finite values: nan={nan_count}, +inf={positive_inf}, -inf={negative_inf}.")
        if self.channel_axis is not None:
            channel_axis = self.channel_axis % value.ndim
            if value.size(channel_axis) != self.channel_size:
                raise ValueError("The input channel size does not match channel_size.")


    def _current_scale(self, value: torch.Tensor) -> torch.Tensor:
        """現在のTensorから符号なしscaleを求める。"""
        detached = value.detach()
        if self.channel_axis is None:
            maximum = detached.amax().reshape(1)
        else:
            channel_axis      = self.channel_axis % value.ndim
            reduce_dimensions = tuple(dimension for dimension in range(value.ndim) if dimension != channel_axis)
            maximum           = detached.amax(dim=reduce_dimensions)
        epsilon = torch.finfo(self.scale.dtype).eps
        return (maximum.to(dtype=self.scale.dtype).clamp_min(0.0) / UFP4_MAX_VALUE).clamp_min(epsilon)


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


def encode_e2m2(value: torch.Tensor, rounding: str = "ties_away_from_zero") -> torch.Tensor:
    """scale適用後の値をE2M2の4bit codeへ符号化する。"""
    if not value.is_floating_point():
        raise TypeError("E2M2 input must be a floating-point Tensor.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("E2M2 input must contain only finite values.")

    round_function = get_rounding_function(rounding)
    levels         = value.new_tensor(E2M2_VALUES)
    bounded        = value.clamp(0.0, UFP4_MAX_VALUE)
    distance       = (bounded.unsqueeze(-1) - levels).abs()
    lower_code     = distance.argmin(dim=-1)
    upper_code     = len(E2M2_VALUES) - 1 - torch.flip(distance, dims=(-1,)).argmin(dim=-1)
    midpoint       = (lower_code + upper_code).to(dtype=bounded.dtype) / 2.0
    midpoint_code  = round_function(midpoint).to(dtype=torch.int64)
    encoded        = torch.where(lower_code == upper_code, lower_code, midpoint_code)
    return encoded.to(dtype=torch.uint8)


def decode_e2m2(value: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """4bitのE2M2 codeを浮動小数点値へ復号する。"""
    if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
        raise TypeError("E2M2 code must use an integer Tensor dtype.")
    if bool(((value < 0) | (value > 15)).any()):
        raise ValueError("E2M2 code must be in [0, 15].")

    levels = torch.tensor(E2M2_VALUES, device=value.device, dtype=dtype)
    return levels[value.to(dtype=torch.int64)]
