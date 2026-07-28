"""整数コードのまま平均プーリングするQAT用レイヤー。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .rounding import get_rounding_function


class QuantizedAvgPool2d(nn.Module):
    """INT8などの整数入力をINT32加算してから平均する処理をFake Quantizationで再現する。"""

    def __init__(self, kernel_size: int, stride: int | None = None, bit_width: int = 8, signed: bool = True, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive.")
        if bit_width <= 0 or bit_width > 16:
            raise ValueError("bit_width must be in [1, 16].")

        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.bit_width = bit_width
        self.signed = signed
        self.rounding = rounding
        self.round_function = get_rounding_function(rounding)

    @property
    def qmin(self) -> int:
        """入力と出力の整数コードが取れる最小値を返す。"""
        return -(2 ** (self.bit_width - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        """入力と出力の整数コードが取れる最大値を返す。"""
        return 2 ** (self.bit_width - 1) - 1 if self.signed else 2**self.bit_width - 1

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """整数コードを加算し、要素数で丸め除算した値と変わらないscaleを返す。"""
        if value.ndim != 4:
            raise ValueError("QuantizedAvgPool2d requires a four-dimensional NCHW tensor.")

        flat_input_scale = input_scale.detach().reshape(-1)
        if flat_input_scale.numel() != 1:
            raise ValueError("QuantizedAvgPool2d requires a per-tensor input scale.")
        if not bool(torch.isfinite(flat_input_scale).all()) or bool((flat_input_scale <= 0.0).any()):
            raise ValueError("input_scale must be finite and positive.")

        scale = flat_input_scale[0].to(device=value.device, dtype=value.dtype).reshape(1, 1, 1, 1)
        scaled_input = value / scale
        rounded_input = scaled_input + (self.round_function(scaled_input) - scaled_input).detach()
        input_integer = rounded_input.clamp(self.qmin, self.qmax)

        summed_integer = F.avg_pool2d(input_integer, kernel_size=self.kernel_size, stride=self.stride, divisor_override=1)
        divisor = float(self.kernel_size * self.kernel_size)
        averaged_integer = summed_integer / divisor
        rounded_output = averaged_integer + (self.round_function(averaged_integer) - averaged_integer).detach()
        output_integer = rounded_output.clamp(self.qmin, self.qmax)
        quantized_output = output_integer * scale

        floating_reference = F.avg_pool2d(value, kernel_size=self.kernel_size, stride=self.stride)
        fake_quantized_output = floating_reference + (quantized_output - floating_reference).detach()
        return fake_quantized_output, flat_input_scale.detach()
