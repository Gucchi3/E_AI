"""量子化Attentionで使用する行列積。"""

from __future__ import annotations

import torch
from torch import nn

from .fp4 import decode_e2m1, encode_e2m1
from .integer import SUPPORTED_BIT_WIDTHS
from .rounding import get_rounding_function


class QuantMatMul(nn.Module):
    """指定したビット幅の整数へ戻して、INT32行列積とscale復元を再現する。"""

    def __init__(self, bit_width: int = 8, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}.")
        self.bit_width = bit_width
        self.rounding = rounding
        self.round_function = get_rounding_function(rounding)

    @property
    def qmin(self) -> int:
        """符号付き整数コードの最小値を返す。"""
        return -(2 ** (self.bit_width - 1))

    @property
    def qmax(self) -> int:
        """符号付き整数コードの最大値を返す。"""
        return 2 ** (self.bit_width - 1) - 1

    def forward(self, left: torch.Tensor, left_scale: torch.Tensor, right: torch.Tensor, right_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """指定ビット幅の整数同士の行列積をFake Quantizationで再現する。"""
        if left.ndim < 2 or right.ndim < 2:
            raise ValueError("QuantMatMul operands must have at least two dimensions.")
        if left.shape[-1] != right.shape[-2]:
            raise ValueError(f"QuantMatMul reduction dimensions do not match: {left.shape[-1]} and {right.shape[-2]}.")
        if not bool(torch.isfinite(left_scale).all()) or bool((left_scale <= 0.0).any()):
            raise ValueError("left_scale must be finite and positive.")
        if not bool(torch.isfinite(right_scale).all()) or bool((right_scale <= 0.0).any()):
            raise ValueError("right_scale must be finite and positive.")

        left_scale = left_scale.detach().to(device=left.device, dtype=left.dtype)
        right_scale = right_scale.detach().to(device=right.device, dtype=right.dtype)

        left_scaled = left / left_scale
        left_integer = left_scaled + (self.round_function(left_scaled) - left_scaled).detach()
        left_integer = left_integer.clamp(self.qmin, self.qmax)

        right_scaled = right / right_scale
        right_integer = right_scaled + (self.round_function(right_scaled) - right_scaled).detach()
        right_integer = right_integer.clamp(self.qmin, self.qmax)

        accumulator = torch.matmul(left_integer, right_integer)
        if bool((accumulator.detach().abs() > 2**31 - 1).any()):
            raise OverflowError("QuantMatMul accumulator exceeded the signed INT32 range.")
        output_scale = left_scale * right_scale
        output = accumulator * output_scale
        return output, output_scale


class FP4MatMul(nn.Module):
    """E2M1を整数係数へ展開し、INT32積和と同値のFP4行列積をQAT上で再現する。"""

    def __init__(self, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        self.rounding = rounding
        get_rounding_function(rounding)

    def forward(self, left: torch.Tensor, left_scale: torch.Tensor, right: torch.Tensor, right_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP4×FP4の出力と、整数係数で積和するときのaccumulator scaleを返す。"""
        if left.ndim < 2 or right.ndim < 2:
            raise ValueError("FP4MatMul operands must have at least two dimensions.")
        if left.shape[-1] != right.shape[-2]:
            raise ValueError(f"FP4MatMul reduction dimensions do not match: {left.shape[-1]} and {right.shape[-2]}.")
        if not bool(torch.isfinite(left_scale).all()) or bool((left_scale <= 0.0).any()):
            raise ValueError("left_scale must be finite and positive.")
        if not bool(torch.isfinite(right_scale).all()) or bool((right_scale <= 0.0).any()):
            raise ValueError("right_scale must be finite and positive.")

        left_scale = left_scale.detach().to(device=left.device, dtype=left.dtype)
        right_scale = right_scale.detach().to(device=right.device, dtype=right.dtype)

        left_scaled = left / left_scale
        left_code = encode_e2m1(left_scaled.detach(), self.rounding)
        left_fp4 = decode_e2m1(left_code, dtype=left.dtype)
        left_fp4 = left_scaled + (left_fp4 - left_scaled).detach()
        left_integer_coefficient = left_fp4 * 2.0

        right_scaled = right / right_scale
        right_code = encode_e2m1(right_scaled.detach(), self.rounding)
        right_fp4 = decode_e2m1(right_code, dtype=right.dtype)
        right_fp4 = right_scaled + (right_fp4 - right_scaled).detach()
        right_integer_coefficient = right_fp4 * 2.0

        accumulator = torch.matmul(left_integer_coefficient, right_integer_coefficient)
        if bool((accumulator.detach().abs() > 2**31 - 1).any()):
            raise OverflowError("FP4MatMul accumulator exceeded the signed INT32 range.")

        output_scale = left_scale * right_scale / 4.0
        output = accumulator * output_scale
        return output, output_scale
