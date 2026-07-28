"""量子化Attentionで使用する行列積。"""

from __future__ import annotations

import torch
from torch import nn

from .rounding import get_rounding_function


class QuantMatMul(nn.Module):
    """入力を整数値へ戻して行列積し、入力scaleの積を掛け戻す。"""

    def __init__(self, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        self.rounding = rounding
        self.round_function = get_rounding_function(rounding)

    def forward(self, left: torch.Tensor, left_scale: torch.Tensor, right: torch.Tensor, right_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """INT8同士の行列積をFake Quantizationで再現する。"""
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

        right_scaled = right / right_scale
        right_integer = right_scaled + (self.round_function(right_scaled) - right_scaled).detach()

        accumulator = torch.matmul(left_integer, right_integer)
        output_scale = left_scale * right_scale
        output = accumulator * output_scale
        return output, output_scale
