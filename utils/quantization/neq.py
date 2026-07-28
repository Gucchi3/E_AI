"""L1 Attentionで使用するNormalization Equivalent Quantization。"""

from __future__ import annotations

import torch
from torch import nn

from .integer import SUPPORTED_BIT_WIDTHS
from .rounding import get_rounding_function


class PerTokenNEQ(nn.Module):
    """Tensorの最後の次元を1トークンとして、符号付きNEQを行う。"""

    def __init__(self, bit_width: int = 8, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}.")

        self.bit_width = bit_width
        self.rounding = rounding
        self.round_function = get_rounding_function(rounding)

    @property
    def qmin(self) -> int:
        """対称量子化で使用する最小整数値を返す。"""
        return -(2 ** (self.bit_width - 1) - 1)

    @property
    def qmax(self) -> int:
        """対称量子化で使用する最大整数値を返す。"""
        return 2 ** (self.bit_width - 1) - 1

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fake QuantizationしたL1正規化値と動的scaleを返す。"""
        if not value.is_floating_point():
            raise TypeError("PerTokenNEQ requires a floating-point tensor for QAT.")
        if value.ndim == 0:
            raise ValueError("PerTokenNEQ requires at least one tensor dimension.")

        epsilon = torch.finfo(value.dtype).eps
        l1_norm = value.abs().sum(dim=-1, keepdim=True)
        safe_l1_norm = l1_norm.clamp_min(epsilon)
        normalized = value / safe_l1_norm

        with torch.no_grad():
            maximum = value.detach().abs().amax(dim=-1, keepdim=True)
            integer_scale = (maximum / float(self.qmax)).clamp_min(epsilon)
            integer_value = self.round_function(value.detach() / integer_scale)
            integer_value = integer_value.clamp(self.qmin, self.qmax)
            normalized_scale = integer_scale / safe_l1_norm.detach()
            quantized = integer_value * normalized_scale

        fake_quantized = normalized + (quantized - normalized).detach()
        return fake_quantized, normalized_scale.detach()
