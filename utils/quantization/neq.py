"""L1 Attentionで使用するNormalization Equivalent Quantization。"""

from __future__ import annotations

import torch
from torch import nn

from .fp4 import decode_e2m1, encode_e2m1, FP4_MAX_VALUE
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


class PerTokenFP4NEQ(nn.Module):
    """L1正規化とE2M1量子化を同じトークン粒度で行い、除算をscaleへ吸収する。"""

    def __init__(self, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        self.rounding = rounding
        get_rounding_function(rounding)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP4 Fake QuantizationされたL1正規化値とトークンごとのscaleを返す。"""
        if not value.is_floating_point():
            raise TypeError("PerTokenFP4NEQ requires a floating-point tensor for QAT.")
        if value.ndim == 0:
            raise ValueError("PerTokenFP4NEQ requires at least one tensor dimension.")

        epsilon = torch.finfo(value.dtype).eps
        l1_norm = value.abs().sum(dim=-1, keepdim=True)
        safe_l1_norm = l1_norm.clamp_min(epsilon)
        normalized = value / safe_l1_norm

        with torch.no_grad():
            maximum = value.detach().abs().amax(dim=-1, keepdim=True)
            fp4_scale = (maximum / FP4_MAX_VALUE).clamp_min(epsilon)
            fp4_code = encode_e2m1(value.detach() / fp4_scale, self.rounding)
            fp4_level = decode_e2m1(fp4_code, dtype=value.dtype)
            normalized_scale = fp4_scale / safe_l1_norm.detach()
            quantized = fp4_level * normalized_scale

        fake_quantized = normalized + (quantized - normalized).detach()
        return fake_quantized, normalized_scale.detach()
