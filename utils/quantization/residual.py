"""共通scaleを使用する量子化残差加算。"""

from __future__ import annotations

import torch
from torch import nn

from .integer import IntegerQuantizer
from .rounding import get_rounding_function


class QuantResidualAdd(nn.Module):
    """2本の残差経路を共通scaleへ再量子化してから加算する。"""

    def __init__(self, bit_width: int = 8, rounding: str = "ties_away_from_zero", range_momentum: float = 0.95) -> None:
        super().__init__()
        self.output_quantizer = IntegerQuantizer(bit_width=bit_width, signed=True, rounding=rounding, range_momentum=range_momentum)
        self.round_function = get_rounding_function(rounding)

    @property
    def output_scale(self) -> torch.Tensor:
        """残差加算後の共通scaleを返す。"""
        return self.output_quantizer.scale

    def forward(self, transformed: torch.Tensor, transformed_scale: torch.Tensor, identity: torch.Tensor, identity_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """両経路を共通scaleへ変換し、整数加算をFake Quantizationで再現する。"""
        if transformed.shape != identity.shape:
            raise ValueError(f"Residual branch shapes must match: {tuple(transformed.shape)} and {tuple(identity.shape)}.")
        if transformed_scale.detach().reshape(-1).numel() != 1:
            raise ValueError("transformed_scale must be a per-tensor scale.")
        if identity_scale.detach().reshape(-1).numel() != 1:
            raise ValueError("identity_scale must be a per-tensor scale.")
        if not bool(torch.isfinite(transformed_scale).all()) or bool((transformed_scale <= 0.0).any()):
            raise ValueError("transformed_scale must be finite and positive.")
        if not bool(torch.isfinite(identity_scale).all()) or bool((identity_scale <= 0.0).any()):
            raise ValueError("identity_scale must be finite and positive.")

        summed = transformed + identity
        output_scale = self.output_quantizer.scale_for(summed)

        transformed_scale = transformed_scale.detach().to(device=transformed.device, dtype=transformed.dtype).reshape((1,) * transformed.ndim)
        identity_scale = identity_scale.detach().to(device=identity.device, dtype=identity.dtype).reshape((1,) * identity.ndim)

        transformed_scaled = transformed / transformed_scale
        transformed_integer = transformed_scaled + (self.round_function(transformed_scaled) - transformed_scaled).detach()

        identity_scaled = identity / identity_scale
        identity_integer = identity_scaled + (self.round_function(identity_scaled) - identity_scaled).detach()

        transformed_common = transformed_integer * transformed_scale / output_scale
        transformed_common = transformed_common + (self.round_function(transformed_common) - transformed_common).detach()

        identity_common = identity_integer * identity_scale / output_scale
        identity_common = identity_common + (self.round_function(identity_common) - identity_common).detach()

        output_integer = transformed_common + identity_common
        output_integer = output_integer.clamp(self.output_quantizer.qmin, self.output_quantizer.qmax)
        quantized = output_integer * output_scale
        fake_quantized = summed + (quantized - summed).detach()
        return fake_quantized, self.output_quantizer.scale
