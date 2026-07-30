"""各残差加算が持つ独立scaleへ直接再量子化するQAT用レイヤー。"""

from __future__ import annotations

import torch
from torch import nn

from .integer import IntegerQuantizer
from .rounding import get_rounding_function


INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


class QuantResidualAdd(nn.Module):
    """本線のINT32 accumulatorと残差を、この加算専用のscaleへ直接再量子化して加算する。"""

    def __init__(self, bit_width: int = 8, rounding: str = "ties_away_from_zero", range_momentum: float = 0.95) -> None:
        super().__init__()
        self.output_quantizer = IntegerQuantizer(bit_width=bit_width, signed=True, rounding=rounding, range_momentum=range_momentum)
        self.round_function   = get_rounding_function(rounding)

    @property
    def output_scale(self) -> torch.Tensor:
        """残差加算後の共通scaleを返す。"""
        return self.output_quantizer.scale

    def forward(self, transformed: torch.Tensor, transformed_scale: torch.Tensor, identity: torch.Tensor, identity_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """本線accumulatorと残差を出力scaleへ直接変換し、整数加算をFake Quantizationで再現する。"""
        if transformed.shape != identity.shape:
            raise ValueError(f"Residual branch shapes must match: {tuple(transformed.shape)} and {tuple(identity.shape)}.")
        if transformed.ndim < 2:
            raise ValueError("Residual tensors must have at least two dimensions.")
        transformed_scale_count = transformed_scale.detach().reshape(-1).numel()
        if transformed_scale_count not in {1, transformed.shape[1]}:
            raise ValueError(f"transformed_scale must contain 1 or {transformed.shape[1]} values.")
        if identity_scale.detach().reshape(-1).numel() != 1:
            raise ValueError("identity_scale must be a per-tensor scale.")
        if not bool(torch.isfinite(transformed_scale).all()) or bool((transformed_scale <= 0.0).any()):
            raise ValueError("transformed_scale must be finite and positive.")
        if not bool(torch.isfinite(identity_scale).all()) or bool((identity_scale <= 0.0).any()):
            raise ValueError("identity_scale must be finite and positive.")

        summed       = transformed + identity
        output_scale = self.output_quantizer.scale_for(summed)

        transformed_scale = transformed_scale.detach().reshape(-1).to(device=transformed.device, dtype=transformed.dtype)
        if transformed_scale.numel() == 1:
            transformed_scale = transformed_scale.reshape((1,) * transformed.ndim)
        else:
            transformed_scale = transformed_scale.reshape((1, transformed_scale.numel()) + (1,) * (transformed.ndim - 2))
        identity_scale    = identity_scale.detach().to(device=identity.device, dtype=identity.dtype).reshape((1,) * identity.ndim)

        transformed_scaled  = transformed / transformed_scale
        transformed_integer = transformed_scaled + (self.round_function(transformed_scaled) - transformed_scaled).detach()
        if bool(((transformed_integer.detach() < INT32_MIN) | (transformed_integer.detach() > INT32_MAX)).any()):
            raise OverflowError("Projection accumulator exceeded the signed INT32 range.")

        identity_scaled  = identity / identity_scale
        identity_integer = identity_scaled + (self.round_function(identity_scaled) - identity_scaled).detach()

        transformed_common = transformed_integer * transformed_scale / output_scale
        transformed_common = transformed_common + (self.round_function(transformed_common) - transformed_common).detach()

        identity_common = identity_integer * identity_scale / output_scale
        identity_common = identity_common + (self.round_function(identity_common) - identity_common).detach()

        output_integer  = transformed_common + identity_common
        output_integer  = output_integer.clamp(self.output_quantizer.qmin, self.output_quantizer.qmax)
        quantized       = output_integer * output_scale
        fake_quantized = summed + (quantized - summed).detach()
        return fake_quantized, self.output_quantizer.scale
