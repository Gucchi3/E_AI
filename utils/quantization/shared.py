"""複数の残差接続で同じ整数scaleを使用するためのQAT用レイヤー。"""

from __future__ import annotations

import torch
from torch import nn

from .integer import SUPPORTED_BIT_WIDTHS
from .rounding import get_rounding_function


class SharedIntegerQuantizer(nn.Module):
    """1回のforward中に固定した1つのscaleを複数の残差接続で共有する。"""

    scale: torch.Tensor
    running_max: torch.Tensor
    range_initialized: torch.Tensor

    def __init__(self, bit_width: int, signed: bool = True, rounding: str = "ties_away_from_zero", range_momentum: float = 0.95) -> None:
        super().__init__()
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}.")
        if not 0.0 <= range_momentum < 1.0:
            raise ValueError("range_momentum must be in [0.0, 1.0).")

        self.bit_width = bit_width
        self.signed = signed
        self.rounding = rounding
        self.range_momentum = range_momentum
        self.round_function = get_rounding_function(rounding)
        self.forward_active = False
        self.just_initialized = False

        self.register_buffer("scale", torch.full((1,), torch.nan, dtype=torch.float32))
        self.register_buffer("running_max", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("range_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("observed_maximum", torch.zeros(1, dtype=torch.float32), persistent=False)

    @property
    def qmin(self) -> int:
        """共有整数コードの最小値を返す。"""
        return -(2 ** (self.bit_width - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        """共有整数コードの最大値を返す。"""
        return 2 ** (self.bit_width - 1) - 1 if self.signed else 2**self.bit_width - 1

    def begin_forward(self, initial_value: torch.Tensor) -> None:
        """Stem出力を観測し、このforwardで使用するscaleを固定する。"""
        current_maximum = self._maximum(initial_value)
        with torch.no_grad():
            self.observed_maximum.copy_(current_maximum)
            if not bool(self.range_initialized.item()):
                self.running_max.copy_(current_maximum)
                self._update_scale_from_running_max()
                self.range_initialized.fill_(True)
                self.just_initialized = True
            else:
                self.just_initialized = False
        self.forward_active = True

    def observe(self, value: torch.Tensor) -> None:
        """次のforwardで使うscaleを決めるため、残差加算候補の最大絶対値を収集する。"""
        if not self.forward_active:
            raise RuntimeError("begin_forward must be called before observe.")
        current_maximum = self._maximum(value)
        with torch.no_grad():
            self.observed_maximum.copy_(torch.maximum(self.observed_maximum, current_maximum))

    def end_forward(self) -> None:
        """全残差を観測した後、次のforwardで使うscaleを1回だけ更新する。"""
        if not self.forward_active:
            raise RuntimeError("begin_forward must be called before end_forward.")
        if self.training:
            with torch.no_grad():
                if self.just_initialized:
                    self.running_max.copy_(self.observed_maximum)
                else:
                    self.running_max.mul_(self.range_momentum).add_(self.observed_maximum, alpha=1.0 - self.range_momentum)
                self._update_scale_from_running_max()
        self.forward_active = False
        self.just_initialized = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """現在固定されている共有scaleへFake Quantizationする。"""
        scale = self.scale_for(value)
        scaled = value / scale
        rounded = scaled + (self.round_function(scaled) - scaled).detach()
        clipped = rounded.clamp(self.qmin, self.qmax)
        return clipped * scale

    def scale_for(self, value: torch.Tensor) -> torch.Tensor:
        """現在の共有scaleをvalueへbroadcastできる形で返す。"""
        if not bool(self.range_initialized.item()):
            raise RuntimeError("Shared scale is not initialized. Call begin_forward first.")
        return self.scale.detach().clone().to(device=value.device, dtype=value.dtype).reshape((1,) * value.ndim)

    def _maximum(self, value: torch.Tensor) -> torch.Tensor:
        """Tensor全体の最大絶対値をscaleと同じ形式で返す。"""
        if not value.is_floating_point():
            raise TypeError("SharedIntegerQuantizer requires floating-point tensors for QAT.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("SharedIntegerQuantizer input must contain only finite values.")
        maximum = value.detach().abs().amax().reshape(1).to(device=self.running_max.device, dtype=self.running_max.dtype)
        epsilon = torch.finfo(self.running_max.dtype).eps
        return maximum.clamp_min(epsilon)

    def _update_scale_from_running_max(self) -> None:
        """観測範囲から対称量子化scaleを更新する。"""
        epsilon = torch.finfo(self.scale.dtype).eps
        self.scale.copy_((self.running_max / float(self.qmax)).clamp_min(epsilon))


class SharedScaleResidualAdd(nn.Module):
    """同じ共有scaleの整数コードへ両枝を変換し、整数加算して同じscaleで出力する。"""

    def __init__(self, rounding: str = "ties_away_from_zero") -> None:
        super().__init__()
        self.rounding = rounding
        self.round_function = get_rounding_function(rounding)

    def forward(self, transformed: torch.Tensor, identity: torch.Tensor, shared_quantizer: SharedIntegerQuantizer) -> tuple[torch.Tensor, torch.Tensor]:
        """両枝を共有scaleへ変換し、INT32加算と飽和をFake Quantizationで再現する。"""
        if transformed.shape != identity.shape:
            raise ValueError(f"Residual branch shapes must match: {tuple(transformed.shape)} and {tuple(identity.shape)}.")

        summed = transformed + identity
        shared_quantizer.observe(summed)
        shared_scale = shared_quantizer.scale_for(summed)

        transformed_scaled = transformed / shared_scale
        transformed_integer = transformed_scaled + (self.round_function(transformed_scaled) - transformed_scaled).detach()
        transformed_integer = transformed_integer.clamp(shared_quantizer.qmin, shared_quantizer.qmax)

        identity_scaled = identity / shared_scale
        identity_integer = identity_scaled + (self.round_function(identity_scaled) - identity_scaled).detach()
        identity_integer = identity_integer.clamp(shared_quantizer.qmin, shared_quantizer.qmax)

        output_accumulator = transformed_integer + identity_integer
        output_integer = output_accumulator.clamp(shared_quantizer.qmin, shared_quantizer.qmax)
        quantized = output_integer * shared_scale
        fake_quantized = summed + (quantized - summed).detach()
        return fake_quantized, shared_quantizer.scale.detach().clone()
