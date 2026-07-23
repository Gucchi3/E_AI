"""モデルのMACsを計測する。"""

from __future__ import annotations

import copy
import warnings

import torch
from torch import nn
from thop import profile

from .quantization import QuantConv2d, QuantLinear


def get_macs(model: nn.Module, device: torch.device, image_size: int) -> str:
    """thopでMACsを計測する。"""
    profile_model      = copy.deepcopy(model).to(device)
    profile_model.train()
    calibration_input  = torch.rand(2, 3, image_size, image_size, device=device)
    input_tensor       = torch.rand(1, 3, image_size, image_size, device=device)
    custom_operations  = {QuantConv2d: _count_quant_conv, QuantLinear: _count_quant_linear}
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        profile_model(calibration_input)
        profile_model.eval()
        macs, _ = profile(profile_model, inputs=(input_tensor,), custom_ops=custom_operations, verbose=False)

    return _format_count(macs)


def _count_quant_conv(module: QuantConv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    """量子化畳み込みのMAC数を加算する。"""
    _set_conv_operations(module, module, output)


def _count_quant_linear(module: QuantLinear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    """量子化線形層のMAC数を加算する。"""
    module.total_ops += torch.DoubleTensor([output.numel() * module.in_features])


def _set_conv_operations(module: nn.Module, convolution: nn.Conv2d, output: torch.Tensor) -> None:
    """畳み込み出力とkernel形状からMAC数を設定する。"""
    kernel_height, kernel_width = convolution.kernel_size
    kernel_operations           = convolution.in_channels // convolution.groups * kernel_height * kernel_width
    module.total_ops           += torch.DoubleTensor([output.numel() * kernel_operations])


def _format_count(value: float) -> str:
    """計算量にK、M、Gの単位を付ける。"""
    if value >= 1e9:
        return f"{int(value):,} ({value / 1e9:.2f}G)"
    if value >= 1e6:
        return f"{int(value):,} ({value / 1e6:.2f}M)"
    if value >= 1e3:
        return f"{int(value):,} ({value / 1e3:.2f}K)"
    return f"{int(value):,}"
