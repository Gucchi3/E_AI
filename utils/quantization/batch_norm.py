"""BatchNormを畳み込みの重みとbiasへ統合する。"""

from __future__ import annotations

import torch
from torch import nn


def batch_norm_scale(batch_norm: nn.BatchNorm2d) -> torch.Tensor:
    """BatchNormが各出力チャンネルへ掛ける倍率を返す。"""
    if batch_norm.running_var is None:
        raise ValueError("BatchNorm must track running statistics for folding.")
    weight = batch_norm.weight if batch_norm.weight is not None else torch.ones_like(batch_norm.running_var)
    return weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)


def fold_batch_norm(weight: torch.Tensor, bias: torch.Tensor | None, batch_norm: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """BatchNormを統合した重みとbiasを返す。"""
    if weight.ndim != 4:
        raise ValueError("BatchNorm folding requires a four-dimensional convolution weight.")
    if weight.size(0) != batch_norm.num_features:
        raise ValueError("The convolution output channels and BatchNorm features must match.")
    if batch_norm.running_mean is None or batch_norm.running_var is None:
        raise ValueError("BatchNorm must track running statistics for folding.")

    running_mean  = batch_norm.running_mean.to(dtype=weight.dtype)
    scale         = batch_norm_scale(batch_norm).to(dtype=weight.dtype)
    shift         = batch_norm.bias.to(dtype=weight.dtype) if batch_norm.bias is not None else torch.zeros_like(running_mean)
    conv_bias     = bias if bias is not None else torch.zeros_like(running_mean)
    weight_shape  = (weight.size(0),) + (1,) * (weight.ndim - 1)
    folded_weight = weight * scale.reshape(weight_shape)
    folded_bias   = shift + (conv_bias - running_mean) * scale
    return folded_weight, folded_bias
