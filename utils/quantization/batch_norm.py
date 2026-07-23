"""モデル内のBatchNormを直前の畳み込みへ統合する。"""

from __future__ import annotations

import copy
from typing import TypeVar

from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def fold_batch_norms(model: ModuleT) -> ModuleT:
    """モデルのコピーをeval modeにし、隣接するConvとBatchNormを融合する。"""
    folded_model = copy.deepcopy(model)
    folded_model.eval()
    _fold_adjacent_modules(folded_model)
    return folded_model


def _fold_adjacent_modules(module: nn.Module) -> None:
    """子moduleを再帰的に調べ、隣接するConvとBatchNormを置換する。"""
    children = list(module.named_children())
    for _, child in children:
        _fold_adjacent_modules(child)

    for (convolution_name, convolution), (batch_norm_name, batch_norm) in zip(children, children[1:]):
        if not _is_foldable_pair(convolution, batch_norm):
            continue
        setattr(module, convolution_name, fuse_conv_bn_eval(convolution, batch_norm))
        setattr(module, batch_norm_name, nn.Identity())


def _is_foldable_pair(convolution: nn.Module, batch_norm: nn.Module) -> bool:
    """同じ次元のConvとBatchNormの組み合わせかを返す。"""
    return (
        isinstance(convolution, nn.Conv1d) and isinstance(batch_norm, nn.BatchNorm1d)
        or isinstance(convolution, nn.Conv2d) and isinstance(batch_norm, nn.BatchNorm2d)
        or isinstance(convolution, nn.Conv3d) and isinstance(batch_norm, nn.BatchNorm3d)
    )
