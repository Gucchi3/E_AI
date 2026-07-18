"""学習と評価のループ。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .augmentation import BatchMixupCutmix


@dataclass(frozen=True)
class EpochMetrics:
    """1 epoch分の集計値。"""

    loss: float
    accuracy: float
    samples: int


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, batch_augmentation: BatchMixupCutmix | None = None) -> EpochMetrics:
    """モデルを1 epoch学習する。"""
    model.train()
    loss_sum = 0.0
    correct  = 0
    samples  = 0

    for images, targets in loader:
        images  = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        loss_targets   = targets
        metric_targets = targets
        if batch_augmentation is not None:
            images, loss_targets, metric_targets = batch_augmentation(images, targets)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss   = criterion(logits, loss_targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum  += loss.item() * batch_size
        correct   += _correct_predictions(logits, metric_targets)
        samples   += batch_size

    return EpochMetrics(loss=loss_sum / samples, accuracy=correct / samples, samples=samples)


def _correct_predictions(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """通常ラベルまたは混合ラベルに対する正解数を返す。"""
    predictions = logits.argmax(dim=1)
    if targets.ndim == 1:
        return float((predictions == targets).sum().item())
    return float(targets.gather(1, predictions.unsqueeze(1)).sum().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> EpochMetrics:
    """モデルを評価する。"""
    model.eval()

    loss_sum = 0.0
    correct  = 0
    samples  = 0

    for images, targets in loader:
        images  = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        logits  = model(images)
        loss    = criterion(logits, targets)

        batch_size = targets.size(0)
        loss_sum  += loss.item() * batch_size
        correct   += (logits.argmax(dim=1) == targets).sum().item()
        samples   += batch_size

    return EpochMetrics(loss=loss_sum / samples, accuracy=correct / samples, samples=samples)
