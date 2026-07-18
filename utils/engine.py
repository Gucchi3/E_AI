"""Small, framework-free training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> EpochMetrics:
    """Run one ordinary FP32 training epoch."""
    model.train()
    loss_sum = 0.0
    correct  = 0
    samples  = 0

    for images, targets in loader:
        images  = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        
        logits = model(images)
        loss   = criterion(logits, targets)
        
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum  += loss.item() * batch_size
        correct   += (logits.argmax(dim=1) == targets).sum().item()
        samples   += batch_size

    return EpochMetrics(loss=loss_sum / samples, accuracy=correct / samples, samples=samples)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> EpochMetrics:
    """Evaluate without changing model parameters or BatchNorm statistics."""
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
