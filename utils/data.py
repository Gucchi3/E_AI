"""CIFAR-10 datasets, transforms, and loaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .config import DataConfig


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def make_cifar10_loaders(config: DataConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """Create the only supported dataset pair: CIFAR-10 train and test."""
    train_dataset  = datasets.CIFAR10(root=config.root, train=True,  download=True, transform=_train_transform(config.image_size, config.normalization))
    test_dataset   = datasets.CIFAR10(root=config.root, train=False, download=True, transform=_test_transform(config.image_size, config.normalization))
    loader_options = {"batch_size": config.batch_size, "num_workers": config.num_workers, "pin_memory": device.type == "cuda", "persistent_workers": config.num_workers > 0}
    train_loader   = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_options)
    test_loader    = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_options)
    
    return train_loader, test_loader


def _train_transform(image_size: int, normalization: str) -> transforms.Compose:
    steps: list[object] = []
    if image_size == 32:
        steps.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    else:
        steps.extend([transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR), transforms.RandomHorizontalFlip()])
    steps.extend(_tensor_and_normalization(normalization))
    
    return transforms.Compose(steps)


def _test_transform(image_size: int, normalization: str) -> transforms.Compose:
    steps: list[object] = []
    if image_size == 256:
        steps.append(transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR))
    steps.extend(_tensor_and_normalization(normalization))
    
    return transforms.Compose(steps)


def _tensor_and_normalization(normalization: str) -> list[object]:
    steps: list[object] = [transforms.ToTensor()]
    if normalization == "cifar10":
        steps.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
        
    return steps
