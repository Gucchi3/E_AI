"""Workflows selected by ``main.py``.

This is the only high-level composition layer. Low-level training utilities do
not know about command-line modes or artifact file names.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from model import build_model
from .artifacts import EpochRecord, RunArtifacts
from .config import AppConfig
from .data import make_cifar10_loaders
from .display import print_training_info
from .engine import evaluate, train_one_epoch
from .runtime import select_device, set_seed


def run_train(config: AppConfig) -> None:
    """Train one model and save self-contained artifacts for that run."""
    set_seed(config.run.seed)
    device                    = select_device(config.run.device)
    train_loader, test_loader = make_cifar10_loaders(config.data, device)

    model     = build_model(config.model.name, config.model.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)

    artifacts       = RunArtifacts.create(config.run.log_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    artifacts.save_json("config.json", asdict(config))

    training_info = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "model_name": config.model.name,
        "parameter_count": parameter_count,
        "image_size": config.data.image_size,
        "normalization": config.data.normalization,
    }
    artifacts.save_json("training_info.json", training_info)
    print_training_info(config, model, device, artifacts.run_dir)

    best_accuracy = -1.0
    for epoch in range(1, config.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics  = evaluate(model, test_loader, criterion, device)
        record        = EpochRecord(epoch=epoch, train_loss=train_metrics.loss, train_accuracy=train_metrics.accuracy, test_loss=test_metrics.loss, test_accuracy=test_metrics.accuracy)
        artifacts.record_epoch(record)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:3d}/{config.train.epochs}]  lr={learning_rate:.2e}  train_loss={record.train_loss:.4f}  test_loss={record.test_loss:.4f}  acc={record.test_accuracy * 100:.2f}%")

        if record.test_accuracy > best_accuracy:
            best_accuracy = record.test_accuracy
            artifacts.save_model(model, "model_best.pth")

    artifacts.save_model(model, "model_final.pth")
    print(f"\n{'=' * 60}")
    print(f"  Training finished - best acc1 = {best_accuracy * 100:.2f}%")
    print(f"{'=' * 60}\n")
