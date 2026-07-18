"""学習処理を組み立てる。"""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from model import build_model
from .artifacts import EpochRecord, RunArtifacts
from .config import AppConfig
from .data import make_cifar10_loaders
from .display import print_epoch, print_training_info, print_weight_loaded
from .engine import evaluate, train_one_epoch
from .profiling import get_macs
from .runtime import select_device, set_seed
from .weights import load_model_weight


def run_train(config: AppConfig) -> None:
    """学習を実行して結果を保存する。"""
    set_seed(config.run.seed)
    device = select_device(config.run.device)
    model  = build_model(config.model.name, config.model.num_classes).to(device)
    if config.model.load_weight:
        weight_path = load_model_weight(model, device, config.model.weight_path)
        print_weight_loaded(weight_path)

    train_loader, test_loader = make_cifar10_loaders(config.data, device)
    optimizer                 = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    criterion                 = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)

    artifacts       = RunArtifacts.create(config.run.log_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    macs            = get_macs(model, device, config.data.image_size)
    artifacts.save_json("config.json", asdict(config))

    training_info = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "model_name": config.model.name,
        "parameter_count": parameter_count,
        "load_weight": config.model.load_weight,
        "weight_path": config.model.weight_path,
        "macs": macs,
        "image_size": config.data.image_size,
        "normalization": config.data.normalization,
    }
    artifacts.save_json("training_info.json", training_info)
    print_training_info(config, model, device, artifacts.run_dir, macs)

    best_accuracy = -1.0
    for epoch in range(1, config.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics  = evaluate(model, test_loader, criterion, device)
        record        = EpochRecord(epoch=epoch, train_loss=train_metrics.loss, train_accuracy=train_metrics.accuracy, test_loss=test_metrics.loss, test_accuracy=test_metrics.accuracy)
        artifacts.record_epoch(record)
        learning_rate = optimizer.param_groups[0]["lr"]
        print_epoch(epoch, config.train.epochs, learning_rate, record.train_loss, record.test_loss, record.test_accuracy)

        if record.test_accuracy > best_accuracy:
            best_accuracy = record.test_accuracy
            artifacts.save_model(model, "model_best.pth")

    artifacts.save_model(model, "model_final.pth")
    print(f"\n{'=' * 60}")
    print(f"  Training finished - best acc1 = {best_accuracy * 100:.2f}%")
    print(f"{'=' * 60}\n")
