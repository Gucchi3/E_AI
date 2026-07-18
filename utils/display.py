"""Human-readable training information shown once before the first epoch."""

from __future__ import annotations

import torch
from rich.console import Console
from rich.table import Table

from .config import AppConfig


def print_training_summary(config: AppConfig, model: torch.nn.Module, device: torch.device, train_batches: int, test_batches: int) -> None:
    """Display only the settings that the current minimal workflow actually uses."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    device_name = _device_name(device)

    table = Table(title="E_AI Training Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Key", style="green")
    table.add_column("Value")

    table.add_row("Environment", "Device", device_name)
    table.add_row("Environment", "PyTorch", torch.__version__)
    table.add_row("Environment", "Seed", str(config.run.seed))
    table.add_row("Environment", "Log Root", config.run.log_dir)

    table.add_row("Model", "Name", config.model.name)
    table.add_row("Model", "Classes", str(config.model.num_classes))
    table.add_row("Model", "Parameters", f"{parameter_count:,}")

    table.add_row("Data", "Dataset", "CIFAR-10")
    table.add_row("Data", "Image Size", f"{config.data.image_size}x{config.data.image_size}")
    table.add_row("Data", "Normalization", config.data.normalization)
    table.add_row("Data", "Batch Size / Workers", f"{config.data.batch_size} / {config.data.num_workers}")
    table.add_row("Data", "Train / Test Batches", f"{train_batches} / {test_batches}")

    table.add_row("Training", "Epochs", str(config.train.epochs))
    table.add_row("Training", "Optimizer", "AdamW")
    table.add_row("Training", "Learning Rate", f"{config.train.learning_rate:g}")
    table.add_row("Training", "Weight Decay", f"{config.train.weight_decay:g}")
    table.add_row("Training", "Label Smoothing", f"{config.train.label_smoothing:g}")

    console = Console()
    console.print()
    console.print(table)
    console.print()


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"
