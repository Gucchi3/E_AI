"""Human-readable training information shown once before the first epoch."""

from __future__ import annotations

from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from .config import AppConfig


def print_training_info(config: AppConfig, model: torch.nn.Module, device: torch.device, run_dir: Path) -> None:
    """Display the active settings with the same Rich table style as Q_ViT."""
    total_params     = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    device_name      = _device_name(device)

    table = Table(title="Training Configuration", show_header=True, header_style="bold")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Key", style="green")
    table.add_column("Value", style="white")

    table.add_row("Environment", "Device", device_name)
    table.add_row("Environment", "PyTorch", torch.__version__)
    table.add_row("Environment", "Seed", str(config.run.seed))
    table.add_row("")

    table.add_row("Model", "Name", config.model.name)
    table.add_row("Model", "Image Size", str(config.data.image_size))
    table.add_row("Model", "Num Classes", str(config.model.num_classes))
    table.add_row("Model", "Total Params", f"{total_params:,} ({total_params / 1e6:.2f}M)")
    table.add_row("Model", "Trainable Params", f"{trainable_params:,}")
    table.add_row("")

    table.add_row("Data", "Dataset", config.data.dataset.upper())
    table.add_row("Data", "Data Dir", config.data.root)
    table.add_row("Data", "Normalization", config.data.normalization)
    table.add_row("")

    table.add_row("Training", "Epochs", str(config.train.epochs))
    table.add_row("Training", "Batch Size", str(config.data.batch_size))
    table.add_row("Training", "Optimizer", "AdamW")
    table.add_row("Training", "LR", str(config.train.learning_rate))
    table.add_row("Training", "Weight Decay", f"{config.train.weight_decay:g}")
    table.add_row("Training", "Label Smooth", f"{config.train.label_smoothing:g}")
    table.add_row("")

    table.add_row("Output", "Run Directory", str(run_dir))

    console = Console()
    console.print()
    console.print(table)
    console.print()


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu (CPU)"
