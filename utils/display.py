"""学習状況を見やすく表示する。"""

from __future__ import annotations

from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import AppConfig, QUANTIZED_MODEL_NAMES


CONSOLE = Console()


def print_training_info(config: AppConfig, model: torch.nn.Module, device: torch.device, run_dir: Path, macs: str) -> None:
    """学習設定を表形式で表示する。"""
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
    table.add_row("Model", "Pretrained", str(config.model.load_weight))
    if config.model.load_weight:
        table.add_row("Model", "Weight Path", config.model.weight_path)
    table.add_row("Model", "Image Size", str(config.data.image_size))
    table.add_row("Model", "Num Classes", str(config.model.num_classes))
    table.add_row("Model", "Total Params", f"{total_params:,} ({total_params / 1e6:.2f}M)")
    table.add_row("Model", "Trainable Params", f"{trainable_params:,}")
    table.add_row("Model", "MACs", macs)
    table.add_row("")

    if config.model.name in QUANTIZED_MODEL_NAMES:
        weight_format, activation_format = _quantization_formats(config)
        table.add_row("Quantization", "Weights", weight_format)
        table.add_row("Quantization", "Activations", activation_format)
        table.add_row("Quantization", "Input", f"uint{config.quantization.input_bits}")
        if config.model.name in {"basic_vit_int8", "basic_vit_int4", "basic_vit_fp4", "basic_vit_ufp4", "basic_vit_test2", "basic_vit_test5", "basic_vit_test6", "basic_vit_test7"}:
            table.add_row("Quantization", "Residual", f"INT{config.quantization.residual_bits} (one scale shared by all 6 additions)")
        if config.model.name in {"basic_vit_test1", "basic_vit_test3", "basic_vit_test4"}:
            table.add_row("Quantization", "Residual", f"INT{config.quantization.residual_bits} (independent scale for each of 6 additions)")
        table.add_row("Quantization", "Rounding", config.quantization.rounding)
        table.add_row("Quantization", "Range Momentum", str(config.quantization.activation_range_momentum))
        table.add_row("")

    table.add_row("Data", "Dataset", config.data.dataset.upper())
    table.add_row("Data", "Data Dir", config.data.root)
    table.add_row("Data", "Normalization", config.data.normalization)
    table.add_row("")

    table.add_row("Training", "Epochs", str(config.train.epochs))
    table.add_row("Training", "Batch Size", str(config.data.batch_size))
    table.add_row("Training", "Optimizer", "AdamW")
    table.add_row("Training", "LR", str(config.train.learning_rate))
    table.add_row("Training", "LR Scheduler", "CosineAnnealingLR")
    table.add_row("Training", "Minimum LR", str(config.train.minimum_learning_rate))
    table.add_row("Training", "Weight Decay", f"{config.train.weight_decay:g}")
    table.add_row("Training", "Label Smooth", f"{config.train.label_smoothing:g}")
    table.add_row("")
    
    table.add_row("Augmentation", "MixUp Alpha", f"{config.train.mixup_alpha:g}")
    table.add_row("Augmentation", "CutMix Alpha", f"{config.train.cutmix_alpha:g}")
    table.add_row("Augmentation", "Mix Probability", f"{config.train.mixup_probability:g}")
    table.add_row("Augmentation", "Switch Probability", f"{config.train.mixup_switch_probability:g}")
    table.add_row("")

    table.add_row("Output", "Run Directory", str(run_dir))

    CONSOLE.print()
    CONSOLE.print(table)
    CONSOLE.print()


def print_weight_loaded(weight_path: Path) -> None:
    """読み込んだ重みのパスを表示する。"""
    line = Text()
    line.append("[OK]", style="bold green")
    line.append(f" Loaded weights from {weight_path}")
    CONSOLE.print(line)


def print_epoch(epoch: int, epochs: int, learning_rate: float, train_loss: float, test_loss: float, accuracy: float) -> None:
    """1 epoch分の結果を色分けして表示する。"""
    line = Text()
    line.append("Epoch", style="bold cyan")
    line.append(f" [{epoch:3d}/{epochs}]  ", style="cyan")
    line.append("lr", style="bold magenta")
    line.append(f"={learning_rate:.2e}  ", style="magenta")
    line.append("train_loss", style="bold yellow")
    line.append(f"={train_loss:.4f}  ", style="yellow")
    line.append("test_loss", style="bold blue")
    line.append(f"={test_loss:.4f}  ", style="blue")
    line.append("acc", style="bold green")
    line.append(f"={accuracy * 100:.2f}%", style="green")
    CONSOLE.print(line)


def _device_name(device: torch.device) -> str:
    """表示用のデバイス名を返す。"""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu (CPU)"


def _quantization_formats(config: AppConfig) -> tuple[str, str]:
    """モデルに対応する重み・活性化形式の表示名を返す。"""
    if config.model.name == "basic_vit_fp4":
        return "INT8 (first/classifier), FP4 E2M1 (body)", "FP4 E2M1 (body), shared INT4 (residual), INT8 (average pool)"
    if config.model.name == "basic_vit_ufp4":
        return "INT8 (first/classifier), FP4 E2M1 (body)", "UFP4 E2M2 (after ReLU), FP4 E2M1 (signed attention), shared INT4 (residual), INT8 (average pool)"
    if config.model.name == "basic_vit_int4":
        return "INT8 (first/classifier), INT4 (body)", "INT4 (body and shared residual), INT8 (average pool)"
    if config.model.name == "basic_vit_test1":
        return "INT8 (first/classifier), FP4 E2M1 (body)", "UFP4 E2M2 (after ReLU), FP4 E2M1 (signed attention), independent INT8 (residual), INT8 (average pool)"
    if config.model.name == "basic_vit_test2":
        return "INT8 (first/classifier), FP4 E2M1 (body)", "UFP4 E2M2 (after ReLU), INT8 (Q/K/V and attention map), FP4 E2M1 (attention output), shared INT4 (residual), INT8 (average pool)"
    if config.model.name == "basic_vit_test3":
        return "INT8 (first/classifier), FP4 E2M1 (body)", "UFP4 E2M2 (after ReLU), INT8 (Q/K/V and attention map), FP4 E2M1 (attention output), independent INT8 (residual), INT8 (average pool)"
    if config.model.name == "basic_vit_test4":
        return "INT8 (first/classifier), FP4 E2M1 (body)", "UFP4 E2M2 (after ReLU), FP4 E2M1 (signed attention), independent INT4 (residual), INT8 (average pool)"
    if config.model.name == "basic_vit_test5":
        return "INT8 (first/classifier), FP4 E2M1 (body)", "UFP4 E2M2 (after ReLU), FP4 E2M1 (signed attention), shared INT8 (residual), INT8 (average pool)"
    if config.model.name == "basic_vit_test6":
        return "INT8 (first/classifier), signed INT4 (body)", "signed INT4 (body and attention), shared INT8 (residual), INT8 (average pool)"
    if config.model.name == "basic_vit_test7":
        return "INT8 (first/classifier), signed INT4 (body)", "UINT4 (after ReLU), signed INT4 (attention), shared INT8 (residual), INT8 (average pool)"
    if config.model.name in {"resnet18_fp4", "mobilenet_v2_fp4"}:
        return "INT8 (stem/classifier), FP4 E2M1 (body)", "FP4 E2M1 (body), UINT8 (average pool)"
    if config.model.name in {"resnet18_ufp4", "mobilenet_v2_ufp4"}:
        return "INT8 (stem/classifier), FP4 E2M1 (body)", "UFP4 E2M2 (after ReLU), FP4 E2M1 (signed branches), UINT8 (average pool)"
    if config.model.name in {"resnet18_int4", "mobilenet_v2_int4"}:
        return "INT8 (stem/classifier), INT4 (body)", "INT4 (body), UINT8 (average pool)"
    if config.model.name in {"basic_vit_int8", "resnet18_int8", "mobilenet_v2_int8"}:
        return "INT8", "INT8"
    if config.model.name in {"fp4_cnn", "mixed_cnn"}:
        return "INT8 (first/last), FP4 (middle)", "FP4 (middle), INT8 (before final)"
    if config.model.name == "test_cnn":
        return "INT8 (first/last), FP4 (middle)", "UINT4 (middle), INT8 (before final)"
    if config.model.name == "ufp4_test_cnn":
        return "INT8 (first/last), FP4 (middle)", "UFP4 E2M2 (middle), INT8 (before final)"
    if config.model.name == "int4_cnn":
        return "INT8 (first/last), INT4 (middle)", "INT4 (middle), INT8 (before final)"
    bit_width = config.quantization.weight_bits
    return f"INT{bit_width}", f"INT{config.quantization.activation_bits}"
