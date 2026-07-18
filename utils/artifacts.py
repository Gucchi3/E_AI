"""Persistent artifacts produced by a single training run.

This module owns output paths and file formats so the training workflow can stay
focused on composing model, data, and optimization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class EpochRecord:
    """Metrics collected after one complete train/evaluation epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float


@dataclass
class RunArtifacts:
    """Files and in-memory metric history for one timestamped training run."""

    run_dir: Path
    history: list[EpochRecord] = field(default_factory=list)

    @classmethod
    def create(cls, log_root: str | Path) -> "RunArtifacts":
        """Create a unique ``YYYYMMDD_HHMMSS`` directory under ``log_root``."""
        run_dir = _create_timestamped_directory(Path(log_root))
        return cls(run_dir=run_dir)

    def save_json(self, filename: str, value: Any) -> Path:
        """Write a small human-readable JSON artifact inside this run directory."""
        path = self.run_dir / filename
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def record_epoch(self, record: EpochRecord) -> None:
        """Append metrics in a machine-readable form and refresh the curve image."""
        self.history.append(record)
        path = self.run_dir / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record)) + "\n")
        self.save_curves()

    def save_model(self, model: torch.nn.Module, filename: str) -> Path:
        """Save only a model ``state_dict``; this is not a resume checkpoint."""
        path = self.run_dir / filename
        torch.save(model.state_dict(), path)
        return path

    def save_curves(self) -> Path:
        """Render loss and accuracy curves from all recorded epochs."""
        path   = self.run_dir / "curves.png"
        epochs = [record.epoch for record in self.history]

        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(epochs, [record.train_loss for record in self.history], label="train")
        axes[0].plot(epochs, [record.test_loss for record in self.history], label="test")
        axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")

        axes[1].plot(epochs, [record.test_accuracy * 100 for record in self.history], label="test")
        axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Percent", ylim=(0, 100))

        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        figure.savefig(path, dpi=150)
        plt.close(figure)

        return path


def _create_timestamped_directory(log_root: Path) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = log_root / timestamp
    suffix    = 1
    while candidate.exists():
        candidate = log_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate
