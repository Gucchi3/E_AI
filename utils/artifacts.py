"""学習結果を保存する。"""

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
    """1 epoch分の学習結果。"""

    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float



@dataclass
class RunArtifacts:
    """実行ごとの保存先と学習履歴。"""

    run_dir: Path
    history: list[EpochRecord] = field(default_factory=list)

    @classmethod
    def create(cls, log_root: str | Path) -> "RunArtifacts":
        """日時を名前にした保存先を作る。"""
        run_dir = _create_timestamped_directory(Path(log_root))
        return cls(run_dir=run_dir)


    def save_json(self, filename: str, value: Any) -> Path:
        """JSONを保存する。"""
        path = self.run_dir / filename
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


    def record_epoch(self, record: EpochRecord) -> None:
        """学習結果を追記してグラフを更新する。"""
        self.history.append(record)
        path = self.run_dir / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record)) + "\n")
        self.save_curves()


    def save_model(self, model: torch.nn.Module, filename: str) -> Path:
        """モデルのstate_dictを保存する。"""
        path = self.run_dir / filename
        torch.save(model.state_dict(), path)
        return path


    def save_curves(self) -> Path:
        """lossとaccuracyのグラフを保存する。"""
        path                     = self.run_dir / "curves.png"
        train_losses             = [record.train_loss for record in self.history]
        test_losses              = [record.test_loss for record in self.history]
        test_accuracies          = [record.test_accuracy * 100 for record in self.history]
        train_label              = f"train (latest: {train_losses[-1]:.4f}, best: {min(train_losses):.4f})" if train_losses else "train"
        test_label               = f"test (latest: {test_losses[-1]:.4f}, best: {min(test_losses):.4f})" if test_losses else "test"
        accuracy_label           = f"test acc (latest: {test_accuracies[-1]:.2f}%, best: {max(test_accuracies):.2f}%)" if test_accuracies else "test acc"
        figure, axes             = plt.subplots(1, 2, figsize=(12, 4))
        loss_axis, accuracy_axis = axes

        loss_axis.plot(train_losses, label=train_label)
        loss_axis.plot(test_losses, label=test_label)
        loss_axis.set_xlabel("epoch")
        loss_axis.set_ylabel("loss")
        loss_axis.legend()
        loss_axis.grid()

        accuracy_axis.plot(test_accuracies, label=accuracy_label, color="r")
        accuracy_axis.set_xlabel("epoch")
        accuracy_axis.set_ylabel("acc (%)")
        accuracy_axis.legend()
        accuracy_axis.grid()

        figure.savefig(path)
        plt.close(figure)

        return path


def _create_timestamped_directory(log_root: Path) -> Path:
    """日時を名前にした重複しない保存先を作る。"""
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = log_root / timestamp
    suffix    = 1
    while candidate.exists():
        candidate = log_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate
