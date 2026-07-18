"""JSON configuration loading and validation for the training workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Settings that control one invocation of ``main.py``."""

    mode   : str
    seed   : int
    device : str
    log_dir: str


@dataclass(frozen=True)
class DataConfig:
    dataset      : str
    root         : str
    image_size   : int
    normalization: str
    batch_size   : int
    num_workers  : int


@dataclass(frozen=True)
class ModelConfig:
    name       : str
    num_classes: int


@dataclass(frozen=True)
class TrainConfig:
    epochs         : int
    learning_rate  : float
    weight_decay   : float
    label_smoothing: float


@dataclass(frozen=True)
class AppConfig:
    run  : RunConfig
    data : DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(path: str | Path) -> AppConfig:
    """Load a JSON configuration into validated, immutable settings."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Config file was not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Config file is not valid JSON: {config_path}") from error

    if not isinstance(raw, dict):
        raise ValueError("The config root must be a JSON object.")

    run_raw   = _section(raw, "run")
    data_raw  = _section(raw, "data")
    model_raw = _section(raw, "model")
    train_raw = _section(raw, "train")

    run_config   = RunConfig(mode=_string(run_raw, "mode"), seed=_integer(run_raw, "seed"), device=_string(run_raw, "device"), log_dir=_string(run_raw, "log_dir"))
    data_config  = DataConfig(dataset=_string(data_raw, "dataset"), root=_string(data_raw, "root"), image_size=_integer(data_raw, "image_size"), normalization=_string(data_raw, "normalization"), batch_size=_integer(data_raw, "batch_size"), num_workers=_integer(data_raw, "num_workers"))
    model_config = ModelConfig(name=_string(model_raw, "name"), num_classes=_integer(model_raw, "num_classes"))
    train_config = TrainConfig(epochs=_integer(train_raw, "epochs"), learning_rate=_number(train_raw, "learning_rate"), weight_decay=_number(train_raw, "weight_decay"), label_smoothing=_number(train_raw, "label_smoothing"))
    config       = AppConfig(run=run_config, data=data_config, model=model_config, train=train_config)
    _validate(config)
    
    return config


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be an object.")
    return value


def _string(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Config value {name!r} must be a string.")
    return value


def _integer(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config value {name!r} must be an integer.")
    return value


def _number(raw: dict[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config value {name!r} must be a number.")
    return float(value)


def _validate(config: AppConfig) -> None:
    if config.run.mode != "train":
        raise ValueError("Only run.mode='train' is implemented currently.")
    if config.run.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("run.device must be 'auto', 'cpu', or 'cuda'.")
    if not config.run.log_dir.strip():
        raise ValueError("run.log_dir must not be empty.")
    if config.data.dataset != "cifar10":
        raise ValueError("Only data.dataset='cifar10' is supported.")
    if config.data.image_size not in {32, 256}:
        raise ValueError("data.image_size must be 32 or 256.")
    if config.data.normalization not in {"cifar10", "zero_one"}:
        raise ValueError("data.normalization must be 'cifar10' or 'zero_one'.")
    if config.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative.")
    if config.model.name != "cifar_cnn":
        raise ValueError("Only model.name='cifar_cnn' is implemented currently.")
    if config.model.num_classes != 10:
        raise ValueError("CIFAR-10 requires model.num_classes=10.")
    if config.train.epochs <= 0:
        raise ValueError("train.epochs must be positive.")
    if config.train.learning_rate <= 0:
        raise ValueError("train.learning_rate must be positive.")
    if config.train.weight_decay < 0:
        raise ValueError("train.weight_decay cannot be negative.")
    if not 0.0 <= config.train.label_smoothing < 1.0:
        raise ValueError("train.label_smoothing must be in [0.0, 1.0).")
