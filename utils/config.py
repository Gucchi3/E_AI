"""JSON設定を読み込み、値を検証する。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_NAMES = frozenset({"basic_vit_fp32", "basic_vit_int8", "basic_vit_int4", "basic_vit_fp4", "basic_vit_ufp4", "basic_vit_test1", "basic_vit_test2", "basic_vit_test3", "basic_vit_test4", "basic_vit_test5", "cnn", "int8_cnn", "int4_cnn", "fp4_cnn", "mixed_cnn", "test_cnn", "ufp4_test_cnn", "resnet18_fp32", "resnet18_int8", "resnet18_int4", "resnet18_fp4", "resnet18_ufp4", "mobilenet_v2_fp32", "mobilenet_v2_int8", "mobilenet_v2_int4", "mobilenet_v2_fp4", "mobilenet_v2_ufp4"})
FP32_MODEL_NAMES = frozenset({"basic_vit_fp32", "cnn", "resnet18_fp32", "mobilenet_v2_fp32"})
QUANTIZED_MODEL_NAMES = MODEL_NAMES - FP32_MODEL_NAMES
MODEL_BIT_WIDTHS = {
    "basic_vit_int8": 8,
    "basic_vit_int4": 4,
    "basic_vit_fp4": 4,
    "basic_vit_ufp4": 4,
    "basic_vit_test1": 4,
    "basic_vit_test2": 4,
    "basic_vit_test3": 4,
    "basic_vit_test4": 4,
    "basic_vit_test5": 4,
    "int8_cnn": 8,
    "int4_cnn": 4,
    "fp4_cnn": 4,
    "mixed_cnn": 4,
    "test_cnn": 4,
    "ufp4_test_cnn": 4,
    "resnet18_int8": 8,
    "resnet18_int4": 4,
    "resnet18_fp4": 4,
    "resnet18_ufp4": 4,
    "mobilenet_v2_int8": 8,
    "mobilenet_v2_int4": 4,
    "mobilenet_v2_fp4": 4,
    "mobilenet_v2_ufp4": 4,
}
MODEL_RESIDUAL_BIT_WIDTHS = {
    "basic_vit_int8": 8,
    "basic_vit_int4": 4,
    "basic_vit_fp4": 4,
    "basic_vit_ufp4": 4,
    "basic_vit_test1": 8,
    "basic_vit_test2": 4,
    "basic_vit_test3": 8,
    "basic_vit_test4": 4,
    "basic_vit_test5": 8,
}


@dataclass(frozen=True)
class RunConfig:
    """実行全体の設定。"""

    mode   : str
    seed   : int
    device : str
    log_dir: str



@dataclass(frozen=True)
class DataConfig:
    """データセットの設定。"""

    dataset      : str
    root         : str
    image_size   : int
    normalization: str
    batch_size   : int
    num_workers  : int



@dataclass(frozen=True)
class ModelConfig:
    """モデルの設定。"""

    name        : str
    num_classes : int
    load_weight : bool
    weight_path : str



@dataclass(frozen=True)
class QuantizationConfig:
    """QATの量子化設定。"""

    weight_bits              : int
    activation_bits          : int
    input_bits               : int
    residual_bits            : int
    rounding                 : str
    activation_range_momentum: float



@dataclass(frozen=True)
class TrainConfig:
    """学習条件の設定。"""

    epochs                      : int
    learning_rate               : float
    minimum_learning_rate       : float
    weight_decay                : float
    label_smoothing             : float
    mixup_alpha                 : float
    cutmix_alpha                : float
    mixup_probability           : float
    mixup_switch_probability    : float



@dataclass(frozen=True)
class AppConfig:
    """E_AIの全設定。"""

    run         : RunConfig
    data        : DataConfig
    model       : ModelConfig
    quantization: QuantizationConfig
    train       : TrainConfig


def load_config(path: str | Path) -> AppConfig:
    """JSON設定を読み込む。"""
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
    quant_raw = _optional_section(raw, "quantization", {"weight_bits": 8, "activation_bits": 8, "input_bits": 8, "rounding": "ties_away_from_zero", "activation_range_momentum": 0.95})
    train_raw = _section(raw, "train")

    run_config          = RunConfig(mode=_string(run_raw, "mode"), seed=_integer(run_raw, "seed"), device=_string(run_raw, "device"), log_dir=_string(run_raw, "log_dir"))
    data_config         = DataConfig(dataset=_string(data_raw, "dataset"), root=_string(data_raw, "root"), image_size=_integer(data_raw, "image_size"), normalization=_string(data_raw, "normalization"), batch_size=_integer(data_raw, "batch_size"), num_workers=_integer(data_raw, "num_workers"))
    model_config        = ModelConfig(name=_string(model_raw, "name"), num_classes=_integer(model_raw, "num_classes"), load_weight=_boolean(model_raw, "load_weight"), weight_path=_string(model_raw, "weight_path"))
    quantization_config = QuantizationConfig(weight_bits=_integer(quant_raw, "weight_bits"), activation_bits=_integer(quant_raw, "activation_bits"), input_bits=_integer(quant_raw, "input_bits"), residual_bits=_integer_or_default(quant_raw, "residual_bits", _integer(quant_raw, "activation_bits")), rounding=_string(quant_raw, "rounding"), activation_range_momentum=_number_or_default(quant_raw, "activation_range_momentum", 0.95))
    train_config        = TrainConfig(epochs=_integer(train_raw, "epochs"), learning_rate=_number(train_raw, "learning_rate"), minimum_learning_rate=_number_or_default(train_raw, "minimum_learning_rate", 0.0), weight_decay=_number(train_raw, "weight_decay"), label_smoothing=_number(train_raw, "label_smoothing"), mixup_alpha=_number_or_default(train_raw, "mixup_alpha", 0.0), cutmix_alpha=_number_or_default(train_raw, "cutmix_alpha", 0.0), mixup_probability=_number_or_default(train_raw, "mixup_probability", 1.0), mixup_switch_probability=_number_or_default(train_raw, "mixup_switch_probability", 0.5))
    config              = AppConfig(run=run_config, data=data_config, model=model_config, quantization=quantization_config, train=train_config)
    _validate(config)

    return config


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """設定のsectionを取り出す。"""
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be an object.")
    return value


def _optional_section(raw: dict[str, Any], name: str, default: dict[str, Any]) -> dict[str, Any]:
    """省略できる設定sectionを取得する。"""
    value = raw.get(name, default)
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be an object.")
    return value


def _string(raw: dict[str, Any], name: str) -> str:
    """文字列の設定値を取り出す。"""
    value = raw.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Config value {name!r} must be a string.")
    return value


def _integer(raw: dict[str, Any], name: str) -> int:
    """整数の設定値を取り出す。"""
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config value {name!r} must be an integer.")
    return value


def _integer_or_default(raw: dict[str, Any], name: str, default: int) -> int:
    """省略可能な整数設定を取り出す。"""
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config value {name!r} must be an integer.")
    return value


def _number(raw: dict[str, Any], name: str) -> float:
    """数値の設定値を取り出す。"""
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config value {name!r} must be a number.")
    return float(value)


def _number_or_default(raw: dict[str, Any], name: str, default: float) -> float:
    """省略できる数値設定を取り出す。"""
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config value {name!r} must be a number.")
    return float(value)


def _boolean(raw: dict[str, Any], name: str) -> bool:
    """真偽値の設定値を取り出す。"""
    value = raw.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"Config value {name!r} must be a boolean.")
    return value


def _validate(config: AppConfig) -> None:
    """設定値の組み合わせを検証する。"""
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
    if config.model.name not in MODEL_NAMES:
        raise ValueError(f"model.name must be one of {sorted(MODEL_NAMES)}.")
    if config.model.num_classes != 10:
        raise ValueError("CIFAR-10 requires model.num_classes=10.")
    if config.model.load_weight and not config.model.weight_path.strip():
        raise ValueError("model.weight_path must not be empty when model.load_weight=true.")
    if config.quantization.weight_bits not in {2, 4, 8, 16}:
        raise ValueError("quantization.weight_bits must be 2, 4, 8, or 16.")
    if config.quantization.activation_bits not in {2, 4, 8, 16}:
        raise ValueError("quantization.activation_bits must be 2, 4, 8, or 16.")
    if config.quantization.input_bits != 8:
        raise ValueError("quantization.input_bits must be 8 because image input is uint8.")
    if config.quantization.residual_bits not in {2, 4, 8, 16}:
        raise ValueError("quantization.residual_bits must be 2, 4, 8, or 16.")
    if config.quantization.rounding not in {"ties_away_from_zero", "ties_to_positive", "ties_to_even"}:
        raise ValueError("Unsupported quantization.rounding value.")
    if not 0.0 <= config.quantization.activation_range_momentum < 1.0:
        raise ValueError("quantization.activation_range_momentum must be in [0.0, 1.0).")
    expected_bits = MODEL_BIT_WIDTHS.get(config.model.name)
    if expected_bits is not None and config.quantization.weight_bits != expected_bits:
        raise ValueError(f"{config.model.name} requires quantization.weight_bits={expected_bits}.")
    if expected_bits is not None and config.quantization.activation_bits != expected_bits:
        raise ValueError(f"{config.model.name} requires quantization.activation_bits={expected_bits}.")
    expected_residual_bits = MODEL_RESIDUAL_BIT_WIDTHS.get(config.model.name)
    if expected_residual_bits is not None and config.quantization.residual_bits != expected_residual_bits:
        raise ValueError(f"{config.model.name} requires quantization.residual_bits={expected_residual_bits}.")
    if config.model.name in QUANTIZED_MODEL_NAMES and config.data.normalization != "zero_one":
        raise ValueError("Quantized models require data.normalization='zero_one' for uint8 input.")
    if config.train.epochs <= 0:
        raise ValueError("train.epochs must be positive.")
    if config.train.learning_rate <= 0:
        raise ValueError("train.learning_rate must be positive.")
    if not 0.0 <= config.train.minimum_learning_rate < config.train.learning_rate:
        raise ValueError("train.minimum_learning_rate must be greater than or equal to 0 and smaller than train.learning_rate.")
    if config.train.weight_decay < 0:
        raise ValueError("train.weight_decay cannot be negative.")
    if not 0.0 <= config.train.label_smoothing < 1.0:
        raise ValueError("train.label_smoothing must be in [0.0, 1.0).")
    if config.train.mixup_alpha < 0.0:
        raise ValueError("train.mixup_alpha cannot be negative.")
    if config.train.cutmix_alpha < 0.0:
        raise ValueError("train.cutmix_alpha cannot be negative.")
    if not 0.0 <= config.train.mixup_probability <= 1.0:
        raise ValueError("train.mixup_probability must be in [0.0, 1.0].")
    if not 0.0 <= config.train.mixup_switch_probability <= 1.0:
        raise ValueError("train.mixup_switch_probability must be in [0.0, 1.0].")
