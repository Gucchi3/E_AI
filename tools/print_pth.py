"""PyTorchの.pthファイルを確認する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SUPPORTED_BIT_WIDTHS = (2, 4, 8, 16)
SUPPORTED_ROUNDING   = ("ties_away_from_zero", "ties_to_positive", "ties_to_even")


@dataclass(frozen=True)
class QuantizationSettings:
    """整数重みの生成に使う設定。"""

    weight_bits   : int
    rounding      : str
    batch_norm_eps: float



def main() -> None:
    """指定された.pthファイルの内容を表示する。"""
    arguments  = _parse_arguments()
    path       = arguments.path.expanduser().resolve()
    checkpoint = _load_checkpoint(path)
    torch.set_printoptions(profile="full" if arguments.full else "default")
    print(f"path: {path}")
    print(f"type: {type(checkpoint).__name__}")

    if arguments.integer:
        settings                  = _load_quantization_settings(path, arguments)
        container_name, state_dict = _extract_state_dict(checkpoint)
        integer_state             = _integer_state_dict(state_dict, settings)
        print(f"weight_bits: {settings.weight_bits}")
        print(f"rounding: {settings.rounding}")
        if not integer_state:
            print("No integer Tensors or quantized weights were found.")
            return
        _print_value(integer_state, container_name)
        return

    _print_value(checkpoint)


def _parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を読み込む。"""
    parser = argparse.ArgumentParser(description="Print the contents of a PyTorch .pth file.")
    parser.add_argument("path", type=Path, help="Path to the .pth file.")
    parser.add_argument("--full", action="store_true", help="Print every Tensor element without abbreviation.")
    parser.add_argument("--integer", action="store_true", help="Convert and print quantized weights and stored integer Tensors.")
    parser.add_argument("--weight-bits", type=int, choices=SUPPORTED_BIT_WIDTHS, help="Weight bit width. The adjacent config.json value is used by default.")
    parser.add_argument("--rounding", choices=SUPPORTED_ROUNDING, help="Rounding method. The adjacent config.json value is used by default.")
    parser.add_argument("--batch-norm-eps", type=float, default=1e-5, help="BatchNorm epsilon used for folding. Default: 1e-5.")
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Any:
    """CPUへ.pthファイルを読み込む。"""
    if not path.is_file():
        raise FileNotFoundError(f".pth file was not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_quantization_settings(path: Path, arguments: argparse.Namespace) -> QuantizationSettings:
    """CLIまたはconfig.jsonから量子化設定を取得する。"""
    config_path  = path.parent / "config.json"
    quantization = {}
    if config_path.is_file():
        config       = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization", {}) if isinstance(config, Mapping) else {}
        if not isinstance(quantization, Mapping):
            quantization = {}

    weight_bits = arguments.weight_bits if arguments.weight_bits is not None else int(quantization.get("weight_bits", 8))
    rounding    = arguments.rounding if arguments.rounding is not None else str(quantization.get("rounding", "ties_away_from_zero"))
    if weight_bits not in SUPPORTED_BIT_WIDTHS:
        raise ValueError(f"weight_bits must be one of {SUPPORTED_BIT_WIDTHS}.")
    if rounding not in SUPPORTED_ROUNDING:
        raise ValueError(f"rounding must be one of {SUPPORTED_ROUNDING}.")
    if arguments.batch_norm_eps <= 0.0:
        raise ValueError("batch_norm_eps must be positive.")
    return QuantizationSettings(weight_bits=weight_bits, rounding=rounding, batch_norm_eps=arguments.batch_norm_eps)


def _extract_state_dict(checkpoint: Any) -> tuple[str | None, Mapping[str, Any]]:
    """一般的なcheckpoint形式からstate_dictを取得する。"""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("The .pth file does not contain a state_dict mapping.")
    for name in ("model", "state_dict"):
        value = checkpoint.get(name)
        if isinstance(value, Mapping):
            return name, value
    return None, checkpoint


def _integer_state_dict(state_dict: Mapping[str, Any], settings: QuantizationSettings) -> dict[str, torch.Tensor]:
    """整数重みと保存済み整数Tensorだけのstate_dictを作る。"""
    integer_state = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            continue
        integer_weight = _quantize_weight(name, value, state_dict, settings)
        if integer_weight is not None:
            integer_name, integer_value = integer_weight
            integer_state[integer_name] = integer_value
        if _is_integer_tensor(value) and not name.endswith("num_batches_tracked"):
            integer_state[name] = value
    return integer_state


def _quantize_weight(name: str, value: torch.Tensor, state_dict: Mapping[str, Any], settings: QuantizationSettings) -> tuple[str, torch.Tensor] | None:
    """量子化層のFP32重みを整数へ変換する。"""
    if not value.is_floating_point():
        return None

    batch_norm_prefix = _batch_norm_prefix(name, state_dict)
    if batch_norm_prefix is not None:
        folded_weight = _fold_batch_norm_weight(value, batch_norm_prefix, state_dict, settings.batch_norm_eps)
        return _join_name(batch_norm_prefix, "weight_integer"), _integer_weight(folded_weight, settings)

    layer_prefix = _layer_prefix(name)
    if layer_prefix is None or _join_name(layer_prefix, "weight_quantizer.scale") not in state_dict:
        return None
    return _join_name(layer_prefix, "weight_integer"), _integer_weight(value, settings)


def _batch_norm_prefix(name: str, state_dict: Mapping[str, Any]) -> str | None:
    """Conv+BN量子化層のprefixを返す。"""
    suffix = ".conv.weight"
    if not name.endswith(suffix):
        return None
    prefix = name[: -len(suffix)]
    required_names = (
        _join_name(prefix, "norm.running_mean"),
        _join_name(prefix, "norm.running_var"),
        _join_name(prefix, "weight_quantizer.scale"),
    )
    return prefix if all(required_name in state_dict for required_name in required_names) else None


def _layer_prefix(name: str) -> str | None:
    """通常の量子化ConvまたはLinearのprefixを返す。"""
    if name == "weight":
        return ""
    suffix = ".weight"
    return name[: -len(suffix)] if name.endswith(suffix) else None


def _fold_batch_norm_weight(weight: torch.Tensor, prefix: str, state_dict: Mapping[str, Any], epsilon: float) -> torch.Tensor:
    """running統計でBatchNormを重みへfoldする。"""
    running_var = _required_tensor(state_dict, _join_name(prefix, "norm.running_var")).to(dtype=weight.dtype)
    norm_weight = state_dict.get(_join_name(prefix, "norm.weight"))
    gamma       = norm_weight.to(dtype=weight.dtype) if isinstance(norm_weight, torch.Tensor) else torch.ones_like(running_var)
    scale       = gamma / torch.sqrt(running_var + epsilon)
    shape       = (weight.size(0),) + (1,) * (weight.ndim - 1)
    return weight * scale.reshape(shape)


def _integer_weight(weight: torch.Tensor, settings: QuantizationSettings) -> torch.Tensor:
    """チャンネル別scaleで重みを整数へ変換する。"""
    if weight.ndim < 2:
        raise ValueError("Quantized weight must have at least two dimensions.")
    qmin       = -(2 ** (settings.weight_bits - 1))
    qmax       = 2 ** (settings.weight_bits - 1) - 1
    dimensions = tuple(range(1, weight.ndim))
    magnitude  = torch.maximum(weight.amin(dim=dimensions).abs(), weight.amax(dim=dimensions).abs()).to(dtype=torch.float32)
    scale      = (magnitude / qmax).clamp_min(torch.finfo(torch.float32).eps)
    shape      = (weight.size(0),) + (1,) * (weight.ndim - 1)
    rounded    = _round(weight / scale.to(dtype=weight.dtype).reshape(shape), settings.rounding)
    dtype      = torch.int8 if settings.weight_bits <= 8 else torch.int16
    return rounded.clamp(qmin, qmax).to(dtype=dtype)


def _round(value: torch.Tensor, rounding: str) -> torch.Tensor:
    """指定された方式でTensorを丸める。"""
    if rounding == "ties_away_from_zero":
        return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)
    if rounding == "ties_to_positive":
        return torch.floor(value + 0.5)
    return torch.round(value)


def _required_tensor(state_dict: Mapping[str, Any], name: str) -> torch.Tensor:
    """state_dictから必須Tensorを取得する。"""
    value = state_dict.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Tensor was not found in state_dict: {name}")
    return value


def _join_name(prefix: str, suffix: str) -> str:
    """state_dictのprefixと名前を連結する。"""
    return f"{prefix}.{suffix}" if prefix else suffix


def _print_value(value: Any, name: str | None = None, indent: int = 0) -> None:
    """入れ子構造をTensor単位で表示する。"""
    prefix = " " * indent
    if isinstance(value, torch.Tensor):
        label = f"{name}: " if name is not None else ""
        print(f"{prefix}{label}shape={tuple(value.shape)}, dtype={value.dtype}")
        print(f"{prefix}{value}")
        return

    if isinstance(value, Mapping):
        if name is not None:
            print(f"{prefix}{name}:")
        for key, item in value.items():
            _print_value(item, str(key), indent + (2 if name is not None else 0))
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if name is not None:
            print(f"{prefix}{name}:")
        child_indent = indent + (2 if name is not None else 0)
        for index, item in enumerate(value):
            _print_value(item, f"[{index}]", child_indent)
        return

    label = f"{name}: " if name is not None else ""
    print(f"{prefix}{label}{value!r}")


def _is_integer_tensor(value: torch.Tensor) -> bool:
    """Tensorがbool以外の整数型かを返す。"""
    return not value.is_floating_point() and not value.is_complex() and value.dtype != torch.bool


if __name__ == "__main__":
    main()
