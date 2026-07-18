"""PyTorchの.pthファイルを確認する。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    """指定された.pthファイルの内容を表示する。"""
    arguments  = _parse_arguments()
    path       = arguments.path.expanduser().resolve()
    checkpoint = _load_checkpoint(path)
    torch.set_printoptions(profile="full" if arguments.full else "default")
    print(f"path: {path}")
    print(f"type: {type(checkpoint).__name__}")
    if not _print_value(checkpoint, integer_only=arguments.integer):
        print("No integer Tensors were found.")


def _parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を読み込む。"""
    parser = argparse.ArgumentParser(description="Print the contents of a PyTorch .pth file.")
    parser.add_argument("path", type=Path, help="Path to the .pth file.")
    parser.add_argument("--full", action="store_true", help="Print every Tensor element without abbreviation.")
    parser.add_argument("--integer", action="store_true", help="Print only integer Tensors.")
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Any:
    """CPUへ.pthファイルを読み込む。"""
    if not path.is_file():
        raise FileNotFoundError(f".pth file was not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _print_value(value: Any, name: str | None = None, indent: int = 0, integer_only: bool = False) -> bool:
    """入れ子構造をTensor単位で表示する。"""
    if not _contains_printable_value(value, integer_only):
        return False

    prefix = " " * indent
    if isinstance(value, torch.Tensor):
        label = f"{name}: " if name is not None else ""
        print(f"{prefix}{label}shape={tuple(value.shape)}, dtype={value.dtype}")
        print(f"{prefix}{value}")
        return True

    if isinstance(value, Mapping):
        if name is not None:
            print(f"{prefix}{name}:")
        for key, item in value.items():
            _print_value(item, str(key), indent + (2 if name is not None else 0), integer_only)
        return True

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if name is not None:
            print(f"{prefix}{name}:")
        child_indent = indent + (2 if name is not None else 0)
        for index, item in enumerate(value):
            _print_value(item, f"[{index}]", child_indent, integer_only)
        return True

    label = f"{name}: " if name is not None else ""
    print(f"{prefix}{label}{value!r}")
    return True


def _contains_printable_value(value: Any, integer_only: bool) -> bool:
    """表示対象の値が含まれるかを返す。"""
    if isinstance(value, torch.Tensor):
        return not integer_only or _is_integer_tensor(value)
    if isinstance(value, Mapping):
        return any(_contains_printable_value(item, integer_only) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_printable_value(item, integer_only) for item in value)
    return not integer_only


def _is_integer_tensor(value: torch.Tensor) -> bool:
    """Tensorがbool以外の整数型かを返す。"""
    return not value.is_floating_point() and not value.is_complex() and value.dtype != torch.bool


if __name__ == "__main__":
    main()
