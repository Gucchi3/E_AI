"""デバイスと乱数seedを設定する。"""

from __future__ import annotations

import random

import torch


def select_device(requested_device: str) -> torch.device:
    """設定値から実行デバイスを選ぶ。"""
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("run.device='cuda' was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """PythonとPyTorchの乱数seedを揃える。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
