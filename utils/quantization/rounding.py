"""量子化で使用する丸め処理。"""

from __future__ import annotations

from collections.abc import Callable

import torch


RoundingFunction = Callable[[torch.Tensor], torch.Tensor]


def round_ties_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    """中間値を絶対値が大きい側へ丸める。"""
    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


def round_ties_to_positive(value: torch.Tensor) -> torch.Tensor:
    """中間値を正の方向へ丸める。"""
    return torch.floor(value + 0.5)


def round_ties_to_even(value: torch.Tensor) -> torch.Tensor:
    """中間値を最も近い偶数へ丸める。"""
    return torch.round(value)


ROUNDING_FUNCTIONS: dict[str, RoundingFunction] = {
    "ties_away_from_zero": round_ties_away_from_zero,
    "ties_to_positive"  : round_ties_to_positive,
    "ties_to_even"      : round_ties_to_even,
}


def get_rounding_function(name: str) -> RoundingFunction:
    """名前に対応する丸め関数を返す。"""
    try:
        return ROUNDING_FUNCTIONS[name]
    except KeyError as error:
        choices = ", ".join(ROUNDING_FUNCTIONS)
        raise ValueError(f"Unsupported rounding mode: {name!r}. Available: {choices}") from error
