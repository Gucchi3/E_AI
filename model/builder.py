"""設定名からモデルを生成する。"""

from __future__ import annotations

from .cnn import CNN
from .fp4_cnn import FP4CNN
from .int4_cnn import Int4CNN
from .int8_cnn import Int8CNN
from .mixed_cnn import MixedCNN
from .test_cnn import TestCNN


Model = CNN | Int8CNN | Int4CNN | FP4CNN | MixedCNN | TestCNN


def build_model(name: str, num_classes: int, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> Model:
    """指定されたモデルを生成する。"""
    if name == "cnn":
        return CNN(num_classes=num_classes, image_size=image_size)
    if name == "int8_cnn":
        return Int8CNN(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "int4_cnn":
        return Int4CNN(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "fp4_cnn":
        return FP4CNN(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "mixed_cnn":
        return MixedCNN(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "test_cnn":
        return TestCNN(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    raise ValueError(f"Unsupported model: {name!r}. Available: ['cnn', 'int8_cnn', 'int4_cnn', 'fp4_cnn', 'mixed_cnn', 'test_cnn']")
