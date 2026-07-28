"""設定名からモデルを生成する。"""

from __future__ import annotations

from .basic_cnn import CNN, FP4CNN, Int4CNN, Int8CNN, MixedCNN, TestCNN, UFP4TestCNN
from .MobileNet_v2 import MobileNetV2FP32
from .ResNet18 import ResNet18FP32


Model = CNN | Int8CNN | Int4CNN | FP4CNN | MixedCNN | TestCNN | UFP4TestCNN | ResNet18FP32 | MobileNetV2FP32

AVAILABLE_MODELS = (
    "cnn",
    "int8_cnn",
    "int4_cnn",
    "fp4_cnn",
    "mixed_cnn",
    "test_cnn",
    "ufp4_test_cnn",
    "resnet18_fp32",
    "mobilenet_v2_fp32",
)


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
    if name == "ufp4_test_cnn":
        return UFP4TestCNN(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "resnet18_fp32":
        return ResNet18FP32(num_classes=num_classes, image_size=image_size)
    if name == "mobilenet_v2_fp32":
        return MobileNetV2FP32(num_classes=num_classes, image_size=image_size)
    raise ValueError(f"Unsupported model: {name!r}. Available: {list(AVAILABLE_MODELS)}")
