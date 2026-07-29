"""設定名からモデルを生成する。"""

from __future__ import annotations

from .basic_vit import BasicViTFP4, BasicViTFP32, BasicViTINT4, BasicViTINT8, BasicViTTest1, BasicViTTest2, BasicViTTest3, BasicViTTest4, BasicViTUFP4
from .basic_cnn import CNN, FP4CNN, Int4CNN, Int8CNN, MixedCNN, TestCNN, UFP4TestCNN
from .MobileNet_v2 import MobileNetV2FP4, MobileNetV2FP32, MobileNetV2INT4, MobileNetV2INT8, MobileNetV2UFP4
from .ResNet18 import ResNet18FP4, ResNet18FP32, ResNet18INT4, ResNet18INT8, ResNet18UFP4


Model = BasicViTFP32 | BasicViTINT8 | BasicViTINT4 | BasicViTFP4 | BasicViTUFP4 | BasicViTTest1 | BasicViTTest2 | BasicViTTest3 | BasicViTTest4 | CNN | Int8CNN | Int4CNN | FP4CNN | MixedCNN | TestCNN | UFP4TestCNN | ResNet18FP32 | ResNet18INT8 | ResNet18INT4 | ResNet18FP4 | ResNet18UFP4 | MobileNetV2FP32 | MobileNetV2INT8 | MobileNetV2INT4 | MobileNetV2FP4 | MobileNetV2UFP4

AVAILABLE_MODELS = (
    "basic_vit_fp32",
    "basic_vit_int8",
    "basic_vit_int4",
    "basic_vit_fp4",
    "basic_vit_ufp4",
    "basic_vit_test1",
    "basic_vit_test2",
    "basic_vit_test3",
    "basic_vit_test4",
    "cnn",
    "int8_cnn",
    "int4_cnn",
    "fp4_cnn",
    "mixed_cnn",
    "test_cnn",
    "ufp4_test_cnn",
    "resnet18_fp32",
    "resnet18_int8",
    "resnet18_int4",
    "resnet18_fp4",
    "resnet18_ufp4",
    "mobilenet_v2_fp32",
    "mobilenet_v2_int8",
    "mobilenet_v2_int4",
    "mobilenet_v2_fp4",
    "mobilenet_v2_ufp4",
)


def build_model(name: str, num_classes: int, input_bits: int = 8, residual_bits: int | None = None, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> Model:
    """指定されたモデルを生成する。"""
    if name == "basic_vit_fp32":
        return BasicViTFP32(num_classes=num_classes, image_size=image_size)
    if name == "basic_vit_int8":
        selected_residual_bits = 8 if residual_bits is None else residual_bits
        return BasicViTINT8(num_classes=num_classes, input_bits=input_bits, residual_bits=selected_residual_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "basic_vit_int4":
        selected_residual_bits = 4 if residual_bits is None else residual_bits
        return BasicViTINT4(num_classes=num_classes, input_bits=input_bits, residual_bits=selected_residual_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "basic_vit_fp4":
        selected_residual_bits = 4 if residual_bits is None else residual_bits
        return BasicViTFP4(num_classes=num_classes, input_bits=input_bits, residual_bits=selected_residual_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "basic_vit_ufp4":
        selected_residual_bits = 4 if residual_bits is None else residual_bits
        return BasicViTUFP4(num_classes=num_classes, input_bits=input_bits, residual_bits=selected_residual_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "basic_vit_test1":
        selected_residual_bits = 8 if residual_bits is None else residual_bits
        return BasicViTTest1(num_classes=num_classes, input_bits=input_bits, residual_bits=selected_residual_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "basic_vit_test2":
        selected_residual_bits = 4 if residual_bits is None else residual_bits
        return BasicViTTest2(num_classes=num_classes, input_bits=input_bits, residual_bits=selected_residual_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "basic_vit_test3":
        selected_residual_bits = 8 if residual_bits is None else residual_bits
        return BasicViTTest3(num_classes=num_classes, input_bits=input_bits, residual_bits=selected_residual_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "basic_vit_test4":
        selected_residual_bits = 4 if residual_bits is None else residual_bits
        return BasicViTTest4(num_classes=num_classes, input_bits=input_bits, residual_bits=selected_residual_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
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
    if name == "resnet18_int8":
        return ResNet18INT8(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "resnet18_int4":
        return ResNet18INT4(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "resnet18_fp4":
        return ResNet18FP4(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "resnet18_ufp4":
        return ResNet18UFP4(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "mobilenet_v2_fp32":
        return MobileNetV2FP32(num_classes=num_classes, image_size=image_size)
    if name == "mobilenet_v2_int8":
        return MobileNetV2INT8(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "mobilenet_v2_int4":
        return MobileNetV2INT4(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "mobilenet_v2_fp4":
        return MobileNetV2FP4(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    if name == "mobilenet_v2_ufp4":
        return MobileNetV2UFP4(num_classes=num_classes, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    raise ValueError(f"Unsupported model: {name!r}. Available: {list(AVAILABLE_MODELS)}")
