"""モデルを名前から生成する。"""

from .cnn import TinyCifarCNN
from .qat_cnn import TinyQATCNN


def build_model(name: str, num_classes: int, weight_bits: int = 8, activation_bits: int = 8, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> TinyCifarCNN | TinyQATCNN:
    """指定されたモデルを生成する。"""
    if name == "cifar_cnn":
        return TinyCifarCNN(num_classes=num_classes, image_size=image_size)
    if name == "qat_cifar_cnn":
        return TinyQATCNN(num_classes=num_classes, weight_bits=weight_bits, activation_bits=activation_bits, input_bits=input_bits, rounding=rounding, activation_range_momentum=activation_range_momentum, image_size=image_size)
    raise ValueError(f"Unsupported model: {name!r}. Available: ['cifar_cnn', 'qat_cifar_cnn']")


__all__ = ["TinyCifarCNN", "TinyQATCNN", "build_model"]
