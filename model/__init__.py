"""モデルを名前から生成する。"""

from .cnn import TinyCifarCNN


def build_model(name: str, num_classes: int) -> TinyCifarCNN:
    """指定されたモデルを生成する。"""
    if name != "cifar_cnn":
        raise ValueError(f"Unsupported model: {name!r}. Available: ['cifar_cnn']")
    return TinyCifarCNN(num_classes=num_classes)


__all__ = ["TinyCifarCNN", "build_model"]
