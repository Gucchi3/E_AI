"""E_AI で使用するモデルの明示的な factory。"""

from .cnn import TinyCifarCNN


def build_model(name: str, num_classes: int) -> TinyCifarCNN:
    """Build a supported model without dynamic imports or filesystem scanning."""
    if name != "cifar_cnn":
        raise ValueError(f"Unsupported model: {name!r}. Available: ['cifar_cnn']")
    return TinyCifarCNN(num_classes=num_classes)


__all__ = ["TinyCifarCNN", "build_model"]
