"""ResNet-18 model variants."""

from .ResNet18_fp4 import ResNet18FP4
from .ResNet18_fp32 import BasicBlock, ResNet18FP32
from .ResNet18_int4 import ResNet18INT4
from .ResNet18_int8 import ResNet18INT8
from .ResNet18_ufp4 import ResNet18UFP4


__all__ = ["BasicBlock", "ResNet18FP4", "ResNet18FP32", "ResNet18INT4", "ResNet18INT8", "ResNet18UFP4"]
