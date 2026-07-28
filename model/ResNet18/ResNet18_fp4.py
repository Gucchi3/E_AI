"""FP4 E2M1 QAT ResNet-18 for 32x32 CIFAR-10 images."""

from __future__ import annotations

import torch
from torch import nn

from utils.quantization import FP4Quantizer, IntegerQuantizer, QuantConv2d, QuantLinear


class FP4BasicBlock(nn.Module):
    """FP4 E2M1 residual block."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, rounding: str, activation_range_momentum: float) -> None:
        super().__init__()
        self.conv1               = QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.bn1                 = nn.Identity()
        self.relu1               = nn.ReLU(inplace=False)
        self.conv1_quantizer     = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        self.conv2               = QuantConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.bn2                 = nn.Identity()
        self.conv2_quantizer     = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        self.shortcut_quantizer  = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum) if stride != 1 or in_channels != out_channels else None
        self.shortcut            = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(QuantConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4"), nn.Identity())
        self.relu2               = nn.ReLU(inplace=False)
        self.output_quantizer    = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)

    @property
    def output_scale(self) -> torch.Tensor:
        return self.output_quantizer.scale

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        if isinstance(self.shortcut, nn.Identity):
            identity = value
        else:
            if self.shortcut_quantizer is None:
                raise RuntimeError("Projection shortcut quantizer is not initialized.")
            identity = self.shortcut_quantizer(self.shortcut[0](value, input_scale))

        value = self.conv1(value, input_scale)
        value = self.conv1_quantizer(self.relu1(value))
        value = self.conv2_quantizer(self.conv2(value, self.conv1_quantizer.scale))
        # Fake-quantized branches are dequantized float values here; the sum is quantized again immediately below.
        return self.output_quantizer(self.relu2(value + identity))


class ResNet18FP4(nn.Module):
    """CIFAR ResNet-18 with FP4 E2M1 body weights and activations."""

    def __init__(self, num_classes: int = 10, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("ResNet18FP4 is adapted specifically for 32x32 CIFAR images.")
        if input_bits != 8:
            raise ValueError("ResNet18FP4 requires input_bits=8.")

        self.in_channels      = 64
        self.input_quantizer  = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, fixed_scale=1.0 / 255.0)
        self.stem             = nn.Sequential(QuantConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True, weight_bits=8, rounding=rounding, quantizer="integer"), nn.Identity(), nn.ReLU(inplace=False))
        self.stem_quantizer   = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        self.stage1           = self._make_stage(out_channels=64, block_count=2, stride=1, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage2           = self._make_stage(out_channels=128, block_count=2, stride=2, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage3           = self._make_stage(out_channels=256, block_count=2, stride=2, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.stage4           = self._make_stage(out_channels=512, block_count=2, stride=2, rounding=rounding, activation_range_momentum=activation_range_momentum)
        self.pool_quantizer   = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, range_momentum=activation_range_momentum)
        self.pool             = nn.AvgPool2d(kernel_size=4, stride=4)
        self.output_quantizer = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, range_momentum=activation_range_momentum)
        self.classifier       = QuantLinear(512, num_classes, weight_bits=8, rounding=rounding, quantizer="integer")

        self._initialize_weights()

    def _make_stage(self, out_channels: int, block_count: int, stride: int, rounding: str, activation_range_momentum: float) -> nn.Sequential:
        blocks = [FP4BasicBlock(self.in_channels, out_channels, stride, rounding, activation_range_momentum)]
        self.in_channels = out_channels
        blocks.extend(FP4BasicBlock(self.in_channels, out_channels, 1, rounding, activation_range_momentum) for _ in range(1, block_count))
        return nn.Sequential(*blocks)

    def _forward_stage(self, stage: nn.Sequential, value: torch.Tensor, input_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = input_scale
        for block in stage:
            if not isinstance(block, FP4BasicBlock):
                raise TypeError("FP4 ResNet stages must contain FP4BasicBlock modules.")
            value = block(value, scale)
            scale = block.output_scale
        return value, scale

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_quantizer(value)
        value = self.stem_quantizer(self.stem[2](self.stem[0](value, self.input_quantizer.scale)))
        value, scale = self._forward_stage(self.stage1, value, self.stem_quantizer.scale)
        value, scale = self._forward_stage(self.stage2, value, scale)
        value, scale = self._forward_stage(self.stage3, value, scale)
        value, _ = self._forward_stage(self.stage4, value, scale)
        value = self.pool_quantizer(value)
        value = self.output_quantizer(self.pool(value))
        value = torch.flatten(value, start_dim=1)
        return self.classifier(value, self.output_quantizer.scale)
