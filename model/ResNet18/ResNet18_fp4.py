"""32x32のCIFAR-10入力に合わせたFP4 E2M1 QAT ResNet-18。"""

from __future__ import annotations

import torch
from torch import nn

from utils.quantization import FP4Quantizer, IntegerQuantizer, QuantConv2d, QuantLinear, QuantResidualAdd, QuantizedAvgPool2d


class FP4BasicBlock(nn.Module):
    """FP4 ConvとINT4残差加算で構成するBasicBlock。"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, rounding: str, activation_range_momentum: float) -> None:
        super().__init__()
        use_projection = stride != 1 or in_channels != out_channels

        self.conv1            = QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.bn1              = nn.Identity()
        self.relu1            = nn.ReLU(inplace=False)
        self.conv1_quantizer  = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        self.conv2            = QuantConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.bn2              = nn.Identity()
        self.conv2_quantizer  = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        self.residual_add     = QuantResidualAdd(bit_width=4, rounding=rounding, range_momentum=activation_range_momentum)
        self.relu2            = nn.ReLU(inplace=False)
        self.output_quantizer = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)

        if use_projection:
            self.shortcut           = nn.Sequential(QuantConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4"), nn.Identity())
            self.shortcut_quantizer = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        else:
            self.shortcut           = nn.Identity()
            self.shortcut_quantizer = None

    @property
    def output_scale(self) -> torch.Tensor:
        """次のblockへ渡すFP4 scaleを返す。"""
        return self.output_quantizer.scale

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        identity       = value
        identity_scale = input_scale

        if not isinstance(self.shortcut, nn.Identity):
            if self.shortcut_quantizer is None:
                raise RuntimeError("Projection shortcut quantizer is not initialized.")
            identity       = self.shortcut[0](value, input_scale / 4.0)
            identity       = self.shortcut[1](identity)
            identity       = self.shortcut_quantizer(identity)
            identity_scale = self.shortcut_quantizer.scale

        transformed = self.conv1(value, input_scale / 4.0)
        transformed = self.bn1(transformed)
        transformed = self.relu1(transformed)
        transformed = self.conv1_quantizer(transformed)

        transformed = self.conv2(transformed, self.conv1_quantizer.scale / 4.0)
        transformed = self.bn2(transformed)
        transformed = self.conv2_quantizer(transformed)

        transformed, _ = self.residual_add(transformed, self.conv2_quantizer.scale, identity, identity_scale)
        transformed    = self.relu2(transformed)
        transformed    = self.output_quantizer(transformed)
        return transformed


class ResNet18FP4(nn.Module):
    """Stem／分類層をINT8、BodyをFP4、残差加算をINT4にしたResNet-18。"""

    def __init__(self, num_classes: int = 10, input_bits: int = 8, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("ResNet18FP4 is adapted specifically for 32x32 CIFAR images.")
        if input_bits != 8:
            raise ValueError("ResNet18FP4 requires input_bits=8.")

        self.input_quantizer  = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, fixed_scale=1.0 / 255.0)
        self.stem             = nn.Sequential(QuantConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True, weight_bits=8, rounding=rounding, quantizer="integer"), nn.Identity(), nn.ReLU(inplace=False))
        self.stem_quantizer   = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        self.stage1           = nn.Sequential(FP4BasicBlock(64, 64, 1, rounding, activation_range_momentum), FP4BasicBlock(64, 64, 1, rounding, activation_range_momentum))
        self.stage2           = nn.Sequential(FP4BasicBlock(64, 128, 2, rounding, activation_range_momentum), FP4BasicBlock(128, 128, 1, rounding, activation_range_momentum))
        self.stage3           = nn.Sequential(FP4BasicBlock(128, 256, 2, rounding, activation_range_momentum), FP4BasicBlock(256, 256, 1, rounding, activation_range_momentum))
        self.stage4           = nn.Sequential(FP4BasicBlock(256, 512, 2, rounding, activation_range_momentum), FP4BasicBlock(512, 512, 1, rounding, activation_range_momentum))
        self.pool_quantizer   = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, range_momentum=activation_range_momentum)
        self.pool             = QuantizedAvgPool2d(kernel_size=4, stride=4, bit_width=8, signed=False, rounding=rounding)
        self.output_quantizer = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, range_momentum=activation_range_momentum)
        self.classifier       = QuantLinear(512, num_classes, weight_bits=8, rounding=rounding, quantizer="integer")

        self._initialize_weights()

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
        stage1_block1 = self.stage1[0]
        stage1_block2 = self.stage1[1]
        stage2_block1 = self.stage2[0]
        stage2_block2 = self.stage2[1]
        stage3_block1 = self.stage3[0]
        stage3_block2 = self.stage3[1]
        stage4_block1 = self.stage4[0]
        stage4_block2 = self.stage4[1]

        value = self.input_quantizer(value)
        value = self.stem[0](value, self.input_quantizer.scale)
        value = self.stem[1](value)
        value = self.stem[2](value)
        value = self.stem_quantizer(value)

        value = stage1_block1(value, self.stem_quantizer.scale)
        value = stage1_block2(value, stage1_block1.output_scale)
        value = stage2_block1(value, stage1_block2.output_scale)
        value = stage2_block2(value, stage2_block1.output_scale)
        value = stage3_block1(value, stage2_block2.output_scale)
        value = stage3_block2(value, stage3_block1.output_scale)
        value = stage4_block1(value, stage3_block2.output_scale)
        value = stage4_block2(value, stage4_block1.output_scale)

        value    = self.pool_quantizer(value)
        value, _ = self.pool(value, self.pool_quantizer.scale)
        value    = self.output_quantizer(value)
        value    = torch.flatten(value, start_dim=1)
        value    = self.classifier(value, self.output_quantizer.scale)
        return value
