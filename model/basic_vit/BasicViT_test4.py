"""残差接続だけを独立INT4 scaleにしたBasicViTのUFP4テストモデル。"""

from __future__ import annotations

import math

import torch
from torch import nn

from utils.quantization import FP4MatMul, FP4Quantizer, IntegerQuantizer, PerTokenFP4NEQ, QuantConv2d, QuantLinear, QuantResidualAdd, QuantizedAvgPool2d, UFP4Quantizer


class Test4BasicViTStem(nn.Module):
    """最初のConvをINT8、ReLU6後のStem内部活性化をUFP4にする。"""

    def __init__(self, embed_dim: int, rounding: str, activation_range_momentum: float) -> None:
        super().__init__()
        middle_channels = embed_dim // 2

        self.conv1      = QuantConv2d(3, middle_channels, kernel_size=3, stride=2, padding=1, bias=True, weight_bits=8, rounding=rounding, quantizer="integer")
        self.bn1        = nn.Identity()
        self.relu1      = nn.ReLU6(inplace=False)
        self.quantizer1 = UFP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)

        self.conv2_depthwise      = QuantConv2d(middle_channels, middle_channels, kernel_size=3, stride=2, padding=1, groups=middle_channels, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.bn2_depthwise        = nn.Identity()
        self.relu2_depthwise      = nn.ReLU6(inplace=False)
        self.quantizer2_depthwise = UFP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)

        self.conv2_pointwise = QuantConv2d(middle_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.bn2_pointwise   = nn.Identity()
        self.relu2_pointwise = nn.ReLU6(inplace=False)

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value, input_scale)
        value = self.bn1(value)
        value = self.relu1(value)
        value = self.quantizer1(value)

        value = self.conv2_depthwise(value, self.quantizer1.scale / 8.0)
        value = self.bn2_depthwise(value)
        value = self.relu2_depthwise(value)
        value = self.quantizer2_depthwise(value)

        value = self.conv2_pointwise(value, self.quantizer2_depthwise.scale / 8.0)
        value = self.bn2_pointwise(value)
        value = self.relu2_pointwise(value)
        return value


class Test4ConvolutionalFeedForward(nn.Module):
    """FP4重み・UFP4活性化とINT4 projection出力を使用する畳み込みFFN。"""

    def __init__(self, channels: int, hidden_channels: int, rounding: str, activation_range_momentum: float) -> None:
        super().__init__()
        self.expand_conv      = QuantConv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.expand_bn        = nn.Identity()
        self.expand_relu      = nn.ReLU6(inplace=False)
        self.expand_quantizer = UFP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)

        self.depthwise_conv      = QuantConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.depthwise_bn        = nn.Identity()
        self.depthwise_relu      = nn.ReLU6(inplace=False)
        self.depthwise_quantizer = UFP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)

        self.project_conv = QuantConv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.project_bn   = nn.Identity()

    @property
    def output_scale(self) -> torch.Tensor:
        """残差加算へ渡すProjection ConvのINT32 accumulator scaleを返す。"""
        return self.project_conv.accumulator_scale

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        value = self.expand_conv(value, input_scale / 2.0)
        value = self.expand_bn(value)
        value = self.expand_relu(value)
        value = self.expand_quantizer(value)

        value = self.depthwise_conv(value, self.expand_quantizer.scale / 8.0)
        value = self.depthwise_bn(value)
        value = self.depthwise_relu(value)
        value = self.depthwise_quantizer(value)

        value = self.project_conv(value, self.depthwise_quantizer.scale / 8.0)
        value = self.project_bn(value)
        return value


class Test4LocalFeedForwardBlock(nn.Module):
    """このblock専用のINT4 scaleで残差加算するローカルブロック。"""

    def __init__(self, channels: int, hidden_channels: int, rounding: str, activation_range_momentum: float) -> None:
        super().__init__()
        self.ffn          = Test4ConvolutionalFeedForward(channels, hidden_channels, rounding, activation_range_momentum)
        self.residual_add = QuantResidualAdd(bit_width=4, rounding=rounding, range_momentum=activation_range_momentum)

    @property
    def output_scale(self) -> torch.Tensor:
        """このblockの残差加算後に得られるINT4 scaleを返す。"""
        return self.residual_add.output_scale

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        identity    = value
        transformed = self.ffn(value, input_scale)
        value, _    = self.residual_add(transformed, self.ffn.output_scale, identity, input_scale)
        return value


class Test4SimpleAttention(nn.Module):
    """符号付きAttentionをFP4、ReLU6後の活性化をUFP4にしたSelf-Attention。"""

    def __init__(self, channels: int, num_heads: int, key_dim: int, value_dim: int, rounding: str, activation_range_momentum: float) -> None:
        super().__init__()
        if channels <= 0 or num_heads <= 0 or key_dim <= 0 or value_dim <= 0:
            raise ValueError("Attention dimensions must be positive.")

        self.num_heads       = num_heads
        self.key_dim         = key_dim
        self.value_dim       = value_dim
        self.attention_scale = key_dim ** -0.5

        query_key_channels = num_heads * key_dim
        value_channels     = num_heads * value_dim

        self.query_conv = QuantConv2d(channels, query_key_channels, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.query_bn   = nn.Identity()
        self.query_neq  = PerTokenFP4NEQ(rounding=rounding)

        self.key_conv = QuantConv2d(channels, query_key_channels, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.key_bn   = nn.Identity()
        self.key_neq  = PerTokenFP4NEQ(rounding=rounding)

        self.value_conv      = QuantConv2d(channels, value_channels, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.value_bn        = nn.Identity()
        self.value_quantizer = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)

        self.query_key_matmul       = FP4MatMul(rounding=rounding)
        self.attention_quantizer    = FP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        self.attention_value_matmul = FP4MatMul(rounding=rounding)

        self.project_input_relu      = nn.ReLU6(inplace=False)
        self.project_input_quantizer = UFP4Quantizer(rounding=rounding, range_momentum=activation_range_momentum)
        self.project_conv = QuantConv2d(value_channels, channels, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.project_bn   = nn.Identity()

    @property
    def output_scale(self) -> torch.Tensor:
        """残差加算へ渡すProjection ConvのINT32 accumulator scaleを返す。"""
        return self.project_conv.accumulator_scale

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = value.shape
        token_count                  = height * width

        query = self.query_conv(value, input_scale / 2.0)
        query = self.query_bn(query)
        query = query.reshape(batch_size, self.num_heads, self.key_dim, token_count)
        query = query.permute(0, 1, 3, 2)
        query, query_scale = self.query_neq(query)

        key = self.key_conv(value, input_scale / 2.0)
        key = self.key_bn(key)
        key = key.reshape(batch_size, self.num_heads, self.key_dim, token_count)
        key = key.permute(0, 1, 3, 2)
        key, key_scale = self.key_neq(key)
        key             = key.transpose(-2, -1)
        key_scale       = key_scale.transpose(-2, -1)

        attention, _ = self.query_key_matmul(query, query_scale, key, key_scale)
        attention    = attention * self.attention_scale
        attention    = self.attention_quantizer(attention)

        attention_value = self.value_conv(value, input_scale / 2.0)
        attention_value = self.value_bn(attention_value)
        attention_value = self.value_quantizer(attention_value)
        attention_value = attention_value.reshape(batch_size, self.num_heads, self.value_dim, token_count)
        attention_value = attention_value.permute(0, 1, 3, 2)

        output, _ = self.attention_value_matmul(attention, self.attention_quantizer.scale, attention_value, self.value_quantizer.scale)
        output    = output.permute(0, 1, 3, 2)
        output    = output.reshape(batch_size, self.num_heads * self.value_dim, height, width)
        output    = self.project_input_relu(output)
        output    = self.project_input_quantizer(output)
        output    = self.project_conv(output, self.project_input_quantizer.scale / 8.0)
        output    = self.project_bn(output)
        return output


class Test4AttentionFeedForwardBlock(nn.Module):
    """Attention残差とFFN残差が別々のINT4 scaleを持つブロック。"""

    def __init__(self, channels: int, hidden_channels: int, num_heads: int, key_dim: int, value_dim: int, rounding: str, activation_range_momentum: float) -> None:
        super().__init__()
        self.attention              = Test4SimpleAttention(channels, num_heads, key_dim, value_dim, rounding, activation_range_momentum)
        self.attention_residual_add = QuantResidualAdd(bit_width=4, rounding=rounding, range_momentum=activation_range_momentum)
        self.ffn                    = Test4ConvolutionalFeedForward(channels, hidden_channels, rounding, activation_range_momentum)
        self.ffn_residual_add       = QuantResidualAdd(bit_width=4, rounding=rounding, range_momentum=activation_range_momentum)

    @property
    def output_scale(self) -> torch.Tensor:
        """FFN残差加算後のINT4 scaleを返す。"""
        return self.ffn_residual_add.output_scale

    def forward(self, value: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        identity         = value
        attention_output = self.attention(value, input_scale)
        value, _         = self.attention_residual_add(attention_output, self.attention.output_scale, identity, input_scale)

        identity   = value
        ffn_output = self.ffn(value, self.attention_residual_add.output_scale)
        value, _   = self.ffn_residual_add(ffn_output, self.ffn.output_scale, identity, self.attention_residual_add.output_scale)
        return value


class BasicViTTest4(nn.Module):
    """UFP4内部活性化と6個の独立INT4残差scaleを持つアブレーションモデル。"""

    def __init__(self, num_classes: int = 10, input_bits: int = 8, residual_bits: int = 4, rounding: str = "ties_away_from_zero", activation_range_momentum: float = 0.95, image_size: int = 32, embed_dim: int = 48, mlp_ratio: int = 2, num_heads: int = 2, key_dim: int = 12, value_dim: int = 24) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("BasicViTTest4 is designed specifically for 32x32 images.")
        if input_bits != 8:
            raise ValueError("BasicViTTest4 requires input_bits=8.")
        if residual_bits != 4:
            raise ValueError("BasicViTTest4 requires residual_bits=4.")
        if embed_dim <= 0 or mlp_ratio <= 0:
            raise ValueError("embed_dim and mlp_ratio must be positive.")

        hidden_channels = embed_dim * mlp_ratio

        self.input_quantizer            = IntegerQuantizer(bit_width=8, signed=False, rounding=rounding, fixed_scale=1.0 / 255.0)
        self.stem                       = Test4BasicViTStem(embed_dim, rounding, activation_range_momentum)
        self.stem_output_quantizer      = IntegerQuantizer(bit_width=4, signed=True, rounding=rounding, range_momentum=activation_range_momentum)
        self.local_block1               = Test4LocalFeedForwardBlock(embed_dim, hidden_channels, rounding, activation_range_momentum)
        self.local_block2               = Test4LocalFeedForwardBlock(embed_dim, hidden_channels, rounding, activation_range_momentum)
        self.attention_block1           = Test4AttentionFeedForwardBlock(embed_dim, hidden_channels, num_heads, key_dim, value_dim, rounding, activation_range_momentum)
        self.attention_block2           = Test4AttentionFeedForwardBlock(embed_dim, hidden_channels, num_heads, key_dim, value_dim, rounding, activation_range_momentum)
        self.head_conv                  = QuantConv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0, bias=True, weight_bits=4, rounding=rounding, quantizer="fp4")
        self.head_bn                    = nn.Identity()
        self.classifier_input_quantizer = IntegerQuantizer(bit_width=8, signed=True, rounding=rounding, range_momentum=activation_range_momentum)
        self.pool                       = QuantizedAvgPool2d(kernel_size=8, stride=8, bit_width=8, signed=True, rounding=rounding)
        self.classifier                 = QuantLinear(embed_dim, num_classes, bias=True, weight_bits=8, rounding=rounding, quantizer="integer")

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_quantizer(value)
        value = self.stem(value, self.input_quantizer.scale)
        value = self.stem_output_quantizer(value)

        value = self.local_block1(value, self.stem_output_quantizer.scale)
        value = self.local_block2(value, self.local_block1.output_scale)
        value = self.attention_block1(value, self.local_block2.output_scale)
        value = self.attention_block2(value, self.attention_block1.output_scale)

        value             = self.head_conv(value, self.attention_block2.output_scale / 2.0)
        value             = self.head_bn(value)
        value             = self.classifier_input_quantizer(value)
        value, pool_scale = self.pool(value, self.classifier_input_quantizer.scale)
        value             = torch.flatten(value, start_dim=1)
        value             = self.classifier(value, pool_scale)
        return value
