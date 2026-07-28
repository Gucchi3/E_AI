"""CIFAR-10向けの小型CNN + Vision Transformer。"""

from __future__ import annotations

import math

import torch
from torch import nn


class BasicViTStem(nn.Module):
    """32x32画像を48x8x8の特徴マップへ変換するCNN Stem。"""

    def __init__(self, embed_dim: int = 48) -> None:
        super().__init__()
        middle_channels = embed_dim // 2

        self.conv1 = nn.Conv2d(3, middle_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu1 = nn.ReLU6(inplace=False)

        self.conv2_depthwise = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride=2, padding=1, groups=middle_channels, bias=False)
        self.bn2_depthwise = nn.BatchNorm2d(middle_channels)
        self.relu2_depthwise = nn.ReLU6(inplace=False)

        self.conv2_pointwise = nn.Conv2d(middle_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_pointwise = nn.BatchNorm2d(embed_dim)
        self.relu2_pointwise = nn.ReLU6(inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value)
        value = self.bn1(value)
        value = self.relu1(value)

        value = self.conv2_depthwise(value)
        value = self.bn2_depthwise(value)
        value = self.relu2_depthwise(value)

        value = self.conv2_pointwise(value)
        value = self.bn2_pointwise(value)
        value = self.relu2_pointwise(value)
        return value


class ConvolutionalFeedForward(nn.Module):
    """1x1 Conv、Depthwise Conv、1x1 Convで構成するFFN。"""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.expand_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_channels)
        self.expand_relu = nn.ReLU6(inplace=False)

        self.depthwise_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_channels)
        self.depthwise_relu = nn.ReLU6(inplace=False)

        self.project_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.expand_conv(value)
        value = self.expand_bn(value)
        value = self.expand_relu(value)

        value = self.depthwise_conv(value)
        value = self.depthwise_bn(value)
        value = self.depthwise_relu(value)

        value = self.project_conv(value)
        value = self.project_bn(value)
        return value


class LocalFeedForwardBlock(nn.Module):
    """畳み込みFFNと残差接続からなるLocal block。"""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.ffn = ConvolutionalFeedForward(channels, hidden_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        transformed = self.ffn(value)
        return transformed + identity


class SimpleAttention(nn.Module):
    """Softmaxを使わず、QとKをL1正規化するSelf-Attention。"""

    def __init__(self, channels: int, num_heads: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        if channels <= 0 or num_heads <= 0 or key_dim <= 0 or value_dim <= 0:
            raise ValueError("Attention dimensions must be positive.")

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.attention_scale = key_dim ** -0.5

        query_key_channels = num_heads * key_dim
        value_channels = num_heads * value_dim

        self.query_conv = nn.Conv2d(channels, query_key_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.query_bn = nn.BatchNorm2d(query_key_channels)

        self.key_conv = nn.Conv2d(channels, query_key_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_bn = nn.BatchNorm2d(query_key_channels)

        self.value_conv = nn.Conv2d(channels, value_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)

        self.project_input_relu = nn.ReLU6(inplace=False)
        self.project_conv = nn.Conv2d(value_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = value.shape
        token_count = height * width

        query = self.query_conv(value)
        query = self.query_bn(query)
        query = query.reshape(batch_size, self.num_heads, self.key_dim, token_count)
        query = query.permute(0, 1, 3, 2)
        query_l1_norm = query.abs().sum(dim=-1, keepdim=True).clamp_min(torch.finfo(query.dtype).eps)
        query = query / query_l1_norm

        key = self.key_conv(value)
        key = self.key_bn(key)
        key = key.reshape(batch_size, self.num_heads, self.key_dim, token_count)
        key = key.permute(0, 1, 3, 2)
        key_l1_norm = key.abs().sum(dim=-1, keepdim=True).clamp_min(torch.finfo(key.dtype).eps)
        key = key / key_l1_norm
        key = key.transpose(-2, -1)

        attention = torch.matmul(query, key)
        attention = attention * self.attention_scale

        attention_value = self.value_conv(value)
        attention_value = self.value_bn(attention_value)
        attention_value = attention_value.reshape(batch_size, self.num_heads, self.value_dim, token_count)
        attention_value = attention_value.permute(0, 1, 3, 2)

        output = torch.matmul(attention, attention_value)
        output = output.permute(0, 1, 3, 2)
        output = output.reshape(batch_size, self.num_heads * self.value_dim, height, width)
        output = self.project_input_relu(output)
        output = self.project_conv(output)
        output = self.project_bn(output)
        return output


class AttentionFeedForwardBlock(nn.Module):
    """Simple Attentionと畳み込みFFNを順番に実行するblock。"""

    def __init__(self, channels: int, hidden_channels: int, num_heads: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.attention = SimpleAttention(channels, num_heads, key_dim, value_dim)
        self.ffn = ConvolutionalFeedForward(channels, hidden_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        attention_output = self.attention(value)
        value = attention_output + identity

        identity = value
        ffn_output = self.ffn(value)
        value = ffn_output + identity
        return value


class BasicViTFP32(nn.Module):
    """4個の主要blockを持つ、32x32画像専用の小型BasicViT。"""

    def __init__(self, num_classes: int = 10, image_size: int = 32, embed_dim: int = 48, mlp_ratio: int = 2, num_heads: int = 2, key_dim: int = 12, value_dim: int = 24) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError("BasicViTFP32 is designed specifically for 32x32 images.")
        if embed_dim <= 0 or mlp_ratio <= 0:
            raise ValueError("embed_dim and mlp_ratio must be positive.")

        hidden_channels = embed_dim * mlp_ratio

        self.stem = BasicViTStem(embed_dim)
        self.local_block1 = LocalFeedForwardBlock(embed_dim, hidden_channels)
        self.local_block2 = LocalFeedForwardBlock(embed_dim, hidden_channels)
        self.attention_block1 = AttentionFeedForwardBlock(embed_dim, hidden_channels, num_heads, key_dim, value_dim)
        self.attention_block2 = AttentionFeedForwardBlock(embed_dim, hidden_channels, num_heads, key_dim, value_dim)

        self.head_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.head_bn = nn.BatchNorm2d(embed_dim)
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        value = self.local_block1(value)
        value = self.local_block2(value)
        value = self.attention_block1(value)
        value = self.attention_block2(value)

        value = self.head_conv(value)
        value = self.head_bn(value)
        value = self.pool(value)
        value = torch.flatten(value, start_dim=1)
        value = self.classifier(value)
        return value
