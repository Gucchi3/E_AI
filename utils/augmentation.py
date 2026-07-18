"""バッチ単位のMixUpとCutMix。"""

from __future__ import annotations

import math
import random

import torch


class BatchMixupCutmix:
    """学習バッチへMixUpまたはCutMixを適用する。"""

    def __init__(self, num_classes: int, mixup_alpha: float, cutmix_alpha: float, probability: float, switch_probability: float) -> None:
        self.num_classes        = num_classes
        self.mixup_alpha        = mixup_alpha
        self.cutmix_alpha       = cutmix_alpha
        self.probability        = probability
        self.switch_probability = switch_probability


    @property
    def enabled(self) -> bool:
        """いずれかのデータ拡張を使用するか返す。"""
        return self.probability > 0.0 and (self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0)


    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """画像、損失用ラベル、accuracy用ラベルを返す。"""
        if not self.enabled or images.size(0) < 2 or random.random() >= self.probability:
            return images, targets, targets

        permutation = torch.randperm(images.size(0), device=images.device)
        if self._use_cutmix():
            mixed_images, ratio = self._cutmix(images, permutation)
        else:
            mixed_images, ratio = self._mixup(images, permutation)
        mixed_targets = self._mix_targets(targets, permutation, ratio)
        return mixed_images, mixed_targets, mixed_targets


    def _use_cutmix(self) -> bool:
        """今回のバッチへCutMixを適用するか返す。"""
        if self.cutmix_alpha <= 0.0:
            return False
        if self.mixup_alpha <= 0.0:
            return True
        return random.random() < self.switch_probability


    def _mixup(self, images: torch.Tensor, permutation: torch.Tensor) -> tuple[torch.Tensor, float]:
        """画像全体を混合する。"""
        ratio        = self._sample_beta(self.mixup_alpha)
        mixed_images = images * ratio + images[permutation] * (1.0 - ratio)
        return mixed_images, ratio


    def _cutmix(self, images: torch.Tensor, permutation: torch.Tensor) -> tuple[torch.Tensor, float]:
        """矩形領域を別画像の領域へ置き換える。"""
        sampled_ratio       = self._sample_beta(self.cutmix_alpha)
        image_height        = images.size(-2)
        image_width         = images.size(-1)
        cut_ratio           = math.sqrt(1.0 - sampled_ratio)
        cut_height          = round(image_height * cut_ratio)
        cut_width           = round(image_width * cut_ratio)
        center_y            = random.randrange(image_height)
        center_x            = random.randrange(image_width)
        top                 = max(center_y - cut_height // 2, 0)
        left                = max(center_x - cut_width // 2, 0)
        bottom              = min(top + cut_height, image_height)
        right               = min(left + cut_width, image_width)
        mixed_images        = images.clone()
        mixed_images[:, :, top:bottom, left:right] = images[permutation, :, top:bottom, left:right]
        replaced_area       = (bottom - top) * (right - left)
        area_adjusted_ratio = 1.0 - replaced_area / (image_height * image_width)
        return mixed_images, area_adjusted_ratio


    def _mix_targets(self, targets: torch.Tensor, permutation: torch.Tensor, ratio: float) -> torch.Tensor:
        """クラス番号を混合比率付きの確率分布へ変換する。"""
        one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).to(dtype=torch.float32)
        return one_hot * ratio + one_hot[permutation] * (1.0 - ratio)


    @staticmethod
    def _sample_beta(alpha: float) -> float:
        """Beta分布から混合比率を取り出す。"""
        return random.betavariate(alpha, alpha)
