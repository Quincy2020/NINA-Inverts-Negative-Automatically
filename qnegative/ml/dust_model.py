"""Lightweight U-Net dust-mask model.

This follows the Spotless-style U-Net flow but keeps the network smaller and
returns logits so training can use numerically stable BCEWithLogits loss.
"""

from __future__ import annotations

from collections.abc import Sequence


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    return torch, nn, functional


class DustUNet:
    """Factory wrapper to avoid importing torch when the core app starts."""

    @staticmethod
    def create(in_channels: int = 1, base_channels: int = 32):
        torch, nn, functional = _torch()

        class ConvBlock(nn.Module):
            def __init__(self, in_c: int, out_c: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        class UNetImpl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                c1 = base_channels
                c2 = c1 * 2
                c3 = c2 * 2
                c4 = c3 * 2
                c5 = c4 * 2
                self.enc1 = ConvBlock(in_channels, c1)
                self.enc2 = ConvBlock(c1, c2)
                self.enc3 = ConvBlock(c2, c3)
                self.enc4 = ConvBlock(c3, c4)
                self.pool = nn.MaxPool2d(2)
                self.middle = ConvBlock(c4, c5)
                self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
                self.dec4 = ConvBlock(c4 + c4, c4)
                self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
                self.dec3 = ConvBlock(c3 + c3, c3)
                self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
                self.dec2 = ConvBlock(c2 + c2, c2)
                self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
                self.dec1 = ConvBlock(c1 + c1, c1)
                self.out = nn.Conv2d(c1, 1, 1)

            @staticmethod
            def _match(skip, x):
                if skip.shape[-2:] == x.shape[-2:]:
                    return skip
                return functional.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                m = self.middle(self.pool(e4))
                d4 = self.up4(m)
                d4 = self.dec4(torch.cat([self._match(e4, d4), d4], dim=1))
                d3 = self.up3(d4)
                d3 = self.dec3(torch.cat([self._match(e3, d3), d3], dim=1))
                d2 = self.up2(d3)
                d2 = self.dec2(torch.cat([self._match(e2, d2), d2], dim=1))
                d1 = self.up1(d2)
                d1 = self.dec1(torch.cat([self._match(e1, d1), d1], dim=1))
                return self.out(d1)

        return UNetImpl()


def dice_loss_from_logits(logits, target, eps: float = 1e-6):
    torch, _, _ = _torch()
    prob = torch.sigmoid(logits)
    dims: Sequence[int] = (1, 2, 3)
    intersection = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits,
    target,
    dice_weight: float = 0.85,
    pos_weight_scale: float = 0.5,
    max_pos_weight: float = 80.0,
    sample_types=None,
    hard_negative_topk_weight: float = 0.0,
    hard_negative_topk_fraction: float = 0.01,
):
    torch, nn, _ = _torch()
    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = torch.clamp(
        (negative / torch.clamp(positive, min=1.0)) * pos_weight_scale,
        min=1.0,
        max=max_pos_weight,
    )
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = dice_loss_from_logits(logits, target)
    loss = bce + dice_weight * dice
    if sample_types is not None and hard_negative_topk_weight > 0.0:
        hard_mask = sample_types.to(logits.device) > 0
        if torch.any(hard_mask):
            prob = torch.sigmoid(logits[hard_mask])
            flat = prob.flatten(start_dim=1)
            k = max(1, int(flat.shape[1] * hard_negative_topk_fraction))
            top_values = torch.topk(flat, k=k, dim=1).values
            loss = loss + hard_negative_topk_weight * top_values.mean()
    return loss
