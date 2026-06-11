from __future__ import annotations

import cv2
import numpy as np

from qnegative.core.models import DetailParams


def apply_detail_to_linear_rgb(linear_rgb: np.ndarray, params: DetailParams) -> np.ndarray:
    if not params.texture_enabled and not params.usm_enabled:
        return linear_rgb
    srgb = _linear_to_srgb(linear_rgb)
    srgb8 = np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(srgb8, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0]

    if params.texture_enabled:
        y = _apply_texture_preserve_to_luminance(
            y,
            amount=params.texture_amount,
            radius=params.texture_radius,
            shadow_protect=params.texture_shadow_protect,
            highlight_protect=params.texture_highlight_protect,
        )
    if params.usm_enabled:
        if params.usm_luminance_only:
            y = _apply_usm_to_luminance(
                y,
                amount=params.usm_amount,
                radius=params.usm_radius,
                threshold=params.usm_threshold,
            )
            ycrcb[..., 0] = y
            srgb8 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            if params.texture_enabled:
                ycrcb[..., 0] = y
                srgb8 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            srgb8 = np.dstack(
                [
                    _apply_usm_to_luminance(
                        srgb8[..., channel],
                        amount=params.usm_amount,
                        radius=params.usm_radius,
                        threshold=params.usm_threshold,
                    )
                    for channel in range(3)
                ]
            )
    else:
        ycrcb[..., 0] = y
        srgb8 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    return _srgb_to_linear(srgb8.astype(np.float32) / 255.0).astype(np.float32, copy=False)


def _apply_texture_preserve_to_luminance(
    channel: np.ndarray,
    *,
    amount: int,
    radius: float,
    shadow_protect: int,
    highlight_protect: int,
) -> np.ndarray:
    amount = int(np.clip(amount, 0, 100))
    if amount <= 0:
        return channel
    base_radius = float(np.clip(radius, 0.30, 2.50))
    source = channel.astype(np.float32)
    small_blur = cv2.GaussianBlur(source, (0, 0), sigmaX=base_radius, sigmaY=base_radius)
    medium_radius = max(0.45, base_radius * 2.35)
    medium_blur = cv2.GaussianBlur(source, (0, 0), sigmaX=medium_radius, sigmaY=medium_radius)

    fine_detail = source - small_blur
    medium_detail = small_blur - medium_blur
    texture_strength = np.abs(fine_detail) + np.abs(medium_detail) * 0.75

    fine_shaped = _shape_texture_detail(fine_detail, threshold=1.5, limit=10.0)
    medium_shaped = _shape_texture_detail(medium_detail, threshold=1.0, limit=16.0)

    luminance = source / 255.0
    amount_scale = float(np.clip(amount / 100.0, 0.0, 1.0))
    shadow_scale = float(np.clip(shadow_protect / 100.0, 0.0, 1.0))
    highlight_scale = float(np.clip(highlight_protect / 100.0, 0.0, 1.0))
    shadow_mask = 1.0 - shadow_scale * np.power(1.0 - luminance, 2.2) * 0.92
    highlight_mask = 1.0 - highlight_scale * np.power(luminance, 2.0) * 0.72
    texture_mask = np.clip((texture_strength - 1.5) / 9.5, 0.0, 1.0)
    blend_mask = shadow_mask * highlight_mask * texture_mask

    enhanced = source + (fine_shaped * (0.16 * amount_scale) + medium_shaped * (0.28 * amount_scale)) * blend_mask
    enhanced = np.where(texture_strength < 2.0, source, enhanced)
    return np.clip(enhanced + 0.5, 0, 255).astype(np.uint8)


def _shape_texture_detail(value: np.ndarray, *, threshold: float, limit: float) -> np.ndarray:
    magnitude = np.abs(value)
    soft = np.maximum(magnitude - threshold, 0.0)
    compression = 1.0 / (1.0 + soft / max(0.1, limit))
    shaped = soft * compression
    return np.where(value >= 0.0, shaped, -shaped).astype(np.float32, copy=False)


def _apply_usm_to_luminance(
    channel: np.ndarray,
    *,
    amount: int,
    radius: float,
    threshold: int,
) -> np.ndarray:
    amount = max(0, int(amount))
    radius = max(0.0, float(radius))
    threshold = max(0, int(threshold))
    if amount <= 0 or radius <= 0.0:
        return channel

    source = channel.astype(np.float32)
    blurred = cv2.GaussianBlur(source, (0, 0), sigmaX=max(0.1, radius), sigmaY=max(0.1, radius))
    difference = source - blurred
    amount_scale = float(amount) / 90.0
    enhanced = source + difference * amount_scale
    output = np.where(np.abs(difference) < float(threshold), source, enhanced)
    return np.clip(output + 0.5, 0, 255).astype(np.uint8)


def _linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    return np.power(np.clip(linear_rgb, 0.0, 1.0), 1.0 / 2.2).astype(np.float32, copy=False)


def _srgb_to_linear(srgb_rgb: np.ndarray) -> np.ndarray:
    return np.power(np.clip(srgb_rgb, 0.0, 1.0), 2.2).astype(np.float32, copy=False)
