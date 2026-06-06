from __future__ import annotations

from dataclasses import fields, replace
import os
from time import perf_counter

import cv2
import numpy as np

from qnegative.core.models import ColorCorrectionParams
from qnegative.core.roll_color_analysis_adapter import positive_linear_to_bgr16


ROLL_COLOR_ENGINE_LEGACY_BGR16 = "legacy_bgr16"
ROLL_COLOR_ENGINE_LINEAR_COMPAT = "linear_compat"
ROLL_COLOR_ENGINE = os.environ.get("NINA_ROLL_COLOR_ENGINE", ROLL_COLOR_ENGINE_LINEAR_COMPAT).strip().lower()
DISPLAY_GAMMA = np.float32(2.2)
DISPLAY_INV_GAMMA = np.float32(1.0 / 2.2)
ROLL_COLOR_LUMA_WEIGHTS_RGB = np.asarray((0.299, 0.587, 0.114), dtype=np.float32)


def bgr16_to_positive_linear(bgr16: np.ndarray) -> np.ndarray:
    # Convert the analyzer output back to NINA's linear RGB working space.
    rgb = bgr16[:, :, :3][:, :, ::-1].astype(np.float32, copy=False) / 65535.0
    return np.power(np.clip(rgb, 0.0, 1.0), 2.2).astype(np.float32, copy=False)


def apply_roll_color_to_linear_rgb(
    linear_rgb: np.ndarray,
    *,
    roll_result: dict | None,
    frame_plan: dict | None,
    settings: ColorCorrectionParams,
    stage_timings: dict[str, float] | None = None,
    engine: str | None = None,
) -> np.ndarray:
    if not settings.enabled or not roll_result or not frame_plan:
        return linear_rgb

    stage_start = perf_counter()
    core = _roll_color_core()
    roll = _dataclass_from_payload(core.RollAnalysisResult, roll_result)
    frame = _dataclass_from_payload(core.FrameAnalysis, frame_plan)
    if stage_timings is not None:
        stage_timings["Lab roll color setup"] = perf_counter() - stage_start

    selected_engine = (engine or ROLL_COLOR_ENGINE or ROLL_COLOR_ENGINE_LINEAR_COMPAT).strip().lower()
    if selected_engine == ROLL_COLOR_ENGINE_LEGACY_BGR16:
        return _apply_roll_color_legacy_bgr16(
            linear_rgb,
            core=core,
            roll=roll,
            frame=frame,
            settings=settings,
            stage_timings=stage_timings,
        )
    return _apply_roll_color_linear_compat(
        linear_rgb,
        core=core,
        roll=roll,
        frame=frame,
        settings=settings,
        stage_timings=stage_timings,
    )


def _apply_roll_color_legacy_bgr16(
    linear_rgb: np.ndarray,
    *,
    core,
    roll,
    frame,
    settings: ColorCorrectionParams,
    stage_timings: dict[str, float] | None,
) -> np.ndarray:
    protection_scale = float(np.clip(settings.protection_strength / 100.0, 0.0, 1.0))
    if protection_scale < 0.999:
        frame = replace(
            frame,
            highlight_protection_strength=(
                float(getattr(frame, "highlight_protection_strength", 0.0)) * protection_scale
            ),
        )
    stage_start = perf_counter()
    bgr = positive_linear_to_bgr16(linear_rgb)
    if stage_timings is not None:
        stage_timings["Lab roll color to BGR16"] = perf_counter() - stage_start

    stage_start = perf_counter()
    corrected = core.apply_roll_plan_to_bgr(
        bgr,
        roll,
        frame,
        roll_strength=float(np.clip(settings.roll_strength / 100.0, 0.0, 1.25)),
        frame_strength=float(np.clip(settings.frame_residual_strength / 100.0, 0.0, 1.0)),
        tone_strength=float(np.clip(settings.tone_balance_strength / 100.0, 0.0, 1.0)),
        exposure_strength=float(np.clip(settings.exposure_match_strength / 100.0, 0.0, 1.0)),
    )
    if stage_timings is not None:
        stage_timings["Lab roll color plan"] = perf_counter() - stage_start

    stage_start = perf_counter()
    output = bgr16_to_positive_linear(corrected)
    if stage_timings is not None:
        stage_timings["Lab roll color from BGR16"] = perf_counter() - stage_start
    return output


def _apply_roll_color_linear_compat(
    linear_rgb: np.ndarray,
    *,
    core,
    roll,
    frame,
    settings: ColorCorrectionParams,
    stage_timings: dict[str, float] | None,
) -> np.ndarray:
    stage_start = perf_counter()
    gains = core.combined_frame_gains(
        roll,
        frame,
        roll_strength=float(np.clip(settings.roll_strength / 100.0, 0.0, 1.25)),
        frame_strength=float(np.clip(settings.frame_residual_strength / 100.0, 0.0, 1.0)),
    )
    tone_strength = float(np.clip(settings.tone_balance_strength / 100.0, 0.0, 1.0))
    clipped = np.clip(linear_rgb, 0.0, 1.0).astype(np.float32, copy=False)
    if frame.color_action in {"none", "protected", "review", "skip-extreme"}:
        corrected = _apply_rgb_gains_to_linear_rgb(clipped, core, gains)
    else:
        corrected = _apply_tone_aware_rgb_gains_to_linear_rgb(
            clipped,
            core,
            gains,
            frame.tone_shadow_rgb_gains,
            frame.tone_mid_rgb_gains,
            frame.tone_highlight_rgb_gains,
            tone_strength=tone_strength,
        )
    if stage_timings is not None:
        stage_timings["Lab roll color linear gains"] = perf_counter() - stage_start

    strength = float(getattr(frame, "highlight_protection_strength", 0.0)) * float(
        np.clip(settings.protection_strength / 100.0, 0.0, 1.0)
    )
    if strength > 0.001:
        stage_start = perf_counter()
        corrected = _apply_highlight_protection_to_linear_rgb(clipped, corrected, core, strength)
        if stage_timings is not None:
            stage_timings["Lab roll color linear highlight protect"] = perf_counter() - stage_start

    stage_start = perf_counter()
    corrected = _apply_exposure_to_linear_rgb(
        corrected,
        frame.exposure_delta_stops,
        strength=float(np.clip(settings.exposure_match_strength / 100.0, 0.0, 1.0)),
    )
    if stage_timings is not None:
        stage_timings["Lab roll color linear exposure"] = perf_counter() - stage_start
    return corrected


def _apply_rgb_gains_to_linear_rgb(
    linear_rgb: np.ndarray,
    core,
    rgb_gains: tuple[float, float, float],
) -> np.ndarray:
    gains = np.asarray(core.normalize_rgb_gains(rgb_gains), dtype=np.float32)
    linear_gains = np.power(np.clip(gains, 1e-4, 1e4), DISPLAY_GAMMA).astype(np.float32, copy=False)
    return np.clip(linear_rgb * linear_gains[None, None, :], 0.0, 1.0).astype(np.float32, copy=False)


def _apply_tone_aware_rgb_gains_to_linear_rgb(
    linear_rgb: np.ndarray,
    core,
    base_rgb_gains: tuple[float, float, float],
    shadow_rgb_gains: tuple[float, float, float],
    mid_rgb_gains: tuple[float, float, float],
    highlight_rgb_gains: tuple[float, float, float],
    *,
    tone_strength: float,
) -> np.ndarray:
    tone_strength = float(np.clip(tone_strength, 0.0, 1.0))
    if tone_strength <= 0.001:
        return _apply_rgb_gains_to_linear_rgb(linear_rgb, core, base_rgb_gains)

    base_log = core._safe_log_gain(core.normalize_rgb_gains(base_rgb_gains))
    shadow_log = core._safe_log_gain(core.normalize_rgb_gains(shadow_rgb_gains))
    mid_log = core._safe_log_gain(core.normalize_rgb_gains(mid_rgb_gains))
    highlight_log = core._safe_log_gain(core.normalize_rgb_gains(highlight_rgb_gains))
    if max(float(np.linalg.norm(shadow_log)), float(np.linalg.norm(mid_log)), float(np.linalg.norm(highlight_log))) <= 0.001:
        return _apply_rgb_gains_to_linear_rgb(linear_rgb, core, base_rgb_gains)

    display_r = np.power(linear_rgb[:, :, 0], DISPLAY_INV_GAMMA)
    display_g = np.power(linear_rgb[:, :, 1], DISPLAY_INV_GAMMA)
    display_b = np.power(linear_rgb[:, :, 2], DISPLAY_INV_GAMMA)
    luma = (
        display_r * ROLL_COLOR_LUMA_WEIGHTS_RGB[0]
        + display_g * ROLL_COLOR_LUMA_WEIGHTS_RGB[1]
        + display_b * ROLL_COLOR_LUMA_WEIGHTS_RGB[2]
    )
    shadow_w = 1.0 - _smoothstep(0.14, 0.42, luma)
    mid_w = _smoothstep(0.18, 0.42, luma) * (1.0 - _smoothstep(0.62, 0.86, luma))
    highlight_w = _smoothstep(0.58, 0.88, luma)
    residual_logs = np.asarray([shadow_log, mid_log, highlight_log], dtype=np.float32) * np.float32(tone_strength)

    output = linear_rgb.copy()
    for channel in range(3):
        gamma_log_gain = (
            np.float32(base_log[channel])
            + shadow_w * residual_logs[0, channel]
            + mid_w * residual_logs[1, channel]
            + highlight_w * residual_logs[2, channel]
        )
        linear_gain = np.exp(gamma_log_gain * DISPLAY_GAMMA).astype(np.float32, copy=False)
        output[:, :, channel] = output[:, :, channel] * linear_gain
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)


def _apply_exposure_to_linear_rgb(linear_rgb: np.ndarray, delta_stops: float, *, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    delta = float(np.clip(float(delta_stops) * strength, -1.2, 1.2))
    if abs(delta) < 0.001:
        return linear_rgb
    # The legacy module applied exposure in gamma-encoded space. Raising the
    # gain by display gamma keeps this compatibility path visually close.
    factor = np.float32(2.0 ** (delta * float(DISPLAY_GAMMA)))
    return np.clip(linear_rgb * factor, 0.0, 1.0).astype(np.float32, copy=False)


def _apply_highlight_protection_to_linear_rgb(
    original_linear: np.ndarray,
    corrected_linear: np.ndarray,
    core,
    strength: float,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.001 or original_linear.shape[:2] != corrected_linear.shape[:2]:
        return corrected_linear
    before_rgb = np.power(np.clip(original_linear, 0.0, 1.0), DISPLAY_INV_GAMMA).astype(np.float32, copy=False)
    before_flat = before_rgb.reshape((-1, 3))
    if before_flat.shape[0] < 128:
        return corrected_linear
    before_luma = before_flat @ ROLL_COLOR_LUMA_WEIGHTS_RGB
    combined_mask = np.zeros(before_flat.shape[0], dtype=bool)
    for _region, mask in core._sky_cloud_region_masks(before_rgb, before_flat, before_luma):
        if int(np.count_nonzero(mask)) >= 96:
            combined_mask |= mask
    if not bool(np.any(combined_mask)):
        return corrected_linear

    height, width = original_linear.shape[:2]
    mask_image = combined_mask.reshape((height, width)).astype(np.float32)
    sigma = max(1.2, min(6.0, min(height, width) / 180.0))
    softened_mask = cv2.GaussianBlur(mask_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask_image = np.maximum(mask_image, softened_mask)
    mask_image = np.clip(mask_image * np.float32(strength), 0.0, 1.0)[:, :, None]

    corrected_rgb = np.power(np.clip(corrected_linear, 0.0, 1.0), DISPLAY_INV_GAMMA).astype(np.float32, copy=False)
    output_rgb = corrected_rgb * (1.0 - mask_image) + before_rgb * mask_image
    return np.power(np.clip(output_rgb, 0.0, 1.0), DISPLAY_GAMMA).astype(np.float32, copy=False)


def _smoothstep(edge0: float, edge1: float, value):
    value = np.asarray(value, dtype=np.float32)
    if abs(edge1 - edge0) <= 1e-6:
        return np.where(value >= edge1, 1.0, 0.0).astype(np.float32)
    x = np.clip((value - np.float32(edge0)) / np.float32(edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def roll_color_frame_key(frame_plan: dict | None) -> tuple | None:
    if not frame_plan:
        return None
    keys = (
        "path",
        "color_action",
        "safe_rgb_gains",
        "tone_shadow_rgb_gains",
        "tone_mid_rgb_gains",
        "tone_highlight_rgb_gains",
        "tone_confidence",
        "highlight_protection_strength",
        "highlight_protected_region",
        "highlight_protection_warning",
        "exposure_delta_stops",
    )
    values: list[object] = []
    for key in keys:
        value = frame_plan.get(key)
        if isinstance(value, list):
            values.append(tuple(round(float(item), 7) for item in value))
        elif isinstance(value, float):
            values.append(round(value, 7))
        else:
            values.append(value)
    return tuple(values)


def _roll_color_core():
    from qnegative.core import roll_color_analysis

    return roll_color_analysis


def _dataclass_from_payload(cls, payload: dict):
    allowed = {field.name for field in fields(cls)}
    values = {key: payload[key] for key in payload if key in allowed}
    if "frames" in values:
        values["frames"] = ()
    if "candidates" in values:
        values["candidates"] = ()
    return cls(**values)
