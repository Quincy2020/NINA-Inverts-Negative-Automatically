from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import tempfile
from typing import Iterable

import cv2
import numpy as np

from qnegative.core.models import ColorCorrectionParams


def positive_linear_to_bgr16(linear_rgb: np.ndarray) -> np.ndarray:
    # The roll analyzer was written for scan-like positive images, so feed it
    # gamma-encoded BGR proxies while keeping NINA's pipeline linear outside.
    clipped = np.clip(linear_rgb, 0.0, 1.0).astype(np.float32, copy=False)
    srgb = np.power(clipped, 1.0 / 2.2)
    rgb16 = np.clip(srgb * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)
    return np.ascontiguousarray(rgb16[:, :, ::-1])


def bgr16_to_positive_linear(bgr16: np.ndarray) -> np.ndarray:
    # Convert the analyzer output back to NINA's linear RGB working space.
    rgb = bgr16[:, :, :3][:, :, ::-1].astype(np.float32, copy=False) / 65535.0
    return np.power(np.clip(rgb, 0.0, 1.0), 2.2).astype(np.float32, copy=False)


def analyze_positive_bgr_roll(
    images: Iterable[tuple[Path, np.ndarray]],
    *,
    crop_percent: float = 4.0,
    analysis_max_size: tuple[int, int] = (768, 768),
) -> tuple[dict, dict[str, dict]]:
    core = _roll_color_core()
    image_items = list(images)
    if not image_items:
        raise ValueError("No positive images are available for roll color analysis.")

    with tempfile.TemporaryDirectory(prefix="nina_roll_color_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_to_original: dict[str, Path] = {}
        temp_paths: list[Path] = []
        for index, (source_path, bgr) in enumerate(image_items):
            temp_path = temp_root / f"{index:04d}_{source_path.stem}.png"
            ok = cv2.imwrite(str(temp_path), np.ascontiguousarray(bgr))
            if not ok:
                raise OSError(f"Could not write roll color proxy: {temp_path}")
            temp_to_original[str(temp_path)] = source_path
            temp_paths.append(temp_path)

        result = core.analyze_roll(
            temp_paths,
            crop_percent=crop_percent,
            analysis_max_size=analysis_max_size,
        )

    payload = result.to_dict()
    frames_by_path: dict[str, dict] = {}
    normalized_frames = []
    for frame_payload in payload.get("frames", []):
        original_path = temp_to_original.get(str(Path(frame_payload.get("path", ""))))
        if original_path is None:
            continue
        normalized = dict(frame_payload)
        normalized["path"] = str(original_path)
        normalized["filename"] = original_path.name
        normalized_frames.append(normalized)
        frames_by_path[str(original_path)] = normalized
    payload["frames"] = normalized_frames
    return payload, frames_by_path


def apply_roll_color_to_linear_rgb(
    linear_rgb: np.ndarray,
    *,
    roll_result: dict | None,
    frame_plan: dict | None,
    settings: ColorCorrectionParams,
) -> np.ndarray:
    if not settings.enabled or not roll_result or not frame_plan:
        return linear_rgb

    core = _roll_color_core()
    roll = _dataclass_from_payload(core.RollAnalysisResult, roll_result)
    frame = _dataclass_from_payload(core.FrameAnalysis, frame_plan)
    bgr = positive_linear_to_bgr16(linear_rgb)
    corrected = core.apply_roll_plan_to_bgr(
        bgr,
        roll,
        frame,
        roll_strength=float(np.clip(settings.roll_strength / 100.0, 0.0, 1.25)),
        frame_strength=float(np.clip(settings.frame_residual_strength / 100.0, 0.0, 1.0)),
        tone_strength=float(np.clip(settings.tone_balance_strength / 100.0, 0.0, 1.0)),
        exposure_strength=float(np.clip(settings.exposure_match_strength / 100.0, 0.0, 1.0)),
    )
    return bgr16_to_positive_linear(corrected)


def roll_color_result_summary(payload: dict | None) -> str:
    if not payload:
        return "Not analyzed"
    analyzed = int(payload.get("analyzed_count", 0))
    used = int(payload.get("used_count", 0))
    confidence = float(payload.get("confidence", 0.0))
    warning = str(payload.get("warning") or "")
    suffix = f", {warning}" if warning else ""
    return f"Analyzed {analyzed}, used {used}, confidence {confidence:.2f}{suffix}"


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
