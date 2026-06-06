from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qnegative.core.models import ImageSize
from qnegative.core.models import LensCorrectionParams
from qnegative.core.raw_loader import load_raw_rgb16


LENS_PROFILE_SCHEMA_VERSION = 1


def default_lens_profile_dir() -> Path:
    return Path.cwd() / "presets" / "lens_profiles"


def lens_params_to_dict(params: LensCorrectionParams) -> dict[str, Any]:
    return asdict(params)


def lens_params_from_dict(payload: dict[str, Any]) -> LensCorrectionParams:
    defaults = LensCorrectionParams()
    enabled = bool(payload.get("enabled", defaults.enabled))
    return LensCorrectionParams(
        enabled=enabled,
        mode=str(payload.get("mode") or ("radial" if enabled else defaults.mode)),
        strength=int(payload.get("strength", defaults.strength)),
        radius=int(payload.get("radius", defaults.radius)),
        center_x=int(payload.get("center_x", defaults.center_x)),
        center_y=int(payload.get("center_y", defaults.center_y)),
        smoothness=int(payload.get("smoothness", defaults.smoothness)),
        max_gain=int(payload.get("max_gain", defaults.max_gain)),
        flat_profile_path=payload.get("flat_profile_path", defaults.flat_profile_path),
        flat_strength=int(payload.get("flat_strength", defaults.flat_strength)),
    )


def save_radial_lens_profile(path: Path, name: str, params: LensCorrectionParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LENS_PROFILE_SCHEMA_VERSION,
        "type": "radial",
        "name": name,
        "params": lens_params_to_dict(params),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_lens_profile(path: Path) -> LensCorrectionParams:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile_type = payload.get("type", "radial")
    if profile_type == "flat_frame":
        _resolve_flat_map_path(path, payload)
        max_gain = float(payload.get("max_gain", 2.0))
        return LensCorrectionParams(
            enabled=True,
            mode="flat_frame",
            max_gain=int(round(max_gain * 100.0)),
            flat_profile_path=str(path),
            flat_strength=100,
        )
    if profile_type != "radial":
        raise ValueError(f"Unsupported lens profile type: {profile_type}")
    params_payload = payload.get("params")
    if not isinstance(params_payload, dict):
        raise ValueError("Lens profile is missing radial parameters.")
    return lens_params_from_dict(params_payload)


def load_radial_lens_profile(path: Path) -> LensCorrectionParams:
    return load_lens_profile(path)


def create_flat_frame_profile(
    source_raw_path: Path,
    output_profile_path: Path,
    *,
    name: str | None = None,
    map_long_edge: int = 512,
    blur_radius: int = 41,
    max_gain: float = 2.0,
    map_mode: str = "linked_luminance",
) -> LensCorrectionParams:
    raw_image = load_raw_rgb16(source_raw_path, half_size=True, include_display_transform=False)
    flat_linear = raw_image.as_float32()
    gain_map = build_flat_frame_gain_map(
        flat_linear,
        map_long_edge=map_long_edge,
        blur_radius=blur_radius,
        max_gain=max_gain,
        map_mode=map_mode,
    )

    output_profile_path.parent.mkdir(parents=True, exist_ok=True)
    map_path = output_profile_path.with_suffix(".npy")
    np.save(map_path, gain_map.astype(np.float32, copy=False))

    payload = {
        "schema_version": LENS_PROFILE_SCHEMA_VERSION,
        "type": "flat_frame",
        "name": name or output_profile_path.stem,
        "source_raw": str(source_raw_path),
        "source_resolution": [raw_image.source_size.width, raw_image.source_size.height],
        "map_resolution": [int(gain_map.shape[1]), int(gain_map.shape[0])],
        "blur_radius": int(blur_radius),
        "map_mode": map_mode,
        "max_gain": float(max_gain),
        "map_file": map_path.name,
    }
    output_profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    clear_flat_frame_gain_cache()
    return LensCorrectionParams(
        enabled=True,
        mode="flat_frame",
        max_gain=int(round(max_gain * 100.0)),
        flat_profile_path=str(output_profile_path),
        flat_strength=100,
    )


def clear_flat_frame_gain_cache() -> None:
    load_flat_frame_gain_map.cache_clear()
    _flat_frame_gain_for_size_cached.cache_clear()
    effective_flat_frame_gain_for_size.cache_clear()


def build_flat_frame_gain_map(
    flat_linear_rgb: np.ndarray,
    *,
    map_long_edge: int,
    blur_radius: int,
    max_gain: float,
    map_mode: str,
) -> np.ndarray:
    if flat_linear_rgb.ndim != 3 or flat_linear_rgb.shape[2] != 3:
        raise ValueError("Flat frame must be an RGB linear image.")

    reduced = _resize_long_edge_float(flat_linear_rgb.astype(np.float32, copy=False), map_long_edge)
    blurred = _blur_flat_frame(reduced, blur_radius)
    eps = 1e-5
    safe = np.maximum(blurred, eps)

    if map_mode == "per_channel":
        center = _center_patch(safe)
        target = np.percentile(center.reshape(-1, 3), 70, axis=0).astype(np.float32)
        gain = target.reshape(1, 1, 3) / safe
    else:
        luminance = (
            safe[:, :, 0] * 0.2126
            + safe[:, :, 1] * 0.7152
            + safe[:, :, 2] * 0.0722
        )
        target = float(np.percentile(_center_patch(luminance).reshape(-1), 70))
        gain = target / np.maximum(luminance, eps)

    gain = _normalize_gain_center(gain)
    return np.clip(gain, 1.0, max(1.0, float(max_gain))).astype(np.float32, copy=False)


@lru_cache(maxsize=8)
def load_flat_frame_gain_map(profile_path_text: str) -> tuple[np.ndarray, float]:
    profile_path = Path(profile_path_text)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("type") != "flat_frame":
        raise ValueError("Lens profile is not a flat-frame profile.")
    map_path = _resolve_flat_map_path(profile_path, payload)
    gain = np.load(map_path).astype(np.float32, copy=False)
    return np.ascontiguousarray(gain), float(payload.get("max_gain", 2.0))


def flat_frame_gain_for_size(profile_path: str, size: ImageSize) -> np.ndarray:
    return _flat_frame_gain_for_size_cached(
        str(profile_path),
        int(size.width),
        int(size.height),
    ).copy()


@lru_cache(maxsize=4)
def _flat_frame_gain_for_size_cached(profile_path: str, width: int, height: int) -> np.ndarray:
    gain, max_gain = load_flat_frame_gain_map(profile_path)
    target_shape = (int(height), int(width))
    if gain.shape[:2] != target_shape:
        gain = cv2.resize(gain, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    gain = _normalize_gain_center(gain)
    return np.clip(gain, 1.0, max(1.0, max_gain)).astype(np.float32, copy=False)


@lru_cache(maxsize=4)
def effective_flat_frame_gain_for_size(
    profile_path: str,
    width: int,
    height: int,
    strength_percent: int,
    max_gain_percent: int,
) -> np.ndarray:
    gain = _flat_frame_gain_for_size_cached(str(profile_path), int(width), int(height))
    strength = float(np.clip(float(strength_percent) / 100.0, 0.0, 2.0))
    max_gain = max(1.0, float(max_gain_percent) / 100.0)
    if strength <= 0.0:
        effective = np.ones_like(gain, dtype=np.float32)
    else:
        effective = np.power(np.maximum(gain, 1e-5), strength, dtype=np.float32)
    return np.clip(effective, 1.0, max_gain).astype(np.float32, copy=False)


def _resolve_flat_map_path(profile_path: Path, payload: dict[str, Any]) -> Path:
    map_file = payload.get("map_file")
    if not map_file:
        raise ValueError("Flat-frame profile is missing map_file.")
    map_path = Path(map_file)
    if not map_path.is_absolute():
        map_path = profile_path.parent / map_path
    if not map_path.exists():
        raise ValueError(f"Flat-frame gain map was not found: {map_path}")
    return map_path


def _resize_long_edge_float(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return np.ascontiguousarray(image)
    scale = max_size / float(longest)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    return np.ascontiguousarray(
        cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    )


def _blur_flat_frame(image: np.ndarray, blur_radius: int) -> np.ndarray:
    radius = max(3, int(blur_radius))
    kernel = radius * 2 + 1
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=radius / 2.0, sigmaY=radius / 2.0)


def _normalize_gain_center(gain: np.ndarray) -> np.ndarray:
    center = _center_patch(np.asarray(gain, dtype=np.float32))
    if center.size == 0:
        return gain.astype(np.float32, copy=False)
    if gain.ndim == 3:
        center_gain = np.median(center.reshape(-1, gain.shape[2]), axis=0).astype(np.float32)
        scale = np.maximum(center_gain.reshape(1, 1, -1), 1e-5)
    else:
        scale = max(1e-5, float(np.median(center.reshape(-1))))
    return (gain.astype(np.float32, copy=False) / scale).astype(np.float32, copy=False)


def _center_patch(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    cut_y = max(1, height // 10)
    cut_x = max(1, width // 10)
    cy0 = max(0, height // 2 - cut_y)
    cy1 = min(height, height // 2 + cut_y)
    cx0 = max(0, width // 2 - cut_x)
    cx1 = min(width, width // 2 + cut_x)
    return image[cy0:cy1, cx0:cx1]
