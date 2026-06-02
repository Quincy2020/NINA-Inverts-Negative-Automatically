from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qnegative.core.models import ImageSize
from qnegative.core.raw_loader import load_raw_rgb16


DEFAULT_PREVIEW_MAX_EDGE = 2048


@dataclass(frozen=True)
class RawPreview:
    path: Path
    source_size: ImageSize
    preview_size: ImageSize
    preview_linear_rgb: np.ndarray
    preview_camera_wb_linear_rgb: np.ndarray
    display_rgb8: np.ndarray
    camera_to_srgb_matrix: np.ndarray

    @property
    def scale_x(self) -> float:
        return self.preview_size.width / self.source_size.width

    @property
    def scale_y(self) -> float:
        return self.preview_size.height / self.source_size.height

    def status_text(self) -> str:
        return (
            f"Source {self.source_size.label()}\n"
            f"Preview {self.preview_size.label()}\n"
            f"Scale {self.scale_x:.3f}x, {self.scale_y:.3f}x"
        )


def make_raw_preview(path: str | Path, *, max_size: int = DEFAULT_PREVIEW_MAX_EDGE) -> RawPreview:
    raw_image = load_raw_rgb16(path, half_size=True)
    preview_linear = resize_long_edge(raw_image.as_float32(), max_size=max_size)
    preview_camera_wb_linear = resize_long_edge(raw_image.camera_wb_as_float32(), max_size=max_size)
    display_linear = resize_long_edge(raw_image.display_as_float32(), max_size=max_size)
    display_rgb8 = linear_to_display_rgb8(display_linear)
    height, width = preview_linear.shape[:2]

    return RawPreview(
        path=Path(path),
        source_size=raw_image.source_size,
        preview_size=ImageSize(width=width, height=height),
        preview_linear_rgb=np.ascontiguousarray(preview_linear),
        preview_camera_wb_linear_rgb=np.ascontiguousarray(preview_camera_wb_linear),
        display_rgb8=display_rgb8,
        camera_to_srgb_matrix=raw_image.camera_to_srgb_matrix,
    )


def resize_long_edge(image: np.ndarray, *, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= max_size:
        return np.ascontiguousarray(image)

    scale = max_size / longest
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized)


def linear_to_display_rgb8(linear_rgb: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear_rgb, 0.0, 1.0)
    sample = clipped.reshape(-1, 3)
    stride = max(1, len(sample) // 500_000)
    sample = sample[::stride]

    low = float(np.percentile(sample, 0.5))
    high = float(np.percentile(sample, 99.5))
    if high <= low:
        low, high = 0.0, 1.0

    normalized = np.clip((clipped - low) / (high - low), 0.0, 1.0)
    display = np.power(normalized, 1.0 / 2.2)
    return np.ascontiguousarray((display * 255.0 + 0.5).astype(np.uint8))
