from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy

from qnegative.core.models import ImageSize


@dataclass(frozen=True)
class RawRgbImage:
    path: Path
    source_size: ImageSize
    rgb16: np.ndarray
    camera_wb_rgb16: np.ndarray
    display_rgb16: np.ndarray
    camera_to_srgb_matrix: np.ndarray

    @property
    def output_size(self) -> ImageSize:
        height, width = self.rgb16.shape[:2]
        return ImageSize(width=width, height=height)

    def as_float32(self) -> np.ndarray:
        return self.rgb16.astype(np.float32) / 65535.0

    def camera_wb_as_float32(self) -> np.ndarray:
        return self.camera_wb_rgb16.astype(np.float32) / 65535.0

    def display_as_float32(self) -> np.ndarray:
        return self.display_rgb16.astype(np.float32) / 65535.0


def load_raw_rgb16(path: str | Path, *, half_size: bool = False) -> RawRgbImage:
    source_path = Path(path)
    with rawpy.imread(str(source_path)) as raw:
        source_size = ImageSize(width=raw.sizes.iwidth, height=raw.sizes.iheight)
        postprocess_kwargs = {
            "use_auto_wb": False,
            "no_auto_bright": True,
            "gamma": (1, 1),
            "output_bps": 16,
            "user_flip": 0,
            "half_size": half_size,
        }
        rgb16 = raw.postprocess(
            **postprocess_kwargs,
            use_camera_wb=False,
            output_color=rawpy.ColorSpace.raw,
        )
        camera_wb_rgb16 = raw.postprocess(
            **postprocess_kwargs,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.raw,
        )
        display_rgb16 = raw.postprocess(
            **postprocess_kwargs,
            use_camera_wb=False,
            output_color=rawpy.ColorSpace.sRGB,
        )

    camera_to_srgb_matrix = fit_camera_to_srgb_matrix(rgb16, display_rgb16)

    return RawRgbImage(
        path=source_path,
        source_size=source_size,
        rgb16=np.ascontiguousarray(rgb16),
        camera_wb_rgb16=np.ascontiguousarray(camera_wb_rgb16),
        display_rgb16=np.ascontiguousarray(display_rgb16),
        camera_to_srgb_matrix=camera_to_srgb_matrix,
    )


def fit_camera_to_srgb_matrix(camera_rgb16: np.ndarray, srgb_rgb16: np.ndarray) -> np.ndarray:
    camera_pixels = camera_rgb16.reshape(-1, 3)
    srgb_pixels = srgb_rgb16.reshape(-1, 3)
    stride = max(1, len(camera_pixels) // 250_000)

    x = camera_pixels[::stride].astype(np.float32) / 65535.0
    y = srgb_pixels[::stride].astype(np.float32) / 65535.0
    valid = np.all(x > 1e-6, axis=1) & np.all(x < 0.98, axis=1) & np.all(y < 0.98, axis=1)
    if int(np.count_nonzero(valid)) >= 128:
        x = x[valid]
        y = y[valid]

    try:
        matrix, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        matrix = np.eye(3, dtype=np.float32)

    return matrix.astype(np.float32, copy=False)
