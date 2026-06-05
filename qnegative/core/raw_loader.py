from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy

from qnegative.core.file_sequence import RAW_EXTENSIONS, TIFF_EXTENSIONS
from qnegative.core.models import ImageSize


@dataclass(frozen=True)
class RawRgbImage:
    path: Path
    source_size: ImageSize
    rgb16: np.ndarray
    camera_wb_rgb16: np.ndarray
    display_rgb16: np.ndarray | None
    camera_to_srgb_matrix: np.ndarray | None

    @property
    def output_size(self) -> ImageSize:
        height, width = self.rgb16.shape[:2]
        return ImageSize(width=width, height=height)

    def as_float32(self) -> np.ndarray:
        return self.rgb16.astype(np.float32) / 65535.0

    def camera_wb_as_float32(self) -> np.ndarray:
        return self.camera_wb_rgb16.astype(np.float32) / 65535.0

    def display_as_float32(self) -> np.ndarray:
        if self.display_rgb16 is None:
            raise ValueError("Display RGB data was not loaded for this RAW image.")
        return self.display_rgb16.astype(np.float32) / 65535.0


def load_raw_rgb16(
    path: str | Path,
    *,
    half_size: bool = False,
    include_display_transform: bool = True,
) -> RawRgbImage:
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

        if include_display_transform:
            display_rgb16 = raw.postprocess(
                **postprocess_kwargs,
                use_camera_wb=False,
                output_color=rawpy.ColorSpace.sRGB,
            )
        else:
            display_rgb16 = None

    camera_to_srgb_matrix = (
        fit_camera_to_srgb_matrix(rgb16, display_rgb16)
        if display_rgb16 is not None
        else None
    )

    return RawRgbImage(
        path=source_path,
        source_size=source_size,
        rgb16=np.ascontiguousarray(rgb16),
        camera_wb_rgb16=np.ascontiguousarray(camera_wb_rgb16),
        display_rgb16=np.ascontiguousarray(display_rgb16) if display_rgb16 is not None else None,
        camera_to_srgb_matrix=camera_to_srgb_matrix,
    )


def load_tiff_rgb16(path: str | Path) -> RawRgbImage:
    import tifffile

    source_path = Path(path)
    image = tifffile.imread(str(source_path))
    rgb16 = _coerce_tiff_to_rgb16(image)
    height, width = rgb16.shape[:2]
    source_size = ImageSize(width=width, height=height)
    identity = np.eye(3, dtype=np.float32)

    return RawRgbImage(
        path=source_path,
        source_size=source_size,
        rgb16=np.ascontiguousarray(rgb16),
        camera_wb_rgb16=np.ascontiguousarray(rgb16),
        display_rgb16=np.ascontiguousarray(rgb16),
        camera_to_srgb_matrix=identity,
    )


def load_source_rgb16(
    path: str | Path,
    *,
    half_size: bool = False,
    include_display_transform: bool = True,
) -> RawRgbImage:
    suffix = Path(path).suffix.lower()
    if suffix in RAW_EXTENSIONS:
        return load_raw_rgb16(
            path,
            half_size=half_size,
            include_display_transform=include_display_transform,
        )
    if suffix in TIFF_EXTENSIONS:
        return load_tiff_rgb16(path)
    raise ValueError(f"Unsupported source file type: {suffix or 'unknown'}")


def _coerce_tiff_to_rgb16(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    array = np.squeeze(array)
    while array.ndim > 3:
        array = array[0]

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3:
        if array.shape[-1] in {3, 4}:
            array = array[:, :, :3]
        elif array.shape[0] in {3, 4}:
            array = np.moveaxis(array[:3], 0, -1)
        else:
            array = np.repeat(array[0, :, :, None], 3, axis=2)
    else:
        raise ValueError("TIFF image must be a 2D grayscale or 3-channel RGB image.")

    return _normalize_to_uint16(array)


def _normalize_to_uint16(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint16:
        return array
    if array.dtype == np.uint8:
        return (array.astype(np.uint16) * 257).astype(np.uint16, copy=False)

    if np.issubdtype(array.dtype, np.floating):
        values = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if values.size and float(np.nanmax(values)) > 1.0:
            max_value = max(float(np.nanmax(values)), 1.0)
            values = values / max_value
        return np.ascontiguousarray((np.clip(values, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16))

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        values = array.astype(np.float32)
        if info.min < 0:
            values = values - float(info.min)
            denominator = float(info.max - info.min)
        else:
            denominator = float(info.max)
        values = values / max(denominator, 1.0)
        return np.ascontiguousarray((np.clip(values, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16))

    raise ValueError(f"Unsupported TIFF dtype: {array.dtype}")


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
