from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Event
from time import perf_counter

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from qnegative.core.models import AdjustmentParams, ImagePoint, ImageRect
from qnegative.core.pipeline import (
    NegativeBasePreview,
    analysis_inset_crop,
    analysis_inset_from_adjustments,
    build_lab_print_color_stage,
    build_lab_print_export_linear,
    build_lab_print_levels_stage,
    build_lab_print_negative_stage,
    build_negative_base_preview,
    suggest_lab_print_luminance_levels,
)
from qnegative.core.raw_loader import load_raw_rgb16


class ExportSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(str, object)
    failed = Signal(str)
    cancelled = Signal(str)


class ExportCancelled(Exception):
    pass


class ImageExportTask(QRunnable):
    def __init__(
        self,
        *,
        source_path: Path,
        output_path: Path,
        mask_point: ImagePoint | None,
        film_rect: ImageRect,
        adjustments: AdjustmentParams,
        flip_horizontal: bool,
        flip_vertical: bool,
        rotation_quarters: int,
        auto_levels_pending: bool,
        export_format: str | None = None,
        preview_cmy_offsets: np.ndarray | None = None,
        roll_color_result: dict | None = None,
        roll_color_frame: dict | None = None,
        cancel_event: Event | None = None,
    ) -> None:
        super().__init__()
        self.source_path = source_path
        self.output_path = output_path
        self.mask_point = mask_point
        self.film_rect = film_rect
        self.adjustments = deepcopy(adjustments)
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.rotation_quarters = rotation_quarters
        self.auto_levels_pending = auto_levels_pending
        self.export_format = export_format or export_format_from_path(output_path)
        self.preview_cmy_offsets = (
            np.asarray(preview_cmy_offsets, dtype=np.float32).copy()
            if preview_cmy_offsets is not None
            else None
        )
        self.roll_color_result = deepcopy(roll_color_result)
        self.roll_color_frame = deepcopy(roll_color_frame)
        self.cancel_event = cancel_event
        self.signals = ExportSignals()

    def run(self) -> None:
        timings: dict[str, float] = {}
        stage_start = perf_counter()
        try:
            self._raise_if_cancelled()
            self.signals.progress.emit(5, "Loading RAW")
            needs_camera_transform = self.adjustments.camera_color_strength > 0
            raw_image = load_raw_rgb16(
                self.source_path,
                half_size=False,
                include_display_transform=needs_camera_transform,
            )
            timings["RAW decode"] = perf_counter() - stage_start
            self._raise_if_cancelled()
            self.signals.progress.emit(30, self._timed_progress_text("Building base", timings))

            stage_start = perf_counter()
            base = build_negative_base_preview(
                raw_image.as_float32(),
                source_size=raw_image.source_size,
                mask_point=self.mask_point,
                film_rect=self.film_rect,
                lens_correction=self.adjustments.lens_correction,
                preview_camera_wb_linear_rgb=raw_image.camera_wb_as_float32(),
                camera_to_srgb_matrix=raw_image.camera_to_srgb_matrix,
            )
            timings["Build base"] = perf_counter() - stage_start
            self._raise_if_cancelled()
            positive_text = (
                "Processing positive with preview CMY WB"
                if self.preview_cmy_offsets is not None
                else "Processing positive"
            )
            self.signals.progress.emit(55, self._timed_progress_text(positive_text, timings))

            stage_start = perf_counter()
            export_linear_rgb = self._process_export(base, timings)
            timings["Lab Print"] = perf_counter() - stage_start
            self._raise_if_cancelled()

            format_label = export_format_label(self.export_format)
            self.signals.progress.emit(75, self._timed_progress_text(f"Preparing {format_label}", timings))
            stage_start = perf_counter()
            linear_rgb = transform_preview_array(
                export_linear_rgb,
                flip_horizontal=self.flip_horizontal,
                flip_vertical=self.flip_vertical,
                rotation_quarters=self.rotation_quarters,
            )
            encoded_rgb = encode_export_rgb(linear_rgb, self.export_format)
            timings[f"Prepare {format_label}"] = perf_counter() - stage_start
            self._raise_if_cancelled()
            self.signals.progress.emit(90, self._timed_progress_text(f"Writing {format_label}", timings))

            stage_start = perf_counter()
            write_export_image(self.output_path, encoded_rgb, self.export_format)
            timings[f"{format_label} write"] = perf_counter() - stage_start
        except ExportCancelled as exc:
            self.signals.cancelled.emit(str(exc))
            return
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(str(self.output_path), timings)

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ExportCancelled("Export cancelled")

    @staticmethod
    def _timed_progress_text(current: str, timings: dict[str, float]) -> str:
        if not timings:
            return current
        elapsed = ", ".join(f"{name} {seconds:.1f}s" for name, seconds in timings.items())
        return f"{current} ({elapsed})"

    def _process_export(self, base: NegativeBasePreview, timings: dict[str, float]) -> np.ndarray:
        stage_start = perf_counter()
        negative_stage = build_lab_print_negative_stage(
            base,
            include_histogram=False,
            analysis_inset=analysis_inset_from_adjustments(self.adjustments),
        )
        timings["Lab negative"] = perf_counter() - stage_start

        effective = deepcopy(self.adjustments)
        if self.auto_levels_pending:
            stage_start = perf_counter()
            auto_levels = suggest_lab_print_luminance_levels(
                analysis_inset_crop(negative_stage.normalized_log, negative_stage.analysis_inset),
                effective,
                camera_to_srgb_matrix=negative_stage.camera_to_srgb_matrix,
            )
            effective.black_point = auto_levels["black_point"]
            effective.mid_point = auto_levels["mid_point"]
            effective.white_point = auto_levels["white_point"]
            timings["Lab auto levels"] = perf_counter() - stage_start
        else:
            auto_levels = _current_levels(effective)
            timings["Lab auto levels"] = 0.0

        stage_start = perf_counter()
        levels_stage = build_lab_print_levels_stage(
            negative_stage,
            effective,
            auto_levels=auto_levels,
        )
        timings["Lab levels"] = perf_counter() - stage_start

        stage_start = perf_counter()
        color_stage = build_lab_print_color_stage(
            levels_stage,
            effective,
            cmy_offsets=self.preview_cmy_offsets if effective.auto_wb else None,
        )
        timings["Lab color print"] = perf_counter() - stage_start

        return build_lab_print_export_linear(
            color_stage,
            effective,
            roll_color_result=self.roll_color_result,
            roll_color_frame=self.roll_color_frame,
            stage_timings=timings,
        )


def _current_levels(adjustments: AdjustmentParams) -> dict[str, int]:
    return {
        "black_point": adjustments.black_point,
        "mid_point": adjustments.mid_point,
        "white_point": adjustments.white_point,
    }


def transform_preview_array(
    image: np.ndarray,
    *,
    flip_horizontal: bool,
    flip_vertical: bool,
    rotation_quarters: int,
) -> np.ndarray:
    transformed = image
    if flip_horizontal:
        transformed = np.flip(transformed, axis=1)
    if flip_vertical:
        transformed = np.flip(transformed, axis=0)
    if rotation_quarters:
        transformed = np.rot90(transformed, k=-(rotation_quarters % 4))
    return np.ascontiguousarray(transformed)


def linear_to_srgb16(linear_rgb: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear_rgb, 0.0, 1.0)
    srgb = np.power(clipped, 1.0 / 2.2)
    return np.ascontiguousarray((srgb * 65535.0 + 0.5).astype(np.uint16))


def linear_to_srgb8(linear_rgb: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear_rgb, 0.0, 1.0)
    srgb = np.power(clipped, 1.0 / 2.2)
    return np.ascontiguousarray((srgb * 255.0 + 0.5).astype(np.uint8))


def export_format_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "jpg"
    if suffix == ".png":
        return "png16"
    return "tiff16"


def export_format_from_filter(selected_filter: str) -> str | None:
    lower = selected_filter.lower()
    if "jpeg" in lower or "jpg" in lower:
        return "jpg"
    if "png" in lower and "8-bit" in lower:
        return "png8"
    if "png" in lower:
        return "png16"
    if ("tiff" in lower or "tif" in lower) and "8-bit" in lower:
        return "tiff8"
    if "tiff" in lower or "tif" in lower:
        return "tiff16"
    return None


def export_format_extension(export_format: str) -> str:
    return {
        "jpg": ".jpg",
        "png8": ".png",
        "png16": ".png",
        "tiff8": ".tif",
        "tiff16": ".tif",
    }.get(export_format, ".tif")


def export_format_label(export_format: str) -> str:
    return {
        "jpg": "JPEG",
        "png8": "PNG 8-bit",
        "png16": "PNG 16-bit",
        "tiff8": "TIFF 8-bit",
        "tiff16": "TIFF 16-bit",
    }.get(export_format, "TIFF")


def encode_export_rgb(linear_rgb: np.ndarray, export_format: str) -> np.ndarray:
    if export_format in {"jpg", "png8", "tiff8"}:
        return linear_to_srgb8(linear_rgb)
    return linear_to_srgb16(linear_rgb)


def write_export_image(path: Path, encoded_rgb: np.ndarray, export_format: str) -> None:
    if export_format in {"tiff8", "tiff16"}:
        import tifffile

        tifffile.imwrite(path, encoded_rgb, photometric="rgb")
        return

    import cv2

    bgr = cv2.cvtColor(encoded_rgb, cv2.COLOR_RGB2BGR)
    if export_format == "jpg":
        ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif export_format in {"png8", "png16"}:
        ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
    if not ok:
        raise OSError(f"Could not write {path}")
