from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import Event
from time import perf_counter

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from qnegative.core.dust_masks import compose_dust_masks
from qnegative.core.dust_masks import dust_auto_mask_params_key, load_dust_mask
from qnegative.core.dust_removal import inpaint_srgb, linear_to_srgb_float, srgb_to_linear_float
from qnegative.core.models import AdjustmentParams, DustMaskState, ImagePoint, ImageRect
from qnegative.core.pipeline import (
    LabPrintBasePreview,
    analysis_inset_crop,
    analysis_inset_from_adjustments,
    build_lab_print_base_preview,
    build_lab_print_color_stage,
    build_lab_print_export_linear,
    build_lab_print_levels_stage,
    build_lab_print_negative_stage,
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
        preview_log_floors: np.ndarray | list[float] | None = None,
        preview_log_ceils: np.ndarray | list[float] | None = None,
        preview_tone_mid_anchor: float | None = None,
        roll_color_result: dict | None = None,
        roll_color_frame: dict | None = None,
        dust_mask: DustMaskState | None = None,
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
        self.preview_log_floors = (
            np.asarray(preview_log_floors, dtype=np.float32).reshape(3).copy()
            if preview_log_floors is not None
            else None
        )
        self.preview_log_ceils = (
            np.asarray(preview_log_ceils, dtype=np.float32).reshape(3).copy()
            if preview_log_ceils is not None
            else None
        )
        self.preview_tone_mid_anchor = (
            float(preview_tone_mid_anchor)
            if preview_tone_mid_anchor is not None
            else None
        )
        self.roll_color_result = deepcopy(roll_color_result)
        self.roll_color_frame = deepcopy(roll_color_frame)
        self.dust_mask = deepcopy(dust_mask) if dust_mask is not None else DustMaskState()
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
            base = build_lab_print_base_preview(
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
            self.signals.progress.emit(70, self._timed_progress_text("Applying orientation", timings))
            stage_start = perf_counter()
            linear_rgb = transform_preview_array(
                export_linear_rgb,
                flip_horizontal=self.flip_horizontal,
                flip_vertical=self.flip_vertical,
                rotation_quarters=self.rotation_quarters,
            )
            timings["Orientation"] = perf_counter() - stage_start
            self._raise_if_cancelled()

            if self.adjustments.dust_removal.enabled:
                self.signals.progress.emit(72, self._timed_progress_text("Removing dust", timings))
                stage_start = perf_counter()
                auto_mask, manual_add, manual_protect, auto_mask_status = self._load_dust_masks(linear_rgb.shape[:2])
                linear_rgb, dust_stats = self._apply_dust_removal(
                    linear_rgb,
                    auto_mask=auto_mask,
                    manual_add_mask=manual_add,
                    manual_protect_mask=manual_protect,
                    auto_mask_status=auto_mask_status,
                )
                timings["Dust removal"] = perf_counter() - stage_start
                for key, value in dust_stats.items():
                    timings[f"Dust {key}"] = float(value)
                self._raise_if_cancelled()

            self.signals.progress.emit(75, self._timed_progress_text(f"Preparing {format_label}", timings))
            stage_start = perf_counter()
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

    def _load_dust_masks(
        self,
        target_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
        auto_mask_path = self.dust_mask.auto_mask_path
        expected_key = dust_auto_mask_params_key(
            self.adjustments.dust_removal
        )
        auto_key_matches = self.dust_mask.auto_mask_params_key == expected_key
        if auto_key_matches and auto_mask_path is None:
            auto_mask = np.zeros(target_shape, dtype=bool)
            auto_mask_status = "empty_reused"
        elif auto_key_matches:
            loaded = load_dust_mask(
                self.source_path,
                auto_mask_path,
                target_shape=target_shape,
            )
            auto_mask = loaded if loaded is not None else np.zeros(target_shape, dtype=bool)
            auto_mask_status = "reused" if loaded is not None else "missing_reused_empty"
        else:
            auto_mask = np.zeros(target_shape, dtype=bool)
            auto_mask_status = "skipped_param_mismatch"
        return (
            auto_mask,
            load_dust_mask(
                self.source_path,
                self.dust_mask.manual_add_mask_path,
                target_shape=target_shape,
            ),
            load_dust_mask(
                self.source_path,
                self.dust_mask.manual_protect_mask_path,
                target_shape=target_shape,
            ),
            auto_mask_status,
        )

    def _apply_dust_removal(
        self,
        linear_rgb: np.ndarray,
        *,
        auto_mask: np.ndarray | None,
        manual_add_mask: np.ndarray | None,
        manual_protect_mask: np.ndarray | None,
        auto_mask_status: str,
    ) -> tuple[np.ndarray, dict[str, float]]:
        srgb = linear_to_srgb_float(linear_rgb)
        mask = auto_mask if auto_mask is not None else np.zeros(srgb.shape[:2], dtype=bool)
        stats = {
            "dust_auto_mask_reused": 1.0 if auto_mask_status in {"reused", "empty_reused"} else 0.0,
            "dust_auto_mask_skipped": 1.0 if auto_mask_status.startswith("skipped") else 0.0,
            "dust_auto_mask_missing": 1.0 if auto_mask_status.startswith("missing") else 0.0,
        }
        final_mask = compose_dust_masks(
            mask,
            manual_add_mask,
            manual_protect_mask,
            target_shape=srgb.shape[:2],
        )
        stats["dust_auto_mask_area"] = float(np.mean(mask > 0))
        stats["dust_manual_add_area"] = (
            float(np.mean(manual_add_mask > 0))
            if manual_add_mask is not None and manual_add_mask.size
            else 0.0
        )
        stats["dust_manual_protect_area"] = (
            float(np.mean(manual_protect_mask > 0))
            if manual_protect_mask is not None and manual_protect_mask.size
            else 0.0
        )
        stats["dust_mask_area"] = float(np.mean(final_mask > 0))
        stats["dust_final_mask_area"] = stats["dust_mask_area"]
        if not np.any(final_mask):
            stats["inpaint_area"] = 0.0
            return linear_rgb, stats

        repaired = inpaint_srgb(
            srgb,
            final_mask,
            radius=max(1, int(self.adjustments.dust_removal.inpaint_radius)),
        )
        stats["inpaint_area"] = float(np.mean(final_mask > 0))
        return srgb_to_linear_float(repaired).astype(np.float32, copy=False), stats

    @staticmethod
    def _timed_progress_text(current: str, timings: dict[str, float]) -> str:
        if not timings:
            return current
        elapsed = ", ".join(f"{name} {seconds:.1f}s" for name, seconds in timings.items())
        return f"{current} ({elapsed})"

    def _process_export(self, base: LabPrintBasePreview, timings: dict[str, float]) -> np.ndarray:
        stage_start = perf_counter()
        negative_stage = build_lab_print_negative_stage(
            base,
            include_histogram=False,
            analysis_inset=analysis_inset_from_adjustments(self.adjustments),
            lab_print_log_floors=self.preview_log_floors,
            lab_print_log_ceils=self.preview_log_ceils,
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
            stage_timings=timings,
        )
        timings["Lab color print"] = perf_counter() - stage_start

        return build_lab_print_export_linear(
            color_stage,
            effective,
            roll_color_result=self.roll_color_result,
            roll_color_frame=self.roll_color_frame,
            tone_mid_anchor=self.preview_tone_mid_anchor,
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

        icc_profile = srgb_icc_profile_bytes()
        extratags = (
            [(34675, "B", len(icc_profile), icc_profile, False)]
            if icc_profile
            else None
        )
        tifffile.imwrite(
            path,
            encoded_rgb,
            photometric="rgb",
            extratags=extratags,
        )
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


@lru_cache(maxsize=1)
def srgb_icc_profile_bytes() -> bytes:
    try:
        from PIL import ImageCms

        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        return bytes(profile.tobytes())
    except Exception:
        return b""
