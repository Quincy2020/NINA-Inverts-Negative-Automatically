from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qnegative.core.models import (
    AdjustmentParams,
    DensityMatrixParams,
    ImagePoint,
    ImageRect,
    LensCorrectionParams,
)
from qnegative.core.pipeline import (
    LabPrintColorStage,
    LabPrintLevelsStage,
    LabPrintNegativeStage,
    NegativeBasePreview,
    NegativePreviewResult,
    log_print_curve_engine,
)
from qnegative.core.preview import RawPreview
from qnegative.core.roll_color_adapter import roll_color_frame_key


@dataclass(frozen=True)
class PreviewStageCache:
    base_key: tuple | None = None
    base: NegativeBasePreview | None = None
    negative_key: tuple | None = None
    negative_stage: LabPrintNegativeStage | None = None
    levels_key: tuple | None = None
    levels_stage: LabPrintLevelsStage | None = None
    color_key: tuple | None = None
    color_stage: LabPrintColorStage | None = None
    display_key: tuple | None = None
    display_result: NegativePreviewResult | None = None


@dataclass(frozen=True)
class PreviewRenderOutput:
    path: Path
    result: NegativePreviewResult
    cache: PreviewStageCache
    quality: str
    render_token: int = 0
    cache_key: tuple | None = None
    mask_point: ImagePoint | None = None
    film_rect: ImageRect | None = None
    adjustments: AdjustmentParams | None = None
    lab_print_cmy_offsets: list[float] | None = None
    roll_color_frame: dict | None = None
    applied_auto_levels: bool = False


@dataclass(frozen=True)
class CachedPreviewResult:
    key: tuple
    result: NegativePreviewResult


@dataclass(frozen=True)
class CachedRawPreview:
    key: tuple
    preview: RawPreview


def current_levels(adjustments: AdjustmentParams) -> dict[str, int]:
    return {
        "black_point": adjustments.black_point,
        "mid_point": adjustments.mid_point,
        "white_point": adjustments.white_point,
    }


def image_point_key(point: ImagePoint | None) -> tuple[int, int] | None:
    if point is None:
        return None
    return (point.x, point.y)


def image_rect_key(rect: ImageRect | None) -> tuple[int, int, int, int, float] | None:
    if rect is None:
        return None
    return (rect.x, rect.y, rect.width, rect.height, round(rect.angle, 4))


def matrix_key(matrix: np.ndarray | None) -> tuple[float, ...] | None:
    if matrix is None:
        return None
    values = np.asarray(matrix, dtype=np.float32).reshape(-1)
    return tuple(round(float(value), 7) for value in values)


def cmy_offsets_key(offsets: list[float] | np.ndarray | None) -> tuple[float, ...] | None:
    if offsets is None:
        return None
    values = np.asarray(offsets, dtype=np.float32).reshape(-1)
    if values.size != 3:
        return None
    return tuple(round(float(value), 7) for value in values)


def file_identity_key(path: str | Path | None) -> tuple | None:
    if not path:
        return None
    profile_path = Path(path)
    try:
        stat = profile_path.stat()
    except OSError:
        return (str(profile_path), "missing")
    return (str(profile_path), stat.st_size, stat.st_mtime_ns)


def balance_axis_key(axis) -> tuple[int, int, int]:
    return (axis.red_cyan, axis.green_magenta, axis.blue_yellow)


def tonal_balance_key(axis) -> tuple[int, int, int, int]:
    return (
        axis.red_cyan,
        axis.green_magenta,
        axis.blue_yellow,
        axis.tonal_range,
    )


def color_balance_key(adjustments: AdjustmentParams) -> tuple:
    params = adjustments.color_balance
    return (
        tonal_balance_key(params.shadows),
        tonal_balance_key(params.midtones),
        tonal_balance_key(params.highlights),
    )


def color_correction_key(adjustments: AdjustmentParams) -> tuple:
    params = adjustments.color_correction
    return (
        params.enabled,
        params.roll_strength,
        params.frame_residual_strength,
        params.tone_balance_strength,
        params.protection_strength,
        params.exposure_match_strength,
    )


def density_matrix_params_key(matrix: DensityMatrixParams) -> tuple[float, ...]:
    return (
        round(float(matrix.m00), 7),
        round(float(matrix.m01), 7),
        round(float(matrix.m02), 7),
        round(float(matrix.m10), 7),
        round(float(matrix.m11), 7),
        round(float(matrix.m12), 7),
        round(float(matrix.m20), 7),
        round(float(matrix.m21), 7),
        round(float(matrix.m22), 7),
    )


def lens_correction_key(params: LensCorrectionParams) -> tuple:
    return (
        params.enabled,
        params.mode,
        params.strength,
        params.radius,
        params.center_x,
        params.center_y,
        params.smoothness,
        params.max_gain,
        file_identity_key(params.flat_profile_path),
        params.flat_strength,
    )


def lab_print_engine_key() -> str:
    return log_print_curve_engine()


def adjustments_preview_cache_key(adjustments: AdjustmentParams) -> tuple:
    return (
        adjustments.invert_mode,
        adjustments.print_curve,
        adjustments.auto_wb,
        balance_axis_key(adjustments.printer_balance),
        color_balance_key(adjustments),
        color_correction_key(adjustments),
        density_matrix_params_key(adjustments.density_matrix),
        lens_correction_key(adjustments.lens_correction),
        adjustments.exposure,
        adjustments.highlights,
        adjustments.shadows,
        adjustments.contrast,
        adjustments.saturation,
        adjustments.camera_color_strength,
        adjustments.soft_highlights,
        adjustments.soft_shadows,
        adjustments.analysis_inset_percent,
        adjustments.black_point,
        adjustments.mid_point,
        adjustments.white_point,
    )


def preview_result_cache_key_for(
    *,
    file_key: tuple,
    preview: RawPreview,
    mask_point: ImagePoint | None,
    film_rect: ImageRect | None,
    adjustments: AdjustmentParams,
    lab_print_cmy_offsets: list[float] | np.ndarray | None = None,
    roll_color_frame: dict | None = None,
) -> tuple:
    return (
        file_key,
        preview.source_size,
        preview.preview_size,
        matrix_key(preview.camera_to_srgb_matrix),
        image_point_key(mask_point),
        image_rect_key(film_rect),
        adjustments_preview_cache_key(adjustments),
        lab_print_engine_key(),
        cmy_offsets_key(lab_print_cmy_offsets) if adjustments.auto_wb else None,
        roll_color_frame_key(roll_color_frame) if adjustments.color_correction.enabled else None,
    )


def base_stage_key(
    preview: RawPreview,
    mask_point: ImagePoint | None,
    film_rect: ImageRect | None,
    adjustments: AdjustmentParams,
) -> tuple:
    return (
        "base",
        preview.path,
        preview.source_size,
        preview.preview_linear_rgb.shape,
        id(preview.preview_linear_rgb),
        id(preview.preview_camera_wb_linear_rgb),
        matrix_key(preview.camera_to_srgb_matrix),
        image_point_key(mask_point),
        image_rect_key(film_rect),
        lens_correction_key(adjustments.lens_correction),
    )


def lab_print_auto_key(adjustments: AdjustmentParams) -> tuple:
    return (
        adjustments.print_curve,
        adjustments.exposure,
        adjustments.contrast,
        adjustments.analysis_inset_percent,
        adjustments.soft_highlights,
        adjustments.soft_shadows,
        adjustments.auto_wb,
        balance_axis_key(adjustments.printer_balance),
        lab_print_engine_key(),
        adjustments.camera_color_strength,
        color_balance_key(adjustments),
        adjustments.highlights,
        adjustments.shadows,
    )


def lab_print_levels_key(
    negative_key: tuple,
    adjustments: AdjustmentParams,
    *,
    auto_levels_pending: bool,
) -> tuple:
    auto_part = lab_print_auto_key(adjustments) if auto_levels_pending else "manual"
    return (
        "lab_print_levels",
        negative_key,
        adjustments.analysis_inset_percent,
        adjustments.black_point,
        adjustments.mid_point,
        adjustments.white_point,
        auto_part,
    )


def lab_print_color_key(
    levels_key: tuple,
    adjustments: AdjustmentParams,
    lab_print_cmy_offsets: list[float] | np.ndarray | None = None,
) -> tuple:
    return (
        "lab_print_color",
        levels_key,
        adjustments.print_curve,
        adjustments.exposure,
        adjustments.contrast,
        adjustments.soft_highlights,
        adjustments.soft_shadows,
        adjustments.auto_wb,
        cmy_offsets_key(lab_print_cmy_offsets) if adjustments.auto_wb else None,
        balance_axis_key(adjustments.printer_balance),
        lab_print_engine_key(),
        adjustments.camera_color_strength,
        color_balance_key(adjustments),
    )


def lab_print_display_key(
    color_key: tuple,
    adjustments: AdjustmentParams,
    roll_color_frame: dict | None = None,
) -> tuple:
    return (
        "lab_print_display",
        color_key,
        color_correction_key(adjustments),
        roll_color_frame_key(roll_color_frame) if adjustments.color_correction.enabled else None,
        adjustments.highlights,
        adjustments.shadows,
        adjustments.saturation,
    )
