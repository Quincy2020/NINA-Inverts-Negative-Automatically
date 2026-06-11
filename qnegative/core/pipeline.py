from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from qnegative.core.geometry import clamp_rect_to_image, scale_point, scale_rect, warp_rotated_rect
from qnegative.core.lens_profiles import effective_flat_frame_gain_for_size
from qnegative.core.models import AdjustmentParams, BalanceAxis, ColorBalanceParams, ImagePoint, ImageRect, ImageSize, LensCorrectionParams, PrintCurveMode, TonalBalance
from qnegative.core.roll_color_apply import apply_roll_color_to_linear_rgb


LOG_MODE_EPSILON = 1e-6
LOG_ANALYSIS_BUFFER = 0.05
LOG_DENSITY_MULTIPLIER = 0.2
LOG_GRADE_MULTIPLIER = 1.75
LOG_D_MAX = 4.0
LOG_TOE_WIDTH = 2.5
LOG_SHOULDER_WIDTH = 2.5
LAB_PRINT_LOG_PERCENTILE_CLIP = 0.02
LAB_PRINT_AUTO_WB_STRENGTH = 0.65
LAB_PRINT_AUTO_WB_MAX_OFFSET = 0.04
LAB_PRINT_MANUAL_CMY_OFFSET_SCALE = 0.0009
LAB_PRINT_MANUAL_CMY_MAX_OFFSET = 0.09
LAB_PRINT_COLOR_SEPARATION_STRENGTH = 0.45
LAB_PRINT_AUTO_EXPOSURE_SAMPLE_LIMIT = 180_000
TONE_MID_ANCHOR_SAMPLE_LIMIT = 180_000
TONE_MID_ANCHOR_PERCENTILE = 55.0
TONE_MID_ANCHOR_MIN = 0.32
TONE_MID_ANCHOR_MAX = 0.62
LAB_PRINT_AUTO_BLACK_PERCENTILE = 0.25
LAB_PRINT_AUTO_WHITE_PERCENTILE = 99.75
LAB_PRINT_AUTO_BLACK_PADDING = 0.060
LAB_PRINT_AUTO_WHITE_PADDING = 0.085
LAB_PRINT_AUTO_MIN_SPAN = 0.54
LAB_PRINT_AUTO_MID_LOW = 0.04
LAB_PRINT_AUTO_MID_HIGH = 0.74
LAB_PRINT_AUTO_VISUAL_TARGET = 0.28
LAB_PRINT_ANALYSIS_INSET = 0.05
GLOBAL_BALANCE_SCALE = 55.0
TONAL_BALANCE_SCALE = 75.0
FILMIC_LUT_SIZE = 4096
TONE_MODIFIER_LUT_SIZE = 16384
LOG_PRINT_CURVE_LUT_4096 = "lut_4096"
LOG_PRINT_CURVE_LUT_8192 = "lut_8192"
LOG_PRINT_CURVE_LUT_DIRECT_16384 = "lut_direct_16384"
LOG_PRINT_CURVE_DIRECT = "direct"
LOG_PRINT_CURVE_ENGINE = LOG_PRINT_CURVE_LUT_DIRECT_16384
LOG_PRINT_CURVE_LUT_SIZES = {
    LOG_PRINT_CURVE_LUT_4096: 4096,
    LOG_PRINT_CURVE_LUT_8192: 8192,
    LOG_PRINT_CURVE_LUT_DIRECT_16384: 16384,
}
GAMUT_EPSILON = 1e-6
GAMUT_LOWER_MARGIN = 1e-5
GAMUT_UPPER_MARGIN = 1.0 - GAMUT_LOWER_MARGIN
GAMUT_SHADOW_CHROMA_LOW = 0.01
GAMUT_SHADOW_CHROMA_HIGH = 0.12
GAMUT_SHADOW_CHROMA_STRENGTH = 0.55
LEVELS_SOFT_BLACK_FLOOR = 0.004
LEVELS_SOFT_BLACK_WIDTH = 0.10
FILMIC_CURVE_PRESETS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    PrintCurveMode.SOFT.value: ((0.30, 0.16), (0.74, 0.86)),
    PrintCurveMode.STANDARD.value: ((0.26, 0.11), (0.76, 0.92)),
    PrintCurveMode.CONTRAST.value: ((0.20, 0.07), (0.82, 0.97)),
    PrintCurveMode.CONTRAST_SHOULDER.value: ((0.20, 0.07), (0.70, 0.90)),
}
RATIONAL_FILMIC_PRINT_MODES = {PrintCurveMode.FILMIC_HABLE.value, PrintCurveMode.FILMIC_ACES.value}
_FILMIC_CURVE_LUTS: dict[str, np.ndarray] = {}
_LOG_PRINT_CURVE_LUTS: dict[tuple, np.ndarray] = {}
_TONE_MODIFIER_LUTS: dict[tuple[int, int, int, int], np.ndarray] = {}

LOG_COLOR_SEPARATION_MATRIX = np.array(
    [
        [1.0, -0.05, -0.02],
        [-0.04, 1.0, -0.08],
        [-0.01, -0.10, 1.0],
    ],
    dtype=np.float32,
)


class PipelineError(ValueError):
    pass


@dataclass(frozen=True)
class LabPrintBasePreview:
    film_linear_rgb: np.ndarray
    film_camera_wb_linear_rgb: np.ndarray | None
    mask_rgb: np.ndarray
    film_rect_preview: ImageRect
    camera_to_srgb_matrix: np.ndarray | None = None

    @property
    def width(self) -> int:
        return self.film_linear_rgb.shape[1]

    @property
    def height(self) -> int:
        return self.film_linear_rgb.shape[0]


@dataclass(frozen=True)
class NegativePreviewResult:
    display_rgb8: np.ndarray
    processed_linear_rgb: np.ndarray
    color_balanced_linear_rgb: np.ndarray
    histogram: np.ndarray
    auto_levels: dict[str, int]
    wb_gains: np.ndarray
    lab_print_log_floors: np.ndarray
    lab_print_log_ceils: np.ndarray
    mask_rgb: np.ndarray
    film_rect_preview: ImageRect
    tone_mid_anchor: float = 0.46

    @property
    def width(self) -> int:
        return self.display_rgb8.shape[1]

    @property
    def height(self) -> int:
        return self.display_rgb8.shape[0]


@dataclass(frozen=True)
class LabPrintNegativeStage:
    normalized_log: np.ndarray
    positive_control: np.ndarray
    histogram: np.ndarray
    lab_print_log_floors: np.ndarray
    lab_print_log_ceils: np.ndarray
    mask_rgb: np.ndarray
    film_rect_preview: ImageRect
    analysis_inset: float
    camera_to_srgb_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class LabPrintLevelsStage:
    normalized_for_print: np.ndarray
    histogram: np.ndarray
    auto_levels: dict[str, int]
    lab_print_log_floors: np.ndarray
    lab_print_log_ceils: np.ndarray
    mask_rgb: np.ndarray
    film_rect_preview: ImageRect
    camera_to_srgb_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class LabPrintColorStage:
    color_linear_rgb: np.ndarray
    histogram: np.ndarray
    auto_levels: dict[str, int]
    wb_gains: np.ndarray
    lab_print_log_floors: np.ndarray
    lab_print_log_ceils: np.ndarray
    mask_rgb: np.ndarray
    film_rect_preview: ImageRect


@dataclass(frozen=True)
class LogPrintCurveParams:
    density: float
    grade: float
    highlight_bias: float = 0.0
    highlight_width: float = 0.55
    shadow_bias: float = 0.0
    shadow_width: float = 0.55


def process_negative_preview(
    preview_linear_rgb: np.ndarray,
    *,
    source_size: ImageSize,
    mask_point: ImagePoint | None,
    film_rect: ImageRect | None,
    adjustments: AdjustmentParams,
    preview_camera_wb_linear_rgb: np.ndarray | None = None,
    camera_to_srgb_matrix: np.ndarray | None = None,
) -> NegativePreviewResult:
    base = build_lab_print_base_preview(
        preview_linear_rgb,
        source_size=source_size,
        mask_point=mask_point,
        film_rect=film_rect,
        lens_correction=adjustments.lens_correction,
        preview_camera_wb_linear_rgb=preview_camera_wb_linear_rgb,
        camera_to_srgb_matrix=camera_to_srgb_matrix,
    )
    return process_negative_base_preview(base, adjustments)


def build_lab_print_base_preview(
    preview_linear_rgb: np.ndarray,
    *,
    source_size: ImageSize,
    mask_point: ImagePoint | None,
    film_rect: ImageRect | None,
    lens_correction: LensCorrectionParams | None = None,
    preview_camera_wb_linear_rgb: np.ndarray | None = None,
    camera_to_srgb_matrix: np.ndarray | None = None,
) -> LabPrintBasePreview:
    if preview_linear_rgb.ndim != 3 or preview_linear_rgb.shape[2] != 3:
        raise PipelineError("Preview image must be an RGB array.")
    if film_rect is None or not film_rect.is_valid():
        raise PipelineError("Select a valid negative frame area first.")

    preview_size = ImageSize(
        width=preview_linear_rgb.shape[1],
        height=preview_linear_rgb.shape[0],
    )
    flat_gain = None
    if (
        lens_correction is not None
        and lens_correction.enabled
        and lens_correction.mode == "flat_frame"
        and lens_correction.flat_profile_path
    ):
        flat_gain = effective_flat_frame_gain_for_size(
            lens_correction.flat_profile_path,
            preview_size.width,
            preview_size.height,
            lens_correction.flat_strength,
            lens_correction.max_gain,
        )
    if mask_point is None:
        mask_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        mask_rgb = sample_mask_rgb(
            preview_linear_rgb,
            source_size=source_size,
            preview_size=preview_size,
            mask_point=mask_point,
        )
        if flat_gain is not None:
            mask_rgb = mask_rgb * sample_gain_rgb(
                flat_gain,
                source_size=source_size,
                preview_size=preview_size,
                mask_point=mask_point,
            )

    film_rect_preview = scale_rect(film_rect, source_size, preview_size)
    film_rect_preview = clamp_rect_to_image(film_rect_preview, preview_size)
    film_linear = warp_rotated_rect(preview_linear_rgb, film_rect_preview)
    film_camera_wb_linear = None
    if preview_camera_wb_linear_rgb is not None:
        if preview_camera_wb_linear_rgb.shape != preview_linear_rgb.shape:
            raise PipelineError("Camera WB preview must match the neutral RAW preview size.")
        film_camera_wb_linear = warp_rotated_rect(preview_camera_wb_linear_rgb, film_rect_preview)
    if flat_gain is not None:
        film_gain = warp_rotated_rect(flat_gain, film_rect_preview)
        film_linear = apply_lens_correction_gain(film_linear, film_gain)
        if film_camera_wb_linear is not None:
            film_camera_wb_linear = apply_lens_correction_gain(film_camera_wb_linear, film_gain)
    if (
        lens_correction is not None
        and lens_correction.enabled
        and lens_correction.mode == "radial"
        and lens_correction.strength != 0
    ):
        gain = radial_lens_correction_gain(film_linear.shape[:2], lens_correction)
        film_linear = apply_lens_correction_gain(film_linear, gain)
        if film_camera_wb_linear is not None:
            film_camera_wb_linear = apply_lens_correction_gain(film_camera_wb_linear, gain)
    return LabPrintBasePreview(
        film_linear_rgb=film_linear,
        film_camera_wb_linear_rgb=film_camera_wb_linear,
        mask_rgb=mask_rgb,
        film_rect_preview=film_rect_preview,
        camera_to_srgb_matrix=camera_to_srgb_matrix,
    )

def process_negative_base_preview(
    base: LabPrintBasePreview,
    adjustments: AdjustmentParams,
) -> NegativePreviewResult:
    return process_lab_print_preview(base, adjustments)


def process_lab_print_preview(
    base: LabPrintBasePreview,
    adjustments: AdjustmentParams,
) -> NegativePreviewResult:
    negative_stage = build_lab_print_negative_stage(
        base,
        analysis_inset=analysis_inset_from_adjustments(adjustments),
    )
    levels_stage = build_lab_print_levels_stage(negative_stage, adjustments)
    color_stage = build_lab_print_color_stage(levels_stage, adjustments)
    return build_lab_print_display_stage(color_stage, adjustments)


def build_lab_print_negative_stage(
    base: LabPrintBasePreview,
    *,
    include_histogram: bool = True,
    analysis_inset: float = LAB_PRINT_ANALYSIS_INSET,
    lab_print_log_floors: np.ndarray | list[float] | None = None,
    lab_print_log_ceils: np.ndarray | list[float] | None = None,
) -> LabPrintNegativeStage:
    source_linear = (
        base.film_camera_wb_linear_rgb
        if base.film_camera_wb_linear_rgb is not None
        else base.film_linear_rgb
    )
    normalized_log, resolved_floors, resolved_ceils = normalize_log_bounds_with_metadata(
        source_linear,
        percentile_clip=LAB_PRINT_LOG_PERCENTILE_CLIP,
        lab_print_log_floors=lab_print_log_floors,
        lab_print_log_ceils=lab_print_log_ceils,
    )
    positive_control = 1.0 - normalized_log
    analysis_control = analysis_inset_crop(positive_control, analysis_inset)
    histogram = (
        luminance_histogram(analysis_control)
        if include_histogram
        else np.zeros(256, dtype=np.float32)
    )

    return LabPrintNegativeStage(
        normalized_log=normalized_log,
        positive_control=positive_control,
        histogram=histogram,
        lab_print_log_floors=resolved_floors,
        lab_print_log_ceils=resolved_ceils,
        mask_rgb=base.mask_rgb,
        film_rect_preview=base.film_rect_preview,
        analysis_inset=analysis_inset,
        camera_to_srgb_matrix=base.camera_to_srgb_matrix,
    )


def build_lab_print_levels_stage(
    negative_stage: LabPrintNegativeStage,
    adjustments: AdjustmentParams,
    *,
    auto_levels: dict[str, int] | None = None,
) -> LabPrintLevelsStage:
    if auto_levels is None:
        auto_levels = suggest_lab_print_luminance_levels(
            analysis_inset_crop(negative_stage.normalized_log, negative_stage.analysis_inset),
            adjustments,
            camera_to_srgb_matrix=negative_stage.camera_to_srgb_matrix,
        )

    positive_control = apply_unit_levels(negative_stage.positive_control, adjustments, clip=True)
    normalized_for_print = np.clip(1.0 - positive_control, 0.0, 1.0)

    return LabPrintLevelsStage(
        normalized_for_print=normalized_for_print,
        histogram=negative_stage.histogram,
        auto_levels=auto_levels,
        lab_print_log_floors=negative_stage.lab_print_log_floors,
        lab_print_log_ceils=negative_stage.lab_print_log_ceils,
        mask_rgb=negative_stage.mask_rgb,
        film_rect_preview=negative_stage.film_rect_preview,
        camera_to_srgb_matrix=negative_stage.camera_to_srgb_matrix,
    )


def build_lab_print_color_stage(
    levels_stage: LabPrintLevelsStage,
    adjustments: AdjustmentParams,
    *,
    cmy_offsets: np.ndarray | None = None,
    stage_timings: dict[str, float] | None = None,
) -> LabPrintColorStage:
    normalized_for_print = levels_stage.normalized_for_print

    stage_start = perf_counter()
    if cmy_offsets is not None:
        cmy_offsets = np.asarray(cmy_offsets, dtype=np.float32)
    elif adjustments.auto_wb:
        cmy_offsets = estimate_lab_print_auto_cmy_offsets(
            normalized_for_print,
            strength=adjustments.auto_cmy_strength / 100.0,
        )
        cmy_offsets = effective_lab_print_cmy_offsets(cmy_offsets, adjustments)
    else:
        cmy_offsets = np.zeros(3, dtype=np.float32)
        cmy_offsets = effective_lab_print_cmy_offsets(cmy_offsets, adjustments)
    if stage_timings is not None:
        stage_timings["Lab color CMY"] = perf_counter() - stage_start

    stage_start = perf_counter()
    processed = apply_log_hd_print_curve(
        normalized_for_print,
        adjustments,
        cmy_offsets=cmy_offsets,
    )
    if stage_timings is not None:
        stage_timings["Lab color print curve"] = perf_counter() - stage_start

    stage_start = perf_counter()
    processed = apply_output_color_transform(
        processed,
        levels_stage.camera_to_srgb_matrix,
        adjustments.camera_color_strength,
    )
    if stage_timings is not None:
        stage_timings["Lab color camera transform"] = perf_counter() - stage_start

    stage_start = perf_counter()
    processed = apply_log_color_separation(
        processed,
        strength=LAB_PRINT_COLOR_SEPARATION_STRENGTH,
    )
    if stage_timings is not None:
        stage_timings["Lab color separation"] = perf_counter() - stage_start

    stage_start = perf_counter()
    processed = apply_color_balance(processed, adjustments.color_balance)
    if stage_timings is not None:
        stage_timings["Lab color balance"] = perf_counter() - stage_start

    return LabPrintColorStage(
        color_linear_rgb=processed,
        histogram=levels_stage.histogram,
        auto_levels=levels_stage.auto_levels,
        wb_gains=cmy_offsets.astype(np.float32, copy=False),
        lab_print_log_floors=levels_stage.lab_print_log_floors,
        lab_print_log_ceils=levels_stage.lab_print_log_ceils,
        mask_rgb=levels_stage.mask_rgb,
        film_rect_preview=levels_stage.film_rect_preview,
    )


def build_lab_print_display_stage(
    color_stage: LabPrintColorStage,
    adjustments: AdjustmentParams,
    *,
    roll_color_result: dict | None = None,
    roll_color_frame: dict | None = None,
) -> NegativePreviewResult:
    corrected = apply_roll_color_to_linear_rgb(
        color_stage.color_linear_rgb,
        roll_result=roll_color_result,
        frame_plan=roll_color_frame,
        settings=adjustments.color_correction,
    )
    tone_active = adjustments.highlights != 0 or adjustments.shadows != 0
    if tone_active:
        tone_luminance = rgb_luminance(np.maximum(corrected, 0.0))
        tone_mid_anchor = estimate_tone_mid_anchor(tone_luminance)
        processed = apply_highlight_shadow_adjustments(
            corrected,
            adjustments,
            mid_anchor=tone_mid_anchor,
            luminance=tone_luminance,
        )
    else:
        tone_mid_anchor = 0.46
        processed = corrected
    color_balanced = processed
    processed = apply_saturation_adjustment(processed, adjustments)
    display_rgb8 = linear_to_srgb8(processed)

    return NegativePreviewResult(
        display_rgb8=display_rgb8,
        processed_linear_rgb=processed,
        color_balanced_linear_rgb=color_balanced,
        histogram=color_stage.histogram,
        auto_levels=color_stage.auto_levels,
        wb_gains=color_stage.wb_gains,
        lab_print_log_floors=color_stage.lab_print_log_floors,
        lab_print_log_ceils=color_stage.lab_print_log_ceils,
        mask_rgb=color_stage.mask_rgb,
        film_rect_preview=color_stage.film_rect_preview,
        tone_mid_anchor=tone_mid_anchor,
    )


def build_lab_print_export_linear(
    color_stage: LabPrintColorStage,
    adjustments: AdjustmentParams,
    *,
    roll_color_result: dict | None = None,
    roll_color_frame: dict | None = None,
    tone_mid_anchor: float | None = None,
    stage_timings: dict[str, float] | None = None,
) -> np.ndarray:
    stage_start = perf_counter()
    corrected = apply_roll_color_to_linear_rgb(
        color_stage.color_linear_rgb,
        roll_result=roll_color_result,
        frame_plan=roll_color_frame,
        settings=adjustments.color_correction,
        stage_timings=stage_timings,
    )
    if stage_timings is not None:
        stage_timings["Lab roll color"] = perf_counter() - stage_start

    stage_start = perf_counter()
    tone_active = adjustments.highlights != 0 or adjustments.shadows != 0
    tone_luminance = None
    if tone_active:
        tone_luminance = rgb_luminance(np.maximum(corrected, 0.0))
        if tone_mid_anchor is None:
            tone_mid_anchor = estimate_tone_mid_anchor(tone_luminance)
        processed = apply_highlight_shadow_adjustments(
            corrected,
            adjustments,
            mid_anchor=tone_mid_anchor,
            luminance=tone_luminance,
        )
    else:
        processed = corrected
    if stage_timings is not None:
        stage_timings["Lab tone modifier"] = perf_counter() - stage_start

    stage_start = perf_counter()
    processed = apply_saturation_adjustment(processed, adjustments)
    if stage_timings is not None:
        stage_timings["Lab saturation"] = perf_counter() - stage_start
    return processed.astype(np.float32, copy=False)


def sample_mask_rgb(
    image: np.ndarray,
    *,
    source_size: ImageSize,
    preview_size: ImageSize,
    mask_point: ImagePoint | None,
    point_radius: int = 12,
) -> np.ndarray:
    if mask_point is not None:
        point = scale_point(mask_point, source_size, preview_size)
        x0 = max(0, point.x - point_radius)
        y0 = max(0, point.y - point_radius)
        x1 = min(preview_size.width, point.x + point_radius + 1)
        y1 = min(preview_size.height, point.y + point_radius + 1)
        sample = image[y0:y1, x0:x1]
    else:
        raise PipelineError("No film base sample point is available.")

    if sample.size == 0:
        raise PipelineError("Film base sample area is empty. Pick the base again.")

    mask_rgb = np.median(sample.reshape(-1, 3), axis=0).astype(np.float32)
    return np.maximum(mask_rgb, np.array([1e-5, 1e-5, 1e-5], dtype=np.float32))


def sample_gain_rgb(
    gain: np.ndarray,
    *,
    source_size: ImageSize,
    preview_size: ImageSize,
    mask_point: ImagePoint,
    point_radius: int = 12,
) -> np.ndarray:
    point = scale_point(mask_point, source_size, preview_size)
    x0 = max(0, point.x - point_radius)
    y0 = max(0, point.y - point_radius)
    x1 = min(preview_size.width, point.x + point_radius + 1)
    y1 = min(preview_size.height, point.y + point_radius + 1)
    sample = gain[y0:y1, x0:x1]
    if sample.size == 0:
        return np.ones(3, dtype=np.float32)
    if sample.ndim == 2:
        value = float(np.median(sample.reshape(-1)))
        return np.array([value, value, value], dtype=np.float32)
    return np.median(sample.reshape(-1, 3), axis=0).astype(np.float32)


def normalize_log_bounds(
    linear_rgb: np.ndarray,
    *,
    percentile_clip: float = 0.00001,
    analysis_buffer: float = LOG_ANALYSIS_BUFFER,
) -> np.ndarray:
    normalized, _floors, _ceils = normalize_log_bounds_with_metadata(
        linear_rgb,
        percentile_clip=percentile_clip,
        analysis_buffer=analysis_buffer,
    )
    return normalized


def normalize_log_bounds_with_metadata(
    linear_rgb: np.ndarray,
    *,
    percentile_clip: float = 0.00001,
    analysis_buffer: float = LOG_ANALYSIS_BUFFER,
    lab_print_log_floors: np.ndarray | list[float] | None = None,
    lab_print_log_ceils: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe = np.clip(
        np.nan_to_num(linear_rgb, nan=LOG_MODE_EPSILON, posinf=1.0, neginf=LOG_MODE_EPSILON),
        LOG_MODE_EPSILON,
        1.0,
    )
    log_rgb = np.log10(safe).astype(np.float32, copy=False)
    if lab_print_log_floors is not None and lab_print_log_ceils is not None:
        floors = np.asarray(lab_print_log_floors, dtype=np.float32).reshape(3)
        ceils = np.asarray(lab_print_log_ceils, dtype=np.float32).reshape(3)
    else:
        sample = log_analysis_crop(log_rgb, analysis_buffer).reshape(-1, 3)
        if sample.size == 0:
            sample = log_rgb.reshape(-1, 3)

        stride = max(1, len(sample) // 800_000)
        sample = sample[::stride]
        clip = float(np.clip(percentile_clip, 0.00001, 1.0))
        floors = np.percentile(sample, clip, axis=0).astype(np.float32)
        ceils = np.percentile(sample, 100.0 - clip, axis=0).astype(np.float32)
    span = ceils - floors
    span = np.where(np.abs(span) < LOG_MODE_EPSILON, LOG_MODE_EPSILON, span).astype(np.float32)

    normalized = (log_rgb - floors.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    return (
        np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False),
        floors.astype(np.float32, copy=True),
        ceils.astype(np.float32, copy=True),
    )


def log_analysis_crop(image: np.ndarray, buffer_ratio: float) -> np.ndarray:
    if buffer_ratio <= 0.0:
        return image

    height, width = image.shape[:2]
    safe_buffer = float(np.clip(buffer_ratio, 0.0, 0.3))
    cut_y = int(height * safe_buffer)
    cut_x = int(width * safe_buffer)
    if cut_y * 2 >= height or cut_x * 2 >= width:
        return image
    return image[cut_y : height - cut_y, cut_x : width - cut_x]


def analysis_inset_crop(image: np.ndarray, inset: float) -> np.ndarray:
    if inset <= 0.0 or image.ndim < 2:
        return image

    height, width = image.shape[:2]
    safe_inset = float(np.clip(inset, 0.0, 0.30))
    cut_y = int(round(height * safe_inset))
    cut_x = int(round(width * safe_inset))
    if cut_y * 2 >= height or cut_x * 2 >= width:
        return image
    return image[cut_y : height - cut_y, cut_x : width - cut_x]


def analysis_inset_from_adjustments(adjustments: AdjustmentParams) -> float:
    return float(np.clip(adjustments.analysis_inset_percent / 100.0, 0.0, 0.20))


def radial_lens_correction_gain(
    shape: tuple[int, int],
    params: LensCorrectionParams,
) -> np.ndarray:
    height, width = shape
    if height <= 0 or width <= 0:
        return np.ones((height, width), dtype=np.float32)

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = (float(params.center_x) / 100.0) * max(1, width - 1)
    cy = (float(params.center_y) / 100.0) * max(1, height - 1)
    radius = max(0.05, float(params.radius) / 100.0) * max(width, height) * 0.5
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    distance = np.clip(distance, 0.0, 1.0)
    smoothness = max(0.25, float(params.smoothness) / 100.0)
    falloff = np.power(distance, smoothness, dtype=np.float32)
    strength = max(0.0, float(params.strength) / 100.0)
    max_gain = max(1.0, float(params.max_gain) / 100.0)
    gain = 1.0 + strength * falloff * (max_gain - 1.0)
    return np.clip(gain, 1.0, max_gain).astype(np.float32, copy=False)


def apply_lens_correction_gain(image: np.ndarray, gain: np.ndarray) -> np.ndarray:
    gain_float = gain.astype(np.float32, copy=False)
    if gain_float.ndim == 2:
        gain_float = gain_float[:, :, None]
    corrected = image.astype(np.float32, copy=False)
    np.multiply(corrected, gain_float, out=corrected)
    np.clip(corrected, 0.0, 1.0, out=corrected)
    return corrected


def apply_log_hd_print_curve(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    cmy_offsets: np.ndarray,
) -> np.ndarray:
    # The direct H&D response uses several exp/power passes per channel.
    # A per-channel LUT keeps preview/export visually identical to the
    # reference path while avoiding that cost on full-resolution exports.
    if LOG_PRINT_CURVE_ENGINE == LOG_PRINT_CURVE_DIRECT:
        return log_hd_print_response(normalized_log, adjustments, cmy_offsets=cmy_offsets)
    if LOG_PRINT_CURVE_ENGINE == LOG_PRINT_CURVE_LUT_DIRECT_16384:
        return log_hd_print_response_lut_direct(
            normalized_log,
            adjustments,
            cmy_offsets=cmy_offsets,
            lut_size=LOG_PRINT_CURVE_LUT_SIZES[LOG_PRINT_CURVE_LUT_DIRECT_16384],
        )
    return log_hd_print_response_lut(
        normalized_log,
        adjustments,
        cmy_offsets=cmy_offsets,
        lut_size=LOG_PRINT_CURVE_LUT_SIZES.get(LOG_PRINT_CURVE_ENGINE, 8192),
    )


def set_log_print_curve_engine(engine: str) -> None:
    global LOG_PRINT_CURVE_ENGINE
    if engine not in {
        LOG_PRINT_CURVE_LUT_4096,
        LOG_PRINT_CURVE_LUT_8192,
        LOG_PRINT_CURVE_LUT_DIRECT_16384,
        LOG_PRINT_CURVE_DIRECT,
    }:
        raise ValueError(f"Unknown print curve engine: {engine}")
    LOG_PRINT_CURVE_ENGINE = engine


def log_print_curve_engine() -> str:
    return LOG_PRINT_CURVE_ENGINE


def log_hd_print_response_lut_direct(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    cmy_offsets: np.ndarray,
    lut_size: int,
) -> np.ndarray:
    clipped = np.clip(normalized_log, 0.0, 1.0).astype(np.float32, copy=False)
    lut = log_hd_print_curve_lut(adjustments, cmy_offsets=cmy_offsets, lut_size=lut_size)
    out = np.empty_like(clipped, dtype=np.float32)
    for channel in range(3):
        index = np.rint(clipped[:, :, channel] * np.float32(lut_size - 1)).astype(np.int32)
        index = np.clip(index, 0, lut_size - 1)
        out[:, :, channel] = lut[channel][index]
    return out


def log_hd_print_response_lut(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    cmy_offsets: np.ndarray,
    lut_size: int,
) -> np.ndarray:
    clipped = np.clip(normalized_log, 0.0, 1.0).astype(np.float32, copy=False)
    lut = log_hd_print_curve_lut(adjustments, cmy_offsets=cmy_offsets, lut_size=lut_size)
    scaled = clipped * np.float32(lut_size - 1)
    index = np.floor(scaled).astype(np.int32)
    index = np.clip(index, 0, lut_size - 2)
    frac = scaled - index

    out = np.empty_like(clipped, dtype=np.float32)
    for channel in range(3):
        channel_index = index[:, :, channel]
        channel_frac = frac[:, :, channel]
        channel_lut = lut[channel]
        low = channel_lut[channel_index]
        high = channel_lut[channel_index + 1]
        out[:, :, channel] = low * (1.0 - channel_frac) + high * channel_frac
    return out


def log_hd_print_curve_lut(
    adjustments: AdjustmentParams,
    *,
    cmy_offsets: np.ndarray,
    lut_size: int,
) -> np.ndarray:
    offsets = np.asarray(cmy_offsets, dtype=np.float32).reshape(3)
    curve_params = log_print_curve_params(adjustments)
    # CMY offsets shift the log input independently for each channel, so they
    # are part of the LUT identity rather than a post-curve adjustment.
    key = (
        int(lut_size),
        adjustments.print_curve,
        int(adjustments.exposure),
        int(adjustments.contrast),
        bool(adjustments.soft_highlights),
        bool(adjustments.soft_shadows),
        bool(adjustments.print_curve_params.enabled),
        round(curve_params.density, 4),
        round(curve_params.grade, 4),
        round(curve_params.highlight_bias, 4),
        round(curve_params.highlight_width, 4),
        round(curve_params.shadow_bias, 4),
        round(curve_params.shadow_width, 4),
        tuple(round(float(value), 7) for value in offsets),
    )
    cached = _LOG_PRINT_CURVE_LUTS.get(key)
    if cached is not None:
        return cached

    samples = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
    if adjustments.print_curve in RATIONAL_FILMIC_PRINT_MODES:
        lut = log_rational_filmic_print_curve_lut(samples, offsets, adjustments.print_curve)
        _LOG_PRINT_CURVE_LUTS[key] = lut
        while len(_LOG_PRINT_CURVE_LUTS) > 32:
            oldest_key = next(iter(_LOG_PRINT_CURVE_LUTS))
            _LOG_PRINT_CURVE_LUTS.pop(oldest_key, None)
        return lut

    pivot = float(np.clip(1.0 - (0.01 + curve_params.density * LOG_DENSITY_MULTIPLIER), 0.02, 0.98))
    slope = float(np.clip(1.0 + curve_params.grade * LOG_GRADE_MULTIPLIER, 0.1, 16.0))
    toe = float(np.clip(0.20 if adjustments.soft_shadows else 0.0, -1.0, 1.0))
    shoulder = float(np.clip(0.20 if adjustments.soft_highlights else 0.0, -1.0, 1.0))

    lut = np.empty((3, lut_size), dtype=np.float32)
    for channel in range(3):
        lut[channel] = log_hd_print_response_1d(
            samples,
            pivot=pivot,
            slope=slope,
            toe=toe,
            shoulder=shoulder,
            cmy_offset=float(offsets[channel]),
            highlight_density_shift=curve_params.highlight_bias,
            highlight_width=curve_params.highlight_width,
            shadow_density_shift=curve_params.shadow_bias,
            shadow_width=curve_params.shadow_width,
        )

    _LOG_PRINT_CURVE_LUTS[key] = lut
    while len(_LOG_PRINT_CURVE_LUTS) > 32:
        oldest_key = next(iter(_LOG_PRINT_CURVE_LUTS))
        _LOG_PRINT_CURVE_LUTS.pop(oldest_key, None)
    return lut


def log_hd_print_response_1d(
    normalized_log: np.ndarray,
    *,
    pivot: float,
    slope: float,
    toe: float,
    shoulder: float,
    cmy_offset: float,
    highlight_density_shift: float = 0.0,
    highlight_width: float = 0.55,
    shadow_density_shift: float = 0.0,
    shadow_width: float = 0.55,
) -> np.ndarray:
    value = np.clip(normalized_log, 0.0, 1.0).astype(np.float32, copy=False) + np.float32(cmy_offset)
    diff = value - np.float32(pivot)

    toe_mask = logistic(LOG_TOE_WIDTH * (diff / max(1.0 - pivot, LOG_MODE_EPSILON) - 0.5))
    shoulder_mask = logistic(-LOG_SHOULDER_WIDTH * (diff / max(pivot, LOG_MODE_EPSILON) + 0.5))
    toe_transition = np.clip(toe_mask * (1.0 - toe_mask) * 4.0, 0.0, 1.0)
    shoulder_transition = np.clip(shoulder_mask * (1.0 - shoulder_mask) * 4.0, 0.0, 1.0)

    toe_lift_mask = toe_transition if toe > 0.0 else toe_mask
    shoulder_lift_mask = shoulder_transition if shoulder > 0.0 else shoulder_mask

    diff_adjusted = (
        diff
        - np.float32(toe) * toe_lift_mask * 0.28
        + np.float32(shoulder) * shoulder_lift_mask * 0.25
    )
    if highlight_density_shift != 0.0:
        highlight_knee = np.float32(np.clip(highlight_width, 0.05, 0.95))
        highlight_mask = smoothstep(0.0, highlight_knee, highlight_knee - np.clip(value, 0.0, 1.0))
        diff_adjusted = diff_adjusted + np.float32(highlight_density_shift) * highlight_mask
    if shadow_density_shift != 0.0:
        shadow_knee = np.float32(1.0 - np.clip(shadow_width, 0.05, 0.95))
        shadow_mask = smoothstep(shadow_knee, 1.0, np.clip(value, 0.0, 1.0))
        diff_adjusted = diff_adjusted + np.float32(shadow_density_shift) * shadow_mask
    slope_mod = np.clip(
        1.0
        - max(toe, 0.0) * toe_transition * 0.55
        - max(shoulder, 0.0) * shoulder_transition * 0.45
        - min(toe, 0.0) * toe_mask * 0.20
        - min(shoulder, 0.0) * shoulder_mask * 0.20,
        0.1,
        2.0,
    )
    print_density = LOG_D_MAX * logistic(np.float32(slope) * diff_adjusted * slope_mod)
    linear = np.power(10.0, -print_density)
    return np.clip(linear, 0.0, 1.0).astype(np.float32, copy=False)


def log_hd_print_response(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    cmy_offsets: np.ndarray,
) -> np.ndarray:
    if adjustments.print_curve in RATIONAL_FILMIC_PRINT_MODES:
        return log_rational_filmic_print_response(
            normalized_log,
            cmy_offsets=cmy_offsets,
            curve_mode=adjustments.print_curve,
        )

    curve_params = log_print_curve_params(adjustments)
    pivot_value = float(np.clip(1.0 - (0.01 + curve_params.density * LOG_DENSITY_MULTIPLIER), 0.02, 0.98))
    slope_value = float(np.clip(1.0 + curve_params.grade * LOG_GRADE_MULTIPLIER, 0.1, 16.0))

    pivot = np.array([pivot_value, pivot_value, pivot_value], dtype=np.float32).reshape(1, 1, 3)
    slope = np.array([slope_value, slope_value, slope_value], dtype=np.float32).reshape(1, 1, 3)
    offsets = cmy_offsets.astype(np.float32, copy=False).reshape(1, 1, 3)

    toe = float(np.clip(0.20 if adjustments.soft_shadows else 0.0, -1.0, 1.0))
    shoulder = float(np.clip(0.20 if adjustments.soft_highlights else 0.0, -1.0, 1.0))

    value = np.clip(normalized_log, 0.0, 1.0) + offsets
    diff = value - pivot

    toe_mask = logistic(LOG_TOE_WIDTH * (diff / np.maximum(1.0 - pivot, LOG_MODE_EPSILON) - 0.5))
    shoulder_mask = logistic(-LOG_SHOULDER_WIDTH * (diff / np.maximum(pivot, LOG_MODE_EPSILON) + 0.5))
    toe_transition = np.clip(toe_mask * (1.0 - toe_mask) * 4.0, 0.0, 1.0)
    shoulder_transition = np.clip(shoulder_mask * (1.0 - shoulder_mask) * 4.0, 0.0, 1.0)

    toe_lift_mask = toe_transition if toe > 0.0 else toe_mask
    shoulder_lift_mask = shoulder_transition if shoulder > 0.0 else shoulder_mask

    toe_density_offset = toe * toe_lift_mask * 0.28
    shoulder_density_offset = shoulder * shoulder_lift_mask * 0.25
    diff_adjusted = diff - toe_density_offset + shoulder_density_offset
    if curve_params.highlight_bias != 0.0:
        highlight_knee = np.float32(np.clip(curve_params.highlight_width, 0.05, 0.95))
        highlight_mask = smoothstep(0.0, highlight_knee, highlight_knee - np.clip(value, 0.0, 1.0))
        diff_adjusted = diff_adjusted + np.float32(curve_params.highlight_bias) * highlight_mask
    if curve_params.shadow_bias != 0.0:
        shadow_knee = np.float32(1.0 - np.clip(curve_params.shadow_width, 0.05, 0.95))
        shadow_mask = smoothstep(shadow_knee, 1.0, np.clip(value, 0.0, 1.0))
        diff_adjusted = diff_adjusted + np.float32(curve_params.shadow_bias) * shadow_mask

    slope_mod = np.clip(
        1.0
        - max(toe, 0.0) * toe_transition * 0.55
        - max(shoulder, 0.0) * shoulder_transition * 0.45
        - min(toe, 0.0) * toe_mask * 0.20
        - min(shoulder, 0.0) * shoulder_mask * 0.20,
        0.1,
        2.0,
    )
    print_density = LOG_D_MAX * logistic(slope * diff_adjusted * slope_mod)
    linear = np.power(10.0, -print_density)
    return np.clip(linear, 0.0, 1.0).astype(np.float32, copy=False)


def log_print_density_grade(adjustments: AdjustmentParams) -> tuple[float, float]:
    params = log_print_curve_params(adjustments)
    return params.density, params.grade


def log_print_curve_params(adjustments: AdjustmentParams) -> LogPrintCurveParams:
    custom = adjustments.print_curve_params
    if custom.enabled:
        return LogPrintCurveParams(
            density=float(np.clip(custom.density, 0.5, 1.5)),
            grade=float(np.clip(custom.grade, 1.0, 4.5)),
            highlight_bias=float(np.clip(custom.highlight_bias, -0.20, 0.30)),
            highlight_width=float(np.clip(custom.highlight_width, 0.20, 0.90)),
            shadow_bias=float(np.clip(custom.shadow_bias, -0.20, 0.30)),
            shadow_width=float(np.clip(custom.shadow_width, 0.20, 0.90)),
        )

    if adjustments.print_curve == PrintCurveMode.LINEAR.value:
        base_density = 0.82
        base_grade = 1.25
        highlight_bias = 0.0
    elif adjustments.print_curve in RATIONAL_FILMIC_PRINT_MODES:
        base_density = 1.0
        base_grade = 2.5
        highlight_bias = 0.0
    elif adjustments.print_curve == PrintCurveMode.SOFT.value:
        base_density = 1.0
        base_grade = 1.85
        highlight_bias = 0.0
    elif adjustments.print_curve == PrintCurveMode.STANDARD.value:
        base_density = 1.0
        base_grade = 3.0
        highlight_bias = 0.12
    elif adjustments.print_curve in {PrintCurveMode.CONTRAST.value, PrintCurveMode.CONTRAST_SHOULDER.value}:
        base_density = 1.06
        base_grade = 3.35
        highlight_bias = 0.12 if adjustments.print_curve == PrintCurveMode.CONTRAST_SHOULDER.value else 0.0
    else:
        base_density = 1.0
        base_grade = 2.5
        highlight_bias = 0.0

    density = base_density - adjustments.exposure / 100.0
    grade = base_grade + adjustments.contrast * 0.025
    return LogPrintCurveParams(
        density=float(np.clip(density, 0.05, 2.0)),
        grade=float(np.clip(grade, 0.1, 6.0)),
        highlight_bias=highlight_bias,
        highlight_width=0.55,
        shadow_bias=0.0,
        shadow_width=0.55,
    )


def log_rational_filmic_print_curve_lut(
    samples: np.ndarray,
    offsets: np.ndarray,
    curve_mode: str,
) -> np.ndarray:
    lut = np.empty((3, len(samples)), dtype=np.float32)
    for channel in range(3):
        positive = np.clip(1.0 - (samples + np.float32(offsets[channel])), 0.0, 1.0)
        lut[channel] = rational_filmic_curve_values(positive, curve_mode)
    return lut


def log_rational_filmic_print_response(
    normalized_log: np.ndarray,
    *,
    cmy_offsets: np.ndarray,
    curve_mode: str,
) -> np.ndarray:
    offsets = cmy_offsets.astype(np.float32, copy=False).reshape(1, 1, 3)
    positive = np.clip(1.0 - (np.clip(normalized_log, 0.0, 1.0) + offsets), 0.0, 1.0)
    return rational_filmic_curve_values(positive, curve_mode)


def rational_filmic_curve_values(values: np.ndarray, curve_mode: str) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    if curve_mode == PrintCurveMode.FILMIC_HABLE.value:
        y = hable_filmic_curve_raw(x)
        white = float(hable_filmic_curve_raw(np.array([1.0], dtype=np.float32))[0])
    elif curve_mode == PrintCurveMode.FILMIC_ACES.value:
        y = aces_filmic_curve_raw(x)
        white = float(aces_filmic_curve_raw(np.array([1.0], dtype=np.float32))[0])
    else:
        return x.astype(np.float32, copy=False)
    return np.clip(y / max(white, 1.0e-6), 0.0, 1.0).astype(np.float32, copy=False)


def hable_filmic_curve_raw(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    a = np.float32(0.15)
    b = np.float32(0.50)
    c = np.float32(0.10)
    d = np.float32(0.20)
    e = np.float32(0.02)
    f = np.float32(0.30)
    numerator = x * (a * x + c * b) + d * e
    denominator = x * (a * x + b) + d * f
    return (numerator / np.maximum(denominator, np.float32(1.0e-6))) - e / f


def aces_filmic_curve_raw(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    a = np.float32(2.51)
    b = np.float32(0.03)
    c = np.float32(2.43)
    d = np.float32(0.59)
    e = np.float32(0.14)
    return (x * (a * x + b)) / np.maximum(x * (c * x + d) + e, np.float32(1.0e-6))


def estimate_lab_print_auto_cmy_offsets(
    normalized_log: np.ndarray,
    *,
    strength: float = LAB_PRINT_AUTO_WB_STRENGTH,
) -> np.ndarray:
    sample = select_lab_print_log_wb_sample(normalized_log)
    if sample.size == 0:
        return np.zeros(3, dtype=np.float32)

    median_log = np.median(sample, axis=0).astype(np.float32)
    # Use a zero-mean CMY correction instead of pinning red/cyan to zero.
    # This lets all channels participate while preserving the average print
    # exposure more than a one-channel reference offset would.
    offsets = np.mean(median_log, dtype=np.float32) - median_log
    offsets *= float(np.clip(strength, 0.0, 1.0))
    return np.clip(offsets, -LAB_PRINT_AUTO_WB_MAX_OFFSET, LAB_PRINT_AUTO_WB_MAX_OFFSET).astype(np.float32, copy=False)


def manual_printer_balance_offsets(axis: BalanceAxis) -> np.ndarray:
    rgb_direction = np.array(
        [axis.red_cyan, axis.green_magenta, axis.blue_yellow],
        dtype=np.float32,
    )
    offsets = -rgb_direction * np.float32(LAB_PRINT_MANUAL_CMY_OFFSET_SCALE)
    return np.clip(
        offsets,
        -LAB_PRINT_MANUAL_CMY_MAX_OFFSET,
        LAB_PRINT_MANUAL_CMY_MAX_OFFSET,
    ).astype(np.float32, copy=False)


def suggest_printer_balance_from_log_sample(
    normalized_log: np.ndarray,
    point: ImagePoint,
    *,
    base_cmy_offsets: np.ndarray | list[float] | None = None,
    radius: int = 14,
    strength: float = 1.0,
) -> tuple[BalanceAxis, np.ndarray, np.ndarray]:
    sample = sample_log_at_point(normalized_log, point, radius=radius)
    median_log = np.median(sample, axis=0).astype(np.float32)
    offset_delta = (np.mean(median_log, dtype=np.float32) - median_log) * float(np.clip(strength, 0.0, 1.0))
    offset_delta = np.clip(
        offset_delta,
        -LAB_PRINT_MANUAL_CMY_MAX_OFFSET,
        LAB_PRINT_MANUAL_CMY_MAX_OFFSET,
    ).astype(np.float32, copy=False)
    base = (
        np.asarray(base_cmy_offsets, dtype=np.float32).reshape(3)
        if base_cmy_offsets is not None
        else np.zeros(3, dtype=np.float32)
    )
    manual_offsets = np.clip(
        offset_delta - base,
        -LAB_PRINT_MANUAL_CMY_MAX_OFFSET,
        LAB_PRINT_MANUAL_CMY_MAX_OFFSET,
    )
    slider_values = np.rint(-manual_offsets / LAB_PRINT_MANUAL_CMY_OFFSET_SCALE).astype(np.int32)

    return (
        BalanceAxis(
            red_cyan=clamp_balance_value(int(slider_values[0])),
            green_magenta=clamp_balance_value(int(slider_values[1])),
            blue_yellow=clamp_balance_value(int(slider_values[2])),
        ),
        median_log,
        offset_delta.astype(np.float32, copy=False),
    )


def sample_log_at_point(image: np.ndarray, point: ImagePoint, *, radius: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise PipelineError("Printer balance picker needs a log RGB image.")

    height, width, _channels = image.shape
    if width <= 0 or height <= 0:
        raise PipelineError("Printer balance sample image is empty.")

    x = min(max(0, int(point.x)), width - 1)
    y = min(max(0, int(point.y)), height - 1)
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(width, x + radius + 1)
    y1 = min(height, y + radius + 1)
    sample = np.clip(image[y0:y1, x0:x1].reshape(-1, 3), 0.0, 1.0)
    if sample.size == 0:
        raise PipelineError("Printer balance sample area is empty.")
    return sample.astype(np.float32, copy=False)


def effective_lab_print_cmy_offsets(
    auto_cmy_offsets: np.ndarray | list[float] | None,
    adjustments: AdjustmentParams,
) -> np.ndarray:
    base = (
        np.asarray(auto_cmy_offsets, dtype=np.float32).reshape(3)
        if auto_cmy_offsets is not None
        else np.zeros(3, dtype=np.float32)
    )
    manual = manual_printer_balance_offsets(adjustments.printer_balance)
    limit = LAB_PRINT_AUTO_WB_MAX_OFFSET + LAB_PRINT_MANUAL_CMY_MAX_OFFSET
    return np.clip(base + manual, -limit, limit).astype(np.float32, copy=False)


def select_lab_print_log_wb_sample(normalized_log: np.ndarray) -> np.ndarray:
    clipped = np.clip(normalized_log, 0.0, 1.0).astype(np.float32, copy=False)
    positive = 1.0 - clipped
    luminance = rgb_luminance(positive)
    flat_log = clipped.reshape(-1, 3)
    flat_luminance = luminance.reshape(-1)
    stride = max(1, len(flat_log) // 800_000)
    flat_log = flat_log[::stride]
    flat_luminance = flat_luminance[::stride]

    low = float(np.percentile(flat_luminance, 22.0))
    high = float(np.percentile(flat_luminance, 78.0))
    mid_mask = (flat_luminance >= low) & (flat_luminance <= high)
    sample = flat_log[mid_mask]
    if len(sample) < 128:
        sample = flat_log

    in_range = np.all((sample > 0.04) & (sample < 0.96), axis=1)
    if int(np.count_nonzero(in_range)) >= 128:
        sample = sample[in_range]

    chroma = sample.max(axis=1) - sample.min(axis=1)
    chroma_limit = float(np.percentile(chroma, 45.0))
    neutralish = chroma <= chroma_limit
    if int(np.count_nonzero(neutralish)) >= 128:
        sample = sample[neutralish]

    return sample


def select_log_positive_wb_sample(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    luminance = rgb_luminance(clipped)
    flat_rgb = clipped.reshape(-1, 3)
    flat_luminance = luminance.reshape(-1)
    stride = max(1, len(flat_rgb) // 800_000)
    flat_rgb = flat_rgb[::stride]
    flat_luminance = flat_luminance[::stride]

    low = float(np.percentile(flat_luminance, 20.0))
    high = float(np.percentile(flat_luminance, 80.0))
    mid_mask = (flat_luminance >= low) & (flat_luminance <= high)
    sample = flat_rgb[mid_mask]
    if len(sample) < 128:
        sample = flat_rgb

    channel_mask = np.all((sample > 0.01) & (sample < 0.99), axis=1)
    if int(np.count_nonzero(channel_mask)) >= 128:
        sample = sample[channel_mask]

    return sample


def apply_log_color_separation(
    image: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    if strength <= 0.0:
        return image

    density = -np.log10(np.clip(image, LOG_MODE_EPSILON, 1.0))
    identity = np.eye(3, dtype=np.float32)
    matrix = identity * (1.0 - strength) + LOG_COLOR_SEPARATION_MATRIX * strength
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(row_sums, LOG_MODE_EPSILON)
    corrected_density = np.einsum("hwc,kc->hwk", density.astype(np.float32, copy=False), matrix)
    corrected_density = np.maximum(corrected_density, 0.0)
    separated = np.power(10.0, -corrected_density)
    return np.clip(separated, 0.0, 1.0).astype(np.float32, copy=False)


def logistic(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)


def channel_percentile_stretch(
    image: np.ndarray,
    *,
    low_percentile: float = 0.1,
    high_percentile: float = 99.8,
) -> np.ndarray:
    stretched = np.empty_like(image, dtype=np.float32)
    pixels = image.reshape(-1, 3)
    stride = max(1, len(pixels) // 800_000)
    sample = pixels[::stride]

    for channel in range(3):
        low = float(np.percentile(sample[:, channel], low_percentile))
        high = float(np.percentile(sample[:, channel], high_percentile))
        if high <= low + 1e-6:
            stretched[:, :, channel] = image[:, :, channel]
        else:
            stretched[:, :, channel] = (image[:, :, channel] - low) / (high - low)

    return np.clip(stretched, 0.0, 1.0)


def apply_adjustments(image: np.ndarray, adjustments: AdjustmentParams) -> np.ndarray:
    adjusted = apply_unit_levels(image, adjustments, clip=False)
    adjusted = apply_exposure(adjusted, adjustments)
    adjusted = apply_highlight_shadow_adjustments(adjusted, adjustments)
    adjusted = apply_soft_tone_adjustments(adjusted, adjustments)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return apply_contrast(adjusted, adjustments)


def apply_unit_levels(
    image: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    clip: bool = True,
    soft_black: bool = False,
) -> np.ndarray:
    adjusted = image.astype(np.float32, copy=True)
    black = adjustments.black_point / 100.0
    mid = adjustments.mid_point / 100.0
    white = adjustments.white_point / 100.0
    if white > black + 1e-5:
        adjusted = (adjusted - black) / (white - black)

        mid_norm = (mid - black) / (white - black)
        mid_norm = float(np.clip(mid_norm, 0.01, 0.99))
        gamma = np.log(0.5) / np.log(mid_norm)
        gamma = float(np.clip(gamma, 0.2, 8.0))
        if soft_black:
            adjusted = apply_soft_black_levels(adjusted, gamma)
        else:
            adjusted = np.power(np.maximum(adjusted, 0.0), gamma)

    if clip:
        return np.clip(adjusted, 0.0, 1.0)
    return np.nan_to_num(adjusted, nan=0.0, posinf=1.0e12, neginf=0.0).astype(np.float32, copy=False)


def apply_soft_black_levels(normalized: np.ndarray, gamma: float) -> np.ndarray:
    values = normalized.astype(np.float32, copy=False)
    positive = np.maximum(values, 0.0)
    mapped = LEVELS_SOFT_BLACK_FLOOR + np.power(positive, gamma) * (1.0 - LEVELS_SOFT_BLACK_FLOOR)

    below_black = values < 0.0
    if np.any(below_black):
        toe = LEVELS_SOFT_BLACK_FLOOR * np.exp(values / LEVELS_SOFT_BLACK_WIDTH)
        mapped = np.where(below_black, toe, mapped)

    return mapped.astype(np.float32, copy=False)


def apply_output_color_transform(
    image: np.ndarray,
    camera_to_srgb_matrix: np.ndarray | None,
    strength_percent: int,
) -> np.ndarray:
    strength = float(np.clip(strength_percent / 100.0, 0.0, 1.0))
    if camera_to_srgb_matrix is None or strength <= 0.0:
        return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    matrix = np.asarray(camera_to_srgb_matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    matrix = np.eye(3, dtype=np.float32) * (1.0 - strength) + matrix * strength
    transformed = image.astype(np.float32, copy=False) @ matrix
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=1.0e6, neginf=-1.0e6).astype(np.float32, copy=False)
    return compress_rgb_gamut(transformed)


def compress_rgb_gamut(image: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(image.astype(np.float32, copy=False), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    if rgb.size == 0:
        return rgb
    min_channel = np.min(rgb, axis=2)
    max_channel = np.max(rgb, axis=2)
    out_of_gamut = (min_channel < 0.0) | (max_channel > 1.0)
    if not bool(np.any(out_of_gamut)):
        return rgb.astype(np.float32, copy=False)

    luminance = np.clip(rgb_luminance(rgb), GAMUT_LOWER_MARGIN, GAMUT_UPPER_MARGIN).astype(np.float32, copy=False)
    neutral = luminance[:, :, None]
    chroma = rgb - neutral

    lower_limits = np.divide(
        neutral - GAMUT_LOWER_MARGIN,
        -chroma,
        out=np.full_like(rgb, np.inf, dtype=np.float32),
        where=chroma < -GAMUT_EPSILON,
    )
    upper_limits = np.divide(
        GAMUT_UPPER_MARGIN - neutral,
        chroma,
        out=np.full_like(rgb, np.inf, dtype=np.float32),
        where=chroma > GAMUT_EPSILON,
    )
    scale = np.minimum(lower_limits, upper_limits).min(axis=2)
    scale = np.clip(scale, 0.0, 1.0).astype(np.float32, copy=False)

    compressed = neutral + chroma * scale[:, :, None]
    shadow_weight = (1.0 - smoothstep(GAMUT_SHADOW_CHROMA_LOW, GAMUT_SHADOW_CHROMA_HIGH, luminance))
    shadow_weight = shadow_weight * out_of_gamut.astype(np.float32)
    if bool(np.any(shadow_weight > 0.0)):
        compressed_luminance = np.clip(
            rgb_luminance(compressed),
            GAMUT_LOWER_MARGIN,
            GAMUT_UPPER_MARGIN,
        ).astype(np.float32, copy=False)
        compressed_neutral = compressed_luminance[:, :, None]
        shadow_scale = 1.0 - GAMUT_SHADOW_CHROMA_STRENGTH * shadow_weight[:, :, None]
        compressed = compressed_neutral + (compressed - compressed_neutral) * shadow_scale

    return np.clip(compressed, 0.0, 1.0).astype(np.float32, copy=False)


def apply_print_curve(image: np.ndarray, curve_mode: str) -> np.ndarray:
    if curve_mode == PrintCurveMode.LINEAR.value:
        return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    return apply_hue_preserving_curve(image, curve_mode)


def print_curve_values(values: np.ndarray, curve_mode: str) -> np.ndarray:
    if curve_mode == PrintCurveMode.LINEAR.value:
        return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    if curve_mode in RATIONAL_FILMIC_PRINT_MODES:
        return rational_filmic_curve_values(values, curve_mode)
    return apply_filmic_curve_lut(values, curve_mode)


def inverse_print_curve_value(target: float, curve_mode: str) -> float:
    target = float(np.clip(target, 0.0, 1.0))
    low = 0.0
    high = 1.0
    for _index in range(24):
        mid = (low + high) * 0.5
        value = float(print_curve_values(np.array([mid], dtype=np.float32), curve_mode)[0])
        if value < target:
            low = mid
        else:
            high = mid
    return (low + high) * 0.5


def filmic_curve_points(curve_mode: str) -> tuple[tuple[float, float], tuple[float, float]]:
    return FILMIC_CURVE_PRESETS.get(curve_mode, FILMIC_CURVE_PRESETS[PrintCurveMode.STANDARD.value])


def filmic_curve_lut(curve_mode: str) -> np.ndarray:
    if curve_mode not in _FILMIC_CURVE_LUTS:
        p1, p2 = filmic_curve_points(curve_mode)
        samples = np.linspace(0.0, 1.0, FILMIC_LUT_SIZE, dtype=np.float32)
        _FILMIC_CURVE_LUTS[curve_mode] = solve_filmic_bezier_curve(samples, p1=p1, p2=p2)
    return _FILMIC_CURVE_LUTS[curve_mode]


def apply_filmic_curve_lut(values: np.ndarray, curve_mode: str) -> np.ndarray:
    x = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    lut = filmic_curve_lut(curve_mode)
    scaled = x * np.float32(FILMIC_LUT_SIZE - 1)
    index = np.floor(scaled).astype(np.int32)
    index = np.clip(index, 0, FILMIC_LUT_SIZE - 2)
    frac = scaled - index
    curved = lut[index] * (1.0 - frac) + lut[index + 1] * frac
    return curved.astype(np.float32, copy=False)


def solve_filmic_bezier_curve(
    values: np.ndarray,
    *,
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> np.ndarray:
    x = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    t = x.astype(np.float32, copy=True)
    x1, y1 = p1
    x2, y2 = p2

    for _iteration in range(6):
        one_minus_t = 1.0 - t
        curve_x = (
            3.0 * one_minus_t * one_minus_t * t * x1
            + 3.0 * one_minus_t * t * t * x2
            + t * t * t
        )
        deriv_x = (
            3.0 * one_minus_t * one_minus_t * x1
            + 6.0 * one_minus_t * t * (x2 - x1)
            + 3.0 * t * t * (1.0 - x2)
        )
        t = np.clip(t - (curve_x - x) / np.maximum(deriv_x, 1e-5), 0.0, 1.0)

    one_minus_t = 1.0 - t
    curve_y = (
        3.0 * one_minus_t * one_minus_t * t * y1
        + 3.0 * one_minus_t * t * t * y2
        + t * t * t
    )
    return np.clip(curve_y, 0.0, 1.0).astype(np.float32, copy=False)


def apply_hue_preserving_curve(image: np.ndarray, curve_mode: str) -> np.ndarray:
    rgb = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    channel_r = rgb[:, :, 0]
    channel_g = rgb[:, :, 1]
    channel_b = rgb[:, :, 2]
    lo_2d = cv2.min(cv2.min(channel_r, channel_g), channel_b)
    hi_2d = cv2.max(cv2.max(channel_r, channel_g), channel_b)
    span_2d = hi_2d - lo_2d

    lo_curved_2d = print_curve_values(lo_2d, curve_mode)
    hi_curved_2d = print_curve_values(hi_2d, curve_mode)
    scale_2d = np.divide(
        hi_curved_2d - lo_curved_2d,
        span_2d,
        out=np.zeros_like(span_2d),
        where=span_2d > 1e-6,
    )

    lo = lo_2d[:, :, None]
    lo_curved = lo_curved_2d[:, :, None]
    scale = scale_2d[:, :, None]
    curved = lo_curved + (rgb - lo) * scale

    return np.clip(curved, 0.0, 1.0).astype(np.float32, copy=False)


def apply_print_s_curve(values: np.ndarray, *, contrast: float = 1.85, strength: float = 0.58) -> np.ndarray:
    x = np.clip(values, 0.0, 1.0)
    y = 1.0 / (1.0 + np.exp(-contrast * (x - 0.5)))
    low = 1.0 / (1.0 + np.exp(contrast * 0.5))
    high = 1.0 / (1.0 + np.exp(-contrast * 0.5))
    curved = np.clip((y - low) / max(high - low, 1e-5), 0.0, 1.0)
    return (x * (1.0 - strength) + curved * strength).astype(np.float32, copy=False)


def apply_exposure(image: np.ndarray, adjustments: AdjustmentParams) -> np.ndarray:
    adjusted = np.maximum(image, 0.0).astype(np.float32, copy=True)
    exposure_multiplier = 2.0 ** (adjustments.exposure / 100.0)
    return (adjusted * exposure_multiplier).astype(np.float32, copy=False)


def apply_contrast(image: np.ndarray, adjustments: AdjustmentParams) -> np.ndarray:
    adjusted = np.clip(image, 0.0, 1.0).astype(np.float32, copy=True)
    contrast = 1.0 + adjustments.contrast / 100.0
    adjusted = (adjusted - 0.5) * contrast + 0.5
    return np.clip(adjusted, 0.0, 1.0)


def apply_saturation_adjustment(image: np.ndarray, adjustments: AdjustmentParams) -> np.ndarray:
    if adjustments.saturation == 0:
        return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    amount = float(np.clip(adjustments.saturation / 100.0, -1.0, 1.0))
    factor = 1.0 + amount if amount < 0.0 else 1.0 + amount * 1.35
    clipped = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    luminance = rgb_luminance(clipped)[:, :, None]
    saturated = luminance + (clipped - luminance) * factor
    return compress_rgb_gamut(saturated)


def apply_highlight_shadow_adjustments(
    image: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    mid_anchor: float | None = None,
    luminance: np.ndarray | None = None,
) -> np.ndarray:
    if adjustments.highlights == 0 and adjustments.shadows == 0:
        return image

    adjusted = np.maximum(image.astype(np.float32, copy=False), 0.0)
    if luminance is None:
        luminance = rgb_luminance(adjusted)
    luminance = np.maximum(luminance.astype(np.float32, copy=False), 0.0)
    if mid_anchor is None:
        mid_anchor = estimate_tone_mid_anchor(luminance)
    tone_lut = highlight_shadow_tone_lut(adjustments, mid_anchor=mid_anchor)
    highlight_recovery = max(0.0, float(-adjustments.highlights) / 100.0)
    headroom_slope = 1.0 if highlight_recovery <= 0.0 else max(0.18, 0.55 * (1.0 - highlight_recovery))
    target_luminance = apply_tone_lut_to_luminance(
        luminance,
        tone_lut,
        headroom_slope=headroom_slope,
    )
    ratio = np.divide(
        target_luminance,
        luminance,
        out=np.ones_like(target_luminance, dtype=np.float32),
        where=luminance > 1e-5,
    )
    if adjustments.shadows > 0:
        ratio = np.minimum(ratio, 1.0 + 2.8 * float(np.clip(adjustments.shadows / 100.0, 0.0, 1.0)))
    processed = adjusted * ratio[:, :, None]
    return np.nan_to_num(processed, nan=0.0, posinf=1.0e6, neginf=0.0).astype(np.float32, copy=False)


def highlight_shadow_tone_lut(
    adjustments: AdjustmentParams,
    *,
    mid_anchor: float = 0.46,
    lut_size: int = TONE_MODIFIER_LUT_SIZE,
) -> np.ndarray:
    mid_anchor = float(np.clip(mid_anchor, TONE_MID_ANCHOR_MIN, TONE_MID_ANCHOR_MAX))
    key = (
        int(lut_size),
        int(adjustments.highlights),
        int(adjustments.shadows),
        int(round(mid_anchor * 1000.0)),
    )
    cached = _TONE_MODIFIER_LUTS.get(key)
    if cached is not None:
        return cached

    x = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
    y = x.copy()
    shadow_controls, highlight_controls = highlight_shadow_segment_control_points(adjustments, mid_anchor)

    if shadow_controls is not None:
        control_x, control_y = shadow_controls
        shadow_mask = x <= mid_anchor
        y[shadow_mask] = fritsch_carlson_values(control_x, control_y, x[shadow_mask])

    if highlight_controls is not None:
        control_x, control_y = highlight_controls
        highlight_mask = x >= mid_anchor
        y[highlight_mask] = fritsch_carlson_values(control_x, control_y, x[highlight_mask])

    y = np.maximum.accumulate(np.clip(y, 0.0, 1.0)).astype(np.float32, copy=False)
    y[0] = 0.0
    y[-1] = 1.0
    _TONE_MODIFIER_LUTS[key] = y
    while len(_TONE_MODIFIER_LUTS) > 48:
        oldest_key = next(iter(_TONE_MODIFIER_LUTS))
        _TONE_MODIFIER_LUTS.pop(oldest_key, None)
    return y


def highlight_shadow_segment_control_points(
    adjustments: AdjustmentParams,
    mid_anchor: float,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, tuple[np.ndarray, np.ndarray] | None]:
    mid_anchor = float(np.clip(mid_anchor, TONE_MID_ANCHOR_MIN, TONE_MID_ANCHOR_MAX))
    shadow_controls: tuple[np.ndarray, np.ndarray] | None = None
    highlight_controls: tuple[np.ndarray, np.ndarray] | None = None

    shadow_amount = float(np.clip(adjustments.shadows / 100.0, -1.0, 1.0))
    if shadow_amount != 0.0:
        amount = abs(shadow_amount)
        shadow_x = mid_anchor * 0.62
        shadow_delta = mid_anchor * 0.24 * amount
        if shadow_amount > 0.0:
            shadow_y = min(mid_anchor - 1e-4, shadow_x + shadow_delta)
        else:
            shadow_y = max(1e-5, shadow_x - min(shadow_delta, shadow_x * 0.92))
        shadow_controls = (
            np.array([0.0, shadow_x, mid_anchor], dtype=np.float32),
            np.array([0.0, shadow_y, mid_anchor], dtype=np.float32),
        )

    highlight_amount = float(np.clip(adjustments.highlights / 100.0, -1.0, 1.0))
    if highlight_amount != 0.0:
        amount = abs(highlight_amount)
        highlight_span = max(1.0 - mid_anchor, 1e-5)
        highlight_x = mid_anchor + highlight_span * 0.38
        highlight_delta = highlight_span * 0.27 * amount
        if highlight_amount > 0.0:
            highlight_y = min(1.0 - 1e-5, highlight_x + highlight_delta)
        else:
            highlight_y = max(mid_anchor + 1e-4, highlight_x - highlight_delta)
        highlight_controls = (
            np.array([mid_anchor, highlight_x, 1.0], dtype=np.float32),
            np.array([mid_anchor, highlight_y, 1.0], dtype=np.float32),
        )

    return shadow_controls, highlight_controls


def highlight_shadow_control_points(
    adjustments: AdjustmentParams,
    mid_anchor: float,
) -> tuple[np.ndarray, np.ndarray]:
    mid_anchor = float(np.clip(mid_anchor, TONE_MID_ANCHOR_MIN, TONE_MID_ANCHOR_MAX))
    shadow_controls, highlight_controls = highlight_shadow_segment_control_points(adjustments, mid_anchor)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    if shadow_controls is not None:
        points.append((float(shadow_controls[0][1]), float(shadow_controls[1][1])))

    points.append((mid_anchor, mid_anchor))
    if highlight_controls is not None:
        points.append((float(highlight_controls[0][1]), float(highlight_controls[1][1])))

    points.append((1.0, 1.0))
    points.sort(key=lambda item: item[0])
    control_x = np.array([point[0] for point in points], dtype=np.float32)
    control_y = np.array([point[1] for point in points], dtype=np.float32)
    return control_x, control_y


def fritsch_carlson_values(
    control_x: np.ndarray,
    control_y: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    x = np.asarray(control_x, dtype=np.float32)
    y = np.asarray(control_y, dtype=np.float32)
    if x.size < 2 or y.size != x.size:
        return np.asarray(samples, dtype=np.float32)

    dx = np.diff(x)
    dy = np.diff(y)
    valid = dx > 1e-6
    if not bool(np.all(valid)):
        return np.asarray(samples, dtype=np.float32)

    slopes = dy / dx
    tangents = np.empty_like(y, dtype=np.float32)
    tangents[0] = slopes[0]
    tangents[-1] = slopes[-1]
    if y.size > 2:
        tangents[1:-1] = (slopes[:-1] + slopes[1:]) * 0.5

    for index, slope in enumerate(slopes):
        if slope <= 1e-8:
            tangents[index] = 0.0
            tangents[index + 1] = 0.0
            continue
        tangents[index] = max(0.0, float(tangents[index]))
        tangents[index + 1] = max(0.0, float(tangents[index + 1]))
        alpha = tangents[index] / slope
        beta = tangents[index + 1] / slope
        radius = alpha * alpha + beta * beta
        if radius > 9.0:
            scale = 3.0 / np.sqrt(radius)
            tangents[index] = scale * alpha * slope
            tangents[index + 1] = scale * beta * slope

    samples = np.asarray(samples, dtype=np.float32)
    indices = np.searchsorted(x, samples, side="right") - 1
    indices = np.clip(indices, 0, x.size - 2)

    x0 = x[indices]
    x1 = x[indices + 1]
    y0 = y[indices]
    y1 = y[indices + 1]
    m0 = tangents[indices]
    m1 = tangents[indices + 1]
    h = x1 - x0
    t = np.divide(samples - x0, h, out=np.zeros_like(samples), where=h > 1e-6)
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    values = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    return values.astype(np.float32, copy=False)


def estimate_tone_mid_anchor(luminance: np.ndarray) -> float:
    flat = np.asarray(luminance, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.46
    stride = max(1, flat.size // TONE_MID_ANCHOR_SAMPLE_LIMIT)
    sample = flat[::stride]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.46
    mid = float(np.percentile(sample, TONE_MID_ANCHOR_PERCENTILE))
    return float(np.clip(mid, TONE_MID_ANCHOR_MIN, TONE_MID_ANCHOR_MAX))


def apply_tone_lut_to_luminance(
    luminance: np.ndarray,
    lut: np.ndarray,
    *,
    headroom_slope: float = 1.0,
) -> np.ndarray:
    clipped = np.clip(luminance, 0.0, 1.0).astype(np.float32, copy=False)
    lut_size = int(len(lut))
    index = np.rint(clipped * np.float32(lut_size - 1)).astype(np.int32)
    index = np.clip(index, 0, lut_size - 1)
    target = lut[index]
    # Above display white, preserve headroom for normal edits but compress that
    # headroom when the highlight slider is pulled down for recovery.
    target = target + np.maximum(luminance - clipped, 0.0) * float(np.clip(headroom_slope, 0.0, 1.0))
    return target.astype(np.float32, copy=False)


def apply_soft_tone_adjustments(image: np.ndarray, adjustments: AdjustmentParams) -> np.ndarray:
    if not adjustments.soft_highlights and not adjustments.soft_shadows:
        return image

    adjusted = np.maximum(image.astype(np.float32, copy=True), 0.0)
    luminance = np.maximum(rgb_luminance(adjusted), 0.0)
    target_luminance = luminance.copy()

    if adjustments.soft_highlights:
        pivot = 0.72
        highlight_sample = luminance[luminance > pivot]
        if highlight_sample.size:
            upper = float(np.percentile(highlight_sample, 99.7))
        else:
            upper = 1.0
        upper = max(1.0, upper, pivot + 1e-4)
        span = upper - pivot
        weight = smoothstep(0.62, min(upper, 1.0), luminance)
        normalized = np.maximum(target_luminance - pivot, 0.0) / span
        shoulder = pivot + (1.0 - pivot) * (1.0 - np.exp(-2.4 * normalized)) / (1.0 - np.exp(-2.4))
        shoulder = np.minimum(shoulder, 1.0)
        target_luminance = target_luminance * (1.0 - 0.42 * weight) + shoulder * (0.42 * weight)

    shadow_floor = 0.018
    shadow_weight = np.zeros_like(luminance, dtype=np.float32)
    if adjustments.soft_shadows:
        shadow_weight = 1.0 - smoothstep(0.0, 0.30, luminance)
        toe = shadow_floor + target_luminance * 0.76
        mix = 0.46 * shadow_weight
        target_luminance = target_luminance * (1.0 - mix) + toe * mix

    ratio = target_luminance / np.maximum(luminance, 1e-5)
    softened = adjusted * ratio[:, :, None]
    if adjustments.soft_shadows:
        floor_weight = (1.0 - smoothstep(0.0, 0.20, luminance))[:, :, None]
        softened = softened * (1.0 - 0.18 * floor_weight) + shadow_floor * 0.65 * floor_weight

    return softened.astype(np.float32, copy=False)


def apply_auto_white_balance(
    image: np.ndarray,
    *,
    strength: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(image, 0.0, 1.0)
    sample = select_auto_wb_sample(clipped)
    if sample.size == 0:
        return clipped.astype(np.float32, copy=True), np.array([1.0, 1.0, 1.0], dtype=np.float32)

    median_rgb = np.median(sample, axis=0).astype(np.float32)
    median_rgb = np.maximum(median_rgb, np.array([1e-5, 1e-5, 1e-5], dtype=np.float32))
    target_gray = float(np.mean(median_rgb))
    raw_gains = target_gray / median_rgb
    raw_gains = np.clip(raw_gains, 0.5, 2.0)
    gains = 1.0 + (raw_gains - 1.0) * strength
    balanced = np.clip(clipped * gains.reshape(1, 1, 3), 0.0, 1.0)
    return balanced.astype(np.float32, copy=False), gains.astype(np.float32)


def select_auto_wb_sample(image: np.ndarray) -> np.ndarray:
    luminance = rgb_luminance(image)
    flat_rgb = image.reshape(-1, 3)
    flat_luminance = luminance.reshape(-1)
    stride = max(1, len(flat_rgb) // 800_000)
    flat_rgb = flat_rgb[::stride]
    flat_luminance = flat_luminance[::stride]

    low = float(np.percentile(flat_luminance, 12.0))
    high = float(np.percentile(flat_luminance, 88.0))
    mid_mask = (flat_luminance >= low) & (flat_luminance <= high)
    sample = flat_rgb[mid_mask]
    if len(sample) < 128:
        sample = flat_rgb

    channel_mask = np.all((sample > 0.02) & (sample < 0.98), axis=1)
    if int(np.count_nonzero(channel_mask)) >= 128:
        sample = sample[channel_mask]

    chroma = sample.max(axis=1) - sample.min(axis=1)
    chroma_limit = float(np.percentile(chroma, 55.0))
    neutral_mask = chroma <= chroma_limit
    if int(np.count_nonzero(neutral_mask)) >= 128:
        sample = sample[neutral_mask]

    return sample


def apply_color_balance(image: np.ndarray, params: ColorBalanceParams) -> np.ndarray:
    active_tonal = [
        ("shadows", params.shadows),
        ("midtones", params.midtones),
        ("highlights", params.highlights),
    ]
    active_tonal = [
        (tonal, balance, axis_to_gains(balance, scale=TONAL_BALANCE_SCALE))
        for tonal, balance in active_tonal
        if not balance_axis_is_neutral(balance)
    ]
    global_gains = axis_to_gains(params.global_balance, scale=GLOBAL_BALANCE_SCALE)
    global_is_neutral = balance_axis_is_neutral(params.global_balance)

    if global_is_neutral and not active_tonal:
        # Lab Print already feeds this stage bounded float32 data. Returning
        # directly avoids a full-resolution clip/copy when WB sliders are zero.
        return image.astype(np.float32, copy=False)

    clipped = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    if not active_tonal:
        balanced = clipped * global_gains.reshape(1, 1, 3)
        np.clip(balanced, 0.0, 1.0, out=balanced)
        return balanced.astype(np.float32, copy=False)

    if global_is_neutral:
        balanced = clipped.astype(np.float32, copy=True)
    else:
        balanced = clipped * global_gains.reshape(1, 1, 3)

    luminance = rgb_luminance(clipped)
    for tonal, balance, gains in active_tonal:
        weight = tonal_weight(luminance, balance.tonal_range, tonal=tonal).astype(np.float32, copy=False)
        log_delta = np.log(np.maximum(gains, 1e-6)).astype(np.float32, copy=False)
        changed_channels = np.flatnonzero(np.abs(log_delta) > 1e-6)
        for channel in changed_channels:
            channel_gain = np.exp(weight * log_delta[channel])
            balanced[:, :, channel] *= channel_gain

    np.clip(balanced, 0.0, 1.0, out=balanced)
    return balanced.astype(np.float32, copy=False)


def balance_axis_is_neutral(axis: BalanceAxis) -> bool:
    return axis.red_cyan == 0 and axis.green_magenta == 0 and axis.blue_yellow == 0


def suggest_global_balance_from_neutral(
    image: np.ndarray,
    point: ImagePoint,
    current_balance: BalanceAxis,
    *,
    radius: int = 14,
    scale: float = GLOBAL_BALANCE_SCALE,
) -> tuple[BalanceAxis, np.ndarray, np.ndarray]:
    sample_rgb = sample_rgb_at_point(image, point, radius=radius)
    sample_rgb = np.maximum(sample_rgb, np.array([1e-5, 1e-5, 1e-5], dtype=np.float32))
    target_gray = float(np.mean(sample_rgb))
    gains = np.clip(target_gray / sample_rgb, 0.25, 4.0).astype(np.float32)
    delta = np.rint(np.log2(gains) * scale).astype(np.int32)

    return (
        BalanceAxis(
            red_cyan=clamp_balance_value(current_balance.red_cyan + int(delta[0])),
            green_magenta=clamp_balance_value(current_balance.green_magenta + int(delta[1])),
            blue_yellow=clamp_balance_value(current_balance.blue_yellow + int(delta[2])),
        ),
        sample_rgb.astype(np.float32),
        gains,
    )


def sample_rgb_at_point(image: np.ndarray, point: ImagePoint, *, radius: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise PipelineError("White balance picker needs an RGB image.")

    height, width, _channels = image.shape
    if width <= 0 or height <= 0:
        raise PipelineError("White balance sample image is empty.")

    x = min(max(0, int(point.x)), width - 1)
    y = min(max(0, int(point.y)), height - 1)
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(width, x + radius + 1)
    y1 = min(height, y + radius + 1)
    sample = image[y0:y1, x0:x1]
    if sample.size == 0:
        raise PipelineError("White balance sample area is empty.")

    pixels = np.clip(sample.reshape(-1, 3), 0.0, 1.0)
    unclipped = np.all((pixels > 0.01) & (pixels < 0.99), axis=1)
    if int(np.count_nonzero(unclipped)) >= 16:
        pixels = pixels[unclipped]

    return np.median(pixels, axis=0).astype(np.float32)


def clamp_balance_value(value: int) -> int:
    return max(-100, min(100, value))


def apply_global_balance(image: np.ndarray, axis: BalanceAxis) -> np.ndarray:
    gains = axis_to_gains(axis, scale=GLOBAL_BALANCE_SCALE)
    return image * gains.reshape(1, 1, 3)


def apply_tonal_balance(
    image: np.ndarray,
    luminance: np.ndarray,
    balance: TonalBalance,
    *,
    tonal: str,
) -> np.ndarray:
    gains = axis_to_gains(balance, scale=TONAL_BALANCE_SCALE)
    if np.allclose(gains, 1.0, atol=1e-5):
        return image

    weight = tonal_weight(luminance, balance.tonal_range, tonal=tonal)
    mixed_gains = 1.0 + weight[:, :, None] * (gains.reshape(1, 1, 3) - 1.0)
    return image * mixed_gains


def axis_to_gains(axis: BalanceAxis, *, scale: float) -> np.ndarray:
    return np.array(
        [
            2.0 ** (axis.red_cyan / scale),
            2.0 ** (axis.green_magenta / scale),
            2.0 ** (axis.blue_yellow / scale),
        ],
        dtype=np.float32,
    )


def tonal_weight(luminance: np.ndarray, tonal_range: int, *, tonal: str) -> np.ndarray:
    amount = np.clip(tonal_range / 100.0, 0.0, 1.0)

    if tonal == "shadows":
        edge = 0.12 + amount * 0.56
        return 1.0 - smoothstep(0.0, edge, luminance)

    if tonal == "highlights":
        width = 0.12 + amount * 0.56
        return smoothstep(1.0 - width, 1.0, luminance)

    if tonal == "midtones":
        width = 0.16 + amount * 0.48
        distance = np.abs(luminance - 0.5) / max(width, 1e-5)
        return np.clip(1.0 - smoothstep(0.0, 1.0, distance), 0.0, 1.0)

    return np.zeros_like(luminance, dtype=np.float32)


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-5), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luminance_histogram(image: np.ndarray, *, bins: int = 256) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    luminance = rgb_luminance(clipped)
    values, _ = np.histogram(luminance, bins=bins, range=(0.0, 1.0))
    return values.astype(np.float32)


def suggest_luminance_levels(
    image: np.ndarray,
    *,
    black_percentile: float = 0.5,
    mid_percentile: float = 55.0,
    white_percentile: float = 99.5,
    black_padding: float = -0.02,
    white_padding: float = 0.04,
    curve_mode: str = PrintCurveMode.LINEAR.value,
    target_output_mid: float = 0.50,
) -> dict[str, int]:
    luminance = rgb_luminance(np.clip(image, 0.0, 1.0))
    sample = luminance.reshape(-1)
    stride = max(1, len(sample) // 800_000)
    sample = sample[::stride]

    left = float(np.percentile(sample, black_percentile))
    mid_source = float(np.percentile(sample, mid_percentile))
    right = float(np.percentile(sample, white_percentile))
    span = max(1e-5, right - left)

    black = left + black_padding * span
    white = right + white_padding * span
    mid = solve_mid_control_point(
        mid_source,
        black,
        white,
        curve_mode=curve_mode,
        target_output_mid=target_output_mid,
    )

    black_point = round(black * 100)
    mid_point = round(mid * 100)
    white_point = round(white * 100)

    black_point = max(0, min(98, black_point))
    white_point = max(black_point + 2, min(100, white_point))
    mid_point = max(black_point + 1, min(white_point - 1, mid_point))

    return {
        "black_point": black_point,
        "mid_point": mid_point,
        "white_point": white_point,
    }


def suggest_lab_print_luminance_levels(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    camera_to_srgb_matrix: np.ndarray | None = None,
) -> dict[str, int]:
    positive_source = np.clip(1.0 - normalized_log, 0.0, 1.0)
    sampled_log = sampled_rgb_pixels(normalized_log, limit=LAB_PRINT_AUTO_EXPOSURE_SAMPLE_LIMIT)
    sampled_positive = np.clip(1.0 - sampled_log, 0.0, 1.0)
    source_sample = rgb_luminance(sampled_positive.reshape(-1, 1, 3)).reshape(-1)

    if source_sample.size < 128:
        return suggest_luminance_levels(
            positive_source,
            black_percentile=1.2,
            mid_percentile=70.0,
            white_percentile=99.0,
            black_padding=-0.006,
            white_padding=0.020,
            curve_mode=PrintCurveMode.LINEAR.value,
            target_output_mid=0.54,
        )

    black = float(np.percentile(source_sample, LAB_PRINT_AUTO_BLACK_PERCENTILE))
    white = float(np.percentile(source_sample, LAB_PRINT_AUTO_WHITE_PERCENTILE))
    source_span = max(1e-5, white - black)
    black -= LAB_PRINT_AUTO_BLACK_PADDING * source_span
    white += LAB_PRINT_AUTO_WHITE_PADDING * source_span

    if white - black < LAB_PRINT_AUTO_MIN_SPAN:
        center = (black + white) * 0.5
        half_span = LAB_PRINT_AUTO_MIN_SPAN * 0.5
        black = center - half_span
        white = center + half_span

    black = float(np.clip(black, 0.0, 0.96))
    white = float(np.clip(white, black + 0.04, 1.0))
    source_span = max(1e-5, white - black)

    mid_low = black + source_span * LAB_PRINT_AUTO_MID_LOW
    mid_high = black + source_span * LAB_PRINT_AUTO_MID_HIGH
    mid = solve_lab_print_visual_midpoint(
        sampled_log,
        black,
        white,
        adjustments,
        camera_to_srgb_matrix=camera_to_srgb_matrix,
    )
    mid = float(np.clip(mid, mid_low, mid_high))

    black_point = round(black * 100)
    mid_point = round(mid * 100)
    white_point = round(white * 100)

    black_point = max(0, min(96, black_point))
    white_point = max(black_point + 4, min(100, white_point))
    mid_point = max(black_point + 1, min(white_point - 1, mid_point))

    return {
        "black_point": black_point,
        "mid_point": mid_point,
        "white_point": white_point,
    }


def sampled_rgb_pixels(image: np.ndarray, *, limit: int) -> np.ndarray:
    pixels = image.reshape(-1, 3)
    if len(pixels) == 0:
        return pixels.astype(np.float32, copy=False)
    stride = max(1, len(pixels) // max(1, limit))
    return pixels[::stride].astype(np.float32, copy=False)


def solve_lab_print_visual_midpoint(
    normalized_log_sample: np.ndarray,
    black: float,
    white: float,
    adjustments: AdjustmentParams,
    *,
    camera_to_srgb_matrix: np.ndarray | None,
) -> float:
    span = max(1e-5, white - black)
    low = black + span * LAB_PRINT_AUTO_MID_LOW
    high = black + span * LAB_PRINT_AUTO_MID_HIGH

    for _index in range(13):
        candidate = (low + high) * 0.5
        brightness = lab_print_visual_brightness_score(
            normalized_log_sample,
            black,
            candidate,
            white,
            adjustments,
            camera_to_srgb_matrix=camera_to_srgb_matrix,
        )
        if brightness > LAB_PRINT_AUTO_VISUAL_TARGET:
            low = candidate
        else:
            high = candidate

    return (low + high) * 0.5


def lab_print_visual_brightness_score(
    normalized_log_sample: np.ndarray,
    black: float,
    mid: float,
    white: float,
    adjustments: AdjustmentParams,
    *,
    camera_to_srgb_matrix: np.ndarray | None,
) -> float:
    span = max(1e-5, white - black)
    mid_norm = float(np.clip((mid - black) / span, 0.01, 0.99))
    gamma = float(np.clip(np.log(0.5) / np.log(mid_norm), 0.2, 8.0))

    positive = np.clip(1.0 - normalized_log_sample, 0.0, 1.0)
    leveled = (positive - black) / span
    leveled = np.power(np.maximum(leveled, 0.0), gamma)
    leveled = np.clip(leveled, 0.0, 1.0).reshape(-1, 1, 3)
    normalized_for_print = np.clip(1.0 - leveled, 0.0, 1.0)

    if adjustments.auto_wb:
        cmy_offsets = estimate_lab_print_auto_cmy_offsets(
            normalized_for_print,
            strength=adjustments.auto_cmy_strength / 100.0,
        )
    else:
        cmy_offsets = np.zeros(3, dtype=np.float32)
    cmy_offsets = effective_lab_print_cmy_offsets(cmy_offsets, adjustments)

    processed = apply_log_hd_print_curve(
        normalized_for_print,
        adjustments,
        cmy_offsets=cmy_offsets,
    )
    processed = apply_output_color_transform(
        processed,
        camera_to_srgb_matrix,
        adjustments.camera_color_strength,
    )
    processed = apply_log_color_separation(
        processed,
        strength=LAB_PRINT_COLOR_SEPARATION_STRENGTH,
    )
    processed = apply_color_balance(processed, adjustments.color_balance)
    processed = apply_highlight_shadow_adjustments(processed, adjustments)

    luminance = rgb_luminance(processed).reshape(-1)
    if luminance.size == 0:
        return 0.0

    p50 = float(np.percentile(luminance, 50.0))
    p60 = float(np.percentile(luminance, 60.0))
    low = float(np.percentile(luminance, 12.0))
    high = float(np.percentile(luminance, 88.0))
    trimmed = luminance[(luminance >= low) & (luminance <= high)]
    trimmed_mean = float(np.mean(trimmed)) if trimmed.size else p50
    return 0.55 * p50 + 0.25 * p60 + 0.20 * trimmed_mean


def log_hd_positive_curve_values(values: np.ndarray, adjustments: AdjustmentParams) -> np.ndarray:
    positive = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    normalized_log = 1.0 - positive
    response = log_hd_print_response(
        normalized_log.reshape(-1, 1, 1).repeat(3, axis=2),
        adjustments,
        cmy_offsets=np.zeros(3, dtype=np.float32),
    )
    return response.reshape(-1, 3)[:, 0].reshape(positive.shape)


def inverse_log_hd_positive_curve_value(target: float, adjustments: AdjustmentParams) -> float:
    target = float(np.clip(target, 1e-5, 1.0 - 1e-5))
    low = 0.0
    high = 1.0
    for _index in range(28):
        mid = (low + high) * 0.5
        value = float(log_hd_positive_curve_values(np.array([mid], dtype=np.float32), adjustments)[0])
        if value < target:
            low = mid
        else:
            high = mid
    return (low + high) * 0.5


def solve_mid_control_point(
    source_mid: float,
    black: float,
    white: float,
    *,
    curve_mode: str,
    target_output_mid: float,
) -> float:
    span = max(1e-5, white - black)
    source_norm = float(np.clip((source_mid - black) / span, 0.01, 0.99))
    pre_curve_target = inverse_print_curve_value(target_output_mid, curve_mode)
    pre_curve_target = float(np.clip(pre_curve_target, 0.05, 0.95))
    gamma = np.log(pre_curve_target) / np.log(source_norm)
    gamma = float(np.clip(gamma, 0.2, 8.0))
    mid_norm = 0.5 ** (1.0 / gamma)
    return black + mid_norm * span


def rgb_luminance(image: np.ndarray) -> np.ndarray:
    return (
        image[:, :, 0] * 0.2126
        + image[:, :, 1] * 0.7152
        + image[:, :, 2] * 0.0722
    )


def linear_to_srgb8(linear_rgb: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear_rgb, 0.0, 1.0)
    srgb = np.power(clipped, 1.0 / 2.2)
    return np.ascontiguousarray((srgb * 255.0 + 0.5).astype(np.uint8))
