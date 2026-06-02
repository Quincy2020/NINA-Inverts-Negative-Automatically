from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qnegative.core.geometry import clamp_rect_to_image, scale_point, scale_rect, warp_rotated_rect
from qnegative.core.models import AdjustmentParams, BalanceAxis, ColorBalanceParams, DensityMatrixParams, ImagePoint, ImageRect, ImageSize, InvertMode, PrintCurveMode, TonalBalance


DENSITY_REFERENCE = 2.046
TRANSMITTANCE_EPSILON = 1e-5
LOG_MODE_EPSILON = 1e-6
LOG_ANALYSIS_BUFFER = 0.05
LOG_CMY_MAX_DENSITY = 0.2
LOG_DENSITY_MULTIPLIER = 0.2
LOG_GRADE_MULTIPLIER = 1.75
LOG_D_MAX = 4.0
LOG_TOE_WIDTH = 2.5
LOG_SHOULDER_WIDTH = 2.5
LOG_COLOR_SEPARATION_STRENGTH = 0.5
LOG_AUTO_WB_MAX_OFFSET = 0.025
NEGPY_LOG_PERCENTILE_CLIP = 0.02
NEGPY_AUTO_WB_STRENGTH = 0.18
NEGPY_AUTO_WB_MAX_OFFSET = 0.04
NEGPY_COLOR_SEPARATION_STRENGTH = 0.45
NEGPY_AUTO_EXPOSURE_SAMPLE_LIMIT = 180_000
NEGPY_AUTO_BLACK_PERCENTILE = 0.25
NEGPY_AUTO_WHITE_PERCENTILE = 99.75
NEGPY_AUTO_BLACK_PADDING = 0.060
NEGPY_AUTO_WHITE_PADDING = 0.085
NEGPY_AUTO_MIN_SPAN = 0.54
NEGPY_AUTO_MID_LOW = 0.04
NEGPY_AUTO_MID_HIGH = 0.74
NEGPY_AUTO_VISUAL_TARGET = 0.30
GLOBAL_BALANCE_SCALE = 55.0
TONAL_BALANCE_SCALE = 75.0
FILMIC_LUT_SIZE = 4096
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
}
_FILMIC_CURVE_LUTS: dict[str, np.ndarray] = {}

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
class NegativeBasePreview:
    film_linear_rgb: np.ndarray
    film_camera_wb_linear_rgb: np.ndarray | None
    transmittance_rgb: np.ndarray
    density_rgb: np.ndarray
    inverted_linear_rgb: np.ndarray
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
class DensityPreviewAnalysis:
    corrected_density_rgb: np.ndarray
    control_rgb: np.ndarray
    histogram: np.ndarray
    auto_levels: dict[str, int]


@dataclass(frozen=True)
class NegativePreviewResult:
    display_rgb8: np.ndarray
    processed_linear_rgb: np.ndarray
    color_balanced_linear_rgb: np.ndarray
    histogram: np.ndarray
    auto_levels: dict[str, int]
    wb_gains: np.ndarray
    mask_rgb: np.ndarray
    film_rect_preview: ImageRect

    @property
    def width(self) -> int:
        return self.display_rgb8.shape[1]

    @property
    def height(self) -> int:
        return self.display_rgb8.shape[0]


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
    base = build_negative_base_preview(
        preview_linear_rgb,
        source_size=source_size,
        mask_point=mask_point,
        film_rect=film_rect,
        preview_camera_wb_linear_rgb=preview_camera_wb_linear_rgb,
        camera_to_srgb_matrix=camera_to_srgb_matrix,
    )
    return process_negative_base_preview(base, adjustments)


def build_negative_base_preview(
    preview_linear_rgb: np.ndarray,
    *,
    source_size: ImageSize,
    mask_point: ImagePoint | None,
    film_rect: ImageRect | None,
    preview_camera_wb_linear_rgb: np.ndarray | None = None,
    camera_to_srgb_matrix: np.ndarray | None = None,
) -> NegativeBasePreview:
    if preview_linear_rgb.ndim != 3 or preview_linear_rgb.shape[2] != 3:
        raise PipelineError("Preview image must be an RGB array.")
    if mask_point is None:
        raise PipelineError("Select the film base with the base picker first.")
    if film_rect is None or not film_rect.is_valid():
        raise PipelineError("Select a valid negative frame area first.")

    preview_size = ImageSize(
        width=preview_linear_rgb.shape[1],
        height=preview_linear_rgb.shape[0],
    )
    mask_rgb = sample_mask_rgb(
        preview_linear_rgb,
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
    transmittance = film_linear / mask_rgb.reshape(1, 1, 3)
    inverted = invert_transmittance_simple(transmittance)
    density = transmittance_to_density(transmittance)

    return NegativeBasePreview(
        film_linear_rgb=film_linear,
        film_camera_wb_linear_rgb=film_camera_wb_linear,
        transmittance_rgb=transmittance.astype(np.float32, copy=False),
        density_rgb=density,
        inverted_linear_rgb=inverted,
        mask_rgb=mask_rgb,
        film_rect_preview=film_rect_preview,
        camera_to_srgb_matrix=camera_to_srgb_matrix,
    )

def process_negative_base_preview(
    base: NegativeBasePreview,
    adjustments: AdjustmentParams,
    *,
    density_analysis: DensityPreviewAnalysis | None = None,
) -> NegativePreviewResult:
    if adjustments.invert_mode == InvertMode.NEGPY_PRINT.value:
        return process_negpy_print_preview(base, adjustments)
    if adjustments.invert_mode == InvertMode.LOG_BOUNDS.value:
        return process_log_bounds_preview(base, adjustments)
    if adjustments.invert_mode == InvertMode.DENSITY.value:
        return process_density_preview(base, adjustments, analysis=density_analysis)
    return process_simple_preview(base, adjustments)


def process_simple_preview(
    base: NegativeBasePreview,
    adjustments: AdjustmentParams,
) -> NegativePreviewResult:
    processed = apply_output_color_transform(
        base.inverted_linear_rgb,
        base.camera_to_srgb_matrix,
        adjustments.camera_color_strength,
    )
    if adjustments.auto_wb:
        processed, wb_gains = apply_auto_white_balance(processed)
    else:
        wb_gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    processed = apply_color_balance(processed, adjustments.color_balance)
    color_balanced = processed
    histogram = luminance_histogram(processed)
    auto_levels = suggest_luminance_levels(processed)
    processed = apply_adjustments(processed, adjustments)
    processed = apply_saturation_adjustment(processed, adjustments)
    display_rgb8 = linear_to_srgb8(processed)

    return NegativePreviewResult(
        display_rgb8=display_rgb8,
        processed_linear_rgb=processed,
        color_balanced_linear_rgb=color_balanced,
        histogram=histogram,
        auto_levels=auto_levels,
        wb_gains=wb_gains,
        mask_rgb=base.mask_rgb,
        film_rect_preview=base.film_rect_preview,
    )


def process_density_preview(
    base: NegativeBasePreview,
    adjustments: AdjustmentParams,
    *,
    analysis: DensityPreviewAnalysis | None = None,
) -> NegativePreviewResult:
    if analysis is None:
        analysis = build_density_preview_analysis(base, adjustments)

    corrected_density = analysis.corrected_density_rgb
    histogram = analysis.histogram
    auto_levels = analysis.auto_levels

    processed = apply_density_levels(corrected_density, adjustments, clip=False)
    processed = apply_exposure(processed, adjustments)
    processed = apply_highlight_shadow_adjustments(processed, adjustments)
    processed = apply_soft_tone_adjustments(processed, adjustments)
    processed = apply_output_color_transform(
        processed,
        base.camera_to_srgb_matrix,
        adjustments.camera_color_strength,
    )
    if adjustments.auto_wb:
        processed, wb_gains = apply_auto_white_balance(processed)
    else:
        wb_gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    processed = apply_color_balance(processed, adjustments.color_balance)
    color_balanced = processed
    processed = apply_print_curve(processed, adjustments.print_curve)
    processed = apply_contrast(processed, adjustments)
    processed = apply_saturation_adjustment(processed, adjustments)
    display_rgb8 = linear_to_srgb8(processed)

    return NegativePreviewResult(
        display_rgb8=display_rgb8,
        processed_linear_rgb=processed,
        color_balanced_linear_rgb=color_balanced,
        histogram=histogram,
        auto_levels=auto_levels,
        wb_gains=wb_gains,
        mask_rgb=base.mask_rgb,
        film_rect_preview=base.film_rect_preview,
    )


def build_density_preview_analysis(
    base: NegativeBasePreview,
    adjustments: AdjustmentParams,
) -> DensityPreviewAnalysis:
    corrected_density = apply_density_matrix(base.density_rgb, adjustments.density_matrix)
    control = density_to_control_image(corrected_density)
    histogram = luminance_histogram(control)
    auto_levels = suggest_density_luminance_levels(control, adjustments.print_curve)
    return DensityPreviewAnalysis(
        corrected_density_rgb=corrected_density,
        control_rgb=control,
        histogram=histogram,
        auto_levels=auto_levels,
    )


def process_log_bounds_preview(
    base: NegativeBasePreview,
    adjustments: AdjustmentParams,
) -> NegativePreviewResult:
    normalized_log = normalize_log_bounds(base.film_linear_rgb)
    positive_control = 1.0 - normalized_log
    histogram = luminance_histogram(positive_control)
    auto_levels = suggest_log_bounds_luminance_levels(positive_control)

    positive_control = apply_unit_levels(positive_control, adjustments, clip=True)
    normalized_for_print = np.clip(1.0 - positive_control, 0.0, 1.0)

    if adjustments.auto_wb:
        cmy_offsets = estimate_log_auto_cmy_offsets(
            normalized_for_print,
            adjustments,
            base.camera_to_srgb_matrix,
        )
    else:
        cmy_offsets = np.zeros(3, dtype=np.float32)

    processed = apply_log_hd_print_curve(
        normalized_for_print,
        adjustments,
        cmy_offsets=cmy_offsets,
    )
    processed = apply_output_color_transform(
        processed,
        base.camera_to_srgb_matrix,
        adjustments.camera_color_strength,
    )
    processed = apply_log_color_separation(
        processed,
        strength=LOG_COLOR_SEPARATION_STRENGTH,
    )
    processed = apply_color_balance(processed, adjustments.color_balance)
    processed = apply_highlight_shadow_adjustments(processed, adjustments)
    color_balanced = processed
    processed = apply_saturation_adjustment(processed, adjustments)
    display_rgb8 = linear_to_srgb8(processed)

    return NegativePreviewResult(
        display_rgb8=display_rgb8,
        processed_linear_rgb=processed,
        color_balanced_linear_rgb=color_balanced,
        histogram=histogram,
        auto_levels=auto_levels,
        wb_gains=cmy_offsets.astype(np.float32, copy=False),
        mask_rgb=base.mask_rgb,
        film_rect_preview=base.film_rect_preview,
    )


def process_negpy_print_preview(
    base: NegativeBasePreview,
    adjustments: AdjustmentParams,
) -> NegativePreviewResult:
    source_linear = (
        base.film_camera_wb_linear_rgb
        if base.film_camera_wb_linear_rgb is not None
        else base.film_linear_rgb
    )
    normalized_log = normalize_log_bounds(
        source_linear,
        percentile_clip=NEGPY_LOG_PERCENTILE_CLIP,
    )
    positive_control = 1.0 - normalized_log
    histogram = luminance_histogram(positive_control)
    auto_levels = suggest_negpy_print_luminance_levels(
        normalized_log,
        adjustments,
        camera_to_srgb_matrix=base.camera_to_srgb_matrix,
    )

    positive_control = apply_unit_levels(positive_control, adjustments, clip=True)
    normalized_for_print = np.clip(1.0 - positive_control, 0.0, 1.0)

    if adjustments.auto_wb:
        cmy_offsets = estimate_negpy_auto_cmy_offsets(normalized_for_print)
    else:
        cmy_offsets = np.zeros(3, dtype=np.float32)

    processed = apply_log_hd_print_curve(
        normalized_for_print,
        adjustments,
        cmy_offsets=cmy_offsets,
    )
    processed = apply_output_color_transform(
        processed,
        base.camera_to_srgb_matrix,
        adjustments.camera_color_strength,
    )
    processed = apply_log_color_separation(
        processed,
        strength=NEGPY_COLOR_SEPARATION_STRENGTH,
    )
    processed = apply_color_balance(processed, adjustments.color_balance)
    processed = apply_highlight_shadow_adjustments(processed, adjustments)
    color_balanced = processed
    processed = apply_saturation_adjustment(processed, adjustments)
    display_rgb8 = linear_to_srgb8(processed)

    return NegativePreviewResult(
        display_rgb8=display_rgb8,
        processed_linear_rgb=processed,
        color_balanced_linear_rgb=color_balanced,
        histogram=histogram,
        auto_levels=auto_levels,
        wb_gains=cmy_offsets.astype(np.float32, copy=False),
        mask_rgb=base.mask_rgb,
        film_rect_preview=base.film_rect_preview,
    )


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


def invert_negative(linear_rgb: np.ndarray, mask_rgb: np.ndarray) -> np.ndarray:
    neutral = linear_rgb / mask_rgb.reshape(1, 1, 3)
    return invert_transmittance_simple(neutral)


def invert_transmittance_simple(transmittance: np.ndarray) -> np.ndarray:
    positive = 1.0 - transmittance
    return np.clip(positive, 0.0, 1.0)


def transmittance_to_density(transmittance: np.ndarray) -> np.ndarray:
    clipped = np.clip(transmittance, TRANSMITTANCE_EPSILON, 1.0)
    density = -np.log10(clipped)
    return np.maximum(density, 0.0).astype(np.float32, copy=False)


def normalize_log_bounds(
    linear_rgb: np.ndarray,
    *,
    percentile_clip: float = 0.00001,
    analysis_buffer: float = LOG_ANALYSIS_BUFFER,
) -> np.ndarray:
    safe = np.clip(
        np.nan_to_num(linear_rgb, nan=LOG_MODE_EPSILON, posinf=1.0, neginf=LOG_MODE_EPSILON),
        LOG_MODE_EPSILON,
        1.0,
    )
    log_rgb = np.log10(safe).astype(np.float32, copy=False)
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
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


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


def apply_log_hd_print_curve(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    cmy_offsets: np.ndarray,
) -> np.ndarray:
    return log_hd_print_response(normalized_log, adjustments, cmy_offsets=cmy_offsets)


def log_hd_print_response(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    cmy_offsets: np.ndarray,
) -> np.ndarray:
    density, grade = log_print_density_grade(adjustments)
    pivot_value = float(np.clip(1.0 - (0.01 + density * LOG_DENSITY_MULTIPLIER), 0.02, 0.98))
    slope_value = float(np.clip(1.0 + grade * LOG_GRADE_MULTIPLIER, 0.1, 16.0))

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
    if adjustments.print_curve == PrintCurveMode.LINEAR.value:
        base_density = 0.82
        base_grade = 1.25
    elif adjustments.print_curve == PrintCurveMode.SOFT.value:
        base_density = 0.92
        base_grade = 1.85
    elif adjustments.print_curve == PrintCurveMode.CONTRAST.value:
        base_density = 1.06
        base_grade = 3.35
    else:
        base_density = 1.0
        base_grade = 2.5

    density = base_density - adjustments.exposure / 100.0
    grade = base_grade + adjustments.contrast * 0.025
    return float(np.clip(density, 0.05, 2.0)), float(np.clip(grade, 0.1, 6.0))


def estimate_log_auto_cmy_offsets(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    camera_to_srgb_matrix: np.ndarray | None,
    *,
    strength: float = 0.32,
) -> np.ndarray:
    probe = apply_log_hd_print_curve(
        normalized_log,
        adjustments,
        cmy_offsets=np.zeros(3, dtype=np.float32),
    )
    probe = apply_output_color_transform(
        probe,
        camera_to_srgb_matrix,
        adjustments.camera_color_strength,
    )
    probe = apply_log_color_separation(
        probe,
        strength=LOG_COLOR_SEPARATION_STRENGTH,
    )
    sample = select_log_positive_wb_sample(probe)
    if sample.size == 0:
        return np.zeros(3, dtype=np.float32)

    median_rgb = np.maximum(
        np.median(sample, axis=0).astype(np.float32),
        np.array([LOG_MODE_EPSILON, LOG_MODE_EPSILON, LOG_MODE_EPSILON], dtype=np.float32),
    )
    magenta = float(np.log10(median_rgb[1]) - np.log10(median_rgb[0]))
    yellow = float(np.log10(median_rgb[2]) - np.log10(median_rgb[0]))
    offsets = np.array([0.0, magenta, yellow], dtype=np.float32) * strength
    return np.clip(offsets, -LOG_AUTO_WB_MAX_OFFSET, LOG_AUTO_WB_MAX_OFFSET).astype(np.float32, copy=False)


def estimate_negpy_auto_cmy_offsets(
    normalized_log: np.ndarray,
    *,
    strength: float = NEGPY_AUTO_WB_STRENGTH,
) -> np.ndarray:
    sample = select_negpy_log_wb_sample(normalized_log)
    if sample.size == 0:
        return np.zeros(3, dtype=np.float32)

    median_log = np.median(sample, axis=0).astype(np.float32)
    offsets = np.array(
        [
            0.0,
            median_log[0] - median_log[1],
            median_log[0] - median_log[2],
        ],
        dtype=np.float32,
    )
    offsets *= float(np.clip(strength, 0.0, 1.0))
    return np.clip(offsets, -NEGPY_AUTO_WB_MAX_OFFSET, NEGPY_AUTO_WB_MAX_OFFSET).astype(np.float32, copy=False)


def select_negpy_log_wb_sample(normalized_log: np.ndarray) -> np.ndarray:
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


def density_to_control_image(density_rgb: np.ndarray) -> np.ndarray:
    return np.clip(density_rgb / DENSITY_REFERENCE, 0.0, 1.0).astype(np.float32, copy=False)


def apply_density_levels(
    density_rgb: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    clip: bool = True,
) -> np.ndarray:
    return apply_unit_levels(
        density_to_control_image(density_rgb),
        adjustments,
        clip=clip,
        soft_black=True,
    )


def apply_soft_black_levels(normalized: np.ndarray, gamma: float) -> np.ndarray:
    values = normalized.astype(np.float32, copy=False)
    positive = np.maximum(values, 0.0)
    mapped = LEVELS_SOFT_BLACK_FLOOR + np.power(positive, gamma) * (1.0 - LEVELS_SOFT_BLACK_FLOOR)

    below_black = values < 0.0
    if np.any(below_black):
        toe = LEVELS_SOFT_BLACK_FLOOR * np.exp(values / LEVELS_SOFT_BLACK_WIDTH)
        mapped = np.where(below_black, toe, mapped)

    return mapped.astype(np.float32, copy=False)


def apply_density_matrix(density_rgb: np.ndarray, params: DensityMatrixParams) -> np.ndarray:
    matrix = np.array(
        [
            [params.m00, params.m01, params.m02],
            [params.m10, params.m11, params.m12],
            [params.m20, params.m21, params.m22],
        ],
        dtype=np.float32,
    )
    corrected = density_rgb @ matrix.T
    return np.maximum(corrected, 0.0).astype(np.float32, copy=False)


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


def apply_highlight_shadow_adjustments(image: np.ndarray, adjustments: AdjustmentParams) -> np.ndarray:
    if adjustments.highlights == 0 and adjustments.shadows == 0:
        return image

    adjusted = np.maximum(image.astype(np.float32, copy=True), 0.0)
    luminance = np.maximum(rgb_luminance(adjusted), 0.0)
    target_luminance = luminance.copy()
    shadow_ratio_limit: float | None = None

    if adjustments.shadows != 0:
        shadow_amount = float(np.clip(adjustments.shadows / 100.0, -1.0, 1.0))
        shadow_pivot = 0.44
        shadow_norm = np.clip(luminance / shadow_pivot, 0.0, 1.0)
        shadow_weight = 1.0 - smoothstep(0.18, 1.0, shadow_norm)

        if shadow_amount > 0:
            gamma = 1.0 / (1.0 + 2.6 * shadow_amount)
            lifted = shadow_pivot * np.power(shadow_norm, gamma)
            black_anchor = smoothstep(0.0, 0.035, luminance)
            mix = np.clip(shadow_amount * 0.95 * shadow_weight * black_anchor, 0.0, 1.0)
            target_luminance = target_luminance * (1.0 - mix) + lifted * mix
            shadow_ratio_limit = 1.0 + 4.5 * shadow_amount
        else:
            amount = abs(shadow_amount)
            gamma = 1.0 + 2.2 * amount
            crushed = shadow_pivot * np.power(shadow_norm, gamma)
            mix = np.clip(amount * 0.90 * shadow_weight, 0.0, 1.0)
            target_luminance = target_luminance * (1.0 - mix) + crushed * mix

    if adjustments.highlights != 0:
        highlight_amount = float(np.clip(adjustments.highlights / 65.0, -1.0, 1.0))
        if highlight_amount > 0:
            highlight_weight = smoothstep(0.36, 0.92, luminance)
            boosted = target_luminance + (1.0 - np.clip(target_luminance, 0.0, 1.0)) * 0.75
            target_luminance += (boosted - target_luminance) * highlight_amount * highlight_weight
        else:
            amount = abs(highlight_amount)
            pivot = 0.46
            highlight_sample = luminance[luminance > pivot]
            if highlight_sample.size:
                upper = float(np.percentile(highlight_sample, 99.2))
            else:
                upper = 1.0
            upper = max(1.0, upper, pivot + 1e-4)
            span = upper - pivot

            highlight_weight = smoothstep(0.34, min(upper, 1.0), luminance)
            highlight_norm = np.maximum(target_luminance - pivot, 0.0) / span
            curve = 2.0 + amount * 12.0
            compressed = pivot + (1.0 - pivot) * (
                np.log1p(highlight_norm * curve) / np.log1p(curve)
            )
            compressed = np.minimum(compressed, 1.0)
            mix = highlight_weight * amount
            target_luminance = target_luminance * (1.0 - mix) + compressed * mix

    ratio = np.divide(
        target_luminance,
        luminance,
        out=np.ones_like(target_luminance, dtype=np.float32),
        where=luminance > 1e-5,
    )
    if shadow_ratio_limit is not None:
        ratio = np.minimum(ratio, shadow_ratio_limit)
    return (adjusted * ratio[:, :, None]).astype(np.float32, copy=False)


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
    balanced = np.clip(image, 0.0, 1.0).astype(np.float32, copy=True)
    balanced = apply_global_balance(balanced, params.global_balance)

    luminance = rgb_luminance(np.clip(balanced, 0.0, 1.0))
    balanced = apply_tonal_balance(balanced, luminance, params.shadows, tonal="shadows")
    balanced = apply_tonal_balance(balanced, luminance, params.midtones, tonal="midtones")
    balanced = apply_tonal_balance(balanced, luminance, params.highlights, tonal="highlights")
    return np.clip(balanced, 0.0, 1.0)


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


def suggest_density_luminance_levels(image: np.ndarray, curve_mode: str) -> dict[str, int]:
    return suggest_luminance_levels(
        image,
        black_percentile=0.6,
        mid_percentile=72.0,
        white_percentile=99.4,
        black_padding=-0.035,
        white_padding=0.035,
        curve_mode=curve_mode,
        target_output_mid=0.54,
    )


def suggest_log_bounds_luminance_levels(image: np.ndarray) -> dict[str, int]:
    return suggest_luminance_levels(
        image,
        black_percentile=0.8,
        mid_percentile=66.0,
        white_percentile=99.2,
        black_padding=0.010,
        white_padding=0.025,
        curve_mode=PrintCurveMode.LINEAR.value,
        target_output_mid=0.52,
    )


def suggest_negpy_print_luminance_levels(
    normalized_log: np.ndarray,
    adjustments: AdjustmentParams,
    *,
    camera_to_srgb_matrix: np.ndarray | None = None,
) -> dict[str, int]:
    positive_source = np.clip(1.0 - normalized_log, 0.0, 1.0)
    sampled_log = sampled_rgb_pixels(normalized_log, limit=NEGPY_AUTO_EXPOSURE_SAMPLE_LIMIT)
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

    black = float(np.percentile(source_sample, NEGPY_AUTO_BLACK_PERCENTILE))
    white = float(np.percentile(source_sample, NEGPY_AUTO_WHITE_PERCENTILE))
    source_span = max(1e-5, white - black)
    black -= NEGPY_AUTO_BLACK_PADDING * source_span
    white += NEGPY_AUTO_WHITE_PADDING * source_span

    if white - black < NEGPY_AUTO_MIN_SPAN:
        center = (black + white) * 0.5
        half_span = NEGPY_AUTO_MIN_SPAN * 0.5
        black = center - half_span
        white = center + half_span

    black = float(np.clip(black, 0.0, 0.96))
    white = float(np.clip(white, black + 0.04, 1.0))
    source_span = max(1e-5, white - black)

    mid_low = black + source_span * NEGPY_AUTO_MID_LOW
    mid_high = black + source_span * NEGPY_AUTO_MID_HIGH
    mid = solve_negpy_visual_midpoint(
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


def solve_negpy_visual_midpoint(
    normalized_log_sample: np.ndarray,
    black: float,
    white: float,
    adjustments: AdjustmentParams,
    *,
    camera_to_srgb_matrix: np.ndarray | None,
) -> float:
    span = max(1e-5, white - black)
    low = black + span * NEGPY_AUTO_MID_LOW
    high = black + span * NEGPY_AUTO_MID_HIGH

    for _index in range(13):
        candidate = (low + high) * 0.5
        brightness = negpy_visual_brightness_score(
            normalized_log_sample,
            black,
            candidate,
            white,
            adjustments,
            camera_to_srgb_matrix=camera_to_srgb_matrix,
        )
        if brightness > NEGPY_AUTO_VISUAL_TARGET:
            low = candidate
        else:
            high = candidate

    return (low + high) * 0.5


def negpy_visual_brightness_score(
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
        cmy_offsets = estimate_negpy_auto_cmy_offsets(normalized_for_print)
    else:
        cmy_offsets = np.zeros(3, dtype=np.float32)

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
        strength=NEGPY_COLOR_SEPARATION_STRENGTH,
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
