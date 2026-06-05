from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from qnegative.core.auto_detect import (
    AutoBaseResult,
    AutoFrameResult,
    detect_film_base,
    detect_film_frame,
    detect_frame_and_base,
)
from qnegative.core.frame_ranker import load_frame_ranker
from qnegative.core.models import AdjustmentParams, ImagePoint, ImageProcessingState, ImageRect, ImageSize
from qnegative.core.pipeline import (
    NegativeBasePreview,
    NegativePreviewResult,
    PipelineError,
    analysis_inset_crop,
    analysis_inset_from_adjustments,
    build_lab_print_color_stage,
    build_lab_print_display_stage,
    build_lab_print_levels_stage,
    build_lab_print_negative_stage,
    build_negative_base_preview,
    suggest_lab_print_luminance_levels,
)
from qnegative.core.preview import RawPreview, make_source_preview, resize_long_edge
from qnegative.ui.preview_cache import (
    PreviewRenderOutput,
    PreviewStageCache,
    base_stage_key,
    cmy_offsets_key,
    current_levels,
    lab_print_color_key,
    lab_print_display_key,
    lab_print_levels_key,
    preview_result_cache_key_for,
)


class PreviewRenderSignals(QObject):
    finished = Signal(int, object, bool)
    failed = Signal(int, str, bool)


class AutoDetectSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)


class RawPreviewSignals(QObject):
    finished = Signal(int, object, object)
    failed = Signal(int, object, str)


class PreInvertSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, object, str)


class ModelWarmupSignals(QObject):
    finished = Signal(bool, str)


@dataclass(frozen=True)
class AutoDetectOutput:
    mode: str
    path: Path | None
    frame_result: AutoFrameResult | None
    base_result: AutoBaseResult | None
    fallback_state: ImageProcessingState | None
    auto_preview: bool = False


@dataclass(frozen=True)
class PreInvertOutput:
    path: Path
    preview: RawPreview
    frame_rect: ImageRect
    adjustments: AdjustmentParams
    result: NegativePreviewResult
    cache_key: tuple | None
    lab_print_cmy_offsets: list[float] | None
    confidence: float


class RawPreviewTask(QRunnable):
    def __init__(self, *, job_id: int, path: Path, max_size: int) -> None:
        super().__init__()
        self.job_id = job_id
        self.path = path
        self.max_size = max_size
        self.signals = RawPreviewSignals()

    def run(self) -> None:
        try:
            preview = make_source_preview(self.path, max_size=self.max_size)
        except Exception as exc:
            self.signals.failed.emit(self.job_id, self.path, str(exc))
            return

        self.signals.finished.emit(self.job_id, self.path, preview)


class PreInvertTask(QRunnable):
    def __init__(
        self,
        *,
        job_id: int,
        path: Path,
        max_size: int,
        format_hint: str,
        file_key: tuple,
        adjustments: AdjustmentParams,
        prior_frame_rect: ImageRect | None = None,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.path = path
        self.max_size = max_size
        self.format_hint = format_hint
        self.file_key = file_key
        self.adjustments = deepcopy(adjustments)
        self.prior_frame_rect = prior_frame_rect
        self.signals = PreInvertSignals()

    def run(self) -> None:
        try:
            preview = make_source_preview(self.path, max_size=self.max_size)
            detected = detect_frame_and_base(
                preview.preview_linear_rgb,
                preview_size=preview.preview_size,
                source_size=preview.source_size,
                format_hint=self.format_hint,
                detect_base=False,
                prior_frame_rect=self.prior_frame_rect,
            )
            frame = detected.frame
            if frame is None or frame.confidence_level not in {"high", "fallback"}:
                raise PipelineError("No high-confidence frame detected.")

            base = build_negative_base_preview(
                preview.preview_linear_rgb,
                source_size=preview.source_size,
                mask_point=None,
                film_rect=frame.rect,
                lens_correction=self.adjustments.lens_correction,
                preview_camera_wb_linear_rgb=preview.preview_camera_wb_linear_rgb,
                camera_to_srgb_matrix=preview.camera_to_srgb_matrix,
            )
            negative_stage = build_lab_print_negative_stage(
                base,
                analysis_inset=analysis_inset_from_adjustments(self.adjustments),
            )
            effective = deepcopy(self.adjustments)
            auto_levels = suggest_lab_print_luminance_levels(
                analysis_inset_crop(negative_stage.normalized_log, negative_stage.analysis_inset),
                effective,
                camera_to_srgb_matrix=negative_stage.camera_to_srgb_matrix,
            )
            effective.black_point = auto_levels["black_point"]
            effective.mid_point = auto_levels["mid_point"]
            effective.white_point = auto_levels["white_point"]
            levels_stage = build_lab_print_levels_stage(
                negative_stage,
                effective,
                auto_levels=auto_levels,
            )
            color_stage = build_lab_print_color_stage(levels_stage, effective)
            result = build_lab_print_display_stage(color_stage, effective)
            lab_print_cmy_offsets = _cmy_offsets_to_state(result.wb_gains if effective.auto_wb else None)
            cache_key = preview_result_cache_key_for(
                file_key=self.file_key,
                preview=preview,
                mask_point=None,
                film_rect=frame.rect,
                adjustments=effective,
                lab_print_cmy_offsets=lab_print_cmy_offsets,
            )
        except Exception as exc:
            self.signals.failed.emit(self.job_id, self.path, str(exc))
            return

        self.signals.finished.emit(
            self.job_id,
            PreInvertOutput(
                path=self.path,
                preview=preview,
                frame_rect=frame.rect,
                adjustments=effective,
                result=result,
                cache_key=cache_key,
                lab_print_cmy_offsets=lab_print_cmy_offsets,
                confidence=frame.confidence,
            ),
        )


class ModelWarmupTask(QRunnable):
    def __init__(self) -> None:
        super().__init__()
        self.signals = ModelWarmupSignals()

    def run(self) -> None:
        loaded = load_frame_ranker()
        self.signals.finished.emit(loaded is not None, loaded[1] if loaded is not None else "")


class AutoDetectTask(QRunnable):
    def __init__(
        self,
        *,
        job_id: int,
        mode: str,
        path: Path | None,
        preview: RawPreview,
        format_hint: str,
        detect_base: bool,
        current_film_rect: ImageRect | None,
        fallback_state: ImageProcessingState | None,
        auto_preview: bool = False,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.mode = mode
        self.path = path
        self.preview = preview
        self.format_hint = format_hint
        self.detect_base = detect_base
        self.current_film_rect = current_film_rect
        self.fallback_state = fallback_state
        self.auto_preview = auto_preview
        self.signals = AutoDetectSignals()

    def run(self) -> None:
        frame_result: AutoFrameResult | None = None
        base_result: AutoBaseResult | None = None
        try:
            prior_frame_rect = (
                self.fallback_state.film_rect
                if self.fallback_state is not None
                else None
            )
            if self.mode == "frame_base":
                result = detect_frame_and_base(
                    self.preview.preview_linear_rgb,
                    preview_size=self.preview.preview_size,
                    source_size=self.preview.source_size,
                    format_hint=self.format_hint,
                    detect_base=self.detect_base,
                    prior_frame_rect=prior_frame_rect,
                )
                frame_result = result.frame
                base_result = result.base
            elif self.mode == "frame":
                frame_result = detect_film_frame(
                    self.preview.preview_linear_rgb,
                    preview_size=self.preview.preview_size,
                    source_size=self.preview.source_size,
                    format_hint=self.format_hint,
                    prior_frame_rect=prior_frame_rect,
                )
            elif self.mode == "base":
                base_result = detect_film_base(
                    self.preview.preview_linear_rgb,
                    preview_size=self.preview.preview_size,
                    source_size=self.preview.source_size,
                    frame_rect=self.current_film_rect,
                )
        except Exception as exc:
            self.signals.failed.emit(self.job_id, str(exc))
            return

        self.signals.finished.emit(
            self.job_id,
            AutoDetectOutput(
                mode=self.mode,
                path=self.path,
                frame_result=frame_result,
                base_result=base_result,
                fallback_state=self.fallback_state,
                auto_preview=self.auto_preview,
            ),
        )


class PreviewRenderTask(QRunnable):
    def __init__(
        self,
        *,
        job_id: int,
        preview: RawPreview,
        mask_point: ImagePoint | None,
        film_rect: ImageRect | None,
        adjustments: AdjustmentParams,
        auto_levels_pending: bool,
        show_errors: bool,
        quality: str,
        file_key: tuple | None = None,
        lab_print_cmy_offsets: list[float] | None = None,
        render_cache: PreviewStageCache | None = None,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.preview = preview
        self.mask_point = mask_point
        self.film_rect = film_rect
        self.adjustments = deepcopy(adjustments)
        self.auto_levels_pending = auto_levels_pending
        self.show_errors = show_errors
        self.quality = quality
        self.file_key = file_key
        self.lab_print_cmy_offsets = (
            _cmy_offsets_to_state(lab_print_cmy_offsets)
            if adjustments.auto_wb
            else None
        )
        self.render_cache = render_cache or PreviewStageCache()
        self.signals = PreviewRenderSignals()

    def run(self) -> None:
        try:
            base_key = base_stage_key(self.preview, self.mask_point, self.film_rect, self.adjustments)
            base = self._base_stage(base_key)
            output = self._lab_print_output(base_key, base)
        except Exception as exc:
            self.signals.failed.emit(self.job_id, str(exc), self.show_errors)
            return

        self.signals.finished.emit(self.job_id, output, self.show_errors)

    def _base_stage(self, base_key: tuple) -> NegativeBasePreview:
        if self.render_cache.base_key == base_key and self.render_cache.base is not None:
            return self.render_cache.base

        return build_negative_base_preview(
            self.preview.preview_linear_rgb,
            source_size=self.preview.source_size,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            lens_correction=self.adjustments.lens_correction,
            preview_camera_wb_linear_rgb=self.preview.preview_camera_wb_linear_rgb,
            camera_to_srgb_matrix=self.preview.camera_to_srgb_matrix,
        )

    def _lab_print_output(
        self,
        base_key: tuple,
        base: NegativeBasePreview,
    ) -> PreviewRenderOutput:
        negative_key = ("lab_print_negative", base_key, self.adjustments.analysis_inset_percent)
        if (
            self.render_cache.negative_key == negative_key
            and self.render_cache.negative_stage is not None
        ):
            negative_stage = self.render_cache.negative_stage
        else:
            negative_stage = build_lab_print_negative_stage(
                base,
                analysis_inset=analysis_inset_from_adjustments(self.adjustments),
            )

        effective_adjustments = deepcopy(self.adjustments)
        applied_auto_levels = False
        if self.auto_levels_pending:
            auto_levels = suggest_lab_print_luminance_levels(
                analysis_inset_crop(negative_stage.normalized_log, negative_stage.analysis_inset),
                effective_adjustments,
                camera_to_srgb_matrix=negative_stage.camera_to_srgb_matrix,
            )
            effective_adjustments.black_point = auto_levels["black_point"]
            effective_adjustments.mid_point = auto_levels["mid_point"]
            effective_adjustments.white_point = auto_levels["white_point"]
            applied_auto_levels = True
        else:
            auto_levels = current_levels(effective_adjustments)

        levels_key = lab_print_levels_key(
            negative_key,
            effective_adjustments,
            auto_levels_pending=False,
        )
        if (
            self.render_cache.levels_key == levels_key
            and self.render_cache.levels_stage is not None
        ):
            levels_stage = self.render_cache.levels_stage
        else:
            levels_stage = build_lab_print_levels_stage(
                negative_stage,
                effective_adjustments,
                auto_levels=auto_levels,
            )

        color_key = lab_print_color_key(
            levels_key,
            effective_adjustments,
            self.lab_print_cmy_offsets,
        )
        if (
            self.render_cache.color_key == color_key
            and self.render_cache.color_stage is not None
        ):
            color_stage = self.render_cache.color_stage
        else:
            color_stage = build_lab_print_color_stage(
                levels_stage,
                effective_adjustments,
                cmy_offsets=self.lab_print_cmy_offsets,
            )
        lab_print_cmy_offsets = _cmy_offsets_to_state(
            color_stage.wb_gains if effective_adjustments.auto_wb else None
        )

        display_key = lab_print_display_key(color_key, effective_adjustments)
        if (
            self.render_cache.display_key == display_key
            and self.render_cache.display_result is not None
        ):
            result = self.render_cache.display_result
        else:
            result = build_lab_print_display_stage(color_stage, effective_adjustments)

        return PreviewRenderOutput(
            path=self.preview.path,
            result=result,
            cache=PreviewStageCache(
                base_key=base_key,
                base=base,
                negative_key=negative_key,
                negative_stage=negative_stage,
                levels_key=levels_key,
                levels_stage=levels_stage,
                color_key=color_key,
                color_stage=color_stage,
                display_key=display_key,
                display_result=result,
            ),
            quality=self.quality,
            cache_key=self._result_cache_key(
                effective_adjustments,
                lab_print_cmy_offsets=lab_print_cmy_offsets,
            ),
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=deepcopy(effective_adjustments),
            lab_print_cmy_offsets=lab_print_cmy_offsets,
            applied_auto_levels=applied_auto_levels,
        )

    def _result_cache_key(
        self,
        adjustments: AdjustmentParams,
        *,
        lab_print_cmy_offsets: list[float] | np.ndarray | None = None,
    ) -> tuple | None:
        if self.file_key is None:
            return None
        return preview_result_cache_key_for(
            file_key=self.file_key,
            preview=self.preview,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=adjustments,
            lab_print_cmy_offsets=(
                lab_print_cmy_offsets
                if lab_print_cmy_offsets is not None
                else self.lab_print_cmy_offsets
            ),
        )


def scaled_raw_preview(preview: RawPreview, *, max_edge: int) -> RawPreview:
    longest = max(preview.preview_size.width, preview.preview_size.height)
    if longest <= max_edge:
        return preview

    preview_linear_rgb = resize_long_edge(preview.preview_linear_rgb, max_size=max_edge)
    preview_camera_wb_linear_rgb = resize_long_edge(
        preview.preview_camera_wb_linear_rgb,
        max_size=max_edge,
    )
    display_rgb8 = resize_long_edge(preview.display_rgb8, max_size=max_edge)
    height, width = preview_linear_rgb.shape[:2]

    return RawPreview(
        path=preview.path,
        source_size=preview.source_size,
        preview_size=ImageSize(width=width, height=height),
        preview_linear_rgb=np.ascontiguousarray(preview_linear_rgb),
        preview_camera_wb_linear_rgb=np.ascontiguousarray(preview_camera_wb_linear_rgb),
        display_rgb8=np.ascontiguousarray(display_rgb8),
        camera_to_srgb_matrix=preview.camera_to_srgb_matrix,
    )


def _cmy_offsets_to_state(offsets: np.ndarray | list[float] | None) -> list[float] | None:
    key = cmy_offsets_key(offsets)
    if key is None:
        return None
    return [float(value) for value in key]
