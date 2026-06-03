from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from time import perf_counter

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.file_sequence import IMAGE_EXTENSIONS, RAW_EXTENSIONS, list_supported_files
from qnegative.core.frame_ranker import load_frame_ranker
from qnegative.core.auto_detect import (
    AutoBaseResult,
    AutoFrameResult,
    detect_film_base,
    detect_frame_and_base,
    detect_film_frame,
)
from qnegative.core.models import (
    AdjustmentParams,
    DensityMatrixParams,
    ImagePoint,
    ImageProcessingState,
    ImageRect,
    ImageSize,
    InvertMode,
    LensCorrectionParams,
    ToolMode,
)
from qnegative.core.pipeline import (
    DensityPreviewAnalysis,
    NegativeBasePreview,
    LabPrintColorStage,
    LabPrintLevelsStage,
    LabPrintNegativeStage,
    NegativePreviewResult,
    PipelineError,
    build_lab_print_color_stage,
    build_lab_print_display_stage,
    build_lab_print_export_linear,
    build_lab_print_levels_stage,
    build_lab_print_negative_stage,
    build_negative_base_preview,
    build_density_preview_analysis,
    log_print_curve_engine,
    process_negative_base_preview,
    analysis_inset_from_adjustments,
    analysis_inset_crop,
    set_log_print_curve_engine,
    suggest_lab_print_luminance_levels,
    suggest_global_balance_from_neutral,
)
from qnegative.core.pipeline import LOG_PRINT_CURVE_DIRECT, LOG_PRINT_CURVE_LUT_4096, LOG_PRINT_CURVE_LUT_8192
from qnegative.core.preview import DEFAULT_PREVIEW_MAX_EDGE, RawPreview, make_raw_preview, resize_long_edge
from qnegative.core.raw_loader import load_raw_rgb16
from qnegative.core.session import load_roll_session, save_roll_session, session_path_for_folder
from qnegative.ui.control_panel import ControlPanel
from qnegative.ui.folder_filmstrip import FolderFilmstrip
from qnegative.ui.gl_preview_view import OpenGLPreviewView
from qnegative.ui.image_view import ImageView


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


INTERACTIVE_PREVIEW_MAX_EDGE = 720
FINAL_RENDER_QUALITY = "final"
INTERACTIVE_RENDER_QUALITY = "interactive"
FINAL_RENDER_DEBOUNCE_MS = 80
INTERACTIVE_RENDER_DEBOUNCE_MS = 115
PREVIEW_RESULT_CACHE_LIMIT = 16
RAW_PREVIEW_CACHE_LIMIT = 16


class BatchExportDialog(QDialog):
    pauseRequested = Signal()
    resumeRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("batchExportDialog")
        self.setWindowTitle("Batch Export")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setMinimumSize(340, 250)
        self.resize(380, 280)

        self.current_label = QLabel("Waiting")
        self.current_label.setObjectName("batchCurrentLabel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Waiting")
        self.queue = QListWidget()
        self.queue.setObjectName("batchQueue")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.cancel_button = QPushButton("Cancel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(QLabel("Current"))
        layout.addWidget(self.current_label)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("Queue"))
        layout.addWidget(self.queue, 1)
        button_row = QHBoxLayout()
        button_row.addWidget(self.pause_button)
        button_row.addWidget(self.resume_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)
        self.pause_button.clicked.connect(self.pauseRequested.emit)
        self.resume_button.clicked.connect(self.resumeRequested.emit)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.set_running(False)
        self._apply_style()

    def set_jobs(self, paths: list[Path]) -> None:
        self.queue.clear()
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            self.queue.addItem(item)
        self.progress.setValue(0)
        self.progress.setFormat("Queued")
        self.current_label.setText("Waiting")
        self.set_running(True)

    def set_current(self, path: Path) -> None:
        self.current_label.setText(path.name)
        for index in range(self.queue.count()):
            item = self.queue.item(index)
            if item.data(Qt.UserRole) == str(path):
                item.setText(f"> {path.name}")
                self.queue.setCurrentRow(index)
            elif not item.text().startswith("Done "):
                item.setText(Path(item.data(Qt.UserRole)).name)

    def update_progress(self, value: int, text: str) -> None:
        self.progress.setValue(max(0, min(100, int(value))))
        self.progress.setFormat(text)

    def mark_done(self, path: Path) -> None:
        for index in range(self.queue.count()):
            item = self.queue.item(index)
            if item.data(Qt.UserRole) == str(path):
                item.setText(f"Done {path.name}")
                break

    def finish(self, text: str, *, auto_close_ms: int | None = None) -> None:
        self.current_label.setText(text)
        self.progress.setValue(100)
        self.progress.setFormat(text)
        self.set_running(False)
        if auto_close_ms is not None:
            QTimer.singleShot(auto_close_ms, self.hide)

    def set_running(self, running: bool, *, paused: bool = False) -> None:
        self.pause_button.setEnabled(running and not paused)
        self.resume_button.setEnabled(running and paused)
        self.cancel_button.setEnabled(running)

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog#batchExportDialog {
                background: #20242b;
                color: #e8eaed;
            }
            QLabel {
                color: #e8eaed;
            }
            QLabel#batchCurrentLabel {
                background: #15191f;
                border: 1px solid #343c47;
                border-radius: 5px;
                padding: 8px;
                font-weight: 600;
            }
            QListWidget#batchQueue {
                background: #15191f;
                border: 1px solid #343c47;
                border-radius: 5px;
                color: #cfd6df;
                outline: 0;
            }
            QListWidget#batchQueue::item {
                padding: 6px;
            }
            QListWidget#batchQueue::item:selected {
                background: #2f5d82;
                color: #ffffff;
            }
            QProgressBar {
                background: #15191f;
                border: 1px solid #343c47;
                border-radius: 5px;
                color: #e8eaed;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background: #4aa3ff;
                border-radius: 4px;
            }
            QPushButton {
                background: #2d333d;
                border: 1px solid #444c59;
                border-radius: 5px;
                color: #f2f4f7;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background: #38414d;
            }
            QPushButton:disabled {
                color: #747d8a;
                background: #22272f;
            }
            """
        )


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
    cache_key: tuple | None = None
    mask_point: ImagePoint | None = None
    film_rect: ImageRect | None = None
    adjustments: AdjustmentParams | None = None
    applied_auto_levels: bool = False


@dataclass(frozen=True)
class CachedPreviewResult:
    key: tuple
    result: NegativePreviewResult


@dataclass(frozen=True)
class CachedRawPreview:
    key: tuple
    preview: RawPreview


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
    confidence: float


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
        balance_axis_key(params.global_balance),
        tonal_balance_key(params.shadows),
        tonal_balance_key(params.midtones),
        tonal_balance_key(params.highlights),
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
        params.strength,
        params.radius,
        params.center_x,
        params.center_y,
        params.smoothness,
        params.max_gain,
    )


def adjustments_preview_cache_key(adjustments: AdjustmentParams) -> tuple:
    return (
        adjustments.invert_mode,
        adjustments.print_curve,
        adjustments.auto_wb,
        color_balance_key(adjustments),
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
) -> tuple:
    return (
        file_key,
        preview.source_size,
        preview.preview_size,
        matrix_key(preview.camera_to_srgb_matrix),
        image_point_key(mask_point),
        image_rect_key(film_rect),
        adjustments_preview_cache_key(adjustments),
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


def lab_print_color_key(levels_key: tuple, adjustments: AdjustmentParams) -> tuple:
    return (
        "lab_print_color",
        levels_key,
        adjustments.print_curve,
        adjustments.exposure,
        adjustments.contrast,
        adjustments.soft_highlights,
        adjustments.soft_shadows,
        adjustments.auto_wb,
        adjustments.camera_color_strength,
        color_balance_key(adjustments),
    )


def lab_print_display_key(color_key: tuple, adjustments: AdjustmentParams) -> tuple:
    return (
        "lab_print_display",
        color_key,
        adjustments.highlights,
        adjustments.shadows,
        adjustments.saturation,
    )


def invert_mode_label(mode: str) -> str:
    labels = {
        InvertMode.LAB_PRINT.value: "Lab Print",
        InvertMode.DENSITY.value: "Density",
        InvertMode.LOG_BOUNDS.value: "Log Bounds",
        InvertMode.SIMPLE.value: "Simple",
    }
    return labels.get(mode, mode)


class RawPreviewTask(QRunnable):
    def __init__(self, *, job_id: int, path: Path, max_size: int) -> None:
        super().__init__()
        self.job_id = job_id
        self.path = path
        self.max_size = max_size
        self.signals = RawPreviewSignals()

    def run(self) -> None:
        try:
            preview = make_raw_preview(self.path, max_size=self.max_size)
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
            preview = make_raw_preview(self.path, max_size=self.max_size)
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
            cache_key = preview_result_cache_key_for(
                file_key=self.file_key,
                preview=preview,
                mask_point=None,
                film_rect=frame.rect,
                adjustments=effective,
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
        self.render_cache = render_cache or PreviewStageCache()
        self.signals = PreviewRenderSignals()

    def run(self) -> None:
        try:
            base_key = base_stage_key(self.preview, self.mask_point, self.film_rect, self.adjustments)
            base = self._base_stage(base_key)
            if self.adjustments.invert_mode == InvertMode.LAB_PRINT.value:
                output = self._lab_print_output(base_key, base)
            else:
                result = process_negative_base_preview(base, self.adjustments)
                output = PreviewRenderOutput(
                    path=self.preview.path,
                    result=result,
                    cache=PreviewStageCache(base_key=base_key, base=base),
                    quality=self.quality,
                    cache_key=self._result_cache_key(self.adjustments),
                    mask_point=self.mask_point,
                    film_rect=self.film_rect,
                    adjustments=deepcopy(self.adjustments),
                    applied_auto_levels=False,
                )
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

        color_key = lab_print_color_key(levels_key, effective_adjustments)
        if (
            self.render_cache.color_key == color_key
            and self.render_cache.color_stage is not None
        ):
            color_stage = self.render_cache.color_stage
        else:
            color_stage = build_lab_print_color_stage(levels_stage, effective_adjustments)

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
            cache_key=self._result_cache_key(effective_adjustments),
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=deepcopy(effective_adjustments),
            applied_auto_levels=applied_auto_levels,
        )

    def _result_cache_key(self, adjustments: AdjustmentParams) -> tuple | None:
        if self.file_key is None:
            return None
        return preview_result_cache_key_for(
            file_key=self.file_key,
            preview=self.preview,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=adjustments,
        )


class ExportSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(str, object)
    failed = Signal(str)
    cancelled = Signal(str)


class ExportCancelled(Exception):
    pass


class TiffExportTask(QRunnable):
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
        preview_cmy_offsets: np.ndarray | None = None,
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
        self.preview_cmy_offsets = (
            np.asarray(preview_cmy_offsets, dtype=np.float32).copy()
            if preview_cmy_offsets is not None
            else None
        )
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
            export_linear_rgb = self._process_export(base)
            timings["Lab Print"] = perf_counter() - stage_start
            self._raise_if_cancelled()

            self.signals.progress.emit(75, self._timed_progress_text("Preparing TIFF", timings))
            stage_start = perf_counter()
            linear_rgb = transform_preview_array(
                export_linear_rgb,
                flip_horizontal=self.flip_horizontal,
                flip_vertical=self.flip_vertical,
                rotation_quarters=self.rotation_quarters,
            )
            tiff_rgb16 = linear_to_srgb16(linear_rgb)
            timings["Prepare TIFF"] = perf_counter() - stage_start
            self._raise_if_cancelled()
            self.signals.progress.emit(90, self._timed_progress_text("Writing TIFF", timings))

            stage_start = perf_counter()
            import tifffile

            tifffile.imwrite(
                self.output_path,
                tiff_rgb16,
                photometric="rgb",
            )
            timings["TIFF write"] = perf_counter() - stage_start
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

    def _process_export(self, base: NegativeBasePreview) -> np.ndarray:
        if self.adjustments.invert_mode != InvertMode.LAB_PRINT.value:
            result = process_negative_base_preview(base, self.adjustments)
            if self.auto_levels_pending:
                adjusted = deepcopy(self.adjustments)
                adjusted.black_point = result.auto_levels["black_point"]
                adjusted.mid_point = result.auto_levels["mid_point"]
                adjusted.white_point = result.auto_levels["white_point"]
                result = process_negative_base_preview(base, adjusted)
            return result.processed_linear_rgb

        negative_stage = build_lab_print_negative_stage(
            base,
            include_histogram=False,
            analysis_inset=analysis_inset_from_adjustments(self.adjustments),
        )
        effective = deepcopy(self.adjustments)
        if self.auto_levels_pending:
            auto_levels = suggest_lab_print_luminance_levels(
                analysis_inset_crop(negative_stage.normalized_log, negative_stage.analysis_inset),
                effective,
                camera_to_srgb_matrix=negative_stage.camera_to_srgb_matrix,
            )
            effective.black_point = auto_levels["black_point"]
            effective.mid_point = auto_levels["mid_point"]
            effective.white_point = auto_levels["white_point"]
        else:
            auto_levels = current_levels(effective)

        levels_stage = build_lab_print_levels_stage(
            negative_stage,
            effective,
            auto_levels=auto_levels,
        )
        color_stage = build_lab_print_color_stage(
            levels_stage,
            effective,
            cmy_offsets=self.preview_cmy_offsets if effective.auto_wb else None,
        )
        return build_lab_print_export_linear(color_stage, effective)


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


class MainWindow(QMainWindow):
    def __init__(self, *, default_invert_mode: str = InvertMode.LAB_PRINT.value) -> None:
        super().__init__()
        self.setWindowTitle("NINA")
        self.default_invert_mode = default_invert_mode
        self.current_path: Path | None = None
        self.current_preview: RawPreview | None = None
        self.folder_files: list[Path] = []
        self.current_index: int = -1
        self.image_states: dict[Path, ImageProcessingState] = {}
        self.raw_preview_cache: dict[Path, CachedRawPreview] = {}
        self.preview_result_cache: dict[Path, CachedPreviewResult] = {}
        self.mask_point: ImagePoint | None = None
        self.film_rect: ImageRect | None = None
        self.white_balance_point: ImagePoint | None = None
        self.adjustments = self._default_adjustments()
        self.negative_preview_active = False
        self.auto_levels_pending = True
        self._applying_auto_levels = False
        self.last_negative_result: NegativePreviewResult | None = None
        self._negative_base_cache: NegativeBasePreview | None = None
        self._negative_base_cache_key: tuple[object, ...] | None = None
        self._density_analysis_cache: DensityPreviewAnalysis | None = None
        self._density_analysis_cache_key: tuple[object, ...] | None = None
        self._preview_stage_caches = {
            FINAL_RENDER_QUALITY: PreviewStageCache(),
            INTERACTIVE_RENDER_QUALITY: PreviewStageCache(),
        }
        self._interactive_adjustment_active = False
        self._interactive_preview_cache_key: tuple[object, ...] | None = None
        self._interactive_preview_cache: RawPreview | None = None
        self._last_untransformed_negative_result: NegativePreviewResult | None = None
        self._preview_flip_horizontal = False
        self._preview_flip_vertical = False
        self._preview_rotation_quarters = 0
        self._thread_pool = QThreadPool.globalInstance()
        self._render_job_id = 0
        self._render_in_progress = False
        self._render_pending = False
        self._render_pending_show_errors = False
        self._raw_preview_job_id = 0
        self._raw_preview_in_progress = False
        self._model_warmup_in_progress = False
        self._auto_detect_job_id = 0
        self._auto_detect_auto_preview_jobs: set[int] = set()
        self._auto_detect_in_progress = False
        self._preinvert_job_id = 0
        self._preinvert_in_progress: set[int] = set()
        self._preinvert_paths: set[Path] = set()
        self._preinvert_queue: list[Path] = []
        self._export_in_progress = False
        self._batch_export_queue: list[dict] = []
        self._batch_export_total = 0
        self._batch_export_done = 0
        self._batch_export_active = False
        self._batch_export_paused = False
        self._batch_export_cancel_requested = False
        self._batch_export_current_path: Path | None = None
        self._export_cancel_event: Event | None = None
        self._default_export_dir: Path | None = None
        self._gpu_preview_enabled = True
        self._auto_invert_after_frame_change = True
        self._auto_frame_new_negatives = True
        self._auto_preinvert_nearby_frames = True
        self._auto_preinvert_radius = 1
        self._roll_session_folder: Path | None = None
        self._roll_session_autosave = True

        self.control_panel = ControlPanel()
        self.origin_view = ImageView()
        self.image_view = self.origin_view
        self.preview_view = OpenGLPreviewView()
        self.preview_view.set_transform_context_enabled(True)
        self.preview_view.set_placeholder("Positive preview waiting")
        self.preview_tabs = QTabWidget()
        self.preview_tabs.setObjectName("previewTabs")
        self.batch_export_dialog = BatchExportDialog(self)
        self.batch_export_dialog.pauseRequested.connect(self.pause_batch_export)
        self.batch_export_dialog.resumeRequested.connect(self.resume_batch_export)
        self.batch_export_dialog.cancelRequested.connect(self.cancel_batch_export)
        self.empty_state = QWidget()
        self.empty_state.setObjectName("emptyState")
        self.open_empty_button = QPushButton("Open Folder")
        self.open_empty_button.setObjectName("emptyOpenButton")
        self.view_stack = QStackedWidget()
        self.filmstrip = FolderFilmstrip()
        self.filmstrip.hide()
        self.preview_refresh_timer = QTimer(self)
        self.preview_refresh_timer.setSingleShot(True)
        self.roll_session_save_timer = QTimer(self)
        self.roll_session_save_timer.setSingleShot(True)
        self.roll_session_save_timer.setInterval(700)
        self.preview_refresh_timer.setInterval(FINAL_RENDER_DEBOUNCE_MS)

        self._build_layout()
        self._build_menus()
        self._connect()
        self._build_shortcuts()
        self._apply_style()
        self.control_panel.set_adjustments(self.adjustments, emit=False)
        self.set_tool_mode(ToolMode.FILM_RECT)

        self.statusBar().showMessage("Ready")
        QTimer.singleShot(120, self._start_frame_ranker_warmup)

    def _build_layout(self) -> None:
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        empty_layout.addStretch(1)
        empty_row = QHBoxLayout()
        empty_row.addStretch(1)
        empty_row.addWidget(self.open_empty_button)
        empty_row.addStretch(1)
        empty_layout.addLayout(empty_row)
        empty_layout.addStretch(1)

        self.preview_tabs.addTab(self.origin_view, "Origin")
        self.preview_tabs.addTab(self.preview_view, "Preview")
        self.view_stack.addWidget(self.empty_state)
        self.view_stack.addWidget(self.preview_tabs)
        self.view_stack.setCurrentWidget(self.empty_state)
        right_layout.addWidget(self.view_stack, 1)
        right_layout.addWidget(self.filmstrip)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.addWidget(self.control_panel)
        splitter.addWidget(right_pane)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 980])

        self.setCentralWidget(splitter)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open RAW / TIFF...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        open_folder_action = QAction("Open Folder...", self)
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        file_menu.addSeparator()

        export_action = QAction("Export TIFF...", self)
        export_action.triggered.connect(self.export_current)
        file_menu.addAction(export_action)

        export_completed_action = QAction("Export Completed TIFFs...", self)
        export_completed_action.triggered.connect(self.export_completed)
        file_menu.addAction(export_completed_action)

        export_dir_action = QAction("Set Default Export Directory...", self)
        export_dir_action.triggered.connect(self.set_default_export_directory)
        file_menu.addAction(export_dir_action)
        file_menu.addSeparator()

        save_session_action = QAction("Save Roll Session", self)
        save_session_action.triggered.connect(self.save_roll_session_now)
        file_menu.addAction(save_session_action)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.quit)
        file_menu.addAction(exit_action)

        edit_menu = self.menuBar().addMenu("Edit")
        invert_action = QAction("Invert Preview", self)
        invert_action.triggered.connect(self.preview_inversion)
        edit_menu.addAction(invert_action)

        reset_action = QAction("Reset Current Image", self)
        reset_action.triggered.connect(self.reset_workspace)
        edit_menu.addAction(reset_action)

        view_menu = self.menuBar().addMenu("View")
        origin_action = QAction("Origin", self)
        origin_action.triggered.connect(lambda: self.preview_tabs.setCurrentWidget(self.origin_view))
        view_menu.addAction(origin_action)
        preview_action = QAction("Preview", self)
        preview_action.triggered.connect(lambda: self.preview_tabs.setCurrentWidget(self.preview_view))
        view_menu.addAction(preview_action)

        settings_menu = self.menuBar().addMenu("Settings")
        self.gpu_preview_action = QAction("GPU Preview Acceleration", self)
        self.gpu_preview_action.setCheckable(True)
        self.gpu_preview_action.setChecked(True)
        self.gpu_preview_action.toggled.connect(self.set_gpu_preview_enabled)
        settings_menu.addAction(self.gpu_preview_action)

        self.auto_invert_after_frame_action = QAction("Auto Invert After Frame Change", self)
        self.auto_invert_after_frame_action.setCheckable(True)
        self.auto_invert_after_frame_action.setChecked(True)
        self.auto_invert_after_frame_action.toggled.connect(self.set_auto_invert_after_frame_change)
        settings_menu.addAction(self.auto_invert_after_frame_action)

        self.auto_frame_new_negatives_action = QAction("Auto Frame New Negatives", self)
        self.auto_frame_new_negatives_action.setCheckable(True)
        self.auto_frame_new_negatives_action.setChecked(True)
        self.auto_frame_new_negatives_action.toggled.connect(self.set_auto_frame_new_negatives)
        settings_menu.addAction(self.auto_frame_new_negatives_action)

        self.auto_preinvert_nearby_action = QAction("Auto Pre-Invert Nearby Frames", self)
        self.auto_preinvert_nearby_action.setCheckable(True)
        self.auto_preinvert_nearby_action.setChecked(True)
        self.auto_preinvert_nearby_action.toggled.connect(self.set_auto_preinvert_nearby_frames)
        settings_menu.addAction(self.auto_preinvert_nearby_action)

        self.roll_session_autosave_action = QAction("Auto Save Roll Session", self)
        self.roll_session_autosave_action.setCheckable(True)
        self.roll_session_autosave_action.setChecked(True)
        self.roll_session_autosave_action.toggled.connect(self.set_roll_session_autosave)
        settings_menu.addAction(self.roll_session_autosave_action)

        preinvert_radius_menu = settings_menu.addMenu("Auto Pre-Invert Range")
        self.preinvert_radius_group = QActionGroup(self)
        self.preinvert_radius_group.setExclusive(True)
        for radius in (1, 2, 3, 5):
            action = QAction(f"Previous/Next {radius}", self)
            action.setCheckable(True)
            action.setData(radius)
            action.setChecked(radius == self._auto_preinvert_radius)
            self.preinvert_radius_group.addAction(action)
            preinvert_radius_menu.addAction(action)
        self.preinvert_radius_group.triggered.connect(self.set_auto_preinvert_radius)
        settings_menu.addSeparator()

        developer_menu = settings_menu.addMenu("Developer")

        self.density_matrix_dock = QDockWidget("Density Matrix", self)
        self.density_matrix_dock.setObjectName("densityMatrixDock")
        self.density_matrix_dock.setWidget(self.control_panel.density_matrix_panel)
        self.density_matrix_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.density_matrix_dock)
        self.density_matrix_dock.hide()

        density_action = QAction("Density Matrix", self)
        density_action.setCheckable(True)
        density_action.toggled.connect(self.density_matrix_dock.setVisible)
        self.density_matrix_dock.visibilityChanged.connect(density_action.setChecked)
        developer_menu.addAction(density_action)

        self.camera_color_dock = QDockWidget("Camera Color", self)
        self.camera_color_dock.setObjectName("cameraColorDock")
        self.camera_color_dock.setWidget(self.control_panel.camera_color_panel)
        self.camera_color_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.camera_color_dock)
        self.camera_color_dock.hide()

        camera_color_action = QAction("Camera Color", self)
        camera_color_action.setCheckable(True)
        camera_color_action.toggled.connect(self.camera_color_dock.setVisible)
        self.camera_color_dock.visibilityChanged.connect(camera_color_action.setChecked)
        developer_menu.addAction(camera_color_action)

        developer_menu.addSeparator()
        density_mode_action = QAction("Use Density Mode", self)
        density_mode_action.triggered.connect(lambda: self._set_developer_invert_mode(InvertMode.DENSITY.value))
        developer_menu.addAction(density_mode_action)

        simple_mode_action = QAction("Use Simple Mode", self)
        simple_mode_action.triggered.connect(lambda: self._set_developer_invert_mode(InvertMode.SIMPLE.value))
        developer_menu.addAction(simple_mode_action)

        base_picker_action = QAction("Base Picker Tool", self)
        base_picker_action.triggered.connect(lambda: self.set_tool_mode(ToolMode.MASK_PICKER))
        developer_menu.addAction(base_picker_action)

        developer_menu.addSeparator()
        export_advanced_menu = developer_menu.addMenu("Export Advanced")
        print_curve_menu = export_advanced_menu.addMenu("Print Curve Engine")
        self.print_curve_engine_group = QActionGroup(self)
        self.print_curve_engine_group.setExclusive(True)
        for label, engine in (
            ("LUT 8192", LOG_PRINT_CURVE_LUT_8192),
            ("LUT 4096", LOG_PRINT_CURVE_LUT_4096),
            ("Direct Reference", LOG_PRINT_CURVE_DIRECT),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setData(engine)
            action.setChecked(engine == log_print_curve_engine())
            self.print_curve_engine_group.addAction(action)
            print_curve_menu.addAction(action)
        self.print_curve_engine_group.triggered.connect(self.set_print_curve_engine)

    def _set_developer_invert_mode(self, mode: str) -> None:
        updated = deepcopy(self.adjustments)
        updated.invert_mode = mode
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"Developer invert mode: {invert_mode_label(mode)}")

    def set_print_curve_engine(self, action: QAction) -> None:
        engine = str(action.data())
        set_log_print_curve_engine(engine)
        self._reset_preview_stage_caches()
        if self.current_path is not None:
            self.preview_result_cache.pop(self.current_path, None)
        label = action.text()
        self.statusBar().showMessage(f"Print curve engine: {label}")
        if self.negative_preview_active:
            self._schedule_preview_if_ready()

    def _connect(self) -> None:
        self.control_panel.openRequested.connect(self.open_file)
        self.control_panel.exportRequested.connect(self.export_current)
        self.control_panel.batchExportRequested.connect(self.export_completed)
        self.control_panel.invertRequested.connect(self.preview_inversion)
        self.control_panel.resetRequested.connect(self.reset_workspace)
        self.control_panel.toolChanged.connect(self.set_tool_mode)
        self.control_panel.autoDetectRequested.connect(self.auto_detect_current)
        self.control_panel.adjustmentsChanged.connect(self.adjustments_changed)
        self.control_panel.adjustmentInteractionStarted.connect(self.adjustment_interaction_started)
        self.control_panel.adjustmentInteractionFinished.connect(self.adjustment_interaction_finished)
        self.open_empty_button.clicked.connect(self.open_folder)

        self.origin_view.maskPointSelected.connect(self.mask_point_selected)
        self.origin_view.filmRectSelected.connect(self.film_rect_selected)
        self.origin_view.filmRectReset.connect(self.film_rect_reset)
        self.origin_view.viewStatusChanged.connect(self.statusBar().showMessage)
        self.preview_view.whiteBalancePointSelected.connect(self.white_balance_point_selected)
        self.preview_view.viewStatusChanged.connect(self.statusBar().showMessage)
        self.preview_view.pickerCancelled.connect(self.cancel_white_balance_picker)
        self.preview_view.flipHorizontalRequested.connect(self.flip_preview_horizontal)
        self.preview_view.flipVerticalRequested.connect(self.flip_preview_vertical)
        self.preview_view.rotateClockwiseRequested.connect(self.rotate_preview_clockwise)

        self.filmstrip.fileSelected.connect(self.select_sequence_file)
        self.filmstrip.previousRequested.connect(self.go_previous_file)
        self.filmstrip.nextRequested.connect(self.go_next_file)
        self.preview_refresh_timer.timeout.connect(self.preview_refresh_timeout)
        self.roll_session_save_timer.timeout.connect(self._save_current_state)

    def _build_shortcuts(self) -> None:
        tab_shortcut = QShortcut(QKeySequence(Qt.Key_Tab), self)
        tab_shortcut.activated.connect(self.toggle_preview_tab)

    def set_gpu_preview_enabled(self, enabled: bool) -> None:
        self._gpu_preview_enabled = bool(enabled)
        self.preview_view.set_gpu_preview_enabled(self._gpu_preview_enabled)
        status = "enabled" if self._gpu_preview_enabled else "disabled"
        self.statusBar().showMessage(f"GPU preview acceleration {status}")

    def set_auto_invert_after_frame_change(self, enabled: bool) -> None:
        self._auto_invert_after_frame_change = bool(enabled)
        status = "enabled" if self._auto_invert_after_frame_change else "disabled"
        self.statusBar().showMessage(f"Auto invert after frame change {status}")

    def set_auto_frame_new_negatives(self, enabled: bool) -> None:
        self._auto_frame_new_negatives = bool(enabled)
        status = "enabled" if self._auto_frame_new_negatives else "disabled"
        self.statusBar().showMessage(f"Auto frame new negatives {status}")

    def set_auto_preinvert_nearby_frames(self, enabled: bool) -> None:
        self._auto_preinvert_nearby_frames = bool(enabled)
        if not self._auto_preinvert_nearby_frames:
            self._preinvert_queue = []
        status = "enabled" if self._auto_preinvert_nearby_frames else "disabled"
        self.statusBar().showMessage(f"Auto pre-invert nearby frames {status}")
        if self._auto_preinvert_nearby_frames:
            self._schedule_nearby_preinvert()

    def set_auto_preinvert_radius(self, action: QAction) -> None:
        self._auto_preinvert_radius = int(action.data())
        self._preinvert_queue = []
        self.statusBar().showMessage(
            f"Auto pre-invert range: previous/next {self._auto_preinvert_radius}"
        )
        self._schedule_nearby_preinvert()

    def set_roll_session_autosave(self, enabled: bool) -> None:
        self._roll_session_autosave = bool(enabled)
        status = "enabled" if self._roll_session_autosave else "disabled"
        self.statusBar().showMessage(f"Roll session autosave {status}")

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open RAW or TIFF",
            str(Path.cwd()),
            "RAW and TIFF (*.arw *.raw *.dng *.cr2 *.cr3 *.nef *.raf *.orf *.rw2 *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return

        self.load_path(Path(path), refresh_sequence=True)

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open Folder",
            str(self.current_path.parent if self.current_path is not None else Path.cwd()),
        )
        if not folder:
            return
        files = list_supported_files(Path(folder))
        if not files:
            QMessageBox.information(self, "Open Folder", "No supported RAW or TIFF files were found.")
            return
        self.load_path(files[0], refresh_sequence=True)

    def set_default_export_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Default Export Directory",
            str(self._default_export_dir or (self.current_path.parent if self.current_path else Path.cwd())),
        )
        if not folder:
            return
        self._default_export_dir = Path(folder)
        self.statusBar().showMessage(f"Default export directory: {self._default_export_dir}")

    def _start_frame_ranker_warmup(self) -> None:
        if self._model_warmup_in_progress:
            return
        self._model_warmup_in_progress = True
        self._refresh_activity_progress()
        task = ModelWarmupTask()
        task.signals.finished.connect(self._frame_ranker_warmup_finished)
        self._thread_pool.start(task)

    def _frame_ranker_warmup_finished(self, loaded: bool, model_name: str) -> None:
        self._model_warmup_in_progress = False
        self._refresh_activity_progress()
        if loaded:
            self.statusBar().showMessage(f"Frame model ready: {model_name}")

    def load_path(self, path: Path, *, refresh_sequence: bool = False) -> None:
        if self.current_path == path:
            if refresh_sequence:
                self._set_folder_sequence(path)
            else:
                self._sync_sequence_position(path)
            self.filmstrip.set_current(path)
            return

        if self.current_path is not None and self.current_path != path:
            self._save_current_state()
            self._start_pending_preview_before_switch()

        self._cancel_preview_render()
        self._cancel_raw_preview()
        self._cancel_auto_detect()

        if refresh_sequence:
            self._set_folder_sequence(path)
        else:
            self._sync_sequence_position(path)

        self.current_path = path
        self.view_stack.setCurrentWidget(self.preview_tabs)
        self.filmstrip.show()
        self.current_preview = None
        self.negative_preview_active = False
        self.auto_levels_pending = True
        self.white_balance_point = None
        self._last_untransformed_negative_result = None
        self._reset_preview_transform()
        self._invalidate_negative_base_cache()
        self.preview_view.set_placeholder("Positive preview waiting")
        extension = path.suffix.lower()
        self.control_panel.set_file_status(path.name)
        self.filmstrip.set_current(path)

        if extension in IMAGE_EXTENSIONS:
            loaded = self.image_view.load_image(path)
            self.control_panel.set_image_loaded(loaded)
            if loaded:
                self.control_panel.set_image_status("Image preview")
                self.control_panel.set_histogram(None)
                self.statusBar().showMessage(f"Opened: {path.name}")
            return

        if extension in RAW_EXTENSIONS:
            if self._restore_cached_raw_preview(path):
                return
            self.load_raw_preview(path)
            return

        self.image_view.set_placeholder(f"Unsupported file type: {extension or 'unknown'}")
        self.control_panel.set_image_loaded(False)
        self.control_panel.set_image_status("Unsupported file type")
        self.control_panel.set_histogram(None)

    def load_raw_preview(self, path: Path) -> None:
        self.statusBar().showMessage(f"Generating RAW {DEFAULT_PREVIEW_MAX_EDGE} preview...")
        self.control_panel.set_image_loaded(False)
        self.control_panel.set_image_status("Decoding RAW...")

        self._raw_preview_job_id += 1
        job_id = self._raw_preview_job_id
        task = RawPreviewTask(job_id=job_id, path=path, max_size=DEFAULT_PREVIEW_MAX_EDGE)
        task.signals.finished.connect(self._raw_preview_finished)
        task.signals.failed.connect(self._raw_preview_failed)
        self._raw_preview_in_progress = True
        self._refresh_activity_progress()
        self._thread_pool.start(task)

    def _raw_preview_finished(self, job_id: int, path: Path, preview: RawPreview) -> None:
        if job_id != self._raw_preview_job_id or path != self.current_path:
            return
        self._raw_preview_in_progress = False
        self._refresh_activity_progress()

        self._store_cached_raw_preview(preview)
        self._show_raw_preview(path, preview)
        restored_state = self._restore_state_for_path(path)
        self._maybe_auto_frame_new_negative(restored_state)
        self._schedule_nearby_preinvert()
        if self.negative_preview_active:
            self.statusBar().showMessage(f"Cached positive preview restored: {path.name}")
        else:
            self.statusBar().showMessage(f"RAW preview ready: {path.name}")

    def _raw_preview_failed(self, job_id: int, path: Path, message: str) -> None:
        if job_id != self._raw_preview_job_id or path != self.current_path:
            return
        self._raw_preview_in_progress = False
        self._refresh_activity_progress()
        self.image_view.set_raw_placeholder(path)
        self.control_panel.set_image_loaded(False)
        self.control_panel.set_image_status("RAW preview failed")
        QMessageBox.warning(self, "RAW Preview Failed", message)
        self.statusBar().showMessage("RAW preview failed")

    def _show_raw_preview(self, path: Path, preview: RawPreview) -> None:
        self.current_preview = preview
        pixmap = self._pixmap_from_rgb8(preview.display_rgb8)
        self.control_panel.set_histogram(None)
        self.image_view.set_preview_pixmap(
            pixmap,
            source_path=path,
            source_size=preview.source_size,
        )
        self.control_panel.set_image_loaded(True)
        self.control_panel.set_image_status(preview.status_text())

    def _file_key_for_path(self, path: Path) -> tuple:
        try:
            stat = path.stat()
        except OSError:
            return (path,)
        return (path, stat.st_size, stat.st_mtime_ns)

    def _raw_preview_cache_key(self, path: Path) -> tuple:
        return (self._file_key_for_path(path), DEFAULT_PREVIEW_MAX_EDGE)

    def _store_cached_raw_preview(self, preview: RawPreview) -> None:
        key = self._raw_preview_cache_key(preview.path)
        self.raw_preview_cache.pop(preview.path, None)
        self.raw_preview_cache[preview.path] = CachedRawPreview(
            key=key,
            preview=preview,
        )
        while len(self.raw_preview_cache) > RAW_PREVIEW_CACHE_LIMIT:
            oldest_path = next(iter(self.raw_preview_cache))
            self.raw_preview_cache.pop(oldest_path, None)

    def _restore_cached_raw_preview(self, path: Path) -> bool:
        cached = self.raw_preview_cache.get(path)
        key = self._raw_preview_cache_key(path)
        if cached is None or cached.key != key:
            return False

        self.raw_preview_cache.pop(path, None)
        self.raw_preview_cache[path] = cached
        self._show_raw_preview(path, cached.preview)
        restored_state = self._restore_state_for_path(path)
        self._maybe_auto_frame_new_negative(restored_state)
        self._schedule_nearby_preinvert()
        if self.negative_preview_active:
            self.statusBar().showMessage(f"Cached positive preview restored: {path.name}")
        else:
            self.statusBar().showMessage(f"Cached RAW preview restored: {path.name}")
        return True

    def set_tool_mode(self, mode: ToolMode) -> None:
        if mode == ToolMode.WB_PICKER:
            self.preview_view.set_tool_mode(mode)
            self.origin_view.set_tool_mode(ToolMode.PAN)
            self.preview_tabs.setCurrentWidget(self.preview_view)
            if not self.negative_preview_active:
                self.statusBar().showMessage("Generate a positive preview before picking white balance")
            return

        self.origin_view.set_tool_mode(mode)
        self.preview_view.set_tool_mode(ToolMode.PAN)
        self.preview_tabs.setCurrentWidget(self.origin_view)

    def cancel_white_balance_picker(self) -> None:
        self.preview_view.set_tool_mode(ToolMode.PAN)
        self.statusBar().showMessage("WB picker cancelled")

    def mask_point_selected(self, point: ImagePoint) -> None:
        self.mask_point = point
        self.white_balance_point = None
        self.auto_levels_pending = True
        self._invalidate_negative_base_cache()
        self.control_panel.set_mask_status(f"Base point: x={point.x}, y={point.y}")
        self.statusBar().showMessage("Base picker point saved")
        self._schedule_preview_if_ready()

    def film_rect_selected(self, rect: ImageRect) -> None:
        self.film_rect = rect
        self.white_balance_point = None
        self.auto_levels_pending = True
        self._invalidate_negative_base_cache()
        self.control_panel.set_film_status(f"Frame: {rect.label()}")
        self.statusBar().showMessage("Frame area saved")
        self._schedule_preview_if_ready()

    def film_rect_reset(self) -> None:
        self.film_rect = None
        self.white_balance_point = None
        self.auto_levels_pending = True
        self.negative_preview_active = False
        self._last_untransformed_negative_result = None
        self._invalidate_negative_base_cache()
        self.control_panel.set_film_status("Not selected")
        self.preview_view.set_placeholder("Positive preview waiting")
        self.statusBar().showMessage("Frame reset")

    def white_balance_point_selected(self, point: ImagePoint) -> None:
        if not self.negative_preview_active or self.last_negative_result is None:
            self.white_balance_point = None
            self.preview_view.restore_selections(
                mask_point=None,
                film_rect=None,
                white_balance_point=None,
            )
            self.statusBar().showMessage("Generate a positive preview before using the WB picker")
            return

        try:
            global_balance, sample_rgb, gains = suggest_global_balance_from_neutral(
                self.last_negative_result.color_balanced_linear_rgb,
                point,
                self.adjustments.color_balance.global_balance,
            )
        except PipelineError as exc:
            self.statusBar().showMessage(str(exc))
            return

        self.white_balance_point = point
        updated = deepcopy(self.adjustments)
        updated.color_balance.global_balance = global_balance
        self.control_panel.set_adjustments(updated, emit=True)

        sample_text = ", ".join(f"{value:.3f}" for value in sample_rgb)
        gain_text = ", ".join(f"{value:.3f}" for value in gains)
        self.statusBar().showMessage(
            f"WB picker: x={point.x}, y={point.y}, sample RGB {sample_text}, gains {gain_text}"
        )

    def auto_detect_current(self, mode: str, *, auto_preview: bool = False) -> None:
        if self.current_preview is None:
            if not auto_preview:
                QMessageBox.information(self, "Auto Detect", "Open a RAW file before auto detection.")
            return
        if self._auto_detect_in_progress:
            self.statusBar().showMessage("Auto detect already running...")
            return

        self._auto_detect_job_id += 1
        job_id = self._auto_detect_job_id
        if auto_preview:
            self._auto_detect_auto_preview_jobs.add(job_id)
        task = AutoDetectTask(
            job_id=job_id,
            mode=mode,
            path=self.current_path,
            preview=self.current_preview,
            format_hint=self.control_panel.auto_format(),
            detect_base=self._film_base_required_for_current_mode(),
            current_film_rect=self.film_rect,
            fallback_state=self._previous_image_state(),
            auto_preview=auto_preview,
        )
        task.signals.finished.connect(self._auto_detect_finished)
        task.signals.failed.connect(self._auto_detect_failed)
        self._auto_detect_in_progress = True
        self._refresh_activity_progress()
        self.statusBar().showMessage(
            "Auto frame running in background..."
            if auto_preview
            else "Auto detect running in background..."
        )
        self._thread_pool.start(task)

    def _auto_detect_finished(self, job_id: int, output: AutoDetectOutput) -> None:
        if job_id != self._auto_detect_job_id:
            self._auto_detect_auto_preview_jobs.discard(job_id)
            return
        self._auto_detect_auto_preview_jobs.discard(job_id)
        self._auto_detect_in_progress = False
        self._refresh_activity_progress()
        if output.path != self.current_path:
            self.statusBar().showMessage("Auto detect result ignored after file change")
            return

        self._apply_auto_detect_output(output)

    def _auto_detect_failed(self, job_id: int, message: str) -> None:
        if job_id != self._auto_detect_job_id:
            self._auto_detect_auto_preview_jobs.discard(job_id)
            return
        auto_preview = job_id in self._auto_detect_auto_preview_jobs
        self._auto_detect_auto_preview_jobs.discard(job_id)
        self._auto_detect_in_progress = False
        self._refresh_activity_progress()
        self.statusBar().showMessage("Auto detect failed")
        if not auto_preview:
            QMessageBox.warning(self, "Auto Detect Failed", message)

    def _apply_auto_detect_output(self, output: AutoDetectOutput) -> None:
        mode = output.mode
        frame_result = output.frame_result
        base_result = output.base_result

        used_fallback = False
        fallback_state = output.fallback_state
        if mode in {"frame", "frame_base"} and frame_result is None and fallback_state is not None:
            frame_result = AutoFrameResult(
                rect=fallback_state.film_rect,
                confidence=0.35,
                confidence_level="fallback",
                format_hint="previous",
                method="previous-image",
            ) if fallback_state.film_rect is not None else None
            used_fallback = frame_result is not None
        if mode in {"base", "frame_base"} and base_result is None and fallback_state is not None:
            base_result = AutoBaseResult(
                point=fallback_state.mask_point,
                rgb=None,
                confidence=0.35,
                confidence_level="fallback",
                source="previous-image",
            ) if fallback_state.mask_point is not None else None
            used_fallback = used_fallback or base_result is not None

        changed = False
        details: list[str] = []
        if (
            frame_result is not None
            and mode in {"frame", "frame_base"}
            and frame_result.confidence_level in {"high", "fallback"}
        ):
            self.film_rect = frame_result.rect
            self.control_panel.set_film_status(
                f"Frame: {frame_result.rect.label()}\nAuto {frame_result.confidence_level} "
                f"{frame_result.confidence:.2f}, {frame_result.format_hint}, {frame_result.method}"
            )
            details.append(f"frame {frame_result.confidence_level} {frame_result.confidence:.2f}")
            changed = True
        elif frame_result is not None and mode in {"frame", "frame_base"}:
            details.append(f"frame candidate rejected {frame_result.confidence:.2f}")
        if base_result is not None and mode in {"base", "frame_base"} and base_result.point is not None:
            self.mask_point = base_result.point
            self.control_panel.set_mask_status(
                f"Base point: x={base_result.point.x}, y={base_result.point.y}\nAuto "
                f"{base_result.confidence_level} {base_result.confidence:.2f}, {base_result.source}"
            )
            details.append(f"base {base_result.confidence_level} {base_result.confidence:.2f}")
            changed = True

        if not changed:
            self.statusBar().showMessage("Auto detect failed. Please select frame/base manually.")
            if not output.auto_preview:
                QMessageBox.information(self, "Auto Detect", "No reliable frame or base was detected. Please select manually.")
            return

        self.white_balance_point = None
        self.auto_levels_pending = True
        self._invalidate_negative_base_cache()
        self.origin_view.restore_selections(
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            white_balance_point=None,
        )
        self.preview_tabs.setCurrentWidget(self.origin_view)
        suffix = " using previous image fallback" if used_fallback else ""
        self.statusBar().showMessage(f"Auto detect applied: {', '.join(details)}{suffix}")
        self._schedule_preview_if_ready(force=output.auto_preview)
        self._schedule_nearby_preinvert()

    def adjustment_interaction_started(self) -> None:
        self._interactive_adjustment_active = True
        self.preview_refresh_timer.setInterval(INTERACTIVE_RENDER_DEBOUNCE_MS)

    def adjustment_interaction_finished(self) -> None:
        was_interactive = self._interactive_adjustment_active
        self._interactive_adjustment_active = False
        self.preview_refresh_timer.setInterval(FINAL_RENDER_DEBOUNCE_MS)
        if not was_interactive or not self.negative_preview_active:
            return

        self.preview_refresh_timer.stop()
        if self._render_in_progress:
            self._render_pending = True
            self._render_pending_show_errors = False
            return
        self._queue_negative_render(show_errors=False, interactive=False)

    def preview_refresh_timeout(self) -> None:
        self._queue_negative_render(
            show_errors=False,
            interactive=self._interactive_adjustment_active,
        )

    def adjustments_changed(self, values: dict) -> None:
        previous = self.adjustments
        self.adjustments = AdjustmentParams(**values)
        mode_changed = previous.invert_mode != self.adjustments.invert_mode
        if mode_changed:
            self.auto_levels_pending = True
        if (
            not self._applying_auto_levels
            and not mode_changed
            and (
                previous.black_point != self.adjustments.black_point
                or previous.mid_point != self.adjustments.mid_point
                or previous.white_point != self.adjustments.white_point
            )
        ):
            self.auto_levels_pending = False
        self.statusBar().showMessage(
            "Adjust: mode {invert_mode}, curve {print_curve}, WB {auto_wb}, exposure {exposure}, highlights {highlights}, shadows {shadows}, contrast {contrast}, saturation {saturation}, boundary {analysis_inset_percent}%, camera color {camera_color_strength}, black {black_point}, mid {mid_point}, white {white_point}".format(
                **values
            )
        )
        if self._apply_gpu_display_adjustment(previous):
            self._schedule_roll_session_save()
            return
        if self.negative_preview_active:
            if self._render_in_progress:
                self._render_pending = True
            self.preview_refresh_timer.setInterval(
                INTERACTIVE_RENDER_DEBOUNCE_MS
                if self._interactive_adjustment_active
                else FINAL_RENDER_DEBOUNCE_MS
            )
            self.preview_refresh_timer.start()
        self._schedule_roll_session_save()

    def _apply_gpu_display_adjustment(self, previous: AdjustmentParams) -> bool:
        del previous
        # The experimental shader path used an 8-bit linear texture, which can
        # quantize deep shadows before display gamma and bring back black-field
        # artifacts. Keep final preview/display on the CPU pipeline for now.
        return False

    def preview_inversion(self) -> None:
        self._queue_negative_render(show_errors=True, interactive=False)

    def _queue_negative_render(self, *, show_errors: bool, interactive: bool = False) -> bool:
        if self.current_preview is None:
            if show_errors:
                QMessageBox.information(self, "Invert Preview", "Open a RAW file and generate a linear preview first.")
            return False
        if self.film_rect is None or not self.film_rect.is_valid():
            if show_errors:
                QMessageBox.information(self, "Invert Preview", "Select a valid frame area first.")
            return False
        if self._film_base_required_for_current_mode() and self.mask_point is None:
            if show_errors:
                QMessageBox.information(self, "Invert Preview", "Pick the film base before using this inversion mode.")
            return False

        interactive = bool(interactive and not show_errors and not self.auto_levels_pending)
        quality = INTERACTIVE_RENDER_QUALITY if interactive else FINAL_RENDER_QUALITY
        render_preview = self._preview_for_render(interactive=interactive)

        if self._render_in_progress:
            self._render_pending = True
            self._render_pending_show_errors = self._render_pending_show_errors or show_errors
            self.statusBar().showMessage("Preview render queued...")
            return True

        self._render_job_id += 1
        job_id = self._render_job_id
        task = PreviewRenderTask(
            job_id=job_id,
            preview=render_preview,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=self.adjustments,
            auto_levels_pending=self.auto_levels_pending,
            show_errors=show_errors,
            quality=quality,
            file_key=self._file_key_for_path(render_preview.path),
            render_cache=self._preview_stage_caches[quality],
        )
        task.signals.finished.connect(self._preview_render_finished)
        task.signals.failed.connect(self._preview_render_failed)
        self._render_in_progress = True
        self.statusBar().showMessage(
            "Rendering interactive preview..."
            if interactive
            else "Rendering preview..."
        )
        self._thread_pool.start(task)
        return True

    def _preview_render_finished(self, job_id: int, output: PreviewRenderOutput, show_errors: bool) -> None:
        if job_id != self._render_job_id:
            self._store_stale_preview_result(output)
            return

        self._render_in_progress = False
        self._preview_stage_caches[output.quality] = output.cache
        result = output.result

        if self._render_pending:
            pending_show_errors = self._render_pending_show_errors
            self._render_pending = False
            self._render_pending_show_errors = False
            self._queue_negative_render(
                show_errors=pending_show_errors,
                interactive=self._interactive_adjustment_active and not pending_show_errors,
            )
            return

        if self.auto_levels_pending and output.applied_auto_levels:
            self._apply_auto_levels(result.auto_levels)
            self.auto_levels_pending = False
        elif self.auto_levels_pending:
            self._apply_auto_levels(result.auto_levels)
            self.auto_levels_pending = False
            self._render_pending = False
            self._render_pending_show_errors = False
            self._queue_negative_render(show_errors=False, interactive=False)
            return

        self._last_untransformed_negative_result = result
        displayed_result, _pixmap = self._update_preview_from_result(
            result,
            update_filmstrip=output.quality == FINAL_RENDER_QUALITY,
        )
        if output.quality == FINAL_RENDER_QUALITY:
            self._store_cached_preview_result(result)
        self._set_negative_preview_status(result, displayed_result)
        self.control_panel.set_histogram(displayed_result.histogram)
        self.negative_preview_active = True
        if show_errors:
            self.preview_tabs.setCurrentWidget(self.preview_view)
        self.statusBar().showMessage(
            "Interactive preview ready"
            if output.quality == INTERACTIVE_RENDER_QUALITY
            else "Inverted preview ready"
        )
        self._schedule_roll_session_save()

    def _start_pending_preview_before_switch(self) -> None:
        if not self.preview_refresh_timer.isActive():
            return
        self.preview_refresh_timer.stop()
        self._queue_negative_render(show_errors=False, interactive=False)

    def _store_stale_preview_result(self, output: PreviewRenderOutput) -> None:
        if output.quality != FINAL_RENDER_QUALITY or output.cache_key is None:
            return
        if output.path == self.current_path:
            return

        self.preview_result_cache.pop(output.path, None)
        self.preview_result_cache[output.path] = CachedPreviewResult(
            key=output.cache_key,
            result=output.result,
        )
        while len(self.preview_result_cache) > PREVIEW_RESULT_CACHE_LIMIT:
            oldest_path = next(iter(self.preview_result_cache))
            self.preview_result_cache.pop(oldest_path, None)

        existing = self.image_states.get(output.path)
        self.image_states[output.path] = ImageProcessingState(
            mask_point=output.mask_point if output.mask_point is not None else (existing.mask_point if existing else None),
            film_rect=output.film_rect if output.film_rect is not None else (existing.film_rect if existing else None),
            white_balance_point=existing.white_balance_point if existing else None,
            adjustments=deepcopy(output.adjustments) if output.adjustments is not None else (deepcopy(existing.adjustments) if existing else self._default_adjustments()),
            negative_preview_active=True,
            auto_levels_pending=False,
            preview_flip_horizontal=existing.preview_flip_horizontal if existing else False,
            preview_flip_vertical=existing.preview_flip_vertical if existing else False,
            preview_rotation_quarters=existing.preview_rotation_quarters if existing else 0,
        )
        self.filmstrip.set_processed_thumbnail(
            output.path,
            self._pixmap_from_rgb8(output.result.display_rgb8),
        )
        self.statusBar().showMessage(f"Background preview cached: {output.path.name}")
        self._autosave_roll_session()

    def _preview_render_failed(self, job_id: int, message: str, show_errors: bool) -> None:
        if job_id != self._render_job_id:
            return

        self._render_in_progress = False
        self.last_negative_result = None
        self._last_untransformed_negative_result = None
        if show_errors:
            QMessageBox.warning(self, "Invert Preview Failed", message)
        self.statusBar().showMessage("Inverted preview failed")

        if self._render_pending:
            pending_show_errors = self._render_pending_show_errors
            self._render_pending = False
            self._render_pending_show_errors = False
            self._queue_negative_render(
                show_errors=pending_show_errors,
                interactive=self._interactive_adjustment_active and not pending_show_errors,
            )

    def _schedule_preview_if_ready(self, *, force: bool = False) -> None:
        if not force and not self._auto_invert_after_frame_change:
            return
        if self.current_preview is None:
            return
        if self.film_rect is None or not self.film_rect.is_valid():
            return
        if self._film_base_required_for_current_mode() and self.mask_point is None:
            return
        self.preview_refresh_timer.start()

    def _film_base_required_for_current_mode(self) -> bool:
        return self.adjustments.invert_mode != InvertMode.LAB_PRINT.value

    def _preview_for_render(self, *, interactive: bool) -> RawPreview:
        if self.current_preview is None:
            raise PipelineError("Open a RAW file and generate a linear preview first.")
        if not interactive:
            return self.current_preview

        key = (
            self.current_preview.path,
            id(self.current_preview.preview_linear_rgb),
            id(self.current_preview.preview_camera_wb_linear_rgb),
            self.current_preview.preview_size,
            INTERACTIVE_PREVIEW_MAX_EDGE,
        )
        if self._interactive_preview_cache_key == key and self._interactive_preview_cache is not None:
            return self._interactive_preview_cache

        self._interactive_preview_cache = scaled_raw_preview(
            self.current_preview,
            max_edge=INTERACTIVE_PREVIEW_MAX_EDGE,
        )
        self._interactive_preview_cache_key = key
        return self._interactive_preview_cache

    def _update_preview_from_result(
        self,
        result: NegativePreviewResult,
        *,
        update_filmstrip: bool = True,
    ) -> tuple[NegativePreviewResult, QPixmap]:
        displayed_result = self._transformed_negative_result(result)
        pixmap = self._pixmap_from_rgb8(displayed_result.display_rgb8)
        source_path = self.current_path or (self.current_preview.path if self.current_preview else "")
        self.preview_view.set_preview_pixmap(
            pixmap,
            source_path=source_path,
            source_size=ImageSize(width=displayed_result.width, height=displayed_result.height),
            reset_navigation=False,
        )
        self.preview_view.restore_selections(
            mask_point=None,
            film_rect=None,
            white_balance_point=self.white_balance_point,
        )
        self.origin_view.restore_selections(
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            white_balance_point=None,
        )
        self.last_negative_result = displayed_result
        if update_filmstrip:
            self.filmstrip.set_processed_thumbnail(self.current_path, pixmap)
        return displayed_result, pixmap

    def _transformed_negative_result(self, result: NegativePreviewResult) -> NegativePreviewResult:
        return replace(
            result,
            display_rgb8=self._transform_preview_array(result.display_rgb8),
            processed_linear_rgb=self._transform_preview_array(result.processed_linear_rgb),
            color_balanced_linear_rgb=self._transform_preview_array(result.color_balanced_linear_rgb),
        )

    def _transform_preview_array(self, image: np.ndarray) -> np.ndarray:
        transformed = image
        if self._preview_flip_horizontal:
            transformed = np.flip(transformed, axis=1)
        if self._preview_flip_vertical:
            transformed = np.flip(transformed, axis=0)
        if self._preview_rotation_quarters:
            transformed = np.rot90(transformed, k=-self._preview_rotation_quarters)
        return np.ascontiguousarray(transformed)

    def flip_preview_horizontal(self) -> None:
        self._preview_flip_horizontal = not self._preview_flip_horizontal
        self._refresh_preview_transform("Preview flipped horizontally")

    def flip_preview_vertical(self) -> None:
        self._preview_flip_vertical = not self._preview_flip_vertical
        self._refresh_preview_transform("Preview flipped vertically")

    def rotate_preview_clockwise(self) -> None:
        self._preview_rotation_quarters = (self._preview_rotation_quarters + 1) % 4
        self._refresh_preview_transform("Preview rotated 90 degrees")

    def _refresh_preview_transform(self, status: str) -> None:
        if self._last_untransformed_negative_result is None:
            self.statusBar().showMessage("Generate a positive preview first")
            return
        self._update_preview_from_result(self._last_untransformed_negative_result)
        self.statusBar().showMessage(status)

    def _reset_preview_transform(self) -> None:
        self._preview_flip_horizontal = False
        self._preview_flip_vertical = False
        self._preview_rotation_quarters = 0

    def _preview_file_key(self) -> tuple | None:
        if self.current_path is None:
            return None
        return self._file_key_for_path(self.current_path)

    def _adjustments_preview_cache_key(self, adjustments: AdjustmentParams) -> tuple:
        return adjustments_preview_cache_key(adjustments)

    def _preview_result_cache_key(self) -> tuple | None:
        if self.current_preview is None:
            return None
        file_key = self._preview_file_key()
        if file_key is None:
            return None
        return preview_result_cache_key_for(
            file_key=file_key,
            preview=self.current_preview,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=self.adjustments,
        )

    def _store_cached_preview_result(self, result: NegativePreviewResult) -> None:
        if self.current_path is None or self.auto_levels_pending:
            return
        key = self._preview_result_cache_key()
        if key is None:
            return

        self.preview_result_cache.pop(self.current_path, None)
        self.preview_result_cache[self.current_path] = CachedPreviewResult(
            key=key,
            result=result,
        )
        while len(self.preview_result_cache) > PREVIEW_RESULT_CACHE_LIMIT:
            oldest_path = next(iter(self.preview_result_cache))
            self.preview_result_cache.pop(oldest_path, None)

    def _restore_cached_preview_result(self) -> bool:
        if self.current_path is None or self.auto_levels_pending:
            return False
        cached = self.preview_result_cache.get(self.current_path)
        key = self._preview_result_cache_key()
        if cached is None or key is None or cached.key != key:
            return False

        self.preview_result_cache.pop(self.current_path, None)
        self.preview_result_cache[self.current_path] = cached
        self._last_untransformed_negative_result = cached.result
        displayed_result, _pixmap = self._update_preview_from_result(
            cached.result,
            update_filmstrip=True,
        )
        self.control_panel.set_histogram(displayed_result.histogram)
        self._set_negative_preview_status(cached.result, displayed_result)
        self.negative_preview_active = True
        self.statusBar().showMessage("Cached positive preview restored")
        return True

    def _set_negative_preview_status(
        self,
        result: NegativePreviewResult,
        displayed_result: NegativePreviewResult,
    ) -> None:
        if self.mask_point is None and self.adjustments.invert_mode == InvertMode.LAB_PRINT.value:
            base_text = "Base fallback: none"
        else:
            mask_rgb = ", ".join(f"{value:.4f}" for value in result.mask_rgb)
            base_text = f"Base RGB {mask_rgb}"
        wb_gains = ", ".join(f"{value:.3f}" for value in displayed_result.wb_gains)
        wb_label = (
            "WB CMY offset"
            if self.adjustments.invert_mode
            in (InvertMode.LOG_BOUNDS.value, InvertMode.LAB_PRINT.value)
            else "WB gain"
        )
        self.control_panel.set_image_status(
            f"Positive preview {displayed_result.width} x {displayed_result.height}\n"
            f"Mode {invert_mode_label(self.adjustments.invert_mode)}\n"
            f"{base_text}\n"
            f"{wb_label} {wb_gains}"
        )

    def _negative_base_for_current(self) -> NegativeBasePreview:
        if self.current_preview is None:
            raise PipelineError("Open a RAW file and generate a linear preview first.")

        key: tuple[object, ...] = (
            self.current_preview.path,
            self.current_preview.source_size,
            self.current_preview.preview_linear_rgb.shape,
            id(self.current_preview.preview_camera_wb_linear_rgb),
            id(self.current_preview.camera_to_srgb_matrix),
            self.mask_point,
            self.film_rect,
            lens_correction_key(self.adjustments.lens_correction),
        )
        if self._negative_base_cache_key == key and self._negative_base_cache is not None:
            return self._negative_base_cache

        base = build_negative_base_preview(
            self.current_preview.preview_linear_rgb,
            source_size=self.current_preview.source_size,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            lens_correction=self.adjustments.lens_correction,
            preview_camera_wb_linear_rgb=self.current_preview.preview_camera_wb_linear_rgb,
            camera_to_srgb_matrix=self.current_preview.camera_to_srgb_matrix,
        )
        self._negative_base_cache = base
        self._negative_base_cache_key = key
        return base

    def _invalidate_negative_base_cache(self) -> None:
        self._negative_base_cache = None
        self._negative_base_cache_key = None
        self._reset_preview_stage_caches()
        self._invalidate_density_analysis_cache()
        self.last_negative_result = None
        self._last_untransformed_negative_result = None

    def _reset_preview_stage_caches(self) -> None:
        self._preview_stage_caches = {
            FINAL_RENDER_QUALITY: PreviewStageCache(),
            INTERACTIVE_RENDER_QUALITY: PreviewStageCache(),
        }
        self._interactive_preview_cache_key = None
        self._interactive_preview_cache = None

    def _density_analysis_for_current(self, base: NegativeBasePreview) -> DensityPreviewAnalysis:
        key: tuple[object, ...] = (
            self._negative_base_cache_key,
            self.adjustments.print_curve,
            self._density_matrix_key(self.adjustments.density_matrix),
        )
        if self._density_analysis_cache_key == key and self._density_analysis_cache is not None:
            return self._density_analysis_cache

        analysis = build_density_preview_analysis(base, self.adjustments)
        self._density_analysis_cache = analysis
        self._density_analysis_cache_key = key
        return analysis

    def _invalidate_density_analysis_cache(self) -> None:
        self._density_analysis_cache = None
        self._density_analysis_cache_key = None

    def _density_matrix_key(self, matrix: DensityMatrixParams) -> tuple[float, ...]:
        return (
            matrix.m00,
            matrix.m01,
            matrix.m02,
            matrix.m10,
            matrix.m11,
            matrix.m12,
            matrix.m20,
            matrix.m21,
            matrix.m22,
        )

    def export_current(self) -> None:
        if self._export_in_progress:
            self.statusBar().showMessage("Export already in progress")
            return

        if self.current_path is None or self.current_path.suffix.lower() not in RAW_EXTENSIONS:
            QMessageBox.information(self, "Export TIFF", "Open a RAW file before exporting.")
            return
        if self._film_base_required_for_current_mode() and self.mask_point is None:
            QMessageBox.information(self, "Export TIFF", "Pick the film base before exporting.")
            return
        if self.film_rect is None or not self.film_rect.is_valid():
            QMessageBox.information(self, "Export TIFF", "Select a valid frame area before exporting.")
            return

        if self._default_export_dir is not None:
            default_path = self._default_export_dir / f"{self.current_path.stem}_positive.tif"
        else:
            default_path = self.current_path.with_name(f"{self.current_path.stem}_positive.tif")
        output, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export 16-bit TIFF",
            str(default_path),
            "TIFF Image (*.tif *.tiff)",
        )
        if not output:
            return

        output_path = Path(output)
        if output_path.suffix.lower() not in {".tif", ".tiff"}:
            output_path = output_path.with_suffix(".tif")

        self._export_cancel_event = Event()
        task = TiffExportTask(
            source_path=self.current_path,
            output_path=output_path,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=self.adjustments,
            flip_horizontal=self._preview_flip_horizontal,
            flip_vertical=self._preview_flip_vertical,
            rotation_quarters=self._preview_rotation_quarters,
            auto_levels_pending=self.auto_levels_pending,
            preview_cmy_offsets=self._current_preview_cmy_offsets_for_export(),
            cancel_event=self._export_cancel_event,
        )
        task.signals.finished.connect(self._export_finished)
        task.signals.failed.connect(self._export_failed)
        task.signals.cancelled.connect(self._export_cancelled)
        task.signals.progress.connect(self._export_progress_updated)
        self._export_in_progress = True
        self.control_panel.export_button.setEnabled(False)
        self.control_panel.batch_export_button.setEnabled(False)
        self.control_panel.set_export_progress(True, value=0, text="Starting export")
        self.statusBar().showMessage("Exporting 16-bit TIFF...")
        self._thread_pool.start(task)

    def export_completed(self) -> None:
        if self._export_in_progress:
            self.statusBar().showMessage("Export already in progress")
            return
        if self.current_path is not None:
            self._save_current_state()

        output_dir_text = QFileDialog.getExistingDirectory(
            self,
            "Export Completed TIFFs",
            str(self._default_export_dir or (self.current_path.parent if self.current_path else Path.cwd())),
        )
        if not output_dir_text:
            return

        output_dir = Path(output_dir_text)
        items = self._completed_export_items(output_dir)
        if not items:
            QMessageBox.information(
                self,
                "Export Completed TIFFs",
                "No completed RAW positives were found in the current sequence.",
            )
            return

        self._batch_export_queue = items
        self._batch_export_total = len(items)
        self._batch_export_done = 0
        self._batch_export_active = True
        self._batch_export_paused = False
        self._batch_export_cancel_requested = False
        self._export_in_progress = True
        self._export_cancel_event = Event()
        self.batch_export_dialog.set_jobs([item["source_path"] for item in items])
        self.batch_export_dialog.show()
        self.batch_export_dialog.raise_()
        self.control_panel.export_button.setEnabled(False)
        self.control_panel.batch_export_button.setEnabled(False)
        self.control_panel.set_export_progress(True, value=0, text="Starting batch export")
        self.statusBar().showMessage(f"Exporting {self._batch_export_total} completed TIFFs...")
        self._start_next_batch_export()

    def _completed_export_items(self, output_dir: Path) -> list[dict]:
        ordered_paths = list(self.folder_files) if self.folder_files else list(self.image_states)
        for path in self.image_states:
            if path not in ordered_paths:
                ordered_paths.append(path)

        items: list[dict] = []
        for path in ordered_paths:
            state = self.image_states.get(path)
            if state is None or not state.negative_preview_active:
                continue
            if path.suffix.lower() not in RAW_EXTENSIONS:
                continue
            if state.film_rect is None or not state.film_rect.is_valid():
                continue
            items.append(
                {
                    "source_path": path,
                    "output_path": output_dir / f"{path.stem}_positive.tif",
                    "mask_point": state.mask_point,
                    "film_rect": state.film_rect,
                    "adjustments": deepcopy(state.adjustments),
                    "flip_horizontal": state.preview_flip_horizontal,
                    "flip_vertical": state.preview_flip_vertical,
                    "rotation_quarters": state.preview_rotation_quarters,
                    "auto_levels_pending": state.auto_levels_pending,
                    "preview_cmy_offsets": self._preview_cmy_offsets_for_path(path, state),
                }
            )
        return items

    def _preview_cmy_offsets_for_path(self, path: Path, state: ImageProcessingState) -> np.ndarray | None:
        if state.adjustments.invert_mode != InvertMode.LAB_PRINT.value or not state.adjustments.auto_wb:
            return None
        cached = self.preview_result_cache.get(path)
        if cached is None:
            return None
        return np.asarray(cached.result.wb_gains, dtype=np.float32).copy()

    def _start_next_batch_export(self) -> None:
        if self._batch_export_cancel_requested:
            self._finish_batch_export(
                f"Batch export cancelled after {self._batch_export_done}/{self._batch_export_total}",
                auto_close_ms=None,
                restart_preinvert=True,
            )
            return
        if self._batch_export_paused:
            self._batch_export_current_path = None
            self.batch_export_dialog.current_label.setText("Paused")
            self.batch_export_dialog.update_progress(
                round(self._batch_export_done / max(1, self._batch_export_total) * 100),
                f"Paused after {self._batch_export_done}/{self._batch_export_total}",
            )
            self.batch_export_dialog.set_running(True, paused=True)
            self.control_panel.update_export_progress(
                round(self._batch_export_done / max(1, self._batch_export_total) * 100),
                "Batch export paused",
            )
            self.statusBar().showMessage("Batch export paused")
            return
        if not self._batch_export_queue:
            self._finish_batch_export(
                f"Batch exported {self._batch_export_done} TIFFs",
                auto_close_ms=1200,
                restart_preinvert=True,
            )
            return

        item = self._batch_export_queue.pop(0)
        self._batch_export_current_path = item["source_path"]
        self.batch_export_dialog.set_current(self._batch_export_current_path)
        self.batch_export_dialog.set_running(True, paused=False)
        self._export_cancel_event = Event()
        item = dict(item)
        item["cancel_event"] = self._export_cancel_event
        task = TiffExportTask(**item)
        task.signals.finished.connect(self._export_finished)
        task.signals.failed.connect(self._export_failed)
        task.signals.cancelled.connect(self._export_cancelled)
        task.signals.progress.connect(self._export_progress_updated)
        self._thread_pool.start(task)

    def pause_batch_export(self) -> None:
        if not self._batch_export_active:
            return
        self._batch_export_paused = True
        self.batch_export_dialog.set_running(True, paused=True)
        self.statusBar().showMessage("Batch export will pause after the current TIFF")

    def resume_batch_export(self) -> None:
        if not self._batch_export_active or not self._batch_export_paused:
            return
        self._batch_export_paused = False
        self.batch_export_dialog.set_running(True, paused=False)
        self.statusBar().showMessage("Batch export resumed")
        if self._batch_export_current_path is None:
            self._start_next_batch_export()

    def cancel_batch_export(self) -> None:
        if not self._batch_export_active and not self._export_in_progress:
            return
        self._batch_export_cancel_requested = True
        self._batch_export_paused = False
        self._batch_export_queue = []
        if self._export_cancel_event is not None:
            self._export_cancel_event.set()
        self.batch_export_dialog.set_running(False)
        self.batch_export_dialog.update_progress(
            round(self._batch_export_done / max(1, self._batch_export_total) * 100),
            "Cancelling...",
        )
        self.statusBar().showMessage("Cancelling export...")
        if self._batch_export_current_path is None:
            self._finish_batch_export(
                f"Batch export cancelled after {self._batch_export_done}/{self._batch_export_total}",
                auto_close_ms=None,
                restart_preinvert=True,
            )

    def _finish_batch_export(
        self,
        text: str,
        *,
        auto_close_ms: int | None,
        restart_preinvert: bool,
    ) -> None:
        self._batch_export_active = False
        self._batch_export_paused = False
        self._batch_export_cancel_requested = False
        self._export_in_progress = False
        self._batch_export_current_path = None
        self._export_cancel_event = None
        self.control_panel.update_export_progress(100, text)
        self.control_panel.set_export_progress(False)
        self.control_panel.export_button.setEnabled(True)
        self.control_panel.batch_export_button.setEnabled(True)
        self.batch_export_dialog.finish(text, auto_close_ms=auto_close_ms)
        self.statusBar().showMessage(text)
        if restart_preinvert:
            self._start_next_preinvert_jobs()

    def _current_preview_cmy_offsets_for_export(self) -> np.ndarray | None:
        if self.current_path is None:
            return None
        if self.adjustments.invert_mode != InvertMode.LAB_PRINT.value or not self.adjustments.auto_wb:
            return None
        if self.auto_levels_pending:
            return None

        # Reuse preview CMY only when the final preview cache exactly matches
        # the current image/selection/adjustments. Otherwise export recomputes
        # auto WB at full resolution instead of risking stale color timing.
        cached = self.preview_result_cache.get(self.current_path)
        key = self._preview_result_cache_key()
        if cached is None or key is None or cached.key != key:
            return None
        return np.asarray(cached.result.wb_gains, dtype=np.float32).copy()

    def _export_progress_updated(self, value: int, text: str) -> None:
        if self._batch_export_active and self._batch_export_total:
            text = f"Batch {self._batch_export_done + 1}/{self._batch_export_total}: {text}"
            self.batch_export_dialog.update_progress(value, text)
        self.control_panel.update_export_progress(value, text)
        self.statusBar().showMessage(f"{text}...")

    def _export_finished(self, output_path: str, timings: dict[str, float]) -> None:
        if self._batch_export_active:
            self._batch_export_done += 1
            if self._batch_export_current_path is not None:
                self.batch_export_dialog.mark_done(self._batch_export_current_path)
            timing_text = self._format_export_timings(timings)
            if timing_text:
                print(f"Export timings: {Path(output_path).name}: {timing_text}", flush=True)
            progress = round(self._batch_export_done / max(1, self._batch_export_total) * 100)
            self.control_panel.update_export_progress(
                progress,
                f"Batch exported {self._batch_export_done}/{self._batch_export_total}",
            )
            self.statusBar().showMessage(
                f"Batch exported {self._batch_export_done}/{self._batch_export_total}: {Path(output_path).name}"
            )
            self._start_next_batch_export()
            return

        self._export_in_progress = False
        self._export_cancel_event = None
        timing_text = self._format_export_timings(timings)
        complete_text = f"Export complete ({timing_text})" if timing_text else "Export complete"
        self.control_panel.update_export_progress(100, complete_text)
        self.control_panel.set_export_progress(False)
        self.control_panel.export_button.setEnabled(True)
        self.control_panel.batch_export_button.setEnabled(True)
        suffix = f" | {timing_text}" if timing_text else ""
        self.statusBar().showMessage(f"Exported TIFF: {output_path}{suffix}")
        if timing_text:
            print(f"Export timings: {timing_text}", flush=True)
        self._start_next_preinvert_jobs()

    @staticmethod
    def _format_export_timings(timings: dict[str, float]) -> str:
        if not timings:
            return ""
        total = sum(float(seconds) for seconds in timings.values())
        parts = [f"{name} {float(seconds):.1f}s" for name, seconds in timings.items()]
        parts.append(f"total {total:.1f}s")
        return ", ".join(parts)

    def _export_failed(self, message: str) -> None:
        if self._batch_export_active:
            self._batch_export_queue = []
            self._finish_batch_export(
                f"Batch export failed after {self._batch_export_done}/{self._batch_export_total}",
                auto_close_ms=None,
                restart_preinvert=True,
            )
        else:
            self._export_in_progress = False
            self._export_cancel_event = None
            self.control_panel.set_export_progress(False)
            self.control_panel.export_button.setEnabled(True)
            self.control_panel.batch_export_button.setEnabled(True)
            self._start_next_preinvert_jobs()
        QMessageBox.warning(self, "Export TIFF Failed", message)
        self.statusBar().showMessage("TIFF export failed")

    def _export_cancelled(self, message: str) -> None:
        if self._batch_export_active:
            self._finish_batch_export(
                f"Batch export cancelled after {self._batch_export_done}/{self._batch_export_total}",
                auto_close_ms=None,
                restart_preinvert=True,
            )
            return
        self._export_in_progress = False
        self._export_cancel_event = None
        self.control_panel.set_export_progress(False)
        self.control_panel.export_button.setEnabled(True)
        self.control_panel.batch_export_button.setEnabled(True)
        self.statusBar().showMessage(message or "Export cancelled")
        self._start_next_preinvert_jobs()

    def reset_workspace(self) -> None:
        self._cancel_preview_render()
        self._cancel_raw_preview()
        self._cancel_auto_detect()
        self.origin_view.clear_selections()
        self.preview_view.set_placeholder("Positive preview waiting")
        self.preview_refresh_timer.stop()
        self.negative_preview_active = False
        self.auto_levels_pending = True
        self.mask_point = None
        self.film_rect = None
        self.white_balance_point = None
        self._reset_preview_transform()
        self._last_untransformed_negative_result = None
        self._invalidate_negative_base_cache()
        self.control_panel.set_mask_status("Not selected")
        self.control_panel.set_film_status("Not selected")
        self.control_panel.set_adjustments(self._default_adjustments(), emit=True)
        self.control_panel.set_histogram(None)
        self.preview_tabs.setCurrentWidget(self.origin_view)
        self.statusBar().showMessage("Workspace reset")

    def _cancel_preview_render(self) -> None:
        self._render_job_id += 1
        self._render_in_progress = False
        self._render_pending = False
        self._render_pending_show_errors = False

    def _cancel_raw_preview(self) -> None:
        self._raw_preview_job_id += 1
        self._raw_preview_in_progress = False
        self._refresh_activity_progress()

    def _cancel_auto_detect(self) -> None:
        self._auto_detect_job_id += 1
        self._auto_detect_auto_preview_jobs.clear()
        self._auto_detect_in_progress = False
        self._refresh_activity_progress()

    def _refresh_activity_progress(self) -> None:
        if self._raw_preview_in_progress:
            self.control_panel.set_activity_progress(True, text="Loading RAW preview...")
        elif self._auto_detect_in_progress:
            self.control_panel.set_activity_progress(True, text="Finding frame...")
        elif self._preinvert_in_progress:
            self.control_panel.set_activity_progress(True, text="Pre-inverting nearby frames...")
        elif self._model_warmup_in_progress:
            self.control_panel.set_activity_progress(True, text="Loading frame model...")
        else:
            self.control_panel.set_activity_progress(False)

    def _apply_auto_levels(self, levels: dict[str, int]) -> None:
        self._applying_auto_levels = True
        try:
            self.control_panel.set_levels(
                levels["black_point"],
                levels["mid_point"],
                levels["white_point"],
                emit=False,
            )
            updated = deepcopy(self.adjustments)
            updated.black_point = levels["black_point"]
            updated.mid_point = levels["mid_point"]
            updated.white_point = levels["white_point"]
            self.adjustments = updated
        finally:
            self._applying_auto_levels = False

    def _save_current_state(self) -> None:
        if self.current_path is None:
            return
        self.image_states[self.current_path] = ImageProcessingState(
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            white_balance_point=self.white_balance_point,
            adjustments=deepcopy(self.adjustments),
            negative_preview_active=self.negative_preview_active,
            auto_levels_pending=self.auto_levels_pending,
            preview_flip_horizontal=self._preview_flip_horizontal,
            preview_flip_vertical=self._preview_flip_vertical,
            preview_rotation_quarters=self._preview_rotation_quarters,
        )
        self._autosave_roll_session()

    def _schedule_roll_session_save(self) -> None:
        if not self._roll_session_autosave:
            return
        if self.current_path is None:
            return
        self.roll_session_save_timer.start()

    def save_roll_session_now(self) -> None:
        if self.current_path is not None:
            self._save_current_state()
        elif self._roll_session_folder is None:
            self.statusBar().showMessage("Open a folder before saving a roll session")
            return
        self._write_roll_session(show_status=True)

    def _restore_state_for_path(self, path: Path) -> bool:
        state = self.image_states.get(path)
        if state is None:
            self.mask_point = None
            self.film_rect = None
            self.white_balance_point = None
            self.adjustments = self._default_adjustments()
            self.negative_preview_active = False
            self.auto_levels_pending = True
            self._reset_preview_transform()
            self.control_panel.set_mask_status("Not selected")
            self.control_panel.set_film_status("Not selected")
            self.control_panel.set_adjustments(self.adjustments, emit=False)
            self.origin_view.restore_selections(mask_point=None, film_rect=None)
            return False

        self.mask_point = state.mask_point
        self.film_rect = state.film_rect
        self.white_balance_point = state.white_balance_point
        self.adjustments = deepcopy(state.adjustments)
        self.negative_preview_active = False
        self.auto_levels_pending = state.auto_levels_pending
        self._preview_flip_horizontal = state.preview_flip_horizontal
        self._preview_flip_vertical = state.preview_flip_vertical
        self._preview_rotation_quarters = state.preview_rotation_quarters % 4

        self.control_panel.set_mask_status(
            f"Base point: x={self.mask_point.x}, y={self.mask_point.y}"
            if self.mask_point is not None
            else "Not selected"
        )
        self.control_panel.set_film_status(
            f"Frame: {self.film_rect.label()}"
            if self.film_rect is not None
            else "Not selected"
        )
        self.control_panel.set_adjustments(self.adjustments, emit=False)
        self.origin_view.restore_selections(
            mask_point=self.mask_point,
            film_rect=self.film_rect,
        )

        if state.negative_preview_active and not self._restore_cached_preview_result():
            self._queue_negative_render(show_errors=False)
        return True

    def _maybe_auto_frame_new_negative(self, restored_state: bool) -> None:
        if restored_state or not self._auto_frame_new_negatives:
            return
        if self.current_preview is None or self.current_path is None:
            return
        if self.current_path.suffix.lower() not in RAW_EXTENSIONS:
            return
        if self.film_rect is not None or self.negative_preview_active:
            return
        self.auto_detect_current("frame_base", auto_preview=True)

    def _schedule_nearby_preinvert(self) -> None:
        if not self._auto_preinvert_nearby_frames:
            return
        if self.current_index < 0 or not self.folder_files:
            return

        radius = int(np.clip(self._auto_preinvert_radius, 0, 5))
        start = max(0, self.current_index - radius)
        end = min(len(self.folder_files), self.current_index + radius + 1)
        candidates = [
            path for path in self.folder_files[start:end]
            if self._should_preinvert_path(path)
        ]
        ordered = sorted(
            candidates,
            key=lambda path: abs(self.folder_files.index(path) - self.current_index),
        )
        for path in ordered:
            if path not in self._preinvert_queue:
                self._preinvert_queue.append(path)
        self._start_next_preinvert_jobs()

    def _should_preinvert_path(self, path: Path) -> bool:
        if path == self.current_path:
            return False
        if path.suffix.lower() not in RAW_EXTENSIONS:
            return False
        if path in self._preinvert_paths or path in self.image_states:
            return False
        if path in self.preview_result_cache:
            return False
        return True

    def _start_next_preinvert_jobs(self) -> None:
        if self._export_in_progress:
            self._refresh_activity_progress()
            return
        while self._preinvert_queue and len(self._preinvert_in_progress) < 2:
            path = self._preinvert_queue.pop(0)
            if not self._should_preinvert_path(path):
                continue
            self._preinvert_job_id += 1
            job_id = self._preinvert_job_id
            self._preinvert_in_progress.add(job_id)
            self._preinvert_paths.add(path)
            task = PreInvertTask(
                job_id=job_id,
                path=path,
                max_size=DEFAULT_PREVIEW_MAX_EDGE,
                format_hint=self.control_panel.auto_format(),
                file_key=self._file_key_for_path(path),
                adjustments=self._default_adjustments(),
                prior_frame_rect=self._preinvert_prior_frame_rect(),
            )
            task.signals.finished.connect(self._preinvert_finished)
            task.signals.failed.connect(self._preinvert_failed)
            self._thread_pool.start(task)
        self._refresh_activity_progress()

    def _preinvert_finished(self, job_id: int, output: PreInvertOutput) -> None:
        self._preinvert_in_progress.discard(job_id)
        self._preinvert_paths.discard(output.path)
        self._cache_preinvert_output(output)
        self._start_next_preinvert_jobs()

    def _preinvert_failed(self, job_id: int, path: Path, message: str) -> None:
        self._preinvert_in_progress.discard(job_id)
        self._preinvert_paths.discard(path)
        self.statusBar().showMessage(f"Auto pre-invert skipped {path.name}: {message}")
        self._start_next_preinvert_jobs()

    def _cache_preinvert_output(self, output: PreInvertOutput) -> None:
        if output.path in self.image_states:
            return
        self.raw_preview_cache.pop(output.path, None)
        self.raw_preview_cache[output.path] = CachedRawPreview(
            key=self._raw_preview_cache_key(output.path),
            preview=output.preview,
        )
        while len(self.raw_preview_cache) > RAW_PREVIEW_CACHE_LIMIT:
            oldest_path = next(iter(self.raw_preview_cache))
            self.raw_preview_cache.pop(oldest_path, None)

        if output.cache_key is not None:
            self.preview_result_cache.pop(output.path, None)
            self.preview_result_cache[output.path] = CachedPreviewResult(
                key=output.cache_key,
                result=output.result,
            )
            while len(self.preview_result_cache) > PREVIEW_RESULT_CACHE_LIMIT:
                oldest_path = next(iter(self.preview_result_cache))
                self.preview_result_cache.pop(oldest_path, None)

        self.image_states[output.path] = ImageProcessingState(
            mask_point=None,
            film_rect=output.frame_rect,
            white_balance_point=None,
            adjustments=deepcopy(output.adjustments),
            negative_preview_active=True,
            auto_levels_pending=False,
        )
        self.filmstrip.set_processed_thumbnail(
            output.path,
            self._pixmap_from_rgb8(output.result.display_rgb8),
        )
        self.statusBar().showMessage(
            f"Auto pre-inverted {output.path.name} ({output.confidence:.2f})"
        )

    def _preinvert_prior_frame_rect(self) -> ImageRect | None:
        if self.film_rect is not None:
            return self.film_rect
        state = self._previous_image_state()
        return state.film_rect if state is not None else None

    def _previous_image_state(self) -> ImageProcessingState | None:
        if not self.folder_files:
            return None
        candidate_paths: list[Path] = []
        if 0 <= self.current_index < len(self.folder_files):
            candidate_paths.extend(reversed(self.folder_files[: self.current_index]))
            candidate_paths.extend(self.folder_files[self.current_index + 1 :])
        candidate_paths.extend(path for path in self.image_states if path not in candidate_paths)

        for path in candidate_paths:
            if path == self.current_path:
                continue
            state = self.image_states.get(path)
            if state is None:
                continue
            if state.film_rect is not None or state.mask_point is not None:
                return state
        return None

    def _default_adjustments(self) -> AdjustmentParams:
        return AdjustmentParams(invert_mode=self.default_invert_mode)

    def select_sequence_file(self, path: Path) -> None:
        self.load_path(path, refresh_sequence=False)

    def go_previous_file(self) -> None:
        if not self.folder_files:
            return
        next_index = max(0, self.current_index - 1)
        self.load_path(self.folder_files[next_index], refresh_sequence=False)

    def go_next_file(self) -> None:
        if not self.folder_files:
            return
        next_index = min(len(self.folder_files) - 1, self.current_index + 1)
        self.load_path(self.folder_files[next_index], refresh_sequence=False)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._handle_color_timing_shortcut(event):
            return
        if event.key() == Qt.Key_Left:
            self.go_previous_file()
            return
        if event.key() == Qt.Key_Right:
            self.go_next_file()
            return
        if event.key() in {Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter}:
            self.confirm_current_and_go_next()
            return
        super().keyPressEvent(event)

    def _handle_color_timing_shortcut(self, event) -> bool:
        modifiers = event.modifiers()
        if modifiers not in (Qt.NoModifier, Qt.ShiftModifier):
            return False
        key = event.key()
        step = 5 if event.modifiers() & Qt.ShiftModifier else 20
        if key in {Qt.Key_Q, Qt.Key_A}:
            self._nudge_global_balance("blue_yellow", -step if key == Qt.Key_Q else step)
            return True
        if key in {Qt.Key_W, Qt.Key_S}:
            self._nudge_global_balance("green_magenta", -step if key == Qt.Key_W else step)
            return True
        if key in {Qt.Key_E, Qt.Key_D}:
            self._nudge_global_balance("red_cyan", -step if key == Qt.Key_E else step)
            return True
        if key in {Qt.Key_R, Qt.Key_F}:
            self._nudge_exposure(step if key == Qt.Key_R else -step)
            return True
        return False

    def _nudge_global_balance(self, axis_name: str, delta: int) -> None:
        updated = deepcopy(self.adjustments)
        axis = updated.color_balance.global_balance
        current = int(getattr(axis, axis_name))
        setattr(axis, axis_name, self._clamp_adjustment(current + delta))
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"{axis_name.replace('_', ' ')} {getattr(axis, axis_name):+d}")

    def _nudge_exposure(self, delta: int) -> None:
        updated = deepcopy(self.adjustments)
        updated.exposure = self._clamp_adjustment(updated.exposure + delta)
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"Exposure {updated.exposure:+d}")

    @staticmethod
    def _clamp_adjustment(value: int) -> int:
        return max(-100, min(100, int(value)))

    def toggle_preview_tab(self) -> None:
        target = self.preview_view if self.preview_tabs.currentWidget() == self.origin_view else self.origin_view
        self.preview_tabs.setCurrentWidget(target)

    def confirm_current_and_go_next(self) -> None:
        if self.current_path is None:
            return
        self._save_current_state()
        self.statusBar().showMessage(f"Confirmed: {self.current_path.name}")
        self.go_next_file()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.roll_session_save_timer.stop()
        self._save_current_state()
        super().closeEvent(event)

    def _set_folder_sequence(self, path: Path) -> None:
        if self.current_path is not None:
            self._save_current_state()
        self._preinvert_queue = []
        self.folder_files = list_supported_files(path.parent)
        self._roll_session_folder = path.parent
        restored = load_roll_session(path.parent, self.folder_files)
        if restored:
            self.image_states.update(restored)
        self._sync_sequence_position(path)
        self.filmstrip.set_files(self.folder_files, path)
        self._restore_filmstrip_session_badges()
        if restored:
            self.statusBar().showMessage(
                f"Loaded roll session: {len(restored)} saved images"
            )

    def _restore_filmstrip_session_badges(self) -> None:
        for source_path in self.folder_files:
            state = self.image_states.get(source_path)
            self.filmstrip.set_processed_badge(
                source_path,
                bool(state is not None and state.negative_preview_active),
            )

    def _autosave_roll_session(self) -> None:
        if not self._roll_session_autosave:
            return
        self._write_roll_session(show_status=False)

    def _write_roll_session(self, *, show_status: bool) -> None:
        folder = self._roll_session_folder or (self.current_path.parent if self.current_path else None)
        if folder is None:
            return
        try:
            save_roll_session(folder, self.image_states, self.folder_files)
        except OSError as exc:
            self.statusBar().showMessage(f"Roll session save failed: {exc}")
            return
        if show_status:
            self.statusBar().showMessage(f"Saved roll session: {session_path_for_folder(folder)}")

    def _sync_sequence_position(self, path: Path) -> None:
        try:
            self.current_index = self.folder_files.index(path)
        except ValueError:
            self.current_index = -1

        if self.current_index >= 0:
            self.control_panel.set_sequence_status(
                f"Sequence {self.current_index + 1} / {len(self.folder_files)}"
            )
        else:
            self.control_panel.set_sequence_status("No sequence")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #15181d;
            }
            QMenuBar {
                background: #111419;
                color: #f2f4f7;
                border-bottom: 1px solid #303640;
                padding: 2px 6px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
                color: #f2f4f7;
            }
            QMenuBar::item:selected {
                background: #2c3440;
                border-radius: 4px;
            }
            QMenu {
                background: #181d23;
                color: #f2f4f7;
                border: 1px solid #343c47;
                padding: 5px 0;
            }
            QMenu::item {
                padding: 7px 28px 7px 24px;
            }
            QMenu::item:selected {
                background: #344150;
            }
            QMenu::separator {
                height: 1px;
                background: #343c47;
                margin: 5px 8px;
            }
            QStatusBar {
                background: #20242b;
                color: #cfd6df;
                border-top: 1px solid #303640;
            }
            QWidget#emptyState {
                background: #15181d;
            }
            QPushButton#emptyOpenButton {
                background: #2f6f91;
                border: 1px solid #63a8c9;
                border-radius: 7px;
                color: #ffffff;
                font-size: 16px;
                font-weight: 600;
                padding: 14px 24px;
                min-width: 220px;
            }
            QPushButton#emptyOpenButton:hover {
                background: #397fa5;
            }
            QSplitter::handle {
                background: #303640;
            }
            QSplitter::handle:horizontal {
                width: 5px;
            }
            QSplitter::handle:hover {
                background: #53606f;
            }
            QTabWidget#previewTabs::pane {
                border: none;
                background: #15181d;
            }
            QTabBar::tab {
                background: #20242b;
                color: #aeb8c5;
                padding: 8px 18px;
                border-right: 1px solid #303640;
            }
            QTabBar::tab:selected {
                background: #2c3440;
                color: #f2f4f7;
            }
            """
        )
        self.setAttribute(Qt.WA_StyledBackground, True)

    def _pixmap_from_rgb8(self, rgb8: np.ndarray) -> QPixmap:
        rgb8 = np.ascontiguousarray(rgb8)
        height, width, channels = rgb8.shape
        if channels != 3:
            raise ValueError("Expected an RGB image with 3 channels.")
        image = QImage(
            rgb8.data,
            width,
            height,
            width * channels,
            QImage.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(image)
