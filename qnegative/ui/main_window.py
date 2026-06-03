from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tifffile
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
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
    process_negative_base_preview,
    analysis_inset_from_adjustments,
    analysis_inset_crop,
    suggest_lab_print_luminance_levels,
    suggest_global_balance_from_neutral,
)
from qnegative.core.preview import DEFAULT_PREVIEW_MAX_EDGE, RawPreview, make_raw_preview, resize_long_edge
from qnegative.core.raw_loader import load_raw_rgb16
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


class ModelWarmupSignals(QObject):
    finished = Signal(bool, str)


INTERACTIVE_PREVIEW_MAX_EDGE = 720
FINAL_RENDER_QUALITY = "final"
INTERACTIVE_RENDER_QUALITY = "interactive"
FINAL_RENDER_DEBOUNCE_MS = 80
INTERACTIVE_RENDER_DEBOUNCE_MS = 115


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
    result: NegativePreviewResult
    cache: PreviewStageCache
    quality: str


@dataclass(frozen=True)
class AutoDetectOutput:
    mode: str
    path: Path | None
    frame_result: AutoFrameResult | None
    base_result: AutoBaseResult | None
    fallback_state: ImageProcessingState | None


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


def base_stage_key(
    preview: RawPreview,
    mask_point: ImagePoint | None,
    film_rect: ImageRect | None,
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
        self.signals = AutoDetectSignals()

    def run(self) -> None:
        frame_result: AutoFrameResult | None = None
        base_result: AutoBaseResult | None = None
        try:
            if self.mode == "frame_base":
                result = detect_frame_and_base(
                    self.preview.preview_linear_rgb,
                    preview_size=self.preview.preview_size,
                    source_size=self.preview.source_size,
                    format_hint=self.format_hint,
                    detect_base=self.detect_base,
                )
                frame_result = result.frame
                base_result = result.base
            elif self.mode == "frame":
                frame_result = detect_film_frame(
                    self.preview.preview_linear_rgb,
                    preview_size=self.preview.preview_size,
                    source_size=self.preview.source_size,
                    format_hint=self.format_hint,
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
        self.render_cache = render_cache or PreviewStageCache()
        self.signals = PreviewRenderSignals()

    def run(self) -> None:
        try:
            base_key = base_stage_key(self.preview, self.mask_point, self.film_rect)
            base = self._base_stage(base_key)
            if self.adjustments.invert_mode == InvertMode.LAB_PRINT.value:
                output = self._lab_print_output(base_key, base)
            else:
                result = process_negative_base_preview(base, self.adjustments)
                output = PreviewRenderOutput(
                    result=result,
                    cache=PreviewStageCache(base_key=base_key, base=base),
                    quality=self.quality,
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

        levels_key = lab_print_levels_key(
            negative_key,
            self.adjustments,
            auto_levels_pending=self.auto_levels_pending,
        )
        if (
            self.render_cache.levels_key == levels_key
            and self.render_cache.levels_stage is not None
        ):
            levels_stage = self.render_cache.levels_stage
        else:
            levels_stage = build_lab_print_levels_stage(
                negative_stage,
                self.adjustments,
                auto_levels=None if self.auto_levels_pending else current_levels(self.adjustments),
            )

        color_key = lab_print_color_key(levels_key, self.adjustments)
        if (
            self.render_cache.color_key == color_key
            and self.render_cache.color_stage is not None
        ):
            color_stage = self.render_cache.color_stage
        else:
            color_stage = build_lab_print_color_stage(levels_stage, self.adjustments)

        display_key = lab_print_display_key(color_key, self.adjustments)
        if (
            self.render_cache.display_key == display_key
            and self.render_cache.display_result is not None
        ):
            result = self.render_cache.display_result
        else:
            result = build_lab_print_display_stage(color_stage, self.adjustments)

        return PreviewRenderOutput(
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
        )


class ExportSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(str)
    failed = Signal(str)


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
        self.signals = ExportSignals()

    def run(self) -> None:
        try:
            self.signals.progress.emit(5, "Loading RAW")
            needs_camera_transform = self.adjustments.camera_color_strength > 0
            raw_image = load_raw_rgb16(
                self.source_path,
                half_size=False,
                include_display_transform=needs_camera_transform,
            )
            self.signals.progress.emit(30, "Building base")
            base = build_negative_base_preview(
                raw_image.as_float32(),
                source_size=raw_image.source_size,
                mask_point=self.mask_point,
                film_rect=self.film_rect,
                preview_camera_wb_linear_rgb=raw_image.camera_wb_as_float32(),
                camera_to_srgb_matrix=raw_image.camera_to_srgb_matrix,
            )
            self.signals.progress.emit(55, "Processing positive")
            export_linear_rgb = self._process_export(base)

            self.signals.progress.emit(75, "Preparing TIFF")
            linear_rgb = transform_preview_array(
                export_linear_rgb,
                flip_horizontal=self.flip_horizontal,
                flip_vertical=self.flip_vertical,
                rotation_quarters=self.rotation_quarters,
            )
            tiff_rgb16 = linear_to_srgb16(linear_rgb)
            self.signals.progress.emit(90, "Writing TIFF")
            tifffile.imwrite(
                self.output_path,
                tiff_rgb16,
                photometric="rgb",
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(str(self.output_path))

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
        color_stage = build_lab_print_color_stage(levels_stage, effective)
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
        self.setWindowTitle("QNegativeLab")
        self.default_invert_mode = default_invert_mode
        self.current_path: Path | None = None
        self.current_preview: RawPreview | None = None
        self.folder_files: list[Path] = []
        self.current_index: int = -1
        self.image_states: dict[Path, ImageProcessingState] = {}
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
        self._auto_detect_in_progress = False
        self._export_in_progress = False

        self.control_panel = ControlPanel()
        self.origin_view = ImageView()
        self.image_view = self.origin_view
        self.preview_view = OpenGLPreviewView()
        self.preview_view.set_transform_context_enabled(True)
        self.preview_view.set_placeholder("Positive preview waiting")
        self.preview_tabs = QTabWidget()
        self.preview_tabs.setObjectName("previewTabs")
        self.filmstrip = FolderFilmstrip()
        self.preview_refresh_timer = QTimer(self)
        self.preview_refresh_timer.setSingleShot(True)
        self.preview_refresh_timer.setInterval(FINAL_RENDER_DEBOUNCE_MS)

        self._build_layout()
        self._build_developer_menu()
        self._connect()
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
        self.preview_tabs.addTab(self.origin_view, "Origin")
        self.preview_tabs.addTab(self.preview_view, "Preview")
        right_layout.addWidget(self.preview_tabs, 1)
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

    def _build_developer_menu(self) -> None:
        developer_menu = self.menuBar().addMenu("Developer")

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

    def _set_developer_invert_mode(self, mode: str) -> None:
        updated = deepcopy(self.adjustments)
        updated.invert_mode = mode
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"Developer invert mode: {invert_mode_label(mode)}")

    def _connect(self) -> None:
        self.control_panel.openRequested.connect(self.open_file)
        self.control_panel.exportRequested.connect(self.export_current)
        self.control_panel.invertRequested.connect(self.preview_inversion)
        self.control_panel.resetRequested.connect(self.reset_workspace)
        self.control_panel.toolChanged.connect(self.set_tool_mode)
        self.control_panel.autoDetectRequested.connect(self.auto_detect_current)
        self.control_panel.adjustmentsChanged.connect(self.adjustments_changed)
        self.control_panel.adjustmentInteractionStarted.connect(self.adjustment_interaction_started)
        self.control_panel.adjustmentInteractionFinished.connect(self.adjustment_interaction_finished)

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

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open RAW or Image",
            str(Path.cwd()),
            "RAW and Images (*.arw *.raw *.dng *.cr2 *.cr3 *.nef *.raf *.orf *.rw2 *.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;All files (*.*)",
        )
        if not path:
            return

        self.load_path(Path(path), refresh_sequence=True)

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

        self._cancel_preview_render()
        self._cancel_raw_preview()
        self._cancel_auto_detect()

        if refresh_sequence:
            self._set_folder_sequence(path)
        else:
            self._sync_sequence_position(path)

        self.current_path = path
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
        self._restore_state_for_path(path)
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

    def auto_detect_current(self, mode: str) -> None:
        if self.current_preview is None:
            QMessageBox.information(self, "Auto Detect", "Open a RAW file before auto detection.")
            return
        if self._auto_detect_in_progress:
            self.statusBar().showMessage("Auto detect already running...")
            return

        self._auto_detect_job_id += 1
        job_id = self._auto_detect_job_id
        task = AutoDetectTask(
            job_id=job_id,
            mode=mode,
            path=self.current_path,
            preview=self.current_preview,
            format_hint=self.control_panel.auto_format(),
            detect_base=self._film_base_required_for_current_mode(),
            current_film_rect=self.film_rect,
            fallback_state=self._previous_image_state(),
        )
        task.signals.finished.connect(self._auto_detect_finished)
        task.signals.failed.connect(self._auto_detect_failed)
        self._auto_detect_in_progress = True
        self._refresh_activity_progress()
        self.statusBar().showMessage("Auto detect running in background...")
        self._thread_pool.start(task)

    def _auto_detect_finished(self, job_id: int, output: AutoDetectOutput) -> None:
        if job_id != self._auto_detect_job_id:
            return
        self._auto_detect_in_progress = False
        self._refresh_activity_progress()
        if output.path != self.current_path:
            self.statusBar().showMessage("Auto detect result ignored after file change")
            return

        self._apply_auto_detect_output(output)

    def _auto_detect_failed(self, job_id: int, message: str) -> None:
        if job_id != self._auto_detect_job_id:
            return
        self._auto_detect_in_progress = False
        self._refresh_activity_progress()
        self.statusBar().showMessage("Auto detect failed")
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
        self._schedule_preview_if_ready()

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
        if self.negative_preview_active:
            if self._render_in_progress:
                self._render_pending = True
            self.preview_refresh_timer.setInterval(
                INTERACTIVE_RENDER_DEBOUNCE_MS
                if self._interactive_adjustment_active
                else FINAL_RENDER_DEBOUNCE_MS
            )
            self.preview_refresh_timer.start()

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

        if self.auto_levels_pending:
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
            f"Positive preview {displayed_result.width} x {displayed_result.height}\nMode {invert_mode_label(self.adjustments.invert_mode)}\n{base_text}\n{wb_label} {wb_gains}"
        )
        self.control_panel.set_histogram(displayed_result.histogram)
        self.negative_preview_active = True
        if show_errors:
            self.preview_tabs.setCurrentWidget(self.preview_view)
        self.statusBar().showMessage(
            "Interactive preview ready"
            if output.quality == INTERACTIVE_RENDER_QUALITY
            else "Inverted preview ready"
        )

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

    def _schedule_preview_if_ready(self) -> None:
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
        )
        if self._negative_base_cache_key == key and self._negative_base_cache is not None:
            return self._negative_base_cache

        base = build_negative_base_preview(
            self.current_preview.preview_linear_rgb,
            source_size=self.current_preview.source_size,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
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
        )
        task.signals.finished.connect(self._export_finished)
        task.signals.failed.connect(self._export_failed)
        task.signals.progress.connect(self._export_progress_updated)
        self._export_in_progress = True
        self.control_panel.export_button.setEnabled(False)
        self.control_panel.set_export_progress(True, value=0, text="Starting export")
        self.statusBar().showMessage("Exporting 16-bit TIFF...")
        self._thread_pool.start(task)

    def _export_progress_updated(self, value: int, text: str) -> None:
        self.control_panel.update_export_progress(value, text)
        self.statusBar().showMessage(f"{text}...")

    def _export_finished(self, output_path: str) -> None:
        self._export_in_progress = False
        self.control_panel.update_export_progress(100, "Export complete")
        self.control_panel.set_export_progress(False)
        self.control_panel.export_button.setEnabled(True)
        self.statusBar().showMessage(f"Exported TIFF: {output_path}")

    def _export_failed(self, message: str) -> None:
        self._export_in_progress = False
        self.control_panel.set_export_progress(False)
        self.control_panel.export_button.setEnabled(True)
        QMessageBox.warning(self, "Export TIFF Failed", message)
        self.statusBar().showMessage("TIFF export failed")

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
        self._auto_detect_in_progress = False
        self._refresh_activity_progress()

    def _refresh_activity_progress(self) -> None:
        if self._raw_preview_in_progress:
            self.control_panel.set_activity_progress(True, text="Loading RAW preview...")
        elif self._auto_detect_in_progress:
            self.control_panel.set_activity_progress(True, text="Finding frame...")
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

    def _restore_state_for_path(self, path: Path) -> None:
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
            return

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

        if state.negative_preview_active:
            self._queue_negative_render(show_errors=False)

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
        if event.key() == Qt.Key_Left:
            self.go_previous_file()
            return
        if event.key() == Qt.Key_Right:
            self.go_next_file()
            return
        super().keyPressEvent(event)

    def _set_folder_sequence(self, path: Path) -> None:
        self.folder_files = list_supported_files(path.parent)
        self._sync_sequence_position(path)
        self.filmstrip.set_files(self.folder_files, path)

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
            QStatusBar {
                background: #20242b;
                color: #cfd6df;
                border-top: 1px solid #303640;
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
