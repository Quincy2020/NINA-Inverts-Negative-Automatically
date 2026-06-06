from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import perf_counter

import numpy as np
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.file_sequence import RAW_EXTENSIONS
from qnegative.core.lens_profiles import (
    create_flat_frame_profile,
    default_lens_profile_dir,
    load_lens_profile,
    save_radial_lens_profile,
)
from qnegative.core.auto_detect import (
    AutoBaseResult,
    AutoFrameResult,
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
    NegativePreviewResult,
    PipelineError,
    build_lab_print_levels_stage,
    build_lab_print_negative_stage,
    build_negative_base_preview,
    build_density_preview_analysis,
    estimate_lab_print_auto_cmy_offsets,
    manual_printer_balance_offsets,
    analysis_inset_from_adjustments,
    set_log_print_curve_engine,
    suggest_printer_balance_from_log_sample,
)
from qnegative.core.preview import DEFAULT_PREVIEW_MAX_EDGE, RawPreview
from qnegative.core.roll_color_adapter import roll_color_result_summary
from qnegative.ui.control_panel import ControlPanel
from qnegative.ui.export_dialogs import BatchExportDialog, BatchExportSettings, BatchExportSettingsDialog
from qnegative.ui.export_tasks import (
    ImageExportTask,
    export_format_extension,
    export_format_from_filter,
    export_format_from_path,
    export_format_label,
)
from qnegative.ui.folder_filmstrip import FolderFilmstrip
from qnegative.ui.gl_preview_view import OpenGLPreviewView
from qnegative.ui.image_view import ImageView
from qnegative.ui.menus import build_main_menus
from qnegative.ui.preview_cache import (
    CachedPreviewResult,
    CachedRawPreview,
    PreviewRenderOutput,
    PreviewStageCache,
    adjustments_preview_cache_key,
    lens_correction_key,
    preview_result_cache_key_for,
)
from qnegative.ui.preview_tasks import (
    AutoDetectOutput,
    AutoDetectTask,
    ModelWarmupTask,
    PreInvertOutput,
    PreInvertTask,
    PreviewRenderTask,
    RawPreviewTask,
    scaled_raw_preview,
)
from qnegative.ui.project_controller import ProjectController
from qnegative.ui.roll_color_tasks import RollColorAnalysisItem, RollColorAnalysisTask
from qnegative.ui.shortcuts import install_main_window_shortcuts
from qnegative.ui.render_controller import PreviewRenderController
from qnegative.ui.state_store import (
    build_current_image_state,
    default_adjustments,
    lab_print_adjustments,
    merge_stale_preview_result_state,
    restored_runtime_for_state,
    should_restore_positive_preview,
    state_from_preinvert_output,
)


INTERACTIVE_PREVIEW_MAX_EDGE = 720
FINAL_RENDER_QUALITY = "final"
INTERACTIVE_RENDER_QUALITY = "interactive"
FINAL_RENDER_DEBOUNCE_MS = 80
INTERACTIVE_RENDER_DEBOUNCE_MS = 115
PREVIEW_RESULT_CACHE_LIMIT = 16
RAW_PREVIEW_CACHE_LIMIT = 16


def invert_mode_label(mode: str) -> str:
    labels = {
        InvertMode.LAB_PRINT.value: "Lab Print",
    }
    return labels.get(mode, mode)


def export_timing_is_top_level(name: str) -> bool:
    return (
        name in {"RAW decode", "Build base", "Lab Print"}
        or name.startswith("Prepare ")
        or name.endswith(" write")
    )


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NINA")
        self.current_path: Path | None = None
        self.current_preview: RawPreview | None = None
        self.folder_files: list[Path] = []
        self.current_index: int = -1
        self._project = ProjectController()
        self.image_states: dict[Path, ImageProcessingState] = {}
        self.raw_preview_cache: dict[Path, CachedRawPreview] = {}
        self.preview_result_cache: dict[Path, CachedPreviewResult] = {}
        self.mask_point: ImagePoint | None = None
        self.film_rect: ImageRect | None = None
        self.white_balance_point: ImagePoint | None = None
        self.lab_print_cmy_offsets: list[float] | None = None
        self.adjustments = default_adjustments()
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
        self._render_controller = PreviewRenderController()
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
        self._roll_session_autosave = True
        self._roll_color_result: dict | None = None
        self._roll_color_analysis_job_id = 0
        self._roll_color_analysis_in_progress = False
        self._undo_stack: list[AdjustmentParams] = []
        self._redo_stack: list[AdjustmentParams] = []
        self._applying_history = False

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
        build_main_menus(self)
        self._connect()
        self._shortcuts = install_main_window_shortcuts(self)
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

    def set_print_curve_engine(self, action: QAction) -> None:
        engine = str(action.data())
        set_log_print_curve_engine(engine)
        self._cancel_preview_render()
        self._reset_preview_stage_caches()
        self.preview_result_cache.clear()
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
        self.control_panel.lensProfileSaveRequested.connect(self.save_lens_profile)
        self.control_panel.lensProfileLoadRequested.connect(self.load_lens_profile)
        self.control_panel.lensFlatProfileCreateRequested.connect(self.create_flat_lens_profile)
        self.control_panel.lensApplyAllRequested.connect(lambda: self.apply_lens_correction("all"))
        self.control_panel.lensApplyUnprocessedRequested.connect(
            lambda: self.apply_lens_correction("unprocessed")
        )
        self.control_panel.lensApplyCompletedRequested.connect(
            lambda: self.apply_lens_correction("completed")
        )
        self.control_panel.rollColorAnalyzeRequested.connect(self.analyze_roll_color)
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
            "Open RAW / DNG",
            str(Path.cwd()),
            "RAW files (*.arw *.raw *.dng *.cr2 *.cr3 *.nef *.raf *.orf *.rw2);;All files (*.*)",
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
        files = self._project.supported_files_for_folder(Path(folder))
        if not files:
            QMessageBox.information(self, "Open Folder", "No supported RAW files were found.")
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

    def save_lens_profile(self) -> None:
        profile_dir = default_lens_profile_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)
        default_path = profile_dir / "radial_lens_profile.json"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Lens Profile",
            str(default_path),
            "NINA Lens Profile (*.json);;All files (*.*)",
        )
        if not selected:
            return

        output_path = Path(selected)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        try:
            save_radial_lens_profile(
                output_path,
                output_path.stem,
                self.adjustments.lens_correction,
            )
        except OSError as exc:
            QMessageBox.warning(self, "Save Lens Profile", str(exc))
            return
        self.statusBar().showMessage(f"Lens profile saved: {output_path.name}")

    def load_lens_profile(self) -> None:
        profile_dir = default_lens_profile_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load Lens Profile",
            str(profile_dir),
            "NINA Lens Profile (*.json);;All files (*.*)",
        )
        if not selected:
            return

        try:
            params = load_lens_profile(Path(selected))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load Lens Profile", str(exc))
            return

        updated = deepcopy(self.adjustments)
        updated.lens_correction = params
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"Lens profile loaded: {Path(selected).name}")

    def create_flat_lens_profile(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self,
            "Select Flat RAW",
            str(Path("D:/QNegativeLab/Len_calibration_data") if Path("D:/QNegativeLab/Len_calibration_data").exists() else Path.cwd()),
            "RAW files (*.arw *.raw *.dng *.cr2 *.cr3 *.nef *.raf *.orf *.rw2);;All files (*.*)",
        )
        if not source:
            return

        profile_dir = default_lens_profile_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)
        default_path = profile_dir / f"{Path(source).stem}_flat_profile.json"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Flat Lens Profile",
            str(default_path),
            "NINA Lens Profile (*.json);;All files (*.*)",
        )
        if not selected:
            return

        output_path = Path(selected)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        self.control_panel.set_activity_progress(True, text="Creating flat lens profile...")
        try:
            params = create_flat_frame_profile(
                Path(source),
                output_path,
                name=output_path.stem,
                map_long_edge=512,
                blur_radius=41,
                max_gain=max(1.0, self.adjustments.lens_correction.max_gain / 100.0),
                map_mode="linked_luminance",
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Create Flat Lens Profile", str(exc))
            return
        finally:
            self.control_panel.set_activity_progress(False)

        updated = deepcopy(self.adjustments)
        updated.lens_correction = params
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"Flat lens profile created: {output_path.name}")

    def apply_lens_correction(self, scope: str) -> None:
        source_params = deepcopy(self.adjustments.lens_correction)
        if self.current_path is not None:
            self._save_current_state()

        targets = self._lens_correction_targets(scope)
        if not targets:
            self.statusBar().showMessage(f"No {scope} negatives to update")
            return

        reply = QMessageBox.question(
            self,
            "Apply Lens Correction",
            f"Apply current lens correction to {len(targets)} image(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for path in targets:
            state = self.image_states.get(path)
            if state is None:
                state = ImageProcessingState(adjustments=default_adjustments())
            updated_adjustments = deepcopy(state.adjustments)
            updated_adjustments.lens_correction = deepcopy(source_params)
            self.image_states[path] = replace(
                state,
                adjustments=updated_adjustments,
                lab_print_cmy_offsets=None,
            )
            self.preview_result_cache.pop(path, None)

        if self.current_path in targets:
            self.adjustments.lens_correction = deepcopy(source_params)
            self.lab_print_cmy_offsets = None
            self.control_panel.set_adjustments(self.adjustments, emit=False)
            self._invalidate_negative_base_cache()
            if self.negative_preview_active:
                self._schedule_preview_if_ready(force=True)

        self._autosave_roll_session()
        self.statusBar().showMessage(f"Lens correction applied to {len(targets)} image(s)")

    def _lens_correction_targets(self, scope: str) -> list[Path]:
        paths = [
            path
            for path in (self.folder_files or ([self.current_path] if self.current_path else []))
            if path is not None and path.suffix.lower() in RAW_EXTENSIONS
        ]
        if scope == "all":
            return paths
        if scope == "completed":
            return [
                path
                for path in paths
                if (self.image_states.get(path) is not None and self.image_states[path].negative_preview_active)
            ]
        if scope == "unprocessed":
            return [
                path
                for path in paths
                if not (self.image_states.get(path) is not None and self.image_states[path].negative_preview_active)
            ]
        return []

    def analyze_roll_color(self) -> None:
        if self._roll_color_analysis_in_progress:
            self.statusBar().showMessage("Roll color analysis already in progress")
            return
        if self.current_path is not None:
            self._save_current_state()

        items = self._roll_color_analysis_items()
        if not items:
            QMessageBox.information(
                self,
                "Analyze Roll Color",
                "Generate positive previews for at least two framed RAW images first.",
            )
            return

        self._roll_color_analysis_job_id += 1
        job_id = self._roll_color_analysis_job_id
        task = RollColorAnalysisTask(job_id=job_id, items=items)
        task.signals.progress.connect(self._roll_color_analysis_progress)
        task.signals.finished.connect(self._roll_color_analysis_finished)
        task.signals.failed.connect(self._roll_color_analysis_failed)
        self._roll_color_analysis_in_progress = True
        self.control_panel.set_roll_color_analyzing(True)
        self.control_panel.set_roll_color_status(f"Analyzing {len(items)} positive frames...")
        self._refresh_activity_progress()
        self.statusBar().showMessage(f"Analyzing roll color for {len(items)} positives...")
        self._thread_pool.start(task)

    def _roll_color_analysis_items(self) -> list[RollColorAnalysisItem]:
        ordered_paths = list(self.folder_files) if self.folder_files else list(self.image_states)
        for path in self.image_states:
            if path not in ordered_paths:
                ordered_paths.append(path)

        items: list[RollColorAnalysisItem] = []
        for path in ordered_paths:
            if path.suffix.lower() not in RAW_EXTENSIONS:
                continue
            state = self.image_states.get(path)
            if state is None or not state.negative_preview_active:
                continue
            if state.film_rect is None or not state.film_rect.is_valid():
                continue
            items.append(RollColorAnalysisItem(path=path, state=deepcopy(state)))
        return items

    def _roll_color_analysis_progress(self, value: int, text: str) -> None:
        self.control_panel.set_roll_color_status(text)
        self.statusBar().showMessage(f"{text} ({value}%)")

    def _roll_color_analysis_finished(self, job_id: int, output) -> None:
        if job_id != self._roll_color_analysis_job_id:
            return
        self._roll_color_analysis_in_progress = False
        self.control_panel.set_roll_color_analyzing(False)
        self._refresh_activity_progress()

        self._roll_color_result = output.result
        updated_count = 0
        for path_text, frame_plan in output.frames_by_path.items():
            path = Path(path_text)
            state = self.image_states.get(path)
            if state is None:
                continue
            adjustments = deepcopy(state.adjustments)
            adjustments.color_correction.enabled = True
            self.image_states[path] = replace(
                state,
                adjustments=adjustments,
                roll_color_frame=deepcopy(frame_plan),
            )
            self.preview_result_cache.pop(path, None)
            updated_count += 1

        if self.current_path is not None and self.current_path in self.image_states:
            current = self.image_states[self.current_path]
            self.adjustments = deepcopy(current.adjustments)
            self.control_panel.set_adjustments(self.adjustments, emit=False)
            self._reset_preview_stage_caches()
            if self.negative_preview_active:
                self._schedule_preview_if_ready(force=True)

        self._update_roll_color_status()
        self._autosave_roll_session()
        self.statusBar().showMessage(f"Roll color analysis ready: {updated_count} frames")

    def _roll_color_analysis_failed(self, job_id: int, message: str) -> None:
        if job_id != self._roll_color_analysis_job_id:
            return
        self._roll_color_analysis_in_progress = False
        self.control_panel.set_roll_color_analyzing(False)
        self.control_panel.set_roll_color_status("Analysis failed")
        self._refresh_activity_progress()
        QMessageBox.warning(self, "Analyze Roll Color", message)
        self.statusBar().showMessage("Roll color analysis failed")

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
        self._update_roll_color_status()
        self.current_preview = None
        self.negative_preview_active = False
        self.auto_levels_pending = True
        self.white_balance_point = None
        self.lab_print_cmy_offsets = None
        self._last_untransformed_negative_result = None
        self._reset_preview_transform()
        self._invalidate_negative_base_cache()
        extension = path.suffix.lower()
        self.control_panel.set_file_status(path.name)
        self.filmstrip.set_current(path)
        if not self._show_cached_preview_result_for_path(path):
            self.preview_view.set_placeholder("Positive preview waiting")

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
        self.control_panel.set_image_status("Decoding source...")

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
        self.lab_print_cmy_offsets = None
        self.auto_levels_pending = True
        self._invalidate_negative_base_cache()
        self.control_panel.set_mask_status(f"Base point: x={point.x}, y={point.y}")
        self.statusBar().showMessage("Base picker point saved")
        self._schedule_preview_if_ready()

    def film_rect_selected(self, rect: ImageRect) -> None:
        self.film_rect = rect
        self.white_balance_point = None
        self.lab_print_cmy_offsets = None
        self.auto_levels_pending = True
        self._invalidate_negative_base_cache()
        self.control_panel.set_film_status(f"Frame: {rect.label()}")
        self.statusBar().showMessage("Frame area saved")
        self._schedule_preview_if_ready()

    def film_rect_reset(self) -> None:
        self.film_rect = None
        self.white_balance_point = None
        self.lab_print_cmy_offsets = None
        self.auto_levels_pending = True
        self.negative_preview_active = False
        self._last_untransformed_negative_result = None
        self._invalidate_negative_base_cache()
        self.control_panel.set_film_status("Not selected")
        self.preview_view.set_placeholder("Positive preview waiting")
        self.statusBar().showMessage("Frame reset")

    def white_balance_point_selected(self, point: ImagePoint) -> None:
        if (
            not self.negative_preview_active
            or self.last_negative_result is None
            or self._last_untransformed_negative_result is None
        ):
            self.white_balance_point = None
            self.preview_view.restore_selections(
                mask_point=None,
                film_rect=None,
                white_balance_point=None,
            )
            self.statusBar().showMessage("Generate a positive preview before using the WB picker")
            return

        try:
            log_point = self._inverse_preview_transform_point(
                point,
                source_size=ImageSize(
                    width=self._last_untransformed_negative_result.width,
                    height=self._last_untransformed_negative_result.height,
                ),
            )
            normalized_for_print = self._current_normalized_for_print()
            base_cmy_offsets = self._printer_picker_base_cmy_offsets(normalized_for_print)
            printer_balance, median_log, offset_delta = suggest_printer_balance_from_log_sample(
                normalized_for_print,
                log_point,
                base_cmy_offsets=base_cmy_offsets,
            )
        except PipelineError as exc:
            self.statusBar().showMessage(str(exc))
            return

        self.white_balance_point = point
        updated = deepcopy(self.adjustments)
        updated.printer_balance = printer_balance
        self.control_panel.set_adjustments(updated, emit=True)
        self.preview_view.set_tool_mode(ToolMode.PAN)

        sample_text = ", ".join(f"{value:.3f}" for value in median_log)
        offset_text = ", ".join(f"{value:+.4f}" for value in offset_delta)
        self.statusBar().showMessage(
            f"WB picker: x={point.x}, y={point.y}, median log {sample_text}, printer delta {offset_text}"
        )

    def _current_normalized_for_print(self) -> np.ndarray:
        if self._last_untransformed_negative_result is None:
            raise PipelineError("Generate a positive preview before using the WB picker.")

        base = self._negative_base_for_current()
        negative_stage = build_lab_print_negative_stage(
            base,
            include_histogram=False,
            analysis_inset=analysis_inset_from_adjustments(self.adjustments),
        )
        levels_stage = build_lab_print_levels_stage(
            negative_stage,
            self.adjustments,
            auto_levels=self._last_untransformed_negative_result.auto_levels,
        )
        return levels_stage.normalized_for_print

    def _printer_picker_base_cmy_offsets(self, normalized_for_print: np.ndarray) -> np.ndarray:
        if not self.adjustments.auto_wb:
            return np.zeros(3, dtype=np.float32)

        manual = manual_printer_balance_offsets(self.adjustments.printer_balance)
        if self.lab_print_cmy_offsets is not None:
            effective = np.asarray(self.lab_print_cmy_offsets, dtype=np.float32).reshape(3)
            return (effective - manual).astype(np.float32, copy=False)

        return estimate_lab_print_auto_cmy_offsets(normalized_for_print)

    def _inverse_preview_transform_point(self, point: ImagePoint, *, source_size: ImageSize) -> ImagePoint:
        width = max(1, int(source_size.width))
        height = max(1, int(source_size.height))
        x = int(point.x)
        y = int(point.y)

        rotation = self._preview_rotation_quarters % 4
        if rotation == 1:
            rotated_x = y
            rotated_y = height - 1 - x
        elif rotation == 2:
            rotated_x = width - 1 - x
            rotated_y = height - 1 - y
        elif rotation == 3:
            rotated_x = width - 1 - y
            rotated_y = x
        else:
            rotated_x = x
            rotated_y = y

        if self._preview_flip_horizontal:
            rotated_x = width - 1 - rotated_x
        if self._preview_flip_vertical:
            rotated_y = height - 1 - rotated_y

        return ImagePoint(
            x=max(0, min(width - 1, int(rotated_x))),
            y=max(0, min(height - 1, int(rotated_y))),
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
        self.lab_print_cmy_offsets = None
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
        if output.auto_preview:
            self._render_auto_detect_preview_if_ready()
        else:
            self._schedule_preview_if_ready()
        self._schedule_nearby_preinvert()

    def _render_auto_detect_preview_if_ready(self) -> None:
        if self.current_preview is None:
            return
        if self.film_rect is None or not self.film_rect.is_valid():
            return
        if self._film_base_required_for_current_mode() and self.mask_point is None:
            return
        self._queue_negative_render(show_errors=False, interactive=False)

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
        if self._render_controller.in_progress:
            self._render_controller.defer(self.current_path, show_errors=False)
            return
        self._queue_negative_render(show_errors=False, interactive=False)

    def preview_refresh_timeout(self) -> None:
        self._queue_negative_render(
            show_errors=False,
            interactive=self._interactive_adjustment_active,
        )

    def adjustments_changed(self, values: dict) -> None:
        previous = self.adjustments
        values["invert_mode"] = InvertMode.LAB_PRINT.value
        self.adjustments = AdjustmentParams(**values)
        if not self._applying_history and previous != self.adjustments:
            self._undo_stack.append(deepcopy(previous))
            if len(self._undo_stack) > 80:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
        mode_changed = previous.invert_mode != self.adjustments.invert_mode
        if mode_changed:
            self.auto_levels_pending = True
            self.lab_print_cmy_offsets = None
        printer_balance_changed = previous.printer_balance != self.adjustments.printer_balance
        if previous.auto_wb != self.adjustments.auto_wb or printer_balance_changed:
            self.lab_print_cmy_offsets = None
            if self.current_path is not None:
                self.preview_result_cache.pop(self.current_path, None)
            self._reset_preview_stage_caches()
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
        self._update_preview_status_overlay()
        if self._apply_gpu_display_adjustment(previous):
            self._schedule_roll_session_save()
            return
        if self.negative_preview_active:
            if self._render_controller.in_progress:
                self._render_controller.defer(self.current_path, show_errors=False)
            self.preview_refresh_timer.setInterval(
                INTERACTIVE_RENDER_DEBOUNCE_MS
                if self._interactive_adjustment_active
                else FINAL_RENDER_DEBOUNCE_MS
            )
            self.preview_refresh_timer.start()
        self._schedule_roll_session_save()

    def undo_adjustments(self) -> None:
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo")
            return
        previous = self._undo_stack.pop()
        self._redo_stack.append(deepcopy(self.adjustments))
        self._apply_adjustments_from_history(previous, "Undo")

    def redo_adjustments(self) -> None:
        if not self._redo_stack:
            self.statusBar().showMessage("Nothing to redo")
            return
        next_adjustments = self._redo_stack.pop()
        self._undo_stack.append(deepcopy(self.adjustments))
        self._apply_adjustments_from_history(next_adjustments, "Redo")

    def _apply_adjustments_from_history(self, adjustments: AdjustmentParams, label: str) -> None:
        self._applying_history = True
        try:
            self.control_panel.set_adjustments(deepcopy(adjustments), emit=True)
        finally:
            self._applying_history = False
        self.statusBar().showMessage(label)

    def _apply_gpu_display_adjustment(self, previous: AdjustmentParams) -> bool:
        del previous
        # The experimental shader path used an 8-bit linear texture, which can
        # quantize deep shadows before display gamma and bring back black-field
        # artifacts. Keep final preview/display on the CPU pipeline for now.
        return False

    def preview_inversion(self) -> None:
        if self._manual_levels_present():
            self.auto_levels_pending = False
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
        auto_levels_pending = self.auto_levels_pending and not self._manual_levels_present()
        if self.auto_levels_pending and not auto_levels_pending:
            self.auto_levels_pending = False

        if self._render_controller.in_progress:
            self._render_controller.defer(self.current_path, show_errors=show_errors)
            self.statusBar().showMessage("Preview render queued...")
            return True

        render_start = self._render_controller.start(render_preview.path)
        task = PreviewRenderTask(
            job_id=render_start.job_id,
            render_token=render_start.render_token,
            preview=render_preview,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=self.adjustments,
            auto_levels_pending=auto_levels_pending,
            show_errors=show_errors,
            quality=quality,
            file_key=self._file_key_for_path(render_preview.path),
            lab_print_cmy_offsets=self.lab_print_cmy_offsets,
            roll_color_result=self._roll_color_result,
            roll_color_frame=self._roll_color_frame_for_path(render_preview.path),
            render_cache=self._preview_stage_caches[quality],
        )
        task.signals.finished.connect(self._preview_render_finished)
        task.signals.failed.connect(self._preview_render_failed)
        self.statusBar().showMessage(
            "Rendering interactive preview..."
            if interactive
            else "Rendering preview..."
        )
        self._thread_pool.start(task)
        return True

    def _preview_render_finished(self, job_id: int, output: PreviewRenderOutput, show_errors: bool) -> None:
        if not self._render_controller.output_is_current(output):
            if self._render_controller.is_latest_job(job_id):
                self._render_controller.mark_idle()
                self._queue_pending_render_if_needed()
            return

        if not self._render_controller.is_latest_job(job_id):
            self._store_stale_preview_result(output)
            return

        self._render_controller.mark_idle()
        if output.path != self.current_path:
            self._store_stale_preview_result(output)
            if self._render_controller.has_pending():
                self._queue_pending_render_if_needed()
            return

        self._preview_stage_caches[output.quality] = output.cache
        result = output.result

        if output.applied_auto_levels:
            self._apply_auto_levels(result.auto_levels)
            self.auto_levels_pending = False

        self._last_untransformed_negative_result = result
        if output.quality == FINAL_RENDER_QUALITY:
            self.lab_print_cmy_offsets = output.lab_print_cmy_offsets
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

        if self._render_controller.has_pending():
            self._queue_pending_render_if_needed()

    def _queue_pending_render_if_needed(self) -> None:
        pending = self._render_controller.consume_pending()
        if pending is None:
            return
        self._queue_negative_render(
            show_errors=pending.show_errors,
            interactive=self._interactive_adjustment_active and not pending.show_errors,
        )

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

        self.image_states[output.path] = merge_stale_preview_result_state(
            self.image_states.get(output.path),
            output,
        )
        self.filmstrip.set_processed_thumbnail(
            output.path,
            self._pixmap_from_rgb8(output.result.display_rgb8),
        )
        self.statusBar().showMessage(f"Background preview cached: {output.path.name}")
        self._autosave_roll_session()

    def _preview_render_failed(self, job_id: int, message: str, show_errors: bool) -> None:
        if not self._render_controller.is_latest_job(job_id):
            return

        self._render_controller.mark_idle()
        self.last_negative_result = None
        self._last_untransformed_negative_result = None
        if show_errors:
            QMessageBox.warning(self, "Invert Preview Failed", message)
        self.statusBar().showMessage("Inverted preview failed")

        if self._render_controller.has_pending():
            self._queue_pending_render_if_needed()

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
        return False

    def _manual_levels_present(self) -> bool:
        return (
            self.adjustments.black_point != 0
            or self.adjustments.mid_point != 50
            or self.adjustments.white_point != 100
        )

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
            lab_print_cmy_offsets=self.lab_print_cmy_offsets,
            roll_color_frame=self._roll_color_frame_for_path(self.current_path),
        )

    def _roll_color_frame_for_path(self, path: Path | None) -> dict | None:
        if path is None:
            return None
        state = self.image_states.get(path)
        if state is not None and state.roll_color_frame is not None:
            return deepcopy(state.roll_color_frame)
        return None

    def _update_roll_color_status(self) -> None:
        text = roll_color_result_summary(self._roll_color_result)
        frame = self._roll_color_frame_for_path(self.current_path)
        if frame:
            details = self._roll_color_frame_status_lines(frame)
            if details:
                text = f"{text}\n\nCurrent frame\n" + "\n".join(details)
        self.control_panel.set_roll_color_status(text)

    def _roll_color_frame_status_lines(self, frame: dict) -> list[str]:
        action = str(frame.get("color_action") or "none")
        confidence = _float_or_none(frame.get("confidence"))
        tone_confidence = _float_or_none(frame.get("tone_confidence"))
        protection_strength = _float_or_none(frame.get("highlight_protection_strength")) or 0.0
        protection_region = str(frame.get("highlight_protected_region") or "")
        protection_warning = str(frame.get("highlight_protection_warning") or "")
        exposure_delta = _float_or_none(frame.get("exposure_delta_stops")) or 0.0
        exposure_action = str(frame.get("exposure_action") or "")

        lines = [f"Action: {action}" + (f" ({confidence:.2f})" if confidence is not None else "")]
        if tone_confidence is not None and tone_confidence > 0:
            lines.append(f"Tone residual: {tone_confidence:.2f}")
        if protection_strength > 0 or protection_region or protection_warning:
            protection = f"Protection: {protection_strength * 100:.0f}%"
            if protection_region:
                protection += f" {protection_region}"
            if protection_warning:
                protection += f" / {protection_warning}"
            lines.append(protection)
        else:
            lines.append("Protection: none")
        if abs(exposure_delta) >= 0.001:
            suffix = f" {exposure_action}" if exposure_action and exposure_action != "none" else ""
            lines.append(f"Exposure match: {exposure_delta:+.2f} stops{suffix}")
        else:
            lines.append("Exposure match: none")
        return lines

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

        return self._show_cached_preview_result_for_path(self.current_path, status="Cached positive preview restored")

    def _show_cached_preview_result_for_path(self, path: Path, *, status: str = "Cached positive preview shown") -> bool:
        cached = self.preview_result_cache.get(path)
        if cached is None:
            return False

        self.preview_result_cache.pop(path, None)
        self.preview_result_cache[path] = cached
        self._last_untransformed_negative_result = cached.result
        displayed_result, _pixmap = self._update_preview_from_result(
            cached.result,
            update_filmstrip=True,
        )
        self.control_panel.set_histogram(displayed_result.histogram)
        self._set_negative_preview_status(cached.result, displayed_result)
        self.negative_preview_active = True
        self.statusBar().showMessage(status)
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
        wb_label = "Printer CMY"
        self.control_panel.set_tone_mid_anchor(result.tone_mid_anchor)
        self.control_panel.set_image_status(
            f"Positive preview {displayed_result.width} x {displayed_result.height}\n"
            f"Mode {invert_mode_label(self.adjustments.invert_mode)}\n"
            f"{base_text}\n"
            f"{wb_label} {wb_gains}"
        )
        self._update_preview_status_overlay()

    def _update_preview_status_overlay(self) -> None:
        axis = self.adjustments.printer_balance
        self.preview_view.set_status_overlay(
            "Printer  R/C {red:+d}  G/M {green:+d}  B/Y {blue:+d}\n"
            "Exp {exposure:+d}   Gray {mid}".format(
                red=axis.red_cyan,
                green=axis.green_magenta,
                blue=axis.blue_yellow,
                exposure=self.adjustments.exposure,
                mid=self.adjustments.mid_point,
            )
        )

    def _negative_base_for_current(self) -> NegativeBasePreview:
        if self.current_preview is None:
            raise PipelineError("Open a RAW file and generate a linear preview first.")

        key: tuple[object, ...] = (
            self.current_preview.path,
            self.current_preview.source_size,
            self.current_preview.preview_linear_rgb.shape,
            id(self.current_preview.preview_linear_rgb),
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
            QMessageBox.information(self, "Export Image", "Open a RAW file before exporting.")
            return
        if self._film_base_required_for_current_mode() and self.mask_point is None:
            QMessageBox.information(self, "Export Image", "Pick the film base before exporting.")
            return
        if self.film_rect is None or not self.film_rect.is_valid():
            QMessageBox.information(self, "Export Image", "Select a valid frame area before exporting.")
            return

        if self._default_export_dir is not None:
            default_path = self._default_export_dir / f"{self.current_path.stem}_positive.tif"
        else:
            default_path = self.current_path.with_name(f"{self.current_path.stem}_positive.tif")
        output, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            str(default_path),
            "TIFF 16-bit RGB (*.tif *.tiff);;TIFF 8-bit RGB (*.tif *.tiff);;JPEG Image (*.jpg *.jpeg);;PNG 16-bit RGB (*.png);;PNG 8-bit RGB (*.png)",
        )
        if not output:
            return

        output_path = Path(output)
        known_suffix = output_path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        export_format = (
            export_format_from_path(output_path)
            if known_suffix
            else export_format_from_filter(selected_filter) or "tiff16"
        )
        if not known_suffix:
            output_path = output_path.with_suffix(export_format_extension(export_format))

        self._export_cancel_event = Event()
        task = ImageExportTask(
            source_path=self.current_path,
            output_path=output_path,
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            adjustments=self.adjustments,
            flip_horizontal=self._preview_flip_horizontal,
            flip_vertical=self._preview_flip_vertical,
            rotation_quarters=self._preview_rotation_quarters,
            auto_levels_pending=self.auto_levels_pending,
            export_format=export_format,
            preview_cmy_offsets=self._current_preview_cmy_offsets_for_export(),
            preview_tone_mid_anchor=self._current_preview_tone_mid_anchor_for_export(),
            roll_color_result=self._roll_color_result,
            roll_color_frame=self._roll_color_frame_for_path(self.current_path),
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
        self.statusBar().showMessage(f"Exporting {export_format_label(export_format)}...")
        self._thread_pool.start(task)

    def export_completed(self) -> None:
        if self._export_in_progress:
            self.statusBar().showMessage("Export already in progress")
            return
        if self.current_path is not None:
            self._save_current_state()

        default_dir = self._default_export_dir or (
            self.current_path.parent if self.current_path else Path.cwd()
        )
        settings_dialog = BatchExportSettingsDialog(
            default_dir=default_dir,
            default_prefix=self._default_batch_prefix(default_dir),
            parent=self,
        )
        if settings_dialog.exec() != QDialog.Accepted:
            return

        settings = settings_dialog.settings()
        self._default_export_dir = settings.output_dir
        items = self._completed_export_items(settings)
        if not items:
            QMessageBox.information(
                self,
                "Export Completed Images",
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
        self.batch_export_dialog.set_jobs(
            [item["source_path"] for item in items],
            [item["output_path"] for item in items],
        )
        self.batch_export_dialog.show()
        self.batch_export_dialog.raise_()
        self.control_panel.export_button.setEnabled(False)
        self.control_panel.batch_export_button.setEnabled(False)
        self.control_panel.set_export_progress(True, value=0, text="Starting batch export")
        self.statusBar().showMessage(f"Exporting {self._batch_export_total} completed images...")
        self._start_next_batch_export()

    def _completed_export_items(self, settings: BatchExportSettings) -> list[dict]:
        ordered_paths = list(self.folder_files) if self.folder_files else list(self.image_states)
        for path in self.image_states:
            if path not in ordered_paths:
                ordered_paths.append(path)

        items: list[dict] = []
        sequence_index = int(settings.start_number)
        for path in ordered_paths:
            state = self.image_states.get(path)
            if state is None or not state.negative_preview_active:
                continue
            if path.suffix.lower() not in RAW_EXTENSIONS:
                continue
            if state.film_rect is None or not state.film_rect.is_valid():
                continue
            output_path = self._batch_export_output_path(
                path,
                settings,
                sequence_index=sequence_index,
            )
            if settings.naming_mode == "sequence":
                sequence_index += 1
            items.append(
                {
                    "source_path": path,
                    "output_path": output_path,
                    "mask_point": state.mask_point,
                    "film_rect": state.film_rect,
                    "adjustments": deepcopy(state.adjustments),
                    "flip_horizontal": state.preview_flip_horizontal,
                    "flip_vertical": state.preview_flip_vertical,
                    "rotation_quarters": state.preview_rotation_quarters,
                    "auto_levels_pending": state.auto_levels_pending,
                    "export_format": settings.export_format,
                    "preview_cmy_offsets": self._preview_cmy_offsets_for_path(path, state),
                    "preview_tone_mid_anchor": self._preview_tone_mid_anchor_for_path(path, state),
                    "roll_color_result": self._roll_color_result,
                    "roll_color_frame": deepcopy(state.roll_color_frame),
                }
            )
        return items

    def _default_batch_prefix(self, default_dir: Path) -> str:
        return self._project.default_batch_prefix(default_dir, self.current_path)

    def _batch_export_output_path(
        self,
        source_path: Path,
        settings: BatchExportSettings,
        *,
        sequence_index: int,
    ) -> Path:
        if settings.naming_mode == "same_name":
            filename = f"{source_path.stem}{export_format_extension(settings.export_format)}"
        else:
            filename = f"{settings.prefix}{sequence_index:03d}{export_format_extension(settings.export_format)}"
        output_path = settings.output_dir / filename
        if settings.overwrite_existing:
            return output_path
        return self._non_conflicting_output_path(output_path)

    @staticmethod
    def _non_conflicting_output_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for index in range(1, 10000):
            candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
            if not candidate.exists():
                return candidate
        return path.with_name(f"{stem}_{int(perf_counter() * 1000)}{suffix}")

    def _preview_cmy_offsets_for_path(self, path: Path, state: ImageProcessingState) -> np.ndarray | None:
        if not state.adjustments.auto_wb:
            return None
        if state.lab_print_cmy_offsets is not None:
            return np.asarray(state.lab_print_cmy_offsets, dtype=np.float32).copy()
        cached = self.preview_result_cache.get(path)
        if cached is None:
            return None
        return np.asarray(cached.result.wb_gains, dtype=np.float32).copy()

    def _preview_tone_mid_anchor_for_path(self, path: Path, state: ImageProcessingState) -> float | None:
        if state.tone_mid_anchor is not None:
            return float(state.tone_mid_anchor)
        cached = self.preview_result_cache.get(path)
        if cached is None:
            return None
        return float(cached.result.tone_mid_anchor)

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
                f"Batch exported {self._batch_export_done} images",
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
        task = ImageExportTask(**item)
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
        self.statusBar().showMessage("Batch export will pause after the current image")

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
        if not self.adjustments.auto_wb:
            return None
        if self.auto_levels_pending:
            return None
        if self.lab_print_cmy_offsets is not None:
            return np.asarray(self.lab_print_cmy_offsets, dtype=np.float32).copy()

        # Reuse preview CMY only when the final preview cache exactly matches
        # the current image/selection/adjustments. Otherwise export recomputes
        # auto WB at full resolution instead of risking stale color timing.
        cached = self.preview_result_cache.get(self.current_path)
        key = self._preview_result_cache_key()
        if cached is None or key is None or cached.key != key:
            return None
        return np.asarray(cached.result.wb_gains, dtype=np.float32).copy()

    def _current_preview_tone_mid_anchor_for_export(self) -> float | None:
        if self.current_path is None or self.auto_levels_pending:
            return None
        if self._last_untransformed_negative_result is not None:
            return float(self._last_untransformed_negative_result.tone_mid_anchor)

        cached = self.preview_result_cache.get(self.current_path)
        key = self._preview_result_cache_key()
        if cached is None or key is None or cached.key != key:
            return None
        return float(cached.result.tone_mid_anchor)

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
        self.statusBar().showMessage(f"Exported image: {output_path}{suffix}")
        if timing_text:
            print(f"Export timings: {timing_text}", flush=True)
        self._start_next_preinvert_jobs()

    @staticmethod
    def _format_export_timings(timings: dict[str, float]) -> str:
        if not timings:
            return ""
        total = sum(
            float(seconds)
            for name, seconds in timings.items()
            if export_timing_is_top_level(name)
        )
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
        QMessageBox.warning(self, "Export Failed", message)
        self.statusBar().showMessage("Export failed")

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
        self.control_panel.set_adjustments(default_adjustments(), emit=True)
        self.control_panel.set_histogram(None)
        self.preview_tabs.setCurrentWidget(self.origin_view)
        self.statusBar().showMessage("Workspace reset")

    def _cancel_preview_render(self) -> None:
        self._render_controller.cancel(self.current_path)

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
        elif self._roll_color_analysis_in_progress:
            self.control_panel.set_activity_progress(True, text="Analyzing roll color...")
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
            self._update_preview_status_overlay()
        finally:
            self._applying_auto_levels = False

    def _save_current_state(self) -> None:
        if self.current_path is None:
            return
        has_positive_result = (
            self.negative_preview_active
            or self._last_untransformed_negative_result is not None
            or self.last_negative_result is not None
        )
        self.image_states[self.current_path] = build_current_image_state(
            existing_state=self.image_states.get(self.current_path),
            mask_point=self.mask_point,
            film_rect=self.film_rect,
            white_balance_point=self.white_balance_point,
            adjustments=self.adjustments,
            lab_print_cmy_offsets=self.lab_print_cmy_offsets,
            tone_mid_anchor=(
                float(self._last_untransformed_negative_result.tone_mid_anchor)
                if self._last_untransformed_negative_result is not None
                else None
            ),
            has_positive_result=has_positive_result,
            manual_levels_present=self._manual_levels_present(),
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
        elif self._project.session_folder_for(self.current_path) is None:
            self.statusBar().showMessage("Open a folder before saving a roll session")
            return
        self._write_roll_session(show_status=True)

    def _restore_state_for_path(self, path: Path) -> bool:
        state = self.image_states.get(path)
        restored = restored_runtime_for_state(
            state,
            fallback_adjustments=self.adjustments,
        )
        self.mask_point = restored.mask_point
        self.film_rect = restored.film_rect
        self.white_balance_point = restored.white_balance_point
        self.lab_print_cmy_offsets = restored.lab_print_cmy_offsets
        self.adjustments = restored.adjustments
        self.negative_preview_active = restored.negative_preview_active
        self.auto_levels_pending = restored.auto_levels_pending
        self._preview_flip_horizontal = restored.preview_flip_horizontal
        self._preview_flip_vertical = restored.preview_flip_vertical
        self._preview_rotation_quarters = restored.preview_rotation_quarters

        if not restored.restored:
            self._reset_preview_transform()
            self.control_panel.set_mask_status("Not selected")
            self.control_panel.set_film_status("Not selected")
            self.control_panel.set_adjustments(self.adjustments, emit=False)
            self._update_preview_status_overlay()
            self._update_roll_color_status()
            self.origin_view.restore_selections(mask_point=None, film_rect=None)
            return False

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
        self._update_preview_status_overlay()
        self._update_roll_color_status()
        self.origin_view.restore_selections(
            mask_point=self.mask_point,
            film_rect=self.film_rect,
        )

        should_restore_positive = should_restore_positive_preview(
            state,
            manual_levels_present=self._manual_levels_present(),
        )
        if should_restore_positive and not self._restore_cached_preview_result():
            if self._manual_levels_present():
                self.auto_levels_pending = False
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
                adjustments=self.adjustments,
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

        self.image_states[output.path] = state_from_preinvert_output(output)
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
        super().keyPressEvent(event)

    def _nudge_global_balance(self, axis_name: str, delta: int) -> None:
        updated = deepcopy(self.adjustments)
        axis = updated.printer_balance
        current = int(getattr(axis, axis_name))
        setattr(axis, axis_name, self._clamp_adjustment(current + delta))
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"Printer {axis_name.replace('_', ' ')} {getattr(axis, axis_name):+d}")

    def _nudge_exposure(self, delta: int) -> None:
        updated = deepcopy(self.adjustments)
        updated.exposure = self._clamp_adjustment(updated.exposure + delta)
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"Exposure {updated.exposure:+d}")

    def _nudge_mid_point(self, delta: int) -> None:
        updated = deepcopy(self.adjustments)
        lower = max(0, updated.black_point + 1)
        upper = min(100, updated.white_point - 1)
        updated.mid_point = max(lower, min(upper, updated.mid_point + int(delta)))
        self.control_panel.set_adjustments(updated, emit=True)
        self.statusBar().showMessage(f"Gray point {updated.mid_point}")

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
        self.preview_refresh_timer.stop()
        self.roll_session_save_timer.stop()
        self._cancel_preview_render()
        self._cancel_raw_preview()
        self._cancel_auto_detect()
        self._preinvert_queue = []
        self._preinvert_in_progress.clear()
        self._preinvert_paths.clear()
        self._batch_export_queue = []
        self._batch_export_cancel_requested = True
        if self._export_cancel_event is not None:
            self._export_cancel_event.set()
        self._save_current_state()
        self._thread_pool.waitForDone(1500)
        super().closeEvent(event)

    def _set_folder_sequence(self, path: Path) -> None:
        if self.current_path is not None:
            self._save_current_state()
        self._preinvert_queue = []
        result = self._project.load_folder(path, self.image_states)
        self.folder_files = result.files
        self.current_index = result.current_index
        self._roll_color_result = result.roll_color_result
        self._update_roll_color_status()
        self._sync_sequence_position(path)
        self.filmstrip.set_files(self.folder_files, path)
        self._restore_filmstrip_session_badges()
        if result.restored_count:
            self.statusBar().showMessage(
                f"Loaded roll session: {result.restored_count} saved images"
            )

    def _restore_filmstrip_session_badges(self) -> None:
        for source_path, processed in self._project.filmstrip_badges(self.image_states):
            self.filmstrip.set_processed_badge(
                source_path,
                processed,
            )

    def _autosave_roll_session(self) -> None:
        if not self._roll_session_autosave:
            return
        self._write_roll_session(show_status=False)

    def _write_roll_session(self, *, show_status: bool) -> None:
        try:
            session_path = self._project.save_session(
                image_states=self.image_states,
                current_path=self.current_path,
                roll_color_result=self._roll_color_result,
            )
        except OSError as exc:
            self.statusBar().showMessage(f"Roll session save failed: {exc}")
            return
        if show_status and session_path is not None:
            self.statusBar().showMessage(f"Saved roll session: {session_path}")

    def _sync_sequence_position(self, path: Path) -> None:
        self.current_index = self._project.sync_position(path)
        self.control_panel.set_sequence_status(self._project.sequence_status_text())

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #1A1A1A;
            }
            QMenuBar {
                background: #121212;
                color: #F2EEE6;
                border-bottom: 1px solid #38312A;
                padding: 2px 6px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
                color: #F2EEE6;
            }
            QMenuBar::item:selected {
                background: #302A24;
                border-radius: 4px;
            }
            QMenu {
                background: #1A1A1A;
                color: #F2EEE6;
                border: 1px solid #3D352D;
                padding: 5px 0;
            }
            QMenu::item {
                padding: 7px 28px 7px 24px;
            }
            QMenu::item:selected {
                background: #3A3026;
            }
            QMenu::separator {
                height: 1px;
                background: #3D352D;
                margin: 5px 8px;
            }
            QStatusBar {
                background: #202020;
                color: #D8D0C2;
                border-top: 1px solid #38312A;
            }
            QWidget#emptyState {
                background: #1A1A1A;
            }
            QPushButton#emptyOpenButton {
                background: #663300;
                border: 1px solid #FFB000;
                border-radius: 7px;
                color: #ffffff;
                font-size: 16px;
                font-weight: 600;
                padding: 14px 24px;
                min-width: 220px;
            }
            QPushButton#emptyOpenButton:hover {
                background: #7A3D00;
            }
            QSplitter::handle {
                background: #38312A;
            }
            QSplitter::handle:horizontal {
                width: 5px;
            }
            QSplitter::handle:hover {
                background: #5C5144;
            }
            QTabWidget#previewTabs::pane {
                border: none;
                background: #1A1A1A;
            }
            QTabBar::tab {
                background: #202020;
                color: #C7BBA8;
                padding: 8px 18px;
                border-right: 1px solid #38312A;
            }
            QTabBar::tab:selected {
                background: #302A24;
                color: #F2EEE6;
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
