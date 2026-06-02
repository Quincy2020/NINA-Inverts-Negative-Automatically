from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.file_sequence import IMAGE_EXTENSIONS, RAW_EXTENSIONS, list_supported_files
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
    build_negative_base_preview,
    build_density_preview_analysis,
    process_negative_base_preview,
    suggest_global_balance_from_neutral,
)
from qnegative.core.preview import DEFAULT_PREVIEW_MAX_EDGE, RawPreview, make_raw_preview
from qnegative.ui.control_panel import ControlPanel
from qnegative.ui.folder_filmstrip import FolderFilmstrip
from qnegative.ui.image_view import ImageView


class MainWindow(QMainWindow):
    def __init__(self, *, default_invert_mode: str = InvertMode.NEGPY_PRINT.value) -> None:
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

        self.control_panel = ControlPanel()
        self.image_view = ImageView()
        self.filmstrip = FolderFilmstrip()
        self.preview_refresh_timer = QTimer(self)
        self.preview_refresh_timer.setSingleShot(True)
        self.preview_refresh_timer.setInterval(80)

        self._build_layout()
        self._build_developer_menu()
        self._connect()
        self._apply_style()
        self.control_panel.set_adjustments(self.adjustments, emit=False)

        self.statusBar().showMessage("Ready")

    def _build_layout(self) -> None:
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.image_view, 1)
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

    def _connect(self) -> None:
        self.control_panel.openRequested.connect(self.open_file)
        self.control_panel.exportRequested.connect(self.export_current)
        self.control_panel.invertRequested.connect(self.preview_inversion)
        self.control_panel.resetRequested.connect(self.reset_workspace)
        self.control_panel.toolChanged.connect(self.set_tool_mode)
        self.control_panel.adjustmentsChanged.connect(self.adjustments_changed)

        self.image_view.maskPointSelected.connect(self.mask_point_selected)
        self.image_view.whiteBalancePointSelected.connect(self.white_balance_point_selected)
        self.image_view.filmRectSelected.connect(self.film_rect_selected)
        self.image_view.filmRectReset.connect(self.film_rect_reset)
        self.image_view.viewStatusChanged.connect(self.statusBar().showMessage)

        self.filmstrip.fileSelected.connect(self.select_sequence_file)
        self.filmstrip.previousRequested.connect(self.go_previous_file)
        self.filmstrip.nextRequested.connect(self.go_next_file)
        self.preview_refresh_timer.timeout.connect(lambda: self._render_negative_preview(show_errors=False))

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

        if refresh_sequence:
            self._set_folder_sequence(path)
        else:
            self._sync_sequence_position(path)

        self.current_path = path
        self.current_preview = None
        self.negative_preview_active = False
        self.auto_levels_pending = True
        self.white_balance_point = None
        self._invalidate_negative_base_cache()
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
            self._restore_state_for_path(path)
            return

        self.image_view.set_placeholder(f"Unsupported file type: {extension or 'unknown'}")
        self.control_panel.set_image_loaded(False)
        self.control_panel.set_image_status("Unsupported file type")
        self.control_panel.set_histogram(None)

    def load_raw_preview(self, path: Path) -> None:
        self.statusBar().showMessage("Generating RAW 1080 preview...")
        self.control_panel.set_image_loaded(False)
        self.control_panel.set_image_status("Decoding RAW...")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            preview = make_raw_preview(path, max_size=DEFAULT_PREVIEW_MAX_EDGE)
            pixmap = self._pixmap_from_rgb8(preview.display_rgb8)
        except Exception as exc:
            self.image_view.set_raw_placeholder(path)
            self.control_panel.set_image_loaded(False)
            self.control_panel.set_image_status("RAW preview failed")
            QMessageBox.warning(self, "RAW Preview Failed", str(exc))
            self.statusBar().showMessage("RAW preview failed")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.current_preview = preview
        self.control_panel.set_histogram(None)
        self.image_view.set_preview_pixmap(
            pixmap,
            source_path=path,
            source_size=preview.source_size,
        )
        self.control_panel.set_image_loaded(True)
        self.control_panel.set_image_status(preview.status_text())
        self.statusBar().showMessage(f"RAW preview ready: {path.name}")

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.image_view.set_tool_mode(mode)

    def mask_point_selected(self, point: ImagePoint) -> None:
        self.mask_point = point
        self.white_balance_point = None
        self.auto_levels_pending = True
        self._invalidate_negative_base_cache()
        self.control_panel.set_mask_status(f"Base point: x={point.x}, y={point.y}")
        self.statusBar().showMessage("Base picker point saved")

    def film_rect_selected(self, rect: ImageRect) -> None:
        self.film_rect = rect
        self.white_balance_point = None
        self.auto_levels_pending = True
        self._invalidate_negative_base_cache()
        self.control_panel.set_film_status(f"Frame: {rect.label()}")
        self.statusBar().showMessage("Frame area saved")

    def film_rect_reset(self) -> None:
        self.film_rect = None
        self.white_balance_point = None
        self.auto_levels_pending = True
        self._invalidate_negative_base_cache()
        self.control_panel.set_film_status("Not selected")
        self.statusBar().showMessage("Frame reset")

    def white_balance_point_selected(self, point: ImagePoint) -> None:
        if not self.negative_preview_active or self.last_negative_result is None:
            self.white_balance_point = None
            self.image_view.restore_selections(
                mask_point=self.mask_point,
                film_rect=self.film_rect,
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
            "Adjust: mode {invert_mode}, curve {print_curve}, WB {auto_wb}, exposure {exposure}, highlights {highlights}, shadows {shadows}, contrast {contrast}, saturation {saturation}, camera color {camera_color_strength}, black {black_point}, mid {mid_point}, white {white_point}".format(
                **values
            )
        )
        if self.negative_preview_active:
            self.preview_refresh_timer.start()

    def preview_inversion(self) -> None:
        self._render_negative_preview(show_errors=True)

    def _render_negative_preview(self, *, show_errors: bool) -> bool:
        if self.current_preview is None:
            if show_errors:
                QMessageBox.information(self, "Invert Preview", "Open a RAW file and generate a linear preview first.")
            return False

        self.statusBar().showMessage("Generating inverted preview...")
        if show_errors:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            base = self._negative_base_for_current()
            density_analysis = (
                self._density_analysis_for_current(base)
                if self.adjustments.invert_mode == InvertMode.DENSITY.value
                else None
            )
            if self.auto_levels_pending and density_analysis is not None:
                self._apply_auto_levels(density_analysis.auto_levels)
                self.auto_levels_pending = False

            result = process_negative_base_preview(
                base,
                self.adjustments,
                density_analysis=density_analysis,
            )
            pixmap = self._pixmap_from_rgb8(result.display_rgb8)
        except PipelineError as exc:
            if show_errors:
                QMessageBox.information(self, "Invert Preview", str(exc))
            self.last_negative_result = None
            self.statusBar().showMessage("Inverted preview not ready")
            return False
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(self, "Invert Preview Failed", str(exc))
            self.last_negative_result = None
            self.statusBar().showMessage("Inverted preview failed")
            return False
        finally:
            if show_errors:
                QApplication.restoreOverrideCursor()

        if self.auto_levels_pending:
            self._apply_auto_levels(result.auto_levels)
            self.auto_levels_pending = False
            return self._render_negative_preview(show_errors=False)

        self.image_view.set_preview_pixmap(
            pixmap,
            source_path=self.current_path or self.current_preview.path,
            source_size=ImageSize(width=result.width, height=result.height),
        )
        self.image_view.restore_selections(
            mask_point=None,
            film_rect=None,
            white_balance_point=self.white_balance_point,
        )
        self.last_negative_result = result
        mask_rgb = ", ".join(f"{value:.4f}" for value in result.mask_rgb)
        wb_gains = ", ".join(f"{value:.3f}" for value in result.wb_gains)
        wb_label = (
            "WB CMY offset"
            if self.adjustments.invert_mode
            in (InvertMode.LOG_BOUNDS.value, InvertMode.NEGPY_PRINT.value)
            else "WB gain"
        )
        self.control_panel.set_image_status(
            f"Positive preview {result.width} x {result.height}\nMode {self.adjustments.invert_mode}\nBase RGB {mask_rgb}\n{wb_label} {wb_gains}"
        )
        self.control_panel.set_histogram(result.histogram)
        self.negative_preview_active = True
        self.statusBar().showMessage("Inverted preview ready")
        return True

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
        self._invalidate_density_analysis_cache()
        self.last_negative_result = None

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
        QMessageBox.information(
            self,
            "Export",
            "Export is reserved for the next pipeline stage.",
        )

    def reset_workspace(self) -> None:
        self.image_view.clear_selections()
        self.preview_refresh_timer.stop()
        self.negative_preview_active = False
        self.auto_levels_pending = True
        self.mask_point = None
        self.film_rect = None
        self.white_balance_point = None
        self._invalidate_negative_base_cache()
        self.control_panel.set_mask_status("Not selected")
        self.control_panel.set_film_status("Not selected")
        self.control_panel.set_adjustments(self._default_adjustments(), emit=True)
        self.control_panel.set_histogram(None)
        self.statusBar().showMessage("Workspace reset")

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
            self.control_panel.set_mask_status("Not selected")
            self.control_panel.set_film_status("Not selected")
            self.control_panel.set_adjustments(self.adjustments, emit=False)
            return

        self.mask_point = state.mask_point
        self.film_rect = state.film_rect
        self.white_balance_point = state.white_balance_point
        self.adjustments = deepcopy(state.adjustments)
        self.negative_preview_active = False
        self.auto_levels_pending = state.auto_levels_pending

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
        self.image_view.restore_selections(
            mask_point=self.mask_point,
            film_rect=self.film_rect,
        )

        if state.negative_preview_active:
            self._render_negative_preview(show_errors=False)

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
