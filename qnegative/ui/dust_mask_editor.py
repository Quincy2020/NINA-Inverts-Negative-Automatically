from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.dust_masks import dust_auto_mask_params_key, resize_mask
from qnegative.core.models import DustRemovalParams, ImagePoint, ImageSize
from qnegative.ui.dust_mask_tasks import DustMaskPreviewTask


class DustMaskCanvas(QWidget):
    masksChanged = Signal()
    statusChanged = Signal(str)

    def __init__(
        self,
        *,
        image_rgb8: np.ndarray,
        auto_mask: np.ndarray | None,
        add_mask: np.ndarray | None,
        protect_mask: np.ndarray | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("dustMaskCanvas")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(900, 620)

        image = np.ascontiguousarray(image_rgb8.astype(np.uint8, copy=True))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Dust mask editor image must be RGB.")
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate([image, alpha], axis=2))
        self._image = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.shape[1] * 4,
            QImage.Format_RGBA8888,
        ).copy()
        self._size = ImageSize(width=self._image.width(), height=self._image.height())
        shape = (self._size.height, self._size.width)
        self._auto_mask = resize_mask(auto_mask, shape)
        self._add_mask = resize_mask(add_mask, shape)
        self._protect_mask = resize_mask(protect_mask, shape)
        if self._add_mask is None:
            self._add_mask = np.zeros(shape, dtype=bool)
        if self._protect_mask is None:
            self._protect_mask = np.zeros(shape, dtype=bool)
        self._overlay_image: QImage | None = None
        self._show_auto = True

        self._display_rect = QRectF()
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._pan_start: QPoint | None = None
        self._pan_offset_start = QPointF(0.0, 0.0)
        self._is_panning = False
        self._space_pan_active = False

        self._brush_mode = "add"
        self._brush_radius = 16
        self._is_brushing = False
        self._brush_last_point: ImagePoint | None = None
        self._cursor_point: ImagePoint | None = None

        self._rebuild_overlay()

    @property
    def add_mask(self) -> np.ndarray:
        return self._add_mask.astype(bool, copy=True)

    @property
    def protect_mask(self) -> np.ndarray:
        return self._protect_mask.astype(bool, copy=True)

    @property
    def auto_mask(self) -> np.ndarray | None:
        return self._auto_mask.astype(bool, copy=True) if self._auto_mask is not None else None

    def set_auto_mask(self, mask: np.ndarray | None) -> None:
        self._auto_mask = resize_mask(mask, (self._size.height, self._size.width))
        self._rebuild_overlay()
        self.update()

    def set_show_auto(self, show: bool) -> None:
        self._show_auto = bool(show)
        self._rebuild_overlay()
        self.update()

    def set_brush_mode(self, mode: str) -> None:
        self._brush_mode = mode if mode in {"add", "protect", "erase"} else "add"
        self._update_cursor()
        self.update()

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, int(radius))
        self.update()

    def set_space_pan_active(self, active: bool) -> None:
        if self._space_pan_active == bool(active):
            return
        self._space_pan_active = bool(active)
        if not self._space_pan_active:
            self._is_panning = False
            self._pan_start = None
        self._update_cursor()
        self.update()

    def clear_manual_masks(self) -> None:
        self._add_mask[:, :] = False
        self._protect_mask[:, :] = False
        self._rebuild_overlay()
        self.masksChanged.emit()
        self.update()

    def reset_view(self) -> None:
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#161616"))
        self._display_rect = self._scaled_display_rect()
        if self._display_rect.isEmpty():
            painter.end()
            return
        painter.fillRect(self._display_rect, QColor("#050505"))
        painter.drawImage(self._display_rect.toRect(), self._image)
        if self._overlay_image is not None:
            painter.drawImage(self._display_rect.toRect(), self._overlay_image)
        painter.setPen(QColor("#443B32"))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self._display_rect.toRect().adjusted(0, 0, -1, -1))
        self._paint_brush_cursor(painter)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MiddleButton, Qt.RightButton) or self._space_pan_active:
            self._begin_pan(event.position().toPoint())
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            return
        point = self._view_to_image_point(event.position())
        if point is None:
            return
        self._is_brushing = True
        self._brush_last_point = point
        self._cursor_point = point
        self._paint_stroke(point, previous=None)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._is_panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_offset = self._pan_offset_start + QPointF(delta)
            self.update()
            event.accept()
            return
        point = self._view_to_image_point(event.position())
        self._cursor_point = point
        if point is not None:
            self.statusChanged.emit(f"x={point.x}, y={point.y}")
        if self._is_brushing and point is not None:
            self._paint_stroke(point, previous=self._brush_last_point)
            self._brush_last_point = point
            event.accept()
            return
        self._update_cursor()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MiddleButton, Qt.RightButton) or self._is_panning:
            self._is_panning = False
            self._pan_start = None
            self._update_cursor()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._is_brushing:
            self._is_brushing = False
            self._brush_last_point = None
            event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._display_rect.isEmpty():
            self._display_rect = self._scaled_display_rect()
        if self._display_rect.isEmpty():
            return
        cursor = event.position()
        if not self._display_rect.contains(cursor):
            cursor = self._display_rect.center()
        x_norm = (cursor.x() - self._display_rect.left()) / max(1.0, self._display_rect.width())
        y_norm = (cursor.y() - self._display_rect.top()) / max(1.0, self._display_rect.height())
        step = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self._zoom_factor = max(0.2, min(12.0, self._zoom_factor * step))
        next_rect = self._scaled_display_rect()
        next_cursor = QPointF(
            next_rect.left() + x_norm * next_rect.width(),
            next_rect.top() + y_norm * next_rect.height(),
        )
        self._pan_offset += cursor - next_cursor
        self.update()
        event.accept()

    def _begin_pan(self, point: QPoint) -> None:
        self._is_panning = True
        self._pan_start = point
        self._pan_offset_start = QPointF(self._pan_offset)
        self.setCursor(Qt.ClosedHandCursor)

    def _paint_stroke(self, point: ImagePoint, *, previous: ImagePoint | None) -> None:
        if self._brush_mode == "add":
            self._draw_on_mask(self._add_mask, point, previous=previous, value=True)
            self._draw_on_mask(self._protect_mask, point, previous=previous, value=False)
        elif self._brush_mode == "protect":
            self._draw_on_mask(self._protect_mask, point, previous=previous, value=True)
            self._draw_on_mask(self._add_mask, point, previous=previous, value=False)
        else:
            self._draw_on_mask(self._add_mask, point, previous=previous, value=False)
            self._draw_on_mask(self._protect_mask, point, previous=previous, value=False)
        self._rebuild_overlay()
        self.masksChanged.emit()
        self.update()

    def _draw_on_mask(
        self,
        mask: np.ndarray,
        point: ImagePoint,
        *,
        previous: ImagePoint | None,
        value: bool,
    ) -> None:
        canvas = mask.astype(np.uint8, copy=True)
        color = 1 if value else 0
        center = (int(point.x), int(point.y))
        if isinstance(previous, ImagePoint):
            cv2.line(
                canvas,
                (int(previous.x), int(previous.y)),
                center,
                color,
                thickness=max(1, self._brush_radius * 2),
                lineType=cv2.LINE_8,
            )
        cv2.circle(canvas, center, self._brush_radius, color, thickness=-1, lineType=cv2.LINE_8)
        mask[:, :] = canvas > 0

    def _scaled_display_rect(self) -> QRectF:
        margin = 18
        available = self.rect().adjusted(margin, margin, -margin, -margin)
        if available.width() <= 0 or available.height() <= 0:
            return QRectF()
        image_ratio = self._size.width / max(1, self._size.height)
        available_ratio = available.width() / max(1, available.height())
        if image_ratio > available_ratio:
            width = available.width()
            height = width / image_ratio
        else:
            height = available.height()
            width = height * image_ratio
        width *= self._zoom_factor
        height *= self._zoom_factor
        return QRectF(
            available.left() + (available.width() - width) / 2.0 + self._pan_offset.x(),
            available.top() + (available.height() - height) / 2.0 + self._pan_offset.y(),
            width,
            height,
        )

    def _view_to_image_point(self, point) -> ImagePoint | None:
        if self._display_rect.isEmpty():
            self._display_rect = self._scaled_display_rect()
        if self._display_rect.isEmpty() or not self._display_rect.contains(point):
            return None
        x_norm = (point.x() - self._display_rect.left()) / max(1.0, self._display_rect.width())
        y_norm = (point.y() - self._display_rect.top()) / max(1.0, self._display_rect.height())
        x = int(round(x_norm * (self._size.width - 1)))
        y = int(round(y_norm * (self._size.height - 1)))
        return ImagePoint(
            x=max(0, min(self._size.width - 1, x)),
            y=max(0, min(self._size.height - 1, y)),
        )

    def _image_point_to_view(self, point: ImagePoint) -> QPointF | None:
        if self._display_rect.isEmpty():
            self._display_rect = self._scaled_display_rect()
        if self._display_rect.isEmpty():
            return None
        x = self._display_rect.left() + point.x / max(1, self._size.width - 1) * self._display_rect.width()
        y = self._display_rect.top() + point.y / max(1, self._size.height - 1) * self._display_rect.height()
        return QPointF(x, y)

    def _paint_brush_cursor(self, painter: QPainter) -> None:
        if self._space_pan_active or self._cursor_point is None:
            return
        view_point = self._image_point_to_view(self._cursor_point)
        if view_point is None:
            return
        radius_x = self._brush_radius / max(1, self._size.width) * self._display_rect.width()
        radius_y = self._brush_radius / max(1, self._size.height) * self._display_rect.height()
        radius = max(2.0, float(radius_x + radius_y) * 0.5)
        color = {
            "add": QColor("#28DCFF"),
            "protect": QColor("#FF46AA"),
            "erase": QColor("#F2EEE6"),
        }.get(self._brush_mode, QColor("#28DCFF"))
        painter.setPen(color)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(view_point.x() - radius, view_point.y() - radius, radius * 2.0, radius * 2.0))

    def _rebuild_overlay(self) -> None:
        rgba = np.zeros((self._size.height, self._size.width, 4), dtype=np.uint8)
        if self._show_auto:
            _apply_overlay_color(rgba, self._auto_mask, (255, 176, 0, 105))
        _apply_overlay_color(rgba, self._add_mask, (40, 220, 255, 135))
        _apply_overlay_color(rgba, self._protect_mask, (255, 70, 170, 145))
        if not np.any(rgba[..., 3]):
            self._overlay_image = None
            return
        self._overlay_image = QImage(
            rgba.data,
            self._size.width,
            self._size.height,
            self._size.width * 4,
            QImage.Format_RGBA8888,
        ).copy()

    def _update_cursor(self) -> None:
        self.setCursor(Qt.OpenHandCursor if self._space_pan_active else Qt.CrossCursor)


class DustMaskEditorDialog(QDialog):
    def __init__(
        self,
        *,
        source_path: Path | None,
        image_rgb8: np.ndarray,
        linear_rgb: np.ndarray,
        params: DustRemovalParams,
        auto_mask: np.ndarray | None,
        add_mask: np.ndarray | None,
        protect_mask: np.ndarray | None,
        thread_pool: QThreadPool,
        auto_mask_params_key: str | None = None,
        params_provider: Callable[[], DustRemovalParams] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("dustMaskEditor")
        self.setWindowTitle("Dust Mask Editor")
        self.resize(1320, 900)
        self.setModal(False)
        self._source_path = source_path
        self._linear_rgb = np.ascontiguousarray(linear_rgb.astype(np.float32, copy=True))
        self._params = deepcopy(params)
        self._params_provider = params_provider
        self._auto_mask_params_key = auto_mask_params_key
        self._pending_auto_mask_params_key: str | None = None
        self._thread_pool = thread_pool
        self._auto_task: DustMaskPreviewTask | None = None
        self._auto_job_id = 0
        self._auto_cancel_event: Event | None = None
        self._closing_after_cancel = False

        self.canvas = DustMaskCanvas(
            image_rgb8=image_rgb8,
            auto_mask=auto_mask,
            add_mask=add_mask,
            protect_mask=protect_mask,
        )
        self.generate_button = QPushButton("Generate Auto Mask")
        self.cancel_auto_button = QPushButton("Cancel")
        self.cancel_auto_button.setEnabled(False)
        self.show_auto_checkbox = QCheckBox("Show auto")
        self.show_auto_checkbox.setChecked(True)
        self.sync_params_checkbox = QCheckBox("Sync sidebar params")
        self.sync_params_checkbox.setChecked(True)
        self.params_label = QLabel(self._params_summary(self._params))
        self.params_label.setObjectName("mutedLabel")
        self.clear_button = QPushButton("Clear Manual")
        self.reset_view_button = QPushButton("Reset View")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("primaryButton")
        self.close_button = QPushButton("Close")
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("mutedLabel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(2, 120)
        self.brush_size_slider.setValue(16)
        self.brush_size_label = QLabel("16")
        self.brush_size_label.setObjectName("sliderValue")

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.add_button = self._mode_button("Add", "add")
        self.protect_button = self._mode_button("Protect", "protect")
        self.erase_button = self._mode_button("Erase", "erase")
        self.add_button.setChecked(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.generate_button)
        toolbar.addWidget(self.cancel_auto_button)
        toolbar.addWidget(self.show_auto_checkbox)
        toolbar.addWidget(self.sync_params_checkbox)
        toolbar.addWidget(self.params_label)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.add_button)
        toolbar.addWidget(self.protect_button)
        toolbar.addWidget(self.erase_button)
        toolbar.addWidget(QLabel("Brush"))
        toolbar.addWidget(self.brush_size_slider)
        toolbar.addWidget(self.brush_size_label)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.clear_button)
        toolbar.addWidget(self.reset_view_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.apply_button)
        toolbar.addWidget(self.close_button)
        root.addLayout(toolbar)
        root.addWidget(self.canvas, 1)
        footer = QHBoxLayout()
        footer.addWidget(self.progress, 0)
        footer.addWidget(self.status_label, 1)
        root.addLayout(footer)

        self.generate_button.clicked.connect(self.generate_auto_mask)
        self.cancel_auto_button.clicked.connect(self.cancel_auto_mask)
        self.show_auto_checkbox.toggled.connect(self.canvas.set_show_auto)
        self.sync_params_checkbox.toggled.connect(self.refresh_synced_params_status)
        self.clear_button.clicked.connect(self.canvas.clear_manual_masks)
        self.reset_view_button.clicked.connect(self.canvas.reset_view)
        self.apply_button.clicked.connect(self.accept)
        self.close_button.clicked.connect(self.reject)
        self.brush_size_slider.valueChanged.connect(self._brush_size_changed)
        for button in (self.add_button, self.protect_button, self.erase_button):
            button.clicked.connect(self._mode_changed)
        self.canvas.statusChanged.connect(self.status_label.setText)
        self._install_editor_key_filters()
        self.canvas.setFocus()
        self._apply_style()

    @property
    def add_mask(self) -> np.ndarray:
        return self.canvas.add_mask

    @property
    def protect_mask(self) -> np.ndarray:
        return self.canvas.protect_mask

    @property
    def auto_mask(self) -> np.ndarray | None:
        return self.canvas.auto_mask

    @property
    def auto_mask_params_key(self) -> str | None:
        return self._auto_mask_params_key

    def refresh_synced_params_status(self, *_args) -> None:
        self._sync_params_if_requested()
        if self.canvas.auto_mask is None:
            return
        if self._auto_mask_params_key != dust_auto_mask_params_key(self._params):
            self.status_label.setText("Auto mask params changed; regenerate to sync")

    def generate_auto_mask(self) -> None:
        if self._auto_task is not None:
            self.status_label.setText("Auto mask already running")
            return
        self._sync_params_if_requested()
        self._auto_job_id += 1
        self._auto_cancel_event = Event()
        self._pending_auto_mask_params_key = dust_auto_mask_params_key(self._params)
        task = DustMaskPreviewTask(
            job_id=self._auto_job_id,
            path=self._source_path,
            linear_rgb=self._linear_rgb,
            params=self._params,
            cancel_event=self._auto_cancel_event,
        )
        task.signals.progress.connect(self._auto_progress)
        task.signals.finished.connect(self._auto_finished)
        task.signals.failed.connect(self._auto_failed)
        self._auto_task = task
        self._set_auto_running(True)
        self.progress.setValue(0)
        self.progress.show()
        self.status_label.setText("Auto mask running...")
        self._thread_pool.start(task)

    def cancel_auto_mask(self) -> None:
        if self._auto_cancel_event is not None:
            self._auto_cancel_event.set()
            self.status_label.setText("Cancelling auto mask...")
            self.cancel_auto_button.setEnabled(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if self._handle_editor_key_press(event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if self._handle_editor_key_release(event):
            return
        super().keyReleaseEvent(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if isinstance(event, QKeyEvent) and event.type() == QEvent.KeyPress:
            return self._handle_editor_key_press(event)
        if isinstance(event, QKeyEvent) and event.type() == QEvent.KeyRelease:
            return self._handle_editor_key_release(event)
        return super().eventFilter(watched, event)

    def _install_editor_key_filters(self) -> None:
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

    def _handle_editor_key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Space:
            if not event.isAutoRepeat():
                self.canvas.set_space_pan_active(True)
            event.accept()
            return True
        if event.key() == Qt.Key_A:
            self.add_button.setChecked(True)
            self._mode_changed()
            event.accept()
            return True
        if event.key() == Qt.Key_P:
            self.protect_button.setChecked(True)
            self._mode_changed()
            event.accept()
            return True
        if event.key() == Qt.Key_E:
            self.erase_button.setChecked(True)
            self._mode_changed()
            event.accept()
            return True
        if event.key() in (Qt.Key_BracketLeft, Qt.Key_Minus):
            self.brush_size_slider.setValue(max(2, self.brush_size_slider.value() - 2))
            event.accept()
            return True
        if event.key() in (Qt.Key_BracketRight, Qt.Key_Equal, Qt.Key_Plus):
            self.brush_size_slider.setValue(min(120, self.brush_size_slider.value() + 2))
            event.accept()
            return True
        return False

    def _handle_editor_key_release(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Space:
            if not event.isAutoRepeat():
                self.canvas.set_space_pan_active(False)
            event.accept()
            return True
        return False

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._auto_task is not None:
            self._closing_after_cancel = True
            self.cancel_auto_mask()
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        if self._auto_task is not None:
            self._closing_after_cancel = True
            self.cancel_auto_mask()
            return
        super().reject()

    def accept(self) -> None:
        if self._auto_task is not None:
            QMessageBox.information(self, "Dust Mask Editor", "Wait for auto mask to finish or cancel it before applying.")
            return
        super().accept()

    def _mode_button(self, text: str, mode: str) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setCheckable(True)
        button.setProperty("mode", mode)
        button.setMinimumHeight(32)
        self.mode_group.addButton(button)
        return button

    def _mode_changed(self, *_args) -> None:
        button = self.mode_group.checkedButton()
        mode = str(button.property("mode") if button is not None else "add")
        self.canvas.set_brush_mode(mode)
        self.status_label.setText(f"Mode: {mode}")

    def _brush_size_changed(self, value: int) -> None:
        self.brush_size_label.setText(str(value))
        self.canvas.set_brush_radius(value)

    def _auto_progress(self, job_id: int, value: int, text: str) -> None:
        if job_id != self._auto_job_id:
            return
        self.progress.setValue(max(0, min(100, int(value))))
        self.status_label.setText(text)

    def _auto_finished(self, job_id: int, path: Path | None, mask, stats) -> None:
        del path, stats
        if job_id != self._auto_job_id:
            return
        self.canvas.set_auto_mask(mask)
        self._auto_mask_params_key = self._pending_auto_mask_params_key
        self._pending_auto_mask_params_key = None
        self._auto_task = None
        self._auto_cancel_event = None
        self._set_auto_running(False)
        self.progress.setValue(100)
        self.status_label.setText("Auto mask ready")
        if self._closing_after_cancel:
            super().reject()

    def _auto_failed(self, job_id: int, path: Path | None, message: str) -> None:
        del path
        if job_id != self._auto_job_id:
            return
        self._auto_task = None
        self._auto_cancel_event = None
        self._pending_auto_mask_params_key = None
        self._set_auto_running(False)
        self.status_label.setText(message)
        if self._closing_after_cancel:
            super().reject()
            return
        if "cancelled" not in message.lower():
            QMessageBox.warning(self, "Dust Mask", message)

    def _set_auto_running(self, running: bool) -> None:
        self.generate_button.setEnabled(not running)
        self.cancel_auto_button.setEnabled(running)
        self.apply_button.setEnabled(not running)
        self.close_button.setEnabled(not running)

    def _sync_params_if_requested(self) -> None:
        if self.sync_params_checkbox.isChecked() and self._params_provider is not None:
            self._params = deepcopy(self._params_provider())
        self.params_label.setText(self._params_summary(self._params))

    @staticmethod
    def _params_summary(params: DustRemovalParams) -> str:
        model = params.model_id or "default"
        return (
            f"{model} / T {params.threshold} / Guard {params.texture_penalty} / "
            f"Max {params.max_threshold}"
        )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog#dustMaskEditor {
                background: #202020;
                color: #E8E1D5;
            }
            QWidget#dustMaskCanvas {
                background: #161616;
                border: 1px solid #443B32;
                border-radius: 6px;
            }
            QLabel {
                color: #E8E1D5;
            }
            QLabel#mutedLabel {
                color: #A69680;
                font-size: 12px;
            }
            QLabel#sliderValue {
                color: #D8D0C2;
                min-width: 28px;
                qproperty-alignment: AlignRight;
            }
            QPushButton, QToolButton {
                background: #2A2520;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #F2EEE6;
                padding: 7px 10px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #342A1D;
            }
            QPushButton:disabled, QToolButton:disabled {
                color: #817666;
                background: #25221F;
                border-color: #3D352D;
            }
            QToolButton:checked, QPushButton#primaryButton {
                background: #663300;
                border-color: #FFB000;
                color: #F2EEE6;
            }
            QPushButton#primaryButton:hover {
                background: #7A3A00;
            }
            QCheckBox {
                color: #E8E1D5;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #4A4034;
                border-radius: 3px;
                background: #1A1A1A;
            }
            QCheckBox::indicator:checked {
                background: #FFB000;
                border-color: #FFB000;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #443B32;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: #D8D0C2;
            }
            QProgressBar {
                background: #1A1A1A;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #E8E1D5;
                height: 18px;
                min-width: 220px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #FFB000;
                border-radius: 4px;
            }
            """
        )


def _apply_overlay_color(
    rgba: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int, int],
) -> None:
    if mask is None:
        return
    clean = np.asarray(mask).astype(bool)
    if clean.shape != rgba.shape[:2] or not np.any(clean):
        return
    rgba[clean, 0] = color[0]
    rgba[clean, 1] = color[1]
    rgba[clean, 2] = color[2]
    rgba[clean, 3] = color[3]
