from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImageReader, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QMenu, QWidget

from qnegative.core.models import ImagePoint, ImageRect, ImageSize, ToolMode


class ImageView(QWidget):
    maskPointSelected = Signal(object)
    whiteBalancePointSelected = Signal(object)
    filmRectSelected = Signal(object)
    filmRectReset = Signal()
    flipHorizontalRequested = Signal()
    flipVerticalRequested = Signal()
    rotateClockwiseRequested = Signal()
    viewStatusChanged = Signal(str)
    pickerCancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(720, 520)

        self._pixmap: QPixmap | None = None
        self._source_path: Path | None = None
        self._source_size: ImageSize | None = None
        self._display_rect = QRectF()
        self._tool_mode = ToolMode.PAN
        self._placeholder = "Open a RAW or image file to begin"
        self._transform_context_enabled = False

        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._is_panning = False
        self._pan_start: QPoint | None = None
        self._pan_offset = QPointF(0.0, 0.0)
        self._pan_offset_start = QPointF(0.0, 0.0)
        self._zoom_factor = 1.0
        self._film_edit_op: str | None = None
        self._film_edit_start: ImagePoint | None = None
        self._film_edit_rect: ImageRect | None = None

        self._mask_point: ImagePoint | None = None
        self._wb_point: ImagePoint | None = None
        self._film_rect: ImageRect | None = None

    def set_transform_context_enabled(self, enabled: bool) -> None:
        self._transform_context_enabled = enabled

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool_mode = mode
        self.setCursor(self._cursor_for_mode(mode))
        self.viewStatusChanged.emit(f"Tool: {self._tool_label(mode)}")

    def load_image(self, path: str | Path) -> bool:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            self.set_placeholder(f"Cannot preview: {Path(path).name}")
            return False

        self._pixmap = QPixmap.fromImage(image)
        self._source_path = Path(path)
        self._source_size = ImageSize(width=self._pixmap.width(), height=self._pixmap.height())
        self._placeholder = ""
        self._reset_navigation()
        self.clear_selections()
        self.update()
        self.viewStatusChanged.emit(
            f"{self._pixmap.width()} x {self._pixmap.height()} px"
        )
        return True

    def set_preview_pixmap(
        self,
        pixmap: QPixmap,
        *,
        source_path: str | Path,
        source_size: ImageSize,
        reset_navigation: bool = True,
    ) -> None:
        self._pixmap = pixmap
        self._source_path = Path(source_path)
        self._source_size = source_size
        self._placeholder = ""
        if reset_navigation:
            self._reset_navigation()
        self.clear_selections()
        self.update()
        self.viewStatusChanged.emit(
            f"Preview {pixmap.width()} x {pixmap.height()} px, source {source_size.label()}"
        )

    def set_raw_placeholder(self, path: str | Path) -> None:
        self._pixmap = None
        self._source_path = Path(path)
        self._source_size = None
        self._placeholder = f"RAW preview pending: {self._source_path.name}"
        self._reset_navigation()
        self.clear_selections()
        self.update()
        self.viewStatusChanged.emit("RAW file selected. Decode preview will be available later.")

    def set_placeholder(self, text: str) -> None:
        self._pixmap = None
        self._source_size = None
        self._placeholder = text
        self._reset_navigation()
        self.clear_selections()
        self.update()
        self.viewStatusChanged.emit(text)

    def clear_selections(self) -> None:
        self._drag_start = None
        self._drag_current = None
        self._is_panning = False
        self._pan_start = None
        self._film_edit_op = None
        self._film_edit_start = None
        self._film_edit_rect = None
        self._mask_point = None
        self._wb_point = None
        self._film_rect = None
        self.update()

    def restore_selections(
        self,
        *,
        mask_point: ImagePoint | None,
        film_rect: ImageRect | None,
        white_balance_point: ImagePoint | None = None,
    ) -> None:
        self._drag_start = None
        self._drag_current = None
        self._film_edit_op = None
        self._film_edit_start = None
        self._film_edit_rect = None
        self._mask_point = mask_point
        self._wb_point = white_balance_point
        self._film_rect = film_rect
        self.update()

    def reset_film_rect(self) -> None:
        self._film_rect = None
        self._film_edit_op = None
        self._film_edit_start = None
        self._film_edit_rect = None
        self.filmRectReset.emit()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1A1A1A"))

        if self._pixmap is None:
            self._paint_placeholder(painter)
            return

        self._display_rect = self._scaled_display_rect(self._pixmap)
        painter.fillRect(self._display_rect, QColor("#101010"))
        painter.drawPixmap(self._display_rect.toRect(), self._pixmap)

        self._paint_saved_selections(painter)
        self._paint_drag_selection(painter)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return

        if event.button() == Qt.RightButton:
            if self._tool_mode == ToolMode.WB_PICKER:
                self.pickerCancelled.emit()
                event.accept()
                return
            self._show_context_menu(event.position().toPoint())
            return

        if event.button() != Qt.LeftButton:
            return

        if not self._display_rect.contains(event.position()):
            return

        if self._tool_mode == ToolMode.MASK_PICKER:
            point = self._view_to_image_point(event.position())
            if point is not None:
                self._mask_point = point
                self.maskPointSelected.emit(point)
                self.update()
            return

        if self._tool_mode == ToolMode.WB_PICKER:
            point = self._view_to_image_point(event.position())
            if point is not None:
                self._wb_point = point
                self.whiteBalancePointSelected.emit(point)
                self.update()
            return

        if self._tool_mode == ToolMode.FILM_RECT and self._film_rect is not None:
            point = self._view_to_image_point(event.position())
            if point is not None:
                op = self._film_hit_test(point)
                if op is not None:
                    self._film_edit_op = op
                    self._film_edit_start = point
                    self._film_edit_rect = self._film_rect
                    self.viewStatusChanged.emit(self._film_op_status(op))
                    return

        if self._tool_mode == ToolMode.FILM_RECT:
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            self.update()
            return

        if self._tool_mode == ToolMode.PAN or self._transform_context_enabled:
            self._is_panning = True
            self._pan_start = event.position().toPoint()
            self._pan_offset_start = QPointF(self._pan_offset)
            self.setCursor(Qt.ClosedHandCursor)
            return

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return

        if self._is_panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_offset = self._pan_offset_start + QPointF(delta)
            self.update()
            return

        if self._film_edit_op is not None:
            point = self._view_to_image_point(event.position())
            if point is not None:
                self._update_film_edit(point)
            return

        if self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self.update()
            return

        if self._display_rect.contains(event.position()):
            point = self._view_to_image_point(event.position())
            if point is not None:
                self._update_hover_cursor(point)
                self.viewStatusChanged.emit(f"x={point.x}, y={point.y}")
        else:
            self.setCursor(self._cursor_for_mode(self._tool_mode))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self._pixmap is None:
            return

        if self._film_edit_op is not None:
            self._film_edit_op = None
            self._film_edit_start = None
            self._film_edit_rect = None
            self.update()
            return

        if self._is_panning:
            self._is_panning = False
            self._pan_start = None
            self.setCursor(self._cursor_for_mode(self._tool_mode))
            return

        if self._drag_start is None:
            return

        self._drag_current = event.position().toPoint()
        image_rect = self._current_drag_image_rect()
        self._drag_start = None
        self._drag_current = None

        if image_rect is None or not image_rect.is_valid():
            self.update()
            return

        if self._tool_mode == ToolMode.FILM_RECT:
            self._film_rect = image_rect
            self.filmRectSelected.emit(image_rect)

        self.update()

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        if self._tool_mode == ToolMode.WB_PICKER:
            self.pickerCancelled.emit()
            event.accept()
            return
        self._show_context_menu(event.pos())

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return

        old_rect = QRectF(self._display_rect)
        if old_rect.isEmpty():
            old_rect = self._scaled_display_rect(self._pixmap)
        cursor = event.position()
        if not old_rect.contains(cursor):
            cursor = old_rect.center()

        x_norm = (cursor.x() - old_rect.left()) / max(1.0, old_rect.width())
        y_norm = (cursor.y() - old_rect.top()) / max(1.0, old_rect.height())
        zoom_step = 1.12 if delta > 0 else 1.0 / 1.12
        self._zoom_factor = float(max(0.25, min(8.0, self._zoom_factor * zoom_step)))

        new_rect = self._scaled_display_rect(self._pixmap)
        new_cursor = QPointF(
            new_rect.left() + x_norm * new_rect.width(),
            new_rect.top() + y_norm * new_rect.height(),
        )
        self._pan_offset += cursor - new_cursor
        self.update()
        event.accept()

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor("#8C8171"))
        font = QFont()
        font.setPointSize(16)
        font.setWeight(QFont.DemiBold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)

    def _paint_saved_selections(self, painter: QPainter) -> None:
        if self._film_rect is not None:
            self._paint_image_rect(painter, self._film_rect, QColor("#FFB000"), "Frame")
        if self._mask_point is not None:
            self._paint_image_point(painter, self._mask_point, QColor("#05070a"))
        if self._wb_point is not None:
            self._paint_image_point(painter, self._wb_point, QColor("#FFB000"))

    def _paint_drag_selection(self, painter: QPainter) -> None:
        if self._drag_start is None or self._drag_current is None:
            return

        color = QColor("#FFB000")

        rect = QRect(self._drag_start, self._drag_current).normalized()
        clipped = rect.intersected(self._display_rect.toRect())
        pen = QPen(color, 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(clipped)

    def _paint_image_rect(
        self,
        painter: QPainter,
        rect: ImageRect,
        color: QColor,
        label: str,
    ) -> None:
        polygon = self._image_rect_to_polygon(rect)
        if polygon is None:
            return
        view_rect = polygon.boundingRect().toRect()
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(polygon)
        if label == "Frame":
            self._paint_film_handles(painter, polygon, color)
        painter.fillRect(
            QRect(view_rect.left(), view_rect.top() - 22, 74, 20),
            QColor(color.red(), color.green(), color.blue(), 170),
        )
        painter.setPen(QColor("#121212"))
        painter.drawText(QRect(view_rect.left() + 6, view_rect.top() - 21, 68, 18), label)

    def _paint_image_point(self, painter: QPainter, point: ImagePoint, color: QColor) -> None:
        view_point = self._image_point_to_view(point)
        if view_point is None:
            return
        painter.setPen(QPen(color, 2))
        painter.setBrush(QColor("#F2EEE6"))
        painter.drawLine(view_point.x() - 8, view_point.y(), view_point.x() + 8, view_point.y())
        painter.drawLine(view_point.x(), view_point.y() - 8, view_point.x(), view_point.y() + 8)
        painter.drawEllipse(view_point, 5, 5)

    def _scaled_display_rect(self, pixmap: QPixmap) -> QRectF:
        margin = 24
        available = self.rect().adjusted(margin, margin, -margin, -margin)
        if available.width() <= 0 or available.height() <= 0:
            return QRectF()

        image_ratio = pixmap.width() / pixmap.height()
        available_ratio = available.width() / available.height()

        if image_ratio > available_ratio:
            width = available.width()
            height = width / image_ratio
        else:
            height = available.height()
            width = height * image_ratio

        width *= self._zoom_factor
        height *= self._zoom_factor
        x = available.left() + (available.width() - width) / 2 + self._pan_offset.x()
        y = available.top() + (available.height() - height) / 2 + self._pan_offset.y()
        return QRectF(x, y, width, height)

    def _view_to_image_point(self, point) -> ImagePoint | None:
        if self._pixmap is None or self._source_size is None or self._display_rect.isEmpty():
            return None
        if not self._display_rect.contains(point):
            return None

        x_norm = (point.x() - self._display_rect.left()) / self._display_rect.width()
        y_norm = (point.y() - self._display_rect.top()) / self._display_rect.height()
        x = round(x_norm * (self._source_size.width - 1))
        y = round(y_norm * (self._source_size.height - 1))
        return ImagePoint(max(0, x), max(0, y))

    def _image_point_to_view(self, point: ImagePoint) -> QPoint | None:
        if self._pixmap is None or self._source_size is None or self._display_rect.isEmpty():
            return None
        x = self._display_rect.left() + point.x / max(1, self._source_size.width - 1) * self._display_rect.width()
        y = self._display_rect.top() + point.y / max(1, self._source_size.height - 1) * self._display_rect.height()
        return QPoint(round(x), round(y))

    def _current_drag_image_rect(self) -> ImageRect | None:
        if self._drag_start is None or self._drag_current is None:
            return None

        rect = QRect(self._drag_start, self._drag_current).normalized()
        clipped = rect.intersected(self._display_rect.toRect())
        if clipped.width() < 2 or clipped.height() < 2:
            return None

        top_left = self._view_to_image_point(QPoint(clipped.left(), clipped.top()))
        bottom_right = self._view_to_image_point(QPoint(clipped.right(), clipped.bottom()))
        if top_left is None or bottom_right is None:
            return None

        x = min(top_left.x, bottom_right.x)
        y = min(top_left.y, bottom_right.y)
        width = abs(bottom_right.x - top_left.x)
        height = abs(bottom_right.y - top_left.y)
        return ImageRect(x, y, width, height)

    def _image_rect_to_view(self, rect: ImageRect) -> QRect | None:
        top_left = self._image_point_to_view(ImagePoint(rect.x, rect.y))
        bottom_right = self._image_point_to_view(ImagePoint(rect.right, rect.bottom))
        if top_left is None or bottom_right is None:
            return None
        return QRect(top_left, bottom_right).normalized()

    def _image_rect_to_polygon(self, rect: ImageRect) -> QPolygonF | None:
        points = [self._source_to_view_float(x, y) for x, y in self._rect_corners(rect)]
        if any(point is None for point in points):
            return None
        return QPolygonF(points)  # type: ignore[arg-type]

    def _paint_film_handles(self, painter: QPainter, polygon: QPolygonF, color: QColor) -> None:
        points = [polygon.at(index) for index in range(polygon.count())]
        if len(points) != 4:
            return

        painter.setBrush(QColor("#1A1A1A"))
        painter.setPen(QPen(color, 2))

        for point in points:
            painter.drawEllipse(point, 5, 5)

        mids = [
            (points[0] + points[1]) / 2,
            (points[1] + points[2]) / 2,
            (points[2] + points[3]) / 2,
            (points[3] + points[0]) / 2,
        ]
        for point in mids:
            painter.drawRect(QRectF(point.x() - 4, point.y() - 4, 8, 8))

    def _cursor_for_mode(self, mode: ToolMode) -> Qt.CursorShape:
        if mode == ToolMode.PAN:
            return Qt.OpenHandCursor
        if mode in (ToolMode.MASK_PICKER, ToolMode.WB_PICKER):
            return Qt.CrossCursor
        return Qt.CrossCursor

    def _tool_label(self, mode: ToolMode) -> str:
        labels = {
            ToolMode.PAN: "Preview",
            ToolMode.MASK_PICKER: "Base Picker",
            ToolMode.WB_PICKER: "WB Picker",
            ToolMode.FILM_RECT: "Frame Area",
        }
        return labels[mode]

    def _source_to_view_float(self, x: float, y: float) -> QPointF | None:
        if self._source_size is None or self._display_rect.isEmpty():
            return None
        view_x = self._display_rect.left() + x / max(1, self._source_size.width - 1) * self._display_rect.width()
        view_y = self._display_rect.top() + y / max(1, self._source_size.height - 1) * self._display_rect.height()
        return QPointF(view_x, view_y)

    def _rect_corners(self, rect: ImageRect) -> list[tuple[float, float]]:
        half_w = rect.width / 2
        half_h = rect.height / 2
        local_points = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        return [
            self._local_to_world(rect, local_x, local_y)
            for local_x, local_y in local_points
        ]

    def _film_hit_test(self, point: ImagePoint) -> str | None:
        if self._film_rect is None:
            return None

        local_x, local_y = self._world_to_local(self._film_rect, point.x, point.y)
        half_w = self._film_rect.width / 2
        half_h = self._film_rect.height / 2
        threshold = self._source_hit_threshold()

        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        for corner_x, corner_y in corners:
            if math.hypot(local_x - corner_x, local_y - corner_y) <= threshold * 1.6:
                return "rotate"

        if abs(local_x + half_w) <= threshold and -half_h - threshold <= local_y <= half_h + threshold:
            return "resize_left"
        if abs(local_x - half_w) <= threshold and -half_h - threshold <= local_y <= half_h + threshold:
            return "resize_right"
        if abs(local_y + half_h) <= threshold and -half_w - threshold <= local_x <= half_w + threshold:
            return "resize_top"
        if abs(local_y - half_h) <= threshold and -half_w - threshold <= local_x <= half_w + threshold:
            return "resize_bottom"
        if -half_w <= local_x <= half_w and -half_h <= local_y <= half_h:
            return "move"

        return None

    def _update_film_edit(self, current: ImagePoint) -> None:
        if self._film_edit_op is None or self._film_edit_start is None or self._film_edit_rect is None:
            return

        start = self._film_edit_start
        rect = self._film_edit_rect
        op = self._film_edit_op

        if op == "move":
            self._film_rect = self._clamp_rect(
                ImageRect(
                    x=round(rect.x + current.x - start.x),
                    y=round(rect.y + current.y - start.y),
                    width=rect.width,
                    height=rect.height,
                    angle=rect.angle,
                )
            )
        elif op == "rotate":
            start_angle = math.atan2(start.y - rect.center_y, start.x - rect.center_x)
            current_angle = math.atan2(current.y - rect.center_y, current.x - rect.center_x)
            degrees = rect.angle + math.degrees(current_angle - start_angle)
            self._film_rect = ImageRect(rect.x, rect.y, rect.width, rect.height, self._normalize_angle(degrees))
        elif op.startswith("resize_"):
            self._film_rect = self._resize_film_rect(rect, current, op)

        if self._film_rect is not None:
            self.filmRectSelected.emit(self._film_rect)
        self.update()

    def _resize_film_rect(self, rect: ImageRect, current: ImagePoint, op: str) -> ImageRect:
        local_x, local_y = self._world_to_local(rect, current.x, current.y)
        half_w = rect.width / 2
        half_h = rect.height / 2
        min_size = 16

        center_local_x = 0.0
        center_local_y = 0.0
        width = rect.width
        height = rect.height

        if op == "resize_left":
            fixed = half_w
            moving = min(local_x, fixed - min_size)
            width = max(min_size, fixed - moving)
            center_local_x = (moving + fixed) / 2
        elif op == "resize_right":
            fixed = -half_w
            moving = max(local_x, fixed + min_size)
            width = max(min_size, moving - fixed)
            center_local_x = (moving + fixed) / 2
        elif op == "resize_top":
            fixed = half_h
            moving = min(local_y, fixed - min_size)
            height = max(min_size, fixed - moving)
            center_local_y = (moving + fixed) / 2
        elif op == "resize_bottom":
            fixed = -half_h
            moving = max(local_y, fixed + min_size)
            height = max(min_size, moving - fixed)
            center_local_y = (moving + fixed) / 2

        center_x, center_y = self._local_to_world(rect, center_local_x, center_local_y)
        resized = ImageRect(
            x=round(center_x - width / 2),
            y=round(center_y - height / 2),
            width=round(width),
            height=round(height),
            angle=rect.angle,
        )
        return self._clamp_rect(resized)

    def _update_hover_cursor(self, point: ImagePoint) -> None:
        if self._tool_mode != ToolMode.FILM_RECT or self._film_rect is None:
            self.setCursor(self._cursor_for_mode(self._tool_mode))
            return

        op = self._film_hit_test(point)
        if op == "move":
            self.setCursor(Qt.SizeAllCursor)
        elif op in ("resize_left", "resize_right"):
            self.setCursor(Qt.SizeHorCursor)
        elif op in ("resize_top", "resize_bottom"):
            self.setCursor(Qt.SizeVerCursor)
        elif op == "rotate":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(self._cursor_for_mode(self._tool_mode))

    def _show_context_menu(self, position: QPoint) -> None:
        if self._transform_context_enabled and self._pixmap is not None:
            menu = QMenu(self)
            flip_h_action = menu.addAction("Flip horizontal")
            flip_v_action = menu.addAction("Flip vertical")
            rotate_action = menu.addAction("Rotate 90 clockwise")
            selected = menu.exec(self.mapToGlobal(position))
            if selected == flip_h_action:
                self.flipHorizontalRequested.emit()
            elif selected == flip_v_action:
                self.flipVerticalRequested.emit()
            elif selected == rotate_action:
                self.rotateClockwiseRequested.emit()
            return

        if self._film_rect is None:
            return
        point = self._view_to_image_point(QPointF(position))
        if point is None or self._film_hit_test(point) is None:
            return

        menu = QMenu(self)
        reset_action = menu.addAction("Reset frame")
        selected = menu.exec(self.mapToGlobal(position))
        if selected == reset_action:
            self.reset_film_rect()

    def _world_to_local(self, rect: ImageRect, x: float, y: float) -> tuple[float, float]:
        radians = math.radians(rect.angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        dx = x - rect.center_x
        dy = y - rect.center_y
        return (
            cos_a * dx + sin_a * dy,
            -sin_a * dx + cos_a * dy,
        )

    def _local_to_world(self, rect: ImageRect, local_x: float, local_y: float) -> tuple[float, float]:
        radians = math.radians(rect.angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        return (
            rect.center_x + cos_a * local_x - sin_a * local_y,
            rect.center_y + sin_a * local_x + cos_a * local_y,
        )

    def _source_hit_threshold(self) -> float:
        if self._source_size is None or self._display_rect.isEmpty():
            return 12.0
        source_per_view = self._source_size.width / max(1.0, self._display_rect.width())
        return max(8.0, 10.0 * source_per_view)

    def _film_op_status(self, op: str) -> str:
        labels = {
            "move": "Move frame",
            "resize_left": "Resize left edge",
            "resize_right": "Resize right edge",
            "resize_top": "Resize top edge",
            "resize_bottom": "Resize bottom edge",
            "rotate": "Rotate frame",
        }
        return labels.get(op, "Edit frame")

    def _normalize_angle(self, angle: float) -> float:
        normalized = (angle + 180.0) % 360.0 - 180.0
        if abs(normalized) < 0.05:
            return 0.0
        return normalized

    def _clamp_rect(self, rect: ImageRect) -> ImageRect:
        if self._source_size is None:
            return rect
        x = min(max(0, rect.x), max(0, self._source_size.width - rect.width))
        y = min(max(0, rect.y), max(0, self._source_size.height - rect.height))
        return ImageRect(x, y, rect.width, rect.height, rect.angle)

    def _reset_navigation(self) -> None:
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._pan_offset_start = QPointF(0.0, 0.0)
        self._is_panning = False
        self._pan_start = None
