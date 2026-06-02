from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QOpenGLFunctions, QPainter, QPixmap
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QMenu

from qnegative.core.models import ImagePoint, ImageRect, ImageSize, ToolMode


GL_COLOR_BUFFER_BIT = 0x00004000
GL_FLOAT = 0x1406
GL_TRIANGLE_STRIP = 0x0005


class OpenGLPreviewView(QOpenGLWidget):
    whiteBalancePointSelected = Signal(object)
    flipHorizontalRequested = Signal()
    flipVerticalRequested = Signal()
    rotateClockwiseRequested = Signal()
    pickerCancelled = Signal()
    viewStatusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(720, 520)

        self._functions: QOpenGLFunctions | None = None
        self._program: QOpenGLShaderProgram | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._vbo: QOpenGLBuffer | None = None
        self._texture: QOpenGLTexture | None = None
        self._texture_dirty = False

        self._image: QImage | None = None
        self._source_path: Path | None = None
        self._source_size: ImageSize | None = None
        self._display_rect = QRectF()
        self._placeholder = "Positive preview waiting"
        self._tool_mode = ToolMode.PAN
        self._transform_context_enabled = False
        self._wb_point: ImagePoint | None = None

        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._pan_offset_start = QPointF(0.0, 0.0)
        self._is_panning = False
        self._pan_start: QPoint | None = None

    def set_transform_context_enabled(self, enabled: bool) -> None:
        self._transform_context_enabled = enabled

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool_mode = mode
        self.setCursor(self._cursor_for_mode(mode))
        self.viewStatusChanged.emit(f"Tool: {self._tool_label(mode)}")

    def set_preview_pixmap(
        self,
        pixmap: QPixmap,
        *,
        source_path: str | Path,
        source_size: ImageSize,
        reset_navigation: bool = True,
    ) -> None:
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        self._image = image.copy()
        self._texture_dirty = True
        self._source_path = Path(source_path)
        self._source_size = source_size
        self._placeholder = ""
        if reset_navigation:
            self._reset_navigation()
        self._wb_point = None
        self.update()
        self.viewStatusChanged.emit(
            f"GPU preview {pixmap.width()} x {pixmap.height()} px, source {source_size.label()}"
        )

    def set_placeholder(self, text: str) -> None:
        self._image = None
        self._texture_dirty = True
        self._source_size = None
        self._placeholder = text
        self._wb_point = None
        self._reset_navigation()
        self.update()
        self.viewStatusChanged.emit(text)

    def restore_selections(
        self,
        *,
        mask_point: ImagePoint | None,
        film_rect: ImageRect | None,
        white_balance_point: ImagePoint | None = None,
    ) -> None:
        del mask_point, film_rect
        self._wb_point = white_balance_point
        self.update()

    def initializeGL(self) -> None:  # noqa: N802
        self._functions = QOpenGLFunctions()
        self._functions.initializeOpenGLFunctions()
        self._program = self._build_program()
        self._vao = QOpenGLVertexArrayObject()
        self._vao.create()
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbo.create()

    def resizeGL(self, width: int, height: int) -> None:  # noqa: N802
        if self._functions is not None:
            self._functions.glViewport(0, 0, width, height)

    def paintGL(self) -> None:  # noqa: N802
        functions = self._functions
        if functions is None:
            return

        functions.glClearColor(0.082, 0.094, 0.114, 1.0)
        functions.glClear(GL_COLOR_BUFFER_BIT)

        if self._image is not None:
            self._ensure_texture()
            self._draw_texture()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._image is None:
            self._paint_placeholder(painter)
        else:
            self._paint_saved_selections(painter)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
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

        if self._tool_mode == ToolMode.WB_PICKER:
            point = self._view_to_image_point(event.position())
            if point is not None:
                self._wb_point = point
                self.whiteBalancePointSelected.emit(point)
                self.update()
            return

        if self._tool_mode == ToolMode.PAN or self._transform_context_enabled:
            self._is_panning = True
            self._pan_start = event.position().toPoint()
            self._pan_offset_start = QPointF(self._pan_offset)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
            return

        if self._is_panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_offset = self._pan_offset_start + QPointF(delta)
            self.update()
            return

        if self._display_rect.contains(event.position()):
            point = self._view_to_image_point(event.position())
            if point is not None:
                self.viewStatusChanged.emit(f"x={point.x}, y={point.y}")
        self.setCursor(self._cursor_for_mode(self._tool_mode))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        if self._is_panning:
            self._is_panning = False
            self._pan_start = None
            self.setCursor(self._cursor_for_mode(self._tool_mode))

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        if self._tool_mode == ToolMode.WB_PICKER:
            self.pickerCancelled.emit()
            event.accept()
            return
        self._show_context_menu(event.pos())

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return

        old_rect = QRectF(self._display_rect)
        if old_rect.isEmpty():
            old_rect = self._scaled_display_rect()
        cursor = event.position()
        if not old_rect.contains(cursor):
            cursor = old_rect.center()

        x_norm = (cursor.x() - old_rect.left()) / max(1.0, old_rect.width())
        y_norm = (cursor.y() - old_rect.top()) / max(1.0, old_rect.height())
        zoom_step = 1.12 if delta > 0 else 1.0 / 1.12
        self._zoom_factor = float(max(0.25, min(8.0, self._zoom_factor * zoom_step)))

        new_rect = self._scaled_display_rect()
        new_cursor = QPointF(
            new_rect.left() + x_norm * new_rect.width(),
            new_rect.top() + y_norm * new_rect.height(),
        )
        self._pan_offset += cursor - new_cursor
        self.update()
        event.accept()

    def _build_program(self) -> QOpenGLShaderProgram:
        program = QOpenGLShaderProgram(self)
        program.addShaderFromSourceCode(
            QOpenGLShader.Vertex,
            """
            attribute vec2 position;
            attribute vec2 texcoord;
            varying vec2 v_texcoord;
            void main() {
                v_texcoord = texcoord;
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """,
        )
        program.addShaderFromSourceCode(
            QOpenGLShader.Fragment,
            """
            uniform sampler2D image_texture;
            varying vec2 v_texcoord;
            void main() {
                gl_FragColor = texture2D(image_texture, v_texcoord);
            }
            """,
        )
        if not program.link():
            raise RuntimeError(f"OpenGL preview shader link failed: {program.log()}")
        return program

    def _ensure_texture(self) -> None:
        if self._image is None or not self._texture_dirty:
            return
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None

        self._texture = QOpenGLTexture(self._image)
        self._texture.setMinificationFilter(QOpenGLTexture.Linear)
        self._texture.setMagnificationFilter(QOpenGLTexture.Linear)
        self._texture.setWrapMode(QOpenGLTexture.ClampToEdge)
        self._texture_dirty = False

    def _draw_texture(self) -> None:
        if (
            self._image is None
            or self._texture is None
            or self._program is None
            or self._vbo is None
            or self._functions is None
        ):
            return

        self._display_rect = self._scaled_display_rect()
        if self._display_rect.isEmpty():
            return

        left = self._display_rect.left()
        right = self._display_rect.right()
        top = self._display_rect.top()
        bottom = self._display_rect.bottom()
        width = max(1.0, float(self.width()))
        height = max(1.0, float(self.height()))
        left_ndc = left / width * 2.0 - 1.0
        right_ndc = right / width * 2.0 - 1.0
        top_ndc = 1.0 - top / height * 2.0
        bottom_ndc = 1.0 - bottom / height * 2.0

        vertices = np.array(
            [
                left_ndc, top_ndc, 0.0, 0.0,
                left_ndc, bottom_ndc, 0.0, 1.0,
                right_ndc, top_ndc, 1.0, 0.0,
                right_ndc, bottom_ndc, 1.0, 1.0,
            ],
            dtype=np.float32,
        )

        self._texture.bind(0)
        self._program.bind()
        self._program.setUniformValue("image_texture", 0)

        self._vbo.bind()
        self._vbo.allocate(vertices.tobytes(), vertices.nbytes)

        position = self._program.attributeLocation("position")
        texcoord = self._program.attributeLocation("texcoord")
        self._program.enableAttributeArray(position)
        self._program.enableAttributeArray(texcoord)
        self._program.setAttributeBuffer(position, GL_FLOAT, 0, 2, 16)
        self._program.setAttributeBuffer(texcoord, GL_FLOAT, 8, 2, 16)
        self._functions.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        self._program.disableAttributeArray(position)
        self._program.disableAttributeArray(texcoord)
        self._vbo.release()
        self._program.release()
        self._texture.release()

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor("#77808d"))
        font = QFont()
        font.setPointSize(16)
        font.setWeight(QFont.DemiBold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)

    def _paint_saved_selections(self, painter: QPainter) -> None:
        if self._wb_point is not None:
            self._paint_image_point(painter, self._wb_point, QColor("#58c7ff"))

    def _paint_image_point(self, painter: QPainter, point: ImagePoint, color: QColor) -> None:
        view_point = self._image_point_to_view(point)
        if view_point is None:
            return
        painter.setPen(color)
        painter.drawLine(view_point.x() - 8, view_point.y(), view_point.x() + 8, view_point.y())
        painter.drawLine(view_point.x(), view_point.y() - 8, view_point.x(), view_point.y() + 8)
        painter.drawEllipse(view_point, 5, 5)

    def _scaled_display_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        margin = 24
        available = self.rect().adjusted(margin, margin, -margin, -margin)
        if available.width() <= 0 or available.height() <= 0:
            return QRectF()

        image_ratio = self._image.width() / self._image.height()
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
        if self._image is None or self._source_size is None:
            return None
        if self._display_rect.isEmpty():
            self._display_rect = self._scaled_display_rect()
        if not self._display_rect.contains(point):
            return None

        x_norm = (point.x() - self._display_rect.left()) / self._display_rect.width()
        y_norm = (point.y() - self._display_rect.top()) / self._display_rect.height()
        x = round(x_norm * (self._source_size.width - 1))
        y = round(y_norm * (self._source_size.height - 1))
        return ImagePoint(max(0, x), max(0, y))

    def _image_point_to_view(self, point: ImagePoint) -> QPoint | None:
        if self._source_size is None:
            return None
        if self._display_rect.isEmpty():
            self._display_rect = self._scaled_display_rect()
        if self._display_rect.isEmpty():
            return None
        x = self._display_rect.left() + point.x / max(1, self._source_size.width - 1) * self._display_rect.width()
        y = self._display_rect.top() + point.y / max(1, self._source_size.height - 1) * self._display_rect.height()
        return QPoint(round(x), round(y))

    def _show_context_menu(self, position: QPoint) -> None:
        if not self._transform_context_enabled or self._image is None:
            return
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

    def _reset_navigation(self) -> None:
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._pan_offset_start = QPointF(0.0, 0.0)
        self._is_panning = False
        self._pan_start = None

    def _cursor_for_mode(self, mode: ToolMode) -> Qt.CursorShape:
        if mode == ToolMode.PAN:
            return Qt.OpenHandCursor
        if mode == ToolMode.WB_PICKER:
            return Qt.CrossCursor
        return Qt.OpenHandCursor

    def _tool_label(self, mode: ToolMode) -> str:
        labels = {
            ToolMode.PAN: "Preview",
            ToolMode.MASK_PICKER: "Base Picker",
            ToolMode.WB_PICKER: "WB Picker",
            ToolMode.FILM_RECT: "Frame Area",
        }
        return labels[mode]
