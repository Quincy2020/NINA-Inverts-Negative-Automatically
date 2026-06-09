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
    rotateCounterClockwiseRequested = Signal()
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
        self._gpu_preview_enabled = True
        self._linear_image: QImage | None = None
        self._linear_texture_dirty = False
        self._gpu_highlights = 0
        self._gpu_shadows = 0
        self._gpu_saturation = 0

        self._image: QImage | None = None
        self._source_path: Path | None = None
        self._source_size: ImageSize | None = None
        self._display_rect = QRectF()
        self._placeholder = "Positive preview waiting"
        self._tool_mode = ToolMode.PAN
        self._transform_context_enabled = False
        self._wb_point: ImagePoint | None = None
        self._status_overlay_text = ""
        self._dust_auto_mask: np.ndarray | None = None
        self._dust_add_mask: np.ndarray | None = None
        self._dust_protect_mask: np.ndarray | None = None
        self._dust_overlay_image: QImage | None = None
        self._dust_show_auto_mask = True

        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._pan_offset_start = QPointF(0.0, 0.0)
        self._is_panning = False
        self._pan_start: QPoint | None = None

    def set_transform_context_enabled(self, enabled: bool) -> None:
        self._transform_context_enabled = enabled

    def set_gpu_preview_enabled(self, enabled: bool) -> None:
        self._gpu_preview_enabled = bool(enabled)
        self._texture_dirty = True
        self._linear_texture_dirty = True
        self.update()

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
        self._linear_image = None
        self._texture_dirty = True
        self._linear_texture_dirty = True
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

    def set_gpu_linear_preview(
        self,
        linear_rgb: np.ndarray,
        *,
        highlights: int,
        shadows: int,
        saturation: int,
    ) -> None:
        if linear_rgb.ndim != 3 or linear_rgb.shape[2] != 3:
            return
        clipped = np.clip(np.nan_to_num(linear_rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        rgb8 = np.ascontiguousarray((clipped * 255.0 + 0.5).astype(np.uint8))
        alpha = np.full((*rgb8.shape[:2], 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate([rgb8, alpha], axis=2))
        image = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.shape[1] * 4,
            QImage.Format_RGBA8888,
        ).copy()
        self._linear_image = image
        self._linear_texture_dirty = True
        self.set_gpu_display_adjustments(
            highlights=highlights,
            shadows=shadows,
            saturation=saturation,
        )

    def set_gpu_display_adjustments(self, *, highlights: int, shadows: int, saturation: int) -> None:
        self._gpu_highlights = int(highlights)
        self._gpu_shadows = int(shadows)
        self._gpu_saturation = int(saturation)
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._image = None
        self._linear_image = None
        self._texture_dirty = True
        self._linear_texture_dirty = True
        self._source_size = None
        self._placeholder = text
        self._wb_point = None
        self.clear_dust_overlay()
        self._reset_navigation()
        self.update()
        self.viewStatusChanged.emit(text)

    def set_status_overlay(self, text: str) -> None:
        self._status_overlay_text = text
        self.update()

    def set_dust_overlay(
        self,
        *,
        auto_mask: np.ndarray | None,
        add_mask: np.ndarray | None,
        protect_mask: np.ndarray | None,
    ) -> None:
        self._dust_auto_mask = _mask_copy(auto_mask)
        self._dust_add_mask = _mask_copy(add_mask)
        self._dust_protect_mask = _mask_copy(protect_mask)
        self._rebuild_dust_overlay_image()
        self.update()

    def set_dust_show_auto_mask(self, enabled: bool) -> None:
        self._dust_show_auto_mask = bool(enabled)
        self._rebuild_dust_overlay_image()
        self.update()

    def clear_dust_overlay(self) -> None:
        self._dust_auto_mask = None
        self._dust_add_mask = None
        self._dust_protect_mask = None
        self._dust_overlay_image = None
        self.update()

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

        if self._image is not None and self._gpu_preview_enabled:
            try:
                self._ensure_texture()
                self._draw_texture()
            except Exception as exc:
                self._gpu_preview_enabled = False
                self.viewStatusChanged.emit(f"GPU preview disabled: {exc}")

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._image is None:
            self._paint_placeholder(painter)
        else:
            if not self._gpu_preview_enabled:
                self._paint_cpu_image(painter)
            self._paint_dust_overlay(painter)
            self._paint_saved_selections(painter)
            self._paint_status_overlay(painter)
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
            uniform bool texture_is_linear;
            uniform float highlights;
            uniform float shadows;
            uniform float saturation;
            varying vec2 v_texcoord;

            float luminance(vec3 rgb) {
                return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
            }

            vec3 applyHighlightShadow(vec3 rgb) {
                rgb = max(rgb, vec3(0.0));
                float luma = max(luminance(rgb), 0.0);
                float target = luma;
                if (abs(shadows) > 0.0001) {
                    float shadowAmount = clamp(shadows / 100.0, -1.0, 1.0);
                    float pivot = 0.44;
                    float normValue = clamp(luma / pivot, 0.0, 1.0);
                    float weight = 1.0 - smoothstep(0.18, 1.0, normValue);
                    if (shadowAmount > 0.0) {
                        float gammaValue = 1.0 / (1.0 + 2.6 * shadowAmount);
                        float lifted = pivot * pow(max(normValue, 0.0), gammaValue);
                        float blackAnchor = smoothstep(0.0, 0.035, luma);
                        float mixValue = clamp(shadowAmount * 0.95 * weight * blackAnchor, 0.0, 1.0);
                        target = mix(target, lifted, mixValue);
                    } else {
                        float amount = abs(shadowAmount);
                        float gammaValue = 1.0 + 2.2 * amount;
                        float crushed = pivot * pow(max(normValue, 0.0), gammaValue);
                        float mixValue = clamp(amount * 0.90 * weight, 0.0, 1.0);
                        target = mix(target, crushed, mixValue);
                    }
                }
                if (abs(highlights) > 0.0001) {
                    float highlightAmount = clamp(highlights / 65.0, -1.0, 1.0);
                    if (highlightAmount > 0.0) {
                        float weight = smoothstep(0.36, 0.92, luma);
                        float boosted = target + (1.0 - clamp(target, 0.0, 1.0)) * 0.75;
                        target = target + (boosted - target) * highlightAmount * weight;
                    } else {
                        float amount = abs(highlightAmount);
                        float pivot = 0.46;
                        float span = 0.54;
                        float weight = smoothstep(0.34, 1.0, luma);
                        float normValue = max(target - pivot, 0.0) / span;
                        float compression = 1.0 + amount * 1.25 * normValue;
                        float compressed = pivot + span * (normValue / compression);
                        compressed = min(compressed, target);
                        target = mix(target, compressed, weight * amount);
                    }
                }
                float ratio = target / max(luma, 0.00001);
                return rgb * ratio;
            }

            vec3 applySaturation(vec3 rgb) {
                float amount = clamp(saturation / 100.0, -1.0, 1.0);
                float factor = amount < 0.0 ? 1.0 + amount : 1.0 + amount * 1.35;
                float luma = luminance(clamp(rgb, 0.0, 1.0));
                return clamp(vec3(luma) + (rgb - vec3(luma)) * factor, 0.0, 1.0);
            }

            void main() {
                vec4 texel = texture2D(image_texture, v_texcoord);
                if (texture_is_linear) {
                    vec3 rgb = applyHighlightShadow(texel.rgb);
                    rgb = applySaturation(rgb);
                    rgb = pow(clamp(rgb, 0.0, 1.0), vec3(1.0 / 2.2));
                    gl_FragColor = vec4(rgb, texel.a);
                } else {
                    gl_FragColor = texel;
                }
            }
            """,
        )
        if not program.link():
            raise RuntimeError(f"OpenGL preview shader link failed: {program.log()}")
        return program

    def _ensure_texture(self) -> None:
        source = self._active_texture_image()
        dirty = self._active_texture_dirty()
        if source is None or not dirty:
            return
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None

        self._texture = QOpenGLTexture(source)
        self._texture.setMinificationFilter(QOpenGLTexture.Linear)
        self._texture.setMagnificationFilter(QOpenGLTexture.Linear)
        self._texture.setWrapMode(QOpenGLTexture.ClampToEdge)
        if self._using_linear_texture():
            self._linear_texture_dirty = False
        else:
            self._texture_dirty = False

    def _using_linear_texture(self) -> bool:
        return self._gpu_preview_enabled and self._linear_image is not None

    def _active_texture_image(self) -> QImage | None:
        if self._using_linear_texture():
            return self._linear_image
        return self._image

    def _active_texture_dirty(self) -> bool:
        if self._using_linear_texture():
            return self._linear_texture_dirty
        return self._texture_dirty

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
        self._set_uniform_int("image_texture", 0)
        self._set_uniform_int("texture_is_linear", 1 if self._using_linear_texture() else 0)
        self._set_uniform_float("highlights", float(self._gpu_highlights))
        self._set_uniform_float("shadows", float(self._gpu_shadows))
        self._set_uniform_float("saturation", float(self._gpu_saturation))

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

    def _set_uniform_int(self, name: str, value: int) -> None:
        if self._program is None:
            return
        location = self._program.uniformLocation(name)
        if location >= 0:
            self._program.setUniformValue(location, int(value))

    def _set_uniform_float(self, name: str, value: float) -> None:
        if self._program is None:
            return
        location = self._program.uniformLocation(name)
        if location >= 0:
            self._program.setUniformValue(location, float(value))

    def _paint_cpu_image(self, painter: QPainter) -> None:
        if self._image is None:
            return
        self._display_rect = self._scaled_display_rect()
        if self._display_rect.isEmpty():
            return
        painter.fillRect(self._display_rect, QColor("#101010"))
        painter.drawImage(self._display_rect.toRect(), self._image)

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor("#8C8171"))
        font = QFont()
        font.setPointSize(16)
        font.setWeight(QFont.DemiBold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)

    def _paint_saved_selections(self, painter: QPainter) -> None:
        if self._wb_point is not None:
            self._paint_image_point(painter, self._wb_point, QColor("#FFB000"))

    def _paint_dust_overlay(self, painter: QPainter) -> None:
        if self._dust_overlay_image is None:
            return
        if self._display_rect.isEmpty():
            self._display_rect = self._scaled_display_rect()
        if self._display_rect.isEmpty():
            return
        painter.drawImage(self._display_rect.toRect(), self._dust_overlay_image)

    def _paint_status_overlay(self, painter: QPainter) -> None:
        if not self._status_overlay_text:
            return

        font = QFont()
        font.setPointSize(10)
        font.setWeight(QFont.Medium)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        lines = self._status_overlay_text.splitlines()
        width = max(metrics.horizontalAdvance(line) for line in lines) + 24
        height = metrics.lineSpacing() * len(lines) + 18
        rect = QRectF(18, 18, width, height)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(10, 12, 15, 178))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QColor("#F2EEE6"))
        text_rect = rect.adjusted(12, 9, -12, -9)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, self._status_overlay_text)

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
        rotate_ccw_action = menu.addAction("Rotate 90 counterclockwise")
        selected = menu.exec(self.mapToGlobal(position))
        if selected == flip_h_action:
            self.flipHorizontalRequested.emit()
        elif selected == flip_v_action:
            self.flipVerticalRequested.emit()
        elif selected == rotate_action:
            self.rotateClockwiseRequested.emit()
        elif selected == rotate_ccw_action:
            self.rotateCounterClockwiseRequested.emit()

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

    def _rebuild_dust_overlay_image(self) -> None:
        height, width = self._dust_overlay_shape()
        if height <= 0 or width <= 0:
            self._dust_overlay_image = None
            return
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        if self._dust_show_auto_mask:
            _apply_overlay_color(rgba, self._dust_auto_mask, (255, 176, 0, 105))
        _apply_overlay_color(rgba, self._dust_add_mask, (40, 220, 255, 135))
        _apply_overlay_color(rgba, self._dust_protect_mask, (255, 70, 170, 145))
        if not np.any(rgba[..., 3]):
            self._dust_overlay_image = None
            return
        self._dust_overlay_image = QImage(
            rgba.data,
            width,
            height,
            width * 4,
            QImage.Format_RGBA8888,
        ).copy()

    def _dust_overlay_shape(self) -> tuple[int, int]:
        if self._source_size is not None:
            return self._source_size.height, self._source_size.width
        for mask in (self._dust_auto_mask, self._dust_add_mask, self._dust_protect_mask):
            if mask is not None:
                return mask.shape[:2]
        return 0, 0


def _mask_copy(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    return np.asarray(mask).astype(bool, copy=True)


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
