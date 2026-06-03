from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class HistogramLevelsWidget(QWidget):
    levelsChanged = Signal(dict)
    interactionStarted = Signal()
    interactionFinished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("histogramLevels")
        self.setMinimumHeight(126)
        self.setMouseTracking(True)

        self._histogram = np.zeros(256, dtype=np.float32)
        self._black = 0
        self._mid = 50
        self._white = 100
        self._active_handle: str | None = None

    def set_histogram(self, histogram: np.ndarray | None) -> None:
        if histogram is None or histogram.size == 0:
            self._histogram = np.zeros(256, dtype=np.float32)
        else:
            values = np.asarray(histogram, dtype=np.float32)
            if values.size != 256:
                values = np.interp(
                    np.linspace(0, values.size - 1, 256),
                    np.arange(values.size),
                    values,
                ).astype(np.float32)
            self._histogram = values
        self.update()

    def set_levels(self, black: int, mid: int, white: int, *, emit: bool = False) -> None:
        self._black = max(0, min(98, black))
        self._white = max(self._black + 2, min(100, white))
        self._mid = max(self._black + 1, min(self._white - 1, mid))
        self.update()
        if emit:
            self._emit_levels()

    def levels(self) -> dict:
        return {
            "black_point": self._black,
            "mid_point": self._mid,
            "white_point": self._white,
        }

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1A1A1A"))

        plot = self._plot_rect()
        painter.fillRect(plot, QColor("#121212"))
        painter.setPen(QPen(QColor("#3D352D"), 1))
        painter.drawRect(plot)

        self._paint_histogram(painter, plot)
        self._paint_handle(painter, plot, self._black, QColor("#F2EEE6"), "B", above=False, inner=QColor("#0E0E0E"))
        self._paint_handle(painter, plot, self._mid, QColor("#d7ca55"), "N", above=True)
        self._paint_handle(painter, plot, self._white, QColor("#E8E1D5"), "W", above=False)

        painter.setPen(QColor("#A69680"))
        painter.drawText(QRectF(0, self.height() - 19, self.width(), 16), Qt.AlignCenter, self._label_text())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        self._active_handle = self._nearest_handle(event.position())
        if self._active_handle is not None:
            self.interactionStarted.emit()
        self._set_handle_from_position(event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._active_handle is None:
            self.setCursor(Qt.PointingHandCursor if self._nearest_handle(event.position()) else Qt.ArrowCursor)
            return
        self._set_handle_from_position(event.position())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            if self._active_handle is not None:
                self.interactionFinished.emit()
            self._active_handle = None

    def _paint_histogram(self, painter: QPainter, plot: QRectF) -> None:
        values = np.log1p(np.maximum(self._histogram, 0.0))
        maximum = float(values.max())
        if maximum <= 0.0:
            painter.setPen(QColor("#443B32"))
            painter.drawText(plot, Qt.AlignCenter, "Waiting for histogram")
            return

        normalized = values / maximum
        step = plot.width() / len(normalized)
        path = QPainterPath()
        path.moveTo(plot.left(), plot.bottom())
        for index, value in enumerate(normalized):
            x = plot.left() + index * step
            y = plot.bottom() - float(value) * plot.height()
            path.lineTo(x, y)
        path.lineTo(plot.right(), plot.bottom())
        path.closeSubpath()

        painter.fillPath(path, QColor("#6A5F51"))
        painter.setPen(QPen(QColor("#9A8B75"), 1))
        painter.drawPath(path)

    def _paint_handle(
        self,
        painter: QPainter,
        plot: QRectF,
        value: int,
        color: QColor,
        label: str,
        *,
        above: bool,
        inner: QColor | None = None,
    ) -> None:
        x = self._x_from_value(value)
        painter.setPen(QPen(color, 3))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        if inner is not None:
            painter.setPen(QPen(inner, 1))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        triangle_y = plot.top() - 1 if above else plot.bottom() + 1
        direction = -1 if above else 1
        points = [
            QPointF(x, triangle_y),
            QPointF(x - 6, triangle_y + direction * 9),
            QPointF(x + 6, triangle_y + direction * 9),
        ]
        painter.setBrush(color)
        painter.drawPolygon(points)
        if inner is not None:
            painter.setPen(QPen(inner, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(points)

        painter.setPen(QColor("#F2EEE6"))
        text_y = plot.top() - 22 if above else plot.bottom() + 11
        painter.drawText(QRectF(x - 18, text_y, 36, 16), Qt.AlignCenter, label)

    def _plot_rect(self) -> QRectF:
        return QRectF(12, 24, max(1, self.width() - 24), max(1, self.height() - 52))

    def _x_from_value(self, value: int) -> float:
        plot = self._plot_rect()
        return plot.left() + value / 100.0 * plot.width()

    def _value_from_x(self, x: float) -> int:
        plot = self._plot_rect()
        ratio = (x - plot.left()) / max(1.0, plot.width())
        return round(max(0.0, min(1.0, ratio)) * 100)

    def _nearest_handle(self, position) -> str | None:
        distances = {
            "black": abs(position.x() - self._x_from_value(self._black)),
            "mid": abs(position.x() - self._x_from_value(self._mid)),
            "white": abs(position.x() - self._x_from_value(self._white)),
        }
        handle, distance = min(distances.items(), key=lambda item: item[1])
        return handle if distance <= 14 else None

    def _set_handle_from_position(self, position) -> None:
        if self._active_handle is None:
            return
        value = self._value_from_x(position.x())

        if self._active_handle == "black":
            self._black = max(0, min(value, self._mid - 1))
        elif self._active_handle == "mid":
            self._mid = max(self._black + 1, min(value, self._white - 1))
        elif self._active_handle == "white":
            self._white = min(100, max(value, self._mid + 1))

        self.update()
        self._emit_levels()

    def _emit_levels(self) -> None:
        self.levelsChanged.emit(self.levels())

    def _label_text(self) -> str:
        return f"Black {self._black}  Mid {self._mid}  White {self._white}"
