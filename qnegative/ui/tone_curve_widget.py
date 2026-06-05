from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from qnegative.core.models import AdjustmentParams
from qnegative.core.pipeline import highlight_shadow_control_points, highlight_shadow_tone_lut


class ToneCurveWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toneCurveWidget")
        self.setMinimumHeight(104)
        self._adjustments = AdjustmentParams()
        self._mid_anchor = 0.46

    def set_tone(self, *, adjustments: AdjustmentParams, mid_anchor: float | None = None) -> None:
        self._adjustments = adjustments
        if mid_anchor is not None:
            self._mid_anchor = float(np.clip(mid_anchor, 0.0, 1.0))
        self.update()

    def set_mid_anchor(self, mid_anchor: float) -> None:
        self._mid_anchor = float(np.clip(mid_anchor, 0.0, 1.0))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1A1A1A"))

        plot = QRectF(12, 10, max(1, self.width() - 24), max(1, self.height() - 22))
        painter.fillRect(plot, QColor("#121212"))
        painter.setPen(QPen(QColor("#3D352D"), 1))
        painter.drawRect(plot)

        painter.setPen(QPen(QColor("#2A2520"), 1))
        for index in range(1, 4):
            x = plot.left() + plot.width() * index / 4.0
            y = plot.top() + plot.height() * index / 4.0
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        diagonal = QPainterPath()
        diagonal.moveTo(plot.left(), plot.bottom())
        diagonal.lineTo(plot.right(), plot.top())
        painter.setPen(QPen(QColor("#5A5045"), 1))
        painter.drawPath(diagonal)

        mid_x = plot.left() + self._mid_anchor * plot.width()
        painter.setPen(QPen(QColor("#806B46"), 1))
        painter.drawLine(QPointF(mid_x, plot.top()), QPointF(mid_x, plot.bottom()))

        samples = np.linspace(0.0, 1.0, 180, dtype=np.float32)
        lut = highlight_shadow_tone_lut(self._adjustments, mid_anchor=self._mid_anchor, lut_size=4096)
        values = np.interp(samples, np.linspace(0.0, 1.0, len(lut), dtype=np.float32), lut)

        path = QPainterPath()
        for index, value in enumerate(values):
            x = plot.left() + float(samples[index]) * plot.width()
            y = plot.bottom() - float(value) * plot.height()
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(QColor("#FFB000"), 2))
        painter.drawPath(path)

        control_x, control_y = highlight_shadow_control_points(self._adjustments, self._mid_anchor)
        painter.setPen(QPen(QColor("#FFCF66"), 1))
        painter.setBrush(QColor("#FFB000"))
        for x_value, y_value in zip(control_x, control_y):
            x = plot.left() + float(x_value) * plot.width()
            y = plot.bottom() - float(y_value) * plot.height()
            painter.drawEllipse(QPointF(x, y), 3.0, 3.0)

        painter.setPen(QColor("#A69680"))
        painter.drawText(
            plot.adjusted(6, 4, -6, -4),
            Qt.AlignLeft | Qt.AlignTop,
            f"Tone Modifier  mid {self._mid_anchor:.2f}",
        )
