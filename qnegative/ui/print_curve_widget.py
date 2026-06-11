from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from qnegative.core.models import PrintCurveMode
from qnegative.core.pipeline import print_curve_values


class PrintCurveWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("printCurveWidget")
        self.setMinimumHeight(96)
        self._curve_mode = PrintCurveMode.STANDARD.value

    def set_curve_mode(self, curve_mode: str) -> None:
        self._curve_mode = curve_mode
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1A1A1A"))

        plot = QRectF(12, 10, max(1, self.width() - 24), max(1, self.height() - 20))
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

        samples = np.linspace(0.0, 1.0, 160, dtype=np.float32)
        values = print_curve_values(samples, self._curve_mode)
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

        painter.setPen(QColor("#A69680"))
        painter.drawText(plot.adjusted(6, 4, -6, -4), Qt.AlignLeft | Qt.AlignTop, self._label())

    def _label(self) -> str:
        labels = {
            PrintCurveMode.LINEAR.value: "Linear",
            PrintCurveMode.FILMIC_HABLE.value: "Filmic Hable",
            PrintCurveMode.FILMIC_ACES.value: "Filmic ACES",
            PrintCurveMode.SOFT.value: "Soft Print",
            PrintCurveMode.STANDARD.value: "Standard Print",
            PrintCurveMode.CONTRAST.value: "Contrast Print",
            PrintCurveMode.CONTRAST_SHOULDER.value: "Contrast Shoulder",
        }
        return labels.get(self._curve_mode, "Standard Print")
