from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.models import AdjustmentParams, ColorCorrectionParams


class ColorCorrectionPanel(QWidget):
    correctionChanged = Signal()
    analyzeRequested = Signal()
    interactionStarted = Signal()
    interactionFinished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("colorCorrectionPanel")
        self._value_labels: dict[QSlider, QLabel] = {}

        self.enabled_checkbox = QCheckBox("Enable Roll Color")
        self.analyze_button = QPushButton("Analyze Roll Color")
        self.status_label = QLabel("Not analyzed")
        self.status_label.setObjectName("mutedLabel")
        self.status_label.setWordWrap(True)

        self.roll_strength_slider = self._make_slider(0, 125, 100)
        self.frame_residual_slider = self._make_slider(0, 100, 80)
        self.tone_balance_slider = self._make_slider(0, 100, 100)
        self.exposure_match_slider = self._make_slider(0, 100, 0)

        self._build_layout()
        self._connect()

    def values(self) -> dict:
        return {
            "color_correction": ColorCorrectionParams(
                enabled=self.enabled_checkbox.isChecked(),
                roll_strength=self.roll_strength_slider.value(),
                frame_residual_strength=self.frame_residual_slider.value(),
                tone_balance_strength=self.tone_balance_slider.value(),
                exposure_match_strength=self.exposure_match_slider.value(),
            )
        }

    def set_adjustments(self, adjustments: AdjustmentParams) -> None:
        widgets = (
            self.enabled_checkbox,
            self.roll_strength_slider,
            self.frame_residual_slider,
            self.tone_balance_slider,
            self.exposure_match_slider,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            params = adjustments.color_correction
            self.enabled_checkbox.setChecked(params.enabled)
            self.roll_strength_slider.setValue(params.roll_strength)
            self.frame_residual_slider.setValue(params.frame_residual_strength)
            self.tone_balance_slider.setValue(params.tone_balance_strength)
            self.exposure_match_slider.setValue(params.exposure_match_strength)
            self._refresh_labels()
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_analyzing(self, analyzing: bool) -> None:
        self.analyze_button.setEnabled(not analyzing)
        self.analyze_button.setText("Analyzing..." if analyzing else "Analyze Roll Color")

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(self.enabled_checkbox)
        header.addStretch(1)
        root.addLayout(header)

        card = QFrame()
        card.setObjectName("colorCorrectionCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(8)
        card_layout.addWidget(self.analyze_button)
        card_layout.addWidget(self.status_label)
        card_layout.addLayout(self._slider_row("Roll Strength", self.roll_strength_slider))
        card_layout.addLayout(self._slider_row("Frame Residual", self.frame_residual_slider))
        card_layout.addLayout(self._slider_row("Tone Balance", self.tone_balance_slider))
        card_layout.addLayout(self._slider_row("Exposure Match", self.exposure_match_slider))
        root.addWidget(card)

    def _connect(self) -> None:
        self.enabled_checkbox.toggled.connect(lambda _checked: self.correctionChanged.emit())
        self.analyze_button.clicked.connect(self.analyzeRequested.emit)
        for slider in (
            self.roll_strength_slider,
            self.frame_residual_slider,
            self.tone_balance_slider,
            self.exposure_match_slider,
        ):
            slider.sliderPressed.connect(self.interactionStarted.emit)
            slider.sliderReleased.connect(self.interactionFinished.emit)
            slider.valueChanged.connect(lambda _value: self.correctionChanged.emit())

    def _make_slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        return slider

    def _slider_row(self, label: str, slider: QSlider) -> QVBoxLayout:
        wrapper = QVBoxLayout()
        header = QHBoxLayout()
        name = QLabel(label)
        value = QLabel(f"{slider.value()}%")
        value.setObjectName("sliderValue")
        self._value_labels[slider] = value
        slider.valueChanged.connect(lambda current, target=value: target.setText(f"{current}%"))
        header.addWidget(name)
        header.addStretch(1)
        header.addWidget(value)
        wrapper.addLayout(header)
        wrapper.addWidget(slider)
        return wrapper

    def _refresh_labels(self) -> None:
        for slider, label in self._value_labels.items():
            label.setText(f"{slider.value()}%")
