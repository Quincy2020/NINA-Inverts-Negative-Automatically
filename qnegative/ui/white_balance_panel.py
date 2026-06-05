from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.models import AdjustmentParams, BalanceAxis, ColorBalanceParams, TonalBalance


class ColorSliderRow(QWidget):
    valueChanged = Signal()
    interactionStarted = Signal()
    interactionFinished = Signal()

    def __init__(
        self,
        label: str,
        *,
        minimum: int,
        maximum: int,
        value: int,
        gradient: tuple[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("colorSliderRow")

        self.name_label = QLabel(label)
        self.name_label.setFixedWidth(48)

        self.minus_button = QToolButton()
        self.minus_button.setText("<")
        self.minus_button.setToolTip("-1")
        self.minus_button.setFixedSize(22, 24)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setMinimumWidth(96)

        self.plus_button = QToolButton()
        self.plus_button.setText(">")
        self.plus_button.setToolTip("+1")
        self.plus_button.setFixedSize(22, 24)

        self.spin_box = QSpinBox()
        self.spin_box.setRange(minimum, maximum)
        self.spin_box.setValue(value)
        self.spin_box.setFixedWidth(58)
        self.spin_box.setButtonSymbols(QSpinBox.NoButtons)
        self.spin_box.setAlignment(Qt.AlignRight)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.name_label)
        layout.addWidget(self.minus_button)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.plus_button)
        layout.addWidget(self.spin_box)

        self.minus_button.clicked.connect(lambda: self.set_value(self.value() - 1))
        self.plus_button.clicked.connect(lambda: self.set_value(self.value() + 1))
        self.slider.sliderPressed.connect(self.interactionStarted.emit)
        self.slider.sliderReleased.connect(self.interactionFinished.emit)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin_box.valueChanged.connect(self._spin_changed)

        self._apply_style(gradient)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int) -> None:
        self.slider.setValue(value)

    def _slider_changed(self, value: int) -> None:
        if self.spin_box.value() != value:
            previous = self.spin_box.blockSignals(True)
            try:
                self.spin_box.setValue(value)
            finally:
                self.spin_box.blockSignals(previous)
        self.valueChanged.emit()

    def _spin_changed(self, value: int) -> None:
        if self.slider.value() != value:
            previous = self.slider.blockSignals(True)
            try:
                self.slider.setValue(value)
            finally:
                self.slider.blockSignals(previous)
        self.valueChanged.emit()

    def _apply_style(self, gradient: tuple[str, str]) -> None:
        left, right = gradient
        self.slider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                height: 7px;
                border-radius: 3px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {left}, stop:0.5 #51483E, stop:1 {right});
            }}
            QSlider::handle:horizontal {{
                width: 15px;
                height: 15px;
                margin: -5px 0;
                border-radius: 8px;
                background: #f0f2f5;
                border: 1px solid #9A8B75;
            }}
            """
        )
        self.setStyleSheet(
            """
            QLabel {
                color: #E8E1D5;
            }
            QToolButton {
                background: #2A2520;
                border: 1px solid #4A4034;
                border-radius: 4px;
                color: #F2EEE6;
                padding: 0;
            }
            QToolButton:hover {
                background: #342A1D;
            }
            QSpinBox {
                background: #1A1A1A;
                border: 1px solid #4A4034;
                border-radius: 4px;
                color: #F2EEE6;
                padding: 3px 5px;
            }
            """
        )


class WhiteBalancePanel(QWidget):
    balanceChanged = Signal()
    pickWhiteBalanceRequested = Signal()
    interactionStarted = Signal()
    interactionFinished = Signal()

    AXIS_GRADIENTS = {
        "red_cyan": ("#00bfd1", "#e14d4d"),
        "green_magenta": ("#c246c8", "#38c56a"),
        "blue_yellow": ("#d0b72f", "#2e70d6"),
        "range": ("#3A332B", "#A69680"),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("whiteBalancePanel")
        self._rows: dict[str, ColorSliderRow] = {}

        self.auto_wb_checkbox = QCheckBox("Auto CMY")
        self.auto_wb_checkbox.setChecked(True)
        self.auto_wb_checkbox.stateChanged.connect(lambda _state: self.balanceChanged.emit())

        self.pick_wb_button = QToolButton()
        self.pick_wb_button.setText("Pick WB")
        self.pick_wb_button.setToolTip("Pick a neutral point and write it to printer CMY")
        self.pick_wb_button.clicked.connect(self.pickWhiteBalanceRequested.emit)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("whiteBalanceTabs")
        self.tabs.addTab(self._axis_widget("printer"), "Printer")
        self.tabs.addTab(self._tonal_widget("midtones"), "Mids")
        self.tabs.addTab(self._tonal_widget("highlights"), "Highs")
        self.tabs.addTab(self._tonal_widget("shadows"), "Shadows")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(8)
        root.addWidget(self.auto_wb_checkbox)
        root.addWidget(self.pick_wb_button)
        root.addWidget(self.tabs)

        self._apply_style()

    def values(self) -> dict:
        return {
            "auto_wb": self.auto_wb_checkbox.isChecked(),
            "printer_balance": self._axis_values("printer"),
            "color_balance": ColorBalanceParams(
                global_balance=BalanceAxis(),
                shadows=self._tonal_values("shadows"),
                midtones=self._tonal_values("midtones"),
                highlights=self._tonal_values("highlights"),
            ),
        }

    def set_adjustments(self, adjustments: AdjustmentParams) -> None:
        previous = self.blockSignals(True)
        try:
            self.auto_wb_checkbox.setChecked(adjustments.auto_wb)
            self._set_axis_values("printer", adjustments.printer_balance)
            self._set_tonal_values("shadows", adjustments.color_balance.shadows)
            self._set_tonal_values("midtones", adjustments.color_balance.midtones)
            self._set_tonal_values("highlights", adjustments.color_balance.highlights)
        finally:
            self.blockSignals(previous)

    def _axis_widget(self, prefix: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 10, 6, 8)
        layout.setSpacing(8)
        layout.addWidget(self._make_row("R / C", f"{prefix}.red_cyan", -100, 100, 0, "red_cyan"))
        layout.addWidget(self._make_row("G / M", f"{prefix}.green_magenta", -100, 100, 0, "green_magenta"))
        layout.addWidget(self._make_row("B / Y", f"{prefix}.blue_yellow", -100, 100, 0, "blue_yellow"))
        if prefix == "printer":
            reset_button = QToolButton()
            reset_button.setText("Reset Printer")
            reset_button.setToolTip("Reset manual printer CMY balance")
            reset_button.clicked.connect(self.reset_printer_balance)
            layout.addWidget(reset_button)
        return widget

    def reset_printer_balance(self) -> None:
        self._set_axis_values("printer", BalanceAxis())
        self.balanceChanged.emit()

    def _tonal_widget(self, prefix: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 10, 6, 8)
        layout.setSpacing(8)
        layout.addWidget(self._make_row("Range", f"{prefix}.range", 0, 100, 50, "range"))
        layout.addWidget(self._make_row("R / C", f"{prefix}.red_cyan", -100, 100, 0, "red_cyan"))
        layout.addWidget(self._make_row("G / M", f"{prefix}.green_magenta", -100, 100, 0, "green_magenta"))
        layout.addWidget(self._make_row("B / Y", f"{prefix}.blue_yellow", -100, 100, 0, "blue_yellow"))
        return widget

    def _make_row(
        self,
        label: str,
        key: str,
        minimum: int,
        maximum: int,
        value: int,
        gradient_key: str,
    ) -> ColorSliderRow:
        row = ColorSliderRow(
            label,
            minimum=minimum,
            maximum=maximum,
            value=value,
            gradient=self.AXIS_GRADIENTS[gradient_key],
        )
        row.valueChanged.connect(self.balanceChanged.emit)
        row.interactionStarted.connect(self.interactionStarted.emit)
        row.interactionFinished.connect(self.interactionFinished.emit)
        self._rows[key] = row
        return row

    def _axis_values(self, prefix: str) -> BalanceAxis:
        return BalanceAxis(
            red_cyan=self._rows[f"{prefix}.red_cyan"].value(),
            green_magenta=self._rows[f"{prefix}.green_magenta"].value(),
            blue_yellow=self._rows[f"{prefix}.blue_yellow"].value(),
        )

    def _tonal_values(self, prefix: str) -> TonalBalance:
        return TonalBalance(
            red_cyan=self._rows[f"{prefix}.red_cyan"].value(),
            green_magenta=self._rows[f"{prefix}.green_magenta"].value(),
            blue_yellow=self._rows[f"{prefix}.blue_yellow"].value(),
            tonal_range=self._rows[f"{prefix}.range"].value(),
        )

    def _set_axis_values(self, prefix: str, axis: BalanceAxis) -> None:
        self._rows[f"{prefix}.red_cyan"].set_value(axis.red_cyan)
        self._rows[f"{prefix}.green_magenta"].set_value(axis.green_magenta)
        self._rows[f"{prefix}.blue_yellow"].set_value(axis.blue_yellow)

    def _set_tonal_values(self, prefix: str, balance: TonalBalance) -> None:
        self._rows[f"{prefix}.range"].set_value(balance.tonal_range)
        self._set_axis_values(prefix, balance)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#whiteBalancePanel {
                color: #E8E1D5;
            }
            QCheckBox {
                color: #E8E1D5;
                spacing: 8px;
                padding: 2px 0;
            }
            QTabWidget::pane {
                border: 1px solid #443B32;
                border-radius: 5px;
                background: #202020;
                top: -1px;
            }
            QTabBar::tab {
                background: #2A2520;
                color: #D8D0C2;
                border: 1px solid #4A4034;
                padding: 6px 9px;
                min-width: 44px;
            }
            QTabBar::tab:selected {
                background: #663300;
                color: #F2EEE6;
                border-color: #FFB000;
            }
            QTabBar::tab:first {
                border-top-left-radius: 5px;
            }
            QTabBar::tab:last {
                border-top-right-radius: 5px;
            }
            """
        )
