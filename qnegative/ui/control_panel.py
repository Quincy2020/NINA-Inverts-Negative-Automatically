from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.models import InvertMode, PrintCurveMode, ToolMode
from qnegative.core.models import AdjustmentParams
from qnegative.ui.collapsible_section import CollapsibleSection
from qnegative.ui.density_matrix_panel import DensityMatrixPanel
from qnegative.ui.histogram_levels import HistogramLevelsWidget
from qnegative.ui.print_curve_widget import PrintCurveWidget
from qnegative.ui.white_balance_panel import WhiteBalancePanel


class ControlPanel(QWidget):
    openRequested = Signal()
    exportRequested = Signal()
    invertRequested = Signal()
    resetRequested = Signal()
    toolChanged = Signal(object)
    adjustmentsChanged = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setMinimumWidth(340)

        self.file_label = QLabel("No file open")
        self.file_label.setObjectName("mutedLabel")
        self.image_label = QLabel("Preview waiting")
        self.image_label.setObjectName("mutedLabel")
        self.sequence_label = QLabel("No sequence")
        self.sequence_label.setObjectName("mutedLabel")
        self.mask_label = QLabel("Not selected")
        self.mask_label.setObjectName("mutedLabel")
        self.film_label = QLabel("Not selected")
        self.film_label.setObjectName("mutedLabel")
        for label in (
            self.file_label,
            self.image_label,
            self.sequence_label,
            self.mask_label,
            self.film_label,
        ):
            label.setWordWrap(True)

        self.open_button = QPushButton("Open RAW / Image")
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)

        self.invert_button = QPushButton("Invert Preview")
        self.invert_button.setEnabled(False)
        self.reset_button = QPushButton("Reset")
        self.invert_mode_combo = QComboBox()
        self.invert_mode_combo.addItem("Lab Print", InvertMode.NEGPY_PRINT.value)
        self.invert_mode_combo.addItem("Density", InvertMode.DENSITY.value)
        self.invert_mode_combo.addItem("Log Bounds", InvertMode.LOG_BOUNDS.value)
        self.invert_mode_combo.addItem("Simple", InvertMode.SIMPLE.value)
        self.print_curve_combo = QComboBox()
        self.print_curve_combo.addItem("Linear", PrintCurveMode.LINEAR.value)
        self.print_curve_combo.addItem("Soft Print", PrintCurveMode.SOFT.value)
        self.print_curve_combo.addItem("Standard Print", PrintCurveMode.STANDARD.value)
        self.print_curve_combo.addItem("Contrast Print", PrintCurveMode.CONTRAST.value)
        self.print_curve_combo.setCurrentIndex(2)
        self.print_curve_widget = PrintCurveWidget()
        self._style_combo_popup(self.invert_mode_combo)
        self._style_combo_popup(self.print_curve_combo)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        self.mask_picker_button = self._make_tool_button("Base Picker", ToolMode.MASK_PICKER)
        self.film_rect_button = self._make_tool_button("Frame Area", ToolMode.FILM_RECT)
        self.mask_picker_button.setChecked(True)

        self.exposure_slider = self._make_slider(-100, 100, 0)
        self.highlights_slider = self._make_slider(-100, 100, 0)
        self.shadows_slider = self._make_slider(-100, 100, 0)
        self.contrast_slider = self._make_slider(-100, 100, 0)
        self.saturation_slider = self._make_slider(-100, 100, 0)
        self.camera_color_slider = self._make_slider(0, 100, 0)
        self.histogram_levels = HistogramLevelsWidget()
        self.density_matrix_panel = DensityMatrixPanel()
        self.camera_color_panel = self._camera_color_developer_panel()
        self.white_balance_panel = WhiteBalancePanel()

        self._build_layout()
        self._connect()
        self._apply_style()

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("controlScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content.setObjectName("controlPanelContent")
        root = QVBoxLayout(content)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        title = QLabel("QNegativeLab")
        title.setObjectName("appTitle")
        root.addWidget(title)

        root.addWidget(self._file_section())
        root.addWidget(self._histogram_section())
        root.addWidget(self._tools_section())
        root.addWidget(self._selection_section())
        root.addWidget(self._invert_section())
        root.addWidget(self._adjustment_section())
        root.addWidget(self._white_balance_section())
        root.addStretch(1)
        root.addWidget(self._output_section())

        scroll_area.setWidget(content)
        outer.addWidget(scroll_area)

    def _file_section(self) -> QGroupBox:
        group = self._section("File")
        layout = QVBoxLayout(group)
        layout.addWidget(self.open_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.sequence_label)
        return group

    def _histogram_section(self) -> QGroupBox:
        group = self._section("Dynamic Range")
        layout = QVBoxLayout(group)
        layout.addWidget(self.histogram_levels)
        return group

    def _tools_section(self) -> QGroupBox:
        group = self._section("Tools")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(self.mask_picker_button)
        row.addWidget(self.film_rect_button)
        layout.addLayout(row)
        return group

    def _selection_section(self) -> QGroupBox:
        group = self._section("Selections")
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Base"))
        layout.addWidget(self.mask_label)
        layout.addWidget(self._divider())
        layout.addWidget(QLabel("Frame"))
        layout.addWidget(self.film_label)
        return group

    def _invert_section(self) -> QGroupBox:
        group = self._section("Invert")
        layout = QVBoxLayout(group)

        mode_row = QHBoxLayout()
        mode_label = QLabel("Invert Model")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.invert_mode_combo, 1)

        action_row = QHBoxLayout()
        action_row.addWidget(self.invert_button)
        action_row.addWidget(self.reset_button)

        layout.addLayout(mode_row)
        layout.addLayout(action_row)
        return group

    def _adjustment_section(self) -> QGroupBox:
        group = self._section("Basic")
        layout = QVBoxLayout(group)
        layout.addLayout(self._slider_row("Exposure", self.exposure_slider, "-1", "+1"))
        layout.addLayout(self._slider_row("Highlights", self.highlights_slider, "-1", "+1"))
        layout.addLayout(self._slider_row("Shadows", self.shadows_slider, "-1", "+1"))
        layout.addLayout(self._slider_row("Contrast", self.contrast_slider, "-1", "+1"))
        layout.addLayout(self._slider_row("Saturation", self.saturation_slider, "-1", "+1"))

        curve_row = QHBoxLayout()
        curve_row.addWidget(QLabel("Print Curve"))
        curve_row.addWidget(self.print_curve_combo, 1)
        layout.addLayout(curve_row)
        layout.addWidget(self.print_curve_widget)
        return group

    def _density_matrix_section(self) -> CollapsibleSection:
        return CollapsibleSection("Density Matrix", self.density_matrix_panel, expanded=False)

    def _white_balance_section(self) -> CollapsibleSection:
        return CollapsibleSection("White Balance", self.white_balance_panel, expanded=False)

    def _output_section(self) -> QGroupBox:
        group = self._section("Output")
        layout = QVBoxLayout(group)
        layout.addWidget(self.export_button)
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 0)
        self.export_progress.setFormat("Exporting TIFF...")
        self.export_progress.setTextVisible(True)
        self.export_progress.hide()
        layout.addWidget(self.export_progress)
        return group

    def _camera_color_developer_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("cameraColorPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addLayout(self._slider_row("Camera Color", self.camera_color_slider, "0", "100"))

        hint = QLabel("Camera transform mix. Keep at 0 for the current NegPy print workflow.")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        panel.setStyleSheet(
            """
            QWidget#cameraColorPanel {
                background: #20242b;
                color: #e8eaed;
            }
            QLabel {
                color: #e8eaed;
            }
            QLabel#mutedLabel {
                color: #9aa4b2;
                font-size: 12px;
            }
            QLabel#sliderValue {
                color: #cfd6df;
                min-width: 34px;
                qproperty-alignment: AlignRight;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3a414c;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: #d7dde5;
            }
            """
        )
        return panel

    def _connect(self) -> None:
        self.open_button.clicked.connect(self.openRequested.emit)
        self.export_button.clicked.connect(self.exportRequested.emit)
        self.invert_button.clicked.connect(self.invertRequested.emit)
        self.reset_button.clicked.connect(self.resetRequested.emit)
        self.tool_group.idClicked.connect(self._emit_tool_mode)
        self.invert_mode_combo.currentIndexChanged.connect(self._emit_adjustments)
        self.print_curve_combo.currentIndexChanged.connect(self._print_curve_changed)

        for slider in (
            self.exposure_slider,
            self.highlights_slider,
            self.shadows_slider,
            self.contrast_slider,
            self.saturation_slider,
            self.camera_color_slider,
        ):
            slider.valueChanged.connect(self._emit_adjustments)
        self.density_matrix_panel.matrixChanged.connect(self._emit_adjustments)
        self.white_balance_panel.balanceChanged.connect(self._emit_adjustments)
        self.white_balance_panel.pickWhiteBalanceRequested.connect(
            lambda: self.toolChanged.emit(ToolMode.WB_PICKER)
        )
        self.histogram_levels.levelsChanged.connect(lambda _levels: self._emit_adjustments())

    def _make_tool_button(self, text: str, mode: ToolMode) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setMinimumHeight(34)
        self.tool_group.addButton(button, list(ToolMode).index(mode))
        return button

    def _emit_tool_mode(self, button_id: int) -> None:
        self.toolChanged.emit(list(ToolMode)[button_id])

    def _emit_adjustments(self) -> None:
        self.adjustmentsChanged.emit(
            {
                "exposure": self.exposure_slider.value(),
                "highlights": self.highlights_slider.value(),
                "shadows": self.shadows_slider.value(),
                "contrast": self.contrast_slider.value(),
                "saturation": self.saturation_slider.value(),
                "camera_color_strength": self.camera_color_slider.value(),
                "invert_mode": self.invert_mode_combo.currentData(),
                "print_curve": self.print_curve_combo.currentData(),
                **self.histogram_levels.levels(),
                **self.density_matrix_panel.values(),
                **self.white_balance_panel.values(),
            }
        )

    def _print_curve_changed(self) -> None:
        self.print_curve_widget.set_curve_mode(self.print_curve_combo.currentData())
        self._emit_adjustments()

    def _make_slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        return slider

    def _style_combo_popup(self, combo: QComboBox) -> None:
        combo.view().setStyleSheet(
            """
            QAbstractItemView {
                background: #15191f;
                border: 1px solid #444c59;
                color: #f2f4f7;
                selection-background-color: #41627a;
                selection-color: #ffffff;
                outline: 0;
            }
            """
        )

    def _slider_row(self, label: str, slider: QSlider, left: str, right: str) -> QVBoxLayout:
        wrapper = QVBoxLayout()
        header = QHBoxLayout()
        name = QLabel(label)
        value = QLabel(str(slider.value()))
        value.setObjectName("sliderValue")
        slider.valueChanged.connect(lambda current, target=value: target.setText(str(current)))
        header.addWidget(name)
        header.addStretch(1)
        header.addWidget(value)

        scale = QHBoxLayout()
        left_label = QLabel(left)
        right_label = QLabel(right)
        left_label.setObjectName("mutedLabel")
        right_label.setObjectName("mutedLabel")
        scale.addWidget(left_label)
        scale.addStretch(1)
        scale.addWidget(right_label)

        wrapper.addLayout(header)
        wrapper.addWidget(slider)
        wrapper.addLayout(scale)
        return wrapper

    def _section(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setObjectName("panelSection")
        return group

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setObjectName("divider")
        return line

    def set_image_loaded(self, loaded: bool) -> None:
        self.invert_button.setEnabled(loaded)
        self.export_button.setEnabled(loaded and not self.export_progress.isVisible())

    def set_file_status(self, text: str) -> None:
        self.file_label.setText(text)

    def set_image_status(self, text: str) -> None:
        self.image_label.setText(text)

    def set_sequence_status(self, text: str) -> None:
        self.sequence_label.setText(text)

    def set_mask_status(self, text: str) -> None:
        self.mask_label.setText(text)

    def set_film_status(self, text: str) -> None:
        self.film_label.setText(text)

    def reset_adjustments(self) -> None:
        self.set_adjustments(AdjustmentParams(), emit=True)

    def set_histogram(self, histogram) -> None:
        self.histogram_levels.set_histogram(histogram)

    def set_levels(self, black: int, mid: int, white: int, *, emit: bool = False) -> None:
        self.histogram_levels.set_levels(black, mid, white, emit=emit)

    def set_export_progress(self, active: bool, *, value: int = 0, text: str = "Exporting TIFF...") -> None:
        self.export_progress.setVisible(active)
        if active:
            self.export_progress.setRange(0, 100)
            self.export_progress.setValue(value)
            self.export_progress.setFormat(f"{text} %p%")
        else:
            self.export_progress.setRange(0, 1)
            self.export_progress.setValue(0)

    def update_export_progress(self, value: int, text: str) -> None:
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(max(0, min(100, value)))
        self.export_progress.setFormat(f"{text} %p%")

    def set_adjustments(self, adjustments: AdjustmentParams, *, emit: bool = False) -> None:
        widgets = (
            self.exposure_slider,
            self.highlights_slider,
            self.shadows_slider,
            self.contrast_slider,
            self.saturation_slider,
            self.camera_color_slider,
            self.histogram_levels,
            self.density_matrix_panel,
            self.white_balance_panel,
            self.invert_mode_combo,
            self.print_curve_combo,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.exposure_slider.setValue(adjustments.exposure)
            self.highlights_slider.setValue(adjustments.highlights)
            self.shadows_slider.setValue(adjustments.shadows)
            self.contrast_slider.setValue(adjustments.contrast)
            self.saturation_slider.setValue(adjustments.saturation)
            self.camera_color_slider.setValue(adjustments.camera_color_strength)
            index = self.invert_mode_combo.findData(adjustments.invert_mode)
            self.invert_mode_combo.setCurrentIndex(0 if index < 0 else index)
            curve_index = self.print_curve_combo.findData(adjustments.print_curve)
            self.print_curve_combo.setCurrentIndex(2 if curve_index < 0 else curve_index)
            self.print_curve_widget.set_curve_mode(self.print_curve_combo.currentData())
            self.histogram_levels.set_levels(
                adjustments.black_point,
                adjustments.mid_point,
                adjustments.white_point,
                emit=False,
            )
            self.density_matrix_panel.set_adjustments(adjustments)
            self.white_balance_panel.set_adjustments(adjustments)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        if emit:
            self._emit_adjustments()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            #controlPanel {
                background: #20242b;
                color: #e8eaed;
            }
            #controlScrollArea {
                background: #20242b;
                border: none;
            }
            #controlPanelContent {
                background: #20242b;
            }
            #appTitle {
                font-size: 22px;
                font-weight: 700;
                padding: 4px 0 8px 0;
            }
            QGroupBox#panelSection {
                border: 1px solid #373d47;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px 8px 8px 8px;
                font-weight: 600;
            }
            QGroupBox#panelSection::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #cfd6df;
            }
            QLabel {
                color: #e8eaed;
            }
            QLabel#mutedLabel {
                color: #9aa4b2;
                font-size: 12px;
            }
            QLabel#sliderValue {
                color: #cfd6df;
                min-width: 34px;
                qproperty-alignment: AlignRight;
            }
            QPushButton, QToolButton {
                background: #2d333d;
                border: 1px solid #444c59;
                border-radius: 5px;
                color: #f2f4f7;
                padding: 7px 9px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #38414d;
            }
            QPushButton:disabled {
                color: #69717d;
                background: #252a31;
                border-color: #343b45;
            }
            QToolButton:checked {
                background: #41627a;
                border-color: #67a4c7;
            }
            QComboBox {
                background: #15191f;
                border: 1px solid #444c59;
                border-radius: 5px;
                color: #f2f4f7;
                padding: 5px 8px;
            }
            QComboBox QAbstractItemView {
                background: #15191f;
                border: 1px solid #444c59;
                color: #f2f4f7;
                selection-background-color: #41627a;
                selection-color: #ffffff;
                outline: 0;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3a414c;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: #d7dde5;
            }
            QFrame#divider {
                color: #3a414c;
                background: #3a414c;
                max-height: 1px;
            }
            QCheckBox {
                color: #e8eaed;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QProgressBar {
                background: #15191f;
                border: 1px solid #444c59;
                border-radius: 5px;
                color: #e8eaed;
                height: 18px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #67a4c7;
                border-radius: 4px;
            }
            """
        )
