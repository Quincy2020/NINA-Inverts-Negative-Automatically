from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.dust_model_registry import default_dust_model_plugin, dust_model_plugins
from qnegative.core.models import DustRemovalParams, InvertMode, LensCorrectionParams, PrintCurveMode, PrintCurveParams, ToolMode
from qnegative.core.models import AdjustmentParams
from qnegative.resources import resource_path
from qnegative.ui.collapsible_section import CollapsibleSection
from qnegative.ui.color_correction_panel import ColorCorrectionPanel
from qnegative.ui.histogram_levels import HistogramLevelsWidget
from qnegative.ui.print_curve_widget import PrintCurveWidget
from qnegative.ui.tone_curve_widget import ToneCurveWidget
from qnegative.ui.white_balance_panel import WhiteBalancePanel


class ControlPanel(QWidget):
    openRequested = Signal()
    exportRequested = Signal()
    batchExportRequested = Signal()
    invertRequested = Signal()
    resetRequested = Signal()
    toolChanged = Signal(object)
    autoDetectRequested = Signal(str)
    adjustmentsChanged = Signal(dict)
    adjustmentInteractionStarted = Signal()
    adjustmentInteractionFinished = Signal()
    lensProfileSaveRequested = Signal()
    lensProfileLoadRequested = Signal()
    lensFlatProfileCreateRequested = Signal()
    lensApplyAllRequested = Signal()
    lensApplyUnprocessedRequested = Signal()
    lensApplyCompletedRequested = Signal()
    rollColorAnalyzeRequested = Signal()
    dustMaskEditorRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setMinimumWidth(340)
        self._slider_value_labels: dict[QSlider, QLabel] = {}

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

        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.batch_export_button = QPushButton("Export Completed")
        self.batch_export_button.setEnabled(False)

        self.invert_button = QPushButton("Invert Preview")
        self.invert_button.setEnabled(False)
        self.reset_button = QPushButton("Reset")
        self.print_curve_combo = QComboBox()
        self.print_curve_combo.addItem("Linear", PrintCurveMode.LINEAR.value)
        self.print_curve_combo.addItem("Filmic Hable", PrintCurveMode.FILMIC_HABLE.value)
        self.print_curve_combo.addItem("Filmic ACES", PrintCurveMode.FILMIC_ACES.value)
        self.print_curve_combo.addItem("Soft Print", PrintCurveMode.SOFT.value)
        self.print_curve_combo.addItem("Standard Print", PrintCurveMode.STANDARD.value)
        self.print_curve_combo.addItem("Contrast Print", PrintCurveMode.CONTRAST.value)
        self.print_curve_combo.addItem("Contrast Shoulder", PrintCurveMode.CONTRAST_SHOULDER.value)
        self.print_curve_combo.setCurrentIndex(4)
        self.print_curve_widget = PrintCurveWidget()
        self.tone_curve_widget = ToneCurveWidget()
        self._tone_mid_anchor = 0.46
        self._style_combo_popup(self.print_curve_combo)
        self.print_curve_advanced_checkbox = QCheckBox("Custom Printer Curve")
        self.print_density_slider = self._make_slider(50, 150, 100)
        self.print_grade_slider = self._make_slider(100, 450, 300)
        self.print_highlight_bias_slider = self._make_slider(-20, 30, 12)
        self.print_highlight_width_slider = self._make_slider(20, 90, 55)
        self.print_shadow_bias_slider = self._make_slider(-20, 30, 0)
        self.print_shadow_width_slider = self._make_slider(20, 90, 55)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        self.mask_picker_button = self._make_tool_button("Base Picker", ToolMode.MASK_PICKER)
        self.film_rect_button = self._make_tool_button("Frame Area", ToolMode.FILM_RECT)
        self.film_rect_button.setChecked(True)
        self.auto_format_combo = QComboBox()
        self.auto_format_combo.addItem("Auto", "auto")
        self.auto_format_combo.addItem("135", "135")
        self.auto_format_combo.addItem("645", "645")
        self.auto_format_combo.addItem("66", "66")
        self.auto_format_combo.addItem("67", "67")
        self.auto_format_combo.addItem("69", "69")
        self._style_combo_popup(self.auto_format_combo)
        self.auto_frame_button = QPushButton("Auto Frame")

        self.exposure_slider = self._make_slider(-100, 100, 0)
        self.highlights_slider = self._make_slider(-100, 100, 0)
        self.shadows_slider = self._make_slider(-100, 100, 0)
        self.contrast_slider = self._make_slider(-100, 100, 0)
        self.saturation_slider = self._make_slider(-100, 100, 0)
        self.camera_color_slider = self._make_slider(0, 100, 0)
        self.analysis_inset_spin = QSpinBox()
        self.analysis_inset_spin.setRange(0, 20)
        self.analysis_inset_spin.setValue(5)
        self.analysis_inset_spin.setSuffix("%")
        self.lens_strength_slider = self._make_slider(0, 100, 0)
        self.lens_radius_slider = self._make_slider(20, 180, 100)
        self.lens_center_x_slider = self._make_slider(0, 100, 50)
        self.lens_center_y_slider = self._make_slider(0, 100, 50)
        self.lens_smoothness_slider = self._make_slider(25, 400, 200)
        self.lens_max_gain_slider = self._make_slider(100, 300, 200)
        self.lens_flat_strength_slider = self._make_slider(0, 200, 100)
        self._flat_profile_path: str | None = None
        self.histogram_levels = HistogramLevelsWidget()
        self.camera_color_panel = self._camera_color_developer_panel()
        self.white_balance_panel = WhiteBalancePanel()
        self.color_correction_panel = ColorCorrectionPanel()
        default_dust_plugin = default_dust_model_plugin()
        dust_plugins = dust_model_plugins()
        if all(plugin.plugin_id != default_dust_plugin.plugin_id for plugin in dust_plugins):
            dust_plugins.insert(0, default_dust_plugin)
        self._dust_model_defaults = {plugin.plugin_id: plugin for plugin in dust_plugins}
        self._dust_default_model_id = default_dust_plugin.plugin_id
        self.dust_model_combo = QComboBox()
        for plugin in dust_plugins:
            self.dust_model_combo.addItem(plugin.name, plugin.plugin_id)
        default_model_index = self.dust_model_combo.findData(self._dust_default_model_id)
        if default_model_index >= 0:
            self.dust_model_combo.setCurrentIndex(default_model_index)
        self._style_combo_popup(self.dust_model_combo)
        self.dust_enable_checkbox = QCheckBox("Enable Dust Removal")
        self.dust_adaptive_checkbox = QCheckBox("Adaptive texture protection")
        self.dust_adaptive_checkbox.setChecked(True)
        self.dust_threshold_slider = self._make_slider(1, 99, default_dust_plugin.threshold)
        self.dust_texture_penalty_slider = self._make_slider(0, 99, default_dust_plugin.texture_penalty)
        self.dust_max_threshold_slider = self._make_slider(1, 99, default_dust_plugin.max_threshold)
        self.dust_inpaint_radius_slider = self._make_slider(1, 16, default_dust_plugin.inpaint_radius)
        self.dust_edit_mask_button = QPushButton("Edit Dust Mask")
        self.dust_mask_status_label = QLabel("No dust mask preview")
        self.dust_mask_status_label.setObjectName("mutedLabel")
        self.dust_mask_status_label.setWordWrap(True)

        self._build_layout()
        self._connect()
        self._apply_style()
        self.set_image_loaded(False)

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

        title = QLabel()
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        banner = self._svg_pixmap("logo/Banner.svg", width=280, height=112)
        if banner is not None:
            title.setPixmap(banner)
            title.setFixedHeight(118)
        else:
            title.setText("NINA")
        root.addWidget(title)

        root.addWidget(self._status_section())
        root.addWidget(self._histogram_section())
        root.addWidget(self._tools_section())
        root.addWidget(self._adjustment_section())
        root.addWidget(self._color_correction_section())
        root.addWidget(self._white_balance_section())
        root.addWidget(self._lens_correction_section())
        root.addWidget(self._dust_removal_section())
        root.addStretch(1)
        root.addWidget(self._output_section())

        scroll_area.setWidget(content)
        outer.addWidget(scroll_area)

    def _status_section(self) -> QGroupBox:
        group = self._section("Status")
        layout = QVBoxLayout(group)
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
        row.addWidget(self.film_rect_button)
        row.addWidget(self.auto_frame_button)
        layout.addLayout(row)

        action_row = QHBoxLayout()
        action_row.addWidget(self.invert_button)
        action_row.addWidget(self.reset_button)
        layout.addLayout(action_row)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format"))
        format_row.addWidget(self.auto_format_combo, 1)
        layout.addLayout(format_row)

        return group

    def _adjustment_section(self) -> QGroupBox:
        group = self._section("Basic")
        layout = QVBoxLayout(group)
        layout.addLayout(self._slider_row("Exposure", self.exposure_slider, "-1", "+1"))
        layout.addLayout(self._slider_row("Highlights", self.highlights_slider, "-1", "+1"))
        layout.addLayout(self._slider_row("Shadows", self.shadows_slider, "-1", "+1"))
        layout.addWidget(self.tone_curve_widget)
        layout.addLayout(self._slider_row("Contrast", self.contrast_slider, "-1", "+1"))
        layout.addLayout(self._slider_row("Saturation", self.saturation_slider, "-1", "+1"))

        curve_row = QHBoxLayout()
        curve_row.addWidget(QLabel("Print Curve"))
        curve_row.addWidget(self.print_curve_combo, 1)
        layout.addLayout(curve_row)
        layout.addWidget(self.print_curve_widget)
        layout.addWidget(self._printer_curve_advanced_section())

        boundary_row = QHBoxLayout()
        boundary_row.addWidget(QLabel("Boundary"))
        boundary_row.addStretch(1)
        boundary_row.addWidget(self.analysis_inset_spin)
        layout.addLayout(boundary_row)
        return group

    def _printer_curve_advanced_section(self) -> CollapsibleSection:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.print_curve_advanced_checkbox)
        layout.addLayout(self._slider_row("Density", self.print_density_slider, "0.50", "1.50"))
        layout.addLayout(self._slider_row("Grade", self.print_grade_slider, "1.00", "4.50"))
        layout.addLayout(self._slider_row("Highlight Bias", self.print_highlight_bias_slider, "-0.20", "+0.30"))
        layout.addLayout(self._slider_row("Highlight Width", self.print_highlight_width_slider, "0.20", "0.90"))
        layout.addLayout(self._slider_row("Shadow Bias", self.print_shadow_bias_slider, "-0.20", "+0.30"))
        layout.addLayout(self._slider_row("Shadow Width", self.print_shadow_width_slider, "0.20", "0.90"))
        return CollapsibleSection("Printer Curve Advanced", panel, expanded=False)

    def _lens_correction_section(self) -> CollapsibleSection:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        frame = QFrame()
        frame.setObjectName("lensCorrectionCard")
        card_layout = QVBoxLayout(frame)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(8)

        self.lens_off_panel = QWidget()
        off_layout = QVBoxLayout(self.lens_off_panel)
        off_layout.setContentsMargins(6, 10, 6, 8)
        off_label = QLabel("Lens falloff correction is disabled.")
        off_label.setObjectName("mutedLabel")
        off_label.setWordWrap(True)
        off_layout.addWidget(off_label)

        self.lens_radial_panel = QWidget()
        radial_layout = QVBoxLayout(self.lens_radial_panel)
        radial_layout.setContentsMargins(6, 10, 6, 8)
        radial_layout.setSpacing(8)
        radial_layout.addLayout(self._slider_row("Strength", self.lens_strength_slider, "0", "100"))
        radial_layout.addLayout(self._slider_row("Radius", self.lens_radius_slider, "20", "180"))
        radial_layout.addLayout(self._slider_row("Center X", self.lens_center_x_slider, "0", "100"))
        radial_layout.addLayout(self._slider_row("Center Y", self.lens_center_y_slider, "0", "100"))
        radial_layout.addLayout(self._slider_row("Smoothness", self.lens_smoothness_slider, "0.25", "4.0"))
        radial_layout.addLayout(self._slider_row("Max Gain", self.lens_max_gain_slider, "1.0x", "3.0x"))

        self.lens_flat_panel = QWidget()
        flat_layout = QVBoxLayout(self.lens_flat_panel)
        flat_layout.setContentsMargins(6, 10, 6, 8)
        flat_layout.setSpacing(8)
        self.lens_flat_profile_card = QFrame()
        self.lens_flat_profile_card.setObjectName("lensProfileCard")
        profile_card_layout = QVBoxLayout(self.lens_flat_profile_card)
        profile_card_layout.setContentsMargins(8, 8, 8, 8)
        self.lens_flat_profile_label = QLabel("No flat-frame profile loaded")
        self.lens_flat_profile_label.setObjectName("mutedLabel")
        self.lens_flat_profile_label.setWordWrap(True)
        profile_card_layout.addWidget(self.lens_flat_profile_label)
        flat_layout.addWidget(self.lens_flat_profile_card)
        self.lens_flat_strength_row = QWidget()
        flat_strength_layout = QVBoxLayout(self.lens_flat_strength_row)
        flat_strength_layout.setContentsMargins(0, 0, 0, 0)
        flat_strength_layout.addLayout(self._slider_row("Strength", self.lens_flat_strength_slider, "0", "200"))
        flat_layout.addWidget(self.lens_flat_strength_row)
        self.lens_create_flat_profile_button = QPushButton("Create From Flat RAW")
        flat_layout.addWidget(self.lens_create_flat_profile_button)

        self.lens_tabs = QTabWidget()
        self.lens_tabs.setObjectName("lensCorrectionTabs")
        self.lens_tabs.addTab(self.lens_off_panel, "Off")
        self.lens_tabs.addTab(self.lens_radial_panel, "Radial")
        self.lens_tabs.addTab(self.lens_flat_panel, "Flat")
        card_layout.addWidget(self.lens_tabs)

        self.lens_profile_row = QWidget()
        profile_row = QHBoxLayout(self.lens_profile_row)
        profile_row.setContentsMargins(0, 0, 0, 0)
        self.lens_save_profile_button = QPushButton("Save")
        self.lens_load_profile_button = QPushButton("Load")
        profile_row.addWidget(self.lens_save_profile_button)
        profile_row.addWidget(self.lens_load_profile_button)
        card_layout.addWidget(self.lens_profile_row)

        self.lens_apply_row = QWidget()
        apply_row = QHBoxLayout(self.lens_apply_row)
        apply_row.setContentsMargins(0, 0, 0, 0)
        self.lens_apply_all_button = QPushButton("Apply All")
        self.lens_apply_unprocessed_button = QPushButton("Unprocessed")
        self.lens_apply_completed_button = QPushButton("Completed")
        apply_row.addWidget(self.lens_apply_all_button)
        apply_row.addWidget(self.lens_apply_unprocessed_button)
        apply_row.addWidget(self.lens_apply_completed_button)
        card_layout.addWidget(self.lens_apply_row)

        layout.addWidget(frame)
        self._set_lens_mode_widgets("off")
        return CollapsibleSection("Lens Correction", panel, expanded=False)

    def _white_balance_section(self) -> CollapsibleSection:
        return CollapsibleSection("White Balance", self.white_balance_panel, expanded=False)

    def _color_correction_section(self) -> CollapsibleSection:
        return CollapsibleSection("Color Correction", self.color_correction_panel, expanded=False)

    def _dust_removal_section(self) -> CollapsibleSection:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model"))
        model_row.addWidget(self.dust_model_combo, 1)
        layout.addLayout(model_row)
        layout.addWidget(self.dust_enable_checkbox)
        layout.addWidget(self.dust_adaptive_checkbox)
        layout.addLayout(self._slider_row("Threshold", self.dust_threshold_slider, "1", "99"))
        layout.addLayout(self._slider_row("Texture Guard", self.dust_texture_penalty_slider, "0", "99"))
        layout.addLayout(self._slider_row("Max Threshold", self.dust_max_threshold_slider, "1", "99"))
        layout.addLayout(self._slider_row("Inpaint Radius", self.dust_inpaint_radius_slider, "1", "16"))
        layout.addWidget(self._divider())
        layout.addWidget(self.dust_edit_mask_button)
        layout.addWidget(self.dust_mask_status_label)
        return CollapsibleSection("Dust Removal", panel, expanded=False)

    def _output_section(self) -> QGroupBox:
        group = self._section("Output")
        layout = QVBoxLayout(group)
        layout.addWidget(self.export_button)
        layout.addWidget(self.batch_export_button)
        self.activity_progress = QProgressBar()
        self.activity_progress.setRange(0, 0)
        self.activity_progress.setTextVisible(True)
        self.activity_progress.hide()
        layout.addWidget(self.activity_progress)
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 0)
        self.export_progress.setFormat("Exporting...")
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

        hint = QLabel("Camera transform mix. Keep at 0 for the current Lab Print workflow.")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        panel.setStyleSheet(
            """
            QWidget#cameraColorPanel {
                background: #202020;
                color: #E8E1D5;
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
                min-width: 34px;
                qproperty-alignment: AlignRight;
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
            """
        )
        return panel

    def _connect(self) -> None:
        self.export_button.clicked.connect(self.exportRequested.emit)
        self.batch_export_button.clicked.connect(self.batchExportRequested.emit)
        self.invert_button.clicked.connect(self.invertRequested.emit)
        self.reset_button.clicked.connect(self.resetRequested.emit)
        self.auto_frame_button.clicked.connect(lambda: self.autoDetectRequested.emit("frame_base"))
        self.tool_group.idClicked.connect(self._emit_tool_mode)
        self.print_curve_combo.currentIndexChanged.connect(self._print_curve_changed)

        for slider in (
            self.exposure_slider,
            self.highlights_slider,
            self.shadows_slider,
            self.contrast_slider,
            self.saturation_slider,
            self.camera_color_slider,
            self.print_density_slider,
            self.print_grade_slider,
            self.print_highlight_bias_slider,
            self.print_highlight_width_slider,
            self.print_shadow_bias_slider,
            self.print_shadow_width_slider,
            self.lens_strength_slider,
            self.lens_radius_slider,
            self.lens_center_x_slider,
            self.lens_center_y_slider,
            self.lens_smoothness_slider,
            self.lens_max_gain_slider,
            self.lens_flat_strength_slider,
            self.dust_threshold_slider,
            self.dust_texture_penalty_slider,
            self.dust_max_threshold_slider,
            self.dust_inpaint_radius_slider,
        ):
            slider.sliderPressed.connect(self.adjustmentInteractionStarted.emit)
            slider.sliderReleased.connect(self.adjustmentInteractionFinished.emit)
            slider.valueChanged.connect(self._emit_adjustments)
            slider.valueChanged.connect(lambda _value, current=slider: self._refresh_slider_value_label(current))
        self.lens_tabs.currentChanged.connect(self._lens_tab_changed)
        self.lens_save_profile_button.clicked.connect(self.lensProfileSaveRequested.emit)
        self.lens_load_profile_button.clicked.connect(self.lensProfileLoadRequested.emit)
        self.lens_create_flat_profile_button.clicked.connect(self.lensFlatProfileCreateRequested.emit)
        self.lens_apply_all_button.clicked.connect(self.lensApplyAllRequested.emit)
        self.lens_apply_unprocessed_button.clicked.connect(self.lensApplyUnprocessedRequested.emit)
        self.lens_apply_completed_button.clicked.connect(self.lensApplyCompletedRequested.emit)
        self.analysis_inset_spin.valueChanged.connect(self._emit_adjustments)
        self.print_curve_advanced_checkbox.toggled.connect(self._emit_adjustments)
        self.white_balance_panel.balanceChanged.connect(self._emit_adjustments)
        self.white_balance_panel.interactionStarted.connect(self.adjustmentInteractionStarted.emit)
        self.white_balance_panel.interactionFinished.connect(self.adjustmentInteractionFinished.emit)
        self.white_balance_panel.pickWhiteBalanceRequested.connect(
            lambda: self.toolChanged.emit(ToolMode.WB_PICKER)
        )
        self.histogram_levels.interactionStarted.connect(self.adjustmentInteractionStarted.emit)
        self.histogram_levels.interactionFinished.connect(self.adjustmentInteractionFinished.emit)
        self.histogram_levels.levelsChanged.connect(lambda _levels: self._emit_adjustments())
        self.color_correction_panel.correctionChanged.connect(self._emit_adjustments)
        self.color_correction_panel.interactionStarted.connect(self.adjustmentInteractionStarted.emit)
        self.color_correction_panel.interactionFinished.connect(self.adjustmentInteractionFinished.emit)
        self.color_correction_panel.analyzeRequested.connect(self.rollColorAnalyzeRequested.emit)
        self.dust_model_combo.currentIndexChanged.connect(self._dust_model_changed)
        self.dust_enable_checkbox.toggled.connect(self._emit_adjustments)
        self.dust_adaptive_checkbox.toggled.connect(self._emit_adjustments)
        self.dust_edit_mask_button.clicked.connect(self.dustMaskEditorRequested.emit)

    def _make_tool_button(self, text: str, mode: ToolMode) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setMinimumHeight(34)
        self.tool_group.addButton(button, list(ToolMode).index(mode))
        return button

    def _lens_tab_changed(self, _index: int) -> None:
        self._set_lens_mode_widgets(self._current_lens_mode())
        self._emit_adjustments()

    def _set_lens_mode_widgets(self, mode: str) -> None:
        normalized = mode if mode in {"off", "radial", "flat_frame"} else "off"
        self.lens_profile_row.setVisible(normalized != "off")
        self.lens_apply_row.setVisible(normalized != "off")
        self.lens_flat_strength_row.setVisible(
            normalized == "flat_frame" and bool(self._flat_profile_path)
        )
        if normalized == "radial":
            self.lens_tabs.setCurrentIndex(1)
        elif normalized == "flat_frame":
            self.lens_tabs.setCurrentIndex(2)
        else:
            self.lens_tabs.setCurrentIndex(0)
        self._update_lens_tabs_height()

    def _current_lens_mode(self) -> str:
        return ("off", "radial", "flat_frame")[max(0, min(2, self.lens_tabs.currentIndex()))]

    def _current_dust_model_id(self) -> str:
        model_id = self.dust_model_combo.currentData()
        return str(model_id or self._dust_default_model_id)

    def _dust_model_changed(self, _index: int) -> None:
        plugin = self._dust_model_defaults.get(self._current_dust_model_id())
        if plugin is not None:
            self._set_dust_defaults(plugin)
        self._emit_adjustments()

    def _set_dust_defaults(self, plugin) -> None:
        sliders = (
            self.dust_threshold_slider,
            self.dust_texture_penalty_slider,
            self.dust_max_threshold_slider,
            self.dust_inpaint_radius_slider,
        )
        for slider in sliders:
            slider.blockSignals(True)
        try:
            self.dust_threshold_slider.setValue(plugin.threshold)
            self.dust_texture_penalty_slider.setValue(plugin.texture_penalty)
            self.dust_max_threshold_slider.setValue(plugin.max_threshold)
            self.dust_inpaint_radius_slider.setValue(plugin.inpaint_radius)
            self._refresh_slider_value_labels()
        finally:
            for slider in sliders:
                slider.blockSignals(False)

    def _update_lens_tabs_height(self) -> None:
        current = self.lens_tabs.currentWidget()
        if current is None:
            return
        page_height = current.sizeHint().height()
        tab_height = self.lens_tabs.tabBar().sizeHint().height()
        self.lens_tabs.setFixedHeight(max(82, page_height + tab_height + 18))

    def _emit_tool_mode(self, button_id: int) -> None:
        self.toolChanged.emit(list(ToolMode)[button_id])

    def _emit_adjustments(self) -> None:
        lens_mode = self._current_lens_mode()
        values = {
            "exposure": self.exposure_slider.value(),
            "highlights": self.highlights_slider.value(),
            "shadows": self.shadows_slider.value(),
            "contrast": self.contrast_slider.value(),
            "saturation": self.saturation_slider.value(),
            "camera_color_strength": self.camera_color_slider.value(),
            "lens_correction": LensCorrectionParams(
                enabled=lens_mode != "off",
                mode=lens_mode,
                strength=self.lens_strength_slider.value(),
                radius=self.lens_radius_slider.value(),
                center_x=self.lens_center_x_slider.value(),
                center_y=self.lens_center_y_slider.value(),
                smoothness=self.lens_smoothness_slider.value(),
                max_gain=self.lens_max_gain_slider.value(),
                flat_profile_path=self._flat_profile_path,
                flat_strength=self.lens_flat_strength_slider.value(),
            ),
            "dust_removal": DustRemovalParams(
                enabled=self.dust_enable_checkbox.isChecked(),
                model_id=self._current_dust_model_id(),
                threshold=self.dust_threshold_slider.value(),
                adaptive=self.dust_adaptive_checkbox.isChecked(),
                texture_penalty=self.dust_texture_penalty_slider.value(),
                max_threshold=self.dust_max_threshold_slider.value(),
                inpaint_radius=self.dust_inpaint_radius_slider.value(),
            ),
            "analysis_inset_percent": self.analysis_inset_spin.value(),
            "invert_mode": InvertMode.LAB_PRINT.value,
            "print_curve": self.print_curve_combo.currentData(),
            "print_curve_params": PrintCurveParams(
                enabled=self.print_curve_advanced_checkbox.isChecked(),
                density=self.print_density_slider.value() / 100.0,
                grade=self.print_grade_slider.value() / 100.0,
                highlight_bias=self.print_highlight_bias_slider.value() / 100.0,
                highlight_width=self.print_highlight_width_slider.value() / 100.0,
                shadow_bias=self.print_shadow_bias_slider.value() / 100.0,
                shadow_width=self.print_shadow_width_slider.value() / 100.0,
            ),
            **self.histogram_levels.levels(),
            **self.white_balance_panel.values(),
            **self.color_correction_panel.values(),
        }
        self._refresh_tone_curve_widget(values)
        self.adjustmentsChanged.emit(values)

    def _print_curve_changed(self) -> None:
        self.print_curve_widget.set_curve_mode(self.print_curve_combo.currentData())
        self._emit_adjustments()

    def _refresh_tone_curve_widget(self, values: dict | None = None) -> None:
        if values is None:
            values = {
                "highlights": self.highlights_slider.value(),
                "shadows": self.shadows_slider.value(),
            }
        self.tone_curve_widget.set_tone(
            adjustments=AdjustmentParams(
                highlights=int(values.get("highlights", 0)),
                shadows=int(values.get("shadows", 0)),
            ),
            mid_anchor=self._tone_mid_anchor,
        )

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
                background: #1A1A1A;
                border: 1px solid #444444;
                color: #F2EEE6;
                selection-background-color: #663300;
                selection-color: #FFB000;
                outline: 0;
            }
            """
        )

    def _svg_pixmap(self, relative_path: str, *, width: int, height: int) -> QPixmap | None:
        path = resource_path(relative_path)
        if not path.exists():
            return None
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer = QSvgRenderer(str(path))
        renderer.render(painter, QRectF(0, 0, width, height))
        painter.end()
        return pixmap

    def _slider_row(self, label: str, slider: QSlider, left: str, right: str) -> QVBoxLayout:
        wrapper = QVBoxLayout()
        header = QHBoxLayout()
        name = QLabel(label)
        value = QLabel(self._format_slider_value(slider, slider.value()))
        value.setObjectName("sliderValue")
        self._slider_value_labels[slider] = value
        slider.valueChanged.connect(
            lambda current, source=slider, target=value: target.setText(
                self._format_slider_value(source, current)
            )
        )
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

    def _format_slider_value(self, slider: QSlider, value: int) -> str:
        if slider in {
            self.print_density_slider,
            self.print_grade_slider,
            self.print_highlight_width_slider,
            self.print_shadow_width_slider,
        }:
            return f"{value / 100.0:.2f}"
        if slider in {self.print_highlight_bias_slider, self.print_shadow_bias_slider}:
            return f"{value / 100.0:+.2f}"
        if slider is self.lens_smoothness_slider:
            return f"{value / 100.0:.2f}"
        if slider is self.lens_max_gain_slider:
            return f"{value / 100.0:.2f}x"
        if slider is self.lens_flat_strength_slider:
            return f"{value}%"
        return str(value)

    def _refresh_slider_value_labels(self) -> None:
        for slider, label in self._slider_value_labels.items():
            self._refresh_slider_value_label(slider)

    def _refresh_slider_value_label(self, slider: QSlider) -> None:
        label = self._slider_value_labels.get(slider)
        if label is not None:
            label.setText(self._format_slider_value(slider, slider.value()))

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
        self.batch_export_button.setEnabled(loaded and not self.export_progress.isVisible())
        self.auto_frame_button.setEnabled(loaded)
        self.dust_edit_mask_button.setEnabled(loaded)

    def auto_format(self) -> str:
        data = self.auto_format_combo.currentData()
        return str(data or "auto")

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

    def set_tone_mid_anchor(self, mid_anchor: float) -> None:
        self._tone_mid_anchor = float(mid_anchor)
        self._refresh_tone_curve_widget()

    def set_export_progress(self, active: bool, *, value: int = 0, text: str = "Exporting...") -> None:
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

    def set_roll_color_status(self, text: str) -> None:
        self.color_correction_panel.set_status(text)

    def set_roll_color_analyzing(self, analyzing: bool) -> None:
        self.color_correction_panel.set_analyzing(analyzing)

    def set_dust_mask_status(self, text: str) -> None:
        self.dust_mask_status_label.setText(text)

    def set_activity_progress(self, active: bool, *, text: str = "Working...") -> None:
        self.activity_progress.setVisible(active)
        if active:
            self.activity_progress.setRange(0, 0)
            self.activity_progress.setFormat(text)
        else:
            self.activity_progress.setRange(0, 1)
            self.activity_progress.setValue(0)

    def set_adjustments(self, adjustments: AdjustmentParams, *, emit: bool = False) -> None:
        widgets = (
            self.exposure_slider,
            self.highlights_slider,
            self.shadows_slider,
            self.contrast_slider,
            self.saturation_slider,
            self.camera_color_slider,
            self.print_curve_advanced_checkbox,
            self.print_density_slider,
            self.print_grade_slider,
            self.print_highlight_bias_slider,
            self.print_highlight_width_slider,
            self.print_shadow_bias_slider,
            self.print_shadow_width_slider,
            self.lens_tabs,
            self.lens_create_flat_profile_button,
            self.lens_strength_slider,
            self.lens_radius_slider,
            self.lens_center_x_slider,
            self.lens_center_y_slider,
            self.lens_smoothness_slider,
            self.lens_max_gain_slider,
            self.lens_flat_strength_slider,
            self.dust_model_combo,
            self.dust_enable_checkbox,
            self.dust_adaptive_checkbox,
            self.dust_threshold_slider,
            self.dust_texture_penalty_slider,
            self.dust_max_threshold_slider,
            self.dust_inpaint_radius_slider,
            self.analysis_inset_spin,
            self.histogram_levels,
            self.white_balance_panel,
            self.color_correction_panel,
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
            self.print_curve_advanced_checkbox.setChecked(adjustments.print_curve_params.enabled)
            self.print_density_slider.setValue(int(round(adjustments.print_curve_params.density * 100.0)))
            self.print_grade_slider.setValue(int(round(adjustments.print_curve_params.grade * 100.0)))
            self.print_highlight_bias_slider.setValue(int(round(adjustments.print_curve_params.highlight_bias * 100.0)))
            self.print_highlight_width_slider.setValue(int(round(adjustments.print_curve_params.highlight_width * 100.0)))
            self.print_shadow_bias_slider.setValue(int(round(adjustments.print_curve_params.shadow_bias * 100.0)))
            self.print_shadow_width_slider.setValue(int(round(adjustments.print_curve_params.shadow_width * 100.0)))
            lens_mode = (
                adjustments.lens_correction.mode
                if adjustments.lens_correction.enabled
                else "off"
            )
            self._flat_profile_path = adjustments.lens_correction.flat_profile_path
            self._set_lens_mode_widgets(lens_mode)
            self.lens_flat_profile_label.setText(
                f"Flat profile\n{self._flat_profile_path.split('/')[-1].split(chr(92))[-1]}"
                if self._flat_profile_path
                else "No flat-frame profile loaded"
            )
            self.lens_strength_slider.setValue(adjustments.lens_correction.strength)
            self.lens_radius_slider.setValue(adjustments.lens_correction.radius)
            self.lens_center_x_slider.setValue(adjustments.lens_correction.center_x)
            self.lens_center_y_slider.setValue(adjustments.lens_correction.center_y)
            self.lens_smoothness_slider.setValue(adjustments.lens_correction.smoothness)
            self.lens_max_gain_slider.setValue(adjustments.lens_correction.max_gain)
            self.lens_flat_strength_slider.setValue(adjustments.lens_correction.flat_strength)
            self.lens_flat_strength_row.setVisible(
                lens_mode == "flat_frame" and bool(self._flat_profile_path)
            )
            dust_model_id = adjustments.dust_removal.model_id or self._dust_default_model_id
            dust_model_index = self.dust_model_combo.findData(dust_model_id)
            if dust_model_index < 0:
                dust_model_index = self.dust_model_combo.findData(self._dust_default_model_id)
            self.dust_model_combo.setCurrentIndex(max(0, dust_model_index))
            self.dust_enable_checkbox.setChecked(adjustments.dust_removal.enabled)
            self.dust_adaptive_checkbox.setChecked(adjustments.dust_removal.adaptive)
            self.dust_threshold_slider.setValue(adjustments.dust_removal.threshold)
            self.dust_texture_penalty_slider.setValue(adjustments.dust_removal.texture_penalty)
            self.dust_max_threshold_slider.setValue(adjustments.dust_removal.max_threshold)
            self.dust_inpaint_radius_slider.setValue(adjustments.dust_removal.inpaint_radius)
            self.analysis_inset_spin.setValue(adjustments.analysis_inset_percent)
            curve_index = self.print_curve_combo.findData(adjustments.print_curve)
            standard_index = self.print_curve_combo.findData(PrintCurveMode.STANDARD.value)
            self.print_curve_combo.setCurrentIndex(standard_index if curve_index < 0 else curve_index)
            self.print_curve_widget.set_curve_mode(self.print_curve_combo.currentData())
            self._refresh_tone_curve_widget(
                {
                    "highlights": adjustments.highlights,
                    "shadows": adjustments.shadows,
                }
            )
            self.histogram_levels.set_levels(
                adjustments.black_point,
                adjustments.mid_point,
                adjustments.white_point,
                emit=False,
            )
            self.white_balance_panel.set_adjustments(adjustments)
            self.color_correction_panel.set_adjustments(adjustments)
            self._refresh_slider_value_labels()
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        if emit:
            self._emit_adjustments()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            #controlPanel {
                background: #202020;
                color: #E8E1D5;
            }
            #controlScrollArea {
                background: #202020;
                border: none;
            }
            #controlPanelContent {
                background: #202020;
            }
            #appTitle {
                font-size: 22px;
                font-weight: 700;
                padding: 4px 0 8px 0;
            }
            QGroupBox#panelSection {
                border: 1px solid #443B32;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px 8px 8px 8px;
                font-weight: 600;
            }
            QGroupBox#panelSection::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #D8D0C2;
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
                min-width: 34px;
                qproperty-alignment: AlignRight;
            }
            QFrame#lensCorrectionCard {
                background: transparent;
                border: none;
            }
            QFrame#lensProfileCard {
                background: #121212;
                border: 1px solid #443B32;
                border-radius: 5px;
            }
            QTabWidget#lensCorrectionTabs::pane {
                border: 1px solid #443B32;
                border-radius: 5px;
                background: #202020;
                top: -1px;
            }
            QTabWidget#lensCorrectionTabs QTabBar::tab {
                background: #2A2520;
                color: #D8D0C2;
                border: 1px solid #4A4034;
                padding: 6px 9px;
                min-width: 44px;
            }
            QTabWidget#lensCorrectionTabs QTabBar::tab:selected {
                background: #663300;
                color: #F2EEE6;
                border-color: #FFB000;
            }
            QTabWidget#lensCorrectionTabs QTabBar::tab:first {
                border-top-left-radius: 5px;
            }
            QTabWidget#lensCorrectionTabs QTabBar::tab:last {
                border-top-right-radius: 5px;
            }
            QPushButton, QToolButton {
                background: #2A2520;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #F2EEE6;
                padding: 7px 9px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #342A1D;
            }
            QPushButton:disabled {
                color: #817666;
                background: #25221F;
                border-color: #3D352D;
            }
            QToolButton:checked {
                background: #663300;
                border-color: #FFB000;
            }
            QComboBox {
                background: #1A1A1A;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #F2EEE6;
                padding: 5px 8px;
            }
            QComboBox QAbstractItemView {
                background: #1A1A1A;
                border: 1px solid #4A4034;
                color: #F2EEE6;
                selection-background-color: #663300;
                selection-color: #ffffff;
                outline: 0;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
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
            QFrame#divider {
                color: #443B32;
                background: #443B32;
                max-height: 1px;
            }
            QCheckBox {
                color: #E8E1D5;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QProgressBar {
                background: #1A1A1A;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #E8E1D5;
                height: 18px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #FFB000;
                border-radius: 4px;
            }
            """
        )
