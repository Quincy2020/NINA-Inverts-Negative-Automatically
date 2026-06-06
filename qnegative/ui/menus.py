from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QApplication, QDockWidget, QHBoxLayout, QLabel, QSpinBox, QWidget, QWidgetAction

from qnegative.core.models import ToolMode
from qnegative.core.pipeline import (
    LOG_PRINT_CURVE_DIRECT,
    LOG_PRINT_CURVE_LUT_DIRECT_16384,
    LOG_PRINT_CURVE_LUT_4096,
    LOG_PRINT_CURVE_LUT_8192,
    log_print_curve_engine,
)


def build_main_menus(window) -> None:
    file_menu = window.file_menu = window.menuBar().addMenu("File")
    open_action = QAction("Open RAW / DNG...", window)
    open_action.triggered.connect(window.open_file)
    file_menu.addAction(open_action)

    open_folder_action = QAction("Open Folder...", window)
    open_folder_action.triggered.connect(window.open_folder)
    file_menu.addAction(open_folder_action)
    file_menu.addSeparator()

    export_action = QAction("Export Image...", window)
    export_action.triggered.connect(window.export_current)
    file_menu.addAction(export_action)

    export_completed_action = QAction("Export Completed Images...", window)
    export_completed_action.triggered.connect(window.export_completed)
    file_menu.addAction(export_completed_action)

    export_dir_action = QAction("Set Default Export Directory...", window)
    export_dir_action.triggered.connect(window.set_default_export_directory)
    file_menu.addAction(export_dir_action)
    file_menu.addSeparator()

    save_session_action = QAction("Save Roll Session", window)
    save_session_action.triggered.connect(window.save_roll_session_now)
    file_menu.addAction(save_session_action)
    file_menu.addSeparator()

    exit_action = QAction("Exit", window)
    exit_action.triggered.connect(QApplication.quit)
    file_menu.addAction(exit_action)

    edit_menu = window.edit_menu = window.menuBar().addMenu("Edit")
    invert_action = QAction("Invert Preview", window)
    invert_action.triggered.connect(window.preview_inversion)
    edit_menu.addAction(invert_action)

    reset_action = QAction("Reset Current Image", window)
    reset_action.triggered.connect(window.reset_workspace)
    edit_menu.addAction(reset_action)

    view_menu = window.view_menu = window.menuBar().addMenu("View")
    origin_action = QAction("Origin", window)
    origin_action.triggered.connect(lambda: window.preview_tabs.setCurrentWidget(window.origin_view))
    view_menu.addAction(origin_action)
    preview_action = QAction("Preview", window)
    preview_action.triggered.connect(lambda: window.preview_tabs.setCurrentWidget(window.preview_view))
    view_menu.addAction(preview_action)

    settings_menu = window.settings_menu = window.menuBar().addMenu("Settings")
    window.gpu_preview_action = QAction("GPU Preview Acceleration", window)
    window.gpu_preview_action.setCheckable(True)
    window.gpu_preview_action.setChecked(window._gpu_preview_enabled)
    window.gpu_preview_action.toggled.connect(window.set_gpu_preview_enabled)
    settings_menu.addAction(window.gpu_preview_action)

    window.auto_invert_after_frame_action = QAction("Auto Invert After Frame Change", window)
    window.auto_invert_after_frame_action.setCheckable(True)
    window.auto_invert_after_frame_action.setChecked(window._auto_invert_after_frame_change)
    window.auto_invert_after_frame_action.toggled.connect(window.set_auto_invert_after_frame_change)
    settings_menu.addAction(window.auto_invert_after_frame_action)

    window.auto_frame_new_negatives_action = QAction("Auto Frame New Negatives", window)
    window.auto_frame_new_negatives_action.setCheckable(True)
    window.auto_frame_new_negatives_action.setChecked(window._frame_automation.auto_frame_new_negatives)
    window.auto_frame_new_negatives_action.toggled.connect(window.set_auto_frame_new_negatives)
    settings_menu.addAction(window.auto_frame_new_negatives_action)

    window.auto_preinvert_nearby_action = QAction("Auto Pre-Invert Nearby Frames", window)
    window.auto_preinvert_nearby_action.setCheckable(True)
    window.auto_preinvert_nearby_action.setChecked(window._frame_automation.auto_preinvert_nearby_frames)
    window.auto_preinvert_nearby_action.toggled.connect(window.set_auto_preinvert_nearby_frames)
    settings_menu.addAction(window.auto_preinvert_nearby_action)

    window.roll_session_autosave_action = QAction("Auto Save Roll Session", window)
    window.roll_session_autosave_action.setCheckable(True)
    window.roll_session_autosave_action.setChecked(window._roll_session_autosave)
    window.roll_session_autosave_action.toggled.connect(window.set_roll_session_autosave)
    settings_menu.addAction(window.roll_session_autosave_action)

    preinvert_radius_menu = window.preinvert_radius_menu = settings_menu.addMenu("Auto Pre-Invert Range")
    window.preinvert_radius_group = QActionGroup(window)
    window.preinvert_radius_group.setExclusive(True)
    for radius in (0, 1, 2, 3, 5):
        label = "Disabled" if radius == 0 else f"Previous/Next {radius}"
        action = QAction(label, window)
        action.setCheckable(True)
        action.setData(radius)
        action.setChecked(radius == window._frame_automation.auto_preinvert_radius)
        window.preinvert_radius_group.addAction(action)
        preinvert_radius_menu.addAction(action)
    window.preinvert_radius_group.triggered.connect(window.set_auto_preinvert_radius)
    settings_menu.addSeparator()

    boundary_menu = window.boundary_settings_menu = settings_menu.addMenu("Frame / Analysis Boundaries")

    safe_crop_action = QWidgetAction(window)
    safe_crop_widget = QWidget()
    safe_crop_widget.setObjectName("menuSpinRow")
    safe_crop_layout = QHBoxLayout(safe_crop_widget)
    safe_crop_layout.setContentsMargins(10, 4, 10, 4)
    safe_crop_layout.addWidget(QLabel("Auto Frame Safe Crop"))
    safe_crop_layout.addStretch(1)
    window.auto_frame_inset_spin = QSpinBox()
    window.auto_frame_inset_spin.setObjectName("menuSpinBox")
    window.auto_frame_inset_spin.setRange(0, 8)
    window.auto_frame_inset_spin.setSuffix("%")
    window.auto_frame_inset_spin.setValue(window._auto_frame_inset_percent)
    window.auto_frame_inset_spin.valueChanged.connect(window.set_auto_frame_inset_percent)
    safe_crop_layout.addWidget(window.auto_frame_inset_spin)
    safe_crop_action.setDefaultWidget(safe_crop_widget)
    boundary_menu.addAction(safe_crop_action)

    analysis_boundary_action = QWidgetAction(window)
    analysis_boundary_widget = QWidget()
    analysis_boundary_widget.setObjectName("menuSpinRow")
    analysis_boundary_layout = QHBoxLayout(analysis_boundary_widget)
    analysis_boundary_layout.setContentsMargins(10, 4, 10, 4)
    analysis_boundary_layout.addWidget(QLabel("Invert Analysis Boundary"))
    analysis_boundary_layout.addStretch(1)
    window.analysis_inset_menu_spin = QSpinBox()
    window.analysis_inset_menu_spin.setObjectName("menuSpinBox")
    window.analysis_inset_menu_spin.setRange(0, 20)
    window.analysis_inset_menu_spin.setSuffix("%")
    window.analysis_inset_menu_spin.setValue(window.adjustments.analysis_inset_percent)
    window.analysis_inset_menu_spin.valueChanged.connect(window.set_analysis_inset_percent)
    analysis_boundary_layout.addWidget(window.analysis_inset_menu_spin)
    analysis_boundary_action.setDefaultWidget(analysis_boundary_widget)
    boundary_menu.addAction(analysis_boundary_action)
    settings_menu.addSeparator()

    developer_menu = window.developer_menu = settings_menu.addMenu("Developer")

    window.camera_color_dock = QDockWidget("Camera Color", window)
    window.camera_color_dock.setObjectName("cameraColorDock")
    window.camera_color_dock.setWidget(window.control_panel.camera_color_panel)
    window.camera_color_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    window.addDockWidget(Qt.LeftDockWidgetArea, window.camera_color_dock)
    window.camera_color_dock.hide()

    camera_color_action = QAction("Camera Color", window)
    camera_color_action.setCheckable(True)
    camera_color_action.toggled.connect(window.camera_color_dock.setVisible)
    window.camera_color_dock.visibilityChanged.connect(camera_color_action.setChecked)
    developer_menu.addAction(camera_color_action)

    base_picker_action = QAction("Base Picker Tool", window)
    base_picker_action.triggered.connect(lambda: window.set_tool_mode(ToolMode.MASK_PICKER))
    developer_menu.addAction(base_picker_action)

    developer_menu.addSeparator()
    export_advanced_menu = window.export_advanced_menu = developer_menu.addMenu("Export Advanced")
    print_curve_menu = window.print_curve_menu = export_advanced_menu.addMenu("Print Curve Engine")
    window.print_curve_engine_group = QActionGroup(window)
    window.print_curve_engine_group.setExclusive(True)
    for label, engine in (
        ("LUT 16384 Direct", LOG_PRINT_CURVE_LUT_DIRECT_16384),
        ("LUT 8192", LOG_PRINT_CURVE_LUT_8192),
        ("LUT 4096", LOG_PRINT_CURVE_LUT_4096),
        ("Direct Reference", LOG_PRINT_CURVE_DIRECT),
    ):
        action = QAction(label, window)
        action.setCheckable(True)
        action.setData(engine)
        action.setChecked(engine == log_print_curve_engine())
        window.print_curve_engine_group.addAction(action)
        print_curve_menu.addAction(action)
    window.print_curve_engine_group.triggered.connect(window.set_print_curve_engine)
