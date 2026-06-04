from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class BatchExportSettings:
    output_dir: Path
    naming_mode: str
    prefix: str
    start_number: int
    export_format: str = "tiff16"
    overwrite_existing: bool = False


class BatchExportSettingsDialog(QDialog):
    def __init__(
        self,
        *,
        default_dir: Path,
        default_prefix: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Export Settings")
        self.setModal(True)
        self.resize(460, 320)

        self.output_dir_edit = QLineEdit(str(default_dir))
        self.browse_button = QPushButton("Browse...")
        self.same_name_radio = QRadioButton("Use original file name")
        self.sequence_radio = QRadioButton("Prefix + 3-digit number")
        self.sequence_radio.setChecked(True)
        self.prefix_edit = QLineEdit(default_prefix)
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 999999)
        self.start_spin.setValue(1)
        self.format_combo = QComboBox()
        self.format_combo.addItem("TIFF (*.tif), 16-bit RGB / 48-bit", "tiff16")
        self.format_combo.addItem("TIFF (*.tif), 8-bit RGB / 24-bit", "tiff8")
        self.format_combo.addItem("JPEG (*.jpg), quality 95", "jpg")
        self.format_combo.addItem("PNG (*.png), 16-bit RGB / 48-bit", "png16")
        self.format_combo.addItem("PNG (*.png), 8-bit RGB / 24-bit", "png8")
        self.overwrite_check = QCheckBox("Overwrite files with the same name")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        location_group = QGroupBox("Location")
        location_layout = QHBoxLayout(location_group)
        location_layout.setSpacing(10)
        location_layout.addWidget(self.output_dir_edit, 1)
        location_layout.addWidget(self.browse_button)
        root.addWidget(location_group)

        name_group = QGroupBox("File Name")
        name_layout = QVBoxLayout(name_group)
        name_layout.setSpacing(8)
        name_layout.addWidget(self.same_name_radio)
        name_layout.addWidget(self.sequence_radio)
        form = QFormLayout()
        form.setSpacing(8)
        form.addRow("Prefix", self.prefix_edit)
        form.addRow("Start Number", self.start_spin)
        name_layout.addLayout(form)
        root.addWidget(name_group)

        format_group = QGroupBox("Image Format")
        format_layout = QFormLayout(format_group)
        format_layout.addRow("Type", self.format_combo)
        format_layout.addRow("Details", QLabel("TIFF/PNG: 8-bit or 16-bit RGB; JPEG: quality 95"))
        root.addWidget(format_group)

        root.addWidget(self.overwrite_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        root.addWidget(buttons)

        self.browse_button.clicked.connect(self._browse)
        self.same_name_radio.toggled.connect(self._refresh_name_controls)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._refresh_name_controls()
        self._apply_style()

    def settings(self) -> BatchExportSettings:
        output_dir = Path(self.output_dir_edit.text()).expanduser()
        prefix = self.prefix_edit.text().strip() or "scan"
        return BatchExportSettings(
            output_dir=output_dir,
            naming_mode="same_name" if self.same_name_radio.isChecked() else "sequence",
            prefix=prefix,
            start_number=int(self.start_spin.value()),
            export_format=str(self.format_combo.currentData() or "tiff16"),
            overwrite_existing=self.overwrite_check.isChecked(),
        )

    def accept(self) -> None:
        settings = self.settings()
        try:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Batch Export Settings", f"Cannot create output folder:\n{exc}")
            return
        super().accept()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Export Folder",
            self.output_dir_edit.text(),
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def _refresh_name_controls(self) -> None:
        sequence = self.sequence_radio.isChecked()
        self.prefix_edit.setEnabled(sequence)
        self.start_spin.setEnabled(sequence)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background: #202020;
                color: #E8E1D5;
            }
            QGroupBox {
                border: 1px solid #443B32;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
                color: #E8E1D5;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #D8D0C2;
            }
            QLabel {
                color: #E8E1D5;
            }
            QLineEdit, QSpinBox, QComboBox {
                background: #1A1A1A;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #F2EEE6;
                padding: 6px 8px;
                selection-background-color: #663300;
                selection-color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border-color: #FFB000;
            }
            QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
                color: #8C8171;
                background: #24211E;
                border-color: #3D352D;
            }
            QComboBox QAbstractItemView {
                background: #1A1A1A;
                border: 1px solid #4A4034;
                color: #F2EEE6;
                selection-background-color: #663300;
                selection-color: #ffffff;
            }
            QRadioButton, QCheckBox {
                color: #E8E1D5;
                spacing: 8px;
            }
            QRadioButton::indicator, QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QRadioButton::indicator:unchecked, QCheckBox::indicator:unchecked {
                background: #1A1A1A;
                border: 1px solid #4A4034;
            }
            QRadioButton::indicator:checked, QCheckBox::indicator:checked {
                background: #FFB000;
                border: 1px solid #FFB000;
            }
            QRadioButton::indicator {
                border-radius: 8px;
            }
            QPushButton {
                background: #2A2520;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #F2EEE6;
                padding: 7px 14px;
                min-width: 72px;
            }
            QPushButton:hover {
                background: #342A1D;
                border-color: #FFB000;
            }
            QPushButton:pressed {
                background: #663300;
            }
            QDialogButtonBox QPushButton {
                min-width: 82px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background: #2A2520;
                border-left: 1px solid #4A4034;
                width: 18px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #342A1D;
            }
            """
        )


class BatchExportDialog(QDialog):
    pauseRequested = Signal()
    resumeRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("batchExportDialog")
        self.setWindowTitle("Batch Export")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setMinimumSize(340, 250)
        self.resize(380, 280)

        self.current_label = QLabel("Waiting")
        self.current_label.setObjectName("batchCurrentLabel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Waiting")
        self.queue = QListWidget()
        self.queue.setObjectName("batchQueue")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.cancel_button = QPushButton("Cancel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(QLabel("Current"))
        layout.addWidget(self.current_label)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("Queue"))
        layout.addWidget(self.queue, 1)
        button_row = QHBoxLayout()
        button_row.addWidget(self.pause_button)
        button_row.addWidget(self.resume_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)
        self.pause_button.clicked.connect(self.pauseRequested.emit)
        self.resume_button.clicked.connect(self.resumeRequested.emit)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.set_running(False)
        self._apply_style()

    def set_jobs(self, paths: list[Path], output_paths: list[Path] | None = None) -> None:
        self.queue.clear()
        for index, path in enumerate(paths):
            output_path = output_paths[index] if output_paths is not None and index < len(output_paths) else None
            item = QListWidgetItem(self._queue_item_text(path, output_path))
            item.setData(Qt.UserRole, str(path))
            item.setData(Qt.UserRole + 1, output_path.name if output_path is not None else "")
            self.queue.addItem(item)
        self.progress.setValue(0)
        self.progress.setFormat("Queued")
        self.current_label.setText("Waiting")
        self.set_running(True)

    def set_current(self, path: Path) -> None:
        self.current_label.setText(path.name)
        for index in range(self.queue.count()):
            item = self.queue.item(index)
            if item.data(Qt.UserRole) == str(path):
                item.setText(f"> {self._queue_item_text(path, self._item_output_name(item))}")
                self.queue.setCurrentRow(index)
            elif not item.text().startswith("Done "):
                source = Path(item.data(Qt.UserRole))
                item.setText(self._queue_item_text(source, self._item_output_name(item)))

    def update_progress(self, value: int, text: str) -> None:
        self.progress.setValue(max(0, min(100, int(value))))
        self.progress.setFormat(text)

    def mark_done(self, path: Path) -> None:
        for index in range(self.queue.count()):
            item = self.queue.item(index)
            if item.data(Qt.UserRole) == str(path):
                item.setText(f"Done {self._queue_item_text(path, self._item_output_name(item))}")
                break

    @staticmethod
    def _queue_item_text(source_path: Path, output_name: str | Path | None) -> str:
        if output_name:
            return f"{source_path.name} -> {Path(output_name).name}"
        return source_path.name

    @staticmethod
    def _item_output_name(item: QListWidgetItem) -> str | None:
        value = item.data(Qt.UserRole + 1)
        return str(value) if value else None

    def finish(self, text: str, *, auto_close_ms: int | None = None) -> None:
        self.current_label.setText(text)
        self.progress.setValue(100)
        self.progress.setFormat(text)
        self.set_running(False)
        if auto_close_ms is not None:
            QTimer.singleShot(auto_close_ms, self.hide)

    def set_running(self, running: bool, *, paused: bool = False) -> None:
        self.pause_button.setEnabled(running and not paused)
        self.resume_button.setEnabled(running and paused)
        self.cancel_button.setEnabled(running)

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog#batchExportDialog {
                background: #202020;
                color: #E8E1D5;
            }
            QLabel {
                color: #E8E1D5;
            }
            QLabel#batchCurrentLabel {
                background: #1A1A1A;
                border: 1px solid #3D352D;
                border-radius: 5px;
                padding: 8px;
                font-weight: 600;
            }
            QListWidget#batchQueue {
                background: #1A1A1A;
                border: 1px solid #3D352D;
                border-radius: 5px;
                color: #D8D0C2;
                outline: 0;
            }
            QListWidget#batchQueue::item {
                padding: 6px;
            }
            QListWidget#batchQueue::item:selected {
                background: #663300;
                color: #ffffff;
            }
            QProgressBar {
                background: #1A1A1A;
                border: 1px solid #3D352D;
                border-radius: 5px;
                color: #E8E1D5;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background: #FFB000;
                border-radius: 4px;
            }
            QPushButton {
                background: #2A2520;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #F2EEE6;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background: #342A1D;
            }
            QPushButton:disabled {
                color: #8C8171;
                background: #24211E;
            }
            """
        )
