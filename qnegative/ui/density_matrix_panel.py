from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.models import AdjustmentParams, DensityMatrixParams


class DensityMatrixPanel(QWidget):
    matrixChanged = Signal()

    FALLBACK_PRESETS: dict[str, DensityMatrixParams] = {
        "Default": DensityMatrixParams(),
        "Fuji C400": DensityMatrixParams(
            m00=0.9654196500778198,
            m01=-0.021040072664618492,
            m02=0.0556204654276371,
            m10=-0.018133480101823807,
            m11=1.0950936079025269,
            m12=-0.07696010917425156,
            m20=0.06534850597381592,
            m21=-0.09767186641693115,
            m22=1.0323233604431152,
        ),
        "Lucky C200": DensityMatrixParams(
            m00=1.0278899669647217,
            m01=-0.03503507375717163,
            m02=0.00714511051774025,
            m10=-0.029770467430353165,
            m11=1.098264455795288,
            m12=-0.06849396228790283,
            m20=0.019769106060266495,
            m21=-0.08775404095649719,
            m22=1.067984938621521,
        ),
        "Identity": DensityMatrixParams(
            m00=1.0,
            m01=0.0,
            m02=0.0,
            m10=0.0,
            m11=1.0,
            m12=0.0,
            m20=0.0,
            m21=0.0,
            m22=1.0,
        ),
    }
    CUSTOM_LABEL = "Custom"

    KEYS = (
        ("m00", "m01", "m02"),
        ("m10", "m11", "m12"),
        ("m20", "m21", "m22"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("densityMatrixPanel")
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._updating = False
        self._presets = self._load_presets()

        self.preset_combo = QComboBox()
        for name in self._presets:
            self.preset_combo.addItem(name)
        self.preset_combo.addItem(self.CUSTOM_LABEL)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self._style_combo_popup(self.preset_combo)

        self.reset_button = QPushButton("Identity")
        self.reset_button.setFixedHeight(28)
        self.reset_button.clicked.connect(self.reset_identity)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(QLabel("Preset"))
        header.addWidget(self.preset_combo, 1)
        header.addWidget(self.reset_button)
        root.addLayout(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for row, keys in enumerate(self.KEYS):
            for column, key in enumerate(keys):
                spin = self._make_spin_box()
                spin.setValue(1.0 if row == column else 0.0)
                spin.valueChanged.connect(self._spin_changed)
                self._spins[key] = spin
                grid.addWidget(spin, row, column)
        root.addLayout(grid)

        hint = QLabel("Applied before density levels.")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self._apply_style()

    def values(self) -> dict:
        return {
            "density_matrix": self._current_matrix_params()
        }

    def set_adjustments(self, adjustments: AdjustmentParams) -> None:
        previous = self.blockSignals(True)
        self._updating = True
        try:
            matrix = adjustments.density_matrix
            self._set_matrix(matrix)
            self._set_matching_preset(matrix)
        finally:
            self._updating = False
            self.blockSignals(previous)

    def reset_identity(self) -> None:
        self._set_preset("Identity", emit=True)

    def _preset_changed(self, _index: int | None = None) -> None:
        if self._updating:
            return
        name = self.preset_combo.currentText()
        if name == self.CUSTOM_LABEL:
            return
        self._set_matrix(self._presets[name])
        self.matrixChanged.emit()

    def _spin_changed(self, _value: float | None = None) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            custom_index = self.preset_combo.findText(self.CUSTOM_LABEL)
            if custom_index >= 0:
                self.preset_combo.setCurrentIndex(custom_index)
        finally:
            self._updating = False
        self.matrixChanged.emit()

    def _set_preset(self, name: str, *, emit: bool) -> None:
        params = self._presets[name]
        self._updating = True
        try:
            index = self.preset_combo.findText(name)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)
            self._set_matrix(params)
        finally:
            self._updating = False
        if emit:
            self.matrixChanged.emit()

    def _set_matrix(self, matrix: DensityMatrixParams) -> None:
        for key, spin in self._spins.items():
            spin.setValue(float(getattr(matrix, key)))

    def _current_matrix_params(self) -> DensityMatrixParams:
        return DensityMatrixParams(**{key: spin.value() for key, spin in self._spins.items()})

    def _set_matching_preset(self, matrix: DensityMatrixParams) -> None:
        for name, preset in self._presets.items():
            if self._matrix_matches(matrix, preset):
                self.preset_combo.setCurrentIndex(self.preset_combo.findText(name))
                return
        self.preset_combo.setCurrentIndex(self.preset_combo.findText(self.CUSTOM_LABEL))

    def _matrix_matches(self, left: DensityMatrixParams, right: DensityMatrixParams) -> bool:
        for keys in self.KEYS:
            for key in keys:
                if abs(float(getattr(left, key)) - float(getattr(right, key))) > 0.0005:
                    return False
        return True

    def _load_presets(self) -> dict[str, DensityMatrixParams]:
        presets = dict(self.FALLBACK_PRESETS)
        presets_dir = Path.cwd() / "presets"
        if not presets_dir.exists():
            return presets

        for path in sorted(presets_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                params = data["density_matrix"]["params"]
                name = self._preset_display_name(str(data.get("name") or path.stem))
                presets[name] = DensityMatrixParams(**{key: float(params[key]) for keys in self.KEYS for key in keys})
            except Exception:
                continue
        return presets

    def _preset_display_name(self, name: str) -> str:
        aliases = {
            "fujiC400": "Fuji C400",
            "LuckyC200": "Lucky C200",
            "luckyC200": "Lucky C200",
        }
        return aliases.get(name, name.replace("_", " "))

    def _make_spin_box(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-2.0, 2.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.01)
        spin.setFixedHeight(28)
        spin.setAlignment(Qt.AlignRight)
        return spin

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

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#densityMatrixPanel {
                color: #e8eaed;
            }
            QLabel {
                color: #e8eaed;
            }
            QLabel#mutedLabel {
                color: #9aa4b2;
                font-size: 12px;
            }
            QDoubleSpinBox {
                background: #15191f;
                border: 1px solid #444c59;
                border-radius: 4px;
                color: #f2f4f7;
                padding: 3px 5px;
            }
            QComboBox {
                background: #15191f;
                border: 1px solid #444c59;
                border-radius: 4px;
                color: #f2f4f7;
                padding: 4px 6px;
            }
            QComboBox QAbstractItemView {
                background: #15191f;
                border: 1px solid #444c59;
                color: #f2f4f7;
                selection-background-color: #41627a;
                selection-color: #ffffff;
                outline: 0;
            }
            QPushButton {
                background: #2d333d;
                border: 1px solid #444c59;
                border-radius: 5px;
                color: #f2f4f7;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background: #38414d;
            }
            """
        )
