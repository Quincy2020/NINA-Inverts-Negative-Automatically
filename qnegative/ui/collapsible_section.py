from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QFrame):
    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("collapsibleSection")
        self.content = content

        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("collapsibleHeader")
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.clicked.connect(self.set_expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)

        self.set_expanded(expanded)
        self._apply_style()

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QFrame#collapsibleSection {
                background: transparent;
            }
            QToolButton#collapsibleHeader {
                background: #2A2520;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #F2EEE6;
                padding: 7px 8px;
                text-align: left;
            }
            QToolButton#collapsibleHeader:hover {
                background: #342A1D;
            }
            """
        )
