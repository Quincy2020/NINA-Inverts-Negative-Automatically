from __future__ import annotations

from pathlib import Path

import rawpy
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from qnegative.core.file_sequence import RAW_EXTENSIONS


class ThumbnailItem(QFrame):
    selected = Signal(object)

    def __init__(self, path: Path, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.index = index
        self.setObjectName("thumbnailItem")
        self.setProperty("active", False)
        self.setFixedSize(112, 104)
        self.setCursor(Qt.PointingHandCursor)

        self.image_label = QLabel()
        self.image_label.setObjectName("thumbnailImage")
        self.image_label.setFixedSize(96, 64)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("...")

        self.name_label = QLabel(path.name)
        self.name_label.setObjectName("thumbnailName")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setFixedHeight(26)
        self.name_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 5)
        layout.setSpacing(5)
        layout.addWidget(self.image_label)
        layout.addWidget(self.name_label)

    def set_active(self, active: bool) -> None:
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_thumbnail(self, pixmap: QPixmap | None) -> None:
        if pixmap is None or pixmap.isNull():
            self.image_label.setText("No preview")
            return
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.selected.emit(self.path)
        super().mousePressEvent(event)


class FolderFilmstrip(QWidget):
    fileSelected = Signal(object)
    previousRequested = Signal()
    nextRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("folderFilmstrip")
        self.setFixedHeight(132)

        self._items: dict[Path, ThumbnailItem] = {}
        self._load_queue: list[Path] = []
        self._current_path: Path | None = None

        self.previous_button = QToolButton()
        self.previous_button.setText("<")
        self.previous_button.setToolTip("Previous")
        self.previous_button.setFixedSize(34, 92)

        self.next_button = QToolButton()
        self.next_button.setText(">")
        self.next_button.setToolTip("Next")
        self.next_button.setFixedSize(34, 92)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.content = QWidget()
        self.content_layout = QHBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(8)
        self.content_layout.addStretch(1)
        self.scroll_area.setWidget(self.content)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self.previous_button)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(self.next_button)

        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._load_next_thumbnail)

        self.previous_button.clicked.connect(self.previousRequested.emit)
        self.next_button.clicked.connect(self.nextRequested.emit)

        self._apply_style()

    def set_files(self, files: list[Path], current_path: Path | None) -> None:
        self._timer.stop()
        self._load_queue = []
        self._items = {}
        self._current_path = current_path
        self._clear_layout()

        for index, path in enumerate(files, start=1):
            item = ThumbnailItem(path, index)
            item.selected.connect(self.fileSelected.emit)
            item.set_active(current_path is not None and path == current_path)
            self.content_layout.insertWidget(self.content_layout.count() - 1, item)
            self._items[path] = item
            self._load_queue.append(path)

        self._update_buttons()
        if self._load_queue:
            self._timer.start()

    def set_current(self, path: Path | None) -> None:
        self._current_path = path
        for item_path, item in self._items.items():
            item.set_active(path is not None and item_path == path)
        self._update_buttons()
        if path in self._items:
            self.scroll_area.ensureWidgetVisible(self._items[path], 24, 0)

    def _clear_layout(self) -> None:
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _load_next_thumbnail(self) -> None:
        if not self._load_queue:
            self._timer.stop()
            return

        path = self._load_queue.pop(0)
        item = self._items.get(path)
        if item is None:
            return
        item.set_thumbnail(load_thumbnail(path))

    def _update_buttons(self) -> None:
        has_items = bool(self._items)
        self.previous_button.setEnabled(has_items)
        self.next_button.setEnabled(has_items)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            #folderFilmstrip {
                background: #1b1f26;
                border-top: 1px solid #303640;
            }
            QToolButton {
                background: #2d333d;
                border: 1px solid #444c59;
                border-radius: 5px;
                color: #f2f4f7;
                font-size: 18px;
            }
            QToolButton:hover {
                background: #38414d;
            }
            QFrame#thumbnailItem {
                background: #242a33;
                border: 1px solid #343c47;
                border-radius: 6px;
            }
            QFrame#thumbnailItem[active="true"] {
                border: 2px solid #67a4c7;
                background: #2b3944;
            }
            QLabel#thumbnailImage {
                background: #111418;
                border-radius: 4px;
                color: #7d8794;
            }
            QLabel#thumbnailName {
                color: #cbd3dd;
                font-size: 10px;
            }
            QScrollArea {
                background: transparent;
            }
            """
        )


def load_thumbnail(path: Path) -> QPixmap | None:
    if path.suffix.lower() in RAW_EXTENSIONS:
        return _load_raw_thumbnail(path)
    return _load_image_thumbnail(path)


def _load_image_thumbnail(path: Path) -> QPixmap | None:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    reader.setScaledSize(QSize(160, 120))
    image = reader.read()
    if image.isNull():
        return None
    return QPixmap.fromImage(image)


def _load_raw_thumbnail(path: Path) -> QPixmap | None:
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
    except Exception:
        return None

    if thumb.format == rawpy.ThumbFormat.JPEG:
        buffer = QBuffer()
        buffer.setData(QByteArray(thumb.data))
        buffer.open(QIODevice.ReadOnly)
        reader = QImageReader(buffer)
        reader.setScaledSize(QSize(160, 120))
        image = reader.read()
        buffer.close()
        if image.isNull():
            return None
        return QPixmap.fromImage(image)

    if thumb.format == rawpy.ThumbFormat.BITMAP:
        data = thumb.data
        height, width = data.shape[:2]
        image = QImage(
            data.data,
            width,
            height,
            width * 3,
            QImage.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(image)

    return None
