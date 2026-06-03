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
    wheelMoved = Signal(int)

    def __init__(self, path: Path, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.index = index
        self._processed = False
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

        self.processed_badge = QLabel("Positive", self)
        self.processed_badge.setObjectName("processedBadge")
        self.processed_badge.setAlignment(Qt.AlignCenter)
        self.processed_badge.setFixedSize(50, 20)
        self.processed_badge.move(54, 8)
        self.processed_badge.hide()

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

    def set_processed_thumbnail(self, pixmap: QPixmap) -> None:
        self.setToolTip("Positive preview generated")
        self.set_thumbnail(pixmap)
        self._processed = True
        self.set_processed_badge(True)

    def set_processed_badge(self, processed: bool) -> None:
        if processed:
            self.setToolTip("Positive preview generated")
            self.processed_badge.show()
            self.processed_badge.raise_()
            return
        self.setToolTip("")
        self.processed_badge.hide()

    def has_processed_thumbnail(self) -> bool:
        return self._processed

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.selected.emit(self.path)
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta:
            self.wheelMoved.emit(delta)
            event.accept()
            return
        super().wheelEvent(event)


class HorizontalWheelScrollArea(QScrollArea):
    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta == 0:
            super().wheelEvent(event)
            return
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - delta)
        event.accept()


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

        self.scroll_area = HorizontalWheelScrollArea()
        self.scroll_area.setObjectName("filmstripScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.content = QWidget()
        self.content.setObjectName("filmstripContent")
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
            item.wheelMoved.connect(self._scroll_by_wheel)
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
            self._center_item(self._items[path])

    def _center_item(self, item: ThumbnailItem) -> None:
        bar = self.scroll_area.horizontalScrollBar()
        viewport_width = max(1, self.scroll_area.viewport().width())
        item_center = item.x() + item.width() // 2
        target = item_center - viewport_width // 2
        bar.setValue(max(bar.minimum(), min(bar.maximum(), target)))

    def _scroll_by_wheel(self, delta: int) -> None:
        bar = self.scroll_area.horizontalScrollBar()
        bar.setValue(bar.value() - delta)

    def set_processed_thumbnail(self, path: Path | None, pixmap: QPixmap) -> None:
        if path is None:
            return
        item = self._items.get(path)
        if item is None:
            return
        item.set_processed_thumbnail(pixmap)

    def set_processed_badge(self, path: Path | None, processed: bool) -> None:
        if path is None:
            return
        item = self._items.get(path)
        if item is None:
            return
        item.set_processed_badge(processed)

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
        if item.has_processed_thumbnail():
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
                background: #1A1A1A;
                border-top: 1px solid #38312A;
            }
            QToolButton {
                background: #2A2520;
                border: 1px solid #4A4034;
                border-radius: 5px;
                color: #F2EEE6;
                font-size: 18px;
            }
            QToolButton:hover {
                background: #342A1D;
            }
            QFrame#thumbnailItem {
                background: #24211E;
                border: 1px solid #3D352D;
                border-radius: 6px;
            }
            QFrame#thumbnailItem[active="true"] {
                border: 3px solid #FFB000;
                background: #2E2418;
            }
            QLabel#thumbnailImage {
                background: #121212;
                border-radius: 4px;
                color: #8C8171;
            }
            QLabel#thumbnailName {
                color: #D8D0C2;
                font-size: 10px;
            }
            QLabel#processedBadge {
                background: #f2c94c;
                border: 1px solid #121212;
                border-radius: 5px;
                color: #121212;
                font-size: 9px;
                font-weight: 700;
            }
            QScrollArea#filmstripScrollArea,
            QScrollArea#filmstripScrollArea QWidget#qt_scrollarea_viewport,
            QWidget#filmstripContent {
                background: #1A1A1A;
            }
            QScrollBar:horizontal {
                background: #1A1A1A;
                border: none;
                height: 10px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #4A4034;
                border-radius: 5px;
                min-width: 32px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #6A5A45;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                background: transparent;
                border: none;
                width: 0;
                height: 0;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: #1A1A1A;
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
