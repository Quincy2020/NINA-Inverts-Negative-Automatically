from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QMenu

from qnegative.ui.folder_filmstrip import FolderFilmstrip


class RollSelectionController(QObject):
    openRequested = Signal(object)
    selectionChanged = Signal(object)
    exportSelectedRequested = Signal(object)
    removeRequested = Signal(object)

    def __init__(self, filmstrip: FolderFilmstrip) -> None:
        super().__init__(filmstrip)
        self._filmstrip = filmstrip
        self._files: list[Path] = []
        self._selected_paths: set[Path] = set()
        self._active_path: Path | None = None
        self._anchor_path: Path | None = None

        filmstrip.itemClicked.connect(self.handle_item_clicked)
        filmstrip.itemDoubleClicked.connect(self.open_path)
        filmstrip.itemContextMenuRequested.connect(self.show_context_menu)

    @property
    def selected_paths(self) -> list[Path]:
        return self._ordered_paths(self._selected_paths)

    def set_files(self, files: list[Path], active_path: Path | None) -> None:
        self._files = list(files)
        file_set = set(self._files)
        self._selected_paths = {path for path in self._selected_paths if path in file_set}
        self._active_path = active_path if active_path in file_set else None
        if self._anchor_path not in file_set:
            self._anchor_path = self._active_path
        self._emit_selection_changed()

    def set_active(self, path: Path | None) -> None:
        if path in self._files:
            self._active_path = path
            if self._anchor_path is None:
                self._anchor_path = path
        elif path is None:
            self._active_path = None

    def handle_item_clicked(self, path: Path, modifiers) -> None:
        if path not in self._files:
            return
        if modifiers & Qt.ControlModifier:
            if path in self._selected_paths:
                self._selected_paths.remove(path)
            else:
                self._selected_paths.add(path)
            self._anchor_path = path
            self._emit_selection_changed()
            return

        if modifiers & Qt.ShiftModifier:
            self._select_range(path)
            self._emit_selection_changed()
            return

        self._active_path = path
        self._anchor_path = path
        self.openRequested.emit(path)

    def open_path(self, path: Path) -> None:
        if path not in self._files:
            return
        self._active_path = path
        self._anchor_path = path
        self.openRequested.emit(path)

    def clear_selection(self) -> None:
        if not self._selected_paths:
            return
        self._selected_paths.clear()
        self._emit_selection_changed()

    def show_context_menu(self, path: Path, global_pos) -> None:
        if path not in self._files:
            return
        targets = self._context_targets(path)
        if not targets:
            return

        menu = QMenu(self._filmstrip)
        open_action = menu.addAction("Open")
        export_label = "Export Selected" if len(targets) > 1 or path in self._selected_paths else "Export This Image"
        export_action = menu.addAction(export_label)
        remove_label = "Remove Selected From Roll" if len(targets) > 1 else "Remove From Roll"
        remove_action = menu.addAction(remove_label)
        menu.addSeparator()
        clear_action = menu.addAction("Clear Selection")
        clear_action.setEnabled(bool(self._selected_paths))

        selected = menu.exec(global_pos)
        if selected == open_action:
            self.open_path(path)
        elif selected == export_action:
            self.exportSelectedRequested.emit(targets)
        elif selected == remove_action:
            self.removeRequested.emit(targets)
        elif selected == clear_action:
            self.clear_selection()

    def _select_range(self, path: Path) -> None:
        anchor = self._anchor_path if self._anchor_path in self._files else self._active_path
        if anchor not in self._files:
            anchor = path
        start = self._files.index(anchor)
        end = self._files.index(path)
        lo, hi = sorted((start, end))
        self._selected_paths.update(self._files[lo : hi + 1])

    def _context_targets(self, path: Path) -> list[Path]:
        if path in self._selected_paths:
            return self._ordered_paths(self._selected_paths)
        return [path]

    def _ordered_paths(self, paths: set[Path]) -> list[Path]:
        selected = set(paths)
        return [path for path in self._files if path in selected]

    def _emit_selection_changed(self) -> None:
        selected = self.selected_paths
        self._filmstrip.set_selected_paths(selected)
        self.selectionChanged.emit(selected)
