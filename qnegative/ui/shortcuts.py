from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget


def install_main_window_shortcuts(window: QWidget) -> list[QShortcut]:
    shortcuts: list[QShortcut] = []

    def add(sequence: str | Qt.Key, callback: Callable[[], None]) -> None:
        shortcut = QShortcut(QKeySequence(sequence), window)
        shortcut.setContext(Qt.WindowShortcut)
        shortcut.activated.connect(callback)
        shortcuts.append(shortcut)

    add(Qt.Key_Tab, window.toggle_preview_tab)
    add(Qt.Key_I, window.preview_inversion)
    add(Qt.Key_K, lambda: window.auto_detect_current("frame_base"))
    add("Ctrl+Z", window.undo_adjustments)
    add("Ctrl+Y", window.redo_adjustments)
    add("Ctrl+Shift+Z", window.redo_adjustments)

    for sequence, callback in (
        ("Q", lambda: window._nudge_global_balance("blue_yellow", -20)),
        ("A", lambda: window._nudge_global_balance("blue_yellow", 20)),
        ("Shift+Q", lambda: window._nudge_global_balance("blue_yellow", -5)),
        ("Shift+A", lambda: window._nudge_global_balance("blue_yellow", 5)),
        ("W", lambda: window._nudge_global_balance("green_magenta", -20)),
        ("S", lambda: window._nudge_global_balance("green_magenta", 20)),
        ("Shift+W", lambda: window._nudge_global_balance("green_magenta", -5)),
        ("Shift+S", lambda: window._nudge_global_balance("green_magenta", 5)),
        ("E", lambda: window._nudge_global_balance("red_cyan", -20)),
        ("D", lambda: window._nudge_global_balance("red_cyan", 20)),
        ("Shift+E", lambda: window._nudge_global_balance("red_cyan", -5)),
        ("Shift+D", lambda: window._nudge_global_balance("red_cyan", 5)),
        ("R", lambda: window._nudge_exposure(20)),
        ("F", lambda: window._nudge_exposure(-20)),
        ("Shift+R", lambda: window._nudge_exposure(5)),
        ("Shift+F", lambda: window._nudge_exposure(-5)),
        ("[", lambda: window._nudge_mid_point(-5)),
        ("]", lambda: window._nudge_mid_point(5)),
        ("Shift+[", lambda: window._nudge_mid_point(-1)),
        ("Shift+]", lambda: window._nudge_mid_point(1)),
        ("Left", window.go_previous_file),
        ("Right", window.go_next_file),
        ("Space", window.confirm_current_and_go_next),
        ("Return", window.confirm_current_and_go_next),
        ("Enter", window.confirm_current_and_go_next),
    ):
        add(sequence, callback)

    return shortcuts
