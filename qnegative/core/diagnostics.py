"""Lightweight runtime diagnostics and crash logging."""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import TextIO

_LOG_LOCK = threading.RLock()
_LOG_FILE: TextIO | None = None
_LOG_PATH: Path | None = None


def enable_crash_logging(log_dir: Path | None = None) -> Path:
    """Enable best-effort logging for Python and native crashes.

    `faulthandler` can dump Python stacks on access violations/segfaults on
    Windows, which is exactly the class of bug we are chasing in auto frame.
    Keep the file handle alive for the entire process.
    """

    global _LOG_FILE, _LOG_PATH
    if _LOG_FILE is not None and _LOG_PATH is not None:
        return _LOG_PATH

    root = log_dir or Path.cwd() / "debug" / "crash_logs"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"nina_{timestamp}_pid{os.getpid()}.log"
    handle = path.open("a", encoding="utf-8", buffering=1)
    _LOG_FILE = handle
    _LOG_PATH = path
    faulthandler.enable(file=handle, all_threads=True)
    _install_excepthook()
    log_event("diagnostics", f"Crash logging enabled: {path}")
    log_event("diagnostics", f"Python {sys.version.replace(chr(10), ' ')}")
    log_event("diagnostics", f"argv={sys.argv!r}")
    return path


def log_event(source: str, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='milliseconds')}] [{threading.current_thread().name}] {source}: {message}\n"
    with _LOG_LOCK:
        if _LOG_FILE is not None:
            _LOG_FILE.write(line)
            _LOG_FILE.flush()
        else:
            try:
                sys.stderr.write(line)
            except Exception:
                pass


def log_exception(source: str, exc: BaseException) -> None:
    with _LOG_LOCK:
        log_event(source, f"{type(exc).__name__}: {exc}")
        if _LOG_FILE is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=_LOG_FILE)
            _LOG_FILE.flush()


def install_qt_message_logger() -> None:
    try:
        from PySide6.QtCore import qInstallMessageHandler
    except Exception:
        return

    def _handler(mode, context, message: str) -> None:  # noqa: ANN001 - Qt callback
        location = ""
        file = getattr(context, "file", None)
        line = getattr(context, "line", 0)
        function = getattr(context, "function", None)
        if file:
            location = f" ({file}:{line} {function or ''})"
        log_event("qt", f"{mode}: {message}{location}")

    qInstallMessageHandler(_handler)
    log_event("diagnostics", "Qt message logger installed")


def _install_excepthook() -> None:
    previous_hook = sys.excepthook

    def _hook(exc_type, exc, tb):  # noqa: ANN001 - sys.excepthook signature
        with _LOG_LOCK:
            if _LOG_FILE is not None:
                traceback.print_exception(exc_type, exc, tb, file=_LOG_FILE)
                _LOG_FILE.flush()
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _hook
