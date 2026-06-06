from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QStandardPaths


@dataclass
class AppSettings:
    gpu_preview_enabled: bool = True
    auto_invert_after_frame_change: bool = True
    auto_frame_new_negatives: bool = True
    auto_preinvert_nearby_frames: bool = True
    auto_preinvert_radius: int = 1
    roll_session_autosave: bool = True
    auto_frame_inset_percent: int = 2
    analysis_inset_percent: int = 5
    default_export_dir: str | None = None


def app_settings_path() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    if not base:
        base = str(Path.home() / ".nina")
    return Path(base) / "settings.json"


def load_app_settings() -> AppSettings:
    path = app_settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppSettings()
    if not isinstance(data, dict):
        return AppSettings()

    defaults = AppSettings()
    return AppSettings(
        gpu_preview_enabled=bool(data.get("gpu_preview_enabled", defaults.gpu_preview_enabled)),
        auto_invert_after_frame_change=bool(
            data.get("auto_invert_after_frame_change", defaults.auto_invert_after_frame_change)
        ),
        auto_frame_new_negatives=bool(data.get("auto_frame_new_negatives", defaults.auto_frame_new_negatives)),
        auto_preinvert_nearby_frames=bool(
            data.get("auto_preinvert_nearby_frames", defaults.auto_preinvert_nearby_frames)
        ),
        auto_preinvert_radius=_clamp_int(
            data.get("auto_preinvert_radius", defaults.auto_preinvert_radius),
            0,
            5,
            defaults.auto_preinvert_radius,
        ),
        roll_session_autosave=bool(data.get("roll_session_autosave", defaults.roll_session_autosave)),
        auto_frame_inset_percent=_clamp_int(
            data.get("auto_frame_inset_percent", defaults.auto_frame_inset_percent),
            0,
            8,
            defaults.auto_frame_inset_percent,
        ),
        analysis_inset_percent=_clamp_int(
            data.get("analysis_inset_percent", defaults.analysis_inset_percent),
            0,
            20,
            defaults.analysis_inset_percent,
        ),
        default_export_dir=_path_text_or_none(data.get("default_export_dir")),
    )


def save_app_settings(settings: AppSettings) -> None:
    path = app_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(asdict(settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _clamp_int(value, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def _path_text_or_none(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
