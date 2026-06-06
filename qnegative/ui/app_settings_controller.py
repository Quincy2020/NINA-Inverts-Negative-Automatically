from __future__ import annotations

from pathlib import Path

from qnegative.ui.app_settings import AppSettings, load_app_settings, save_app_settings


class AppSettingsController:
    def __init__(self) -> None:
        self.settings = load_app_settings()

    def apply_to_window(self, window) -> None:
        settings = self.settings
        window._frame_automation.set_auto_frame_new_negatives(
            settings.auto_frame_new_negatives
        )
        window._frame_automation.set_auto_preinvert_nearby_frames(
            settings.auto_preinvert_nearby_frames
        )
        window._frame_automation.set_auto_preinvert_radius(settings.auto_preinvert_radius)
        window.adjustments.analysis_inset_percent = settings.analysis_inset_percent
        window._default_export_dir = (
            Path(settings.default_export_dir) if settings.default_export_dir else None
        )
        window._gpu_preview_enabled = settings.gpu_preview_enabled
        window._auto_invert_after_frame_change = settings.auto_invert_after_frame_change
        window._auto_frame_inset_percent = settings.auto_frame_inset_percent
        window._roll_session_autosave = settings.roll_session_autosave

    def save_from_window(self, window) -> None:
        self.settings = AppSettings(
            gpu_preview_enabled=window._gpu_preview_enabled,
            auto_invert_after_frame_change=window._auto_invert_after_frame_change,
            auto_frame_new_negatives=window._frame_automation.auto_frame_new_negatives,
            auto_preinvert_nearby_frames=window._frame_automation.auto_preinvert_nearby_frames,
            auto_preinvert_radius=window._frame_automation.auto_preinvert_radius,
            roll_session_autosave=window._roll_session_autosave,
            auto_frame_inset_percent=window._auto_frame_inset_percent,
            analysis_inset_percent=window.adjustments.analysis_inset_percent,
            default_export_dir=(
                str(window._default_export_dir)
                if window._default_export_dir is not None
                else None
            ),
        )
        save_app_settings(self.settings)
