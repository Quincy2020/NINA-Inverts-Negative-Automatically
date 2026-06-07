from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any

import numpy as np
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from qnegative.core.file_sequence import RAW_EXTENSIONS
from qnegative.core.models import ImageProcessingState
from qnegative.ui.export_dialogs import (
    BatchExportDialog,
    BatchExportSettings,
    BatchExportSettingsDialog,
)
from qnegative.ui.export_tasks import (
    ImageExportTask,
    export_format_extension,
    export_format_from_filter,
    export_format_from_path,
    export_format_label,
)
from qnegative.ui.white_balance_controller import WhiteBalanceController


def export_timing_is_top_level(name: str) -> bool:
    return (
        name in {"RAW decode", "Build base", "Lab Print"}
        or name.startswith("Prepare ")
        or name.endswith(" write")
    )


class ExportController:
    """Owns single-image and batch export state.

    MainWindow still provides the active image state and the visible widgets, but
    queue state, export task wiring, and batch item construction live here so
    export behavior does not keep spreading through the main window.
    """

    def __init__(
        self,
        owner: Any,
        *,
        thread_pool: QThreadPool,
        control_panel: Any,
        batch_dialog: BatchExportDialog,
    ) -> None:
        self.owner = owner
        self.thread_pool = thread_pool
        self.control_panel = control_panel
        self.batch_dialog = batch_dialog

        self._export_in_progress = False
        self._batch_export_queue: list[dict] = []
        self._batch_export_total = 0
        self._batch_export_done = 0
        self._batch_export_active = False
        self._batch_export_paused = False
        self._batch_export_cancel_requested = False
        self._batch_export_current_path: Path | None = None
        self._export_cancel_event: Event | None = None

    @property
    def export_in_progress(self) -> bool:
        return self._export_in_progress

    def export_current(self) -> None:
        window = self.owner
        if self._export_in_progress:
            window.statusBar().showMessage("Export already in progress")
            return

        if window.current_path is None or window.current_path.suffix.lower() not in RAW_EXTENSIONS:
            QMessageBox.information(window, "Export Image", "Open a RAW file before exporting.")
            return
        if window._film_base_required_for_current_mode() and window.mask_point is None:
            QMessageBox.information(window, "Export Image", "Pick the film base before exporting.")
            return
        if window.film_rect is None or not window.film_rect.is_valid():
            QMessageBox.information(window, "Export Image", "Select a valid frame area before exporting.")
            return

        if window._default_export_dir is not None:
            default_path = window._default_export_dir / f"{window.current_path.stem}_positive.tif"
        else:
            default_path = window.current_path.with_name(f"{window.current_path.stem}_positive.tif")
        output, selected_filter = QFileDialog.getSaveFileName(
            window,
            "Export Image",
            str(default_path),
            "TIFF 16-bit RGB (*.tif *.tiff);;TIFF 8-bit RGB (*.tif *.tiff);;JPEG Image (*.jpg *.jpeg);;PNG 16-bit RGB (*.png);;PNG 8-bit RGB (*.png)",
        )
        if not output:
            return

        output_path = Path(output)
        known_suffix = output_path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        export_format = (
            export_format_from_path(output_path)
            if known_suffix
            else export_format_from_filter(selected_filter) or "tiff16"
        )
        if not known_suffix:
            output_path = output_path.with_suffix(export_format_extension(export_format))
        window._default_export_dir = output_path.parent
        window._save_app_settings()

        self._export_cancel_event = Event()
        preview_log_floors, preview_log_ceils = self._current_preview_log_bounds_for_export()
        task = ImageExportTask(
            source_path=window.current_path,
            output_path=output_path,
            mask_point=window.mask_point,
            film_rect=window.film_rect,
            adjustments=window.adjustments,
            flip_horizontal=window._preview_flip_horizontal,
            flip_vertical=window._preview_flip_vertical,
            rotation_quarters=window._preview_rotation_quarters,
            auto_levels_pending=window.auto_levels_pending,
            export_format=export_format,
            preview_cmy_offsets=self._current_preview_cmy_offsets_for_export(),
            preview_log_floors=preview_log_floors,
            preview_log_ceils=preview_log_ceils,
            preview_tone_mid_anchor=self._current_preview_tone_mid_anchor_for_export(),
            roll_color_result=window._roll_color_result,
            roll_color_frame=window._roll_color_frame_for_path(window.current_path),
            cancel_event=self._export_cancel_event,
        )
        self._wire_task(task)
        self._export_in_progress = True
        self.control_panel.export_button.setEnabled(False)
        self.control_panel.batch_export_button.setEnabled(False)
        self.control_panel.set_export_progress(True, value=0, text="Starting export")
        window.statusBar().showMessage(f"Exporting {export_format_label(export_format)}...")
        self.thread_pool.start(task)

    def export_completed(self) -> None:
        window = self.owner
        if self._export_in_progress:
            window.statusBar().showMessage("Export already in progress")
            return
        if window.current_path is not None:
            window._save_current_state()

        default_dir = window._default_export_dir or (
            window.current_path.parent if window.current_path else Path.cwd()
        )
        settings_dialog = BatchExportSettingsDialog(
            default_dir=default_dir,
            default_prefix=self._default_batch_prefix(default_dir),
            parent=window,
        )
        if settings_dialog.exec() != QDialog.Accepted:
            return

        settings = settings_dialog.settings()
        window._default_export_dir = settings.output_dir
        window._save_app_settings()
        items = self._completed_export_items(settings)
        if not items:
            QMessageBox.information(
                window,
                "Export Completed Images",
                "No completed RAW positives were found in the current sequence.",
            )
            return

        self._start_batch_export_items(items, f"Exporting {len(items)} completed images...")

    def export_selected(self, paths=None) -> None:
        window = self.owner
        if self._export_in_progress:
            window.statusBar().showMessage("Export already in progress")
            return
        selected_paths = self.selected_export_paths(paths)
        if not selected_paths:
            QMessageBox.information(
                window,
                "Export Selected Images",
                "Select one or more images in the filmstrip first.",
            )
            return
        if window.current_path is not None:
            window._save_current_state()

        completed_paths = self.completed_export_source_paths(selected_paths)
        if not completed_paths:
            QMessageBox.information(
                window,
                "Export Selected Images",
                "None of the selected images have completed RAW positives ready to export.",
            )
            return

        incomplete_count = len(selected_paths) - len(completed_paths)
        if incomplete_count:
            incomplete_message = (
                "1 selected image is not completed. Export completed only?"
                if incomplete_count == 1
                else f"{incomplete_count} selected images are not completed. Export completed only?"
            )
            answer = QMessageBox.question(
                window,
                "Export Selected Images",
                incomplete_message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return

        default_dir = window._default_export_dir or (
            window.current_path.parent if window.current_path else selected_paths[0].parent
        )
        settings_dialog = BatchExportSettingsDialog(
            default_dir=default_dir,
            default_prefix=self._default_batch_prefix(default_dir),
            parent=window,
        )
        if settings_dialog.exec() != QDialog.Accepted:
            return

        settings = settings_dialog.settings()
        window._default_export_dir = settings.output_dir
        window._save_app_settings()
        items = self._completed_export_items(settings, source_paths=completed_paths)
        if not items:
            QMessageBox.information(
                window,
                "Export Selected Images",
                "None of the selected images have completed RAW positives ready to export.",
            )
            return

        self._start_batch_export_items(items, f"Exporting {len(items)} selected images...")

    def selected_export_paths(self, paths) -> list[Path]:
        if isinstance(paths, bool) or paths is None:
            return self.owner.roll_selection.selected_paths
        selected: list[Path] = []
        for path in paths:
            candidate = Path(path)
            if candidate not in selected:
                selected.append(candidate)
        return selected

    def completed_export_source_paths(self, source_paths: list[Path]) -> list[Path]:
        completed: list[Path] = []
        for path in source_paths:
            state = self.owner.image_states.get(path)
            if state is None or not state.negative_preview_active:
                continue
            if path.suffix.lower() not in RAW_EXTENSIONS:
                continue
            if state.film_rect is None or not state.film_rect.is_valid():
                continue
            completed.append(path)
        return completed

    def pause_batch_export(self) -> None:
        if not self._batch_export_active:
            return
        self._batch_export_paused = True
        self.batch_dialog.set_running(True, paused=True)
        self.owner.statusBar().showMessage("Batch export will pause after the current image")

    def resume_batch_export(self) -> None:
        if not self._batch_export_active or not self._batch_export_paused:
            return
        self._batch_export_paused = False
        self.batch_dialog.set_running(True, paused=False)
        self.owner.statusBar().showMessage("Batch export resumed")
        if self._batch_export_current_path is None:
            self._start_next_batch_export()

    def cancel_batch_export(self) -> None:
        if not self._batch_export_active and not self._export_in_progress:
            return
        self._batch_export_cancel_requested = True
        self._batch_export_paused = False
        self._batch_export_queue = []
        if self._export_cancel_event is not None:
            self._export_cancel_event.set()
        self.batch_dialog.set_running(False)
        self.batch_dialog.update_progress(
            round(self._batch_export_done / max(1, self._batch_export_total) * 100),
            "Cancelling...",
        )
        self.owner.statusBar().showMessage("Cancelling export...")
        if self._batch_export_current_path is None:
            self._finish_batch_export(
                f"Batch export cancelled after {self._batch_export_done}/{self._batch_export_total}",
                auto_close_ms=None,
                restart_preinvert=True,
            )

    def cancel_all(self) -> None:
        self._batch_export_queue = []
        self._batch_export_cancel_requested = True
        if self._export_cancel_event is not None:
            self._export_cancel_event.set()

    def _start_batch_export_items(self, items: list[dict], status: str) -> None:
        self._batch_export_queue = items
        self._batch_export_total = len(items)
        self._batch_export_done = 0
        self._batch_export_active = True
        self._batch_export_paused = False
        self._batch_export_cancel_requested = False
        self._export_in_progress = True
        self._export_cancel_event = Event()
        self.batch_dialog.set_jobs(
            [item["source_path"] for item in items],
            [item["output_path"] for item in items],
        )
        self.batch_dialog.show()
        self.batch_dialog.raise_()
        self.control_panel.export_button.setEnabled(False)
        self.control_panel.batch_export_button.setEnabled(False)
        self.control_panel.set_export_progress(True, value=0, text="Starting batch export")
        self.owner.statusBar().showMessage(status)
        self._start_next_batch_export()

    def _completed_export_items(
        self,
        settings: BatchExportSettings,
        *,
        source_paths: list[Path] | None = None,
    ) -> list[dict]:
        window = self.owner
        if source_paths is not None:
            ordered_paths = list(source_paths)
        else:
            ordered_paths = list(window.folder_files) if window.folder_files else list(window.image_states)
            for path in window.image_states:
                if path not in ordered_paths:
                    ordered_paths.append(path)

        items: list[dict] = []
        sequence_index = int(settings.start_number)
        for path in ordered_paths:
            state = window.image_states.get(path)
            if state is None or not state.negative_preview_active:
                continue
            if path.suffix.lower() not in RAW_EXTENSIONS:
                continue
            if state.film_rect is None or not state.film_rect.is_valid():
                continue
            output_path = self._batch_export_output_path(
                path,
                settings,
                sequence_index=sequence_index,
            )
            if settings.naming_mode == "sequence":
                sequence_index += 1
            preview_log_floors, preview_log_ceils = self._preview_log_bounds_for_path(path, state)
            items.append(
                {
                    "source_path": path,
                    "output_path": output_path,
                    "mask_point": state.mask_point,
                    "film_rect": state.film_rect,
                    "adjustments": deepcopy(state.adjustments),
                    "flip_horizontal": state.preview_flip_horizontal,
                    "flip_vertical": state.preview_flip_vertical,
                    "rotation_quarters": state.preview_rotation_quarters,
                    "auto_levels_pending": state.auto_levels_pending,
                    "export_format": settings.export_format,
                    "preview_cmy_offsets": self._preview_cmy_offsets_for_path(path, state),
                    "preview_log_floors": preview_log_floors,
                    "preview_log_ceils": preview_log_ceils,
                    "preview_tone_mid_anchor": self._preview_tone_mid_anchor_for_path(path, state),
                    "roll_color_result": window._roll_color_result,
                    "roll_color_frame": deepcopy(state.roll_color_frame),
                }
            )
        return items

    def _default_batch_prefix(self, default_dir: Path) -> str:
        return self.owner._project.default_batch_prefix(default_dir, self.owner.current_path)

    def _batch_export_output_path(
        self,
        source_path: Path,
        settings: BatchExportSettings,
        *,
        sequence_index: int,
    ) -> Path:
        if settings.naming_mode == "same_name":
            filename = f"{source_path.stem}{export_format_extension(settings.export_format)}"
        else:
            filename = f"{settings.prefix}{sequence_index:03d}{export_format_extension(settings.export_format)}"
        output_path = settings.output_dir / filename
        if settings.overwrite_existing:
            return output_path
        return self._non_conflicting_output_path(output_path)

    @staticmethod
    def _non_conflicting_output_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for index in range(1, 10000):
            candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
            if not candidate.exists():
                return candidate
        return path.with_name(f"{stem}_{int(perf_counter() * 1000)}{suffix}")

    def _preview_cmy_offsets_for_path(self, path: Path, state: ImageProcessingState) -> np.ndarray | None:
        return WhiteBalanceController.cmy_offsets_for_state(
            state,
            self.owner.preview_result_cache.get(path),
        )

    def _preview_tone_mid_anchor_for_path(self, path: Path, state: ImageProcessingState) -> float | None:
        if state.tone_mid_anchor is not None:
            return float(state.tone_mid_anchor)
        cached = self.owner.preview_result_cache.get(path)
        if cached is None:
            return None
        return float(cached.result.tone_mid_anchor)

    def _preview_log_bounds_for_path(
        self,
        path: Path,
        state: ImageProcessingState,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if state.lab_print_log_floors is not None and state.lab_print_log_ceils is not None:
            return (
                np.asarray(state.lab_print_log_floors, dtype=np.float32).copy(),
                np.asarray(state.lab_print_log_ceils, dtype=np.float32).copy(),
            )
        cached = self.owner.preview_result_cache.get(path)
        if cached is None:
            return None, None
        return (
            np.asarray(cached.result.lab_print_log_floors, dtype=np.float32).copy(),
            np.asarray(cached.result.lab_print_log_ceils, dtype=np.float32).copy(),
        )

    def _start_next_batch_export(self) -> None:
        if self._batch_export_cancel_requested:
            self._finish_batch_export(
                f"Batch export cancelled after {self._batch_export_done}/{self._batch_export_total}",
                auto_close_ms=None,
                restart_preinvert=True,
            )
            return
        if self._batch_export_paused:
            self._batch_export_current_path = None
            self.batch_dialog.current_label.setText("Paused")
            self.batch_dialog.update_progress(
                round(self._batch_export_done / max(1, self._batch_export_total) * 100),
                f"Paused after {self._batch_export_done}/{self._batch_export_total}",
            )
            self.batch_dialog.set_running(True, paused=True)
            self.control_panel.update_export_progress(
                round(self._batch_export_done / max(1, self._batch_export_total) * 100),
                "Batch export paused",
            )
            self.owner.statusBar().showMessage("Batch export paused")
            return
        if not self._batch_export_queue:
            self._finish_batch_export(
                f"Batch exported {self._batch_export_done} images",
                auto_close_ms=1200,
                restart_preinvert=True,
            )
            return

        item = self._batch_export_queue.pop(0)
        self._batch_export_current_path = item["source_path"]
        self.batch_dialog.set_current(self._batch_export_current_path)
        self.batch_dialog.set_running(True, paused=False)
        self._export_cancel_event = Event()
        item = dict(item)
        item["cancel_event"] = self._export_cancel_event
        task = ImageExportTask(**item)
        self._wire_task(task)
        self.thread_pool.start(task)

    def _wire_task(self, task: ImageExportTask) -> None:
        task.signals.finished.connect(self._export_finished)
        task.signals.failed.connect(self._export_failed)
        task.signals.cancelled.connect(self._export_cancelled)
        task.signals.progress.connect(self._export_progress_updated)

    def _finish_batch_export(
        self,
        text: str,
        *,
        auto_close_ms: int | None,
        restart_preinvert: bool,
    ) -> None:
        self._batch_export_active = False
        self._batch_export_paused = False
        self._batch_export_cancel_requested = False
        self._export_in_progress = False
        self._batch_export_current_path = None
        self._export_cancel_event = None
        self.control_panel.update_export_progress(100, text)
        self.control_panel.set_export_progress(False)
        self.control_panel.export_button.setEnabled(True)
        self.control_panel.batch_export_button.setEnabled(True)
        self.batch_dialog.finish(text, auto_close_ms=auto_close_ms)
        self.owner.statusBar().showMessage(text)
        if restart_preinvert:
            self.owner._start_next_preinvert_jobs()

    def _current_preview_cmy_offsets_for_export(self) -> np.ndarray | None:
        window = self.owner
        if window.current_path is None:
            return None
        if not window.adjustments.auto_wb:
            return None
        if window.auto_levels_pending:
            return None
        current_cmy = window.white_balance.current_cmy_offsets(window.adjustments)
        if current_cmy is not None:
            return current_cmy

        # Reuse preview CMY only when the final preview cache exactly matches
        # the current image/selection/adjustments. Otherwise export recomputes
        # auto WB at full resolution instead of risking stale color timing.
        cached = window.preview_result_cache.get(window.current_path)
        key = window._preview_result_cache_key()
        if cached is None or key is None or cached.key != key:
            return None
        return np.asarray(cached.result.wb_gains, dtype=np.float32).copy()

    def _current_preview_tone_mid_anchor_for_export(self) -> float | None:
        window = self.owner
        if window.current_path is None or window.auto_levels_pending:
            return None
        if window._last_untransformed_negative_result is not None:
            return float(window._last_untransformed_negative_result.tone_mid_anchor)

        cached = window.preview_result_cache.get(window.current_path)
        key = window._preview_result_cache_key()
        if cached is None or key is None or cached.key != key:
            return None
        return float(cached.result.tone_mid_anchor)

    def _current_preview_log_bounds_for_export(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        window = self.owner
        if window.current_path is None:
            return None, None
        if window._last_untransformed_negative_result is not None:
            return (
                np.asarray(
                    window._last_untransformed_negative_result.lab_print_log_floors,
                    dtype=np.float32,
                ).copy(),
                np.asarray(
                    window._last_untransformed_negative_result.lab_print_log_ceils,
                    dtype=np.float32,
                ).copy(),
            )

        cached = window.preview_result_cache.get(window.current_path)
        key = window._preview_result_cache_key()
        if cached is not None and key is not None and cached.key == key:
            return (
                np.asarray(cached.result.lab_print_log_floors, dtype=np.float32).copy(),
                np.asarray(cached.result.lab_print_log_ceils, dtype=np.float32).copy(),
            )

        state = window.image_states.get(window.current_path)
        if (
            state is not None
            and state.lab_print_log_floors is not None
            and state.lab_print_log_ceils is not None
        ):
            return (
                np.asarray(state.lab_print_log_floors, dtype=np.float32).copy(),
                np.asarray(state.lab_print_log_ceils, dtype=np.float32).copy(),
            )
        return None, None

    def _export_progress_updated(self, value: int, text: str) -> None:
        if self._batch_export_active and self._batch_export_total:
            text = f"Batch {self._batch_export_done + 1}/{self._batch_export_total}: {text}"
            self.batch_dialog.update_progress(value, text)
        self.control_panel.update_export_progress(value, text)
        self.owner.statusBar().showMessage(f"{text}...")

    def _export_finished(self, output_path: str, timings: dict[str, float]) -> None:
        if self._batch_export_active:
            self._batch_export_done += 1
            if self._batch_export_current_path is not None:
                self.batch_dialog.mark_done(self._batch_export_current_path)
            timing_text = self._format_export_timings(timings)
            if timing_text:
                print(f"Export timings: {Path(output_path).name}: {timing_text}", flush=True)
            progress = round(self._batch_export_done / max(1, self._batch_export_total) * 100)
            self.control_panel.update_export_progress(
                progress,
                f"Batch exported {self._batch_export_done}/{self._batch_export_total}",
            )
            self.owner.statusBar().showMessage(
                f"Batch exported {self._batch_export_done}/{self._batch_export_total}: {Path(output_path).name}"
            )
            self._start_next_batch_export()
            return

        self._export_in_progress = False
        self._export_cancel_event = None
        timing_text = self._format_export_timings(timings)
        complete_text = f"Export complete ({timing_text})" if timing_text else "Export complete"
        self.control_panel.update_export_progress(100, complete_text)
        self.control_panel.set_export_progress(False)
        self.control_panel.export_button.setEnabled(True)
        self.control_panel.batch_export_button.setEnabled(True)
        suffix = f" | {timing_text}" if timing_text else ""
        self.owner.statusBar().showMessage(f"Exported image: {output_path}{suffix}")
        if timing_text:
            print(f"Export timings: {timing_text}", flush=True)
        self.owner._start_next_preinvert_jobs()

    @staticmethod
    def _format_export_timings(timings: dict[str, float]) -> str:
        if not timings:
            return ""
        total = sum(
            float(seconds)
            for name, seconds in timings.items()
            if export_timing_is_top_level(name)
        )
        parts = [f"{name} {float(seconds):.1f}s" for name, seconds in timings.items()]
        parts.append(f"total {total:.1f}s")
        return ", ".join(parts)

    def _export_failed(self, message: str) -> None:
        if self._batch_export_active:
            self._batch_export_queue = []
            self._finish_batch_export(
                f"Batch export failed after {self._batch_export_done}/{self._batch_export_total}",
                auto_close_ms=None,
                restart_preinvert=True,
            )
        else:
            self._export_in_progress = False
            self._export_cancel_event = None
            self.control_panel.set_export_progress(False)
            self.control_panel.export_button.setEnabled(True)
            self.control_panel.batch_export_button.setEnabled(True)
            self.owner._start_next_preinvert_jobs()
        QMessageBox.warning(self.owner, "Export Failed", message)
        self.owner.statusBar().showMessage("Export failed")

    def _export_cancelled(self, message: str) -> None:
        if self._batch_export_active:
            self._finish_batch_export(
                f"Batch export cancelled after {self._batch_export_done}/{self._batch_export_total}",
                auto_close_ms=None,
                restart_preinvert=True,
            )
            return
        self._export_in_progress = False
        self._export_cancel_event = None
        self.control_panel.set_export_progress(False)
        self.control_panel.export_button.setEnabled(True)
        self.control_panel.batch_export_button.setEnabled(True)
        self.owner.statusBar().showMessage(message or "Export cancelled")
        self.owner._start_next_preinvert_jobs()
