from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class AutoDetectJob:
    job_id: int
    auto_preview: bool


@dataclass(frozen=True)
class AutoDetectCompletion:
    is_current: bool
    auto_preview: bool


@dataclass(frozen=True)
class PreInvertJob:
    job_id: int
    path: Path


class FrameAutomationController:
    def __init__(self) -> None:
        self.auto_frame_new_negatives = True
        self.auto_preinvert_nearby_frames = True
        self.auto_preinvert_radius = 1
        self.model_warmup_in_progress = False

        self._auto_detect_job_id = 0
        self._auto_detect_auto_preview_jobs: set[int] = set()
        self._auto_detect_in_progress = False

        self._preinvert_job_id = 0
        self._preinvert_in_progress: set[int] = set()
        self._preinvert_paths: set[Path] = set()
        self._preinvert_queue: list[Path] = []

    @property
    def auto_detect_in_progress(self) -> bool:
        return self._auto_detect_in_progress

    @property
    def preinvert_in_progress(self) -> bool:
        return bool(self._preinvert_in_progress)

    def set_auto_frame_new_negatives(self, enabled: bool) -> None:
        self.auto_frame_new_negatives = bool(enabled)

    def set_auto_preinvert_nearby_frames(self, enabled: bool) -> None:
        self.auto_preinvert_nearby_frames = bool(enabled)
        if not self.auto_preinvert_nearby_frames:
            self._preinvert_queue.clear()

    def set_auto_preinvert_radius(self, radius: int) -> None:
        self.auto_preinvert_radius = int(radius)
        self._preinvert_queue.clear()

    def begin_model_warmup(self) -> bool:
        if self.model_warmup_in_progress:
            return False
        self.model_warmup_in_progress = True
        return True

    def finish_model_warmup(self) -> None:
        self.model_warmup_in_progress = False

    def begin_auto_detect(self, *, auto_preview: bool) -> AutoDetectJob | None:
        if self._auto_detect_in_progress:
            return None
        self._auto_detect_job_id += 1
        job = AutoDetectJob(job_id=self._auto_detect_job_id, auto_preview=auto_preview)
        if auto_preview:
            self._auto_detect_auto_preview_jobs.add(job.job_id)
        self._auto_detect_in_progress = True
        return job

    def finish_auto_detect(self, job_id: int) -> AutoDetectCompletion:
        if job_id != self._auto_detect_job_id:
            self._auto_detect_auto_preview_jobs.discard(job_id)
            return AutoDetectCompletion(is_current=False, auto_preview=False)
        auto_preview = job_id in self._auto_detect_auto_preview_jobs
        self._auto_detect_auto_preview_jobs.discard(job_id)
        self._auto_detect_in_progress = False
        return AutoDetectCompletion(is_current=True, auto_preview=auto_preview)

    def cancel_auto_detect(self) -> None:
        self._auto_detect_job_id += 1
        self._auto_detect_auto_preview_jobs.clear()
        self._auto_detect_in_progress = False

    def should_auto_frame_new_negative(
        self,
        *,
        path: Path | None,
        has_current_preview: bool,
        has_film_rect: bool,
        negative_preview_active: bool,
        raw_extensions: set[str],
    ) -> bool:
        if not self.auto_frame_new_negatives:
            return False
        if not has_current_preview or path is None:
            return False
        if path.suffix.lower() not in raw_extensions:
            return False
        return not has_film_rect and not negative_preview_active

    def schedule_nearby_preinvert(
        self,
        *,
        folder_files: list[Path],
        current_index: int,
        current_path: Path | None,
        image_states: dict[Path, object],
        preview_result_cache: dict[Path, object],
        raw_extensions: set[str],
    ) -> None:
        if not self.auto_preinvert_nearby_frames:
            return
        if current_index < 0 or not folder_files:
            return

        radius = max(0, min(5, int(self.auto_preinvert_radius)))
        start = max(0, current_index - radius)
        end = min(len(folder_files), current_index + radius + 1)
        candidates = [
            path
            for path in folder_files[start:end]
            if self.should_preinvert_path(
                path,
                current_path=current_path,
                image_states=image_states,
                preview_result_cache=preview_result_cache,
                raw_extensions=raw_extensions,
            )
        ]
        ordered = sorted(
            candidates,
            key=lambda path: abs(folder_files.index(path) - current_index),
        )
        for path in ordered:
            if path not in self._preinvert_queue:
                self._preinvert_queue.append(path)

    def should_preinvert_path(
        self,
        path: Path,
        *,
        current_path: Path | None,
        image_states: dict[Path, object],
        preview_result_cache: dict[Path, object],
        raw_extensions: set[str],
    ) -> bool:
        if path == current_path:
            return False
        if path.suffix.lower() not in raw_extensions:
            return False
        if path in self._preinvert_paths or path in image_states:
            return False
        if path in preview_result_cache:
            return False
        return True

    def start_next_preinvert_jobs(
        self,
        *,
        can_start_path: Callable[[Path], bool],
        export_in_progress: bool,
        max_jobs: int = 2,
    ) -> list[PreInvertJob]:
        if export_in_progress:
            return []

        jobs: list[PreInvertJob] = []
        while self._preinvert_queue and len(self._preinvert_in_progress) < max_jobs:
            path = self._preinvert_queue.pop(0)
            if not can_start_path(path):
                continue
            self._preinvert_job_id += 1
            job = PreInvertJob(job_id=self._preinvert_job_id, path=path)
            self._preinvert_in_progress.add(job.job_id)
            self._preinvert_paths.add(path)
            jobs.append(job)
        return jobs

    def finish_preinvert(self, job_id: int, path: Path) -> bool:
        if job_id not in self._preinvert_in_progress:
            self._preinvert_paths.discard(path)
            return False
        self._preinvert_in_progress.discard(job_id)
        self._preinvert_paths.discard(path)
        return True

    def cancel_preinvert(self) -> None:
        self._preinvert_job_id += 1
        self._preinvert_queue.clear()
        self._preinvert_in_progress.clear()
        self._preinvert_paths.clear()

    def clear_preinvert_queue(self) -> None:
        self._preinvert_queue.clear()
