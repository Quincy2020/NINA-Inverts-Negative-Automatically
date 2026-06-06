from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qnegative.core.file_sequence import list_supported_files
from qnegative.core.models import ImageProcessingState
from qnegative.core.session import (
    load_roll_color_result,
    load_roll_session,
    save_roll_session,
    session_path_for_folder,
)


@dataclass(frozen=True)
class FolderLoadResult:
    files: list[Path]
    current_index: int
    session_folder: Path
    restored_count: int
    roll_color_result: dict | None


class ProjectController:
    def __init__(self) -> None:
        self.folder_files: list[Path] = []
        self.current_index: int = -1
        self.roll_session_folder: Path | None = None

    def load_folder(self, path: Path, image_states: dict[Path, ImageProcessingState]) -> FolderLoadResult:
        folder = path.parent
        self.folder_files = self.supported_files_for_folder(folder)
        self.roll_session_folder = folder
        restored = load_roll_session(folder, self.folder_files)
        if restored:
            image_states.update(restored)
        self.current_index = self.index_for_path(path)
        return FolderLoadResult(
            files=list(self.folder_files),
            current_index=self.current_index,
            session_folder=folder,
            restored_count=len(restored),
            roll_color_result=load_roll_color_result(folder),
        )

    def supported_files_for_folder(self, folder: Path) -> list[Path]:
        return list_supported_files(folder)

    def index_for_path(self, path: Path) -> int:
        try:
            return self.folder_files.index(path)
        except ValueError:
            return -1

    def sync_position(self, path: Path) -> int:
        self.current_index = self.index_for_path(path)
        return self.current_index

    def sequence_status_text(self) -> str:
        if self.current_index >= 0:
            return f"Sequence {self.current_index + 1} / {len(self.folder_files)}"
        return "No sequence"

    def filmstrip_badges(self, image_states: dict[Path, ImageProcessingState]) -> list[tuple[Path, bool]]:
        return [
            (path, bool((state := image_states.get(path)) is not None and state.negative_preview_active))
            for path in self.folder_files
        ]

    def session_folder_for(self, current_path: Path | None) -> Path | None:
        return self.roll_session_folder or (current_path.parent if current_path else None)

    def save_session(
        self,
        *,
        image_states: dict[Path, ImageProcessingState],
        current_path: Path | None,
        roll_color_result: dict | None,
    ) -> Path | None:
        folder = self.session_folder_for(current_path)
        if folder is None:
            return None
        save_roll_session(
            folder,
            image_states,
            self.folder_files,
            roll_color_result=roll_color_result,
        )
        return session_path_for_folder(folder)

    def default_batch_prefix(self, default_dir: Path, current_path: Path | None) -> str:
        if self.roll_session_folder is not None and self.roll_session_folder.name:
            return self.roll_session_folder.name
        if current_path is not None and current_path.parent.name:
            return current_path.parent.name
        return default_dir.name or "scan"
