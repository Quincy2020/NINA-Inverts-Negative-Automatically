from __future__ import annotations

from pathlib import Path


RAW_EXTENSIONS = {".arw", ".raw", ".dng", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | IMAGE_EXTENSIONS


def list_supported_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []

    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
