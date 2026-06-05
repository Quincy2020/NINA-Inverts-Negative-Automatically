from __future__ import annotations

from pathlib import Path


RAW_EXTENSIONS = {".arw", ".raw", ".dng", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_EXTENSIONS = RAW_EXTENSIONS
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | IMAGE_EXTENSIONS
DEFAULT_SEQUENCE_EXTENSIONS = SOURCE_EXTENSIONS
EXCLUDED_OUTPUT_STEM_SUFFIXES = ("_positive", "-positive", " positive")


def is_default_sequence_file(path: Path) -> bool:
    if path.suffix.lower() not in DEFAULT_SEQUENCE_EXTENSIONS:
        return False
    stem = path.stem.lower()
    return not stem.endswith(EXCLUDED_OUTPUT_STEM_SUFFIXES)


def list_supported_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []

    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and is_default_sequence_file(path)
        ),
        key=lambda path: path.name.lower(),
    )
