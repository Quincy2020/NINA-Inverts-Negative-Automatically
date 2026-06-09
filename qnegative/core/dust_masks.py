from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from qnegative.core.models import DustRemovalParams


DUST_MASK_DIR_NAME = ".nina/dust_masks"


def dust_mask_paths_for_source(source_path: Path) -> tuple[Path, Path, Path]:
    mask_dir = source_path.parent / DUST_MASK_DIR_NAME
    safe_stem = source_path.stem.replace("/", "_").replace("\\", "_")
    return (
        mask_dir / f"{safe_stem}_dust_auto.png",
        mask_dir / f"{safe_stem}_dust_add.png",
        mask_dir / f"{safe_stem}_dust_protect.png",
    )


def stored_mask_path(source_path: Path, path: Path) -> str:
    try:
        return path.relative_to(source_path.parent).as_posix()
    except ValueError:
        return str(path)


def resolve_stored_mask_path(source_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return source_path.parent / path


def load_dust_mask(
    source_path: Path,
    stored_path: str | None,
    *,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    path = resolve_stored_mask_path(source_path, stored_path)
    if path is None or not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    mask = image > 0
    if target_shape is not None:
        mask = resize_mask(mask, target_shape)
    return mask.astype(bool, copy=False)


def save_dust_mask(path: Path, mask: np.ndarray | None) -> bool:
    if mask is None or not np.any(mask):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (mask.astype(np.uint8) * 255)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Could not write dust mask: {path}")
    return True


def resize_mask(mask: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    height, width = int(target_shape[0]), int(target_shape[1])
    if height <= 0 or width <= 0:
        return None
    clean = np.asarray(mask).astype(bool)
    if clean.shape == (height, width):
        return clean
    resized = cv2.resize(
        clean.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def compose_dust_masks(
    auto_mask: np.ndarray | None,
    manual_add_mask: np.ndarray | None,
    manual_protect_mask: np.ndarray | None,
    *,
    target_shape: tuple[int, int],
) -> np.ndarray:
    height, width = int(target_shape[0]), int(target_shape[1])
    final = np.zeros((height, width), dtype=bool)
    auto = resize_mask(auto_mask, (height, width))
    add = resize_mask(manual_add_mask, (height, width))
    protect = resize_mask(manual_protect_mask, (height, width))
    if auto is not None:
        final |= auto
    if add is not None:
        final |= add
    if protect is not None:
        final &= ~protect
    return final


def dust_auto_mask_params_key(params: DustRemovalParams) -> str:
    payload = {
        "model_id": params.model_id,
        "model_path": params.model_path,
        "threshold": int(params.threshold),
        "adaptive": bool(params.adaptive),
        "texture_penalty": int(params.texture_penalty),
        "max_threshold": int(params.max_threshold),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
