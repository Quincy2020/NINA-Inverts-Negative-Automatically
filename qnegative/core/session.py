from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from qnegative.core.models import (
    AdjustmentParams,
    BalanceAxis,
    ColorBalanceParams,
    DensityMatrixParams,
    ImagePoint,
    ImageProcessingState,
    ImageRect,
    LensCorrectionParams,
    TonalBalance,
)


SESSION_DIR_NAME = ".nina"
SESSION_FILE_NAME = "roll_session.json"
SESSION_SCHEMA_VERSION = 1


def session_path_for_folder(folder: Path) -> Path:
    return folder / SESSION_DIR_NAME / SESSION_FILE_NAME


def state_to_json_dict(state: ImageProcessingState, source_path: Path) -> dict[str, Any]:
    stat = _file_stat(source_path)
    payload = _dataclass_to_dict(state)
    payload["file_size"] = stat["file_size"]
    payload["mtime_ns"] = stat["mtime_ns"]
    return payload


def state_from_json_dict(payload: dict[str, Any], source_path: Path) -> ImageProcessingState | None:
    stat = _file_stat(source_path)
    if payload.get("file_size") != stat["file_size"]:
        return None
    if payload.get("mtime_ns") != stat["mtime_ns"]:
        return None

    try:
        return ImageProcessingState(
            mask_point=_point_from_dict(payload.get("mask_point")),
            film_rect=_rect_from_dict(payload.get("film_rect")),
            white_balance_point=_point_from_dict(payload.get("white_balance_point")),
            adjustments=_adjustments_from_dict(payload.get("adjustments") or {}),
            negative_preview_active=bool(payload.get("negative_preview_active", False)),
            auto_levels_pending=bool(payload.get("auto_levels_pending", True)),
            preview_flip_horizontal=bool(payload.get("preview_flip_horizontal", False)),
            preview_flip_vertical=bool(payload.get("preview_flip_vertical", False)),
            preview_rotation_quarters=int(payload.get("preview_rotation_quarters", 0)) % 4,
        )
    except (TypeError, ValueError):
        return None


def load_roll_session(folder: Path, files: list[Path]) -> dict[Path, ImageProcessingState]:
    path = session_path_for_folder(folder)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
        return {}

    file_by_name = {item.name: item for item in files}
    states: dict[Path, ImageProcessingState] = {}
    for name, state_payload in (payload.get("images") or {}).items():
        source_path = file_by_name.get(name)
        if source_path is None or not isinstance(state_payload, dict):
            continue
        state = state_from_json_dict(state_payload, source_path)
        if state is not None:
            states[source_path] = state
    return states


def save_roll_session(folder: Path, states: dict[Path, ImageProcessingState], files: list[Path]) -> None:
    path = session_path_for_folder(folder)
    file_set = set(files)
    images: dict[str, dict[str, Any]] = {}
    for source_path in files:
        state = states.get(source_path)
        if state is None:
            continue
        images[source_path.name] = state_to_json_dict(state, source_path)

    for source_path, state in states.items():
        if source_path in file_set or source_path.parent != folder:
            continue
        images[source_path.name] = state_to_json_dict(state, source_path)

    payload = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "app": "NINA",
        "images": images,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(path)


def _file_stat(path: Path) -> dict[str, int | None]:
    try:
        stat = path.stat()
    except OSError:
        return {"file_size": None, "mtime_ns": None}
    return {"file_size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dataclass_to_dict(item) for item in value]
    return value


def _point_from_dict(payload: dict[str, Any] | None) -> ImagePoint | None:
    if not payload:
        return None
    return ImagePoint(x=int(payload["x"]), y=int(payload["y"]))


def _rect_from_dict(payload: dict[str, Any] | None) -> ImageRect | None:
    if not payload:
        return None
    return ImageRect(
        x=int(payload["x"]),
        y=int(payload["y"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        angle=float(payload.get("angle", 0.0)),
    )


def _adjustments_from_dict(payload: dict[str, Any]) -> AdjustmentParams:
    allowed = {item.name for item in fields(AdjustmentParams)}
    values = {key: payload[key] for key in payload if key in allowed}
    values["color_balance"] = _color_balance_from_dict(payload.get("color_balance") or {})
    values["density_matrix"] = _density_matrix_from_dict(payload.get("density_matrix") or {})
    values["lens_correction"] = _lens_correction_from_dict(payload.get("lens_correction") or {})
    return AdjustmentParams(**values)


def _color_balance_from_dict(payload: dict[str, Any]) -> ColorBalanceParams:
    return ColorBalanceParams(
        global_balance=_balance_axis_from_dict(payload.get("global_balance") or {}),
        shadows=_tonal_balance_from_dict(payload.get("shadows") or {}),
        midtones=_tonal_balance_from_dict(payload.get("midtones") or {}),
        highlights=_tonal_balance_from_dict(payload.get("highlights") or {}),
    )


def _balance_axis_from_dict(payload: dict[str, Any]) -> BalanceAxis:
    return BalanceAxis(
        red_cyan=int(payload.get("red_cyan", 0)),
        green_magenta=int(payload.get("green_magenta", 0)),
        blue_yellow=int(payload.get("blue_yellow", 0)),
    )


def _tonal_balance_from_dict(payload: dict[str, Any]) -> TonalBalance:
    return TonalBalance(
        red_cyan=int(payload.get("red_cyan", 0)),
        green_magenta=int(payload.get("green_magenta", 0)),
        blue_yellow=int(payload.get("blue_yellow", 0)),
        tonal_range=int(payload.get("tonal_range", 50)),
    )


def _density_matrix_from_dict(payload: dict[str, Any]) -> DensityMatrixParams:
    allowed = {item.name for item in fields(DensityMatrixParams)}
    values = {key: float(payload[key]) for key in payload if key in allowed}
    return DensityMatrixParams(**values)


def _lens_correction_from_dict(payload: dict[str, Any]) -> LensCorrectionParams:
    enabled = bool(payload.get("enabled", False))
    mode = str(payload.get("mode") or ("radial" if enabled else "off"))
    return LensCorrectionParams(
        enabled=enabled,
        mode=mode,
        strength=int(payload.get("strength", 0)),
        radius=int(payload.get("radius", 100)),
        center_x=int(payload.get("center_x", 50)),
        center_y=int(payload.get("center_y", 50)),
        smoothness=int(payload.get("smoothness", 200)),
        max_gain=int(payload.get("max_gain", 200)),
        flat_profile_path=payload.get("flat_profile_path"),
        flat_strength=int(payload.get("flat_strength", 100)),
    )
