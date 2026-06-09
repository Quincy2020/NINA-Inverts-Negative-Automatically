from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DUST_MODEL_CONFIG = Path("configs/dust_models.json")
DEFAULT_DUST_MODELS_DIR = Path("models")
DEFAULT_DUST_MODEL_ID = "lint_ice_stage2"
FALLBACK_DUST_MODEL = Path("models/dust_lint_ice_stage2/best.pth")
PLUGIN_FILE_NAMES = ("dust_plugin.json", "plugin.json")


@dataclass(frozen=True)
class DustModelPlugin:
    plugin_id: str
    name: str
    model_path: Path
    model_type: str = "dust-mask-unet"
    threshold: int = 20
    texture_penalty: int = 10
    max_threshold: int = 75
    inpaint_radius: int = 3
    notes: str = ""


def default_dust_model_path(model_root: Path | None = None) -> Path:
    plugin = default_dust_model_plugin(model_root=model_root)
    return plugin.model_path


def default_dust_model_plugin(model_root: Path | None = None) -> DustModelPlugin:
    root = model_root or Path.cwd()
    plugins, default_id = _load_plugins(root)
    if default_id in plugins:
        return plugins[default_id]
    if plugins:
        return next(iter(plugins.values()))
    return DustModelPlugin(
        plugin_id="lint_ice_stage2",
        name="Lint + ICE Stage2",
        model_path=(root / FALLBACK_DUST_MODEL).resolve(),
    )


def dust_model_plugins(model_root: Path | None = None) -> list[DustModelPlugin]:
    root = model_root or Path.cwd()
    plugins, _default_id = _load_plugins(root)
    if plugins:
        return list(plugins.values())
    return [default_dust_model_plugin(model_root=root)]


def dust_model_plugin(plugin_id: str, model_root: Path | None = None) -> DustModelPlugin | None:
    root = model_root or Path.cwd()
    plugins, _default_id = _load_plugins(root)
    return plugins.get(plugin_id)

def _load_plugins(root: Path) -> tuple[dict[str, DustModelPlugin], str]:
    config_path = root / DEFAULT_DUST_MODEL_CONFIG
    data = _read_json(config_path) if config_path.exists() else {}
    plugins: dict[str, DustModelPlugin] = {}

    for metadata_path in _plugin_metadata_paths(root):
        item = _read_json(metadata_path)
        plugin = _plugin_from_dict(
            item,
            root=root,
            base_dir=metadata_path.parent,
            fallback_id=metadata_path.parent.name,
        )
        plugins[plugin.plugin_id] = plugin

    for item in data.get("plugins", []):
        plugin_id = str(item.get("id") or item.get("plugin_id") or "")
        existing = plugins.get(plugin_id) if plugin_id else None
        plugin = _plugin_from_dict(
            item,
            root=root,
            base_dir=root,
            existing=existing,
            fallback_id=plugin_id or None,
        )
        plugins[plugin.plugin_id] = plugin

    return plugins, str(data.get("default", DEFAULT_DUST_MODEL_ID))


def _plugin_metadata_paths(root: Path) -> list[Path]:
    models_root = root / DEFAULT_DUST_MODELS_DIR
    if not models_root.exists():
        return []
    paths: list[Path] = []
    seen_dirs: set[Path] = set()
    for file_name in PLUGIN_FILE_NAMES:
        for path in sorted(models_root.glob(f"dust_*/{file_name}")):
            parent = path.parent.resolve()
            if parent in seen_dirs:
                continue
            seen_dirs.add(parent)
            paths.append(path)
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _plugin_from_dict(
    item: dict[str, Any],
    *,
    root: Path,
    base_dir: Path,
    existing: DustModelPlugin | None = None,
    fallback_id: str | None = None,
) -> DustModelPlugin:
    plugin_id = str(
        item.get("id")
        or item.get("plugin_id")
        or (existing.plugin_id if existing else fallback_id)
        or DEFAULT_DUST_MODEL_ID
    )
    path_value = item.get("model_path") or item.get("weights")
    if path_value is None and existing is not None:
        path = existing.model_path
    elif path_value is None:
        path = _default_model_path_for_dir(base_dir)
    else:
        path = Path(str(path_value))
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    defaults = _defaults_from_dict(item)
    threshold = _percent_slider_value(
        defaults.get("threshold"),
        existing.threshold if existing else 20,
    )
    texture_penalty = _percent_slider_value(
        defaults.get("texture_penalty"),
        existing.texture_penalty if existing else 10,
    )
    max_threshold = _percent_slider_value(
        defaults.get("max_threshold"),
        existing.max_threshold if existing else 75,
    )
    inpaint_radius = int(defaults.get(
        "inpaint_radius",
        existing.inpaint_radius if existing else 3,
    ))
    return DustModelPlugin(
        plugin_id=plugin_id,
        name=str(item.get("name") or (existing.name if existing else plugin_id)),
        model_path=path,
        model_type=str(item.get("type") or item.get("model_type") or (existing.model_type if existing else "dust-mask-unet")),
        threshold=threshold,
        texture_penalty=texture_penalty,
        max_threshold=max_threshold,
        inpaint_radius=inpaint_radius,
        notes=str(item.get("notes") or (existing.notes if existing else "")),
    )


def _defaults_from_dict(item: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key in ("defaults", "runtime_defaults", "parameters", "dust_removal"):
        payload = item.get(key)
        if isinstance(payload, dict):
            defaults.update(payload)
    for key in ("threshold", "texture_penalty", "max_threshold", "inpaint_radius"):
        if key in item:
            defaults[key] = item[key]
    return defaults


def _percent_slider_value(value: Any, fallback: int) -> int:
    if value is None:
        return int(fallback)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            text = text[:-1]
        value = text
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(fallback)
    if 0.0 <= numeric <= 1.0:
        numeric *= 100.0
    return int(round(numeric))


def _default_model_path_for_dir(directory: Path) -> Path:
    for name in (
        "best.pth",
        "candidate.pth",
        "nina_dust_candidate.pth",
        "nina_dust_finetuned_candidate.pth",
        "latest.pth",
    ):
        path = directory / name
        if path.exists():
            return path
    candidates = sorted(directory.glob("*.pth"))
    if candidates:
        return candidates[0]
    return directory / "best.pth"
