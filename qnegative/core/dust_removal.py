from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import Callable

import cv2
import numpy as np

from qnegative.core.dust_masks import compose_dust_masks
from qnegative.core.models import DustRemovalParams
from qnegative.core.dust_model_registry import default_dust_model_path, dust_model_plugin
from qnegative.ml.dust_model import DustUNet


def apply_dust_removal_to_linear_rgb(
    linear_rgb: np.ndarray,
    params: DustRemovalParams,
    *,
    model_root: Path | None = None,
    auto_mask: np.ndarray | None = None,
    manual_add_mask: np.ndarray | None = None,
    manual_protect_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Detect and repair dust on a positive image.

    The dust model was trained on positive-display luma, so detection and
    inpainting happen in sRGB display space. The repaired image is converted
    back to linear RGB so the existing export encoder can keep its format path.
    """
    if not params.enabled:
        return linear_rgb, {}

    srgb = linear_to_srgb_float(linear_rgb)
    if auto_mask is None:
        mask, stats = predict_dust_mask(srgb, params, model_root=model_root)
        stats["dust_auto_mask_reused"] = 0.0
    else:
        mask = auto_mask
        stats = {"dust_auto_mask_reused": 1.0}
    final_mask = compose_dust_masks(
        mask,
        manual_add_mask,
        manual_protect_mask,
        target_shape=srgb.shape[:2],
    )
    stats["dust_auto_mask_area"] = float(np.mean(mask > 0))
    stats["dust_manual_add_area"] = (
        float(np.mean(manual_add_mask > 0))
        if manual_add_mask is not None and manual_add_mask.size
        else 0.0
    )
    stats["dust_manual_protect_area"] = (
        float(np.mean(manual_protect_mask > 0))
        if manual_protect_mask is not None and manual_protect_mask.size
        else 0.0
    )
    stats["dust_mask_area"] = float(np.mean(final_mask > 0))
    stats["dust_final_mask_area"] = stats["dust_mask_area"]
    if not np.any(final_mask):
        stats["inpaint_area"] = 0.0
        return linear_rgb, stats

    repaired = inpaint_srgb(srgb, final_mask, radius=max(1, int(params.inpaint_radius)))
    stats["inpaint_area"] = float(np.mean(final_mask > 0))
    return srgb_to_linear_float(repaired).astype(np.float32, copy=False), stats


def predict_dust_mask(
    srgb_rgb: np.ndarray,
    params: DustRemovalParams,
    *,
    model_root: Path | None = None,
    tile: int = 512,
    overlap: int = 96,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_event: Event | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    model, device = _load_model(_resolve_model_path(params, model_root))
    luma = rgb_to_luma(srgb_rgb).astype(np.float32, copy=False)
    probability = tiled_predict_luma(
        model,
        device,
        luma,
        tile=tile,
        overlap=overlap,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    _raise_if_cancelled(cancel_event)
    if progress_callback is not None:
        progress_callback(86, "Thresholding mask")
    threshold = max(0.01, min(0.99, params.threshold / 100.0))
    if params.adaptive:
        texture = texture_map(luma)
        max_threshold = max(threshold, min(0.99, params.max_threshold / 100.0))
        threshold_map = np.clip(
            threshold + texture * max(0.0, params.texture_penalty / 100.0),
            threshold,
            max_threshold,
        )
        mask = probability >= threshold_map
    else:
        texture = np.zeros_like(luma, dtype=np.float32)
        threshold_map = np.full_like(luma, threshold, dtype=np.float32)
        mask = probability >= threshold

    mask = cleanup_mask(mask)
    _raise_if_cancelled(cancel_event)
    if progress_callback is not None:
        progress_callback(92, "Protecting texture")
    component_stats = protect_high_texture_components(mask, probability, texture, threshold_map, params)
    mask = cleanup_mask(component_stats["mask"], dilate=False)
    stats = {
        "dust_probability_mean": float(np.mean(probability)),
        "dust_probability_p95": float(np.percentile(probability, 95.0)),
        "dust_mask_area": float(np.mean(mask)),
        "dust_texture_mean": float(np.mean(texture)),
        "dust_components_total": float(component_stats["total"]),
        "dust_components_rejected": float(component_stats["rejected"]),
        "dust_components_high_texture": float(component_stats["high_texture"]),
        "dust_components_line_rejected": float(component_stats["line_rejected"]),
    }
    if progress_callback is not None:
        progress_callback(100, "Auto mask ready")
    return mask.astype(np.uint8), stats


def tiled_predict_luma(
    model,
    device,
    luma: np.ndarray,
    *,
    tile: int,
    overlap: int,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_event: Event | None = None,
) -> np.ndarray:
    import torch

    h, w = luma.shape
    if h <= tile and w <= tile:
        _raise_if_cancelled(cancel_event)
        padded = _pad_to(luma, tile, tile)
        pred = _predict_tile(model, device, padded)
        if progress_callback is not None:
            progress_callback(82, "Model inference")
        return pred[:h, :w]

    stride = max(64, tile - overlap)
    probability = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    window = _blend_window(tile)
    ys = _tile_starts(h, tile, stride)
    xs = _tile_starts(w, tile, stride)
    total_tiles = max(1, len(ys) * len(xs))
    completed = 0
    with torch.no_grad():
        for y in ys:
            for x in xs:
                _raise_if_cancelled(cancel_event)
                patch = luma[y : y + tile, x : x + tile]
                ph, pw = patch.shape
                if ph != tile or pw != tile:
                    patch = _pad_to(patch, tile, tile)
                pred = _predict_tile(model, device, patch)[:ph, :pw]
                tile_weight = window[:ph, :pw]
                probability[y : y + ph, x : x + pw] += pred * tile_weight
                weight[y : y + ph, x : x + pw] += tile_weight
                completed += 1
                if progress_callback is not None:
                    value = 8 + int(round(completed / total_tiles * 74))
                    progress_callback(value, f"Model inference {completed}/{total_tiles}")
    return probability / np.maximum(weight, 1.0e-6)


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Dust mask generation cancelled.")


def texture_map(luma: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(luma.astype(np.float32), (0, 0), 1.25)
    coarse = cv2.GaussianBlur(smooth, (0, 0), 7.0)
    detail = np.abs(smooth - coarse)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    # A single clean edge should not be treated like dense texture. Use local
    # edge density and gradient-direction incoherence so hair, foliage,
    # buildings, text, and other repeated structures score higher than one
    # isolated border line.
    grad_scale = float(np.percentile(grad, 92.0)) + 1.0e-6
    edge_density = cv2.GaussianBlur((grad > grad_scale * 0.35).astype(np.float32), (0, 0), 4.0)
    local_energy = cv2.GaussianBlur(grad, (0, 0), 2.5)
    jxx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), 3.0)
    jyy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), 3.0)
    jxy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), 3.0)
    energy = jxx + jyy
    coherence = np.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy) / (energy + 1.0e-6)
    direction_clutter = 1.0 - np.clip(coherence, 0.0, 1.0)
    repeated_structure = local_energy * (0.35 + edge_density * 0.85) * (0.45 + direction_clutter * 0.75)
    structure = repeated_structure * 0.72 + detail * edge_density * 0.28
    lo = float(np.percentile(structure, 55.0))
    hi = float(np.percentile(structure, 98.5))
    if hi <= lo + 1.0e-6:
        return np.zeros_like(luma, dtype=np.float32)
    texture = np.clip((structure - lo) / (hi - lo), 0.0, 1.0)
    return cv2.GaussianBlur(texture.astype(np.float32), (0, 0), 2.0)


def protect_high_texture_components(
    mask: np.ndarray,
    probability: np.ndarray,
    texture: np.ndarray,
    threshold_map: np.ndarray,
    params: DustRemovalParams,
) -> dict[str, object]:
    """Remove low-confidence components in busy image regions.

    The pixel-level adaptive threshold already raises the threshold in hair,
    branches, text, and other high-frequency areas. This second pass is more
    conservative at the object level: if a whole connected component lives in a
    high-texture region, it needs stronger model confidence to be repaired.
    Low-frequency components are intentionally kept because isolated defects in
    smooth sky/shadow/wall areas are very likely to be real dust or lint.
    """
    clean = mask.astype(np.uint8)
    if not np.any(clean) or not params.adaptive or params.texture_penalty <= 0:
        return {"mask": clean.astype(bool), "total": 0, "rejected": 0, "high_texture": 0, "line_rejected": 0}

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    filtered = np.zeros_like(clean)
    total = rejected = high_texture_count = line_rejected = 0
    guard = np.clip(params.texture_penalty / 50.0, 0.0, 1.0)
    low_texture_cutoff = 0.30
    high_texture_cutoff = 0.58 - 0.10 * guard

    for index in range(1, count):
        component = labels == index
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 3:
            rejected += 1
            continue
        total += 1
        tex_mean = float(np.mean(texture[component]))
        tex_p75 = float(np.percentile(texture[component], 75.0))
        prob_mean = float(np.mean(probability[component]))
        prob_max = float(np.max(probability[component]))
        threshold_mean = float(np.mean(threshold_map[component]))
        line_like = _component_is_line_like(component)

        keep = True
        if tex_mean <= low_texture_cutoff and tex_p75 <= high_texture_cutoff:
            keep = True
        elif tex_mean >= high_texture_cutoff or tex_p75 >= high_texture_cutoff:
            high_texture_count += 1
            mean_margin = 0.045 + 0.030 * guard
            max_margin = 0.075 + 0.040 * guard
            keep = (prob_mean >= threshold_mean + mean_margin) or (prob_max >= threshold_mean + max_margin)
            if line_like:
                keep = keep and (
                    prob_mean >= threshold_mean + mean_margin + 0.030
                    or prob_max >= min(0.99, threshold_mean + max_margin + 0.060)
                )
        else:
            keep = (prob_mean >= threshold_mean + 0.015) or (prob_max >= threshold_mean + 0.080)

        if keep:
            filtered[component] = 1
        else:
            rejected += 1
            if line_like:
                line_rejected += 1

    return {
        "mask": filtered.astype(bool),
        "total": total,
        "rejected": rejected,
        "high_texture": high_texture_count,
        "line_rejected": line_rejected,
    }


def _component_is_line_like(component: np.ndarray) -> bool:
    ys, xs = np.where(component)
    if len(xs) < 5:
        return False
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(points.reshape(-1, 1, 2))
    width, height = rect[1]
    major = max(float(width), float(height), 1.0)
    minor = max(min(float(width), float(height)), 1.0)
    aspect = major / minor
    fill = len(xs) / max(major * minor, 1.0)
    return aspect >= 4.5 and fill <= 0.55


def cleanup_mask(mask: np.ndarray, *, dilate: bool = True) -> np.ndarray:
    clean = mask.astype(np.uint8)
    if not np.any(clean):
        return clean
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    filtered = np.zeros_like(clean)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 3:
            filtered[labels == index] = 1
    if not dilate:
        return filtered.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    clean = cv2.dilate(filtered, kernel, iterations=1)
    return clean.astype(bool)


def inpaint_srgb(srgb_rgb: np.ndarray, mask: np.ndarray, *, radius: int) -> np.ndarray:
    srgb8 = np.clip(srgb_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    mask8 = (mask.astype(np.uint8) * 255)
    repaired = cv2.inpaint(srgb8, mask8, float(radius), cv2.INPAINT_TELEA)
    return repaired.astype(np.float32) / 255.0


def linear_to_srgb_float(linear_rgb: np.ndarray) -> np.ndarray:
    return np.power(np.clip(linear_rgb, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)


def srgb_to_linear_float(srgb_rgb: np.ndarray) -> np.ndarray:
    return np.power(np.clip(srgb_rgb, 0.0, 1.0), 2.2).astype(np.float32)


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    ).astype(np.float32)


def _resolve_model_path(params: DustRemovalParams, model_root: Path | None) -> Path:
    if params.model_path:
        path = Path(params.model_path)
    elif params.model_id:
        plugin = dust_model_plugin(params.model_id, model_root)
        if plugin is not None:
            return plugin.model_path
        return default_dust_model_path(model_root)
    else:
        return default_dust_model_path(model_root)
    if path.is_absolute():
        return path.resolve()
    return ((model_root or Path.cwd()) / path).resolve()


@lru_cache(maxsize=2)
def _load_model(path: Path):
    import torch

    if not path.exists():
        raise FileNotFoundError(f"Dust model not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    model = DustUNet.create(in_channels=1, base_channels=int(checkpoint.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    return model, device


def _predict_tile(model, device, patch: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(patch[None, None, ...].astype(np.float32)).to(device)
    return torch.sigmoid(model(tensor)).detach().cpu().numpy()[0, 0].astype(np.float32)


def _pad_to(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    pad_h = max(0, height - arr.shape[0])
    pad_w = max(0, width - arr.shape[1])
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")


def _tile_starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, max(1, size - tile + 1), stride))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


@lru_cache(maxsize=8)
def _blend_window(tile: int) -> np.ndarray:
    one = np.hanning(tile).astype(np.float32)
    one = np.maximum(one, 0.08)
    return np.outer(one, one).astype(np.float32)
