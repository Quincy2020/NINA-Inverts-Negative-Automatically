from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from qnegative.core.geometry import rotated_rect_corners, scale_rect
from qnegative.core.models import ImageRect, ImageSize
from qnegative.core.preview import resize_long_edge


FRAME_RANKER_MAX_EDGE = 320
FRAME_RANKER_GLOBAL_CANDIDATES = 900
FRAME_RANKER_KEEP_CANDIDATES = 240
FRAME_RANKER_ANGLE_SNAP_DEGREES = 4.0

FORMAT_RATIOS = {
    "135": 1.50,
    "645": 4.0 / 3.0,
    "66": 1.00,
    "67": 7.0 / 6.0,
    "69": 1.50,
}

FEATURE_NAMES = [
    "cx",
    "cy",
    "width",
    "height",
    "area",
    "aspect",
    "angle_abs",
    "border_clearance",
    "format_score",
    "inside_luma_p10",
    "inside_luma_p50",
    "inside_luma_p90",
    "inside_luma_std",
    "inside_luma_range",
    "inside_chroma_p50",
    "inside_chroma_std",
    "inside_edge_density",
    "inside_black_fraction",
    "outside_luma_p10",
    "outside_luma_p50",
    "outside_luma_p90",
    "outside_luma_std",
    "outside_chroma_p50",
    "outside_chroma_std",
    "outside_edge_density",
    "outside_black_fraction",
    "outside_valid_fraction",
    "luma_contrast",
    "texture_ratio",
    "edge_support",
    "top_edge_support",
    "right_edge_support",
    "bottom_edge_support",
    "left_edge_support",
]


@dataclass(frozen=True)
class ImageFeatures:
    rgb: np.ndarray
    luma: np.ndarray
    chroma: np.ndarray
    edge: np.ndarray
    size: ImageSize


@dataclass(frozen=True)
class RankedFrameCandidate:
    rect: ImageRect
    confidence: float
    score: float
    format_hint: str
    method: str


def default_model_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "models" / "frame_ranker.joblib",
        root / "models" / "frame_ranker_dual_smoke.joblib",
    )


def detect_ranked_frame_candidates(
    preview_linear_rgb: np.ndarray,
    *,
    preview_size: ImageSize,
    source_size: ImageSize,
    format_hint: str = "auto",
    top_k: int = 3,
    model_path: Path | None = None,
) -> list[RankedFrameCandidate]:
    loaded = load_frame_ranker(model_path)
    if loaded is None:
        return []

    model, model_name = loaded
    image = np.nan_to_num(
        preview_linear_rgb.astype(np.float32, copy=False),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    scaled = resize_long_edge(image, max_size=FRAME_RANKER_MAX_EDGE)
    ranker_size = ImageSize(width=scaled.shape[1], height=scaled.shape[0])
    image_features = prepare_image_features(scaled)

    rng = np.random.default_rng(20260603)
    raw_rects = generate_global_candidates(
        ranker_size,
        count=FRAME_RANKER_GLOBAL_CANDIDATES,
        rng=rng,
    )
    x_all = np.vstack(
        [extract_candidate_features(image_features, rect) for rect in raw_rects]
    ).astype(np.float32)
    rects, x, _kept_indices = prefilter_candidates(
        raw_rects,
        x_all,
        keep=FRAME_RANKER_KEEP_CANDIDATES,
    )
    if len(rects) == 0:
        return []

    predictions = np.asarray(model.predict(x), dtype=np.float32)
    predictions = apply_format_hint_bias(predictions, rects, format_hint)
    order = np.argsort(predictions)[::-1]

    results: list[RankedFrameCandidate] = []
    for index in order[: max(1, top_k)]:
        rect = rects[int(index)]
        preview_rect = scale_rect(rect, ranker_size, preview_size)
        source_rect = snap_small_angle_rect(scale_rect(preview_rect, preview_size, source_size))
        score = float(predictions[int(index)])
        results.append(
            RankedFrameCandidate(
                rect=source_rect,
                confidence=float(np.clip(score, 0.0, 0.98)),
                score=score,
                format_hint=best_format_label(rect),
                method=f"ranker:{model_name}",
            )
        )
    return results


def snap_small_angle_rect(rect: ImageRect) -> ImageRect:
    if abs(rect.angle) > FRAME_RANKER_ANGLE_SNAP_DEGREES:
        return rect
    return ImageRect(
        x=rect.x,
        y=rect.y,
        width=rect.width,
        height=rect.height,
        angle=0.0,
    )


@lru_cache(maxsize=4)
def _load_frame_ranker_cached(path_text: str) -> tuple[object, str] | None:
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        import joblib
    except ImportError:
        return None

    try:
        payload = joblib.load(path)
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else payload
    feature_names = payload.get("feature_names") if isinstance(payload, dict) else FEATURE_NAMES
    if model is None or list(feature_names) != FEATURE_NAMES:
        return None
    return model, path.name


def load_frame_ranker(model_path: Path | None = None) -> tuple[object, str] | None:
    if model_path is not None:
        return _load_frame_ranker_cached(str(model_path.resolve()))
    for path in default_model_paths():
        loaded = _load_frame_ranker_cached(str(path.resolve()))
        if loaded is not None:
            return loaded
    return None


def apply_format_hint_bias(
    predictions: np.ndarray,
    rects: list[ImageRect],
    format_hint: str,
) -> np.ndarray:
    if format_hint == "auto" or format_hint not in FORMAT_RATIOS:
        return predictions

    target = FORMAT_RATIOS[format_hint]
    adjusted = predictions.copy()
    for index, rect in enumerate(rects):
        aspect = rect.width / max(rect.height, 1)
        if aspect < 1.0:
            aspect = 1.0 / max(aspect, 1e-5)
        delta = abs(np.log(max(aspect, 0.05)) - np.log(target))
        match = float(1.0 - np.clip(delta / 0.36, 0.0, 1.0))
        adjusted[index] = adjusted[index] * (0.88 + match * 0.12)
    return adjusted


def best_format_label(rect: ImageRect) -> str:
    aspect = rect.width / max(rect.height, 1)
    if aspect < 1.0:
        aspect = 1.0 / max(aspect, 1e-5)

    best_label = "auto"
    best_score = -1.0
    for label, target in FORMAT_RATIOS.items():
        delta = abs(np.log(max(aspect, 0.05)) - np.log(max(target, 0.05)))
        score = float(1.0 - np.clip(delta / 0.36, 0.0, 1.0))
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def prepare_image_features(image: np.ndarray) -> ImageFeatures:
    rgb = np.clip(
        np.nan_to_num(
            image.astype(np.float32, copy=False),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    luma = (rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722).astype(np.float32)
    chroma = (rgb.max(axis=2) - rgb.min(axis=2)).astype(np.float32)
    gray = normalized_uint8(luma)
    edge = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 130)
    height, width = image.shape[:2]
    return ImageFeatures(
        rgb=rgb,
        luma=luma,
        chroma=chroma,
        edge=edge,
        size=ImageSize(width=width, height=height),
    )


def normalized_uint8(values: np.ndarray) -> np.ndarray:
    sample = values.reshape(-1)
    low = float(np.percentile(sample, 1.0))
    high = float(np.percentile(sample, 99.0))
    if high <= low + 1e-6:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.ascontiguousarray((normalized * 255.0 + 0.5).astype(np.uint8))


def generate_global_candidates(size: ImageSize, *, count: int, rng: np.random.Generator) -> list[ImageRect]:
    rects: list[ImageRect] = []
    ratios = sorted(set(float(value) for value in FORMAT_RATIOS.values()))
    area_values = (0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78, 0.84)
    center_x_values = np.linspace(0.28, 0.72, 9)
    center_y_values = np.linspace(0.28, 0.72, 9)
    angle_values = (-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0)

    deterministic: list[tuple[float, float, float, float, float]] = []
    for ratio in ratios:
        for area_ratio in area_values:
            for cx_norm in center_x_values:
                for cy_norm in center_y_values:
                    for angle in angle_values:
                        deterministic.append((ratio, area_ratio, float(cx_norm), float(cy_norm), angle))
    rng.shuffle(deterministic)
    for ratio, area_ratio, cx_norm, cy_norm, angle in deterministic[:count]:
        rects.append(rect_from_format(size, ratio, area_ratio, cx_norm, cy_norm, angle))

    while len(rects) < count:
        ratio = float(rng.choice(ratios))
        if rng.random() < 0.06:
            ratio = 1.0 / ratio
        area_ratio = float(rng.uniform(0.34, 0.84))
        cx_norm = float(np.clip(rng.normal(0.50, 0.13), 0.18, 0.82))
        cy_norm = float(np.clip(rng.normal(0.50, 0.13), 0.18, 0.82))
        angle = float(rng.uniform(-12.0, 12.0))
        rects.append(rect_from_format(size, ratio, area_ratio, cx_norm, cy_norm, angle))
    return rects


def rect_from_format(
    size: ImageSize,
    aspect: float,
    area_ratio: float,
    cx_norm: float,
    cy_norm: float,
    angle: float,
) -> ImageRect:
    area = size.width * size.height * float(area_ratio)
    width = int(round(np.sqrt(area * aspect)))
    height = int(round(np.sqrt(area / aspect)))
    width = max(12, min(width, size.width))
    height = max(12, min(height, size.height))
    cx = cx_norm * size.width
    cy = cy_norm * size.height
    return clamp_rect(
        ImageRect(
            x=int(round(cx - width / 2)),
            y=int(round(cy - height / 2)),
            width=width,
            height=height,
            angle=normalize_angle(angle),
        ),
        size,
    )


def normalize_angle(angle: float) -> float:
    normalized = float(angle)
    while normalized > 45.0:
        normalized -= 90.0
    while normalized <= -45.0:
        normalized += 90.0
    return normalized


def clamp_rect(rect: ImageRect, size: ImageSize) -> ImageRect:
    width = max(1, min(rect.width, size.width))
    height = max(1, min(rect.height, size.height))
    x = max(0, min(rect.x, max(0, size.width - width)))
    y = max(0, min(rect.y, max(0, size.height - height)))
    return ImageRect(x=x, y=y, width=width, height=height, angle=rect.angle)


def extract_candidate_features(image_features: ImageFeatures, rect: ImageRect) -> np.ndarray:
    size = image_features.size
    inside_mask = rect_mask(rect, size)
    ring_mask = outside_ring_mask(rect, size)
    inside = mask_stats(image_features, inside_mask)
    outside = mask_stats(image_features, ring_mask)
    edge_values = edge_supports(image_features.edge, rotated_rect_corners(rect))
    aspect = rect.width / max(rect.height, 1)
    if aspect < 1.0:
        aspect = 1.0 / max(aspect, 1e-5)
    feature_values = [
        rect.center_x / max(1.0, size.width),
        rect.center_y / max(1.0, size.height),
        rect.width / max(1.0, size.width),
        rect.height / max(1.0, size.height),
        (rect.width * rect.height) / max(1.0, size.width * size.height),
        aspect,
        abs(rect.angle) / 18.0,
        border_clearance(rect, size),
        runtime_format_score(aspect),
        inside["luma_p10"],
        inside["luma_p50"],
        inside["luma_p90"],
        inside["luma_std"],
        inside["luma_p90"] - inside["luma_p10"],
        inside["chroma_p50"],
        inside["chroma_std"],
        inside["edge_density"],
        inside["black_fraction"],
        outside["luma_p10"],
        outside["luma_p50"],
        outside["luma_p90"],
        outside["luma_std"],
        outside["chroma_p50"],
        outside["chroma_std"],
        outside["edge_density"],
        outside["black_fraction"],
        outside["valid_fraction"],
        inside["luma_p50"] - outside["luma_p50"],
        inside["luma_std"] / max(outside["luma_std"], 1e-5),
        float(np.mean(edge_values)),
        edge_values[0],
        edge_values[1],
        edge_values[2],
        edge_values[3],
    ]
    return np.nan_to_num(np.array(feature_values, dtype=np.float32), nan=0.0, posinf=1e4, neginf=-1e4)


def runtime_format_score(aspect: float) -> float:
    best = 0.0
    for target in FORMAT_RATIOS.values():
        delta = abs(np.log(max(aspect, 0.05)) - np.log(target))
        best = max(best, float(1.0 - np.clip(delta / 0.36, 0.0, 1.0)))
    return best


def rect_mask(rect: ImageRect, size: ImageSize) -> np.ndarray:
    mask = np.zeros((size.height, size.width), dtype=np.uint8)
    points = np.round(rotated_rect_corners(rect)).astype(np.int32)
    cv2.fillConvexPoly(mask, points, 255)
    return mask > 0


def outside_ring_mask(rect: ImageRect, size: ImageSize) -> np.ndarray:
    pad = max(8.0, min(rect.width, rect.height) * 0.075)
    expanded = ImageRect(
        x=int(round(rect.x - pad)),
        y=int(round(rect.y - pad)),
        width=int(round(rect.width + pad * 2)),
        height=int(round(rect.height + pad * 2)),
        angle=rect.angle,
    )
    expanded_mask = rect_mask(expanded, size)
    inner_mask = rect_mask(rect, size)
    return expanded_mask & ~inner_mask


def mask_stats(image_features: ImageFeatures, mask: np.ndarray) -> dict[str, float]:
    count = int(np.count_nonzero(mask))
    total = int(mask.size)
    if count < 16:
        return {
            "luma_p10": 0.0,
            "luma_p50": 0.0,
            "luma_p90": 0.0,
            "luma_std": 0.0,
            "chroma_p50": 0.0,
            "chroma_std": 0.0,
            "edge_density": 0.0,
            "black_fraction": 1.0,
            "valid_fraction": count / max(1, total),
        }
    luma = image_features.luma[mask]
    chroma = image_features.chroma[mask]
    edge = image_features.edge[mask]
    return {
        "luma_p10": float(np.percentile(luma, 10.0)),
        "luma_p50": float(np.percentile(luma, 50.0)),
        "luma_p90": float(np.percentile(luma, 90.0)),
        "luma_std": float(np.std(luma)),
        "chroma_p50": float(np.percentile(chroma, 50.0)),
        "chroma_std": float(np.std(chroma)),
        "edge_density": float(np.mean(edge > 0)),
        "black_fraction": float(np.mean(luma < 0.030)),
        "valid_fraction": count / max(1, total),
    }


def edge_supports(edge: np.ndarray, corners: np.ndarray) -> np.ndarray:
    supports = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        distance = float(np.linalg.norm(end - start))
        steps = max(8, int(distance / 5.0))
        hits = 0
        total = 0
        for t in np.linspace(0.0, 1.0, steps, dtype=np.float32):
            x = int(round(start[0] * (1.0 - t) + end[0] * t))
            y = int(round(start[1] * (1.0 - t) + end[1] * t))
            if 0 <= x < edge.shape[1] and 0 <= y < edge.shape[0]:
                hits += 1 if edge[y, x] > 0 else 0
                total += 1
        supports.append(float(hits / total) if total else 0.0)
    return np.array(supports, dtype=np.float32)


def border_clearance(rect: ImageRect, size: ImageSize) -> float:
    points = rotated_rect_corners(rect)
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    margin = min(x_min, y_min, size.width - x_max, size.height - y_max)
    required = max(8.0, min(size.width, size.height) * 0.04)
    return float(np.clip(margin / required, 0.0, 1.0))


def prefilter_candidates(
    rects: list[ImageRect],
    features: np.ndarray,
    *,
    keep: int,
) -> tuple[list[ImageRect], np.ndarray, np.ndarray]:
    if len(rects) <= keep:
        indices = np.arange(len(rects), dtype=np.int32)
        return rects, features, indices

    index = {name: position for position, name in enumerate(FEATURE_NAMES)}

    cx = features[:, index["cx"]]
    cy = features[:, index["cy"]]
    area = features[:, index["area"]]
    angle_abs = features[:, index["angle_abs"]]
    border = features[:, index["border_clearance"]]
    format_match = features[:, index["format_score"]]
    inside_std = features[:, index["inside_luma_std"]]
    inside_range = features[:, index["inside_luma_range"]]
    outside_luma = features[:, index["outside_luma_p50"]]
    outside_std = features[:, index["outside_luma_std"]]
    outside_black = features[:, index["outside_black_fraction"]]
    outside_valid = features[:, index["outside_valid_fraction"]]
    edge_support = features[:, index["edge_support"]]

    inside_content = (
        np.clip(inside_range / 0.34, 0.0, 1.0) * 0.55
        + np.clip(inside_std / 0.18, 0.0, 1.0) * 0.45
    )
    base_ring_score = (
        np.clip((outside_luma - 0.035) / 0.18, 0.0, 1.0) * 0.32
        + np.clip(1.0 - outside_std / 0.20, 0.0, 1.0) * 0.23
        + outside_valid * 0.15
        + np.clip(1.0 - outside_black * 1.2, 0.0, 1.0) * 0.10
        + edge_support * 0.20
    )
    tight_crop_score = (
        np.clip(outside_black * 1.45, 0.0, 1.0) * 0.24
        + edge_support * 0.26
        + inside_content * 0.25
        + border * 0.10
        + format_match * 0.15
    )
    center_distance = np.abs(cx - 0.5) * 2.0 + np.abs(cy - 0.5) * 2.0
    center_score = np.clip(1.0 - center_distance / 1.35, 0.0, 1.0)
    area_score = np.clip(1.0 - np.abs(area - 0.66) / 0.46, 0.0, 1.0)
    geometry_score = (
        format_match * 0.36
        + area_score * 0.24
        + border * 0.12
        + center_score * 0.18
        + np.clip(1.0 - angle_abs / 0.95, 0.0, 1.0) * 0.10
    )
    oversize_touch = np.clip((area - 0.84) / 0.12, 0.0, 1.0) * np.clip((0.10 - border) / 0.10, 0.0, 1.0)
    empty_inside = np.clip((0.20 - inside_content) / 0.20, 0.0, 1.0)
    score = (
        np.maximum(base_ring_score, tight_crop_score) * 0.46
        + geometry_score * 0.34
        + inside_content * 0.20
        - oversize_touch * 0.18
        - empty_inside * 0.12
    )

    keep_count = min(len(rects), keep)
    selected = np.argsort(score)[-keep_count:][::-1].astype(np.int32)
    return [rects[int(index)] for index in selected], features[selected], selected
