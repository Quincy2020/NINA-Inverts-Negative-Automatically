from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from qnegative.core.geometry import rotated_rect_corners, scale_rect
from qnegative.core.models import ImageRect, ImageSize
from qnegative.core.preview import resize_long_edge


FRAME_RANKER_MAX_EDGE = 256
FRAME_RANKER_GLOBAL_CANDIDATES = 160
FRAME_RANKER_PRIOR_CANDIDATES = 48
FRAME_RANKER_CORNER_CANDIDATES = 48
FRAME_RANKER_BOUNDARY_CANDIDATES = 120
FRAME_RANKER_KEEP_CANDIDATES = 80
FRAME_RANKER_ANGLE_SNAP_DEGREES = 3.0
FRAME_RANKER_OUTPUT_INSET_RATIO = 0.0

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


@dataclass(frozen=True)
class BoundaryLineCandidate:
    position: int
    score: float


@dataclass(frozen=True)
class CornerCandidate:
    x: int
    y: int
    score: float
    quadrant: str
    polarity: str


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
    prior_frame_rect: ImageRect | None = None,
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
    prior_rects: list[ImageRect] = []
    if prior_frame_rect is not None and prior_frame_rect.is_valid():
        prior_rect = scale_rect(prior_frame_rect, source_size, ranker_size)
        prior_rects = generate_prior_frame_candidates(
            ranker_size,
            prior_rect,
            format_hint=format_hint,
            count=FRAME_RANKER_PRIOR_CANDIDATES,
        )
    corner_rects = generate_center_out_corner_candidates(
        image_features,
        format_hint=format_hint,
        count=FRAME_RANKER_CORNER_CANDIDATES,
    )
    boundary_rects = generate_boundary_candidates(
        image_features,
        format_hint=format_hint,
        count=FRAME_RANKER_BOUNDARY_CANDIDATES,
    )
    raw_rects = dedupe_rects(corner_rects + prior_rects + boundary_rects)
    fallback_count = max(FRAME_RANKER_GLOBAL_CANDIDATES - len(raw_rects), 24)
    raw_rects.extend(generate_global_candidates(
        ranker_size,
        count=fallback_count,
        rng=rng,
        format_hint=format_hint,
    ))
    raw_rects = dedupe_rects(raw_rects)[:FRAME_RANKER_GLOBAL_CANDIDATES]
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
        source_rect = inset_frame_rect(
            snap_small_angle_rect(scale_rect(preview_rect, preview_size, source_size)),
            inset_ratio=FRAME_RANKER_OUTPUT_INSET_RATIO,
        )
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


def inset_frame_rect(rect: ImageRect, *, inset_ratio: float) -> ImageRect:
    if inset_ratio <= 0.0:
        return rect
    inset = float(np.clip(inset_ratio, 0.0, 0.12))
    dx = int(round(rect.width * inset))
    dy = int(round(rect.height * inset))
    width = max(1, rect.width - dx * 2)
    height = max(1, rect.height - dy * 2)
    return ImageRect(
        x=rect.x + dx,
        y=rect.y + dy,
        width=width,
        height=height,
        angle=rect.angle,
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


def format_ratios_for_hint(format_hint: str, *, include_inverse: bool = True) -> tuple[float, ...]:
    base_ratios = (
        (FORMAT_RATIOS[format_hint],)
        if format_hint in FORMAT_RATIOS
        else (1.50, 4.0 / 3.0, 1.0, 7.0 / 6.0)
    )
    ratios: list[float] = []
    for ratio in base_ratios:
        ratio = float(ratio)
        if not any(abs(ratio - existing) < 1e-4 for existing in ratios):
            ratios.append(ratio)
        inverse = 1.0 / max(ratio, 1e-5)
        if include_inverse and abs(inverse - ratio) > 0.03:
            if not any(abs(inverse - existing) < 1e-4 for existing in ratios):
                ratios.append(inverse)
    return tuple(ratios)


def generate_prior_frame_candidates(
    size: ImageSize,
    prior_rect: ImageRect,
    *,
    format_hint: str,
    count: int,
) -> list[ImageRect]:
    if count <= 0:
        return []

    base = clamp_rect(prior_rect, size)
    ratios = format_ratios_for_hint(format_hint) if format_hint in FORMAT_RATIOS else ()
    if ratios:
        aspect = base.width / max(base.height, 1)
        ratio = min(ratios, key=lambda item: abs(np.log(max(aspect, 0.05)) - np.log(max(item, 0.05))))
        base = fit_format_rect_to_boundary_box(
            size,
            left=base.x,
            right=base.x + base.width,
            top=base.y,
            bottom=base.y + base.height,
            aspect=ratio,
            angle=base.angle,
        )

    shift_x = max(1, int(round(base.width * 0.018)))
    shift_y = max(1, int(round(base.height * 0.018)))
    shifts = (
        (0, 0),
        (-shift_x, 0),
        (shift_x, 0),
        (0, -shift_y),
        (0, shift_y),
        (-shift_x, -shift_y),
        (shift_x, -shift_y),
        (-shift_x, shift_y),
        (shift_x, shift_y),
        (-shift_x * 2, 0),
        (shift_x * 2, 0),
        (0, -shift_y * 2),
        (0, shift_y * 2),
    )
    scales = (1.0, 0.985, 1.015, 0.970, 1.030)
    angles = dedupe_float_values((base.angle, 0.0, base.angle - 1.2, base.angle + 1.2))

    candidates: list[ImageRect] = []
    for dx, dy in shifts:
        for scale in scales:
            width = max(12, int(round(base.width * scale)))
            height = max(12, int(round(base.height * scale)))
            center_x = base.center_x + dx
            center_y = base.center_y + dy
            for angle in angles:
                candidates.append(
                    clamp_rect(
                        ImageRect(
                            x=int(round(center_x - width * 0.5)),
                            y=int(round(center_y - height * 0.5)),
                            width=width,
                            height=height,
                            angle=normalize_angle(angle),
                        ),
                        size,
                    )
                )
                if len(candidates) >= count:
                    return dedupe_rects(candidates)
    return dedupe_rects(candidates)[:count]


def dedupe_float_values(values: tuple[float, ...], *, epsilon: float = 0.35) -> tuple[float, ...]:
    deduped: list[float] = []
    for value in values:
        value = normalize_angle(float(value))
        if any(abs(value - existing) < epsilon for existing in deduped):
            continue
        deduped.append(value)
    return tuple(deduped)


def generate_center_out_corner_candidates(
    image_features: ImageFeatures,
    *,
    format_hint: str,
    count: int,
) -> list[ImageRect]:
    if count <= 0:
        return []

    size = image_features.size
    min_side = max(1, min(size.width, size.height))
    enhanced = enhanced_luma_for_corners(image_features.luma)
    edges = cv2.Canny(cv2.GaussianBlur(enhanced, (3, 3), 0), 42, 138)
    raw_points = cv2.goodFeaturesToTrack(
        enhanced,
        maxCorners=96,
        qualityLevel=0.018,
        minDistance=max(5, int(round(min_side * 0.028))),
        blockSize=5,
        useHarrisDetector=True,
        k=0.04,
    )
    if raw_points is None:
        return []

    center_x = size.width * 0.5
    center_y = size.height * 0.5
    by_quadrant: dict[str, list[CornerCandidate]] = {
        "tl": [],
        "tr": [],
        "br": [],
        "bl": [],
    }
    for point in raw_points[:, 0, :]:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        quadrant = point_quadrant(x, y, center_x=center_x, center_y=center_y)
        if quadrant is None:
            continue
        score, polarity = score_center_out_corner(image_features, edges, x, y, quadrant)
        if score < 0.34:
            continue
        by_quadrant[quadrant].append(
            CornerCandidate(
                x=x,
                y=y,
                score=score,
                quadrant=quadrant,
                polarity=polarity,
            )
        )

    for quadrant, values in by_quadrant.items():
        values.sort(
            key=lambda item: (
                1 if item.polarity == "bright_base" else 0,
                item.score,
            ),
            reverse=True,
        )
        by_quadrant[quadrant] = values[:5]

    ratios = format_ratios_for_hint(format_hint)
    bright_quadrants = {
        quadrant: [candidate for candidate in candidates if candidate.polarity == "bright_base"]
        for quadrant, candidates in by_quadrant.items()
    }
    dark_quadrants = {
        quadrant: [candidate for candidate in candidates if candidate.polarity == "dark_table_fallback"]
        for quadrant, candidates in by_quadrant.items()
    }

    bright_rects = build_corner_rect_candidates(size, bright_quadrants, ratios)
    if len(bright_rects) >= min(count, 18):
        bright_rects.sort(key=lambda item: item[0], reverse=True)
        return dedupe_rects([rect for _score, rect in bright_rects])[:count]

    fallback_quadrants = {
        quadrant: (bright_quadrants[quadrant] + dark_quadrants[quadrant])[:5]
        for quadrant in by_quadrant
    }
    scored_rects = bright_rects + build_corner_rect_candidates(size, fallback_quadrants, ratios)
    if not scored_rects:
        return []
    scored_rects.sort(key=lambda item: item[0], reverse=True)
    return dedupe_rects([rect for _score, rect in scored_rects])[:count]


def build_corner_rect_candidates(
    size: ImageSize,
    by_quadrant: dict[str, list[CornerCandidate]],
    ratios: tuple[float, ...],
) -> list[tuple[float, ImageRect]]:
    scored_rects: list[tuple[float, ImageRect]] = []
    scored_rects.extend(corner_opposite_rects(size, by_quadrant["tl"], by_quadrant["br"], ratios))
    scored_rects.extend(corner_opposite_rects(size, by_quadrant["tr"], by_quadrant["bl"], ratios))
    scored_rects.extend(corner_same_edge_rects(size, by_quadrant["tl"], by_quadrant["tr"], ratios, edge_name="top"))
    scored_rects.extend(corner_same_edge_rects(size, by_quadrant["bl"], by_quadrant["br"], ratios, edge_name="bottom"))
    scored_rects.extend(corner_same_edge_rects(size, by_quadrant["tl"], by_quadrant["bl"], ratios, edge_name="left"))
    scored_rects.extend(corner_same_edge_rects(size, by_quadrant["tr"], by_quadrant["br"], ratios, edge_name="right"))
    return scored_rects


def enhanced_luma_for_corners(luma: np.ndarray) -> np.ndarray:
    gray = normalized_uint8(luma)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def point_quadrant(x: int, y: int, *, center_x: float, center_y: float) -> str | None:
    if x < center_x and y < center_y:
        return "tl"
    if x >= center_x and y < center_y:
        return "tr"
    if x >= center_x and y >= center_y:
        return "br"
    if x < center_x and y >= center_y:
        return "bl"
    return None


def quadrant_inside_dirs(quadrant: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if quadrant == "tl":
        return (1, 0), (0, 1), (1, 1)
    if quadrant == "tr":
        return (-1, 0), (0, 1), (-1, 1)
    if quadrant == "br":
        return (-1, 0), (0, -1), (-1, -1)
    return (1, 0), (0, -1), (1, -1)


def score_center_out_corner(
    image_features: ImageFeatures,
    edge: np.ndarray,
    x: int,
    y: int,
    quadrant: str,
) -> tuple[float, str]:
    min_side = max(1, min(image_features.size.width, image_features.size.height))
    arm_length = int(np.clip(min_side * 0.16, 16, 56))
    patch_radius = int(np.clip(min_side * 0.030, 4, 12))
    dir_a, dir_b, inner_diag = quadrant_inside_dirs(quadrant)
    edge_score = (
        sample_edge_ray(edge, x, y, dir_a[0], dir_a[1], arm_length)
        + sample_edge_ray(edge, x, y, dir_b[0], dir_b[1], arm_length)
    ) * 0.5

    inside = sample_luma_patch(
        image_features.luma,
        x + inner_diag[0] * patch_radius * 2,
        y + inner_diag[1] * patch_radius * 2,
        patch_radius,
    )
    outside = sample_luma_patch(
        image_features.luma,
        x - inner_diag[0] * patch_radius * 2,
        y - inner_diag[1] * patch_radius * 2,
        patch_radius,
    )
    if inside is None or outside is None:
        return 0.0, "weak"

    inside_mean, inside_std = inside
    outside_mean, outside_std = outside
    inside_content = float(np.clip(inside_std / 0.16, 0.0, 1.0))
    outside_stable = float(np.clip(1.0 - outside_std / 0.12, 0.0, 1.0))
    bright_base = float(np.clip((outside_mean - inside_mean + 0.005) / 0.22, 0.0, 1.0))
    dark_table = (
        float(np.clip((inside_mean - outside_mean + 0.005) / 0.22, 0.0, 1.0))
        * float(np.clip((0.060 - outside_mean) / 0.060, 0.0, 1.0))
        * outside_stable
    )
    if bright_base >= max(0.18, dark_table * 0.85):
        polarity = "bright_base"
        outside_prior = bright_base
        polarity_weight = 1.0
    elif dark_table >= 0.28:
        polarity = "dark_table_fallback"
        outside_prior = dark_table
        polarity_weight = 0.72
    else:
        return 0.0, "weak"
    contrast = float(np.clip(abs(outside_mean - inside_mean) / 0.28, 0.0, 1.0))
    center_distance = abs(x / max(1.0, image_features.size.width) - 0.5) + abs(
        y / max(1.0, image_features.size.height) - 0.5
    )
    center_out = float(np.clip(center_distance / 0.55, 0.0, 1.0))

    score = float(
        np.clip(
            edge_score * 0.34
            + outside_stable * 0.16
            + outside_prior * 0.18
            + contrast * 0.12
            + inside_content * 0.10
            + center_out * 0.10,
            0.0,
            1.0,
        )
    )
    return score * polarity_weight, polarity


def sample_edge_ray(edge: np.ndarray, x: int, y: int, dx: int, dy: int, length: int) -> float:
    hits = 0
    total = 0
    for step in range(2, max(3, length)):
        px = x + dx * step
        py = y + dy * step
        if px < 1 or py < 1 or px >= edge.shape[1] - 1 or py >= edge.shape[0] - 1:
            break
        hits += 1 if np.max(edge[py - 1 : py + 2, px - 1 : px + 2]) > 0 else 0
        total += 1
    return float(hits / total) if total else 0.0


def sample_luma_patch(luma: np.ndarray, x: int, y: int, radius: int) -> tuple[float, float] | None:
    x0 = max(0, int(round(x)) - radius)
    y0 = max(0, int(round(y)) - radius)
    x1 = min(luma.shape[1], int(round(x)) + radius + 1)
    y1 = min(luma.shape[0], int(round(y)) + radius + 1)
    patch = luma[y0:y1, x0:x1]
    if patch.size < 12:
        return None
    return float(np.median(patch)), float(np.std(patch))


def corner_opposite_rects(
    size: ImageSize,
    first: list[CornerCandidate],
    second: list[CornerCandidate],
    ratios: tuple[float, ...],
) -> list[tuple[float, ImageRect]]:
    scored: list[tuple[float, ImageRect]] = []
    for a in first:
        for b in second:
            left = min(a.x, b.x)
            right = max(a.x, b.x)
            top = min(a.y, b.y)
            bottom = max(a.y, b.y)
            width = right - left
            height = bottom - top
            if width < size.width * 0.24 or height < size.height * 0.24:
                continue
            box_area = width * height
            area_ratio = box_area / max(1.0, float(size.width * size.height))
            if area_ratio < 0.10 or area_ratio > 0.94:
                continue
            box_aspect = width / max(height, 1)
            for ratio in ratios:
                rect = fit_format_rect_to_boundary_box(
                    size,
                    left=left,
                    right=right,
                    top=top,
                    bottom=bottom,
                    aspect=ratio,
                    angle=0.0,
                )
                format_score = aspect_match_score(box_aspect, ratio)
                fill_ratio = (rect.width * rect.height) / max(1.0, float(box_area))
                score = (a.score + b.score) * 0.38 + format_score * 0.25 + fill_ratio * 0.22 + center_prior_from_rect(rect, size) * 0.15
                scored.append((float(score), rect))
    return scored


def corner_same_edge_rects(
    size: ImageSize,
    first: list[CornerCandidate],
    second: list[CornerCandidate],
    ratios: tuple[float, ...],
    *,
    edge_name: str,
) -> list[tuple[float, ImageRect]]:
    scored: list[tuple[float, ImageRect]] = []
    center_x = size.width * 0.5
    center_y = size.height * 0.5
    for a in first:
        for b in second:
            if edge_name in {"top", "bottom"}:
                span = abs(b.x - a.x)
                edge_pos = int(round((a.y + b.y) * 0.5))
                left = min(a.x, b.x)
                if span < size.width * 0.24:
                    continue
                for ratio in ratios:
                    height = span / max(ratio, 1e-5)
                    top = edge_pos if edge_name == "top" else edge_pos - height
                    bottom = top + height
                    if not (top <= center_y <= bottom):
                        continue
                    rect = clamp_rect(
                        ImageRect(x=int(round(left)), y=int(round(top)), width=int(round(span)), height=int(round(height))),
                        size,
                    )
                    scored.append((same_edge_rect_score(a, b, rect, size, ratio), rect))
            else:
                span = abs(b.y - a.y)
                edge_pos = int(round((a.x + b.x) * 0.5))
                top = min(a.y, b.y)
                if span < size.height * 0.24:
                    continue
                for ratio in ratios:
                    width = span * ratio
                    left = edge_pos if edge_name == "left" else edge_pos - width
                    right = left + width
                    if not (left <= center_x <= right):
                        continue
                    rect = clamp_rect(
                        ImageRect(x=int(round(left)), y=int(round(top)), width=int(round(width)), height=int(round(span))),
                        size,
                    )
                    scored.append((same_edge_rect_score(a, b, rect, size, ratio), rect))
    return scored


def same_edge_rect_score(
    a: CornerCandidate,
    b: CornerCandidate,
    rect: ImageRect,
    size: ImageSize,
    ratio: float,
) -> float:
    aspect = rect.width / max(rect.height, 1)
    return float(
        (a.score + b.score) * 0.42
        + aspect_match_score(aspect, ratio) * 0.25
        + center_prior_from_rect(rect, size) * 0.18
        + border_clearance(rect, size) * 0.15
    )


def generate_boundary_candidates(
    image_features: ImageFeatures,
    *,
    format_hint: str,
    count: int,
) -> list[ImageRect]:
    if count <= 0:
        return []

    size = image_features.size
    gray = normalized_uint8(image_features.luma).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    col_profile = smooth_1d_signal(np.percentile(gray, 58.0, axis=0), max(5, size.width // 42))
    row_profile = smooth_1d_signal(np.percentile(gray, 58.0, axis=1), max(5, size.height // 42))

    left_lines = boundary_lines(col_profile, side="left", max_candidates=6)
    right_lines = boundary_lines(col_profile, side="right", max_candidates=6)
    top_lines = boundary_lines(row_profile, side="top", max_candidates=6)
    bottom_lines = boundary_lines(row_profile, side="bottom", max_candidates=6)
    if not left_lines or not right_lines or not top_lines or not bottom_lines:
        return []

    x_pairs = boundary_pairs(
        left_lines,
        right_lines,
        length=size.width,
        min_span=max(16, int(size.width * 0.24)),
        max_pairs=14,
    )
    y_pairs = boundary_pairs(
        top_lines,
        bottom_lines,
        length=size.height,
        min_span=max(16, int(size.height * 0.24)),
        max_pairs=14,
    )
    if not x_pairs or not y_pairs:
        return []

    ratios = format_ratios_for_hint(format_hint)
    angle_candidates = estimate_frame_angles(image_features.edge)
    candidates: list[tuple[float, ImageRect]] = []
    for left, right, x_score in x_pairs:
        for top, bottom, y_score in y_pairs:
            box_width = right - left
            box_height = bottom - top
            if box_width <= 0 or box_height <= 0:
                continue
            box_area = float(box_width * box_height)
            area_ratio = box_area / max(1.0, float(size.width * size.height))
            if area_ratio < 0.10 or area_ratio > 0.92:
                continue

            box_aspect = box_width / max(box_height, 1)
            for ratio in ratios:
                aspect_match = aspect_match_score(box_aspect, ratio)
                rect = fit_format_rect_to_boundary_box(
                    size,
                    left=left,
                    right=right,
                    top=top,
                    bottom=bottom,
                    aspect=ratio,
                    angle=0.0,
                )
                fill_ratio = (rect.width * rect.height) / max(1.0, box_area)
                if fill_ratio < 0.52:
                    continue
                center_score = center_prior_from_rect(rect, size)
                for angle in angle_candidates:
                    angled = ImageRect(
                        x=rect.x,
                        y=rect.y,
                        width=rect.width,
                        height=rect.height,
                        angle=normalize_angle(angle),
                    )
                    score = (
                        (x_score + y_score) * 0.44
                        + aspect_match * 0.22
                        + fill_ratio * 0.16
                        + center_score * 0.10
                        + (1.0 - min(abs(angle) / 10.0, 1.0)) * 0.08
                    )
                    candidates.append((float(score), angled))

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return dedupe_rects([rect for _score, rect in candidates])[:count]


def smooth_1d_signal(signal: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if values.size == 0 or window <= 1:
        return values
    window = int(max(3, min(window, values.size)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def boundary_lines(profile: np.ndarray, *, side: str, max_candidates: int) -> list[BoundaryLineCandidate]:
    length = int(profile.size)
    if length < 32:
        return []
    window = max(3, int(round(length * 0.022)))
    margin = max(window * 3, int(round(length * 0.035)))
    if side in {"left", "top"}:
        start = margin
        stop = max(start + 1, int(round(length * 0.58)))
        outside_before = True
    else:
        start = min(length - margin - 1, int(round(length * 0.42)))
        stop = length - margin
        outside_before = False

    start = max(window * 3, min(start, length - window * 3 - 1))
    stop = max(start + 1, min(stop, length - window * 3))
    scores = np.zeros(length, dtype=np.float32)
    for position in range(start, stop):
        before = profile[position - window * 3 : position - window]
        after = profile[position + window : position + window * 3]
        if before.size == 0 or after.size == 0:
            continue
        before_mean = float(np.mean(before))
        after_mean = float(np.mean(after))
        if outside_before:
            outside = before_mean
            inside = after_mean
        else:
            outside = after_mean
            inside = before_mean
        contrast = outside - inside
        gradient = abs(after_mean - before_mean)
        scores[position] = max(0.0, contrast) * 0.78 + gradient * 0.22

    search = scores[start:stop]
    if search.size == 0:
        return []
    floor = max(float(np.percentile(search, 82.0)) * 1.15, 0.030)
    separation = max(4, int(round(length * 0.035)))
    ranked = np.argsort(search)[::-1]
    candidates: list[BoundaryLineCandidate] = []
    for offset in ranked:
        position = int(start + offset)
        score = float(scores[position])
        if score < floor:
            break
        if any(abs(position - candidate.position) < separation for candidate in candidates):
            continue
        candidates.append(BoundaryLineCandidate(position=position, score=score))
        if len(candidates) >= max_candidates:
            break
    return candidates


def boundary_pairs(
    first_lines: list[BoundaryLineCandidate],
    second_lines: list[BoundaryLineCandidate],
    *,
    length: int,
    min_span: int,
    max_pairs: int,
) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    max_span = int(round(length * 0.96))
    for first in first_lines:
        for second in second_lines:
            span = second.position - first.position
            if span < min_span or span > max_span:
                continue
            span_ratio = span / max(1.0, float(length))
            span_prior = float(1.0 - np.clip(abs(span_ratio - 0.68) / 0.52, 0.0, 1.0))
            score = (first.score + second.score) * 0.5 + span_prior * 0.10
            pairs.append((first.position, second.position, float(score)))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:max_pairs]


def fit_format_rect_to_boundary_box(
    size: ImageSize,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    aspect: float,
    angle: float,
) -> ImageRect:
    box_width = max(1.0, float(right - left))
    box_height = max(1.0, float(bottom - top))
    aspect = max(0.05, float(aspect))
    if box_width / box_height >= aspect:
        height = box_height
        width = height * aspect
    else:
        width = box_width
        height = width / aspect
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    return clamp_rect(
        ImageRect(
            x=int(round(center_x - width * 0.5)),
            y=int(round(center_y - height * 0.5)),
            width=max(12, int(round(width))),
            height=max(12, int(round(height))),
            angle=normalize_angle(angle),
        ),
        size,
    )


def estimate_frame_angles(edge: np.ndarray) -> tuple[float, ...]:
    height, width = edge.shape[:2]
    min_side = min(width, height)
    threshold = max(16, int(round(min_side * 0.14)))
    min_length = max(18, int(round(min_side * 0.20)))
    max_gap = max(4, int(round(min_side * 0.035)))
    lines = cv2.HoughLinesP(
        edge,
        1,
        np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if lines is None:
        return (0.0,)

    angles: list[float] = []
    weights: list[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_length:
            continue
        angle = normalize_angle(float(np.degrees(np.arctan2(dy, dx))))
        if abs(angle) > 12.0:
            continue
        angles.append(angle)
        weights.append(length)
    if not angles:
        return (0.0,)

    estimate = float(np.average(np.asarray(angles, dtype=np.float32), weights=np.asarray(weights, dtype=np.float32)))
    values = [0.0]
    if abs(estimate) >= 0.8:
        values.extend([estimate, estimate - 1.5, estimate + 1.5])
    deduped: list[float] = []
    for value in values:
        if abs(value) > 12.0:
            continue
        if not any(abs(value - existing) < 0.5 for existing in deduped):
            deduped.append(float(value))
    return tuple(deduped) if deduped else (0.0,)


def aspect_match_score(aspect: float, target: float) -> float:
    delta = abs(np.log(max(aspect, 0.05)) - np.log(max(target, 0.05)))
    return float(1.0 - np.clip(delta / 0.42, 0.0, 1.0))


def center_prior_from_rect(rect: ImageRect, size: ImageSize) -> float:
    dx = abs(rect.center_x / max(1.0, size.width) - 0.5) * 2.0
    dy = abs(rect.center_y / max(1.0, size.height) - 0.5) * 2.0
    return float(1.0 - np.clip((dx + dy) / 1.45, 0.0, 1.0))


def dedupe_rects(rects: list[ImageRect]) -> list[ImageRect]:
    deduped: list[ImageRect] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for rect in rects:
        key = (
            int(round(rect.x)),
            int(round(rect.y)),
            int(round(rect.width)),
            int(round(rect.height)),
            int(round(rect.angle * 10.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rect)
    return deduped


def generate_global_candidates(
    size: ImageSize,
    *,
    count: int,
    rng: np.random.Generator,
    format_hint: str = "auto",
) -> list[ImageRect]:
    rects: list[ImageRect] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    ratios = format_ratios_for_hint(format_hint)

    def append_candidate(ratio: float, area_ratio: float, cx_norm: float, cy_norm: float, angle: float) -> None:
        if len(rects) >= count:
            return
        rect = rect_from_format(size, ratio, area_ratio, cx_norm, cy_norm, angle)
        key = (rect.x, rect.y, rect.width, rect.height, int(round(rect.angle * 10.0)))
        if key in seen:
            return
        seen.add(key)
        rects.append(rect)

    priority_centers = (
        (0.50, 0.50),
        (0.485, 0.50),
        (0.515, 0.50),
        (0.50, 0.485),
        (0.50, 0.515),
        (0.47, 0.50),
        (0.53, 0.50),
        (0.50, 0.47),
        (0.50, 0.53),
    )
    priority_areas = (0.72, 0.78, 0.66, 0.60, 0.84, 0.54, 0.48)
    priority_angles = (0.0, -2.5, 2.5, -5.0, 5.0)
    for cx_norm, cy_norm in priority_centers:
        for area_ratio in priority_areas:
            for angle in priority_angles:
                for ratio in ratios:
                    append_candidate(ratio, area_ratio, cx_norm, cy_norm, angle)
                    if len(rects) >= count:
                        return rects

    area_values = (0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78, 0.84)
    center_x_values = np.linspace(0.28, 0.72, 9)
    center_y_values = np.linspace(0.28, 0.72, 9)
    angle_values = (-8.0, -5.0, -2.5, 0.0, 2.5, 5.0, 8.0)

    deterministic: list[tuple[float, float, float, float, float]] = []
    for cx_norm in center_x_values:
        for cy_norm in center_y_values:
            for area_ratio in area_values:
                for angle in angle_values:
                    for ratio in ratios:
                        deterministic.append((ratio, area_ratio, float(cx_norm), float(cy_norm), angle))
    rng.shuffle(deterministic)
    for ratio, area_ratio, cx_norm, cy_norm, angle in deterministic[:count]:
        append_candidate(ratio, area_ratio, cx_norm, cy_norm, angle)
        if len(rects) >= count:
            return rects

    while len(rects) < count:
        ratio = float(rng.choice(ratios))
        if rng.random() < 0.06:
            ratio = 1.0 / ratio
        area_ratio = float(rng.uniform(0.34, 0.84))
        cx_norm = float(np.clip(rng.normal(0.50, 0.13), 0.18, 0.82))
        cy_norm = float(np.clip(rng.normal(0.50, 0.13), 0.18, 0.82))
        angle = float(rng.uniform(-8.0, 8.0))
        append_candidate(ratio, area_ratio, cx_norm, cy_norm, angle)
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
    inside_luma = features[:, index["inside_luma_p50"]]
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
    base_to_negative_transition = (
        np.clip((outside_luma - inside_luma + 0.015) / 0.24, 0.0, 1.0) * 0.58
        + np.clip(edge_support / 0.32, 0.0, 1.0) * 0.22
        + inside_content * 0.20
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
        np.maximum(base_ring_score * 0.62 + base_to_negative_transition * 0.38, tight_crop_score * 0.82) * 0.46
        + geometry_score * 0.34
        + inside_content * 0.20
        - oversize_touch * 0.18
        - empty_inside * 0.12
    )

    keep_count = min(len(rects), keep)
    selected = np.argsort(score)[-keep_count:][::-1].astype(np.int32)
    return [rects[int(index)] for index in selected], features[selected], selected
