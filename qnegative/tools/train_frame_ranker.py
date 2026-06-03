from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.ensemble import ExtraTreesRegressor

from qnegative.core.geometry import rotated_rect_corners, scale_rect
from qnegative.core.models import ImageRect, ImageSize
from qnegative.core.preview import make_raw_preview


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
class FrameLabel:
    name: str
    negative_path: Path
    source_size: ImageSize
    rect_source: ImageRect
    confidence: float
    format_hint: str


@dataclass(frozen=True)
class ImageFeatures:
    rgb: np.ndarray
    luma: np.ndarray
    chroma: np.ndarray
    edge: np.ndarray
    size: ImageSize


@dataclass(frozen=True)
class CandidateSet:
    rects: list[ImageRect]
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight frame candidate ranker.")
    parser.add_argument("--labels", type=Path, default=Path("calibration/frame_labels_expanded.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("calibration/frame_ranker_smoke"))
    parser.add_argument("--model-out", type=Path, default=Path("models/frame_ranker.joblib"))
    parser.add_argument("--max-labels", type=int, default=80)
    parser.add_argument("--preview-max-size", type=int, default=640)
    parser.add_argument("--candidates-per-image", type=int, default=140)
    parser.add_argument("--global-candidates", type=int, default=2400)
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--debug-count", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_state)
    labels = read_labels(args.labels, min_confidence=args.min_confidence)
    if args.max_labels > 0:
        labels = labels[: args.max_labels]
    if len(labels) < 12:
        raise SystemExit("Need at least 12 labels for a useful smoke test.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    order = rng.permutation(len(labels))
    test_count = max(4, int(round(len(labels) * args.test_ratio)))
    test_indices = set(int(index) for index in order[:test_count])
    train_labels = [label for index, label in enumerate(labels) if index not in test_indices]
    test_labels = [label for index, label in enumerate(labels) if index in test_indices]

    print(f"Labels: {len(labels)} train={len(train_labels)} test={len(test_labels)}")
    x_train, y_train, w_train = build_split(
        train_labels,
        preview_max_size=args.preview_max_size,
        candidates_per_image=args.candidates_per_image,
        augmentations=args.augmentations,
        rng=rng,
    )
    print(f"Train samples: {len(y_train)}")

    model = ExtraTreesRegressor(
        n_estimators=160,
        max_depth=18,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, sample_weight=w_train)

    report, debug_items = evaluate_model(
        model,
        test_labels,
        preview_max_size=args.preview_max_size,
        candidates_per_image=args.candidates_per_image,
        global_candidates=args.global_candidates,
        rng=np.random.default_rng(args.random_state + 99),
    )
    report.update(
        {
            "labels": len(labels),
            "train_labels": len(train_labels),
            "test_labels": len(test_labels),
            "train_samples": int(len(y_train)),
            "preview_max_size": args.preview_max_size,
            "candidates_per_image": args.candidates_per_image,
            "global_candidates": args.global_candidates,
            "augmentations": args.augmentations,
            "model": "ExtraTreesRegressor",
            "feature_names": FEATURE_NAMES,
        }
    )

    (args.out_dir / "frame_ranker_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    draw_debug_sheet(debug_items[: args.debug_count], args.out_dir / "topk_debug.jpg")
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "metadata": report,
        },
        args.model_out,
    )

    print("\nFrame ranker smoke report")
    print(f"Top1 mean IoU: {report['top1_mean_iou']:.3f}")
    print(f"Top3 mean best IoU: {report['top3_mean_best_iou']:.3f}")
    print(f"Top1 >= 0.85: {report['top1_iou_ge_085_rate']:.3f}")
    print(f"Top3 >= 0.85: {report['top3_iou_ge_085_rate']:.3f}")
    print(f"Oracle top mean IoU: {report['oracle_top_mean_iou']:.3f}")
    print(f"Report: {args.out_dir / 'frame_ranker_report.json'}")
    print(f"Debug: {args.out_dir / 'topk_debug.jpg'}")
    print(f"Model: {args.model_out}")
    return 0


def read_labels(path: Path, *, min_confidence: float) -> list[FrameLabel]:
    labels: list[FrameLabel] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            if not data.get("accepted", True):
                continue
            confidence = float(data.get("confidence", 1.0))
            if confidence < min_confidence:
                continue
            rect_data = data["frame_rect_source"]
            source_width, source_height = data["source_size"]
            label = FrameLabel(
                name=str(data["name"]),
                negative_path=Path(data["negative_path"]),
                source_size=ImageSize(width=int(source_width), height=int(source_height)),
                rect_source=ImageRect(
                    x=int(round(rect_data["x"])),
                    y=int(round(rect_data["y"])),
                    width=int(round(rect_data["width"])),
                    height=int(round(rect_data["height"])),
                    angle=float(rect_data.get("angle", 0.0)),
                ),
                confidence=confidence,
                format_hint=str(data.get("format", "auto")),
            )
            if label.negative_path.exists():
                labels.append(label)
    labels.sort(key=lambda item: item.name)
    return labels


def build_split(
    labels: list[FrameLabel],
    *,
    preview_max_size: int,
    candidates_per_image: int,
    augmentations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for label in labels:
        for variant_index, (image, gt_rect, weight_scale) in enumerate(
            image_variants(label, preview_max_size=preview_max_size, augmentations=augmentations, rng=rng)
        ):
            del variant_index
            image_features = prepare_image_features(image)
            candidate_set = build_candidate_set(
                image_features,
                gt_rect,
                label,
                candidates_per_image=candidates_per_image,
                weight_scale=weight_scale,
                rng=rng,
            )
            features.append(candidate_set.x)
            targets.append(candidate_set.y)
            weights.append(candidate_set.weights)
    return (
        np.vstack(features).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
        np.concatenate(weights).astype(np.float32),
    )


def image_variants(
    label: FrameLabel,
    *,
    preview_max_size: int,
    augmentations: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, ImageRect, float]]:
    preview = make_raw_preview(label.negative_path, max_size=preview_max_size)
    gt_rect = scale_rect(label.rect_source, label.source_size, preview.preview_size)
    variants = [(preview.preview_linear_rgb, gt_rect, label_weight(label.confidence))]

    if label.confidence < 0.90:
        return variants

    for _index in range(max(0, augmentations)):
        angle = float(rng.uniform(-8.0, 8.0))
        scale = float(rng.uniform(0.975, 1.025))
        tx = float(rng.uniform(-0.025, 0.025) * preview.preview_size.width)
        ty = float(rng.uniform(-0.025, 0.025) * preview.preview_size.height)
        aug_image, aug_rect = rotate_image_and_rect(
            preview.preview_linear_rgb,
            gt_rect,
            angle=angle,
            scale=scale,
            translate=(tx, ty),
        )
        variants.append((aug_image, aug_rect, label_weight(label.confidence) * 0.90))
    return variants


def label_weight(confidence: float) -> float:
    if confidence >= 0.95:
        return 1.0
    if confidence >= 0.85:
        return 0.72
    if confidence >= 0.70:
        return 0.42
    return 0.20


def rotate_image_and_rect(
    image: np.ndarray,
    rect: ImageRect,
    *,
    angle: float,
    scale: float,
    translate: tuple[float, float],
) -> tuple[np.ndarray, ImageRect]:
    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += translate[0]
    matrix[1, 2] += translate[1]
    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    corners = rotated_rect_corners(rect)
    homogeneous = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    transformed_corners = homogeneous @ matrix.T
    transformed_rect = quad_to_image_rect(transformed_corners.astype(np.float32))
    return np.ascontiguousarray(transformed_image), transformed_rect


def quad_to_image_rect(quad: np.ndarray) -> ImageRect:
    rect = cv2.minAreaRect(quad.astype(np.float32).reshape(-1, 1, 2))
    (center_x, center_y), (width, height), angle = rect
    if width < height:
        width, height = height, width
        angle += 90.0
    angle = normalize_angle(angle)
    return ImageRect(
        x=int(round(center_x - width / 2.0)),
        y=int(round(center_y - height / 2.0)),
        width=max(1, int(round(width))),
        height=max(1, int(round(height))),
        angle=angle,
    )


def normalize_angle(angle: float) -> float:
    normalized = float(angle)
    while normalized > 45.0:
        normalized -= 90.0
    while normalized <= -45.0:
        normalized += 90.0
    return normalized


def prepare_image_features(image: np.ndarray) -> ImageFeatures:
    rgb = np.clip(np.nan_to_num(image.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
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
    return np.ascontiguousarray((np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def build_candidate_set(
    image_features: ImageFeatures,
    gt_rect: ImageRect,
    label: FrameLabel,
    *,
    candidates_per_image: int,
    weight_scale: float,
    rng: np.random.Generator,
) -> CandidateSet:
    rects = generate_candidates(gt_rect, image_features.size, candidates_per_image=candidates_per_image, rng=rng)
    x = np.vstack([extract_candidate_features(image_features, rect) for rect in rects]).astype(np.float32)
    y = np.array([rect_iou(rect, gt_rect) for rect in rects], dtype=np.float32)
    weights = np.full(len(rects), weight_scale, dtype=np.float32)
    weights *= np.where(y >= 0.80, 1.6, np.where(y <= 0.20, 0.75, 1.0)).astype(np.float32)
    del label
    return CandidateSet(rects=rects, x=x, y=y, weights=weights)


def generate_candidates(
    gt_rect: ImageRect,
    size: ImageSize,
    *,
    candidates_per_image: int,
    rng: np.random.Generator,
) -> list[ImageRect]:
    rects: list[ImageRect] = [gt_rect]
    high_count = int(candidates_per_image * 0.34)
    mid_count = int(candidates_per_image * 0.24)
    inner_outer_count = int(candidates_per_image * 0.18)
    remaining = max(0, candidates_per_image - len(rects) - high_count - mid_count - inner_outer_count)

    rects.extend(jitter_rects(gt_rect, high_count, size, rng, center_sigma=0.018, scale_sigma=0.035, angle_sigma=2.2))
    rects.extend(jitter_rects(gt_rect, mid_count, size, rng, center_sigma=0.075, scale_sigma=0.12, angle_sigma=7.0))
    rects.extend(scale_error_rects(gt_rect, inner_outer_count, size, rng))
    rects.extend(random_format_rects(remaining, size, rng))
    return rects[:candidates_per_image]


def jitter_rects(
    rect: ImageRect,
    count: int,
    size: ImageSize,
    rng: np.random.Generator,
    *,
    center_sigma: float,
    scale_sigma: float,
    angle_sigma: float,
) -> list[ImageRect]:
    rects: list[ImageRect] = []
    for _index in range(count):
        dx = float(rng.normal(0.0, center_sigma) * size.width)
        dy = float(rng.normal(0.0, center_sigma) * size.height)
        scale_w = float(np.exp(rng.normal(0.0, scale_sigma)))
        scale_h = float(np.exp(rng.normal(0.0, scale_sigma)))
        width = max(10, int(round(rect.width * scale_w)))
        height = max(10, int(round(rect.height * scale_h)))
        cx = rect.center_x + dx
        cy = rect.center_y + dy
        angle = normalize_angle(rect.angle + float(rng.normal(0.0, angle_sigma)))
        rects.append(clamp_rect(ImageRect(int(round(cx - width / 2)), int(round(cy - height / 2)), width, height, angle), size))
    return rects


def scale_error_rects(rect: ImageRect, count: int, size: ImageSize, rng: np.random.Generator) -> list[ImageRect]:
    rects: list[ImageRect] = []
    for index in range(count):
        if index % 2 == 0:
            scale = float(rng.uniform(0.62, 0.92))
        else:
            scale = float(rng.uniform(1.08, 1.34))
        width = max(10, int(round(rect.width * scale * rng.uniform(0.94, 1.06))))
        height = max(10, int(round(rect.height * scale * rng.uniform(0.94, 1.06))))
        dx = float(rng.normal(0.0, 0.035) * size.width)
        dy = float(rng.normal(0.0, 0.035) * size.height)
        angle = normalize_angle(rect.angle + float(rng.normal(0.0, 4.0)))
        candidate = ImageRect(
            x=int(round(rect.center_x + dx - width / 2)),
            y=int(round(rect.center_y + dy - height / 2)),
            width=width,
            height=height,
            angle=angle,
        )
        rects.append(clamp_rect(candidate, size))
    return rects


def random_format_rects(count: int, size: ImageSize, rng: np.random.Generator) -> list[ImageRect]:
    rects: list[ImageRect] = []
    ratios = list(FORMAT_RATIOS.values())
    for _index in range(count):
        aspect = float(rng.choice(ratios))
        if rng.random() < 0.08:
            aspect = 1.0 / aspect
        area_ratio = float(rng.uniform(0.10, 0.86))
        area = size.width * size.height * area_ratio
        width = int(round(math.sqrt(area * aspect)))
        height = int(round(math.sqrt(area / aspect)))
        width = max(12, min(width, size.width))
        height = max(12, min(height, size.height))
        cx = float(rng.uniform(width * 0.20, size.width - width * 0.20))
        cy = float(rng.uniform(height * 0.20, size.height - height * 0.20))
        angle = normalize_angle(float(rng.uniform(-13.0, 13.0)))
        rects.append(clamp_rect(ImageRect(int(round(cx - width / 2)), int(round(cy - height / 2)), width, height, angle), size))
    return rects


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
        format_score(aspect),
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


def format_score(aspect: float) -> float:
    best = 0.0
    for target in FORMAT_RATIOS.values():
        delta = abs(math.log(max(aspect, 0.05)) - math.log(target))
        best = max(best, float(1.0 - np.clip(delta / 0.36, 0.0, 1.0)))
    return best


def rect_iou(a: ImageRect, b: ImageRect) -> float:
    poly_a = rotated_rect_corners(a).astype(np.float32)
    poly_b = rotated_rect_corners(b).astype(np.float32)
    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    intersection_area, _intersection_poly = cv2.intersectConvexConvex(poly_a, poly_b)
    intersection = max(0.0, float(intersection_area))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 1e-6 else 0.0


def evaluate_model(
    model: ExtraTreesRegressor,
    labels: list[FrameLabel],
    *,
    preview_max_size: int,
    candidates_per_image: int,
    global_candidates: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    top1: list[float] = []
    top3: list[float] = []
    oracle: list[float] = []
    raw_oracle: list[float] = []
    debug_items: list[dict[str, object]] = []

    for label in labels:
        preview = make_raw_preview(label.negative_path, max_size=preview_max_size)
        gt_rect = scale_rect(label.rect_source, label.source_size, preview.preview_size)
        image_features = prepare_image_features(preview.preview_linear_rgb)
        raw_rects = generate_global_candidates(
            image_features.size,
            count=max(global_candidates, candidates_per_image * 8, 640),
            rng=rng,
        )
        x_all = np.vstack([extract_candidate_features(image_features, rect) for rect in raw_rects]).astype(np.float32)
        y_all = np.array([rect_iou(rect, gt_rect) for rect in raw_rects], dtype=np.float32)
        raw_oracle_iou = float(np.max(y_all))
        rects, x, kept_indices = prefilter_candidates(
            raw_rects,
            x_all,
            keep=max(candidates_per_image * 4, 420),
        )
        y = y_all[kept_indices]
        prediction = model.predict(x)
        order = np.argsort(prediction)[::-1]
        best_indices = order[:3]
        top1_iou = float(y[order[0]])
        top3_iou = float(np.max(y[best_indices]))
        oracle_iou = float(np.max(y))
        top1.append(top1_iou)
        top3.append(top3_iou)
        oracle.append(oracle_iou)
        raw_oracle.append(raw_oracle_iou)
        debug_items.append(
            {
                "name": label.name,
                "image_rgb8": linear_to_preview_rgb8(preview.preview_linear_rgb),
                "gt": gt_rect,
                "rects": [rects[int(index)] for index in best_indices],
                "scores": [float(prediction[int(index)]) for index in best_indices],
                "ious": [float(y[int(index)]) for index in best_indices],
                "top1_iou": top1_iou,
            }
        )

    top1_arr = np.array(top1, dtype=np.float32)
    top3_arr = np.array(top3, dtype=np.float32)
    oracle_arr = np.array(oracle, dtype=np.float32)
    raw_oracle_arr = np.array(raw_oracle, dtype=np.float32)
    return (
        {
            "top1_mean_iou": float(np.mean(top1_arr)),
            "top1_median_iou": float(np.median(top1_arr)),
            "top3_mean_best_iou": float(np.mean(top3_arr)),
            "top3_median_best_iou": float(np.median(top3_arr)),
            "oracle_top_mean_iou": float(np.mean(oracle_arr)),
            "raw_oracle_top_mean_iou": float(np.mean(raw_oracle_arr)),
            "top1_iou_ge_080_rate": float(np.mean(top1_arr >= 0.80)),
            "top1_iou_ge_085_rate": float(np.mean(top1_arr >= 0.85)),
            "top1_iou_ge_090_rate": float(np.mean(top1_arr >= 0.90)),
            "top3_iou_ge_080_rate": float(np.mean(top3_arr >= 0.80)),
            "top3_iou_ge_085_rate": float(np.mean(top3_arr >= 0.85)),
            "top3_iou_ge_090_rate": float(np.mean(top3_arr >= 0.90)),
        },
        sorted(debug_items, key=lambda item: float(item["top1_iou"])),
    )


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
    width = int(round(math.sqrt(area * aspect)))
    height = int(round(math.sqrt(area / aspect)))
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


def linear_to_preview_rgb8(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    sample = clipped.reshape(-1, 3)
    low = float(np.percentile(sample, 0.5))
    high = float(np.percentile(sample, 99.5))
    if high <= low + 1e-6:
        low, high = 0.0, 1.0
    normalized = np.clip((clipped - low) / (high - low), 0.0, 1.0)
    display = np.power(normalized, 1.0 / 2.2)
    return np.ascontiguousarray((display * 255.0 + 0.5).astype(np.uint8))


def draw_debug_sheet(items: list[dict[str, object]], path: Path) -> None:
    if not items:
        return
    tile_w, tile_h = 420, 300
    columns = 3
    rows = int(math.ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (18, 20, 24))
    for index, item in enumerate(items):
        image = Image.fromarray(item["image_rgb8"]).convert("RGB")
        image.thumbnail((tile_w, tile_h - 36), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), (18, 20, 24))
        offset = ((tile_w - image.width) // 2, 0)
        tile.paste(image, offset)
        draw = ImageDraw.Draw(tile, "RGBA")
        scale_x = image.width / item["image_rgb8"].shape[1]
        scale_y = image.height / item["image_rgb8"].shape[0]
        draw_rect(draw, item["gt"], offset, scale_x, scale_y, (255, 220, 90, 230), width=3)
        colors = [(82, 207, 174, 230), (95, 164, 255, 210), (235, 120, 120, 210)]
        for rect, color in zip(item["rects"], colors):
            draw_rect(draw, rect, offset, scale_x, scale_y, color, width=2)
        label = f"{item['name']} top1={item['top1_iou']:.3f}"
        draw.rectangle((0, tile_h - 32, tile_w, tile_h), fill=(18, 20, 24, 230))
        draw.text((8, tile_h - 23), label, fill=(235, 240, 246, 255))
        sheet.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))
    sheet.save(path, quality=92)


def draw_rect(draw: ImageDraw.ImageDraw, rect: ImageRect, offset: tuple[int, int], scale_x: float, scale_y: float, color: tuple[int, int, int, int], *, width: int) -> None:
    points = rotated_rect_corners(rect)
    mapped = [
        (float(x) * scale_x + offset[0], float(y) * scale_y + offset[1])
        for x, y in points
    ]
    draw.line(mapped + [mapped[0]], fill=color, width=width)


if __name__ == "__main__":
    raise SystemExit(main())
