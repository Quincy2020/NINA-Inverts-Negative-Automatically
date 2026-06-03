from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from qnegative.core.geometry import scale_rect
from qnegative.core.file_sequence import IMAGE_EXTENSIONS, RAW_EXTENSIONS
from qnegative.core.models import ImageRect, ImageSize
from qnegative.core.preview import make_raw_preview
from qnegative.tools.calibrate_reference import (
    load_reference_srgb,
    prepare_negative_match_gray,
    prepare_reference_match_gray,
    estimate_homography,
)


FORMAT_RATIOS = {
    "135": 1.50,
    "645": 4.0 / 3.0,
    "66": 1.00,
    "67": 7.0 / 6.0,
    "69": 1.50,
}

REFERENCE_ORIENTATIONS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_h_rot90",
    "flip_h_rot180",
    "flip_h_rot270",
)


@dataclass(frozen=True)
class FrameLabel:
    name: str
    accepted: bool
    negative_path: str
    positive_path: str
    source_size: list[int]
    preview_size: list[int]
    frame_quad_preview: list[list[float]]
    frame_quad_source: list[list[float]]
    frame_rect_preview: dict[str, float]
    frame_rect_source: dict[str, float]
    format: str
    confidence: float
    quality: dict[str, float | int | str | bool]
    alignment: dict


@dataclass(frozen=True)
class PairPaths:
    name: str
    negative_path: Path
    positive_path: Path


@dataclass(frozen=True)
class PairDiscovery:
    pairs: list[PairPaths]
    negative_count: int
    positive_count: int
    matched_count: int
    duplicate_negative_stems: list[str]
    duplicate_positive_stems: list[str]
    negatives_without_positive: int
    positives_without_negative: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate frame labels by matching cropped positive references back to negative RAW previews."
    )
    parser.add_argument("--negative-dir", type=Path, default=Path("negative file"))
    parser.add_argument("--positive-dir", type=Path, default=Path("posituve file"))
    parser.add_argument("--out", type=Path, default=Path("calibration/frame_labels.jsonl"))
    parser.add_argument("--debug-dir", type=Path, default=Path("calibration/frame_label_debug"))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=1600)
    parser.add_argument("--min-inliers", type=int, default=40)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.22)
    parser.add_argument("--write-rejected", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover_pairs(args.negative_dir, args.positive_dir, start=args.start, end=args.end)
    pairs = discovery.pairs
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No matching negative/positive pairs found.")

    labels: list[FrameLabel] = []
    failures: list[dict[str, str]] = []
    accepted_overlays: list[Path] = []
    rejected_overlays: list[Path] = []

    with args.out.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            try:
                label, overlay_path = label_pair(
                    pair.name,
                    pair.negative_path,
                    pair.positive_path,
                    debug_dir=args.debug_dir,
                    max_size=args.max_size,
                    min_inliers=args.min_inliers,
                    min_inlier_ratio=args.min_inlier_ratio,
                    write_rejected=args.write_rejected,
                )
            except Exception as exc:
                print(f"{pair.name}: failed: {exc}")
                failures.append({"name": pair.name, "reason": str(exc)})
                continue

            labels.append(label)
            if label.accepted:
                handle.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
                accepted_overlays.append(overlay_path)
            elif overlay_path is not None:
                rejected_overlays.append(overlay_path)

            status = "accepted" if label.accepted else "rejected"
            print(
                f"{pair.name}: {status} conf={label.confidence:.3f} "
                f"inliers={label.quality['inliers']} ratio={label.quality['inlier_ratio']:.3f} "
                f"area={label.quality['area_ratio']:.3f} aspect={label.quality['aspect']:.3f} "
                f"format={label.format} reason={label.quality['reason']}"
            )

    summary = build_summary(labels, discovery=discovery, processed_pairs=len(pairs), failures=failures)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(accepted_overlays, args.debug_dir / "accepted_contact_sheet.jpg")
    if rejected_overlays:
        make_contact_sheet(rejected_overlays, args.debug_dir / "rejected_contact_sheet.jpg")

    print("\nFrame label dataset")
    print(f"Discovered negative RAW files: {discovery.negative_count}")
    print(f"Discovered positive image files: {discovery.positive_count}")
    print(f"Matched filename pairs: {discovery.matched_count}")
    print(f"Processed pairs: {len(pairs)}")
    print(f"Labels written: {summary['accepted']}")
    print(f"Rejected: {summary['rejected']}")
    print(f"JSONL: {args.out}")
    print(f"Summary: {summary_path}")
    print(f"Debug: {args.debug_dir}")
    return 0


def discover_pairs(negative_dir: Path, positive_dir: Path, *, start: str | None, end: str | None) -> PairDiscovery:
    raw_groups = group_files_by_stem(negative_dir, RAW_EXTENSIONS)
    positive_groups = group_files_by_stem(positive_dir, IMAGE_EXTENSIONS)

    raw_by_name, duplicate_negative = select_unique_raws(raw_groups)
    positive_by_name, duplicate_positive = select_unique_positives(positive_groups)

    duplicate_names = set(duplicate_negative) | set(duplicate_positive)
    names = sorted((set(raw_by_name) & set(positive_by_name)) - duplicate_names)
    if start is not None:
        names = [name for name in names if name >= start]
    if end is not None:
        names = [name for name in names if name <= end]

    pairs = [
        PairPaths(
            name=name,
            negative_path=raw_by_name[name],
            positive_path=positive_by_name[name],
        )
        for name in names
    ]
    return PairDiscovery(
        pairs=pairs,
        negative_count=sum(len(paths) for paths in raw_groups.values()),
        positive_count=sum(len(paths) for paths in positive_groups.values()),
        matched_count=len(pairs),
        duplicate_negative_stems=duplicate_negative,
        duplicate_positive_stems=duplicate_positive,
        negatives_without_positive=len(set(raw_by_name) - set(positive_by_name)),
        positives_without_negative=len(set(positive_by_name) - set(raw_by_name)),
    )


def group_files_by_stem(root: Path, extensions: set[str]) -> dict[str, list[Path]]:
    if not root.exists():
        return {}
    groups: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        groups.setdefault(path.stem, []).append(path)
    return groups


def select_unique_raws(groups: dict[str, list[Path]]) -> tuple[dict[str, Path], list[str]]:
    selected: dict[str, Path] = {}
    duplicates: list[str] = []
    for stem, paths in groups.items():
        raw_paths = sorted(paths, key=lambda item: str(item).lower())
        if len(raw_paths) == 1:
            selected[stem] = raw_paths[0]
        else:
            duplicates.append(stem)
    return selected, duplicates


def select_unique_positives(groups: dict[str, list[Path]]) -> tuple[dict[str, Path], list[str]]:
    extension_priority = {
        ".png": 0,
        ".tif": 1,
        ".tiff": 1,
        ".jpg": 2,
        ".jpeg": 2,
        ".webp": 3,
        ".bmp": 4,
    }
    selected: dict[str, Path] = {}
    duplicates: list[str] = []
    for stem, paths in groups.items():
        ranked = sorted(
            paths,
            key=lambda item: (extension_priority.get(item.suffix.lower(), 99), str(item).lower()),
        )
        best_priority = extension_priority.get(ranked[0].suffix.lower(), 99)
        tied = [
            path
            for path in ranked
            if extension_priority.get(path.suffix.lower(), 99) == best_priority
        ]
        if len(tied) == 1:
            selected[stem] = tied[0]
        else:
            duplicates.append(stem)
    return selected, duplicates


def label_pair(
    name: str,
    negative_path: Path,
    positive_path: Path,
    *,
    debug_dir: Path,
    max_size: int,
    min_inliers: int,
    min_inlier_ratio: float,
    write_rejected: bool,
) -> tuple[FrameLabel, Path | None]:
    preview = make_raw_preview(negative_path, max_size=max_size)
    reference_srgb = load_reference_srgb(positive_path, max_size=max_size)

    display_bgr = cv2.cvtColor(preview.display_rgb8, cv2.COLOR_RGB2BGR)
    source_gray = prepare_negative_match_gray(display_bgr)

    best: dict[str, object] | None = None
    errors: list[str] = []
    for orientation in REFERENCE_ORIENTATIONS:
        oriented_reference = orient_reference(reference_srgb, orientation)
        reference_gray = prepare_reference_match_gray(oriented_reference)
        try:
            homography, alignment = estimate_homography(source_gray, reference_gray)
        except Exception as exc:
            errors.append(f"{orientation}: {exc}")
            continue

        frame_quad_preview = reference_corners_to_source_quad(
            homography,
            reference_size=ImageSize(width=reference_gray.shape[1], height=reference_gray.shape[0]),
        )
        frame_rect_preview = quad_to_image_rect(frame_quad_preview)
        frame_quad_source = scale_quad(frame_quad_preview, preview.preview_size, preview.source_size)
        quality = score_label_geometry(
            frame_quad_preview,
            frame_rect_preview,
            preview.preview_size,
            alignment,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
        )
        alignment = dict(alignment)
        alignment["reference_orientation"] = orientation
        rank = float(quality["confidence"]) + (1.0 if quality["accepted"] else 0.0)
        if best is None or rank > float(best["rank"]):
            best = {
                "rank": rank,
                "orientation": orientation,
                "reference": oriented_reference,
                "homography": homography,
                "alignment": alignment,
                "quad_preview": frame_quad_preview,
                "rect_preview": frame_rect_preview,
                "quad_source": frame_quad_source,
                "quality": quality,
            }

    if best is None:
        raise RuntimeError("; ".join(errors[:4]) if errors else "No reference orientation matched.")

    homography = best["homography"]
    alignment = best["alignment"]
    frame_quad_preview = best["quad_preview"]
    frame_rect_preview = best["rect_preview"]
    frame_rect_source = scale_rect(frame_rect_preview, preview.preview_size, preview.source_size)
    frame_quad_source = best["quad_source"]
    quality = best["quality"]
    accepted = bool(quality["accepted"])
    confidence = float(quality["confidence"])
    matched_format = str(quality["format"])

    overlay_path: Path | None = None
    if accepted or write_rejected:
        overlay_path = debug_dir / f"{name}_{'accepted' if accepted else 'rejected'}_overlay.jpg"
        write_overlay(
            overlay_path,
            preview.display_rgb8,
            reference_srgb,
            frame_quad_preview,
            label=f"{name} {matched_format} {confidence:.2f} {alignment['reference_orientation']}",
            accepted=accepted,
        )
        write_alignment_debug(
            debug_dir / f"{name}_{'accepted' if accepted else 'rejected'}_alignment.jpg",
            preview.display_rgb8,
            best["reference"],
            homography,
        )

    label = FrameLabel(
        name=name,
        accepted=accepted,
        negative_path=str(negative_path),
        positive_path=str(positive_path),
        source_size=[preview.source_size.width, preview.source_size.height],
        preview_size=[preview.preview_size.width, preview.preview_size.height],
        frame_quad_preview=frame_quad_preview.round(3).tolist(),
        frame_quad_source=frame_quad_source.round(3).tolist(),
        frame_rect_preview=rect_to_dict(frame_rect_preview),
        frame_rect_source=rect_to_dict(frame_rect_source),
        format=matched_format,
        confidence=round(confidence, 4),
        quality=quality,
        alignment=alignment,
    )
    return label, overlay_path


def reference_corners_to_source_quad(homography_source_to_reference: np.ndarray, *, reference_size: ImageSize) -> np.ndarray:
    inverse = np.linalg.inv(homography_source_to_reference)
    corners = np.array(
        [
            [0.0, 0.0],
            [reference_size.width - 1.0, 0.0],
            [reference_size.width - 1.0, reference_size.height - 1.0],
            [0.0, reference_size.height - 1.0],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    source = cv2.perspectiveTransform(corners, inverse).reshape(-1, 2)
    return source.astype(np.float32)


def orient_reference(reference_srgb: np.ndarray, orientation: str) -> np.ndarray:
    image = reference_srgb
    if orientation.startswith("flip_h"):
        image = np.flip(image, axis=1)
    if "rot90" in orientation:
        image = np.rot90(image, k=1)
    elif "rot180" in orientation:
        image = np.rot90(image, k=2)
    elif "rot270" in orientation:
        image = np.rot90(image, k=3)
    return np.ascontiguousarray(image)


def quad_to_image_rect(quad: np.ndarray) -> ImageRect:
    rect = cv2.minAreaRect(quad.astype(np.float32).reshape(-1, 1, 2))
    (center_x, center_y), (width, height), angle = rect
    if width < height:
        width, height = height, width
        angle += 90.0
    angle = normalize_rect_angle(angle)
    return ImageRect(
        x=int(round(center_x - width / 2.0)),
        y=int(round(center_y - height / 2.0)),
        width=max(1, int(round(width))),
        height=max(1, int(round(height))),
        angle=angle,
    )


def normalize_rect_angle(angle: float) -> float:
    normalized = float(angle)
    while normalized > 45.0:
        normalized -= 90.0
    while normalized <= -45.0:
        normalized += 90.0
    return normalized


def scale_quad(quad: np.ndarray, from_size: ImageSize, to_size: ImageSize) -> np.ndarray:
    scaled = quad.astype(np.float32, copy=True)
    scaled[:, 0] *= to_size.width / max(1, from_size.width)
    scaled[:, 1] *= to_size.height / max(1, from_size.height)
    return scaled


def score_label_geometry(
    quad: np.ndarray,
    rect: ImageRect,
    preview_size: ImageSize,
    alignment: dict,
    *,
    min_inliers: int,
    min_inlier_ratio: float,
) -> dict[str, float | int | str | bool]:
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    image_area = float(preview_size.width * preview_size.height)
    area_ratio = area / max(image_area, 1.0)
    points_ok = bool(np.all(np.isfinite(quad)))
    margin_x = preview_size.width * 0.08
    margin_y = preview_size.height * 0.08
    inside_bounds = bool(
        np.all(quad[:, 0] >= -margin_x)
        and np.all(quad[:, 0] <= preview_size.width + margin_x)
        and np.all(quad[:, 1] >= -margin_y)
        and np.all(quad[:, 1] <= preview_size.height + margin_y)
    )
    convex = bool(cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)))
    inliers = int(alignment.get("inliers", 0))
    inlier_ratio = float(alignment.get("inlier_ratio", 0.0))
    aspect = max(rect.width / max(rect.height, 1), rect.height / max(rect.width, 1))
    format_label, aspect_score = closest_format(aspect)
    area_score = 1.0 - np.clip(abs(area_ratio - 0.58) / 0.58, 0.0, 1.0)
    inlier_score = np.clip(inliers / max(min_inliers * 3.0, 1.0), 0.0, 1.0)
    ratio_score = np.clip(inlier_ratio / max(min_inlier_ratio * 2.0, 1e-5), 0.0, 1.0)
    confidence = float(
        np.clip(
            inlier_score * 0.34
            + ratio_score * 0.24
            + aspect_score * 0.20
            + area_score * 0.14
            + (1.0 if inside_bounds else 0.0) * 0.04
            + (1.0 if convex else 0.0) * 0.04,
            0.0,
            1.0,
        )
    )

    reason = "ok"
    accepted = True
    if not points_ok:
        accepted = False
        reason = "non_finite_quad"
    elif not inside_bounds:
        accepted = False
        reason = "quad_out_of_bounds"
    elif not convex:
        accepted = False
        reason = "quad_not_convex"
    elif inliers < min_inliers:
        accepted = False
        reason = "too_few_inliers"
    elif inlier_ratio < min_inlier_ratio:
        accepted = False
        reason = "low_inlier_ratio"
    elif area_ratio < 0.08 or area_ratio > 0.88:
        accepted = False
        reason = "bad_area"
    elif aspect_score < 0.50:
        accepted = False
        reason = "bad_aspect"
    elif confidence < 0.50:
        accepted = False
        reason = "low_confidence"

    return {
        "accepted": accepted,
        "reason": reason,
        "confidence": round(confidence, 4),
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "good_matches": int(alignment.get("good_matches", 0)),
        "area_ratio": round(float(area_ratio), 4),
        "aspect": round(float(aspect), 4),
        "aspect_score": round(float(aspect_score), 4),
        "format": format_label,
    }


def closest_format(aspect: float) -> tuple[str, float]:
    best_label = "unknown"
    best_score = 0.0
    for label, target in FORMAT_RATIOS.items():
        delta = abs(np.log(max(aspect, 0.05)) - np.log(target))
        score = float(1.0 - np.clip(delta / 0.36, 0.0, 1.0))
        if score > best_score:
            best_score = score
            best_label = label
    return best_label, best_score


def rect_to_dict(rect: ImageRect) -> dict[str, float]:
    return {
        "x": int(rect.x),
        "y": int(rect.y),
        "width": int(rect.width),
        "height": int(rect.height),
        "angle": round(float(rect.angle), 4),
    }


def write_overlay(
    path: Path,
    negative_rgb8: np.ndarray,
    reference_srgb: np.ndarray,
    quad: np.ndarray,
    *,
    label: str,
    accepted: bool,
) -> None:
    negative = Image.fromarray(negative_rgb8).convert("RGB")
    reference = Image.fromarray((np.clip(reference_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).convert("RGB")
    reference.thumbnail((360, 360), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(negative, "RGBA")
    points = [(float(x), float(y)) for x, y in quad]
    color = (69, 212, 168, 230) if accepted else (235, 92, 92, 230)
    fill = (69, 212, 168, 28) if accepted else (235, 92, 92, 28)
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=color, width=4)
    for point in points:
        x, y = point
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    draw.rectangle((12, 12, 520, 48), fill=(18, 20, 24, 200))
    draw.text((22, 21), label, fill=(245, 248, 252, 255))

    canvas_width = negative.width + reference.width + 24
    canvas_height = max(negative.height, reference.height + 36)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 20, 24))
    canvas.paste(negative, (0, 0))
    canvas.paste(reference, (negative.width + 24, 36))
    ref_draw = ImageDraw.Draw(canvas)
    ref_draw.text((negative.width + 24, 12), "cropped positive reference", fill=(230, 235, 242))
    canvas.save(path, quality=92)


def write_alignment_debug(
    path: Path,
    negative_rgb8: np.ndarray,
    reference_srgb: np.ndarray,
    homography_source_to_reference: np.ndarray,
) -> None:
    reference_rgb8 = (np.clip(reference_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    height, width = reference_rgb8.shape[:2]
    warped = cv2.warpPerspective(
        negative_rgb8,
        homography_source_to_reference,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    blend = cv2.addWeighted(warped, 0.50, reference_rgb8, 0.50, 0)
    canvas = Image.new("RGB", (width * 3, height + 28), (18, 20, 24))
    canvas.paste(Image.fromarray(warped), (0, 28))
    canvas.paste(Image.fromarray(reference_rgb8), (width, 28))
    canvas.paste(Image.fromarray(blend), (width * 2, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "warped negative", fill=(230, 235, 242))
    draw.text((width + 8, 8), "reference", fill=(230, 235, 242))
    draw.text((width * 2 + 8, 8), "blend", fill=(230, 235, 242))
    canvas.save(path, quality=92)


def make_contact_sheet(paths: list[Path], out_path: Path, *, columns: int = 4, thumb_size: tuple[int, int] = (420, 280)) -> None:
    if not paths:
        return
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", thumb_size, (18, 20, 24))
        tile.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, thumb_size[1] - 24, thumb_size[0], thumb_size[1]), fill=(18, 20, 24))
        draw.text((8, thumb_size[1] - 18), path.stem, fill=(230, 235, 242))
        thumbs.append(tile)

    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), (18, 20, 24))
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * thumb_size[1]
        sheet.paste(thumb, (x, y))
    sheet.save(out_path, quality=92)


def build_summary(
    labels: list[FrameLabel],
    *,
    discovery: PairDiscovery | None = None,
    processed_pairs: int | None = None,
    failures: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    accepted = [label for label in labels if label.accepted]
    rejected = [label for label in labels if not label.accepted]
    reasons: dict[str, int] = {}
    for label in rejected:
        reason = str(label.quality.get("reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    formats: dict[str, int] = {}
    for label in accepted:
        formats[label.format] = formats.get(label.format, 0) + 1
    summary: dict[str, object] = {
        "total": len(labels),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": reasons,
        "failed": len(failures or []),
        "failure_examples": (failures or [])[:24],
        "formats": formats,
        "mean_confidence": float(np.mean([label.confidence for label in accepted])) if accepted else 0.0,
        "median_confidence": float(np.median([label.confidence for label in accepted])) if accepted else 0.0,
    }
    if discovery is not None:
        summary["discovery"] = {
            "negative_count": discovery.negative_count,
            "positive_count": discovery.positive_count,
            "matched_count": discovery.matched_count,
            "processed_pairs": processed_pairs if processed_pairs is not None else discovery.matched_count,
            "negatives_without_positive": discovery.negatives_without_positive,
            "positives_without_negative": discovery.positives_without_negative,
            "duplicate_negative_stems": discovery.duplicate_negative_stems,
            "duplicate_positive_stems": discovery.duplicate_positive_stems,
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
