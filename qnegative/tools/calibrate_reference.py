from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from qnegative.core.preview import make_raw_preview


LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DENSITY_EPSILON = 1e-5


@dataclass(frozen=True)
class CropMargins:
    left: float = 0.075
    top: float = 0.065
    right: float = 0.950
    bottom: float = 0.915


@dataclass(frozen=True)
class PairPaths:
    name: str
    negative_path: Path
    positive_path: Path


@dataclass
class ImageSamples:
    name: str
    density_rgb: np.ndarray
    reference_rgb: np.ndarray
    tone_rgb: np.ndarray | None
    match_scores: np.ndarray
    alignment: dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate QNegativeLab against positive reference scans.")
    parser.add_argument("--negative-dir", required=True, type=Path)
    parser.add_argument("--positive-dir", required=True, type=Path)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default=Path("calibration_debug"), type=Path)
    parser.add_argument("--max-size", default=1600, type=int)
    parser.add_argument("--step", default=18, type=int)
    parser.add_argument("--patch-radius", default=4, type=int)
    parser.add_argument("--search-radius", default=8, type=int)
    parser.add_argument("--limit", default=0, type=int, help="Optional maximum number of pairs.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = pair_files(args.negative_dir, args.positive_dir, start=args.start, end=args.end)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No matching negative/positive pairs found.")

    print(f"Pairs: {len(pairs)}")
    samples_by_image: list[ImageSamples] = []
    for pair in pairs:
        try:
            samples = process_pair(
                pair,
                out_dir=args.out,
                max_size=args.max_size,
                step=args.step,
                patch_radius=args.patch_radius,
                search_radius=args.search_radius,
            )
        except Exception as exc:
            print(f"{pair.name}: failed: {exc}")
            continue
        samples_by_image.append(samples)
        alignment = samples.alignment
        print(
            f"{pair.name}: samples={len(samples.density_rgb)}, "
            f"good={alignment['good_matches']}, inliers={alignment['inliers']}, "
            f"score_med={float(np.median(samples.match_scores)):.3f}"
        )

    if not samples_by_image:
        raise SystemExit("All pairs failed.")

    all_density = np.concatenate([item.density_rgb for item in samples_by_image], axis=0)
    all_reference = np.concatenate([item.reference_rgb for item in samples_by_image], axis=0)

    tone_curve = fit_luminance_curve(all_density, all_reference)
    for item in samples_by_image:
        item.tone_rgb = apply_tone_curve(item.density_rgb, tone_curve)
    all_tone = np.concatenate([item.tone_rgb for item in samples_by_image if item.tone_rgb is not None], axis=0)

    wb_gains = estimate_wb_gains(all_tone, all_reference)
    matrix, bias, robust_mask = fit_color_matrix(all_tone * wb_gains.reshape(1, 3), all_reference)

    before_stats = error_stats(all_tone, all_reference)
    wb_stats = error_stats(np.clip(all_tone * wb_gains.reshape(1, 3), 0.0, 1.0), all_reference)
    after = np.clip((all_tone * wb_gains.reshape(1, 3)) @ matrix + bias.reshape(1, 3), 0.0, 1.0)
    after_stats = error_stats(after, all_reference)

    per_image = []
    for item in samples_by_image:
        assert item.tone_rgb is not None
        image_matrix, image_bias, image_mask = fit_color_matrix(item.tone_rgb * wb_gains.reshape(1, 3), item.reference_rgb)
        pred = np.clip((item.tone_rgb * wb_gains.reshape(1, 3)) @ image_matrix + image_bias.reshape(1, 3), 0.0, 1.0)
        per_image.append(
            {
                "name": item.name,
                "samples": int(len(item.density_rgb)),
                "matrix": image_matrix.tolist(),
                "bias": image_bias.tolist(),
                "kept_samples": int(np.count_nonzero(image_mask)),
                "error": error_stats(pred, item.reference_rgb),
                "alignment": item.alignment,
            }
        )

    matrices = np.array([entry["matrix"] for entry in per_image], dtype=np.float32)
    biases = np.array([entry["bias"] for entry in per_image], dtype=np.float32)
    film_bases = np.array([entry["alignment"]["film_base_rgb"] for entry in per_image], dtype=np.float32)
    film_base_chroma = film_bases / np.maximum(film_bases[:, 1:2], DENSITY_EPSILON)
    matrix_mean = matrices.mean(axis=0)
    matrix_std = matrices.std(axis=0)
    matrix_relative_std = matrix_std / np.maximum(np.abs(matrix_mean), 1e-6)
    report = {
        "pairs": [pair.name for pair in pairs],
        "used_pairs": [item.name for item in samples_by_image],
        "total_samples": int(len(all_density)),
        "tone_curve": {
            "density_luma": tone_curve["x"].tolist(),
            "reference_luma": tone_curve["y"].tolist(),
            "density_targets": tone_curve_targets(tone_curve),
        },
        "white_balance_gains": wb_gains.tolist(),
        "color_matrix": matrix.tolist(),
        "color_bias": bias.tolist(),
        "robust_kept_samples": int(np.count_nonzero(robust_mask)),
        "errors": {
            "tone_only": before_stats,
            "tone_plus_wb": wb_stats,
            "tone_wb_matrix": after_stats,
        },
        "per_image": per_image,
        "matrix_mean": matrix_mean.tolist(),
        "matrix_std": matrix_std.tolist(),
        "bias_mean": biases.mean(axis=0).tolist(),
        "bias_std": biases.std(axis=0).tolist(),
        "stability": {
            "film_base_rgb_mean": film_bases.mean(axis=0).tolist(),
            "film_base_rgb_std": film_bases.std(axis=0).tolist(),
            "film_base_chroma_mean": {
                "r_over_g": float(film_base_chroma[:, 0].mean()),
                "b_over_g": float(film_base_chroma[:, 2].mean()),
            },
            "film_base_chroma_std": {
                "r_over_g": float(film_base_chroma[:, 0].std()),
                "b_over_g": float(film_base_chroma[:, 2].std()),
            },
            "matrix_relative_std": matrix_relative_std.tolist(),
        },
    }

    write_json(args.out / "calibration_result.json", report)
    write_samples_csv(args.out / "calibration_samples.csv", samples_by_image)
    draw_tone_plot(args.out / "tone_curve.png", all_density, all_reference, tone_curve)
    draw_color_plot(args.out / "color_error_before_after.png", all_tone, all_reference, wb_gains, matrix, bias)
    write_preview_comparisons(args.out, samples_by_image[:3], tone_curve, wb_gains, matrix, bias)

    print("\nCalibration result")
    print(f"Total samples: {report['total_samples']}")
    print(f"Tone only RMSE: {before_stats['rmse']:.5f}")
    print(f"Tone + WB RMSE: {wb_stats['rmse']:.5f}")
    print(f"Tone + WB + Matrix RMSE: {after_stats['rmse']:.5f}")
    print("WB gains:", np.array2string(wb_gains, precision=5))
    print("Color matrix:\n", np.array2string(matrix, precision=5))
    print("Color bias:", np.array2string(bias, precision=5))
    print(
        "Film base chroma R/G, B/G:",
        f"{film_base_chroma[:, 0].mean():.5f} +/- {film_base_chroma[:, 0].std():.5f},",
        f"{film_base_chroma[:, 2].mean():.5f} +/- {film_base_chroma[:, 2].std():.5f}",
    )
    print("Report:", args.out / "calibration_result.json")
    return 0


def pair_files(negative_dir: Path, positive_dir: Path, *, start: str | None, end: str | None) -> list[PairPaths]:
    negatives = {path.stem: path for path in negative_dir.glob("*.ARW")}
    positives = {
        path.stem: path
        for path in positive_dir.glob("*.png")
        if not path.name.startswith("._")
    }
    names = sorted(set(negatives) & set(positives))
    if start is not None:
        names = [name for name in names if name >= start]
    if end is not None:
        names = [name for name in names if name <= end]
    return [PairPaths(name=name, negative_path=negatives[name], positive_path=positives[name]) for name in names]


def process_pair(
    pair: PairPaths,
    *,
    out_dir: Path,
    max_size: int,
    step: int,
    patch_radius: int,
    search_radius: int,
    write_debug: bool = True,
) -> ImageSamples:
    preview = make_raw_preview(pair.negative_path, max_size=max_size)
    reference_srgb = load_reference_srgb(pair.positive_path, max_size=max_size)
    reference_linear = srgb_to_linear(reference_srgb)

    display_bgr = cv2.cvtColor(preview.display_rgb8, cv2.COLOR_RGB2BGR)
    crop_rect = rough_frame_crop(display_bgr.shape, CropMargins())
    x0, y0, x1, y1 = crop_rect
    display_crop = display_bgr[y0:y1, x0:x1]
    camera_crop = preview.preview_linear_rgb[y0:y1, x0:x1]
    mask_rgb = estimate_film_base(preview.preview_linear_rgb, crop_rect)
    density_crop = transmittance_to_density(camera_crop / mask_rgb.reshape(1, 1, 3))

    raw_work = prepare_negative_match_gray(display_crop)
    ref_work = prepare_reference_match_gray(reference_srgb)
    homography, alignment = estimate_homography(raw_work, ref_work)
    alignment.update(
        {
            "crop_rect_preview": [int(x0), int(y0), int(x1), int(y1)],
            "film_base_rgb": mask_rgb.tolist(),
        }
    )

    height, width = reference_linear.shape[:2]
    warped_density = cv2.warpPerspective(
        density_crop,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_match = cv2.warpPerspective(
        raw_work,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_mask = cv2.warpPerspective(
        np.ones(density_crop.shape[:2], dtype=np.uint8) * 255,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    density_samples, reference_samples, match_scores = collect_patch_samples(
        warped_density,
        reference_linear,
        warped_match,
        ref_work,
        valid_mask,
        step=step,
        patch_radius=patch_radius,
        search_radius=search_radius,
    )

    if write_debug:
        write_alignment_debug(
            pair.name,
            out_dir,
            display_crop,
            reference_srgb,
            raw_work,
            ref_work,
            homography,
            alignment,
        )
    return ImageSamples(
        name=pair.name,
        density_rgb=density_samples,
        reference_rgb=reference_samples,
        tone_rgb=None,
        match_scores=match_scores,
        alignment=alignment,
    )


def rough_frame_crop(shape: tuple[int, int, int], margins: CropMargins) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x0 = int(width * margins.left)
    y0 = int(height * margins.top)
    x1 = int(width * margins.right)
    y1 = int(height * margins.bottom)
    return x0, y0, x1, y1


def estimate_film_base(camera_rgb: np.ndarray, crop_rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, _y1 = crop_rect
    height, width = camera_rgb.shape[:2]
    band_h = max(8, int(height * 0.025))
    y_start = max(0, y0 - band_h - 4)
    y_end = max(y_start + 1, y0 - 4)
    x_start = max(0, x0 + int((x1 - x0) * 0.15))
    x_end = min(width, x1 - int((x1 - x0) * 0.15))
    sample = camera_rgb[y_start:y_end, x_start:x_end]
    if sample.size == 0:
        sample = camera_rgb[max(0, y0 - band_h) : y0, x0:x1]
    pixels = sample.reshape(-1, 3)
    luminance = pixels @ LUMA_WEIGHTS
    low = np.percentile(luminance, 20.0)
    high = np.percentile(luminance, 95.0)
    keep = (luminance >= low) & (luminance <= high)
    if int(np.count_nonzero(keep)) >= 64:
        pixels = pixels[keep]
    return np.maximum(np.median(pixels, axis=0).astype(np.float32), np.array([DENSITY_EPSILON] * 3, dtype=np.float32))


def transmittance_to_density(transmittance: np.ndarray) -> np.ndarray:
    clipped = np.clip(transmittance, DENSITY_EPSILON, 1.0)
    return np.maximum(-np.log10(clipped), 0.0).astype(np.float32, copy=False)


def load_reference_srgb(path: Path, *, max_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb8(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear, 0.0, 1.0)
    srgb = np.where(clipped <= 0.0031308, clipped * 12.92, 1.055 * np.power(clipped, 1.0 / 2.4) - 0.055)
    return np.ascontiguousarray((np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def prepare_negative_match_gray(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(255 - gray)


def prepare_reference_match_gray(image_srgb: np.ndarray) -> np.ndarray:
    image_u8 = np.ascontiguousarray((np.clip(image_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_homography(source_gray: np.ndarray, reference_gray: np.ndarray) -> tuple[np.ndarray, dict]:
    try:
        detector = cv2.SIFT_create(nfeatures=5000)
        norm = cv2.NORM_L2
        detector_name = "SIFT"
    except Exception:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
        detector_name = "ORB"

    kp_source, des_source = detector.detectAndCompute(source_gray, None)
    kp_reference, des_reference = detector.detectAndCompute(reference_gray, None)
    if des_source is None or des_reference is None:
        raise RuntimeError("Feature detection failed.")

    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(des_source, des_reference, k=2)
    good = [m for m, n in matches if m.distance < 0.72 * n.distance]
    if len(good) < 8:
        raise RuntimeError(f"Not enough feature matches: {len(good)}.")

    source_points = np.float32([kp_source[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    reference_points = np.float32([kp_reference[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(source_points, reference_points, cv2.RANSAC, 5.0)
    if homography is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed.")

    inliers = int(inlier_mask.sum())
    if inliers < 16:
        raise RuntimeError(f"Homography has too few inliers: {inliers}.")

    scale_x = float((homography[0, 0] ** 2 + homography[1, 0] ** 2) ** 0.5)
    scale_y = float((homography[0, 1] ** 2 + homography[1, 1] ** 2) ** 0.5)
    return homography.astype(np.float32), {
        "detector": detector_name,
        "source_keypoints": int(len(kp_source)),
        "reference_keypoints": int(len(kp_reference)),
        "good_matches": int(len(good)),
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(len(good), 1)),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "homography": homography.tolist(),
    }


def collect_patch_samples(
    warped_density: np.ndarray,
    reference_linear: np.ndarray,
    warped_match_gray: np.ndarray,
    reference_match_gray: np.ndarray,
    valid_mask: np.ndarray,
    *,
    step: int,
    patch_radius: int,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = reference_linear.shape[:2]
    density_samples: list[np.ndarray] = []
    reference_samples: list[np.ndarray] = []
    match_scores: list[float] = []

    reference_luma = reference_linear @ LUMA_WEIGHTS
    ref_grad_x = cv2.Sobel(reference_match_gray, cv2.CV_32F, 1, 0, ksize=3)
    ref_grad_y = cv2.Sobel(reference_match_gray, cv2.CV_32F, 0, 1, ksize=3)
    reference_gradient = cv2.magnitude(ref_grad_x, ref_grad_y)

    y_values = range(search_radius + patch_radius, height - search_radius - patch_radius, step)
    x_values = range(search_radius + patch_radius, width - search_radius - patch_radius, step)
    for y in y_values:
        for x in x_values:
            y0 = y - patch_radius
            y1 = y + patch_radius + 1
            x0 = x - patch_radius
            x1 = x + patch_radius + 1

            if np.mean(valid_mask[y0:y1, x0:x1] > 0) < 0.98:
                continue
            ref_patch = reference_linear[y0:y1, x0:x1]
            ref_luma_patch = reference_luma[y0:y1, x0:x1]
            if float(np.median(ref_luma_patch)) < 0.025 or float(np.median(ref_luma_patch)) > 0.97:
                continue
            if np.any(np.median(ref_patch.reshape(-1, 3), axis=0) > 0.985):
                continue
            if float(np.mean(reference_gradient[y0:y1, x0:x1])) > 95.0:
                continue

            score, match_x, match_y = local_match(
                reference_match_gray,
                warped_match_gray,
                x,
                y,
                patch_radius=patch_radius,
                search_radius=search_radius,
            )
            if score < 0.36:
                continue

            wy0 = match_y - patch_radius
            wy1 = match_y + patch_radius + 1
            wx0 = match_x - patch_radius
            wx1 = match_x + patch_radius + 1
            if wy0 < 0 or wx0 < 0 or wy1 > height or wx1 > width:
                continue
            if np.mean(valid_mask[wy0:wy1, wx0:wx1] > 0) < 0.98:
                continue

            density_patch = warped_density[wy0:wy1, wx0:wx1]
            if not np.all(np.isfinite(density_patch)):
                continue
            density_median = np.median(density_patch.reshape(-1, 3), axis=0)
            ref_median = np.median(ref_patch.reshape(-1, 3), axis=0)
            if float(density_median.max()) <= DENSITY_EPSILON:
                continue

            density_samples.append(density_median.astype(np.float32))
            reference_samples.append(ref_median.astype(np.float32))
            match_scores.append(float(score))

    if not density_samples:
        raise RuntimeError("No usable color samples.")
    return (
        np.vstack(density_samples).astype(np.float32),
        np.vstack(reference_samples).astype(np.float32),
        np.array(match_scores, dtype=np.float32),
    )


def local_match(
    reference_gray: np.ndarray,
    warped_gray: np.ndarray,
    x: int,
    y: int,
    *,
    patch_radius: int,
    search_radius: int,
) -> tuple[float, int, int]:
    template = reference_gray[y - patch_radius : y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
    search = warped_gray[
        y - search_radius - patch_radius : y + search_radius + patch_radius + 1,
        x - search_radius - patch_radius : x + search_radius + patch_radius + 1,
    ]
    if template.size == 0 or search.size == 0:
        return 0.0, x, y
    if float(np.std(template)) < 4.0:
        return 1.0, x, y
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(result)
    matched_x = x - search_radius + max_location[0]
    matched_y = y - search_radius + max_location[1]
    return float(max_value), int(matched_x), int(matched_y)


def rgb_luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA_WEIGHTS


def fit_luminance_curve(density_rgb: np.ndarray, reference_rgb: np.ndarray, *, bins: int = 48) -> dict[str, np.ndarray]:
    x = rgb_luminance(density_rgb)
    y = rgb_luminance(reference_rgb)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0) & (y < 0.995)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) < bins * 8:
        raise RuntimeError("Not enough samples to fit luminance curve.")

    edges = np.linspace(0, len(x), bins + 1, dtype=np.int32)
    curve_x = []
    curve_y = []
    for index in range(bins):
        lo = edges[index]
        hi = edges[index + 1]
        if hi <= lo:
            continue
        curve_x.append(float(np.median(x[lo:hi])))
        curve_y.append(float(np.median(y[lo:hi])))

    curve_x_arr = np.array(curve_x, dtype=np.float32)
    curve_y_arr = np.array(curve_y, dtype=np.float32)
    curve_y_arr = np.maximum.accumulate(curve_y_arr)
    curve_y_arr = np.clip(curve_y_arr, 0.0, 1.0)

    unique_x, unique_indices = np.unique(curve_x_arr, return_index=True)
    curve_y_arr = curve_y_arr[unique_indices]
    return {"x": unique_x.astype(np.float32), "y": curve_y_arr.astype(np.float32)}


def tone_curve_targets(tone_curve: dict[str, np.ndarray]) -> dict[str, float | None]:
    x = tone_curve["x"]
    y = tone_curve["y"]
    targets = {}
    for target in (0.02, 0.05, 0.18, 0.50, 0.82, 0.95):
        if target < float(y.min()) or target > float(y.max()):
            targets[f"luma_{target:.2f}"] = None
        else:
            targets[f"luma_{target:.2f}"] = float(np.interp(target, y, x))
    return targets


def apply_tone_curve(density_rgb: np.ndarray, tone_curve: dict[str, np.ndarray]) -> np.ndarray:
    x = tone_curve["x"]
    y = tone_curve["y"]
    flat = density_rgb.reshape(-1)
    toned = np.interp(flat, x, y, left=float(y[0]), right=float(y[-1]))
    return toned.reshape(density_rgb.shape).astype(np.float32)


def estimate_wb_gains(tone_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    ref_luma = rgb_luminance(reference_rgb)
    ref_chroma = reference_rgb.max(axis=1) - reference_rgb.min(axis=1)
    ref_saturation = ref_chroma / np.maximum(ref_luma, 1e-4)
    mask = (ref_luma > 0.10) & (ref_luma < 0.88) & (ref_saturation < 0.16)
    if int(np.count_nonzero(mask)) < 128:
        mask = (ref_luma > 0.08) & (ref_luma < 0.92) & (ref_saturation < 0.24)
    if int(np.count_nonzero(mask)) < 32:
        mask = (ref_luma > 0.08) & (ref_luma < 0.92)

    ratios = reference_rgb[mask] / np.maximum(tone_rgb[mask], 1e-4)
    ratios = ratios[np.all(np.isfinite(ratios), axis=1)]
    ratios = np.clip(ratios, 0.1, 10.0)
    gains = np.median(ratios, axis=0).astype(np.float32)
    gains = gains / np.exp(np.mean(np.log(np.maximum(gains, 1e-4))))
    return np.clip(gains, 0.25, 4.0).astype(np.float32)


def fit_color_matrix(source_rgb: np.ndarray, reference_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        np.all(np.isfinite(source_rgb), axis=1)
        & np.all(np.isfinite(reference_rgb), axis=1)
        & np.all(source_rgb > 0.002, axis=1)
        & np.all(reference_rgb > 0.002, axis=1)
        & np.all(reference_rgb < 0.985, axis=1)
    )
    x = source_rgb[valid]
    y = reference_rgb[valid]
    if len(x) < 32:
        x = source_rgb
        y = reference_rgb
        valid = np.ones(len(x), dtype=bool)

    keep = np.ones(len(x), dtype=bool)
    coeff = None
    for _iteration in range(4):
        x_aug = np.concatenate([x[keep], np.ones((int(np.count_nonzero(keep)), 1), dtype=np.float32)], axis=1)
        coeff, *_ = np.linalg.lstsq(x_aug, y[keep], rcond=None)
        prediction = np.clip(np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1) @ coeff, 0.0, 1.0)
        error = np.linalg.norm(prediction - y, axis=1)
        threshold = np.percentile(error, 88.0)
        keep = error <= threshold
        if int(np.count_nonzero(keep)) < 32:
            keep = np.ones(len(x), dtype=bool)
            break

    assert coeff is not None
    full_keep = np.zeros(len(source_rgb), dtype=bool)
    full_keep[np.flatnonzero(valid)[keep]] = True
    return coeff[:3, :].astype(np.float32), coeff[3, :].astype(np.float32), full_keep


def error_stats(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    clipped = np.clip(prediction, 0.0, 1.0)
    delta = clipped - reference
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "p95_abs": float(np.percentile(np.abs(delta), 95.0)),
    }


def write_alignment_debug(
    name: str,
    out_dir: Path,
    display_crop_bgr: np.ndarray,
    reference_srgb: np.ndarray,
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    homography: np.ndarray,
    alignment: dict,
) -> None:
    ref_u8 = np.ascontiguousarray((reference_srgb * 255.0 + 0.5).astype(np.uint8))
    ref_bgr = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(display_crop_bgr, homography, (ref_bgr.shape[1], ref_bgr.shape[0]))
    blend = cv2.addWeighted(warped, 0.5, ref_bgr, 0.5, 0)
    cv2.imwrite(str(out_dir / f"{name}_alignment_blend.jpg"), blend)
    cv2.imwrite(str(out_dir / f"{name}_warped_negative_crop.jpg"), warped)
    Image.fromarray(ref_u8).save(out_dir / f"{name}_reference_preview.jpg", quality=92)

    match_overlay = np.zeros((ref_bgr.shape[0], ref_bgr.shape[1], 3), dtype=np.uint8)
    match_overlay[:, :, 1] = reference_gray
    match_overlay[:, :, 2] = cv2.warpPerspective(source_gray, homography, (ref_bgr.shape[1], ref_bgr.shape[0]))
    cv2.imwrite(str(out_dir / f"{name}_match_gray_overlay.jpg"), match_overlay)


def write_preview_comparisons(
    out_dir: Path,
    samples_by_image: list[ImageSamples],
    tone_curve: dict[str, np.ndarray],
    wb_gains: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
) -> None:
    for item in samples_by_image:
        assert item.tone_rgb is not None
        before = linear_to_srgb8(item.tone_rgb)
        after = linear_to_srgb8(np.clip((item.tone_rgb * wb_gains.reshape(1, 3)) @ matrix + bias.reshape(1, 3), 0.0, 1.0))
        ref = linear_to_srgb8(item.reference_rgb)
        width = 420
        image = Image.new("RGB", (width * 3, width), (20, 22, 26))
        for col, data in enumerate((before, after, ref)):
            swatch = color_cloud_image(data, width, width)
            image.paste(swatch, (col * width, 0))
        image.save(out_dir / f"{item.name}_sample_color_clouds.jpg", quality=92)


def color_cloud_image(rgb8: np.ndarray, width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(42)
    image = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(image)
    pixels = rgb8.reshape(-1, 3)
    count = min(3500, len(pixels))
    indices = rng.choice(len(pixels), size=count, replace=False)
    colors = pixels[indices]
    xs = rng.integers(0, width, size=count)
    ys = rng.integers(0, height, size=count)
    for x, y, color in zip(xs, ys, colors):
        draw.point((int(x), int(y)), fill=tuple(int(v) for v in color))
    return image


def draw_tone_plot(path: Path, density_rgb: np.ndarray, reference_rgb: np.ndarray, tone_curve: dict[str, np.ndarray]) -> None:
    width, height = 900, 640
    margin = 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    x = rgb_luminance(density_rgb)
    y = rgb_luminance(reference_rgb)
    x_min, x_max = float(np.percentile(x, 0.5)), float(np.percentile(x, 99.5))
    y_min, y_max = 0.0, 1.0

    rng = np.random.default_rng(123)
    count = min(12000, len(x))
    indices = rng.choice(len(x), size=count, replace=False)
    for index in indices:
        px = margin + (float(x[index]) - x_min) / max(x_max - x_min, 1e-6) * (width - 2 * margin)
        py = height - margin - (float(y[index]) - y_min) / max(y_max - y_min, 1e-6) * (height - 2 * margin)
        draw.point((int(px), int(py)), fill=(80, 120, 180))

    points = []
    for cx, cy in zip(tone_curve["x"], tone_curve["y"]):
        px = margin + (float(cx) - x_min) / max(x_max - x_min, 1e-6) * (width - 2 * margin)
        py = height - margin - (float(cy) - y_min) / max(y_max - y_min, 1e-6) * (height - 2 * margin)
        points.append((px, py))
    if len(points) >= 2:
        draw.line(points, fill=(220, 45, 35), width=3)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(20, 20, 20), width=1)
    draw.text((margin, 18), "density luminance -> reference luminance", fill=(0, 0, 0))
    image.save(path)


def draw_color_plot(
    path: Path,
    tone_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    wb_gains: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
) -> None:
    before = np.clip(tone_rgb, 0.0, 1.0)
    after = np.clip((tone_rgb * wb_gains.reshape(1, 3)) @ matrix + bias.reshape(1, 3), 0.0, 1.0)
    width, height = 900, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(456)
    count = min(9000, len(reference_rgb))
    indices = rng.choice(len(reference_rgb), size=count, replace=False)
    left = (55, 65, 420, 585)
    right = (480, 65, 845, 585)
    draw_channel_scatter(draw, left, before[indices], reference_rgb[indices], (60, 90, 190))
    draw_channel_scatter(draw, right, after[indices], reference_rgb[indices], (20, 150, 75))
    draw.text((55, 24), "before matrix", fill=(0, 0, 0))
    draw.text((480, 24), "after matrix", fill=(0, 0, 0))
    image.save(path)


def draw_channel_scatter(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], source: np.ndarray, reference: np.ndarray, color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(20, 20, 20), width=1)
    draw.line((x0, y1, x1, y0), fill=(200, 60, 60), width=1)
    for channel in range(3):
        for sx, ry in zip(source[:, channel], reference[:, channel]):
            px = x0 + float(sx) * (x1 - x0)
            py = y1 - float(ry) * (y1 - y0)
            draw.point((int(px), int(py)), fill=color)


def write_samples_csv(path: Path, samples_by_image: list[ImageSamples]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "density_r", "density_g", "density_b", "reference_r", "reference_g", "reference_b", "match_score"])
        for item in samples_by_image:
            for density, reference, score in zip(item.density_rgb, item.reference_rgb, item.match_scores):
                writer.writerow([item.name, *density.tolist(), *reference.tolist(), float(score)])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
