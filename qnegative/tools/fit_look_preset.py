from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from qnegative.core.models import DensityMatrixParams
from qnegative.tools.calibrate_reference import (
    DENSITY_EPSILON,
    ImageSamples,
    apply_tone_curve,
    color_cloud_image,
    error_stats,
    fit_luminance_curve,
    linear_to_srgb8,
    pair_files,
    process_pair,
    rgb_luminance,
)


DEFAULT_PRIOR = DensityMatrixParams()
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit an empirical QNegativeLab look preset from reference scans.")
    parser.add_argument("--negative-dir", required=True, type=Path)
    parser.add_argument("--positive-dir", required=True, type=Path)
    parser.add_argument("--name", required=True, help="Preset name, e.g. fujiC400.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--debug-dir", type=Path, default=Path("calibration_look"))
    parser.add_argument("--max-size", type=int, default=1200)
    parser.add_argument("--step", type=int, default=34)
    parser.add_argument("--patch-radius", type=int, default=4)
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--regularization", type=float, default=5000.0)
    parser.add_argument("--fit-strength", type=float, default=0.55)
    parser.add_argument("--min-inliers", type=int, default=160)
    parser.add_argument("--debug-alignments", action="store_true")
    args = parser.parse_args()

    out_path = args.out or Path("presets") / f"{args.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = args.debug_dir / args.name
    debug_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_files(args.negative_dir, args.positive_dir, start=args.start, end=args.end)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No matching negative/positive pairs found.")

    print(f"Preset: {args.name}")
    print(f"Pairs: {len(pairs)} ({args.start} - {args.end})")

    samples_by_image: list[ImageSamples] = []
    for pair in pairs:
        try:
            samples = process_pair(
                pair,
                out_dir=debug_dir,
                max_size=args.max_size,
                step=args.step,
                patch_radius=args.patch_radius,
                search_radius=args.search_radius,
                write_debug=args.debug_alignments,
            )
        except Exception as exc:
            print(f"{pair.name}: failed: {exc}")
            continue
        alignment = samples.alignment
        if int(alignment["inliers"]) < args.min_inliers:
            print(f"{pair.name}: skipped: inliers={alignment['inliers']} below {args.min_inliers}")
            continue
        samples_by_image.append(samples)
        print(
            f"{pair.name}: samples={len(samples.density_rgb)}, "
            f"inliers={alignment['inliers']}, score_med={float(np.median(samples.match_scores)):.3f}"
        )

    if len(samples_by_image) < 2:
        raise SystemExit("Need at least two usable pairs to fit a look preset.")

    all_density = np.concatenate([item.density_rgb for item in samples_by_image], axis=0)
    all_reference = np.concatenate([item.reference_rgb for item in samples_by_image], axis=0)
    tone_bins = int(np.clip(len(all_density) // 12, 12, 48))
    tone_curve = fit_luminance_curve(all_density, all_reference, bins=tone_bins)

    prior = normalize_matrix_rows(density_params_to_matrix(DEFAULT_PRIOR))
    fit_source, fit_target, per_image_norm = build_density_fit_set(samples_by_image, tone_curve)
    fitted_matrix, robust_mask = fit_row_sum_density_matrix(
        fit_source,
        fit_target,
        prior,
        regularization=args.regularization,
    )
    strength = float(np.clip(args.fit_strength, 0.0, 1.0))
    recommended_matrix = normalize_matrix_rows(prior * (1.0 - strength) + fitted_matrix * strength)

    before_pred = np.maximum(fit_source @ prior.T, 0.0)
    fitted_pred = np.maximum(fit_source @ fitted_matrix.T, 0.0)
    recommended_pred = np.maximum(fit_source @ recommended_matrix.T, 0.0)
    before_chroma_error = chroma_error_stats(before_pred, fit_target)
    fitted_chroma_error = chroma_error_stats(fitted_pred, fit_target)
    recommended_chroma_error = chroma_error_stats(recommended_pred, fit_target)

    positive_before = apply_tone_curve(before_pred, tone_curve)
    positive_recommended = apply_tone_curve(recommended_pred, tone_curve)
    reference_proxy = apply_tone_curve(fit_target, tone_curve)
    positive_before_stats = error_stats(positive_before, reference_proxy)
    positive_recommended_stats = error_stats(positive_recommended, reference_proxy)

    film_bases = np.array([item.alignment["film_base_rgb"] for item in samples_by_image], dtype=np.float32)
    film_base_chroma = film_bases / np.maximum(film_bases[:, 1:2], DENSITY_EPSILON)

    report = {
        "name": args.name,
        "type": "empirical_density_look_preset",
        "range": {"start": args.start, "end": args.end},
        "source": {
            "negative_dir": str(args.negative_dir),
            "positive_dir": str(args.positive_dir),
            "used_pairs": [item.name for item in samples_by_image],
            "failed_or_missing_pairs": [pair.name for pair in pairs if pair.name not in {item.name for item in samples_by_image}],
            "total_samples": int(len(fit_source)),
            "robust_kept_samples": int(np.count_nonzero(robust_mask)),
        },
        "method": {
            "wb_exposure_normalization": "per-image density chroma normalization using reference low-saturation patches",
            "matrix_constraint": "row sums normalized to 1.0 to preserve neutral density brightness",
            "prior": "current app default density matrix, row-normalized",
            "regularization": float(args.regularization),
            "fit_strength": strength,
            "min_inliers": int(args.min_inliers),
        },
        "tone_curve_reference": {
            "density_luma": tone_curve["x"].tolist(),
            "reference_luma": tone_curve["y"].tolist(),
        },
        "density_matrix": {
            "prior": prior.tolist(),
            "raw_fit": fitted_matrix.tolist(),
            "recommended": recommended_matrix.tolist(),
            "params": matrix_to_density_params(recommended_matrix),
        },
        "quality": {
            "density_chroma_before": before_chroma_error,
            "density_chroma_raw_fit": fitted_chroma_error,
            "density_chroma_recommended": recommended_chroma_error,
            "positive_proxy_before": positive_before_stats,
            "positive_proxy_recommended": positive_recommended_stats,
        },
        "stability": {
            "film_base_chroma_mean": {
                "r_over_g": float(film_base_chroma[:, 0].mean()),
                "b_over_g": float(film_base_chroma[:, 2].mean()),
            },
            "film_base_chroma_std": {
                "r_over_g": float(film_base_chroma[:, 0].std()),
                "b_over_g": float(film_base_chroma[:, 2].std()),
            },
            "per_image_normalization": per_image_norm,
        },
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_matrix_comparison(debug_dir / "look_matrix_color_clouds.jpg", fit_source, fit_target, prior, recommended_matrix, tone_curve)

    print("\nLook preset result")
    print(f"Used pairs: {len(samples_by_image)}")
    print(f"Total samples: {len(fit_source)}")
    print("Recommended density matrix:\n", np.array2string(recommended_matrix, precision=5))
    print("Density matrix params:", json.dumps(report["density_matrix"]["params"], indent=2))
    print(f"Density chroma RMSE before: {before_chroma_error['rmse']:.5f}")
    print(f"Density chroma RMSE recommended: {recommended_chroma_error['rmse']:.5f}")
    print(
        "Film base chroma R/G, B/G:",
        f"{film_base_chroma[:, 0].mean():.5f} +/- {film_base_chroma[:, 0].std():.5f},",
        f"{film_base_chroma[:, 2].mean():.5f} +/- {film_base_chroma[:, 2].std():.5f}",
    )
    print("Preset:", out_path)
    return 0


def density_params_to_matrix(params: DensityMatrixParams) -> np.ndarray:
    return np.array(
        [
            [params.m00, params.m01, params.m02],
            [params.m10, params.m11, params.m12],
            [params.m20, params.m21, params.m22],
        ],
        dtype=np.float32,
    )


def matrix_to_density_params(matrix: np.ndarray) -> dict[str, float]:
    keys = ("m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22")
    return {key: float(value) for key, value in zip(keys, matrix.reshape(-1))}


def normalize_matrix_rows(matrix: np.ndarray) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=np.float32).copy()
    row_sums = normalized.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) < 1e-6, 1.0, row_sums)
    return (normalized / row_sums).astype(np.float32)


def build_density_fit_set(
    samples_by_image: list[ImageSamples],
    tone_curve: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_image: list[dict[str, object]] = []

    for item in samples_by_image:
        source_density = item.density_rgb.astype(np.float32, copy=False)
        target_density = invert_tone_curve(item.reference_rgb, tone_curve)
        source_luma = rgb_luminance(source_density)
        target_chroma = density_chroma(target_density)
        neutral_mask = reference_neutral_mask(item.reference_rgb)
        if int(np.count_nonzero(neutral_mask)) < 32:
            neutral_mask = np.ones(len(source_density), dtype=bool)

        target_chroma_bias = np.median(target_chroma[neutral_mask], axis=0).astype(np.float32)
        normalized_target = source_luma[:, None] + (target_chroma - target_chroma_bias.reshape(1, 3))

        valid = (
            np.all(np.isfinite(source_density), axis=1)
            & np.all(np.isfinite(normalized_target), axis=1)
            & (source_luma > 0.04)
            & (source_luma < 1.75)
            & np.all(normalized_target > -0.35, axis=1)
            & np.all(normalized_target < 2.5, axis=1)
        )
        if int(np.count_nonzero(valid)) < 16:
            continue

        sources.append(source_density[valid])
        targets.append(normalized_target[valid].astype(np.float32))
        per_image.append(
            {
                "name": item.name,
                "samples": int(np.count_nonzero(valid)),
                "target_chroma_bias": target_chroma_bias.tolist(),
                "neutral_samples": int(np.count_nonzero(neutral_mask)),
            }
        )

    if not sources:
        raise RuntimeError("No usable samples after look normalization.")
    return np.vstack(sources).astype(np.float32), np.vstack(targets).astype(np.float32), per_image


def invert_tone_curve(reference_rgb: np.ndarray, tone_curve: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(tone_curve["x"], dtype=np.float32)
    y = np.asarray(tone_curve["y"], dtype=np.float32)
    unique_y, indices = np.unique(y, return_index=True)
    unique_x = x[indices]
    if len(unique_y) < 2:
        raise RuntimeError("Tone curve cannot be inverted.")
    flat = np.clip(reference_rgb.reshape(-1), float(unique_y[0]), float(unique_y[-1]))
    inverted = np.interp(flat, unique_y, unique_x, left=float(unique_x[0]), right=float(unique_x[-1]))
    return inverted.reshape(reference_rgb.shape).astype(np.float32)


def density_chroma(density_rgb: np.ndarray) -> np.ndarray:
    luma = rgb_luminance(density_rgb)
    return (density_rgb - luma[:, None]).astype(np.float32)


def reference_neutral_mask(reference_rgb: np.ndarray) -> np.ndarray:
    luma = rgb_luminance(reference_rgb)
    chroma = reference_rgb.max(axis=1) - reference_rgb.min(axis=1)
    saturation = chroma / np.maximum(luma, 1e-4)
    mid = (luma > 0.08) & (luma < 0.88)
    if int(np.count_nonzero(mid)) < 64:
        mid = (luma > 0.04) & (luma < 0.94)
    if int(np.count_nonzero(mid)) < 64:
        mid = np.ones(len(reference_rgb), dtype=bool)
    cutoff = float(np.percentile(saturation[mid], 28.0))
    mask = mid & (saturation <= max(cutoff, 0.08))
    if int(np.count_nonzero(mask)) < 32:
        cutoff = float(np.percentile(saturation[mid], 45.0))
        mask = mid & (saturation <= cutoff)
    return mask


def fit_row_sum_density_matrix(
    source: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray,
    *,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    x = source[valid]
    y = target[valid]
    if len(x) < 128:
        raise RuntimeError("Not enough samples to fit matrix.")

    keep = np.ones(len(x), dtype=bool)
    matrix = prior.copy()
    for _iteration in range(5):
        for row in range(3):
            matrix[row] = solve_row_sum_ridge(x[keep], y[keep, row], prior[row], regularization=regularization)
        prediction = np.maximum(x @ matrix.T, 0.0)
        error = np.linalg.norm(density_chroma(prediction) - density_chroma(y), axis=1)
        threshold = float(np.percentile(error, 88.0))
        next_keep = error <= threshold
        if int(np.count_nonzero(next_keep)) < 128:
            break
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep

    full_keep = np.zeros(len(source), dtype=bool)
    full_keep[np.flatnonzero(valid)[keep]] = True
    return normalize_matrix_rows(matrix), full_keep


def solve_row_sum_ridge(
    x: np.ndarray,
    y: np.ndarray,
    prior_row: np.ndarray,
    *,
    regularization: float,
) -> np.ndarray:
    hessian = x.T @ x + np.eye(3, dtype=np.float32) * float(regularization)
    gradient = x.T @ y + prior_row.astype(np.float32) * float(regularization)
    constraint = np.ones((1, 3), dtype=np.float32)
    lhs = np.block(
        [
            [hessian, constraint.T],
            [constraint, np.zeros((1, 1), dtype=np.float32)],
        ]
    )
    rhs = np.concatenate([gradient, np.array([1.0], dtype=np.float32)])
    solution = np.linalg.solve(lhs, rhs)
    return solution[:3].astype(np.float32)


def chroma_error_stats(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta = density_chroma(prediction) - density_chroma(target)
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "p95_abs": float(np.percentile(np.abs(delta), 95.0)),
    }


def draw_matrix_comparison(
    path: Path,
    source_density: np.ndarray,
    target_density: np.ndarray,
    prior: np.ndarray,
    recommended: np.ndarray,
    tone_curve: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(8721)
    count = min(6000, len(source_density))
    indices = rng.choice(len(source_density), size=count, replace=False)
    before = linear_to_srgb8(apply_tone_curve(np.maximum(source_density[indices] @ prior.T, 0.0), tone_curve))
    after = linear_to_srgb8(apply_tone_curve(np.maximum(source_density[indices] @ recommended.T, 0.0), tone_curve))
    target = linear_to_srgb8(apply_tone_curve(np.maximum(target_density[indices], 0.0), tone_curve))

    width = 420
    image = Image.new("RGB", (width * 3, width + 42), (20, 22, 26))
    draw = ImageDraw.Draw(image)
    for column, (label, data) in enumerate((("Before", before), ("Preset", after), ("Reference proxy", target))):
        image.paste(color_cloud_image(data, width, width), (column * width, 42))
        draw.text((column * width + 16, 14), label, fill=(235, 238, 242))
    image.save(path, quality=92)


if __name__ == "__main__":
    raise SystemExit(main())
