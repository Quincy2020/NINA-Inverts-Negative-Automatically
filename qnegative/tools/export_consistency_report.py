from __future__ import annotations

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from qnegative.core.file_sequence import RAW_EXTENSIONS
from qnegative.core.models import ImageSize, ImageProcessingState
from qnegative.core.pipeline import (
    analysis_inset_crop,
    analysis_inset_from_adjustments,
    build_lab_print_base_preview,
    build_lab_print_color_stage,
    build_lab_print_export_linear,
    build_lab_print_levels_stage,
    build_lab_print_negative_stage,
    build_lab_print_display_stage,
    linear_to_srgb8,
    suggest_lab_print_luminance_levels,
)
from qnegative.core.preview import DEFAULT_PREVIEW_MAX_EDGE, RawPreview, make_raw_preview, resize_long_edge
from qnegative.core.raw_loader import RawRgbImage, load_raw_rgb16
from qnegative.core.session import load_roll_color_result, load_roll_session
from qnegative.ui.export_tasks import encode_export_rgb, transform_preview_array, write_export_image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare NINA preview, full-resolution render, and TIFF readback color."
    )
    parser.add_argument("--source", type=Path, default=None, help="RAW/DNG source file.")
    parser.add_argument("--folder", type=Path, default=Path(r"D:\QNegativeLab\negative file\seaweed"))
    parser.add_argument("--name", default=None, help="File name inside --folder.")
    parser.add_argument("--out", type=Path, default=Path("debug/export_consistency"))
    parser.add_argument("--preview-max-edge", type=int, default=DEFAULT_PREVIEW_MAX_EDGE)
    parser.add_argument("--format", default="tiff16", choices=["tiff16", "tiff8", "png16", "png8", "jpg"])
    args = parser.parse_args()

    source_path = resolve_source_path(args.source, args.folder, args.name)
    folder = source_path.parent
    files = sorted(path for path in folder.iterdir() if path.suffix.lower() in RAW_EXTENSIONS)
    states = load_roll_session(folder, files)
    state = states.get(source_path)
    if state is None:
        raise SystemExit(f"No saved .nina state for {source_path.name}")
    if state.film_rect is None or not state.film_rect.is_valid():
        raise SystemExit(f"No valid frame in saved .nina state for {source_path.name}")

    roll_color_result = load_roll_color_result(folder)
    out_dir = args.out / source_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fast_preview = make_raw_preview(source_path, max_size=args.preview_max_edge)
    fast = render_preview_pipeline(
        fast_preview,
        state,
        roll_color_result=roll_color_result,
    )

    raw_full = load_raw_rgb16(
        source_path,
        half_size=False,
        include_display_transform=state.adjustments.camera_color_strength > 0,
    )
    full = render_full_export_pipeline(
        source_path,
        raw_full,
        state,
        roll_color_result=roll_color_result,
        tone_mid_anchor=fast["tone_mid_anchor"],
        preview_log_floors=fast["log_floors"],
        preview_log_ceils=fast["log_ceils"],
    )

    export_readback = write_and_read_export(
        full["linear_rgb"],
        state,
        export_format=args.format,
    )

    target_size = (fast["srgb8"].shape[1], fast["srgb8"].shape[0])
    accurate_down = resize_to(full["srgb8"], target_size)
    readback_down = resize_to(export_readback["srgb8"], target_size)

    write_png(out_dir / "fast_preview.png", fast["srgb8"])
    write_png(out_dir / "accurate_fullres_downscaled.png", accurate_down)
    write_png(out_dir / "export_readback_downscaled.png", readback_down)
    write_png(out_dir / "delta_fast_vs_export.png", delta_heatmap(fast["srgb8"], readback_down))
    write_png(out_dir / "delta_accurate_vs_export.png", delta_heatmap(accurate_down, readback_down))
    write_png(out_dir / "yellow_green_fast_vs_export.png", yellow_green_map(fast["srgb8"], readback_down))
    write_png(out_dir / "yellow_green_accurate_vs_export.png", yellow_green_map(accurate_down, readback_down))
    write_histogram_plot(
        out_dir / "histogram_compare.png",
        {
            "fast": fast["srgb8"],
            "accurate": accurate_down,
            "export": readback_down,
        },
    )

    report = {
        "source": str(source_path),
        "preview_max_edge": int(args.preview_max_edge),
        "export_format": args.format,
        "fast_preview_shape": list(fast["srgb8"].shape),
        "full_export_shape": list(full["srgb8"].shape),
        "cmy": {
            "fast_preview": float_list(fast["cmy_offsets"]),
            "full_export": float_list(full["cmy_offsets"]),
            "delta_export_minus_fast": float_list(full["cmy_offsets"] - fast["cmy_offsets"]),
        },
        "levels": {
            "fast_preview": fast["levels"],
            "full_export": full["levels"],
        },
        "tone_mid_anchor": {
            "fast_preview": float(fast["tone_mid_anchor"]),
            "used_for_export": float(full["tone_mid_anchor"]),
        },
        "comparisons": {
            "export_minus_fast_preview": compare_images(readback_down, fast["srgb8"]),
            "export_minus_accurate_fullres": compare_images(readback_down, accurate_down),
            "accurate_fullres_minus_fast_preview": compare_images(accurate_down, fast["srgb8"]),
        },
        "notes": [
            "Positive Lab delta_b means the first image is yellower than the second.",
            "Negative Lab delta_a means the first image is greener than the second.",
            "The accurate_fullres image is generated before TIFF write; export_readback is read from an actual written file.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_summary(report, out_dir)
    return 0


def resolve_source_path(source: Path | None, folder: Path, name: str | None) -> Path:
    if source is not None:
        return source
    if name:
        return folder / name

    files = sorted(path for path in folder.iterdir() if path.suffix.lower() in RAW_EXTENSIONS)
    states = load_roll_session(folder, files)
    for path in files:
        state = states.get(path)
        if (
            state is not None
            and state.negative_preview_active
            and state.film_rect is not None
            and state.film_rect.is_valid()
        ):
            return path
    raise SystemExit(f"No completed framed RAW found in {folder}")


def render_preview_pipeline(
    preview: RawPreview,
    state: ImageProcessingState,
    *,
    roll_color_result: dict | None,
) -> dict[str, Any]:
    adjustments = deepcopy(state.adjustments)
    base = build_lab_print_base_preview(
        preview.preview_linear_rgb,
        source_size=preview.source_size,
        mask_point=state.mask_point,
        film_rect=state.film_rect,
        lens_correction=adjustments.lens_correction,
        preview_camera_wb_linear_rgb=preview.preview_camera_wb_linear_rgb,
        camera_to_srgb_matrix=preview.camera_to_srgb_matrix,
    )
    negative_stage = build_lab_print_negative_stage(
        base,
        analysis_inset=analysis_inset_from_adjustments(adjustments),
    )
    levels = resolved_levels(negative_stage, adjustments, state)
    levels_stage = build_lab_print_levels_stage(
        negative_stage,
        adjustments,
        auto_levels=levels,
    )
    cmy_offsets = state_cmy_offsets(state)
    color_stage = build_lab_print_color_stage(
        levels_stage,
        adjustments,
        cmy_offsets=cmy_offsets if adjustments.auto_wb else None,
    )
    result = build_lab_print_display_stage(
        color_stage,
        adjustments,
        roll_color_result=roll_color_result,
        roll_color_frame=state.roll_color_frame,
    )
    display = transform_preview_array(
        result.display_rgb8,
        flip_horizontal=state.preview_flip_horizontal,
        flip_vertical=state.preview_flip_vertical,
        rotation_quarters=state.preview_rotation_quarters,
    )
    return {
        "srgb8": display,
        "linear_rgb": result.processed_linear_rgb,
        "cmy_offsets": color_stage.wb_gains.copy(),
        "log_floors": negative_stage.lab_print_log_floors.copy(),
        "log_ceils": negative_stage.lab_print_log_ceils.copy(),
        "levels": levels,
        "tone_mid_anchor": float(result.tone_mid_anchor),
    }


def render_full_export_pipeline(
    source_path: Path,
    raw_image: RawRgbImage,
    state: ImageProcessingState,
    *,
    roll_color_result: dict | None,
    tone_mid_anchor: float | None,
    preview_log_floors: np.ndarray | None = None,
    preview_log_ceils: np.ndarray | None = None,
) -> dict[str, Any]:
    del source_path
    adjustments = deepcopy(state.adjustments)
    base = build_lab_print_base_preview(
        raw_image.as_float32(),
        source_size=raw_image.source_size,
        mask_point=state.mask_point,
        film_rect=state.film_rect,
        lens_correction=adjustments.lens_correction,
        preview_camera_wb_linear_rgb=raw_image.camera_wb_as_float32(),
        camera_to_srgb_matrix=raw_image.camera_to_srgb_matrix,
    )
    negative_stage = build_lab_print_negative_stage(
        base,
        include_histogram=False,
        analysis_inset=analysis_inset_from_adjustments(adjustments),
        lab_print_log_floors=preview_log_floors,
        lab_print_log_ceils=preview_log_ceils,
    )
    levels = resolved_levels(negative_stage, adjustments, state)
    levels_stage = build_lab_print_levels_stage(
        negative_stage,
        adjustments,
        auto_levels=levels,
    )
    cmy_offsets = state_cmy_offsets(state)
    color_stage = build_lab_print_color_stage(
        levels_stage,
        adjustments,
        cmy_offsets=cmy_offsets if adjustments.auto_wb else None,
    )
    linear = build_lab_print_export_linear(
        color_stage,
        adjustments,
        roll_color_result=roll_color_result,
        roll_color_frame=state.roll_color_frame,
        tone_mid_anchor=tone_mid_anchor,
    )
    transformed_linear = transform_preview_array(
        linear,
        flip_horizontal=state.preview_flip_horizontal,
        flip_vertical=state.preview_flip_vertical,
        rotation_quarters=state.preview_rotation_quarters,
    )
    return {
        "linear_rgb": transformed_linear,
        "srgb8": linear_to_srgb8(transformed_linear),
        "cmy_offsets": color_stage.wb_gains.copy(),
        "log_floors": negative_stage.lab_print_log_floors.copy(),
        "log_ceils": negative_stage.lab_print_log_ceils.copy(),
        "levels": levels,
        "tone_mid_anchor": float(tone_mid_anchor) if tone_mid_anchor is not None else None,
    }


def resolved_levels(negative_stage, adjustments, state: ImageProcessingState) -> dict[str, int]:
    if state.auto_levels_pending:
        levels = suggest_lab_print_luminance_levels(
            analysis_inset_crop(negative_stage.normalized_log, negative_stage.analysis_inset),
            adjustments,
            camera_to_srgb_matrix=negative_stage.camera_to_srgb_matrix,
        )
        adjustments.black_point = levels["black_point"]
        adjustments.mid_point = levels["mid_point"]
        adjustments.white_point = levels["white_point"]
        return levels
    return {
        "black_point": int(adjustments.black_point),
        "mid_point": int(adjustments.mid_point),
        "white_point": int(adjustments.white_point),
    }


def state_cmy_offsets(state: ImageProcessingState) -> np.ndarray | None:
    if not state.adjustments.auto_wb:
        return None
    if state.lab_print_cmy_offsets is None:
        return None
    return np.asarray(state.lab_print_cmy_offsets, dtype=np.float32).reshape(3).copy()


def write_and_read_export(
    linear_rgb: np.ndarray,
    state: ImageProcessingState,
    *,
    export_format: str,
) -> dict[str, np.ndarray]:
    encoded = encode_export_rgb(linear_rgb, export_format)
    suffix = ".tif" if export_format.startswith("tiff") else ".png"
    if export_format == "jpg":
        suffix = ".jpg"
    with tempfile.TemporaryDirectory(prefix="nina_export_report_") as temp_dir:
        path = Path(temp_dir) / f"export_readback{suffix}"
        write_export_image(path, encoded, export_format)
        if export_format.startswith("tiff"):
            readback = tifffile.imread(path)
            srgb8 = encoded_to_srgb8(readback)
        else:
            bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise OSError(f"Could not read {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            srgb8 = encoded_to_srgb8(rgb)
    del state
    return {"srgb8": srgb8}


def encoded_to_srgb8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image[:, :, :3])
    if image.dtype == np.uint16:
        return np.ascontiguousarray((image[:, :, :3].astype(np.uint32) + 128) // 257).astype(np.uint8)
    clipped = np.clip(image[:, :, :3].astype(np.float32), 0.0, 1.0)
    return np.ascontiguousarray((clipped * 255.0 + 0.5).astype(np.uint8))


def resize_to(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    if image.shape[1] == width and image.shape[0] == height:
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))


def compare_images(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    if first.shape != second.shape:
        raise ValueError(f"Image shapes differ: {first.shape} vs {second.shape}")
    first_f = first.astype(np.float32)
    second_f = second.astype(np.float32)
    diff = first_f - second_f
    abs_diff = np.abs(diff)
    first_lab = rgb8_to_lab_float(first)
    second_lab = rgb8_to_lab_float(second)
    lab_delta = first_lab - second_lab
    first_luma = rgb_luma(first_f)
    second_luma = rgb_luma(second_f)
    first_chroma = lab_chroma(first_lab)
    second_chroma = lab_chroma(second_lab)
    shadow_mask = second_luma <= np.percentile(second_luma, 25.0)
    return {
        "mean_rgb_delta": float_list(diff.reshape(-1, 3).mean(axis=0)),
        "median_rgb_delta": float_list(np.median(diff.reshape(-1, 3), axis=0)),
        "mean_abs_rgb_diff": float(abs_diff.mean()),
        "p95_abs_rgb_diff": float(np.percentile(abs_diff, 95.0)),
        "max_abs_rgb_diff": float(abs_diff.max()),
        "mean_lab_delta": float_list(lab_delta.reshape(-1, 3).mean(axis=0)),
        "median_lab_delta": float_list(np.median(lab_delta.reshape(-1, 3), axis=0)),
        "mean_delta_a_green_negative": float(lab_delta[:, :, 1].mean()),
        "mean_delta_b_yellow_positive": float(lab_delta[:, :, 2].mean()),
        "luma": {
            "first": percentile_summary(first_luma),
            "second": percentile_summary(second_luma),
            "delta_mean": float((first_luma - second_luma).mean()),
            "delta_shadow_mean": float((first_luma[shadow_mask] - second_luma[shadow_mask]).mean()),
        },
        "contrast_p95_minus_p5": {
            "first": float(np.percentile(first_luma, 95.0) - np.percentile(first_luma, 5.0)),
            "second": float(np.percentile(second_luma, 95.0) - np.percentile(second_luma, 5.0)),
        },
        "chroma": {
            "first_mean": float(first_chroma.mean()),
            "second_mean": float(second_chroma.mean()),
            "delta_mean": float((first_chroma - second_chroma).mean()),
        },
    }


def rgb8_to_lab_float(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def lab_chroma(lab: np.ndarray) -> np.ndarray:
    return np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)


def rgb_luma(image: np.ndarray) -> np.ndarray:
    return image[:, :, 0] * 0.2126 + image[:, :, 1] * 0.7152 + image[:, :, 2] * 0.0722


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "p1": float(np.percentile(values, 1.0)),
        "p5": float(np.percentile(values, 5.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "mean": float(values.mean()),
    }


def delta_heatmap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    diff = np.abs(first.astype(np.float32) - second.astype(np.float32)).mean(axis=2)
    scaled = np.clip(diff * 4.0, 0.0, 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def yellow_green_map(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = rgb8_to_lab_float(first) - rgb8_to_lab_float(second)
    yellow = np.clip(delta[:, :, 2], 0.0, 24.0) / 24.0
    green = np.clip(-delta[:, :, 1], 0.0, 24.0) / 24.0
    magnitude = np.clip((yellow + green) * 0.5, 0.0, 1.0)
    out = np.zeros((*magnitude.shape, 3), dtype=np.float32)
    out[:, :, 0] = yellow * 255.0
    out[:, :, 1] = np.maximum(yellow, green) * 255.0
    out[:, :, 2] = green * 120.0
    return np.clip(out * np.maximum(magnitude[:, :, None], 0.25), 0.0, 255.0).astype(np.uint8)


def write_histogram_plot(path: Path, images: dict[str, np.ndarray]) -> None:
    canvas = np.full((360, 780, 3), 18, dtype=np.uint8)
    colors = {
        "fast": (255, 176, 0),
        "accurate": (90, 210, 255),
        "export": (230, 230, 230),
    }
    for row, channel_name in enumerate(("R", "G", "B")):
        y0 = 22 + row * 112
        cv2.putText(canvas, channel_name, (14, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.line(canvas, (52, y0 + 90), (750, y0 + 90), (65, 58, 48), 1)
        for label, image in images.items():
            channel = image[:, :, row].reshape(-1)
            hist, _bins = np.histogram(channel, bins=256, range=(0, 255))
            hist = hist.astype(np.float32)
            hist = hist / max(1.0, float(hist.max()))
            points = []
            for index, value in enumerate(hist):
                x = 54 + int(index / 255.0 * 694)
                y = y0 + 90 - int(value * 82)
                points.append((x, y))
            bgr = colors[label][::-1]
            for a, b in zip(points[:-1], points[1:]):
                cv2.line(canvas, a, b, bgr, 1)
    cv2.putText(canvas, "fast", (560, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["fast"][::-1], 1)
    cv2.putText(canvas, "accurate", (620, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["accurate"][::-1], 1)
    cv2.putText(canvas, "export", (710, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["export"][::-1], 1)
    cv2.imwrite(str(path), canvas)


def write_png(path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def float_list(values: np.ndarray | list[float]) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def print_summary(report: dict[str, Any], out_dir: Path) -> None:
    print(f"Report: {out_dir}")
    print(f"Source: {report['source']}")
    print(f"CMY delta export-fast: {report['cmy']['delta_export_minus_fast']}")
    for name, comparison in report["comparisons"].items():
        lab = comparison["mean_lab_delta"]
        rgb = comparison["mean_rgb_delta"]
        print(
            f"{name}: mean RGB delta {np.round(rgb, 3).tolist()}, "
            f"mean Lab delta L/a/b {np.round(lab, 3).tolist()}, "
            f"abs mean {comparison['mean_abs_rgb_diff']:.3f}, "
            f"p95 {comparison['p95_abs_rgb_diff']:.3f}, "
            f"chroma delta {comparison['chroma']['delta_mean']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
