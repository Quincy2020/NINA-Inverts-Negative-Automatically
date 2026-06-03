from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from qnegative.core.frame_ranker import detect_ranked_frame_candidates
from qnegative.core.geometry import rotated_rect_corners, scale_point, scale_rect
from qnegative.core.models import ImagePoint, ImageRect, ImageSize
from qnegative.core.preview import resize_long_edge


DETECT_MAX_EDGE = 1400
FRAME_CONFIDENCE_HIGH = 0.72
FRAME_CONFIDENCE_MEDIUM = 0.68
FRAME_MAX_AREA_RATIO = 0.70
CENTERED_EDGE_INSET_RATIO = 0.025
CENTERED_EDGE_CONFIDENCE_HIGH = 0.68
BASE_CONFIDENCE_HIGH = 0.68
BASE_CONFIDENCE_MEDIUM = 0.45

FORMAT_RATIOS = {
    "auto": (1.50, 1.33, 1.00, 7.0 / 6.0, 1.50),
    "135": (1.50,),
    "645": (4.0 / 3.0,),
    "66": (1.00,),
    "67": (7.0 / 6.0,),
    "69": (1.50,),
}


@dataclass(frozen=True)
class AutoFrameResult:
    rect: ImageRect
    confidence: float
    confidence_level: str
    format_hint: str
    method: str


@dataclass(frozen=True)
class AutoBaseResult:
    point: ImagePoint | None
    rgb: np.ndarray | None
    confidence: float
    confidence_level: str
    source: str


@dataclass(frozen=True)
class AutoDetectResult:
    frame: AutoFrameResult | None
    base: AutoBaseResult | None


@dataclass(frozen=True)
class _FrameCandidate:
    rect: ImageRect
    confidence: float
    format_hint: str
    method: str


@dataclass(frozen=True)
class _BaseCandidate:
    point: ImagePoint
    rgb: np.ndarray
    score: float
    confidence: float
    source: str


def detect_frame_and_base(
    preview_linear_rgb: np.ndarray,
    *,
    preview_size: ImageSize,
    source_size: ImageSize,
    format_hint: str = "auto",
    detect_base: bool = True,
    prior_frame_rect: ImageRect | None = None,
) -> AutoDetectResult:
    frame = detect_film_frame(
        preview_linear_rgb,
        preview_size=preview_size,
        source_size=source_size,
        format_hint=format_hint,
        prior_frame_rect=prior_frame_rect,
    )
    if not detect_base:
        return AutoDetectResult(frame=frame, base=None)

    frame_for_base = frame.rect if frame is not None and frame.confidence_level == "high" else None
    base = detect_film_base(
        preview_linear_rgb,
        preview_size=preview_size,
        source_size=source_size,
        frame_rect=frame_for_base,
    )
    return AutoDetectResult(frame=frame, base=base)


def detect_film_frame(
    preview_linear_rgb: np.ndarray,
    *,
    preview_size: ImageSize,
    source_size: ImageSize,
    format_hint: str = "auto",
    prior_frame_rect: ImageRect | None = None,
) -> AutoFrameResult | None:
    image = _prepare_detection_image(preview_linear_rgb)
    if image.size == 0:
        return None

    scaled = resize_long_edge(image, max_size=DETECT_MAX_EDGE)
    detect_size = ImageSize(width=scaled.shape[1], height=scaled.shape[0])
    gray = _normalize_luminance_to_uint8(scaled)
    edges = _edge_map(gray)
    centered_candidate = _centered_edge_frame_candidate(
        gray,
        edges,
        image_size=detect_size,
        format_hint=format_hint,
    )
    if centered_candidate is not None and centered_candidate.confidence >= CENTERED_EDGE_CONFIDENCE_HIGH:
        preview_rect = scale_rect(centered_candidate.rect, detect_size, preview_size)
        source_rect = _fit_rect_to_format_hint(
            scale_rect(preview_rect, preview_size, source_size),
            source_size,
            format_hint,
        )
        source_rect = _inset_axis_aligned_rect(source_rect, CENTERED_EDGE_INSET_RATIO)
        return AutoFrameResult(
            rect=source_rect,
            confidence=round(float(centered_candidate.confidence), 3),
            confidence_level=_confidence_level(
                centered_candidate.confidence,
                FRAME_CONFIDENCE_HIGH,
                FRAME_CONFIDENCE_MEDIUM,
            ),
            format_hint=centered_candidate.format_hint,
            method=centered_candidate.method,
        )

    ranked = _ranker_frame_candidates(
        image,
        preview_size=preview_size,
        source_size=source_size,
        format_hint=format_hint,
        prior_frame_rect=prior_frame_rect,
    )
    if ranked is not None:
        return ranked

    best: _FrameCandidate | None = None
    masks = _candidate_masks(gray, edges)

    image_area = float(gray.shape[0] * gray.shape[1])
    for method, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:48]
        for contour in contours:
            candidate = _score_frame_contour(
                contour,
                edges,
                image_area=image_area,
                image_size=detect_size,
                format_hint=format_hint,
                method=method,
            )
            if candidate is None:
                continue
            if best is None or candidate.confidence > best.confidence:
                best = candidate

    projection_candidate = _projection_frame_candidate(
        gray,
        edges,
        image_size=detect_size,
        format_hint=format_hint,
    )
    if projection_candidate is not None and (best is None or projection_candidate.confidence > best.confidence):
        best = projection_candidate
    if centered_candidate is not None and centered_candidate.confidence >= CENTERED_EDGE_CONFIDENCE_HIGH:
        best = centered_candidate
    elif centered_candidate is not None and (best is None or centered_candidate.confidence > best.confidence):
        best = centered_candidate

    if best is None or best.confidence < FRAME_CONFIDENCE_MEDIUM:
        return None

    preview_rect = scale_rect(best.rect, detect_size, preview_size)
    source_rect = _fit_rect_to_format_hint(
        scale_rect(preview_rect, preview_size, source_size),
        source_size,
        format_hint,
    )
    if best.method == "centered-edge-inset":
        source_rect = _inset_axis_aligned_rect(source_rect, CENTERED_EDGE_INSET_RATIO)
    return AutoFrameResult(
        rect=source_rect,
        confidence=round(float(best.confidence), 3),
        confidence_level=_confidence_level(best.confidence, FRAME_CONFIDENCE_HIGH, FRAME_CONFIDENCE_MEDIUM),
        format_hint=best.format_hint,
        method=best.method,
    )


def _ranker_frame_candidates(
    image: np.ndarray,
    *,
    preview_size: ImageSize,
    source_size: ImageSize,
    format_hint: str,
    prior_frame_rect: ImageRect | None,
) -> AutoFrameResult | None:
    try:
        candidates = detect_ranked_frame_candidates(
            image,
            preview_size=preview_size,
            source_size=source_size,
            format_hint=format_hint,
            top_k=1,
            prior_frame_rect=prior_frame_rect,
        )
    except Exception:
        return None
    if not candidates:
        return None

    best = candidates[0]
    if best.confidence < FRAME_CONFIDENCE_MEDIUM:
        return None
    return AutoFrameResult(
        rect=best.rect,
        confidence=round(float(best.confidence), 3),
        confidence_level=_confidence_level(best.confidence, FRAME_CONFIDENCE_HIGH, FRAME_CONFIDENCE_MEDIUM),
        format_hint=best.format_hint,
        method=best.method,
    )


def detect_film_base(
    preview_linear_rgb: np.ndarray,
    *,
    preview_size: ImageSize,
    source_size: ImageSize,
    frame_rect: ImageRect | None = None,
) -> AutoBaseResult | None:
    image = _prepare_detection_image(preview_linear_rgb)
    if image.size == 0:
        return None

    preview_frame = scale_rect(frame_rect, source_size, preview_size) if frame_rect is not None else None
    candidates = _frame_base_candidates(image, preview_frame)
    if not candidates:
        candidates = _edge_base_candidates(image)
    if not candidates:
        return None

    candidates.sort(key=lambda item: item.score, reverse=True)
    selected = candidates[: min(7, max(3, len(candidates) // 4))]
    rgb_values = np.stack([candidate.rgb for candidate in selected], axis=0)
    median_rgb = np.median(rgb_values, axis=0).astype(np.float32)
    best_point = selected[0].point

    consistency = float(np.mean(np.percentile(rgb_values, 90, axis=0) - np.percentile(rgb_values, 10, axis=0)))
    base_confidence = float(
        np.clip(
            np.mean([candidate.confidence for candidate in selected]) * 0.72
            + np.clip(1.0 - consistency / 0.16, 0.0, 1.0) * 0.20
            + np.clip(len(selected) / 6.0, 0.0, 1.0) * 0.08,
            0.0,
            0.98,
        )
    )
    if base_confidence < BASE_CONFIDENCE_MEDIUM:
        return None

    return AutoBaseResult(
        point=scale_point(best_point, preview_size, source_size),
        rgb=median_rgb,
        confidence=round(base_confidence, 3),
        confidence_level=_confidence_level(base_confidence, BASE_CONFIDENCE_HIGH, BASE_CONFIDENCE_MEDIUM),
        source=selected[0].source,
    )


def _prepare_detection_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        return np.empty((0, 0, 3), dtype=np.float32)
    return np.nan_to_num(image.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)


def _normalize_luminance_to_uint8(image: np.ndarray) -> np.ndarray:
    luminance = image[:, :, 0] * 0.2126 + image[:, :, 1] * 0.7152 + image[:, :, 2] * 0.0722
    valid = luminance[np.isfinite(luminance)]
    if valid.size == 0:
        return np.zeros(luminance.shape, dtype=np.uint8)
    low = float(np.percentile(valid, 1.0))
    high = float(np.percentile(valid, 99.0))
    if high <= low + 1e-6:
        high = low + 1.0
    gray = np.clip((luminance - low) / (high - low), 0.0, 1.0)
    return np.ascontiguousarray((gray * 255.0 + 0.5).astype(np.uint8))


def _edge_map(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 36, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(edges, kernel, iterations=1)


def _candidate_masks(gray: np.ndarray, edges: np.ndarray) -> list[tuple[str, np.ndarray]]:
    masks: list[tuple[str, np.ndarray]] = []
    close_15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    close_31 = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    open_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    edge_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_15, iterations=2)
    masks.append(("edges", edge_mask))

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, close_31, iterations=2)
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, open_5, iterations=1)
    masks.append(("threshold-dark", inv))

    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_31, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, open_5, iterations=1)
    masks.append(("threshold-bright", bright))
    return masks


def _centered_edge_frame_candidate(
    gray: np.ndarray,
    edges: np.ndarray,
    *,
    image_size: ImageSize,
    format_hint: str,
) -> _FrameCandidate | None:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    col_signal = _smooth_signal(np.percentile(np.abs(grad_x), 96, axis=0), 17)
    row_signal = _smooth_signal(np.percentile(np.abs(grad_y), 96, axis=1), 17)

    center_x = image_size.width // 2
    center_y = image_size.height // 2
    left = _center_out_boundary(col_signal, center_x, -1)
    right = _center_out_boundary(col_signal, center_x, 1)
    top = _center_out_boundary(row_signal, center_y, -1)
    bottom = _center_out_boundary(row_signal, center_y, 1)
    if left is None or right is None or top is None or bottom is None:
        return None

    left_idx, left_strength = left
    right_idx, right_strength = right
    top_idx, top_strength = top
    bottom_idx, bottom_strength = bottom
    width = right_idx - left_idx + 1
    height = bottom_idx - top_idx + 1
    if width <= 0 or height <= 0:
        return None
    if width < image_size.width * 0.30 or height < image_size.height * 0.30:
        return None

    rect = ImageRect(
        x=int(left_idx),
        y=int(top_idx),
        width=int(width),
        height=int(height),
        angle=0.0,
    )
    area_ratio = (width * height) / max(1.0, float(image_size.width * image_size.height))
    if area_ratio < 0.14 or area_ratio > 0.98:
        return None

    strength_values = np.array(
        [left_strength, right_strength, top_strength, bottom_strength],
        dtype=np.float32,
    )
    signal_values = np.concatenate([col_signal.reshape(-1), row_signal.reshape(-1)])
    signal_reference = max(1.0, float(np.percentile(signal_values, 92)))
    strength_score = float(np.clip(np.mean(strength_values) / signal_reference, 0.0, 1.0))
    balance_score = 1.0 - float(
        np.clip(np.std(strength_values) / max(1.0, np.mean(strength_values)), 0.0, 1.0)
    )
    aspect_score, matched_format = _aspect_score(width / max(height, 1), format_hint)
    edge_support = _edge_support(edges, rotated_rect_corners(rect))
    center_prior = _center_prior(rect.center_x, rect.center_y, image_size)
    area_score = 1.0 - float(np.clip(abs(area_ratio - 0.55) / 0.55, 0.0, 1.0))

    confidence = float(
        np.clip(
            strength_score * 0.34
            + edge_support * 0.18
            + aspect_score * 0.20
            + center_prior * 0.14
            + area_score * 0.08
            + balance_score * 0.06,
            0.0,
            0.96,
        )
    )
    if confidence < FRAME_CONFIDENCE_MEDIUM:
        return None

    return _FrameCandidate(
        rect=rect,
        confidence=confidence,
        format_hint=matched_format,
        method="centered-edge-inset",
    )


def _center_out_boundary(signal: np.ndarray, center: int, direction: int) -> tuple[int, float] | None:
    length = signal.size
    if length < 32:
        return None

    margin = max(2, int(round(length * 0.018)))
    min_distance = max(6, int(round(length * 0.10)))
    search_start = margin if direction < 0 else min(length - margin, center + min_distance)
    search_stop = max(margin, center - min_distance) if direction < 0 else length - margin
    search = signal[search_start:search_stop]
    if search.size == 0:
        return None
    noise_floor = float(np.percentile(signal, 72))
    idx = int(search_start + np.argmax(search))
    value = float(signal[idx])
    if value >= max(noise_floor * 1.12, noise_floor + 1.0):
        return idx, value
    return None


def _projection_frame_candidate(
    gray: np.ndarray,
    edges: np.ndarray,
    *,
    image_size: ImageSize,
    format_hint: str,
) -> _FrameCandidate | None:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    col_signal = _smooth_signal(np.percentile(np.abs(grad_x), 92, axis=0), 25)
    row_signal = _smooth_signal(np.percentile(np.abs(grad_y), 92, axis=1), 25)

    left_candidates = _inner_boundaries(col_signal, from_start=True)
    right_candidates = _inner_boundaries(col_signal, from_start=False)
    top_candidates = _inner_boundaries(row_signal, from_start=True)
    bottom_candidates = _inner_boundaries(row_signal, from_start=False)
    if not left_candidates or not right_candidates or not top_candidates or not bottom_candidates:
        return None

    best: _FrameCandidate | None = None
    min_width = int(image_size.width * 0.28)
    min_height = int(image_size.height * 0.28)
    for left_idx, left_strength in left_candidates:
        for right_idx, right_strength in right_candidates:
            width = right_idx - left_idx + 1
            if width < min_width:
                continue
            for top_idx, top_strength in top_candidates:
                for bottom_idx, bottom_strength in bottom_candidates:
                    height = bottom_idx - top_idx + 1
                    if height < min_height:
                        continue
                    if right_idx <= left_idx or bottom_idx <= top_idx:
                        continue

                    rect = ImageRect(
                        x=int(left_idx),
                        y=int(top_idx),
                        width=int(width),
                        height=int(height),
                        angle=0.0,
                    )
                    area_ratio = (width * height) / max(1.0, float(image_size.width * image_size.height))
                    if area_ratio < 0.10 or area_ratio > FRAME_MAX_AREA_RATIO:
                        continue
                    if min(width, height) < min(image_size.width, image_size.height) * 0.16:
                        continue

                    corners = rotated_rect_corners(rect)
                    aspect_score, matched_format = _aspect_score(width / max(height, 1), format_hint)
                    edge_support = _edge_support(edges, corners)
                    border_clearance = _border_clearance(corners, image_size)
                    if border_clearance < 0.12 and area_ratio > 0.62:
                        continue
                    strength_values = np.array(
                        [left_strength, right_strength, top_strength, bottom_strength],
                        dtype=np.float32,
                    )
                    strength_score = float(np.clip(np.mean(strength_values) / 38.0, 0.0, 1.0))
                    area_score = 1.0 - np.clip(abs(area_ratio - 0.38) / 0.38, 0.0, 1.0)
                    center_prior = _center_prior(rect.center_x, rect.center_y, image_size)

                    confidence = float(
                        np.clip(
                            strength_score * 0.28
                            + edge_support * 0.20
                            + aspect_score * 0.22
                            + border_clearance * 0.10
                            + area_score * 0.12
                            + center_prior * 0.08,
                            0.0,
                            0.98,
                        )
                    )
                    if confidence < FRAME_CONFIDENCE_MEDIUM:
                        continue
                    candidate = _FrameCandidate(
                        rect=rect,
                        confidence=confidence,
                        format_hint=matched_format,
                        method="projection-inner",
                    )
                    if best is None or candidate.confidence > best.confidence:
                        best = candidate

    return best


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if signal.size == 0 or window <= 1:
        return signal.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def _inner_boundaries(signal: np.ndarray, *, from_start: bool) -> list[tuple[int, float]]:
    length = signal.size
    if length < 32:
        return []
    low = int(round(length * 0.02))
    high = int(round(length * 0.49))
    if not from_start:
        start = int(round(length * 0.51))
        stop = int(round(length * 0.98))
    else:
        start = low
        stop = high
    start = max(0, min(start, length - 1))
    stop = max(start + 1, min(stop, length))
    search = signal[start:stop]
    if search.size == 0:
        return []
    noise_floor = float(np.percentile(signal, 74))
    ranked = np.argsort(search)[::-1]
    candidates: list[tuple[int, float]] = []
    for offset in ranked[:32]:
        idx = int(start + offset)
        value = float(signal[idx])
        if value < max(noise_floor * 1.4, noise_floor + 3.5):
            break
        if any(abs(idx - existing_idx) < 6 for existing_idx, _ in candidates):
            continue
        candidates.append((idx, value))
        if len(candidates) >= 4:
            break
    return candidates


def _score_frame_contour(
    contour: np.ndarray,
    edges: np.ndarray,
    *,
    image_area: float,
    image_size: ImageSize,
    format_hint: str,
    method: str,
) -> _FrameCandidate | None:
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rect
    if width <= 1 or height <= 1:
        return None
    if width < height:
        width, height = height, width
        angle += 90.0
    angle = _normalize_rect_angle(angle)

    rect_area = float(width * height)
    area_ratio = rect_area / max(image_area, 1.0)
    if area_ratio < 0.10 or area_ratio > FRAME_MAX_AREA_RATIO:
        return None
    if min(width, height) < min(image_size.width, image_size.height) * 0.16:
        return None

    contour_area = float(cv2.contourArea(contour))
    rectangularity = np.clip(contour_area / max(rect_area, 1.0), 0.0, 1.0)
    aspect = width / max(height, 1.0)
    aspect_score, matched_format = _aspect_score(aspect, format_hint)
    edge_support = _edge_support(edges, cv2.boxPoints(rect))
    center_prior = _center_prior(center_x, center_y, image_size)
    border_clearance = _border_clearance(cv2.boxPoints(rect), image_size)
    if border_clearance < 0.12 and area_ratio > 0.62:
        return None
    angle_score = 1.0 - np.clip(abs(angle) / 18.0, 0.0, 1.0) * 0.35
    area_score = 1.0 - np.clip(abs(area_ratio - 0.38) / 0.38, 0.0, 1.0)

    confidence = float(
        np.clip(
            area_score * 0.14
            + rectangularity * 0.20
            + edge_support * 0.22
            + center_prior * 0.12
            + aspect_score * 0.20
            + border_clearance * 0.08
            + angle_score * 0.04,
            0.0,
            1.0,
        )
    )
    return _FrameCandidate(
        rect=ImageRect(
            x=max(0, int(round(center_x - width / 2.0))),
            y=max(0, int(round(center_y - height / 2.0))),
            width=max(1, int(round(width))),
            height=max(1, int(round(height))),
            angle=angle,
        ),
        confidence=confidence,
        format_hint=matched_format,
        method=method,
    )


def _normalize_rect_angle(angle: float) -> float:
    normalized = float(angle)
    while normalized > 45.0:
        normalized -= 90.0
    while normalized <= -45.0:
        normalized += 90.0
    return normalized


def _aspect_score(aspect: float, format_hint: str) -> tuple[float, str]:
    targets = FORMAT_RATIOS.get(format_hint, FORMAT_RATIOS["auto"])
    best_score = 0.0
    best_label = "auto"
    labels = ["135", "645", "66", "67", "69"] if format_hint == "auto" else [format_hint]
    for index, target in enumerate(targets):
        target_aspect = max(0.05, float(target))
        delta = abs(math.log(max(aspect, 0.05)) - math.log(target_aspect))
        score = 1.0 - np.clip(delta / 0.42, 0.0, 1.0)
        if score > best_score:
            best_score = float(score)
            best_label = labels[min(index, len(labels) - 1)]
    return best_score, best_label


def _fit_rect_to_format_hint(rect: ImageRect, image_size: ImageSize, format_hint: str) -> ImageRect:
    if format_hint == "auto" or format_hint not in FORMAT_RATIOS:
        return rect
    target = float(FORMAT_RATIOS[format_hint][0])
    current = rect.width / max(rect.height, 1)
    if current < 1.0:
        target = 1.0 / max(target, 1e-5)

    width = float(rect.width)
    height = float(rect.height)
    if width / max(height, 1e-5) > target:
        width = height * target
    else:
        height = width / max(target, 1e-5)
    if width > image_size.width:
        scale = image_size.width / max(width, 1e-5)
        width *= scale
        height *= scale
    if height > image_size.height:
        scale = image_size.height / max(height, 1e-5)
        width *= scale
        height *= scale

    center_x = rect.center_x
    center_y = rect.center_y
    fitted = ImageRect(
        x=int(round(center_x - width * 0.5)),
        y=int(round(center_y - height * 0.5)),
        width=max(1, int(round(width))),
        height=max(1, int(round(height))),
        angle=rect.angle,
    )
    return ImageRect(
        x=max(0, min(fitted.x, max(0, image_size.width - fitted.width))),
        y=max(0, min(fitted.y, max(0, image_size.height - fitted.height))),
        width=min(fitted.width, image_size.width),
        height=min(fitted.height, image_size.height),
        angle=fitted.angle,
    )


def _inset_axis_aligned_rect(rect: ImageRect, inset_ratio: float) -> ImageRect:
    inset = float(np.clip(inset_ratio, 0.0, 0.12))
    dx = int(round(rect.width * inset))
    dy = int(round(rect.height * inset))
    return ImageRect(
        x=rect.x + dx,
        y=rect.y + dy,
        width=max(1, rect.width - dx * 2),
        height=max(1, rect.height - dy * 2),
        angle=0.0,
    )


def _edge_support(edges: np.ndarray, box_points: np.ndarray) -> float:
    points = box_points.astype(np.float32)
    hits = 0
    total = 0
    for index in range(4):
        start = points[index]
        end = points[(index + 1) % 4]
        distance = float(np.linalg.norm(end - start))
        steps = max(8, int(distance / 8))
        for t in np.linspace(0.0, 1.0, steps, dtype=np.float32):
            x = int(round(start[0] * (1.0 - t) + end[0] * t))
            y = int(round(start[1] * (1.0 - t) + end[1] * t))
            if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                hits += 1 if edges[y, x] > 0 else 0
                total += 1
    return float(hits / total) if total else 0.0


def _center_prior(center_x: float, center_y: float, size: ImageSize) -> float:
    dx = abs(center_x / max(1.0, size.width) - 0.5) * 2.0
    dy = abs(center_y / max(1.0, size.height) - 0.5) * 2.0
    return float(1.0 - np.clip((dx + dy) / 1.35, 0.0, 1.0))


def _border_clearance(box_points: np.ndarray, size: ImageSize) -> float:
    x_min = float(np.min(box_points[:, 0]))
    x_max = float(np.max(box_points[:, 0]))
    y_min = float(np.min(box_points[:, 1]))
    y_max = float(np.max(box_points[:, 1]))
    margin = min(x_min, y_min, size.width - x_max, size.height - y_max)
    required = max(8.0, min(size.width, size.height) * 0.025)
    return float(np.clip(margin / required, 0.0, 1.0))


def _frame_base_candidates(image: np.ndarray, frame_rect: ImageRect | None) -> list[_BaseCandidate]:
    if frame_rect is None:
        return []
    corners = rotated_rect_corners(frame_rect)
    center = np.array([frame_rect.center_x, frame_rect.center_y], dtype=np.float32)
    min_side = max(12.0, min(frame_rect.width, frame_rect.height))
    offset = max(8.0, min_side * 0.055)
    radius = int(np.clip(min_side * 0.022, 4, 42))
    fractions = (0.18, 0.34, 0.50, 0.66, 0.82)

    candidates: list[_BaseCandidate] = []
    side_names = ("top", "right", "bottom", "left")
    for side_index, side_name in enumerate(side_names):
        start = corners[side_index]
        end = corners[(side_index + 1) % 4]
        edge = end - start
        normal = np.array([edge[1], -edge[0]], dtype=np.float32)
        normal_len = float(np.linalg.norm(normal))
        if normal_len < 1e-5:
            continue
        normal /= normal_len
        midpoint = (start + end) * 0.5
        if float(np.dot(midpoint + normal * offset - center, normal)) < float(np.dot(midpoint - center, normal)):
            normal *= -1.0
        for fraction in fractions:
            point = start * (1.0 - fraction) + end * fraction + normal * offset
            candidate = _sample_base_candidate(image, point, radius, side_name)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _edge_base_candidates(image: np.ndarray) -> list[_BaseCandidate]:
    height, width = image.shape[:2]
    min_side = max(1, min(width, height))
    offset = int(np.clip(min_side * 0.045, 5, min_side // 2))
    radius = int(np.clip(min_side * 0.022, 4, 42))
    fractions = (0.10, 0.22, 0.36, 0.50, 0.64, 0.78, 0.90)
    candidates: list[_BaseCandidate] = []
    for fraction in fractions:
        x = width * fraction
        candidates.append(candidate) if (candidate := _sample_base_candidate(image, np.array([x, offset]), radius, "edge-top")) else None
        candidates.append(candidate) if (candidate := _sample_base_candidate(image, np.array([x, height - offset - 1]), radius, "edge-bottom")) else None
        y = height * fraction
        candidates.append(candidate) if (candidate := _sample_base_candidate(image, np.array([offset, y]), radius, "edge-left")) else None
        candidates.append(candidate) if (candidate := _sample_base_candidate(image, np.array([width - offset - 1, y]), radius, "edge-right")) else None
    return candidates


def _sample_base_candidate(image: np.ndarray, point: np.ndarray, radius: int, source: str) -> _BaseCandidate | None:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
        return None
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(image.shape[1], x + radius + 1)
    y1 = min(image.shape[0], y + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    pixels = np.clip(patch.reshape(-1, 3), 0.0, 1.0)
    rgb = np.median(pixels, axis=0).astype(np.float32)
    luminance = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mean_luma = float(np.mean(luminance))
    spread = float(np.percentile(luminance, 90) - np.percentile(luminance, 10))
    channel_spread = float(np.mean(np.percentile(pixels, 90, axis=0) - np.percentile(pixels, 10, axis=0)))
    clipped = float(np.mean(np.any(pixels >= 0.995, axis=1)))
    too_dark = mean_luma < 0.025
    saturation = float(np.mean(np.max(pixels, axis=1) - np.min(pixels, axis=1)))

    stability = np.clip(1.0 - (spread + channel_spread) / 0.28, 0.0, 1.0)
    brightness = np.clip((mean_luma - 0.02) / 0.55, 0.0, 1.0)
    not_clipped = np.clip(1.0 - clipped * 12.0, 0.0, 1.0)
    saturation_ok = np.clip(1.0 - saturation / 0.55, 0.0, 1.0)
    confidence = float(np.clip(stability * 0.44 + brightness * 0.24 + not_clipped * 0.22 + saturation_ok * 0.10, 0.0, 1.0))
    if too_dark or confidence < 0.22:
        return None

    score = confidence * 100.0 + mean_luma * 12.0 - spread * 35.0 - clipped * 30.0
    return _BaseCandidate(
        point=ImagePoint(x=x, y=y),
        rgb=rgb,
        score=float(score),
        confidence=confidence,
        source=source,
    )


def _confidence_level(confidence: float, high: float, medium: float) -> str:
    if confidence >= high:
        return "high"
    if confidence >= medium:
        return "medium"
    return "low"
