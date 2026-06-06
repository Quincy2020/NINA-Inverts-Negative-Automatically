from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

import math

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency check
    cv2 = None

try:
    import tifffile
except ImportError:  # pragma: no cover - runtime dependency check
    tifffile = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency check
    np = None


SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
DEFAULT_ANALYSIS_MAX_SIZE = (768, 768)
DEFAULT_ANALYSIS_CROP_PERCENT = 4.0
SRGB_LUMA_WEIGHTS_RGB = (0.299, 0.587, 0.114)
IDENTITY_GAINS_RGB = (1.0, 1.0, 1.0)
MIN_OBVIOUS_CAST_MAGNITUDE = 0.005
MIN_ROLL_MEMBERSHIP_FOR_ROLL = 0.30


@dataclass(frozen=True)
class CastMetrics:
    magnitude: float = 0.0
    cast_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    neutral_fraction: float = 0.0
    neutral_pixels: int = 0
    sampled_pixels: int = 0
    median_saturation: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExposureMetrics:
    p05: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    shadow_clip_fraction: float = 0.0
    highlight_clip_fraction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BalanceCandidate:
    algorithm: str
    rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB
    score: float = 0.0
    confidence: float = 0.0
    after_cast: float = 0.0
    regional_sample_count: int = 0
    regional_agreement: float = 0.0
    scene_bias: str = ""
    accepted_region_count: int = 0
    rejected_region_count: int = 0
    region_rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    warning: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rgb_gains"] = [float(value) for value in self.rgb_gains]
        return data


@dataclass(frozen=True)
class FrameAnalysis:
    path: str
    filename: str
    width: int = 0
    height: int = 0
    dtype: str = ""
    bit_depth: int = 0
    algorithm: str = "identity"
    rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB
    safe_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB
    color_action: str = "protected"
    confidence: float = 0.0
    candidate_agreement: float = 0.0
    roll_membership: float = 0.0
    regional_sample_count: int = 0
    regional_agreement: float = 0.0
    scene_bias: str = ""
    accepted_region_count: int = 0
    rejected_region_count: int = 0
    region_rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    tone_shadow_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB
    tone_mid_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB
    tone_highlight_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB
    tone_confidence: float = 0.0
    perceptual_risk: float = 0.0
    perceptual_warning: str = ""
    highlight_luma_delta: float = 0.0
    highlight_green_bias_delta: float = 0.0
    highlight_protected_region: str = ""
    highlight_protection_warning: str = ""
    highlight_protection_strength: float = 0.0
    before_cast: float = 0.0
    after_cast: float = 0.0
    safe_after_cast: float = 0.0
    neutral_fraction: float = 0.0
    neutral_pixels: int = 0
    luma_p50: float = 0.0
    luma_p95: float = 0.0
    exposure_class: str = "normal"
    baseline_exclusion_reason: str = ""
    exposure_delta_stops: float = 0.0
    exposure_action: str = "none"
    exposure_confidence: float = 0.0
    exposure_suggestion_stops: float = 0.0
    exposure_suggestion_action: str = "none"
    used_for_roll: bool = False
    warning: str = ""
    candidates: tuple[BalanceCandidate, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rgb_gains"] = [float(value) for value in self.rgb_gains]
        data["safe_rgb_gains"] = [float(value) for value in self.safe_rgb_gains]
        data["tone_shadow_rgb_gains"] = [float(value) for value in self.tone_shadow_rgb_gains]
        data["tone_mid_rgb_gains"] = [float(value) for value in self.tone_mid_rgb_gains]
        data["tone_highlight_rgb_gains"] = [float(value) for value in self.tone_highlight_rgb_gains]
        data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return data


@dataclass(frozen=True)
class RollAnalysisResult:
    roll_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB
    exposure_target_p50: float = 0.0
    exposure_target_p95: float = 0.0
    roll_cast_strength: float = 1.0
    baseline_strength_hint: str = "normal"
    confidence: float = 0.0
    analyzed_count: int = 0
    used_count: int = 0
    warning: str = ""
    frames: tuple[FrameAnalysis, ...] = field(default_factory=tuple)

    def frame_for_path(self, path: str | Path) -> FrameAnalysis | None:
        target = str(Path(path))
        for frame in self.frames:
            if frame.path == target:
                return frame
        return None

    def to_dict(self) -> dict:
        return {
            "roll_rgb_gains": [float(value) for value in self.roll_rgb_gains],
            "exposure_target_p50": float(self.exposure_target_p50),
            "exposure_target_p95": float(self.exposure_target_p95),
            "roll_cast_strength": float(self.roll_cast_strength),
            "baseline_strength_hint": self.baseline_strength_hint,
            "confidence": float(self.confidence),
            "analyzed_count": int(self.analyzed_count),
            "used_count": int(self.used_count),
            "warning": self.warning,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True)
class _FrameWork:
    frame: FrameAnalysis
    proxy: object
    cast_metrics: CastMetrics
    exposure_metrics: ExposureMetrics


@dataclass(frozen=True)
class _RegionalTileResult:
    gains: tuple[float, float, float] = IDENTITY_GAINS_RGB
    weight: float = 0.0
    scene_bias: str = "mixed-neutral"
    reject_reason: str = ""


def collect_roll_image_paths(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    paths = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda item: item.name.lower())


def load_bgr_image(path: str | Path):
    _require_cv2_np()
    source_path = Path(path)
    if source_path.suffix.lower() in {".tif", ".tiff"}:
        return _load_tiff_bgr_image(source_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not open image: {path}")
    return _ensure_bgr(image)


def _load_tiff_bgr_image(path: Path):
    _require_tifffile_np()
    try:
        image = tifffile.imread(str(path))
    except Exception as exc:
        raise ValueError(f"Could not open TIFF image: {path}") from exc
    return _tiff_array_to_bgr(image, path)


def save_bgr_image(path: str | Path, image) -> None:
    _require_cv2_np()
    output_path = Path(path)
    extension = output_path.suffix or ".tif"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Could not encode image as {extension}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(output_path))


def analyze_roll(
    paths: Iterable[str | Path],
    *,
    analysis_max_size: tuple[int, int] = DEFAULT_ANALYSIS_MAX_SIZE,
    crop_percent: float = DEFAULT_ANALYSIS_CROP_PERCENT,
) -> RollAnalysisResult:
    work_items = tuple(
        _analyze_frame_work(path, analysis_max_size=analysis_max_size, crop_percent=crop_percent)
        for path in paths
    )
    frames = tuple(item.frame for item in work_items)
    usable = [frame for frame in frames if frame.used_for_roll]
    if not frames:
        return RollAnalysisResult(warning="no images found")

    if len(usable) < 2:
        exposure_target_p50, exposure_target_p95 = _roll_exposure_targets(
            frames,
            set(frame.path for frame in usable),
        )
        warning = "not enough reliable frames for roll baseline"
        if usable:
            gains = usable[0].rgb_gains
            confidence = usable[0].confidence * 0.45
        else:
            gains = IDENTITY_GAINS_RGB
            confidence = 0.0
        roll_cast_strength, baseline_strength_hint = _roll_cast_strength_hint(
            usable,
            gains,
            set(frame.path for frame in usable),
        )
        return RollAnalysisResult(
            roll_rgb_gains=gains,
            exposure_target_p50=exposure_target_p50,
            exposure_target_p95=exposure_target_p95,
            roll_cast_strength=roll_cast_strength,
            baseline_strength_hint=baseline_strength_hint,
            confidence=round(float(confidence), 3),
            analyzed_count=len(frames),
            used_count=len(usable),
            warning=warning,
            frames=_finalize_frame_plans(
                work_items,
                gains,
                set(frame.path for frame in usable),
                exposure_target_p50,
                exposure_target_p95,
                roll_cast_strength,
            ),
        )

    log_gains = np.asarray([_safe_log_gain(frame.rgb_gains) for frame in usable], dtype=np.float64)
    weights = np.asarray([max(0.01, frame.confidence) for frame in usable], dtype=np.float64)
    cluster_keep = _main_gain_cluster_mask(log_gains, weights)
    cluster_logs = log_gains[cluster_keep]
    cluster_weights = weights[cluster_keep]
    preliminary = _weighted_median_vector(cluster_logs, cluster_weights)
    distances = np.linalg.norm(log_gains - preliminary[None, :], axis=1)
    median_distance = float(np.median(distances[cluster_keep])) if int(np.count_nonzero(cluster_keep)) else 0.0
    mad = float(np.median(np.abs(distances[cluster_keep] - median_distance))) if int(np.count_nonzero(cluster_keep)) else 0.0
    keep_limit = max(0.16, median_distance + 2.5 * max(mad, 0.018))
    keep = cluster_keep & (distances <= keep_limit)
    if int(np.count_nonzero(keep)) >= 2:
        kept_logs = log_gains[keep]
        kept_weights = weights[keep]
        kept_paths = {usable[index].path for index, is_kept in enumerate(keep.tolist()) if is_kept}
    else:
        kept_logs = log_gains
        kept_weights = weights
        kept_paths = {frame.path for frame in usable}

    exposure_target_p50, exposure_target_p95 = _roll_exposure_targets(frames, kept_paths)
    final_log = _weighted_median_vector(kept_logs, kept_weights)
    gains = normalize_rgb_gains(tuple(float(math.exp(value)) for value in final_log))
    roll_cast_strength, baseline_strength_hint = _roll_cast_strength_hint(frames, gains, kept_paths)
    dispersion = float(np.average(np.linalg.norm(kept_logs - final_log[None, :], axis=1), weights=kept_weights))
    count_score = min(1.0, len(kept_logs) / 8.0)
    used_fraction_score = min(1.0, len(kept_logs) / max(1, len(frames)) / 0.55)
    dispersion_score = max(0.0, min(1.0, 1.0 - dispersion / 0.18))
    confidence = round(0.35 * count_score + 0.30 * used_fraction_score + 0.35 * dispersion_score, 3)
    warning = "" if confidence >= 0.45 else "low roll confidence"
    final_frames = _finalize_frame_plans(
        work_items,
        gains,
        kept_paths,
        exposure_target_p50,
        exposure_target_p95,
        roll_cast_strength,
    )
    return RollAnalysisResult(
        roll_rgb_gains=gains,
        exposure_target_p50=exposure_target_p50,
        exposure_target_p95=exposure_target_p95,
        roll_cast_strength=roll_cast_strength,
        baseline_strength_hint=baseline_strength_hint,
        confidence=confidence,
        analyzed_count=len(frames),
        used_count=int(len(kept_logs)),
        warning=warning,
        frames=final_frames,
    )


def analyze_frame(
    path: str | Path,
    *,
    analysis_max_size: tuple[int, int] = DEFAULT_ANALYSIS_MAX_SIZE,
    crop_percent: float = DEFAULT_ANALYSIS_CROP_PERCENT,
) -> FrameAnalysis:
    return _analyze_frame_work(path, analysis_max_size=analysis_max_size, crop_percent=crop_percent).frame


def summarize_roll_result(result: RollAnalysisResult, *, worsened_threshold: float = 0.003) -> dict:
    frames = tuple(result.frames)
    before_values = [float(frame.before_cast) for frame in frames]
    after_values = [float(frame.safe_after_cast) for frame in frames]
    return {
        "roll_rgb_gains": [round(float(value), 6) for value in result.roll_rgb_gains],
        "roll_cast_strength": float(result.roll_cast_strength),
        "baseline_strength_hint": result.baseline_strength_hint,
        "confidence": float(result.confidence),
        "analyzed_count": int(result.analyzed_count),
        "used_count": int(result.used_count),
        "warning": result.warning,
        "median_before_cast": _median_or_zero(before_values),
        "median_after_cast": _median_or_zero(after_values),
        "max_before_cast": max(before_values, default=0.0),
        "max_after_cast": max(after_values, default=0.0),
        "worsened_count": int(
            sum(1 for frame in frames if frame.safe_after_cast > frame.before_cast + float(worsened_threshold))
        ),
        "action_distribution": _count_values(frame.color_action for frame in frames),
        "exposure_distribution": _count_values(frame.exposure_action for frame in frames),
        "exposure_class_distribution": _count_values(
            getattr(frame, "exposure_class", "normal") or "normal" for frame in frames
        ),
        "baseline_exclusion_distribution": _count_values(
            getattr(frame, "baseline_exclusion_reason", "") or "none" for frame in frames
        ),
        "scene_bias_distribution": _count_values(getattr(frame, "scene_bias", "") or "unknown" for frame in frames),
        "perceptual_warning_distribution": _count_values(
            getattr(frame, "perceptual_warning", "") or "none" for frame in frames
        ),
        "highlight_protection_warning_distribution": _count_values(
            getattr(frame, "highlight_protection_warning", "") or "none" for frame in frames
        ),
        "highlight_risk_count": int(sum(1 for frame in frames if getattr(frame, "highlight_protection_warning", ""))),
        "regional_frames": int(sum(1 for frame in frames if frame.regional_sample_count > 0)),
        "tone_frames": int(sum(1 for frame in frames if frame.tone_confidence > 0.0)),
    }


def _analyze_frame_work(
    path: str | Path,
    *,
    analysis_max_size: tuple[int, int] = DEFAULT_ANALYSIS_MAX_SIZE,
    crop_percent: float = DEFAULT_ANALYSIS_CROP_PERCENT,
) -> _FrameWork:
    _require_cv2_np()
    image = load_bgr_image(path)
    analysis_image = crop_bgr_border(image, crop_percent)
    proxy = resize_bgr_to_fit(analysis_image, analysis_max_size)
    metrics = estimate_cast_metrics(proxy)
    exposure = estimate_exposure_metrics(proxy)
    candidates = tuple(_candidate_corrections(proxy, metrics))
    regional_candidates = [candidate for candidate in candidates if candidate.regional_sample_count > 0]
    if regional_candidates:
        regional_candidate = max(regional_candidates, key=lambda candidate: candidate.regional_sample_count)
        regional_sample_count = regional_candidate.regional_sample_count
        regional_agreement = regional_candidate.regional_agreement
        scene_bias = regional_candidate.scene_bias
        accepted_region_count = regional_candidate.accepted_region_count
        rejected_region_count = regional_candidate.rejected_region_count
        region_rejection_reasons = regional_candidate.region_rejection_reasons
    else:
        regional_sample_count = 0
        regional_agreement = 0.0
        scene_bias = ""
        accepted_region_count = 0
        rejected_region_count = 0
        region_rejection_reasons = ()
    ensemble = _ensemble_candidate(candidates, proxy, metrics)
    accepted = _candidate_accepted_for_frame(ensemble, metrics)
    gain_log_norm = float(np.linalg.norm(_safe_log_gain(ensemble.rgb_gains)))
    exposure_class = _exposure_class_for_roll(exposure)
    baseline_exclusion_reason = _baseline_exclusion_reason(
        accepted,
        metrics,
        gain_log_norm,
        exposure,
        exposure_class,
    )
    used_for_roll = accepted and not baseline_exclusion_reason
    if not accepted:
        algorithm = "identity"
        gains = IDENTITY_GAINS_RGB
        confidence = 0.0
        candidate_agreement = ensemble.confidence
        after_cast = metrics.magnitude
        warning = ensemble.warning or "no reliable correction"
    else:
        algorithm = ensemble.algorithm
        gains = ensemble.rgb_gains
        confidence = ensemble.confidence
        candidate_agreement = ensemble.score
        after_cast = ensemble.after_cast
        warning = "" if used_for_roll else baseline_exclusion_reason or "frame correction only"
        if exposure_class == "extreme protected":
            warning = "exposure outlier"

    height, width = image.shape[:2]
    frame = FrameAnalysis(
        path=str(Path(path)),
        filename=Path(path).name,
        width=int(width),
        height=int(height),
        dtype=str(image.dtype),
        bit_depth=_bit_depth_for_image(image),
        algorithm=algorithm,
        rgb_gains=gains,
        safe_rgb_gains=IDENTITY_GAINS_RGB,
        color_action="pending",
        confidence=round(float(confidence), 3),
        candidate_agreement=round(float(candidate_agreement), 3),
        regional_sample_count=int(regional_sample_count),
        regional_agreement=round(float(regional_agreement), 3),
        scene_bias=scene_bias,
        accepted_region_count=int(accepted_region_count),
        rejected_region_count=int(rejected_region_count),
        region_rejection_reasons=tuple(region_rejection_reasons),
        before_cast=round(float(metrics.magnitude), 5),
        after_cast=round(float(after_cast), 5),
        safe_after_cast=round(float(metrics.magnitude), 5),
        neutral_fraction=round(float(metrics.neutral_fraction), 5),
        neutral_pixels=int(metrics.neutral_pixels),
        luma_p50=round(float(exposure.p50), 5),
        luma_p95=round(float(exposure.p95), 5),
        exposure_class=exposure_class,
        baseline_exclusion_reason=baseline_exclusion_reason,
        used_for_roll=bool(used_for_roll),
        warning=warning,
        candidates=candidates,
    )
    return _FrameWork(frame=frame, proxy=proxy, cast_metrics=metrics, exposure_metrics=exposure)


def _finalize_frame_plans(
    work_items: tuple[_FrameWork, ...],
    roll_gains: tuple[float, float, float],
    kept_paths: set[str],
    exposure_target_p50: float,
    exposure_target_p95: float,
    roll_cast_strength: float = 1.0,
) -> tuple[FrameAnalysis, ...]:
    frames: list[FrameAnalysis] = []
    roll_available = _roll_baseline_available(roll_gains, kept_paths)
    for item in work_items:
        frame = item.frame
        if roll_available:
            roll_after_for_membership = float(
                estimate_cast_metrics(apply_rgb_gains_to_bgr(item.proxy, roll_gains)).magnitude
            )
        else:
            roll_after_for_membership = float(item.cast_metrics.magnitude)
        roll_membership = _roll_membership_score(
            frame,
            roll_gains,
            item.exposure_metrics,
            roll_after_for_membership,
        )
        frame_roll_gains = _strengthened_roll_gains(roll_gains, roll_cast_strength, roll_membership, frame)
        color_action, safe_gains, safe_after, color_warning = _select_v3_color_plan(
            item.proxy,
            item.cast_metrics,
            item.exposure_metrics,
            frame_roll_gains,
            frame,
            roll_available,
            roll_membership,
        )
        tone_shadow, tone_mid, tone_highlight, tone_confidence = _tone_residual_gains(
            item.proxy,
            safe_gains,
            color_action,
        )
        if getattr(frame, "exposure_class", "normal") == "low-light salvageable":
            tone_shadow = _scale_rgb_gains_to_identity(tone_shadow, 0.30)
            tone_mid = _scale_rgb_gains_to_identity(tone_mid, 0.50)
            tone_highlight = _scale_rgb_gains_to_identity(tone_highlight, 0.25)
            tone_confidence *= 0.50
        if _should_run_perceptual_check(frame, color_action):
            perceptual_risk, perceptual_warning = _perceptual_side_effect_check(
                item.proxy,
                safe_gains,
                color_action,
                tone_shadow,
                tone_mid,
                tone_highlight,
            )
        else:
            perceptual_risk, perceptual_warning = 0.0, ""
        if perceptual_warning and color_action == "roll+frame" and roll_available:
            roll_only_gains = normalize_rgb_gains(frame_roll_gains)
            roll_tone_shadow, roll_tone_mid, roll_tone_highlight, roll_tone_confidence = _tone_residual_gains(
                item.proxy,
                roll_only_gains,
                "roll",
            )
            roll_only_risk, roll_only_warning = _perceptual_side_effect_check(
                item.proxy,
                roll_only_gains,
                "roll",
                roll_tone_shadow,
                roll_tone_mid,
                roll_tone_highlight,
            )
            roll_only_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(item.proxy, roll_only_gains)).magnitude)
            if roll_only_risk <= perceptual_risk and roll_only_after <= item.cast_metrics.magnitude + 0.003:
                color_action = "roll"
                safe_gains = roll_only_gains
                safe_after = roll_only_after
                tone_shadow = roll_tone_shadow
                tone_mid = roll_tone_mid
                tone_highlight = roll_tone_highlight
                tone_confidence = roll_tone_confidence
                color_warning = "; ".join(part for part in (color_warning, "frame residual skipped") if part)
                perceptual_risk = roll_only_risk
                perceptual_warning = roll_only_warning
        highlight_region, highlight_luma_delta, highlight_green_delta, highlight_warning = _highlight_rendering_diagnostics(
            item.proxy,
            safe_gains,
            color_action,
            tone_shadow,
            tone_mid,
            tone_highlight,
        )
        highlight_protection_strength = 0.0
        if (
            (perceptual_warning == "highlight sky tint drift" or highlight_warning)
            and color_action in {"roll", "roll+frame", "frame-only"}
        ):
            corrected_proxy = apply_tone_aware_rgb_gains_to_bgr(
                item.proxy,
                safe_gains,
                tone_shadow,
                tone_mid,
                tone_highlight,
            )
            chosen_highlight_protection: tuple[float, float, str, float, float, str] | None = None
            before_cast = float(item.cast_metrics.magnitude)
            for candidate_strength in (0.35, 0.50, 0.65, 0.80, 0.92, 1.0):
                protected_proxy = _apply_highlight_protection_to_bgr(
                    item.proxy,
                    corrected_proxy,
                    candidate_strength,
                )
                candidate_safe_after = float(estimate_cast_metrics(protected_proxy).magnitude)
                candidate_region, candidate_luma_delta, candidate_green_delta, candidate_warning = (
                    _highlight_rendering_diagnostics_for_images(item.proxy, protected_proxy)
                )
                candidate = (
                    candidate_strength,
                    candidate_safe_after,
                    candidate_region,
                    candidate_luma_delta,
                    candidate_green_delta,
                    candidate_warning,
                )
                if chosen_highlight_protection is None:
                    chosen_highlight_protection = candidate
                elif not candidate_warning and chosen_highlight_protection[5]:
                    chosen_highlight_protection = candidate
                elif candidate_warning == chosen_highlight_protection[5] and candidate_safe_after < chosen_highlight_protection[1]:
                    chosen_highlight_protection = candidate
                if not candidate_warning and candidate_safe_after <= before_cast + max(0.003, before_cast * 0.08):
                    chosen_highlight_protection = candidate
                    break
            if chosen_highlight_protection is not None:
                (
                    highlight_protection_strength,
                    safe_after,
                    highlight_region,
                    highlight_luma_delta,
                    highlight_green_delta,
                    highlight_warning,
                ) = chosen_highlight_protection
            if perceptual_warning == "highlight sky tint drift" and not highlight_warning:
                perceptual_risk = 0.0
                perceptual_warning = ""
            highlight_action_warning = "highlight rendering protected"
            if highlight_action_warning:
                color_warning = "; ".join(part for part in (color_warning, highlight_action_warning) if part)
        if (
            perceptual_warning
            and getattr(frame, "exposure_class", "normal") == "low-light salvageable"
            and color_action in {"roll", "frame-only", "roll+frame"}
        ):
            improvement = float(item.cast_metrics.magnitude) - float(safe_after)
            if perceptual_risk >= 0.35 or improvement < max(0.003, float(item.cast_metrics.magnitude) * 0.18):
                color_action = "review"
                safe_gains = IDENTITY_GAINS_RGB
                safe_after = float(item.cast_metrics.magnitude)
                tone_shadow = IDENTITY_GAINS_RGB
                tone_mid = IDENTITY_GAINS_RGB
                tone_highlight = IDENTITY_GAINS_RGB
                tone_confidence = 0.0
                highlight_region = ""
                highlight_luma_delta = 0.0
                highlight_green_delta = 0.0
                highlight_warning = ""
                highlight_protection_strength = 0.0
                color_warning = "; ".join(part for part in (color_warning, "low-light perceptual risk") if part)
        exposure_suggestion, exposure_suggestion_action, exposure_confidence = _frame_exposure_delta(
            item.exposure_metrics,
            exposure_target_p50,
            exposure_target_p95,
        )
        exposure_delta = exposure_suggestion
        exposure_action = exposure_suggestion_action
        if color_action in {"none", "protected", "review", "skip-extreme"} and abs(exposure_suggestion) >= 0.001:
            exposure_delta = 0.0
            exposure_action = "exposure suggestion only"
        if abs(exposure_delta) >= 0.001:
            color_test = apply_rgb_gains_to_bgr(item.proxy, safe_gains)
            exposure_test = apply_exposure_to_bgr(color_test, exposure_delta, strength=0.65)
            exposure_cast = float(estimate_cast_metrics(exposure_test).magnitude)
            reference_cast = min(float(item.cast_metrics.magnitude), float(safe_after))
            if exposure_cast > reference_cast + max(0.004, reference_cast * 0.40):
                exposure_delta = 0.0
                exposure_action = "exposure color protected"
        warning_parts = []
        if frame.warning:
            warning_parts.append(frame.warning)
        if frame.used_for_roll and frame.path not in kept_paths:
            warning_parts.append("roll outlier")
        if color_warning:
            warning_parts.append(color_warning)
        if perceptual_warning:
            warning_parts.append(perceptual_warning)
        if highlight_warning:
            warning_parts.append(highlight_warning)
        if roll_membership < 0.20 and color_action in {"roll", "roll+frame"}:
            warning_parts.append("low roll membership")
        if exposure_action in {"exposure protected", "exposure review"}:
            warning_parts.append(exposure_action)
        unique_warning = "; ".join(dict.fromkeys(part for part in warning_parts if part))
        frames.append(
            replace(
                frame,
                used_for_roll=frame.path in kept_paths,
                safe_rgb_gains=safe_gains,
                color_action=color_action,
                roll_membership=round(float(roll_membership), 3),
                tone_shadow_rgb_gains=tone_shadow,
                tone_mid_rgb_gains=tone_mid,
                tone_highlight_rgb_gains=tone_highlight,
                tone_confidence=round(float(tone_confidence), 3),
                perceptual_risk=round(float(perceptual_risk), 3),
                perceptual_warning=perceptual_warning,
                highlight_luma_delta=round(float(highlight_luma_delta), 5),
                highlight_green_bias_delta=round(float(highlight_green_delta), 5),
                highlight_protected_region=highlight_region,
                highlight_protection_warning=highlight_warning,
                highlight_protection_strength=round(float(highlight_protection_strength), 3),
                safe_after_cast=round(float(safe_after), 5),
                exposure_delta_stops=round(float(exposure_delta), 4),
                exposure_action=exposure_action,
                exposure_confidence=round(float(exposure_confidence), 3),
                exposure_suggestion_stops=round(float(exposure_suggestion), 4),
                exposure_suggestion_action=exposure_suggestion_action,
                warning=unique_warning,
            )
        )
    return tuple(frames)


def _exposure_class_for_roll(exposure: ExposureMetrics) -> str:
    tonal_span = float(exposure.p75 - exposure.p25)
    if (
        exposure.shadow_clip_fraction >= 0.12
        or exposure.highlight_clip_fraction >= 0.07
        or exposure.p50 <= 0.10
        or exposure.p95 <= 0.35
        or exposure.p50 >= 0.90
        or exposure.p95 >= 0.992
        or tonal_span <= 0.045
    ):
        return "extreme protected"
    if exposure.p50 <= 0.18 or (exposure.p50 <= 0.22 and exposure.p95 <= 0.68):
        return "low-light salvageable"
    return "normal"


def _protect_highlight_rendering(
    proxy,
    before_cast: float,
    color_action: str,
    safe_gains: tuple[float, float, float],
    tone_shadow: tuple[float, float, float],
    tone_mid: tuple[float, float, float],
    tone_highlight: tuple[float, float, float],
    tone_confidence: float,
) -> tuple[
    str,
    tuple[float, float, float],
    float,
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
    str,
    str,
    float,
    float,
    str,
    str,
]:
    candidates: list[
        tuple[
            str,
            tuple[float, float, float],
            float,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
            str,
            str,
            float,
            float,
            str,
            str,
        ]
    ] = []

    def add_candidate(
        action: str,
        gains: tuple[float, float, float],
        shadow: tuple[float, float, float],
        mid: tuple[float, float, float],
        highlight: tuple[float, float, float],
        confidence: float,
        label: str,
    ) -> None:
        after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, gains)).magnitude)
        risk, perceptual_warning = _perceptual_side_effect_check(proxy, gains, action, shadow, mid, highlight)
        region, luma_delta, green_delta, highlight_warning = _highlight_rendering_diagnostics(
            proxy,
            gains,
            action,
            shadow,
            mid,
            highlight,
        )
        if perceptual_warning == "highlight sky tint drift" or highlight_warning:
            return
        if after > before_cast + max(0.004, before_cast * 0.25):
            return
        candidates.append(
            (
                action,
                gains,
                after,
                shadow,
                mid,
                highlight,
                confidence,
                risk,
                perceptual_warning,
                region,
                luma_delta,
                green_delta,
                highlight_warning,
                label,
            )
        )

    add_candidate(
        color_action,
        safe_gains,
        tone_shadow,
        tone_mid,
        IDENTITY_GAINS_RGB,
        tone_confidence * 0.75,
        "highlight residual skipped",
    )
    add_candidate(
        color_action,
        safe_gains,
        IDENTITY_GAINS_RGB,
        IDENTITY_GAINS_RGB,
        IDENTITY_GAINS_RGB,
        0.0,
        "tone residual skipped",
    )
    for strength in (0.90, 0.75, 0.60, 0.45, 0.30):
        add_candidate(
            color_action,
            _scale_rgb_gains_to_identity(safe_gains, strength),
            IDENTITY_GAINS_RGB,
            IDENTITY_GAINS_RGB,
            IDENTITY_GAINS_RGB,
            0.0,
            "highlight correction weakened",
        )

    if candidates:
        return candidates[0]
    return (
        "review",
        IDENTITY_GAINS_RGB,
        before_cast,
        IDENTITY_GAINS_RGB,
        IDENTITY_GAINS_RGB,
        IDENTITY_GAINS_RGB,
        0.0,
        0.0,
        "",
        "",
        0.0,
        0.0,
        "",
        "highlight rendering protected",
    )


def _protect_highlight_tint_drift(
    proxy,
    before_cast: float,
    color_action: str,
    safe_gains: tuple[float, float, float],
    tone_shadow: tuple[float, float, float],
    tone_mid: tuple[float, float, float],
    tone_highlight: tuple[float, float, float],
    tone_confidence: float,
) -> tuple[
    str,
    tuple[float, float, float],
    float,
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
    str,
    str,
]:
    protected = _protect_highlight_rendering(
        proxy,
        before_cast,
        color_action,
        safe_gains,
        tone_shadow,
        tone_mid,
        tone_highlight,
        tone_confidence,
    )
    return (
        protected[0],
        protected[1],
        protected[2],
        protected[3],
        protected[4],
        protected[5],
        protected[6],
        protected[7],
        protected[8],
        "highlight tint protected" if protected[13] == "highlight rendering protected" else protected[13],
    )


def _highlight_rendering_diagnostics(
    proxy,
    rgb_gains: tuple[float, float, float],
    color_action: str,
    shadow_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
    mid_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
    highlight_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
) -> tuple[str, float, float, str]:
    if color_action in {"none", "protected", "review", "skip-extreme"}:
        return "", 0.0, 0.0, ""
    corrected = apply_tone_aware_rgb_gains_to_bgr(
        proxy,
        rgb_gains,
        shadow_rgb_gains,
        mid_rgb_gains,
        highlight_rgb_gains,
        tone_strength=1.0,
    )
    return _highlight_rendering_diagnostics_for_images(proxy, corrected)


def _highlight_rendering_diagnostics_for_images(before_bgr, after_bgr) -> tuple[str, float, float, str]:
    before_rgb = _rgb_float01(before_bgr)
    after_rgb = _rgb_float01(after_bgr)
    before_flat = before_rgb.reshape((-1, 3))
    after_flat = after_rgb.reshape((-1, 3))
    if before_flat.shape[0] < 128:
        return "", 0.0, 0.0, ""
    before_luma = _rgb_luma(before_flat)
    after_luma = _rgb_luma(after_flat)
    region_masks = _sky_cloud_region_masks(before_rgb, before_flat, before_luma)
    best: tuple[float, str, float, float, str] | None = None
    for region, mask in region_masks:
        if int(np.count_nonzero(mask)) < 96:
            continue
        if float(np.mean(mask)) < 0.008:
            continue
        luma_delta = float(np.median(after_luma[mask]) - np.median(before_luma[mask]))
        green_delta = _green_bias_delta(before_flat, after_flat, mask)
        green_limit = 0.010 if region != "blue_sky" else 0.013
        dark_limit = -0.030 if region != "blue_sky" else -0.026
        green_score = max(0.0, green_delta / green_limit)
        dark_score = max(0.0, luma_delta / min(dark_limit, -1e-6))
        score = max(green_score, dark_score)
        warning = ""
        if green_score >= 1.0 and dark_score >= 1.0:
            warning = f"{region.replace('_', ' ')} green/dark highlight risk"
        elif green_score >= 1.0:
            warning = f"{region.replace('_', ' ')} green highlight risk"
        elif dark_score >= 1.0:
            warning = f"{region.replace('_', ' ')} dark highlight risk"
        if best is None or score > best[0]:
            best = (score, region, luma_delta, green_delta, warning)
    if best is None:
        return "", 0.0, 0.0, ""
    _score, region, luma_delta, green_delta, warning = best
    return region, luma_delta, green_delta, warning


def _apply_highlight_protection_to_bgr(original_bgr, corrected_bgr, strength: float = 1.0):
    _require_cv2_np()
    strength = _clamp_float(float(strength), 0.0, 1.0)
    if strength <= 0.001:
        return corrected_bgr.copy()
    if original_bgr.shape[:2] != corrected_bgr.shape[:2]:
        return corrected_bgr.copy()
    if original_bgr.ndim != 3 or corrected_bgr.ndim != 3 or original_bgr.shape[2] < 3 or corrected_bgr.shape[2] < 3:
        return corrected_bgr.copy()

    before_rgb = _rgb_float01(original_bgr)
    before_flat = before_rgb.reshape((-1, 3))
    if before_flat.shape[0] < 128:
        return corrected_bgr.copy()
    before_luma = _rgb_luma(before_flat)
    combined_mask = np.zeros(before_flat.shape[0], dtype=bool)
    for _region, mask in _sky_cloud_region_masks(before_rgb, before_flat, before_luma):
        if int(np.count_nonzero(mask)) >= 96:
            combined_mask |= mask
    if not bool(np.any(combined_mask)):
        return corrected_bgr.copy()

    height, width = original_bgr.shape[:2]
    mask_image = combined_mask.reshape((height, width)).astype(np.float32)
    sigma = max(1.2, min(6.0, min(height, width) / 180.0))
    softened_mask = cv2.GaussianBlur(mask_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask_image = np.maximum(mask_image, softened_mask)
    mask_image = np.clip(mask_image * np.float32(strength), 0.0, 1.0)[:, :, None]

    max_value = float(_max_value_for_dtype(corrected_bgr.dtype))
    original_rgb = original_bgr[:, :, :3].astype(np.float32, copy=False)
    corrected_rgb = corrected_bgr[:, :, :3].astype(np.float32, copy=False)
    output_rgb = corrected_rgb * (1.0 - mask_image) + original_rgb * mask_image
    output_rgb = np.clip(np.rint(output_rgb), 0, max_value).astype(corrected_bgr.dtype)
    if corrected_bgr.shape[2] > 3:
        return np.dstack((output_rgb, corrected_bgr[:, :, 3:]))
    return output_rgb


def _sky_cloud_region_masks(before_rgb, before_flat, before_luma) -> tuple[tuple[str, object], ...]:
    max_channel = np.max(before_flat, axis=1)
    min_channel = np.min(before_flat, axis=1)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    red = before_flat[:, 0]
    green = before_flat[:, 1]
    blue = before_flat[:, 2]
    height, width = before_rgb.shape[:2]
    y_position = np.repeat(np.arange(height, dtype=np.float32)[:, None], width, axis=1).reshape(-1) / max(1, height - 1)
    blue_sky = (
        (before_luma >= 0.50)
        & (before_luma <= 0.985)
        & (y_position <= 0.82)
        & (blue >= red * 1.01)
        & (blue >= green * 0.95)
        & (saturation >= 0.035)
        & (saturation <= 0.42)
    )
    white_cloud = (
        (before_luma >= 0.62)
        & (before_luma <= 0.988)
        & (y_position <= 0.86)
        & (saturation <= 0.14)
        & (max_channel <= 0.992)
    )
    overcast_cloud = (
        (before_luma >= 0.44)
        & (before_luma <= 0.92)
        & (y_position <= 0.88)
        & (saturation <= 0.11)
        & (max_channel <= 0.985)
    )
    return (("blue_sky", blue_sky), ("white_cloud", white_cloud), ("overcast_cloud", overcast_cloud))


def _green_bias_delta(before_flat, after_flat, mask) -> float:
    before_green = before_flat[:, 1] - (before_flat[:, 0] + before_flat[:, 2]) * 0.5
    after_green = after_flat[:, 1] - (after_flat[:, 0] + after_flat[:, 2]) * 0.5
    return float(np.median(after_green[mask]) - np.median(before_green[mask]))


def _baseline_exclusion_reason(
    accepted: bool,
    metrics: CastMetrics,
    gain_log_norm: float,
    exposure: ExposureMetrics,
    exposure_class: str,
) -> str:
    if not accepted:
        return "unreliable correction"
    if metrics.neutral_fraction < 0.015:
        return "low neutral evidence"
    if gain_log_norm > 0.78:
        return "large correction"
    if exposure_class == "low-light salvageable":
        return "low-light excluded from baseline"
    if exposure_class == "extreme protected":
        return "extreme exposure"
    if not _exposure_usable_for_roll_baseline(exposure):
        return "exposure outlier"
    return ""


def _select_v3_color_plan(
    proxy,
    before_metrics: CastMetrics,
    exposure: ExposureMetrics,
    roll_gains: tuple[float, float, float],
    frame: FrameAnalysis,
    roll_available: bool,
    roll_membership: float = 1.0,
) -> tuple[str, tuple[float, float, float], float, str]:
    before = float(before_metrics.magnitude)
    frame_reliable = _frame_candidate_reliable(frame)
    exposure_class = getattr(frame, "exposure_class", "normal")
    exposure_extreme = exposure_class == "extreme protected" or not _exposure_usable_for_roll_baseline(exposure)

    if exposure_extreme:
        if frame_reliable:
            after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, frame.rgb_gains)).magnitude)
            return "frame-only", normalize_rgb_gains(frame.rgb_gains), after, "excluded from roll baseline"
        return "review", IDENTITY_GAINS_RGB, before, "extreme exposure"

    if exposure_class == "low-light salvageable":
        return _select_low_light_color_plan(
            proxy,
            before_metrics,
            roll_gains,
            frame,
            roll_available,
            roll_membership,
            frame_reliable,
        )

    if not roll_available:
        if frame_reliable:
            after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, frame.rgb_gains)).magnitude)
            return "frame-only", normalize_rgb_gains(frame.rgb_gains), after, "no roll baseline"
        return "none", IDENTITY_GAINS_RGB, before, "no roll baseline"

    roll_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, roll_gains)).magnitude)
    if _roll_membership_blocks_roll(frame, before, roll_after, roll_membership):
        if frame_reliable:
            frame_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, frame.rgb_gains)).magnitude)
            if frame_after < before - max(0.0025, before * 0.12):
                return "frame-only", normalize_rgb_gains(frame.rgb_gains), frame_after, "low roll membership"
        if before <= 0.014:
            return "none", IDENTITY_GAINS_RGB, before, "low roll membership"
        return "review", IDENTITY_GAINS_RGB, before, "low roll membership"

    if _roll_would_cause_severe_local_harm(before, roll_after):
        if frame_reliable and _candidate_agrees_with_roll(frame.rgb_gains, roll_gains):
            frame_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, frame.rgb_gains)).magnitude)
            return "frame-only", normalize_rgb_gains(frame.rgb_gains), frame_after, "roll conflict"
        return "review", IDENTITY_GAINS_RGB, before, "roll conflict"

    if frame_reliable and roll_after > before + max(0.003, before * 0.12):
        frame_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, frame.rgb_gains)).magnitude)
        if frame_after < before - max(0.0025, before * 0.12):
            return "frame-only", normalize_rgb_gains(frame.rgb_gains), frame_after, "roll local mismatch"

    if not frame_reliable:
        return "roll", normalize_rgb_gains(roll_gains), roll_after, ""

    if _candidate_conflicts_with_roll(frame.rgb_gains, roll_gains):
        return "review", IDENTITY_GAINS_RGB, before, "frame conflicts with roll"

    residual = _safe_log_gain(frame.rgb_gains) - _safe_log_gain(roll_gains)
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 0.09:
        return "roll", normalize_rgb_gains(roll_gains), roll_after, ""

    combined_gains = _default_roll_frame_gains(roll_gains, frame.rgb_gains)
    combined_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, combined_gains)).magnitude)
    if _roll_would_cause_severe_local_harm(before, combined_after):
        return "roll", normalize_rgb_gains(roll_gains), roll_after, "frame residual skipped"
    if combined_after > min(before, roll_after) + 0.003:
        return "roll", normalize_rgb_gains(roll_gains), roll_after, "frame residual skipped"
    return "roll+frame", normalize_rgb_gains(combined_gains), combined_after, ""


def _select_low_light_color_plan(
    proxy,
    before_metrics: CastMetrics,
    roll_gains: tuple[float, float, float],
    frame: FrameAnalysis,
    roll_available: bool,
    roll_membership: float,
    frame_reliable: bool,
) -> tuple[str, tuple[float, float, float], float, str]:
    before = float(before_metrics.magnitude)
    weak_frame_gains = _scale_rgb_gains_to_identity(frame.rgb_gains, 0.45)
    if not roll_available:
        if frame_reliable:
            frame_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, weak_frame_gains)).magnitude)
            if frame_after < before - max(0.002, before * 0.10):
                return "frame-only", weak_frame_gains, frame_after, "low-light weak correction"
        if before <= 0.014:
            return "none", IDENTITY_GAINS_RGB, before, "low-light no safe correction"
        return "review", IDENTITY_GAINS_RGB, before, "low-light no safe correction"

    roll_strength = 0.55 + 0.18 * max(0.0, min(1.0, (roll_membership - 0.55) / 0.35))
    weak_roll_gains = _scale_rgb_gains_to_identity(roll_gains, roll_strength)
    roll_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, weak_roll_gains)).magnitude)
    if roll_after <= before - max(0.0015, before * 0.08) and not _roll_would_cause_severe_local_harm(before, roll_after):
        return "roll", weak_roll_gains, roll_after, "low-light weak correction"

    if frame_reliable:
        frame_after = float(estimate_cast_metrics(apply_rgb_gains_to_bgr(proxy, weak_frame_gains)).magnitude)
        if frame_after < before - max(0.002, before * 0.10):
            return "frame-only", weak_frame_gains, frame_after, "low-light weak correction"

    if before <= 0.014 and roll_after <= before + max(0.002, before * 0.20):
        return "none", IDENTITY_GAINS_RGB, before, "low-light protected"
    return "review", IDENTITY_GAINS_RGB, before, "low-light protected"


def _roll_baseline_available(roll_gains: tuple[float, float, float], kept_paths: set[str]) -> bool:
    if len(kept_paths) < 2:
        return False
    return float(np.linalg.norm(_safe_log_gain(roll_gains))) >= 0.006


def _roll_membership_score(
    frame: FrameAnalysis,
    roll_gains: tuple[float, float, float],
    exposure: ExposureMetrics,
    roll_after_cast: float | None = None,
) -> float:
    exposure_class = getattr(frame, "exposure_class", "normal")
    if exposure_class == "low-light salvageable":
        exposure_score = 0.45
    elif exposure_class == "extreme protected":
        exposure_score = 0.20
    elif not _exposure_usable_for_roll_baseline(exposure):
        exposure_score = 0.35
    else:
        exposure_score = 1.0
    roll_norm = float(np.linalg.norm(_safe_log_gain(roll_gains)))
    if roll_norm < 0.006:
        return 0.0
    before = max(0.0, float(frame.before_cast))
    if roll_after_cast is None:
        roll_effect_score = 0.50
    else:
        roll_after = max(0.0, float(roll_after_cast))
        improvement = before - roll_after
        scale = max(0.006, before * 0.80)
        roll_effect_score = max(0.0, min(1.0, 0.50 + improvement / scale))
        near_neutral_worsening = before <= 0.012 and roll_after > before + max(0.002, before * 0.25)
        clear_worsening = roll_after > before + max(0.004, before * 0.50)
        if near_neutral_worsening:
            roll_effect_score *= 0.15
        elif clear_worsening:
            roll_effect_score *= 0.45
    if frame.algorithm == "identity" or frame.confidence <= 0.0:
        candidate_score = 0.15 + 0.45 * roll_effect_score
    else:
        frame_log = _safe_log_gain(frame.rgb_gains)
        roll_log = _safe_log_gain(roll_gains)
        frame_norm = float(np.linalg.norm(frame_log))
        if frame_norm < 0.015:
            direction_score = 0.65
        else:
            cosine = float(np.dot(frame_log, roll_log) / max(1e-6, frame_norm * roll_norm))
            direction_score = max(0.0, min(1.0, (cosine + 1.0) * 0.5))
        distance = float(np.linalg.norm(frame_log - roll_log))
        distance_score = max(0.0, min(1.0, 1.0 - distance / 0.42))
        candidate_score = (
            0.38 * direction_score
            + 0.28 * distance_score
            + 0.22 * frame.candidate_agreement
            + 0.12 * roll_effect_score
        )
    evidence_score = min(1.0, frame.neutral_fraction / 0.08)
    membership = 0.42 * candidate_score + 0.33 * roll_effect_score + 0.15 * evidence_score + 0.10 * exposure_score
    return max(0.0, min(1.0, membership))


def _roll_cast_strength_hint(
    frames: tuple[FrameAnalysis, ...] | list[FrameAnalysis],
    roll_gains: tuple[float, float, float],
    kept_paths: set[str],
) -> tuple[float, str]:
    roll_log = _safe_log_gain(roll_gains)
    roll_norm = float(np.linalg.norm(roll_log))
    if roll_norm < 0.035 or len(kept_paths) < 5:
        return 1.0, "normal"
    reliable = _strong_roll_reliable_frames(frames, kept_paths)
    if len(reliable) < 5:
        return 1.0, "normal"
    cosines: list[float] = []
    for frame in reliable:
        frame_log = _safe_log_gain(frame.rgb_gains)
        frame_norm = float(np.linalg.norm(frame_log))
        if frame_norm < 0.025:
            continue
        cosines.append(float(np.dot(frame_log, roll_log) / max(1e-6, frame_norm * roll_norm)))
    if len(cosines) < 5:
        return 1.0, "normal"

    values = np.asarray(cosines, dtype=np.float64)
    positive_cluster = values[values >= 0.35]
    agreement = float(positive_cluster.size / max(1, values.size))
    if positive_cluster.size < max(5, int(math.ceil(values.size * 0.60))):
        return 1.0, "normal"
    cluster_median = float(np.median(positive_cluster))
    if agreement < 0.62 or cluster_median < 0.65:
        return 1.0, "normal"

    strength = 1.0 + min(0.18, (agreement - 0.60) * 0.28 + min(0.08, roll_norm * 0.45))
    if strength >= 1.12:
        return round(float(strength), 3), "strong"
    if strength >= 1.055:
        return round(float(strength), 3), "moderate"
    return 1.0, "normal"


def _strong_roll_reliable_frames(
    frames: tuple[FrameAnalysis, ...] | list[FrameAnalysis],
    kept_paths: set[str],
) -> list[FrameAnalysis]:
    reliable = [
        frame
        for frame in frames
        if frame.path in kept_paths
        and frame.confidence >= 0.30
        and frame.algorithm != "identity"
        and getattr(frame, "exposure_class", "normal") == "normal"
        and frame.scene_bias not in {"sky-heavy", "foliage-heavy", "skin/object-heavy", "high-saturation"}
        and not getattr(frame, "perceptual_warning", "")
    ]
    if len(reliable) >= 5:
        return reliable
    return [
        frame
        for frame in frames
        if frame.path in kept_paths
        and frame.confidence >= 0.30
        and frame.algorithm != "identity"
        and getattr(frame, "exposure_class", "normal") == "normal"
    ]


def _strengthened_roll_gains(
    roll_gains: tuple[float, float, float],
    roll_cast_strength: float,
    roll_membership: float,
    frame: FrameAnalysis,
) -> tuple[float, float, float]:
    base = normalize_rgb_gains(roll_gains)
    strength = _clamp_float(float(roll_cast_strength), 1.0, 1.20)
    if strength <= 1.001 or roll_membership < 0.80:
        return base
    if getattr(frame, "exposure_class", "normal") != "normal":
        return base
    if frame.before_cast <= 0.012 and roll_membership < 0.92:
        return base
    if frame.scene_bias in {"sky-heavy", "skin/object-heavy", "high-saturation"} and roll_membership < 0.92:
        return base
    if frame.perceptual_warning:
        return base
    factor = 1.0 + (strength - 1.0) * min(1.0, max(0.0, (roll_membership - 0.80) / 0.18))
    roll_log = _safe_log_gain(base) * factor
    return normalize_rgb_gains(tuple(float(math.exp(value)) for value in roll_log))


def _frame_candidate_reliable(frame: FrameAnalysis) -> bool:
    return frame.algorithm != "identity" and frame.confidence >= 0.28


def _roll_membership_blocks_roll(
    frame: FrameAnalysis,
    before_cast: float,
    roll_after_cast: float,
    roll_membership: float,
) -> bool:
    if roll_after_cast <= before_cast + max(0.0015, before_cast * 0.10):
        return False
    if roll_membership < MIN_ROLL_MEMBERSHIP_FOR_ROLL:
        return True
    if frame.algorithm == "identity" and before_cast <= 0.014 and roll_membership < 0.48:
        return True
    return False


def _candidate_agrees_with_roll(
    frame_gains: tuple[float, float, float],
    roll_gains: tuple[float, float, float],
) -> bool:
    frame_log = _safe_log_gain(frame_gains)
    roll_log = _safe_log_gain(roll_gains)
    frame_norm = float(np.linalg.norm(frame_log))
    roll_norm = float(np.linalg.norm(roll_log))
    if frame_norm < 0.04 or roll_norm < 0.04:
        return True
    cosine = float(np.dot(frame_log, roll_log) / max(1e-6, frame_norm * roll_norm))
    return cosine >= 0.20


def _candidate_conflicts_with_roll(
    frame_gains: tuple[float, float, float],
    roll_gains: tuple[float, float, float],
) -> bool:
    frame_log = _safe_log_gain(frame_gains)
    roll_log = _safe_log_gain(roll_gains)
    frame_norm = float(np.linalg.norm(frame_log))
    roll_norm = float(np.linalg.norm(roll_log))
    if frame_norm < 0.11 or roll_norm < 0.04:
        return False
    cosine = float(np.dot(frame_log, roll_log) / max(1e-6, frame_norm * roll_norm))
    return cosine < -0.25


def _roll_would_cause_severe_local_harm(before_cast: float, after_cast: float) -> bool:
    if before_cast <= 0.010 and after_cast >= 0.018:
        return True
    if before_cast <= 0.012 and after_cast >= 0.022:
        return True
    if before_cast <= 0.024 and after_cast >= 0.035:
        return True
    return after_cast > before_cast + 0.055


def _default_roll_frame_gains(
    roll_gains: tuple[float, float, float],
    frame_gains: tuple[float, float, float],
    *,
    residual_strength: float = 0.45,
    max_residual_log: float = 0.14,
) -> tuple[float, float, float]:
    roll_log = _safe_log_gain(roll_gains)
    frame_log = _safe_log_gain(frame_gains)
    residual = np.clip(frame_log - roll_log, -abs(max_residual_log), abs(max_residual_log))
    combined = roll_log + residual * residual_strength
    return normalize_rgb_gains(tuple(float(math.exp(value)) for value in combined))


def _scale_rgb_gains_to_identity(
    gains: tuple[float, float, float],
    strength: float,
) -> tuple[float, float, float]:
    strength = _clamp_float(float(strength), 0.0, 1.0)
    gain_log = _safe_log_gain(normalize_rgb_gains(gains)) * strength
    return normalize_rgb_gains(tuple(float(math.exp(value)) for value in gain_log))


def _roll_exposure_targets(frames: tuple[FrameAnalysis, ...], preferred_paths: set[str] | None = None) -> tuple[float, float]:
    preferred_paths = preferred_paths or set()
    preferred = [
        frame
        for frame in frames
        if frame.path in preferred_paths and _frame_exposure_usable_values(frame)
    ]
    source = preferred
    if len(source) < 3:
        source = [frame for frame in frames if _frame_exposure_usable_values(frame)]
    p50_values = np.asarray(
        [frame.luma_p50 for frame in source],
        dtype=np.float64,
    )
    p95_values = np.asarray(
        [frame.luma_p95 for frame in source],
        dtype=np.float64,
    )
    if p50_values.size == 0:
        p50 = 0.38
    else:
        p50 = float(np.median(p50_values))
    if p95_values.size == 0:
        p95 = 0.86
    else:
        p95 = float(np.median(p95_values))
    return p50, p95


def _frame_exposure_usable_values(frame: FrameAnalysis) -> bool:
    if getattr(frame, "exposure_class", "normal") != "normal":
        return False
    if frame.luma_p50 <= 0.10 or frame.luma_p50 >= 0.88:
        return False
    if frame.luma_p95 <= 0.35 or frame.luma_p95 >= 0.985:
        return False
    return True


def _exposure_usable_for_roll_baseline(exposure: ExposureMetrics) -> bool:
    if exposure.shadow_clip_fraction >= 0.10 or exposure.highlight_clip_fraction >= 0.06:
        return False
    if exposure.p50 <= 0.10 or exposure.p50 >= 0.88:
        return False
    if exposure.p95 <= 0.35 or exposure.p95 >= 0.992:
        return False
    if (exposure.p75 - exposure.p25) <= 0.045:
        return False
    return True


def _frame_exposure_delta(
    exposure: ExposureMetrics,
    target_p50: float,
    target_p95: float,
) -> tuple[float, str, float]:
    if exposure.p50 <= 0.001 or exposure.p95 <= 0.001:
        return 0.0, "exposure protected", 0.0
    if exposure.shadow_clip_fraction >= 0.12 or exposure.highlight_clip_fraction >= 0.07:
        return 0.0, "exposure protected", 0.0
    tonal_span = exposure.p75 - exposure.p25
    if (
        exposure.p50 <= 0.08
        or exposure.p95 <= 0.35
        or exposure.p50 >= 0.90
        or exposure.p95 >= 0.992
        or tonal_span <= 0.045
    ):
        return 0.0, "exposure protected", 0.0

    midtone_delta = math.log(max(0.001, target_p50) / max(0.001, exposure.p50), 2.0)
    highlight_delta = math.log(max(0.001, target_p95) / max(0.001, exposure.p95), 2.0)
    if midtone_delta >= 0.50 and highlight_delta >= 0.22 and exposure.p95 <= 0.90:
        delta = midtone_delta * 0.70 + highlight_delta * 0.30
        delta = _clamp_float(delta, 0.0, 0.55)
        confidence = min(1.0, 0.55 * (midtone_delta / 0.90) + 0.45 * (highlight_delta / 0.70))
        if delta >= 0.10:
            return delta, "exposure brighten", confidence
        return 0.0, "none", 0.0

    if (
        midtone_delta <= -0.48
        and highlight_delta <= -0.20
        and exposure.p95 <= 0.965
        and exposure.highlight_clip_fraction <= 0.01
    ):
        delta = midtone_delta * 0.65 + highlight_delta * 0.35
        delta = _clamp_float(delta, -0.32, 0.0)
        confidence = min(1.0, 0.55 * (abs(midtone_delta) / 0.85) + 0.45 * (abs(highlight_delta) / 0.65))
        if delta <= -0.10:
            return delta, "exposure darken", confidence
    return 0.0, "none", 0.0


def _tone_residual_gains(
    proxy,
    base_gains: tuple[float, float, float],
    color_action: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
    if color_action in {"none", "protected", "review", "skip-extreme"}:
        return IDENTITY_GAINS_RGB, IDENTITY_GAINS_RGB, IDENTITY_GAINS_RGB, 0.0
    corrected = apply_rgb_gains_to_bgr(proxy, base_gains)
    rgb = _rgb_float01(corrected)
    flat = rgb.reshape((-1, 3))
    if flat.shape[0] < 192:
        return IDENTITY_GAINS_RGB, IDENTITY_GAINS_RGB, IDENTITY_GAINS_RGB, 0.0
    luma = _rgb_luma(flat)
    shadow = _tone_band_residual(flat, luma, 0.08, 0.38, max_log=0.055)
    mid = _tone_band_residual(flat, luma, 0.25, 0.72, max_log=0.105)
    highlight = _tone_band_residual(flat, luma, 0.58, 0.93, max_log=0.050)
    valid_weights = [shadow[1], mid[1], highlight[1]]
    if max(valid_weights) <= 0.0:
        return IDENTITY_GAINS_RGB, IDENTITY_GAINS_RGB, IDENTITY_GAINS_RGB, 0.0
    confidence = min(1.0, (0.22 * shadow[1]) + (0.56 * mid[1]) + (0.22 * highlight[1]))
    return shadow[0], mid[0], highlight[0], confidence


def _tone_band_residual(
    flat_rgb,
    luma,
    low: float,
    high: float,
    *,
    max_log: float,
) -> tuple[tuple[float, float, float], float]:
    mask = (luma >= low) & (luma <= high)
    if int(np.count_nonzero(mask)) < 96:
        return IDENTITY_GAINS_RGB, 0.0
    band_pixels = flat_rgb[mask]
    band_luma = luma[mask]
    max_channel = np.max(band_pixels, axis=1)
    min_channel = np.min(band_pixels, axis=1)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    clean = (
        (band_luma >= 0.04)
        & (band_luma <= 0.97)
        & (min_channel >= 0.004)
        & (max_channel <= 0.992)
    )
    selected = _neutral_mask(band_pixels, band_luma, saturation, clean)
    selected_count = int(np.count_nonzero(selected))
    if selected_count < 64:
        return IDENTITY_GAINS_RGB, 0.0
    selected_pixels = band_pixels[selected]
    selected_luma = band_luma[selected]
    channel_medians = np.median(selected_pixels, axis=0)
    target = float(np.median(selected_luma))
    if target <= 1e-6 or float(np.min(channel_medians)) <= 1e-6:
        return IDENTITY_GAINS_RGB, 0.0
    residual = normalize_rgb_gains(tuple(float(target / channel_medians[index]) for index in range(3)))
    residual_log = _safe_log_gain(residual)
    residual_norm = float(np.linalg.norm(residual_log))
    if residual_norm > max_log:
        residual_log = residual_log * (max_log / max(residual_norm, 1e-6))
        residual = normalize_rgb_gains(tuple(float(math.exp(value)) for value in residual_log))
    count_score = min(1.0, selected_count / 1600.0)
    fraction_score = min(1.0, selected_count / max(1, int(np.count_nonzero(mask))) / 0.10)
    saturation_score = max(0.0, min(1.0, 1.0 - float(np.median(saturation[selected])) / 0.32))
    confidence = 0.35 * count_score + 0.35 * fraction_score + 0.30 * saturation_score
    return residual, confidence


def _perceptual_side_effect_check(
    proxy,
    rgb_gains: tuple[float, float, float],
    color_action: str,
    shadow_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
    mid_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
    highlight_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
) -> tuple[float, str]:
    if color_action in {"none", "protected", "review", "skip-extreme"}:
        return 0.0, ""
    before_rgb = _rgb_float01(proxy)
    after_rgb = _rgb_float01(
        apply_tone_aware_rgb_gains_to_bgr(
            proxy,
            rgb_gains,
            shadow_rgb_gains,
            mid_rgb_gains,
            highlight_rgb_gains,
            tone_strength=1.0,
        )
    )
    before_flat = before_rgb.reshape((-1, 3))
    after_flat = after_rgb.reshape((-1, 3))
    if before_flat.shape[0] < 128:
        return 0.0, ""
    before_luma = _rgb_luma(before_flat)
    after_luma = _rgb_luma(after_flat)
    max_channel = np.max(before_flat, axis=1)
    min_channel = np.min(before_flat, axis=1)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    red = before_flat[:, 0]
    green = before_flat[:, 1]
    blue = before_flat[:, 2]
    height, width = before_rgb.shape[:2]
    y_position = np.repeat(np.arange(height, dtype=np.float32)[:, None], width, axis=1).reshape(-1) / max(1, height - 1)

    risks: list[tuple[float, str]] = []
    highlight_sky = (
        (before_luma >= 0.62)
        & (before_luma <= 0.97)
        & (y_position <= 0.72)
        & (blue >= red * 1.01)
        & (blue >= green * 0.97)
        & (saturation >= 0.035)
        & (saturation <= 0.32)
    )
    risks.append(
        _perceptual_chroma_drift(
            before_flat,
            after_flat,
            before_luma,
            after_luma,
            highlight_sky,
            "highlight sky tint drift",
            drift_limit=0.028,
            chroma_limit=0.010,
            minimum_pixels=96,
            minimum_fraction=0.06,
        )
    )

    foliage = (
        (green > red * 1.04)
        & (green > blue * 1.08)
        & (before_luma >= 0.12)
        & (saturation >= 0.18)
    )
    risks.append(
        _perceptual_chroma_drift(
            before_flat,
            after_flat,
            before_luma,
            after_luma,
            foliage,
            "foliage hue drift",
            drift_limit=0.040,
            chroma_limit=0.015,
            minimum_pixels=128,
        )
    )

    red_object = (
        (red > green * 1.08)
        & (red > blue * 1.15)
        & (before_luma >= 0.16)
        & (saturation >= 0.28)
    )
    risks.append(
        _perceptual_chroma_drift(
            before_flat,
            after_flat,
            before_luma,
            after_luma,
            red_object,
            "skin/red object hue drift",
            drift_limit=0.040,
            chroma_limit=0.015,
            minimum_pixels=128,
        )
    )

    shadow = (before_luma <= 0.22) & (before_luma >= 0.025)
    shadow_risk = _shadow_color_noise_risk(before_flat, after_flat, before_luma, after_luma, shadow)
    if shadow_risk[0] > 0.0:
        risks.append(shadow_risk)

    risk, warning = max(risks, key=lambda item: item[0], default=(0.0, ""))
    if risk <= 0.0:
        return 0.0, ""
    return min(1.0, float(risk)), warning


def _should_run_perceptual_check(frame: FrameAnalysis, color_action: str) -> bool:
    if color_action in {"none", "protected", "review", "skip-extreme"}:
        return False
    if getattr(frame, "exposure_class", "normal") == "low-light salvageable":
        return True
    if color_action == "roll+frame":
        return True
    if frame.scene_bias in {"sky-heavy", "foliage-heavy", "skin/object-heavy", "high-saturation", "shadow-dominant"}:
        return True
    if frame.rejected_region_count >= 6 and frame.region_rejection_reasons:
        return True
    return False


def _perceptual_chroma_drift(
    before_flat,
    after_flat,
    before_luma,
    after_luma,
    mask,
    warning: str,
    *,
    drift_limit: float,
    chroma_limit: float,
    minimum_pixels: int,
    minimum_fraction: float = 0.0,
) -> tuple[float, str]:
    if int(np.count_nonzero(mask)) < minimum_pixels:
        return 0.0, ""
    if float(np.mean(mask)) < float(minimum_fraction):
        return 0.0, ""
    before_chroma = _masked_chroma_center(before_flat, before_luma, mask)
    after_chroma = _masked_chroma_center(after_flat, after_luma, mask)
    before_mag = float(np.linalg.norm(before_chroma))
    after_mag = float(np.linalg.norm(after_chroma))
    drift = float(np.linalg.norm(after_chroma - before_chroma))
    if after_mag <= before_mag + chroma_limit and drift <= drift_limit:
        return 0.0, ""
    risk = max(
        0.0,
        min(
            1.0,
            0.55 * max(0.0, (after_mag - before_mag) / max(chroma_limit * 2.5, 1e-6))
            + 0.45 * max(0.0, drift / max(drift_limit * 2.0, 1e-6)),
        ),
    )
    if risk < 0.30:
        return 0.0, ""
    return risk, warning


def _masked_chroma_center(flat_rgb, luma, mask):
    residual = flat_rgb[mask] - luma[mask, None]
    if residual.shape[0] <= 0:
        return np.zeros(3, dtype=np.float64)
    center = np.median(residual, axis=0)
    return _remove_luma_component(center)


def _shadow_color_noise_risk(before_flat, after_flat, before_luma, after_luma, mask) -> tuple[float, str]:
    if int(np.count_nonzero(mask)) < 128:
        return 0.0, ""
    before_residual = before_flat[mask] - before_luma[mask, None]
    after_residual = after_flat[mask] - after_luma[mask, None]
    before_noise = float(np.median(np.linalg.norm(before_residual - np.median(before_residual, axis=0), axis=1)))
    after_noise = float(np.median(np.linalg.norm(after_residual - np.median(after_residual, axis=0), axis=1)))
    if after_noise <= before_noise + 0.018:
        return 0.0, ""
    risk = min(1.0, (after_noise - before_noise) / 0.06)
    if risk < 0.30:
        return 0.0, ""
    return risk, "shadow color noise"


def combined_frame_gains(
    roll: RollAnalysisResult,
    frame: FrameAnalysis,
    *,
    roll_strength: float = 1.0,
    frame_strength: float = 0.45,
    max_residual_log: float = 0.14,
) -> tuple[float, float, float]:
    roll_strength = _clamp_float(float(roll_strength), 0.0, 1.25)
    frame_strength = _clamp_float(float(frame_strength), 0.0, 1.0)
    if frame.color_action in {"none", "protected", "review", "skip-extreme"}:
        return IDENTITY_GAINS_RGB
    if frame.color_action in {"safe", "frame", "roll", "roll_frame", "roll+frame", "frame-only"}:
        safe_log = _safe_log_gain(frame.safe_rgb_gains)
        if frame.color_action == "roll":
            strength = roll_strength
        elif frame.color_action in {"frame", "frame-only"}:
            strength = max(roll_strength, frame_strength)
        else:
            strength = max(roll_strength, frame_strength)
        return normalize_rgb_gains(tuple(float(math.exp(value * strength)) for value in safe_log))

    roll_log = _safe_log_gain(roll.roll_rgb_gains)
    frame_log = _safe_log_gain(frame.rgb_gains)
    residual = frame_log - roll_log
    residual = np.clip(residual, -abs(max_residual_log), abs(max_residual_log))
    if frame.confidence < 0.28 or frame.algorithm == "identity":
        residual *= 0.0
    if frame.color_action == "frame":
        combined = frame_log * frame_strength
    elif frame.color_action == "roll":
        combined = roll_log * roll_strength
    else:
        combined = roll_log * roll_strength + residual * frame_strength
    return normalize_rgb_gains(tuple(float(math.exp(value)) for value in combined))


def apply_rgb_gains_to_bgr(image, rgb_gains: tuple[float, float, float]):
    _require_cv2_np()
    gains = normalize_rgb_gains(rgb_gains)
    b_gain, g_gain, r_gain = float(gains[2]), float(gains[1]), float(gains[0])
    if image.ndim == 3 and image.shape[2] == 3 and hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "applyChannelGains"):
        try:
            return cv2.xphoto.applyChannelGains(image, b_gain, g_gain, r_gain)
        except Exception:
            pass
    max_value = _max_value_for_dtype(image.dtype)
    output = image.astype(np.float32, copy=True)
    output[:, :, :3] = output[:, :, :3] * np.asarray([b_gain, g_gain, r_gain], dtype=np.float32)[None, None, :]
    output[:, :, :3] = np.clip(np.rint(output[:, :, :3]), 0, max_value)
    return output.astype(image.dtype)


def apply_tone_aware_rgb_gains_to_bgr(
    image,
    base_rgb_gains: tuple[float, float, float],
    shadow_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
    mid_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
    highlight_rgb_gains: tuple[float, float, float] = IDENTITY_GAINS_RGB,
    *,
    tone_strength: float = 1.0,
):
    _require_cv2_np()
    tone_strength = _clamp_float(float(tone_strength), 0.0, 1.0)
    if tone_strength <= 0.001:
        return apply_rgb_gains_to_bgr(image, base_rgb_gains)

    base_log = _safe_log_gain(normalize_rgb_gains(base_rgb_gains))
    shadow_log = _safe_log_gain(normalize_rgb_gains(shadow_rgb_gains))
    mid_log = _safe_log_gain(normalize_rgb_gains(mid_rgb_gains))
    highlight_log = _safe_log_gain(normalize_rgb_gains(highlight_rgb_gains))
    if max(float(np.linalg.norm(shadow_log)), float(np.linalg.norm(mid_log)), float(np.linalg.norm(highlight_log))) <= 0.001:
        return apply_rgb_gains_to_bgr(image, base_rgb_gains)

    max_value = float(_max_value_for_dtype(image.dtype))
    working = image[:, :, :3].astype(np.float32, copy=False)
    b = working[:, :, 0] / max_value
    g = working[:, :, 1] / max_value
    r = working[:, :, 2] / max_value
    luma = (
        r * np.float32(SRGB_LUMA_WEIGHTS_RGB[0])
        + g * np.float32(SRGB_LUMA_WEIGHTS_RGB[1])
        + b * np.float32(SRGB_LUMA_WEIGHTS_RGB[2])
    )
    shadow_w = 1.0 - _smoothstep(0.14, 0.42, luma)
    mid_w = _smoothstep(0.18, 0.42, luma) * (1.0 - _smoothstep(0.62, 0.86, luma))
    highlight_w = _smoothstep(0.58, 0.88, luma)
    residual_logs = np.asarray([shadow_log, mid_log, highlight_log], dtype=np.float32) * np.float32(tone_strength)
    output = working.copy()
    for bgr_channel, rgb_channel in enumerate((2, 1, 0)):
        log_gain = (
            np.float32(base_log[rgb_channel])
            + shadow_w * residual_logs[0, rgb_channel]
            + mid_w * residual_logs[1, rgb_channel]
            + highlight_w * residual_logs[2, rgb_channel]
        )
        output[:, :, bgr_channel] = output[:, :, bgr_channel] * np.exp(log_gain).astype(np.float32, copy=False)
    output = np.clip(np.rint(output), 0, max_value).astype(image.dtype)
    if image.ndim == 3 and image.shape[2] > 3:
        return np.dstack((output, image[:, :, 3:]))
    return output


def apply_exposure_to_bgr(image, delta_stops: float, *, strength: float = 1.0):
    _require_cv2_np()
    strength = _clamp_float(float(strength), 0.0, 1.0)
    delta = _clamp_float(float(delta_stops) * strength, -1.2, 1.2)
    if abs(delta) < 0.001:
        return image.copy()
    factor = float(2.0 ** delta)
    max_value = _max_value_for_dtype(image.dtype)
    output = image.astype(np.float32, copy=True)
    output[:, :, :3] = output[:, :, :3] * factor
    output[:, :, :3] = np.clip(np.rint(output[:, :, :3]), 0, max_value)
    return output.astype(image.dtype)


def apply_roll_plan_to_bgr(
    image,
    roll: RollAnalysisResult,
    frame: FrameAnalysis,
    *,
    roll_strength: float = 1.0,
    frame_strength: float = 0.45,
    tone_strength: float = 1.0,
    exposure_strength: float = 0.0,
):
    gains = combined_frame_gains(roll, frame, roll_strength=roll_strength, frame_strength=frame_strength)
    if frame.color_action in {"none", "protected", "review", "skip-extreme"}:
        corrected = apply_rgb_gains_to_bgr(image, gains)
    else:
        corrected = apply_tone_aware_rgb_gains_to_bgr(
            image,
            gains,
            frame.tone_shadow_rgb_gains,
            frame.tone_mid_rgb_gains,
            frame.tone_highlight_rgb_gains,
            tone_strength=tone_strength,
        )
    if getattr(frame, "highlight_protection_strength", 0.0) > 0.0:
        corrected = _apply_highlight_protection_to_bgr(
            image,
            corrected,
            getattr(frame, "highlight_protection_strength", 0.0),
        )
    return apply_exposure_to_bgr(corrected, frame.exposure_delta_stops, strength=exposure_strength)


def resize_bgr_to_fit(image, max_size: tuple[int, int]):
    max_width, max_height = max_size
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image.copy()
    scale = min(max_width / max(1, width), max_height / max(1, height))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def crop_bgr_border(image, crop_percent: float):
    crop = _clamp_float(float(crop_percent), 0.0, 20.0) / 100.0
    if crop <= 0.0:
        return image.copy()
    height, width = image.shape[:2]
    left = int(round(width * crop))
    right = int(round(width * (1.0 - crop)))
    top = int(round(height * crop))
    bottom = int(round(height * (1.0 - crop)))
    if right - left < max(32, width * 0.25) or bottom - top < max(32, height * 0.25):
        return image.copy()
    return image[top:bottom, left:right].copy()


def bgr_to_rgb8_preview(image):
    _require_cv2_np()
    if image.ndim != 3 or image.shape[2] < 3:
        image = _ensure_bgr(image)
    max_value = _max_value_for_dtype(image.dtype)
    if image.dtype == np.uint8:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        return rgb.copy()
    scaled = np.clip((image[:, :, :3].astype(np.float32) / float(max_value)) * 255.0, 0.0, 255.0)
    return cv2.cvtColor(scaled.astype(np.uint8), cv2.COLOR_BGR2RGB)


def normalize_rgb_gains(gains: tuple[float, float, float]) -> tuple[float, float, float]:
    values = np.asarray(gains, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 1.0)
    values = np.clip(values, 0.05, 20.0)
    luma_scale = float(values @ np.asarray(SRGB_LUMA_WEIGHTS_RGB, dtype=np.float64))
    if luma_scale <= 1e-6:
        return IDENTITY_GAINS_RGB
    values = values / luma_scale
    return tuple(float(value) for value in values)


def estimate_cast_metrics(image) -> CastMetrics:
    _require_cv2_np()
    rgb = _rgb_float01(image)
    flat = rgb.reshape((-1, 3))
    sampled = int(flat.shape[0])
    if sampled <= 0:
        return CastMetrics()
    luma = _rgb_luma(flat)
    max_channel = np.max(flat, axis=1)
    min_channel = np.min(flat, axis=1)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    clean = (
        (luma >= 0.08)
        & (luma <= 0.94)
        & (min_channel >= 0.01)
        & (max_channel <= 0.985)
    )
    selected = _neutral_mask(flat, luma, saturation, clean)
    neutral_pixels = int(np.count_nonzero(selected))
    if neutral_pixels < 16:
        selected = clean
        neutral_pixels = int(np.count_nonzero(selected))
    if neutral_pixels < 16:
        return CastMetrics(sampled_pixels=sampled)
    selected_pixels = flat[selected]
    selected_luma = luma[selected]
    residuals = selected_pixels - selected_luma[:, None]
    center = np.median(residuals, axis=0)
    distances = np.linalg.norm(residuals - center[None, :], axis=1)
    if distances.size >= 32:
        keep = distances <= float(np.quantile(distances, 0.85))
        if int(np.count_nonzero(keep)) >= 16:
            residuals = residuals[keep]
    cast = np.median(residuals, axis=0)
    cast = _remove_luma_component(cast)
    magnitude = float(np.linalg.norm(cast))
    return CastMetrics(
        magnitude=magnitude,
        cast_rgb=(float(cast[0]), float(cast[1]), float(cast[2])),
        neutral_fraction=float(neutral_pixels / max(1, sampled)),
        neutral_pixels=neutral_pixels,
        sampled_pixels=sampled,
        median_saturation=float(np.median(saturation[selected])),
    )


def estimate_exposure_metrics(image) -> ExposureMetrics:
    _require_cv2_np()
    rgb = _rgb_float01(image)
    flat = rgb.reshape((-1, 3))
    if flat.shape[0] <= 0:
        return ExposureMetrics()
    luma = _rgb_luma(flat)
    finite = luma[np.isfinite(luma)]
    if finite.size <= 0:
        return ExposureMetrics()
    p05, p25, p50, p75, p95 = (float(np.percentile(finite, value)) for value in (5, 25, 50, 75, 95))
    return ExposureMetrics(
        p05=p05,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
        shadow_clip_fraction=float(np.mean(finite <= 0.01)),
        highlight_clip_fraction=float(np.mean(finite >= 0.99)),
    )


def _ensemble_candidate(
    candidates: tuple[BalanceCandidate, ...],
    proxy,
    before_metrics: CastMetrics,
) -> BalanceCandidate:
    valid = [
        candidate
        for candidate in candidates
        if candidate.algorithm != "identity"
        and candidate.confidence >= 0.18
        and candidate.score >= 0.12
        and candidate.after_cast < before_metrics.magnitude
    ]
    if not valid:
        best = max(candidates, key=lambda candidate: candidate.score, default=BalanceCandidate("identity"))
        return BalanceCandidate(
            algorithm="identity",
            confidence=0.0,
            score=0.0,
            after_cast=before_metrics.magnitude,
            warning=best.warning or "no candidate consensus",
        )
    if len(valid) == 1:
        return valid[0]

    logs = np.asarray([_safe_log_gain(candidate.rgb_gains) for candidate in valid], dtype=np.float64)
    weights = np.asarray(
        [max(0.01, candidate.confidence) * max(0.01, candidate.score) for candidate in valid],
        dtype=np.float64,
    )
    keep = _main_gain_cluster_mask(logs, weights)
    if int(np.count_nonzero(keep)) < 1:
        keep = np.ones(len(valid), dtype=bool)
    kept_logs = logs[keep]
    kept_weights = weights[keep]
    kept_candidates = [candidate for candidate, is_kept in zip(valid, keep.tolist()) if is_kept]
    center = _weighted_median_vector(kept_logs, kept_weights)
    dispersion = float(np.average(np.linalg.norm(kept_logs - center[None, :], axis=1), weights=kept_weights))
    agreement = max(0.0, min(1.0, 1.0 - dispersion / 0.22))
    count_score = min(1.0, len(kept_candidates) / 3.0)
    gains = normalize_rgb_gains(tuple(float(math.exp(value)) for value in center))
    corrected = apply_rgb_gains_to_bgr(proxy, gains)
    after = float(estimate_cast_metrics(corrected).magnitude)
    candidate_confidence = float(np.average([candidate.confidence for candidate in kept_candidates], weights=kept_weights))
    confidence = candidate_confidence * (0.55 + 0.45 * agreement) * (0.75 + 0.25 * count_score)
    names = "+".join(candidate.algorithm for candidate in kept_candidates)
    warning = "" if len(kept_candidates) >= 2 else "single candidate cluster"
    return BalanceCandidate(
        algorithm=f"ensemble({names})" if len(kept_candidates) >= 2 else kept_candidates[0].algorithm,
        rgb_gains=gains,
        score=round(float(agreement * count_score), 4),
        confidence=round(float(confidence), 3),
        after_cast=round(float(after), 5),
        warning=warning,
    )


def _candidate_accepted_for_frame(candidate: BalanceCandidate, metrics: CastMetrics) -> bool:
    return (
        candidate.algorithm != "identity"
        and candidate.confidence >= 0.24
        and candidate.score >= 0.18
        and candidate.after_cast <= max(metrics.magnitude * 0.97, metrics.magnitude - 0.0025)
    )


def _candidate_corrections(image, before_metrics: CastMetrics) -> list[BalanceCandidate]:
    candidates = [BalanceCandidate("identity", after_cast=before_metrics.magnitude)]
    candidates.append(_neutral_percentile_candidate(image, before_metrics))
    candidates.append(_regional_neutral_candidate(image, before_metrics))
    if cv2 is None or not hasattr(cv2, "xphoto"):
        return candidates
    for name, maker in (
        ("learning", getattr(cv2.xphoto, "createLearningBasedWB", None)),
        ("grayworld", getattr(cv2.xphoto, "createGrayworldWB", None)),
    ):
        if maker is None:
            continue
        candidate = _opencv_candidate(image, before_metrics, name, maker)
        candidates.append(candidate)
    return candidates


def _regional_neutral_candidate(image, before_metrics: CastMetrics, *, grid_size: int = 4) -> BalanceCandidate:
    try:
        height, width = image.shape[:2]
        if height < 96 or width < 96:
            return BalanceCandidate("regional_neutral", warning="image too small for regional analysis")

        logs: list[object] = []
        weights: list[float] = []
        scene_counts: dict[str, int] = {}
        reject_counts: dict[str, int] = {}
        accepted_regions = 0
        rejected_regions = 0
        for row in range(grid_size):
            top = int(round(height * row / grid_size))
            bottom = int(round(height * (row + 1) / grid_size))
            for col in range(grid_size):
                left = int(round(width * col / grid_size))
                right = int(round(width * (col + 1) / grid_size))
                tile = image[top:bottom, left:right]
                tile_result = _regional_tile_gain(tile)
                scene_counts[tile_result.scene_bias] = scene_counts.get(tile_result.scene_bias, 0) + 1
                if tile_result.reject_reason:
                    rejected_regions += 1
                    reject_counts[tile_result.reject_reason] = reject_counts.get(tile_result.reject_reason, 0) + 1
                    continue
                logs.append(_safe_log_gain(tile_result.gains))
                weights.append(float(tile_result.weight))
                accepted_regions += 1

        dominant_scene = _dominant_count_key(scene_counts, default="mixed-neutral")
        rejection_reasons = tuple(_top_count_keys(reject_counts, limit=3))
        if accepted_regions < 3:
            return BalanceCandidate(
                "regional_neutral",
                scene_bias=dominant_scene,
                accepted_region_count=accepted_regions,
                rejected_region_count=rejected_regions,
                region_rejection_reasons=rejection_reasons,
                warning="not enough reliable regions",
            )

        log_values = np.asarray(logs, dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)
        keep = _main_gain_cluster_mask(log_values, weight_values)
        if int(np.count_nonzero(keep)) >= 3:
            kept_logs = log_values[keep]
            kept_weights = weight_values[keep]
        else:
            kept_logs = log_values
            kept_weights = weight_values

        center = _weighted_median_vector(kept_logs, kept_weights)
        distances = np.linalg.norm(kept_logs - center[None, :], axis=1)
        dispersion = float(np.average(distances, weights=kept_weights))
        agreement = max(0.0, min(1.0, 1.0 - dispersion / 0.26))
        gains = normalize_rgb_gains(tuple(float(math.exp(value)) for value in center))
        corrected = apply_rgb_gains_to_bgr(image, gains)
        after = estimate_cast_metrics(corrected)
        score, confidence, warning = _score_candidate(before_metrics, after, gains)
        region_score = min(1.0, int(kept_logs.shape[0]) / 8.0)
        confidence = confidence * (0.60 + 0.40 * agreement) * (0.70 + 0.30 * region_score)
        score = score * (0.70 + 0.30 * agreement)
        if int(kept_logs.shape[0]) < accepted_regions:
            warning = warning or "regional outliers rejected"
        return BalanceCandidate(
            algorithm="regional_neutral",
            rgb_gains=gains,
            score=round(float(score), 4),
            confidence=round(float(confidence), 3),
            after_cast=round(float(after.magnitude), 5),
            regional_sample_count=int(kept_logs.shape[0]),
            regional_agreement=round(float(agreement), 3),
            scene_bias=dominant_scene,
            accepted_region_count=accepted_regions,
            rejected_region_count=rejected_regions,
            region_rejection_reasons=rejection_reasons,
            warning=warning,
        )
    except Exception as exc:
        return BalanceCandidate("regional_neutral", warning=str(exc))


def _regional_tile_gain(tile) -> _RegionalTileResult:
    if tile.size <= 0:
        return _RegionalTileResult(reject_reason="empty")
    exposure = estimate_exposure_metrics(tile)
    tonal_span = float(exposure.p75 - exposure.p25)
    if exposure.p50 <= 0.08 or exposure.p50 >= 0.88:
        return _RegionalTileResult(reject_reason="extreme midtone")
    if exposure.p95 <= 0.28 or exposure.p95 >= 0.992:
        return _RegionalTileResult(reject_reason="extreme highlight")
    if exposure.shadow_clip_fraction >= 0.08 or exposure.highlight_clip_fraction >= 0.05:
        return _RegionalTileResult(reject_reason="clipping")
    if tonal_span <= 0.045:
        return _RegionalTileResult(reject_reason="low dynamic range")

    rgb = _rgb_float01(tile)
    flat = rgb.reshape((-1, 3))
    if flat.shape[0] < 96:
        return _RegionalTileResult(reject_reason="too few pixels")
    luma = _rgb_luma(flat)
    max_channel = np.max(flat, axis=1)
    min_channel = np.min(flat, axis=1)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    scene_bias, scene_weight, scene_reject = _classify_region_scene(flat, luma, saturation, exposure)
    if scene_reject:
        return _RegionalTileResult(scene_bias=scene_bias, reject_reason=scene_reject)
    clean = (
        (luma >= 0.07)
        & (luma <= 0.95)
        & (min_channel >= 0.008)
        & (max_channel <= 0.988)
    )
    if int(np.count_nonzero(clean)) < 96:
        return _RegionalTileResult(scene_bias=scene_bias, reject_reason="too few clean pixels")
    if float(np.median(saturation[clean])) >= 0.40:
        return _RegionalTileResult(scene_bias=scene_bias, reject_reason="high saturation")

    selected = _neutral_mask(flat, luma, saturation, clean)
    selected_count = int(np.count_nonzero(selected))
    if selected_count < 64:
        return _RegionalTileResult(scene_bias=scene_bias, reject_reason="not enough neutral pixels")
    neutral_fraction = selected_count / max(1, int(np.count_nonzero(clean)))
    if neutral_fraction < 0.045:
        return _RegionalTileResult(scene_bias=scene_bias, reject_reason="low neutral fraction")

    selected_pixels = flat[selected]
    selected_luma = luma[selected]
    center = np.median(selected_pixels, axis=0)
    distances = np.linalg.norm(selected_pixels - center[None, :], axis=1)
    if distances.size >= 96:
        keep = distances <= float(np.quantile(distances, 0.78))
        if int(np.count_nonzero(keep)) >= 64:
            selected_pixels = selected_pixels[keep]
            selected_luma = selected_luma[keep]
    channel_medians = np.median(selected_pixels, axis=0)
    target = float(np.median(selected_luma))
    if target <= 1e-6 or float(np.min(channel_medians)) <= 1e-6:
        return _RegionalTileResult(scene_bias=scene_bias, reject_reason="unstable neutral sample")

    gains = normalize_rgb_gains(tuple(float(target / channel_medians[index]) for index in range(3)))
    gain_norm = float(np.linalg.norm(_safe_log_gain(gains)))
    if gain_norm > 0.80:
        return _RegionalTileResult(scene_bias=scene_bias, reject_reason="large regional correction")
    saturation_score = max(0.0, min(1.0, 1.0 - float(np.median(saturation[selected])) / 0.32))
    dynamic_score = max(0.0, min(1.0, tonal_span / 0.22))
    count_score = min(1.0, selected_count / 800.0)
    weight = (0.30 + 0.30 * saturation_score + 0.25 * dynamic_score + 0.15 * count_score) * scene_weight
    return _RegionalTileResult(gains=gains, weight=weight, scene_bias=scene_bias)


def _classify_region_scene(flat_rgb, luma, saturation, exposure: ExposureMetrics) -> tuple[str, float, str]:
    if flat_rgb.shape[0] <= 0:
        return "mixed-neutral", 1.0, "empty"
    red = flat_rgb[:, 0]
    green = flat_rgb[:, 1]
    blue = flat_rgb[:, 2]
    saturated = saturation >= 0.30
    strongly_saturated = saturation >= 0.42
    high_saturation_fraction = float(np.mean(saturation >= 0.48))
    sky_fraction = float(np.mean((blue > red * 1.05) & (blue >= green * 0.98) & (luma >= 0.34) & (saturation >= 0.08)))
    foliage_fraction = float(np.mean((green > red * 1.04) & (green > blue * 1.08) & (luma >= 0.12) & saturated))
    red_object_fraction = float(np.mean((red > green * 1.08) & (red > blue * 1.15) & (luma >= 0.16) & strongly_saturated))
    shadow_fraction = float(np.mean(luma <= 0.18))
    if shadow_fraction >= 0.70 or (exposure.p50 <= 0.18 and exposure.p75 <= 0.34):
        return "shadow-dominant", 0.45, "shadow dominant"
    if sky_fraction >= 0.46:
        return "sky-heavy", 0.35, "sky dominated"
    if foliage_fraction >= 0.46:
        return "foliage-heavy", 0.30, "foliage dominated"
    if red_object_fraction >= 0.40:
        return "skin/object-heavy", 0.40, "skin or red object dominated"
    if high_saturation_fraction >= 0.42:
        return "high-saturation", 0.20, "high saturation scene"
    if sky_fraction >= 0.28:
        return "sky-heavy", 0.55, ""
    if foliage_fraction >= 0.28:
        return "foliage-heavy", 0.50, ""
    if red_object_fraction >= 0.24:
        return "skin/object-heavy", 0.55, ""
    return "mixed-neutral", 1.0, ""


def _neutral_percentile_candidate(image, before_metrics: CastMetrics) -> BalanceCandidate:
    try:
        rgb = _rgb_float01(image)
        flat = rgb.reshape((-1, 3))
        if flat.shape[0] < 64:
            return BalanceCandidate("neutral_percentile", warning="not enough pixels")
        luma = _rgb_luma(flat)
        max_channel = np.max(flat, axis=1)
        min_channel = np.min(flat, axis=1)
        saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
        clean = (
            (luma >= 0.06)
            & (luma <= 0.96)
            & (min_channel >= 0.006)
            & (max_channel <= 0.992)
        )
        selected = _neutral_mask(flat, luma, saturation, clean)
        if int(np.count_nonzero(selected)) < 96:
            return BalanceCandidate("neutral_percentile", warning="not enough neutral pixels")
        selected_pixels = flat[selected]
        selected_luma = luma[selected]
        center = np.median(selected_pixels, axis=0)
        distances = np.linalg.norm(selected_pixels - center[None, :], axis=1)
        if distances.size >= 128:
            keep = distances <= float(np.quantile(distances, 0.80))
            if int(np.count_nonzero(keep)) >= 96:
                selected_pixels = selected_pixels[keep]
                selected_luma = selected_luma[keep]
        channel_medians = np.median(selected_pixels, axis=0)
        target = float(np.median(selected_luma))
        if target <= 1e-6 or float(np.min(channel_medians)) <= 1e-6:
            return BalanceCandidate("neutral_percentile", warning="unstable neutral sample")
        gains = normalize_rgb_gains(tuple(float(target / channel_medians[index]) for index in range(3)))
        gain_log = _safe_log_gain(gains)
        gain_norm = float(np.linalg.norm(gain_log))
        max_log = 0.90 if before_metrics.magnitude >= 0.04 else 0.62
        if gain_norm > max_log:
            gain_log = gain_log * (max_log / max(gain_norm, 1e-6))
            gains = normalize_rgb_gains(tuple(float(math.exp(value)) for value in gain_log))
        corrected = apply_rgb_gains_to_bgr(image, gains)
        after = estimate_cast_metrics(corrected)
        score, confidence, warning = _score_candidate(before_metrics, after, gains)
        return BalanceCandidate(
            algorithm="neutral_percentile",
            rgb_gains=gains,
            score=round(float(score), 4),
            confidence=round(float(confidence), 3),
            after_cast=round(float(after.magnitude), 5),
            warning=warning,
        )
    except Exception as exc:
        return BalanceCandidate("neutral_percentile", warning=str(exc))


def _opencv_candidate(image, before_metrics: CastMetrics, name: str, maker) -> BalanceCandidate:
    try:
        wb = maker()
        max_value = _max_value_for_dtype(image.dtype)
        if hasattr(wb, "setRangeMaxVal"):
            wb.setRangeMaxVal(int(max_value))
        if name == "grayworld" and hasattr(wb, "setSaturationThreshold"):
            wb.setSaturationThreshold(0.35)
        balanced = wb.balanceWhite(image)
        gains = _fit_rgb_gains(image, balanced)
        corrected = apply_rgb_gains_to_bgr(image, gains)
        after = estimate_cast_metrics(corrected)
        score, confidence, warning = _score_candidate(before_metrics, after, gains)
        return BalanceCandidate(
            algorithm=name,
            rgb_gains=gains,
            score=round(float(score), 4),
            confidence=round(float(confidence), 3),
            after_cast=round(float(after.magnitude), 5),
            warning=warning,
        )
    except Exception as exc:
        return BalanceCandidate(name, warning=str(exc))


def _score_candidate(
    before: CastMetrics,
    after: CastMetrics,
    rgb_gains: tuple[float, float, float],
) -> tuple[float, float, str]:
    before_mag = float(before.magnitude)
    after_mag = float(after.magnitude)
    if before_mag <= MIN_OBVIOUS_CAST_MAGNITUDE:
        return 0.0, 0.0, "no obvious cast"
    improvement = before_mag - after_mag
    improvement_ratio = improvement / max(before_mag, 1e-6)
    gain_log_norm = float(np.linalg.norm(_safe_log_gain(rgb_gains)))
    count_score = min(1.0, before.neutral_pixels / 2400.0)
    fraction_score = min(1.0, before.neutral_fraction / 0.08)
    saturation_score = max(0.0, min(1.0, 1.0 - before.median_saturation / 0.36))
    evidence = 0.35 * count_score + 0.35 * fraction_score + 0.30 * saturation_score
    max_safe_log = 0.88 if before_mag >= 0.04 else 0.62
    gain_safety = max(0.0, min(1.0, 1.0 - gain_log_norm / max_safe_log))
    improvement_score = max(0.0, min(1.0, improvement_ratio))
    score = 0.55 * improvement_score + 0.25 * evidence + 0.20 * gain_safety
    confidence = score * evidence * (0.35 + 0.65 * gain_safety)
    warning = ""
    if improvement <= 0:
        warning = "does not improve neutral cast"
    elif gain_log_norm > max_safe_log:
        warning = "large correction"
    elif confidence < 0.28:
        warning = "low confidence"
    return score, confidence, warning


def _fit_rgb_gains(source_bgr, target_bgr) -> tuple[float, float, float]:
    max_value = float(_max_value_for_dtype(source_bgr.dtype))
    source = source_bgr[:, :, :3].astype(np.float64)
    target = target_bgr[:, :, :3].astype(np.float64)
    bgr_gains: list[float] = []
    for channel in range(3):
        src = source[:, :, channel].reshape(-1)
        dst = target[:, :, channel].reshape(-1)
        mask = (
            (src > max_value * 0.015)
            & (src < max_value * 0.985)
            & (dst > 0)
            & (dst < max_value)
        )
        if int(np.count_nonzero(mask)) < 32:
            mask = src > max_value * 0.015
        if int(np.count_nonzero(mask)) < 32:
            bgr_gains.append(1.0)
            continue
        denominator = float(np.dot(src[mask], src[mask]))
        if denominator <= 1e-6:
            bgr_gains.append(1.0)
        else:
            bgr_gains.append(float(np.dot(src[mask], dst[mask]) / denominator))
    rgb_gains = (bgr_gains[2], bgr_gains[1], bgr_gains[0])
    return normalize_rgb_gains(rgb_gains)


def _neutral_mask(flat_rgb, luma, saturation, clean):
    if int(np.count_nonzero(clean)) < 16:
        return clean & False
    tone_saturation = saturation[clean]
    first_limit = min(0.30, max(0.055, float(np.quantile(tone_saturation, 0.35))))
    selected = clean & (saturation <= first_limit)
    if int(np.count_nonzero(selected)) >= 64:
        return selected
    relaxed_limit = min(0.42, max(first_limit, float(np.quantile(tone_saturation, 0.55))))
    selected = clean & (saturation <= relaxed_limit)
    if int(np.count_nonzero(selected)) >= 64:
        return selected
    channel_spread = np.max(np.abs(flat_rgb - luma[:, None]), axis=1)
    spread_limit = min(0.14, max(0.045, float(np.quantile(channel_spread[clean], 0.30))))
    return clean & (channel_spread <= spread_limit)


def _weighted_median_vector(values, weights):
    return np.asarray([_weighted_median(values[:, index], weights) for index in range(values.shape[1])])


def _main_gain_cluster_mask(values, weights):
    if values.shape[0] <= 2:
        return np.ones(values.shape[0], dtype=bool)
    radius = 0.20
    best_mask = np.ones(values.shape[0], dtype=bool)
    best_score = (-1.0, -1.0, -1.0)
    for index in range(values.shape[0]):
        distances = np.linalg.norm(values - values[index][None, :], axis=1)
        mask = distances <= radius
        count = int(np.count_nonzero(mask))
        if count < 2:
            continue
        cluster_weight = float(np.sum(weights[mask]))
        dispersion = float(np.average(distances[mask], weights=weights[mask]))
        score = (cluster_weight, float(count), -dispersion)
        if score > best_score:
            best_score = score
            best_mask = mask
    if int(np.count_nonzero(best_mask)) < max(2, int(math.ceil(values.shape[0] * 0.25))):
        return np.ones(values.shape[0], dtype=bool)
    return best_mask


def _weighted_median(values, weights):
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = float(sorted_weights.sum()) * 0.5
    return float(sorted_values[min(len(sorted_values) - 1, int(np.searchsorted(cumulative, cutoff)))])


def _safe_log_gain(gains: tuple[float, float, float]):
    values = np.asarray(normalize_rgb_gains(gains), dtype=np.float64)
    return np.log(np.clip(values, 1e-4, 1e4))


def _rgb_float01(image):
    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / float(_max_value_for_dtype(image.dtype))


def _rgb_luma(flat_rgb):
    weights = np.asarray(SRGB_LUMA_WEIGHTS_RGB, dtype=np.float32)
    return flat_rgb @ weights


def _remove_luma_component(vector):
    weights = np.asarray(SRGB_LUMA_WEIGHTS_RGB, dtype=np.float64)
    luma = float(np.asarray(vector, dtype=np.float64) @ weights)
    return np.asarray(vector, dtype=np.float64) - luma


def _ensure_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :4].copy()
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3].copy()
    raise ValueError("Unsupported image shape")


def _tiff_array_to_bgr(image, path: Path):
    array = np.asarray(image)
    if array.ndim >= 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] not in {3, 4}:
        array = array[0]
    if array.dtype.kind not in {"u", "i"}:
        raise ValueError(f"Unsupported TIFF dtype for {path}: {array.dtype}")
    if array.ndim == 2:
        return cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return np.ascontiguousarray(array[:, :, ::-1])
    if array.ndim == 3 and array.shape[2] == 4:
        return np.ascontiguousarray(array[:, :, [2, 1, 0, 3]])
    raise ValueError(f"Unsupported TIFF shape for {path}: {array.shape}")


def _bit_depth_for_image(image) -> int:
    if image.dtype == np.uint8:
        return 8
    if image.dtype == np.uint16:
        return 16
    if image.dtype == np.int16:
        return 16
    if image.dtype == np.float32:
        return 32
    return int(image.dtype.itemsize * 8)


def _max_value_for_dtype(dtype) -> int:
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return 255
    if dtype == np.uint16:
        return 65535
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return int(info.max)
    return 1


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _smoothstep(edge0: float, edge1: float, value):
    value = np.asarray(value, dtype=np.float32)
    if abs(edge1 - edge0) <= 1e-6:
        return np.where(value >= edge1, 1.0, 0.0).astype(np.float32)
    x = np.clip((value - np.float32(edge0)) / np.float32(edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _dominant_count_key(counts: dict[str, int], *, default: str = "") -> str:
    if not counts:
        return default
    return max(sorted(counts), key=lambda key: counts[key])


def _top_count_keys(counts: dict[str, int], *, limit: int) -> list[str]:
    return [
        key
        for key, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(0, int(limit))]
    ]


def _require_cv2_np() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python with xphoto support is required")
    if np is None:
        raise RuntimeError("numpy is required")


def _require_tifffile_np() -> None:
    if tifffile is None:
        raise RuntimeError("tifffile and imagecodecs are required for TIFF input")
    if np is None:
        raise RuntimeError("numpy is required")
