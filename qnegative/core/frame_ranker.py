from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from qnegative.core.geometry import scale_rect
from qnegative.core.models import ImageRect, ImageSize
from qnegative.core.preview import resize_long_edge
from qnegative.tools.train_frame_ranker import (
    FEATURE_NAMES,
    extract_candidate_features,
    generate_global_candidates,
    prepare_image_features,
    prefilter_candidates,
)


FRAME_RANKER_MAX_EDGE = 384
FRAME_RANKER_GLOBAL_CANDIDATES = 1400
FRAME_RANKER_KEEP_CANDIDATES = 360

FORMAT_RATIOS = {
    "135": 1.50,
    "645": 4.0 / 3.0,
    "66": 1.00,
    "67": 7.0 / 6.0,
    "69": 1.50,
}


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
        source_rect = scale_rect(preview_rect, preview_size, source_size)
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


@lru_cache(maxsize=4)
def _load_frame_ranker_cached(path_text: str) -> tuple[object, str] | None:
    path = Path(path_text)
    if not path.exists():
        return None
    payload = joblib.load(path)
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
