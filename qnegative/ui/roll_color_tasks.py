from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

from qnegative.core.models import ImageProcessingState
from qnegative.core.pipeline import (
    analysis_inset_crop,
    analysis_inset_from_adjustments,
    build_lab_print_color_stage,
    build_lab_print_display_stage,
    build_lab_print_levels_stage,
    build_lab_print_negative_stage,
    build_negative_base_preview,
    suggest_lab_print_luminance_levels,
)
from qnegative.core.preview import make_raw_preview
from qnegative.core.roll_color_adapter import analyze_positive_bgr_roll, positive_linear_to_bgr16


class RollColorAnalysisSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(int, object)
    failed = Signal(int, str)


@dataclass(frozen=True)
class RollColorAnalysisItem:
    path: Path
    state: ImageProcessingState


@dataclass(frozen=True)
class RollColorAnalysisOutput:
    result: dict
    frames_by_path: dict[str, dict]


class RollColorAnalysisTask(QRunnable):
    def __init__(
        self,
        *,
        job_id: int,
        items: list[RollColorAnalysisItem],
        max_size: int = 768,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.items = list(items)
        self.max_size = max_size
        self.signals = RollColorAnalysisSignals()

    def run(self) -> None:
        try:
            proxies = []
            total = max(1, len(self.items))
            for index, item in enumerate(self.items):
                self.signals.progress.emit(
                    round(index / total * 80),
                    f"Building positive proxy {index + 1}/{len(self.items)}",
                )
                proxies.append((item.path, self._positive_proxy(item)))

            self.signals.progress.emit(85, "Analyzing roll color")
            result, frames_by_path = analyze_positive_bgr_roll(proxies)
        except Exception as exc:
            self.signals.failed.emit(self.job_id, str(exc))
            return

        self.signals.finished.emit(
            self.job_id,
            RollColorAnalysisOutput(result=result, frames_by_path=frames_by_path),
        )

    def _positive_proxy(self, item: RollColorAnalysisItem):
        state = item.state
        if state.film_rect is None or not state.film_rect.is_valid():
            raise ValueError(f"No valid frame area for {item.path.name}")

        preview = make_raw_preview(item.path, max_size=self.max_size)
        adjustments = deepcopy(state.adjustments)
        adjustments.color_correction.enabled = False
        base = build_negative_base_preview(
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
        if state.auto_levels_pending:
            auto_levels = suggest_lab_print_luminance_levels(
                analysis_inset_crop(negative_stage.normalized_log, negative_stage.analysis_inset),
                adjustments,
                camera_to_srgb_matrix=negative_stage.camera_to_srgb_matrix,
            )
            adjustments.black_point = auto_levels["black_point"]
            adjustments.mid_point = auto_levels["mid_point"]
            adjustments.white_point = auto_levels["white_point"]
        else:
            auto_levels = {
                "black_point": adjustments.black_point,
                "mid_point": adjustments.mid_point,
                "white_point": adjustments.white_point,
            }

        levels_stage = build_lab_print_levels_stage(
            negative_stage,
            adjustments,
            auto_levels=auto_levels,
        )
        color_stage = build_lab_print_color_stage(
            levels_stage,
            adjustments,
            cmy_offsets=state.lab_print_cmy_offsets if adjustments.auto_wb else None,
        )
        result = build_lab_print_display_stage(color_stage, adjustments)
        return positive_linear_to_bgr16(result.processed_linear_rgb)
