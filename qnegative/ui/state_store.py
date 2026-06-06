from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from qnegative.core.models import AdjustmentParams, ImageProcessingState, ImageRect, InvertMode


@dataclass(frozen=True)
class RestoredImageRuntime:
    mask_point: object | None
    film_rect: ImageRect | None
    white_balance_point: object | None
    lab_print_cmy_offsets: list[float] | None
    adjustments: AdjustmentParams
    negative_preview_active: bool
    auto_levels_pending: bool
    preview_flip_horizontal: bool
    preview_flip_vertical: bool
    preview_rotation_quarters: int
    restored: bool


def default_adjustments() -> AdjustmentParams:
    return AdjustmentParams(invert_mode=InvertMode.LAB_PRINT.value)


def lab_print_adjustments(adjustments: AdjustmentParams) -> AdjustmentParams:
    normalized = deepcopy(adjustments)
    normalized.invert_mode = InvertMode.LAB_PRINT.value
    return normalized


def restored_runtime_for_state(
    state: ImageProcessingState | None,
    *,
    fallback_adjustments: AdjustmentParams,
) -> RestoredImageRuntime:
    if state is None:
        return RestoredImageRuntime(
            mask_point=None,
            film_rect=None,
            white_balance_point=None,
            lab_print_cmy_offsets=None,
            adjustments=deepcopy(fallback_adjustments),
            negative_preview_active=False,
            auto_levels_pending=True,
            preview_flip_horizontal=False,
            preview_flip_vertical=False,
            preview_rotation_quarters=0,
            restored=False,
        )

    return RestoredImageRuntime(
        mask_point=state.mask_point,
        film_rect=state.film_rect,
        white_balance_point=state.white_balance_point,
        lab_print_cmy_offsets=(
            deepcopy(state.lab_print_cmy_offsets) if state.adjustments.auto_wb else None
        ),
        adjustments=lab_print_adjustments(state.adjustments),
        negative_preview_active=False,
        auto_levels_pending=state.auto_levels_pending,
        preview_flip_horizontal=state.preview_flip_horizontal,
        preview_flip_vertical=state.preview_flip_vertical,
        preview_rotation_quarters=state.preview_rotation_quarters % 4,
        restored=True,
    )


def should_restore_positive_preview(
    state: ImageProcessingState,
    *,
    manual_levels_present: bool,
) -> bool:
    return bool(
        state.negative_preview_active
        or (
            state.film_rect is not None
            and state.film_rect.is_valid()
            and manual_levels_present
        )
    )


def build_current_image_state(
    *,
    existing_state: ImageProcessingState | None,
    mask_point,
    film_rect,
    white_balance_point,
    adjustments: AdjustmentParams,
    lab_print_cmy_offsets: list[float] | None,
    tone_mid_anchor: float | None,
    has_positive_result: bool,
    manual_levels_present: bool,
    auto_levels_pending: bool,
    preview_flip_horizontal: bool,
    preview_flip_vertical: bool,
    preview_rotation_quarters: int,
) -> ImageProcessingState:
    return ImageProcessingState(
        mask_point=mask_point,
        film_rect=film_rect,
        white_balance_point=white_balance_point,
        adjustments=deepcopy(adjustments),
        lab_print_cmy_offsets=(
            deepcopy(lab_print_cmy_offsets) if adjustments.auto_wb else None
        ),
        tone_mid_anchor=tone_mid_anchor,
        roll_color_frame=deepcopy(existing_state.roll_color_frame) if existing_state else None,
        negative_preview_active=has_positive_result,
        auto_levels_pending=(
            False if has_positive_result or manual_levels_present else auto_levels_pending
        ),
        preview_flip_horizontal=preview_flip_horizontal,
        preview_flip_vertical=preview_flip_vertical,
        preview_rotation_quarters=preview_rotation_quarters % 4,
    )


def merge_stale_preview_result_state(
    existing_state: ImageProcessingState | None,
    output,
) -> ImageProcessingState:
    adjustments = (
        deepcopy(existing_state.adjustments)
        if existing_state is not None
        else (
            deepcopy(output.adjustments)
            if output.adjustments is not None
            else default_adjustments()
        )
    )
    return ImageProcessingState(
        mask_point=(
            output.mask_point
            if output.mask_point is not None
            else (existing_state.mask_point if existing_state else None)
        ),
        film_rect=(
            output.film_rect
            if output.film_rect is not None
            else (existing_state.film_rect if existing_state else None)
        ),
        white_balance_point=existing_state.white_balance_point if existing_state else None,
        adjustments=adjustments,
        lab_print_cmy_offsets=(
            existing_state.lab_print_cmy_offsets
            if existing_state is not None and existing_state.lab_print_cmy_offsets is not None
            else output.lab_print_cmy_offsets
        ),
        tone_mid_anchor=(
            existing_state.tone_mid_anchor
            if existing_state is not None and existing_state.tone_mid_anchor is not None
            else output.result.tone_mid_anchor
        ),
        roll_color_frame=(
            deepcopy(output.roll_color_frame)
            if output.roll_color_frame is not None
            else (deepcopy(existing_state.roll_color_frame) if existing_state else None)
        ),
        negative_preview_active=True,
        auto_levels_pending=existing_state.auto_levels_pending if existing_state is not None else False,
        preview_flip_horizontal=existing_state.preview_flip_horizontal if existing_state else False,
        preview_flip_vertical=existing_state.preview_flip_vertical if existing_state else False,
        preview_rotation_quarters=existing_state.preview_rotation_quarters if existing_state else 0,
    )


def state_from_preinvert_output(output: Any) -> ImageProcessingState:
    return ImageProcessingState(
        mask_point=None,
        film_rect=output.frame_rect,
        white_balance_point=None,
        adjustments=deepcopy(output.adjustments),
        lab_print_cmy_offsets=output.lab_print_cmy_offsets,
        tone_mid_anchor=output.result.tone_mid_anchor,
        roll_color_frame=None,
        negative_preview_active=True,
        auto_levels_pending=False,
    )
