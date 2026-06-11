from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from qnegative.core.models import (
    AdjustmentParams,
    DustMaskState,
    ImageProcessingState,
    ImageRect,
    InvertMode,
)


@dataclass(frozen=True)
class RestoredImageRuntime:
    mask_point: object | None
    film_rect: ImageRect | None
    white_balance_point: object | None
    lab_print_log_floors: list[float] | None
    lab_print_log_ceils: list[float] | None
    lab_print_cmy_offsets: list[float] | None
    lab_print_cmy_strength: int | None
    adjustments: AdjustmentParams
    negative_preview_active: bool
    auto_levels_pending: bool
    preview_flip_horizontal: bool
    preview_flip_vertical: bool
    preview_rotation_quarters: int
    dust_mask: DustMaskState
    restored: bool


def default_adjustments() -> AdjustmentParams:
    return AdjustmentParams(invert_mode=InvertMode.LAB_PRINT.value)


def lab_print_adjustments(adjustments: AdjustmentParams) -> AdjustmentParams:
    normalized = deepcopy(adjustments)
    normalized.invert_mode = InvertMode.LAB_PRINT.value
    return normalized


def restored_runtime_for_state(
    state: ImageProcessingState | None,
) -> RestoredImageRuntime:
    if state is None:
        return RestoredImageRuntime(
            mask_point=None,
            film_rect=None,
            white_balance_point=None,
            lab_print_log_floors=None,
            lab_print_log_ceils=None,
            lab_print_cmy_offsets=None,
            lab_print_cmy_strength=None,
            adjustments=default_adjustments(),
            negative_preview_active=False,
            auto_levels_pending=True,
            preview_flip_horizontal=False,
            preview_flip_vertical=False,
            preview_rotation_quarters=0,
            dust_mask=DustMaskState(),
            restored=False,
        )

    cmy_offsets = state.lab_print_cmy_offsets if state.adjustments.auto_wb else None
    cmy_strength = state.lab_print_cmy_strength
    if cmy_strength != state.adjustments.auto_cmy_strength:
        cmy_offsets = None
        cmy_strength = None

    return RestoredImageRuntime(
        mask_point=state.mask_point,
        film_rect=state.film_rect,
        white_balance_point=state.white_balance_point,
        lab_print_log_floors=(
            deepcopy(state.lab_print_log_floors)
            if state.lab_print_log_floors is not None
            else None
        ),
        lab_print_log_ceils=(
            deepcopy(state.lab_print_log_ceils)
            if state.lab_print_log_ceils is not None
            else None
        ),
        lab_print_cmy_offsets=(
            deepcopy(cmy_offsets) if cmy_offsets is not None else None
        ),
        lab_print_cmy_strength=cmy_strength,
        adjustments=lab_print_adjustments(state.adjustments),
        negative_preview_active=False,
        auto_levels_pending=state.auto_levels_pending,
        preview_flip_horizontal=state.preview_flip_horizontal,
        preview_flip_vertical=state.preview_flip_vertical,
        preview_rotation_quarters=state.preview_rotation_quarters % 4,
        dust_mask=deepcopy(state.dust_mask),
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
    lab_print_log_floors: list[float] | None,
    lab_print_log_ceils: list[float] | None,
    tone_mid_anchor: float | None,
    has_positive_result: bool,
    manual_levels_present: bool,
    auto_levels_pending: bool,
    preview_flip_horizontal: bool,
    preview_flip_vertical: bool,
    preview_rotation_quarters: int,
    dust_mask: DustMaskState | None = None,
) -> ImageProcessingState:
    existing_positive = bool(existing_state is not None and existing_state.negative_preview_active)
    positive_active = bool(has_positive_result or existing_positive)
    return ImageProcessingState(
        mask_point=mask_point,
        film_rect=film_rect,
        white_balance_point=white_balance_point,
        adjustments=deepcopy(adjustments),
        lab_print_log_floors=(
            deepcopy(lab_print_log_floors) if lab_print_log_floors is not None else None
        ),
        lab_print_log_ceils=(
            deepcopy(lab_print_log_ceils) if lab_print_log_ceils is not None else None
        ),
        lab_print_cmy_offsets=(
            deepcopy(lab_print_cmy_offsets) if adjustments.auto_wb else None
        ),
        lab_print_cmy_strength=(
            adjustments.auto_cmy_strength
            if adjustments.auto_wb and lab_print_cmy_offsets is not None
            else None
        ),
        tone_mid_anchor=tone_mid_anchor,
        roll_color_frame=deepcopy(existing_state.roll_color_frame) if existing_state else None,
        negative_preview_active=positive_active,
        auto_levels_pending=(
            False if positive_active or manual_levels_present else auto_levels_pending
        ),
        preview_flip_horizontal=preview_flip_horizontal,
        preview_flip_vertical=preview_flip_vertical,
        preview_rotation_quarters=preview_rotation_quarters % 4,
        dust_mask=deepcopy(dust_mask) if dust_mask is not None else DustMaskState(),
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
    existing_cmy_current = (
        existing_state is not None
        and existing_state.lab_print_cmy_offsets is not None
        and existing_state.lab_print_cmy_strength == adjustments.auto_cmy_strength
    )
    output_cmy_current = (
        output.lab_print_cmy_offsets is not None
        and getattr(output, "lab_print_cmy_strength", None) == adjustments.auto_cmy_strength
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
        lab_print_log_floors=(
            deepcopy(output.lab_print_log_floors)
            if getattr(output, "lab_print_log_floors", None) is not None
            else (
                deepcopy(existing_state.lab_print_log_floors)
                if existing_state and existing_state.lab_print_log_floors is not None
                else None
            )
        ),
        lab_print_log_ceils=(
            deepcopy(output.lab_print_log_ceils)
            if getattr(output, "lab_print_log_ceils", None) is not None
            else (
                deepcopy(existing_state.lab_print_log_ceils)
                if existing_state and existing_state.lab_print_log_ceils is not None
                else None
            )
        ),
        lab_print_cmy_offsets=(
            existing_state.lab_print_cmy_offsets
            if existing_cmy_current
            else output.lab_print_cmy_offsets
            if output_cmy_current
            else None
        ),
        lab_print_cmy_strength=(
            adjustments.auto_cmy_strength
            if existing_cmy_current or output_cmy_current
            else None
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
        dust_mask=deepcopy(existing_state.dust_mask) if existing_state else DustMaskState(),
    )


def state_from_preinvert_output(output: Any) -> ImageProcessingState:
    return ImageProcessingState(
        mask_point=None,
        film_rect=output.frame_rect,
        white_balance_point=None,
        adjustments=deepcopy(output.adjustments),
        lab_print_log_floors=getattr(output, "lab_print_log_floors", None),
        lab_print_log_ceils=getattr(output, "lab_print_log_ceils", None),
        lab_print_cmy_offsets=output.lab_print_cmy_offsets,
        lab_print_cmy_strength=getattr(output, "lab_print_cmy_strength", None),
        tone_mid_anchor=output.result.tone_mid_anchor,
        roll_color_frame=None,
        negative_preview_active=True,
        auto_levels_pending=False,
        dust_mask=DustMaskState(),
    )
