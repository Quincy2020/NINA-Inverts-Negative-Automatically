from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnegative.core.models import AdjustmentParams, BalanceAxis, ImagePoint, ImageProcessingState
from qnegative.core.pipeline import (
    estimate_lab_print_auto_cmy_offsets,
    manual_printer_balance_offsets,
    suggest_printer_balance_from_log_sample,
)


@dataclass(frozen=True)
class WhiteBalancePickResult:
    point: ImagePoint
    printer_balance: BalanceAxis
    median_log: np.ndarray
    offset_delta: np.ndarray

    def status_text(self) -> str:
        sample_text = ", ".join(f"{value:.3f}" for value in self.median_log)
        offset_text = ", ".join(f"{value:+.4f}" for value in self.offset_delta)
        return (
            f"WB picker: x={self.point.x}, y={self.point.y}, "
            f"median log {sample_text}, printer delta {offset_text}"
        )


class WhiteBalanceController:
    def __init__(self) -> None:
        self.point: ImagePoint | None = None
        self.cmy_offsets: list[float] | None = None
        self.cmy_strength: int | None = None

    def clear_point(self) -> None:
        self.point = None

    def clear_cmy_offsets(self) -> None:
        self.cmy_offsets = None
        self.cmy_strength = None

    def clear_point_and_cmy(self) -> None:
        self.clear_point()
        self.clear_cmy_offsets()

    def restore(
        self,
        *,
        point: ImagePoint | None,
        cmy_offsets: list[float] | None,
        cmy_strength: int | None,
    ) -> None:
        self.point = point
        self.cmy_offsets = list(cmy_offsets) if cmy_offsets is not None else None
        self.cmy_strength = int(cmy_strength) if cmy_strength is not None else None

    def set_cmy_offsets(
        self,
        cmy_offsets: list[float] | np.ndarray | None,
        adjustments: AdjustmentParams,
    ) -> None:
        if not adjustments.auto_wb or cmy_offsets is None:
            self.clear_cmy_offsets()
            return
        values = np.asarray(cmy_offsets, dtype=np.float32).reshape(3)
        self.cmy_offsets = [float(value) for value in values]
        self.cmy_strength = int(adjustments.auto_cmy_strength)

    def current_cmy_offsets(self, adjustments: AdjustmentParams) -> np.ndarray | None:
        if not adjustments.auto_wb:
            return None
        if self.cmy_offsets is None:
            return None
        if self.cmy_strength != adjustments.auto_cmy_strength:
            return None
        return np.asarray(self.cmy_offsets, dtype=np.float32).reshape(3).copy()

    def base_cmy_offsets_for_picker(
        self,
        normalized_for_print: np.ndarray,
        adjustments: AdjustmentParams,
    ) -> np.ndarray:
        if not adjustments.auto_wb:
            return np.zeros(3, dtype=np.float32)

        manual = manual_printer_balance_offsets(adjustments.printer_balance)
        effective = self.current_cmy_offsets(adjustments)
        if effective is not None:
            return (effective - manual).astype(np.float32, copy=False)

        return estimate_lab_print_auto_cmy_offsets(
            normalized_for_print,
            strength=adjustments.auto_cmy_strength / 100.0,
        )

    def pick_printer_balance(
        self,
        *,
        point: ImagePoint,
        log_point: ImagePoint,
        normalized_for_print: np.ndarray,
        adjustments: AdjustmentParams,
    ) -> WhiteBalancePickResult:
        printer_balance, median_log, offset_delta = suggest_printer_balance_from_log_sample(
            normalized_for_print,
            log_point,
            base_cmy_offsets=self.base_cmy_offsets_for_picker(normalized_for_print, adjustments),
        )
        self.point = point
        return WhiteBalancePickResult(
            point=point,
            printer_balance=printer_balance,
            median_log=median_log,
            offset_delta=offset_delta,
        )

    @staticmethod
    def cmy_offsets_for_state(
        state: ImageProcessingState,
        cached_result,
    ) -> np.ndarray | None:
        if not state.adjustments.auto_wb:
            return None
        if (
            state.lab_print_cmy_offsets is not None
            and state.lab_print_cmy_strength == state.adjustments.auto_cmy_strength
        ):
            return np.asarray(state.lab_print_cmy_offsets, dtype=np.float32).reshape(3).copy()
        if cached_result is None:
            return None
        return np.asarray(cached_result.result.wb_gains, dtype=np.float32).reshape(3).copy()

    @staticmethod
    def status_overlay_text(adjustments: AdjustmentParams) -> str:
        axis = adjustments.printer_balance
        return (
            "Printer  R/C {red:+d}  G/M {green:+d}  B/Y {blue:+d}\n"
            "Exp {exposure:+d}   Gray {mid}"
        ).format(
            red=axis.red_cyan,
            green=axis.green_magenta,
            blue=axis.blue_yellow,
            exposure=adjustments.exposure,
            mid=adjustments.mid_point,
        )
