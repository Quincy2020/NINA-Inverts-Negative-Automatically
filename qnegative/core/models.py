from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isclose


class ToolMode(str, Enum):
    PAN = "pan"
    MASK_PICKER = "mask_picker"
    WB_PICKER = "wb_picker"
    FILM_RECT = "film_rect"


class InvertMode(str, Enum):
    SIMPLE = "simple"
    DENSITY = "density"
    LOG_BOUNDS = "log_bounds"
    NEGPY_PRINT = "negpy_print"


class PrintCurveMode(str, Enum):
    LINEAR = "linear"
    SOFT = "soft"
    STANDARD = "standard"
    CONTRAST = "contrast"


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def label(self) -> str:
        return f"{self.width} x {self.height}"


@dataclass(frozen=True)
class ImagePoint:
    x: int
    y: int


@dataclass(frozen=True)
class ImageRect:
    x: int
    y: int
    width: int
    height: int
    angle: float = 0.0

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def label(self) -> str:
        angle = "" if isclose(self.angle, 0.0, abs_tol=0.05) else f", angle={self.angle:.1f}"
        return f"x={self.x}, y={self.y}, w={self.width}, h={self.height}{angle}"


@dataclass
class BalanceAxis:
    red_cyan: int = 0
    green_magenta: int = 0
    blue_yellow: int = 0


@dataclass
class TonalBalance(BalanceAxis):
    tonal_range: int = 50


@dataclass
class ColorBalanceParams:
    global_balance: BalanceAxis = field(default_factory=BalanceAxis)
    shadows: TonalBalance = field(default_factory=TonalBalance)
    midtones: TonalBalance = field(default_factory=TonalBalance)
    highlights: TonalBalance = field(default_factory=TonalBalance)


@dataclass
class DensityMatrixParams:
    m00: float = 1.0
    m01: float = -0.035
    m02: float = -0.015
    m10: float = -0.030
    m11: float = 1.0
    m12: float = -0.060
    m20: float = -0.010
    m21: float = -0.080
    m22: float = 1.0


@dataclass
class AdjustmentParams:
    invert_mode: str = InvertMode.NEGPY_PRINT.value
    print_curve: str = PrintCurveMode.STANDARD.value
    auto_wb: bool = True
    color_balance: ColorBalanceParams = field(default_factory=ColorBalanceParams)
    density_matrix: DensityMatrixParams = field(default_factory=DensityMatrixParams)
    exposure: int = 0
    highlights: int = 0
    shadows: int = 0
    contrast: int = 0
    saturation: int = 0
    camera_color_strength: int = 50
    soft_highlights: bool = False
    soft_shadows: bool = False
    black_point: int = 0
    mid_point: int = 50
    white_point: int = 100


@dataclass
class ImageProcessingState:
    mask_point: ImagePoint | None = None
    film_rect: ImageRect | None = None
    white_balance_point: ImagePoint | None = None
    adjustments: AdjustmentParams = field(default_factory=AdjustmentParams)
    negative_preview_active: bool = False
    auto_levels_pending: bool = True
