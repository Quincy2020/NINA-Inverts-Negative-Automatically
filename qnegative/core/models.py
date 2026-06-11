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
    LAB_PRINT = "lab_print"


class PrintCurveMode(str, Enum):
    LINEAR = "linear"
    FILMIC_HABLE = "filmic_hable"
    FILMIC_ACES = "filmic_aces"
    SOFT = "soft"
    STANDARD = "standard"
    CONTRAST = "contrast"
    CONTRAST_SHOULDER = "contrast_shoulder"


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
class LensCorrectionParams:
    enabled: bool = False
    mode: str = "off"
    strength: int = 0
    radius: int = 100
    center_x: int = 50
    center_y: int = 50
    smoothness: int = 200
    max_gain: int = 200
    flat_profile_path: str | None = None
    flat_strength: int = 100


@dataclass
class ColorCorrectionParams:
    enabled: bool = False
    roll_strength: int = 100
    frame_residual_strength: int = 80
    tone_balance_strength: int = 100
    protection_strength: int = 100
    exposure_match_strength: int = 0


@dataclass
class DustRemovalParams:
    enabled: bool = False
    model_id: str | None = None
    threshold: int = 20
    adaptive: bool = True
    texture_penalty: int = 10
    max_threshold: int = 75
    inpaint_radius: int = 5
    model_path: str | None = None


@dataclass
class DustMaskState:
    manual_add_mask_path: str | None = None
    manual_protect_mask_path: str | None = None
    mask_width: int = 0
    mask_height: int = 0
    auto_mask_path: str | None = None
    auto_mask_params_key: str | None = None


@dataclass
class PrintCurveParams:
    enabled: bool = False
    density: float = 1.0
    grade: float = 3.0
    highlight_bias: float = 0.12
    highlight_width: float = 0.55
    shadow_bias: float = 0.0
    shadow_width: float = 0.55


@dataclass
class AdjustmentParams:
    invert_mode: str = InvertMode.LAB_PRINT.value
    print_curve: str = PrintCurveMode.STANDARD.value
    print_curve_params: PrintCurveParams = field(default_factory=PrintCurveParams)
    auto_wb: bool = True
    auto_cmy_strength: int = 65
    printer_balance: BalanceAxis = field(default_factory=BalanceAxis)
    color_balance: ColorBalanceParams = field(default_factory=ColorBalanceParams)
    lens_correction: LensCorrectionParams = field(default_factory=LensCorrectionParams)
    color_correction: ColorCorrectionParams = field(default_factory=ColorCorrectionParams)
    dust_removal: DustRemovalParams = field(default_factory=DustRemovalParams)
    exposure: int = 0
    highlights: int = 0
    shadows: int = 0
    contrast: int = 0
    saturation: int = 0
    camera_color_strength: int = 0
    soft_highlights: bool = False
    soft_shadows: bool = False
    analysis_inset_percent: int = 5
    black_point: int = 0
    mid_point: int = 50
    white_point: int = 100


@dataclass
class ImageProcessingState:
    mask_point: ImagePoint | None = None
    film_rect: ImageRect | None = None
    white_balance_point: ImagePoint | None = None
    adjustments: AdjustmentParams = field(default_factory=AdjustmentParams)
    lab_print_log_floors: list[float] | None = None
    lab_print_log_ceils: list[float] | None = None
    lab_print_cmy_offsets: list[float] | None = None
    lab_print_cmy_strength: int | None = None
    tone_mid_anchor: float | None = None
    roll_color_frame: dict | None = None
    negative_preview_active: bool = False
    auto_levels_pending: bool = True
    preview_flip_horizontal: bool = False
    preview_flip_vertical: bool = False
    preview_rotation_quarters: int = 0
    dust_mask: DustMaskState = field(default_factory=DustMaskState)
