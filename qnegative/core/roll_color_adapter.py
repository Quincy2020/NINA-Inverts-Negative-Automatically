from __future__ import annotations

# Compatibility facade kept for older imports while the roll color code is
# split into analysis proxying and image-application modules.
from qnegative.core.roll_color_analysis_adapter import (
    analyze_positive_bgr_roll,
    positive_linear_to_bgr16,
    roll_color_result_summary,
)
from qnegative.core.roll_color_apply import (
    ROLL_COLOR_ENGINE,
    ROLL_COLOR_ENGINE_LEGACY_BGR16,
    ROLL_COLOR_ENGINE_LINEAR_COMPAT,
    apply_roll_color_to_linear_rgb,
    bgr16_to_positive_linear,
    roll_color_frame_key,
)

__all__ = [
    "ROLL_COLOR_ENGINE",
    "ROLL_COLOR_ENGINE_LEGACY_BGR16",
    "ROLL_COLOR_ENGINE_LINEAR_COMPAT",
    "analyze_positive_bgr_roll",
    "apply_roll_color_to_linear_rgb",
    "bgr16_to_positive_linear",
    "positive_linear_to_bgr16",
    "roll_color_frame_key",
    "roll_color_result_summary",
]
