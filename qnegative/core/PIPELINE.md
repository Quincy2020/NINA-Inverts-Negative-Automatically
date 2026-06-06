# NINA Lab Print Pipeline

This document records the current processing order so the UI, preview, and
export paths stay maintainable.

## Main Flow

1. RAW decode
   - Preview uses a downscaled RAW preview.
   - Export decodes full-resolution RAW.
   - Output is linear RGB plus camera metadata/matrices when available.

2. Base geometry
   - `build_negative_base_preview()`
   - Applies frame crop/warp.
   - Applies lens correction before inversion when enabled.
   - Produces the cropped film linear RGB used by later stages.

3. Negative stage
   - `build_lab_print_negative_stage()`
   - Uses camera white-balanced linear RGB when available.
   - Normalizes log bounds.
   - Builds `positive_control = 1 - normalized_log`.
   - Builds the analysis histogram from the inset area.

4. Levels stage
   - `build_lab_print_levels_stage()`
   - Applies automatic or manual black/mid/white points.
   - Converts back to `normalized_for_print` for the print response.

5. Color / print stage
   - `build_lab_print_color_stage()`
   - Estimates or reuses CMY auto white-balance offsets.
   - Applies `apply_log_hd_print_curve()`.
   - Exposure and contrast currently live inside this base print response:
     exposure changes print density, contrast changes print grade.
   - Applies camera color transform.
   - Applies log color separation.
   - Applies manual/global/tonal white-balance sliders.

6. Roll color correction
   - `apply_roll_color_to_linear_rgb()`
   - Runs after the base Lab Print color stage.
   - Runs before high/shadow tone modifier and saturation.
   - Treat this as roll-level color calibration, not per-image final tone.

7. Tone modifier
   - `apply_highlight_shadow_adjustments()`
   - Estimates a dynamic `mid_anchor` from roll-corrected luminance.
   - Uses a monotone Fritsch-Carlson LUT.
   - Shadows affect only black-to-mid.
   - Highlights affect only mid-to-white.
   - Black, mid, and white anchors are preserved.

8. Saturation
   - `apply_saturation_adjustment()`
   - Runs after tone shaping.

9. Output
   - Preview converts to 8-bit sRGB for display.
   - Export returns linear RGB to `export_tasks.py`, then encodes TIFF/JPEG/PNG.

## Preview And Export Consistency

Preview and export share the same Lab Print stages:

- Preview: `build_lab_print_display_stage()`
- Export: `build_lab_print_export_linear()`

Both call roll color correction, tone modifier, and saturation in the same
order. Export differs mainly by resolution and output encoding.

## Export Timing Label

The `Lab Print` export timing in `qnegative/ui/export_tasks.py` wraps the whole
`_process_export()` call. Export also reports sub-stage timings:

- `Lab negative`
- `Lab auto levels`
- `Lab levels`
- `Lab color print`
- `Lab roll color`
- `Lab tone modifier`
- `Lab saturation`
- `Lab Print` total

Use these sub-stage timings before guessing which part is responsible.
