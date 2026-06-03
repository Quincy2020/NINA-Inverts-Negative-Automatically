# QNegativeLab

QNegativeLab is a desktop RAW negative conversion tool focused on camera-scanned film. The current MVP is built with PySide6, rawpy, NumPy, and OpenCV.

## Run

```powershell
python -m qnegative.app
```

Open a RAW/image from `File > Open RAW / Image...` or open a folder from `File > Open Folder...`.

## Current MVP

- Folder sequence browsing with a bottom filmstrip.
- Origin/Preview tabs: the origin view keeps frame selection editable, while the preview view shows the cropped positive.
- Rotated frame selection with move/resize/rotate controls.
- Lab Print inversion mode as the default workflow.
- Auto frame detection MVP with a lightweight ranker path.
- Per-image cached adjustments and preview results.
- Histogram levels with black/mid/white controls.
- Auto CMY white balance plus manual white balance controls.
- Exposure, contrast, highlights, shadows, saturation, print curve, and analysis boundary controls.
- 16-bit TIFF export.

## Processing Notes

The default Lab Print path is:

```text
RAW linear
-> frame warp
-> normalized log signal
-> levels / auto levels
-> CMY auto white balance
-> H&D print curve
-> color separation
-> manual color balance
-> highlights / shadows / saturation
-> sRGB preview or 16-bit TIFF export
```

Preview uses a 1080px RAW preview for speed. Export uses full-resolution RAW data.

When a final preview cache exactly matches the current image, selection, and adjustments, export reuses the preview CMY auto white-balance offsets. If the cache does not match, export recomputes auto WB at full resolution.

## Performance

The Lab Print H&D curve defaults to a per-channel LUT engine. This keeps preview and export close to the direct reference curve while avoiding repeated full-image `exp`/`power` calculations.

Developer switch:

```text
Settings > Developer > Export Advanced > Print Curve Engine
```

Options:

- `LUT 8192` default
- `LUT 4096`
- `Direct Reference`

On a measured `6138 x 4079` export, the LUT path reduced total export time from about `16.6s` to about `12.4s`, with max reference error around `1e-7`.

## Developer Notes

- The experimental GPU color-adjustment shader is currently disabled for final color operations because the 8-bit linear texture path can quantize deep shadows. OpenGL is still used for display behavior.
- `Density` and `Simple` inversion modes are kept as developer options.
- Calibration datasets, generated labels, model artifacts, RAW files, positive references, and TIFF outputs should not be committed unless explicitly intended.

