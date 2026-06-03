# NINA

NINA is a desktop RAW negative conversion tool focused on camera-scanned film.

NINA stands for "NINA Inverts Negative Automatically". The current MVP is built with PySide6, rawpy, NumPy, and OpenCV.

## Run

```powershell
python -m qnegative.app
```

Open a RAW/TIFF from `File > Open RAW / TIFF...` or open a folder from `File > Open Folder...`.

## Build

Windows one-folder build:

```powershell
.\scripts\build_windows.ps1 -Clean
```

The executable is written to:

```text
dist\NINA\NINA.exe
```

This build uses PyInstaller and keeps Qt/rawpy dependencies next to the executable. The one-folder layout is preferred for now because it starts faster and is less fragile than a single-file bundle.

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

## Roll Sessions

NINA stores per-folder processing state in:

```text
.nina/roll_session.json
```

The session file records per-image selections and adjustments, including frame area, film base point, levels, white balance, color controls, preview orientation, and completion state. Preview images are not stored; they are rebuilt from the saved parameters when the folder is opened again.

Session entries are matched by filename, file size, and modification time, so replacing a RAW file will not silently reuse stale parameters. Roll sessions are auto-saved by default and can also be written manually from `File > Save Roll Session`.

## Shortcuts

Adjustment shortcuts use a coarse step of `20`. Hold `Shift` for a fine step of `5`.

| Shortcut | Action |
| --- | --- |
| `Left` / `Right` | Move to previous / next file in the filmstrip. |
| `Space` / `Enter` | Confirm the current image and move to the next file. |
| `Tab` | Toggle between `Origin` and `Preview`. |
| `Q` / `A` | Push global color balance toward yellow / blue. |
| `W` / `S` | Push global color balance toward magenta / green. |
| `E` / `D` | Push global color balance toward cyan / red. |
| `R` / `F` | Increase / decrease exposure. |

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
- Folder-local `.nina/roll_session.json` files are user/session data and should not be committed.
