# Usage

The UI is organized into six numbered sidebar sections, executed top-down.

## 001 · Source

Path to a folder containing input portraits. Type the path or click **Browse** to open the native folder picker (AppleScript on macOS, Tkinter subprocess fallback elsewhere).

Supported extensions (case-insensitive): `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`. See `VALID_EXT` in [`pipeline.py:90`](../pipeline.py).

The source folder is scanned once at job start; images added mid-run are not picked up.

## 002 · Destination

Where to write the outputs. The folder is created if it does not exist. Outputs are placed flat (no subfolders): one transparent PNG plus one composed JPEG per background per input image.

## 003 · Backgrounds

Upload one or more background images (JPG / PNG). Each upload appears as a thumbnail chip; click the × on a chip to remove it, or use **Clear backgrounds** to reset.

You can run with multiple backgrounds in a single batch — every input image is composed against every uploaded background.

## 004 · Quality

Toggle between two models:

- **Off (default)** — `isnet-general-use`. Fast, very good for typical portraits.
- **On (high quality)** — `birefnet-portrait`. State-of-the-art portrait matting. Slower; CPU-only on macOS (CoreML is intentionally disabled for transformer models — see [Performance-Tuning](Performance-Tuning)).

## 005 · Parallelism & Resume

- **Workers** — slider from 1 to the auto-tuned cap (`MAX_RECOMMENDED_WORKERS`, derived from CPU and RAM; see [Performance-Tuning](Performance-Tuning)).
- **Skip already-processed images** — when checked (default), an input is skipped if its `_nobg.png` and all expected `_bg_*.jpg` files already exist in the destination. Filename match only — there is no checksum verification.

## 006 · Execute

Click **Begin processing** to start. Inputs are validated before the job kicks off:

- Source folder exists and contains at least one supported image.
- Destination folder is writable (created if missing).
- At least one background is uploaded.

## Output naming

For an input `martino.jpg` and a background named `aula_magna.jpg`:

```
martino_nobg.png                       transparent cutout (RGBA, lossless)
martino_bg_aula_magna_4x5.jpg          composed (RGB, JPEG q=95)
```

The `_AR` suffix encodes the background's aspect ratio (`ar_label()` in [`pipeline.py:669`](../pipeline.py)). Output canvas adapts per background — the subject is never cropped; transparent or background-fitted padding is added as needed.

## Status console

The right-hand panel shows live progress. Each line is prefixed with a glyph:

| Glyph | Meaning |
|-------|---------|
| `▸` | Info / start |
| `✓` | Image processed successfully — followed by `total · matting Xs · composite Xs` timing |
| `✕` | Image failed — error message attached |

The console caches the last 128 entries.
