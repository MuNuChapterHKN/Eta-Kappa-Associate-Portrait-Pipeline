# HKN PoliTO — Associate Portrait Pipeline

A local batch processor that takes raw associate photos, removes the background, and composites them onto chapter-standard backgrounds. Everything runs on your machine — no data leaves, no accounts, no cloud.

Built for the Mu Nu chapter of IEEE-HKN at Politecnico di Torino.

---

## What it does

When a new associate cohort comes in, someone has to process dozens of portrait photos: crop them consistently, strip the background, and drop each face onto the chapter background. This tool automates that in six deterministic steps:

1. **Detect** — Mediapipe locates the face in the image. OpenCV Haar cascade kicks in automatically if Mediapipe isn't available for the running Python version.
2. **Frame** — The image gets cropped to a square. For landscape shots the crop centers on the face horizontally. For portrait shots the face sits 30% from the top, which is standard for ID-style portraits.
3. **Isolate** — rembg strips the background (ISNet or BiRefNet depending on quality setting) with alpha matting enabled. The matting trimap is intentionally wide so the solver has enough room to produce continuous alpha values on frizzy and curly hair instead of a binary cutout.
4. **Archive** — A transparent PNG (`_nobg.png`) is saved. Useful for later reuse without reprocessing.
5. **Compose** — Each uploaded background gets cover-fit to the subject's size and alpha-composited underneath.
6. **Publish** — Final RGB exports saved as JPEG at q=95, 4:4:4 chroma.

---

## Quick start

```bash
git clone <repo>
cd HKN_BG_REMOVAL
./run.sh
```

The launcher script handles everything else: picks the best available Python (prefers 3.12, then 3.11, down to 3.10), creates a virtualenv, installs dependencies, and opens the Streamlit UI. Model weights (~175–970 MB depending on mode) download automatically on first run.

On subsequent runs it skips the install if `requirements.txt` hasn't changed (SHA-256 stamped).

---

## Using the UI

The sidebar is split into five labeled sections:

**001 · Source** — Click "Browse folder…" to pick the folder with raw portraits. On macOS the native Finder dialog opens via AppleScript. On other platforms a tkinter dialog runs in a subprocess (to avoid Streamlit's threading restrictions).

**002 · Destination** — Where processed files go. Created automatically if it doesn't exist.

**003 · Backgrounds** — Drop in one or more images (JPG or PNG). A preview chip shows a real thumbnail, resolution and file size for each. "Clear all" resets the uploader.

**004 · Quality** — Toggle between two modes:
- **Standard**: `isnet-general-use`, ~175 MB, fast. Uses CoreML on macOS for 3–6× acceleration.
- **High Quality**: `birefnet-portrait`, ~970 MB, slower. Runs on CPU (transformer ops cause CoreML graph compilation hangs, so it's hardcoded to CPU).

**005 · Execute** — "Begin processing" validates inputs, warms up models if needed (live progress), then runs the batch. The console shows per-image timing with a breakdown: matting time, composite time, and a running ETA.

---

## How the hair edges actually work

Getting clean hair edges on curly or frizzy subjects is the hardest part of this kind of pipeline. The approach here has a few layers:

**Supersampling.** Both ISNet and BiRefNet run at 1024² internally. When the output mask gets upsampled to 2048², you get stair-step artifacts on hair edges. The fix: upsample the input image before passing it to rembg (1.25× standard, 1.5× HQ), let the model run at a proportionally larger effective resolution, then Lanczos-downscale the RGBA output back to the original size. The downscale anti-aliases the stair-steps because neighboring pixels average together.

**Alpha matting.** rembg's `alpha_matting=True` runs pymatting's closed-form solver on the uncertain trimap region. The thresholds are tuned deliberately loose (FG=250, BG=15, erode=30) so the uncertain band is wide. This forces the solver to compute real continuous alpha values on hair wisps instead of inheriting the model's binary mask.

**Alpha gamma lift.** Very fine hair strands — thinner than ~2 pixels in model space — come out of the solver at 5–12% alpha. Lanczos downscaling then pushes them even lower. Applying `α^0.80` before the downscale boosts those values enough that they survive the averaging. Pixels at 10% alpha become 16%, at 5% become ~9%. The fully opaque body of the hair is unaffected since `1.0^0.80 = 1.0`.

**Color detail injection.** pymatting's foreground color estimator smooths locally, which causes hair wisps to come out as a flat uniform tone even when individual strands are correctly resolved as separate alpha values. The fix: take the decontaminated color (correct base tone, no old-background bleed) and add back high-frequency luminance detail from the original image using an unsharp mask (`detail = I − Gaussian(I, σ=5)`). The Gaussian removes the original background (uniform low-frequency content); what remains is local strand-level variation. That gets added back weighted by alpha so background pixels don't pick it up.

---

## CoreML acceleration

On macOS, onnxruntime uses `CoreMLExecutionProvider` as the primary backend, routing inference through the Apple Neural Engine and Metal GPU. For convolutional models (u2net, isnet-general-use) this typically gives 3–6× speedup.

BiRefNet and other transformer-based models skip CoreML (`COREML_BLACKLIST` in pipeline.py). The CoreML graph compiler can hang for several minutes — or indefinitely — trying to convert complex attention ops. CPU is slower but at least it finishes.

The warm-up log shows which providers are active: `Metal / ANE` or `CPU only`.

---

## Output files

For each input image `photo.jpg` with backgrounds `blue.png` and `white.jpg`:

```
output/
  photo_nobg.png           # RGBA, transparent background
  photo_bg_blue.jpg        # RGB, composited on blue
  photo_bg_white.jpg       # RGB, composited on white
```

Output filenames come from the stem of the source file and the stem of the background file.

---

## Tuning

All the quality knobs are constants at the top of `pipeline.py`:

| Constant | Default | What it does |
|---|---|---|
| `MAX_SIDE` | 2048 | Output resolution cap (px) |
| `SUPERSAMPLE_STANDARD` | 1.25 | Input upscale factor before rembg in standard mode |
| `SUPERSAMPLE_HIGH_QUALITY` | 1.5 | Input upscale factor in HQ mode |
| `MAX_WORK_SIDE` | 2560 | Hard ceiling on working resolution (prevents solver stalls) |
| `ALPHA_MATTING_FG` | 250 | Foreground threshold; higher = wider uncertain band |
| `ALPHA_MATTING_BG` | 15 | Background threshold; lower = wider uncertain band |
| `ALPHA_MATTING_ERODE` | 30 | Trimap erosion size (px at output resolution) |
| `ALPHA_LIFT_GAMMA` | 0.80 | Gamma applied to alpha before downscale; lower = stronger wisp lift |
| `ALPHA_SNAP_HIGH` | 0.992 | Alpha values above this snap to 1.0 |
| `PORTRAIT_FACE_TOP_RATIO` | 0.30 | Vertical face position in portrait crops (fraction from top) |

---

## File structure

```
HKN_BG_REMOVAL/
├── app.py              # Streamlit UI
├── pipeline.py         # Image processing pipeline
├── run.sh              # Zero-setup launcher
├── requirements.txt    # Python dependencies
├── assets/
│   ├── theme.css       # Custom UI styles
│   ├── hkn_logo_white.svg
│   └── hkn_logo_society.png
└── .streamlit/
    └── config.toml     # Streamlit server config
```

Model weights go in `~/.u2net/` (managed by rembg). `run.sh` creates a `venv/` folder locally; it's gitignored.

---

## Requirements

Python 3.10–3.13. 3.14 is untested; some wheels may not be available yet.

Core dependencies: `streamlit`, `rembg`, `onnxruntime`, `mediapipe`, `opencv-python-headless`, `pillow`, `numpy`, `pymatting`, `scikit-image`.

No GPU required. CoreML acceleration is optional and kicks in automatically on macOS when available.

---

*Mu Nu Chapter · IEEE-HKN · Politecnico di Torino*
