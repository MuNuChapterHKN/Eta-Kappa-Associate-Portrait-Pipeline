# Configuration

## Dependencies

[`requirements.txt`](../requirements.txt) pins lower bounds only (the file's own header explains why):

| Package | Lower bound | Used for |
|---------|-------------|----------|
| `streamlit` | 1.38.0 | UI framework |
| `mediapipe` | 0.10.30 | Primary face detector |
| `rembg` | 2.0.59 | Background removal (ONNX models + alpha matting) |
| `onnxruntime` | 1.19.2 | Runs the rembg models. Picks CoreML on macOS where allowed. |
| `opencv-python-headless` | 4.10.0.84 | Image I/O, resize, bilateral filter, Haar fallback detector |
| `pillow` | 10.4.0 | RGBA canvas operations and JPEG encode |
| `numpy` | 1.26.4 | Underlying array math |
| `pymatting` | 1.1.12 | Alpha matting solver behind rembg |
| `scikit-image` | 0.24.0 | Color-space helpers in the matting refinement step |

For reproducible builds, the file's header recommends `pip freeze > requirements.lock.txt`.

## Streamlit configuration

[`.streamlit/config.toml`](../.streamlit/config.toml) hard-codes operator-friendly defaults:

```toml
[server]
headless = true            # never auto-open the browser; run.sh does it
runOnSave = false          # don't auto-rerun on file save
maxUploadSize = 500        # MB — large enough for high-res backgrounds
fileWatcherType = "none"   # silence the watcher; saves CPU

[browser]
gatherUsageStats = false   # no telemetry

[client]
toolbarMode = "minimal"    # keep the editorial look clean
showErrorDetails = true

[theme]
base = "dark"
primaryColor = "#D4A24C"           # HKN gold
backgroundColor = "#0C0E12"
secondaryBackgroundColor = "#14171D"
textColor = "#ECE4D2"
font = "serif"
```

Layered on top of this is [`assets/theme.css`](../assets/theme.css) (~25 KB), injected via `inject_css()` in [`app.py:70`](../app.py).

## Runtime knobs

Most behavior is exposed in the sidebar (see [Usage](Usage)). The values that are not user-facing live as module constants in [`pipeline.py`](../pipeline.py):

| Constant | Default | Line | Effect |
|----------|---------|------|--------|
| `MAX_SIDE` | 2048 | [91](../pipeline.py) | Longest-edge cap for working images |
| `ALPHA_MATTING_FG` | 250 | [100](../pipeline.py) | Trimap foreground threshold |
| `ALPHA_MATTING_BG` | 15 | [101](../pipeline.py) | Trimap background threshold |
| `ALPHA_MATTING_ERODE` | 30 | [102](../pipeline.py) | Trimap erosion (px) |
| `DECONTAM_BLEND_LO/HI` | 0.92 / 0.99 | [122 – 123](../pipeline.py) | Smoothstep blend bounds for hair-texture preservation |
| `ORIG_RGB_BILATERAL_*` | 7 / 18 / 5 | [133 – 135](../pipeline.py) | Bilateral filter on original RGB before α-blend |
| `FACE_TOP_RATIO` | 0.30 | [137](../pipeline.py) | Face vertical anchor (from top) |
| `DEFAULT_WORKERS` | 2 | [147](../pipeline.py) | Initial slider value |
| `SUBJECT_ALPHA_THRESHOLD` | 12 | [187](../pipeline.py) | Alpha below this is "background" for bbox |
| `DEFAULT_MODEL` | `isnet-general-use` | [196](../pipeline.py) | Fast model |
| `HIGH_QUALITY_MODEL` | `birefnet-portrait` | [197](../pipeline.py) | High-quality toggle |
| `COREML_BLACKLIST` | `{"birefnet-portrait", ...}` | [212](../pipeline.py) | Models forced to CPU |

Tweaking these requires editing the file and restarting; there is no `.env` or settings UI for them — they encode the chapter's quality contract.

## VERSION

[`VERSION`](../VERSION) is a one-line plain-text file (`1.5.1`) consumed by:

- The masthead in `app.py` (rendered as `v1.5.1` in the strip under the headline).
- The release workflow in [`.github/workflows/release.yml`](../.github/workflows/release.yml) (used to tag releases).

Bump it when shipping. See [Contributing](Contributing).

## Where weights live

`rembg` caches ONNX weights in `~/.u2net/` (one file per model name). They are never bundled in the repo or the venv.
