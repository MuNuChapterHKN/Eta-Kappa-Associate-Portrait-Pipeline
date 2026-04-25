# Pipeline Internals

This page walks the per-image pipeline in [`pipeline.py`](../pipeline.py). All line numbers refer to that file.

## Six stages

| # | Stage | Function | Line | Output |
|---|-------|----------|------|--------|
| 1 | Detect | `detect_face_center()` | [412](../pipeline.py) | `(face_x, face_y, detected_bool)` |
| 2 | Scale | `smart_crop_to_ar()` | [459](../pipeline.py) | Lanczos-downscaled BGR with face anchored |
| 3 | Matte | `remove_background()` | [711](../pipeline.py) | RGBA cutout at full input resolution |
| 4 | Archive | `canvas_for_ar()` + save | [560](../pipeline.py) | Transparent canvas-anchored RGBA |
| 5 | Compose | `composite_on_background()` | [790](../pipeline.py) | Final RGB composite per background |
| 6 | Batch | `process_batch()` | [949](../pipeline.py) | Drives all of the above in a thread pool |

The per-image driver `_process_one_image()` at [857](../pipeline.py) chains stages 1 – 5 and produces an `ItemResult` ([808](../pipeline.py)) with timing breakdown for the console.

## Stage 1 — Detect

Mediapipe's short-range face detector runs first. If Mediapipe is unavailable or finds nothing, the function falls back to OpenCV's Haar cascade. If neither finds a face, the image is still processed but the face position defaults to the geometric center; the boolean third return value reports the actual detection state.

`get_face_detector()` at [284](../pipeline.py) memoizes the chosen detector for the process lifetime.

## Stage 2 — Scale

`smart_crop_to_ar()` downscales the image so its longest edge equals `MAX_SIDE = 2048` ([91](../pipeline.py)) using Lanczos. The face is anchored at `FACE_TOP_RATIO = 0.30` ([137](../pipeline.py)) — i.e., 30 % from the top — which is what produces the editorial framing seen across the chapter portraits. The aspect ratio of the *output* is determined per background, not per input.

## Stage 3 — Matte (the hot stage)

`remove_background()` runs `rembg` with alpha matting on the **full raw resolution image** (no pre-downscale). Three constants control the matting solver ([100 – 102](../pipeline.py)):

```python
ALPHA_MATTING_FG = 250    # foreground threshold (out of 255)
ALPHA_MATTING_BG = 15     # background threshold
ALPHA_MATTING_ERODE = 30  # px erosion of the trimap
```

`FG = 250` is intentionally aggressive: it widens the "uncertain" band that the matting solver must compute, which is what gives translucent hair wisps continuous alpha instead of binary on/off.

After matting, `_refine_alpha()` at [687](../pipeline.py) runs the **hair-texture preservation pass**. At high alpha values (opaque interior of the subject), the original RGB is preferred over `pymatting`'s decontaminated RGB, because decontam over-smooths solid hair. The blend uses a smoothstep gated by:

```python
DECONTAM_BLEND_LO = 0.92   # alpha below this -> decontam
DECONTAM_BLEND_HI = 0.99   # alpha above this -> original RGB
```

Before the blend, the original RGB is bilateral-filtered ([133 – 135](../pipeline.py)) to damp JPEG DCT block artifacts without softening hair-strand edges:

```python
ORIG_RGB_BILATERAL_D = 7
ORIG_RGB_BILATERAL_SIGMA_COLOR = 18
ORIG_RGB_BILATERAL_SIGMA_SPACE = 5
```

This combination is what shipped in commits c3eb111 (bilateral deblock), bf1a3a3 (decontam → original RGB at high α), and 748a426 (purity-first matting). See [Changelog](Changelog).

## Stage 4 — Archive

`canvas_for_ar()` builds a transparent RGBA canvas sized for the *current background's* aspect ratio and pastes the cutout with three rules:

1. **Face horizontally centered** — the canvas is shifted so the detected face X lands at canvas-center X.
2. **Subject bottom-flush** — the lowest non-transparent pixel of the cutout sits on the canvas's bottom edge. No padding below; half-bust portraits "stand on the floor".
3. **Subject never cropped** — if the canvas would clip the cutout, it is widened/heightened with transparent padding instead.

Wider target ARs add transparent padding on the sides; taller ARs pad above the subject only. `_subject_bbox()` at [538](../pipeline.py) finds the cutout's tight bounds using `SUBJECT_ALPHA_THRESHOLD = 12` ([187](../pipeline.py)).

The transparent result is also saved as `{stem}_nobg.png` for archival.

## Stage 5 — Compose

For each uploaded background, `fit_background()` ([775](../pipeline.py)) cover-fits the background to the canvas size and `composite_on_background()` ([790](../pipeline.py)) alpha-composites the cutout on top. Output is JPEG quality 95.

The aspect-ratio label that appears in the filename is computed by `ar_label()` at [669](../pipeline.py); it produces fractions like `4x5`, `3x2`, or `1x1` by reducing to a canonical form with a max denominator of 20.

## Stage 6 — Batch

`process_batch()` iterates input images through a `ThreadPoolExecutor`. See [Architecture](Architecture) and [Performance-Tuning](Performance-Tuning) for the threading model and worker-count derivation.

Resume detection runs *before* a worker is dispatched: `_all_outputs_present()` at [845](../pipeline.py) checks for the `_nobg.png` and every expected `_bg_*.jpg`. If all are present, the input is skipped and reported as such in the console.

## Models

| Constant | Value | Notes |
|----------|-------|-------|
| `DEFAULT_MODEL` ([196](../pipeline.py)) | `isnet-general-use` | Fast convolutional model. CoreML-accelerated on Apple Silicon. |
| `HIGH_QUALITY_MODEL` ([197](../pipeline.py)) | `birefnet-portrait` | Transformer model. Forced to CPU via `COREML_BLACKLIST` ([212](../pipeline.py)) because graph compilation hangs. |

`get_rembg_session()` at [257](../pipeline.py) caches the loaded model per name; switching the high-quality toggle re-warms.
