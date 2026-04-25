# Troubleshooting

## Model download is stuck or slow

`rembg` downloads ONNX weights into `~/.u2net/` on first use. If the download stalls:

1. Check your network — the files are pulled from the rembg release host (~175 MB for `isnet-general-use`, ~440 MB for `birefnet-portrait`).
2. Look for a partial file in `~/.u2net/` (e.g., a `.tmp` or zero-byte file). Delete it.
3. Restart the app and run a single-image test batch to retrigger the download.

Once a model is cached locally, subsequent runs are offline.

## Out-of-memory or swapping

Each worker peaks at 3 – 5 GB on a 24 MP input. The auto-tuned cap (`ram_gb // 8`, see [Performance-Tuning](Performance-Tuning)) is conservative but not infallible — if your machine has other heavy processes running (browsers, video apps), drop the **Workers** slider to 1.

Resume-on-restart is your friend here: stop the batch, lower the slider, restart it, and the already-processed images are skipped.

## Folder picker does nothing on macOS

`_pick_directory_macos()` at [`app.py:89`](../app.py) drives the native picker via AppleScript. The first time it runs, macOS asks for **Automation** permission for the terminal/app launching `streamlit`. Approve it in **System Settings → Privacy & Security → Automation**.

If AppleScript is blocked or unavailable, the code falls back to a Tkinter subprocess (`_pick_directory_tk_subprocess()` at [`app.py:134`](../app.py)). If that also fails, type the path into the text input directly.

## High-quality model freezes or hangs

The transformer-backed `birefnet-portrait` is intentionally **forced to CPU** via `COREML_BLACKLIST` ([`pipeline.py:212`](../pipeline.py)). This is because CoreML's graph compiler hangs on transformer attention kernels.

If you observe a freeze on the high-quality toggle:

- Confirm the blacklist is in effect — `get_active_providers()` should *not* include `CoreMLExecutionProvider` for that model.
- Allow extra time on the first run (CPU initialization is slow).
- For daily runs, leave high-quality off; reserve it for the final pass on the canonical photos.

## Streamlit cold start is slow

Heavy libraries (`cv2`, `mediapipe`, `rembg`, `pymatting`, `scikit-image`) are imported lazily — the first batch run pays a one-time warm-up cost (rendered as `st.status()` while `prewarm()` at [`pipeline.py:328`](../pipeline.py) loads the face detector and rembg session). This is normal. Subsequent runs reuse cached objects within the same process.

## "Already processed" but I changed the source image

Resume detection is by **filename only** — see [Performance-Tuning](Performance-Tuning#resume-on-restart-semantics). To force a re-run:

- Untick **Skip already-processed images** in the sidebar, or
- Delete the corresponding `{stem}_nobg.png` and `{stem}_bg_*.jpg` from the destination.

## Numba parallel runtime warning

If you see a Numba threading warning at startup, the probe in `_probe_threadsafe_numba_backend()` ([`pipeline.py:62`](../pipeline.py)) failed to find `tbb` or `omp` and fell back to the `workqueue` layer. The app still works — matting is just serialized via `_matting_lock`. Install `tbb` (`pip install tbb`) and restart for full parallelism.

## Ruff E402 fails CI on a PR

Imports must be at the top of the file, even when they are heavy. The lazy-import pattern in `app.py` is not lazy in the import-statement sense — every import line still sits at module scope; `app.py` simply doesn't import the heavy libs at all (they are imported inside the functions in `pipeline.py`). See commit `ba87057` for the precedent.
