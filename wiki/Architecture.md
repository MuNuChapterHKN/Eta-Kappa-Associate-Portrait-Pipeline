# Architecture

## High-level shape

```
+--------------------+        +---------------------------+
|   app.py (UI)      |        |   pipeline.py (engine)    |
|                    |        |                           |
|  Streamlit widgets | -----> |  process_batch()          |
|  Sidebar callbacks |        |   |                       |
|  Session state     |        |   +-> ThreadPoolExecutor  |
|  Status console    | <----- |        |                  |
|  Progress / ETA    |  cb    |        +-> _process_one_image()
+--------------------+        |              |            |
                              |              v            |
                              |   detect -> crop -> matte |
                              |        -> canvas -> compose
                              +---------------------------+
                                       |
                                       v
                           ~/.u2net/  (rembg model cache)
                           {output}/  (results)
```

`app.py` owns all UI and orchestration callbacks. `pipeline.py` is callable as a library: it never imports Streamlit and never touches `st.session_state`. The two are decoupled by a small set of callback types (`ProgressFn`, `StartFn`, `on_step`).

## File map

| Path | Purpose |
|------|---------|
| [`app.py`](../app.py) | Streamlit UI: sidebar, masthead, validation, console, callbacks. ~970 lines. |
| [`pipeline.py`](../pipeline.py) | Pure-Python image pipeline: detection, matting, canvas, composition, batch orchestration. ~1100 lines. |
| [`requirements.txt`](../requirements.txt) | Pinned-with-lower-bounds dependency list. |
| [`.streamlit/config.toml`](../.streamlit/config.toml) | Headless mode, dark theme tokens, max upload size. |
| [`assets/hkn_logo_white.svg`](../assets/hkn_logo_white.svg) | Chapter logo, embedded as data URI in the sidebar header. |
| [`assets/theme.css`](../assets/theme.css) | Custom CSS layered on top of the Streamlit dark theme (~25 KB). |
| [`run.sh`](../run.sh) / [`run.bat`](../run.bat) | Zero-setup launchers. |
| [`VERSION`](../VERSION) | Plain-text version string consumed by the masthead and release workflow. |
| [`ruff.toml`](../ruff.toml) | Lint config enforced in CI. |

## Lazy-import strategy

`app.py` imports only Streamlit and standard library at module load. Heavy libraries — `cv2`, `mediapipe`, `rembg`, `PIL`, `numpy`, `pymatting`, `scikit-image` — are imported **inside the functions that use them** (or behind `ensure_models_ready()` in `app.py`). This keeps cold-start time of the Streamlit page in the low hundreds of milliseconds, even though the cumulative import cost is several seconds.

The trade-off: the first batch run pays a one-time warm-up cost (rendered as `st.status()` while `prewarm()` in [`pipeline.py:328`](../pipeline.py) loads the face detector and rembg session). Subsequent runs reuse the cached objects.

## Session state keys

`_init_state()` in [`app.py:225`](../app.py) seeds the following keys:

| Key | Role |
|-----|------|
| `log` | List of console entries (capped at 128) |
| `running` | Set true while a batch is in flight; disables sidebar inputs |
| `input_path`, `output_path` | User-typed or picker-supplied folder paths |
| `high_quality` | Boolean — selects model |
| `max_workers` | Slider value |
| `skip_existing` | Resume toggle |
| `_bg_uploader_key` | Bumped to reset the file uploader widget |
| `models_ready`, `models_ready_for`, `_warmup_secs` | Warm-up gating and timing |

## Threading model

`process_batch()` in [`pipeline.py:949`](../pipeline.py) uses a `ThreadPoolExecutor` with `max_workers` from the slider. The matting step (`rembg.remove()` + `pymatting`) is serialized either by a thread-safe Numba backend (`tbb` or `omp`, probed in [`pipeline.py:62`](../pipeline.py)) or, when neither is available, by a module-level `_matting_lock`. Other stages (decode, face detect, canvas, composite, encode) run free in parallel.

This is why `DEFAULT_WORKERS = 2` ([`pipeline.py:147`](../pipeline.py)): one worker can be in CoreML / GPU inference while the other is in CPU matting, hiding most of the serialization cost.

## I/O layout

Inputs are read sequentially from disk. Each worker writes its outputs directly to the destination folder; there are no temp files. Resume detection is by filename: if the expected `_nobg.png` plus every `_bg_<bg>_<AR>.jpg` already exists, the input is skipped (`_all_outputs_present()` at [`pipeline.py:845`](../pipeline.py)).
