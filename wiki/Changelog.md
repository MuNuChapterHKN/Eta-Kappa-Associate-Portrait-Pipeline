# Changelog

A curated, theme-grouped summary of the v1.x line. For the exhaustive log, see `git log` in the repository.

## v1.5.1 — current

- `ba87057` **fix:** move module-level imports to top of file (ruff E402 compliance).
- `b37ed01` **chore:** bump version to 1.5.1; standardize string formatting across `app.py`.

## v1.5.0 — parallel & resilient

Two large changes shipped together to make multi-hundred-image batches practical on operator laptops:

- `cbdc3e6` **Resume on restart, RAM-aware worker cap, no more OOM on big batches.**
  - `_recommended_worker_cap()` derives `MAX_RECOMMENDED_WORKERS` from CPU count and detected RAM.
  - `_all_outputs_present()` skips an input whose `_nobg.png` and all `_bg_*.jpg` outputs already exist.
- `9aac1fd` **Parallel batch processing with a thread-pool worker model.**
  - `process_batch()` switches from sequential to `ThreadPoolExecutor`.
  - `_probe_threadsafe_numba_backend()` selects `tbb` / `omp` / falls back to `_matting_lock` for thread-safe matting.

## v1.4.x — quality refinements

A focused arc on hair edges and JPEG-artifact handling:

- `c3eb111` **Bilateral-deblock the original RGB before the α-blend.**
  - `ORIG_RGB_BILATERAL_D = 7`, `SIGMA_COLOR = 18`, `SIGMA_SPACE = 5`.
  - Damps JPEG DCT block artifacts in the source without softening hair-strand edges.
- `bf1a3a3` **Replace decontam with original RGB at high alpha.**
  - `DECONTAM_BLEND_LO = 0.92`, `DECONTAM_BLEND_HI = 0.99`.
  - At opaque interior alpha values, prefer the original RGB; `pymatting`'s decontaminated RGB is reserved for the matting band.
- `748a426` **Purity-first matting** — alpha matting runs on the full raw resolution; no pre-downscale.
- `8bfb215` Preserve hair texture on dense regions.

## v1.3.x — output framing

- `8c46642` **Canvas-padded AR output** — subject is never cropped; transparent padding is added when the target aspect ratio would clip.
- `9b0888a` Output aspect ratio adapts to each background.

## Themes

| Theme | Commits |
|-------|---------|
| Quality of edges | `c3eb111`, `bf1a3a3`, `748a426`, `8bfb215` |
| Performance and resilience | `cbdc3e6`, `9aac1fd` |
| Output framing | `8c46642`, `9b0888a` |
| Hygiene | `ba87057`, `b37ed01` |

## How to read this file

This page is hand-curated, not auto-generated. When you ship a notable change, add a one-line entry under the matching theme (or open a new theme section) and reference the short commit SHA. Drop entries that have been superseded.
