# HKN PoliTO Background Removal

A fully-local Streamlit application that batch-processes associate portraits for the **IEEE-HKN Mu Nu Chapter at Politecnico di Torino**. It detects the subject, removes the background with alpha-matted edges, and composites the cutout onto chapter-standard backgrounds — all on the operator's machine, with no cloud calls and no data leaving the laptop.

| | |
|---|---|
| Version | `1.5.1` |
| Python | 3.10 – 3.13 (probed in that order: 3.12 → 3.11 → 3.13 → 3.10) |
| License | See [LICENSE](../LICENSE) |
| Entry point | `./run.sh` (macOS / Linux) · `run.bat` (Windows) |

## What it does

1. **Detect** — Mediapipe (with OpenCV Haar fallback) locates the face in every input image.
2. **Matte** — `rembg` runs at full input resolution with alpha-matting tuned for translucent hair.
3. **Compose** — the cutout is anchored on a transparent canvas (face horizontally centered, subject bottom-flush) and composited onto each chapter background in the matching aspect ratio.

Outputs land beside the input as `{stem}_nobg.png` (transparent) and `{stem}_bg_{bgname}_{AR}.jpg` (composed).

## Wiki map

- [Installation](Installation) — prerequisites, launchers, manual venv path
- [Usage](Usage) — sidebar walkthrough and output layout
- [Architecture](Architecture) — files, modules, lazy-import strategy
- [Pipeline-Internals](Pipeline-Internals) — six stages, key functions, matting tuning
- [Configuration](Configuration) — dependencies, Streamlit theme, runtime knobs
- [Performance-Tuning](Performance-Tuning) — workers, RAM, CoreML, resume semantics
- [Troubleshooting](Troubleshooting) — common failures and recovery
- [Contributing](Contributing) — branch model, CI workflows, release flow
- [Changelog](Changelog) — curated highlights of the v1.x line

## Design tenets

- **Local-only** — no telemetry, no uploads, no SaaS dependency. GDPR-safe by construction.
- **Zero-setup operator path** — `run.sh` / `run.bat` provision the venv, pin dependencies, and launch the UI.
- **Quality over throughput** — full-resolution matting; bilateral pre-blend on the original RGB to damp JPEG artifacts before alpha compositing.
- **Resume on restart** — re-running the same job skips images whose outputs already exist.
