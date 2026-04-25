# Performance Tuning

## Worker-count derivation

`MAX_RECOMMENDED_WORKERS` is computed once at module load by `_recommended_worker_cap()` ([`pipeline.py:165`](../pipeline.py)):

```
MAX_RECOMMENDED_WORKERS = min(cpu_count // 4, ram_gb // 8, 4)
```

Worked examples:

| Machine | cpu // 4 | ram // 8 | Cap |
|---------|----------|----------|-----|
| MacBook Air M2, 8 cores, 16 GB | 2 | 2 | **2** |
| MacBook Pro M3 Pro, 12 cores, 32 GB | 3 | 4 | **3** |
| Mac Studio M2 Ultra, 24 cores, 64 GB | 6 | 8 | **4** (hard cap) |
| Linux workstation, 16 cores, 16 GB | 4 | 2 | **2** |

The hard cap of 4 reflects diminishing returns once the matting lock dominates (see below). The slider in `005 Parallelism & Resume` ranges from 1 to this cap.

`DEFAULT_WORKERS = 2` because in practice one worker is in GPU/CoreML inference while the other is doing CPU matting — beyond two, contention on the matting lock cuts into throughput.

## Memory budget

Each worker peaks at **3 – 5 GB** on a 24 MP input (the matting solver allocates several full-resolution alpha buffers). The RAM-aware cap (`ram_gb // 8`) is the safety margin: it leaves OS and Streamlit memory free, plus headroom for transient allocations.

If the machine starts swapping, lower the slider to 1 — sequential mode is rock-solid even on 16 GB. Resume-on-restart means you can stop the batch, lower the worker count, and resume without redoing finished work.

## Threading backends

`pymatting` uses Numba under the hood. Numba's parallel runtime is not thread-safe by default, which would corrupt results when multiple workers call `pymatting` concurrently. `_probe_threadsafe_numba_backend()` ([`pipeline.py:62`](../pipeline.py)) tries layers in this order:

1. `tbb` (Intel TBB) — fastest, requires the `tbb` package
2. `omp` (OpenMP) — ships with Numba on most platforms
3. `workqueue` (Numba's pure-Python fallback)

If a thread-safe layer is found, `_NUMBA_THREADSAFE_LAYER` is set and the matting calls run free in parallel. If not, the module-level `_matting_lock` serializes `rembg.remove()` across all workers — the rest of each pipeline (decode, face detect, canvas, composite, encode) still runs in parallel, so throughput is roughly 1.5× a single-worker run instead of 2 – 4×.

You can verify which layer was selected by reading `_NUMBA_THREADSAFE_LAYER` after import.

## CoreML acceleration (Apple Silicon)

`_preferred_providers()` at [`pipeline.py:221`](../pipeline.py) picks the ONNX Runtime execution providers in priority order. On macOS, CoreML is preferred unless the model is in `COREML_BLACKLIST` ([`pipeline.py:212`](../pipeline.py)):

```python
COREML_BLACKLIST = {
    "birefnet-portrait",
    # ... (transformer-backed models)
}
```

Why the blacklist exists: CoreML's graph compiler hangs (or takes minutes) when it tries to lower transformer attention kernels. Convolutional models like `isnet-general-use` and `u2net` compile in seconds and run **3 – 6× faster** on Apple Silicon.

On non-Apple platforms, the providers list collapses to `["CPUExecutionProvider"]`. Active providers are reported by `get_active_providers()` at [`pipeline.py:278`](../pipeline.py).

## Resume-on-restart semantics

When **Skip already-processed images** is checked (default), each input is tested by `_all_outputs_present()` at [`pipeline.py:845`](../pipeline.py) before being dispatched:

- The expected output set (`_nobg.png` plus one `_bg_<bg>_<AR>.jpg` per uploaded background) is computed by `_expected_outputs()` at [`pipeline.py:829`](../pipeline.py).
- If every expected file exists in the destination, the input is skipped.
- Match is by **filename only** — there is no checksum, no metadata, no mtime check. If you change the source image but keep the filename, you must clear the destination or untick the resume box.

This is intentional: chapter operators run the same batch many times across re-uploads, and full-checksum scans of large folders would dominate a re-run that only needs to add a handful of new images.

## Tuning checklist

When throughput is below expectations:

1. Confirm `_NUMBA_THREADSAFE_LAYER` is set to `tbb` or `omp` — if it's `None`, install `tbb` (`pip install tbb`) and restart.
2. Confirm CoreML is active for non-blacklisted models — `get_active_providers()` should include `CoreMLExecutionProvider` on macOS.
3. Watch RAM during a batch — if swap activity appears, lower the worker count by 1.
4. For the high-quality model, expect ~3× the per-image time of the default model. Use it for a final pass, not for daily runs.
