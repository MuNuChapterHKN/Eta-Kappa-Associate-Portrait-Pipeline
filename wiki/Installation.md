# Installation

## Prerequisites

- **Operating system:** macOS, Linux, or Windows.
- **Python:** any of 3.10, 3.11, 3.12, or 3.13. The launcher probes them in this order: `python3.12` → `python3.11` → `python3.13` → `python3.10`. Whichever is found first is used to build the venv.
- **Disk:** ~1 GB for the venv plus 175 MB – 970 MB for ONNX model weights downloaded on first run.
- **RAM:** 8 GB minimum; 16 GB+ recommended for batches of 24 MP photos. Each worker peaks at 3 – 5 GB.
- **GPU:** not required. On Apple Silicon, CoreML acceleration is enabled automatically for compatible models (3 – 6× speedup).

## Quick start

### macOS / Linux

```bash
./run.sh
```

### Windows

```bat
run.bat
```

The launcher will:

1. Locate a supported Python interpreter.
2. Create `./venv/` if missing.
3. Install `requirements.txt`. A SHA-256 stamp of the requirements file is recorded in the venv so re-launches skip pip when nothing changed.
4. Start Streamlit and open the UI in the default browser.

## Manual installation

If you prefer to manage the environment yourself:

```bash
python3.12 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The full dependency list is in [`requirements.txt`](../requirements.txt) — see [Configuration](Configuration) for what each package is used for.

## First-run model downloads

`rembg` downloads its ONNX models lazily into `~/.u2net/` the first time a model is requested:

| Model | Approx. size | Used when |
|-------|--------------|-----------|
| `isnet-general-use` | ~175 MB | Default — fast, good quality |
| `birefnet-portrait` | ~440 MB | High-quality toggle in the sidebar |

Subsequent runs reuse the cached weights. If a download is interrupted, delete the partial file in `~/.u2net/` and re-run.

## Verifying the install

After launch, the Streamlit UI shows the masthead, the sidebar with six numbered sections (`001 Source` through `006 Execute`), and a status console. If you see those, the install is healthy.
