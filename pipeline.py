"""
HKN PoliTO — Associate ID Photo Pipeline
Local, GDPR-safe batch processor.

Pipeline (per image):
    1. Face detection (Mediapipe, CPU)
    2. Smart square crop + resize (≤ 2048 px, LANCZOS)
    3. Background removal (rembg / u2net, alpha matting ON)
    4. Save <name>_nobg.png (RGBA)
    5. For each background: fit + alpha-composite
    6. Save <name>_bg_<bgname>.jpg (RGB, q=95)

NOTE: heavy dependencies (cv2, mediapipe, rembg, PIL) are imported lazily
inside the functions that need them so the Streamlit app can render
immediately on cold start and defer the multi-second import cost until the
user actually clicks "Begin processing".
"""

from __future__ import annotations

import io
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_SIDE = 2048

# Alpha-matting thresholds tuned for curly / frizzy hair edges.
# Higher FG + lower BG + wider erode = more pixels fall into the "unknown"
# band, which forces the matting solver to compute *continuous* alpha
# values for them (translucent hair wisps) instead of snapping to 0/1.
# Pushing FG up from 240 → 250 preserves wisp depth: pixels the model
# thinks are 95–99% foreground stay in the solver instead of being frozen
# to α=1, which flattens the silhouette into a cut-out look.
ALPHA_MATTING_FG = 250
ALPHA_MATTING_BG = 15
ALPHA_MATTING_ERODE = 30

# Alpha refinement — kept *very* gentle so faint wisps (α ≈ 0.05–0.15) that
# form the "halo" around the main hair strands survive to the final output.
# No bilateral filter at all (eats isolated thin strands) and no low snap
# (would zero the halo). Only a sub-pixel Gaussian and a near-one snap.
ALPHA_POLISH_SIGMA = 0.3
ALPHA_SNAP_HIGH = 0.992

# Gamma < 1 on α *lifts* low-alpha values: α ^ 0.80 takes 0.10 → 0.16
# (+60 %) and 0.05 → 0.087 (+74 %) while leaving 1.0 unchanged. This brings
# back the faint translucent halo of fine hair that the model+Lanczos
# downscale pipeline tends to attenuate.
ALPHA_LIFT_GAMMA = 0.80

# Supersampling factors — the matting model runs at 1024² internally.
# Processing the input at >1× and Lanczos-downsampling the resulting RGBA
# anti-aliases the stair-step artifacts from the model's mask upscaling.
# Capped by MAX_WORK_SIDE below so pymatting doesn't stall on huge images.
SUPERSAMPLE_STANDARD = 1.25  # modest slowdown, clean edges
SUPERSAMPLE_HIGH_QUALITY = 1.5  # stronger AA, noticeable slowdown

# Hard ceiling on the working resolution passed to rembg + pymatting.
# Above ~2560 the alpha-matting solver stalls (minutes per image) with
# no intermediate feedback, which looks to the user like a hang.
MAX_WORK_SIDE = 2560

PORTRAIT_FACE_TOP_RATIO = 0.30  # face at ~30% from top of square for portraits

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# u2net                 ≈ 170 MB · native 320×320 · fastest, softer hair
# u2net_human_seg       ≈ 170 MB · tuned for humans, similar resolution
# isnet-general-use     ≈ 175 MB · native 1024×1024 · DEFAULT · sharp hair edges
# birefnet-portrait     ≈ 440 MB · 1024+ · state-of-the-art for human portraits
DEFAULT_MODEL = "isnet-general-use"
HIGH_QUALITY_MODEL = "birefnet-portrait"

# ---------------------------------------------------------------------------
# Lazy singletons.
# ---------------------------------------------------------------------------
_rembg_sessions: dict = {}
_face_detector = None
_face_backend: Optional[str] = None  # "mediapipe" | "opencv"
_active_providers: Optional[List[str]] = None  # set by get_rembg_session


# Models that hang or take unreasonably long to compile under the CoreML
# provider. These are transformer-based graphs whose ops stress CoreML's
# graph converter — safer to run them on CPU than risk a multi-minute hang
# on first session creation. Convolutional models (u2net, isnet) are fine.
COREML_BLACKLIST = {
    "birefnet-portrait",
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-massive",
    "sam",
}


def _preferred_providers(model_name: str = DEFAULT_MODEL) -> List[str]:
    """
    Return the onnxruntime execution-provider list to use for rembg.

    On macOS (Apple Silicon or Intel with a GPU) we prefer CoreML, which
    dispatches to the Apple Neural Engine + Metal GPU and typically gives
    3–10× speedup vs pure CPU for u2net / isnet.

    CoreML gracefully falls back op-by-op to CPU when it encounters
    unsupported ops, so listing it first is *usually* safe. The exception
    is transformer-heavy graphs (the birefnet family, SAM): CoreML's graph
    compiler can spend minutes — or hang — converting them, so those
    models are force-routed to pure CPU via COREML_BLACKLIST.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    preferred: List[str] = []

    if (
        platform.system() == "Darwin"
        and "CoreMLExecutionProvider" in available
        and model_name not in COREML_BLACKLIST
    ):
        preferred.append("CoreMLExecutionProvider")

    # CUDA / DirectML left off by design: this app is targeted at macOS
    # portable installs with no system-wide GPU runtime requirement.

    preferred.append("CPUExecutionProvider")
    return preferred


def get_rembg_session(model_name: str = DEFAULT_MODEL):
    """Lazily create and cache a rembg session for ``model_name``."""
    global _active_providers
    if model_name not in _rembg_sessions:
        from rembg import new_session  # slow import

        providers = _preferred_providers(model_name)
        # Try with the preferred list first (CoreML + CPU on macOS). If
        # session creation fails — e.g. a rare model op is rejected by
        # CoreML at graph-build time instead of falling through — drop to
        # CPU-only so the user still gets a working pipeline.
        try:
            _rembg_sessions[model_name] = new_session(
                model_name, providers=providers
            )
            _active_providers = providers
        except Exception:
            cpu_only = ["CPUExecutionProvider"]
            _rembg_sessions[model_name] = new_session(
                model_name, providers=cpu_only
            )
            _active_providers = cpu_only
    return _rembg_sessions[model_name]


def get_active_providers() -> List[str]:
    """Return the provider list that was actually used for the last
    session creation (empty list before the first session)."""
    return list(_active_providers) if _active_providers else []


def get_face_detector() -> Tuple[object, str]:
    """
    Return (detector, backend) where backend is "mediapipe" or "opencv".

    Tries Mediapipe's full-range face detector first (best quality). If the
    submodule import fails — which happens on some wheel builds where
    ``mediapipe.solutions`` isn't auto-populated — falls back to OpenCV's
    Haar cascade bundled with opencv-python.
    """
    global _face_detector, _face_backend
    if _face_detector is not None:
        return _face_detector, _face_backend  # type: ignore[return-value]

    # --- attempt 1: mediapipe ----------------------------------------------
    try:
        # The explicit submodule import forces population of ``mp.solutions``
        # which is *not* guaranteed by ``import mediapipe`` alone on recent
        # wheels (notably the Python 3.14 builds).
        import mediapipe.solutions.face_detection as mp_face  # noqa: F401
        import mediapipe as mp

        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        _face_backend = "mediapipe"
        return _face_detector, _face_backend
    except Exception:
        pass

    # --- attempt 2: OpenCV Haar cascade (always available) ----------------
    import cv2
    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(str(cascade))
    if det.empty():
        raise RuntimeError(
            f"OpenCV face cascade not found at {cascade}. "
            "Install opencv-python (not just opencv-python-headless-core)."
        )
    _face_detector = det
    _face_backend = "opencv"
    return _face_detector, _face_backend


def prewarm(
    model_name: str = DEFAULT_MODEL,
    on_step: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Eagerly import heavy deps + build models so subsequent calls are fast.
    Designed to be wrapped in ``st.status()`` so the UI stays responsive.
    """
    def _step(msg: str) -> None:
        if on_step is not None:
            on_step(msg)

    _step("Loading OpenCV")
    import cv2  # noqa: F401

    _step("Loading NumPy & Pillow")
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401

    _step("Loading face detector")
    _, backend = get_face_detector()
    if backend == "mediapipe":
        _step("→ Mediapipe full-range detector ready")
    else:
        _step("→ Mediapipe unavailable · using OpenCV Haar cascade fallback")

    size_hint = "≈ 440 MB" if model_name == HIGH_QUALITY_MODEL else "≈ 175 MB"
    _step(f"Loading rembg · {model_name} (first run downloads {size_hint})")
    session = get_rembg_session(model_name)

    # Surface the actual rembg session class so the user can visually confirm
    # which backend is running (e.g. BiRefNetSessionPortrait vs IsnetGeneralUseSession).
    _step(f"→ rembg session class: {type(session).__name__}")

    # Surface the onnxruntime execution providers — "CoreMLExecutionProvider"
    # means the model is dispatching to the Apple Neural Engine + Metal GPU.
    # "CPUExecutionProvider" alone means we're on CPU fallback.
    providers = get_active_providers()
    if providers:
        tag = "Metal / ANE" if "CoreMLExecutionProvider" in providers else "CPU only"
        _step(f"→ onnxruntime providers: {' + '.join(providers)}  ({tag})")

    _step("Ready")


# ---------------------------------------------------------------------------
# Step 1 — face detection
# ---------------------------------------------------------------------------
def detect_face_center(image_bgr) -> Tuple[int, int, bool]:
    """Return (face_x, face_y, detected). Backend-agnostic."""
    import cv2
    h, w = image_bgr.shape[:2]
    detector, backend = get_face_detector()

    if backend == "mediapipe":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if results.detections:
            best = max(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height,
            )
            bbox = best.location_data.relative_bounding_box
            cx = int((bbox.xmin + bbox.width / 2) * w)
            cy = int((bbox.ymin + bbox.height / 2) * h)
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            return cx, cy, True
    else:  # opencv Haar cascade
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Equalize to help under uneven lighting (common for ID photos).
        gray = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(60, min(w, h) // 15), max(60, min(w, h) // 15)),
        )
        if len(faces) > 0:
            # pick the largest bounding box
            x, y, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
            cx = int(x + fw / 2)
            cy = int(y + fh / 2)
            return cx, cy, True

    return w // 2, h // 2, False


# ---------------------------------------------------------------------------
# Step 2 — smart square crop
# ---------------------------------------------------------------------------
def smart_square_crop(image_bgr, face_x: int, face_y: int):
    import cv2
    h, w = image_bgr.shape[:2]
    L = min(w, h)

    if w >= h:
        # Landscape (or square): crop width, keep height; center on face_x
        x0 = int(face_x - L / 2)
        x0 = max(0, min(w - L, x0))
        y0 = 0
    else:
        # Portrait: crop height, keep width; face at ~30% from top
        y0 = int(face_y - L * PORTRAIT_FACE_TOP_RATIO)
        y0 = max(0, min(h - L, y0))
        x0 = 0

    crop = image_bgr[y0 : y0 + L, x0 : x0 + L]

    if L > MAX_SIDE:
        crop = cv2.resize(
            crop, (MAX_SIDE, MAX_SIDE), interpolation=cv2.INTER_LANCZOS4
        )
    return crop


# ---------------------------------------------------------------------------
# Step 3 — background removal with alpha matting
# ---------------------------------------------------------------------------
def _refine_alpha(rgba):
    """
    Minimal alpha refinement — run *after* alpha matting.

        1. Sub-pixel Gaussian polish: light dithering of pixel-boundary
           stair-steps.
        2. Snap near-one to exact 1 (keeps the opaque body perfectly
           opaque without affecting any translucent pixel).

    Intentionally does **not** bilateral-filter or low-snap α: both erase
    the faint wisps (α in ~5-15 %) that surround the main hair strands and
    are what separate "cutout" from "photoreal".
    """
    import numpy as np
    import cv2
    from PIL import Image

    arr = np.array(rgba)  # H×W×4 uint8
    alpha = arr[..., 3]

    # 1. sub-pixel Gaussian polish
    k = max(3, int(ALPHA_POLISH_SIGMA * 6) | 1)
    alpha = cv2.GaussianBlur(
        alpha.astype(np.float32),
        (k, k),
        sigmaX=ALPHA_POLISH_SIGMA,
        sigmaY=ALPHA_POLISH_SIGMA,
    )

    # 2. snap only near-one (keep faint wisps alive)
    a = alpha / 255.0
    a = np.where(a > ALPHA_SNAP_HIGH, 1.0, a)
    arr[..., 3] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


def remove_background(
    image_bgr,
    model_name: str = DEFAULT_MODEL,
    supersample: float = SUPERSAMPLE_STANDARD,
):
    """
    Run rembg with supersampling + alpha matting + multi-stage α refinement.

    The model runs at its fixed internal resolution (1024² for
    isnet / birefnet, 320² for u2net), so the output mask is always a
    bilinear upsample of that internal result. Processing at ``supersample ×``
    the target resolution and Lanczos-downsampling the RGBA afterwards
    anti-aliases those upsampling steps.
    """
    import cv2
    from PIL import Image
    from rembg import remove

    orig_h, orig_w = image_bgr.shape[:2]

    # --- 1. supersample input, but cap so pymatting doesn't stall ---------
    eff_supersample = supersample
    if supersample > 1.0:
        longest = max(orig_w, orig_h)
        if longest * supersample > MAX_WORK_SIDE:
            eff_supersample = MAX_WORK_SIDE / longest

    if eff_supersample > 1.0:
        ss_w = int(round(orig_w * eff_supersample))
        ss_h = int(round(orig_h * eff_supersample))
        work_bgr = cv2.resize(
            image_bgr, (ss_w, ss_h), interpolation=cv2.INTER_LANCZOS4
        )
    else:
        work_bgr = image_bgr

    # --- 2. rembg + alpha matting -----------------------------------------
    # Scale erode_size with the *effective* factor so the matting band
    # covers the same relative region in input-space regardless of scale.
    erode_size = max(3, int(round(ALPHA_MATTING_ERODE * max(1.0, eff_supersample))))

    import numpy as np

    rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    session = get_rembg_session(model_name)
    out = remove(
        pil,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=ALPHA_MATTING_FG,
        alpha_matting_background_threshold=ALPHA_MATTING_BG,
        alpha_matting_erode_size=erode_size,
    ).convert("RGBA")

    # --- 3. high-pass detail injection ------------------------------------
    # rembg's pymatting foreground-ML estimator smooths the foreground
    # color locally → monotone wisps. We can't just reuse the raw original
    # RGB (it still carries the old background colour on semi-transparent
    # pixels) and manual per-pixel decontamination is numerically unstable
    # at low α.
    #
    # Clean approach: use pymatting's decontaminated colour as the *base*
    # (correct pure hair tone, free of bleach) and *add* the original's
    # high-frequency detail on top. The detail is isolated with an
    # unsharp-mask kernel:
    #
    #     detail = I_orig − GaussianBlur(I_orig, σ)
    #
    # The Gaussian blur kills everything low-frequency — including the
    # uniform old background, which is *by construction* the dominant
    # low-frequency content of the image. What survives is local variation:
    # individual hair strands, highlights, shadows. Adding that back on top
    # of the decontaminated base gives natural colour variation without
    # reintroducing background bleach.
    #
    # Weighted by α so pure-background pixels (α=0) get no detail added.
    out_arr = np.array(out, dtype=np.uint8)
    alpha_f = out_arr[..., 3:4].astype(np.float32) / 255.0

    orig_rgb_f = rgb.astype(np.float32)
    decontam_rgb_f = out_arr[..., :3].astype(np.float32)

    # Unsharp mask on the ORIGINAL: high-frequency detail, background-safe.
    blur = cv2.GaussianBlur(
        orig_rgb_f, ksize=(0, 0), sigmaX=5.0, sigmaY=5.0
    )
    detail = orig_rgb_f - blur  # ∈ [-~80, +~80] practically

    # Scale the detail by α so wisps with α≈0 don't get detail added (those
    # pixels are invisible anyway but avoiding the addition keeps the image
    # array numerically clean). Strength=0.9 reintroduces most of the
    # variation without amplifying noise.
    detail_strength = 0.9
    enriched = decontam_rgb_f + detail * alpha_f * detail_strength

    # --- 3b. α gamma-lift: rescue faint wisps -----------------------------
    # The model runs at 1024² internally; wisps thinner than ~2 model pixels
    # come out with very low α (~8-12 %). Lanczos downscale then averages
    # them toward 0. Applying α ← α^0.80 *before* the downscale boosts those
    # faint values so they survive averaging, restoring the halo of fine
    # hair visible in the source photo.
    alpha_ch = out_arr[..., 3].astype(np.float32) / 255.0
    alpha_ch = np.power(alpha_ch, ALPHA_LIFT_GAMMA)
    out_arr[..., 3] = np.clip(alpha_ch * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out_arr[..., :3] = np.clip(enriched, 0, 255).astype(np.uint8)
    out = Image.fromarray(out_arr, "RGBA")

    # --- 4. Lanczos downscale back to target size -------------------------
    # This is where the stair-step anti-aliasing actually happens: 4 pixels
    # of the supersampled RGBA collapse into 1 pixel of output, averaging
    # the jaggies. With the α gamma-lift above, faint wisps survive this.
    if out.size != (orig_w, orig_h):
        out = out.resize((orig_w, orig_h), Image.LANCZOS)

    # --- 5. multi-stage α refinement --------------------------------------
    return _refine_alpha(out)


# ---------------------------------------------------------------------------
# Step 5 — background compositing
# ---------------------------------------------------------------------------
def fit_background(bg_pil, target_size: Tuple[int, int]):
    """Cover-fit: resize keeping aspect, center-crop to target."""
    from PIL import Image
    tw, th = target_size
    bg = bg_pil.convert("RGB")
    bw, bh = bg.size
    scale = max(tw / bw, th / bh)
    nw, nh = max(tw, int(round(bw * scale))), max(th, int(round(bh * scale)))
    bg = bg.resize((nw, nh), Image.LANCZOS)
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    return bg.crop((x0, y0, x0 + tw, y0 + th))


def composite_on_background(subject_rgba, bg_rgb):
    from PIL import Image
    bg = fit_background(bg_rgb, subject_rgba.size).convert("RGBA")
    composed = Image.alpha_composite(bg, subject_rgba)
    return composed.convert("RGB")


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------
def list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXT
        ]
    )


@dataclass
class ItemResult:
    index: int
    total: int
    filename: str
    ok: bool
    detail: str
    produced: List[str]
    elapsed_s: float = 0.0
    # Fine-grained timing of the pipeline stages for this image. Keys:
    # "decode", "face", "crop", "matting", "save_png", "composite".
    stage_times: dict = field(default_factory=dict)


ProgressFn = Callable[[ItemResult], None]
StartFn = Callable[[int, int, str], None]


def process_batch(
    input_dir: Path,
    output_dir: Path,
    backgrounds: List[Tuple[str, bytes]],
    on_progress: Optional[ProgressFn] = None,
    model_name: str = DEFAULT_MODEL,
    supersample: float = SUPERSAMPLE_STANDARD,
    on_start: Optional[StartFn] = None,
) -> List[ItemResult]:
    """
    Execute the full 6-step pipeline for every image in ``input_dir``.
    Calls ``on_progress`` after each image (both success and failure).
    """
    import cv2
    import numpy as np
    from PIL import Image

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preload backgrounds once — reused for every subject.
    bg_pils: List[Tuple[str, "Image.Image"]] = []
    bg_errors: List[str] = []
    for name, data in backgrounds:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            bg_pils.append((Path(name).stem, img))
        except Exception as e:  # noqa: BLE001
            bg_errors.append(f"background '{name}' unreadable: {e}")

    images = list_images(input_dir)
    total = len(images)
    results: List[ItemResult] = []

    for idx, img_path in enumerate(images, start=1):
        produced: List[str] = []
        stage_times: dict = {}
        if on_start is not None:
            try:
                on_start(idx, total, img_path.name)
            except Exception:  # noqa: BLE001
                pass
        t_start = time.perf_counter()
        try:
            t0 = time.perf_counter()
            raw = cv2.imdecode(
                np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if raw is None:
                raise ValueError("cannot decode image (unsupported or corrupted)")
            stage_times["decode"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            fx, fy, detected = detect_face_center(raw)
            stage_times["face"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            squared = smart_square_crop(raw, fx, fy)
            stage_times["crop"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            subject = remove_background(
                squared, model_name=model_name, supersample=supersample
            )
            stage_times["matting"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            nobg_name = f"{img_path.stem}_nobg.png"
            subject.save(output_dir / nobg_name, format="PNG", optimize=True)
            produced.append(nobg_name)
            stage_times["save_png"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            for bg_name, bg_pil in bg_pils:
                composed = composite_on_background(subject, bg_pil)
                out_name = f"{img_path.stem}_bg_{bg_name}.jpg"
                composed.save(
                    output_dir / out_name,
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                    optimize=True,
                )
                produced.append(out_name)
            stage_times["composite"] = time.perf_counter() - t0

            detail = (
                "face detected"
                if detected
                else "no face found → centered fallback"
            )
            elapsed = time.perf_counter() - t_start
            res = ItemResult(
                idx, total, img_path.name, True, detail, produced,
                elapsed_s=elapsed, stage_times=stage_times,
            )
        except Exception as e:  # noqa: BLE001
            elapsed = time.perf_counter() - t_start
            res = ItemResult(
                idx, total, img_path.name, False, str(e), produced,
                elapsed_s=elapsed, stage_times=stage_times,
            )

        results.append(res)
        if on_progress is not None:
            on_progress(res)

    if bg_errors and on_progress is not None:
        for msg in bg_errors:
            on_progress(
                ItemResult(0, total, "(backgrounds)", False, msg, [])
            )

    return results
