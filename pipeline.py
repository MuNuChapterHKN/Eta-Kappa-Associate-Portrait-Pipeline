"""
HKN PoliTO — Associate ID Photo Pipeline
Local, GDPR-safe batch processor.

Pipeline (per image):
    1. Face detection (Mediapipe, CPU).
    2. Background removal (rembg, alpha matting ON) on the **full-resolution
       raw** — no pre-downscale, no supersampling. Whatever pymatting
       produces for each pixel is the colour we keep. The only α
       post-processing is a near-one snap so the opaque body is perfectly
       opaque; no detail injection, no unsharp mask, no gamma lift.
    3. For each output (archival 1:1 and per-background composite):
        · build a transparent canvas at the target aspect ratio that
          contains the whole subject (no crop), with the face horizontally
          centered, at FACE_TOP_RATIO from top when possible, and the
          subject's bottom flush with the canvas bottom
        · paste the subject RGBA at the computed offset
    4. Archival: save <name>_nobg.png (1:1).
    5. Composite: cover-fit the background under the canvas and save
       <name>_bg_<bgname>_<WxH>.jpg (RGB, q=95).

The subject silhouette is aspect-ratio-invariant: once the background is
gone, each output just parks the same silhouette on a canvas sized for its
target AR. Wider AR than the subject → transparent side padding; taller AR
→ transparent padding above (never below — half-bust subjects always stand
on the canvas floor).

NOTE: heavy dependencies (cv2, mediapipe, rembg, PIL) are imported lazily
inside the functions that need them so the Streamlit app can render
immediately on cold start and defer the multi-second import cost until the
user actually clicks "Begin processing".
"""

from __future__ import annotations

import io
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from fractions import Fraction
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

# Alpha refinement — kept *very* gentle. The only post-matting touch: snap
# α values above ALPHA_SNAP_HIGH to exactly 1.0 so the opaque body of the
# subject doesn't drift to 0.998 / 0.996 from rounding in rembg's pipeline.
# No Gaussian polish, no low-α snap, no gamma lift — we want pymatting's
# raw output to show through.
ALPHA_SNAP_HIGH = 0.992

# α-blend window between pymatting's decontaminated RGB and the original
# image RGB. pymatting's foreground estimator smooths locally, so dense
# opaque hair regions come out as flat patches of average tone — "paint
# bucket" artefact. At α ≥ HI the original image is already an excellent
# estimate of the pure foreground colour (background bleed ≤ 2 %), so we
# use it directly. Below LO the decontamination is still needed (semi-
# transparent hair with real bg bleed). Smoothstep in between.
#
# This only *replaces* decontam with original — no unsharp mask, no detail
# injection, no sharpening. The texture at α≈1 pixels is exactly the
# texture that was in the source photo.
DECONTAM_BLEND_LO = 0.92
DECONTAM_BLEND_HI = 0.99

# Bilateral filter applied to the original RGB *before* the α-blend, to
# damp JPEG compression blocks without touching strand edges.
#   cv2.bilateralFilter smooths pixels whose colours are close within a
#   spatial neighbourhood, so 8×8 DCT blocks (near-uniform colour inside)
#   get averaged together while real strand boundaries (large colour
#   differences) stay crisp. sigmaColor is kept tight so clump-level
#   variation survives — only JPEG quantisation noise (< ~15 units) gets
#   smoothed.
ORIG_RGB_BILATERAL_D = 7
ORIG_RGB_BILATERAL_SIGMA_COLOR = 18
ORIG_RGB_BILATERAL_SIGMA_SPACE = 5

FACE_TOP_RATIO = 0.30  # face vertical position in crop (from top), universal
PORTRAIT_FACE_TOP_RATIO = FACE_TOP_RATIO  # backward-compat alias

# Parallelism. Default to 2 workers: one in ANE/GPU inference while another
# is in pymatting on the CPU, typically 1.5–2× faster than sequential with
# minimal memory pressure. More helps only on beefy machines — each worker
# can peak at ~3–5 GB of scratch on a 24 MP input (raw BGR + pymatting
# float64 intermediates + decontam + bilateral + RGBA output), so 4 workers
# can legitimately touch ~20 GB. We cap conservatively to avoid the OS
# oom-killer silently nuking the process mid-batch.
DEFAULT_WORKERS = 2


def _detect_total_ram_gb() -> Optional[float]:
    """Return total installed RAM in GB, or None if unavailable.

    Uses POSIX sysconf when available (macOS + Linux). No psutil dependency.
    """
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024 ** 3)
    except Exception:
        pass
    return None


def _recommended_worker_cap() -> int:
    """Compute a safe upper bound for parallel workers on this machine.

    Heuristic: min(cpu_count // 4, ram_gb // 8, 4). Floors at 2. This keeps
    a 12-core / 32 GB Apple Silicon at 3 workers, a 16-core / 64 GB at 4,
    and a 10-core / 16 GB base-config Mac at 2. The hard ceiling of 4 is
    intentional — beyond 4 the ANE/GPU saturates and extra workers just
    eat RAM for no throughput gain.
    """
    cpu_cap = max(2, (os.cpu_count() or 2) // 4)
    ram_gb = _detect_total_ram_gb()
    ram_cap = max(2, int(ram_gb // 8)) if ram_gb else 3
    return max(2, min(cpu_cap, ram_cap, 4))


MAX_RECOMMENDED_WORKERS = _recommended_worker_cap()

# Alpha threshold (0-255) for deciding which pixels belong to the subject
# when computing the bounding box for canvas sizing. 12 ≈ 5% opacity: low
# enough to include faint hair wisps (the "halo") as part of the subject so
# the canvas doesn't clip them, high enough to ignore pure numerical noise
# left by the matting solver in nominally-transparent regions.
SUBJECT_ALPHA_THRESHOLD = 12

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
def smart_crop_to_ar(
    image_bgr,
    face_x: int,
    face_y: int,
    target_ar: float,
    face_top_ratio: float = FACE_TOP_RATIO,
    max_side: int = MAX_SIDE,
) -> Tuple["object", Tuple[int, int]]:
    """
    Crop ``image_bgr`` to aspect ratio ``target_ar`` (= width/height), with the
    face horizontally centered and placed at ``face_top_ratio`` from the top.

    Picks the largest possible crop that satisfies the AR and face-position
    constraints, then clamps the origin to the image bounds (so a face near
    an edge still produces a valid crop, just with the face slightly off the
    ideal 30 %).

    Longest side is capped to ``max_side`` via Lanczos downscale so downstream
    matting doesn't stall on 50 MP inputs.

    Returns ``(crop_bgr, (face_x_in_crop, face_y_in_crop))``.
    """
    import cv2
    h, w = image_bgr.shape[:2]
    fty = face_top_ratio

    # Largest (cw, ch) with cw/ch = target_ar and face at (cw/2, fty*ch):
    #   cx = face_x - cw/2    — requires cx >= 0  →  cw <= 2*face_x
    #                         — requires cx+cw <= w  →  cw <= 2*(w-face_x)
    #   cy = face_y - fty*ch  — requires cy >= 0  →  ch <= face_y / fty
    #                         — requires cy+ch <= h  →  ch <= (h-face_y)/(1-fty)
    ch_max = min(
        float(h),
        face_y / fty if fty > 0 else float("inf"),
        (h - face_y) / (1 - fty) if fty < 1 else float("inf"),
        (2 * face_x) / target_ar if target_ar > 0 else float("inf"),
        (2 * (w - face_x)) / target_ar if target_ar > 0 else float("inf"),
        w / target_ar if target_ar > 0 else float("inf"),
    )
    ch = max(1.0, ch_max)
    cw = ch * target_ar

    cx = face_x - cw / 2
    cy = face_y - fty * ch

    # Clamp defensively — upstream math should already keep us in bounds, but
    # rounding can push one pixel over.
    cx = max(0.0, min(float(w) - cw, cx))
    cy = max(0.0, min(float(h) - ch, cy))

    x0, y0 = int(round(cx)), int(round(cy))
    cw_i, ch_i = int(round(cw)), int(round(ch))
    x1, y1 = min(w, x0 + cw_i), min(h, y0 + ch_i)
    crop = image_bgr[y0:y1, x0:x1]

    # Downscale if the longest side exceeds max_side.
    out_fx = face_x - x0
    out_fy = face_y - y0
    crop_h, crop_w = crop.shape[:2]
    longest = max(crop_w, crop_h)
    if longest > max_side:
        scale = max_side / longest
        new_w = int(round(crop_w * scale))
        new_h = int(round(crop_h * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        out_fx = int(round(out_fx * scale))
        out_fy = int(round(out_fy * scale))

    return crop, (int(out_fx), int(out_fy))


def smart_square_crop(image_bgr, face_x: int, face_y: int):
    """Legacy 1:1 crop, kept for backward compatibility. Prefer
    ``smart_crop_to_ar`` which supports arbitrary aspect ratios."""
    crop, _ = smart_crop_to_ar(image_bgr, face_x, face_y, 1.0)
    return crop


def _subject_bbox(subject_rgba, alpha_threshold: int = SUBJECT_ALPHA_THRESHOLD):
    """
    Return ``(x0, y0, x1, y1)`` bounding the pixels whose α exceeds
    ``alpha_threshold``. Falls back to the full image extent if nothing
    passes the threshold (shouldn't happen in practice — matting always
    leaves some opaque content for a real portrait).
    """
    import numpy as np
    arr = np.asarray(subject_rgba)
    if arr.ndim != 3 or arr.shape[2] < 4:
        h, w = arr.shape[:2]
        return 0, 0, w, h
    alpha = arr[..., 3]
    mask = alpha > alpha_threshold
    if not mask.any():
        h, w = alpha.shape
        return 0, 0, w, h
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def canvas_for_ar(
    subject_rgba,
    face_x: int,
    face_y: int,
    target_ar: float,
    face_top_ratio: float = FACE_TOP_RATIO,
    max_side: int = MAX_SIDE,
):
    """
    Build a transparent canvas at ``target_ar`` containing the whole subject
    (no pixels cropped), with these placement rules:

      · face horizontally centered: ``face_x_canvas = CW / 2``
      · face at ``face_top_ratio`` from top when possible (30 % default)
      · **bottom-anchored**: the subject's bottom edge is the canvas's
        bottom edge — never pad below. Associates are framed as half-busts
        with crossed arms, so adding space beneath them would look wrong.

    Algorithm
    ---------
    Let ``above = face_y − subject_top`` and ``below = subject_bottom −
    face_y`` (both in pixels). With the bottom anchored:

        face_in_canvas_y = CH − below

    Placing the face at ``face_top_ratio · CH`` fixes ``CH`` at
    ``below / (1 − face_top_ratio)``. If the subject's above-face portion
    is taller than the resulting above-face slack, grow ``CH`` to fit the
    whole subject (``above + below``) and accept a face slightly higher
    than the ideal 30 %. The extra height becomes transparent padding
    above the head.

    Horizontally, ``CW ≥ 2 · max(left, right)`` so the face stays
    centered without clipping either side. Then lock the AR:

        if CH · target_ar ≥ CW_min   →   CW = CH · target_ar
        else                         →   CW = CW_min, grow CH to CW/target_ar

    Growing CH in the AR-lock branch only adds more transparent padding
    *above* the head — the bottom anchor is preserved.

    Finally the canvas is Lanczos-downscaled to ``max_side`` on its longest
    edge so the archival / composite outputs stay bounded regardless of how
    much padding we added.
    """
    from PIL import Image
    sx0, sy0, sx1, sy1 = _subject_bbox(subject_rgba)
    sw = max(1, sx1 - sx0)
    sh = max(1, sy1 - sy0)

    # Clamp face coords to the subject bbox. If detection puts the face
    # slightly outside the silhouette (e.g. on a hair wisp that fell below
    # the α threshold), snap it to the nearest subject pixel so the math
    # stays sane.
    fx = max(sx0, min(sx1 - 1, int(face_x)))
    fy = max(sy0, min(sy1 - 1, int(face_y)))

    above = fy - sy0
    below = sy1 - fy
    left = fx - sx0
    right = sx1 - fx

    fty = face_top_ratio
    # CH from the face-at-30 % rule (bottom anchored) and from the
    # no-crop-subject rule. Take the larger.
    ch_from_face = below / (1 - fty) if fty < 1 else float("inf")
    ch = max(ch_from_face, float(above + below))

    # CW must fit the wider of the two face-to-edge distances (doubled,
    # because the face is horizontally centered).
    cw_min = 2.0 * max(left, right)

    # Lock aspect ratio. If the AR-implied CW is smaller than cw_min, grow
    # CW and re-grow CH so the AR holds — the extra CH becomes transparent
    # padding above the head (bottom stays anchored).
    cw = ch * target_ar if target_ar > 0 else cw_min
    if cw < cw_min:
        cw = cw_min
        ch = cw / target_ar if target_ar > 0 else ch

    CW = max(1, int(round(cw)))
    CH = max(1, int(round(ch)))

    # Paste position. Bottom-anchored vertically; face-centered horizontally.
    sub_top = CH - sh
    sub_left = int(round(CW / 2.0 - left))

    # Clamp in case of rounding (rare; sub_top/sub_left are already ≥ 0 by
    # construction, but rounding can push them by one pixel).
    sub_top = max(0, min(CH - sh, sub_top))
    sub_left = max(0, min(CW - sw, sub_left))

    subject_crop = subject_rgba.crop((sx0, sy0, sx1, sy1))
    canvas = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    canvas.paste(subject_crop, (sub_left, sub_top), subject_crop)

    # Cap final size — padding can push the canvas well above the matting
    # resolution when the target AR differs a lot from the subject's own AR.
    longest = max(CW, CH)
    if longest > max_side:
        scale = max_side / longest
        new_w = max(1, int(round(CW * scale)))
        new_h = max(1, int(round(CH * scale)))
        canvas = canvas.resize((new_w, new_h), Image.LANCZOS)

    return canvas


def ar_label(width: int, height: int, max_denom: int = 20) -> str:
    """
    Produce a human-readable aspect-ratio tag like ``"16x9"`` or ``"3x2"``.

    Snaps to the closest fraction with denominator ≤ ``max_denom`` so odd
    sensor sizes (3024×4032, 1920×1081) collapse to the canonical tag
    (``3x4``, ``16x9``) instead of leaking raw pixel dimensions into the
    filename.
    """
    if width <= 0 or height <= 0:
        return "na"
    frac = Fraction(width, height).limit_denominator(max_denom)
    return f"{frac.numerator}x{frac.denominator}"


# ---------------------------------------------------------------------------
# Step 3 — background removal with alpha matting
# ---------------------------------------------------------------------------
def _refine_alpha(rgba):
    """
    Minimal alpha refinement — run *after* alpha matting.

        Snap α values above ALPHA_SNAP_HIGH to exactly 1.0 so the opaque
        body is perfectly opaque (without that, rembg's pipeline leaves
        α ≈ 0.996 on the body, which can subtly tint it against bright
        backgrounds during compositing).

    Intentionally does **not** touch anything else: no Gaussian polish,
    no bilateral filter, no low-α snap. Whatever pymatting computed for a
    pixel is what reaches the output — "purity first" is the design goal
    for this stage.
    """
    import numpy as np
    from PIL import Image

    arr = np.array(rgba)  # H×W×4 uint8
    a = arr[..., 3].astype(np.float32) / 255.0
    a = np.where(a > ALPHA_SNAP_HIGH, 1.0, a)
    arr[..., 3] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def remove_background(
    image_bgr,
    model_name: str = DEFAULT_MODEL,
):
    """
    Run rembg + alpha matting on the **full-resolution** input. No
    supersampling, no downscaling, no unsharp mask, no gamma lift.

    One targeted fix over pure pymatting output: α-weighted blend between
    pymatting's decontaminated RGB and the original image RGB. At α ≳ 0.99
    we use the original directly (its bg bleed is negligible), which
    eliminates the "paint bucket" flat-colour artefact that the decontam
    estimator produces in dense opaque hair. Below α ≈ 0.92 we keep the
    decontaminated colour (real bg bleed needs removing). Smoothstep in
    between keeps the transition invisible.
    """
    import cv2
    import numpy as np
    from PIL import Image
    from rembg import remove

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    session = get_rembg_session(model_name)
    out = remove(
        pil,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=ALPHA_MATTING_FG,
        alpha_matting_background_threshold=ALPHA_MATTING_BG,
        alpha_matting_erode_size=ALPHA_MATTING_ERODE,
    ).convert("RGBA")

    # Bilateral-smooth the original RGB to damp JPEG quantisation blocks
    # without touching strand edges. This is run on the *original* image
    # (not on the decontam output) because we're about to blend toward it
    # at high α — we want its JPEG artefacts gone before they reach the
    # composite.
    orig_smoothed = cv2.bilateralFilter(
        rgb,
        d=ORIG_RGB_BILATERAL_D,
        sigmaColor=ORIG_RGB_BILATERAL_SIGMA_COLOR,
        sigmaSpace=ORIG_RGB_BILATERAL_SIGMA_SPACE,
    )

    # α-smoothstep blend: decontam → (bilateral-smoothed) original as α rises.
    out_arr = np.array(out, dtype=np.uint8)
    alpha_f = out_arr[..., 3:4].astype(np.float32) / 255.0
    orig_rgb_f = orig_smoothed.astype(np.float32)
    decontam_rgb_f = out_arr[..., :3].astype(np.float32)

    lo, hi = DECONTAM_BLEND_LO, DECONTAM_BLEND_HI
    t = np.clip((alpha_f - lo) / (hi - lo), 0.0, 1.0)
    blend_w = t * t * (3.0 - 2.0 * t)  # smoothstep → 0..1
    blended = decontam_rgb_f * (1.0 - blend_w) + orig_rgb_f * blend_w
    out_arr[..., :3] = np.clip(blended, 0, 255).astype(np.uint8)

    return _refine_alpha(Image.fromarray(out_arr, "RGBA"))


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
    # True when the item was short-circuited because the expected output
    # files already existed on disk (resume path). ok is also True in that
    # case and ``produced`` carries the *pre-existing* output filenames.
    skipped: bool = False


ProgressFn = Callable[[ItemResult], None]
StartFn = Callable[[int, int, str], None]


def _expected_outputs(
    img_path: Path,
    bg_items: List[Tuple[str, "object", float, str]],
) -> List[str]:
    """Return the exact filenames ``_process_one_image`` would emit for
    this input, given the current background set.

    Used by the resume path: if every one of these files is already on
    disk in the output directory, we skip the image entirely.
    """
    names = [f"{img_path.stem}_nobg.png"]
    for bg_stem, _bg_pil, _bg_ar, bg_ar_tag in bg_items:
        names.append(f"{img_path.stem}_bg_{bg_stem}_{bg_ar_tag}.jpg")
    return names


def _all_outputs_present(
    img_path: Path,
    bg_items: List[Tuple[str, "object", float, str]],
    output_dir: Path,
) -> bool:
    """True iff every expected output for ``img_path`` already exists."""
    return all(
        (output_dir / name).exists() and (output_dir / name).stat().st_size > 0
        for name in _expected_outputs(img_path, bg_items)
    )


def _process_one_image(
    idx: int,
    total: int,
    img_path: Path,
    output_dir: Path,
    bg_items: List[Tuple[str, "object", float, str]],
    model_name: str,
) -> ItemResult:
    """
    Run the full per-image pipeline (decode → face → matting → canvas →
    archival PNG → per-BG composites). Pure function: no callbacks, no
    shared mutable state, safe to call from multiple threads concurrently
    on independent inputs.
    """
    import cv2
    import numpy as np

    produced: List[str] = []
    stage_times: dict = {}
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

        # Matting runs on the raw, full resolution. No pre-downscale, no
        # supersampling — whatever the raw is, pymatting sees. Face coords
        # stay in raw space for the canvas step below.
        t0 = time.perf_counter()
        subject = remove_background(raw, model_name=model_name)
        stage_times["matting"] = time.perf_counter() - t0

        # Archival 1:1 PNG — same canvas algorithm as the composites so the
        # 1:1 nobg and any 1:1 background output share identical framing of
        # the subject (only the background layer differs).
        t0 = time.perf_counter()
        nobg_canvas = canvas_for_ar(subject, fx, fy, 1.0)
        nobg_name = f"{img_path.stem}_nobg.png"
        nobg_canvas.save(output_dir / nobg_name, format="PNG", optimize=True)
        produced.append(nobg_name)
        stage_times["save_png"] = time.perf_counter() - t0

        # Per-background composites: each output takes on its background's
        # aspect ratio. Filename carries the AR tag (e.g. "_16x9").
        t0 = time.perf_counter()
        for bg_name, bg_pil, bg_ar, bg_ar_tag in bg_items:
            subject_for_bg = canvas_for_ar(subject, fx, fy, bg_ar)
            composed = composite_on_background(subject_for_bg, bg_pil)
            out_name = f"{img_path.stem}_bg_{bg_name}_{bg_ar_tag}.jpg"
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
            "face detected" if detected else "no face found → centered fallback"
        )
        elapsed = time.perf_counter() - t_start
        return ItemResult(
            idx, total, img_path.name, True, detail, produced,
            elapsed_s=elapsed, stage_times=stage_times,
        )
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - t_start
        return ItemResult(
            idx, total, img_path.name, False, str(e), produced,
            elapsed_s=elapsed, stage_times=stage_times,
        )


def process_batch(
    input_dir: Path,
    output_dir: Path,
    backgrounds: List[Tuple[str, bytes]],
    on_progress: Optional[ProgressFn] = None,
    model_name: str = DEFAULT_MODEL,
    on_start: Optional[StartFn] = None,
    max_workers: int = 1,
    skip_existing: bool = True,
) -> List[ItemResult]:
    """
    Execute the full pipeline for every image in ``input_dir``.

    When ``max_workers > 1`` images are processed in parallel on a
    ThreadPoolExecutor. onnxruntime's ``InferenceSession.run`` is
    thread-safe (the session is a shared singleton), and pymatting /
    OpenCV release the GIL during heavy work, so threading actually
    scales. While one image is in pymatting on the CPU, another can be
    in ONNX inference on the ANE/Metal GPU.

    ``on_progress`` is always fired from the main thread (the one that
    called ``process_batch``), so it's safe to call Streamlit APIs from
    inside it. ``on_start`` is only fired in the **sequential** path
    (``max_workers == 1``) — in parallel mode it would have to fire from
    worker threads, which produces the "missing ScriptRunContext"
    warnings and serves no useful purpose when multiple images are in
    flight simultaneously.

    When ``skip_existing`` is True (default) any input whose complete
    output set is already on disk is short-circuited: no decode, no
    matting, no composite. A synthetic ``ItemResult`` with
    ``skipped=True`` is emitted for each so callers can show "N skipped"
    stats. Matching is purely filename-based: ``{stem}_nobg.png`` plus
    ``{stem}_bg_{bgstem}_{artag}.jpg`` for every current background.
    """
    from PIL import Image

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preload backgrounds once — reused for every subject. For each background
    # we also precompute its aspect ratio and canonical AR label (e.g. "16x9")
    # so the per-subject composite stage only does cropping + compositing.
    bg_items: List[Tuple[str, "Image.Image", float, str]] = []
    bg_errors: List[str] = []
    for name, data in backgrounds:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            bw, bh = img.size
            if bw <= 0 or bh <= 0:
                raise ValueError("zero-size image")
            bg_ar = bw / bh
            label = ar_label(bw, bh)
            bg_items.append((Path(name).stem, img, bg_ar, label))
        except Exception as e:  # noqa: BLE001
            bg_errors.append(f"background '{name}' unreadable: {e}")

    images = list_images(input_dir)
    total = len(images)
    results: List[ItemResult] = []

    cb_lock = threading.Lock()

    def _fire_start(idx: int, filename: str) -> None:
        if on_start is None:
            return
        with cb_lock:
            try:
                on_start(idx, total, filename)
            except Exception:  # noqa: BLE001
                pass

    def _fire_progress(res: ItemResult) -> None:
        if on_progress is None:
            return
        with cb_lock:
            try:
                on_progress(res)
            except Exception:  # noqa: BLE001
                pass

    # --- Resume pass: partition inputs into "already done" vs "to process" ---
    # This runs on the main thread before any heavy work, so Streamlit sees
    # the skip log lines immediately.
    to_process: List[Tuple[int, Path]] = []
    for idx, img_path in enumerate(images, start=1):
        if skip_existing and _all_outputs_present(img_path, bg_items, output_dir):
            outs = _expected_outputs(img_path, bg_items)
            skip_res = ItemResult(
                idx, total, img_path.name, True,
                "already processed · skipped",
                outs,
                elapsed_s=0.0,
                stage_times={},
                skipped=True,
            )
            results.append(skip_res)
            _fire_progress(skip_res)
        else:
            to_process.append((idx, img_path))

    # Warm up the lazy singletons only if we actually have work to do. If the
    # whole batch was cached (resume on identical inputs) we skip the ~2 s
    # model load entirely.
    if to_process:
        get_rembg_session(model_name)
        get_face_detector()

    workers = max(1, min(int(max_workers), len(to_process) or 1))

    if workers <= 1:
        # Sequential path — preserves exact legacy semantics, including
        # on_start callbacks (safe because we're on the main thread).
        for idx, img_path in to_process:
            _fire_start(idx, img_path.name)
            res = _process_one_image(
                idx, total, img_path, output_dir, bg_items, model_name
            )
            results.append(res)
            _fire_progress(res)
    else:
        # Parallel path: we deliberately do NOT fire on_start from workers.
        # Streamlit APIs must be called from the main thread; firing from a
        # worker triggers "missing ScriptRunContext" warnings and leaves the
        # UI in an undefined state. on_progress is still fired here (main
        # thread) as each future completes.
        def _run(idx: int, img_path: Path) -> ItemResult:
            return _process_one_image(
                idx, total, img_path, output_dir, bg_items, model_name
            )

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="hkn-worker"
        ) as ex:
            futures = {
                ex.submit(_run, idx, p): (idx, p) for idx, p in to_process
            }
            for fut in as_completed(futures):
                idx, img_path = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    res = ItemResult(
                        idx, total, img_path.name, False, str(e), [],
                        elapsed_s=0.0, stage_times={},
                    )
                results.append(res)
                _fire_progress(res)

    # Return results in original input order regardless of completion
    # order — callers (and tests) expect stable ordering.
    results.sort(key=lambda r: r.index)

    if bg_errors and on_progress is not None:
        for msg in bg_errors:
            on_progress(
                ItemResult(0, total, "(backgrounds)", False, msg, [])
            )

    return results
