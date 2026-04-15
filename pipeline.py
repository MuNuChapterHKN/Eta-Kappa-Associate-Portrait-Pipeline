"""
HKN PoliTO — Associate ID Photo Pipeline
Local, GDPR-safe batch processor.

Pipeline (per image):
    1. Face detection (Mediapipe, CPU)
    2. Frame at the input's native aspect ratio, minimally cropped with face
       at FACE_TOP_RATIO from top (≤ MAX_SIDE on longest side, LANCZOS)
    3. Background removal (rembg, alpha matting ON) → "standard-AR" RGBA
    4. Re-crop the RGBA to 1:1 and save <name>_nobg.png (archival)
    5. For each background:
        · re-crop the RGBA to the background's aspect ratio
        · cover-fit the background under it
        · save <name>_bg_<bgname>_<WxH>.jpg (RGB, q=95)

The background removal runs *once* per input at the native aspect ratio;
each output composite is a cheap re-crop + alpha-compose, not a re-matting.

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

FACE_TOP_RATIO = 0.30  # face vertical position in crop (from top), universal
PORTRAIT_FACE_TOP_RATIO = FACE_TOP_RATIO  # backward-compat alias

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


def crop_rgba_to_ar(
    subject_rgba,
    face_x: int,
    face_y: int,
    target_ar: float,
    face_top_ratio: float = FACE_TOP_RATIO,
):
    """
    Re-crop an already-matted RGBA subject to ``target_ar``, keeping the face
    at ``(new_W/2, face_top_ratio * new_H)``.

    Used for:
      · the 1:1 archival PNG (``target_ar = 1.0``)
      · each per-background composite (``target_ar = bg_w / bg_h``)

    If an edge is hit (rare, happens when the face is close to the border of
    the native-AR standard frame), the crop clamps to the image bounds and
    accepts a small deviation in face position rather than upscaling.
    """
    from PIL import Image  # noqa: F401 — just to ensure import is sane
    w, h = subject_rgba.size
    fty = face_top_ratio

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

    cx = max(0.0, min(float(w) - cw, cx))
    cy = max(0.0, min(float(h) - ch, cy))

    x0, y0 = int(round(cx)), int(round(cy))
    x1 = min(w, x0 + int(round(cw)))
    y1 = min(h, y0 + int(round(ch)))
    return subject_rgba.crop((x0, y0, x1, y1))


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

            # Frame at the input's NATIVE aspect ratio (minimally cropped to
            # place the face at FACE_TOP_RATIO from top). This is the "standard
            # frame" on which background removal runs — one matting pass, then
            # every output re-crops the resulting RGBA to its own target AR.
            t0 = time.perf_counter()
            raw_h, raw_w = raw.shape[:2]
            native_ar = raw_w / raw_h if raw_h else 1.0
            framed, (face_in_crop_x, face_in_crop_y) = smart_crop_to_ar(
                raw, fx, fy, native_ar
            )
            stage_times["crop"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            subject = remove_background(
                framed, model_name=model_name, supersample=supersample
            )
            stage_times["matting"] = time.perf_counter() - t0

            # Archival 1:1 PNG — a re-crop of the standard-AR RGBA centered on
            # the face. Same role as before; no re-matting.
            t0 = time.perf_counter()
            nobg_crop = crop_rgba_to_ar(
                subject, face_in_crop_x, face_in_crop_y, 1.0
            )
            nobg_name = f"{img_path.stem}_nobg.png"
            nobg_crop.save(output_dir / nobg_name, format="PNG", optimize=True)
            produced.append(nobg_name)
            stage_times["save_png"] = time.perf_counter() - t0

            # Per-background composites: each output takes on its background's
            # aspect ratio. Filename carries the AR tag (e.g. "_16x9") so you
            # can tell variants apart at a glance.
            t0 = time.perf_counter()
            for bg_name, bg_pil, bg_ar, bg_ar_tag in bg_items:
                subject_for_bg = crop_rgba_to_ar(
                    subject, face_in_crop_x, face_in_crop_y, bg_ar
                )
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
