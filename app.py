"""
HKN PoliTO — Associate Portrait Pipeline
Streamlit GUI, local processing, GDPR-safe.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Lightweight imports from pipeline — heavy ones (cv2/mediapipe/rembg) are
# deferred inside pipeline.py functions so the app renders instantly.
from pipeline import (
    DEFAULT_MODEL,
    DEFAULT_WORKERS,
    HIGH_QUALITY_MODEL,
    ItemResult,
    MAX_RECOMMENDED_WORKERS,
    VALID_EXT,
    list_images,
    prewarm,
    process_batch,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HKN PoliTO · Associate Portrait Pipeline",
    page_icon="⟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str:
    p = ASSETS / "hkn_logo_white.svg"
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/svg+xml;base64,{b64}"


def inject_css() -> None:
    css = _read_text(ASSETS / "theme.css")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def now_ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def fmt_secs(s: float) -> str:
    """Human-readable duration: 0.8s, 12.4s, 1m23s, 4m07s."""
    if s < 0.1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m, r = divmod(s, 60)
    return f"{int(m)}m{int(r):02d}s"


def _pick_directory_macos(initial: str = "") -> Optional[str]:
    """
    Native Cocoa folder picker via AppleScript.

    Tkinter inside a Streamlit server process on macOS frequently opens the
    dialog *behind* the browser or fails silently because ``Tk()`` isn't on
    the main thread. osascript sidesteps this: the OS itself owns the dialog
    and forces it to the foreground.
    """
    default = ""
    try:
        if initial:
            p = Path(initial).expanduser()
            if p.exists() and p.is_dir():
                # escape for AppleScript string
                safe = str(p).replace("\\", "\\\\").replace('"', '\\"')
                default = f' default location (POSIX file "{safe}")'
    except Exception:
        default = ""

    script = (
        'tell application "System Events" to activate\n'
        f'POSIX path of (choose folder with prompt '
        f'"Select folder — HKN PoliTO"{default})'
    )

    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None

    if proc.returncode != 0:
        # user cancelled → osascript returns non-zero and prints on stderr
        return None
    path = proc.stdout.strip()
    return path.rstrip("/") or None


def _pick_directory_tk_subprocess(initial: str = "") -> Optional[str]:
    """
    Cross-platform fallback: run tkinter in an *isolated subprocess* so it
    doesn't fight Streamlit's main thread. The subprocess prints the chosen
    path on its last stdout line and exits.
    """
    helper = textwrap.dedent(
        """
        import sys
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            print("")
            sys.exit(0)
        initial = sys.argv[1] if len(sys.argv) > 1 else ""
        root = tk.Tk()
        root.withdraw()
        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass
        root.update()
        path = filedialog.askdirectory(
            parent=root,
            initialdir=initial or None,
            title="Select folder — HKN PoliTO",
            mustexist=False,
        )
        try:
            root.destroy()
        except Exception:
            pass
        print(path or "")
        """
    ).strip()

    try:
        proc = subprocess.run(
            [sys.executable, "-c", helper, initial or ""],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return last or None


def _pick_directory(initial: str = "") -> Optional[str]:
    """
    Open a native folder picker. Returns the chosen absolute path or None
    if the user cancelled / no picker is available.
    """
    if platform.system() == "Darwin":
        result = _pick_directory_macos(initial)
        if result is not None:
            return result
        # fall through to tkinter subprocess if AppleScript failed
    return _pick_directory_tk_subprocess(initial)


def _browse_cb(key: str, flash_key: str):
    """Callback factory: opens picker, writes the choice into session state.
    Stores a status message in ``flash_key`` so the UI can surface failures."""
    def _cb():
        picked = _pick_directory(st.session_state.get(key, ""))
        if picked:
            st.session_state[key] = picked
            st.session_state[flash_key] = None
        else:
            # None means cancel OR picker failed — we can't tell the two apart
            # from osascript, so only flash a warning if nothing was picked
            # and the field is still empty.
            if not st.session_state.get(key):
                st.session_state[flash_key] = (
                    "No folder selected. If the picker didn't appear, "
                    "check behind the browser window or paste the path "
                    "manually above."
                )
    return _cb


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("log", [])
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("last_run", None)
    st.session_state.setdefault("input_path", "")
    st.session_state.setdefault("output_path", "")
    st.session_state.setdefault("_flash_in", None)
    st.session_state.setdefault("_flash_out", None)
    st.session_state.setdefault("models_ready", False)
    st.session_state.setdefault("models_ready_for", None)  # which model was warmed
    st.session_state.setdefault("high_quality", False)
    st.session_state.setdefault("max_workers", DEFAULT_WORKERS)
    # key used by the file_uploader — bump to reset it (Streamlit pattern)
    st.session_state.setdefault("_bg_uploader_key", 0)


def _clear_backgrounds():
    """Reset the uploader by rotating its widget key."""
    st.session_state["_bg_uploader_key"] += 1


@st.cache_data(show_spinner=False, max_entries=128)
def _thumb_data_uri(name: str, data: bytes) -> tuple:
    """
    Build a small JPEG thumbnail of an uploaded image and return
    (data_uri, width_px, height_px, display_size_kb).
    Cached by (name, bytes) so reruns don't re-encode.
    """
    try:
        from PIL import Image
    except Exception:
        return ("", 0, 0, 0)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        w, h = img.size
        thumb = img.convert("RGB").copy()
        thumb.thumbnail((96, 96), Image.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, "JPEG", quality=78)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return (f"data:image/jpeg;base64,{b64}", w, h, round(len(data) / 1024))
    except Exception:
        return ("", 0, 0, round(len(data) / 1024))


def render_background_previews(uploads, sidebar) -> None:
    """Render our own chip list with real thumbnails + a 'Clear all' button.
    The built-in Streamlit pills are hidden via CSS."""
    if not uploads:
        return
    n = len(uploads)
    sidebar.markdown(
        f'<div class="hkn-bg-preview-label">Selected'
        f'<span class="count">· {n:02d}</span></div>',
        unsafe_allow_html=True,
    )

    chips_html = []
    for f in uploads:
        data = f.getvalue()
        uri, w, h, kb = _thumb_data_uri(f.name, data)
        thumb_el = (
            f'<img class="thumb" src="{uri}" alt="" />'
            if uri
            else '<div class="thumb placeholder"></div>'
        )
        dim = f'{w} × {h}' if w and h else '—'
        chips_html.append(
            f'<div class="hkn-bg-chip">'
            f'{thumb_el}'
            f'<div class="meta">'
            f'<span class="name" title="{html.escape(f.name)}">{html.escape(f.name)}</span>'
            f'<span class="dim">{dim} · {kb} KB</span>'
            f'</div>'
            f'</div>'
        )
    sidebar.markdown(
        f'<div class="hkn-bg-chips">{"".join(chips_html)}</div>',
        unsafe_allow_html=True,
    )

    sidebar.button(
        "Clear all backgrounds",
        key="_clear_bg",
        on_click=_clear_backgrounds,
        use_container_width=True,
    )


def push_log(kind: str, msg_html: str) -> None:
    st.session_state["log"].append(
        {"ts": now_ts(), "kind": kind, "msg": msg_html}
    )


# ---------------------------------------------------------------------------
# Model warm-up (called from within st.status so user sees progress)
# ---------------------------------------------------------------------------
def ensure_models_ready(model_name: str) -> None:
    """Eager-load heavy deps + rembg/mediapipe models with live status UI.
    Re-warms if the active model changed since the last run."""
    if (
        st.session_state.get("models_ready")
        and st.session_state.get("models_ready_for") == model_name
    ):
        return

    with st.status(
        f"Preparing models · {model_name}",
        expanded=True,
        state="running",
    ) as status:
        def _step(msg: str) -> None:
            status.write(f"**›** {msg}")

        try:
            t0 = time.perf_counter()
            prewarm(model_name=model_name, on_step=_step)
            dt_warm = time.perf_counter() - t0
            _step(f"Warm-up completed in {fmt_secs(dt_warm)}")
            status.update(
                label=f"Models ready · {model_name} · {fmt_secs(dt_warm)}",
                state="complete",
                expanded=False,
            )
            st.session_state["models_ready"] = True
            st.session_state["models_ready_for"] = model_name
            st.session_state["_warmup_secs"] = dt_warm
        except Exception as e:  # noqa: BLE001
            status.update(
                label=f"Warm-up failed: {e}",
                state="error",
                expanded=True,
            )
            raise


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    logo = _logo_data_uri()
    brand_html = f"""
    <div class="hkn-side-brand">
        {f'<img class="logo" src="{logo}" alt="HKN PoliTO" />' if logo else ''}
        <div class="chapter">IEEE · Eta Kappa Nu<br/><b>Mu Nu Chapter</b></div>
        <div class="tag">Politecnico di Torino · Since 2017</div>
    </div>
    """
    st.sidebar.markdown(brand_html, unsafe_allow_html=True)

    # ---- SOURCE -----------------------------------------------------------
    st.sidebar.markdown(
        '<div class="hkn-side-label"><span class="idx">001 ·</span> Source</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="hkn-side-hint">Folder holding the associate portraits.</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.text_input(
        "Input folder",
        key="input_path",
        placeholder="No folder selected — click Browse",
        label_visibility="collapsed",
    )
    st.sidebar.button(
        "▸  Browse folder…",
        key="_browse_in",
        on_click=_browse_cb("input_path", "_flash_in"),
        use_container_width=True,
    )
    if st.session_state.get("_flash_in"):
        st.sidebar.markdown(
            f'<div class="hkn-flash">{html.escape(st.session_state["_flash_in"])}</div>',
            unsafe_allow_html=True,
        )

    # ---- DESTINATION ------------------------------------------------------
    st.sidebar.markdown(
        '<div class="hkn-side-label"><span class="idx">002 ·</span> Destination</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="hkn-side-hint">Where processed files will be written.</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.text_input(
        "Output folder",
        key="output_path",
        placeholder="No folder selected — click Browse",
        label_visibility="collapsed",
    )
    st.sidebar.button(
        "▸  Browse folder…",
        key="_browse_out",
        on_click=_browse_cb("output_path", "_flash_out"),
        use_container_width=True,
    )
    if st.session_state.get("_flash_out"):
        st.sidebar.markdown(
            f'<div class="hkn-flash">{html.escape(st.session_state["_flash_out"])}</div>',
            unsafe_allow_html=True,
        )

    # ---- BACKGROUNDS ------------------------------------------------------
    st.sidebar.markdown(
        '<div class="hkn-side-label"><span class="idx">003 ·</span> Backgrounds</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="hkn-side-hint">Drop in one or more images. JPG / JPEG / PNG.</div>',
        unsafe_allow_html=True,
    )
    uploads = st.sidebar.file_uploader(
        "Backgrounds",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"_bg_uploader_{st.session_state['_bg_uploader_key']}",
    )
    render_background_previews(uploads, st.sidebar)

    # ---- QUALITY ----------------------------------------------------------
    st.sidebar.markdown(
        '<div class="hkn-side-label"><span class="idx">004 ·</span> Quality</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="hkn-side-hint">Standard: <code>isnet-general-use</code> '
        '(~175 MB, fast). <b style="color:var(--gold)">High quality</b>: '
        '<code>birefnet-portrait</code> (~440 MB first-time download, slower, '
        'but noticeably cleaner hair edges on curly / frizzy subjects).</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.toggle(
        "High quality · birefnet-portrait",
        key="high_quality",
        help="Recommended for portraits with curly or fly-away hair.",
    )

    # ---- PARALLELISM ------------------------------------------------------
    st.sidebar.markdown(
        '<div class="hkn-side-label"><span class="idx">005 ·</span> Parallelism</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="hkn-side-hint">Workers processing images in parallel. '
        '2 is a safe default: one in ANE/GPU inference while another is in '
        'CPU matting. More helps on bigger machines but each worker adds '
        '~1–2&nbsp;GB of memory pressure per 24&nbsp;MP image.</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.number_input(
        "Workers",
        min_value=1,
        max_value=max(MAX_RECOMMENDED_WORKERS, DEFAULT_WORKERS),
        step=1,
        key="max_workers",
        help=(
            "1 = sequential (legacy behaviour). "
            f"Recommended upper bound on this machine: {MAX_RECOMMENDED_WORKERS}."
        ),
    )

    # ---- EXECUTE ----------------------------------------------------------
    st.sidebar.markdown(
        '<div class="hkn-side-label"><span class="idx">006 ·</span> Execute</div>',
        unsafe_allow_html=True,
    )
    start = st.sidebar.button(
        "Begin processing",
        type="primary",
        key="_start_btn",
        use_container_width=True,
        disabled=st.session_state.get("running", False),
    )

    st.sidebar.markdown(
        """
        <div class="hkn-side-footer">
            Driven by passion · Guided by values
        </div>
        """,
        unsafe_allow_html=True,
    )

    return (
        st.session_state.get("input_path", ""),
        st.session_state.get("output_path", ""),
        uploads,
        start,
    )


# ---------------------------------------------------------------------------
# Masthead, headline, protocol strip
# ---------------------------------------------------------------------------
def render_masthead():
    st.markdown(
        """
        <div class="hkn-masthead">
            <div class="mark">HKN <em>PoliTO</em> · Associate Operations</div>
            <div class="meta">Local · On-device · v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_headline():
    st.markdown(
        """
        <div class="hkn-kicker">Instrument № 01 — Identity Portraits</div>
        <h1 class="hkn-headline">
            The Associate <em>Portrait</em><br/>
            Pipeline <span class="amp">&amp;</span> Compositor
        </h1>
        <p class="hkn-lede">
            A six-step, fully local protocol that prepares new-associate
            portraits for the Mu Nu chapter — <em>face-aware cropping</em>,
            alpha-matted background removal, and compositing over the
            chapter's chosen backgrounds. No pixel leaves this machine.
        </p>
        <div class="hkn-badges">
            <span class="hkn-badge"><span class="dot"></span>Fully Offline</span>
            <span class="hkn-badge">GDPR · Art. 25 Ready</span>
            <span class="hkn-badge">rembg · u2net + Alpha Matting</span>
            <span class="hkn-badge">Mediapipe Face Detect</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


PROTOCOL = [
    ("01", "Detect", "Mediapipe locates the subject's face, or falls back to image center."),
    ("02", "Frame", "Square crop on the shortest side; face anchored at 30% from the top."),
    ("03", "Isolate", "rembg / u2net with alpha matting — curly & frizzy hair preserved."),
    ("04", "Archive", "Transparent RGBA saved as _nobg.png for future reuse."),
    ("05", "Compose", "Each chapter background is fit-cropped and alpha-composited."),
    ("06", "Publish", "Final RGB exports saved as high-quality JPEG (q = 95)."),
]


def render_protocol():
    cells = "".join(
        f"""<div class="hkn-step">
            <div class="num">{num}</div>
            <span class="name">{name}</span>
            <span class="note">{note}</span>
        </div>"""
        for num, name, note in PROTOCOL
    )
    st.markdown(f'<div class="hkn-protocol">{cells}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stats & log rendering
# ---------------------------------------------------------------------------
def render_stats(total: int, done: int, ok: int, err: int):
    st.markdown(
        f"""
        <div class="hkn-stats">
            <div class="hkn-stat">
                <span class="k">Queued</span>
                <span class="v">{total:02d}</span>
            </div>
            <div class="hkn-stat">
                <span class="k">Processed</span>
                <span class="v gold">{done:02d}</span>
            </div>
            <div class="hkn-stat">
                <span class="k">Successes</span>
                <span class="v ok">{ok:02d}</span>
            </div>
            <div class="hkn-stat">
                <span class="k">Errors</span>
                <span class="v err">{err:02d}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_log(container):
    rows = st.session_state.get("log", [])
    if not rows:
        container.markdown(
            '<div class="hkn-log"><div class="hkn-log-empty">'
            'The console is silent. Configure the run in the sidebar '
            'and press <b>Begin processing</b>.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    html_rows = []
    for r in rows:
        glyph_cls = {
            "ok": "glyph-ok",
            "err": "glyph-err",
            "info": "glyph-info",
        }.get(r["kind"], "glyph-info")
        glyph = {"ok": "✓", "err": "✕", "info": "▸"}.get(r["kind"], "·")
        html_rows.append(
            f'<div class="hkn-log-row">'
            f'<span class="ts">{r["ts"]}</span>'
            f'<span class="{glyph_cls}">{glyph}</span>'
            f'<span class="msg">{r["msg"]}</span>'
            f'</div>'
        )
    container.markdown(
        f'<div class="hkn-log">{"".join(html_rows)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_inputs(in_path: str, out_path: str):
    errors = []
    if not in_path.strip():
        errors.append("Input folder is not set — click Browse under **Source**.")
    if not out_path.strip():
        errors.append("Output folder is not set — click Browse under **Destination**.")
    if in_path.strip():
        p = Path(in_path).expanduser()
        if not p.exists():
            errors.append(f"Input folder does not exist: `{p}`")
        elif not p.is_dir():
            errors.append(f"Input path is not a directory: `{p}`")
        elif not list_images(p):
            errors.append(
                f"No supported images found in `{p}` "
                f"(accepted: {', '.join(sorted(VALID_EXT))})."
            )
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _init_state()
    inject_css()

    input_path, output_path, uploads, start_clicked = render_sidebar()

    render_masthead()
    render_headline()

    st.markdown(
        '<div class="hkn-section-label"><span class="dot"></span>The protocol</div>',
        unsafe_allow_html=True,
    )
    render_protocol()

    st.markdown(
        '<div class="hkn-section-label"><span class="dot"></span>Execution</div>',
        unsafe_allow_html=True,
    )
    status_slot = st.empty()
    stats_slot = st.empty()
    progress_slot = st.empty()
    warmup_slot = st.empty()  # reserved for st.status

    st.markdown(
        '<div class="hkn-section-label" style="margin-top:1.8rem;">'
        '<span class="dot"></span>Console</div>',
        unsafe_allow_html=True,
    )
    log_slot = st.empty()

    # ---- idle view --------------------------------------------------------
    if not st.session_state["running"]:
        total = 0
        if input_path.strip():
            p = Path(input_path).expanduser()
            if p.exists() and p.is_dir():
                total = len(list_images(p))

        with stats_slot.container():
            render_stats(total, 0, 0, 0)
        status_slot.markdown(
            f"""
            <div class="hkn-status">
                <span class="label">Status</span>
                <span class="value">Idle · awaiting instructions</span>
                <span class="count">{total:02d} queued</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress_slot.progress(0.0)
        render_log(log_slot)

    # ---- run --------------------------------------------------------------
    if start_clicked and not st.session_state["running"]:
        errors = validate_inputs(input_path, output_path)
        if errors:
            for e in errors:
                st.error(e)
            return

        st.session_state["running"] = True
        st.session_state["log"] = []
        in_dir = Path(input_path).expanduser()
        out_dir = Path(output_path).expanduser()

        high_quality = bool(st.session_state.get("high_quality"))
        model_name = HIGH_QUALITY_MODEL if high_quality else DEFAULT_MODEL
        max_workers = int(st.session_state.get("max_workers") or DEFAULT_WORKERS)

        # ---- warm-up with live status ------------------------------------
        try:
            with warmup_slot.container():
                ensure_models_ready(model_name)
        except Exception as e:  # noqa: BLE001
            st.session_state["running"] = False
            st.error(f"Model warm-up failed: {e}")
            return

        push_log(
            "info",
            f'Run initiated · source <span class="path">{html.escape(str(in_dir))}</span>',
        )
        push_log(
            "info",
            f'Destination <span class="path">{html.escape(str(out_dir))}</span>',
        )
        push_log(
            "info",
            f'Matting model · <span class="path">{html.escape(model_name)}</span>'
            f' · full-res, no supersample · '
            f'<span class="path">{max_workers}</span> worker(s)',
        )

        bgs: List[tuple] = []
        if uploads:
            for f in uploads:
                bgs.append((f.name, f.getvalue()))
            push_log(
                "info",
                f'{len(bgs)} background(s) loaded: '
                f'<span class="dim">{html.escape(", ".join(f.name for f in uploads))}</span>',
            )
        else:
            push_log(
                "info",
                'No backgrounds supplied — <span class="dim">transparent _nobg.png '
                'will be produced, no composites.</span>',
            )

        ok_count = 0
        err_count = 0
        finished_count = 0
        total_elapsed = 0.0
        batch_t0 = time.perf_counter()
        progress_bar = progress_slot.progress(0.0)

        def on_start(idx: int, total: int, filename: str):
            # In parallel mode several images are in flight at once; keep the
            # start log minimal so the stream stays coherent.
            push_log(
                "info",
                f'Starting <span class="path">{html.escape(filename)}</span> '
                f'· <span class="dim">{idx}/{total}</span>',
            )
            render_log(log_slot)

        def on_progress(r: ItemResult):
            nonlocal ok_count, err_count, finished_count, total_elapsed
            if r.index == 0:
                push_log("err", html.escape(r.detail))
                render_log(log_slot)
                return

            # Completion order differs from dispatch order in parallel mode,
            # so base the progress bar / ETA on "how many are done" rather
            # than the input-order index of the last finisher.
            finished_count += 1
            total_elapsed += r.elapsed_s
            avg = total_elapsed / max(finished_count, 1)
            remaining = (r.total - finished_count) * avg

            # Compact stage breakdown: matting is the heavy one, worth
            # calling out. Others are in the tooltip-ish dim line.
            stages = r.stage_times or {}
            matting_s = stages.get("matting", 0.0)
            compose_s = stages.get("composite", 0.0)
            other_s = max(
                0.0,
                r.elapsed_s - matting_s - compose_s
                - stages.get("save_png", 0.0),
            )
            time_tag = (
                f'<span class="dim"> · '
                f'<b>{fmt_secs(r.elapsed_s)}</b> total · '
                f'matting {fmt_secs(matting_s)} · '
                f'composite {fmt_secs(compose_s)} · '
                f'other {fmt_secs(other_s)}</span>'
            )

            if r.ok:
                ok_count += 1
                produced = (
                    f' <span class="dim">→ {html.escape(", ".join(r.produced))}</span>'
                    if r.produced else ""
                )
                push_log(
                    "ok",
                    f'<span class="path">{html.escape(r.filename)}</span> '
                    f'· {html.escape(r.detail)}{produced}{time_tag}',
                )
            else:
                err_count += 1
                push_log(
                    "err",
                    f'<span class="path">{html.escape(r.filename)}</span> '
                    f'· {html.escape(r.detail)}'
                    f'<span class="dim"> · {fmt_secs(r.elapsed_s)}</span>',
                )

            frac = finished_count / max(r.total, 1)
            progress_bar.progress(min(frac, 1.0))
            eta_str = (
                f' · ETA {fmt_secs(remaining)}'
                if finished_count < r.total else ''
            )
            status_slot.markdown(
                f"""
                <div class="hkn-status">
                    <span class="label">Processing</span>
                    <span class="value">{finished_count} of {r.total} done · last · {html.escape(r.filename)}
                        <span class="dim"> · avg {fmt_secs(avg)}/img{eta_str}</span></span>
                    <span class="count">{int(frac * 100):02d}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            stats_slot.empty()
            with stats_slot.container():
                render_stats(r.total, finished_count, ok_count, err_count)
            render_log(log_slot)

        try:
            process_batch(
                in_dir,
                out_dir,
                bgs,
                on_progress=on_progress,
                on_start=on_start,
                model_name=model_name,
                max_workers=max_workers,
            )
            wall = time.perf_counter() - batch_t0
            processed = ok_count + err_count
            avg_all = (total_elapsed / processed) if processed else 0.0
            push_log(
                "info",
                f'Run complete · <span class="path">{ok_count} ok</span> · '
                f'{err_count} error(s) · '
                f'wall <b>{fmt_secs(wall)}</b> · '
                f'avg <b>{fmt_secs(avg_all)}</b>/img'
                + (
                    f' · warm-up '
                    f'{fmt_secs(st.session_state.get("_warmup_secs", 0.0))}'
                    if st.session_state.get("_warmup_secs")
                    else ""
                ),
            )
            progress_bar.progress(1.0)
            status_slot.markdown(
                f"""
                <div class="hkn-status">
                    <span class="label">Status</span>
                    <span class="value">Complete · {ok_count} processed, {err_count} failed
                        <span class="dim"> · {fmt_secs(wall)} total · {fmt_secs(avg_all)}/img</span></span>
                    <span class="count">100%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:  # noqa: BLE001
            push_log("err", f"Fatal error: {html.escape(str(e))}")
            render_log(log_slot)
        finally:
            st.session_state["running"] = False
            render_log(log_slot)

    # ---- colophon ---------------------------------------------------------
    st.markdown('<div class="hkn-ornament">· · ·</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hkn-colophon">
            <span>Mu Nu Chapter · IEEE-HKN · Politecnico di Torino</span>
            <em>"Driven by passion, guided by values."</em>
            <span>Local · No telemetry · MIT tooling</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
