#!/usr/bin/env bash
# ------------------------------------------------------------------
# HKN PoliTO · Associate Portrait Pipeline — zero-setup launcher
# ------------------------------------------------------------------
# 1. Creates a Python virtual environment (./venv) if missing.
# 2. Activates it.
# 3. Upgrades pip + installs requirements.txt (only when needed).
# 4. Launches the Streamlit UI.
#
# Model weights (Mediapipe face detector, rembg u2net ≈ 170 MB) are
# downloaded automatically by the libraries on first run.
# ------------------------------------------------------------------

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- colors ---
RESET=$'\033[0m'
BOLD=$'\033[1m'
GOLD=$'\033[38;5;179m'
DIM=$'\033[2m'

banner() {
    printf "\n%s%s HKN PoliTO %s ·%s Associate Portrait Pipeline\n" \
        "$BOLD" "$GOLD" "$RESET$DIM" "$RESET"
    printf "%s------------------------------------------------%s\n" "$DIM" "$RESET"
}

PYBIN=""
require_python() {
    # Prefer 3.12 → 3.11 → 3.13 → 3.10 → generic python3.
    # Mediapipe / rembg / onnxruntime currently ship the widest wheel
    # coverage for 3.11 and 3.12; fall through to whatever is available.
    local candidates=(python3.12 python3.11 python3.13 python3.10 python3)
    for c in "${candidates[@]}"; do
        if command -v "$c" >/dev/null 2>&1; then
            PYBIN="$c"
            break
        fi
    done
    if [[ -z "$PYBIN" ]]; then
        echo "ERROR: no Python 3 interpreter found in PATH." >&2
        echo "Install Python 3.11 or 3.12 (https://python.org) and retry." >&2
        exit 1
    fi
    local v
    v="$($PYBIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    printf " %s·%s using %s (Python %s)\n" "$GOLD" "$RESET" "$PYBIN" "$v"

    # Warn if user is on a bleeding-edge version where wheels may be missing.
    local major minor
    major="${v%.*}"; minor="${v#*.}"
    if (( major == 3 && minor >= 14 )); then
        printf " %s!%s Python %s is very new — if pip fails building wheels,\n" "$GOLD" "$RESET" "$v"
        printf "   install Python 3.12 via Homebrew: brew install python@3.12\n"
    fi
}

setup_venv() {
    if [[ ! -d "venv" ]]; then
        printf " %s·%s venv not found · creating ./venv with %s\n" "$GOLD" "$RESET" "$PYBIN"
        "$PYBIN" -m venv venv
    else
        # Detect mismatch between existing venv and selected interpreter.
        local venv_v cur_v
        venv_v="$(venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo '?')"
        cur_v="$($PYBIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        if [[ "$venv_v" != "$cur_v" ]]; then
            printf " %s!%s existing venv uses Python %s, but %s is %s\n" \
                "$GOLD" "$RESET" "$venv_v" "$PYBIN" "$cur_v"
            printf "   rebuilding ./venv to match\n"
            rm -rf venv
            "$PYBIN" -m venv venv
        else
            printf " %s·%s venv present · reusing (Python %s)\n" "$GOLD" "$RESET" "$venv_v"
        fi
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
}

install_deps() {
    printf " %s·%s upgrading pip · installing requirements\n" "$GOLD" "$RESET"
    python -m pip install --upgrade pip wheel setuptools >/dev/null

    # Only reinstall if requirements.txt changed (hash stamp).
    local stamp="venv/.requirements.sha256"
    local current
    current="$(shasum -a 256 requirements.txt | awk '{print $1}')"
    if [[ -f "$stamp" ]] && [[ "$(cat "$stamp")" == "$current" ]]; then
        printf " %s·%s requirements unchanged · skipping install\n" "$GOLD" "$RESET"
    else
        python -m pip install -r requirements.txt
        echo "$current" >"$stamp"
    fi
}

launch() {
    printf "\n %s·%s launching Streamlit · Ctrl-C to quit\n\n" "$GOLD" "$RESET"
    exec streamlit run app.py \
        --server.headless true \
        --browser.gatherUsageStats false
}

banner
require_python
setup_venv
install_deps
launch
