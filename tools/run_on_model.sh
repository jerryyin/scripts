#!/bin/bash
# Set up AM or FFM environment and run a command.
#
# Auto-detects the simulator package (/am-ffm or /ffm) and backend.
# This script is environment-agnostic: run it directly inside a container,
# on bare metal, or pipe it through `docker exec`.
#
# Usage:
#   run_on_model.sh -- python3 kernel.py
#   run_on_model.sh --backend ffm -- python3 kernel.py --arg val
#
#   # Inside docker:
#   docker exec my-container /path/to/run_on_model.sh -- python3 k.py
#
# Options:
#   --backend am|ffm    Which simulator backend. Auto-detected if omitted:
#                        AM if /am-ffm exists, FFM otherwise.
#   -- COMMAND [ARGS...] Everything after -- is the command to run.
set -euo pipefail

BACKEND=""

usage() {
    sed -n '2,/^set /{ /^#/s/^# \?//p }' "$0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) BACKEND="$2"; shift 2 ;;
        -h|--help) usage ;;
        --)        shift; break ;;
        *)         echo "Unknown option: $1" >&2; usage ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: no command specified after --" >&2
    usage
fi

# ---------- auto-detect package directory ----------

PKG_DIR=""
if [[ -d /am-ffm ]]; then
    PKG_DIR=/am-ffm
elif [[ -d /ffm ]]; then
    PKG_DIR=/ffm
else
    echo "Error: neither /am-ffm nor /ffm found" >&2
    exit 1
fi

# ---------- auto-detect / validate backend ----------

if [[ -z "$BACKEND" ]]; then
    if [[ -f "$PKG_DIR/am_env.sh" ]]; then
        BACKEND=am
    elif [[ -f "$PKG_DIR/ffmlite_env.sh" ]]; then
        BACKEND=ffm
    else
        echo "Error: no env script found in $PKG_DIR" >&2
        exit 1
    fi
fi

case "$BACKEND" in
    am)
        if [[ ! -f "$PKG_DIR/am_env.sh" ]]; then
            echo "Error: AM requested but $PKG_DIR/am_env.sh not found" >&2
            exit 1
        fi
        source "$PKG_DIR/am_env.sh"
        ;;
    ffm)
        if [[ ! -f "$PKG_DIR/ffmlite_env.sh" ]]; then
            echo "Error: FFM requested but $PKG_DIR/ffmlite_env.sh not found" >&2
            exit 1
        fi
        source "$PKG_DIR/ffmlite_env.sh"
        ;;
    *)
        echo "Error: --backend must be 'am' or 'ffm', got '$BACKEND'" >&2
        exit 1
        ;;
esac

echo "[run_on_model] pkg=$PKG_DIR backend=$BACKEND" >&2

# ---------- ROCm overlay ----------
# Symlink bundled ROCm libs, skipping libamd_smi (conflicts with system).
# Only needed when the package ships its own rocm/ directory.

if [[ -d "$PKG_DIR/rocm" && ! -d /tmp/rocm-overlay ]]; then
    mkdir -p /tmp/rocm-overlay
    for f in "$PKG_DIR"/rocm/*.so*; do
        [[ -e "$f" ]] || continue
        base=$(basename "$f")
        case "$base" in
            libamd_smi*) ;;
            *) ln -sf "$f" "/tmp/rocm-overlay/$base" ;;
        esac
    done
fi

if [[ -d /tmp/rocm-overlay ]]; then
    # Replace the package's rocm/ dir with the overlay (which excludes
    # libamd_smi) so the system version from /opt/rocm/lib is found instead.
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//$PKG_DIR\/rocm//tmp/rocm-overlay}"
fi

if [[ -d /opt/rocm/lib ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/opt/rocm/lib"
fi

# ---------- run ----------

exec "$@"
