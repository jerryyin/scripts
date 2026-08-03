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
#   run_on_model.sh --capture -- ./hip_tdm_1d 3
#
#   # Inside docker:
#   docker exec my-container /path/to/run_on_model.sh -- python3 k.py
#
# Options:
#   --backend am|ffm    Which simulator backend. Auto-detected if omitted:
#                        AM if /am-ffm exists, FFM otherwise.
#   --am-profile NAME   AM environment: trace (default), profile (1-XCC,
#                        no trace, per-dispatch counters), or 8xcc-profile
#                        (8-XCC, no trace, per-dispatch counters).
#   --capture           Capture an AQL packet trace (.cap file) via roccap.
#                        Forces FFM backend. Output: roc_capture_<binary>.cap
#   -- COMMAND [ARGS...] Everything after -- is the command to run.
set -euo pipefail

BACKEND=""
CAPTURE=0
AM_PROFILE="trace"
AM_PROFILE_SET=0

usage() {
    sed -n '2,/^set /{ /^#/s/^# \?//p }' "$0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)  BACKEND="$2"; shift 2 ;;
        --am-profile) AM_PROFILE="$2"; AM_PROFILE_SET=1; shift 2 ;;
        --capture)  CAPTURE=1; shift ;;
        -h|--help)  usage ;;
        --)         shift; break ;;
        *)          echo "Unknown option: $1" >&2; usage ;;
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

# ---------- capture mode forces FFM ----------

if [[ $CAPTURE -eq 1 && -n "$BACKEND" && "$BACKEND" != "ffm" ]]; then
    echo "Warning: --capture forces FFM backend (ignoring --backend $BACKEND)" >&2
fi
if [[ $CAPTURE -eq 1 ]]; then
    BACKEND=ffm
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
if [[ "$BACKEND" != "am" && "$AM_PROFILE_SET" -eq 1 ]]; then
    echo "Error: --am-profile is valid only with the AM backend" >&2
    exit 1
fi

# The vendor env scripts (am_env.sh / ffmlite_env.sh) append to $LD_PRELOAD by
# reading it. Under `set -u` that aborts ("LD_PRELOAD: unbound variable") if the
# caller never exported it. Define it (empty) here so callers don't have to.
export LD_PRELOAD="${LD_PRELOAD:-}"

case "$BACKEND" in
    am)
        case "$AM_PROFILE" in
            trace) AM_ENV_FILE="$PKG_DIR/am_env.sh" ;;
            profile) AM_ENV_FILE="$PKG_DIR/am_profile_env.sh" ;;
            8xcc-profile) AM_ENV_FILE="$PKG_DIR/am_8xcc_env.sh" ;;
            *)
                echo "Error: --am-profile must be trace, profile, or 8xcc-profile; got '$AM_PROFILE'" >&2
                exit 1
                ;;
        esac
        if [[ ! -f "$AM_ENV_FILE" ]]; then
            echo "Error: AM profile $AM_PROFILE requires $AM_ENV_FILE" >&2
            exit 1
        fi
        source "$AM_ENV_FILE"

        # The vendor 8-XCC file enables ttrace and lacks per-dispatch dumps.
        # Fidelity timing needs the same topology with both instruction/timing
        # traces disabled and a counter snapshot at every dispatch boundary.
        if [[ "$AM_PROFILE" == "8xcc-profile" ]]; then
            export DtifExtraModelArgs="-om=am"
            export DtifExtraTestArgs="-tg_chunksize=2 -num_xcds=8 -no_itrace"
            DtifGeneralArgs="${DtifGeneralArgs//test.enable_ttrace=true/test.enable_ttrace=false}"
            DtifGeneralArgs="${DtifGeneralArgs%\"} monitors.counters.perf.dump_on_draw=true\""
            export DtifGeneralArgs
        fi

        case "$AM_PROFILE" in
            trace)
                PROFILE_XCC=1
                PROFILE_CP=1
                PROFILE_TRACE=1
                PROFILE_DISPATCH_COUNTERS=0
                PROFILE_COUNTER_SOURCE="am-log-dispatch-clock"
                PROFILE_MODEL_CONFIG="make_mi400_16cu_2se_1xcc_cu_cache_l0_64k_lds_320k"
                ;;
            profile)
                PROFILE_XCC=1
                PROFILE_CP=1
                PROFILE_TRACE=0
                PROFILE_DISPATCH_COUNTERS=1
                PROFILE_COUNTER_SOURCE="am-log-dispatch-clock+per-dispatch-counters"
                PROFILE_MODEL_CONFIG="make_mi400_16cu_2se_1xcc_cu_cache_l0_64k_lds_320k"
                ;;
            8xcc-profile)
                PROFILE_XCC=8
                PROFILE_CP=8
                PROFILE_TRACE=0
                PROFILE_DISPATCH_COUNTERS=1
                PROFILE_COUNTER_SOURCE="am-log-dispatch-clock+per-dispatch-counters"
                PROFILE_MODEL_CONFIG="make_mi400_16cu_2se_8xcc_8cp_cu_cache_l0_64k_lds_320k"
                ;;
        esac

        if [[ "$PROFILE_TRACE" -eq 0 ]]; then
            [[ "$DtifExtraModelArgs" != *sq_ttrace* ]] || {
                echo "Error: AM profile $AM_PROFILE unexpectedly enables ttrace" >&2
                exit 1
            }
            [[ "$DtifExtraTestArgs" == *-no_itrace* ]] || {
                echo "Error: AM profile $AM_PROFILE does not disable itrace" >&2
                exit 1
            }
        fi
        if [[ "$PROFILE_TRACE" -eq 1 ]]; then
            [[ "$DtifExtraModelArgs" == *sq_ttrace* && "$DtifGeneralArgs" == *test.enable_ttrace=true* ]] || {
                echo "Error: AM trace profile does not enable ttrace" >&2
                exit 1
            }
        fi
        if [[ "$PROFILE_DISPATCH_COUNTERS" -eq 1 ]]; then
            [[ "$DtifGeneralArgs" == *monitors.counters.perf.dump_on_draw=true* ]] || {
                echo "Error: AM profile $AM_PROFILE lacks per-dispatch counters" >&2
                exit 1
            }
        fi
        if [[ "$PROFILE_XCC" -eq 8 ]]; then
            [[ "${DtifNumXcc:-}" == "8" && "$DtifGeneralArgs" == *make_mi400_16cu_2se_8xcc_8cp_cu_cache_l0_64k_lds_320k* ]] || {
                echo "Error: AM profile $AM_PROFILE failed its 8-XCC topology validation" >&2
                exit 1
            }
        else
            [[ -z "${DtifNumXcc:-}" && "$DtifGeneralArgs" == *make_mi400_16cu_2se_1xcc_cu_cache_l0_64k_lds_320k* ]] || {
                echo "Error: AM profile $AM_PROFILE failed its 1-XCC topology validation" >&2
                exit 1
            }
        fi

        export RUN_ON_MODEL_AM_PROFILE="$AM_PROFILE"
        export RUN_ON_MODEL_AM_ENV_FILE="$AM_ENV_FILE"
        export RUN_ON_MODEL_AM_MODEL_CONFIG="$PROFILE_MODEL_CONFIG"
        export RUN_ON_MODEL_AM_TOPOLOGY_XCC="$PROFILE_XCC"
        export RUN_ON_MODEL_AM_TOPOLOGY_CP="$PROFILE_CP"
        export RUN_ON_MODEL_AM_TOPOLOGY_SE_PER_XCC=2
        export RUN_ON_MODEL_AM_TOPOLOGY_CU_PER_XCC=16
        # make_mi400_base fixes sclk to 555 ps (documented as 1.8 GHz).
        export RUN_ON_MODEL_AM_CLOCK_PERIOD_PS=555
        export RUN_ON_MODEL_AM_COUNTER_SOURCE="$PROFILE_COUNTER_SOURCE"
        export RUN_ON_MODEL_AM_TRACE_ENABLED="$PROFILE_TRACE"
        export RUN_ON_MODEL_AM_PER_DISPATCH_COUNTERS="$PROFILE_DISPATCH_COUNTERS"
        PROFILE_PACKAGE_VERSION=$(sed -n 's/^Package Name[[:space:]]*:[[:space:]]*//p' "$PKG_DIR/VERSION" | head -1)
        PROFILE_MODEL_VERSION=$(sed -n 's/^Model Version[[:space:]]*:[[:space:]]*//p' "$PKG_DIR/VERSION" | head -1)
        [[ -n "$PROFILE_PACKAGE_VERSION" && -n "$PROFILE_MODEL_VERSION" ]] || {
            echo "Error: cannot parse AM package/model version from $PKG_DIR/VERSION" >&2
            exit 1
        }
        export RUN_ON_MODEL_AM_PACKAGE_VERSION="$PROFILE_PACKAGE_VERSION"
        export RUN_ON_MODEL_AM_MODEL_VERSION="$PROFILE_MODEL_VERSION"
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

if [[ "$BACKEND" == "am" ]]; then
    echo "[run_on_model] pkg=$PKG_DIR package_version=$RUN_ON_MODEL_AM_PACKAGE_VERSION model_version=$RUN_ON_MODEL_AM_MODEL_VERSION backend=$BACKEND profile=$RUN_ON_MODEL_AM_PROFILE env=$RUN_ON_MODEL_AM_ENV_FILE topology=${RUN_ON_MODEL_AM_TOPOLOGY_XCC}xcc/${RUN_ON_MODEL_AM_TOPOLOGY_CP}cp/${RUN_ON_MODEL_AM_TOPOLOGY_SE_PER_XCC}se/${RUN_ON_MODEL_AM_TOPOLOGY_CU_PER_XCC}cu clock_period_ps=$RUN_ON_MODEL_AM_CLOCK_PERIOD_PS model_config=$RUN_ON_MODEL_AM_MODEL_CONFIG counter_source=$RUN_ON_MODEL_AM_COUNTER_SOURCE cwd=$PWD" >&2
else
    echo "[run_on_model] pkg=$PKG_DIR backend=$BACKEND" >&2
fi

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

if [[ $CAPTURE -eq 1 ]]; then
    export HSA_KMT_MODEL_GPUVM_BASE=0x200000000
    export HSA_KMT_MODEL_GPUVM_SIZE=0xF00000000

    ROCCAP=""
    for candidate in "$PKG_DIR/tools/roccap/bin/roccap" \
                     "$(command -v roccap 2>/dev/null)"; do
        if [[ -x "$candidate" ]]; then
            ROCCAP="$candidate"
            break
        fi
    done
    if [[ -z "$ROCCAP" ]]; then
        echo "Error: roccap not found in $PKG_DIR/tools/roccap/bin/ or PATH" >&2
        exit 1
    fi

    echo "[run_on_model] capture via $ROCCAP" >&2
    exec "$ROCCAP" capture --loglevel info "$@"
fi

# ---------- FFM teardown fix ----------
# FFM simulator threads don't shut down during Py_Finalize, causing the process
# to hang indefinitely after tests pass.  For pytest: inject a plugin that calls
# hipDeviceReset + os._exit.  For plain python: wrap with a small -c shim.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$1" in
    pytest|*/pytest)
        export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:+$PYTHONPATH}"
        exec "$@" -p ffm_teardown
        ;;
    python3|python|*/python3|*/python)
        exec "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
