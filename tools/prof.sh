#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_usage() {
  echo "Usage:"
  echo "  $0 trace <command> [args...]   # Run rocprofv3 kernel trace"
  echo "  $0 att <command> [args...]     # Run rocprofv3 ATT profiling"
  exit 1
}

# Resolve the ROCm root WITHOUT mutating the filesystem, supporting both layouts:
#   - normal ROCm image:   /opt/rocm  (or a versioned /opt/rocm-*)
#   - therock image:       ROCm lives in the venv as _rocm_sdk_devel, NO /opt/rocm
# Sets global ROCM_DIR. rocprofv3 ATT does NOT need /opt/rocm to exist as long as
# --preload / --att-library-path / LD_LIBRARY_PATH point at the right lib dir, so we
# never create a symlink here (forcing /opt/rocm in base setup is too invasive).
resolve_rocm_dir() {
  if [[ -e /opt/rocm ]]; then
    ROCM_DIR=$(readlink -f /opt/rocm)
  else
    local versioned venv
    versioned=$(ls -d /opt/rocm-* 2>/dev/null | head -1)
    venv=$(ls -d /opt/venv/lib/python*/site-packages/_rocm_sdk_devel 2>/dev/null | head -1)
    if [[ -n "$versioned" ]]; then
      ROCM_DIR=$(readlink -f "$versioned")
    elif [[ -n "$venv" ]]; then
      ROCM_DIR="$venv"   # therock layout
    else
      echo "Error: no ROCm found (/opt/rocm, /opt/rocm-*, or venv _rocm_sdk_devel)"
      return 1
    fi
  fi
  echo "[ROCm] using ROCM_DIR=$ROCM_DIR"
}

# Set up environment for the profiler / IREE with HIP.
setup_hip_env() {
  # IREE_HIP_DYLIB_PATH tells IREE where to find libamdhip64.so
  export IREE_HIP_DYLIB_PATH="$ROCM_DIR/lib"
  # Also add to LD_LIBRARY_PATH for the profiler and the trace decoder
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ROCM_DIR/lib"
}

trace() {
  local OUTBASE="rocprof_ktrace"

  resolve_rocm_dir || exit 1
  setup_hip_env

  rocprofv3 --kernel-trace -o "$OUTBASE" --output-format csv -- "$@"
  local TRACE_FILE="${OUTBASE}_kernel_trace.csv"

  if [[ ! -f "$TRACE_FILE" ]]; then
    echo "Error: Kernel trace file not found!"
    exit 1
  fi

  awk -F',' 'NR>1 {
    runtime_us = ($11 - $10)/1000.0
    printf "%s:\n", $8
    printf "  runtime: %.3f us\n", runtime_us
    printf "  block_size: %s\n", $12
    printf "  scratch_size: %s\n", $13
    printf "  vgpr_count: %s\n", $14
    printf "  accum_vgpr_count: %s\n", $15
    printf "  sgpr_count: %s\n", $16
    printf "  workgroup_size: %sx%sx%s\n", $17, $18, $19
    printf "  grid_size: %sx%sx%s\n", $20, $21, $22
    print ""
  }' "$TRACE_FILE"

  rm -f ${OUTBASE}_*.csv
}

att() {
  local OUTBASE="/zyin/rocprof_att"

  resolve_rocm_dir || exit 1
  setup_hip_env

  # Ensure output directory's parent exists
  mkdir -p "$(dirname "$OUTBASE")"

  # If output folder exists, move aside
  if [[ -d "$OUTBASE" ]]; then
    rm -rf "${OUTBASE}_bkp"
    mv "$OUTBASE" "${OUTBASE}_bkp"
  fi

  # The SQTT trace decoder. On therock it ships in the venv ROCm; on normal images
  # att.sh installs it under /opt/rocm/lib. Pass its dir explicitly via
  # --att-library-path so it is found regardless of whether /opt/rocm exists.
  local DECODER="$ROCM_DIR/lib/librocprof-trace-decoder.so"
  if [[ ! -f "$DECODER" ]]; then
    echo "[WARN] trace decoder not found at $DECODER"
    echo "[HINT] normal ROCm images: run ~/scripts/docker/env/att.sh to install it"
  fi

  echo "[ATT] Profiling: $*"
  echo "[ATT] Output directory: $OUTBASE"
  echo "[ATT] ROCm: $ROCM_DIR"

  rocprofv3 \
    --att-library-path "$ROCM_DIR/lib" \
    --preload "$ROCM_DIR/lib/libamdhip64.so" \
    -i "$SCRIPT_DIR/att.json" \
    -d "$OUTBASE" -- "$@"

  # Check if output was generated
  if [[ -d "$OUTBASE" ]]; then
    echo ""
    echo "[ATT] Output generated in: $OUTBASE"
    echo "[ATT] Files:"
    ls -la "$OUTBASE"/*.csv 2>/dev/null || echo "  (no CSV files)"
    ls -la "$OUTBASE"/*.json 2>/dev/null || echo "  (no JSON files)"
  else
    echo "[ATT] Warning: No output directory created"
  fi
}

# --- Main entry point ---
if [[ $# -lt 2 ]]; then
  show_usage
fi

MODE="$1"
shift

case "$MODE" in
  trace) trace "$@" ;;
  att) att "$@" ;;
  *)       show_usage ;;
esac
