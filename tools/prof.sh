#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_usage() {
  echo "Usage:"
  echo "  $0 trace <command> [args...]   # Run rocprofv3 kernel trace"
  echo "  $0 att <command> [args...]     # Run rocprofv3 ATT profiling"
  exit 1
}

# Ensure /opt/rocm symlink points to the versioned ROCm directory.
# The ATT decoder installer can overwrite this symlink, breaking things.
ensure_rocm_symlink() {
  # Find the versioned ROCm directory
  local ROCM_VERSIONED
  ROCM_VERSIONED=$(ls -d /opt/rocm-* 2>/dev/null | head -1)
  
  if [[ -z "$ROCM_VERSIONED" ]]; then
    echo "Error: No versioned ROCm directory found in /opt/"
    return 1
  fi

  # Check if /opt/rocm is a valid symlink to the versioned directory
  if [[ -L /opt/rocm ]]; then
    local CURRENT_TARGET
    CURRENT_TARGET=$(readlink -f /opt/rocm)
    if [[ "$CURRENT_TARGET" == "$ROCM_VERSIONED" ]]; then
      # Symlink is correct
      return 0
    fi
    echo "[WARN] /opt/rocm points to $CURRENT_TARGET instead of $ROCM_VERSIONED"
    echo "[FIX] Correcting symlink..."
    rm -f /opt/rocm
  elif [[ -d /opt/rocm ]]; then
    echo "[WARN] /opt/rocm is a directory, not a symlink"
    echo "[FIX] Converting to symlink..."
    # Backup and remove
    mv /opt/rocm /opt/rocm_backup_$$ 2>/dev/null || rm -rf /opt/rocm
  fi

  # Create the symlink
  ln -sf "$ROCM_VERSIONED" /opt/rocm
  echo "[FIX] Created symlink: /opt/rocm -> $ROCM_VERSIONED"
}

# Set up environment for IREE with HIP
setup_hip_env() {
  # IREE_HIP_DYLIB_PATH tells IREE where to find libamdhip64.so
  export IREE_HIP_DYLIB_PATH=/opt/rocm/lib
  
  # Also add to LD_LIBRARY_PATH for the profiler
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/rocm/lib"
}

trace() {
  local OUTBASE="rocprof_ktrace"

  # Ensure ROCm symlink is correct
  ensure_rocm_symlink || exit 1
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

  # Ensure ROCm symlink is correct (ATT installer may have broken it)
  ensure_rocm_symlink || exit 1
  setup_hip_env

  # Ensure output directory's parent exists
  mkdir -p "$(dirname "$OUTBASE")"

  # If output folder exists, move aside
  if [[ -d "$OUTBASE" ]]; then
    rm -rf "${OUTBASE}_bkp"
    mv "$OUTBASE" "${OUTBASE}_bkp"
  fi

  # Check for ATT decoder
  local ATT_DECODER="/opt/rocm/libexec/rocprofiler-sdk/att/att_decoder.py"
  if [[ ! -f "$ATT_DECODER" ]]; then
    echo "[WARN] ATT decoder not found at $ATT_DECODER"
    echo "[HINT] Run ~/scripts/docker/env/att.sh to install it"
  fi

  # att.json is in the same directory as this script (tools/)
  echo "[ATT] Profiling: $*"
  echo "[ATT] Output directory: $OUTBASE"
  echo "[ATT] Using HIP library: $IREE_HIP_DYLIB_PATH"
  
  rocprofv3 \
    --preload /opt/rocm/lib/libamdhip64.so \
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
