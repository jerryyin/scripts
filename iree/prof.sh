#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_usage() {
  echo "Usage:"
  echo "  $0 trace <command> [args...]   # Run rocprofv3 kernel trace"
  echo "  $0 att <command> [args...] # Run rocprofv3 ATT profiling"
  exit 1
}

trace() {
  local OUTBASE="rocprof_ktrace"

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

  # If output folder exists, move aside
  if [[ -d "$OUTBASE" ]]; then
    rm -rf "${OUTBASE}_bkp"
    mv "$OUTBASE" "${OUTBASE}_bkp"
  fi

  # Use /opt/rocm (symlinked to ROCm 7.0) for all profiling
  # IREE_HIP_DYLIB_PATH: tells IREE where to find libamdhip64.so
  export IREE_HIP_DYLIB_PATH=/opt/rocm/lib

  /opt/rocm/bin/rocprofv3 \
    --preload /opt/rocm/lib/libamdhip64.so \
    -i "$SCRIPT_DIR/att.json" \
    -d "$OUTBASE" -- "$@"
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

