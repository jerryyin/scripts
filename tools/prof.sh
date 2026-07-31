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

  # Never rotate or delete trace data implicitly.  The Stage 2 control plane
  # requires each capture to be archived and hashed before the next one.
  if [[ -d "$OUTBASE" ]]; then
    echo "Error: $OUTBASE already exists; archive it before the next ATT capture"
    return 1
  fi

  # The SQTT trace decoder. On therock it ships in the venv ROCm; on normal images
  # att.sh installs it under /opt/rocm/lib. Pass its dir explicitly via
  # --att-library-path so it is found regardless of whether /opt/rocm exists.
  local DECODER="$ROCM_DIR/lib/librocprof-trace-decoder.so"
  if [[ ! -f "$DECODER" ]]; then
    echo "[WARN] trace decoder not found at $DECODER"
    echo "[HINT] normal ROCm images: run ~/scripts/docker/env/att.sh to install it"
  fi

  # Optional capture overrides.  Kernel iteration selection is important for ATT:
  # without it rocprofiler normally traces the first matching kernel occurrence,
  # which may be a cold/warmup dispatch rather than the intended steady-state one.
  local ATT_CFG_SOURCE="${ATT_CONFIG_PATH:-$SCRIPT_DIR/att.json}"
  local ATT_CFG="$ATT_CFG_SOURCE"
  local GENERATED_ATT_CFG=0
  if [[ ! -f "$ATT_CFG_SOURCE" ]]; then
    echo "Error: ATT config does not exist: $ATT_CFG_SOURCE"
    return 1
  fi
  if [[ -n "${ATT_KERNEL_REGEX:-}" || -n "${ATT_KERNEL_ITERATION_RANGE:-}" || \
        -n "${ATT_TARGET_CU:-}" || -n "${ATT_SHADER_ENGINE_MASK:-}" || \
        -n "${ATT_SIMD_SELECT:-}" || -n "${ATT_PERFCOUNTERS+x}" || \
        -n "${ATT_PERFCOUNTER_CTRL:-}" ]]; then
    ATT_CFG="$(mktemp --suffix=.att.json)"
    GENERATED_ATT_CFG=1
    python3 - "$ATT_CFG_SOURCE" "$ATT_CFG" <<'PY'
import json
import os
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as stream:
    config = json.load(stream)
for job in config["jobs"]:
    if os.environ.get("ATT_KERNEL_REGEX"):
        job["kernel_include_regex"] = os.environ["ATT_KERNEL_REGEX"]
    if os.environ.get("ATT_KERNEL_ITERATION_RANGE"):
        job["kernel_iteration_range"] = os.environ["ATT_KERNEL_ITERATION_RANGE"]
    if os.environ.get("ATT_TARGET_CU"):
        job["att_target_cu"] = int(os.environ["ATT_TARGET_CU"], 0)
    if os.environ.get("ATT_SHADER_ENGINE_MASK"):
        job["att_shader_engine_mask"] = os.environ["ATT_SHADER_ENGINE_MASK"]
    if os.environ.get("ATT_SIMD_SELECT"):
        job["att_simd_select"] = int(os.environ["ATT_SIMD_SELECT"], 0)
    if "ATT_PERFCOUNTERS" in os.environ:
        counters = [name.strip() for name in os.environ["ATT_PERFCOUNTERS"].split(",") if name.strip()]
        job["att_perfcounters"] = ", ".join(counters)
        if counters and "ATT_PERFCOUNTER_CTRL" not in os.environ:
            job["att_perfcounter_ctrl"] = 3
    if os.environ.get("ATT_PERFCOUNTER_CTRL"):
        job["att_perfcounter_ctrl"] = int(os.environ["ATT_PERFCOUNTER_CTRL"], 0)
with open(destination, "w", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
  fi
  if [[ -n "${ATT_KERNEL_REGEX:-}" ]]; then
    echo "[ATT] kernel filter: $ATT_KERNEL_REGEX"
  fi
  if [[ -n "${ATT_KERNEL_ITERATION_RANGE:-}" ]]; then
    echo "[ATT] kernel iteration range: $ATT_KERNEL_ITERATION_RANGE"
  fi
  if [[ -n "${ATT_TARGET_CU:-}" || -n "${ATT_SHADER_ENGINE_MASK:-}" || -n "${ATT_SIMD_SELECT:-}" ]]; then
    echo "[ATT] placement: target_cu=${ATT_TARGET_CU:-config} se_mask=${ATT_SHADER_ENGINE_MASK:-config} simd=${ATT_SIMD_SELECT:-config}"
  fi
  if [[ -n "${ATT_PERFCOUNTERS+x}" ]]; then
    echo "[ATT] performance counters: ${ATT_PERFCOUNTERS:-<none>}"
  fi

  echo "[ATT] Profiling: $*"
  echo "[ATT] Output directory: $OUTBASE"
  echo "[ATT] ROCm: $ROCM_DIR"

  local ROCPROF_STATUS=0
  rocprofv3 \
    --att-library-path "$ROCM_DIR/lib" \
    --preload "$ROCM_DIR/lib/libamdhip64.so" \
    -i "$ATT_CFG" \
    -d "$OUTBASE" -- "$@" || ROCPROF_STATUS=$?

  if [[ -d "$OUTBASE" ]]; then
    cp "$ATT_CFG" "$OUTBASE/effective_att_config.json"
  fi
  [[ $GENERATED_ATT_CFG -eq 1 ]] && rm -f "$ATT_CFG"
  if [[ $ROCPROF_STATUS -ne 0 ]]; then
    return "$ROCPROF_STATUS"
  fi

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
