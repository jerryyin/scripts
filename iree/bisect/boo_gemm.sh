#!/usr/bin/env bash
set -euo pipefail

# Thresholds (microseconds)
THRESH_BAD=55000.0    # > this => bad
THRESH_GOOD=40000.0   # < this => good

# Where to run the benchmark (exact path you provided)
BENCH_CWD="/root/iree-turbine/iree/turbine/kernel/boo/driver"
BENCH_CMD=(
  python driver.py
  --backend iree_boo_experimental
  --verbose
  aten::addmm
  "[[4096], [150000, 16384], [16384, 4096], [], []]"
  "['c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16', 'Scalar', 'Scalar']"
  "[[1], [16384, 1], [1, 16384], [], []]"
  "['', '', '', '1', '1']"
)

BUILD_CMD="git submodule update --init && cmake --build ./build/model"

# Logging
LOG_DIR="/tmp/git-bisect-logs"
mkdir -p "$LOG_DIR"
COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LOG_FILE="$LOG_DIR/${COMMIT_HASH}_$(date +%Y%m%dT%H%M%S).log"

# Timeout for the benchmark run (seconds) — protects against hangs
TIMEOUT_SECS=600

# ----------------- Helpers -----------------
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_FILE" >&2; }

fail_skip() {
  log "Skipping commit (exit 125)"
  exit 125
}

# ----------------- Start -----------------
log "==== git-bisect test start ===="
log "Commit: $COMMIT_HASH"
log "Log file: $LOG_FILE"

# Ensure we are at repo root so git submodule update runs in correct repo
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -n "$REPO_ROOT" ]; then
  cd "$REPO_ROOT"
  log "Changed to repo root: $REPO_ROOT"
fi

log "Updating submodules and building"
if ! bash -lc "$BUILD_CMD" 2>&1 | tee -a "$LOG_FILE"; then
  log "Build step failed. See log above."
  fail_skip
fi

if [ ! -d "$BENCH_CWD" ]; then
  log "Benchmark directory does not exist: $BENCH_CWD"
  fail_skip
fi

log "Running benchmark in $BENCH_CWD"
cd "$BENCH_CWD"

# Run the benchmark with timeout; capture stdout/stderr to a temp file and log it.
OUTFILE=$(mktemp)
trap 'rm -f "$OUTFILE"' EXIT

if ! timeout "$TIMEOUT_SECS" env BOO_CACHE_ON=0 ROCR_VISIBLE_DEVICES=3,4 "${BENCH_CMD[@]}" >"$OUTFILE" 2>&1; then
  rc=$?
  log "Benchmark command failed or timed out (rc=$rc). Dumping output to log."
  sed -n '1,200p' "$OUTFILE" | sed -n '1,200p' >> "$LOG_FILE"
  fail_skip
fi

# Save full run output to the log (tail so logs don't explode)
log "Benchmark output (tail 200 lines):"
tail -n 200 "$OUTFILE" | tee -a "$LOG_FILE"

# 3) Extract mean=...us from output
# Look for patterns like: mean=60869.60us  or mean=60869.597416666635us
MEAN_VAL_RAW=$(grep -oP 'mean=\s*\K[0-9]+(?:\.[0-9]+)?(?=us)' "$OUTFILE" | head -n 1 || true)

if [ -z "$MEAN_VAL_RAW" ]; then
  # fallback: try the "Per-launch GPU mean time (...): 60869.5974us" style
  MEAN_VAL_RAW=$(grep -oP 'Per-launch GPU mean time.*:\s*\K[0-9]+(?:\.[0-9]+)?(?=us)' "$OUTFILE" | head -n 1 || true)
fi

if [ -z "$MEAN_VAL_RAW" ]; then
  log "Failed to extract mean value from benchmark output. Skipping commit."
  fail_skip
fi

# Ensure numeric format
MEAN_VAL=$(printf '%s' "$MEAN_VAL_RAW" | sed 's/,//g')
log "Extracted mean (microseconds): $MEAN_VAL"

# 4) Decision logic using numeric comparisons (awk handles numeric compare reliably)
is_bad=$(awk -v m="$MEAN_VAL" -v t="$THRESH_BAD" 'BEGIN{print (m > t) ? 1 : 0}')
is_good=$(awk -v m="$MEAN_VAL" -v t="$THRESH_GOOD" 'BEGIN{print (m < t) ? 1 : 0}')

if [ "$is_bad" -eq 1 ]; then
  log "Mean $MEAN_VAL > $THRESH_BAD -> classified as BAD (regression). Exiting 1."
  exit 1
fi

if [ "$is_good" -eq 1 ]; then
  log "Mean $MEAN_VAL < $THRESH_GOOD -> classified as GOOD. Exiting 0."
  exit 0
fi

# Ambiguous case: between thresholds -> skip
log "Mean $MEAN_VAL is between $THRESH_GOOD and $THRESH_BAD -> ambiguous. Skipping commit."
fail_skip

