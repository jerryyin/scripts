#!/usr/bin/env bash
# =============================================================================
# run_decode_itrace.sh
#
# End-to-end, IDEMPOTENT capture of the decode a8w4 MoE GEMM1 instruction trace
# (itrace) under the AM model, for the triton and gluon backends, then extract +
# visualize + analyze per-WGP00.
#
# Pipeline (see MOE_DECODE_ITRACE_CHRONICLE.md for the why behind every step):
#   1. ensure `m4` exists            (AM model bring-up needs it)
#   2. enable itrace in am_env.sh    (sed; reverted on exit if we turned it on)
#   3. precompute routing+weights    (CPU only; skipped if the .pt already exists)
#   4. run GEMM1-only under AM        per backend (skipped if a good trace exists)
#   5. extract WGP00 + gen HTML + analyze instruction mix
#
# Idempotency: every expensive artifact (the .pt payload, each .mon trace, each
# .html) is reused if already present. Re-running only does the missing work.
#
# Known result (documented in the chronicle):
#   - triton decode: traces cleanly.
#   - gluon  decode: aborts AM on the TDM async-copy tracker
#     (tcp.cpp:4894 "Can't find tracker table entry for async direct copy").
#     The tracker depth `tcp_async_copy_depth` in
#     /am-ffm/package/etc/am/conf/model.conf (lines ~3566=128, ~3821=256) cannot
#     be raised enough (2048 -> invalid/param-parse FATAL; 256 -> still aborts),
#     so this script does NOT edit model.conf; it captures gluon's partial
#     (prologue) trace and reports the FATAL.
# =============================================================================

set -uo pipefail   # NOT -e: the gluon run is expected to FATAL; handled explicitly.

# -------------------------- configuration ------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_HOME="${AITER_HOME:-/root/aiter}"
OUT_ROOT="${OUT_ROOT:-/root/itrace_runs}"
PKG="${PKG:-/am-ffm}"
AM_ENV="$PKG/am_env.sh"
RUNNER="${RUNNER:-$HOME/scripts/tools/run_on_model.sh}"
ITRACEVIZ="${ITRACEVIZ:-/root/ItraceViz}"

# Decode shape. K x N = 2048 x 7168 are the ticket dims (keep them: they set the
# per-tile/per-WGP instruction mix). EXPERTS_TOT=32 gives block_m=16 (decode
# tiling) with ~1/8 the tiles of the ticket's 256 -> tractable under cycle-
# accurate AM and a representative steady-state WGP00. Set to 256 8 / batch 128
# for the ticket-exact (very large / slow) run.
SHAPE_K="${SHAPE_K:-2048}"
SHAPE_N="${SHAPE_N:-7168}"
EXPERTS_TOT="${EXPERTS_TOT:-32}"
EXPERTS_ACT="${EXPERTS_ACT:-8}"
BATCH="${BATCH:-64}"
BACKENDS="${BACKENDS:-triton gluon}"
WGP="${WGP:-0}"                       # which WGP to extract/visualize
AM_TIMEOUT="${AM_TIMEOUT:-5400}"      # hard cap per AM run (seconds)

PAYLOAD="$OUT_ROOT/moe_decode_${EXPERTS_TOT}x${EXPERTS_ACT}_b${BATCH}.pt"

# AM needs these; bind LD_PRELOAD so run_on_model.sh's `set -u` doesn't trip.
export GPU_ARCHS="${GPU_ARCHS:-gfx1250}"
export LD_PRELOAD="${LD_PRELOAD:-}"
export AITER_HOME

log()  { printf '\n========== %s ==========\n' "$*"; }
info() { printf '  -> %s\n' "$*"; }

# -------------------------- 1. m4 dependency ---------------------------------
ensure_m4() {
  log "step 1: ensure m4 (AM model bring-up requires it)"
  if [[ -x /usr/bin/m4 ]]; then info "m4 present"; return 0; fi
  info "m4 missing -> apt-get install -y m4"
  apt-get install -y m4 >/dev/null 2>&1 || { echo "ERROR: m4 install failed"; exit 1; }
  [[ -x /usr/bin/m4 ]] && info "m4 installed" || { echo "ERROR: m4 still missing"; exit 1; }
}

# -------------------------- 2. itrace env toggle (sed) -----------------------
itrace_is_enabled() { grep -qE '^[[:space:]]*"test\.enable_itrace=true"' "$AM_ENV"; }

enable_itrace() {
  log "step 2: enable itrace in $AM_ENV (sed; idempotent)"
  PRE_ENABLED=0; itrace_is_enabled && PRE_ENABLED=1
  if [[ "$PRE_ENABLED" == 1 ]]; then info "itrace already enabled (will leave as-is on exit)"; return 0; fi
  # drop -no_itrace, then uncomment the two flags
  sed -i 's/^export DtifExtraTestArgs="-no_itrace"/export DtifExtraTestArgs=""/' "$AM_ENV"
  sed -i 's/^\([[:space:]]*\)#"test\.enable_itrace=true"/\1"test.enable_itrace=true"/' "$AM_ENV"
  sed -i 's/^\([[:space:]]*\)#"test\.itrace_perf_detail=true"/\1"test.itrace_perf_detail=true"/' "$AM_ENV"
  itrace_is_enabled && info "itrace enabled" || { echo "ERROR: failed to enable itrace"; exit 1; }
}

revert_itrace() {
  # Only revert if WE turned it on (preserve a pre-existing enabled state).
  [[ "${PRE_ENABLED:-1}" == 1 ]] && return 0
  info "reverting itrace in $AM_ENV (sed)"
  sed -i 's/^export DtifExtraTestArgs=""/export DtifExtraTestArgs="-no_itrace"/' "$AM_ENV"
  sed -i 's/^\([[:space:]]*\)"test\.enable_itrace=true"/\1#"test.enable_itrace=true"/' "$AM_ENV"
  sed -i 's/^\([[:space:]]*\)"test\.itrace_perf_detail=true"/\1#"test.itrace_perf_detail=true"/' "$AM_ENV"
}
trap revert_itrace EXIT

# -------------------------- device serialization -----------------------------
# The AM model is a single simulated device; a killed/stuck run doesn't release
# it instantly. Wait until no GEMM1 python process remains. (pgrep -f also
# matches this shell, so filter to comm==python3.)
free_device() {
  local p tries=0
  while :; do
    local found=0
    for p in $(pgrep -f "run_a8w4_gemm1.py" 2>/dev/null); do
      [[ "$(cat /proc/$p/comm 2>/dev/null)" == "python3" ]] && { kill -9 "$p" 2>/dev/null; found=1; }
    done
    [[ "$found" == 0 ]] && { [[ $tries -gt 0 ]] && sleep 3; return 0; }
    tries=$((tries+1)); sleep 4
    [[ $tries -gt 30 ]] && { echo "WARN: device still busy after kills"; return 0; }
  done
}

# -------------------------- 3. precompute (CPU, idempotent) -------------------
precompute() {
  log "step 3: precompute routing(cpu)+weights -> $PAYLOAD"
  if [[ -s "$PAYLOAD" ]]; then info "payload exists ($(du -h "$PAYLOAD" | cut -f1)) -> skip"; return 0; fi
  info "building (CPU only, fabricated mxfp4 weights)..."
  GPU_ARCHS="$GPU_ARCHS" python3 "$SCRIPT_DIR/../precompute_routing.py" \
      --out "$PAYLOAD" --shape "$SHAPE_K" "$SHAPE_N" \
      --experts "$EXPERTS_TOT" "$EXPERTS_ACT" --batch "$BATCH" \
      2>&1 | grep -E "arch=|weights:|saved|WARNING|Error" || true
  [[ -s "$PAYLOAD" ]] || { echo "ERROR: precompute did not produce $PAYLOAD"; exit 1; }
}

# -------------------------- 4. one AM backend run (idempotent) ---------------
# Launches the GEMM1-only AM run, then monitors the log for success
# ("GEMM1 done") or the known TDM FATAL, killing the (otherwise hanging) process
# as soon as the outcome is known.
run_backend() {
  local be="$1"
  local dir="$OUT_ROOT/decode_${be}"
  local mon="$dir/xcc0se0sa0_itrace_emu.mon"
  local logf="$dir/run.log"

  log "step 4: AM itrace -- backend=$be"
  if [[ -s "$mon" ]] && grep -qE "GEMM1 done|Can't find tracker table entry" "$logf" 2>/dev/null; then
    info "trace already present ($(du -h "$mon" | cut -f1)) -> skip"
    return 0
  fi

  free_device
  rm -rf "$dir"; mkdir -p "$dir"
  info "running (cwd=$dir) ..."
  ( cd "$dir" && exec env LD_PRELOAD= GPU_ARCHS="$GPU_ARCHS" \
        "$RUNNER" --backend am -- \
        python3 "$SCRIPT_DIR/../run_a8w4_gemm1.py" --backend "$be" --data "$PAYLOAD" \
  ) >"$logf" 2>&1 &
  local pid=$!

  local waited=0 status="running"
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "GEMM1 done" "$logf" 2>/dev/null; then status="done"; break; fi
    if grep -qE "Can't find tracker table entry|signal 6|\[FATAL\]|Traceback" "$logf" 2>/dev/null; then status="fatal"; break; fi
    sleep 10; waited=$((waited+10))
    [[ $waited -ge $AM_TIMEOUT ]] && { status="timeout"; break; }
  done
  # The process can die (SIGABRT on FATAL, or normal exit) between polls, exiting
  # the loop with status still "running" -- reclassify from the log.
  if [[ "$status" == "running" ]]; then
    if grep -q "GEMM1 done" "$logf" 2>/dev/null; then status="done"
    elif grep -qE "Can't find tracker table entry|signal 6|\[FATAL\]|Traceback" "$logf" 2>/dev/null; then status="fatal"
    else status="exited"; fi
  fi
  sleep 2   # let the last trace bytes flush
  free_device

  case "$status" in
    done)    info "backend=$be: GEMM1 completed; trace $(du -h "$mon" 2>/dev/null | cut -f1)" ;;
    fatal)   info "backend=$be: AM FATAL (expected for gluon TDM); partial trace $(du -h "$mon" 2>/dev/null | cut -f1)";
             grep -m1 -E "Can't find tracker table entry|\[FATAL\]" "$logf" | sed 's/^/       /' ;;
    timeout) info "backend=$be: hit AM_TIMEOUT=${AM_TIMEOUT}s; using partial trace" ;;
    *)       info "backend=$be: ended ($status)" ;;
  esac
}

# -------------------------- 5. extract + visualize + analyze ------------------
postprocess() {
  local be="$1"
  local dir="$OUT_ROOT/decode_${be}"
  local mon="$dir/xcc0se0sa0_itrace_emu.mon"
  local wgptxt="$dir/wgp$(printf '%02d' "$WGP")_${be}.txt"
  local html="$dir/${be}_decode_wgp$(printf '%02d' "$WGP").html"

  log "step 5: postprocess -- backend=$be"
  [[ -s "$mon" ]] || { info "no trace for $be -> skip"; return 0; }

  if [[ ! -s "$html" ]]; then
    info "extract WGP$(printf '%02d' "$WGP") + gen_timeline -> $html"
    grep -A1 "WGP$(printf '%02d' "$WGP")" "$mon" > "$wgptxt" 2>/dev/null || true
    python3 "$ITRACEVIZ/gen_timeline.py" "$wgptxt" "$html" >/dev/null 2>&1 \
      && info "html: $html ($(du -h "$html" | cut -f1))" \
      || info "gen_timeline failed (trace may be empty)"
  else
    info "html exists -> skip ($html)"
  fi

  info "instruction mix (WGP$(printf '%02d' "$WGP")):"
  python3 "$SCRIPT_DIR/itrace_analyze.py" mix "$mon" "$WGP" 2>/dev/null | sed 's/^/     /' || true
}

# -------------------------------- main ---------------------------------------
mkdir -p "$OUT_ROOT"
ensure_m4
enable_itrace
precompute
for be in $BACKENDS; do
  run_backend "$be"
  postprocess "$be"
done
log "done. artifacts under $OUT_ROOT/decode_<backend>/  (HTML timelines + run.log)"
