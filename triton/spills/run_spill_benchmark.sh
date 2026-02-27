#!/bin/bash
# Benchmark: Measure how AM execution speed responds to register spills
# by running f16_gemm_gfx1250.py with increasing block_k values.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRITON_DIR="${TRITON_DIR:-/home/mirror/triton}"
GEMM_SCRIPT="$TRITON_DIR/third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/spill_benchmark_runs}"
RESULTS_FILE="$RESULTS_DIR/spill_benchmark_results.md"

BLOCK_K_VALUES=(64)

# BLOCK_M=256, BLOCK_N=256, NUM_WARPS=2 (patched), NUM_BUFFERS=2.
# LDS: block_k=64 → 128KB, block_k=128 → 256KB (both fit 320KB limit).
# With 2 warps the accumulator (256x256 = 65536 elems) is split across fewer waves,
# doubling per-wave VGPR pressure vs 4 warps.
M=256; N=256; K=1024
BLOCK_M=256; BLOCK_N=256
NUM_WARPS=2; NUM_BUFFERS=2

# AM hang detection: kill the run if the simulator reports a hang or no progress for this long
AM_TIMEOUT_S=${AM_TIMEOUT_S:-1800}

if [ -z "${TRITON_GFX1250_MODEL_PATH:-}" ]; then
    echo "ERROR: TRITON_GFX1250_MODEL_PATH is not set."
    echo "Please: export TRITON_GFX1250_MODEL_PATH=/am-ffm"
    exit 1
fi

if [ ! -f "$GEMM_SCRIPT" ]; then
    echo "ERROR: GEMM script not found at $GEMM_SCRIPT"
    exit 1
fi

# Source AM environment (sets HSA/DTIF vars, LD_LIBRARY_PATH, etc.)
source "$TRITON_GFX1250_MODEL_PATH/am_env.sh"

# The AM package's libamd_smi.so is too old for the installed amdsmi Python wrapper.
# Create an overlay that uses all AM/FFM rocm libs except libamd_smi.so, falling back
# to the system ROCm version.
ROCM_OVERLAY="/tmp/rocm-overlay"
mkdir -p "$ROCM_OVERLAY"
for f in "$TRITON_GFX1250_MODEL_PATH/rocm"/*; do
    base=$(basename "$f")
    case "$base" in
        libamd_smi*) ;;
        *) ln -sf "$f" "$ROCM_OVERLAY/$base" 2>/dev/null || true ;;
    esac
done

# AM needs: its rocm overlay + package libs + system ROCm (for libroctx64, libamd_smi)
export LD_LIBRARY_PATH="$ROCM_OVERLAY:$TRITON_GFX1250_MODEL_PATH/package:$TRITON_GFX1250_MODEL_PATH/package/lib64:$TRITON_GFX1250_MODEL_PATH/package/bin:/opt/rocm/lib"

echo "=========================================="
echo " AM Register Spill Benchmark"
echo "=========================================="
echo "Fixed params: M=$M, N=$N, K=$K, BLOCK_M=$BLOCK_M, BLOCK_N=$BLOCK_N"
echo "              NUM_WARPS=$NUM_WARPS, NUM_BUFFERS=$NUM_BUFFERS"
echo "Sweep:        block_k = ${BLOCK_K_VALUES[*]}"
echo "Results dir:  $RESULTS_DIR"
echo "=========================================="
echo ""

mkdir -p "$RESULTS_DIR"

# Header for the CSV results
CSV_FILE="$RESULTS_DIR/results.csv"
echo "block_k,sgpr_count,sgpr_spill_count,vgpr_count,vgpr_spill_count,scratch_size,code_len_in_byte,occupancy,am_cycles,wall_time_s,status" > "$CSV_FILE"

# Markdown table header
{
    echo "# AM Register Spill Benchmark Results"
    echo ""
    echo "**Parameters**: M=$M, N=$N, K=$K, BLOCK_M=$BLOCK_M, BLOCK_N=$BLOCK_N, NUM_WARPS=$NUM_WARPS, NUM_BUFFERS=$NUM_BUFFERS"
    echo ""
    echo "| block_k | sgpr | sgpr_spill | vgpr | vgpr_spill | scratch | code_bytes | occupancy | am_cycles | wall_time_s | status |"
    echo "|---------|------|------------|------|------------|---------|------------|-----------|-----------|-------------|--------|"
} > "$RESULTS_FILE"

parse_metric() {
    local log_file="$1"
    local metric_name="$2"
    grep -oP "^- ${metric_name}: \K\d+" "$log_file" 2>/dev/null || echo "N/A"
}

parse_am_cycles() {
    local log_file="$1"
    local start_clk end_clk
    start_clk=$(grep -oP 'DispatchId 0:: CP_clk =\K\d+' "$log_file" 2>/dev/null | head -1)
    end_clk=$(grep -oP 'DumpDispatchEndTime .* clk \K\d+' "$log_file" 2>/dev/null | head -1)
    if [[ -n "$start_clk" && -n "$end_clk" ]]; then
        echo $(( end_clk - start_clk ))
    else
        echo "N/A"
    fi
}

# Watchdog: monitors the log file for AM hang or timeout.
# Kills $1 (PID) if hang detected or $AM_TIMEOUT_S exceeded.
# Sets STATUS in the parent via a status file.
am_watchdog() {
    local pid="$1"
    local log_file="$2"
    local status_file="$3"
    local elapsed=0

    while kill -0 "$pid" 2>/dev/null; do
        sleep 5
        elapsed=$((elapsed + 5))

        if grep -q "Chip is hung" "$log_file" 2>/dev/null; then
            echo "AM_HANG" > "$status_file"
            echo "  [WATCHDOG] AM hang detected at ${elapsed}s -- killing process"
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return
        fi

        if [ "$elapsed" -ge "$AM_TIMEOUT_S" ]; then
            echo "TIMEOUT" > "$status_file"
            echo "  [WATCHDOG] Timeout after ${elapsed}s -- killing process"
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return
        fi
    done
}

for BK in "${BLOCK_K_VALUES[@]}"; do
    echo "--- Running block_k=$BK ---"

    RUN_DIR="$RESULTS_DIR/run_bk${BK}"
    mkdir -p "$RUN_DIR"
    LOG_FILE="$RUN_DIR/output.log"

    # Check constraint: ceil(K/block_k) >= NUM_BUFFERS
    K_TILES=$(( (K + BK - 1) / BK ))
    if [ "$K_TILES" -lt "$NUM_BUFFERS" ]; then
        echo "  SKIP: K/block_k=$K_TILES < NUM_BUFFERS=$NUM_BUFFERS"
        echo "$BK,,,,,,,,,SKIP_CONSTRAINT" >> "$CSV_FILE"
        echo "| $BK | - | - | - | - | - | - | - | - | SKIP |" >> "$RESULTS_FILE"
        continue
    fi

    STATUS="PASS"
    WATCHDOG_STATUS_FILE="$RUN_DIR/.watchdog_status"
    rm -f "$WATCHDOG_STATUS_FILE"
    START_NS=$(date +%s%N)

    # Clear triton cache to force recompilation with new block_k
    rm -rf ~/.triton/cache/

    # Run kernel in background so we can monitor it with a watchdog
    (cd "$RUN_DIR" && PYTHONPATH="$TRITON_DIR/python:${PYTHONPATH:-}" python3 "$GEMM_SCRIPT" \
        -M "$M" -N "$N" -K "$K" \
        --block_m="$BLOCK_M" --block_n="$BLOCK_N" --block_k="$BK" \
        --num-warps "$NUM_WARPS" --num-buffers "$NUM_BUFFERS" \
        > "$LOG_FILE" 2>&1) &
    KERNEL_PID=$!

    # Start watchdog in background to detect AM hangs
    am_watchdog "$KERNEL_PID" "$LOG_FILE" "$WATCHDOG_STATUS_FILE" &
    WATCHDOG_PID=$!

    # Wait for the kernel process to finish (either naturally or killed by watchdog)
    # Temporarily disable errexit so wait's non-zero exit doesn't kill the script
    set +e
    wait "$KERNEL_PID" 2>/dev/null
    EXIT_CODE=$?

    # Stop the watchdog (ignore errors since it may have already exited)
    kill "$WATCHDOG_PID" 2>/dev/null
    wait "$WATCHDOG_PID" 2>/dev/null
    set -e

    # Determine status
    if [ -f "$WATCHDOG_STATUS_FILE" ]; then
        STATUS=$(cat "$WATCHDOG_STATUS_FILE")
    elif [ "$EXIT_CODE" -ne 0 ]; then
        STATUS="FAIL(exit=$EXIT_CODE)"
    fi

    END_NS=$(date +%s%N)
    ELAPSED_S=$(python3 -c "print(f'{($END_NS - $START_NS) / 1e9:.2f}')")

    # Parse static_profile metrics
    SGPR=$(parse_metric "$LOG_FILE" "sgpr_count")
    SGPR_SPILL=$(parse_metric "$LOG_FILE" "sgpr_spill_count")
    VGPR=$(parse_metric "$LOG_FILE" "vgpr_count")
    VGPR_SPILL=$(parse_metric "$LOG_FILE" "vgpr_spill_count")
    SCRATCH=$(parse_metric "$LOG_FILE" "scratch_size")
    CODE_LEN=$(parse_metric "$LOG_FILE" "code_len_in_byte")
    OCC=$(parse_metric "$LOG_FILE" "occupancy")

    # Parse AM simulated cycle count (dispatch start to dispatch end)
    AM_CYCLES=$(parse_am_cycles "$LOG_FILE")

    echo "  sgpr=$SGPR (spill=$SGPR_SPILL), vgpr=$VGPR (spill=$VGPR_SPILL), scratch=$SCRATCH, occupancy=$OCC"
    echo "  am_cycles=$AM_CYCLES, wall_time=${ELAPSED_S}s [$STATUS]"

    # Append to CSV
    echo "$BK,$SGPR,$SGPR_SPILL,$VGPR,$VGPR_SPILL,$SCRATCH,$CODE_LEN,$OCC,$AM_CYCLES,$ELAPSED_S,$STATUS" >> "$CSV_FILE"

    # Append to markdown
    echo "| $BK | $SGPR | $SGPR_SPILL | $VGPR | $VGPR_SPILL | $SCRATCH | $CODE_LEN | $OCC | $AM_CYCLES | $ELAPSED_S | $STATUS |" >> "$RESULTS_FILE"
done

echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Raw CSV:      $CSV_FILE"
echo "Markdown:     $RESULTS_FILE"
echo "Run logs:     $RESULTS_DIR/run_bk*/output.log"
