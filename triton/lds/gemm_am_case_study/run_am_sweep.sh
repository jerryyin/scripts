#!/bin/bash
# Sweep multiple GEMM configurations on the AM simulator to measure LDS bank
# conflicts.  Each configuration is run via run_am_bank_conflict.sh and the
# key counters are collected into a summary table.
#
# Usage:
#   ./run_am_sweep.sh              # run all predefined configurations
#   ./run_am_sweep.sh --dry-run    # print configs without running
#
# The predefined sweep varies warp count and tile size to isolate the source
# of GL0_LDS_READ_BANK_CONFLICT.  Results show that bank conflicts come
# entirely from ds_load_tr16_b128 (B operand transposed load), at exactly
# 2 conflict cycles per execution, while ds_load_b128 (A operand) contributes
# zero.
#
# Environment variables:
#   AM_TIMEOUT_S   Per-run timeout in seconds (default: 1800)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_am_bank_conflict.sh"

if [[ ! -x "$RUN_SCRIPT" ]]; then
    echo "Error: run_am_bank_conflict.sh not found at $RUN_SCRIPT" >&2
    exit 1
fi

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Dtypes to sweep ──────────────────────────────────────────────────────────
# The descriptor-load GEMM kernel supports fp16 and fp8.
# fp32 has no ds_load_tr instruction and is not supported by this kernel.
DTYPES=("fp16" "fp8")

# ── Configuration table ──────────────────────────────────────────────────────
# Each line: WARPS BLOCK_M BLOCK_N BLOCK_K M N K
CONFIGS=(
    "1   16   16  64   16   16  1024"
    "1   16   32  64   16   32  1024"
    "1   16   64  64   16   64  1024"
    "2   32   32  64   32   32  1024"
    "4   64   64  64   64   64  1024"
    "8  128  128  64  128  128  1024"
)

RESULTS_DIR="$SCRIPT_DIR/sweep_results"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.txt"

header=$(printf "%-5s %-6s %-8s %-8s %-8s %-5s %-5s %-6s | %-12s %-12s %-12s" \
    "dtype" "Warps" "BLOCK_M" "BLOCK_N" "BLOCK_K" "M" "N" "K" \
    "SQ_INSTS_LDS" "BANK_CONFL" "PART_CONFL")

echo "=================================================================="
echo " AM Bank Conflict Sweep"
echo "=================================================================="
echo ""
echo "$header"
echo "$(printf -- '-%.0s' {1..105})"

# Also write to summary file
{
    echo "AM Bank Conflict Sweep"
    echo "$(date)"
    echo ""
    echo "$header"
    echo "$(printf -- '-%.0s' {1..105})"
} > "$SUMMARY"

total_runs=$(( ${#DTYPES[@]} * ${#CONFIGS[@]} ))
run_idx=0
for DTYPE in "${DTYPES[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        read -r WARPS BM BN BK M N K <<< "$cfg"
        run_idx=$((run_idx + 1))

        if $DRY_RUN; then
            printf "%-5s %-6s %-8s %-8s %-8s %-5s %-5s %-6s | (dry run)\n" \
                "$DTYPE" "$WARPS" "$BM" "$BN" "$BK" "$M" "$N" "$K"
            continue
        fi

        echo "" >&2
        echo "── Run $run_idx/$total_runs: ${DTYPE} ${WARPS}w ${BM}x${BN}x${BK} ──" >&2

        LOG="$RESULTS_DIR/run_${DTYPE}_${WARPS}w_${BM}x${BN}x${BK}.log"

        if "$RUN_SCRIPT" \
            --num-warps "$WARPS" \
            --block_m "$BM" --block_n "$BN" --block_k "$BK" \
            -M "$M" -N "$N" -K "$K" \
            --dtype "$DTYPE" \
            > "$LOG" 2>&1; then

            sq_lds=$(awk '$1 == "SQ_INSTS_LDS" {print $NF}' "$LOG" | head -1)
            bank_confl=$(awk '$1 == "GL0_LDS_READ_BANK_CONFLICT" {print $NF}' "$LOG")
            part_confl=$(awk '$1 == "GL0_PARTITION_READ_CONFLICT" {print $NF}' "$LOG")

            line=$(printf "%-5s %-6s %-8s %-8s %-8s %-5s %-5s %-6s | %-12s %-12s %-12s" \
                "$DTYPE" "$WARPS" "$BM" "$BN" "$BK" "$M" "$N" "$K" \
                "${sq_lds:-?}" "${bank_confl:-?}" "${part_confl:-?}")
        else
            line=$(printf "%-5s %-6s %-8s %-8s %-8s %-5s %-5s %-6s | FAILED (see %s)" \
                "$DTYPE" "$WARPS" "$BM" "$BN" "$BK" "$M" "$N" "$K" "$LOG")
        fi

        echo "$line"
        echo "$line" >> "$SUMMARY"
    done

    # Visual separator between dtypes
    sep="$(printf -- '-%.0s' {1..105})"
    echo "$sep"
    echo "$sep" >> "$SUMMARY"
done

if ! $DRY_RUN; then
    echo "" | tee -a "$SUMMARY"
    echo "Results saved to $RESULTS_DIR/" | tee -a "$SUMMARY"
    echo "Summary:       $SUMMARY"
fi
