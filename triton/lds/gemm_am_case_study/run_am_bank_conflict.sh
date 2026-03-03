#!/bin/bash
# Turn-key AM bank conflict measurement for the descriptor-load GEMM kernel.
#
# Runs the kernel on the AM simulator, parses performance counters, and prints
# LDS bank conflict results.  Handles environment setup, the libamd_smi ROCm
# overlay, Triton cache clearing, and temp-directory management automatically.
#
# Usage:
#   ./run_am_bank_conflict.sh                                    # defaults
#   ./run_am_bank_conflict.sh --dtype fp8
#   ./run_am_bank_conflict.sh --block_n 256 -N 256
#   ./run_am_bank_conflict.sh --block_m 128 --block_n 128 --block_k 64 \
#                             -M 128 -N 128 -K 1024 --num-warps 8
#
# All arguments are forwarded to gemm_descriptor_load_kernel.py.
#
# Environment variables:
#   AM_TIMEOUT_S   Timeout in seconds for the AM run (default: 1800)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ON_MODEL="$(cd "$SCRIPT_DIR/../../../tools" && pwd)/run_on_model.sh"
KERNEL_SCRIPT="$SCRIPT_DIR/gemm_descriptor_load_kernel.py"
AM_TIMEOUT_S="${AM_TIMEOUT_S:-1800}"

for f in "$RUN_ON_MODEL" "$KERNEL_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: required file not found: $f" >&2
        exit 1
    fi
done

# Work in a temp directory so AM output files don't pollute the source tree.
WORK_DIR=$(mktemp -d -t am_bank_conflict.XXXXXX)
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

echo "=== AM Bank Conflict Measurement ==="
echo "Kernel args: ${*:-<defaults>}"
echo "Work dir:    $WORK_DIR"
echo ""

rm -rf ~/.triton/cache/*

# Run the kernel on AM.  run_on_model.sh handles env setup and the ROCm
# overlay; we just cd into the work dir so AM drops its output files there.
cd "$WORK_DIR"
timeout "$AM_TIMEOUT_S" \
    "$RUN_ON_MODEL" --backend am -- python3 "$KERNEL_SCRIPT" "$@"

# ---------------------------------------------------------------------------
# Parse results (mirrors the counter extraction from generate_am_report.py)
# ---------------------------------------------------------------------------
GFXPERF="$WORK_DIR/perf_counters_gfxperf_absolute.txt"
MIPERF="$WORK_DIR/perf_counters_miperf_absolute.txt"
DISPATCH="$WORK_DIR/dumpPerDrawPerf.csv"

if [[ ! -f "$GFXPERF" ]]; then
    echo "Error: AM did not produce perf_counters_gfxperf_absolute.txt" >&2
    exit 1
fi

echo ""
echo "=== LDS Bank Conflict Counters ==="
echo ""

BANK_CONFLICT_KEYS=(
    SQ_INSTS_LDS
    SP_LDS_CYCLES
    GL0_LDS_REQ
    GL0_LDS_READ_BANK_CONFLICT
    GL0_LDS_WRITE_BANK_CONFLICT
    GL0_TCP_READ_BANK_CONFLICT
    GL0_TCP_WRITE_BANK_CONFLICT
    GL0_PARTITION_READ_CONFLICT
    GL0_PARTITION_WRITE_CONFLICT
)

printf "%-42s %s\n" "Counter" "Value"
printf "%-42s %s\n" "$(printf -- '-%.0s' {1..42})" "----------"

for key in "${BANK_CONFLICT_KEYS[@]}"; do
    val=$(awk -v k="$key" '$1 == k {print $NF}' "$GFXPERF")
    if [[ -n "$val" ]]; then
        printf "%-42s %s\n" "$key" "$val"
    fi
done

if [[ -f "$MIPERF" ]]; then
    for key in DS_READ_BANK_CONFLICTS_SUM CU_CACHE_LDS_PARTITION_READ_CONFLICTS VMEM_READ_BANK_CONFLICTS; do
        val=$(awk -v k="$key" '$1 == k {print $NF}' "$MIPERF")
        if [[ -n "$val" ]]; then
            printf "%-42s %s\n" "$key (miperf)" "$val"
        fi
    done
fi

echo ""
echo "=== Execution Counters ==="
echo ""

EXEC_KEYS=(SCLK GRBM_GUI_ACTIVE SPI_WAVES_LAUNCH SPI_WAVES_DONE SQ_INSTS SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_LDS SQ_INSTS_SCA SQ_INSTS_SMEM)
printf "%-42s %s\n" "Counter" "Value"
printf "%-42s %s\n" "$(printf -- '-%.0s' {1..42})" "----------"
for key in "${EXEC_KEYS[@]}"; do
    val=$(awk -v k="$key" '$1 == k {print $NF}' "$GFXPERF")
    if [[ -n "$val" ]]; then
        printf "%-42s %s\n" "$key" "$val"
    fi
done

if [[ -f "$DISPATCH" ]]; then
    cycles=$(awk -F',' '/drawID:.*Clock Duration:/ {
        for (i=1; i<=NF; i++) if ($i ~ /Clock Duration:/) { print $(i+1)+0; exit }
    }' "$DISPATCH")
    if [[ -n "$cycles" ]]; then
        echo ""
        printf "%-42s %s\n" "Dispatch Clock Duration" "$cycles"
    fi
fi

echo ""
echo "=== Done ==="
