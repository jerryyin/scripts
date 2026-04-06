#!/bin/bash
#
# Benchmark script for direct-load (gather_to_lds) pipelined GEMM performance.
# Compiles and benchmarks square GEMMs with --iree-llvmgpu-use-direct-load.
#
# Usage:
#   ./bench_direct_load.sh [--target gfx950] [--reps 10] [--warmup 3.0] [--stages N]
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_SCRIPTS="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
TARGET="gfx950"
DTYPE="f32"
REPS=10
WARMUP="3.0"
STAGES=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift ;;
        --dtype) DTYPE="$2"; shift ;;
        --reps) REPS="$2"; shift ;;
        --warmup) WARMUP="$2"; shift ;;
        --stages) STAGES="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --target CHIP    GPU target (default: gfx950)"
            echo "  --dtype TYPE     Data type (default: f32)"
            echo "  --reps N         Benchmark repetitions (default: 10)"
            echo "  --warmup SECS    Warmup time in seconds (default: 3.0)"
            echo "  --stages N       Pipeline stages (default: compiler default, 0 disables)"
            echo ""
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Test configurations: "M N K"
SIZES=(
    "1024 1024 1024"
    "2048 2048 2048"
    "4096 4096 4096"
    "8192 8192 8192"
)

# Build stages flags
STAGES_FLAG=""
STAGES_LABEL="default"
if [ -n "$STAGES" ]; then
    STAGES_FLAG="--iree-llvmgpu-prefetch-num-stages=${STAGES}"
    STAGES_LABEL="${STAGES}"
fi

echo "============================================================"
echo "Direct Load GEMM Benchmark"
echo "============================================================"
echo "Target:    ${TARGET}"
echo "Dtype:     ${DTYPE}"
echo "Stages:    ${STAGES_LABEL}"
echo "Reps:      ${REPS}"
echo "Warmup:    ${WARMUP}s"
echo "Sizes:     ${#SIZES[@]} configuration(s)"
echo "============================================================"
echo ""

# Map dtype to iree element type
case "$DTYPE" in
    f32)  ELEM_TYPE="f32" ;;
    f16)  ELEM_TYPE="f16" ;;
    bf16) ELEM_TYPE="bf16" ;;
    *)    echo "[ERROR] Unsupported dtype: $DTYPE"; exit 1 ;;
esac

# Extract median from benchmark output
# iree-benchmark-module prints lines like:
#   BM_.../real_time_median   1.22 ms   1.28 ms   3   items_per_second=818/s
extract_median() {
    local output="$1"
    local median_line
    median_line=$(echo "$output" | grep "real_time_median" | head -1)
    if [ -z "$median_line" ]; then
        echo "N/A"
        return
    fi
    # Extract the first time value and unit (e.g. "1.22 ms" or "456 us")
    echo "$median_line" | awk '{
        for(i=1;i<=NF;i++) {
            if ($i ~ /^[0-9]+\.?[0-9]*$/ && ($(i+1) == "us" || $(i+1) == "ms" || $(i+1) == "s")) {
                printf "%s %s", $i, $(i+1)
                exit
            }
        }
    }'
}

printf "%-20s %15s\n" "Size" "Median (us)"
printf "%-20s %15s\n" "----" "----------"

for size in "${SIZES[@]}"; do
    read -r M N K <<< "$size"
    TEST_NAME="${M}x${N}x${K}"
    TEST_DIR="/tmp/bench_direct_load_s${STAGES_LABEL}_${TEST_NAME}"

    # Setup
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    rm -f *.bin *.vmfb *.mlir

    # Generate MLIR
    "$IREE_SCRIPTS/gen_gemm.sh" -m "$M" -n "$N" -k "$K" -d "$DTYPE" > /dev/null 2>&1

    # Compile with direct load + benchmark funcs
    if ! iree-compile \
        --iree-hal-target-backends=rocm \
        --iree-hip-target="${TARGET}" \
        --iree-llvmgpu-set-workgroup-distribution-along=x \
        --iree-llvmgpu-use-direct-load \
        --iree-flow-export-benchmark-funcs \
        ${STAGES_FLAG} \
        test_mm.mlir -o bench.vmfb > /dev/null 2>&1; then
        printf "%-20s %15s\n" "$TEST_NAME" "COMPILE_FAIL"
        continue
    fi

    # Generate random inputs
    INPUT_ARGS=""
    INPUT_ARGS+="--input=${M}x${K}x${ELEM_TYPE}=random "
    INPUT_ARGS+="--input=${K}x${N}x${ELEM_TYPE}=random "

    # Benchmark
    BENCH_OUTPUT=$(iree-benchmark-module \
        --device=hip \
        --module=bench.vmfb \
        ${INPUT_ARGS} \
        --benchmark_repetitions="${REPS}" \
        --benchmark_min_warmup_time="${WARMUP}" 2>&1) || true

    MEDIAN=$(extract_median "$BENCH_OUTPUT")
    printf "%-20s %15s\n" "$TEST_NAME" "${MEDIAN}"
done

echo ""
echo "============================================================"
echo "Done"
echo "============================================================"
