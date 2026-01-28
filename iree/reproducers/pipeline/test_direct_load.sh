#!/bin/bash
#
# Test script to verify --iree-llvmgpu-use-direct-load produces numerically
# equivalent results across various matrix sizes.
#
# Usage:
#   ./test_direct_load.sh [--target gfx950] [--threshold 0.001] [--id N] [-q]
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_SCRIPTS="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
TARGET="gfx950"
THRESHOLD="0.001"
DTYPE="f32"
QUIET=""
TEST_ID=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --dtype) DTYPE="$2"; shift ;;
        --id) TEST_ID="$2"; shift ;;
        -q|--quiet) QUIET="-q" ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --target CHIP    GPU target (default: gfx950)"
            echo "  --threshold N    Comparison threshold (default: 0.001)"
            echo "  --dtype TYPE     Data type (default: f32)"
            echo "  --id N           Run only test N (0-6), omit to run all"
            echo "  -q, --quiet      Minimal output from sub-scripts"
            echo ""
            echo "Test IDs:"
            echo "  0: 32x32x64"
            echo "  1: 32x64x64"
            echo "  2: 32x128x128"
            echo "  3: 128x256x512"
            echo "  4: 1024x1024x1024"
            echo "  5: 2048x2048x2048"
            echo "  6: 4096x4096x4096"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all tests"
            echo "  $0 --id 6                    # Run only 4096x4096x4096"
            echo "  $0 --id 4 -q                 # Run 1024x1024x1024 quietly"
            echo "  $0 --target gfx942 --id 0    # Run smallest test on gfx942"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Test configurations: "M N K"
ALL_SIZES=(
    "32 32 64"
    "32 64 64"
    "32 128 128"
    "128 256 512"
    "1024 1024 1024"
    "2048 2048 2048"
    "4096 4096 4096"
)

# Select tests to run
if [ -n "$TEST_ID" ]; then
    if ! [[ "$TEST_ID" =~ ^[0-9]+$ ]] || [ "$TEST_ID" -lt 0 ] || [ "$TEST_ID" -ge ${#ALL_SIZES[@]} ]; then
        echo "[ERROR] Invalid test ID: $TEST_ID (must be 0-$((${#ALL_SIZES[@]}-1)))"
        exit 1
    fi
    SIZES=("${ALL_SIZES[$TEST_ID]}")
else
    SIZES=("${ALL_SIZES[@]}")
fi

echo "============================================================"
echo "Direct Load Flag Comparison Test"
echo "============================================================"
echo "Target:    ${TARGET}"
echo "Dtype:     ${DTYPE}"
echo "Threshold: ${THRESHOLD}"
if [ -n "$TEST_ID" ]; then
    echo "Test ID:   ${TEST_ID}"
fi
echo "Tests:     ${#SIZES[@]} configuration(s)"
[ -n "$QUIET" ] && echo "Mode:      Quiet"
echo "============================================================"
echo ""

PASSED=0
FAILED=0
FAILED_TESTS=()

for idx in "${!SIZES[@]}"; do
    size="${SIZES[$idx]}"
    read -r M N K <<< "$size"
    TEST_NAME="${M}x${N}x${K}"
    
    # Calculate actual test ID for display
    if [ -n "$TEST_ID" ]; then
        DISPLAY_ID="$TEST_ID"
    else
        DISPLAY_ID="$idx"
    fi
    
    TEST_DIR="/tmp/test_direct_load_${TEST_NAME}"

    echo "------------------------------------------------------------"
    echo "[TEST ${DISPLAY_ID}] ${TEST_NAME} (${DTYPE})"
    echo "------------------------------------------------------------"

    # Setup test directory
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    rm -f *.bin *.vmfb *.mlir

    # Generate MLIR (always quiet for gen_gemm)
    "$IREE_SCRIPTS/gen_gemm.sh" -m "$M" -n "$N" -k "$K" -d "$DTYPE" > /dev/null 2>&1

    # Run test
    if "$IREE_SCRIPTS/test.sh" \
        -f test_mm.mlir \
        -d "$DTYPE" \
        -i "${M}x${K}" \
        -i "${K}x${N}" \
        --target "$TARGET" \
        --compare-flags '--iree-llvmgpu-use-direct-load' \
        --threshold "$THRESHOLD" \
        $QUIET; then
        echo "[PASS] ${TEST_NAME}"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] ${TEST_NAME}"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("${DISPLAY_ID}:${TEST_NAME}")
    fi
    echo ""
done

# Summary
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Passed: ${PASSED}/${#SIZES[@]}"
echo "Failed: ${FAILED}/${#SIZES[@]}"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - ${test}"
    done
    exit 1
fi

echo ""
echo "All tests passed!"
exit 0
