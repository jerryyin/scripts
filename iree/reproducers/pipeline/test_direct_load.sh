#!/bin/bash
#
# Test script to verify --iree-llvmgpu-use-direct-load produces numerically
# equivalent results across various matrix sizes.
#
# Usage:
#   ./test_direct_load.sh [--target gfx950] [--threshold 0.001]
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_SCRIPTS="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
TARGET="gfx950"
THRESHOLD="0.001"
DTYPE="f32"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --dtype) DTYPE="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--target CHIP] [--threshold N] [--dtype TYPE]"
            echo ""
            echo "Options:"
            echo "  --target CHIP    GPU target (default: gfx950)"
            echo "  --threshold N    Comparison threshold (default: 0.001)"
            echo "  --dtype TYPE     Data type (default: f32)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Test configurations: "M N K"
SIZES=(
    "32 32 64"
    "32 64 64"
    "32 128 128"
    "128 256 512"
    "1024 1024 1024"
    "2048 2048 2048"
    "4096 4096 4096"
)

echo "============================================================"
echo "Direct Load Flag Comparison Test"
echo "============================================================"
echo "Target:    ${TARGET}"
echo "Dtype:     ${DTYPE}"
echo "Threshold: ${THRESHOLD}"
echo "Sizes:     ${#SIZES[@]} configurations"
echo "============================================================"
echo ""

PASSED=0
FAILED=0
FAILED_TESTS=()

for size in "${SIZES[@]}"; do
    read -r M N K <<< "$size"
    TEST_NAME="${M}x${N}x${K}"
    TEST_DIR="/tmp/test_direct_load_${TEST_NAME}"

    echo "------------------------------------------------------------"
    echo "[TEST] ${TEST_NAME} (${DTYPE})"
    echo "------------------------------------------------------------"

    # Setup test directory
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    rm -f *.bin *.vmfb *.mlir

    # Generate MLIR
    "$IREE_SCRIPTS/gen_gemm.sh" -m "$M" -n "$N" -k "$K" -d "$DTYPE"

    # Run test
    if "$IREE_SCRIPTS/test.sh" \
        -f test_mm.mlir \
        -d "$DTYPE" \
        -i "${M}x${K}" \
        -i "${K}x${N}" \
        --target "$TARGET" \
        --compare-flags '--iree-llvmgpu-use-direct-load' \
        --threshold "$THRESHOLD"; then
        echo "[PASS] ${TEST_NAME}"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] ${TEST_NAME}"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$TEST_NAME")
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
