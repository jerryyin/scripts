#!/bin/bash
# Verify that the pipelined async copy assembly does not contain a
# conservative vmcnt(0) between DMA writes and ds_read in the main loop.
#
# Usage:
#   ./verify_pipeline_vmcnt.sh [--iree-compile PATH] [--input MLIR_FILE] [--target TARGET]
#
# The script compiles a GEMM kernel, extracts the assembly, and checks that
# the main loop body does NOT have a vmcnt(0) between buffer_load_dwordx4...lds
# and the first ds_read instruction.

set -euo pipefail

IREE_COMPILE="${IREE_COMPILE:-/root/iree/build/dbg/tools/iree-compile}"
INPUT="${INPUT:-/tmp/bench_f16f32_s2_4096x4096x4096/test_mm.mlir}"
TARGET="${TARGET:-gfx950}"
WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iree-compile) IREE_COMPILE="$2"; shift 2;;
    --input) INPUT="$2"; shift 2;;
    --target) TARGET="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: Input file not found: $INPUT"
  exit 1
fi

echo "=== Pipeline vmcnt Verification ==="
echo "Compiler: $IREE_COMPILE"
echo "Input:    $INPUT"
echo "Target:   $TARGET"
echo ""

# Compile with assembly dump
"$IREE_COMPILE" \
  --iree-hal-target-backends=rocm \
  --iree-hip-target="$TARGET" \
  --iree-llvmgpu-set-workgroup-distribution-along=x \
  --iree-llvmgpu-use-direct-load \
  --iree-hal-dump-executable-intermediates-to="$WORKDIR/asm" \
  "$INPUT" \
  -o "$WORKDIR/output.vmfb" 2>&1

ASM_FILE=$(find "$WORKDIR/asm" -name '*.rocmasm' | head -1)
if [[ -z "$ASM_FILE" ]]; then
  echo "ERROR: No .rocmasm file found"
  exit 1
fi

echo "Assembly: $ASM_FILE"
echo ""

# Find the main loop label (typically .LBB0_1 for 2-stage pipelined loops)
LOOP_LABEL=$(grep -n '^\.LBB0_1:' "$ASM_FILE" | head -1 | cut -d: -f1)
if [[ -z "$LOOP_LABEL" ]]; then
  echo "WARNING: Could not find .LBB0_1 loop label"
  echo "Falling back to searching entire file"
  LOOP_LABEL=1
fi

# Find the loop end (s_cbranch back to .LBB0_1)
LOOP_END=$(grep -n 's_cbranch.*\.LBB0_1' "$ASM_FILE" | head -1 | cut -d: -f1)
if [[ -z "$LOOP_END" ]]; then
  echo "WARNING: Could not find loop branch back"
  LOOP_END=$(wc -l < "$ASM_FILE")
fi

echo "Loop body: lines $LOOP_LABEL - $LOOP_END"
echo ""

# Extract loop body
LOOP_BODY=$(sed -n "${LOOP_LABEL},${LOOP_END}p" "$ASM_FILE")

# Count vmcnt(0) in the loop body
VMCNT_COUNT=$(echo "$LOOP_BODY" | grep -c 's_waitcnt vmcnt(0)' || true)

echo "vmcnt(0) occurrences in main loop: $VMCNT_COUNT"
echo ""

# The first vmcnt(0) before s_barrier is expected (waits for prior iteration).
# A second vmcnt(0) between buffer_load...lds and ds_read is the problem.

# Check: find buffer_load_dwordx4...lds, then check if there's vmcnt(0) between
# the last buffer_load...lds and the first ds_read
LAST_BUF_LOAD=$(echo "$LOOP_BODY" | grep -n 'buffer_load_dwordx4.*lds' | tail -1 | cut -d: -f1)
FIRST_DS_READ=$(echo "$LOOP_BODY" | grep -n 'ds_read' | head -1 | cut -d: -f1)

if [[ -n "$LAST_BUF_LOAD" && -n "$FIRST_DS_READ" ]]; then
  BETWEEN=$(echo "$LOOP_BODY" | sed -n "${LAST_BUF_LOAD},${FIRST_DS_READ}p")
  VMCNT_BETWEEN=$(echo "$BETWEEN" | grep -c 's_waitcnt vmcnt(0)' || true)

  echo "--- Between last buffer_load...lds and first ds_read ---"
  echo "$BETWEEN"
  echo "---"
  echo ""

  if [[ "$VMCNT_BETWEEN" -gt 0 ]]; then
    echo "FAIL: Found conservative vmcnt(0) between DMA writes and ds_reads"
    echo "      This blocks overlap between new-iteration DMA and old-iteration reads."
    exit 1
  else
    echo "PASS: No vmcnt(0) between DMA writes and ds_reads"
    echo "      DMA writes can overlap with ds_reads from previous iteration's slot."
    exit 0
  fi
else
  echo "WARNING: Could not find buffer_load...lds or ds_read in loop body"
  echo "         Cannot verify vmcnt placement"
  # Print all vmcnt info
  echo ""
  echo "All s_waitcnt in assembly:"
  grep -n 's_waitcnt' "$ASM_FILE" || echo "(none)"
  exit 2
fi
