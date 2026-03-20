#!/bin/bash
#
# reproduce_from_asm.sh — Reproduce the asyncmark bug from pre-built assembly
#
# Uses the checked-in assembly files (original/assembly.s and fixed/assembly.s)
# to build HSACOs, substitutes each into an IREE vmfb, and compares results
# against a baseline.
#
# Both vmfbs go through the exact same IREE compilation with the same flags
# and dispatch setup. The only difference is which .hsaco gets substituted.
#
# EXPECTED RESULT:
#   original/assembly.s (asyncmark scheduling) → FAIL  (~40% elements wrong)
#   fixed/assembly.s    (s_waitcnt scheduling) → PASS  (exact match)
#
# DEPENDENCIES:
#   - llvm-mc and ld.lld (from IREE's LLVM build)
#   - iree-compile and iree-run-module
#   - python3 with numpy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Configuration (edit these paths) ─────────────────────────────────────────

IREE_BUILD=/root/iree/build/dbg

LLVM_MC=$IREE_BUILD/llvm-project/bin/llvm-mc
LLD=/usr/bin/ld.lld
IREE_COMPILE=$IREE_BUILD/tools/iree-compile
IREE_RUN=$IREE_BUILD/tools/iree-run-module

TARGET=gfx950
WORKDIR=/tmp/asyncmark_asm_repro

# ─── Verify tools exist ──────────────────────────────────────────────────────

for tool in "$LLVM_MC" "$LLD" "$IREE_COMPILE" "$IREE_RUN"; do
    [ -x "$tool" ] || { echo "ERROR: not found: $tool"; exit 1; }
done

ORIGINAL_ASM=$SCRIPT_DIR/original/assembly.s
FIXED_ASM=$SCRIPT_DIR/fixed/assembly.s
TEST_MLIR=$SCRIPT_DIR/test_mm.mlir

for f in "$ORIGINAL_ASM" "$FIXED_ASM" "$TEST_MLIR"; do
    [ -f "$f" ] || { echo "ERROR: missing: $f"; exit 1; }
done

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# ─── Step 1: Assemble .s → .o → .hsaco ───────────────────────────────────────

echo "Step 1: Assemble to HSACOs"

"$LLVM_MC" -triple=amdgcn-amd-amdhsa -mcpu=$TARGET -filetype=obj \
    "$ORIGINAL_ASM" -o "$WORKDIR/original.o"
"$LLD" -shared "$WORKDIR/original.o" -o "$WORKDIR/original.hsaco"

"$LLVM_MC" -triple=amdgcn-amd-amdhsa -mcpu=$TARGET -filetype=obj \
    "$FIXED_ASM" -o "$WORKDIR/fixed.o"
"$LLD" -shared "$WORKDIR/fixed.o" -o "$WORKDIR/fixed.hsaco"

echo "  original.hsaco: $(stat -c %s "$WORKDIR/original.hsaco") bytes"
echo "  fixed.hsaco:    $(stat -c %s "$WORKDIR/fixed.hsaco") bytes"

# ─── Step 2: Build 3 vmfbs ───────────────────────────────────────────────────

COMMON_FLAGS=(
    --iree-hal-target-backends=rocm
    --iree-rocm-target=$TARGET
    --iree-llvmgpu-set-workgroup-distribution-along=x
    --iree-llvmgpu-use-direct-load
    --iree-llvmgpu-prefetch-num-stages=3
)

echo "Step 2: Build vmfbs"

echo "  baseline (no direct-load)..."
"$IREE_COMPILE" \
    --iree-hal-target-backends=rocm --iree-rocm-target=$TARGET \
    "$TEST_MLIR" -o "$WORKDIR/baseline.vmfb" 2>&1

echo "  with_original (original.hsaco substituted)..."
"$IREE_COMPILE" "${COMMON_FLAGS[@]}" \
    "--iree-hal-substitute-executable-object=matmul_dispatch_0=$WORKDIR/original.hsaco" \
    "$TEST_MLIR" -o "$WORKDIR/with_original.vmfb" 2>&1

echo "  with_fixed (fixed.hsaco substituted)..."
"$IREE_COMPILE" "${COMMON_FLAGS[@]}" \
    "--iree-hal-substitute-executable-object=matmul_dispatch_0=$WORKDIR/fixed.hsaco" \
    "$TEST_MLIR" -o "$WORKDIR/with_fixed.vmfb" 2>&1

# ─── Step 3: Generate random inputs ──────────────────────────────────────────

echo "Step 3: Generate inputs (seed=42)"
python3 -c "
import numpy as np
np.random.seed(42)
np.save('$WORKDIR/a.npy', np.random.randn(4096,4096).astype(np.float32))
np.save('$WORKDIR/b.npy', np.random.randn(4096,4096).astype(np.float32))
"

# ─── Step 4: Run all 3 variants ──────────────────────────────────────────────

echo "Step 4: Run"

echo "  baseline..."
"$IREE_RUN" --module="$WORKDIR/baseline.vmfb" --device=hip \
    --input=@"$WORKDIR/a.npy" --input=@"$WORKDIR/b.npy" \
    --output=@"$WORKDIR/out_baseline.npy" 2>&1

echo "  with_original..."
"$IREE_RUN" --module="$WORKDIR/with_original.vmfb" --device=hip \
    --input=@"$WORKDIR/a.npy" --input=@"$WORKDIR/b.npy" \
    --output=@"$WORKDIR/out_original.npy" 2>&1

echo "  with_fixed..."
"$IREE_RUN" --module="$WORKDIR/with_fixed.vmfb" --device=hip \
    --input=@"$WORKDIR/a.npy" --input=@"$WORKDIR/b.npy" \
    --output=@"$WORKDIR/out_fixed.npy" 2>&1

# ─── Step 5: Compare results ─────────────────────────────────────────────────

echo ""
echo "Step 5: Compare against baseline"
echo ""

python3 << 'PYEOF' - "$WORKDIR"
import numpy as np, sys
workdir = sys.argv[1]

baseline = np.load(f'{workdir}/out_baseline.npy')
original = np.load(f'{workdir}/out_original.npy')
fixed    = np.load(f'{workdir}/out_fixed.npy')

for name, arr in [("original/assembly.s", original), ("fixed/assembly.s", fixed)]:
    diff = np.abs(arr - baseline)
    max_diff = float(diff.max())
    wrong = int((diff > 0.1).sum())
    total = baseline.size
    ok = np.allclose(arr, baseline, rtol=0.001, atol=0.1)
    print(f"  {name}:")
    print(f"    max_abs_diff   = {max_diff:.6f}")
    print(f"    wrong elements = {wrong:,} / {total:,} ({100*wrong/total:.1f}%)")
    print(f"    verdict: {'PASS' if ok else 'FAIL'}")
    print()
PYEOF
