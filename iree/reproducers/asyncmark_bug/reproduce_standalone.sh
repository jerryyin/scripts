#!/bin/bash
#
# reproduce_standalone.sh — Reproduce the asyncmark bug without IREE
#
# Assembles original/assembly.s and fixed/assembly.s to HSACOs, builds
# a standalone HIP driver, and launches each kernel against a host reference.
#
# No IREE tools required. Only needs: llvm-mc, ld.lld, hipcc.
#
# EXPECTED RESULT:
#   original/assembly.s (asyncmark scheduling) → FAIL  (~40% elements wrong)
#   fixed/assembly.s    (s_waitcnt scheduling) → PASS  (exact match)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Configuration (edit these paths) ─────────────────────────────────────────

LLVM_MC=/root/iree/build/dbg/llvm-project/bin/llvm-mc
LLD=/usr/bin/ld.lld
HIPCC=hipcc

TARGET=gfx950
WORKDIR=/tmp/asyncmark_standalone

# ─── Verify tools ─────────────────────────────────────────────────────────────

for tool in "$LLVM_MC" "$LLD"; do
    [ -x "$tool" ] || { echo "ERROR: not found: $tool"; exit 1; }
done
command -v "$HIPCC" >/dev/null || { echo "ERROR: $HIPCC not in PATH"; exit 1; }

ORIGINAL_ASM=$SCRIPT_DIR/original/assembly.s
FIXED_ASM=$SCRIPT_DIR/fixed/assembly.s
DRIVER_SRC=$SCRIPT_DIR/driver.cpp

for f in "$ORIGINAL_ASM" "$FIXED_ASM" "$DRIVER_SRC"; do
    [ -f "$f" ] || { echo "ERROR: missing: $f"; exit 1; }
done

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# ─── Step 1: Assemble .s → .o → .hsaco ───────────────────────────────────────

echo "Step 1: Assemble to HSACOs"

"$LLVM_MC" -triple=amdgcn-amd-amdhsa -mcpu=$TARGET -filetype=obj \
    "$ORIGINAL_ASM" -o "$WORKDIR/original.o"
"$LLD" -shared "$WORKDIR/original.o" -o "$WORKDIR/original.hsaco"
echo "  original.hsaco: $(stat -c %s "$WORKDIR/original.hsaco") bytes"

"$LLVM_MC" -triple=amdgcn-amd-amdhsa -mcpu=$TARGET -filetype=obj \
    "$FIXED_ASM" -o "$WORKDIR/fixed.o"
"$LLD" -shared "$WORKDIR/fixed.o" -o "$WORKDIR/fixed.hsaco"
echo "  fixed.hsaco:    $(stat -c %s "$WORKDIR/fixed.hsaco") bytes"

# ─── Step 2: Build HIP driver ────────────────────────────────────────────────

echo "Step 2: Build HIP driver"
"$HIPCC" "$DRIVER_SRC" -o "$WORKDIR/driver" --offload-arch=$TARGET 2>&1
echo "  driver built"

# ─── Step 3: Run both ────────────────────────────────────────────────────────

echo ""
echo "=== original/assembly.s (asyncmark scheduling) ==="
"$WORKDIR/driver" "$WORKDIR/original.hsaco" || true

echo ""
echo "=== fixed/assembly.s (s_waitcnt scheduling) ==="
"$WORKDIR/driver" "$WORKDIR/fixed.hsaco"
