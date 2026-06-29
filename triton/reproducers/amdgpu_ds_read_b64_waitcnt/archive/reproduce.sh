#!/bin/bash
# Reproducer for the AMDGPU ds_read_b64 / s_waitcnt miscompile on gfx950.
# Requires only an LLVM toolchain (llc) + ld.lld + hipcc.  No Triton/PyTorch.
#
#   ./reproduce.sh            # codegen diff (always) + runtime A/B demo (needs GPU)
#   ./reproduce.sh codegen    # just the deterministic codegen diff (no GPU)
#
# Two IRs for the SAME kernel/launch, differing only in the LDS read lowering:
#   ir/attn_fwd.ll           -> backend emits ds_read_b64       (racy at -O3)
#   ir/attn_fwd_strided.ll   -> backend emits ds_read2st64_b32  (correct, control)
# Generated artifacts go under build/ (transient).
set -e
cd "$(dirname "$0")"
MODE="${1:-all}"

LLVM_BIN="${LLVM_BIN:-/root/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin}"
LLC="$LLVM_BIN/llc"
LLD="${LLD:-ld.lld}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
ARCH="${ARCH:-gfx950}"
NRUNS="${NRUNS:-10}"
TRIPLE="amdgcn-amd-amdhsa"
mkdir -p build

# ---- Phase 1: codegen difference (deterministic, no GPU) --------------------
echo "== llc: $("$LLC" --version | grep -i 'LLVM version')"
printf "%-20s %-5s %-12s %-16s %-10s\n" IR opt s_waitcnt ds_read_b64 ds_read2st64
for IR in attn_fwd attn_fwd_strided; do
  for OPT in O0 O3; do
    "$LLC" -mtriple=$TRIPLE -mcpu=$ARCH -$OPT ir/$IR.ll -o build/${IR}_$OPT.s
    printf "%-20s -%-4s %-12s %-16s %-10s\n" "$IR" "$OPT" \
      "$(grep -c s_waitcnt build/${IR}_$OPT.s)" \
      "$(grep -cE 'ds_read_b64 ' build/${IR}_$OPT.s)" \
      "$(grep -cE 'ds_read2st64_b32' build/${IR}_$OPT.s)"
  done
done
echo "(-O3 drops ~half the s_waitcnt for the ds_read_b64 IR; the strided IR keeps a"
echo " correct schedule. Both are the same kernel/launch.)"
[ "$MODE" = "codegen" ] && exit 0

# ---- Phase 2: runtime A/B (gfx950 GPU) -------------------------------------
echo; echo "== runtime: building hsacos + driver =="
for IR in attn_fwd attn_fwd_strided; do
  "$LLC" -mtriple=$TRIPLE -mcpu=$ARCH -O3 -filetype=obj ir/$IR.ll -o build/${IR}_O3.o
  "$LLD" -shared build/${IR}_O3.o -o build/${IR}_O3.hsaco
done
"$HIPCC" -O2 driver.cpp -o build/driver
echo; echo "###### ir/attn_fwd.ll @ -O3  (ds_read_b64 — expected NONDETERMINISTIC) ######"
./build/driver build/attn_fwd_O3.hsaco "$NRUNS" || true
echo; echo "###### ir/attn_fwd_strided.ll @ -O3  (ds_read2st64_b32 — expected STABLE) ######"
./build/driver build/attn_fwd_strided_O3.hsaco "$NRUNS" || true
