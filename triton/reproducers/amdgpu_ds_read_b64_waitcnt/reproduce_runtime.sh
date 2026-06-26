#!/bin/bash
# Standalone reproducer for the AMDGPU ds_read_b64 / s_waitcnt miscompile.
# Requires only an LLVM toolchain (llc) + ld.lld + hipcc on a gfx950 GPU.
# NO Triton, NO PyTorch, NO IREE.
set -e
cd "$(dirname "$0")"

# --- toolchain (override as needed) ---
LLVM_BIN="${LLVM_BIN:-/root/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin}"
LLC="$LLVM_BIN/llc"
LLD="${LLD:-ld.lld}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
ARCH="${ARCH:-gfx950}"
NRUNS="${NRUNS:-10}"

echo "== using LLC=$LLC =="; "$LLC" --version | grep -i version | head -1

for OPT in O0 O3; do
  echo "== llc -$OPT =="
  "$LLC" -mtriple=amdgcn-amd-amdhsa -mcpu=$ARCH -$OPT ir/kernel.ll -o asm/kernel_$OPT.s
  "$LLC" -mtriple=amdgcn-amd-amdhsa -mcpu=$ARCH -$OPT -filetype=obj ir/kernel.ll -o asm/kernel_$OPT.o
  "$LLD" -shared asm/kernel_$OPT.o -o asm/kernel_$OPT.hsaco
  echo "   ds_read_b64=$(grep -cE 'ds_read_b64 ' asm/kernel_$OPT.s)  s_waitcnt=$(grep -c s_waitcnt asm/kernel_$OPT.s)  s_barrier=$(grep -c s_barrier asm/kernel_$OPT.s)"
done

echo "== building driver =="
"$HIPCC" -O2 driver.cpp -o driver

echo
echo "############ O0 (backend -O0, expected CORRECT / stable) ############"
./driver asm/kernel_O0.hsaco $NRUNS || true
echo
echo "############ O3 (backend -O3, expected MISCOMPILED / varying) ############"
./driver asm/kernel_O3.hsaco $NRUNS || true
