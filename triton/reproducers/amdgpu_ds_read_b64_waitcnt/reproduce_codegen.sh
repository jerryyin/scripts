#!/bin/bash
# Deterministic, GPU-free reproducer of the codegen difference.
# Compiles ir/kernel.ll with the AMDGPU backend at -O0 and -O3 and reports the
# s_waitcnt / ds_read_b64 difference. Requires only `llc`. No Triton, no GPU.
set -e
cd "$(dirname "$0")"
LLC="${LLC:-/root/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin/llc}"
ARCH="${ARCH:-gfx950}"
echo "llc: $("$LLC" --version | grep -i 'LLVM version')"
for OPT in O0 O3; do
  "$LLC" -mtriple=amdgcn-amd-amdhsa -mcpu=$ARCH -$OPT ir/kernel.ll -o asm/kernel_$OPT.s
done
printf "%-6s %-14s %-12s %-12s\n" OPT s_waitcnt ds_read_b64 s_barrier
for OPT in O0 O3; do
  printf "%-6s %-14s %-12s %-12s\n" "-$OPT" \
    "$(grep -c s_waitcnt asm/kernel_$OPT.s)" \
    "$(grep -cE 'ds_read_b64 ' asm/kernel_$OPT.s)" \
    "$(grep -c s_barrier asm/kernel_$OPT.s)"
done
echo
echo "The kernel contains no atomics; with identical inputs a correct compilation"
echo "must be deterministic. -O3 drops ~half the s_waitcnt instructions present at"
echo "-O0 while keeping the same 5 ds_read_b64 and 29 s_barrier."
