#!/usr/bin/env bash
# Isolated (no Triton, no GPU) proof of the widening root cause: AMDGPU has no
# wide/vectorized sub-dword (i16) SMEM load, so a contiguous i16 uniform load
# falls to VMEM (global_load), while a dword (i32) one coalesces to wide SMEM.
#
#   ./reproduce.sh            (set LLC=/path/to/llc to override)
#
# Expected (gfx1250):
#   @t16   contiguous i16 -> global_load_u16   (VMEM)   <- the a8w4 problem
#   @t16nc scattered  i16 -> s_load_u16        (SMEM, scalar sub-dword OK)
#   @t32   contiguous i32 -> s_load_b32/b*     (wide SMEM) <- what moe_gfx1250 gets
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LLC="${LLC:-/root/.triton/llvm/llvm-56421f92-ubuntu-x64-1/bin/llc}"
FLAGS="-mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm"

body() { awk -v f="$1" '$0 ~ (f":"){p=1} p{print} p&&/s_endpgm/{exit}'; }
kind() { # fn file  -> report s_load/global for the DATA loads (skip the arg s_load_b128 s[4:5])
  local fn="$1" file="$2"
  local g s
  g=$(body "$fn" < "$file" | grep -c 'global_load')
  s=$(body "$fn" < "$file" | grep -cE 's_load_(u16|b32|b64|b128|b256|b512)' )
  # subtract 1 for the kernel-arg s_load_b128 from s[4:5]
  printf "  %-8s global_load=%-2s s_load(incl 1 arg-load)=%-2s\n" "$fn" "$g" "$s"
  body "$fn" < "$file" | grep -E '_load_' | sed 's/^/      /'
}

$LLC $FLAGS "$HERE/subdword.ll"  -o /tmp/sd.s  2>/dev/null
$LLC $FLAGS "$HERE/subdword2.ll" -o /tmp/sd2.s 2>/dev/null
echo "=== subdword.ll ==="
kind t16 /tmp/sd.s
kind t32 /tmp/sd.s
echo "=== subdword2.ll ==="
kind t16nc /tmp/sd2.s
echo
echo "Verdict: i16-contiguous uses global_load (VMEM); i16-scattered and i32 use s_load (SMEM)."
echo "=> a8w4's uint16 index can only wide-load via VMEM; wide SMEM needs dword (i32) granularity."
