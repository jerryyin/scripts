#!/bin/bash
# gfx950 ds_read_b64 LDS/MFMA race — micro-dot base case.
# Backend-only: llc + llvm-mc + ld.lld + hipcc. No Triton/PyTorch needed to run.
#
# Three experiments, each built FROM LLVM IR (assembly is saved only as an
# inspection intermediate under build/, mirroring the committed asm/):
#   A  ir/micro-dot.racy.ll    -O0  -> ds_read_b64
#   B  ir/micro-dot.racy.ll    -O3  -> ds_read_b64
#   C  ir/micro-dot.stable.ll  -O3  -> ds_read2_b32
#
#   ./reproduce.sh            # build A/B/C from IR + runtime (needs gfx950)
#   ./reproduce.sh codegen    # build A/B/C + opcode/wait table only (no GPU)
#   ./reproduce.sh irdiff     # clean LLVM IR diff (racy vs stable, no GPU)
#   ./reproduce.sh attn       # (optional knob) attention corroboration run
set -e
cd "$(dirname "$0")"
MODE="${1:-base}"

LLC="${LLC:-/root/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin/llc}"
LB="${LLVM_BIN:-/opt/rocm-7.2.4/lib/llvm/bin}"
MC="$LB/llvm-mc"; LLD="${LLD:-$LB/ld.lld}"; HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
ARCH="${ARCH:-gfx950}"; T="amdgcn-amd-amdhsa"
RUNS="${RUNS:-5000}"; GRID="${GRID:-2048}"; TRIALS="${TRIALS:-3}"; ATTN_RUNS="${ATTN_RUNS:-200}"
mkdir -p build
norm(){ sed 's/amdgcn-amd-amdhsa-unknown-gfx950/amdgcn-amd-amdhsa--gfx950/'; }

# ---- optional knob: attention corroboration ---------------------------------
if [ "$MODE" = "attn" ]; then
  "$LLC" -mtriple=$T -mcpu=$ARCH -O3 ir/attn_fwd.ll         -o - | norm > build/attn_racy.s
  "$LLC" -mtriple=$T -mcpu=$ARCH -O3 ir/attn_fwd_strided.ll -o - | norm > build/attn_strided.s
  "$HIPCC" --offload-arch=$ARCH -O2 -std=c++17 driver_attn.cpp -o build/driver_attn
  for k in attn_racy attn_strided; do "$MC" -triple=$T -mcpu=$ARCH -filetype=obj build/$k.s -o build/$k.o; "$LLD" -shared build/$k.o -o build/$k.hsaco; done
  echo "-- attention ds_read_b64       --"; ./build/driver_attn build/attn_racy.hsaco    "$ATTN_RUNS" || true
  echo "-- attention ds_read2st64_b32  --"; ./build/driver_attn build/attn_strided.hsaco "$ATTN_RUNS" || true
  exit 0
fi

# ---- clean LLVM IR diff (no GPU, no build) ----------------------------------
if [ "$MODE" = "irdiff" ]; then
  nm(){ sed -E 's/;.*//; s/[[:space:]]+$//; s/%[0-9]+/%N/g; s/![0-9]+/!M/g' "$1" | grep -vE '^\s*$|^\s*!'; }
  echo "== micro-dot LLVM IR diff: racy (ds_read_b64) -> stable (ds_read2_b32)"
  echo "   entire difference = the convert read + its address swizzle:"
  diff <(nm ir/micro-dot.racy.ll) <(nm ir/micro-dot.stable.ll) || true
  exit 0
fi

# ---- build A/B/C from LLVM IR -----------------------------------------------
# emit <name> <src.ll> <optflag>: IR -> llc -> build/<name>.s (saved intermediate) -> hsaco
emit(){ "$LLC" -mtriple=$T -mcpu=$ARCH "$3" "$2" -o "build/$1.s"; "$MC" -triple=$T -mcpu=$ARCH -filetype=obj "build/$1.s" -o "build/$1.o"; "$LLD" -shared "build/$1.o" -o "build/$1.hsaco"; }
echo "== llc $("$LLC" --version | sed -n 's/.*LLVM version //p')  (assembly saved under build/ for inspection)"
emit A_racy_O0   ir/micro-dot.racy.ll   -O0
emit B_racy_O3   ir/micro-dot.racy.ll   -O3
emit C_stable_O3 ir/micro-dot.stable.ll -O3
printf "%-14s %-10s %-12s %-14s %-10s\n" experiment opt ds_read_b64 ds_read2_b32 s_waitcnt
row(){ printf "%-14s %-10s %-12s %-14s %-10s\n" "$1" "$2" "$(grep -c 'ds_read_b64 v\[' build/$1.s)" "$(grep -c 'ds_read2_b32' build/$1.s)" "$(grep -c s_waitcnt build/$1.s)"; }
row A_racy_O0 -O0; row B_racy_O3 -O3; row C_stable_O3 -O3
[ "$MODE" = "codegen" ] && exit 0

# ---- runtime ----------------------------------------------------------------
echo; echo "== runtime: building driver =="
"$HIPCC" --offload-arch=$ARCH -O2 -std=c++17 driver_microdot.cpp -o build/driver
run(){ ./build/driver "build/$1.hsaco" "$RUNS" "$GRID" 2>&1 | sed -n 's/.*\(worst=[0-9.]*\).*/\1/p'; }
echo; printf "%-14s %-8s %s\n" experiment expect "$TRIALS trials x $RUNS runs"
for e in "A_racy_O0 STABLE" "B_racy_O3 RACES" "C_stable_O3 STABLE"; do
  set -- $e; printf "%-14s %-8s " "$1" "$2"; for t in $(seq 1 "$TRIALS"); do printf "%s  " "$(run $1)"; done; echo
done