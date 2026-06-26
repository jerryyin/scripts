#!/bin/bash
# git bisect run script for the AMDGPU ds_read_b64 / s_waitcnt regression.
# Deterministic predicate: compile the portable IR with this commit's llc at -O3
# and count s_waitcnt. Bug present (BAD) => waitcnt count <= THRESHOLD.
#   exit 0   = good (bug absent)
#   exit 1   = bad  (bug present)
#   exit 125 = skip (build/parse failed -> untestable commit)
set -u
cd /root/llvm-tip
IR=/tmp/kernel_portable.ll
THRESHOLD=${THRESHOLD:-70}    # good O3 >> 70 waitcnt; bad O3 = 55 (set after probe)

# incremental build of just llc
if ! ninja -C build llc > /tmp/bisect_build.log 2>&1; then
  echo "BUILD FAILED -> skip"; exit 125
fi
LLC=build/bin/llc
if ! "$LLC" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 "$IR" -o /tmp/bisect_O3.s 2>/tmp/bisect_llc.log; then
  echo "LLC PARSE/CODEGEN FAILED -> skip"; cat /tmp/bisect_llc.log | head -3; exit 125
fi
WC=$(grep -c s_waitcnt /tmp/bisect_O3.s)
DR=$(grep -cE 'ds_read_b64 ' /tmp/bisect_O3.s)
echo "commit $(git rev-parse --short HEAD): s_waitcnt=$WC ds_read_b64=$DR"
if [ "$DR" -eq 0 ]; then echo "no ds_read_b64 (different codegen) -> skip"; exit 125; fi
if [ "$WC" -le "$THRESHOLD" ]; then echo "=> BAD (bug present)"; exit 1; else echo "=> GOOD"; exit 0; fi
