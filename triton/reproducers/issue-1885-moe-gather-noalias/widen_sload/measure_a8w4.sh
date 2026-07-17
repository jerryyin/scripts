#!/usr/bin/env bash
# End-to-end demonstration on the real a8w4 gluon prefill kernel that the narrow
# 32x s_load_u16 is entangled: dropping the `% M` (the only "widen" lever) coalesces
# the index but into VMEM (global_load_b128) + reintroduces 16 in-loop readfirstlane.
# Requires a triton built with the noalias contract + FFM env.
#
#   ./measure_a8w4.sh
#
# The "diff" this applies (temporarily, then restores) to the aiter gluon kernel
# _gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py, at `offs_x_m = offs_x_m % M`:
#   -        offs_x_m = offs_x_m % M
#   +        pass  # dropped % M (widen experiment)  -- unsound without host padding
set -u
RUNNER=/root/scripts/triton/moe/ffm_verification/run_moe_gemm_ffm.py
K=/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py
STOCK="${LLC:-/root/.triton/llvm/llvm-56421f92-ubuntu-x64-1/bin/llc}"
source /zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh >/dev/null 2>&1
export HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 AITER_HOME=/root/aiter TRITON_ALWAYS_COMPILE=1
inloop() { awk '/Inner Loop Header/{i=1} i&&/v_readfirstlane/{c++} i&&/s_cbranch.*\.LBB/{i=0} END{print c+0}' "$1"; }
measure() { local tag="$1"; local C; C=$(mktemp -d)
  ( export TRITON_CACHE_DIR="$C"; cd "$(dirname "$RUNNER")"; timeout 600 python3 "$(basename "$RUNNER")" --kernel a8w4 --backend gluon --phase prefill >"$C/log" 2>&1 )
  local ll; ll=$(find "$C" -name "_moe_gemm_a8w4_prefill.llir" -exec ls -S {} + 2>/dev/null|head -1)
  "$STOCK" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm "$ll" -o "$C/k.s" 2>/dev/null
  printf "  %-16s in_loop_rfl=%-3s s_load_u16=%-3s global_load_b128(index)=%-2s  %s\n" "$tag" \
    "$(inloop "$C/k.s")" "$(grep -c s_load_u16 "$C/k.s")" "$(grep -c global_load_b128 "$C/k.s")" "$(grep -iE '^RESULT' "$C/log"|tail -1)"
  rm -rf "$C"; }
cp "$K" "$K.bak"
echo "=== with % M (current: narrow SMEM) ==="; measure "with %M"
sed -i 's/^\(\s*\)offs_x_m = offs_x_m % M/\1pass  # dropped % M (widen experiment)/' "$K"
echo "=== no % M (coalesced, but VMEM + churn) ==="; measure "no %M"
cp "$K.bak" "$K"; rm -f "$K.bak"; echo "restored aiter kernel"
