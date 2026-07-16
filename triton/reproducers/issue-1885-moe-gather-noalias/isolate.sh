#!/usr/bin/env bash
# The two experiments that establish the #1885 conclusions. Run against a triton
# built with the noalias contract (LoadStore lowering removed).
#
#   ./isolate.sh
#
# Experiment A (readonly x noalias 2x2 on a8w4 gluon prefill): noalias is the
#   switch (0 with it, churn without); readonly is irrelevant (LLVM infers it).
#   Toggles are on the aiter kernel: noalias via noalias_args, readonly via tl.const.
# Experiment B (same IR, old vs new stock llc): LLVM version is invariant, proving
#   the fix is the contract (a triton IR change), not the LLVM pin bump.
set -u

RUNNER=/root/scripts/triton/moe/ffm_verification/run_moe_gemm_ffm.py
K=/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py
OLD=/root/.triton/llvm/llvm-62b7cf96-ubuntu-x64-2/bin/llc
NEW=/root/.triton/llvm/llvm-56421f92-ubuntu-x64-1/bin/llc
source /zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh >/dev/null 2>&1
export HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 AITER_HOME=/root/aiter TRITON_ALWAYS_COMPILE=1
inloop() { awk '/Inner Loop Header/{i=1} i&&/v_readfirstlane/{c++} i&&/s_cbranch.*\.LBB/{i=0} END{print c+0}' "$1"; }
BAK=$(mktemp); cp "$K" "$BAK"
setcfg() {  # readonly(1/0) noalias(1/0)
  cp "$BAK" "$K"
  [ "$1" = 0 ] && sed -i 's/GatherIndx: tl.const,/GatherIndx,/g' "$K"
  [ "$2" = 0 ] && sed -i 's/@gluon.jit(launch_metadata=matmul_launch_metadata, noalias_args=\["GatherIndx"\])/@gluon.jit(launch_metadata=matmul_launch_metadata)/' "$K"
}
compile_prefill() {  # -> echoes path to the prefill llir; also leaves NEW-llc asm in /tmp/isoA.s
  local C; C=$(mktemp -d)
  ( export TRITON_CACHE_DIR="$C" TRITON_AMD_DISABLE_UNIFORM_SLOAD=1  # harmless if lowering already gone
    cd "$(dirname "$RUNNER")"; timeout 500 python3 "$(basename "$RUNNER")" --kernel a8w4 --backend gluon --phase prefill >"$C/log" 2>&1 )
  local ll; ll=$(find "$C" -name "_moe_gemm_a8w4_prefill.llir" -exec ls -S {} + 2>/dev/null | head -1)
  cp "$ll" /tmp/iso.llir; rm -rf "$C"; echo /tmp/iso.llir
}

echo "=== Experiment A: readonly x noalias (a8w4 gluon prefill, NEW stock llc) ==="
for ro in 1 0; do for na in 1 0; do
  setcfg "$ro" "$na"; ll=$(compile_prefill)
  "$NEW" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm "$ll" -o /tmp/isoA.s 2>/dev/null
  printf "  readonly=%s noalias=%s -> in_loop_rfl=%s\n" "$ro" "$na" "$(inloop /tmp/isoA.s)"
done; done

echo "=== Experiment B: same noalias-only IR, OLD vs NEW stock llc ==="
setcfg 1 1; ll=$(compile_prefill)
"$OLD" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm "$ll" -o /tmp/isoB_old.s 2>/dev/null
"$NEW" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm "$ll" -o /tmp/isoB_new.s 2>/dev/null
printf "  OLD(62b7cf96)=%s  NEW(56421f92)=%s\n" "$(inloop /tmp/isoB_old.s)" "$(inloop /tmp/isoB_new.s)"

cp "$BAK" "$K"; rm -f "$BAK"; echo "aiter kernel restored"
