#!/usr/bin/env bash
set -u
RUNNER=/root/scripts/triton/moe/ffm_verification/run_moe_gemm_ffm.py
AITER=/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py
STOCK=/root/.triton/llvm/llvm-56421f92-ubuntu-x64-1/bin/llc
source /zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh >/dev/null 2>&1
export HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 AITER_HOME=/root/aiter TRITON_ALWAYS_COMPILE=1
inloop() { awk '/Inner Loop Header/{i=1} i&&/v_readfirstlane/{c++} i&&/s_cbranch.*\.LBB/{i=0} END{print c+0}' "$1"; }
run() { local tag="$1"; shift; local C; C=$(mktemp -d)
  ( export TRITON_CACHE_DIR="$C" "$@"; cd "$(dirname "$RUNNER")"; timeout 500 python3 "$(basename "$RUNNER")" --kernel a8w4 --backend gluon --phase prefill >"$C/log" 2>&1 )
  local ll; ll=$(find "$C" -name "_moe_gemm_a8w4_prefill.llir" -exec ls -S {} + 2>/dev/null|head -1)
  "$STOCK" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm "$ll" -o "$C/k.s" 2>/dev/null
  printf "%-40s in_loop_rfl=%-4s s_load_u16=%-3s rfl.i32(peel)=%-3s rfl.p1(fallback)=%-3s %s\n" "$tag" \
    "$(inloop "$C/k.s")" "$(grep -c s_load_u16 "$C/k.s")" \
    "$(grep -c 'readfirstlane.i32' "$ll" 2>/dev/null)" "$(grep -c 'readfirstlane.p1' "$ll" 2>/dev/null)" \
    "$(grep -iE '^RESULT' "$C/log"|tail -1)"
  rm -rf "$C"; }
# disable the noalias contract to isolate the ticket route
cp "$AITER" "$AITER.bak"
sed -i 's/@gluon.jit(launch_metadata=matmul_launch_metadata, noalias_args=\["GatherIndx"\])/@gluon.jit(launch_metadata=matmul_launch_metadata)/' "$AITER"
echo "=== noalias contract DISABLED (aiter) -> isolate ticket route ==="
run "baseline (no noalias, env OFF)"
run "TICKET route (no noalias, env ON)" TRITON_AMD_UNIFORM_SLOAD=1
cp "$AITER.bak" "$AITER"; rm -f "$AITER.bak"
echo "=== for reference: noalias contract ENABLED, env OFF (the shipped route) ==="
run "noalias route (contract, env OFF)"
echo "aiter restored"
