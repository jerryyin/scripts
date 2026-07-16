#!/usr/bin/env bash
# Regenerate the BEFORE/AFTER (noalias off vs on) prefill assembly for both MoE
# kernels into ./asm/, using stock LLVM llc. Toggles the noalias_args annotation
# on the aiter a8w4 gluon kernel and the in-tree moe_gfx1250.py example, then
# restores both. Reports in-loop v_readfirstlane and the hot-loop line range.
#
#   ./gen_before_after.sh
#
# See README.md. Requires a triton built with the noalias contract (branch
# users/jerryyin/moe-gather-sload-contract) and the FFM env.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/asm"; mkdir -p "$OUT"
RUNNER=/root/scripts/triton/moe/ffm_verification/run_moe_gemm_ffm.py
EX=/root/triton/third_party/amd/python/examples/gluon/moe_gfx1250.py
AITER=/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py
STOCK=/root/.triton/llvm/llvm-56421f92-ubuntu-x64-1/bin/llc

source /zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh >/dev/null 2>&1
export HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 AITER_HOME=/root/aiter TRITON_ALWAYS_COMPILE=1

report() {  # tag sfile
  local tag="$1" s="$2" hdr bk cnt
  hdr=$(awk '/Inner Loop Header/{print NR; exit}' "$s")
  bk=$(awk -v h="$hdr" 'NR>h && /s_cbranch.*\.LBB/{print NR; exit}' "$s")
  cnt=$(awk -v A="$hdr" -v B="$bk" 'NR>=A && NR<=B && /v_readfirstlane/{c++} END{print c+0}' "$s")
  awk -v A="$hdr" -v B="$bk" 'NR>=A && NR<=B' "$s" > "$OUT/${tag}_loopbody.txt"
  printf "  %-32s in_loop_rfl=%-4s  hot loop %s..%s\n" "$tag" "$cnt" "$hdr" "$bk"
}
gen_a8w4() {  # tag
  local tag="$1" C; C=$(mktemp -d)
  ( export TRITON_CACHE_DIR="$C"; cd "$(dirname "$RUNNER")"
    timeout 500 python3 "$(basename "$RUNNER")" --kernel a8w4 --backend gluon --phase prefill >"$C/log" 2>&1 )
  local ll; ll=$(find "$C" -name "_moe_gemm_a8w4_prefill.llir" -exec ls -S {} + 2>/dev/null | head -1)
  "$STOCK" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm "$ll" -o "$OUT/$tag.s" 2>/dev/null
  cp "$ll" "$OUT/$tag.llir"; rm -rf "$C"
}
gen_moe() {  # tag
  local tag="$1" C; C=$(mktemp -d)
  ( export TRITON_CACHE_DIR="$C" PYTHONPATH=/root/triton/python/triton_kernels; cd "$(dirname "$EX")"
    timeout 500 python3 "$(basename "$EX")" -b 1024 -d1 512 -d2 512 -et 8 -ea 2 -a dispatch >"$C/log" 2>&1 )
  local ll; ll=$(find "$C" -name "_matmul*.llir" -exec ls -S {} + 2>/dev/null | head -1)
  "$STOCK" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm "$ll" -o "$OUT/$tag.s" 2>/dev/null
  cp "$ll" "$OUT/$tag.llir"; rm -rf "$C"
}

# a8w4 gluon (toggle aiter noalias_args)
cp "$AITER" "$AITER.bak"
sed -i 's/@gluon.jit(launch_metadata=matmul_launch_metadata, noalias_args=\["GatherIndx"\])/@gluon.jit(launch_metadata=matmul_launch_metadata)/' "$AITER"
gen_a8w4 a8w4_gluon_prefill_BEFORE
cp "$AITER.bak" "$AITER"
gen_a8w4 a8w4_gluon_prefill_AFTER
rm -f "$AITER.bak"

# moe_gfx1250 (toggle the in-tree annotation; assumes it is present = AFTER)
cp "$EX" "$EX.bak"
gen_moe moe_gfx1250_prefill_AFTER
sed -i 's/@gluon.jit(noalias_args=\["GatherIndx"\])/@gluon.jit/' "$EX"
gen_moe moe_gfx1250_prefill_BEFORE
cp "$EX.bak" "$EX"; rm -f "$EX.bak"

echo "=== a8w4 gluon prefill (stock LLVM) ==="
report a8w4_gluon_prefill_BEFORE "$OUT/a8w4_gluon_prefill_BEFORE.s"
report a8w4_gluon_prefill_AFTER  "$OUT/a8w4_gluon_prefill_AFTER.s"
echo "=== moe_gfx1250 prefill (stock LLVM) ==="
report moe_gfx1250_prefill_BEFORE "$OUT/moe_gfx1250_prefill_BEFORE.s"
report moe_gfx1250_prefill_AFTER  "$OUT/moe_gfx1250_prefill_AFTER.s"
echo "asm written to $OUT ; aiter + example restored"
