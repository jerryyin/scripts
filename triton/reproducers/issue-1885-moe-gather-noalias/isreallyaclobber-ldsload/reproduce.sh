#!/bin/bash
# Guided inspection of the AMDGPU isReallyAClobber missed-optimization.
#
# This prints, for each comparison chunk, the exact file:line to open (LLVM IR and
# assembly), the snippet itself, and a one-line "what/why" — so you can jump into
# ir/ asm/ annotate/ and read them directly. It is READ-ONLY over the committed
# artifacts; it does not run any tool.
#
# To regenerate / verify the before-vs-after with your own opt+llc (build fix.patch),
# see README.md "Reproduce" (BEFORE -> build the fix -> AFTER via LLVM_BUILD).
set -e
cd "$(dirname "$0")"

# print file:line: for a range of lines
at(){ awk -v s="$2" -v e="$((${2}+${3:-1}-1))" 'NR>=s&&NR<=e{printf "    %s:%d:%s\n",FILENAME,NR,$0}' "$1"; }
# first line number matching a fixed string
findn(){ grep -n -m1 -F -- "$2" "$1" | cut -d: -f1; }
h(){ echo; echo "=================================================================================="; echo "$1"; echo "=================================================================================="; }

h "[1] REAL KERNEL — the in-loop block-table prefetch load (MoE MLA-decode, gfx1250)"

echo
echo "(a) LLVM IR — the load itself:"
at ir/mla_decode.ll "$(findn ir/mla_decode.ll '%1146 = load <1 x i32>')"
echo "    what: the KV block index, loaded every K-loop iteration (software-pipelined"
echo "          prefetch of the next block). Uniform across lanes, never written."

echo
echo "(b) annotate-uniform — the fix's DIRECT effect (does the load get !amdgpu.noclobber?):"
echo "  BEFORE:"; at annotate/mla_decode.before.uniform.ll "$(findn annotate/mla_decode.before.uniform.ll '%1146 = load <1 x i32>')"
echo "  AFTER :"; at annotate/mla_decode.after.uniform.ll  "$(findn annotate/mla_decode.after.uniform.ll  '%1146 = load <1 x i32>')"
echo "    why: upstream isReallyAClobber counts the in-loop llvm.amdgcn.tensor.load.to.lds"
echo "         (LDS, addrspace 3) as a clobber WITHOUT asking AA. The fix asks AA -> it"
echo "         cannot touch this addrspace(1) load -> the load is tagged !amdgpu.noclobber."

echo
echo "(c) ASSEMBLY, BEFORE — no noclobber => VMEM load + a VGPR->SGPR round-trip:"
at asm/mla_decode.before.s "$(findn asm/mla_decode.before.s 'global_load_b32 v253')"
at asm/mla_decode.before.s "$(findn asm/mla_decode.before.s 'v_readfirstlane_b32 s25, v253')"
echo "    why: global_load_b32 (VMEM) writes a VGPR (v253); feeding the SCALAR TDM"
echo "         descriptor operand then needs v_readfirstlane s25,v253 — once per iteration."

echo
echo "(d) ASSEMBLY, AFTER — noclobber => scalar s_load, round-trip gone:"
at asm/mla_decode.after.s "$(findn asm/mla_decode.after.s 's_load_b32 s78, s[54:55], 0x0')"
echo "    why: s_load (SMEM) writes an SGPR (s78) directly, so the value is born scalar"
echo "         and the per-iteration v_readfirstlane disappears (in-loop rfl 1 -> 0)."

h "[2] CONTRAST — the OUT-OF-LOOP sibling block-table loads are already scalar in BOTH"

echo
echo "Same load, but not inside the pipelined loop (e.g.):"
at ir/mla_decode.ll "$(findn ir/mla_decode.ll '%619 = load <1 x i32>')"
echo "  BEFORE annotate already has noclobber:"; at annotate/mla_decode.before.uniform.ll "$(findn annotate/mla_decode.before.uniform.ll '%619 = load <1 x i32>')"
echo "    why: their MemorySSA clobber walk reaches liveOnEntry cleanly (no loop MemoryPhi"
echo "         surfacing the intrinsic), so they get noclobber and select s_load even upstream."
echo "         => the in-loop load is fully s_load-representable; ONLY the missing tag differs."

h "[3] MINIMAL (18-line, llvm-reduce'd) — same mechanism, isolated"

echo
echo "The clobber (LDS-writing intrinsic):"
at ir/minimal.ll "$(findn ir/minimal.ll 'tensor.load.to.lds(')"
echo "The uniform read-only load it wrongly 'clobbers':"
at ir/minimal.ll "$(findn ir/minimal.ll '= load <1 x i32>')"
echo "    check: opt -passes=amdgpu-annotate-uniform -mcpu=gfx1250 -S ir/minimal.ll | grep -c amdgpu.noclobber"
echo "           BEFORE = 0   AFTER = 1   (compile-only; loops forever / loads null — never executed)"

h "SUMMARY"
cat <<'EOF'
                       BEFORE (upstream)        AFTER (fix.patch)
  minimal noclobber    0                        1
  mla    noclobber     4                        5
  mla    prefetch load global_load_b32 (VMEM)   s_load_b32 (SMEM)
  mla    in-loop rfl   1                        0

  Files: ir/{minimal,mla_decode}.ll  asm/mla_decode.{before,after}.s
         annotate/mla_decode.{before,after}.uniform.ll  fix.patch
  Live re-verify with your own opt/llc: see README.md "Reproduce".
EOF