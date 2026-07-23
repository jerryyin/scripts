# [AMDGPU] `isReallyAClobber` — AA-blind clobber check blocks scalar `s_load`

**Component:** backend:AMDGPU · **Kind:** missed-optimization (codegen quality, not a correctness bug) · **Targets:** all AMDGPU (observed on gfx1250; mechanism is target-independent)

A uniform, read-only global (addrspace(1)) load whose only in-loop "clobber" is an LDS (addrspace(3)) write intrinsic is denied `!amdgpu.noclobber`, and is therefore selected as a vector `global_load` + a `v_readfirstlane` round-trip instead of a scalar `s_load`. Alias analysis can prove the LDS write cannot touch the global load, but `AMDGPU::isReallyAClobber` only consults AA for *atomics*; every other memory-writing def falls through to an unconditional "it's a clobber."

## Minimal reproducer

```llvm
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k() {
.lr.ph:
  tail call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %0
0:
  %1 = load <1 x i32>, ptr addrspace(1) null, align 4     ; uniform, read-only
  fence release
  br label %0
}
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #0
attributes #0 = { ... memory(argmem: readwrite, inaccessiblemem: readwrite) }
```

```
$ opt -passes=amdgpu-annotate-uniform -mcpu=gfx1250 -S ir/minimal.ll | grep -c amdgpu.noclobber
0        # upstream: load NOT tagged noclobber   (with the fix: 1)
```

The clobber `llvm.amdgcn.tensor.load.to.lds` touches only `argmem`/`inaccessiblemem` (LDS), which cannot alias the `addrspace(1)` load; `getModRefInfo` returns `NoModRef`, but `isReallyAClobber` never asks.

## Root-cause trace (file:line)

1. `SIISelLowering.cpp:13302-13313` `LowerLOAD`: the scalar-global path needs `(Load->isInvariant() || isMemOpHasNoClobberedMemOperand(Load))` **and** `(!Op->isDivergent() || isUniformMMO(MMO))`. The load's pointer *is* tagged `!amdgpu.uniform`, but its MMO lacks `MONoClobber` → first clause fails → VMEM.
2. `AMDGPUAnnotateUniformValues.cpp:81` sets `noclobber` only if `!isClobberedInFunction`.
3. `isClobberedInFunction` (`AMDGPUMemoryUtils.cpp:409`) walks MemorySSA; for an in-loop load it expands the loop-header `MemoryPhi` to the loop-body defs and calls `isReallyAClobber` on each.
4. `isReallyAClobber` (`AMDGPUMemoryUtils.cpp:367`) returns `false` only for fences, a fixed barrier-intrinsic whitelist, and AA-NoAlias *atomics*; the `tensor.load.to.lds` def is none of these → blanket `return true`, with no AA query.

(MemorySSA's own AA-refined walker skips a plain non-aliasing `store`, so this only bites for defs it surfaces conservatively — memory intrinsics/calls reached via the phi.)

## Fix

Consult AA generally (`getModRefInfo`) for the fall-through case, keeping the fence/barrier whitelist and the pointer-level atomic special case (a synchronizing atomic on unrelated memory must stay non-clobbering — `getModRefInfo` over-reports `Mod` for its ordering effects; folding atomics into it regresses `noclobber-barrier.ll`). The load's `MemoryLocation` is threaded through for addressing/size precision.

```diff
-bool isReallyAClobber(const Value *Ptr, MemoryDef *Def, AAResults *AA) {
+bool isReallyAClobber(const MemoryLocation &Loc, MemoryDef *Def, AAResults *AA) {
   ...                                  // fence + barrier-intrinsic whitelist unchanged
-  const auto checkNoAlias = [AA, Ptr](auto I) -> bool {
-    return I && AA->isNoAlias(I->getPointerOperand(), Ptr);
+  const auto checkNoAlias = [AA, &Loc](auto I) -> bool {
+    return I && AA->isNoAlias(I->getPointerOperand(), Loc.Ptr);
   };
   if (checkNoAlias(dyn_cast<AtomicCmpXchgInst>(DefInst)) ||
       checkNoAlias(dyn_cast<AtomicRMWInst>(DefInst)))
     return false;
-  return true;
+  // Any other memory-writing def is a real clobber only if AA says it can
+  // modify the loaded location.
+  return isModSet(AA->getModRefInfo(DefInst, Loc));
 }
```

(+ caller passes `Loc`; header signature + `class MemoryLocation;` fwd-decl updated.) Sound by construction: `getModRefInfo` returns `NoModRef` only when the def provably cannot modify the location, so `noclobber` is never granted unsoundly. Full patch: `fix.patch` (`git am`-able).

## Effect on a real kernel (`ir/mla_decode.ll`, gfx1250)

The AITer paged/MLA-decode kernel's software-pipelined block-table prefetch `physical_block_idx = load(block_tables + …)` feeds a scalar TDM `tensor_load_to_lds` descriptor. Its out-of-loop sibling block-table loads already get `noclobber` and select `s_load` in both configs — proving the in-loop load is `s_load`-representable and only the missing tag differs. Single input delta = `fix.patch`, same `llc`:

| | BEFORE | AFTER (fix) |
|---|---|---|
| `!amdgpu.noclobber` count (annotate-uniform) | 4 | **5** |
| block-table prefetch load (mla.py:847, in-loop) | `global_load_b32` (VGPR) | **`s_load_b32`** (SGPR) |
| in-loop `v_readfirstlane` | 1 | **0** |

`noalias` is not involved: the prefetch carries none — the fix rests on addrspace(3) ↛ addrspace(1), which AA proves unconditionally.

## Reproduce (no Triton, no GPU)

BEFORE needs only an unmodified upstream `opt`+`llc`; AFTER needs a build of the fix:

```bash
./reproduce.sh                                   # BEFORE (override with OPT=/LLC=)
# build the fix: apply fix.patch (or check out the branch below), then:
#   cmake -G Ninja -S llvm -B build -DLLVM_TARGETS_TO_BUILD=AMDGPU && ninja -C build opt llc
LLVM_BUILD=/path/to/llvm-project/build ./reproduce.sh   # BEFORE + AFTER + delta
```

`reproduce.sh` is also a guided walkthrough (prints the `file:line` + snippet + why for each chunk). Committed BEFORE/AFTER evidence is in `asm/` and `annotate/`.

Fix branch: `users/jerryyin/amdgpu-isreallyaclobber-aa` (commit `2958d9f07`, rebased on `llvm/llvm-project` main) — https://github.com/llvm/llvm-project/tree/users/jerryyin/amdgpu-isreallyaclobber-aa
