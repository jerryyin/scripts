# [AMDGPU] MachineLICM does not hoist a loop-invariant `v_readfirstlane_b32` out of a uniform loop

## Summary
When a wave-uniform value is broadcast with `v_readfirstlane_b32` and consumed by a
scalar-operand instruction inside a **uniform** loop, MachineLICM leaves the
broadcast in the loop even though it is loop-invariant, so it re-executes every
iteration. `readfirstlane` is `isConvergent` and MachineLICM bails on all
convergent instructions — but a `readfirstlane` whose operands (including its
implicit `EXEC` use) are loop-invariant is safe to hoist: in a uniform loop the
"first active lane" is the same on every iteration.

## Reproduce (no GPU)
`repro.ll` (attached; `@bug` is the essence):
```llvm
define amdgpu_kernel void @bug(ptr addrspace(1) %p, i32 %n) {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %t0  = load i32, ptr addrspace(1) %p
  %g1  = getelementptr i32, ptr addrspace(1) %p, i32 %tid
  %t1  = load i32, ptr addrspace(1) %g1
  %v0  = insertelement <4 x i32> poison, i32 %t0, i64 0
  %v1  = insertelement <4 x i32> %v0, i32 %t1, i64 1
  %v2  = insertelement <4 x i32> %v1, i32 %tid, i64 2
  %desc = insertelement <4 x i32> %v2, i32 %t0, i64 3   ; loop-invariant descriptor
  br label %loop
loop:                                                    ; uniform loop, EXEC never redefined
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  tail call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %desc, <8 x i32> zeroinitializer,
       <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %inc  = add nuw nsw i32 %i, 1
  %cond = icmp slt i32 %inc, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
```
```
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 repro.ll -o -
```

## Actual (loop body of `@bug`)
The `<4 x i32>` descriptor is built once in the preheader (VGPRs `v1,v2,v4,v7`),
but its four VGPR→SGPR copies are re-issued every iteration:
```asm
.LBB0_1:                                ; %loop
	v_readfirstlane_b32 s12, v4
	v_readfirstlane_b32 s13, v1
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v7
	s_add_co_i32 s8, s8, 1
	s_cmp_lt_i32 s8, s10
	tensor_load_to_lds s[12:15], s[0:7]
	s_cbranch_scc1 .LBB0_1
```
`v1,v2,v4,v7` are defined in the preheader and the loop contains no write to
`EXEC`, so the four `v_readfirstlane_b32` are trivially loop-invariant.

## Expected
The four broadcasts hoisted to the preheader; the loop body is just
`tensor_load_to_lds s[8:11], s[0:7]` + the counter.

## Longstanding, not a regression (bisect not needed)
Compiling the **same IR** with two LLVM revisions gives the **same** in-loop count,
so this is not a codegen regression — it reproduces wherever the IR shape occurs:

| IR (identical file) | `llc @ 62b7cf96` | `llc @ 56421f92` |
|---|---|---|
| this `repro.ll` `@bug` | 4 | 4 |
| a real MoE GEMM (gfx1250) | 16 | 16 |

## It is safe, and specific to uniform loops
`repro.ll` also contains `@safe`: identical descriptor and intrinsic, but a per-lane
(divergent) branch redefines `EXEC` inside the loop. There the broadcast result
depends on which lanes are live per iteration and must **not** be hoisted. Any
correct fix must hoist `@bug` and leave `@safe`.

## Root cause
1. `MachineLICMImpl::IsLICMCandidate` returns false for every `isConvergent()`
   instruction.
2. Even relaxing (1), `MachineLoop::isLoopInvariant` must prove the implicit `EXEC`
   use loop-invariant — but AMDGPU does not implement
   `shouldAnalyzePhysregInMachineLoopInfo`, so `EXEC` is never analyzed and a
   `readfirstlane` (whose `resultDependsOnExec` is true) is never loop-invariant.

## Real-world impact
On gfx1250 the wave-uniform TDM gather descriptor (`<4 x i32>` row-index groups)
feeds `tensor_load_to_lds` in the K-loop of MoE GEMMs, emitting **16 (prefill) /
8 (decode) `v_readfirstlane` per iteration** — pure overhead, since the descriptor
is loop-invariant.

## A candidate direction (not proposed as the fix)
A small MachineLICM hook that consults the target for convergent instructions it
may hoist, opted in for `V_READFIRSTLANE_B32`, plus AMDGPU tracking `EXEC` in
MachineLoopInfo so the existing loop-invariance check gates hoisting to uniform
loops. A WIP branch that does this (`@bug`→0, `@safe` unchanged, byte-identical
numerics) is available for reference; happy to share if useful. The right shape of
the fix is a maintainer call — this report is about the missed optimization.
