; Minimal standalone reproducer (no Triton) for:
;   MachineLICM does not hoist a loop-invariant, wave-uniform v_readfirstlane_b32
;   out of a UNIFORM loop, so the broadcast is recomputed every iteration.
;
; Both functions build a loop-invariant <4 x i32> "descriptor" and feed it to the
; scalar-operand intrinsic @llvm.amdgcn.tensor.load.to.lds inside a loop. ISel
; must copy the (VGPR) descriptor to SGPRs with v_readfirstlane_b32.
;
;   @bug   : UNIFORM loop  -> the 4 readfirstlanes are loop-invariant and safe to
;                            hoist, but stay in the loop (the missed optimization).
;   @safe  : DIVERGENT loop (a per-lane branch redefines EXEC) -> the readfirstlanes
;                            are NOT loop-invariant and must stay in the loop.
;
; A correct fix hoists @bug's readfirstlanes to the preheader while leaving
; @safe's untouched.

target triple = "amdgcn-amd-amdhsa"

declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @bug(ptr addrspace(1) %p, i32 %n) {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %t0  = load i32, ptr addrspace(1) %p
  %g1  = getelementptr i32, ptr addrspace(1) %p, i32 %tid
  %t1  = load i32, ptr addrspace(1) %g1
  %v0  = insertelement <4 x i32> poison, i32 %t0, i64 0
  %v1  = insertelement <4 x i32> %v0, i32 %t1, i64 1
  %v2  = insertelement <4 x i32> %v1, i32 %tid, i64 2
  %desc = insertelement <4 x i32> %v2, i32 %t0, i64 3
  br label %loop
loop:                                              ; uniform loop: EXEC never redefined
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  tail call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %desc, <8 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %inc  = add nuw nsw i32 %i, 1
  %cond = icmp slt i32 %inc, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define amdgpu_kernel void @safe(ptr addrspace(1) %p, i32 %n) {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %t0  = load i32, ptr addrspace(1) %p
  %g1  = getelementptr i32, ptr addrspace(1) %p, i32 %tid
  %t1  = load i32, ptr addrspace(1) %g1
  %v0  = insertelement <4 x i32> poison, i32 %t0, i64 0
  %v1  = insertelement <4 x i32> %v0, i32 %t1, i64 1
  %v2  = insertelement <4 x i32> %v1, i32 %tid, i64 2
  %desc = insertelement <4 x i32> %v2, i32 %t0, i64 3
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %cont ]
  tail call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %desc, <8 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %dc = icmp slt i32 %tid, %i                       ; per-lane => divergent branch
  br i1 %dc, label %side, label %cont
side:
  store i32 %i, ptr addrspace(1) %p
  br label %cont
cont:                                               ; divergent loop: EXEC redefined inside
  %inc  = add nuw nsw i32 %i, 1
  %cond = icmp slt i32 %inc, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
