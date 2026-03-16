; To reproduce the .optimized.ll from the .linked.ll, run:
; opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 --passes='verify,memprof-remove-attributes,annotation2metadata,forceattrs,inferattrs,coro-early,function<eager-inv>(ee-instrument<>,lower-expect,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;no-switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,sroa<modify-cfg>,early-cse<>),openmp-opt,amdgpu-printf-runtime-binding,ipsccp,called-value-propagation,globalopt,function<eager-inv>(mem2reg,instcombine<max-iterations=1;no-verify-fixpoint>,amdgpu-usenative,amdgpu-simplifylib,amdgpu-uniform-intrinsic-combine,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>),always-inline,require<globals-aa>,function(invalidate<aa>),require<profile-summary>,cgscc(devirt<4>(inline,function-attrs<skip-non-recursive-function-attrs>,openmp-opt-cgscc,function(amdgpu-promote-kernel-arguments,infer-address-spaces,amdgpu-lower-kernel-attributes,amdgpu-promote-alloca-to-vector),function<eager-inv;no-rerun>(sroa<modify-cfg>,early-cse<memssa>,speculative-execution<only-if-divergent-target>,jump-threading,correlated-propagation,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>,aggressive-instcombine,libcalls-shrinkwrap,amdgpu-usenative,amdgpu-simplifylib,amdgpu-uniform-intrinsic-combine,tailcallelim,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,reassociate,constraint-elimination,loop-mssa(loop-instsimplify,loop-simplifycfg,licm<no-allowspeculation>,loop-rotate<header-duplication;no-prepare-for-lto>,licm<allowspeculation>,simple-loop-unswitch<no-nontrivial;trivial>),simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>,loop(loop-idiom,indvars,extra-simple-loop-unswitch-passes,loop-deletion,loop-unroll-full),sroa<modify-cfg>,vector-combine,mldst-motion<no-split-footer-bb>,gvn<>,sccp,bdce,instcombine<max-iterations=1;no-verify-fixpoint>,amdgpu-usenative,amdgpu-simplifylib,amdgpu-uniform-intrinsic-combine,jump-threading,correlated-propagation,adce,memcpyopt,dse,move-auto-init,loop-mssa(licm<allowspeculation>),coro-elide,infer-address-spaces,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;switch-to-arithmetic;no-switch-to-lookup;keep-loops;hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>,amdgpu-usenative,amdgpu-simplifylib,amdgpu-uniform-intrinsic-combine),function-attrs,function(require<should-not-run-function-passes>),coro-split,coro-annotation-elide)),deadargelim,coro-cleanup,globalopt,globaldce,elim-avail-extern,rpo-function-attrs,recompute-globalsaa,function<eager-inv>(drop-unnecessary-assumes,float2int,lower-constant-intrinsics,loop(loop-rotate<header-duplication;no-prepare-for-lto>,loop-deletion),loop-distribute,inject-tli-mappings,loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only;>,drop-unnecessary-assumes,infer-alignment,loop-load-elim,instcombine<max-iterations=1;no-verify-fixpoint>,simplifycfg<bonus-inst-threshold=1;forward-switch-cond;switch-range-to-icmp;switch-to-arithmetic;switch-to-lookup;no-keep-loops;hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,slp-vectorizer,vector-combine,instcombine<max-iterations=1;no-verify-fixpoint>,loop-unroll<O2>,transform-warning,sroa<preserve-cfg>,infer-alignment,instcombine<max-iterations=1;no-verify-fixpoint>,loop-mssa(licm<allowspeculation>),alignment-from-assumptions,infer-address-spaces,loop-sink,instsimplify,div-rem-pairs,tailcallelim,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;speculate-unpredictables>),alloc-token,amdgpu-attributor,globaldce,constmerge,cg-profile,rel-lookup-table-converter,function(annotation-remarks),verify' <.linked.ll>
; The flag '-S' is to emit LLVMIR.
; The behavior of some passes depends on '-mtriple' and '-mcpu'.

; ModuleID = 'matmul_dispatch_0'
source_filename = "matmul_dispatch_0"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__shared_memory___0 = private addrspace(3) global [3 x [16 x [128 x float]]] undef, align 16
@__shared_memory__ = private addrspace(3) global [3 x [128 x [16 x float]]] undef, align 16

; Function Attrs: alwaysinline nounwind
define amdgpu_kernel void @matmul_dispatch_0_matmul_4096x4096x4096_f32(ptr addrspace(1) inreg noalias noundef nonnull readonly align 16 captures(none) %0, ptr addrspace(1) inreg noalias noundef nonnull readonly align 16 captures(none) %1, ptr addrspace(1) inreg noalias noundef nonnull writeonly align 16 %2) local_unnamed_addr #0 !reqd_work_group_size !2 {
  %4 = tail call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x()
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %0, i64 64) ]
  %5 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %0, i16 0, i64 67108864, i32 159744)
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %1, i64 64) ]
  %6 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %1, i16 0, i64 67108864, i32 159744)
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %2, i64 64) ]
  %7 = lshr i32 %4, 6
  %8 = and i32 %4, 63
  %9 = lshr i32 %8, 4
  %10 = and i32 %4, 15
  %11 = shl nuw nsw i32 %8, 2
  %12 = lshr i32 %8, 2
  %13 = and i32 %11, 12
  %14 = or disjoint i32 %11, 256
  %15 = lshr i32 %14, 4
  %16 = lshr i32 %8, 5
  %17 = and i32 %11, 124
  %18 = lshr i32 %14, 7
  %19 = shl nuw nsw i32 %7, 2
  %20 = shl nuw nsw i32 %7, 5
  %21 = or disjoint i32 %19, 16
  %22 = tail call range(i32 0, 1024) i32 @llvm.amdgcn.workgroup.id.x()
  %23 = shl nuw nsw i32 %22, 7
  %24 = and i32 %23, 3968
  %25 = shl nuw nsw i32 %22, 2
  %26 = and i32 %25, 3968
  %27 = or disjoint i32 %20, %26
  %28 = or disjoint i32 %27, %12
  %29 = shl nuw nsw i32 %28, 12
  %30 = getelementptr float, ptr addrspace(7) %5, i32 %29
  %31 = getelementptr float, ptr addrspace(7) %30, i32 %13
  %32 = shl nuw nsw i32 %7, 9
  %33 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %32
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %31, ptr addrspace(3) %33, i32 16, i32 0, i32 0)
  %34 = or disjoint i32 %15, %27
  %35 = shl nuw nsw i32 %34, 12
  %36 = getelementptr float, ptr addrspace(7) %5, i32 %35
  %37 = getelementptr float, ptr addrspace(7) %36, i32 %13
  %38 = or disjoint i32 %32, 256
  %39 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %38
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %37, ptr addrspace(3) %39, i32 16, i32 0, i32 0)
  %40 = or disjoint i32 %19, %16
  %41 = or disjoint i32 %17, %24
  %.idx = shl nuw nsw i32 %40, 14
  %42 = getelementptr i8, ptr addrspace(7) %6, i32 %.idx
  %43 = getelementptr float, ptr addrspace(7) %42, i32 %41
  %44 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %32
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %43, ptr addrspace(3) %44, i32 16, i32 0, i32 0)
  %45 = or disjoint i32 %18, %19
  %.idx1 = shl nuw nsw i32 %45, 14
  %46 = getelementptr i8, ptr addrspace(7) %6, i32 %.idx1
  %47 = getelementptr float, ptr addrspace(7) %46, i32 %41
  %48 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %38
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %47, ptr addrspace(3) %48, i32 16, i32 0, i32 0)
  tail call void @llvm.amdgcn.asyncmark()
  %49 = getelementptr float, ptr addrspace(7) %5, i32 %13
  %50 = getelementptr float, ptr addrspace(7) %49, i32 %29
  %51 = getelementptr i8, ptr addrspace(7) %50, i32 64
  %52 = getelementptr i8, ptr addrspace(3) %33, i32 8192
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %51, ptr addrspace(3) %52, i32 16, i32 0, i32 0)
  %53 = getelementptr float, ptr addrspace(7) %49, i32 %35
  %54 = getelementptr i8, ptr addrspace(7) %53, i32 64
  %55 = getelementptr i8, ptr addrspace(3) %33, i32 9216
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %54, ptr addrspace(3) %55, i32 16, i32 0, i32 0)
  %56 = or disjoint i32 %21, %16
  %.idx2 = shl nuw nsw i32 %56, 14
  %57 = getelementptr i8, ptr addrspace(7) %6, i32 %.idx2
  %58 = getelementptr float, ptr addrspace(7) %57, i32 %41
  %59 = getelementptr i8, ptr addrspace(3) %44, i32 8192
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %58, ptr addrspace(3) %59, i32 16, i32 0, i32 0)
  %60 = or disjoint i32 %18, %21
  %.idx3 = shl nuw nsw i32 %60, 14
  %61 = getelementptr i8, ptr addrspace(7) %6, i32 %.idx3
  %62 = getelementptr float, ptr addrspace(7) %61, i32 %41
  %63 = getelementptr i8, ptr addrspace(3) %44, i32 9216
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %62, ptr addrspace(3) %63, i32 16, i32 0, i32 0)
  tail call void @llvm.amdgcn.asyncmark()
  %64 = lshr i32 %4, 1
  %65 = and i32 %64, 64
  %66 = or disjoint i32 %65, %10
  %67 = or disjoint i32 %9, 4
  %68 = or disjoint i32 %9, 8
  %69 = or disjoint i32 %9, 12
  %70 = and i32 %4, 79
  %71 = or disjoint i32 %70, 16
  %72 = or disjoint i32 %70, 32
  %73 = or disjoint i32 %70, 48
  %invariant.gep16 = getelementptr float, ptr addrspace(7) %6, i32 %41
  %74 = shl nuw nsw i32 %66, 4
  %75 = shl nuw nsw i32 %66, 4
  %76 = or disjoint i32 %75, 256
  %77 = shl nuw nsw i32 %66, 4
  %78 = or disjoint i32 %77, 512
  %79 = shl nuw nsw i32 %66, 4
  %80 = or disjoint i32 %79, 768
  %81 = shl nuw nsw i32 %9, 7
  %82 = shl nuw nsw i32 %67, 7
  %83 = shl nuw nsw i32 %68, 7
  %84 = shl nuw nsw i32 %69, 7
  br label %85

85:                                               ; preds = %3, %85
  %86 = phi { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } [ { ptr addrspace(3) inttoptr (i64 3735928559 to ptr addrspace(3)), ptr addrspace(3) @__shared_memory__, i64 2048, [2 x i64] [i64 128, i64 16], [2 x i64] [i64 16, i64 1] }, %3 ], [ %106, %85 ]
  %87 = phi { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } [ { ptr addrspace(3) inttoptr (i64 3735928559 to ptr addrspace(3)), ptr addrspace(3) @__shared_memory__, i64 0, [2 x i64] [i64 128, i64 16], [2 x i64] [i64 16, i64 1] }, %3 ], [ %86, %85 ]
  %88 = phi { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } [ { ptr addrspace(3) inttoptr (i64 3735928559 to ptr addrspace(3)), ptr addrspace(3) @__shared_memory___0, i64 2048, [2 x i64] [i64 16, i64 128], [2 x i64] [i64 128, i64 1] }, %3 ], [ %101, %85 ]
  %89 = phi { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } [ { ptr addrspace(3) inttoptr (i64 3735928559 to ptr addrspace(3)), ptr addrspace(3) @__shared_memory___0, i64 0, [2 x i64] [i64 16, i64 128], [2 x i64] [i64 128, i64 1] }, %3 ], [ %88, %85 ]
  %90 = phi [4 x [4 x [4 x <1 x float>]]] [ zeroinitializer, %3 ], [ %574, %85 ]
  %91 = phi i32 [ 0, %3 ], [ %575, %85 ]
  %92 = add nuw nsw i32 %91, 8
  %93 = lshr exact i32 %92, 2
  %.lhs.trunc = trunc nuw i32 %93 to i8
  %94 = urem i8 %.lhs.trunc, 3
  %.zext = zext nneg i8 %94 to i32
  %95 = shl nuw nsw i32 %.zext, 11
  %96 = zext nneg i32 %95 to i64
  %97 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } { ptr addrspace(3) inttoptr (i64 3735928559 to ptr addrspace(3)), ptr addrspace(3) @__shared_memory___0, i64 poison, [2 x i64] poison, [2 x i64] poison }, i64 %96, 2
  %98 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %97, i64 16, 3, 0
  %99 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %98, i64 128, 4, 0
  %100 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %99, i64 128, 3, 1
  %101 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %100, i64 1, 4, 1
  %102 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } { ptr addrspace(3) inttoptr (i64 3735928559 to ptr addrspace(3)), ptr addrspace(3) @__shared_memory__, i64 poison, [2 x i64] poison, [2 x i64] poison }, i64 %96, 2
  %103 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %102, i64 128, 3, 0
  %104 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %103, i64 16, 4, 0
  %105 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %104, i64 16, 3, 1
  %106 = insertvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %105, i64 1, 4, 1
  %107 = shl nuw nsw i32 %92, 2
  %108 = or disjoint i32 %107, %19
  fence syncscope("workgroup") release, !mmra !3
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !3
  %gep = getelementptr float, ptr addrspace(7) %31, i32 %107
  %gep11 = getelementptr float, ptr addrspace(3) %33, i32 %95
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %gep, ptr addrspace(3) %gep11, i32 16, i32 0, i32 0)
  %gep13 = getelementptr float, ptr addrspace(7) %37, i32 %107
  %gep15 = getelementptr float, ptr addrspace(3) %39, i32 %95
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %gep13, ptr addrspace(3) %gep15, i32 16, i32 0, i32 0)
  %109 = or disjoint i32 %108, %16
  %.idx4 = shl nuw nsw i32 %109, 14
  %gep17 = getelementptr i8, ptr addrspace(7) %invariant.gep16, i32 %.idx4
  %gep19 = getelementptr float, ptr addrspace(3) %44, i32 %95
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %gep17, ptr addrspace(3) %gep19, i32 16, i32 0, i32 0)
  %110 = or disjoint i32 %108, %18
  %.idx5 = shl nuw nsw i32 %110, 14
  %gep21 = getelementptr i8, ptr addrspace(7) %invariant.gep16, i32 %.idx5
  %gep23 = getelementptr float, ptr addrspace(3) %48, i32 %95
  tail call void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) %gep21, ptr addrspace(3) %gep23, i32 16, i32 0, i32 0)
  tail call void @llvm.amdgcn.asyncmark()
  tail call void @llvm.amdgcn.wait.asyncmark(i16 2)
  fence syncscope("workgroup") release, !mmra !3
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !3
  %111 = extractvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %87, 1
  %112 = extractvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %87, 2
  %113 = trunc i64 %112 to i32
  %114 = getelementptr float, ptr addrspace(3) %111, i32 %113
  %115 = getelementptr float, ptr addrspace(3) %114, i32 %9
  %116 = getelementptr float, ptr addrspace(3) %115, i32 %74
  %117 = load <1 x float>, ptr addrspace(3) %116, align 4
  %118 = getelementptr float, ptr addrspace(3) %114, i32 %67
  %119 = getelementptr float, ptr addrspace(3) %118, i32 %74
  %120 = load <1 x float>, ptr addrspace(3) %119, align 4
  %121 = getelementptr float, ptr addrspace(3) %114, i32 %68
  %122 = getelementptr float, ptr addrspace(3) %121, i32 %74
  %123 = load <1 x float>, ptr addrspace(3) %122, align 4
  %124 = getelementptr float, ptr addrspace(3) %114, i32 %69
  %125 = getelementptr float, ptr addrspace(3) %124, i32 %74
  %126 = load <1 x float>, ptr addrspace(3) %125, align 4
  %127 = getelementptr float, ptr addrspace(3) %115, i32 %76
  %128 = load <1 x float>, ptr addrspace(3) %127, align 4
  %129 = getelementptr float, ptr addrspace(3) %118, i32 %76
  %130 = load <1 x float>, ptr addrspace(3) %129, align 4
  %131 = getelementptr float, ptr addrspace(3) %121, i32 %76
  %132 = load <1 x float>, ptr addrspace(3) %131, align 4
  %133 = getelementptr float, ptr addrspace(3) %124, i32 %76
  %134 = load <1 x float>, ptr addrspace(3) %133, align 4
  %135 = getelementptr float, ptr addrspace(3) %115, i32 %78
  %136 = load <1 x float>, ptr addrspace(3) %135, align 4
  %137 = getelementptr float, ptr addrspace(3) %118, i32 %78
  %138 = load <1 x float>, ptr addrspace(3) %137, align 4
  %139 = getelementptr float, ptr addrspace(3) %121, i32 %78
  %140 = load <1 x float>, ptr addrspace(3) %139, align 4
  %141 = getelementptr float, ptr addrspace(3) %124, i32 %78
  %142 = load <1 x float>, ptr addrspace(3) %141, align 4
  %143 = getelementptr float, ptr addrspace(3) %115, i32 %80
  %144 = load <1 x float>, ptr addrspace(3) %143, align 4
  %145 = getelementptr float, ptr addrspace(3) %118, i32 %80
  %146 = load <1 x float>, ptr addrspace(3) %145, align 4
  %147 = getelementptr float, ptr addrspace(3) %121, i32 %80
  %148 = load <1 x float>, ptr addrspace(3) %147, align 4
  %149 = getelementptr float, ptr addrspace(3) %124, i32 %80
  %150 = load <1 x float>, ptr addrspace(3) %149, align 4
  %151 = extractvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %89, 1
  %152 = extractvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %89, 2
  %153 = trunc i64 %152 to i32
  %154 = getelementptr float, ptr addrspace(3) %151, i32 %153
  %155 = getelementptr float, ptr addrspace(3) %154, i32 %70
  %156 = getelementptr float, ptr addrspace(3) %155, i32 %81
  %157 = load <1 x float>, ptr addrspace(3) %156, align 4
  %158 = getelementptr float, ptr addrspace(3) %154, i32 %71
  %159 = getelementptr float, ptr addrspace(3) %158, i32 %81
  %160 = load <1 x float>, ptr addrspace(3) %159, align 4
  %161 = getelementptr float, ptr addrspace(3) %154, i32 %72
  %162 = getelementptr float, ptr addrspace(3) %161, i32 %81
  %163 = load <1 x float>, ptr addrspace(3) %162, align 4
  %164 = getelementptr float, ptr addrspace(3) %154, i32 %73
  %165 = getelementptr float, ptr addrspace(3) %164, i32 %81
  %166 = load <1 x float>, ptr addrspace(3) %165, align 4
  %167 = getelementptr float, ptr addrspace(3) %155, i32 %82
  %168 = load <1 x float>, ptr addrspace(3) %167, align 4
  %169 = getelementptr float, ptr addrspace(3) %158, i32 %82
  %170 = load <1 x float>, ptr addrspace(3) %169, align 4
  %171 = getelementptr float, ptr addrspace(3) %161, i32 %82
  %172 = load <1 x float>, ptr addrspace(3) %171, align 4
  %173 = getelementptr float, ptr addrspace(3) %164, i32 %82
  %174 = load <1 x float>, ptr addrspace(3) %173, align 4
  %175 = getelementptr float, ptr addrspace(3) %155, i32 %83
  %176 = load <1 x float>, ptr addrspace(3) %175, align 4
  %177 = getelementptr float, ptr addrspace(3) %158, i32 %83
  %178 = load <1 x float>, ptr addrspace(3) %177, align 4
  %179 = getelementptr float, ptr addrspace(3) %161, i32 %83
  %180 = load <1 x float>, ptr addrspace(3) %179, align 4
  %181 = getelementptr float, ptr addrspace(3) %164, i32 %83
  %182 = load <1 x float>, ptr addrspace(3) %181, align 4
  %183 = getelementptr float, ptr addrspace(3) %155, i32 %84
  %184 = load <1 x float>, ptr addrspace(3) %183, align 4
  %185 = getelementptr float, ptr addrspace(3) %158, i32 %84
  %186 = load <1 x float>, ptr addrspace(3) %185, align 4
  %187 = getelementptr float, ptr addrspace(3) %161, i32 %84
  %188 = load <1 x float>, ptr addrspace(3) %187, align 4
  %189 = getelementptr float, ptr addrspace(3) %164, i32 %84
  %190 = load <1 x float>, ptr addrspace(3) %189, align 4
  %191 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 0, 0
  %192 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 0, 1
  %193 = shufflevector <1 x float> %191, <1 x float> %192, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %194 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 0, 2
  %195 = shufflevector <1 x float> %194, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %196 = shufflevector <4 x float> %193, <4 x float> %195, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %197 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 0, 3
  %198 = shufflevector <1 x float> %197, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %199 = shufflevector <4 x float> %196, <4 x float> %198, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %200 = extractelement <1 x float> %117, i64 0
  %201 = extractelement <1 x float> %157, i64 0
  %202 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %200, float %201, <4 x float> %199, i32 0, i32 0, i32 0)
  %203 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 1, 0
  %204 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 1, 1
  %205 = shufflevector <1 x float> %203, <1 x float> %204, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %206 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 1, 2
  %207 = shufflevector <1 x float> %206, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %208 = shufflevector <4 x float> %205, <4 x float> %207, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %209 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 1, 3
  %210 = shufflevector <1 x float> %209, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %211 = shufflevector <4 x float> %208, <4 x float> %210, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %212 = extractelement <1 x float> %160, i64 0
  %213 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %200, float %212, <4 x float> %211, i32 0, i32 0, i32 0)
  %214 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 2, 0
  %215 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 2, 1
  %216 = shufflevector <1 x float> %214, <1 x float> %215, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %217 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 2, 2
  %218 = shufflevector <1 x float> %217, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %219 = shufflevector <4 x float> %216, <4 x float> %218, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %220 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 2, 3
  %221 = shufflevector <1 x float> %220, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %222 = shufflevector <4 x float> %219, <4 x float> %221, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %223 = extractelement <1 x float> %163, i64 0
  %224 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %200, float %223, <4 x float> %222, i32 0, i32 0, i32 0)
  %225 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 3, 0
  %226 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 3, 1
  %227 = shufflevector <1 x float> %225, <1 x float> %226, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %228 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 3, 2
  %229 = shufflevector <1 x float> %228, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %230 = shufflevector <4 x float> %227, <4 x float> %229, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %231 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 0, 3, 3
  %232 = shufflevector <1 x float> %231, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %233 = shufflevector <4 x float> %230, <4 x float> %232, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %234 = extractelement <1 x float> %166, i64 0
  %235 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %200, float %234, <4 x float> %233, i32 0, i32 0, i32 0)
  %236 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 0, 0
  %237 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 0, 1
  %238 = shufflevector <1 x float> %236, <1 x float> %237, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %239 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 0, 2
  %240 = shufflevector <1 x float> %239, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %241 = shufflevector <4 x float> %238, <4 x float> %240, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %242 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 0, 3
  %243 = shufflevector <1 x float> %242, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %244 = shufflevector <4 x float> %241, <4 x float> %243, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %245 = extractelement <1 x float> %128, i64 0
  %246 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %245, float %201, <4 x float> %244, i32 0, i32 0, i32 0)
  %247 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 1, 0
  %248 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 1, 1
  %249 = shufflevector <1 x float> %247, <1 x float> %248, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %250 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 1, 2
  %251 = shufflevector <1 x float> %250, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %252 = shufflevector <4 x float> %249, <4 x float> %251, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %253 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 1, 3
  %254 = shufflevector <1 x float> %253, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %255 = shufflevector <4 x float> %252, <4 x float> %254, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %256 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %245, float %212, <4 x float> %255, i32 0, i32 0, i32 0)
  %257 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 2, 0
  %258 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 2, 1
  %259 = shufflevector <1 x float> %257, <1 x float> %258, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %260 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 2, 2
  %261 = shufflevector <1 x float> %260, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %262 = shufflevector <4 x float> %259, <4 x float> %261, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %263 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 2, 3
  %264 = shufflevector <1 x float> %263, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %265 = shufflevector <4 x float> %262, <4 x float> %264, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %266 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %245, float %223, <4 x float> %265, i32 0, i32 0, i32 0)
  %267 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 3, 0
  %268 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 3, 1
  %269 = shufflevector <1 x float> %267, <1 x float> %268, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %270 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 3, 2
  %271 = shufflevector <1 x float> %270, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %272 = shufflevector <4 x float> %269, <4 x float> %271, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %273 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 1, 3, 3
  %274 = shufflevector <1 x float> %273, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %275 = shufflevector <4 x float> %272, <4 x float> %274, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %276 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %245, float %234, <4 x float> %275, i32 0, i32 0, i32 0)
  %277 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 0, 0
  %278 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 0, 1
  %279 = shufflevector <1 x float> %277, <1 x float> %278, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %280 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 0, 2
  %281 = shufflevector <1 x float> %280, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %282 = shufflevector <4 x float> %279, <4 x float> %281, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %283 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 0, 3
  %284 = shufflevector <1 x float> %283, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %285 = shufflevector <4 x float> %282, <4 x float> %284, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %286 = extractelement <1 x float> %136, i64 0
  %287 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %286, float %201, <4 x float> %285, i32 0, i32 0, i32 0)
  %288 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 1, 0
  %289 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 1, 1
  %290 = shufflevector <1 x float> %288, <1 x float> %289, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %291 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 1, 2
  %292 = shufflevector <1 x float> %291, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %293 = shufflevector <4 x float> %290, <4 x float> %292, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %294 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 1, 3
  %295 = shufflevector <1 x float> %294, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %296 = shufflevector <4 x float> %293, <4 x float> %295, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %297 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %286, float %212, <4 x float> %296, i32 0, i32 0, i32 0)
  %298 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 2, 0
  %299 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 2, 1
  %300 = shufflevector <1 x float> %298, <1 x float> %299, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %301 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 2, 2
  %302 = shufflevector <1 x float> %301, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %303 = shufflevector <4 x float> %300, <4 x float> %302, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %304 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 2, 3
  %305 = shufflevector <1 x float> %304, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %306 = shufflevector <4 x float> %303, <4 x float> %305, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %307 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %286, float %223, <4 x float> %306, i32 0, i32 0, i32 0)
  %308 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 3, 0
  %309 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 3, 1
  %310 = shufflevector <1 x float> %308, <1 x float> %309, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %311 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 3, 2
  %312 = shufflevector <1 x float> %311, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %313 = shufflevector <4 x float> %310, <4 x float> %312, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %314 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 2, 3, 3
  %315 = shufflevector <1 x float> %314, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %316 = shufflevector <4 x float> %313, <4 x float> %315, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %317 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %286, float %234, <4 x float> %316, i32 0, i32 0, i32 0)
  %318 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 0, 0
  %319 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 0, 1
  %320 = shufflevector <1 x float> %318, <1 x float> %319, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %321 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 0, 2
  %322 = shufflevector <1 x float> %321, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %323 = shufflevector <4 x float> %320, <4 x float> %322, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %324 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 0, 3
  %325 = shufflevector <1 x float> %324, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %326 = shufflevector <4 x float> %323, <4 x float> %325, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %327 = extractelement <1 x float> %144, i64 0
  %328 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %327, float %201, <4 x float> %326, i32 0, i32 0, i32 0)
  %329 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 1, 0
  %330 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 1, 1
  %331 = shufflevector <1 x float> %329, <1 x float> %330, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %332 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 1, 2
  %333 = shufflevector <1 x float> %332, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %334 = shufflevector <4 x float> %331, <4 x float> %333, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %335 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 1, 3
  %336 = shufflevector <1 x float> %335, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %337 = shufflevector <4 x float> %334, <4 x float> %336, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %338 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %327, float %212, <4 x float> %337, i32 0, i32 0, i32 0)
  %339 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 2, 0
  %340 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 2, 1
  %341 = shufflevector <1 x float> %339, <1 x float> %340, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %342 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 2, 2
  %343 = shufflevector <1 x float> %342, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %344 = shufflevector <4 x float> %341, <4 x float> %343, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %345 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 2, 3
  %346 = shufflevector <1 x float> %345, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %347 = shufflevector <4 x float> %344, <4 x float> %346, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %348 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %327, float %223, <4 x float> %347, i32 0, i32 0, i32 0)
  %349 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 3, 0
  %350 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 3, 1
  %351 = shufflevector <1 x float> %349, <1 x float> %350, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %352 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 3, 2
  %353 = shufflevector <1 x float> %352, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %354 = shufflevector <4 x float> %351, <4 x float> %353, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>
  %355 = extractvalue [4 x [4 x [4 x <1 x float>]]] %90, 3, 3, 3
  %356 = shufflevector <1 x float> %355, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  %357 = shufflevector <4 x float> %354, <4 x float> %356, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %358 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %327, float %234, <4 x float> %357, i32 0, i32 0, i32 0)
  %359 = extractelement <1 x float> %120, i64 0
  %360 = extractelement <1 x float> %168, i64 0
  %361 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %359, float %360, <4 x float> %202, i32 0, i32 0, i32 0)
  %362 = extractelement <1 x float> %170, i64 0
  %363 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %359, float %362, <4 x float> %213, i32 0, i32 0, i32 0)
  %364 = extractelement <1 x float> %172, i64 0
  %365 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %359, float %364, <4 x float> %224, i32 0, i32 0, i32 0)
  %366 = extractelement <1 x float> %174, i64 0
  %367 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %359, float %366, <4 x float> %235, i32 0, i32 0, i32 0)
  %368 = extractelement <1 x float> %130, i64 0
  %369 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %368, float %360, <4 x float> %246, i32 0, i32 0, i32 0)
  %370 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %368, float %362, <4 x float> %256, i32 0, i32 0, i32 0)
  %371 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %368, float %364, <4 x float> %266, i32 0, i32 0, i32 0)
  %372 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %368, float %366, <4 x float> %276, i32 0, i32 0, i32 0)
  %373 = extractelement <1 x float> %138, i64 0
  %374 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %373, float %360, <4 x float> %287, i32 0, i32 0, i32 0)
  %375 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %373, float %362, <4 x float> %297, i32 0, i32 0, i32 0)
  %376 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %373, float %364, <4 x float> %307, i32 0, i32 0, i32 0)
  %377 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %373, float %366, <4 x float> %317, i32 0, i32 0, i32 0)
  %378 = extractelement <1 x float> %146, i64 0
  %379 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %378, float %360, <4 x float> %328, i32 0, i32 0, i32 0)
  %380 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %378, float %362, <4 x float> %338, i32 0, i32 0, i32 0)
  %381 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %378, float %364, <4 x float> %348, i32 0, i32 0, i32 0)
  %382 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %378, float %366, <4 x float> %358, i32 0, i32 0, i32 0)
  %383 = extractelement <1 x float> %123, i64 0
  %384 = extractelement <1 x float> %176, i64 0
  %385 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %383, float %384, <4 x float> %361, i32 0, i32 0, i32 0)
  %386 = extractelement <1 x float> %178, i64 0
  %387 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %383, float %386, <4 x float> %363, i32 0, i32 0, i32 0)
  %388 = extractelement <1 x float> %180, i64 0
  %389 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %383, float %388, <4 x float> %365, i32 0, i32 0, i32 0)
  %390 = extractelement <1 x float> %182, i64 0
  %391 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %383, float %390, <4 x float> %367, i32 0, i32 0, i32 0)
  %392 = extractelement <1 x float> %132, i64 0
  %393 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %392, float %384, <4 x float> %369, i32 0, i32 0, i32 0)
  %394 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %392, float %386, <4 x float> %370, i32 0, i32 0, i32 0)
  %395 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %392, float %388, <4 x float> %371, i32 0, i32 0, i32 0)
  %396 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %392, float %390, <4 x float> %372, i32 0, i32 0, i32 0)
  %397 = extractelement <1 x float> %140, i64 0
  %398 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %397, float %384, <4 x float> %374, i32 0, i32 0, i32 0)
  %399 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %397, float %386, <4 x float> %375, i32 0, i32 0, i32 0)
  %400 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %397, float %388, <4 x float> %376, i32 0, i32 0, i32 0)
  %401 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %397, float %390, <4 x float> %377, i32 0, i32 0, i32 0)
  %402 = extractelement <1 x float> %148, i64 0
  %403 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %402, float %384, <4 x float> %379, i32 0, i32 0, i32 0)
  %404 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %402, float %386, <4 x float> %380, i32 0, i32 0, i32 0)
  %405 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %402, float %388, <4 x float> %381, i32 0, i32 0, i32 0)
  %406 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %402, float %390, <4 x float> %382, i32 0, i32 0, i32 0)
  %407 = extractelement <1 x float> %126, i64 0
  %408 = extractelement <1 x float> %184, i64 0
  %409 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %407, float %408, <4 x float> %385, i32 0, i32 0, i32 0)
  %410 = extractelement <1 x float> %186, i64 0
  %411 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %407, float %410, <4 x float> %387, i32 0, i32 0, i32 0)
  %412 = extractelement <1 x float> %188, i64 0
  %413 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %407, float %412, <4 x float> %389, i32 0, i32 0, i32 0)
  %414 = extractelement <1 x float> %190, i64 0
  %415 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %407, float %414, <4 x float> %391, i32 0, i32 0, i32 0)
  %416 = extractelement <1 x float> %134, i64 0
  %417 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %416, float %408, <4 x float> %393, i32 0, i32 0, i32 0)
  %418 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %416, float %410, <4 x float> %394, i32 0, i32 0, i32 0)
  %419 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %416, float %412, <4 x float> %395, i32 0, i32 0, i32 0)
  %420 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %416, float %414, <4 x float> %396, i32 0, i32 0, i32 0)
  %421 = extractelement <1 x float> %142, i64 0
  %422 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %421, float %408, <4 x float> %398, i32 0, i32 0, i32 0)
  %423 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %421, float %410, <4 x float> %399, i32 0, i32 0, i32 0)
  %424 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %421, float %412, <4 x float> %400, i32 0, i32 0, i32 0)
  %425 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %421, float %414, <4 x float> %401, i32 0, i32 0, i32 0)
  %426 = extractelement <1 x float> %150, i64 0
  %427 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %426, float %408, <4 x float> %403, i32 0, i32 0, i32 0)
  %428 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %426, float %410, <4 x float> %404, i32 0, i32 0, i32 0)
  %429 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %426, float %412, <4 x float> %405, i32 0, i32 0, i32 0)
  %430 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %426, float %414, <4 x float> %406, i32 0, i32 0, i32 0)
  %431 = shufflevector <4 x float> %409, <4 x float> poison, <1 x i32> zeroinitializer
  %432 = insertvalue [4 x <1 x float>] poison, <1 x float> %431, 0
  %433 = shufflevector <4 x float> %409, <4 x float> poison, <1 x i32> <i32 1>
  %434 = insertvalue [4 x <1 x float>] %432, <1 x float> %433, 1
  %435 = shufflevector <4 x float> %409, <4 x float> poison, <1 x i32> <i32 2>
  %436 = insertvalue [4 x <1 x float>] %434, <1 x float> %435, 2
  %437 = shufflevector <4 x float> %409, <4 x float> poison, <1 x i32> <i32 3>
  %438 = insertvalue [4 x <1 x float>] %436, <1 x float> %437, 3
  %439 = insertvalue [4 x [4 x [4 x <1 x float>]]] zeroinitializer, [4 x <1 x float>] %438, 0, 0
  %440 = shufflevector <4 x float> %411, <4 x float> poison, <1 x i32> zeroinitializer
  %441 = insertvalue [4 x <1 x float>] poison, <1 x float> %440, 0
  %442 = shufflevector <4 x float> %411, <4 x float> poison, <1 x i32> <i32 1>
  %443 = insertvalue [4 x <1 x float>] %441, <1 x float> %442, 1
  %444 = shufflevector <4 x float> %411, <4 x float> poison, <1 x i32> <i32 2>
  %445 = insertvalue [4 x <1 x float>] %443, <1 x float> %444, 2
  %446 = shufflevector <4 x float> %411, <4 x float> poison, <1 x i32> <i32 3>
  %447 = insertvalue [4 x <1 x float>] %445, <1 x float> %446, 3
  %448 = insertvalue [4 x [4 x [4 x <1 x float>]]] %439, [4 x <1 x float>] %447, 0, 1
  %449 = shufflevector <4 x float> %413, <4 x float> poison, <1 x i32> zeroinitializer
  %450 = insertvalue [4 x <1 x float>] poison, <1 x float> %449, 0
  %451 = shufflevector <4 x float> %413, <4 x float> poison, <1 x i32> <i32 1>
  %452 = insertvalue [4 x <1 x float>] %450, <1 x float> %451, 1
  %453 = shufflevector <4 x float> %413, <4 x float> poison, <1 x i32> <i32 2>
  %454 = insertvalue [4 x <1 x float>] %452, <1 x float> %453, 2
  %455 = shufflevector <4 x float> %413, <4 x float> poison, <1 x i32> <i32 3>
  %456 = insertvalue [4 x <1 x float>] %454, <1 x float> %455, 3
  %457 = insertvalue [4 x [4 x [4 x <1 x float>]]] %448, [4 x <1 x float>] %456, 0, 2
  %458 = shufflevector <4 x float> %415, <4 x float> poison, <1 x i32> zeroinitializer
  %459 = insertvalue [4 x <1 x float>] poison, <1 x float> %458, 0
  %460 = shufflevector <4 x float> %415, <4 x float> poison, <1 x i32> <i32 1>
  %461 = insertvalue [4 x <1 x float>] %459, <1 x float> %460, 1
  %462 = shufflevector <4 x float> %415, <4 x float> poison, <1 x i32> <i32 2>
  %463 = insertvalue [4 x <1 x float>] %461, <1 x float> %462, 2
  %464 = shufflevector <4 x float> %415, <4 x float> poison, <1 x i32> <i32 3>
  %465 = insertvalue [4 x <1 x float>] %463, <1 x float> %464, 3
  %466 = insertvalue [4 x [4 x [4 x <1 x float>]]] %457, [4 x <1 x float>] %465, 0, 3
  %467 = shufflevector <4 x float> %417, <4 x float> poison, <1 x i32> zeroinitializer
  %468 = insertvalue [4 x <1 x float>] poison, <1 x float> %467, 0
  %469 = shufflevector <4 x float> %417, <4 x float> poison, <1 x i32> <i32 1>
  %470 = insertvalue [4 x <1 x float>] %468, <1 x float> %469, 1
  %471 = shufflevector <4 x float> %417, <4 x float> poison, <1 x i32> <i32 2>
  %472 = insertvalue [4 x <1 x float>] %470, <1 x float> %471, 2
  %473 = shufflevector <4 x float> %417, <4 x float> poison, <1 x i32> <i32 3>
  %474 = insertvalue [4 x <1 x float>] %472, <1 x float> %473, 3
  %475 = insertvalue [4 x [4 x [4 x <1 x float>]]] %466, [4 x <1 x float>] %474, 1, 0
  %476 = shufflevector <4 x float> %418, <4 x float> poison, <1 x i32> zeroinitializer
  %477 = insertvalue [4 x <1 x float>] poison, <1 x float> %476, 0
  %478 = shufflevector <4 x float> %418, <4 x float> poison, <1 x i32> <i32 1>
  %479 = insertvalue [4 x <1 x float>] %477, <1 x float> %478, 1
  %480 = shufflevector <4 x float> %418, <4 x float> poison, <1 x i32> <i32 2>
  %481 = insertvalue [4 x <1 x float>] %479, <1 x float> %480, 2
  %482 = shufflevector <4 x float> %418, <4 x float> poison, <1 x i32> <i32 3>
  %483 = insertvalue [4 x <1 x float>] %481, <1 x float> %482, 3
  %484 = insertvalue [4 x [4 x [4 x <1 x float>]]] %475, [4 x <1 x float>] %483, 1, 1
  %485 = shufflevector <4 x float> %419, <4 x float> poison, <1 x i32> zeroinitializer
  %486 = insertvalue [4 x <1 x float>] poison, <1 x float> %485, 0
  %487 = shufflevector <4 x float> %419, <4 x float> poison, <1 x i32> <i32 1>
  %488 = insertvalue [4 x <1 x float>] %486, <1 x float> %487, 1
  %489 = shufflevector <4 x float> %419, <4 x float> poison, <1 x i32> <i32 2>
  %490 = insertvalue [4 x <1 x float>] %488, <1 x float> %489, 2
  %491 = shufflevector <4 x float> %419, <4 x float> poison, <1 x i32> <i32 3>
  %492 = insertvalue [4 x <1 x float>] %490, <1 x float> %491, 3
  %493 = insertvalue [4 x [4 x [4 x <1 x float>]]] %484, [4 x <1 x float>] %492, 1, 2
  %494 = shufflevector <4 x float> %420, <4 x float> poison, <1 x i32> zeroinitializer
  %495 = insertvalue [4 x <1 x float>] poison, <1 x float> %494, 0
  %496 = shufflevector <4 x float> %420, <4 x float> poison, <1 x i32> <i32 1>
  %497 = insertvalue [4 x <1 x float>] %495, <1 x float> %496, 1
  %498 = shufflevector <4 x float> %420, <4 x float> poison, <1 x i32> <i32 2>
  %499 = insertvalue [4 x <1 x float>] %497, <1 x float> %498, 2
  %500 = shufflevector <4 x float> %420, <4 x float> poison, <1 x i32> <i32 3>
  %501 = insertvalue [4 x <1 x float>] %499, <1 x float> %500, 3
  %502 = insertvalue [4 x [4 x [4 x <1 x float>]]] %493, [4 x <1 x float>] %501, 1, 3
  %503 = shufflevector <4 x float> %422, <4 x float> poison, <1 x i32> zeroinitializer
  %504 = insertvalue [4 x <1 x float>] poison, <1 x float> %503, 0
  %505 = shufflevector <4 x float> %422, <4 x float> poison, <1 x i32> <i32 1>
  %506 = insertvalue [4 x <1 x float>] %504, <1 x float> %505, 1
  %507 = shufflevector <4 x float> %422, <4 x float> poison, <1 x i32> <i32 2>
  %508 = insertvalue [4 x <1 x float>] %506, <1 x float> %507, 2
  %509 = shufflevector <4 x float> %422, <4 x float> poison, <1 x i32> <i32 3>
  %510 = insertvalue [4 x <1 x float>] %508, <1 x float> %509, 3
  %511 = insertvalue [4 x [4 x [4 x <1 x float>]]] %502, [4 x <1 x float>] %510, 2, 0
  %512 = shufflevector <4 x float> %423, <4 x float> poison, <1 x i32> zeroinitializer
  %513 = insertvalue [4 x <1 x float>] poison, <1 x float> %512, 0
  %514 = shufflevector <4 x float> %423, <4 x float> poison, <1 x i32> <i32 1>
  %515 = insertvalue [4 x <1 x float>] %513, <1 x float> %514, 1
  %516 = shufflevector <4 x float> %423, <4 x float> poison, <1 x i32> <i32 2>
  %517 = insertvalue [4 x <1 x float>] %515, <1 x float> %516, 2
  %518 = shufflevector <4 x float> %423, <4 x float> poison, <1 x i32> <i32 3>
  %519 = insertvalue [4 x <1 x float>] %517, <1 x float> %518, 3
  %520 = insertvalue [4 x [4 x [4 x <1 x float>]]] %511, [4 x <1 x float>] %519, 2, 1
  %521 = shufflevector <4 x float> %424, <4 x float> poison, <1 x i32> zeroinitializer
  %522 = insertvalue [4 x <1 x float>] poison, <1 x float> %521, 0
  %523 = shufflevector <4 x float> %424, <4 x float> poison, <1 x i32> <i32 1>
  %524 = insertvalue [4 x <1 x float>] %522, <1 x float> %523, 1
  %525 = shufflevector <4 x float> %424, <4 x float> poison, <1 x i32> <i32 2>
  %526 = insertvalue [4 x <1 x float>] %524, <1 x float> %525, 2
  %527 = shufflevector <4 x float> %424, <4 x float> poison, <1 x i32> <i32 3>
  %528 = insertvalue [4 x <1 x float>] %526, <1 x float> %527, 3
  %529 = insertvalue [4 x [4 x [4 x <1 x float>]]] %520, [4 x <1 x float>] %528, 2, 2
  %530 = shufflevector <4 x float> %425, <4 x float> poison, <1 x i32> zeroinitializer
  %531 = insertvalue [4 x <1 x float>] poison, <1 x float> %530, 0
  %532 = shufflevector <4 x float> %425, <4 x float> poison, <1 x i32> <i32 1>
  %533 = insertvalue [4 x <1 x float>] %531, <1 x float> %532, 1
  %534 = shufflevector <4 x float> %425, <4 x float> poison, <1 x i32> <i32 2>
  %535 = insertvalue [4 x <1 x float>] %533, <1 x float> %534, 2
  %536 = shufflevector <4 x float> %425, <4 x float> poison, <1 x i32> <i32 3>
  %537 = insertvalue [4 x <1 x float>] %535, <1 x float> %536, 3
  %538 = insertvalue [4 x [4 x [4 x <1 x float>]]] %529, [4 x <1 x float>] %537, 2, 3
  %539 = shufflevector <4 x float> %427, <4 x float> poison, <1 x i32> zeroinitializer
  %540 = insertvalue [4 x <1 x float>] poison, <1 x float> %539, 0
  %541 = shufflevector <4 x float> %427, <4 x float> poison, <1 x i32> <i32 1>
  %542 = insertvalue [4 x <1 x float>] %540, <1 x float> %541, 1
  %543 = shufflevector <4 x float> %427, <4 x float> poison, <1 x i32> <i32 2>
  %544 = insertvalue [4 x <1 x float>] %542, <1 x float> %543, 2
  %545 = shufflevector <4 x float> %427, <4 x float> poison, <1 x i32> <i32 3>
  %546 = insertvalue [4 x <1 x float>] %544, <1 x float> %545, 3
  %547 = insertvalue [4 x [4 x [4 x <1 x float>]]] %538, [4 x <1 x float>] %546, 3, 0
  %548 = shufflevector <4 x float> %428, <4 x float> poison, <1 x i32> zeroinitializer
  %549 = insertvalue [4 x <1 x float>] poison, <1 x float> %548, 0
  %550 = shufflevector <4 x float> %428, <4 x float> poison, <1 x i32> <i32 1>
  %551 = insertvalue [4 x <1 x float>] %549, <1 x float> %550, 1
  %552 = shufflevector <4 x float> %428, <4 x float> poison, <1 x i32> <i32 2>
  %553 = insertvalue [4 x <1 x float>] %551, <1 x float> %552, 2
  %554 = shufflevector <4 x float> %428, <4 x float> poison, <1 x i32> <i32 3>
  %555 = insertvalue [4 x <1 x float>] %553, <1 x float> %554, 3
  %556 = insertvalue [4 x [4 x [4 x <1 x float>]]] %547, [4 x <1 x float>] %555, 3, 1
  %557 = shufflevector <4 x float> %429, <4 x float> poison, <1 x i32> zeroinitializer
  %558 = insertvalue [4 x <1 x float>] poison, <1 x float> %557, 0
  %559 = shufflevector <4 x float> %429, <4 x float> poison, <1 x i32> <i32 1>
  %560 = insertvalue [4 x <1 x float>] %558, <1 x float> %559, 1
  %561 = shufflevector <4 x float> %429, <4 x float> poison, <1 x i32> <i32 2>
  %562 = insertvalue [4 x <1 x float>] %560, <1 x float> %561, 2
  %563 = shufflevector <4 x float> %429, <4 x float> poison, <1 x i32> <i32 3>
  %564 = insertvalue [4 x <1 x float>] %562, <1 x float> %563, 3
  %565 = insertvalue [4 x [4 x [4 x <1 x float>]]] %556, [4 x <1 x float>] %564, 3, 2
  %566 = shufflevector <4 x float> %430, <4 x float> poison, <1 x i32> zeroinitializer
  %567 = insertvalue [4 x <1 x float>] poison, <1 x float> %566, 0
  %568 = shufflevector <4 x float> %430, <4 x float> poison, <1 x i32> <i32 1>
  %569 = insertvalue [4 x <1 x float>] %567, <1 x float> %568, 1
  %570 = shufflevector <4 x float> %430, <4 x float> poison, <1 x i32> <i32 2>
  %571 = insertvalue [4 x <1 x float>] %569, <1 x float> %570, 2
  %572 = shufflevector <4 x float> %430, <4 x float> poison, <1 x i32> <i32 3>
  %573 = insertvalue [4 x <1 x float>] %571, <1 x float> %572, 3
  %574 = insertvalue [4 x [4 x [4 x <1 x float>]]] %565, [4 x <1 x float>] %573, 3, 3
  %575 = add nuw nsw i32 %91, 4
  %576 = icmp samesign ult i32 %91, 1012
  br i1 %576, label %85, label %577

577:                                              ; preds = %85
  %578 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %2, i16 0, i64 67108864, i32 159744)
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0)
  fence syncscope("workgroup") release, !mmra !3
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !3
  %579 = extractvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %86, 1
  %580 = extractvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %86, 2
  %581 = trunc i64 %580 to i32
  %582 = getelementptr float, ptr addrspace(3) %579, i32 %581
  %583 = or disjoint i32 %74, %9
  %584 = getelementptr float, ptr addrspace(3) %582, i32 %583
  %585 = load <1 x float>, ptr addrspace(3) %584, align 4
  %586 = or disjoint i32 %74, %67
  %587 = getelementptr float, ptr addrspace(3) %582, i32 %586
  %588 = load <1 x float>, ptr addrspace(3) %587, align 4
  %589 = or disjoint i32 %74, %68
  %590 = getelementptr float, ptr addrspace(3) %582, i32 %589
  %591 = load <1 x float>, ptr addrspace(3) %590, align 4
  %592 = or disjoint i32 %74, %69
  %593 = getelementptr float, ptr addrspace(3) %582, i32 %592
  %594 = load <1 x float>, ptr addrspace(3) %593, align 4
  %595 = or disjoint i32 %76, %9
  %596 = getelementptr float, ptr addrspace(3) %582, i32 %595
  %597 = load <1 x float>, ptr addrspace(3) %596, align 4
  %598 = or disjoint i32 %76, %67
  %599 = getelementptr float, ptr addrspace(3) %582, i32 %598
  %600 = load <1 x float>, ptr addrspace(3) %599, align 4
  %601 = or disjoint i32 %76, %68
  %602 = getelementptr float, ptr addrspace(3) %582, i32 %601
  %603 = load <1 x float>, ptr addrspace(3) %602, align 4
  %604 = or disjoint i32 %76, %69
  %605 = getelementptr float, ptr addrspace(3) %582, i32 %604
  %606 = load <1 x float>, ptr addrspace(3) %605, align 4
  %607 = or disjoint i32 %78, %9
  %608 = getelementptr float, ptr addrspace(3) %582, i32 %607
  %609 = load <1 x float>, ptr addrspace(3) %608, align 4
  %610 = or disjoint i32 %78, %67
  %611 = getelementptr float, ptr addrspace(3) %582, i32 %610
  %612 = load <1 x float>, ptr addrspace(3) %611, align 4
  %613 = or disjoint i32 %78, %68
  %614 = getelementptr float, ptr addrspace(3) %582, i32 %613
  %615 = load <1 x float>, ptr addrspace(3) %614, align 4
  %616 = or disjoint i32 %78, %69
  %617 = getelementptr float, ptr addrspace(3) %582, i32 %616
  %618 = load <1 x float>, ptr addrspace(3) %617, align 4
  %619 = or disjoint i32 %80, %9
  %620 = getelementptr float, ptr addrspace(3) %582, i32 %619
  %621 = load <1 x float>, ptr addrspace(3) %620, align 4
  %622 = or disjoint i32 %80, %67
  %623 = getelementptr float, ptr addrspace(3) %582, i32 %622
  %624 = load <1 x float>, ptr addrspace(3) %623, align 4
  %625 = or disjoint i32 %80, %68
  %626 = getelementptr float, ptr addrspace(3) %582, i32 %625
  %627 = load <1 x float>, ptr addrspace(3) %626, align 4
  %628 = or disjoint i32 %80, %69
  %629 = getelementptr float, ptr addrspace(3) %582, i32 %628
  %630 = load <1 x float>, ptr addrspace(3) %629, align 4
  %631 = extractvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %88, 1
  %632 = extractvalue { ptr addrspace(3), ptr addrspace(3), i64, [2 x i64], [2 x i64] } %88, 2
  %633 = trunc i64 %632 to i32
  %634 = getelementptr float, ptr addrspace(3) %631, i32 %633
  %635 = or disjoint i32 %81, %70
  %636 = getelementptr float, ptr addrspace(3) %634, i32 %635
  %637 = load <1 x float>, ptr addrspace(3) %636, align 4
  %638 = or disjoint i32 %81, %71
  %639 = getelementptr float, ptr addrspace(3) %634, i32 %638
  %640 = load <1 x float>, ptr addrspace(3) %639, align 4
  %641 = or disjoint i32 %81, %72
  %642 = getelementptr float, ptr addrspace(3) %634, i32 %641
  %643 = load <1 x float>, ptr addrspace(3) %642, align 4
  %644 = or disjoint i32 %81, %73
  %645 = getelementptr float, ptr addrspace(3) %634, i32 %644
  %646 = load <1 x float>, ptr addrspace(3) %645, align 4
  %647 = or disjoint i32 %82, %70
  %648 = getelementptr float, ptr addrspace(3) %634, i32 %647
  %649 = load <1 x float>, ptr addrspace(3) %648, align 4
  %650 = or disjoint i32 %82, %71
  %651 = getelementptr float, ptr addrspace(3) %634, i32 %650
  %652 = load <1 x float>, ptr addrspace(3) %651, align 4
  %653 = or disjoint i32 %82, %72
  %654 = getelementptr float, ptr addrspace(3) %634, i32 %653
  %655 = load <1 x float>, ptr addrspace(3) %654, align 4
  %656 = or disjoint i32 %82, %73
  %657 = getelementptr float, ptr addrspace(3) %634, i32 %656
  %658 = load <1 x float>, ptr addrspace(3) %657, align 4
  %659 = or disjoint i32 %83, %70
  %660 = getelementptr float, ptr addrspace(3) %634, i32 %659
  %661 = load <1 x float>, ptr addrspace(3) %660, align 4
  %662 = or disjoint i32 %83, %71
  %663 = getelementptr float, ptr addrspace(3) %634, i32 %662
  %664 = load <1 x float>, ptr addrspace(3) %663, align 4
  %665 = or disjoint i32 %83, %72
  %666 = getelementptr float, ptr addrspace(3) %634, i32 %665
  %667 = load <1 x float>, ptr addrspace(3) %666, align 4
  %668 = or disjoint i32 %83, %73
  %669 = getelementptr float, ptr addrspace(3) %634, i32 %668
  %670 = load <1 x float>, ptr addrspace(3) %669, align 4
  %671 = or disjoint i32 %84, %70
  %672 = getelementptr float, ptr addrspace(3) %634, i32 %671
  %673 = load <1 x float>, ptr addrspace(3) %672, align 4
  %674 = or disjoint i32 %84, %71
  %675 = getelementptr float, ptr addrspace(3) %634, i32 %674
  %676 = load <1 x float>, ptr addrspace(3) %675, align 4
  %677 = or disjoint i32 %84, %72
  %678 = getelementptr float, ptr addrspace(3) %634, i32 %677
  %679 = load <1 x float>, ptr addrspace(3) %678, align 4
  %680 = or disjoint i32 %84, %73
  %681 = getelementptr float, ptr addrspace(3) %634, i32 %680
  %682 = load <1 x float>, ptr addrspace(3) %681, align 4
  %683 = extractelement <1 x float> %585, i64 0
  %684 = extractelement <1 x float> %637, i64 0
  %685 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %683, float %684, <4 x float> %409, i32 0, i32 0, i32 0)
  %686 = extractelement <1 x float> %640, i64 0
  %687 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %683, float %686, <4 x float> %411, i32 0, i32 0, i32 0)
  %688 = extractelement <1 x float> %643, i64 0
  %689 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %683, float %688, <4 x float> %413, i32 0, i32 0, i32 0)
  %690 = extractelement <1 x float> %646, i64 0
  %691 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %683, float %690, <4 x float> %415, i32 0, i32 0, i32 0)
  %692 = extractelement <1 x float> %597, i64 0
  %693 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %692, float %684, <4 x float> %417, i32 0, i32 0, i32 0)
  %694 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %692, float %686, <4 x float> %418, i32 0, i32 0, i32 0)
  %695 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %692, float %688, <4 x float> %419, i32 0, i32 0, i32 0)
  %696 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %692, float %690, <4 x float> %420, i32 0, i32 0, i32 0)
  %697 = extractelement <1 x float> %609, i64 0
  %698 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %697, float %684, <4 x float> %422, i32 0, i32 0, i32 0)
  %699 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %697, float %686, <4 x float> %423, i32 0, i32 0, i32 0)
  %700 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %697, float %688, <4 x float> %424, i32 0, i32 0, i32 0)
  %701 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %697, float %690, <4 x float> %425, i32 0, i32 0, i32 0)
  %702 = extractelement <1 x float> %621, i64 0
  %703 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %702, float %684, <4 x float> %427, i32 0, i32 0, i32 0)
  %704 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %702, float %686, <4 x float> %428, i32 0, i32 0, i32 0)
  %705 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %702, float %688, <4 x float> %429, i32 0, i32 0, i32 0)
  %706 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %702, float %690, <4 x float> %430, i32 0, i32 0, i32 0)
  %707 = extractelement <1 x float> %588, i64 0
  %708 = extractelement <1 x float> %649, i64 0
  %709 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %707, float %708, <4 x float> %685, i32 0, i32 0, i32 0)
  %710 = extractelement <1 x float> %652, i64 0
  %711 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %707, float %710, <4 x float> %687, i32 0, i32 0, i32 0)
  %712 = extractelement <1 x float> %655, i64 0
  %713 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %707, float %712, <4 x float> %689, i32 0, i32 0, i32 0)
  %714 = extractelement <1 x float> %658, i64 0
  %715 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %707, float %714, <4 x float> %691, i32 0, i32 0, i32 0)
  %716 = extractelement <1 x float> %600, i64 0
  %717 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %716, float %708, <4 x float> %693, i32 0, i32 0, i32 0)
  %718 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %716, float %710, <4 x float> %694, i32 0, i32 0, i32 0)
  %719 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %716, float %712, <4 x float> %695, i32 0, i32 0, i32 0)
  %720 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %716, float %714, <4 x float> %696, i32 0, i32 0, i32 0)
  %721 = extractelement <1 x float> %612, i64 0
  %722 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %721, float %708, <4 x float> %698, i32 0, i32 0, i32 0)
  %723 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %721, float %710, <4 x float> %699, i32 0, i32 0, i32 0)
  %724 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %721, float %712, <4 x float> %700, i32 0, i32 0, i32 0)
  %725 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %721, float %714, <4 x float> %701, i32 0, i32 0, i32 0)
  %726 = extractelement <1 x float> %624, i64 0
  %727 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %726, float %708, <4 x float> %703, i32 0, i32 0, i32 0)
  %728 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %726, float %710, <4 x float> %704, i32 0, i32 0, i32 0)
  %729 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %726, float %712, <4 x float> %705, i32 0, i32 0, i32 0)
  %730 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %726, float %714, <4 x float> %706, i32 0, i32 0, i32 0)
  %731 = extractelement <1 x float> %591, i64 0
  %732 = extractelement <1 x float> %661, i64 0
  %733 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %731, float %732, <4 x float> %709, i32 0, i32 0, i32 0)
  %734 = extractelement <1 x float> %664, i64 0
  %735 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %731, float %734, <4 x float> %711, i32 0, i32 0, i32 0)
  %736 = extractelement <1 x float> %667, i64 0
  %737 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %731, float %736, <4 x float> %713, i32 0, i32 0, i32 0)
  %738 = extractelement <1 x float> %670, i64 0
  %739 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %731, float %738, <4 x float> %715, i32 0, i32 0, i32 0)
  %740 = extractelement <1 x float> %603, i64 0
  %741 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %740, float %732, <4 x float> %717, i32 0, i32 0, i32 0)
  %742 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %740, float %734, <4 x float> %718, i32 0, i32 0, i32 0)
  %743 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %740, float %736, <4 x float> %719, i32 0, i32 0, i32 0)
  %744 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %740, float %738, <4 x float> %720, i32 0, i32 0, i32 0)
  %745 = extractelement <1 x float> %615, i64 0
  %746 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %745, float %732, <4 x float> %722, i32 0, i32 0, i32 0)
  %747 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %745, float %734, <4 x float> %723, i32 0, i32 0, i32 0)
  %748 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %745, float %736, <4 x float> %724, i32 0, i32 0, i32 0)
  %749 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %745, float %738, <4 x float> %725, i32 0, i32 0, i32 0)
  %750 = extractelement <1 x float> %627, i64 0
  %751 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %750, float %732, <4 x float> %727, i32 0, i32 0, i32 0)
  %752 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %750, float %734, <4 x float> %728, i32 0, i32 0, i32 0)
  %753 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %750, float %736, <4 x float> %729, i32 0, i32 0, i32 0)
  %754 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %750, float %738, <4 x float> %730, i32 0, i32 0, i32 0)
  %755 = extractelement <1 x float> %594, i64 0
  %756 = extractelement <1 x float> %673, i64 0
  %757 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %755, float %756, <4 x float> %733, i32 0, i32 0, i32 0)
  %758 = extractelement <1 x float> %676, i64 0
  %759 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %755, float %758, <4 x float> %735, i32 0, i32 0, i32 0)
  %760 = extractelement <1 x float> %679, i64 0
  %761 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %755, float %760, <4 x float> %737, i32 0, i32 0, i32 0)
  %762 = extractelement <1 x float> %682, i64 0
  %763 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %755, float %762, <4 x float> %739, i32 0, i32 0, i32 0)
  %764 = extractelement <1 x float> %606, i64 0
  %765 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %764, float %756, <4 x float> %741, i32 0, i32 0, i32 0)
  %766 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %764, float %758, <4 x float> %742, i32 0, i32 0, i32 0)
  %767 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %764, float %760, <4 x float> %743, i32 0, i32 0, i32 0)
  %768 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %764, float %762, <4 x float> %744, i32 0, i32 0, i32 0)
  %769 = extractelement <1 x float> %618, i64 0
  %770 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %769, float %756, <4 x float> %746, i32 0, i32 0, i32 0)
  %771 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %769, float %758, <4 x float> %747, i32 0, i32 0, i32 0)
  %772 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %769, float %760, <4 x float> %748, i32 0, i32 0, i32 0)
  %773 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %769, float %762, <4 x float> %749, i32 0, i32 0, i32 0)
  %774 = extractelement <1 x float> %630, i64 0
  %775 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %774, float %756, <4 x float> %751, i32 0, i32 0, i32 0)
  %776 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %774, float %758, <4 x float> %752, i32 0, i32 0, i32 0)
  %777 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %774, float %760, <4 x float> %753, i32 0, i32 0, i32 0)
  %778 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %774, float %762, <4 x float> %754, i32 0, i32 0, i32 0)
  %779 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %583
  %780 = load <1 x float>, ptr addrspace(3) %779, align 4
  %781 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %586
  %782 = load <1 x float>, ptr addrspace(3) %781, align 4
  %783 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %589
  %784 = load <1 x float>, ptr addrspace(3) %783, align 4
  %785 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %592
  %786 = load <1 x float>, ptr addrspace(3) %785, align 4
  %787 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %595
  %788 = load <1 x float>, ptr addrspace(3) %787, align 4
  %789 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %598
  %790 = load <1 x float>, ptr addrspace(3) %789, align 4
  %791 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %601
  %792 = load <1 x float>, ptr addrspace(3) %791, align 4
  %793 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %604
  %794 = load <1 x float>, ptr addrspace(3) %793, align 4
  %795 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %607
  %796 = load <1 x float>, ptr addrspace(3) %795, align 4
  %797 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %610
  %798 = load <1 x float>, ptr addrspace(3) %797, align 4
  %799 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %613
  %800 = load <1 x float>, ptr addrspace(3) %799, align 4
  %801 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %616
  %802 = load <1 x float>, ptr addrspace(3) %801, align 4
  %803 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %619
  %804 = load <1 x float>, ptr addrspace(3) %803, align 4
  %805 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %622
  %806 = load <1 x float>, ptr addrspace(3) %805, align 4
  %807 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %625
  %808 = load <1 x float>, ptr addrspace(3) %807, align 4
  %809 = getelementptr float, ptr addrspace(3) @__shared_memory__, i32 %628
  %810 = load <1 x float>, ptr addrspace(3) %809, align 4
  %811 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %635
  %812 = load <1 x float>, ptr addrspace(3) %811, align 4
  %813 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %638
  %814 = load <1 x float>, ptr addrspace(3) %813, align 4
  %815 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %641
  %816 = load <1 x float>, ptr addrspace(3) %815, align 4
  %817 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %644
  %818 = load <1 x float>, ptr addrspace(3) %817, align 4
  %819 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %647
  %820 = load <1 x float>, ptr addrspace(3) %819, align 4
  %821 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %650
  %822 = load <1 x float>, ptr addrspace(3) %821, align 4
  %823 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %653
  %824 = load <1 x float>, ptr addrspace(3) %823, align 4
  %825 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %656
  %826 = load <1 x float>, ptr addrspace(3) %825, align 4
  %827 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %659
  %828 = load <1 x float>, ptr addrspace(3) %827, align 4
  %829 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %662
  %830 = load <1 x float>, ptr addrspace(3) %829, align 4
  %831 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %665
  %832 = load <1 x float>, ptr addrspace(3) %831, align 4
  %833 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %668
  %834 = load <1 x float>, ptr addrspace(3) %833, align 4
  %835 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %671
  %836 = load <1 x float>, ptr addrspace(3) %835, align 4
  %837 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %674
  %838 = load <1 x float>, ptr addrspace(3) %837, align 4
  %839 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %677
  %840 = load <1 x float>, ptr addrspace(3) %839, align 4
  %841 = getelementptr float, ptr addrspace(3) @__shared_memory___0, i32 %680
  %842 = load <1 x float>, ptr addrspace(3) %841, align 4
  %843 = extractelement <1 x float> %780, i64 0
  %844 = extractelement <1 x float> %812, i64 0
  %845 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %843, float %844, <4 x float> %757, i32 0, i32 0, i32 0)
  %846 = extractelement <1 x float> %814, i64 0
  %847 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %843, float %846, <4 x float> %759, i32 0, i32 0, i32 0)
  %848 = extractelement <1 x float> %816, i64 0
  %849 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %843, float %848, <4 x float> %761, i32 0, i32 0, i32 0)
  %850 = extractelement <1 x float> %818, i64 0
  %851 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %843, float %850, <4 x float> %763, i32 0, i32 0, i32 0)
  %852 = extractelement <1 x float> %788, i64 0
  %853 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %852, float %844, <4 x float> %765, i32 0, i32 0, i32 0)
  %854 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %852, float %846, <4 x float> %766, i32 0, i32 0, i32 0)
  %855 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %852, float %848, <4 x float> %767, i32 0, i32 0, i32 0)
  %856 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %852, float %850, <4 x float> %768, i32 0, i32 0, i32 0)
  %857 = extractelement <1 x float> %796, i64 0
  %858 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %857, float %844, <4 x float> %770, i32 0, i32 0, i32 0)
  %859 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %857, float %846, <4 x float> %771, i32 0, i32 0, i32 0)
  %860 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %857, float %848, <4 x float> %772, i32 0, i32 0, i32 0)
  %861 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %857, float %850, <4 x float> %773, i32 0, i32 0, i32 0)
  %862 = extractelement <1 x float> %804, i64 0
  %863 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %862, float %844, <4 x float> %775, i32 0, i32 0, i32 0)
  %864 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %862, float %846, <4 x float> %776, i32 0, i32 0, i32 0)
  %865 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %862, float %848, <4 x float> %777, i32 0, i32 0, i32 0)
  %866 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %862, float %850, <4 x float> %778, i32 0, i32 0, i32 0)
  %867 = extractelement <1 x float> %782, i64 0
  %868 = extractelement <1 x float> %820, i64 0
  %869 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %867, float %868, <4 x float> %845, i32 0, i32 0, i32 0)
  %870 = extractelement <1 x float> %822, i64 0
  %871 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %867, float %870, <4 x float> %847, i32 0, i32 0, i32 0)
  %872 = extractelement <1 x float> %824, i64 0
  %873 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %867, float %872, <4 x float> %849, i32 0, i32 0, i32 0)
  %874 = extractelement <1 x float> %826, i64 0
  %875 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %867, float %874, <4 x float> %851, i32 0, i32 0, i32 0)
  %876 = extractelement <1 x float> %790, i64 0
  %877 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %876, float %868, <4 x float> %853, i32 0, i32 0, i32 0)
  %878 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %876, float %870, <4 x float> %854, i32 0, i32 0, i32 0)
  %879 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %876, float %872, <4 x float> %855, i32 0, i32 0, i32 0)
  %880 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %876, float %874, <4 x float> %856, i32 0, i32 0, i32 0)
  %881 = extractelement <1 x float> %798, i64 0
  %882 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %881, float %868, <4 x float> %858, i32 0, i32 0, i32 0)
  %883 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %881, float %870, <4 x float> %859, i32 0, i32 0, i32 0)
  %884 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %881, float %872, <4 x float> %860, i32 0, i32 0, i32 0)
  %885 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %881, float %874, <4 x float> %861, i32 0, i32 0, i32 0)
  %886 = extractelement <1 x float> %806, i64 0
  %887 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %886, float %868, <4 x float> %863, i32 0, i32 0, i32 0)
  %888 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %886, float %870, <4 x float> %864, i32 0, i32 0, i32 0)
  %889 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %886, float %872, <4 x float> %865, i32 0, i32 0, i32 0)
  %890 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %886, float %874, <4 x float> %866, i32 0, i32 0, i32 0)
  %891 = extractelement <1 x float> %784, i64 0
  %892 = extractelement <1 x float> %828, i64 0
  %893 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %891, float %892, <4 x float> %869, i32 0, i32 0, i32 0)
  %894 = extractelement <1 x float> %830, i64 0
  %895 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %891, float %894, <4 x float> %871, i32 0, i32 0, i32 0)
  %896 = extractelement <1 x float> %832, i64 0
  %897 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %891, float %896, <4 x float> %873, i32 0, i32 0, i32 0)
  %898 = extractelement <1 x float> %834, i64 0
  %899 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %891, float %898, <4 x float> %875, i32 0, i32 0, i32 0)
  %900 = extractelement <1 x float> %792, i64 0
  %901 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %900, float %892, <4 x float> %877, i32 0, i32 0, i32 0)
  %902 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %900, float %894, <4 x float> %878, i32 0, i32 0, i32 0)
  %903 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %900, float %896, <4 x float> %879, i32 0, i32 0, i32 0)
  %904 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %900, float %898, <4 x float> %880, i32 0, i32 0, i32 0)
  %905 = extractelement <1 x float> %800, i64 0
  %906 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %905, float %892, <4 x float> %882, i32 0, i32 0, i32 0)
  %907 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %905, float %894, <4 x float> %883, i32 0, i32 0, i32 0)
  %908 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %905, float %896, <4 x float> %884, i32 0, i32 0, i32 0)
  %909 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %905, float %898, <4 x float> %885, i32 0, i32 0, i32 0)
  %910 = extractelement <1 x float> %808, i64 0
  %911 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %910, float %892, <4 x float> %887, i32 0, i32 0, i32 0)
  %912 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %910, float %894, <4 x float> %888, i32 0, i32 0, i32 0)
  %913 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %910, float %896, <4 x float> %889, i32 0, i32 0, i32 0)
  %914 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %910, float %898, <4 x float> %890, i32 0, i32 0, i32 0)
  %915 = extractelement <1 x float> %786, i64 0
  %916 = extractelement <1 x float> %836, i64 0
  %917 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %915, float %916, <4 x float> %893, i32 0, i32 0, i32 0)
  %918 = extractelement <1 x float> %838, i64 0
  %919 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %915, float %918, <4 x float> %895, i32 0, i32 0, i32 0)
  %920 = extractelement <1 x float> %840, i64 0
  %921 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %915, float %920, <4 x float> %897, i32 0, i32 0, i32 0)
  %922 = extractelement <1 x float> %842, i64 0
  %923 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %915, float %922, <4 x float> %899, i32 0, i32 0, i32 0)
  %924 = extractelement <1 x float> %794, i64 0
  %925 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %924, float %916, <4 x float> %901, i32 0, i32 0, i32 0)
  %926 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %924, float %918, <4 x float> %902, i32 0, i32 0, i32 0)
  %927 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %924, float %920, <4 x float> %903, i32 0, i32 0, i32 0)
  %928 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %924, float %922, <4 x float> %904, i32 0, i32 0, i32 0)
  %929 = extractelement <1 x float> %802, i64 0
  %930 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %929, float %916, <4 x float> %906, i32 0, i32 0, i32 0)
  %931 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %929, float %918, <4 x float> %907, i32 0, i32 0, i32 0)
  %932 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %929, float %920, <4 x float> %908, i32 0, i32 0, i32 0)
  %933 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %929, float %922, <4 x float> %909, i32 0, i32 0, i32 0)
  %934 = extractelement <1 x float> %810, i64 0
  %935 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %934, float %916, <4 x float> %911, i32 0, i32 0, i32 0)
  %936 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %934, float %918, <4 x float> %912, i32 0, i32 0, i32 0)
  %937 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %934, float %920, <4 x float> %913, i32 0, i32 0, i32 0)
  %938 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %934, float %922, <4 x float> %914, i32 0, i32 0, i32 0)
  %939 = shufflevector <4 x float> %917, <4 x float> poison, <1 x i32> zeroinitializer
  %940 = shufflevector <4 x float> %917, <4 x float> poison, <1 x i32> <i32 1>
  %941 = shufflevector <4 x float> %917, <4 x float> poison, <1 x i32> <i32 2>
  %942 = shufflevector <4 x float> %917, <4 x float> poison, <1 x i32> <i32 3>
  %943 = shufflevector <4 x float> %919, <4 x float> poison, <1 x i32> zeroinitializer
  %944 = shufflevector <4 x float> %919, <4 x float> poison, <1 x i32> <i32 1>
  %945 = shufflevector <4 x float> %919, <4 x float> poison, <1 x i32> <i32 2>
  %946 = shufflevector <4 x float> %919, <4 x float> poison, <1 x i32> <i32 3>
  %947 = shufflevector <4 x float> %921, <4 x float> poison, <1 x i32> zeroinitializer
  %948 = shufflevector <4 x float> %921, <4 x float> poison, <1 x i32> <i32 1>
  %949 = shufflevector <4 x float> %921, <4 x float> poison, <1 x i32> <i32 2>
  %950 = shufflevector <4 x float> %921, <4 x float> poison, <1 x i32> <i32 3>
  %951 = shufflevector <4 x float> %923, <4 x float> poison, <1 x i32> zeroinitializer
  %952 = shufflevector <4 x float> %923, <4 x float> poison, <1 x i32> <i32 1>
  %953 = shufflevector <4 x float> %923, <4 x float> poison, <1 x i32> <i32 2>
  %954 = shufflevector <4 x float> %923, <4 x float> poison, <1 x i32> <i32 3>
  %955 = shufflevector <4 x float> %925, <4 x float> poison, <1 x i32> zeroinitializer
  %956 = shufflevector <4 x float> %925, <4 x float> poison, <1 x i32> <i32 1>
  %957 = shufflevector <4 x float> %925, <4 x float> poison, <1 x i32> <i32 2>
  %958 = shufflevector <4 x float> %925, <4 x float> poison, <1 x i32> <i32 3>
  %959 = shufflevector <4 x float> %926, <4 x float> poison, <1 x i32> zeroinitializer
  %960 = shufflevector <4 x float> %926, <4 x float> poison, <1 x i32> <i32 1>
  %961 = shufflevector <4 x float> %926, <4 x float> poison, <1 x i32> <i32 2>
  %962 = shufflevector <4 x float> %926, <4 x float> poison, <1 x i32> <i32 3>
  %963 = shufflevector <4 x float> %927, <4 x float> poison, <1 x i32> zeroinitializer
  %964 = shufflevector <4 x float> %927, <4 x float> poison, <1 x i32> <i32 1>
  %965 = shufflevector <4 x float> %927, <4 x float> poison, <1 x i32> <i32 2>
  %966 = shufflevector <4 x float> %927, <4 x float> poison, <1 x i32> <i32 3>
  %967 = shufflevector <4 x float> %928, <4 x float> poison, <1 x i32> zeroinitializer
  %968 = shufflevector <4 x float> %928, <4 x float> poison, <1 x i32> <i32 1>
  %969 = shufflevector <4 x float> %928, <4 x float> poison, <1 x i32> <i32 2>
  %970 = shufflevector <4 x float> %928, <4 x float> poison, <1 x i32> <i32 3>
  %971 = shufflevector <4 x float> %930, <4 x float> poison, <1 x i32> zeroinitializer
  %972 = shufflevector <4 x float> %930, <4 x float> poison, <1 x i32> <i32 1>
  %973 = shufflevector <4 x float> %930, <4 x float> poison, <1 x i32> <i32 2>
  %974 = shufflevector <4 x float> %930, <4 x float> poison, <1 x i32> <i32 3>
  %975 = shufflevector <4 x float> %931, <4 x float> poison, <1 x i32> zeroinitializer
  %976 = shufflevector <4 x float> %931, <4 x float> poison, <1 x i32> <i32 1>
  %977 = shufflevector <4 x float> %931, <4 x float> poison, <1 x i32> <i32 2>
  %978 = shufflevector <4 x float> %931, <4 x float> poison, <1 x i32> <i32 3>
  %979 = shufflevector <4 x float> %932, <4 x float> poison, <1 x i32> zeroinitializer
  %980 = shufflevector <4 x float> %932, <4 x float> poison, <1 x i32> <i32 1>
  %981 = shufflevector <4 x float> %932, <4 x float> poison, <1 x i32> <i32 2>
  %982 = shufflevector <4 x float> %932, <4 x float> poison, <1 x i32> <i32 3>
  %983 = shufflevector <4 x float> %933, <4 x float> poison, <1 x i32> zeroinitializer
  %984 = shufflevector <4 x float> %933, <4 x float> poison, <1 x i32> <i32 1>
  %985 = shufflevector <4 x float> %933, <4 x float> poison, <1 x i32> <i32 2>
  %986 = shufflevector <4 x float> %933, <4 x float> poison, <1 x i32> <i32 3>
  %987 = shufflevector <4 x float> %935, <4 x float> poison, <1 x i32> zeroinitializer
  %988 = shufflevector <4 x float> %935, <4 x float> poison, <1 x i32> <i32 1>
  %989 = shufflevector <4 x float> %935, <4 x float> poison, <1 x i32> <i32 2>
  %990 = shufflevector <4 x float> %935, <4 x float> poison, <1 x i32> <i32 3>
  %991 = shufflevector <4 x float> %936, <4 x float> poison, <1 x i32> zeroinitializer
  %992 = shufflevector <4 x float> %936, <4 x float> poison, <1 x i32> <i32 1>
  %993 = shufflevector <4 x float> %936, <4 x float> poison, <1 x i32> <i32 2>
  %994 = shufflevector <4 x float> %936, <4 x float> poison, <1 x i32> <i32 3>
  %995 = shufflevector <4 x float> %937, <4 x float> poison, <1 x i32> zeroinitializer
  %996 = shufflevector <4 x float> %937, <4 x float> poison, <1 x i32> <i32 1>
  %997 = shufflevector <4 x float> %937, <4 x float> poison, <1 x i32> <i32 2>
  %998 = shufflevector <4 x float> %937, <4 x float> poison, <1 x i32> <i32 3>
  %999 = shufflevector <4 x float> %938, <4 x float> poison, <1 x i32> zeroinitializer
  %1000 = shufflevector <4 x float> %938, <4 x float> poison, <1 x i32> <i32 1>
  %1001 = shufflevector <4 x float> %938, <4 x float> poison, <1 x i32> <i32 2>
  %1002 = shufflevector <4 x float> %938, <4 x float> poison, <1 x i32> <i32 3>
  %1003 = and i32 %22, 992
  %1004 = or disjoint i32 %9, %1003
  %1005 = or disjoint i32 %70, %24
  %.idx42 = shl nuw nsw i32 %1004, 16
  %1006 = getelementptr i8, ptr addrspace(7) %578, i32 %.idx42
  %.idx43 = shl nuw nsw i32 %65, 14
  %1007 = getelementptr i8, ptr addrspace(7) %1006, i32 %.idx43
  %1008 = getelementptr float, ptr addrspace(7) %1007, i32 %1005
  store <1 x float> %939, ptr addrspace(7) %1008, align 4
  %1009 = or disjoint i32 %1005, 16
  %1010 = getelementptr float, ptr addrspace(7) %1007, i32 %1009
  store <1 x float> %943, ptr addrspace(7) %1010, align 4
  %1011 = or disjoint i32 %1005, 32
  %1012 = getelementptr float, ptr addrspace(7) %1007, i32 %1011
  store <1 x float> %947, ptr addrspace(7) %1012, align 4
  %1013 = or disjoint i32 %1005, 48
  %1014 = getelementptr float, ptr addrspace(7) %1007, i32 %1013
  store <1 x float> %951, ptr addrspace(7) %1014, align 4
  %1015 = getelementptr i8, ptr addrspace(7) %1007, i32 16384
  %1016 = getelementptr float, ptr addrspace(7) %1015, i32 %1005
  store <1 x float> %940, ptr addrspace(7) %1016, align 4
  %1017 = getelementptr float, ptr addrspace(7) %1015, i32 %1009
  store <1 x float> %944, ptr addrspace(7) %1017, align 4
  %1018 = getelementptr float, ptr addrspace(7) %1015, i32 %1011
  store <1 x float> %948, ptr addrspace(7) %1018, align 4
  %1019 = getelementptr float, ptr addrspace(7) %1015, i32 %1013
  store <1 x float> %952, ptr addrspace(7) %1019, align 4
  %1020 = getelementptr i8, ptr addrspace(7) %1007, i32 32768
  %1021 = getelementptr float, ptr addrspace(7) %1020, i32 %1005
  store <1 x float> %941, ptr addrspace(7) %1021, align 4
  %1022 = getelementptr float, ptr addrspace(7) %1020, i32 %1009
  store <1 x float> %945, ptr addrspace(7) %1022, align 4
  %1023 = getelementptr float, ptr addrspace(7) %1020, i32 %1011
  store <1 x float> %949, ptr addrspace(7) %1023, align 4
  %1024 = getelementptr float, ptr addrspace(7) %1020, i32 %1013
  store <1 x float> %953, ptr addrspace(7) %1024, align 4
  %1025 = getelementptr i8, ptr addrspace(7) %1007, i32 49152
  %1026 = getelementptr float, ptr addrspace(7) %1025, i32 %1005
  store <1 x float> %942, ptr addrspace(7) %1026, align 4
  %1027 = getelementptr float, ptr addrspace(7) %1025, i32 %1009
  store <1 x float> %946, ptr addrspace(7) %1027, align 4
  %1028 = getelementptr float, ptr addrspace(7) %1025, i32 %1011
  store <1 x float> %950, ptr addrspace(7) %1028, align 4
  %1029 = getelementptr float, ptr addrspace(7) %1025, i32 %1013
  store <1 x float> %954, ptr addrspace(7) %1029, align 4
  %1030 = getelementptr i8, ptr addrspace(7) %1007, i32 262144
  %1031 = getelementptr float, ptr addrspace(7) %1030, i32 %1005
  store <1 x float> %955, ptr addrspace(7) %1031, align 4
  %1032 = getelementptr float, ptr addrspace(7) %1030, i32 %1009
  store <1 x float> %959, ptr addrspace(7) %1032, align 4
  %1033 = getelementptr float, ptr addrspace(7) %1030, i32 %1011
  store <1 x float> %963, ptr addrspace(7) %1033, align 4
  %1034 = getelementptr float, ptr addrspace(7) %1030, i32 %1013
  store <1 x float> %967, ptr addrspace(7) %1034, align 4
  %1035 = getelementptr i8, ptr addrspace(7) %1007, i32 278528
  %1036 = getelementptr float, ptr addrspace(7) %1035, i32 %1005
  store <1 x float> %956, ptr addrspace(7) %1036, align 4
  %1037 = getelementptr float, ptr addrspace(7) %1035, i32 %1009
  store <1 x float> %960, ptr addrspace(7) %1037, align 4
  %1038 = getelementptr float, ptr addrspace(7) %1035, i32 %1011
  store <1 x float> %964, ptr addrspace(7) %1038, align 4
  %1039 = getelementptr float, ptr addrspace(7) %1035, i32 %1013
  store <1 x float> %968, ptr addrspace(7) %1039, align 4
  %1040 = getelementptr i8, ptr addrspace(7) %1007, i32 294912
  %1041 = getelementptr float, ptr addrspace(7) %1040, i32 %1005
  store <1 x float> %957, ptr addrspace(7) %1041, align 4
  %1042 = getelementptr float, ptr addrspace(7) %1040, i32 %1009
  store <1 x float> %961, ptr addrspace(7) %1042, align 4
  %1043 = getelementptr float, ptr addrspace(7) %1040, i32 %1011
  store <1 x float> %965, ptr addrspace(7) %1043, align 4
  %1044 = getelementptr float, ptr addrspace(7) %1040, i32 %1013
  store <1 x float> %969, ptr addrspace(7) %1044, align 4
  %1045 = getelementptr i8, ptr addrspace(7) %1007, i32 311296
  %1046 = getelementptr float, ptr addrspace(7) %1045, i32 %1005
  store <1 x float> %958, ptr addrspace(7) %1046, align 4
  %1047 = getelementptr float, ptr addrspace(7) %1045, i32 %1009
  store <1 x float> %962, ptr addrspace(7) %1047, align 4
  %1048 = getelementptr float, ptr addrspace(7) %1045, i32 %1011
  store <1 x float> %966, ptr addrspace(7) %1048, align 4
  %1049 = getelementptr float, ptr addrspace(7) %1045, i32 %1013
  store <1 x float> %970, ptr addrspace(7) %1049, align 4
  %1050 = getelementptr i8, ptr addrspace(7) %1007, i32 524288
  %1051 = getelementptr float, ptr addrspace(7) %1050, i32 %1005
  store <1 x float> %971, ptr addrspace(7) %1051, align 4
  %1052 = getelementptr float, ptr addrspace(7) %1050, i32 %1009
  store <1 x float> %975, ptr addrspace(7) %1052, align 4
  %1053 = getelementptr float, ptr addrspace(7) %1050, i32 %1011
  store <1 x float> %979, ptr addrspace(7) %1053, align 4
  %1054 = getelementptr float, ptr addrspace(7) %1050, i32 %1013
  store <1 x float> %983, ptr addrspace(7) %1054, align 4
  %1055 = getelementptr i8, ptr addrspace(7) %1007, i32 540672
  %1056 = getelementptr float, ptr addrspace(7) %1055, i32 %1005
  store <1 x float> %972, ptr addrspace(7) %1056, align 4
  %1057 = getelementptr float, ptr addrspace(7) %1055, i32 %1009
  store <1 x float> %976, ptr addrspace(7) %1057, align 4
  %1058 = getelementptr float, ptr addrspace(7) %1055, i32 %1011
  store <1 x float> %980, ptr addrspace(7) %1058, align 4
  %1059 = getelementptr float, ptr addrspace(7) %1055, i32 %1013
  store <1 x float> %984, ptr addrspace(7) %1059, align 4
  %1060 = getelementptr i8, ptr addrspace(7) %1007, i32 557056
  %1061 = getelementptr float, ptr addrspace(7) %1060, i32 %1005
  store <1 x float> %973, ptr addrspace(7) %1061, align 4
  %1062 = getelementptr float, ptr addrspace(7) %1060, i32 %1009
  store <1 x float> %977, ptr addrspace(7) %1062, align 4
  %1063 = getelementptr float, ptr addrspace(7) %1060, i32 %1011
  store <1 x float> %981, ptr addrspace(7) %1063, align 4
  %1064 = getelementptr float, ptr addrspace(7) %1060, i32 %1013
  store <1 x float> %985, ptr addrspace(7) %1064, align 4
  %1065 = getelementptr i8, ptr addrspace(7) %1007, i32 573440
  %1066 = getelementptr float, ptr addrspace(7) %1065, i32 %1005
  store <1 x float> %974, ptr addrspace(7) %1066, align 4
  %1067 = getelementptr float, ptr addrspace(7) %1065, i32 %1009
  store <1 x float> %978, ptr addrspace(7) %1067, align 4
  %1068 = getelementptr float, ptr addrspace(7) %1065, i32 %1011
  store <1 x float> %982, ptr addrspace(7) %1068, align 4
  %1069 = getelementptr float, ptr addrspace(7) %1065, i32 %1013
  store <1 x float> %986, ptr addrspace(7) %1069, align 4
  %1070 = getelementptr i8, ptr addrspace(7) %1007, i32 786432
  %1071 = getelementptr float, ptr addrspace(7) %1070, i32 %1005
  store <1 x float> %987, ptr addrspace(7) %1071, align 4
  %1072 = getelementptr float, ptr addrspace(7) %1070, i32 %1009
  store <1 x float> %991, ptr addrspace(7) %1072, align 4
  %1073 = getelementptr float, ptr addrspace(7) %1070, i32 %1011
  store <1 x float> %995, ptr addrspace(7) %1073, align 4
  %1074 = getelementptr float, ptr addrspace(7) %1070, i32 %1013
  store <1 x float> %999, ptr addrspace(7) %1074, align 4
  %1075 = getelementptr i8, ptr addrspace(7) %1007, i32 802816
  %1076 = getelementptr float, ptr addrspace(7) %1075, i32 %1005
  store <1 x float> %988, ptr addrspace(7) %1076, align 4
  %1077 = getelementptr float, ptr addrspace(7) %1075, i32 %1009
  store <1 x float> %992, ptr addrspace(7) %1077, align 4
  %1078 = getelementptr float, ptr addrspace(7) %1075, i32 %1011
  store <1 x float> %996, ptr addrspace(7) %1078, align 4
  %1079 = getelementptr float, ptr addrspace(7) %1075, i32 %1013
  store <1 x float> %1000, ptr addrspace(7) %1079, align 4
  %1080 = getelementptr i8, ptr addrspace(7) %1007, i32 819200
  %1081 = getelementptr float, ptr addrspace(7) %1080, i32 %1005
  store <1 x float> %989, ptr addrspace(7) %1081, align 4
  %1082 = getelementptr float, ptr addrspace(7) %1080, i32 %1009
  store <1 x float> %993, ptr addrspace(7) %1082, align 4
  %1083 = getelementptr float, ptr addrspace(7) %1080, i32 %1011
  store <1 x float> %997, ptr addrspace(7) %1083, align 4
  %1084 = getelementptr float, ptr addrspace(7) %1080, i32 %1013
  store <1 x float> %1001, ptr addrspace(7) %1084, align 4
  %1085 = getelementptr i8, ptr addrspace(7) %1007, i32 835584
  %1086 = getelementptr float, ptr addrspace(7) %1085, i32 %1005
  store <1 x float> %990, ptr addrspace(7) %1086, align 4
  %1087 = getelementptr float, ptr addrspace(7) %1085, i32 %1009
  store <1 x float> %994, ptr addrspace(7) %1087, align 4
  %1088 = getelementptr float, ptr addrspace(7) %1085, i32 %1011
  store <1 x float> %998, ptr addrspace(7) %1088, align 4
  %1089 = getelementptr float, ptr addrspace(7) %1085, i32 %1013
  store <1 x float> %1002, ptr addrspace(7) %1089, align 4
  ret void
}

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #2

; Function Attrs: alwaysinline mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) readnone, i16, i64, i32) #3

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: alwaysinline mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.load.async.to.lds.p7(ptr addrspace(7) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32 immarg, i32 immarg) #4

; Function Attrs: alwaysinline nounwind
declare void @llvm.amdgcn.asyncmark() #5

; Function Attrs: alwaysinline nounwind
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #5

; Function Attrs: alwaysinline convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: alwaysinline convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #7

attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size"="true" }
attributes #1 = { alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { alwaysinline mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { alwaysinline mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { alwaysinline mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { alwaysinline nounwind }
attributes #6 = { alwaysinline convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { alwaysinline convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!2 = !{i32 256, i32 1, i32 1}
!3 = !{!"amdgpu-synchronize-as", !"local"}
