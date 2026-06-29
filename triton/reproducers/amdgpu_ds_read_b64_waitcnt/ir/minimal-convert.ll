; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_smem = external local_unnamed_addr addrspace(3) global [0 x i8], align 16

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define amdgpu_kernel void @repro(ptr addrspace(1) inreg readonly captures(none) %0, ptr addrspace(1) inreg writeonly captures(none) %1, ptr addrspace(1) inreg readnone captures(none) %2, ptr addrspace(1) inreg readnone captures(none) %3) local_unnamed_addr #0 !dbg !4 {
  %5 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !11
  %6 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %5), !dbg !11
  %7 = lshr i32 %6, 6, !dbg !11
  %8 = and i32 %7, 3, !dbg !11
  %9 = and i32 %5, 63, !dbg !12
  %10 = shl nuw nsw i32 %8, 6, !dbg !12
  %11 = or disjoint i32 %10, %9, !dbg !12
  %12 = and i32 %5, 7, !dbg !13
  %13 = shl nuw nsw i32 %12, 2, !dbg !13
  %14 = shl nuw nsw i32 %11, 4, !dbg !14
  %15 = and i32 %14, 3968, !dbg !14
  %16 = and i32 %14, 3968, !dbg !14
  %17 = or disjoint i32 %16, %13, !dbg !15
  %18 = or disjoint i32 %15, %13, !dbg !15
  %19 = zext nneg i32 %17 to i64, !dbg !16
  %20 = getelementptr [4 x i8], ptr addrspace(1) %0, i64 %19, !dbg !16
  %21 = getelementptr i8, ptr addrspace(1) %20, i64 128, !dbg !16
  %22 = getelementptr i8, ptr addrspace(1) %20, i64 256, !dbg !16
  %23 = zext nneg i32 %18 to i64, !dbg !16
  %24 = getelementptr [4 x i8], ptr addrspace(1) %0, i64 %23, !dbg !16
  %25 = getelementptr i8, ptr addrspace(1) %24, i64 384, !dbg !16
  %26 = load <4 x float>, ptr addrspace(1) %20, align 16, !dbg !17
  %27 = load <4 x float>, ptr addrspace(1) %21, align 16, !dbg !17
  %28 = load <4 x float>, ptr addrspace(1) %22, align 16, !dbg !17
  %29 = load <4 x float>, ptr addrspace(1) %25, align 16, !dbg !17
  %30 = shl nuw nsw i32 %5, 8, !dbg !18
  %31 = and i32 %30, 12288, !dbg !18
  %32 = shl nuw nsw i32 %12, 5, !dbg !18
  %33 = shl nuw nsw i32 %11, 2, !dbg !18
  %34 = and i32 %33, 960, !dbg !18
  %35 = shl nuw nsw i32 %5, 1, !dbg !18
  %36 = and i32 %35, 16, !dbg !18
  %37 = or disjoint i32 %31, %32, !dbg !18
  %38 = xor i32 %34, %37, !dbg !18
  %39 = or disjoint i32 %38, %36, !dbg !18
  %40 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %39, !dbg !18
  store <4 x float> %26, ptr addrspace(3) %40, align 16, !dbg !18
  %41 = xor i32 %39, 1040, !dbg !18
  %42 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %41, !dbg !18
  store <4 x float> %27, ptr addrspace(3) %42, align 16, !dbg !18
  %43 = xor i32 %39, 2080, !dbg !18
  %44 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %43, !dbg !18
  store <4 x float> %28, ptr addrspace(3) %44, align 16, !dbg !18
  %45 = xor i32 %39, 3120, !dbg !18
  %46 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %45, !dbg !18
  store <4 x float> %29, ptr addrspace(3) %46, align 16, !dbg !18
  tail call void @llvm.amdgcn.wave.barrier(), !dbg !18
  %47 = and i32 %5, 3, !dbg !18
  %48 = shl nuw nsw i32 %47, 10, !dbg !18
  %49 = and i32 %5, 24, !dbg !18
  %50 = shl nuw nsw i32 %49, 9, !dbg !18
  %51 = shl nuw nsw i32 %47, 4, !dbg !18
  %52 = shl nuw nsw i32 %49, 3, !dbg !18
  %53 = and i32 %33, 784, !dbg !18
  %54 = and i32 %5, 32, !dbg !18
  %55 = or disjoint i32 %48, %50, !dbg !18
  %56 = or disjoint i32 %51, %52, !dbg !18
  %57 = or disjoint i32 %53, %54, !dbg !18
  %58 = xor i32 %57, %56, !dbg !18
  %59 = or disjoint i32 %58, %55, !dbg !18
  %60 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %59, !dbg !18
  %61 = load <4 x float>, ptr addrspace(3) %60, align 16, !dbg !18
  %62 = extractelement <4 x float> %61, i64 0, !dbg !18
  %63 = extractelement <4 x float> %61, i64 1, !dbg !18
  %64 = extractelement <4 x float> %61, i64 2, !dbg !18
  %65 = extractelement <4 x float> %61, i64 3, !dbg !18
  %66 = xor i32 %59, 64, !dbg !18
  %67 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %66, !dbg !18
  %68 = load <4 x float>, ptr addrspace(3) %67, align 16, !dbg !18
  %69 = extractelement <4 x float> %68, i64 0, !dbg !18
  %70 = extractelement <4 x float> %68, i64 1, !dbg !18
  %71 = extractelement <4 x float> %68, i64 2, !dbg !18
  %72 = extractelement <4 x float> %68, i64 3, !dbg !18
  %73 = xor i32 %59, 128, !dbg !18
  %74 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %73, !dbg !18
  %75 = load <4 x float>, ptr addrspace(3) %74, align 16, !dbg !18
  %76 = extractelement <4 x float> %75, i64 0, !dbg !18
  %77 = extractelement <4 x float> %75, i64 1, !dbg !18
  %78 = extractelement <4 x float> %75, i64 2, !dbg !18
  %79 = extractelement <4 x float> %75, i64 3, !dbg !18
  %80 = xor i32 %59, 192, !dbg !18
  %81 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %80, !dbg !18
  %82 = load <4 x float>, ptr addrspace(3) %81, align 16, !dbg !18
  %83 = extractelement <4 x float> %82, i64 0, !dbg !18
  %84 = extractelement <4 x float> %82, i64 1, !dbg !18
  %85 = extractelement <4 x float> %82, i64 2, !dbg !18
  %86 = extractelement <4 x float> %82, i64 3, !dbg !18
  %87 = tail call float @llvm.maxnum.f32(float %62, float %63), !dbg !19
  %88 = tail call float @llvm.maxnum.f32(float %87, float %64), !dbg !19
  %89 = tail call float @llvm.maxnum.f32(float %65, float %69), !dbg !19
  %90 = tail call float @llvm.maxnum.f32(float %89, float %70), !dbg !19
  %91 = tail call float @llvm.maxnum.f32(float %71, float %72), !dbg !19
  %92 = tail call float @llvm.maxnum.f32(float %91, float %76), !dbg !19
  %93 = tail call float @llvm.maxnum.f32(float %77, float %78), !dbg !19
  %94 = tail call float @llvm.maxnum.f32(float %93, float %79), !dbg !19
  %95 = tail call float @llvm.maxnum.f32(float %83, float %84), !dbg !19
  %96 = tail call float @llvm.maxnum.f32(float %95, float %85), !dbg !19
  %97 = tail call float @llvm.maxnum.f32(float %88, float %90), !dbg !19
  %98 = tail call float @llvm.maxnum.f32(float %97, float %92), !dbg !19
  %99 = tail call float @llvm.maxnum.f32(float %94, float %96), !dbg !19
  %100 = tail call float @llvm.maxnum.f32(float %99, float %86), !dbg !19
  %101 = tail call float @llvm.maxnum.f32(float %98, float %100), !dbg !19
  %102 = bitcast float %101 to i32, !dbg !20
  %103 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %102, i32 %102, i1 false, i1 false), !dbg !20
  %104 = extractvalue { i32, i32 } %103, 0, !dbg !20
  %105 = extractvalue { i32, i32 } %103, 1, !dbg !20
  %106 = bitcast i32 %104 to float, !dbg !20
  %107 = bitcast i32 %105 to float, !dbg !20
  %108 = tail call float @llvm.maxnum.f32(float %106, float %107), !dbg !19
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  %109 = shl nuw nsw i32 %5, 3, !dbg !21
  %110 = and i32 %109, 248, !dbg !21
  %111 = shl nuw nsw i32 %8, 8, !dbg !21
  %112 = and i32 %111, 256, !dbg !21
  %113 = and i32 %10, 128, !dbg !21
  %114 = lshr exact i32 %113, 5, !dbg !21
  %115 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %110, !dbg !21
  %116 = getelementptr inbounds nuw i8, ptr addrspace(3) %115, i32 %114, !dbg !21
  %117 = getelementptr inbounds nuw i8, ptr addrspace(3) %116, i32 %112, !dbg !21
  %118 = insertelement <1 x float> poison, float %108, i64 0, !dbg !21
  store <1 x float> %118, ptr addrspace(3) %117, align 4, !dbg !21
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !21
  tail call void @llvm.amdgcn.s.barrier(), !dbg !21
  %119 = and i32 %109, 120, !dbg !21
  %120 = shl nuw nsw i32 %8, 7, !dbg !21
  %121 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %119, !dbg !21
  %122 = getelementptr inbounds nuw i8, ptr addrspace(3) %121, i32 %120, !dbg !21
  %123 = load <2 x float>, ptr addrspace(3) %122, align 8, !dbg !21
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !22
  tail call void @llvm.amdgcn.s.barrier(), !dbg !22
  %124 = shl nuw nsw i32 %5, 4, !dbg !22
  %125 = and i32 %124, 224, !dbg !22
  %126 = and i32 %109, 8, !dbg !22
  %127 = lshr exact i32 %113, 3, !dbg !22
  %128 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %125, !dbg !22
  %129 = getelementptr inbounds nuw i8, ptr addrspace(3) %128, i32 %126, !dbg !22
  %130 = getelementptr inbounds nuw i8, ptr addrspace(3) %129, i32 %112, !dbg !22
  %131 = getelementptr inbounds nuw i8, ptr addrspace(3) %130, i32 %127, !dbg !22
  store <2 x float> %123, ptr addrspace(3) %131, align 8, !dbg !22
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !22
  tail call void @llvm.amdgcn.s.barrier(), !dbg !22
  %132 = and i32 %5, 56, !dbg !11
  %133 = or disjoint i32 %8, %132, !dbg !11
  %134 = icmp eq i32 %133, 0, !dbg !11
  br i1 %134, label %.critedge, label %.critedge6, !dbg !11

.critedge:                                        ; preds = %4
  %135 = shl nuw nsw i32 %12, 1, !dbg !23
  %136 = zext nneg i32 %135 to i64, !dbg !24
  %137 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %136, !dbg !24
  %138 = getelementptr i8, ptr addrspace(1) %137, i64 448, !dbg !24
  %139 = getelementptr i8, ptr addrspace(1) %137, i64 384, !dbg !24
  %140 = getelementptr i8, ptr addrspace(1) %137, i64 320, !dbg !24
  %141 = getelementptr i8, ptr addrspace(1) %137, i64 256, !dbg !24
  %142 = getelementptr i8, ptr addrspace(1) %137, i64 192, !dbg !24
  %143 = getelementptr i8, ptr addrspace(1) %137, i64 128, !dbg !24
  %144 = getelementptr i8, ptr addrspace(1) %137, i64 64, !dbg !24
  %145 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %32, !dbg !22
  %146 = getelementptr inbounds nuw i8, ptr addrspace(3) %145, i32 272, !dbg !22
  %147 = load <4 x float>, ptr addrspace(3) %146, align 16, !dbg !22
  %148 = getelementptr inbounds nuw i8, ptr addrspace(3) %145, i32 16, !dbg !22
  %149 = load <4 x float>, ptr addrspace(3) %148, align 16, !dbg !22
  %150 = getelementptr inbounds nuw i8, ptr addrspace(3) %145, i32 256, !dbg !22
  %151 = load <4 x float>, ptr addrspace(3) %150, align 16, !dbg !22
  %152 = load <4 x float>, ptr addrspace(3) %145, align 16, !dbg !22
  %153 = shufflevector <4 x float> %152, <4 x float> poison, <2 x i32> <i32 0, i32 2>, !dbg !11
  store <2 x float> %153, ptr addrspace(1) %137, align 8, !dbg !11
  %154 = shufflevector <4 x float> %151, <4 x float> poison, <2 x i32> <i32 0, i32 2>, !dbg !11
  store <2 x float> %154, ptr addrspace(1) %144, align 8, !dbg !11
  %155 = shufflevector <4 x float> %149, <4 x float> poison, <2 x i32> <i32 0, i32 2>, !dbg !11
  store <2 x float> %155, ptr addrspace(1) %143, align 8, !dbg !11
  %156 = shufflevector <4 x float> %147, <4 x float> poison, <2 x i32> <i32 0, i32 2>, !dbg !11
  store <2 x float> %156, ptr addrspace(1) %142, align 8, !dbg !11
  %157 = shufflevector <4 x float> %152, <4 x float> poison, <2 x i32> <i32 1, i32 3>, !dbg !11
  store <2 x float> %157, ptr addrspace(1) %141, align 8, !dbg !11
  %158 = shufflevector <4 x float> %151, <4 x float> poison, <2 x i32> <i32 1, i32 3>, !dbg !11
  store <2 x float> %158, ptr addrspace(1) %140, align 8, !dbg !11
  %159 = shufflevector <4 x float> %149, <4 x float> poison, <2 x i32> <i32 1, i32 3>, !dbg !11
  store <2 x float> %159, ptr addrspace(1) %139, align 8, !dbg !11
  %160 = shufflevector <4 x float> %147, <4 x float> poison, <2 x i32> <i32 1, i32 3>, !dbg !11
  store <2 x float> %160, ptr addrspace(1) %138, align 8, !dbg !11
  br label %.critedge6, !dbg !11

.critedge6:                                       ; preds = %4, %.critedge
  ret void, !dbg !25
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wave.barrier() #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1 immarg, i1 immarg) #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #3

attributes #0 = { mustprogress nofree norecurse nounwind willreturn "amdgpu-agpr-alloc"="0" "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "amdgpu-waves-per-eu"="0, 0" "denormal-fp-math-f32"="ieee" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #4 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "redu.ttgir", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!4 = distinct !DISubprogram(name: "repro", linkageName: "repro", scope: !1, file: !1, line: 6, type: !5, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{null, !7, !7, !9, !9}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !8, size: 64, dwarfAddressSpace: 1)
!8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !10, size: 64, dwarfAddressSpace: 1)
!10 = !DIBasicType(name: "unknown_type", encoding: DW_ATE_signed)
!11 = !DILocation(line: 32, column: 5, scope: !4)
!12 = !DILocation(line: 8, column: 12, scope: !4)
!13 = !DILocation(line: 10, column: 13, scope: !4)
!14 = !DILocation(line: 12, column: 13, scope: !4)
!15 = !DILocation(line: 15, column: 13, scope: !4)
!16 = !DILocation(line: 17, column: 11, scope: !4)
!17 = !DILocation(line: 18, column: 11, scope: !4)
!18 = !DILocation(line: 19, column: 11, scope: !4)
!19 = !DILocation(line: 22, column: 13, scope: !4)
!20 = !DILocation(line: 20, column: 10, scope: !4)
!21 = !DILocation(line: 26, column: 14, scope: !4)
!22 = !DILocation(line: 27, column: 13, scope: !4)
!23 = !DILocation(line: 28, column: 10, scope: !4)
!24 = !DILocation(line: 31, column: 12, scope: !4)
!25 = !DILocation(line: 33, column: 5, scope: !4)
