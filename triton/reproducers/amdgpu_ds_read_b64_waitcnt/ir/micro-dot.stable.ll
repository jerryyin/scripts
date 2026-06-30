; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define amdgpu_kernel void @repro(ptr addrspace(1) inreg writeonly captures(none) %0, float inreg %1, ptr addrspace(1) inreg readnone captures(none) %2, ptr addrspace(1) inreg readnone captures(none) %3) local_unnamed_addr #0 !dbg !4 {
  %5 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10
  %6 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %5), !dbg !11
  %7 = and i32 %5, 63, !dbg !12
  %8 = insertelement <2 x i32> poison, i32 %6, i64 0, !dbg !10
  %9 = insertelement <2 x i32> %8, i32 %5, i64 1, !dbg !10
  %10 = and <2 x i32> %9, <i32 192, i32 32>, !dbg !10
  %11 = extractelement <2 x i32> %10, i64 0, !dbg !13
  %12 = or disjoint i32 %11, %7, !dbg !12
  %13 = and i32 %5, 31, !dbg !12
  %14 = lshr exact <2 x i32> %10, <i32 1, i32 3>, !dbg !12
  %15 = extractelement <2 x i32> %14, i64 1, !dbg !14
  %16 = or disjoint i32 %15, 1, !dbg !14
  %17 = shufflevector <2 x i32> %14, <2 x i32> poison, <2 x i32> <i32 1, i32 1>, !dbg !14
  %18 = or disjoint <2 x i32> %17, <i32 2, i32 3>, !dbg !14
  %19 = insertelement <2 x i32> <i32 poison, i32 10>, i32 %13, i64 0, !dbg !12
  %20 = or disjoint <2 x i32> %14, %19, !dbg !12
  %21 = or disjoint i32 %15, 11, !dbg !14
  %22 = or disjoint i32 %15, 9, !dbg !14
  %23 = or disjoint i32 %15, 8, !dbg !14
  %24 = or disjoint i32 %15, 16, !dbg !14
  %25 = or disjoint i32 %15, 27, !dbg !14
  %26 = uitofp nneg <2 x i32> %20 to <2 x float>, !dbg !15
  %27 = insertelement <4 x i32> poison, i32 %23, i64 0, !dbg !16
  %28 = insertelement <4 x i32> %27, i32 %22, i64 1, !dbg !16
  %29 = shufflevector <2 x i32> %20, <2 x i32> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>, !dbg !16
  %30 = shufflevector <4 x i32> %28, <4 x i32> %29, <4 x i32> <i32 0, i32 1, i32 4, i32 poison>, !dbg !16
  %31 = insertelement <4 x i32> %30, i32 %21, i64 3, !dbg !16
  %32 = shufflevector <2 x i32> %14, <2 x i32> poison, <8 x i32> <i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %33 = insertelement <8 x i32> %32, i32 %16, i64 1, !dbg !16
  %34 = shufflevector <4 x i32> %31, <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %35 = shufflevector <8 x i32> %33, <8 x i32> %34, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %36 = shufflevector <2 x i32> %18, <2 x i32> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !16
  %37 = shufflevector <8 x i32> %35, <8 x i32> %36, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %38 = uitofp <8 x i32> %37 to <8 x float>, !dbg !16
  %39 = uitofp nneg i32 %24 to float, !dbg !16
  %40 = uitofp nneg i32 %25 to float, !dbg !16
  %41 = fmul nnan <2 x float> %26, <float 3.906250e-03, float 3.125000e-02>, !dbg !17
  %42 = shufflevector <2 x float> %41, <2 x float> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>, !dbg !17
  %43 = fmul nnan <8 x float> %38, <float 3.125000e-02, float 3.125000e-02, float 3.125000e-02, float 3.125000e-02, float 3.125000e-02, float 3.125000e-02, float 3.906250e-03, float 3.125000e-02>, !dbg !18
  %44 = fmul nnan float %39, 3.125000e-02, !dbg !18
  %45 = insertelement <2 x float> poison, float %1, i64 0, !dbg !19
  %46 = insertelement <2 x float> %45, float %40, i64 1, !dbg !19
  %47 = fmul <2 x float> %46, <float 0x3FF7154760000000, float 3.125000e-02>, !dbg !19
  %48 = extractelement <2 x float> %47, i64 0, !dbg !19
  %49 = fadd <8 x float> %43, %42, !dbg !20
  %50 = extractelement <8 x float> %43, i64 6, !dbg !20
  %51 = fadd float %44, %50, !dbg !20
  %52 = extractelement <2 x float> %47, i64 1, !dbg !20
  %53 = fadd float %52, %50, !dbg !20
  %54 = shufflevector <2 x float> %47, <2 x float> poison, <8 x i32> zeroinitializer, !dbg !21
  %55 = fadd <8 x float> %54, %49, !dbg !21
  %56 = fadd float %48, %51, !dbg !21
  %57 = shufflevector <2 x i32> %14, <2 x i32> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !14
  %58 = or disjoint <4 x i32> %57, <i32 17, i32 18, i32 19, i32 24>, !dbg !14
  %59 = uitofp nneg <4 x i32> %58 to <4 x float>, !dbg !16
  %60 = fmul nnan <4 x float> %59, splat (float 3.125000e-02), !dbg !18
  %61 = shufflevector <8 x float> %43, <8 x float> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>, !dbg !20
  %62 = fadd <4 x float> %60, %61, !dbg !20
  %63 = shufflevector <2 x float> %47, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %64 = fadd <4 x float> %63, %62, !dbg !21
  %65 = fadd float %48, %53, !dbg !21
  %66 = extractelement <8 x float> %55, i64 0, !dbg !22
  %67 = extractelement <8 x float> %55, i64 1, !dbg !22
  %68 = tail call float @llvm.maxnum.f32(float %66, float %67), !dbg !23
  %69 = extractelement <8 x float> %55, i64 2, !dbg !22
  %70 = tail call float @llvm.maxnum.f32(float %68, float %69), !dbg !23
  %71 = extractelement <8 x float> %55, i64 3, !dbg !22
  %72 = extractelement <8 x float> %55, i64 4, !dbg !22
  %73 = tail call float @llvm.maxnum.f32(float %71, float %72), !dbg !23
  %74 = extractelement <8 x float> %55, i64 5, !dbg !22
  %75 = tail call float @llvm.maxnum.f32(float %73, float %74), !dbg !23
  %76 = extractelement <8 x float> %55, i64 6, !dbg !22
  %77 = extractelement <8 x float> %55, i64 7, !dbg !22
  %78 = tail call float @llvm.maxnum.f32(float %76, float %77), !dbg !23
  %79 = tail call float @llvm.maxnum.f32(float %78, float %56), !dbg !23
  %80 = extractelement <4 x float> %64, i64 0, !dbg !22
  %81 = extractelement <4 x float> %64, i64 1, !dbg !22
  %82 = tail call float @llvm.maxnum.f32(float %80, float %81), !dbg !23
  %83 = extractelement <4 x float> %64, i64 2, !dbg !22
  %84 = tail call float @llvm.maxnum.f32(float %82, float %83), !dbg !23
  %85 = extractelement <4 x float> %64, i64 3, !dbg !22
  %86 = tail call float @llvm.maxnum.f32(float %70, float %75), !dbg !23
  %87 = tail call float @llvm.maxnum.f32(float %86, float %79), !dbg !23
  %88 = shufflevector <8 x float> %55, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !22
  %89 = shufflevector <8 x float> %55, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !22
  %90 = shufflevector <8 x float> %55, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !22
  %91 = shufflevector <8 x float> %55, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !22
  %92 = shufflevector <4 x float> %64, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !22
  %93 = shufflevector <4 x float> %64, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !22
  %94 = or disjoint <2 x i32> %17, <i32 25, i32 26>, !dbg !14
  %95 = uitofp nneg <2 x i32> %94 to <2 x float>, !dbg !16
  %96 = fmul nnan <2 x float> %95, splat (float 3.125000e-02), !dbg !18
  %97 = shufflevector <8 x float> %43, <8 x float> poison, <2 x i32> <i32 6, i32 6>, !dbg !20
  %98 = fadd <2 x float> %96, %97, !dbg !20
  %99 = shufflevector <2 x float> %47, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !21
  %100 = fadd <2 x float> %99, %98, !dbg !21
  %101 = extractelement <2 x float> %100, i64 0, !dbg !23
  %102 = tail call float @llvm.maxnum.f32(float %85, float %101), !dbg !23
  %103 = extractelement <2 x float> %100, i64 1, !dbg !23
  %104 = tail call float @llvm.maxnum.f32(float %102, float %103), !dbg !23
  %105 = tail call float @llvm.maxnum.f32(float %84, float %104), !dbg !23
  %106 = tail call float @llvm.maxnum.f32(float %105, float %65), !dbg !23
  %107 = tail call float @llvm.maxnum.f32(float %87, float %106), !dbg !23
  %108 = bitcast float %107 to i32, !dbg !24
  %109 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %108, i32 %108, i1 false, i1 false), !dbg !24
  %110 = extractvalue { i32, i32 } %109, 0, !dbg !24
  %111 = extractvalue { i32, i32 } %109, 1, !dbg !24
  %112 = bitcast i32 %110 to float, !dbg !24
  %113 = bitcast i32 %111 to float, !dbg !24
  %114 = tail call float @llvm.maxnum.f32(float %112, float %113), !dbg !23
  %115 = tail call float @llvm.maxnum.f32(float %114, float 0xFFF0000000000000), !dbg !25
  %116 = insertelement <2 x float> poison, float %115, i64 0, !dbg !22
  %117 = shufflevector <2 x float> %116, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !22
  %118 = fsub <2 x float> %88, %117, !dbg !22
  %119 = fsub <2 x float> %89, %117, !dbg !22
  %120 = fsub <2 x float> %90, %117, !dbg !22
  %121 = fsub <2 x float> %91, %117, !dbg !22
  %122 = fsub float %56, %115, !dbg !22
  %123 = fsub <2 x float> %92, %117, !dbg !22
  %124 = fsub <2 x float> %93, %117, !dbg !22
  %125 = fsub <2 x float> %100, %117, !dbg !22
  %126 = fsub float %65, %115, !dbg !22
  %127 = extractelement <2 x float> %118, i64 0, !dbg !26
  %128 = tail call float @llvm.amdgcn.exp2.f32(float %127), !dbg !26
  %129 = extractelement <2 x float> %118, i64 1, !dbg !26
  %130 = tail call float @llvm.amdgcn.exp2.f32(float %129), !dbg !26
  %131 = extractelement <2 x float> %119, i64 0, !dbg !26
  %132 = tail call float @llvm.amdgcn.exp2.f32(float %131), !dbg !26
  %133 = extractelement <2 x float> %119, i64 1, !dbg !26
  %134 = tail call float @llvm.amdgcn.exp2.f32(float %133), !dbg !26
  %135 = extractelement <2 x float> %120, i64 0, !dbg !26
  %136 = tail call float @llvm.amdgcn.exp2.f32(float %135), !dbg !26
  %137 = extractelement <2 x float> %120, i64 1, !dbg !26
  %138 = tail call float @llvm.amdgcn.exp2.f32(float %137), !dbg !26
  %139 = extractelement <2 x float> %121, i64 0, !dbg !26
  %140 = tail call float @llvm.amdgcn.exp2.f32(float %139), !dbg !26
  %141 = extractelement <2 x float> %121, i64 1, !dbg !26
  %142 = tail call float @llvm.amdgcn.exp2.f32(float %141), !dbg !26
  %143 = tail call float @llvm.amdgcn.exp2.f32(float %122), !dbg !26
  %144 = extractelement <2 x float> %123, i64 0, !dbg !26
  %145 = tail call float @llvm.amdgcn.exp2.f32(float %144), !dbg !26
  %146 = extractelement <2 x float> %123, i64 1, !dbg !26
  %147 = tail call float @llvm.amdgcn.exp2.f32(float %146), !dbg !26
  %148 = extractelement <2 x float> %124, i64 0, !dbg !26
  %149 = tail call float @llvm.amdgcn.exp2.f32(float %148), !dbg !26
  %150 = extractelement <2 x float> %124, i64 1, !dbg !26
  %151 = tail call float @llvm.amdgcn.exp2.f32(float %150), !dbg !26
  %152 = extractelement <2 x float> %125, i64 0, !dbg !26
  %153 = tail call float @llvm.amdgcn.exp2.f32(float %152), !dbg !26
  %154 = extractelement <2 x float> %125, i64 1, !dbg !26
  %155 = tail call float @llvm.amdgcn.exp2.f32(float %154), !dbg !26
  %156 = tail call float @llvm.amdgcn.exp2.f32(float %126), !dbg !26
  %157 = insertelement <2 x float> poison, float %128, i64 0, !dbg !27
  %158 = insertelement <2 x float> %157, float %130, i64 1, !dbg !27
  %159 = insertelement <2 x float> poison, float %132, i64 0, !dbg !27
  %160 = insertelement <2 x float> %159, float %134, i64 1, !dbg !27
  %161 = insertelement <2 x float> poison, float %136, i64 0, !dbg !27
  %162 = insertelement <2 x float> %161, float %138, i64 1, !dbg !27
  %163 = insertelement <2 x float> poison, float %140, i64 0, !dbg !27
  %164 = insertelement <2 x float> %163, float %142, i64 1, !dbg !27
  %165 = insertelement <2 x float> poison, float %143, i64 0, !dbg !27
  %166 = insertelement <2 x float> %165, float %145, i64 1, !dbg !27
  %167 = insertelement <2 x float> poison, float %147, i64 0, !dbg !27
  %168 = insertelement <2 x float> %167, float %149, i64 1, !dbg !27
  %169 = insertelement <2 x float> poison, float %151, i64 0, !dbg !27
  %170 = insertelement <2 x float> %169, float %153, i64 1, !dbg !27
  %171 = insertelement <2 x float> poison, float %155, i64 0, !dbg !27
  %172 = insertelement <2 x float> %171, float %156, i64 1, !dbg !27
  %173 = fadd <2 x float> %158, %160, !dbg !27
  %174 = fadd <2 x float> %162, %164, !dbg !27
  %175 = fadd <2 x float> %166, %168, !dbg !27
  %176 = fadd <2 x float> %170, %172, !dbg !27
  %177 = fadd <2 x float> %173, %174, !dbg !27
  %178 = fadd <2 x float> %175, %176, !dbg !27
  %179 = fadd <2 x float> %177, %178, !dbg !27
  %shift = shufflevector <2 x float> %179, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !28
  %foldExtExtBinop = fadd <2 x float> %179, %shift, !dbg !28
  %bc = bitcast <2 x float> %foldExtExtBinop to <2 x i32>, !dbg !27
  %180 = extractelement <2 x i32> %bc, i64 0, !dbg !27
  %181 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %180, i32 %180, i1 false, i1 false), !dbg !27
  %182 = extractvalue { i32, i32 } %181, 0, !dbg !27
  %183 = extractvalue { i32, i32 } %181, 1, !dbg !27
  %184 = bitcast i32 %182 to float, !dbg !27
  %185 = bitcast i32 %183 to float, !dbg !27
  %186 = fadd float %184, %185, !dbg !28
  %187 = shl nuw nsw i32 %12, 2, !dbg !29
  %188 = and i32 %187, 764, !dbg !29
  %189 = and i32 %6, 64, !dbg !29
  %190 = icmp eq i32 %189, 0, !dbg !29
  %191 = select i1 %190, i32 0, i32 272, !dbg !29
  %192 = xor i32 %188, %191, !dbg !29
  %193 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %192, !dbg !29
  store <2 x bfloat> splat (bfloat 0xR3F80), ptr addrspace(3) %193, align 4, !dbg !29
  %194 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %192, !dbg !30
  store <2 x bfloat> splat (bfloat 0xR3F80), ptr addrspace(3) %194, align 4, !dbg !30
  %195 = select i1 %190, i32 0, i32 264, !dbg !31
  %196 = xor i32 %188, %195, !dbg !31
  %197 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 2048), i32 %196, !dbg !31
  store <2 x bfloat> splat (bfloat 0xR3F80), ptr addrspace(3) %197, align 4, !dbg !31
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !32
  tail call void @llvm.amdgcn.s.barrier(), !dbg !32
  %198 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 9216), i32 %192, !dbg !32
  store <2 x bfloat> splat (bfloat 0xR3F80), ptr addrspace(3) %198, align 4, !dbg !32
  %199 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 1024), i32 %192, !dbg !33
  store <2 x bfloat> splat (bfloat 0xR3F80), ptr addrspace(3) %199, align 4, !dbg !33
  %200 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 3072), i32 %196, !dbg !34
  store <2 x bfloat> splat (bfloat 0xR3F80), ptr addrspace(3) %200, align 4, !dbg !34
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !35
  tail call void @llvm.amdgcn.s.barrier(), !dbg !35
  %201 = shl nuw nsw i32 %13, 3, !dbg !35
  %202 = extractelement <2 x i32> %10, i64 1, !dbg !35
  %203 = icmp eq i32 %202, 0, !dbg !35
  %204 = select i1 %203, i32 0, i32 264, !dbg !35
  %205 = xor i32 %204, %201, !dbg !35
  %206 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 2048), i32 %205, !dbg !35
  %207 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %206), !dbg !35
  %208 = getelementptr inbounds nuw i8, ptr addrspace(3) %206, i32 512, !dbg !35
  %209 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %208), !dbg !35
  %210 = and i32 %187, 380, !dbg !36
  %211 = and i32 %6, 128, !dbg !36
  %212 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %210, !dbg !36
  %213 = getelementptr inbounds nuw i8, ptr addrspace(3) %212, i32 %211, !dbg !36
  %214 = insertelement <1 x float> poison, float %186, i64 0, !dbg !36
  store <1 x float> %214, ptr addrspace(3) %213, align 4, !dbg !36
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !36
  tail call void @llvm.amdgcn.s.barrier(), !dbg !36
  %215 = and i32 %5, 15, !dbg !36
  %216 = shl nuw nsw i32 %215, 2, !dbg !36
  %217 = shl nuw nsw i32 %211, 1, !dbg !36
  %218 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %217, !dbg !36
  %219 = getelementptr inbounds nuw i8, ptr addrspace(3) %218, i32 %216, !dbg !36
  %220 = getelementptr inbounds nuw i8, ptr addrspace(3) %219, i32 %189, !dbg !36
  %221 = load <1 x float>, ptr addrspace(3) %220, align 4, !dbg !36
  %222 = extractelement <1 x float> %221, i64 0, !dbg !36
  %223 = getelementptr inbounds nuw i8, ptr addrspace(3) %220, i32 128, !dbg !36
  %224 = load <1 x float>, ptr addrspace(3) %223, align 4, !dbg !36
  %225 = extractelement <1 x float> %224, i64 0, !dbg !36
  %226 = fmul float %222, 0.000000e+00, !dbg !37
  %227 = fmul float %225, 0.000000e+00, !dbg !37
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  %228 = shl nuw nsw i32 %13, 1, !dbg !13
  %229 = select i1 %203, i32 0, i32 1056, !dbg !13
  %230 = or disjoint i32 %11, %228, !dbg !13
  %231 = xor i32 %230, %229, !dbg !13
  %232 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %231, !dbg !13
  %233 = shufflevector <2 x float> %157, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !13
  %234 = fptrunc <1 x float> %233 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %234, ptr addrspace(3) %232, align 2, !dbg !13
  %235 = getelementptr inbounds nuw i8, ptr addrspace(3) %232, i32 4096, !dbg !13
  %236 = shufflevector <2 x float> %165, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !13
  %237 = fptrunc <1 x float> %236 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %237, ptr addrspace(3) %235, align 2, !dbg !13
  %238 = xor i32 %231, 264, !dbg !13
  %239 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %238, !dbg !13
  %240 = shufflevector <2 x float> %158, <2 x float> poison, <1 x i32> <i32 1>, !dbg !13
  %241 = fptrunc <1 x float> %240 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %241, ptr addrspace(3) %239, align 2, !dbg !13
  %242 = getelementptr inbounds nuw i8, ptr addrspace(3) %239, i32 4096, !dbg !13
  %243 = shufflevector <2 x float> %166, <2 x float> poison, <1 x i32> <i32 1>, !dbg !13
  %244 = fptrunc <1 x float> %243 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %244, ptr addrspace(3) %242, align 2, !dbg !13
  %245 = xor i32 %231, 528, !dbg !13
  %246 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %245, !dbg !13
  %247 = shufflevector <2 x float> %159, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !13
  %248 = fptrunc <1 x float> %247 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %248, ptr addrspace(3) %246, align 2, !dbg !13
  %249 = getelementptr inbounds nuw i8, ptr addrspace(3) %246, i32 4096, !dbg !13
  %250 = shufflevector <2 x float> %167, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !13
  %251 = fptrunc <1 x float> %250 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %251, ptr addrspace(3) %249, align 2, !dbg !13
  %252 = xor i32 %231, 792, !dbg !13
  %253 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %252, !dbg !13
  %254 = shufflevector <2 x float> %160, <2 x float> poison, <1 x i32> <i32 1>, !dbg !13
  %255 = fptrunc <1 x float> %254 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %255, ptr addrspace(3) %253, align 2, !dbg !13
  %256 = getelementptr inbounds nuw i8, ptr addrspace(3) %253, i32 4096, !dbg !13
  %257 = shufflevector <2 x float> %168, <2 x float> poison, <1 x i32> <i32 1>, !dbg !13
  %258 = fptrunc <1 x float> %257 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %258, ptr addrspace(3) %256, align 2, !dbg !13
  %259 = xor i32 %231, 2112, !dbg !13
  %260 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %259, !dbg !13
  %261 = shufflevector <2 x float> %161, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !13
  %262 = fptrunc <1 x float> %261 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %262, ptr addrspace(3) %260, align 2, !dbg !13
  %263 = getelementptr inbounds nuw i8, ptr addrspace(3) %260, i32 4096, !dbg !13
  %264 = shufflevector <2 x float> %169, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !13
  %265 = fptrunc <1 x float> %264 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %265, ptr addrspace(3) %263, align 2, !dbg !13
  %266 = xor i32 %231, 2376, !dbg !13
  %267 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %266, !dbg !13
  %268 = shufflevector <2 x float> %162, <2 x float> poison, <1 x i32> <i32 1>, !dbg !13
  %269 = fptrunc <1 x float> %268 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %269, ptr addrspace(3) %267, align 2, !dbg !13
  %270 = getelementptr inbounds nuw i8, ptr addrspace(3) %267, i32 4096, !dbg !13
  %271 = shufflevector <2 x float> %170, <2 x float> poison, <1 x i32> <i32 1>, !dbg !13
  %272 = fptrunc <1 x float> %271 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %272, ptr addrspace(3) %270, align 2, !dbg !13
  %273 = xor i32 %231, 2640, !dbg !13
  %274 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %273, !dbg !13
  %275 = shufflevector <2 x float> %163, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !13
  %276 = fptrunc <1 x float> %275 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %276, ptr addrspace(3) %274, align 2, !dbg !13
  %277 = getelementptr inbounds nuw i8, ptr addrspace(3) %274, i32 4096, !dbg !13
  %278 = shufflevector <2 x float> %171, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !13
  %279 = fptrunc <1 x float> %278 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %279, ptr addrspace(3) %277, align 2, !dbg !13
  %280 = xor i32 %231, 2904, !dbg !13
  %281 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %280, !dbg !13
  %282 = shufflevector <2 x float> %164, <2 x float> poison, <1 x i32> <i32 1>, !dbg !13
  %283 = fptrunc <1 x float> %282 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %283, ptr addrspace(3) %281, align 2, !dbg !13
  %284 = getelementptr inbounds nuw i8, ptr addrspace(3) %281, i32 4096, !dbg !13
  %285 = shufflevector <2 x float> %172, <2 x float> poison, <1 x i32> <i32 1>, !dbg !13
  %286 = fptrunc <1 x float> %285 to <1 x bfloat>, !dbg !13
  store <1 x bfloat> %286, ptr addrspace(3) %284, align 2, !dbg !13
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  %287 = and i32 %5, 60, !dbg !38
  %288 = shl nuw nsw i32 %287, 6, !dbg !38
  %289 = shl nuw nsw i32 %5, 3, !dbg !38
  %290 = and i32 %289, 24, !dbg !38
  %291 = shl nuw nsw i32 %287, 1, !dbg !38
  %292 = or disjoint i32 %288, %290, !dbg !38
  %293 = xor i32 %292, %291, !dbg !38
  %294 = extractelement <2 x i32> %14, i64 0, !dbg !38
  %295 = xor i32 %293, %294, !dbg !38
  %296 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %295, !dbg !38
  %297 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %296), !dbg !38
  %298 = getelementptr inbounds nuw i8, ptr addrspace(3) %296, i32 4096, !dbg !38
  %299 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %298), !dbg !38
  %300 = getelementptr inbounds nuw i8, ptr addrspace(3) %296, i32 128, !dbg !38
  %301 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %300), !dbg !38
  %302 = getelementptr inbounds nuw i8, ptr addrspace(3) %296, i32 4224, !dbg !38
  %303 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %302), !dbg !38
  %304 = shufflevector <4 x bfloat> %297, <4 x bfloat> %299, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %305 = shufflevector <4 x bfloat> %301, <4 x bfloat> %303, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %306 = shufflevector <4 x bfloat> %207, <4 x bfloat> %209, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %307 = insertelement <4 x float> poison, float %226, i64 0, !dbg !39
  %308 = shufflevector <4 x float> %307, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %309 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %306, <8 x bfloat> %304, <4 x float> %308, i32 0, i32 0, i32 0), !dbg !39
  %310 = insertelement <4 x float> poison, float %227, i64 0, !dbg !39
  %311 = shufflevector <4 x float> %310, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %312 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %306, <8 x bfloat> %305, <4 x float> %311, i32 0, i32 0, i32 0), !dbg !39
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !40
  tail call void @llvm.amdgcn.s.barrier(), !dbg !40
  store <1 x float> %214, ptr addrspace(3) %213, align 4, !dbg !40
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !40
  tail call void @llvm.amdgcn.s.barrier(), !dbg !40
  %313 = load <1 x float>, ptr addrspace(3) %220, align 4, !dbg !40
  %314 = load <1 x float>, ptr addrspace(3) %223, align 4, !dbg !40
  %315 = fadd float %186, %186, !dbg !41
  %316 = fdiv float 1.000000e+00, %315, !dbg !42
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !43
  tail call void @llvm.amdgcn.s.barrier(), !dbg !43
  %317 = insertelement <1 x float> poison, float %316, i64 0, !dbg !43
  store <1 x float> %317, ptr addrspace(3) %213, align 4, !dbg !43
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !43
  tail call void @llvm.amdgcn.s.barrier(), !dbg !43
  %318 = load <1 x float>, ptr addrspace(3) %220, align 4, !dbg !43
  %319 = load <1 x float>, ptr addrspace(3) %223, align 4, !dbg !43
  %320 = shufflevector <4 x float> %309, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !44
  %321 = shufflevector <1 x float> %313, <1 x float> poison, <2 x i32> zeroinitializer, !dbg !44
  %322 = fmul <2 x float> %320, %321, !dbg !44
  %323 = shufflevector <1 x float> %318, <1 x float> poison, <2 x i32> zeroinitializer, !dbg !45
  %324 = fmul <2 x float> %322, %323, !dbg !45
  %325 = shufflevector <4 x float> %309, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !44
  %326 = fmul <2 x float> %325, %321, !dbg !44
  %327 = fmul <2 x float> %326, %323, !dbg !45
  %328 = shufflevector <4 x float> %312, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !44
  %329 = shufflevector <1 x float> %314, <1 x float> poison, <2 x i32> zeroinitializer, !dbg !44
  %330 = fmul <2 x float> %328, %329, !dbg !44
  %331 = shufflevector <1 x float> %319, <1 x float> poison, <2 x i32> zeroinitializer, !dbg !45
  %332 = fmul <2 x float> %330, %331, !dbg !45
  %333 = shufflevector <4 x float> %312, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !44
  %334 = fmul <2 x float> %333, %329, !dbg !44
  %335 = fmul <2 x float> %334, %331, !dbg !45
  %336 = fptrunc <2 x float> %324 to <2 x bfloat>, !dbg !46
  %337 = fptrunc <2 x float> %327 to <2 x bfloat>, !dbg !46
  %338 = fptrunc <2 x float> %332 to <2 x bfloat>, !dbg !46
  %339 = fptrunc <2 x float> %335 to <2 x bfloat>, !dbg !46
  %340 = load <2 x bfloat>, ptr addrspace(3) %193, align 4, !dbg !47
  %341 = lshr exact i32 %11, 2, !dbg !48
  %342 = or disjoint i32 %341, %215, !dbg !48
  %343 = lshr i32 %5, 2, !dbg !49
  %344 = and i32 %343, 12, !dbg !49
  %345 = shl nuw nsw i32 %342, 4, !dbg !50
  %346 = or disjoint i32 %345, %344, !dbg !51
  %347 = zext nneg i32 %346 to i64, !dbg !52
  %348 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %347, !dbg !52
  %349 = getelementptr i8, ptr addrspace(1) %348, i64 2048, !dbg !52
  %350 = shufflevector <2 x bfloat> %336, <2 x bfloat> %337, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  store <4 x bfloat> %350, ptr addrspace(1) %348, align 8, !dbg !53
  %351 = shufflevector <2 x bfloat> %338, <2 x bfloat> %339, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  store <4 x bfloat> %351, ptr addrspace(1) %349, align 8, !dbg !53
  %352 = shl nuw nsw i32 %5, 6, !dbg !54
  %353 = and i32 %352, 448, !dbg !54
  %354 = lshr i32 %12, 3, !dbg !11
  %355 = or disjoint i32 %354, %353, !dbg !55
  %356 = zext nneg i32 %355 to i64, !dbg !56
  %357 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %356, !dbg !56
  %358 = getelementptr i8, ptr addrspace(1) %357, i64 4096, !dbg !56
  %359 = getelementptr i8, ptr addrspace(1) %357, i64 4160, !dbg !56
  %360 = shufflevector <2 x bfloat> %340, <2 x bfloat> poison, <1 x i32> zeroinitializer, !dbg !10
  store <1 x bfloat> %360, ptr addrspace(1) %358, align 2, !dbg !10
  %361 = shufflevector <2 x bfloat> %340, <2 x bfloat> poison, <1 x i32> <i32 1>, !dbg !10
  store <1 x bfloat> %361, ptr addrspace(1) %359, align 2, !dbg !10
  ret void, !dbg !57
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1 immarg, i1 immarg) #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) captures(none)) #7

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #8

attributes #0 = { mustprogress nofree norecurse nounwind willreturn "amdgpu-agpr-alloc"="0" "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "amdgpu-waves-per-eu"="0, 0" "denormal-fp-math-f32"="ieee" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { convergent mustprogress nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #8 = { convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "micro-dot.ttgir", directory: "ir")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!4 = distinct !DISubprogram(name: "repro", linkageName: "repro", scope: !1, file: !1, line: 11, type: !5, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{null, !7, !9, !7, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !8, size: 64, dwarfAddressSpace: 1)
!8 = !DIBasicType(name: "unknown_type", encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!10 = !DILocation(line: 129, column: 5, scope: !4)
!11 = !DILocation(line: 114, column: 11, scope: !4)
!12 = !DILocation(line: 29, column: 11, scope: !4)
!13 = !DILocation(line: 80, column: 14, scope: !4)
!14 = !DILocation(line: 30, column: 11, scope: !4)
!15 = !DILocation(line: 33, column: 12, scope: !4)
!16 = !DILocation(line: 34, column: 12, scope: !4)
!17 = !DILocation(line: 37, column: 17, scope: !4)
!18 = !DILocation(line: 38, column: 17, scope: !4)
!19 = !DILocation(line: 27, column: 20, scope: !4)
!20 = !DILocation(line: 39, column: 16, scope: !4)
!21 = !DILocation(line: 40, column: 15, scope: !4)
!22 = !DILocation(line: 49, column: 17, scope: !4)
!23 = !DILocation(line: 43, column: 13, scope: !4)
!24 = !DILocation(line: 41, column: 17, scope: !4)
!25 = !DILocation(line: 46, column: 16, scope: !4)
!26 = !DILocation(line: 50, column: 14, scope: !4)
!27 = !DILocation(line: 51, column: 16, scope: !4)
!28 = !DILocation(line: 53, column: 14, scope: !4)
!29 = !DILocation(line: 63, column: 5, scope: !4)
!30 = !DILocation(line: 64, column: 5, scope: !4)
!31 = !DILocation(line: 65, column: 5, scope: !4)
!32 = !DILocation(line: 69, column: 5, scope: !4)
!33 = !DILocation(line: 70, column: 5, scope: !4)
!34 = !DILocation(line: 71, column: 5, scope: !4)
!35 = !DILocation(line: 72, column: 14, scope: !4)
!36 = !DILocation(line: 75, column: 18, scope: !4)
!37 = !DILocation(line: 77, column: 12, scope: !4)
!38 = !DILocation(line: 81, column: 14, scope: !4)
!39 = !DILocation(line: 83, column: 12, scope: !4)
!40 = !DILocation(line: 85, column: 16, scope: !4)
!41 = !DILocation(line: 88, column: 12, scope: !4)
!42 = !DILocation(line: 90, column: 12, scope: !4)
!43 = !DILocation(line: 91, column: 14, scope: !4)
!44 = !DILocation(line: 87, column: 15, scope: !4)
!45 = !DILocation(line: 93, column: 19, scope: !4)
!46 = !DILocation(line: 94, column: 17, scope: !4)
!47 = !DILocation(line: 95, column: 15, scope: !4)
!48 = !DILocation(line: 97, column: 10, scope: !4)
!49 = !DILocation(line: 98, column: 10, scope: !4)
!50 = !DILocation(line: 104, column: 16, scope: !4)
!51 = !DILocation(line: 107, column: 14, scope: !4)
!52 = !DILocation(line: 110, column: 14, scope: !4)
!53 = !DILocation(line: 111, column: 5, scope: !4)
!54 = !DILocation(line: 113, column: 11, scope: !4)
!55 = !DILocation(line: 123, column: 15, scope: !4)
!56 = !DILocation(line: 128, column: 16, scope: !4)
!57 = !DILocation(line: 130, column: 5, scope: !4)
