; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nounwind
define amdgpu_kernel void @_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0(ptr addrspace(1) inreg readonly captures(none) %0, ptr addrspace(1) inreg readonly captures(none) %1, ptr addrspace(1) inreg readonly captures(none) %2, ptr addrspace(1) inreg writeonly captures(none) %3, ptr addrspace(1) inreg writeonly captures(none) %4, i32 inreg %5, i32 inreg %6, i32 inreg %7, i32 inreg %8, i32 inreg %9, i32 inreg %10, i32 inreg %11, i32 inreg %12, i32 inreg %13, i32 inreg %14, i32 inreg %15, i32 inreg %16, i32 inreg %17, i32 inreg %18, i32 inreg %19, i32 inreg %20, i32 inreg %21, i32 inreg %22, i32 inreg %23, i32 inreg %24, i32 inreg %25, i32 inreg %26, i32 inreg %27, float inreg %28, float inreg %29, i32 inreg %30, i32 inreg %31, i32 inreg %32, i32 inreg %33, i32 inreg %34, ptr addrspace(1) inreg readnone captures(none) %35, ptr addrspace(1) inreg readnone captures(none) %36) local_unnamed_addr #0 !dbg !6 {
  %38 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !14
  %39 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !14
  %40 = lshr i32 %39, 6, !dbg !14
  %41 = and i32 %40, 3, !dbg !14
  %42 = add i32 %32, 127, !dbg !15
  %43 = sdiv i32 %42, 128, !dbg !16
  %44 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !17
  %45 = sdiv i32 %44, 32, !dbg !18
  %46 = mul i32 %45, 32, !dbg !19
  %.decomposed = sub i32 %44, %46, !dbg !19
  %.lhs.trunc = trunc nsw i32 %.decomposed to i8, !dbg !20
  %47 = sdiv i8 %.lhs.trunc, 8, !dbg !20
  %.sext = sext i8 %47 to i32, !dbg !20
  %48 = shl nsw i32 %.decomposed, 2, !dbg !25
  %49 = mul nsw i32 %.sext, -31, !dbg !25
  %50 = add nsw i32 %49, %48, !dbg !25
  %51 = srem i32 %45, %43, !dbg !26
  %52 = shl nsw i32 %43, 5, !dbg !27
  %53 = sdiv i32 %44, %52, !dbg !28
  %54 = srem i32 %53, %34, !dbg !29
  %55 = shl nsw i32 %51, 7, !dbg !30
  %56 = shl nuw nsw i32 %41, 6, !dbg !31
  %57 = and i32 %38, 15, !dbg !31
  %58 = shl nuw nsw i32 %38, 1, !dbg !32
  %59 = and i32 %58, 14, !dbg !32
  %60 = shl nuw nsw i32 %38, 3, !dbg !32
  %61 = and i32 %60, 8, !dbg !32
  %62 = or disjoint i32 %59, 16, !dbg !33
  %63 = or disjoint i32 %61, 16, !dbg !33
  %64 = zext i32 %8 to i64, !dbg !34
  %65 = zext i32 %9 to i64, !dbg !35
  %66 = sext i32 %10 to i64, !dbg !36
  %67 = zext i32 %11 to i64, !dbg !37
  %68 = zext i32 %12 to i64, !dbg !38
  %69 = sext i32 %13 to i64, !dbg !39
  %70 = zext i32 %26 to i64, !dbg !40
  %71 = zext i32 %27 to i64, !dbg !41
  %72 = add i32 %33, 31, !dbg !42
  %73 = sdiv i32 %72, 32, !dbg !44
  %.lhs.trunc87 = trunc nsw i32 %50 to i16, !dbg !45
  %74 = sdiv i16 %.lhs.trunc87, 8, !dbg !45
  %75 = zext i32 %50 to i64, !dbg !46
  %76 = mul i32 %6, %50, !dbg !46
  %77 = sext i16 %74 to i64, !dbg !47
  %78 = mul nsw i64 %65, %77, !dbg !47
  %79 = mul nsw i64 %68, %77, !dbg !48
  %80 = zext i32 %54 to i64, !dbg !49
  %81 = mul i32 %54, %5, !dbg !49
  %82 = add i32 %81, %76, !dbg !49
  %83 = zext nneg i32 %59 to i64, !dbg !50
  %84 = mul nuw i64 %80, %64, !dbg !51
  %85 = add i64 %84, %78, !dbg !51
  %86 = and i32 %38, 32, !dbg !52
  %87 = and i32 %38, 63, !dbg !31
  %88 = or disjoint i32 %56, %87, !dbg !31
  %89 = lshr i32 %88, 1, !dbg !31
  %90 = and i32 %88, 127, !dbg !31
  %91 = or disjoint i32 %89, %55, !dbg !30
  %92 = or disjoint i32 %90, %55, !dbg !30
  %93 = lshr i32 %88, 3, !dbg !53
  %94 = mul i32 %91, %7, !dbg !54
  %95 = add i32 %94, %82, !dbg !54
  %96 = add i32 %61, %95, !dbg !54
  %97 = add i32 %63, %95, !dbg !55
  %98 = lshr exact i32 %86, 3, !dbg !52
  %99 = zext nneg i32 %93 to i64, !dbg !52
  %100 = mul nsw i64 %99, %66, !dbg !56
  %101 = add nsw i64 %100, %83, !dbg !56
  %102 = zext nneg i32 %62 to i64, !dbg !57
  %103 = add nsw i64 %100, %102, !dbg !58
  %104 = mul nuw i64 %80, %67, !dbg !59
  %105 = add i64 %104, %79, !dbg !59
  %106 = mul nsw i64 %99, %69, !dbg !60
  %107 = add nsw i64 %106, %83, !dbg !60
  %108 = icmp slt i32 %91, %32, !dbg !61
  %109 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %0, i16 0, i64 2147483646, i32 159744), !dbg !62
  %110 = shl i32 %97, 1, !dbg !62
  %111 = select i1 %108, i32 %110, i32 -2147483648, !dbg !62
  %112 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %109, i32 %111, i32 0, i32 3), !dbg !62
  %.extract = extractelement <4 x i32> %112, i64 0, !dbg !62
  %.extract6 = extractelement <4 x i32> %112, i64 1, !dbg !62
  %.extract7 = extractelement <4 x i32> %112, i64 2, !dbg !62
  %.extract8 = extractelement <4 x i32> %112, i64 3, !dbg !62
  %113 = and i32 %58, 62, !dbg !62
  %114 = lshr i32 %38, 5, !dbg !62
  %115 = and i32 %114, 1, !dbg !62
  %116 = or disjoint i32 %113, %115, !dbg !62
  %117 = shl nuw nsw i32 %116, 2, !dbg !62
  %118 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %117, i32 %.extract), !dbg !62
  %119 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %117, i32 %.extract6), !dbg !62
  %120 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %117, i32 %.extract7), !dbg !62
  %121 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %117, i32 %.extract8), !dbg !62
  %122 = bitcast i32 %118 to <2 x bfloat>, !dbg !62
  %123 = bitcast i32 %119 to <2 x bfloat>, !dbg !62
  %124 = bitcast i32 %120 to <2 x bfloat>, !dbg !62
  %125 = bitcast i32 %121 to <2 x bfloat>, !dbg !62
  %126 = shl i32 %96, 1, !dbg !63
  %127 = select i1 %108, i32 %126, i32 -2147483648, !dbg !63
  %128 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %109, i32 %127, i32 0, i32 3), !dbg !63
  %.extract9 = extractelement <4 x i32> %128, i64 0, !dbg !63
  %.extract10 = extractelement <4 x i32> %128, i64 1, !dbg !63
  %.extract11 = extractelement <4 x i32> %128, i64 2, !dbg !63
  %.extract12 = extractelement <4 x i32> %128, i64 3, !dbg !63
  %129 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %117, i32 %.extract9), !dbg !63
  %130 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %117, i32 %.extract10), !dbg !63
  %131 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %117, i32 %.extract11), !dbg !63
  %132 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %117, i32 %.extract12), !dbg !63
  %133 = bitcast i32 %129 to <2 x bfloat>, !dbg !63
  %134 = bitcast i32 %130 to <2 x bfloat>, !dbg !63
  %135 = bitcast i32 %131 to <2 x bfloat>, !dbg !63
  %136 = bitcast i32 %132 to <2 x bfloat>, !dbg !63
  %137 = icmp slt i32 %33, 32, !dbg !64
  %138 = and i32 %33, 31, !dbg !65
  %139 = icmp ne i32 %138, 0, !dbg !65
  %140 = or i1 %137, %139, !dbg !65
  %141 = zext i1 %140 to i32, !dbg !66
  %142 = tail call i32 @llvm.smin.i32(i32 %141, i32 %73), !dbg !66
  %143 = sub nsw i32 %73, %142, !dbg !67
  %144 = shl nsw i32 %73, 5, !dbg !68
  %145 = icmp sgt i32 %143, 0, !dbg !69
  br i1 %145, label %146, label %1033, !dbg !70

146:                                              ; preds = %37
  %147 = add i64 %107, %105, !dbg !60
  %148 = trunc i64 %147 to i32, !dbg !60
  %149 = add i64 %103, %85, !dbg !58
  %150 = trunc i64 %149 to i32, !dbg !58
  %151 = add i64 %101, %85, !dbg !56
  %152 = trunc i64 %151 to i32, !dbg !56
  %153 = shl nuw i32 %143, 5, !dbg !71
  %154 = fmul float %28, 0x3FF7154760000000, !dbg !72
  %155 = shl nsw i64 %66, 5, !dbg !74
  %156 = shl nsw i64 %69, 5, !dbg !75
  %157 = icmp sgt i32 %153, 0, !dbg !76
  %158 = shl nuw nsw i32 %88, 1, !dbg !77
  %159 = and i32 %158, 382, !dbg !77
  %160 = and i32 %56, 64, !dbg !77
  %161 = icmp eq i32 %160, 0, !dbg !77
  %162 = select i1 %161, i32 0, i32 136, !dbg !77
  %163 = xor i32 %159, %162, !dbg !77
  %164 = sub nsw i32 %163, %158, !dbg !77
  %165 = ashr exact i32 %164, 1, !dbg !77
  %166 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %1, i16 0, i64 2147483646, i32 159744), !dbg !77
  %167 = shl nuw nsw i32 %41, 8, !dbg !77
  %168 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 14400), i32 %167, !dbg !77
  %169 = add nsw i32 %165, %87, !dbg !77
  %170 = shl nsw i32 %169, 2, !dbg !77
  %171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %170, i32 %152), !dbg !77
  %172 = tail call i64 @llvm.amdgcn.ballot.i64(i1 %157), !dbg !77
  %173 = zext i32 %169 to i64, !dbg !77
  %174 = lshr i64 %172, %173, !dbg !77
  %175 = trunc i64 %174 to i1, !dbg !77
  %176 = shl i32 %171, 1, !dbg !77
  %177 = select i1 %175, i32 %176, i32 -2147483648, !dbg !77
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %166, ptr addrspace(3) %168, i32 4, i32 %177, i32 0, i32 0, i32 0), !dbg !77, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !77
  %178 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 11328), i32 %167, !dbg !82
  %179 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %170, i32 %150), !dbg !82
  %180 = shl i32 %179, 1, !dbg !82
  %181 = select i1 %175, i32 %180, i32 -2147483648, !dbg !82
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %166, ptr addrspace(3) %178, i32 4, i32 %181, i32 0, i32 0, i32 0), !dbg !82, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !82
  %182 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 2147483646, i32 159744), !dbg !84
  %183 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %167, !dbg !84
  %184 = shl i32 %148, 1, !dbg !84
  %185 = select i1 %157, i32 %184, i32 -2147483648, !dbg !84
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %182, ptr addrspace(3) %183, i32 4, i32 %185, i32 0, i32 0, i32 0), !dbg !84, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !84
  %186 = icmp sgt i32 %153, 32, !dbg !76
  %187 = add i64 %151, %155, !dbg !86
  %188 = trunc i64 %187 to i32, !dbg !86
  %189 = add i64 %149, %155, !dbg !87
  %190 = trunc i64 %189 to i32, !dbg !87
  %191 = add i64 %147, %156, !dbg !88
  %192 = trunc i64 %191 to i32, !dbg !88
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !77
  tail call void @llvm.amdgcn.s.barrier(), !dbg !77
  %193 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 15424), i32 %167, !dbg !77
  %194 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %170, i32 %188), !dbg !77
  %195 = tail call i64 @llvm.amdgcn.ballot.i64(i1 %186), !dbg !77
  %196 = lshr i64 %195, %173, !dbg !77
  %197 = trunc i64 %196 to i1, !dbg !77
  %198 = shl i32 %194, 1, !dbg !77
  %199 = select i1 %197, i32 %198, i32 -2147483648, !dbg !77
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %166, ptr addrspace(3) %193, i32 4, i32 %199, i32 0, i32 0, i32 0), !dbg !77, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !77
  %200 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12352), i32 %167, !dbg !82
  %201 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %170, i32 %190), !dbg !82
  %202 = shl i32 %201, 1, !dbg !82
  %203 = select i1 %197, i32 %202, i32 -2147483648, !dbg !82
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %166, ptr addrspace(3) %200, i32 4, i32 %203, i32 0, i32 0, i32 0), !dbg !82, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !82
  %204 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 9248), i32 %167, !dbg !84
  %205 = shl i32 %192, 1, !dbg !84
  %206 = select i1 %186, i32 %205, i32 -2147483648, !dbg !84
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %182, ptr addrspace(3) %204, i32 4, i32 %206, i32 0, i32 0, i32 0), !dbg !84, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !84
  tail call void @llvm.amdgcn.wait.asyncmark(i16 3), !dbg !77
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !77
  tail call void @llvm.amdgcn.s.barrier(), !dbg !77
  %207 = add i32 %153, -64, !dbg !76
  %208 = icmp sgt i32 %207, 0, !dbg !76
  %209 = shl nuw nsw i32 %38, 5
  %210 = and i32 %209, 736
  %211 = and i32 %38, 8
  br i1 %208, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !76

.._crit_edge_crit_edge:                           ; preds = %146
  %.pre140 = lshr exact i32 %86, 1, !dbg !77
  %.pre142 = shl nuw nsw i32 %87, 3, !dbg !84
  %212 = insertelement <2 x i32> poison, i32 %.pre140, i64 0, !dbg !84
  %213 = insertelement <2 x i32> %212, i32 %.pre142, i64 1, !dbg !84
  br label %._crit_edge, !dbg !76

.lr.ph:                                           ; preds = %146
  %214 = icmp eq i32 %211, 0
  %215 = select i1 %214, i32 0, i32 272
  %216 = insertelement <2 x i32> poison, i32 %86, i64 0
  %217 = insertelement <2 x i32> %216, i32 %87, i64 1
  %218 = shl nuw nsw <2 x i32> %217, <i32 1, i32 3>
  %219 = lshr exact <2 x i32> %217, <i32 1, i32 3>
  %220 = shufflevector <2 x i32> %219, <2 x i32> %218, <2 x i32> <i32 0, i32 3>
  %221 = extractelement <2 x i32> %219, i64 0
  %222 = xor i32 %215, %221
  %223 = or disjoint i32 %222, %210
  %224 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %225 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %226 = shufflevector <8 x bfloat> %224, <8 x bfloat> %225, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %227 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %228 = shufflevector <8 x bfloat> %226, <8 x bfloat> %227, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %229 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %230 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %231 = shufflevector <8 x bfloat> %229, <8 x bfloat> %230, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %232 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %233 = shufflevector <8 x bfloat> %231, <8 x bfloat> %232, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %234 = and i32 %38, 31
  %235 = shl nuw nsw i32 %234, 3
  %236 = shl nuw nsw i32 %160, 2
  %237 = shl nuw nsw i32 %40, 1
  %238 = and i32 %237, 4
  %239 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %235
  %240 = getelementptr inbounds nuw i8, ptr addrspace(3) %239, i32 %236
  %241 = getelementptr inbounds nuw i8, ptr addrspace(3) %240, i32 %238
  %242 = shl nuw nsw i32 %57, 3
  %243 = shl nuw nsw i32 %41, 7
  %244 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %242
  %245 = getelementptr inbounds nuw i8, ptr addrspace(3) %244, i32 %243
  %246 = shl nuw nsw i32 %234, 1
  %247 = icmp eq i32 %86, 0
  %248 = select i1 %247, i32 0, i32 1056
  %249 = or disjoint i32 %56, %246
  %250 = xor i32 %249, %248
  %251 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %250
  %252 = getelementptr inbounds nuw i8, ptr addrspace(3) %251, i32 4096
  %253 = xor i32 %250, 264
  %254 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %253
  %255 = getelementptr inbounds nuw i8, ptr addrspace(3) %254, i32 4096
  %256 = xor i32 %250, 528
  %257 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %256
  %258 = getelementptr inbounds nuw i8, ptr addrspace(3) %257, i32 4096
  %259 = xor i32 %250, 792
  %260 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %259
  %261 = getelementptr inbounds nuw i8, ptr addrspace(3) %260, i32 4096
  %262 = xor i32 %250, 2112
  %263 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %262
  %264 = getelementptr inbounds nuw i8, ptr addrspace(3) %263, i32 4096
  %265 = xor i32 %250, 2376
  %266 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %265
  %267 = getelementptr inbounds nuw i8, ptr addrspace(3) %266, i32 4096
  %268 = xor i32 %250, 2640
  %269 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %268
  %270 = getelementptr inbounds nuw i8, ptr addrspace(3) %269, i32 4096
  %271 = xor i32 %250, 2904
  %272 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %271
  %273 = getelementptr inbounds nuw i8, ptr addrspace(3) %272, i32 4096
  %274 = and i32 %38, 60
  %275 = shl nuw nsw i32 %274, 6
  %276 = and i32 %60, 24
  %277 = shl nuw nsw i32 %274, 1
  %278 = shl nuw nsw i32 %41, 5
  %279 = or disjoint i32 %275, %276
  %280 = xor i32 %279, %277
  %281 = xor i32 %280, %278
  %282 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %281
  %283 = getelementptr inbounds nuw i8, ptr addrspace(3) %282, i32 4096
  %284 = getelementptr inbounds nuw i8, ptr addrspace(3) %282, i32 128
  %285 = getelementptr inbounds nuw i8, ptr addrspace(3) %282, i32 4224
  %286 = extractelement <2 x i32> %218, i64 1, !dbg !84
  %287 = insertelement <8 x float> poison, float %154, i64 0, !dbg !89
  %288 = shufflevector <8 x float> %287, <8 x float> poison, <8 x i32> zeroinitializer, !dbg !89
  %289 = insertelement <4 x float> poison, float %154, i64 0, !dbg !89
  %290 = shufflevector <4 x float> %289, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !89
  %291 = insertelement <2 x float> poison, float %154, i64 0, !dbg !89
  %292 = shufflevector <2 x float> %291, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !89
  br label %293, !dbg !76

293:                                              ; preds = %.lr.ph, %293
  %294 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 9248), %.lr.ph ], [ %341, %293 ]
  %295 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), %.lr.ph ], [ %294, %293 ]
  %296 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 12352), %.lr.ph ], [ %331, %293 ]
  %297 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 11328), %.lr.ph ], [ %296, %293 ]
  %298 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 15424), %.lr.ph ], [ %321, %293 ]
  %299 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 14400), %.lr.ph ], [ %298, %293 ]
  %300 = phi i32 [ 1, %.lr.ph ], [ %319, %293 ]
  %.pn28104 = phi i64 [ %191, %.lr.ph ], [ %315, %293 ]
  %.pn23102 = phi i64 [ %189, %.lr.ph ], [ %312, %293 ]
  %.pn18100 = phi i64 [ %187, %.lr.ph ], [ %309, %293 ]
  %301 = phi float [ 0xFFF0000000000000, %.lr.ph ], [ %402, %293 ]
  %302 = phi float [ 1.000000e+00, %.lr.ph ], [ %479, %293 ]
  %303 = phi i32 [ 0, %.lr.ph ], [ %527, %293 ]
  %304 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %530, %293 ]
  %305 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %529, %293 ]
  %306 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %532, %293 ]
  %307 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %531, %293 ]
  %sext81 = shl i64 %.pn18100, 32, !dbg !86
  %308 = ashr exact i64 %sext81, 32, !dbg !86
  %309 = add nsw i64 %308, %155, !dbg !86
  %310 = trunc i64 %309 to i32, !dbg !86
  %sext83 = shl i64 %.pn23102, 32, !dbg !87
  %311 = ashr exact i64 %sext83, 32, !dbg !87
  %312 = add nsw i64 %311, %155, !dbg !87
  %313 = trunc i64 %312 to i32, !dbg !87
  %sext85 = shl i64 %.pn28104, 32, !dbg !88
  %314 = ashr exact i64 %sext85, 32, !dbg !88
  %315 = add nsw i64 %314, %156, !dbg !88
  %316 = trunc i64 %315 to i32, !dbg !88
  %317 = add i32 %300, 1, !dbg !76
  %318 = icmp slt i32 %317, 3, !dbg !76
  %319 = select i1 %318, i32 %317, i32 0, !dbg !76
  %320 = shl i32 %319, 9, !dbg !77
  %321 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 14400), i32 %320, !dbg !77
  %322 = getelementptr inbounds nuw i8, ptr addrspace(3) %321, i32 %167, !dbg !77
  %323 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %170, i32 %310), !dbg !77
  %324 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true), !dbg !77
  %325 = lshr i64 %324, %173, !dbg !77
  %326 = trunc i64 %325 to i1, !dbg !77
  %327 = shl i32 %323, 1, !dbg !77
  %328 = select i1 %326, i32 %327, i32 -2147483648, !dbg !77
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %166, ptr addrspace(3) %322, i32 4, i32 %328, i32 0, i32 0, i32 0), !dbg !77, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !77
  %329 = getelementptr inbounds nuw i8, ptr addrspace(3) %299, i32 %223, !dbg !77
  %330 = load <8 x bfloat>, ptr addrspace(3) %329, align 16, !dbg !77, !alias.scope !90, !noalias !79
  %331 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 11328), i32 %320, !dbg !82
  %332 = getelementptr inbounds nuw i8, ptr addrspace(3) %331, i32 %167, !dbg !82
  %333 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %170, i32 %313), !dbg !82
  %334 = shl i32 %333, 1, !dbg !82
  %335 = select i1 %326, i32 %334, i32 -2147483648, !dbg !82
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %166, ptr addrspace(3) %332, i32 4, i32 %335, i32 0, i32 0, i32 0), !dbg !82, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !82
  %336 = getelementptr inbounds nuw i8, ptr addrspace(3) %297, i32 %223, !dbg !82
  %337 = load <8 x bfloat>, ptr addrspace(3) %336, align 16, !dbg !82, !alias.scope !90, !noalias !79
  %338 = shl i32 %319, 4, !dbg !84
  %339 = and i32 %338, 134217712, !dbg !84
  %340 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %320, !dbg !84
  %341 = getelementptr [2 x i8], ptr addrspace(3) %340, i32 %339, !dbg !84
  %342 = getelementptr inbounds nuw i8, ptr addrspace(3) %341, i32 %167, !dbg !84
  %343 = shl i32 %316, 1, !dbg !84
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %182, ptr addrspace(3) %342, i32 4, i32 %343, i32 0, i32 0, i32 0), !dbg !84, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !84
  %344 = getelementptr inbounds nuw i8, ptr addrspace(3) %295, i32 %286, !dbg !84
  %345 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %344), !dbg !84, !alias.scope !90, !noalias !79
  %346 = getelementptr inbounds nuw i8, ptr addrspace(3) %344, i32 512, !dbg !84
  %347 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %346), !dbg !84, !alias.scope !90, !noalias !79
  %348 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %337, <8 x bfloat> %228, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !92
  %349 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %330, <8 x bfloat> %233, <16 x float> %348, i32 0, i32 0, i32 0), !dbg !93
  %350 = extractelement <16 x float> %349, i64 8, !dbg !93
  %351 = extractelement <16 x float> %349, i64 15, !dbg !93
  %352 = shufflevector <16 x float> %349, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !89
  %353 = fmul <8 x float> %288, %352, !dbg !89
  %354 = fmul float %154, %350, !dbg !89
  %355 = shufflevector <16 x float> %349, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !89
  %356 = fmul <4 x float> %290, %355, !dbg !89
  %357 = fmul float %154, %351, !dbg !89
  %358 = extractelement <8 x float> %353, i64 0, !dbg !94
  %359 = extractelement <8 x float> %353, i64 1, !dbg !94
  %360 = tail call float @llvm.maxnum.f32(float %358, float %359), !dbg !95
  %361 = extractelement <8 x float> %353, i64 2, !dbg !94
  %362 = tail call float @llvm.maxnum.f32(float %360, float %361), !dbg !95
  %363 = extractelement <8 x float> %353, i64 3, !dbg !94
  %364 = extractelement <8 x float> %353, i64 4, !dbg !94
  %365 = tail call float @llvm.maxnum.f32(float %363, float %364), !dbg !95
  %366 = extractelement <8 x float> %353, i64 5, !dbg !94
  %367 = tail call float @llvm.maxnum.f32(float %365, float %366), !dbg !95
  %368 = extractelement <8 x float> %353, i64 6, !dbg !94
  %369 = extractelement <8 x float> %353, i64 7, !dbg !94
  %370 = tail call float @llvm.maxnum.f32(float %368, float %369), !dbg !95
  %371 = tail call float @llvm.maxnum.f32(float %370, float %354), !dbg !95
  %372 = extractelement <4 x float> %356, i64 0, !dbg !94
  %373 = extractelement <4 x float> %356, i64 1, !dbg !94
  %374 = tail call float @llvm.maxnum.f32(float %372, float %373), !dbg !95
  %375 = extractelement <4 x float> %356, i64 2, !dbg !94
  %376 = tail call float @llvm.maxnum.f32(float %374, float %375), !dbg !95
  %377 = extractelement <4 x float> %356, i64 3, !dbg !94
  %378 = tail call float @llvm.maxnum.f32(float %362, float %367), !dbg !95
  %379 = tail call float @llvm.maxnum.f32(float %378, float %371), !dbg !95
  %380 = shufflevector <8 x float> %353, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %381 = shufflevector <8 x float> %353, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %382 = shufflevector <8 x float> %353, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !94
  %383 = shufflevector <8 x float> %353, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !94
  %384 = shufflevector <4 x float> %356, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %385 = shufflevector <4 x float> %356, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %386 = shufflevector <16 x float> %349, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !89
  %387 = fmul <2 x float> %292, %386, !dbg !89
  %388 = extractelement <2 x float> %387, i64 0, !dbg !95
  %389 = tail call float @llvm.maxnum.f32(float %377, float %388), !dbg !95
  %390 = extractelement <2 x float> %387, i64 1, !dbg !95
  %391 = tail call float @llvm.maxnum.f32(float %389, float %390), !dbg !95
  %392 = tail call float @llvm.maxnum.f32(float %376, float %391), !dbg !95
  %393 = tail call float @llvm.maxnum.f32(float %392, float %357), !dbg !95
  %394 = tail call float @llvm.maxnum.f32(float %379, float %393), !dbg !95
  %395 = bitcast float %394 to i32, !dbg !98
  %396 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %395, i32 %395, i1 false, i1 false), !dbg !98
  %397 = extractvalue { i32, i32 } %396, 0, !dbg !98
  %398 = extractvalue { i32, i32 } %396, 1, !dbg !98
  %399 = bitcast i32 %397 to float, !dbg !98
  %400 = bitcast i32 %398 to float, !dbg !98
  %401 = tail call float @llvm.maxnum.f32(float %399, float %400), !dbg !95
  %402 = tail call float @llvm.maxnum.f32(float %301, float %401), !dbg !100
  %403 = insertelement <2 x float> poison, float %402, i64 0, !dbg !94
  %404 = shufflevector <2 x float> %403, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !94
  %405 = fsub <2 x float> %380, %404, !dbg !94
  %406 = fsub <2 x float> %381, %404, !dbg !94
  %407 = fsub <2 x float> %382, %404, !dbg !94
  %408 = fsub <2 x float> %383, %404, !dbg !94
  %409 = fsub float %354, %402, !dbg !94
  %410 = fsub <2 x float> %384, %404, !dbg !94
  %411 = fsub <2 x float> %385, %404, !dbg !94
  %412 = fsub <2 x float> %387, %404, !dbg !94
  %413 = fsub float %357, %402, !dbg !94
  %414 = extractelement <2 x float> %405, i64 0, !dbg !101
  %415 = tail call float @llvm.amdgcn.exp2.f32(float %414), !dbg !101
  %416 = extractelement <2 x float> %405, i64 1, !dbg !101
  %417 = tail call float @llvm.amdgcn.exp2.f32(float %416), !dbg !101
  %418 = extractelement <2 x float> %406, i64 0, !dbg !101
  %419 = tail call float @llvm.amdgcn.exp2.f32(float %418), !dbg !101
  %420 = extractelement <2 x float> %406, i64 1, !dbg !101
  %421 = tail call float @llvm.amdgcn.exp2.f32(float %420), !dbg !101
  %422 = extractelement <2 x float> %407, i64 0, !dbg !101
  %423 = tail call float @llvm.amdgcn.exp2.f32(float %422), !dbg !101
  %424 = extractelement <2 x float> %407, i64 1, !dbg !101
  %425 = tail call float @llvm.amdgcn.exp2.f32(float %424), !dbg !101
  %426 = extractelement <2 x float> %408, i64 0, !dbg !101
  %427 = tail call float @llvm.amdgcn.exp2.f32(float %426), !dbg !101
  %428 = extractelement <2 x float> %408, i64 1, !dbg !101
  %429 = tail call float @llvm.amdgcn.exp2.f32(float %428), !dbg !101
  %430 = tail call float @llvm.amdgcn.exp2.f32(float %409), !dbg !101
  %431 = extractelement <2 x float> %410, i64 0, !dbg !101
  %432 = tail call float @llvm.amdgcn.exp2.f32(float %431), !dbg !101
  %433 = extractelement <2 x float> %410, i64 1, !dbg !101
  %434 = tail call float @llvm.amdgcn.exp2.f32(float %433), !dbg !101
  %435 = extractelement <2 x float> %411, i64 0, !dbg !101
  %436 = tail call float @llvm.amdgcn.exp2.f32(float %435), !dbg !101
  %437 = extractelement <2 x float> %411, i64 1, !dbg !101
  %438 = tail call float @llvm.amdgcn.exp2.f32(float %437), !dbg !101
  %439 = extractelement <2 x float> %412, i64 0, !dbg !101
  %440 = tail call float @llvm.amdgcn.exp2.f32(float %439), !dbg !101
  %441 = extractelement <2 x float> %412, i64 1, !dbg !101
  %442 = tail call float @llvm.amdgcn.exp2.f32(float %441), !dbg !101
  %443 = tail call float @llvm.amdgcn.exp2.f32(float %413), !dbg !101
  %444 = insertelement <2 x float> poison, float %415, i64 0, !dbg !102
  %445 = insertelement <2 x float> %444, float %417, i64 1, !dbg !102
  %446 = insertelement <2 x float> poison, float %419, i64 0, !dbg !102
  %447 = insertelement <2 x float> %446, float %421, i64 1, !dbg !102
  %448 = insertelement <2 x float> poison, float %423, i64 0, !dbg !102
  %449 = insertelement <2 x float> %448, float %425, i64 1, !dbg !102
  %450 = insertelement <2 x float> poison, float %427, i64 0, !dbg !102
  %451 = insertelement <2 x float> %450, float %429, i64 1, !dbg !102
  %452 = insertelement <2 x float> poison, float %430, i64 0, !dbg !102
  %453 = insertelement <2 x float> %452, float %432, i64 1, !dbg !102
  %454 = insertelement <2 x float> poison, float %434, i64 0, !dbg !102
  %455 = insertelement <2 x float> %454, float %436, i64 1, !dbg !102
  %456 = insertelement <2 x float> poison, float %438, i64 0, !dbg !102
  %457 = insertelement <2 x float> %456, float %440, i64 1, !dbg !102
  %458 = insertelement <2 x float> poison, float %442, i64 0, !dbg !102
  %459 = insertelement <2 x float> %458, float %443, i64 1, !dbg !102
  %460 = fadd <2 x float> %445, %447, !dbg !102
  %461 = fadd <2 x float> %449, %451, !dbg !102
  %462 = fadd <2 x float> %453, %455, !dbg !102
  %463 = fadd <2 x float> %457, %459, !dbg !102
  %464 = fadd <2 x float> %460, %461, !dbg !102
  %465 = fadd <2 x float> %462, %463, !dbg !102
  %466 = fadd <2 x float> %464, %465, !dbg !102
  %shift = shufflevector <2 x float> %466, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !104
  %foldExtExtBinop = fadd <2 x float> %466, %shift, !dbg !104
  %bc = bitcast <2 x float> %foldExtExtBinop to <2 x i32>, !dbg !102
  %467 = extractelement <2 x i32> %bc, i64 0, !dbg !102
  %468 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %467, i32 %467, i1 false, i1 false), !dbg !102
  %469 = extractvalue { i32, i32 } %468, 0, !dbg !102
  %470 = extractvalue { i32, i32 } %468, 1, !dbg !102
  %471 = bitcast i32 %469 to float, !dbg !102
  %472 = bitcast i32 %470 to float, !dbg !102
  %473 = fadd float %471, %472, !dbg !104
  %474 = fsub float %301, %402, !dbg !105
  %475 = tail call float @llvm.amdgcn.exp2.f32(float %474), !dbg !106
  %476 = insertelement <1 x float> poison, float %475, i64 0, !dbg !107
  store <1 x float> %476, ptr addrspace(3) %241, align 4, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !107
  tail call void @llvm.amdgcn.s.barrier(), !dbg !107
  %477 = load <2 x float>, ptr addrspace(3) %245, align 8, !dbg !107
  %478 = fmul float %302, %475, !dbg !108
  %479 = fadd float %473, %478, !dbg !108
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %480 = shufflevector <2 x float> %444, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %481 = fptrunc <1 x float> %480 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %481, ptr addrspace(3) %251, align 2, !dbg !109
  %482 = shufflevector <2 x float> %452, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %483 = fptrunc <1 x float> %482 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %483, ptr addrspace(3) %252, align 2, !dbg !109
  %484 = shufflevector <2 x float> %445, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %485 = fptrunc <1 x float> %484 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %485, ptr addrspace(3) %254, align 2, !dbg !109
  %486 = shufflevector <2 x float> %453, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %487 = fptrunc <1 x float> %486 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %487, ptr addrspace(3) %255, align 2, !dbg !109
  %488 = shufflevector <2 x float> %446, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %489 = fptrunc <1 x float> %488 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %489, ptr addrspace(3) %257, align 2, !dbg !109
  %490 = shufflevector <2 x float> %454, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %491 = fptrunc <1 x float> %490 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %491, ptr addrspace(3) %258, align 2, !dbg !109
  %492 = shufflevector <2 x float> %447, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %493 = fptrunc <1 x float> %492 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %493, ptr addrspace(3) %260, align 2, !dbg !109
  %494 = shufflevector <2 x float> %455, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %495 = fptrunc <1 x float> %494 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %495, ptr addrspace(3) %261, align 2, !dbg !109
  %496 = shufflevector <2 x float> %448, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %497 = fptrunc <1 x float> %496 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %497, ptr addrspace(3) %263, align 2, !dbg !109
  %498 = shufflevector <2 x float> %456, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %499 = fptrunc <1 x float> %498 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %499, ptr addrspace(3) %264, align 2, !dbg !109
  %500 = shufflevector <2 x float> %449, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %501 = fptrunc <1 x float> %500 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %501, ptr addrspace(3) %266, align 2, !dbg !109
  %502 = shufflevector <2 x float> %457, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %503 = fptrunc <1 x float> %502 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %503, ptr addrspace(3) %267, align 2, !dbg !109
  %504 = shufflevector <2 x float> %450, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %505 = fptrunc <1 x float> %504 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %505, ptr addrspace(3) %269, align 2, !dbg !109
  %506 = shufflevector <2 x float> %458, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %507 = fptrunc <1 x float> %506 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %507, ptr addrspace(3) %270, align 2, !dbg !109
  %508 = shufflevector <2 x float> %451, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %509 = fptrunc <1 x float> %508 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %509, ptr addrspace(3) %272, align 2, !dbg !109
  %510 = shufflevector <2 x float> %459, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %511 = fptrunc <1 x float> %510 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %511, ptr addrspace(3) %273, align 2, !dbg !109
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %512 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %282), !dbg !109
  %513 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %283), !dbg !109
  %514 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %284), !dbg !109
  %515 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %285), !dbg !109
  %516 = shufflevector <4 x bfloat> %512, <4 x bfloat> %513, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %517 = shufflevector <4 x bfloat> %514, <4 x bfloat> %515, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %518 = shufflevector <4 x bfloat> %345, <4 x bfloat> %347, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %519 = shufflevector <2 x float> %306, <2 x float> %307, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %520 = shufflevector <2 x float> %477, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %521 = fmul <4 x float> %519, %520, !dbg !110
  %522 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %518, <8 x bfloat> %516, <4 x float> %521, i32 0, i32 0, i32 0), !dbg !110
  %523 = shufflevector <2 x float> %304, <2 x float> %305, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %524 = shufflevector <2 x float> %477, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !110
  %525 = fmul <4 x float> %523, %524, !dbg !110
  %526 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %518, <8 x bfloat> %517, <4 x float> %525, i32 0, i32 0, i32 0), !dbg !110
  tail call void @llvm.amdgcn.wait.asyncmark(i16 3), !dbg !77
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !77
  tail call void @llvm.amdgcn.s.barrier(), !dbg !77
  %527 = add nuw nsw i32 %303, 32, !dbg !76
  %528 = icmp slt i32 %527, %207, !dbg !76
  %529 = shufflevector <4 x float> %526, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %530 = shufflevector <4 x float> %526, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %531 = shufflevector <4 x float> %522, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %532 = shufflevector <4 x float> %522, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  br i1 %528, label %293, label %._crit_edge, !dbg !76

._crit_edge:                                      ; preds = %293, %.._crit_edge_crit_edge
  %.lcssa98 = phi float [ 1.000000e+00, %.._crit_edge_crit_edge ], [ %479, %293 ]
  %.lcssa97 = phi float [ 0xFFF0000000000000, %.._crit_edge_crit_edge ], [ %402, %293 ]
  %533 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 14400), %.._crit_edge_crit_edge ], [ %298, %293 ], !dbg !111
  %.lcssa95 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 15424), %.._crit_edge_crit_edge ], [ %321, %293 ], !dbg !111
  %534 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 11328), %.._crit_edge_crit_edge ], [ %296, %293 ], !dbg !111
  %.lcssa93 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 12352), %.._crit_edge_crit_edge ], [ %331, %293 ], !dbg !111
  %535 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), %.._crit_edge_crit_edge ], [ %294, %293 ], !dbg !111
  %.lcssa91 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 9248), %.._crit_edge_crit_edge ], [ %341, %293 ], !dbg !111
  %536 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %529, %293 ]
  %537 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %530, %293 ]
  %538 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %531, %293 ]
  %539 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %532, %293 ]
  %540 = phi <2 x i32> [ %213, %.._crit_edge_crit_edge ], [ %220, %293 ], !dbg !84
  %541 = or disjoint i32 %153, 31, !dbg !76
  %542 = icmp sgt i32 %541, 63, !dbg !76
  %543 = icmp eq i32 %211, 0, !dbg !77
  %544 = select i1 %543, i32 0, i32 272, !dbg !77
  %545 = extractelement <2 x i32> %540, i64 0, !dbg !77
  %546 = xor i32 %544, %545, !dbg !77
  %547 = or disjoint i32 %546, %210, !dbg !77
  %548 = extractelement <2 x i32> %540, i64 1, !dbg !84
  %549 = getelementptr inbounds nuw i8, ptr addrspace(3) %535, i32 %548, !dbg !84
  %550 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %549), !dbg !84, !alias.scope !90, !noalias !79
  %551 = or disjoint i32 %548, 512, !dbg !84
  %552 = getelementptr inbounds nuw i8, ptr addrspace(3) %535, i32 %551, !dbg !84
  %553 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %552), !dbg !84, !alias.scope !90, !noalias !79
  br i1 %157, label %554, label %576, !dbg !93

554:                                              ; preds = %._crit_edge
  %555 = getelementptr inbounds nuw i8, ptr addrspace(3) %534, i32 %547, !dbg !82
  %556 = load <8 x bfloat>, ptr addrspace(3) %555, align 16, !dbg !82, !alias.scope !90, !noalias !79
  %557 = getelementptr inbounds nuw i8, ptr addrspace(3) %533, i32 %547, !dbg !77
  %558 = load <8 x bfloat>, ptr addrspace(3) %557, align 16, !dbg !77, !alias.scope !90, !noalias !79
  %559 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %560 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %561 = shufflevector <8 x bfloat> %559, <8 x bfloat> %560, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !92
  %562 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %563 = shufflevector <8 x bfloat> %561, <8 x bfloat> %562, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !92
  %564 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %556, <8 x bfloat> %563, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !92
  %565 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %566 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %567 = shufflevector <8 x bfloat> %565, <8 x bfloat> %566, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !93
  %568 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %569 = shufflevector <8 x bfloat> %567, <8 x bfloat> %568, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !93
  %570 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %558, <8 x bfloat> %569, <16 x float> %564, i32 0, i32 0, i32 0), !dbg !93
  %571 = extractelement <16 x float> %570, i64 8, !dbg !93
  %572 = extractelement <16 x float> %570, i64 15, !dbg !93
  %573 = shufflevector <16 x float> %570, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !111
  %574 = shufflevector <16 x float> %570, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !111
  %575 = shufflevector <16 x float> %570, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !111
  br label %576, !dbg !93

576:                                              ; preds = %554, %._crit_edge
  %577 = phi float [ %571, %554 ], [ 0.000000e+00, %._crit_edge ], !dbg !111
  %578 = phi float [ %572, %554 ], [ 0.000000e+00, %._crit_edge ], !dbg !111
  %579 = phi <8 x float> [ %573, %554 ], [ zeroinitializer, %._crit_edge ], !dbg !111
  %580 = phi <4 x float> [ %574, %554 ], [ zeroinitializer, %._crit_edge ], !dbg !111
  %581 = phi <2 x float> [ %575, %554 ], [ zeroinitializer, %._crit_edge ], !dbg !111
  %582 = insertelement <8 x float> poison, float %154, i64 0, !dbg !89
  %583 = shufflevector <8 x float> %582, <8 x float> poison, <8 x i32> zeroinitializer, !dbg !89
  %584 = fmul <8 x float> %583, %579, !dbg !89
  %585 = fmul float %154, %577, !dbg !89
  %586 = insertelement <4 x float> poison, float %154, i64 0, !dbg !89
  %587 = shufflevector <4 x float> %586, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !89
  %588 = fmul <4 x float> %587, %580, !dbg !89
  %589 = insertelement <2 x float> poison, float %154, i64 0, !dbg !89
  %590 = shufflevector <2 x float> %589, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !89
  %591 = fmul <2 x float> %590, %581, !dbg !89
  %592 = fmul float %154, %578, !dbg !89
  %593 = extractelement <8 x float> %584, i64 0, !dbg !94
  %594 = extractelement <8 x float> %584, i64 1, !dbg !94
  %595 = tail call float @llvm.maxnum.f32(float %593, float %594), !dbg !95
  %596 = extractelement <8 x float> %584, i64 2, !dbg !94
  %597 = tail call float @llvm.maxnum.f32(float %595, float %596), !dbg !95
  %598 = extractelement <8 x float> %584, i64 3, !dbg !94
  %599 = extractelement <8 x float> %584, i64 4, !dbg !94
  %600 = tail call float @llvm.maxnum.f32(float %598, float %599), !dbg !95
  %601 = extractelement <8 x float> %584, i64 5, !dbg !94
  %602 = tail call float @llvm.maxnum.f32(float %600, float %601), !dbg !95
  %603 = extractelement <8 x float> %584, i64 6, !dbg !94
  %604 = extractelement <8 x float> %584, i64 7, !dbg !94
  %605 = tail call float @llvm.maxnum.f32(float %603, float %604), !dbg !95
  %606 = tail call float @llvm.maxnum.f32(float %605, float %585), !dbg !95
  %607 = extractelement <4 x float> %588, i64 0, !dbg !94
  %608 = extractelement <4 x float> %588, i64 1, !dbg !94
  %609 = tail call float @llvm.maxnum.f32(float %607, float %608), !dbg !95
  %610 = extractelement <4 x float> %588, i64 2, !dbg !94
  %611 = tail call float @llvm.maxnum.f32(float %609, float %610), !dbg !95
  %612 = extractelement <4 x float> %588, i64 3, !dbg !94
  %613 = extractelement <2 x float> %591, i64 0, !dbg !95
  %614 = tail call float @llvm.maxnum.f32(float %612, float %613), !dbg !95
  %615 = extractelement <2 x float> %591, i64 1, !dbg !95
  %616 = tail call float @llvm.maxnum.f32(float %614, float %615), !dbg !95
  %617 = tail call float @llvm.maxnum.f32(float %597, float %602), !dbg !95
  %618 = tail call float @llvm.maxnum.f32(float %617, float %606), !dbg !95
  %619 = tail call float @llvm.maxnum.f32(float %611, float %616), !dbg !95
  %620 = tail call float @llvm.maxnum.f32(float %619, float %592), !dbg !95
  %621 = tail call float @llvm.maxnum.f32(float %618, float %620), !dbg !95
  %622 = bitcast float %621 to i32, !dbg !98
  %623 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %622, i32 %622, i1 false, i1 false), !dbg !98
  %624 = extractvalue { i32, i32 } %623, 0, !dbg !98
  %625 = extractvalue { i32, i32 } %623, 1, !dbg !98
  %626 = bitcast i32 %624 to float, !dbg !98
  %627 = bitcast i32 %625 to float, !dbg !98
  %628 = tail call float @llvm.maxnum.f32(float %626, float %627), !dbg !95
  %629 = tail call float @llvm.maxnum.f32(float %.lcssa97, float %628), !dbg !100
  %630 = shufflevector <8 x float> %584, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %631 = insertelement <2 x float> poison, float %629, i64 0, !dbg !94
  %632 = shufflevector <2 x float> %631, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !94
  %633 = fsub <2 x float> %630, %632, !dbg !94
  %634 = shufflevector <8 x float> %584, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %635 = fsub <2 x float> %634, %632, !dbg !94
  %636 = shufflevector <8 x float> %584, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !94
  %637 = fsub <2 x float> %636, %632, !dbg !94
  %638 = shufflevector <8 x float> %584, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !94
  %639 = fsub <2 x float> %638, %632, !dbg !94
  %640 = fsub float %585, %629, !dbg !94
  %641 = shufflevector <4 x float> %588, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %642 = fsub <2 x float> %641, %632, !dbg !94
  %643 = shufflevector <4 x float> %588, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %644 = fsub <2 x float> %643, %632, !dbg !94
  %645 = fsub <2 x float> %591, %632, !dbg !94
  %646 = fsub float %592, %629, !dbg !94
  %647 = extractelement <2 x float> %633, i64 0, !dbg !101
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !101
  %649 = extractelement <2 x float> %633, i64 1, !dbg !101
  %650 = tail call float @llvm.amdgcn.exp2.f32(float %649), !dbg !101
  %651 = extractelement <2 x float> %635, i64 0, !dbg !101
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !101
  %653 = extractelement <2 x float> %635, i64 1, !dbg !101
  %654 = tail call float @llvm.amdgcn.exp2.f32(float %653), !dbg !101
  %655 = extractelement <2 x float> %637, i64 0, !dbg !101
  %656 = tail call float @llvm.amdgcn.exp2.f32(float %655), !dbg !101
  %657 = extractelement <2 x float> %637, i64 1, !dbg !101
  %658 = tail call float @llvm.amdgcn.exp2.f32(float %657), !dbg !101
  %659 = extractelement <2 x float> %639, i64 0, !dbg !101
  %660 = tail call float @llvm.amdgcn.exp2.f32(float %659), !dbg !101
  %661 = extractelement <2 x float> %639, i64 1, !dbg !101
  %662 = tail call float @llvm.amdgcn.exp2.f32(float %661), !dbg !101
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %640), !dbg !101
  %664 = extractelement <2 x float> %642, i64 0, !dbg !101
  %665 = tail call float @llvm.amdgcn.exp2.f32(float %664), !dbg !101
  %666 = extractelement <2 x float> %642, i64 1, !dbg !101
  %667 = tail call float @llvm.amdgcn.exp2.f32(float %666), !dbg !101
  %668 = extractelement <2 x float> %644, i64 0, !dbg !101
  %669 = tail call float @llvm.amdgcn.exp2.f32(float %668), !dbg !101
  %670 = extractelement <2 x float> %644, i64 1, !dbg !101
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %670), !dbg !101
  %672 = extractelement <2 x float> %645, i64 0, !dbg !101
  %673 = tail call float @llvm.amdgcn.exp2.f32(float %672), !dbg !101
  %674 = extractelement <2 x float> %645, i64 1, !dbg !101
  %675 = tail call float @llvm.amdgcn.exp2.f32(float %674), !dbg !101
  %676 = tail call float @llvm.amdgcn.exp2.f32(float %646), !dbg !101
  %677 = insertelement <2 x float> poison, float %648, i64 0, !dbg !102
  %678 = insertelement <2 x float> %677, float %650, i64 1, !dbg !102
  %679 = insertelement <2 x float> poison, float %652, i64 0, !dbg !102
  %680 = insertelement <2 x float> %679, float %654, i64 1, !dbg !102
  %681 = insertelement <2 x float> poison, float %656, i64 0, !dbg !102
  %682 = insertelement <2 x float> %681, float %658, i64 1, !dbg !102
  %683 = insertelement <2 x float> poison, float %660, i64 0, !dbg !102
  %684 = insertelement <2 x float> %683, float %662, i64 1, !dbg !102
  %685 = insertelement <2 x float> poison, float %663, i64 0, !dbg !102
  %686 = insertelement <2 x float> %685, float %665, i64 1, !dbg !102
  %687 = insertelement <2 x float> poison, float %667, i64 0, !dbg !102
  %688 = insertelement <2 x float> %687, float %669, i64 1, !dbg !102
  %689 = insertelement <2 x float> poison, float %671, i64 0, !dbg !102
  %690 = insertelement <2 x float> %689, float %673, i64 1, !dbg !102
  %691 = insertelement <2 x float> poison, float %675, i64 0, !dbg !102
  %692 = insertelement <2 x float> %691, float %676, i64 1, !dbg !102
  %693 = fadd <2 x float> %678, %680, !dbg !102
  %694 = fadd <2 x float> %682, %684, !dbg !102
  %695 = fadd <2 x float> %686, %688, !dbg !102
  %696 = fadd <2 x float> %690, %692, !dbg !102
  %697 = fadd <2 x float> %693, %694, !dbg !102
  %698 = fadd <2 x float> %695, %696, !dbg !102
  %699 = fadd <2 x float> %697, %698, !dbg !102
  %shift184 = shufflevector <2 x float> %699, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !104
  %foldExtExtBinop185 = fadd <2 x float> %699, %shift184, !dbg !104
  %bc193 = bitcast <2 x float> %foldExtExtBinop185 to <2 x i32>, !dbg !102
  %700 = extractelement <2 x i32> %bc193, i64 0, !dbg !102
  %701 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %700, i32 %700, i1 false, i1 false), !dbg !102
  %702 = fsub float %.lcssa97, %629, !dbg !105
  %703 = tail call float @llvm.amdgcn.exp2.f32(float %702), !dbg !106
  %704 = and i32 %38, 31, !dbg !107
  %705 = shl nuw nsw i32 %704, 3, !dbg !107
  %706 = shl nuw nsw i32 %160, 2, !dbg !107
  %707 = shl nuw nsw i32 %40, 1, !dbg !107
  %708 = and i32 %707, 4, !dbg !107
  %709 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %705, !dbg !107
  %710 = getelementptr inbounds nuw i8, ptr addrspace(3) %709, i32 %706, !dbg !107
  %711 = getelementptr inbounds nuw i8, ptr addrspace(3) %710, i32 %708, !dbg !107
  %712 = insertelement <1 x float> poison, float %703, i64 0, !dbg !107
  store <1 x float> %712, ptr addrspace(3) %711, align 4, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !107
  tail call void @llvm.amdgcn.s.barrier(), !dbg !107
  %713 = shl nuw nsw i32 %57, 3, !dbg !107
  %714 = shl nuw nsw i32 %41, 7, !dbg !107
  %715 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %713, !dbg !107
  %716 = getelementptr inbounds nuw i8, ptr addrspace(3) %715, i32 %714, !dbg !107
  %717 = load <2 x float>, ptr addrspace(3) %716, align 8, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %718 = shl nuw nsw i32 %704, 1, !dbg !109
  %719 = icmp eq i32 %86, 0, !dbg !109
  %720 = select i1 %719, i32 0, i32 1056, !dbg !109
  %721 = or disjoint i32 %56, %718, !dbg !109
  %722 = xor i32 %721, %720, !dbg !109
  %723 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %722, !dbg !109
  %724 = shufflevector <2 x float> %677, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %725 = fptrunc <1 x float> %724 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %725, ptr addrspace(3) %723, align 2, !dbg !109
  %726 = getelementptr inbounds nuw i8, ptr addrspace(3) %723, i32 4096, !dbg !109
  %727 = shufflevector <2 x float> %685, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %728 = fptrunc <1 x float> %727 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %728, ptr addrspace(3) %726, align 2, !dbg !109
  %729 = xor i32 %722, 264, !dbg !109
  %730 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %729, !dbg !109
  %731 = shufflevector <2 x float> %678, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %732 = fptrunc <1 x float> %731 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %732, ptr addrspace(3) %730, align 2, !dbg !109
  %733 = getelementptr inbounds nuw i8, ptr addrspace(3) %730, i32 4096, !dbg !109
  %734 = shufflevector <2 x float> %686, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %735 = fptrunc <1 x float> %734 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %735, ptr addrspace(3) %733, align 2, !dbg !109
  %736 = xor i32 %722, 528, !dbg !109
  %737 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %736, !dbg !109
  %738 = shufflevector <2 x float> %679, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %739 = fptrunc <1 x float> %738 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %739, ptr addrspace(3) %737, align 2, !dbg !109
  %740 = getelementptr inbounds nuw i8, ptr addrspace(3) %737, i32 4096, !dbg !109
  %741 = shufflevector <2 x float> %687, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %742 = fptrunc <1 x float> %741 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %742, ptr addrspace(3) %740, align 2, !dbg !109
  %743 = xor i32 %722, 792, !dbg !109
  %744 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %743, !dbg !109
  %745 = shufflevector <2 x float> %680, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %746 = fptrunc <1 x float> %745 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %746, ptr addrspace(3) %744, align 2, !dbg !109
  %747 = getelementptr inbounds nuw i8, ptr addrspace(3) %744, i32 4096, !dbg !109
  %748 = shufflevector <2 x float> %688, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %749 = fptrunc <1 x float> %748 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %749, ptr addrspace(3) %747, align 2, !dbg !109
  %750 = xor i32 %722, 2112, !dbg !109
  %751 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %750, !dbg !109
  %752 = shufflevector <2 x float> %681, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %753 = fptrunc <1 x float> %752 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %753, ptr addrspace(3) %751, align 2, !dbg !109
  %754 = getelementptr inbounds nuw i8, ptr addrspace(3) %751, i32 4096, !dbg !109
  %755 = shufflevector <2 x float> %689, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %756 = fptrunc <1 x float> %755 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %756, ptr addrspace(3) %754, align 2, !dbg !109
  %757 = xor i32 %722, 2376, !dbg !109
  %758 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %757, !dbg !109
  %759 = shufflevector <2 x float> %682, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %760 = fptrunc <1 x float> %759 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %760, ptr addrspace(3) %758, align 2, !dbg !109
  %761 = getelementptr inbounds nuw i8, ptr addrspace(3) %758, i32 4096, !dbg !109
  %762 = shufflevector <2 x float> %690, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %763 = fptrunc <1 x float> %762 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %763, ptr addrspace(3) %761, align 2, !dbg !109
  %764 = xor i32 %722, 2640, !dbg !109
  %765 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %764, !dbg !109
  %766 = shufflevector <2 x float> %683, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %767 = fptrunc <1 x float> %766 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %767, ptr addrspace(3) %765, align 2, !dbg !109
  %768 = getelementptr inbounds nuw i8, ptr addrspace(3) %765, i32 4096, !dbg !109
  %769 = shufflevector <2 x float> %691, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %770 = fptrunc <1 x float> %769 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %770, ptr addrspace(3) %768, align 2, !dbg !109
  %771 = xor i32 %722, 2904, !dbg !109
  %772 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %771, !dbg !109
  %773 = shufflevector <2 x float> %684, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %774 = fptrunc <1 x float> %773 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %774, ptr addrspace(3) %772, align 2, !dbg !109
  %775 = getelementptr inbounds nuw i8, ptr addrspace(3) %772, i32 4096, !dbg !109
  %776 = shufflevector <2 x float> %692, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %777 = fptrunc <1 x float> %776 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %777, ptr addrspace(3) %775, align 2, !dbg !109
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %778 = and i32 %38, 60, !dbg !109
  %779 = shl nuw nsw i32 %778, 6, !dbg !109
  %780 = and i32 %60, 24, !dbg !109
  %781 = shl nuw nsw i32 %778, 1, !dbg !109
  %782 = shl nuw nsw i32 %41, 5, !dbg !109
  %783 = or disjoint i32 %779, %780, !dbg !109
  %784 = xor i32 %783, %781, !dbg !109
  %785 = xor i32 %784, %782, !dbg !109
  %786 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %785, !dbg !109
  %787 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %786), !dbg !109
  %788 = getelementptr inbounds nuw i8, ptr addrspace(3) %786, i32 4096, !dbg !109
  %789 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %788), !dbg !109
  %790 = getelementptr inbounds nuw i8, ptr addrspace(3) %786, i32 128, !dbg !109
  %791 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %790), !dbg !109
  %792 = getelementptr inbounds nuw i8, ptr addrspace(3) %786, i32 4224, !dbg !109
  %793 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %792), !dbg !109
  br i1 %157, label %794, label %817, !dbg !110

794:                                              ; preds = %576
  %795 = fmul float %.lcssa98, %703, !dbg !108
  %796 = extractvalue { i32, i32 } %701, 0, !dbg !102
  %797 = bitcast i32 %796 to float, !dbg !102
  %798 = extractvalue { i32, i32 } %701, 1, !dbg !102
  %799 = bitcast i32 %798 to float, !dbg !102
  %800 = fadd float %797, %799, !dbg !104
  %801 = fadd float %800, %795, !dbg !108
  %802 = shufflevector <4 x bfloat> %787, <4 x bfloat> %789, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %803 = shufflevector <4 x bfloat> %791, <4 x bfloat> %793, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %804 = shufflevector <4 x bfloat> %550, <4 x bfloat> %553, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %805 = shufflevector <2 x float> %539, <2 x float> %538, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %806 = shufflevector <2 x float> %717, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %807 = fmul <4 x float> %805, %806, !dbg !110
  %808 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %804, <8 x bfloat> %802, <4 x float> %807, i32 0, i32 0, i32 0), !dbg !110
  %809 = shufflevector <2 x float> %537, <2 x float> %536, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %810 = shufflevector <2 x float> %717, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !110
  %811 = fmul <4 x float> %809, %810, !dbg !110
  %812 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %804, <8 x bfloat> %803, <4 x float> %811, i32 0, i32 0, i32 0), !dbg !110
  %813 = shufflevector <4 x float> %812, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %814 = shufflevector <4 x float> %812, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %815 = shufflevector <4 x float> %808, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %816 = shufflevector <4 x float> %808, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  br label %817, !dbg !110

817:                                              ; preds = %794, %576
  %818 = phi float [ %629, %794 ], [ %.lcssa97, %576 ]
  %819 = phi float [ %801, %794 ], [ %.lcssa98, %576 ]
  %820 = phi <2 x float> [ %813, %794 ], [ %536, %576 ]
  %821 = phi <2 x float> [ %814, %794 ], [ %537, %576 ]
  %822 = phi <2 x float> [ %815, %794 ], [ %538, %576 ]
  %823 = phi <2 x float> [ %816, %794 ], [ %539, %576 ]
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !77
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !77
  tail call void @llvm.amdgcn.s.barrier(), !dbg !77
  %824 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa91, i32 %548, !dbg !84
  %825 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %824), !dbg !84, !alias.scope !90, !noalias !79
  %826 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa91, i32 %551, !dbg !84
  %827 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %826), !dbg !84, !alias.scope !90, !noalias !79
  br i1 %542, label %828, label %850, !dbg !93

828:                                              ; preds = %817
  %829 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa93, i32 %547, !dbg !82
  %830 = load <8 x bfloat>, ptr addrspace(3) %829, align 16, !dbg !82, !alias.scope !90, !noalias !79
  %831 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa95, i32 %547, !dbg !77
  %832 = load <8 x bfloat>, ptr addrspace(3) %831, align 16, !dbg !77, !alias.scope !90, !noalias !79
  %833 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %834 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %835 = shufflevector <8 x bfloat> %833, <8 x bfloat> %834, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !92
  %836 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %837 = shufflevector <8 x bfloat> %835, <8 x bfloat> %836, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !92
  %838 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %830, <8 x bfloat> %837, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !92
  %839 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %840 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %841 = shufflevector <8 x bfloat> %839, <8 x bfloat> %840, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !93
  %842 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %843 = shufflevector <8 x bfloat> %841, <8 x bfloat> %842, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !93
  %844 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %832, <8 x bfloat> %843, <16 x float> %838, i32 0, i32 0, i32 0), !dbg !93
  %845 = extractelement <16 x float> %844, i64 8, !dbg !93
  %846 = extractelement <16 x float> %844, i64 15, !dbg !93
  %847 = shufflevector <16 x float> %844, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !111
  %848 = shufflevector <16 x float> %844, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !111
  %849 = shufflevector <16 x float> %844, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !111
  br label %850, !dbg !93

850:                                              ; preds = %828, %817
  %851 = phi float [ %845, %828 ], [ 0.000000e+00, %817 ], !dbg !111
  %852 = phi float [ %846, %828 ], [ 0.000000e+00, %817 ], !dbg !111
  %853 = phi <8 x float> [ %847, %828 ], [ zeroinitializer, %817 ], !dbg !111
  %854 = phi <4 x float> [ %848, %828 ], [ zeroinitializer, %817 ], !dbg !111
  %855 = phi <2 x float> [ %849, %828 ], [ zeroinitializer, %817 ], !dbg !111
  %856 = fmul <8 x float> %583, %853, !dbg !89
  %857 = fmul float %154, %851, !dbg !89
  %858 = fmul <4 x float> %587, %854, !dbg !89
  %859 = fmul <2 x float> %590, %855, !dbg !89
  %860 = fmul float %154, %852, !dbg !89
  %861 = extractelement <8 x float> %856, i64 0, !dbg !94
  %862 = extractelement <8 x float> %856, i64 1, !dbg !94
  %863 = tail call float @llvm.maxnum.f32(float %861, float %862), !dbg !95
  %864 = extractelement <8 x float> %856, i64 2, !dbg !94
  %865 = tail call float @llvm.maxnum.f32(float %863, float %864), !dbg !95
  %866 = extractelement <8 x float> %856, i64 3, !dbg !94
  %867 = extractelement <8 x float> %856, i64 4, !dbg !94
  %868 = tail call float @llvm.maxnum.f32(float %866, float %867), !dbg !95
  %869 = extractelement <8 x float> %856, i64 5, !dbg !94
  %870 = tail call float @llvm.maxnum.f32(float %868, float %869), !dbg !95
  %871 = extractelement <8 x float> %856, i64 6, !dbg !94
  %872 = extractelement <8 x float> %856, i64 7, !dbg !94
  %873 = tail call float @llvm.maxnum.f32(float %871, float %872), !dbg !95
  %874 = tail call float @llvm.maxnum.f32(float %873, float %857), !dbg !95
  %875 = extractelement <4 x float> %858, i64 0, !dbg !94
  %876 = extractelement <4 x float> %858, i64 1, !dbg !94
  %877 = tail call float @llvm.maxnum.f32(float %875, float %876), !dbg !95
  %878 = extractelement <4 x float> %858, i64 2, !dbg !94
  %879 = tail call float @llvm.maxnum.f32(float %877, float %878), !dbg !95
  %880 = extractelement <4 x float> %858, i64 3, !dbg !94
  %881 = extractelement <2 x float> %859, i64 0, !dbg !95
  %882 = tail call float @llvm.maxnum.f32(float %880, float %881), !dbg !95
  %883 = extractelement <2 x float> %859, i64 1, !dbg !95
  %884 = tail call float @llvm.maxnum.f32(float %882, float %883), !dbg !95
  %885 = tail call float @llvm.maxnum.f32(float %865, float %870), !dbg !95
  %886 = tail call float @llvm.maxnum.f32(float %885, float %874), !dbg !95
  %887 = tail call float @llvm.maxnum.f32(float %879, float %884), !dbg !95
  %888 = tail call float @llvm.maxnum.f32(float %887, float %860), !dbg !95
  %889 = tail call float @llvm.maxnum.f32(float %886, float %888), !dbg !95
  %890 = bitcast float %889 to i32, !dbg !98
  %891 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %890, i32 %890, i1 false, i1 false), !dbg !98
  %892 = extractvalue { i32, i32 } %891, 0, !dbg !98
  %893 = extractvalue { i32, i32 } %891, 1, !dbg !98
  %894 = bitcast i32 %892 to float, !dbg !98
  %895 = bitcast i32 %893 to float, !dbg !98
  %896 = tail call float @llvm.maxnum.f32(float %894, float %895), !dbg !95
  %897 = tail call float @llvm.maxnum.f32(float %818, float %896), !dbg !100
  %898 = shufflevector <8 x float> %856, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %899 = insertelement <2 x float> poison, float %897, i64 0, !dbg !94
  %900 = shufflevector <2 x float> %899, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !94
  %901 = fsub <2 x float> %898, %900, !dbg !94
  %902 = shufflevector <8 x float> %856, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %903 = fsub <2 x float> %902, %900, !dbg !94
  %904 = shufflevector <8 x float> %856, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !94
  %905 = fsub <2 x float> %904, %900, !dbg !94
  %906 = shufflevector <8 x float> %856, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !94
  %907 = fsub <2 x float> %906, %900, !dbg !94
  %908 = fsub float %857, %897, !dbg !94
  %909 = shufflevector <4 x float> %858, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %910 = fsub <2 x float> %909, %900, !dbg !94
  %911 = shufflevector <4 x float> %858, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %912 = fsub <2 x float> %911, %900, !dbg !94
  %913 = fsub <2 x float> %859, %900, !dbg !94
  %914 = fsub float %860, %897, !dbg !94
  %915 = extractelement <2 x float> %901, i64 0, !dbg !101
  %916 = tail call float @llvm.amdgcn.exp2.f32(float %915), !dbg !101
  %917 = extractelement <2 x float> %901, i64 1, !dbg !101
  %918 = tail call float @llvm.amdgcn.exp2.f32(float %917), !dbg !101
  %919 = extractelement <2 x float> %903, i64 0, !dbg !101
  %920 = tail call float @llvm.amdgcn.exp2.f32(float %919), !dbg !101
  %921 = extractelement <2 x float> %903, i64 1, !dbg !101
  %922 = tail call float @llvm.amdgcn.exp2.f32(float %921), !dbg !101
  %923 = extractelement <2 x float> %905, i64 0, !dbg !101
  %924 = tail call float @llvm.amdgcn.exp2.f32(float %923), !dbg !101
  %925 = extractelement <2 x float> %905, i64 1, !dbg !101
  %926 = tail call float @llvm.amdgcn.exp2.f32(float %925), !dbg !101
  %927 = extractelement <2 x float> %907, i64 0, !dbg !101
  %928 = tail call float @llvm.amdgcn.exp2.f32(float %927), !dbg !101
  %929 = extractelement <2 x float> %907, i64 1, !dbg !101
  %930 = tail call float @llvm.amdgcn.exp2.f32(float %929), !dbg !101
  %931 = tail call float @llvm.amdgcn.exp2.f32(float %908), !dbg !101
  %932 = extractelement <2 x float> %910, i64 0, !dbg !101
  %933 = tail call float @llvm.amdgcn.exp2.f32(float %932), !dbg !101
  %934 = extractelement <2 x float> %910, i64 1, !dbg !101
  %935 = tail call float @llvm.amdgcn.exp2.f32(float %934), !dbg !101
  %936 = extractelement <2 x float> %912, i64 0, !dbg !101
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !101
  %938 = extractelement <2 x float> %912, i64 1, !dbg !101
  %939 = tail call float @llvm.amdgcn.exp2.f32(float %938), !dbg !101
  %940 = extractelement <2 x float> %913, i64 0, !dbg !101
  %941 = tail call float @llvm.amdgcn.exp2.f32(float %940), !dbg !101
  %942 = extractelement <2 x float> %913, i64 1, !dbg !101
  %943 = tail call float @llvm.amdgcn.exp2.f32(float %942), !dbg !101
  %944 = tail call float @llvm.amdgcn.exp2.f32(float %914), !dbg !101
  %945 = insertelement <2 x float> poison, float %916, i64 0, !dbg !102
  %946 = insertelement <2 x float> %945, float %918, i64 1, !dbg !102
  %947 = insertelement <2 x float> poison, float %920, i64 0, !dbg !102
  %948 = insertelement <2 x float> %947, float %922, i64 1, !dbg !102
  %949 = insertelement <2 x float> poison, float %924, i64 0, !dbg !102
  %950 = insertelement <2 x float> %949, float %926, i64 1, !dbg !102
  %951 = insertelement <2 x float> poison, float %928, i64 0, !dbg !102
  %952 = insertelement <2 x float> %951, float %930, i64 1, !dbg !102
  %953 = insertelement <2 x float> poison, float %931, i64 0, !dbg !102
  %954 = insertelement <2 x float> %953, float %933, i64 1, !dbg !102
  %955 = insertelement <2 x float> poison, float %935, i64 0, !dbg !102
  %956 = insertelement <2 x float> %955, float %937, i64 1, !dbg !102
  %957 = insertelement <2 x float> poison, float %939, i64 0, !dbg !102
  %958 = insertelement <2 x float> %957, float %941, i64 1, !dbg !102
  %959 = insertelement <2 x float> poison, float %943, i64 0, !dbg !102
  %960 = insertelement <2 x float> %959, float %944, i64 1, !dbg !102
  %961 = fadd <2 x float> %946, %948, !dbg !102
  %962 = fadd <2 x float> %950, %952, !dbg !102
  %963 = fadd <2 x float> %954, %956, !dbg !102
  %964 = fadd <2 x float> %958, %960, !dbg !102
  %965 = fadd <2 x float> %961, %962, !dbg !102
  %966 = fadd <2 x float> %963, %964, !dbg !102
  %967 = fadd <2 x float> %965, %966, !dbg !102
  %shift187 = shufflevector <2 x float> %967, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !104
  %foldExtExtBinop188 = fadd <2 x float> %967, %shift187, !dbg !104
  %bc194 = bitcast <2 x float> %foldExtExtBinop188 to <2 x i32>, !dbg !102
  %968 = extractelement <2 x i32> %bc194, i64 0, !dbg !102
  %969 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %968, i32 %968, i1 false, i1 false), !dbg !102
  %970 = fsub float %818, %897, !dbg !105
  %971 = tail call float @llvm.amdgcn.exp2.f32(float %970), !dbg !106
  %972 = insertelement <1 x float> poison, float %971, i64 0, !dbg !107
  store <1 x float> %972, ptr addrspace(3) %711, align 4, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !107
  tail call void @llvm.amdgcn.s.barrier(), !dbg !107
  %973 = load <2 x float>, ptr addrspace(3) %716, align 8, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %974 = shufflevector <2 x float> %945, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %975 = fptrunc <1 x float> %974 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %975, ptr addrspace(3) %723, align 2, !dbg !109
  %976 = shufflevector <2 x float> %953, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %977 = fptrunc <1 x float> %976 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %977, ptr addrspace(3) %726, align 2, !dbg !109
  %978 = shufflevector <2 x float> %946, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %979 = fptrunc <1 x float> %978 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %979, ptr addrspace(3) %730, align 2, !dbg !109
  %980 = shufflevector <2 x float> %954, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %981 = fptrunc <1 x float> %980 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %981, ptr addrspace(3) %733, align 2, !dbg !109
  %982 = shufflevector <2 x float> %947, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %983 = fptrunc <1 x float> %982 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %983, ptr addrspace(3) %737, align 2, !dbg !109
  %984 = shufflevector <2 x float> %955, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %985 = fptrunc <1 x float> %984 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %985, ptr addrspace(3) %740, align 2, !dbg !109
  %986 = shufflevector <2 x float> %948, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %987 = fptrunc <1 x float> %986 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %987, ptr addrspace(3) %744, align 2, !dbg !109
  %988 = shufflevector <2 x float> %956, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %989 = fptrunc <1 x float> %988 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %989, ptr addrspace(3) %747, align 2, !dbg !109
  %990 = shufflevector <2 x float> %949, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %991 = fptrunc <1 x float> %990 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %991, ptr addrspace(3) %751, align 2, !dbg !109
  %992 = shufflevector <2 x float> %957, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %993 = fptrunc <1 x float> %992 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %993, ptr addrspace(3) %754, align 2, !dbg !109
  %994 = shufflevector <2 x float> %950, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %995 = fptrunc <1 x float> %994 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %995, ptr addrspace(3) %758, align 2, !dbg !109
  %996 = shufflevector <2 x float> %958, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %997 = fptrunc <1 x float> %996 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %997, ptr addrspace(3) %761, align 2, !dbg !109
  %998 = shufflevector <2 x float> %951, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %999 = fptrunc <1 x float> %998 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %999, ptr addrspace(3) %765, align 2, !dbg !109
  %1000 = shufflevector <2 x float> %959, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %1001 = fptrunc <1 x float> %1000 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %1001, ptr addrspace(3) %768, align 2, !dbg !109
  %1002 = shufflevector <2 x float> %952, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %1003 = fptrunc <1 x float> %1002 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %1003, ptr addrspace(3) %772, align 2, !dbg !109
  %1004 = shufflevector <2 x float> %960, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %1005 = fptrunc <1 x float> %1004 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %1005, ptr addrspace(3) %775, align 2, !dbg !109
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %1006 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %786), !dbg !109
  %1007 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %788), !dbg !109
  %1008 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %790), !dbg !109
  %1009 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %792), !dbg !109
  br i1 %542, label %1010, label %1033, !dbg !110

1010:                                             ; preds = %850
  %1011 = fmul float %819, %971, !dbg !108
  %1012 = extractvalue { i32, i32 } %969, 0, !dbg !102
  %1013 = bitcast i32 %1012 to float, !dbg !102
  %1014 = extractvalue { i32, i32 } %969, 1, !dbg !102
  %1015 = bitcast i32 %1014 to float, !dbg !102
  %1016 = fadd float %1013, %1015, !dbg !104
  %1017 = fadd float %1016, %1011, !dbg !108
  %1018 = shufflevector <4 x bfloat> %1006, <4 x bfloat> %1007, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %1019 = shufflevector <4 x bfloat> %1008, <4 x bfloat> %1009, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %1020 = shufflevector <4 x bfloat> %825, <4 x bfloat> %827, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %1021 = shufflevector <2 x float> %823, <2 x float> %822, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %1022 = shufflevector <2 x float> %973, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %1023 = fmul <4 x float> %1021, %1022, !dbg !110
  %1024 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1020, <8 x bfloat> %1018, <4 x float> %1023, i32 0, i32 0, i32 0), !dbg !110
  %1025 = shufflevector <2 x float> %821, <2 x float> %820, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %1026 = shufflevector <2 x float> %973, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !110
  %1027 = fmul <4 x float> %1025, %1026, !dbg !110
  %1028 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1020, <8 x bfloat> %1019, <4 x float> %1027, i32 0, i32 0, i32 0), !dbg !110
  %1029 = shufflevector <4 x float> %1028, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !70
  %1030 = shufflevector <4 x float> %1028, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !70
  %1031 = shufflevector <4 x float> %1024, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !70
  %1032 = shufflevector <4 x float> %1024, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !70
  br label %1033, !dbg !110

1033:                                             ; preds = %850, %1010, %37
  %1034 = phi float [ 0xFFF0000000000000, %37 ], [ %897, %1010 ], [ %818, %850 ], !dbg !70
  %1035 = phi float [ 1.000000e+00, %37 ], [ %1017, %1010 ], [ %819, %850 ], !dbg !70
  %1036 = phi i32 [ 0, %37 ], [ %153, %1010 ], [ %153, %850 ], !dbg !70
  %1037 = phi <2 x float> [ zeroinitializer, %37 ], [ %1029, %1010 ], [ %820, %850 ], !dbg !70
  %1038 = phi <2 x float> [ zeroinitializer, %37 ], [ %1030, %1010 ], [ %821, %850 ], !dbg !70
  %1039 = phi <2 x float> [ zeroinitializer, %37 ], [ %1031, %1010 ], [ %822, %850 ], !dbg !70
  %1040 = phi <2 x float> [ zeroinitializer, %37 ], [ %1032, %1010 ], [ %823, %850 ], !dbg !70
  %1041 = icmp sgt i32 %142, 0, !dbg !112
  br i1 %1041, label %1042, label %.loopexit, !dbg !113

1042:                                             ; preds = %1033
  %1043 = fmul float %28, 0x3FF7154760000000, !dbg !114
  %1044 = shl nsw i64 %66, 5, !dbg !116
  %1045 = shl nsw i64 %69, 5, !dbg !117
  %1046 = icmp slt i32 %1036, %144, !dbg !118
  br i1 %1046, label %.lr.ph121, label %.loopexit, !dbg !118

.lr.ph121:                                        ; preds = %1042
  %1047 = shl i32 %143, 5, !dbg !119
  %1048 = zext i32 %1047 to i64, !dbg !119
  %1049 = mul nsw i64 %1048, %69, !dbg !120
  %1050 = add i64 %105, %1049, !dbg !121
  %1051 = add i64 %107, %1050, !dbg !121
  %1052 = mul nsw i64 %1048, %66, !dbg !119
  %1053 = add i64 %85, %1052, !dbg !121
  %1054 = add i64 %103, %1053, !dbg !121
  %1055 = add i64 %101, %1053, !dbg !121
  %1056 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %1, i16 0, i64 2147483646, i32 159744)
  %1057 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 2147483646, i32 159744)
  %1058 = shl nuw nsw i32 %88, 2
  %1059 = and i32 %1058, 764
  %1060 = and i32 %56, 64
  %1061 = icmp eq i32 %1060, 0
  %1062 = select i1 %1061, i32 0, i32 272
  %1063 = xor i32 %1059, %1062
  %1064 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1063
  %1065 = shl nuw nsw i32 %38, 5
  %1066 = and i32 %1065, 736
  %1067 = and i32 %38, 8
  %1068 = icmp eq i32 %1067, 0
  %1069 = select i1 %1068, i32 0, i32 272
  %1070 = lshr exact i32 %86, 1
  %1071 = xor i32 %1069, %1070
  %1072 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1066
  %1073 = getelementptr inbounds nuw i8, ptr addrspace(3) %1072, i32 %1071
  %1074 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1075 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1076 = shufflevector <8 x bfloat> %1074, <8 x bfloat> %1075, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1077 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1078 = shufflevector <8 x bfloat> %1076, <8 x bfloat> %1077, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1079 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1080 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1081 = shufflevector <8 x bfloat> %1079, <8 x bfloat> %1080, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1082 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1083 = shufflevector <8 x bfloat> %1081, <8 x bfloat> %1082, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1084 = and i32 %38, 31
  %1085 = shl nuw nsw i32 %1084, 3
  %1086 = shl nuw nsw i32 %1060, 2
  %1087 = shl nuw nsw i32 %40, 1
  %1088 = and i32 %1087, 4
  %1089 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1086
  %1090 = getelementptr inbounds nuw i8, ptr addrspace(3) %1089, i32 %1088
  %1091 = getelementptr inbounds nuw i8, ptr addrspace(3) %1090, i32 %1085
  %1092 = shl nuw nsw i32 %57, 3
  %1093 = shl nuw nsw i32 %41, 7
  %1094 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1092
  %1095 = getelementptr inbounds nuw i8, ptr addrspace(3) %1094, i32 %1093
  %1096 = shl nuw nsw i32 %1084, 1
  %1097 = icmp eq i32 %86, 0
  %1098 = select i1 %1097, i32 0, i32 1056
  %1099 = or disjoint i32 %56, %1096
  %1100 = xor i32 %1099, %1098
  %1101 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1100
  %1102 = getelementptr inbounds nuw i8, ptr addrspace(3) %1101, i32 4096
  %1103 = xor i32 %1100, 264
  %1104 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1103
  %1105 = getelementptr inbounds nuw i8, ptr addrspace(3) %1104, i32 4096
  %1106 = xor i32 %1100, 528
  %1107 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1106
  %1108 = getelementptr inbounds nuw i8, ptr addrspace(3) %1107, i32 4096
  %1109 = xor i32 %1100, 792
  %1110 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1109
  %1111 = getelementptr inbounds nuw i8, ptr addrspace(3) %1110, i32 4096
  %1112 = xor i32 %1100, 2112
  %1113 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1112
  %1114 = getelementptr inbounds nuw i8, ptr addrspace(3) %1113, i32 4096
  %1115 = xor i32 %1100, 2376
  %1116 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1115
  %1117 = getelementptr inbounds nuw i8, ptr addrspace(3) %1116, i32 4096
  %1118 = xor i32 %1100, 2640
  %1119 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1118
  %1120 = getelementptr inbounds nuw i8, ptr addrspace(3) %1119, i32 4096
  %1121 = xor i32 %1100, 2904
  %1122 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1121
  %1123 = getelementptr inbounds nuw i8, ptr addrspace(3) %1122, i32 4096
  %1124 = and i32 %38, 60
  %1125 = shl nuw nsw i32 %1124, 6
  %1126 = and i32 %60, 24
  %1127 = shl nuw nsw i32 %1124, 1
  %1128 = shl nuw nsw i32 %41, 5
  %1129 = or disjoint i32 %1125, %1126
  %1130 = xor i32 %1129, %1127
  %1131 = xor i32 %1130, %1128
  %1132 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1131
  %1133 = getelementptr inbounds nuw i8, ptr addrspace(3) %1132, i32 4096
  %1134 = getelementptr inbounds nuw i8, ptr addrspace(3) %1132, i32 128
  %1135 = getelementptr inbounds nuw i8, ptr addrspace(3) %1132, i32 4224
  %1136 = select i1 %1061, i32 0, i32 264
  %1137 = xor i32 %1059, %1136
  %1138 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1137
  %1139 = select i1 %1097, i32 0, i32 264
  %1140 = xor i32 %1139, %1085
  %1141 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1140
  %1142 = getelementptr inbounds nuw i8, ptr addrspace(3) %1141, i32 512
  br label %1143, !dbg !118

1143:                                             ; preds = %.lr.ph121, %1143
  %.pn73.in118 = phi i64 [ %1051, %.lr.ph121 ], [ %1392, %1143 ]
  %.pn69.in116 = phi i64 [ %1054, %.lr.ph121 ], [ %1390, %1143 ]
  %.pn65.in114 = phi i64 [ %1055, %.lr.ph121 ], [ %1388, %1143 ]
  %1144 = phi float [ %1034, %.lr.ph121 ], [ %1257, %1143 ]
  %1145 = phi float [ %1035, %.lr.ph121 ], [ %1337, %1143 ]
  %1146 = phi i32 [ %1036, %.lr.ph121 ], [ %1162, %1143 ]
  %1147 = phi <2 x float> [ %1038, %.lr.ph121 ], [ %1395, %1143 ]
  %1148 = phi <2 x float> [ %1037, %.lr.ph121 ], [ %1394, %1143 ]
  %1149 = phi <2 x float> [ %1040, %.lr.ph121 ], [ %1397, %1143 ]
  %1150 = phi <2 x float> [ %1039, %.lr.ph121 ], [ %1396, %1143 ]
  %.pn73 = trunc i64 %.pn73.in118 to i32, !dbg !121
  %.pn69 = trunc i64 %.pn69.in116 to i32, !dbg !121
  %.pn65 = trunc i64 %.pn65.in114 to i32, !dbg !121
  %1151 = or disjoint i32 %1146, %93, !dbg !122
  %1152 = icmp slt i32 %1151, %33, !dbg !123
  %1153 = shl i32 %.pn65, 1, !dbg !125
  %1154 = select i1 %1152, i32 %1153, i32 -2147483648, !dbg !125
  %1155 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1056, i32 %1154, i32 0, i32 0), !dbg !125
  %1156 = shl i32 %.pn69, 1, !dbg !126
  %1157 = select i1 %1152, i32 %1156, i32 -2147483648, !dbg !126
  %1158 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1056, i32 %1157, i32 0, i32 0), !dbg !126
  %1159 = shl i32 %.pn73, 1, !dbg !128
  %1160 = select i1 %1152, i32 %1159, i32 -2147483648, !dbg !128
  %1161 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1057, i32 %1160, i32 0, i32 0), !dbg !128
  %1162 = add nsw i32 %1146, 32, !dbg !130
  %1163 = icmp eq i32 %1162, %144, !dbg !130
  %1164 = and i1 %140, %1163, !dbg !131
  %1165 = or disjoint i32 %1146, %98, !dbg !132
  %1166 = or disjoint i32 %1165, 16, !dbg !132
  %1167 = icmp sge i32 %1165, %33, !dbg !133
  %1168 = icmp sge i32 %1166, %33, !dbg !133
  %.not74 = select i1 %1164, i1 %1167, i1 false, !dbg !134
  %.not75 = select i1 %1164, i1 %1168, i1 false, !dbg !126
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !126
  tail call void @llvm.amdgcn.s.barrier(), !dbg !126
  store i32 %1158, ptr addrspace(3) %1064, align 4, !dbg !126
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !126
  tail call void @llvm.amdgcn.s.barrier(), !dbg !126
  %1169 = load <8 x bfloat>, ptr addrspace(3) %1073, align 16, !dbg !126
  %1170 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %1169, <8 x bfloat> %1078, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !135
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !125
  tail call void @llvm.amdgcn.s.barrier(), !dbg !125
  store i32 %1155, ptr addrspace(3) %1064, align 4, !dbg !125
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !125
  tail call void @llvm.amdgcn.s.barrier(), !dbg !125
  %1171 = load <8 x bfloat>, ptr addrspace(3) %1073, align 16, !dbg !125
  %1172 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %1171, <8 x bfloat> %1083, <16 x float> %1170, i32 0, i32 0, i32 0), !dbg !136
  %1173 = extractelement <16 x float> %1172, i64 0, !dbg !136
  %1174 = extractelement <16 x float> %1172, i64 1, !dbg !136
  %1175 = extractelement <16 x float> %1172, i64 2, !dbg !136
  %1176 = extractelement <16 x float> %1172, i64 3, !dbg !136
  %1177 = extractelement <16 x float> %1172, i64 4, !dbg !136
  %1178 = extractelement <16 x float> %1172, i64 5, !dbg !136
  %1179 = extractelement <16 x float> %1172, i64 6, !dbg !136
  %1180 = extractelement <16 x float> %1172, i64 7, !dbg !136
  %1181 = extractelement <16 x float> %1172, i64 8, !dbg !136
  %1182 = extractelement <16 x float> %1172, i64 9, !dbg !136
  %1183 = extractelement <16 x float> %1172, i64 10, !dbg !136
  %1184 = extractelement <16 x float> %1172, i64 11, !dbg !136
  %1185 = extractelement <16 x float> %1172, i64 12, !dbg !136
  %1186 = extractelement <16 x float> %1172, i64 13, !dbg !136
  %1187 = extractelement <16 x float> %1172, i64 14, !dbg !136
  %1188 = extractelement <16 x float> %1172, i64 15, !dbg !136
  %1189 = fmul float %1043, %1174, !dbg !137
  %1190 = fmul float %1043, %1173, !dbg !137
  %1191 = select i1 %.not74, float 0xFFF0000000000000, float %1189, !dbg !138
  %1192 = select i1 %.not74, float 0xFFF0000000000000, float %1190, !dbg !138
  %1193 = tail call float @llvm.maxnum.f32(float %1192, float %1191), !dbg !139
  %1194 = insertelement <2 x float> poison, float %1192, i64 0, !dbg !142
  %1195 = insertelement <2 x float> %1194, float %1191, i64 1, !dbg !142
  %1196 = fmul float %1043, %1176, !dbg !137
  %1197 = fmul float %1043, %1175, !dbg !137
  %1198 = select i1 %.not74, float 0xFFF0000000000000, float %1196, !dbg !138
  %1199 = select i1 %.not74, float 0xFFF0000000000000, float %1197, !dbg !138
  %1200 = tail call float @llvm.maxnum.f32(float %1193, float %1199), !dbg !139
  %1201 = insertelement <2 x float> poison, float %1199, i64 0, !dbg !142
  %1202 = insertelement <2 x float> %1201, float %1198, i64 1, !dbg !142
  %1203 = fmul float %1043, %1178, !dbg !137
  %1204 = fmul float %1043, %1177, !dbg !137
  %1205 = select i1 %.not74, float 0xFFF0000000000000, float %1203, !dbg !138
  %1206 = select i1 %.not74, float 0xFFF0000000000000, float %1204, !dbg !138
  %1207 = tail call float @llvm.maxnum.f32(float %1198, float %1206), !dbg !139
  %1208 = tail call float @llvm.maxnum.f32(float %1207, float %1205), !dbg !139
  %1209 = tail call float @llvm.maxnum.f32(float %1200, float %1208), !dbg !139
  %1210 = insertelement <2 x float> poison, float %1206, i64 0, !dbg !142
  %1211 = insertelement <2 x float> %1210, float %1205, i64 1, !dbg !142
  %1212 = fmul float %1043, %1180, !dbg !137
  %1213 = fmul float %1043, %1179, !dbg !137
  %1214 = select i1 %.not74, float 0xFFF0000000000000, float %1212, !dbg !138
  %1215 = select i1 %.not74, float 0xFFF0000000000000, float %1213, !dbg !138
  %1216 = tail call float @llvm.maxnum.f32(float %1215, float %1214), !dbg !139
  %1217 = insertelement <2 x float> poison, float %1215, i64 0, !dbg !142
  %1218 = insertelement <2 x float> %1217, float %1214, i64 1, !dbg !142
  %1219 = fmul float %1043, %1182, !dbg !137
  %1220 = fmul float %1043, %1181, !dbg !137
  %1221 = select i1 %.not75, float 0xFFF0000000000000, float %1219, !dbg !138
  %1222 = select i1 %.not75, float 0xFFF0000000000000, float %1220, !dbg !138
  %1223 = tail call float @llvm.maxnum.f32(float %1216, float %1222), !dbg !139
  %1224 = tail call float @llvm.maxnum.f32(float %1209, float %1223), !dbg !139
  %1225 = insertelement <2 x float> poison, float %1222, i64 0, !dbg !142
  %1226 = insertelement <2 x float> %1225, float %1221, i64 1, !dbg !142
  %1227 = fmul float %1043, %1184, !dbg !137
  %1228 = fmul float %1043, %1183, !dbg !137
  %1229 = select i1 %.not75, float 0xFFF0000000000000, float %1227, !dbg !138
  %1230 = select i1 %.not75, float 0xFFF0000000000000, float %1228, !dbg !138
  %1231 = tail call float @llvm.maxnum.f32(float %1221, float %1230), !dbg !139
  %1232 = tail call float @llvm.maxnum.f32(float %1231, float %1229), !dbg !139
  %1233 = insertelement <2 x float> poison, float %1230, i64 0, !dbg !142
  %1234 = insertelement <2 x float> %1233, float %1229, i64 1, !dbg !142
  %1235 = fmul float %1043, %1186, !dbg !137
  %1236 = fmul float %1043, %1185, !dbg !137
  %1237 = select i1 %.not75, float 0xFFF0000000000000, float %1235, !dbg !138
  %1238 = select i1 %.not75, float 0xFFF0000000000000, float %1236, !dbg !138
  %1239 = tail call float @llvm.maxnum.f32(float %1238, float %1237), !dbg !139
  %1240 = insertelement <2 x float> poison, float %1238, i64 0, !dbg !142
  %1241 = insertelement <2 x float> %1240, float %1237, i64 1, !dbg !142
  %1242 = fmul float %1043, %1188, !dbg !137
  %1243 = fmul float %1043, %1187, !dbg !137
  %1244 = select i1 %.not75, float 0xFFF0000000000000, float %1242, !dbg !138
  %1245 = select i1 %.not75, float 0xFFF0000000000000, float %1243, !dbg !138
  %1246 = tail call float @llvm.maxnum.f32(float %1239, float %1245), !dbg !139
  %1247 = tail call float @llvm.maxnum.f32(float %1232, float %1246), !dbg !139
  %1248 = tail call float @llvm.maxnum.f32(float %1247, float %1244), !dbg !139
  %1249 = tail call float @llvm.maxnum.f32(float %1224, float %1248), !dbg !139
  %1250 = bitcast float %1249 to i32, !dbg !140
  %1251 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %1250, i32 %1250, i1 false, i1 false), !dbg !140
  %1252 = extractvalue { i32, i32 } %1251, 0, !dbg !140
  %1253 = extractvalue { i32, i32 } %1251, 1, !dbg !140
  %1254 = bitcast i32 %1252 to float, !dbg !140
  %1255 = bitcast i32 %1253 to float, !dbg !140
  %1256 = tail call float @llvm.maxnum.f32(float %1254, float %1255), !dbg !139
  %1257 = tail call float @llvm.maxnum.f32(float %1144, float %1256), !dbg !143
  %1258 = insertelement <2 x float> poison, float %1257, i64 0, !dbg !142
  %1259 = shufflevector <2 x float> %1258, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !142
  %1260 = fsub <2 x float> %1195, %1259, !dbg !142
  %1261 = fsub <2 x float> %1202, %1259, !dbg !142
  %1262 = fsub <2 x float> %1211, %1259, !dbg !142
  %1263 = fsub <2 x float> %1218, %1259, !dbg !142
  %1264 = fsub <2 x float> %1226, %1259, !dbg !142
  %1265 = fsub <2 x float> %1234, %1259, !dbg !142
  %1266 = fsub <2 x float> %1241, %1259, !dbg !142
  %1267 = insertelement <2 x float> poison, float %1245, i64 0, !dbg !142
  %1268 = insertelement <2 x float> %1267, float %1244, i64 1, !dbg !142
  %1269 = fsub <2 x float> %1268, %1259, !dbg !142
  %1270 = extractelement <2 x float> %1260, i64 0, !dbg !144
  %1271 = tail call float @llvm.amdgcn.exp2.f32(float %1270), !dbg !144
  %1272 = extractelement <2 x float> %1260, i64 1, !dbg !144
  %1273 = tail call float @llvm.amdgcn.exp2.f32(float %1272), !dbg !144
  %1274 = extractelement <2 x float> %1261, i64 0, !dbg !144
  %1275 = tail call float @llvm.amdgcn.exp2.f32(float %1274), !dbg !144
  %1276 = extractelement <2 x float> %1261, i64 1, !dbg !144
  %1277 = tail call float @llvm.amdgcn.exp2.f32(float %1276), !dbg !144
  %1278 = extractelement <2 x float> %1262, i64 0, !dbg !144
  %1279 = tail call float @llvm.amdgcn.exp2.f32(float %1278), !dbg !144
  %1280 = extractelement <2 x float> %1262, i64 1, !dbg !144
  %1281 = tail call float @llvm.amdgcn.exp2.f32(float %1280), !dbg !144
  %1282 = extractelement <2 x float> %1263, i64 0, !dbg !144
  %1283 = tail call float @llvm.amdgcn.exp2.f32(float %1282), !dbg !144
  %1284 = extractelement <2 x float> %1263, i64 1, !dbg !144
  %1285 = tail call float @llvm.amdgcn.exp2.f32(float %1284), !dbg !144
  %1286 = extractelement <2 x float> %1264, i64 0, !dbg !144
  %1287 = tail call float @llvm.amdgcn.exp2.f32(float %1286), !dbg !144
  %1288 = extractelement <2 x float> %1264, i64 1, !dbg !144
  %1289 = tail call float @llvm.amdgcn.exp2.f32(float %1288), !dbg !144
  %1290 = extractelement <2 x float> %1265, i64 0, !dbg !144
  %1291 = tail call float @llvm.amdgcn.exp2.f32(float %1290), !dbg !144
  %1292 = extractelement <2 x float> %1265, i64 1, !dbg !144
  %1293 = tail call float @llvm.amdgcn.exp2.f32(float %1292), !dbg !144
  %1294 = extractelement <2 x float> %1266, i64 0, !dbg !144
  %1295 = tail call float @llvm.amdgcn.exp2.f32(float %1294), !dbg !144
  %1296 = extractelement <2 x float> %1266, i64 1, !dbg !144
  %1297 = tail call float @llvm.amdgcn.exp2.f32(float %1296), !dbg !144
  %1298 = extractelement <2 x float> %1269, i64 0, !dbg !144
  %1299 = tail call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !144
  %1300 = extractelement <2 x float> %1269, i64 1, !dbg !144
  %1301 = tail call float @llvm.amdgcn.exp2.f32(float %1300), !dbg !144
  %1302 = insertelement <2 x float> poison, float %1271, i64 0, !dbg !145
  %1303 = insertelement <2 x float> %1302, float %1273, i64 1, !dbg !145
  %1304 = insertelement <2 x float> poison, float %1275, i64 0, !dbg !145
  %1305 = insertelement <2 x float> %1304, float %1277, i64 1, !dbg !145
  %1306 = insertelement <2 x float> poison, float %1279, i64 0, !dbg !145
  %1307 = insertelement <2 x float> %1306, float %1281, i64 1, !dbg !145
  %1308 = insertelement <2 x float> poison, float %1283, i64 0, !dbg !145
  %1309 = insertelement <2 x float> %1308, float %1285, i64 1, !dbg !145
  %1310 = insertelement <2 x float> poison, float %1287, i64 0, !dbg !145
  %1311 = insertelement <2 x float> %1310, float %1289, i64 1, !dbg !145
  %1312 = insertelement <2 x float> poison, float %1291, i64 0, !dbg !145
  %1313 = insertelement <2 x float> %1312, float %1293, i64 1, !dbg !145
  %1314 = insertelement <2 x float> poison, float %1295, i64 0, !dbg !145
  %1315 = insertelement <2 x float> %1314, float %1297, i64 1, !dbg !145
  %1316 = insertelement <2 x float> poison, float %1299, i64 0, !dbg !145
  %1317 = insertelement <2 x float> %1316, float %1301, i64 1, !dbg !145
  %1318 = fadd <2 x float> %1303, %1305, !dbg !145
  %1319 = fadd <2 x float> %1307, %1309, !dbg !145
  %1320 = fadd <2 x float> %1311, %1313, !dbg !145
  %1321 = fadd <2 x float> %1315, %1317, !dbg !145
  %1322 = fadd <2 x float> %1318, %1319, !dbg !145
  %1323 = fadd <2 x float> %1320, %1321, !dbg !145
  %1324 = fadd <2 x float> %1322, %1323, !dbg !145
  %shift190 = shufflevector <2 x float> %1324, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !147
  %foldExtExtBinop191 = fadd <2 x float> %1324, %shift190, !dbg !147
  %bc195 = bitcast <2 x float> %foldExtExtBinop191 to <2 x i32>, !dbg !145
  %1325 = extractelement <2 x i32> %bc195, i64 0, !dbg !145
  %1326 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %1325, i32 %1325, i1 false, i1 false), !dbg !145
  %1327 = extractvalue { i32, i32 } %1326, 0, !dbg !145
  %1328 = extractvalue { i32, i32 } %1326, 1, !dbg !145
  %1329 = bitcast i32 %1327 to float, !dbg !145
  %1330 = bitcast i32 %1328 to float, !dbg !145
  %1331 = fadd float %1329, %1330, !dbg !147
  %1332 = fsub float %1144, %1257, !dbg !148
  %1333 = tail call float @llvm.amdgcn.exp2.f32(float %1332), !dbg !149
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !150
  tail call void @llvm.amdgcn.s.barrier(), !dbg !150
  %1334 = insertelement <1 x float> poison, float %1333, i64 0, !dbg !150
  store <1 x float> %1334, ptr addrspace(3) %1091, align 4, !dbg !150
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !150
  tail call void @llvm.amdgcn.s.barrier(), !dbg !150
  %1335 = load <2 x float>, ptr addrspace(3) %1095, align 8, !dbg !150
  %1336 = fmul float %1145, %1333, !dbg !151
  %1337 = fadd float %1331, %1336, !dbg !151
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !152
  tail call void @llvm.amdgcn.s.barrier(), !dbg !152
  %1338 = shufflevector <2 x float> %1302, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1339 = fptrunc <1 x float> %1338 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1339, ptr addrspace(3) %1101, align 2, !dbg !152
  %1340 = shufflevector <2 x float> %1310, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1341 = fptrunc <1 x float> %1340 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1341, ptr addrspace(3) %1102, align 2, !dbg !152
  %1342 = shufflevector <2 x float> %1303, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1343 = fptrunc <1 x float> %1342 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1343, ptr addrspace(3) %1104, align 2, !dbg !152
  %1344 = shufflevector <2 x float> %1311, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1345 = fptrunc <1 x float> %1344 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1345, ptr addrspace(3) %1105, align 2, !dbg !152
  %1346 = shufflevector <2 x float> %1304, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1347 = fptrunc <1 x float> %1346 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1347, ptr addrspace(3) %1107, align 2, !dbg !152
  %1348 = shufflevector <2 x float> %1312, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1349 = fptrunc <1 x float> %1348 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1349, ptr addrspace(3) %1108, align 2, !dbg !152
  %1350 = shufflevector <2 x float> %1305, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1351 = fptrunc <1 x float> %1350 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1351, ptr addrspace(3) %1110, align 2, !dbg !152
  %1352 = shufflevector <2 x float> %1313, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1353 = fptrunc <1 x float> %1352 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1353, ptr addrspace(3) %1111, align 2, !dbg !152
  %1354 = shufflevector <2 x float> %1306, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1355 = fptrunc <1 x float> %1354 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1355, ptr addrspace(3) %1113, align 2, !dbg !152
  %1356 = shufflevector <2 x float> %1314, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1357 = fptrunc <1 x float> %1356 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1357, ptr addrspace(3) %1114, align 2, !dbg !152
  %1358 = shufflevector <2 x float> %1307, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1359 = fptrunc <1 x float> %1358 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1359, ptr addrspace(3) %1116, align 2, !dbg !152
  %1360 = shufflevector <2 x float> %1315, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1361 = fptrunc <1 x float> %1360 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1361, ptr addrspace(3) %1117, align 2, !dbg !152
  %1362 = shufflevector <2 x float> %1308, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1363 = fptrunc <1 x float> %1362 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1363, ptr addrspace(3) %1119, align 2, !dbg !152
  %1364 = shufflevector <2 x float> %1316, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1365 = fptrunc <1 x float> %1364 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1365, ptr addrspace(3) %1120, align 2, !dbg !152
  %1366 = shufflevector <2 x float> %1309, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1367 = fptrunc <1 x float> %1366 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1367, ptr addrspace(3) %1122, align 2, !dbg !152
  %1368 = shufflevector <2 x float> %1317, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1369 = fptrunc <1 x float> %1368 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1369, ptr addrspace(3) %1123, align 2, !dbg !152
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !152
  tail call void @llvm.amdgcn.s.barrier(), !dbg !152
  %1370 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %1132), !dbg !152
  %1371 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1133), !dbg !152
  %1372 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1134), !dbg !152
  %1373 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1135), !dbg !152
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !128
  tail call void @llvm.amdgcn.s.barrier(), !dbg !128
  store i32 %1161, ptr addrspace(3) %1138, align 4, !dbg !128
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !128
  tail call void @llvm.amdgcn.s.barrier(), !dbg !128
  %1374 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %1141), !dbg !128
  %1375 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1142), !dbg !128
  %1376 = shufflevector <4 x bfloat> %1370, <4 x bfloat> %1371, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !153
  %1377 = shufflevector <4 x bfloat> %1372, <4 x bfloat> %1373, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !153
  %1378 = shufflevector <4 x bfloat> %1374, <4 x bfloat> %1375, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !153
  %1379 = shufflevector <2 x float> %1149, <2 x float> %1150, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !153
  %1380 = shufflevector <2 x float> %1335, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !153
  %1381 = fmul <4 x float> %1379, %1380, !dbg !153
  %1382 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1378, <8 x bfloat> %1376, <4 x float> %1381, i32 0, i32 0, i32 0), !dbg !153
  %1383 = shufflevector <2 x float> %1147, <2 x float> %1148, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !153
  %1384 = shufflevector <2 x float> %1335, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !153
  %1385 = fmul <4 x float> %1383, %1384, !dbg !153
  %1386 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1378, <8 x bfloat> %1377, <4 x float> %1385, i32 0, i32 0, i32 0), !dbg !153
  %sext = shl i64 %.pn65.in114, 32, !dbg !154
  %1387 = ashr exact i64 %sext, 32, !dbg !154
  %1388 = add nsw i64 %1387, %1044, !dbg !154
  %sext77 = shl i64 %.pn69.in116, 32, !dbg !155
  %1389 = ashr exact i64 %sext77, 32, !dbg !155
  %1390 = add nsw i64 %1389, %1044, !dbg !155
  %sext79 = shl i64 %.pn73.in118, 32, !dbg !156
  %1391 = ashr exact i64 %sext79, 32, !dbg !156
  %1392 = add nsw i64 %1391, %1045, !dbg !156
  %1393 = icmp slt i32 %1162, %144, !dbg !118
  %1394 = shufflevector <4 x float> %1386, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !113
  %1395 = shufflevector <4 x float> %1386, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !113
  %1396 = shufflevector <4 x float> %1382, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !113
  %1397 = shufflevector <4 x float> %1382, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !113
  br i1 %1393, label %1143, label %.loopexit, !dbg !118

.loopexit:                                        ; preds = %1143, %1042, %1033
  %1398 = phi float [ %1034, %1033 ], [ %1034, %1042 ], [ %1257, %1143 ], !dbg !113
  %1399 = phi float [ %1035, %1033 ], [ %1035, %1042 ], [ %1337, %1143 ], !dbg !113
  %1400 = phi <2 x float> [ %1037, %1033 ], [ %1037, %1042 ], [ %1394, %1143 ], !dbg !113
  %1401 = phi <2 x float> [ %1038, %1033 ], [ %1038, %1042 ], [ %1395, %1143 ], !dbg !113
  %1402 = phi <2 x float> [ %1039, %1033 ], [ %1039, %1042 ], [ %1396, %1143 ], !dbg !113
  %1403 = phi <2 x float> [ %1040, %1033 ], [ %1040, %1042 ], [ %1397, %1143 ], !dbg !113
  %1404 = fdiv float 1.000000e+00, %1399, !dbg !157
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !158
  tail call void @llvm.amdgcn.s.barrier(), !dbg !158
  %1405 = and i32 %38, 31, !dbg !158
  %1406 = shl nuw nsw i32 %1405, 3, !dbg !158
  %1407 = shl nuw nsw i32 %41, 8, !dbg !158
  %1408 = and i32 %1407, 256, !dbg !158
  %1409 = shl nuw nsw i32 %40, 1, !dbg !158
  %1410 = and i32 %1409, 4, !dbg !158
  %1411 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1406, !dbg !158
  %1412 = getelementptr inbounds nuw i8, ptr addrspace(3) %1411, i32 %1408, !dbg !158
  %1413 = getelementptr inbounds nuw i8, ptr addrspace(3) %1412, i32 %1410, !dbg !158
  %1414 = insertelement <1 x float> poison, float %1404, i64 0, !dbg !158
  store <1 x float> %1414, ptr addrspace(3) %1413, align 4, !dbg !158
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !158
  tail call void @llvm.amdgcn.s.barrier(), !dbg !158
  %1415 = shl nuw nsw i32 %57, 3, !dbg !158
  %1416 = shl nuw nsw i32 %41, 7, !dbg !158
  %1417 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1415, !dbg !158
  %1418 = getelementptr inbounds nuw i8, ptr addrspace(3) %1417, i32 %1416, !dbg !158
  %1419 = load <2 x float>, ptr addrspace(3) %1418, align 8, !dbg !158
  %reass.sub = sub i32 %55, %32, !dbg !159
  %1420 = add i32 %reass.sub, 128, !dbg !159
  %1421 = tail call noundef float @llvm.log2.f32(float %1399), !dbg !160
  %1422 = fadd float %1398, %1421, !dbg !161
  %1423 = fmul float %1422, 0x3FE62E4300000000, !dbg !162
  %1424 = mul nuw i64 %80, %70, !dbg !163
  %1425 = mul nuw i64 %71, %75, !dbg !164
  %1426 = add i64 %1424, %1425, !dbg !163
  %1427 = icmp slt i32 %1420, 1, !dbg !165
  br i1 %1427, label %1446, label %1428, !dbg !166

1428:                                             ; preds = %.loopexit
  %1429 = sub nsw i32 0, %reass.sub, !dbg !167
  %1430 = icmp slt i32 %90, %1429, !dbg !168
  %1431 = trunc i64 %1426 to i32, !dbg !169
  %1432 = add i32 %92, %1431, !dbg !169
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !170
  tail call void @llvm.amdgcn.s.barrier(), !dbg !170
  %1433 = shl nuw nsw i32 %1405, 2, !dbg !170
  %1434 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1433, !dbg !170
  %1435 = getelementptr inbounds nuw i8, ptr addrspace(3) %1434, i32 %1416, !dbg !170
  %1436 = insertelement <1 x float> poison, float %1423, i64 0, !dbg !170
  store <1 x float> %1436, ptr addrspace(3) %1435, align 4, !dbg !170
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !170
  tail call void @llvm.amdgcn.s.barrier(), !dbg !170
  %1437 = shl nuw nsw i32 %90, 2, !dbg !170
  %1438 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1437, !dbg !170
  %1439 = load float, ptr addrspace(3) %1438, align 4, !dbg !170
  %1440 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 2147483646, i32 159744), !dbg !170
  %1441 = and i32 %39, 128, !dbg !170
  %1442 = icmp eq i32 %1441, 0, !dbg !170
  %1443 = and i1 %1442, %1430, !dbg !170
  %1444 = shl i32 %1432, 2, !dbg !170
  %1445 = select i1 %1443, i32 %1444, i32 -2147483648, !dbg !170
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %1439, ptr addrspace(8) %1440, i32 %1445, i32 0, i32 0), !dbg !170
  br label %1461, !dbg !166

1446:                                             ; preds = %.loopexit
  %1447 = trunc i64 %1426 to i32, !dbg !171
  %1448 = add i32 %92, %1447, !dbg !171
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  %1449 = shl nuw nsw i32 %1405, 2, !dbg !14
  %1450 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1449, !dbg !14
  %1451 = getelementptr inbounds nuw i8, ptr addrspace(3) %1450, i32 %1416, !dbg !14
  %1452 = insertelement <1 x float> poison, float %1423, i64 0, !dbg !14
  store <1 x float> %1452, ptr addrspace(3) %1451, align 4, !dbg !14
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  %1453 = shl nuw nsw i32 %90, 2, !dbg !14
  %1454 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1453, !dbg !14
  %1455 = load float, ptr addrspace(3) %1454, align 4, !dbg !14
  %1456 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 2147483646, i32 159744), !dbg !14
  %1457 = and i32 %39, 128, !dbg !14
  %1458 = icmp eq i32 %1457, 0, !dbg !14
  %1459 = shl i32 %1448, 2, !dbg !14
  %1460 = select i1 %1458, i32 %1459, i32 -2147483648, !dbg !14
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %1455, ptr addrspace(8) %1456, i32 %1460, i32 0, i32 0), !dbg !14
  br label %1461, !dbg !166

1461:                                             ; preds = %1428, %1446
  %1462 = shufflevector <2 x float> %1419, <2 x float> poison, <2 x i32> <i32 1, i32 1>, !dbg !158
  %1463 = fmul <2 x float> %1400, %1462, !dbg !158
  %1464 = fmul <2 x float> %1401, %1462, !dbg !158
  %1465 = shufflevector <2 x float> %1419, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !158
  %1466 = fmul <2 x float> %1402, %1465, !dbg !158
  %1467 = fmul <2 x float> %1403, %1465, !dbg !158
  %1468 = shl nuw nsw i32 %41, 4, !dbg !31
  %1469 = or disjoint i32 %1468, %57, !dbg !31
  %1470 = or disjoint i32 %55, %1469, !dbg !30
  %1471 = or disjoint i32 %1470, 64, !dbg !30
  %1472 = icmp slt i32 %1471, %32, !dbg !61
  %1473 = icmp slt i32 %1470, %32, !dbg !61
  %1474 = lshr i32 %38, 2, !dbg !50
  %1475 = and i32 %1474, 12, !dbg !50
  %1476 = zext nneg i32 %1475 to i64, !dbg !50
  %1477 = zext i32 %1471 to i64, !dbg !172
  %1478 = zext i32 %1470 to i64, !dbg !172
  %1479 = zext i32 %19 to i64, !dbg !173
  %1480 = zext i32 %18 to i64, !dbg !174
  %1481 = zext i32 %17 to i64, !dbg !175
  %1482 = mul nuw i64 %80, %1481, !dbg !176
  %1483 = mul nuw i64 %1480, %75, !dbg !177
  %1484 = add i64 %1482, %1483, !dbg !176
  %1485 = select i1 %1427, i1 true, i1 %1473, !dbg !178
  %1486 = select i1 %1427, i1 true, i1 %1472, !dbg !178
  %1487 = fptrunc <2 x float> %1467 to <2 x bfloat>, !dbg !179
  %1488 = fptrunc <2 x float> %1466 to <2 x bfloat>, !dbg !179
  %1489 = fptrunc <2 x float> %1464 to <2 x bfloat>, !dbg !179
  %1490 = fptrunc <2 x float> %1463 to <2 x bfloat>, !dbg !179
  %1491 = mul nuw i64 %1478, %1479, !dbg !180
  %1492 = mul nuw i64 %1477, %1479, !dbg !180
  %1493 = add i64 %1484, %1476, !dbg !180
  %1494 = add i64 %1493, %1491, !dbg !180
  %1495 = add i64 %1493, %1492, !dbg !180
  %1496 = trunc i64 %1494 to i32, !dbg !180
  %1497 = trunc i64 %1495 to i32, !dbg !180
  %1498 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %3, i16 0, i64 2147483646, i32 159744), !dbg !181
  %1499 = shufflevector <2 x bfloat> %1487, <2 x bfloat> %1488, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !181
  %1500 = bitcast <4 x bfloat> %1499 to <2 x i32>, !dbg !181
  %1501 = shl i32 %1496, 1, !dbg !181
  %1502 = select i1 %1485, i32 %1501, i32 -2147483648, !dbg !181
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1500, ptr addrspace(8) %1498, i32 %1502, i32 0, i32 0), !dbg !181
  %1503 = shufflevector <2 x bfloat> %1489, <2 x bfloat> %1490, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !181
  %1504 = bitcast <4 x bfloat> %1503 to <2 x i32>, !dbg !181
  %1505 = shl i32 %1497, 1, !dbg !181
  %1506 = select i1 %1486, i32 %1505, i32 -2147483648, !dbg !181
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1504, ptr addrspace(8) %1498, i32 %1506, i32 0, i32 0), !dbg !181
  ret void, !dbg !182
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone, i16, i64, i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #3

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #6

; Function Attrs: nounwind
declare void @llvm.amdgcn.asyncmark() #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #8

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #9

; Function Attrs: nounwind
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) captures(none)) #10

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #11

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1 immarg, i1 immarg) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #3

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #11

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #12

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #12

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.log2.f32(float) #3

attributes #0 = { nounwind "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="2, 2" "denormal-fp-math-f32"="ieee" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nounwind }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn }
attributes #9 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #10 = { convergent mustprogress nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #11 = { convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #12 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "mha.py", directory: "/root/aiter/aiter/ops/triton/_triton_kernels/attention")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 0}
!6 = distinct !DISubprogram(name: "_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0", linkageName: "_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0", scope: !1, file: !1, line: 297, type: !7, scopeLine: 297, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DISubroutineType(cc: DW_CC_normal, types: !8)
!8 = !{null, !9, !9, !9, !9, !11, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !12, !12, !13, !13, !13, !13, !13, !9, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !10, size: 64, dwarfAddressSpace: 1)
!10 = !DIBasicType(name: "unknown_type", encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !12, size: 64, dwarfAddressSpace: 1)
!12 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 912, column: 13, scope: !6)
!15 = !DILocation(line: 367, column: 19, scope: !6)
!16 = !DILocation(line: 367, column: 18, scope: !6)
!17 = !DILocation(line: 369, column: 11, scope: !6)
!18 = !DILocation(line: 376, column: 16, scope: !6)
!19 = !DILocation(line: 374, column: 18, scope: !6)
!20 = !DILocation(line: 41, column: 17, scope: !21, inlinedAt: !23)
!21 = distinct !DILexicalBlockFile(scope: !6, file: !22, discriminator: 0)
!22 = !DIFile(filename: "pid_preprocessing.py", directory: "/root/aiter/aiter/ops/triton/utils/_triton")
!23 = !DILocation(line: 375, column: 18, scope: !24)
!24 = distinct !DILexicalBlockFile(scope: !6, file: !1, discriminator: 0)
!25 = !DILocation(line: 47, column: 15, scope: !21, inlinedAt: !23)
!26 = !DILocation(line: 376, column: 15, scope: !6)
!27 = !DILocation(line: 377, column: 22, scope: !6)
!28 = !DILocation(line: 377, column: 14, scope: !6)
!29 = !DILocation(line: 377, column: 13, scope: !6)
!30 = !DILocation(line: 380, column: 14, scope: !6)
!31 = !DILocation(line: 380, column: 34, scope: !6)
!32 = !DILocation(line: 382, column: 14, scope: !6)
!33 = !DILocation(line: 385, column: 19, scope: !6)
!34 = !DILocation(line: 401, column: 21, scope: !6)
!35 = !DILocation(line: 402, column: 21, scope: !6)
!36 = !DILocation(line: 403, column: 21, scope: !6)
!37 = !DILocation(line: 405, column: 21, scope: !6)
!38 = !DILocation(line: 406, column: 21, scope: !6)
!39 = !DILocation(line: 407, column: 21, scope: !6)
!40 = !DILocation(line: 426, column: 24, scope: !6)
!41 = !DILocation(line: 427, column: 24, scope: !6)
!42 = !DILocation(line: 19, column: 13, scope: !24, inlinedAt: !43)
!43 = !DILocation(line: 510, column: 16, scope: !24)
!44 = !DILocation(line: 19, column: 12, scope: !24, inlinedAt: !43)
!45 = !DILocation(line: 563, column: 22, scope: !6)
!46 = !DILocation(line: 573, column: 14, scope: !6)
!47 = !DILocation(line: 574, column: 14, scope: !6)
!48 = !DILocation(line: 575, column: 14, scope: !6)
!49 = !DILocation(line: 582, column: 9, scope: !6)
!50 = !DILocation(line: 586, column: 11, scope: !6)
!51 = !DILocation(line: 602, column: 9, scope: !6)
!52 = !DILocation(line: 606, column: 11, scope: !6)
!53 = !DILocation(line: 381, column: 14, scope: !6)
!54 = !DILocation(line: 588, column: 14, scope: !6)
!55 = !DILocation(line: 597, column: 21, scope: !6)
!56 = !DILocation(line: 608, column: 14, scope: !6)
!57 = !DILocation(line: 614, column: 15, scope: !6)
!58 = !DILocation(line: 617, column: 21, scope: !6)
!59 = !DILocation(line: 622, column: 9, scope: !6)
!60 = !DILocation(line: 628, column: 14, scope: !6)
!61 = !DILocation(line: 679, column: 18, scope: !6)
!62 = !DILocation(line: 689, column: 16, scope: !6)
!63 = !DILocation(line: 692, column: 9, scope: !6)
!64 = !DILocation(line: 701, column: 8, scope: !6)
!65 = !DILocation(line: 701, column: 5, scope: !6)
!66 = !DILocation(line: 728, column: 21, scope: !6)
!67 = !DILocation(line: 729, column: 21, scope: !6)
!68 = !DILocation(line: 731, column: 17, scope: !6)
!69 = !DILocation(line: 744, column: 8, scope: !6)
!70 = !DILocation(line: 744, column: 5, scope: !6)
!71 = !DILocation(line: 745, column: 33, scope: !6)
!72 = !DILocation(line: 177, column: 20, scope: !24, inlinedAt: !73)
!73 = !DILocation(line: 746, column: 25, scope: !24)
!74 = !DILocation(line: 261, column: 19, scope: !24, inlinedAt: !73)
!75 = !DILocation(line: 264, column: 19, scope: !24, inlinedAt: !73)
!76 = !DILocation(line: 132, column: 5, scope: !24, inlinedAt: !73)
!77 = !DILocation(line: 36, column: 18, scope: !24, inlinedAt: !78)
!78 = !DILocation(line: 140, column: 13, scope: !24, inlinedAt: !73)
!79 = !{!80}
!80 = !{!"amdg.AsyncCopies", !81, !"Scope containing all AsyncCopyGlobalToLocal and BufferLoadToLocal ops"}
!81 = !{!"amdg.AsyncOps", !"Domain to hold alias scopes to specify aliasing information between AsyncCopyGlobalToLocal, BufferLoadToLocal and LocalLoad ops"}
!82 = !DILocation(line: 36, column: 18, scope: !24, inlinedAt: !83)
!83 = !DILocation(line: 142, column: 20, scope: !24, inlinedAt: !73)
!84 = !DILocation(line: 36, column: 18, scope: !24, inlinedAt: !85)
!85 = !DILocation(line: 150, column: 17, scope: !24, inlinedAt: !73)
!86 = !DILocation(line: 261, column: 9, scope: !24, inlinedAt: !73)
!87 = !DILocation(line: 263, column: 13, scope: !24, inlinedAt: !73)
!88 = !DILocation(line: 264, column: 9, scope: !24, inlinedAt: !73)
!89 = !DILocation(line: 186, column: 18, scope: !24, inlinedAt: !73)
!90 = !{!91}
!91 = !{!"amdg.LocalLoads", !81, !"Scope containing all LocalLoad ops"}
!92 = !DILocation(line: 181, column: 18, scope: !24, inlinedAt: !73)
!93 = !DILocation(line: 182, column: 14, scope: !24, inlinedAt: !73)
!94 = !DILocation(line: 212, column: 26, scope: !24, inlinedAt: !73)
!95 = !DILocation(line: 170, column: 12, scope: !96, inlinedAt: !98)
!96 = distinct !DILexicalBlockFile(scope: !6, file: !97, discriminator: 0)
!97 = !DIFile(filename: "standard.py", directory: "/root/triton/python/triton/language")
!98 = !DILocation(line: 191, column: 16, scope: !96, inlinedAt: !99)
!99 = !DILocation(line: 209, column: 32, scope: !24, inlinedAt: !73)
!100 = !DILocation(line: 209, column: 16, scope: !24, inlinedAt: !73)
!101 = !DILocation(line: 212, column: 13, scope: !24, inlinedAt: !73)
!102 = !DILocation(line: 293, column: 12, scope: !96, inlinedAt: !103)
!103 = !DILocation(line: 220, column: 16, scope: !24, inlinedAt: !73)
!104 = !DILocation(line: 263, column: 12, scope: !96, inlinedAt: !102)
!105 = !DILocation(line: 241, column: 30, scope: !24, inlinedAt: !73)
!106 = !DILocation(line: 241, column: 17, scope: !24, inlinedAt: !73)
!107 = !DILocation(line: 246, column: 15, scope: !24, inlinedAt: !73)
!108 = !DILocation(line: 248, column: 15, scope: !24, inlinedAt: !73)
!109 = !DILocation(line: 259, column: 26, scope: !24, inlinedAt: !73)
!110 = !DILocation(line: 259, column: 19, scope: !24, inlinedAt: !73)
!111 = !DILocation(line: 746, column: 25, scope: !6)
!112 = !DILocation(line: 798, column: 8, scope: !6)
!113 = !DILocation(line: 798, column: 5, scope: !6)
!114 = !DILocation(line: 177, column: 20, scope: !24, inlinedAt: !115)
!115 = !DILocation(line: 811, column: 25, scope: !24)
!116 = !DILocation(line: 261, column: 19, scope: !24, inlinedAt: !115)
!117 = !DILocation(line: 264, column: 19, scope: !24, inlinedAt: !115)
!118 = !DILocation(line: 132, column: 5, scope: !24, inlinedAt: !115)
!119 = !DILocation(line: 803, column: 19, scope: !6)
!120 = !DILocation(line: 806, column: 19, scope: !6)
!121 = !DILocation(line: 0, scope: !6)
!122 = !DILocation(line: 136, column: 24, scope: !24, inlinedAt: !115)
!123 = !DILocation(line: 33, column: 16, scope: !24, inlinedAt: !124)
!124 = !DILocation(line: 140, column: 13, scope: !24, inlinedAt: !115)
!125 = !DILocation(line: 34, column: 18, scope: !24, inlinedAt: !124)
!126 = !DILocation(line: 34, column: 18, scope: !24, inlinedAt: !127)
!127 = !DILocation(line: 142, column: 20, scope: !24, inlinedAt: !115)
!128 = !DILocation(line: 31, column: 18, scope: !24, inlinedAt: !129)
!129 = !DILocation(line: 150, column: 17, scope: !24, inlinedAt: !115)
!130 = !DILocation(line: 168, column: 27, scope: !24, inlinedAt: !115)
!131 = !DILocation(line: 168, column: 26, scope: !24, inlinedAt: !115)
!132 = !DILocation(line: 169, column: 22, scope: !24, inlinedAt: !115)
!133 = !DILocation(line: 170, column: 28, scope: !24, inlinedAt: !115)
!134 = !DILocation(line: 171, column: 20, scope: !24, inlinedAt: !115)
!135 = !DILocation(line: 181, column: 18, scope: !24, inlinedAt: !115)
!136 = !DILocation(line: 182, column: 14, scope: !24, inlinedAt: !115)
!137 = !DILocation(line: 186, column: 18, scope: !24, inlinedAt: !115)
!138 = !DILocation(line: 198, column: 14, scope: !24, inlinedAt: !115)
!139 = !DILocation(line: 170, column: 12, scope: !96, inlinedAt: !140)
!140 = !DILocation(line: 191, column: 16, scope: !96, inlinedAt: !141)
!141 = !DILocation(line: 209, column: 32, scope: !24, inlinedAt: !115)
!142 = !DILocation(line: 212, column: 26, scope: !24, inlinedAt: !115)
!143 = !DILocation(line: 209, column: 16, scope: !24, inlinedAt: !115)
!144 = !DILocation(line: 212, column: 13, scope: !24, inlinedAt: !115)
!145 = !DILocation(line: 293, column: 12, scope: !96, inlinedAt: !146)
!146 = !DILocation(line: 220, column: 16, scope: !24, inlinedAt: !115)
!147 = !DILocation(line: 263, column: 12, scope: !96, inlinedAt: !145)
!148 = !DILocation(line: 241, column: 30, scope: !24, inlinedAt: !115)
!149 = !DILocation(line: 241, column: 17, scope: !24, inlinedAt: !115)
!150 = !DILocation(line: 246, column: 15, scope: !24, inlinedAt: !115)
!151 = !DILocation(line: 248, column: 15, scope: !24, inlinedAt: !115)
!152 = !DILocation(line: 259, column: 26, scope: !24, inlinedAt: !115)
!153 = !DILocation(line: 259, column: 19, scope: !24, inlinedAt: !115)
!154 = !DILocation(line: 261, column: 9, scope: !24, inlinedAt: !115)
!155 = !DILocation(line: 263, column: 13, scope: !24, inlinedAt: !115)
!156 = !DILocation(line: 264, column: 9, scope: !24, inlinedAt: !115)
!157 = !DILocation(line: 861, column: 15, scope: !6)
!158 = !DILocation(line: 862, column: 11, scope: !6)
!159 = !DILocation(line: 884, column: 21, scope: !6)
!160 = !DILocation(line: 888, column: 29, scope: !6)
!161 = !DILocation(line: 888, column: 23, scope: !6)
!162 = !DILocation(line: 890, column: 9, scope: !6)
!163 = !DILocation(line: 900, column: 13, scope: !6)
!164 = !DILocation(line: 901, column: 15, scope: !6)
!165 = !DILocation(line: 905, column: 12, scope: !6)
!166 = !DILocation(line: 905, column: 9, scope: !6)
!167 = !DILocation(line: 906, column: 44, scope: !6)
!168 = !DILocation(line: 907, column: 24, scope: !6)
!169 = !DILocation(line: 909, column: 17, scope: !6)
!170 = !DILocation(line: 908, column: 13, scope: !6)
!171 = !DILocation(line: 913, column: 17, scope: !6)
!172 = !DILocation(line: 585, column: 11, scope: !6)
!173 = !DILocation(line: 415, column: 21, scope: !6)
!174 = !DILocation(line: 414, column: 21, scope: !6)
!175 = !DILocation(line: 413, column: 21, scope: !6)
!176 = !DILocation(line: 918, column: 9, scope: !6)
!177 = !DILocation(line: 919, column: 11, scope: !6)
!178 = !DILocation(line: 925, column: 5, scope: !6)
!179 = !DILocation(line: 929, column: 10, scope: !6)
!180 = !DILocation(line: 930, column: 14, scope: !6)
!181 = !DILocation(line: 930, column: 5, scope: !6)
!182 = !DILocation(line: 297, column: 1, scope: !6)
