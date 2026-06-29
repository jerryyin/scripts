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
  br i1 %145, label %146, label %1028, !dbg !70

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
  %160 = and i32 %39, 64, !dbg !77
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
  br i1 %208, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !76

.._crit_edge_crit_edge:                           ; preds = %146
  %.pre = shl nuw nsw i32 %38, 5, !dbg !77
  %.pre138 = and i32 %.pre, 736, !dbg !77
  %.pre140 = and i32 %38, 8, !dbg !77
  %.pre142 = lshr exact i32 %86, 1, !dbg !77
  %.pre144 = shl nuw nsw i32 %87, 3, !dbg !84
  %209 = insertelement <2 x i32> poison, i32 %.pre142, i64 0, !dbg !84
  %210 = insertelement <2 x i32> %209, i32 %.pre144, i64 1, !dbg !84
  br label %._crit_edge, !dbg !76

.lr.ph:                                           ; preds = %146
  %invariant.gep = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %56, !dbg !76
  %211 = shl nuw nsw i32 %38, 5
  %212 = and i32 %211, 736
  %213 = and i32 %38, 8
  %214 = icmp eq i32 %213, 0
  %215 = select i1 %214, i32 0, i32 272
  %216 = insertelement <2 x i32> poison, i32 %86, i64 0
  %217 = insertelement <2 x i32> %216, i32 %87, i64 1
  %218 = shl nuw nsw <2 x i32> %217, <i32 1, i32 3>
  %219 = lshr exact <2 x i32> %217, <i32 1, i32 3>
  %220 = shufflevector <2 x i32> %219, <2 x i32> %218, <2 x i32> <i32 0, i32 3>
  %221 = extractelement <2 x i32> %219, i64 0
  %222 = xor i32 %215, %221
  %223 = or disjoint i32 %222, %212
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
  %235 = shl nuw nsw i32 %234, 2
  %236 = shl nuw nsw i32 %41, 7
  %237 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %235
  %238 = getelementptr inbounds nuw i8, ptr addrspace(3) %237, i32 %236
  %239 = shl nuw nsw i32 %57, 2
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %invariant.gep, i32 %239
  %240 = getelementptr inbounds nuw i8, ptr addrspace(3) %gep, i32 256
  %241 = shl nuw nsw i32 %234, 1
  %242 = icmp eq i32 %86, 0
  %243 = select i1 %242, i32 0, i32 1056
  %244 = or disjoint i32 %56, %241
  %245 = xor i32 %244, %243
  %246 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %245
  %247 = getelementptr inbounds nuw i8, ptr addrspace(3) %246, i32 4096
  %248 = xor i32 %245, 264
  %249 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %248
  %250 = getelementptr inbounds nuw i8, ptr addrspace(3) %249, i32 4096
  %251 = xor i32 %245, 528
  %252 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %251
  %253 = getelementptr inbounds nuw i8, ptr addrspace(3) %252, i32 4096
  %254 = xor i32 %245, 792
  %255 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %254
  %256 = getelementptr inbounds nuw i8, ptr addrspace(3) %255, i32 4096
  %257 = xor i32 %245, 2112
  %258 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %257
  %259 = getelementptr inbounds nuw i8, ptr addrspace(3) %258, i32 4096
  %260 = xor i32 %245, 2376
  %261 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %260
  %262 = getelementptr inbounds nuw i8, ptr addrspace(3) %261, i32 4096
  %263 = xor i32 %245, 2640
  %264 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %263
  %265 = getelementptr inbounds nuw i8, ptr addrspace(3) %264, i32 4096
  %266 = xor i32 %245, 2904
  %267 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %266
  %268 = getelementptr inbounds nuw i8, ptr addrspace(3) %267, i32 4096
  %269 = and i32 %38, 60
  %270 = shl nuw nsw i32 %269, 6
  %271 = and i32 %60, 24
  %272 = shl nuw nsw i32 %269, 1
  %273 = shl nuw nsw i32 %41, 5
  %274 = or disjoint i32 %270, %271
  %275 = xor i32 %274, %272
  %276 = xor i32 %275, %273
  %277 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %276
  %278 = getelementptr inbounds nuw i8, ptr addrspace(3) %277, i32 4096
  %279 = getelementptr inbounds nuw i8, ptr addrspace(3) %277, i32 128
  %280 = getelementptr inbounds nuw i8, ptr addrspace(3) %277, i32 4224
  %281 = extractelement <2 x i32> %218, i64 1, !dbg !84
  %282 = insertelement <8 x float> poison, float %154, i64 0, !dbg !89
  %283 = shufflevector <8 x float> %282, <8 x float> poison, <8 x i32> zeroinitializer, !dbg !89
  %284 = insertelement <4 x float> poison, float %154, i64 0, !dbg !89
  %285 = shufflevector <4 x float> %284, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !89
  %286 = insertelement <2 x float> poison, float %154, i64 0, !dbg !89
  %287 = shufflevector <2 x float> %286, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !89
  br label %288, !dbg !76

288:                                              ; preds = %.lr.ph, %288
  %289 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 9248), %.lr.ph ], [ %336, %288 ]
  %290 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), %.lr.ph ], [ %289, %288 ]
  %291 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 12352), %.lr.ph ], [ %326, %288 ]
  %292 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 11328), %.lr.ph ], [ %291, %288 ]
  %293 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 15424), %.lr.ph ], [ %316, %288 ]
  %294 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 14400), %.lr.ph ], [ %293, %288 ]
  %295 = phi i32 [ 1, %.lr.ph ], [ %314, %288 ]
  %.pn28104 = phi i64 [ %191, %.lr.ph ], [ %310, %288 ]
  %.pn23102 = phi i64 [ %189, %.lr.ph ], [ %307, %288 ]
  %.pn18100 = phi i64 [ %187, %.lr.ph ], [ %304, %288 ]
  %296 = phi float [ 0xFFF0000000000000, %.lr.ph ], [ %397, %288 ]
  %297 = phi float [ 1.000000e+00, %.lr.ph ], [ %475, %288 ]
  %298 = phi i32 [ 0, %.lr.ph ], [ %523, %288 ]
  %299 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %526, %288 ]
  %300 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %525, %288 ]
  %301 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %528, %288 ]
  %302 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %527, %288 ]
  %sext81 = shl i64 %.pn18100, 32, !dbg !86
  %303 = ashr exact i64 %sext81, 32, !dbg !86
  %304 = add nsw i64 %303, %155, !dbg !86
  %305 = trunc i64 %304 to i32, !dbg !86
  %sext83 = shl i64 %.pn23102, 32, !dbg !87
  %306 = ashr exact i64 %sext83, 32, !dbg !87
  %307 = add nsw i64 %306, %155, !dbg !87
  %308 = trunc i64 %307 to i32, !dbg !87
  %sext85 = shl i64 %.pn28104, 32, !dbg !88
  %309 = ashr exact i64 %sext85, 32, !dbg !88
  %310 = add nsw i64 %309, %156, !dbg !88
  %311 = trunc i64 %310 to i32, !dbg !88
  %312 = add i32 %295, 1, !dbg !76
  %313 = icmp slt i32 %312, 3, !dbg !76
  %314 = select i1 %313, i32 %312, i32 0, !dbg !76
  %315 = shl i32 %314, 9, !dbg !77
  %316 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 14400), i32 %315, !dbg !77
  %317 = getelementptr inbounds nuw i8, ptr addrspace(3) %316, i32 %167, !dbg !77
  %318 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %170, i32 %305), !dbg !77
  %319 = tail call i64 @llvm.amdgcn.ballot.i64(i1 true), !dbg !77
  %320 = lshr i64 %319, %173, !dbg !77
  %321 = trunc i64 %320 to i1, !dbg !77
  %322 = shl i32 %318, 1, !dbg !77
  %323 = select i1 %321, i32 %322, i32 -2147483648, !dbg !77
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %166, ptr addrspace(3) %317, i32 4, i32 %323, i32 0, i32 0, i32 0), !dbg !77, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !77
  %324 = getelementptr inbounds nuw i8, ptr addrspace(3) %294, i32 %223, !dbg !77
  %325 = load <8 x bfloat>, ptr addrspace(3) %324, align 16, !dbg !77, !alias.scope !90, !noalias !79
  %326 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 11328), i32 %315, !dbg !82
  %327 = getelementptr inbounds nuw i8, ptr addrspace(3) %326, i32 %167, !dbg !82
  %328 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %170, i32 %308), !dbg !82
  %329 = shl i32 %328, 1, !dbg !82
  %330 = select i1 %321, i32 %329, i32 -2147483648, !dbg !82
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %166, ptr addrspace(3) %327, i32 4, i32 %330, i32 0, i32 0, i32 0), !dbg !82, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !82
  %331 = getelementptr inbounds nuw i8, ptr addrspace(3) %292, i32 %223, !dbg !82
  %332 = load <8 x bfloat>, ptr addrspace(3) %331, align 16, !dbg !82, !alias.scope !90, !noalias !79
  %333 = shl i32 %314, 4, !dbg !84
  %334 = and i32 %333, 134217712, !dbg !84
  %335 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %315, !dbg !84
  %336 = getelementptr [2 x i8], ptr addrspace(3) %335, i32 %334, !dbg !84
  %337 = getelementptr inbounds nuw i8, ptr addrspace(3) %336, i32 %167, !dbg !84
  %338 = shl i32 %311, 1, !dbg !84
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %182, ptr addrspace(3) %337, i32 4, i32 %338, i32 0, i32 0, i32 0), !dbg !84, !alias.scope !79
  tail call void @llvm.amdgcn.asyncmark(), !dbg !84
  %339 = getelementptr inbounds nuw i8, ptr addrspace(3) %290, i32 %281, !dbg !84
  %340 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %339), !dbg !84, !alias.scope !90, !noalias !79
  %341 = getelementptr inbounds nuw i8, ptr addrspace(3) %339, i32 512, !dbg !84
  %342 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %341), !dbg !84, !alias.scope !90, !noalias !79
  %343 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %332, <8 x bfloat> %228, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !92
  %344 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %325, <8 x bfloat> %233, <16 x float> %343, i32 0, i32 0, i32 0), !dbg !93
  %345 = extractelement <16 x float> %344, i64 8, !dbg !93
  %346 = extractelement <16 x float> %344, i64 15, !dbg !93
  %347 = shufflevector <16 x float> %344, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !89
  %348 = fmul <8 x float> %283, %347, !dbg !89
  %349 = fmul float %154, %345, !dbg !89
  %350 = shufflevector <16 x float> %344, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !89
  %351 = fmul <4 x float> %285, %350, !dbg !89
  %352 = fmul float %154, %346, !dbg !89
  %353 = extractelement <8 x float> %348, i64 0, !dbg !94
  %354 = extractelement <8 x float> %348, i64 1, !dbg !94
  %355 = tail call float @llvm.maxnum.f32(float %353, float %354), !dbg !95
  %356 = extractelement <8 x float> %348, i64 2, !dbg !94
  %357 = tail call float @llvm.maxnum.f32(float %355, float %356), !dbg !95
  %358 = extractelement <8 x float> %348, i64 3, !dbg !94
  %359 = extractelement <8 x float> %348, i64 4, !dbg !94
  %360 = tail call float @llvm.maxnum.f32(float %358, float %359), !dbg !95
  %361 = extractelement <8 x float> %348, i64 5, !dbg !94
  %362 = tail call float @llvm.maxnum.f32(float %360, float %361), !dbg !95
  %363 = extractelement <8 x float> %348, i64 6, !dbg !94
  %364 = extractelement <8 x float> %348, i64 7, !dbg !94
  %365 = tail call float @llvm.maxnum.f32(float %363, float %364), !dbg !95
  %366 = tail call float @llvm.maxnum.f32(float %365, float %349), !dbg !95
  %367 = extractelement <4 x float> %351, i64 0, !dbg !94
  %368 = extractelement <4 x float> %351, i64 1, !dbg !94
  %369 = tail call float @llvm.maxnum.f32(float %367, float %368), !dbg !95
  %370 = extractelement <4 x float> %351, i64 2, !dbg !94
  %371 = tail call float @llvm.maxnum.f32(float %369, float %370), !dbg !95
  %372 = extractelement <4 x float> %351, i64 3, !dbg !94
  %373 = tail call float @llvm.maxnum.f32(float %357, float %362), !dbg !95
  %374 = tail call float @llvm.maxnum.f32(float %373, float %366), !dbg !95
  %375 = shufflevector <8 x float> %348, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %376 = shufflevector <8 x float> %348, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %377 = shufflevector <8 x float> %348, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !94
  %378 = shufflevector <8 x float> %348, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !94
  %379 = shufflevector <4 x float> %351, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %380 = shufflevector <4 x float> %351, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %381 = shufflevector <16 x float> %344, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !89
  %382 = fmul <2 x float> %287, %381, !dbg !89
  %383 = extractelement <2 x float> %382, i64 0, !dbg !95
  %384 = tail call float @llvm.maxnum.f32(float %372, float %383), !dbg !95
  %385 = extractelement <2 x float> %382, i64 1, !dbg !95
  %386 = tail call float @llvm.maxnum.f32(float %384, float %385), !dbg !95
  %387 = tail call float @llvm.maxnum.f32(float %371, float %386), !dbg !95
  %388 = tail call float @llvm.maxnum.f32(float %387, float %352), !dbg !95
  %389 = tail call float @llvm.maxnum.f32(float %374, float %388), !dbg !95
  %390 = bitcast float %389 to i32, !dbg !98
  %391 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %390, i32 %390, i1 false, i1 false), !dbg !98
  %392 = extractvalue { i32, i32 } %391, 0, !dbg !98
  %393 = extractvalue { i32, i32 } %391, 1, !dbg !98
  %394 = bitcast i32 %392 to float, !dbg !98
  %395 = bitcast i32 %393 to float, !dbg !98
  %396 = tail call float @llvm.maxnum.f32(float %394, float %395), !dbg !95
  %397 = tail call float @llvm.maxnum.f32(float %296, float %396), !dbg !100
  %398 = insertelement <2 x float> poison, float %397, i64 0, !dbg !94
  %399 = shufflevector <2 x float> %398, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !94
  %400 = fsub <2 x float> %375, %399, !dbg !94
  %401 = fsub <2 x float> %376, %399, !dbg !94
  %402 = fsub <2 x float> %377, %399, !dbg !94
  %403 = fsub <2 x float> %378, %399, !dbg !94
  %404 = fsub float %349, %397, !dbg !94
  %405 = fsub <2 x float> %379, %399, !dbg !94
  %406 = fsub <2 x float> %380, %399, !dbg !94
  %407 = fsub <2 x float> %382, %399, !dbg !94
  %408 = fsub float %352, %397, !dbg !94
  %409 = extractelement <2 x float> %400, i64 0, !dbg !101
  %410 = tail call float @llvm.amdgcn.exp2.f32(float %409), !dbg !101
  %411 = extractelement <2 x float> %400, i64 1, !dbg !101
  %412 = tail call float @llvm.amdgcn.exp2.f32(float %411), !dbg !101
  %413 = extractelement <2 x float> %401, i64 0, !dbg !101
  %414 = tail call float @llvm.amdgcn.exp2.f32(float %413), !dbg !101
  %415 = extractelement <2 x float> %401, i64 1, !dbg !101
  %416 = tail call float @llvm.amdgcn.exp2.f32(float %415), !dbg !101
  %417 = extractelement <2 x float> %402, i64 0, !dbg !101
  %418 = tail call float @llvm.amdgcn.exp2.f32(float %417), !dbg !101
  %419 = extractelement <2 x float> %402, i64 1, !dbg !101
  %420 = tail call float @llvm.amdgcn.exp2.f32(float %419), !dbg !101
  %421 = extractelement <2 x float> %403, i64 0, !dbg !101
  %422 = tail call float @llvm.amdgcn.exp2.f32(float %421), !dbg !101
  %423 = extractelement <2 x float> %403, i64 1, !dbg !101
  %424 = tail call float @llvm.amdgcn.exp2.f32(float %423), !dbg !101
  %425 = tail call float @llvm.amdgcn.exp2.f32(float %404), !dbg !101
  %426 = extractelement <2 x float> %405, i64 0, !dbg !101
  %427 = tail call float @llvm.amdgcn.exp2.f32(float %426), !dbg !101
  %428 = extractelement <2 x float> %405, i64 1, !dbg !101
  %429 = tail call float @llvm.amdgcn.exp2.f32(float %428), !dbg !101
  %430 = extractelement <2 x float> %406, i64 0, !dbg !101
  %431 = tail call float @llvm.amdgcn.exp2.f32(float %430), !dbg !101
  %432 = extractelement <2 x float> %406, i64 1, !dbg !101
  %433 = tail call float @llvm.amdgcn.exp2.f32(float %432), !dbg !101
  %434 = extractelement <2 x float> %407, i64 0, !dbg !101
  %435 = tail call float @llvm.amdgcn.exp2.f32(float %434), !dbg !101
  %436 = extractelement <2 x float> %407, i64 1, !dbg !101
  %437 = tail call float @llvm.amdgcn.exp2.f32(float %436), !dbg !101
  %438 = tail call float @llvm.amdgcn.exp2.f32(float %408), !dbg !101
  %439 = insertelement <2 x float> poison, float %410, i64 0, !dbg !102
  %440 = insertelement <2 x float> %439, float %412, i64 1, !dbg !102
  %441 = insertelement <2 x float> poison, float %414, i64 0, !dbg !102
  %442 = insertelement <2 x float> %441, float %416, i64 1, !dbg !102
  %443 = insertelement <2 x float> poison, float %418, i64 0, !dbg !102
  %444 = insertelement <2 x float> %443, float %420, i64 1, !dbg !102
  %445 = insertelement <2 x float> poison, float %422, i64 0, !dbg !102
  %446 = insertelement <2 x float> %445, float %424, i64 1, !dbg !102
  %447 = insertelement <2 x float> poison, float %425, i64 0, !dbg !102
  %448 = insertelement <2 x float> %447, float %427, i64 1, !dbg !102
  %449 = insertelement <2 x float> poison, float %429, i64 0, !dbg !102
  %450 = insertelement <2 x float> %449, float %431, i64 1, !dbg !102
  %451 = insertelement <2 x float> poison, float %433, i64 0, !dbg !102
  %452 = insertelement <2 x float> %451, float %435, i64 1, !dbg !102
  %453 = insertelement <2 x float> poison, float %437, i64 0, !dbg !102
  %454 = insertelement <2 x float> %453, float %438, i64 1, !dbg !102
  %455 = fadd <2 x float> %440, %442, !dbg !102
  %456 = fadd <2 x float> %444, %446, !dbg !102
  %457 = fadd <2 x float> %448, %450, !dbg !102
  %458 = fadd <2 x float> %452, %454, !dbg !102
  %459 = fadd <2 x float> %455, %456, !dbg !102
  %460 = fadd <2 x float> %457, %458, !dbg !102
  %461 = fadd <2 x float> %459, %460, !dbg !102
  %shift = shufflevector <2 x float> %461, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !104
  %foldExtExtBinop = fadd <2 x float> %461, %shift, !dbg !104
  %bc = bitcast <2 x float> %foldExtExtBinop to <2 x i32>, !dbg !102
  %462 = extractelement <2 x i32> %bc, i64 0, !dbg !102
  %463 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %462, i32 %462, i1 false, i1 false), !dbg !102
  %464 = extractvalue { i32, i32 } %463, 0, !dbg !102
  %465 = extractvalue { i32, i32 } %463, 1, !dbg !102
  %466 = bitcast i32 %464 to float, !dbg !102
  %467 = bitcast i32 %465 to float, !dbg !102
  %468 = fadd float %466, %467, !dbg !104
  %469 = fsub float %296, %397, !dbg !105
  %470 = tail call float @llvm.amdgcn.exp2.f32(float %469), !dbg !106
  %471 = insertelement <1 x float> poison, float %470, i64 0, !dbg !107
  store <1 x float> %471, ptr addrspace(3) %238, align 4, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !107
  tail call void @llvm.amdgcn.s.barrier(), !dbg !107
  %472 = load <1 x float>, ptr addrspace(3) %gep, align 4, !dbg !107
  %473 = load <1 x float>, ptr addrspace(3) %240, align 4, !dbg !107
  %474 = fmul float %297, %470, !dbg !108
  %475 = fadd float %468, %474, !dbg !108
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %476 = shufflevector <2 x float> %439, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %477 = fptrunc <1 x float> %476 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %477, ptr addrspace(3) %246, align 2, !dbg !109
  %478 = shufflevector <2 x float> %447, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %479 = fptrunc <1 x float> %478 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %479, ptr addrspace(3) %247, align 2, !dbg !109
  %480 = shufflevector <2 x float> %440, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %481 = fptrunc <1 x float> %480 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %481, ptr addrspace(3) %249, align 2, !dbg !109
  %482 = shufflevector <2 x float> %448, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %483 = fptrunc <1 x float> %482 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %483, ptr addrspace(3) %250, align 2, !dbg !109
  %484 = shufflevector <2 x float> %441, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %485 = fptrunc <1 x float> %484 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %485, ptr addrspace(3) %252, align 2, !dbg !109
  %486 = shufflevector <2 x float> %449, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %487 = fptrunc <1 x float> %486 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %487, ptr addrspace(3) %253, align 2, !dbg !109
  %488 = shufflevector <2 x float> %442, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %489 = fptrunc <1 x float> %488 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %489, ptr addrspace(3) %255, align 2, !dbg !109
  %490 = shufflevector <2 x float> %450, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %491 = fptrunc <1 x float> %490 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %491, ptr addrspace(3) %256, align 2, !dbg !109
  %492 = shufflevector <2 x float> %443, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %493 = fptrunc <1 x float> %492 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %493, ptr addrspace(3) %258, align 2, !dbg !109
  %494 = shufflevector <2 x float> %451, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %495 = fptrunc <1 x float> %494 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %495, ptr addrspace(3) %259, align 2, !dbg !109
  %496 = shufflevector <2 x float> %444, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %497 = fptrunc <1 x float> %496 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %497, ptr addrspace(3) %261, align 2, !dbg !109
  %498 = shufflevector <2 x float> %452, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %499 = fptrunc <1 x float> %498 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %499, ptr addrspace(3) %262, align 2, !dbg !109
  %500 = shufflevector <2 x float> %445, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %501 = fptrunc <1 x float> %500 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %501, ptr addrspace(3) %264, align 2, !dbg !109
  %502 = shufflevector <2 x float> %453, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %503 = fptrunc <1 x float> %502 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %503, ptr addrspace(3) %265, align 2, !dbg !109
  %504 = shufflevector <2 x float> %446, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %505 = fptrunc <1 x float> %504 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %505, ptr addrspace(3) %267, align 2, !dbg !109
  %506 = shufflevector <2 x float> %454, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %507 = fptrunc <1 x float> %506 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %507, ptr addrspace(3) %268, align 2, !dbg !109
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %508 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %277), !dbg !109
  %509 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %278), !dbg !109
  %510 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %279), !dbg !109
  %511 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %280), !dbg !109
  %512 = shufflevector <4 x bfloat> %508, <4 x bfloat> %509, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %513 = shufflevector <4 x bfloat> %510, <4 x bfloat> %511, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %514 = shufflevector <4 x bfloat> %340, <4 x bfloat> %342, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %515 = shufflevector <2 x float> %301, <2 x float> %302, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %516 = shufflevector <1 x float> %472, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %517 = fmul <4 x float> %515, %516, !dbg !110
  %518 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %514, <8 x bfloat> %512, <4 x float> %517, i32 0, i32 0, i32 0), !dbg !110
  %519 = shufflevector <2 x float> %299, <2 x float> %300, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %520 = shufflevector <1 x float> %473, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %521 = fmul <4 x float> %519, %520, !dbg !110
  %522 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %514, <8 x bfloat> %513, <4 x float> %521, i32 0, i32 0, i32 0), !dbg !110
  tail call void @llvm.amdgcn.wait.asyncmark(i16 3), !dbg !77
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !77
  tail call void @llvm.amdgcn.s.barrier(), !dbg !77
  %523 = add nuw nsw i32 %298, 32, !dbg !76
  %524 = icmp slt i32 %523, %207, !dbg !76
  %525 = shufflevector <4 x float> %522, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %526 = shufflevector <4 x float> %522, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %527 = shufflevector <4 x float> %518, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %528 = shufflevector <4 x float> %518, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  br i1 %524, label %288, label %._crit_edge, !dbg !76

._crit_edge:                                      ; preds = %288, %.._crit_edge_crit_edge
  %.pre-phi141 = phi i32 [ %.pre140, %.._crit_edge_crit_edge ], [ %213, %288 ], !dbg !77
  %.pre-phi139 = phi i32 [ %.pre138, %.._crit_edge_crit_edge ], [ %212, %288 ], !dbg !77
  %.lcssa98 = phi float [ 1.000000e+00, %.._crit_edge_crit_edge ], [ %475, %288 ]
  %.lcssa97 = phi float [ 0xFFF0000000000000, %.._crit_edge_crit_edge ], [ %397, %288 ]
  %529 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 14400), %.._crit_edge_crit_edge ], [ %293, %288 ], !dbg !111
  %.lcssa95 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 15424), %.._crit_edge_crit_edge ], [ %316, %288 ], !dbg !111
  %530 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 11328), %.._crit_edge_crit_edge ], [ %291, %288 ], !dbg !111
  %.lcssa93 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 12352), %.._crit_edge_crit_edge ], [ %326, %288 ], !dbg !111
  %531 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), %.._crit_edge_crit_edge ], [ %289, %288 ], !dbg !111
  %.lcssa91 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 9248), %.._crit_edge_crit_edge ], [ %336, %288 ], !dbg !111
  %532 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %525, %288 ]
  %533 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %526, %288 ]
  %534 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %527, %288 ]
  %535 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %528, %288 ]
  %536 = phi <2 x i32> [ %210, %.._crit_edge_crit_edge ], [ %220, %288 ], !dbg !84
  %537 = or disjoint i32 %153, 31, !dbg !76
  %538 = icmp sgt i32 %537, 63, !dbg !76
  %539 = icmp eq i32 %.pre-phi141, 0, !dbg !77
  %540 = select i1 %539, i32 0, i32 272, !dbg !77
  %541 = extractelement <2 x i32> %536, i64 0, !dbg !77
  %542 = xor i32 %540, %541, !dbg !77
  %543 = or disjoint i32 %542, %.pre-phi139, !dbg !77
  %544 = extractelement <2 x i32> %536, i64 1, !dbg !84
  %545 = getelementptr inbounds nuw i8, ptr addrspace(3) %531, i32 %544, !dbg !84
  %546 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %545), !dbg !84, !alias.scope !90, !noalias !79
  %547 = or disjoint i32 %544, 512, !dbg !84
  %548 = getelementptr inbounds nuw i8, ptr addrspace(3) %531, i32 %547, !dbg !84
  %549 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %548), !dbg !84, !alias.scope !90, !noalias !79
  br i1 %157, label %550, label %572, !dbg !93

550:                                              ; preds = %._crit_edge
  %551 = getelementptr inbounds nuw i8, ptr addrspace(3) %530, i32 %543, !dbg !82
  %552 = load <8 x bfloat>, ptr addrspace(3) %551, align 16, !dbg !82, !alias.scope !90, !noalias !79
  %553 = getelementptr inbounds nuw i8, ptr addrspace(3) %529, i32 %543, !dbg !77
  %554 = load <8 x bfloat>, ptr addrspace(3) %553, align 16, !dbg !77, !alias.scope !90, !noalias !79
  %555 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %556 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %557 = shufflevector <8 x bfloat> %555, <8 x bfloat> %556, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !92
  %558 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %559 = shufflevector <8 x bfloat> %557, <8 x bfloat> %558, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !92
  %560 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %552, <8 x bfloat> %559, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !92
  %561 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %562 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %563 = shufflevector <8 x bfloat> %561, <8 x bfloat> %562, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !93
  %564 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %565 = shufflevector <8 x bfloat> %563, <8 x bfloat> %564, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !93
  %566 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %554, <8 x bfloat> %565, <16 x float> %560, i32 0, i32 0, i32 0), !dbg !93
  %567 = extractelement <16 x float> %566, i64 8, !dbg !93
  %568 = extractelement <16 x float> %566, i64 15, !dbg !93
  %569 = shufflevector <16 x float> %566, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !111
  %570 = shufflevector <16 x float> %566, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !111
  %571 = shufflevector <16 x float> %566, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !111
  br label %572, !dbg !93

572:                                              ; preds = %550, %._crit_edge
  %573 = phi float [ %567, %550 ], [ 0.000000e+00, %._crit_edge ], !dbg !111
  %574 = phi float [ %568, %550 ], [ 0.000000e+00, %._crit_edge ], !dbg !111
  %575 = phi <8 x float> [ %569, %550 ], [ zeroinitializer, %._crit_edge ], !dbg !111
  %576 = phi <4 x float> [ %570, %550 ], [ zeroinitializer, %._crit_edge ], !dbg !111
  %577 = phi <2 x float> [ %571, %550 ], [ zeroinitializer, %._crit_edge ], !dbg !111
  %578 = insertelement <8 x float> poison, float %154, i64 0, !dbg !89
  %579 = shufflevector <8 x float> %578, <8 x float> poison, <8 x i32> zeroinitializer, !dbg !89
  %580 = fmul <8 x float> %579, %575, !dbg !89
  %581 = fmul float %154, %573, !dbg !89
  %582 = insertelement <4 x float> poison, float %154, i64 0, !dbg !89
  %583 = shufflevector <4 x float> %582, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !89
  %584 = fmul <4 x float> %583, %576, !dbg !89
  %585 = insertelement <2 x float> poison, float %154, i64 0, !dbg !89
  %586 = shufflevector <2 x float> %585, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !89
  %587 = fmul <2 x float> %586, %577, !dbg !89
  %588 = fmul float %154, %574, !dbg !89
  %589 = extractelement <8 x float> %580, i64 0, !dbg !94
  %590 = extractelement <8 x float> %580, i64 1, !dbg !94
  %591 = tail call float @llvm.maxnum.f32(float %589, float %590), !dbg !95
  %592 = extractelement <8 x float> %580, i64 2, !dbg !94
  %593 = tail call float @llvm.maxnum.f32(float %591, float %592), !dbg !95
  %594 = extractelement <8 x float> %580, i64 3, !dbg !94
  %595 = extractelement <8 x float> %580, i64 4, !dbg !94
  %596 = tail call float @llvm.maxnum.f32(float %594, float %595), !dbg !95
  %597 = extractelement <8 x float> %580, i64 5, !dbg !94
  %598 = tail call float @llvm.maxnum.f32(float %596, float %597), !dbg !95
  %599 = extractelement <8 x float> %580, i64 6, !dbg !94
  %600 = extractelement <8 x float> %580, i64 7, !dbg !94
  %601 = tail call float @llvm.maxnum.f32(float %599, float %600), !dbg !95
  %602 = tail call float @llvm.maxnum.f32(float %601, float %581), !dbg !95
  %603 = extractelement <4 x float> %584, i64 0, !dbg !94
  %604 = extractelement <4 x float> %584, i64 1, !dbg !94
  %605 = tail call float @llvm.maxnum.f32(float %603, float %604), !dbg !95
  %606 = extractelement <4 x float> %584, i64 2, !dbg !94
  %607 = tail call float @llvm.maxnum.f32(float %605, float %606), !dbg !95
  %608 = extractelement <4 x float> %584, i64 3, !dbg !94
  %609 = extractelement <2 x float> %587, i64 0, !dbg !95
  %610 = tail call float @llvm.maxnum.f32(float %608, float %609), !dbg !95
  %611 = extractelement <2 x float> %587, i64 1, !dbg !95
  %612 = tail call float @llvm.maxnum.f32(float %610, float %611), !dbg !95
  %613 = tail call float @llvm.maxnum.f32(float %593, float %598), !dbg !95
  %614 = tail call float @llvm.maxnum.f32(float %613, float %602), !dbg !95
  %615 = tail call float @llvm.maxnum.f32(float %607, float %612), !dbg !95
  %616 = tail call float @llvm.maxnum.f32(float %615, float %588), !dbg !95
  %617 = tail call float @llvm.maxnum.f32(float %614, float %616), !dbg !95
  %618 = bitcast float %617 to i32, !dbg !98
  %619 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %618, i32 %618, i1 false, i1 false), !dbg !98
  %620 = extractvalue { i32, i32 } %619, 0, !dbg !98
  %621 = extractvalue { i32, i32 } %619, 1, !dbg !98
  %622 = bitcast i32 %620 to float, !dbg !98
  %623 = bitcast i32 %621 to float, !dbg !98
  %624 = tail call float @llvm.maxnum.f32(float %622, float %623), !dbg !95
  %625 = tail call float @llvm.maxnum.f32(float %.lcssa97, float %624), !dbg !100
  %626 = shufflevector <8 x float> %580, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %627 = insertelement <2 x float> poison, float %625, i64 0, !dbg !94
  %628 = shufflevector <2 x float> %627, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !94
  %629 = fsub <2 x float> %626, %628, !dbg !94
  %630 = shufflevector <8 x float> %580, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %631 = fsub <2 x float> %630, %628, !dbg !94
  %632 = shufflevector <8 x float> %580, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !94
  %633 = fsub <2 x float> %632, %628, !dbg !94
  %634 = shufflevector <8 x float> %580, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !94
  %635 = fsub <2 x float> %634, %628, !dbg !94
  %636 = fsub float %581, %625, !dbg !94
  %637 = shufflevector <4 x float> %584, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %638 = fsub <2 x float> %637, %628, !dbg !94
  %639 = shufflevector <4 x float> %584, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %640 = fsub <2 x float> %639, %628, !dbg !94
  %641 = fsub <2 x float> %587, %628, !dbg !94
  %642 = fsub float %588, %625, !dbg !94
  %643 = extractelement <2 x float> %629, i64 0, !dbg !101
  %644 = tail call float @llvm.amdgcn.exp2.f32(float %643), !dbg !101
  %645 = extractelement <2 x float> %629, i64 1, !dbg !101
  %646 = tail call float @llvm.amdgcn.exp2.f32(float %645), !dbg !101
  %647 = extractelement <2 x float> %631, i64 0, !dbg !101
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !101
  %649 = extractelement <2 x float> %631, i64 1, !dbg !101
  %650 = tail call float @llvm.amdgcn.exp2.f32(float %649), !dbg !101
  %651 = extractelement <2 x float> %633, i64 0, !dbg !101
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !101
  %653 = extractelement <2 x float> %633, i64 1, !dbg !101
  %654 = tail call float @llvm.amdgcn.exp2.f32(float %653), !dbg !101
  %655 = extractelement <2 x float> %635, i64 0, !dbg !101
  %656 = tail call float @llvm.amdgcn.exp2.f32(float %655), !dbg !101
  %657 = extractelement <2 x float> %635, i64 1, !dbg !101
  %658 = tail call float @llvm.amdgcn.exp2.f32(float %657), !dbg !101
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %636), !dbg !101
  %660 = extractelement <2 x float> %638, i64 0, !dbg !101
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %660), !dbg !101
  %662 = extractelement <2 x float> %638, i64 1, !dbg !101
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %662), !dbg !101
  %664 = extractelement <2 x float> %640, i64 0, !dbg !101
  %665 = tail call float @llvm.amdgcn.exp2.f32(float %664), !dbg !101
  %666 = extractelement <2 x float> %640, i64 1, !dbg !101
  %667 = tail call float @llvm.amdgcn.exp2.f32(float %666), !dbg !101
  %668 = extractelement <2 x float> %641, i64 0, !dbg !101
  %669 = tail call float @llvm.amdgcn.exp2.f32(float %668), !dbg !101
  %670 = extractelement <2 x float> %641, i64 1, !dbg !101
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %670), !dbg !101
  %672 = tail call float @llvm.amdgcn.exp2.f32(float %642), !dbg !101
  %673 = insertelement <2 x float> poison, float %644, i64 0, !dbg !102
  %674 = insertelement <2 x float> %673, float %646, i64 1, !dbg !102
  %675 = insertelement <2 x float> poison, float %648, i64 0, !dbg !102
  %676 = insertelement <2 x float> %675, float %650, i64 1, !dbg !102
  %677 = insertelement <2 x float> poison, float %652, i64 0, !dbg !102
  %678 = insertelement <2 x float> %677, float %654, i64 1, !dbg !102
  %679 = insertelement <2 x float> poison, float %656, i64 0, !dbg !102
  %680 = insertelement <2 x float> %679, float %658, i64 1, !dbg !102
  %681 = insertelement <2 x float> poison, float %659, i64 0, !dbg !102
  %682 = insertelement <2 x float> %681, float %661, i64 1, !dbg !102
  %683 = insertelement <2 x float> poison, float %663, i64 0, !dbg !102
  %684 = insertelement <2 x float> %683, float %665, i64 1, !dbg !102
  %685 = insertelement <2 x float> poison, float %667, i64 0, !dbg !102
  %686 = insertelement <2 x float> %685, float %669, i64 1, !dbg !102
  %687 = insertelement <2 x float> poison, float %671, i64 0, !dbg !102
  %688 = insertelement <2 x float> %687, float %672, i64 1, !dbg !102
  %689 = fadd <2 x float> %674, %676, !dbg !102
  %690 = fadd <2 x float> %678, %680, !dbg !102
  %691 = fadd <2 x float> %682, %684, !dbg !102
  %692 = fadd <2 x float> %686, %688, !dbg !102
  %693 = fadd <2 x float> %689, %690, !dbg !102
  %694 = fadd <2 x float> %691, %692, !dbg !102
  %695 = fadd <2 x float> %693, %694, !dbg !102
  %shift186 = shufflevector <2 x float> %695, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !104
  %foldExtExtBinop187 = fadd <2 x float> %695, %shift186, !dbg !104
  %bc195 = bitcast <2 x float> %foldExtExtBinop187 to <2 x i32>, !dbg !102
  %696 = extractelement <2 x i32> %bc195, i64 0, !dbg !102
  %697 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %696, i32 %696, i1 false, i1 false), !dbg !102
  %698 = fsub float %.lcssa97, %625, !dbg !105
  %699 = tail call float @llvm.amdgcn.exp2.f32(float %698), !dbg !106
  %700 = and i32 %38, 31, !dbg !107
  %701 = shl nuw nsw i32 %700, 2, !dbg !107
  %702 = shl nuw nsw i32 %41, 7, !dbg !107
  %703 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %701, !dbg !107
  %704 = getelementptr inbounds nuw i8, ptr addrspace(3) %703, i32 %702, !dbg !107
  %705 = insertelement <1 x float> poison, float %699, i64 0, !dbg !107
  store <1 x float> %705, ptr addrspace(3) %704, align 4, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !107
  tail call void @llvm.amdgcn.s.barrier(), !dbg !107
  %706 = shl nuw nsw i32 %57, 2, !dbg !107
  %707 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %706, !dbg !107
  %708 = getelementptr inbounds nuw i8, ptr addrspace(3) %707, i32 %56, !dbg !107
  %709 = load <1 x float>, ptr addrspace(3) %708, align 4, !dbg !107
  %710 = getelementptr inbounds nuw i8, ptr addrspace(3) %708, i32 256, !dbg !107
  %711 = load <1 x float>, ptr addrspace(3) %710, align 4, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %712 = shl nuw nsw i32 %700, 1, !dbg !109
  %713 = icmp eq i32 %86, 0, !dbg !109
  %714 = select i1 %713, i32 0, i32 1056, !dbg !109
  %715 = or disjoint i32 %56, %712, !dbg !109
  %716 = xor i32 %715, %714, !dbg !109
  %717 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %716, !dbg !109
  %718 = shufflevector <2 x float> %673, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %719 = fptrunc <1 x float> %718 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %719, ptr addrspace(3) %717, align 2, !dbg !109
  %720 = getelementptr inbounds nuw i8, ptr addrspace(3) %717, i32 4096, !dbg !109
  %721 = shufflevector <2 x float> %681, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %722 = fptrunc <1 x float> %721 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %722, ptr addrspace(3) %720, align 2, !dbg !109
  %723 = xor i32 %716, 264, !dbg !109
  %724 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %723, !dbg !109
  %725 = shufflevector <2 x float> %674, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %726 = fptrunc <1 x float> %725 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %726, ptr addrspace(3) %724, align 2, !dbg !109
  %727 = getelementptr inbounds nuw i8, ptr addrspace(3) %724, i32 4096, !dbg !109
  %728 = shufflevector <2 x float> %682, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %729 = fptrunc <1 x float> %728 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %729, ptr addrspace(3) %727, align 2, !dbg !109
  %730 = xor i32 %716, 528, !dbg !109
  %731 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %730, !dbg !109
  %732 = shufflevector <2 x float> %675, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %733 = fptrunc <1 x float> %732 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %733, ptr addrspace(3) %731, align 2, !dbg !109
  %734 = getelementptr inbounds nuw i8, ptr addrspace(3) %731, i32 4096, !dbg !109
  %735 = shufflevector <2 x float> %683, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %736 = fptrunc <1 x float> %735 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %736, ptr addrspace(3) %734, align 2, !dbg !109
  %737 = xor i32 %716, 792, !dbg !109
  %738 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %737, !dbg !109
  %739 = shufflevector <2 x float> %676, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %740 = fptrunc <1 x float> %739 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %740, ptr addrspace(3) %738, align 2, !dbg !109
  %741 = getelementptr inbounds nuw i8, ptr addrspace(3) %738, i32 4096, !dbg !109
  %742 = shufflevector <2 x float> %684, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %743 = fptrunc <1 x float> %742 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %743, ptr addrspace(3) %741, align 2, !dbg !109
  %744 = xor i32 %716, 2112, !dbg !109
  %745 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %744, !dbg !109
  %746 = shufflevector <2 x float> %677, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %747 = fptrunc <1 x float> %746 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %747, ptr addrspace(3) %745, align 2, !dbg !109
  %748 = getelementptr inbounds nuw i8, ptr addrspace(3) %745, i32 4096, !dbg !109
  %749 = shufflevector <2 x float> %685, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %750 = fptrunc <1 x float> %749 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %750, ptr addrspace(3) %748, align 2, !dbg !109
  %751 = xor i32 %716, 2376, !dbg !109
  %752 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %751, !dbg !109
  %753 = shufflevector <2 x float> %678, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %754 = fptrunc <1 x float> %753 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %754, ptr addrspace(3) %752, align 2, !dbg !109
  %755 = getelementptr inbounds nuw i8, ptr addrspace(3) %752, i32 4096, !dbg !109
  %756 = shufflevector <2 x float> %686, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %757 = fptrunc <1 x float> %756 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %757, ptr addrspace(3) %755, align 2, !dbg !109
  %758 = xor i32 %716, 2640, !dbg !109
  %759 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %758, !dbg !109
  %760 = shufflevector <2 x float> %679, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %761 = fptrunc <1 x float> %760 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %761, ptr addrspace(3) %759, align 2, !dbg !109
  %762 = getelementptr inbounds nuw i8, ptr addrspace(3) %759, i32 4096, !dbg !109
  %763 = shufflevector <2 x float> %687, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %764 = fptrunc <1 x float> %763 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %764, ptr addrspace(3) %762, align 2, !dbg !109
  %765 = xor i32 %716, 2904, !dbg !109
  %766 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %765, !dbg !109
  %767 = shufflevector <2 x float> %680, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %768 = fptrunc <1 x float> %767 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %768, ptr addrspace(3) %766, align 2, !dbg !109
  %769 = getelementptr inbounds nuw i8, ptr addrspace(3) %766, i32 4096, !dbg !109
  %770 = shufflevector <2 x float> %688, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %771 = fptrunc <1 x float> %770 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %771, ptr addrspace(3) %769, align 2, !dbg !109
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %772 = and i32 %38, 60, !dbg !109
  %773 = shl nuw nsw i32 %772, 6, !dbg !109
  %774 = and i32 %60, 24, !dbg !109
  %775 = shl nuw nsw i32 %772, 1, !dbg !109
  %776 = shl nuw nsw i32 %41, 5, !dbg !109
  %777 = or disjoint i32 %773, %774, !dbg !109
  %778 = xor i32 %777, %775, !dbg !109
  %779 = xor i32 %778, %776, !dbg !109
  %780 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %779, !dbg !109
  %781 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %780), !dbg !109
  %782 = getelementptr inbounds nuw i8, ptr addrspace(3) %780, i32 4096, !dbg !109
  %783 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %782), !dbg !109
  %784 = getelementptr inbounds nuw i8, ptr addrspace(3) %780, i32 128, !dbg !109
  %785 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %784), !dbg !109
  %786 = getelementptr inbounds nuw i8, ptr addrspace(3) %780, i32 4224, !dbg !109
  %787 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %786), !dbg !109
  br i1 %157, label %788, label %811, !dbg !110

788:                                              ; preds = %572
  %789 = fmul float %.lcssa98, %699, !dbg !108
  %790 = extractvalue { i32, i32 } %697, 0, !dbg !102
  %791 = bitcast i32 %790 to float, !dbg !102
  %792 = extractvalue { i32, i32 } %697, 1, !dbg !102
  %793 = bitcast i32 %792 to float, !dbg !102
  %794 = fadd float %791, %793, !dbg !104
  %795 = fadd float %794, %789, !dbg !108
  %796 = shufflevector <4 x bfloat> %781, <4 x bfloat> %783, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %797 = shufflevector <4 x bfloat> %785, <4 x bfloat> %787, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %798 = shufflevector <4 x bfloat> %546, <4 x bfloat> %549, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %799 = shufflevector <2 x float> %535, <2 x float> %534, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %800 = shufflevector <1 x float> %709, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %801 = fmul <4 x float> %799, %800, !dbg !110
  %802 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %798, <8 x bfloat> %796, <4 x float> %801, i32 0, i32 0, i32 0), !dbg !110
  %803 = shufflevector <2 x float> %533, <2 x float> %532, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %804 = shufflevector <1 x float> %711, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %805 = fmul <4 x float> %803, %804, !dbg !110
  %806 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %798, <8 x bfloat> %797, <4 x float> %805, i32 0, i32 0, i32 0), !dbg !110
  %807 = shufflevector <4 x float> %806, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %808 = shufflevector <4 x float> %806, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %809 = shufflevector <4 x float> %802, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %810 = shufflevector <4 x float> %802, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  br label %811, !dbg !110

811:                                              ; preds = %788, %572
  %812 = phi float [ %625, %788 ], [ %.lcssa97, %572 ]
  %813 = phi float [ %795, %788 ], [ %.lcssa98, %572 ]
  %814 = phi <2 x float> [ %807, %788 ], [ %532, %572 ]
  %815 = phi <2 x float> [ %808, %788 ], [ %533, %572 ]
  %816 = phi <2 x float> [ %809, %788 ], [ %534, %572 ]
  %817 = phi <2 x float> [ %810, %788 ], [ %535, %572 ]
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !77
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !77
  tail call void @llvm.amdgcn.s.barrier(), !dbg !77
  %818 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa91, i32 %544, !dbg !84
  %819 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %818), !dbg !84, !alias.scope !90, !noalias !79
  %820 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa91, i32 %547, !dbg !84
  %821 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %820), !dbg !84, !alias.scope !90, !noalias !79
  br i1 %538, label %822, label %844, !dbg !93

822:                                              ; preds = %811
  %823 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa93, i32 %543, !dbg !82
  %824 = load <8 x bfloat>, ptr addrspace(3) %823, align 16, !dbg !82, !alias.scope !90, !noalias !79
  %825 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa95, i32 %543, !dbg !77
  %826 = load <8 x bfloat>, ptr addrspace(3) %825, align 16, !dbg !77, !alias.scope !90, !noalias !79
  %827 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %828 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %829 = shufflevector <8 x bfloat> %827, <8 x bfloat> %828, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !92
  %830 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !92
  %831 = shufflevector <8 x bfloat> %829, <8 x bfloat> %830, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !92
  %832 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %824, <8 x bfloat> %831, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !92
  %833 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %834 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %835 = shufflevector <8 x bfloat> %833, <8 x bfloat> %834, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !93
  %836 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !93
  %837 = shufflevector <8 x bfloat> %835, <8 x bfloat> %836, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !93
  %838 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %826, <8 x bfloat> %837, <16 x float> %832, i32 0, i32 0, i32 0), !dbg !93
  %839 = extractelement <16 x float> %838, i64 8, !dbg !93
  %840 = extractelement <16 x float> %838, i64 15, !dbg !93
  %841 = shufflevector <16 x float> %838, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !111
  %842 = shufflevector <16 x float> %838, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !111
  %843 = shufflevector <16 x float> %838, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !111
  br label %844, !dbg !93

844:                                              ; preds = %822, %811
  %845 = phi float [ %839, %822 ], [ 0.000000e+00, %811 ], !dbg !111
  %846 = phi float [ %840, %822 ], [ 0.000000e+00, %811 ], !dbg !111
  %847 = phi <8 x float> [ %841, %822 ], [ zeroinitializer, %811 ], !dbg !111
  %848 = phi <4 x float> [ %842, %822 ], [ zeroinitializer, %811 ], !dbg !111
  %849 = phi <2 x float> [ %843, %822 ], [ zeroinitializer, %811 ], !dbg !111
  %850 = fmul <8 x float> %579, %847, !dbg !89
  %851 = fmul float %154, %845, !dbg !89
  %852 = fmul <4 x float> %583, %848, !dbg !89
  %853 = fmul <2 x float> %586, %849, !dbg !89
  %854 = fmul float %154, %846, !dbg !89
  %855 = extractelement <8 x float> %850, i64 0, !dbg !94
  %856 = extractelement <8 x float> %850, i64 1, !dbg !94
  %857 = tail call float @llvm.maxnum.f32(float %855, float %856), !dbg !95
  %858 = extractelement <8 x float> %850, i64 2, !dbg !94
  %859 = tail call float @llvm.maxnum.f32(float %857, float %858), !dbg !95
  %860 = extractelement <8 x float> %850, i64 3, !dbg !94
  %861 = extractelement <8 x float> %850, i64 4, !dbg !94
  %862 = tail call float @llvm.maxnum.f32(float %860, float %861), !dbg !95
  %863 = extractelement <8 x float> %850, i64 5, !dbg !94
  %864 = tail call float @llvm.maxnum.f32(float %862, float %863), !dbg !95
  %865 = extractelement <8 x float> %850, i64 6, !dbg !94
  %866 = extractelement <8 x float> %850, i64 7, !dbg !94
  %867 = tail call float @llvm.maxnum.f32(float %865, float %866), !dbg !95
  %868 = tail call float @llvm.maxnum.f32(float %867, float %851), !dbg !95
  %869 = extractelement <4 x float> %852, i64 0, !dbg !94
  %870 = extractelement <4 x float> %852, i64 1, !dbg !94
  %871 = tail call float @llvm.maxnum.f32(float %869, float %870), !dbg !95
  %872 = extractelement <4 x float> %852, i64 2, !dbg !94
  %873 = tail call float @llvm.maxnum.f32(float %871, float %872), !dbg !95
  %874 = extractelement <4 x float> %852, i64 3, !dbg !94
  %875 = extractelement <2 x float> %853, i64 0, !dbg !95
  %876 = tail call float @llvm.maxnum.f32(float %874, float %875), !dbg !95
  %877 = extractelement <2 x float> %853, i64 1, !dbg !95
  %878 = tail call float @llvm.maxnum.f32(float %876, float %877), !dbg !95
  %879 = tail call float @llvm.maxnum.f32(float %859, float %864), !dbg !95
  %880 = tail call float @llvm.maxnum.f32(float %879, float %868), !dbg !95
  %881 = tail call float @llvm.maxnum.f32(float %873, float %878), !dbg !95
  %882 = tail call float @llvm.maxnum.f32(float %881, float %854), !dbg !95
  %883 = tail call float @llvm.maxnum.f32(float %880, float %882), !dbg !95
  %884 = bitcast float %883 to i32, !dbg !98
  %885 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %884, i32 %884, i1 false, i1 false), !dbg !98
  %886 = extractvalue { i32, i32 } %885, 0, !dbg !98
  %887 = extractvalue { i32, i32 } %885, 1, !dbg !98
  %888 = bitcast i32 %886 to float, !dbg !98
  %889 = bitcast i32 %887 to float, !dbg !98
  %890 = tail call float @llvm.maxnum.f32(float %888, float %889), !dbg !95
  %891 = tail call float @llvm.maxnum.f32(float %812, float %890), !dbg !100
  %892 = shufflevector <8 x float> %850, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %893 = insertelement <2 x float> poison, float %891, i64 0, !dbg !94
  %894 = shufflevector <2 x float> %893, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !94
  %895 = fsub <2 x float> %892, %894, !dbg !94
  %896 = shufflevector <8 x float> %850, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %897 = fsub <2 x float> %896, %894, !dbg !94
  %898 = shufflevector <8 x float> %850, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !94
  %899 = fsub <2 x float> %898, %894, !dbg !94
  %900 = shufflevector <8 x float> %850, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !94
  %901 = fsub <2 x float> %900, %894, !dbg !94
  %902 = fsub float %851, %891, !dbg !94
  %903 = shufflevector <4 x float> %852, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !94
  %904 = fsub <2 x float> %903, %894, !dbg !94
  %905 = shufflevector <4 x float> %852, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !94
  %906 = fsub <2 x float> %905, %894, !dbg !94
  %907 = fsub <2 x float> %853, %894, !dbg !94
  %908 = fsub float %854, %891, !dbg !94
  %909 = extractelement <2 x float> %895, i64 0, !dbg !101
  %910 = tail call float @llvm.amdgcn.exp2.f32(float %909), !dbg !101
  %911 = extractelement <2 x float> %895, i64 1, !dbg !101
  %912 = tail call float @llvm.amdgcn.exp2.f32(float %911), !dbg !101
  %913 = extractelement <2 x float> %897, i64 0, !dbg !101
  %914 = tail call float @llvm.amdgcn.exp2.f32(float %913), !dbg !101
  %915 = extractelement <2 x float> %897, i64 1, !dbg !101
  %916 = tail call float @llvm.amdgcn.exp2.f32(float %915), !dbg !101
  %917 = extractelement <2 x float> %899, i64 0, !dbg !101
  %918 = tail call float @llvm.amdgcn.exp2.f32(float %917), !dbg !101
  %919 = extractelement <2 x float> %899, i64 1, !dbg !101
  %920 = tail call float @llvm.amdgcn.exp2.f32(float %919), !dbg !101
  %921 = extractelement <2 x float> %901, i64 0, !dbg !101
  %922 = tail call float @llvm.amdgcn.exp2.f32(float %921), !dbg !101
  %923 = extractelement <2 x float> %901, i64 1, !dbg !101
  %924 = tail call float @llvm.amdgcn.exp2.f32(float %923), !dbg !101
  %925 = tail call float @llvm.amdgcn.exp2.f32(float %902), !dbg !101
  %926 = extractelement <2 x float> %904, i64 0, !dbg !101
  %927 = tail call float @llvm.amdgcn.exp2.f32(float %926), !dbg !101
  %928 = extractelement <2 x float> %904, i64 1, !dbg !101
  %929 = tail call float @llvm.amdgcn.exp2.f32(float %928), !dbg !101
  %930 = extractelement <2 x float> %906, i64 0, !dbg !101
  %931 = tail call float @llvm.amdgcn.exp2.f32(float %930), !dbg !101
  %932 = extractelement <2 x float> %906, i64 1, !dbg !101
  %933 = tail call float @llvm.amdgcn.exp2.f32(float %932), !dbg !101
  %934 = extractelement <2 x float> %907, i64 0, !dbg !101
  %935 = tail call float @llvm.amdgcn.exp2.f32(float %934), !dbg !101
  %936 = extractelement <2 x float> %907, i64 1, !dbg !101
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !101
  %938 = tail call float @llvm.amdgcn.exp2.f32(float %908), !dbg !101
  %939 = insertelement <2 x float> poison, float %910, i64 0, !dbg !102
  %940 = insertelement <2 x float> %939, float %912, i64 1, !dbg !102
  %941 = insertelement <2 x float> poison, float %914, i64 0, !dbg !102
  %942 = insertelement <2 x float> %941, float %916, i64 1, !dbg !102
  %943 = insertelement <2 x float> poison, float %918, i64 0, !dbg !102
  %944 = insertelement <2 x float> %943, float %920, i64 1, !dbg !102
  %945 = insertelement <2 x float> poison, float %922, i64 0, !dbg !102
  %946 = insertelement <2 x float> %945, float %924, i64 1, !dbg !102
  %947 = insertelement <2 x float> poison, float %925, i64 0, !dbg !102
  %948 = insertelement <2 x float> %947, float %927, i64 1, !dbg !102
  %949 = insertelement <2 x float> poison, float %929, i64 0, !dbg !102
  %950 = insertelement <2 x float> %949, float %931, i64 1, !dbg !102
  %951 = insertelement <2 x float> poison, float %933, i64 0, !dbg !102
  %952 = insertelement <2 x float> %951, float %935, i64 1, !dbg !102
  %953 = insertelement <2 x float> poison, float %937, i64 0, !dbg !102
  %954 = insertelement <2 x float> %953, float %938, i64 1, !dbg !102
  %955 = fadd <2 x float> %940, %942, !dbg !102
  %956 = fadd <2 x float> %944, %946, !dbg !102
  %957 = fadd <2 x float> %948, %950, !dbg !102
  %958 = fadd <2 x float> %952, %954, !dbg !102
  %959 = fadd <2 x float> %955, %956, !dbg !102
  %960 = fadd <2 x float> %957, %958, !dbg !102
  %961 = fadd <2 x float> %959, %960, !dbg !102
  %shift189 = shufflevector <2 x float> %961, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !104
  %foldExtExtBinop190 = fadd <2 x float> %961, %shift189, !dbg !104
  %bc196 = bitcast <2 x float> %foldExtExtBinop190 to <2 x i32>, !dbg !102
  %962 = extractelement <2 x i32> %bc196, i64 0, !dbg !102
  %963 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %962, i32 %962, i1 false, i1 false), !dbg !102
  %964 = fsub float %812, %891, !dbg !105
  %965 = tail call float @llvm.amdgcn.exp2.f32(float %964), !dbg !106
  %966 = insertelement <1 x float> poison, float %965, i64 0, !dbg !107
  store <1 x float> %966, ptr addrspace(3) %704, align 4, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !107
  tail call void @llvm.amdgcn.s.barrier(), !dbg !107
  %967 = load <1 x float>, ptr addrspace(3) %708, align 4, !dbg !107
  %968 = load <1 x float>, ptr addrspace(3) %710, align 4, !dbg !107
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %969 = shufflevector <2 x float> %939, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %970 = fptrunc <1 x float> %969 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %970, ptr addrspace(3) %717, align 2, !dbg !109
  %971 = shufflevector <2 x float> %947, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %972 = fptrunc <1 x float> %971 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %972, ptr addrspace(3) %720, align 2, !dbg !109
  %973 = shufflevector <2 x float> %940, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %974 = fptrunc <1 x float> %973 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %974, ptr addrspace(3) %724, align 2, !dbg !109
  %975 = shufflevector <2 x float> %948, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %976 = fptrunc <1 x float> %975 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %976, ptr addrspace(3) %727, align 2, !dbg !109
  %977 = shufflevector <2 x float> %941, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %978 = fptrunc <1 x float> %977 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %978, ptr addrspace(3) %731, align 2, !dbg !109
  %979 = shufflevector <2 x float> %949, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %980 = fptrunc <1 x float> %979 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %980, ptr addrspace(3) %734, align 2, !dbg !109
  %981 = shufflevector <2 x float> %942, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %982 = fptrunc <1 x float> %981 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %982, ptr addrspace(3) %738, align 2, !dbg !109
  %983 = shufflevector <2 x float> %950, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %984 = fptrunc <1 x float> %983 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %984, ptr addrspace(3) %741, align 2, !dbg !109
  %985 = shufflevector <2 x float> %943, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %986 = fptrunc <1 x float> %985 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %986, ptr addrspace(3) %745, align 2, !dbg !109
  %987 = shufflevector <2 x float> %951, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %988 = fptrunc <1 x float> %987 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %988, ptr addrspace(3) %748, align 2, !dbg !109
  %989 = shufflevector <2 x float> %944, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %990 = fptrunc <1 x float> %989 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %990, ptr addrspace(3) %752, align 2, !dbg !109
  %991 = shufflevector <2 x float> %952, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %992 = fptrunc <1 x float> %991 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %992, ptr addrspace(3) %755, align 2, !dbg !109
  %993 = shufflevector <2 x float> %945, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %994 = fptrunc <1 x float> %993 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %994, ptr addrspace(3) %759, align 2, !dbg !109
  %995 = shufflevector <2 x float> %953, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !109
  %996 = fptrunc <1 x float> %995 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %996, ptr addrspace(3) %762, align 2, !dbg !109
  %997 = shufflevector <2 x float> %946, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %998 = fptrunc <1 x float> %997 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %998, ptr addrspace(3) %766, align 2, !dbg !109
  %999 = shufflevector <2 x float> %954, <2 x float> poison, <1 x i32> <i32 1>, !dbg !109
  %1000 = fptrunc <1 x float> %999 to <1 x bfloat>, !dbg !109
  store <1 x bfloat> %1000, ptr addrspace(3) %769, align 2, !dbg !109
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !109
  %1001 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %780), !dbg !109
  %1002 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %782), !dbg !109
  %1003 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %784), !dbg !109
  %1004 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %786), !dbg !109
  br i1 %538, label %1005, label %1028, !dbg !110

1005:                                             ; preds = %844
  %1006 = fmul float %813, %965, !dbg !108
  %1007 = extractvalue { i32, i32 } %963, 0, !dbg !102
  %1008 = bitcast i32 %1007 to float, !dbg !102
  %1009 = extractvalue { i32, i32 } %963, 1, !dbg !102
  %1010 = bitcast i32 %1009 to float, !dbg !102
  %1011 = fadd float %1008, %1010, !dbg !104
  %1012 = fadd float %1011, %1006, !dbg !108
  %1013 = shufflevector <4 x bfloat> %1001, <4 x bfloat> %1002, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %1014 = shufflevector <4 x bfloat> %1003, <4 x bfloat> %1004, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %1015 = shufflevector <4 x bfloat> %819, <4 x bfloat> %821, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !110
  %1016 = shufflevector <2 x float> %817, <2 x float> %816, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %1017 = shufflevector <1 x float> %967, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %1018 = fmul <4 x float> %1016, %1017, !dbg !110
  %1019 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1015, <8 x bfloat> %1013, <4 x float> %1018, i32 0, i32 0, i32 0), !dbg !110
  %1020 = shufflevector <2 x float> %815, <2 x float> %814, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !110
  %1021 = shufflevector <1 x float> %968, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !110
  %1022 = fmul <4 x float> %1020, %1021, !dbg !110
  %1023 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1015, <8 x bfloat> %1014, <4 x float> %1022, i32 0, i32 0, i32 0), !dbg !110
  %1024 = shufflevector <4 x float> %1023, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !70
  %1025 = shufflevector <4 x float> %1023, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !70
  %1026 = shufflevector <4 x float> %1019, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !70
  %1027 = shufflevector <4 x float> %1019, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !70
  br label %1028, !dbg !110

1028:                                             ; preds = %844, %1005, %37
  %1029 = phi float [ 0xFFF0000000000000, %37 ], [ %891, %1005 ], [ %812, %844 ], !dbg !70
  %1030 = phi float [ 1.000000e+00, %37 ], [ %1012, %1005 ], [ %813, %844 ], !dbg !70
  %1031 = phi i32 [ 0, %37 ], [ %153, %1005 ], [ %153, %844 ], !dbg !70
  %1032 = phi <2 x float> [ zeroinitializer, %37 ], [ %1024, %1005 ], [ %814, %844 ], !dbg !70
  %1033 = phi <2 x float> [ zeroinitializer, %37 ], [ %1025, %1005 ], [ %815, %844 ], !dbg !70
  %1034 = phi <2 x float> [ zeroinitializer, %37 ], [ %1026, %1005 ], [ %816, %844 ], !dbg !70
  %1035 = phi <2 x float> [ zeroinitializer, %37 ], [ %1027, %1005 ], [ %817, %844 ], !dbg !70
  %1036 = icmp sgt i32 %142, 0, !dbg !112
  br i1 %1036, label %1037, label %.loopexit, !dbg !113

1037:                                             ; preds = %1028
  %1038 = fmul float %28, 0x3FF7154760000000, !dbg !114
  %1039 = shl nsw i64 %66, 5, !dbg !116
  %1040 = shl nsw i64 %69, 5, !dbg !117
  %1041 = icmp slt i32 %1031, %144, !dbg !118
  br i1 %1041, label %.lr.ph123, label %.loopexit, !dbg !118

.lr.ph123:                                        ; preds = %1037
  %invariant.gep114 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %56, !dbg !118
  %1042 = shl i32 %143, 5, !dbg !119
  %1043 = zext i32 %1042 to i64, !dbg !119
  %1044 = mul nsw i64 %1043, %69, !dbg !120
  %1045 = add i64 %105, %1044, !dbg !121
  %1046 = add i64 %107, %1045, !dbg !121
  %1047 = mul nsw i64 %1043, %66, !dbg !119
  %1048 = add i64 %85, %1047, !dbg !121
  %1049 = add i64 %103, %1048, !dbg !121
  %1050 = add i64 %101, %1048, !dbg !121
  %1051 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %1, i16 0, i64 2147483646, i32 159744)
  %1052 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 2147483646, i32 159744)
  %1053 = shl nuw nsw i32 %88, 2
  %1054 = and i32 %1053, 764
  %1055 = and i32 %39, 64
  %1056 = icmp eq i32 %1055, 0
  %1057 = select i1 %1056, i32 0, i32 272
  %1058 = xor i32 %1054, %1057
  %1059 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1058
  %1060 = shl nuw nsw i32 %38, 5
  %1061 = and i32 %1060, 736
  %1062 = and i32 %38, 8
  %1063 = icmp eq i32 %1062, 0
  %1064 = select i1 %1063, i32 0, i32 272
  %1065 = lshr exact i32 %86, 1
  %1066 = xor i32 %1064, %1065
  %1067 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1061
  %1068 = getelementptr inbounds nuw i8, ptr addrspace(3) %1067, i32 %1066
  %1069 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1070 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1071 = shufflevector <8 x bfloat> %1069, <8 x bfloat> %1070, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1072 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1073 = shufflevector <8 x bfloat> %1071, <8 x bfloat> %1072, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1074 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1075 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1076 = shufflevector <8 x bfloat> %1074, <8 x bfloat> %1075, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1077 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1078 = shufflevector <8 x bfloat> %1076, <8 x bfloat> %1077, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1079 = and i32 %38, 31
  %1080 = shl nuw nsw i32 %1079, 2
  %1081 = shl nuw nsw i32 %41, 7
  %1082 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1080
  %1083 = getelementptr inbounds nuw i8, ptr addrspace(3) %1082, i32 %1081
  %1084 = shl nuw nsw i32 %57, 2
  %gep115 = getelementptr inbounds nuw i8, ptr addrspace(3) %invariant.gep114, i32 %1084
  %1085 = getelementptr inbounds nuw i8, ptr addrspace(3) %gep115, i32 256
  %1086 = shl nuw nsw i32 %1079, 1
  %1087 = icmp eq i32 %86, 0
  %1088 = select i1 %1087, i32 0, i32 1056
  %1089 = or disjoint i32 %56, %1086
  %1090 = xor i32 %1089, %1088
  %1091 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1090
  %1092 = getelementptr inbounds nuw i8, ptr addrspace(3) %1091, i32 4096
  %1093 = xor i32 %1090, 264
  %1094 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1093
  %1095 = getelementptr inbounds nuw i8, ptr addrspace(3) %1094, i32 4096
  %1096 = xor i32 %1090, 528
  %1097 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1096
  %1098 = getelementptr inbounds nuw i8, ptr addrspace(3) %1097, i32 4096
  %1099 = xor i32 %1090, 792
  %1100 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1099
  %1101 = getelementptr inbounds nuw i8, ptr addrspace(3) %1100, i32 4096
  %1102 = xor i32 %1090, 2112
  %1103 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1102
  %1104 = getelementptr inbounds nuw i8, ptr addrspace(3) %1103, i32 4096
  %1105 = xor i32 %1090, 2376
  %1106 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1105
  %1107 = getelementptr inbounds nuw i8, ptr addrspace(3) %1106, i32 4096
  %1108 = xor i32 %1090, 2640
  %1109 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1108
  %1110 = getelementptr inbounds nuw i8, ptr addrspace(3) %1109, i32 4096
  %1111 = xor i32 %1090, 2904
  %1112 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1111
  %1113 = getelementptr inbounds nuw i8, ptr addrspace(3) %1112, i32 4096
  %1114 = and i32 %38, 60
  %1115 = shl nuw nsw i32 %1114, 6
  %1116 = and i32 %60, 24
  %1117 = shl nuw nsw i32 %1114, 1
  %1118 = shl nuw nsw i32 %41, 5
  %1119 = or disjoint i32 %1115, %1116
  %1120 = xor i32 %1119, %1117
  %1121 = xor i32 %1120, %1118
  %1122 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1121
  %1123 = getelementptr inbounds nuw i8, ptr addrspace(3) %1122, i32 4096
  %1124 = getelementptr inbounds nuw i8, ptr addrspace(3) %1122, i32 128
  %1125 = getelementptr inbounds nuw i8, ptr addrspace(3) %1122, i32 4224
  %1126 = select i1 %1056, i32 0, i32 264
  %1127 = xor i32 %1054, %1126
  %1128 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1127
  %1129 = shl nuw nsw i32 %1079, 3
  %1130 = select i1 %1087, i32 0, i32 264
  %1131 = xor i32 %1130, %1129
  %1132 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1131
  %1133 = getelementptr inbounds nuw i8, ptr addrspace(3) %1132, i32 512
  br label %1134, !dbg !118

1134:                                             ; preds = %.lr.ph123, %1134
  %.pn73.in120 = phi i64 [ %1046, %.lr.ph123 ], [ %1384, %1134 ]
  %.pn69.in118 = phi i64 [ %1049, %.lr.ph123 ], [ %1382, %1134 ]
  %.pn65.in116 = phi i64 [ %1050, %.lr.ph123 ], [ %1380, %1134 ]
  %1135 = phi float [ %1029, %.lr.ph123 ], [ %1248, %1134 ]
  %1136 = phi float [ %1030, %.lr.ph123 ], [ %1329, %1134 ]
  %1137 = phi i32 [ %1031, %.lr.ph123 ], [ %1153, %1134 ]
  %1138 = phi <2 x float> [ %1033, %.lr.ph123 ], [ %1387, %1134 ]
  %1139 = phi <2 x float> [ %1032, %.lr.ph123 ], [ %1386, %1134 ]
  %1140 = phi <2 x float> [ %1035, %.lr.ph123 ], [ %1389, %1134 ]
  %1141 = phi <2 x float> [ %1034, %.lr.ph123 ], [ %1388, %1134 ]
  %.pn73 = trunc i64 %.pn73.in120 to i32, !dbg !121
  %.pn69 = trunc i64 %.pn69.in118 to i32, !dbg !121
  %.pn65 = trunc i64 %.pn65.in116 to i32, !dbg !121
  %1142 = or disjoint i32 %1137, %93, !dbg !122
  %1143 = icmp slt i32 %1142, %33, !dbg !123
  %1144 = shl i32 %.pn65, 1, !dbg !125
  %1145 = select i1 %1143, i32 %1144, i32 -2147483648, !dbg !125
  %1146 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1051, i32 %1145, i32 0, i32 0), !dbg !125
  %1147 = shl i32 %.pn69, 1, !dbg !126
  %1148 = select i1 %1143, i32 %1147, i32 -2147483648, !dbg !126
  %1149 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1051, i32 %1148, i32 0, i32 0), !dbg !126
  %1150 = shl i32 %.pn73, 1, !dbg !128
  %1151 = select i1 %1143, i32 %1150, i32 -2147483648, !dbg !128
  %1152 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1052, i32 %1151, i32 0, i32 0), !dbg !128
  %1153 = add nsw i32 %1137, 32, !dbg !130
  %1154 = icmp eq i32 %1153, %144, !dbg !130
  %1155 = and i1 %140, %1154, !dbg !131
  %1156 = or disjoint i32 %1137, %98, !dbg !132
  %1157 = or disjoint i32 %1156, 16, !dbg !132
  %1158 = icmp sge i32 %1156, %33, !dbg !133
  %1159 = icmp sge i32 %1157, %33, !dbg !133
  %.not74 = select i1 %1155, i1 %1158, i1 false, !dbg !134
  %.not75 = select i1 %1155, i1 %1159, i1 false, !dbg !126
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !126
  tail call void @llvm.amdgcn.s.barrier(), !dbg !126
  store i32 %1149, ptr addrspace(3) %1059, align 4, !dbg !126
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !126
  tail call void @llvm.amdgcn.s.barrier(), !dbg !126
  %1160 = load <8 x bfloat>, ptr addrspace(3) %1068, align 16, !dbg !126
  %1161 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %1160, <8 x bfloat> %1073, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !135
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !125
  tail call void @llvm.amdgcn.s.barrier(), !dbg !125
  store i32 %1146, ptr addrspace(3) %1059, align 4, !dbg !125
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !125
  tail call void @llvm.amdgcn.s.barrier(), !dbg !125
  %1162 = load <8 x bfloat>, ptr addrspace(3) %1068, align 16, !dbg !125
  %1163 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %1162, <8 x bfloat> %1078, <16 x float> %1161, i32 0, i32 0, i32 0), !dbg !136
  %1164 = extractelement <16 x float> %1163, i64 0, !dbg !136
  %1165 = extractelement <16 x float> %1163, i64 1, !dbg !136
  %1166 = extractelement <16 x float> %1163, i64 2, !dbg !136
  %1167 = extractelement <16 x float> %1163, i64 3, !dbg !136
  %1168 = extractelement <16 x float> %1163, i64 4, !dbg !136
  %1169 = extractelement <16 x float> %1163, i64 5, !dbg !136
  %1170 = extractelement <16 x float> %1163, i64 6, !dbg !136
  %1171 = extractelement <16 x float> %1163, i64 7, !dbg !136
  %1172 = extractelement <16 x float> %1163, i64 8, !dbg !136
  %1173 = extractelement <16 x float> %1163, i64 9, !dbg !136
  %1174 = extractelement <16 x float> %1163, i64 10, !dbg !136
  %1175 = extractelement <16 x float> %1163, i64 11, !dbg !136
  %1176 = extractelement <16 x float> %1163, i64 12, !dbg !136
  %1177 = extractelement <16 x float> %1163, i64 13, !dbg !136
  %1178 = extractelement <16 x float> %1163, i64 14, !dbg !136
  %1179 = extractelement <16 x float> %1163, i64 15, !dbg !136
  %1180 = fmul float %1038, %1165, !dbg !137
  %1181 = fmul float %1038, %1164, !dbg !137
  %1182 = select i1 %.not74, float 0xFFF0000000000000, float %1180, !dbg !138
  %1183 = select i1 %.not74, float 0xFFF0000000000000, float %1181, !dbg !138
  %1184 = tail call float @llvm.maxnum.f32(float %1183, float %1182), !dbg !139
  %1185 = insertelement <2 x float> poison, float %1183, i64 0, !dbg !142
  %1186 = insertelement <2 x float> %1185, float %1182, i64 1, !dbg !142
  %1187 = fmul float %1038, %1167, !dbg !137
  %1188 = fmul float %1038, %1166, !dbg !137
  %1189 = select i1 %.not74, float 0xFFF0000000000000, float %1187, !dbg !138
  %1190 = select i1 %.not74, float 0xFFF0000000000000, float %1188, !dbg !138
  %1191 = tail call float @llvm.maxnum.f32(float %1184, float %1190), !dbg !139
  %1192 = insertelement <2 x float> poison, float %1190, i64 0, !dbg !142
  %1193 = insertelement <2 x float> %1192, float %1189, i64 1, !dbg !142
  %1194 = fmul float %1038, %1169, !dbg !137
  %1195 = fmul float %1038, %1168, !dbg !137
  %1196 = select i1 %.not74, float 0xFFF0000000000000, float %1194, !dbg !138
  %1197 = select i1 %.not74, float 0xFFF0000000000000, float %1195, !dbg !138
  %1198 = tail call float @llvm.maxnum.f32(float %1189, float %1197), !dbg !139
  %1199 = tail call float @llvm.maxnum.f32(float %1198, float %1196), !dbg !139
  %1200 = tail call float @llvm.maxnum.f32(float %1191, float %1199), !dbg !139
  %1201 = insertelement <2 x float> poison, float %1197, i64 0, !dbg !142
  %1202 = insertelement <2 x float> %1201, float %1196, i64 1, !dbg !142
  %1203 = fmul float %1038, %1171, !dbg !137
  %1204 = fmul float %1038, %1170, !dbg !137
  %1205 = select i1 %.not74, float 0xFFF0000000000000, float %1203, !dbg !138
  %1206 = select i1 %.not74, float 0xFFF0000000000000, float %1204, !dbg !138
  %1207 = tail call float @llvm.maxnum.f32(float %1206, float %1205), !dbg !139
  %1208 = insertelement <2 x float> poison, float %1206, i64 0, !dbg !142
  %1209 = insertelement <2 x float> %1208, float %1205, i64 1, !dbg !142
  %1210 = fmul float %1038, %1173, !dbg !137
  %1211 = fmul float %1038, %1172, !dbg !137
  %1212 = select i1 %.not75, float 0xFFF0000000000000, float %1210, !dbg !138
  %1213 = select i1 %.not75, float 0xFFF0000000000000, float %1211, !dbg !138
  %1214 = tail call float @llvm.maxnum.f32(float %1207, float %1213), !dbg !139
  %1215 = tail call float @llvm.maxnum.f32(float %1200, float %1214), !dbg !139
  %1216 = insertelement <2 x float> poison, float %1213, i64 0, !dbg !142
  %1217 = insertelement <2 x float> %1216, float %1212, i64 1, !dbg !142
  %1218 = fmul float %1038, %1175, !dbg !137
  %1219 = fmul float %1038, %1174, !dbg !137
  %1220 = select i1 %.not75, float 0xFFF0000000000000, float %1218, !dbg !138
  %1221 = select i1 %.not75, float 0xFFF0000000000000, float %1219, !dbg !138
  %1222 = tail call float @llvm.maxnum.f32(float %1212, float %1221), !dbg !139
  %1223 = tail call float @llvm.maxnum.f32(float %1222, float %1220), !dbg !139
  %1224 = insertelement <2 x float> poison, float %1221, i64 0, !dbg !142
  %1225 = insertelement <2 x float> %1224, float %1220, i64 1, !dbg !142
  %1226 = fmul float %1038, %1177, !dbg !137
  %1227 = fmul float %1038, %1176, !dbg !137
  %1228 = select i1 %.not75, float 0xFFF0000000000000, float %1226, !dbg !138
  %1229 = select i1 %.not75, float 0xFFF0000000000000, float %1227, !dbg !138
  %1230 = tail call float @llvm.maxnum.f32(float %1229, float %1228), !dbg !139
  %1231 = insertelement <2 x float> poison, float %1229, i64 0, !dbg !142
  %1232 = insertelement <2 x float> %1231, float %1228, i64 1, !dbg !142
  %1233 = fmul float %1038, %1179, !dbg !137
  %1234 = fmul float %1038, %1178, !dbg !137
  %1235 = select i1 %.not75, float 0xFFF0000000000000, float %1233, !dbg !138
  %1236 = select i1 %.not75, float 0xFFF0000000000000, float %1234, !dbg !138
  %1237 = tail call float @llvm.maxnum.f32(float %1230, float %1236), !dbg !139
  %1238 = tail call float @llvm.maxnum.f32(float %1223, float %1237), !dbg !139
  %1239 = tail call float @llvm.maxnum.f32(float %1238, float %1235), !dbg !139
  %1240 = tail call float @llvm.maxnum.f32(float %1215, float %1239), !dbg !139
  %1241 = bitcast float %1240 to i32, !dbg !140
  %1242 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %1241, i32 %1241, i1 false, i1 false), !dbg !140
  %1243 = extractvalue { i32, i32 } %1242, 0, !dbg !140
  %1244 = extractvalue { i32, i32 } %1242, 1, !dbg !140
  %1245 = bitcast i32 %1243 to float, !dbg !140
  %1246 = bitcast i32 %1244 to float, !dbg !140
  %1247 = tail call float @llvm.maxnum.f32(float %1245, float %1246), !dbg !139
  %1248 = tail call float @llvm.maxnum.f32(float %1135, float %1247), !dbg !143
  %1249 = insertelement <2 x float> poison, float %1248, i64 0, !dbg !142
  %1250 = shufflevector <2 x float> %1249, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !142
  %1251 = fsub <2 x float> %1186, %1250, !dbg !142
  %1252 = fsub <2 x float> %1193, %1250, !dbg !142
  %1253 = fsub <2 x float> %1202, %1250, !dbg !142
  %1254 = fsub <2 x float> %1209, %1250, !dbg !142
  %1255 = fsub <2 x float> %1217, %1250, !dbg !142
  %1256 = fsub <2 x float> %1225, %1250, !dbg !142
  %1257 = fsub <2 x float> %1232, %1250, !dbg !142
  %1258 = insertelement <2 x float> poison, float %1236, i64 0, !dbg !142
  %1259 = insertelement <2 x float> %1258, float %1235, i64 1, !dbg !142
  %1260 = fsub <2 x float> %1259, %1250, !dbg !142
  %1261 = extractelement <2 x float> %1251, i64 0, !dbg !144
  %1262 = tail call float @llvm.amdgcn.exp2.f32(float %1261), !dbg !144
  %1263 = extractelement <2 x float> %1251, i64 1, !dbg !144
  %1264 = tail call float @llvm.amdgcn.exp2.f32(float %1263), !dbg !144
  %1265 = extractelement <2 x float> %1252, i64 0, !dbg !144
  %1266 = tail call float @llvm.amdgcn.exp2.f32(float %1265), !dbg !144
  %1267 = extractelement <2 x float> %1252, i64 1, !dbg !144
  %1268 = tail call float @llvm.amdgcn.exp2.f32(float %1267), !dbg !144
  %1269 = extractelement <2 x float> %1253, i64 0, !dbg !144
  %1270 = tail call float @llvm.amdgcn.exp2.f32(float %1269), !dbg !144
  %1271 = extractelement <2 x float> %1253, i64 1, !dbg !144
  %1272 = tail call float @llvm.amdgcn.exp2.f32(float %1271), !dbg !144
  %1273 = extractelement <2 x float> %1254, i64 0, !dbg !144
  %1274 = tail call float @llvm.amdgcn.exp2.f32(float %1273), !dbg !144
  %1275 = extractelement <2 x float> %1254, i64 1, !dbg !144
  %1276 = tail call float @llvm.amdgcn.exp2.f32(float %1275), !dbg !144
  %1277 = extractelement <2 x float> %1255, i64 0, !dbg !144
  %1278 = tail call float @llvm.amdgcn.exp2.f32(float %1277), !dbg !144
  %1279 = extractelement <2 x float> %1255, i64 1, !dbg !144
  %1280 = tail call float @llvm.amdgcn.exp2.f32(float %1279), !dbg !144
  %1281 = extractelement <2 x float> %1256, i64 0, !dbg !144
  %1282 = tail call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !144
  %1283 = extractelement <2 x float> %1256, i64 1, !dbg !144
  %1284 = tail call float @llvm.amdgcn.exp2.f32(float %1283), !dbg !144
  %1285 = extractelement <2 x float> %1257, i64 0, !dbg !144
  %1286 = tail call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !144
  %1287 = extractelement <2 x float> %1257, i64 1, !dbg !144
  %1288 = tail call float @llvm.amdgcn.exp2.f32(float %1287), !dbg !144
  %1289 = extractelement <2 x float> %1260, i64 0, !dbg !144
  %1290 = tail call float @llvm.amdgcn.exp2.f32(float %1289), !dbg !144
  %1291 = extractelement <2 x float> %1260, i64 1, !dbg !144
  %1292 = tail call float @llvm.amdgcn.exp2.f32(float %1291), !dbg !144
  %1293 = insertelement <2 x float> poison, float %1262, i64 0, !dbg !145
  %1294 = insertelement <2 x float> %1293, float %1264, i64 1, !dbg !145
  %1295 = insertelement <2 x float> poison, float %1266, i64 0, !dbg !145
  %1296 = insertelement <2 x float> %1295, float %1268, i64 1, !dbg !145
  %1297 = insertelement <2 x float> poison, float %1270, i64 0, !dbg !145
  %1298 = insertelement <2 x float> %1297, float %1272, i64 1, !dbg !145
  %1299 = insertelement <2 x float> poison, float %1274, i64 0, !dbg !145
  %1300 = insertelement <2 x float> %1299, float %1276, i64 1, !dbg !145
  %1301 = insertelement <2 x float> poison, float %1278, i64 0, !dbg !145
  %1302 = insertelement <2 x float> %1301, float %1280, i64 1, !dbg !145
  %1303 = insertelement <2 x float> poison, float %1282, i64 0, !dbg !145
  %1304 = insertelement <2 x float> %1303, float %1284, i64 1, !dbg !145
  %1305 = insertelement <2 x float> poison, float %1286, i64 0, !dbg !145
  %1306 = insertelement <2 x float> %1305, float %1288, i64 1, !dbg !145
  %1307 = insertelement <2 x float> poison, float %1290, i64 0, !dbg !145
  %1308 = insertelement <2 x float> %1307, float %1292, i64 1, !dbg !145
  %1309 = fadd <2 x float> %1294, %1296, !dbg !145
  %1310 = fadd <2 x float> %1298, %1300, !dbg !145
  %1311 = fadd <2 x float> %1302, %1304, !dbg !145
  %1312 = fadd <2 x float> %1306, %1308, !dbg !145
  %1313 = fadd <2 x float> %1309, %1310, !dbg !145
  %1314 = fadd <2 x float> %1311, %1312, !dbg !145
  %1315 = fadd <2 x float> %1313, %1314, !dbg !145
  %shift192 = shufflevector <2 x float> %1315, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !147
  %foldExtExtBinop193 = fadd <2 x float> %1315, %shift192, !dbg !147
  %bc197 = bitcast <2 x float> %foldExtExtBinop193 to <2 x i32>, !dbg !145
  %1316 = extractelement <2 x i32> %bc197, i64 0, !dbg !145
  %1317 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %1316, i32 %1316, i1 false, i1 false), !dbg !145
  %1318 = extractvalue { i32, i32 } %1317, 0, !dbg !145
  %1319 = extractvalue { i32, i32 } %1317, 1, !dbg !145
  %1320 = bitcast i32 %1318 to float, !dbg !145
  %1321 = bitcast i32 %1319 to float, !dbg !145
  %1322 = fadd float %1320, %1321, !dbg !147
  %1323 = fsub float %1135, %1248, !dbg !148
  %1324 = tail call float @llvm.amdgcn.exp2.f32(float %1323), !dbg !149
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !150
  tail call void @llvm.amdgcn.s.barrier(), !dbg !150
  %1325 = insertelement <1 x float> poison, float %1324, i64 0, !dbg !150
  store <1 x float> %1325, ptr addrspace(3) %1083, align 4, !dbg !150
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !150
  tail call void @llvm.amdgcn.s.barrier(), !dbg !150
  %1326 = load <1 x float>, ptr addrspace(3) %gep115, align 4, !dbg !150
  %1327 = load <1 x float>, ptr addrspace(3) %1085, align 4, !dbg !150
  %1328 = fmul float %1136, %1324, !dbg !151
  %1329 = fadd float %1322, %1328, !dbg !151
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !152
  tail call void @llvm.amdgcn.s.barrier(), !dbg !152
  %1330 = shufflevector <2 x float> %1293, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1331 = fptrunc <1 x float> %1330 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1331, ptr addrspace(3) %1091, align 2, !dbg !152
  %1332 = shufflevector <2 x float> %1301, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1333 = fptrunc <1 x float> %1332 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1333, ptr addrspace(3) %1092, align 2, !dbg !152
  %1334 = shufflevector <2 x float> %1294, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1335 = fptrunc <1 x float> %1334 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1335, ptr addrspace(3) %1094, align 2, !dbg !152
  %1336 = shufflevector <2 x float> %1302, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1337 = fptrunc <1 x float> %1336 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1337, ptr addrspace(3) %1095, align 2, !dbg !152
  %1338 = shufflevector <2 x float> %1295, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1339 = fptrunc <1 x float> %1338 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1339, ptr addrspace(3) %1097, align 2, !dbg !152
  %1340 = shufflevector <2 x float> %1303, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1341 = fptrunc <1 x float> %1340 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1341, ptr addrspace(3) %1098, align 2, !dbg !152
  %1342 = shufflevector <2 x float> %1296, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1343 = fptrunc <1 x float> %1342 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1343, ptr addrspace(3) %1100, align 2, !dbg !152
  %1344 = shufflevector <2 x float> %1304, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1345 = fptrunc <1 x float> %1344 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1345, ptr addrspace(3) %1101, align 2, !dbg !152
  %1346 = shufflevector <2 x float> %1297, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1347 = fptrunc <1 x float> %1346 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1347, ptr addrspace(3) %1103, align 2, !dbg !152
  %1348 = shufflevector <2 x float> %1305, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1349 = fptrunc <1 x float> %1348 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1349, ptr addrspace(3) %1104, align 2, !dbg !152
  %1350 = shufflevector <2 x float> %1298, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1351 = fptrunc <1 x float> %1350 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1351, ptr addrspace(3) %1106, align 2, !dbg !152
  %1352 = shufflevector <2 x float> %1306, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1353 = fptrunc <1 x float> %1352 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1353, ptr addrspace(3) %1107, align 2, !dbg !152
  %1354 = shufflevector <2 x float> %1299, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1355 = fptrunc <1 x float> %1354 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1355, ptr addrspace(3) %1109, align 2, !dbg !152
  %1356 = shufflevector <2 x float> %1307, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !152
  %1357 = fptrunc <1 x float> %1356 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1357, ptr addrspace(3) %1110, align 2, !dbg !152
  %1358 = shufflevector <2 x float> %1300, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1359 = fptrunc <1 x float> %1358 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1359, ptr addrspace(3) %1112, align 2, !dbg !152
  %1360 = shufflevector <2 x float> %1308, <2 x float> poison, <1 x i32> <i32 1>, !dbg !152
  %1361 = fptrunc <1 x float> %1360 to <1 x bfloat>, !dbg !152
  store <1 x bfloat> %1361, ptr addrspace(3) %1113, align 2, !dbg !152
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !152
  tail call void @llvm.amdgcn.s.barrier(), !dbg !152
  %1362 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %1122), !dbg !152
  %1363 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1123), !dbg !152
  %1364 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1124), !dbg !152
  %1365 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1125), !dbg !152
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !128
  tail call void @llvm.amdgcn.s.barrier(), !dbg !128
  store i32 %1152, ptr addrspace(3) %1128, align 4, !dbg !128
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !128
  tail call void @llvm.amdgcn.s.barrier(), !dbg !128
  %1366 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %1132), !dbg !128
  %1367 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1133), !dbg !128
  %1368 = shufflevector <4 x bfloat> %1362, <4 x bfloat> %1363, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !153
  %1369 = shufflevector <4 x bfloat> %1364, <4 x bfloat> %1365, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !153
  %1370 = shufflevector <4 x bfloat> %1366, <4 x bfloat> %1367, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !153
  %1371 = shufflevector <2 x float> %1140, <2 x float> %1141, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !153
  %1372 = shufflevector <1 x float> %1326, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !153
  %1373 = fmul <4 x float> %1371, %1372, !dbg !153
  %1374 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1370, <8 x bfloat> %1368, <4 x float> %1373, i32 0, i32 0, i32 0), !dbg !153
  %1375 = shufflevector <2 x float> %1138, <2 x float> %1139, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !153
  %1376 = shufflevector <1 x float> %1327, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !153
  %1377 = fmul <4 x float> %1375, %1376, !dbg !153
  %1378 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1370, <8 x bfloat> %1369, <4 x float> %1377, i32 0, i32 0, i32 0), !dbg !153
  %sext = shl i64 %.pn65.in116, 32, !dbg !154
  %1379 = ashr exact i64 %sext, 32, !dbg !154
  %1380 = add nsw i64 %1379, %1039, !dbg !154
  %sext77 = shl i64 %.pn69.in118, 32, !dbg !155
  %1381 = ashr exact i64 %sext77, 32, !dbg !155
  %1382 = add nsw i64 %1381, %1039, !dbg !155
  %sext79 = shl i64 %.pn73.in120, 32, !dbg !156
  %1383 = ashr exact i64 %sext79, 32, !dbg !156
  %1384 = add nsw i64 %1383, %1040, !dbg !156
  %1385 = icmp slt i32 %1153, %144, !dbg !118
  %1386 = shufflevector <4 x float> %1378, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !113
  %1387 = shufflevector <4 x float> %1378, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !113
  %1388 = shufflevector <4 x float> %1374, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !113
  %1389 = shufflevector <4 x float> %1374, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !113
  br i1 %1385, label %1134, label %.loopexit, !dbg !118

.loopexit:                                        ; preds = %1134, %1037, %1028
  %1390 = phi float [ %1029, %1028 ], [ %1029, %1037 ], [ %1248, %1134 ], !dbg !113
  %1391 = phi float [ %1030, %1028 ], [ %1030, %1037 ], [ %1329, %1134 ], !dbg !113
  %1392 = phi <2 x float> [ %1032, %1028 ], [ %1032, %1037 ], [ %1386, %1134 ], !dbg !113
  %1393 = phi <2 x float> [ %1033, %1028 ], [ %1033, %1037 ], [ %1387, %1134 ], !dbg !113
  %1394 = phi <2 x float> [ %1034, %1028 ], [ %1034, %1037 ], [ %1388, %1134 ], !dbg !113
  %1395 = phi <2 x float> [ %1035, %1028 ], [ %1035, %1037 ], [ %1389, %1134 ], !dbg !113
  %1396 = fdiv float 1.000000e+00, %1391, !dbg !157
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !158
  tail call void @llvm.amdgcn.s.barrier(), !dbg !158
  %1397 = shl nuw nsw i32 %38, 2, !dbg !158
  %1398 = and i32 %1397, 124, !dbg !158
  %1399 = shl nuw nsw i32 %41, 7, !dbg !158
  %1400 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1398, !dbg !158
  %1401 = getelementptr inbounds nuw i8, ptr addrspace(3) %1400, i32 %1399, !dbg !158
  %1402 = insertelement <1 x float> poison, float %1396, i64 0, !dbg !158
  store <1 x float> %1402, ptr addrspace(3) %1401, align 4, !dbg !158
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !158
  tail call void @llvm.amdgcn.s.barrier(), !dbg !158
  %1403 = shl nuw nsw i32 %57, 2, !dbg !158
  %1404 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1403, !dbg !158
  %1405 = getelementptr inbounds nuw i8, ptr addrspace(3) %1404, i32 %56, !dbg !158
  %1406 = load <1 x float>, ptr addrspace(3) %1405, align 4, !dbg !158
  %1407 = getelementptr inbounds nuw i8, ptr addrspace(3) %1405, i32 256, !dbg !158
  %1408 = load <1 x float>, ptr addrspace(3) %1407, align 4, !dbg !158
  %reass.sub = sub i32 %55, %32, !dbg !159
  %1409 = add i32 %reass.sub, 128, !dbg !159
  %1410 = tail call noundef float @llvm.log2.f32(float %1391), !dbg !160
  %1411 = fadd float %1390, %1410, !dbg !161
  %1412 = fmul float %1411, 0x3FE62E4300000000, !dbg !162
  %1413 = mul nuw i64 %80, %70, !dbg !163
  %1414 = mul nuw i64 %71, %75, !dbg !164
  %1415 = add i64 %1413, %1414, !dbg !163
  %1416 = icmp slt i32 %1409, 1, !dbg !165
  br i1 %1416, label %1432, label %1417, !dbg !166

1417:                                             ; preds = %.loopexit
  %1418 = sub nsw i32 0, %reass.sub, !dbg !167
  %1419 = icmp slt i32 %90, %1418, !dbg !168
  %1420 = trunc i64 %1415 to i32, !dbg !169
  %1421 = add i32 %92, %1420, !dbg !169
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !170
  tail call void @llvm.amdgcn.s.barrier(), !dbg !170
  %1422 = insertelement <1 x float> poison, float %1412, i64 0, !dbg !170
  store <1 x float> %1422, ptr addrspace(3) %1401, align 4, !dbg !170
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !170
  tail call void @llvm.amdgcn.s.barrier(), !dbg !170
  %1423 = shl nuw nsw i32 %90, 2, !dbg !170
  %1424 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1423, !dbg !170
  %1425 = load float, ptr addrspace(3) %1424, align 4, !dbg !170
  %1426 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 2147483646, i32 159744), !dbg !170
  %1427 = and i32 %39, 128, !dbg !170
  %1428 = icmp eq i32 %1427, 0, !dbg !170
  %1429 = and i1 %1428, %1419, !dbg !170
  %1430 = shl i32 %1421, 2, !dbg !170
  %1431 = select i1 %1429, i32 %1430, i32 -2147483648, !dbg !170
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %1425, ptr addrspace(8) %1426, i32 %1431, i32 0, i32 0), !dbg !170
  br label %1444, !dbg !166

1432:                                             ; preds = %.loopexit
  %1433 = trunc i64 %1415 to i32, !dbg !171
  %1434 = add i32 %92, %1433, !dbg !171
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  %1435 = insertelement <1 x float> poison, float %1412, i64 0, !dbg !14
  store <1 x float> %1435, ptr addrspace(3) %1401, align 4, !dbg !14
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  %1436 = shl nuw nsw i32 %90, 2, !dbg !14
  %1437 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1436, !dbg !14
  %1438 = load float, ptr addrspace(3) %1437, align 4, !dbg !14
  %1439 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 2147483646, i32 159744), !dbg !14
  %1440 = and i32 %39, 128, !dbg !14
  %1441 = icmp eq i32 %1440, 0, !dbg !14
  %1442 = shl i32 %1434, 2, !dbg !14
  %1443 = select i1 %1441, i32 %1442, i32 -2147483648, !dbg !14
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %1438, ptr addrspace(8) %1439, i32 %1443, i32 0, i32 0), !dbg !14
  br label %1444, !dbg !166

1444:                                             ; preds = %1417, %1432
  %1445 = shufflevector <1 x float> %1408, <1 x float> poison, <2 x i32> zeroinitializer, !dbg !158
  %1446 = fmul <2 x float> %1392, %1445, !dbg !158
  %1447 = fmul <2 x float> %1393, %1445, !dbg !158
  %1448 = shufflevector <1 x float> %1406, <1 x float> poison, <2 x i32> zeroinitializer, !dbg !158
  %1449 = fmul <2 x float> %1394, %1448, !dbg !158
  %1450 = fmul <2 x float> %1395, %1448, !dbg !158
  %1451 = shl nuw nsw i32 %41, 4, !dbg !31
  %1452 = or disjoint i32 %1451, %57, !dbg !31
  %1453 = or disjoint i32 %55, %1452, !dbg !30
  %1454 = or disjoint i32 %1453, 64, !dbg !30
  %1455 = icmp slt i32 %1454, %32, !dbg !61
  %1456 = icmp slt i32 %1453, %32, !dbg !61
  %1457 = lshr i32 %38, 2, !dbg !50
  %1458 = and i32 %1457, 12, !dbg !50
  %1459 = zext nneg i32 %1458 to i64, !dbg !50
  %1460 = zext i32 %1454 to i64, !dbg !172
  %1461 = zext i32 %1453 to i64, !dbg !172
  %1462 = zext i32 %19 to i64, !dbg !173
  %1463 = zext i32 %18 to i64, !dbg !174
  %1464 = zext i32 %17 to i64, !dbg !175
  %1465 = mul nuw i64 %80, %1464, !dbg !176
  %1466 = mul nuw i64 %1463, %75, !dbg !177
  %1467 = add i64 %1465, %1466, !dbg !176
  %1468 = select i1 %1416, i1 true, i1 %1456, !dbg !178
  %1469 = select i1 %1416, i1 true, i1 %1455, !dbg !178
  %1470 = fptrunc <2 x float> %1450 to <2 x bfloat>, !dbg !179
  %1471 = fptrunc <2 x float> %1449 to <2 x bfloat>, !dbg !179
  %1472 = fptrunc <2 x float> %1447 to <2 x bfloat>, !dbg !179
  %1473 = fptrunc <2 x float> %1446 to <2 x bfloat>, !dbg !179
  %1474 = mul nuw i64 %1461, %1462, !dbg !180
  %1475 = mul nuw i64 %1460, %1462, !dbg !180
  %1476 = add i64 %1467, %1459, !dbg !180
  %1477 = add i64 %1476, %1474, !dbg !180
  %1478 = add i64 %1476, %1475, !dbg !180
  %1479 = trunc i64 %1477 to i32, !dbg !180
  %1480 = trunc i64 %1478 to i32, !dbg !180
  %1481 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %3, i16 0, i64 2147483646, i32 159744), !dbg !181
  %1482 = shufflevector <2 x bfloat> %1470, <2 x bfloat> %1471, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !181
  %1483 = bitcast <4 x bfloat> %1482 to <2 x i32>, !dbg !181
  %1484 = shl i32 %1479, 1, !dbg !181
  %1485 = select i1 %1468, i32 %1484, i32 -2147483648, !dbg !181
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1483, ptr addrspace(8) %1481, i32 %1485, i32 0, i32 0), !dbg !181
  %1486 = shufflevector <2 x bfloat> %1472, <2 x bfloat> %1473, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !181
  %1487 = bitcast <4 x bfloat> %1486 to <2 x i32>, !dbg !181
  %1488 = shl i32 %1480, 1, !dbg !181
  %1489 = select i1 %1469, i32 %1488, i32 -2147483648, !dbg !181
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1487, ptr addrspace(8) %1481, i32 %1489, i32 0, i32 0), !dbg !181
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
