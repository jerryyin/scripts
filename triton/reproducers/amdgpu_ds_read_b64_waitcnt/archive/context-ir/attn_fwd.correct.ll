; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nofree norecurse nounwind
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
  %56 = and i32 %38, 63, !dbg !31
  %57 = shl nuw nsw i32 %41, 6, !dbg !31
  %58 = or disjoint i32 %57, %56, !dbg !31
  %59 = and i32 %38, 15, !dbg !31
  %60 = lshr i32 %58, 1, !dbg !31
  %61 = and i32 %58, 127, !dbg !31
  %62 = or disjoint i32 %60, %55, !dbg !30
  %63 = or disjoint i32 %61, %55, !dbg !30
  %64 = lshr i32 %58, 3, !dbg !32
  %65 = shl nuw nsw i32 %38, 1, !dbg !33
  %66 = and i32 %65, 14, !dbg !33
  %67 = shl nuw nsw i32 %38, 3, !dbg !33
  %68 = and i32 %67, 8, !dbg !33
  %69 = or disjoint i32 %66, 16, !dbg !34
  %70 = or disjoint i32 %68, 16, !dbg !34
  %71 = zext i32 %8 to i64, !dbg !35
  %72 = zext i32 %9 to i64, !dbg !36
  %73 = sext i32 %10 to i64, !dbg !37
  %74 = zext i32 %11 to i64, !dbg !38
  %75 = zext i32 %12 to i64, !dbg !39
  %76 = sext i32 %13 to i64, !dbg !40
  %77 = zext i32 %26 to i64, !dbg !41
  %78 = zext i32 %27 to i64, !dbg !42
  %79 = add i32 %33, 31, !dbg !43
  %80 = sdiv i32 %79, 32, !dbg !45
  %.lhs.trunc87 = trunc nsw i32 %50 to i16, !dbg !46
  %81 = sdiv i16 %.lhs.trunc87, 8, !dbg !46
  %82 = zext i32 %50 to i64, !dbg !47
  %83 = mul i32 %6, %50, !dbg !47
  %84 = sext i16 %81 to i64, !dbg !48
  %85 = mul nsw i64 %72, %84, !dbg !48
  %86 = mul nsw i64 %75, %84, !dbg !49
  %87 = zext i32 %54 to i64, !dbg !50
  %88 = mul i32 %54, %5, !dbg !50
  %89 = add i32 %88, %83, !dbg !50
  %90 = zext nneg i32 %66 to i64, !dbg !51
  %91 = mul i32 %62, %7, !dbg !52
  %92 = add i32 %91, %89, !dbg !52
  %93 = add i32 %68, %92, !dbg !52
  %94 = add i32 %70, %92, !dbg !53
  %95 = mul nuw i64 %87, %71, !dbg !54
  %96 = add i64 %95, %85, !dbg !54
  %97 = and i32 %38, 32, !dbg !55
  %98 = lshr exact i32 %97, 3, !dbg !55
  %99 = zext nneg i32 %64 to i64, !dbg !55
  %100 = mul nsw i64 %99, %73, !dbg !56
  %101 = add nsw i64 %100, %90, !dbg !56
  %102 = zext nneg i32 %69 to i64, !dbg !57
  %103 = add nsw i64 %100, %102, !dbg !58
  %104 = mul nuw i64 %87, %74, !dbg !59
  %105 = add i64 %104, %86, !dbg !59
  %106 = mul nsw i64 %99, %76, !dbg !60
  %107 = add nsw i64 %106, %90, !dbg !60
  %108 = icmp slt i32 %62, %32, !dbg !61
  %109 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %0, i16 0, i64 2147483646, i32 159744), !dbg !62
  %110 = shl i32 %94, 1, !dbg !62
  %111 = select i1 %108, i32 %110, i32 -2147483648, !dbg !62
  %112 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %109, i32 %111, i32 0, i32 3), !dbg !62
  %.extract = extractelement <4 x i32> %112, i64 0, !dbg !62
  %.extract6 = extractelement <4 x i32> %112, i64 1, !dbg !62
  %.extract7 = extractelement <4 x i32> %112, i64 2, !dbg !62
  %.extract8 = extractelement <4 x i32> %112, i64 3, !dbg !62
  %113 = and i32 %65, 62, !dbg !62
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
  %126 = shl i32 %93, 1, !dbg !63
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
  %142 = tail call i32 @llvm.smin.i32(i32 %141, i32 %80), !dbg !66
  %143 = sub nsw i32 %80, %142, !dbg !67
  %144 = shl nsw i32 %80, 5, !dbg !68
  %145 = icmp sgt i32 %143, 0, !dbg !69
  br i1 %145, label %146, label %1012, !dbg !70

146:                                              ; preds = %37
  %147 = add i64 %107, %105, !dbg !60
  %148 = trunc i64 %147 to i32, !dbg !60
  %149 = add i64 %103, %96, !dbg !58
  %150 = trunc i64 %149 to i32, !dbg !58
  %151 = add i64 %101, %96, !dbg !56
  %152 = trunc i64 %151 to i32, !dbg !56
  %153 = shl nuw i32 %143, 5, !dbg !71
  %154 = icmp sgt i32 %153, 0, !dbg !72
  %155 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %1, i16 0, i64 2147483646, i32 159744), !dbg !74
  %156 = shl i32 %152, 1, !dbg !74
  %157 = select i1 %154, i32 %156, i32 -2147483648, !dbg !74
  %158 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %155, i32 %157, i32 0, i32 0), !dbg !74
  %159 = shl i32 %150, 1, !dbg !76
  %160 = select i1 %154, i32 %159, i32 -2147483648, !dbg !76
  %161 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %155, i32 %160, i32 0, i32 0), !dbg !76
  %162 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 2147483646, i32 159744), !dbg !78
  %163 = shl i32 %148, 1, !dbg !78
  %164 = select i1 %154, i32 %163, i32 -2147483648, !dbg !78
  %165 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %162, i32 %164, i32 0, i32 0), !dbg !78
  %166 = fmul float %28, 0x3FF7154760000000, !dbg !80
  %167 = shl nsw i64 %73, 5, !dbg !81
  %168 = shl nsw i64 %76, 5, !dbg !82
  %169 = shl nuw nsw i32 %58, 2, !dbg !74
  %170 = and i32 %169, 764, !dbg !74
  %171 = and i32 %57, 64, !dbg !74
  %172 = icmp eq i32 %171, 0, !dbg !74
  %173 = select i1 %172, i32 0, i32 272, !dbg !74
  %174 = xor i32 %170, %173, !dbg !74
  %175 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 10240), i32 %174, !dbg !74
  store i32 %158, ptr addrspace(3) %175, align 4, !dbg !74
  %176 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %174, !dbg !76
  store i32 %161, ptr addrspace(3) %176, align 4, !dbg !76
  %177 = select i1 %172, i32 0, i32 264, !dbg !78
  %178 = xor i32 %170, %177, !dbg !78
  %179 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288), i32 %178, !dbg !78
  store i32 %165, ptr addrspace(3) %179, align 4, !dbg !78
  %180 = icmp sgt i32 %153, 32, !dbg !72
  %181 = add i64 %151, %167, !dbg !83
  %182 = trunc i64 %181 to i32, !dbg !83
  %183 = shl i32 %182, 1, !dbg !74
  %184 = select i1 %180, i32 %183, i32 -2147483648, !dbg !74
  %185 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %155, i32 %184, i32 0, i32 0), !dbg !74
  %186 = add i64 %149, %167, !dbg !84
  %187 = trunc i64 %186 to i32, !dbg !84
  %188 = shl i32 %187, 1, !dbg !76
  %189 = select i1 %180, i32 %188, i32 -2147483648, !dbg !76
  %190 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %155, i32 %189, i32 0, i32 0), !dbg !76
  %191 = add i64 %147, %168, !dbg !85
  %192 = trunc i64 %191 to i32, !dbg !85
  %193 = shl i32 %192, 1, !dbg !78
  %194 = select i1 %180, i32 %193, i32 -2147483648, !dbg !78
  %195 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %162, i32 %194, i32 0, i32 0), !dbg !78
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !74
  tail call void @llvm.amdgcn.s.barrier(), !dbg !74
  %196 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 11264), i32 %174, !dbg !74
  store i32 %185, ptr addrspace(3) %196, align 4, !dbg !74
  %197 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 9216), i32 %174, !dbg !76
  store i32 %190, ptr addrspace(3) %197, align 4, !dbg !76
  %198 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 13312), i32 %178, !dbg !78
  store i32 %195, ptr addrspace(3) %198, align 4, !dbg !78
  %199 = add i32 %153, -64, !dbg !72
  %200 = icmp sgt i32 %199, 0, !dbg !72
  %201 = shl nuw nsw i32 %38, 5
  %202 = and i32 %201, 736
  %203 = and i32 %38, 8
  br i1 %200, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !72

.._crit_edge_crit_edge:                           ; preds = %146
  %.pre140 = lshr exact i32 %97, 1, !dbg !74
  %.pre142 = and i32 %38, 31, !dbg !78
  %.pre144 = shl nuw nsw i32 %.pre142, 3, !dbg !78
  br label %._crit_edge, !dbg !72

.lr.ph:                                           ; preds = %146
  %204 = icmp eq i32 %203, 0
  %205 = select i1 %204, i32 0, i32 272
  %206 = lshr exact i32 %97, 1
  %207 = xor i32 %205, %206
  %208 = or disjoint i32 %207, %202
  %209 = and i32 %38, 31
  %210 = shl nuw nsw i32 %209, 3
  %211 = icmp eq i32 %97, 0
  %212 = select i1 %211, i32 0, i32 264
  %213 = xor i32 %212, %210
  %214 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %215 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %216 = shufflevector <8 x bfloat> %214, <8 x bfloat> %215, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %217 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %218 = shufflevector <8 x bfloat> %216, <8 x bfloat> %217, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %219 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %220 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %221 = shufflevector <8 x bfloat> %219, <8 x bfloat> %220, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %222 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %223 = shufflevector <8 x bfloat> %221, <8 x bfloat> %222, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %224 = and i32 %169, 380
  %225 = and i32 %57, 128
  %226 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %224
  %227 = getelementptr inbounds nuw i8, ptr addrspace(3) %226, i32 %225
  %228 = shl nuw nsw i32 %59, 2
  %229 = shl nuw nsw i32 %225, 1
  %230 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %229
  %231 = getelementptr inbounds nuw i8, ptr addrspace(3) %230, i32 %228
  %232 = getelementptr inbounds nuw i8, ptr addrspace(3) %231, i32 %171
  %233 = getelementptr inbounds nuw i8, ptr addrspace(3) %232, i32 128
  %234 = shl nuw nsw i32 %209, 1
  %235 = select i1 %211, i32 0, i32 1056
  %236 = or disjoint i32 %57, %234
  %237 = xor i32 %236, %235
  %238 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %237
  %239 = getelementptr inbounds nuw i8, ptr addrspace(3) %238, i32 4096
  %240 = xor i32 %237, 264
  %241 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %240
  %242 = getelementptr inbounds nuw i8, ptr addrspace(3) %241, i32 4096
  %243 = xor i32 %237, 528
  %244 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %243
  %245 = getelementptr inbounds nuw i8, ptr addrspace(3) %244, i32 4096
  %246 = xor i32 %237, 792
  %247 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %246
  %248 = getelementptr inbounds nuw i8, ptr addrspace(3) %247, i32 4096
  %249 = xor i32 %237, 2112
  %250 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %249
  %251 = getelementptr inbounds nuw i8, ptr addrspace(3) %250, i32 4096
  %252 = xor i32 %237, 2376
  %253 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %252
  %254 = getelementptr inbounds nuw i8, ptr addrspace(3) %253, i32 4096
  %255 = xor i32 %237, 2640
  %256 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %255
  %257 = getelementptr inbounds nuw i8, ptr addrspace(3) %256, i32 4096
  %258 = xor i32 %237, 2904
  %259 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %258
  %260 = getelementptr inbounds nuw i8, ptr addrspace(3) %259, i32 4096
  %261 = and i32 %38, 60
  %262 = shl nuw nsw i32 %261, 6
  %263 = and i32 %67, 24
  %264 = shl nuw nsw i32 %261, 1
  %265 = shl nuw nsw i32 %41, 5
  %266 = or disjoint i32 %262, %263
  %267 = xor i32 %266, %264
  %268 = xor i32 %267, %265
  %269 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %268
  %270 = getelementptr inbounds nuw i8, ptr addrspace(3) %269, i32 4096
  %271 = getelementptr inbounds nuw i8, ptr addrspace(3) %269, i32 128
  %272 = getelementptr inbounds nuw i8, ptr addrspace(3) %269, i32 4224
  %273 = insertelement <8 x float> poison, float %166, i64 0, !dbg !86
  %274 = shufflevector <8 x float> %273, <8 x float> poison, <8 x i32> zeroinitializer, !dbg !86
  %275 = insertelement <4 x float> poison, float %166, i64 0, !dbg !86
  %276 = shufflevector <4 x float> %275, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !86
  %277 = insertelement <2 x float> poison, float %166, i64 0, !dbg !86
  %278 = shufflevector <2 x float> %277, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !86
  br label %279, !dbg !72

279:                                              ; preds = %.lr.ph, %279
  %280 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 13312), %.lr.ph ], [ %505, %279 ]
  %281 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288), %.lr.ph ], [ %280, %279 ]
  %282 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 9216), %.lr.ph ], [ %503, %279 ]
  %283 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), %.lr.ph ], [ %282, %279 ]
  %284 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 11264), %.lr.ph ], [ %501, %279 ]
  %285 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 10240), %.lr.ph ], [ %284, %279 ]
  %286 = phi i32 [ 1, %.lr.ph ], [ %499, %279 ]
  %.pn28104 = phi i64 [ %191, %.lr.ph ], [ %301, %279 ]
  %.pn23102 = phi i64 [ %186, %.lr.ph ], [ %298, %279 ]
  %.pn18100 = phi i64 [ %181, %.lr.ph ], [ %295, %279 ]
  %287 = phi float [ 0xFFF0000000000000, %.lr.ph ], [ %371, %279 ]
  %288 = phi float [ 1.000000e+00, %.lr.ph ], [ %449, %279 ]
  %289 = phi i32 [ 0, %.lr.ph ], [ %507, %279 ]
  %290 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %510, %279 ]
  %291 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %509, %279 ]
  %292 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %512, %279 ]
  %293 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %511, %279 ]
  %sext81 = shl i64 %.pn18100, 32, !dbg !83
  %294 = ashr exact i64 %sext81, 32, !dbg !83
  %295 = add nsw i64 %294, %167, !dbg !83
  %296 = trunc i64 %295 to i32, !dbg !83
  %sext83 = shl i64 %.pn23102, 32, !dbg !84
  %297 = ashr exact i64 %sext83, 32, !dbg !84
  %298 = add nsw i64 %297, %167, !dbg !84
  %299 = trunc i64 %298 to i32, !dbg !84
  %sext85 = shl i64 %.pn28104, 32, !dbg !85
  %300 = ashr exact i64 %sext85, 32, !dbg !85
  %301 = add nsw i64 %300, %168, !dbg !85
  %302 = trunc i64 %301 to i32, !dbg !85
  %303 = shl i32 %296, 1, !dbg !74
  %304 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %155, i32 %303, i32 0, i32 0), !dbg !74
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !74
  tail call void @llvm.amdgcn.s.barrier(), !dbg !74
  %305 = getelementptr inbounds nuw i8, ptr addrspace(3) %285, i32 %208, !dbg !74
  %306 = load <8 x bfloat>, ptr addrspace(3) %305, align 16, !dbg !74
  %307 = shl i32 %299, 1, !dbg !76
  %308 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %155, i32 %307, i32 0, i32 0), !dbg !76
  %309 = getelementptr inbounds nuw i8, ptr addrspace(3) %283, i32 %208, !dbg !76
  %310 = load <8 x bfloat>, ptr addrspace(3) %309, align 16, !dbg !76
  %311 = shl i32 %302, 1, !dbg !78
  %312 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %162, i32 %311, i32 0, i32 0), !dbg !78
  %313 = getelementptr inbounds nuw i8, ptr addrspace(3) %281, i32 %213, !dbg !78
  %314 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %313), !dbg !78
  %315 = getelementptr inbounds nuw i8, ptr addrspace(3) %313, i32 512, !dbg !78
  %316 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %315), !dbg !78
  %317 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %310, <8 x bfloat> %218, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !87
  %318 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %306, <8 x bfloat> %223, <16 x float> %317, i32 0, i32 0, i32 0), !dbg !88
  %319 = extractelement <16 x float> %318, i64 8, !dbg !88
  %320 = extractelement <16 x float> %318, i64 15, !dbg !88
  %321 = shufflevector <16 x float> %318, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !86
  %322 = fmul <8 x float> %274, %321, !dbg !86
  %323 = fmul float %166, %319, !dbg !86
  %324 = shufflevector <16 x float> %318, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !86
  %325 = fmul <4 x float> %276, %324, !dbg !86
  %326 = fmul float %166, %320, !dbg !86
  %327 = extractelement <8 x float> %322, i64 0, !dbg !89
  %328 = extractelement <8 x float> %322, i64 1, !dbg !89
  %329 = tail call float @llvm.maxnum.f32(float %327, float %328), !dbg !90
  %330 = extractelement <8 x float> %322, i64 2, !dbg !89
  %331 = tail call float @llvm.maxnum.f32(float %329, float %330), !dbg !90
  %332 = extractelement <8 x float> %322, i64 3, !dbg !89
  %333 = extractelement <8 x float> %322, i64 4, !dbg !89
  %334 = tail call float @llvm.maxnum.f32(float %332, float %333), !dbg !90
  %335 = extractelement <8 x float> %322, i64 5, !dbg !89
  %336 = tail call float @llvm.maxnum.f32(float %334, float %335), !dbg !90
  %337 = extractelement <8 x float> %322, i64 6, !dbg !89
  %338 = extractelement <8 x float> %322, i64 7, !dbg !89
  %339 = tail call float @llvm.maxnum.f32(float %337, float %338), !dbg !90
  %340 = tail call float @llvm.maxnum.f32(float %339, float %323), !dbg !90
  %341 = extractelement <4 x float> %325, i64 0, !dbg !89
  %342 = extractelement <4 x float> %325, i64 1, !dbg !89
  %343 = tail call float @llvm.maxnum.f32(float %341, float %342), !dbg !90
  %344 = extractelement <4 x float> %325, i64 2, !dbg !89
  %345 = tail call float @llvm.maxnum.f32(float %343, float %344), !dbg !90
  %346 = extractelement <4 x float> %325, i64 3, !dbg !89
  %347 = tail call float @llvm.maxnum.f32(float %331, float %336), !dbg !90
  %348 = tail call float @llvm.maxnum.f32(float %347, float %340), !dbg !90
  %349 = shufflevector <8 x float> %322, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !89
  %350 = shufflevector <8 x float> %322, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !89
  %351 = shufflevector <8 x float> %322, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !89
  %352 = shufflevector <8 x float> %322, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !89
  %353 = shufflevector <4 x float> %325, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !89
  %354 = shufflevector <4 x float> %325, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !89
  %355 = shufflevector <16 x float> %318, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !86
  %356 = fmul <2 x float> %278, %355, !dbg !86
  %357 = extractelement <2 x float> %356, i64 0, !dbg !90
  %358 = tail call float @llvm.maxnum.f32(float %346, float %357), !dbg !90
  %359 = extractelement <2 x float> %356, i64 1, !dbg !90
  %360 = tail call float @llvm.maxnum.f32(float %358, float %359), !dbg !90
  %361 = tail call float @llvm.maxnum.f32(float %345, float %360), !dbg !90
  %362 = tail call float @llvm.maxnum.f32(float %361, float %326), !dbg !90
  %363 = tail call float @llvm.maxnum.f32(float %348, float %362), !dbg !90
  %364 = bitcast float %363 to i32, !dbg !93
  %365 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %364, i32 %364, i1 false, i1 false), !dbg !93
  %366 = extractvalue { i32, i32 } %365, 0, !dbg !93
  %367 = extractvalue { i32, i32 } %365, 1, !dbg !93
  %368 = bitcast i32 %366 to float, !dbg !93
  %369 = bitcast i32 %367 to float, !dbg !93
  %370 = tail call float @llvm.maxnum.f32(float %368, float %369), !dbg !90
  %371 = tail call float @llvm.maxnum.f32(float %287, float %370), !dbg !95
  %372 = insertelement <2 x float> poison, float %371, i64 0, !dbg !89
  %373 = shufflevector <2 x float> %372, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !89
  %374 = fsub <2 x float> %349, %373, !dbg !89
  %375 = fsub <2 x float> %350, %373, !dbg !89
  %376 = fsub <2 x float> %351, %373, !dbg !89
  %377 = fsub <2 x float> %352, %373, !dbg !89
  %378 = fsub float %323, %371, !dbg !89
  %379 = fsub <2 x float> %353, %373, !dbg !89
  %380 = fsub <2 x float> %354, %373, !dbg !89
  %381 = fsub <2 x float> %356, %373, !dbg !89
  %382 = fsub float %326, %371, !dbg !89
  %383 = extractelement <2 x float> %374, i64 0, !dbg !96
  %384 = tail call float @llvm.amdgcn.exp2.f32(float %383), !dbg !96
  %385 = extractelement <2 x float> %374, i64 1, !dbg !96
  %386 = tail call float @llvm.amdgcn.exp2.f32(float %385), !dbg !96
  %387 = extractelement <2 x float> %375, i64 0, !dbg !96
  %388 = tail call float @llvm.amdgcn.exp2.f32(float %387), !dbg !96
  %389 = extractelement <2 x float> %375, i64 1, !dbg !96
  %390 = tail call float @llvm.amdgcn.exp2.f32(float %389), !dbg !96
  %391 = extractelement <2 x float> %376, i64 0, !dbg !96
  %392 = tail call float @llvm.amdgcn.exp2.f32(float %391), !dbg !96
  %393 = extractelement <2 x float> %376, i64 1, !dbg !96
  %394 = tail call float @llvm.amdgcn.exp2.f32(float %393), !dbg !96
  %395 = extractelement <2 x float> %377, i64 0, !dbg !96
  %396 = tail call float @llvm.amdgcn.exp2.f32(float %395), !dbg !96
  %397 = extractelement <2 x float> %377, i64 1, !dbg !96
  %398 = tail call float @llvm.amdgcn.exp2.f32(float %397), !dbg !96
  %399 = tail call float @llvm.amdgcn.exp2.f32(float %378), !dbg !96
  %400 = extractelement <2 x float> %379, i64 0, !dbg !96
  %401 = tail call float @llvm.amdgcn.exp2.f32(float %400), !dbg !96
  %402 = extractelement <2 x float> %379, i64 1, !dbg !96
  %403 = tail call float @llvm.amdgcn.exp2.f32(float %402), !dbg !96
  %404 = extractelement <2 x float> %380, i64 0, !dbg !96
  %405 = tail call float @llvm.amdgcn.exp2.f32(float %404), !dbg !96
  %406 = extractelement <2 x float> %380, i64 1, !dbg !96
  %407 = tail call float @llvm.amdgcn.exp2.f32(float %406), !dbg !96
  %408 = extractelement <2 x float> %381, i64 0, !dbg !96
  %409 = tail call float @llvm.amdgcn.exp2.f32(float %408), !dbg !96
  %410 = extractelement <2 x float> %381, i64 1, !dbg !96
  %411 = tail call float @llvm.amdgcn.exp2.f32(float %410), !dbg !96
  %412 = tail call float @llvm.amdgcn.exp2.f32(float %382), !dbg !96
  %413 = insertelement <2 x float> poison, float %384, i64 0, !dbg !97
  %414 = insertelement <2 x float> %413, float %386, i64 1, !dbg !97
  %415 = insertelement <2 x float> poison, float %388, i64 0, !dbg !97
  %416 = insertelement <2 x float> %415, float %390, i64 1, !dbg !97
  %417 = insertelement <2 x float> poison, float %392, i64 0, !dbg !97
  %418 = insertelement <2 x float> %417, float %394, i64 1, !dbg !97
  %419 = insertelement <2 x float> poison, float %396, i64 0, !dbg !97
  %420 = insertelement <2 x float> %419, float %398, i64 1, !dbg !97
  %421 = insertelement <2 x float> poison, float %399, i64 0, !dbg !97
  %422 = insertelement <2 x float> %421, float %401, i64 1, !dbg !97
  %423 = insertelement <2 x float> poison, float %403, i64 0, !dbg !97
  %424 = insertelement <2 x float> %423, float %405, i64 1, !dbg !97
  %425 = insertelement <2 x float> poison, float %407, i64 0, !dbg !97
  %426 = insertelement <2 x float> %425, float %409, i64 1, !dbg !97
  %427 = insertelement <2 x float> poison, float %411, i64 0, !dbg !97
  %428 = insertelement <2 x float> %427, float %412, i64 1, !dbg !97
  %429 = fadd <2 x float> %414, %416, !dbg !97
  %430 = fadd <2 x float> %418, %420, !dbg !97
  %431 = fadd <2 x float> %422, %424, !dbg !97
  %432 = fadd <2 x float> %426, %428, !dbg !97
  %433 = fadd <2 x float> %429, %430, !dbg !97
  %434 = fadd <2 x float> %431, %432, !dbg !97
  %435 = fadd <2 x float> %433, %434, !dbg !97
  %shift = shufflevector <2 x float> %435, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !99
  %foldExtExtBinop = fadd <2 x float> %435, %shift, !dbg !99
  %bc = bitcast <2 x float> %foldExtExtBinop to <2 x i32>, !dbg !97
  %436 = extractelement <2 x i32> %bc, i64 0, !dbg !97
  %437 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %436, i32 %436, i1 false, i1 false), !dbg !97
  %438 = extractvalue { i32, i32 } %437, 0, !dbg !97
  %439 = extractvalue { i32, i32 } %437, 1, !dbg !97
  %440 = bitcast i32 %438 to float, !dbg !97
  %441 = bitcast i32 %439 to float, !dbg !97
  %442 = fadd float %440, %441, !dbg !99
  %443 = fsub float %287, %371, !dbg !100
  %444 = tail call float @llvm.amdgcn.exp2.f32(float %443), !dbg !101
  %445 = insertelement <1 x float> poison, float %444, i64 0, !dbg !102
  store <1 x float> %445, ptr addrspace(3) %227, align 4, !dbg !102
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !102
  tail call void @llvm.amdgcn.s.barrier(), !dbg !102
  %446 = load <1 x float>, ptr addrspace(3) %232, align 4, !dbg !102
  %447 = load <1 x float>, ptr addrspace(3) %233, align 4, !dbg !102
  %448 = fmul float %288, %444, !dbg !103
  %449 = fadd float %442, %448, !dbg !103
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !104
  tail call void @llvm.amdgcn.s.barrier(), !dbg !104
  %450 = shufflevector <2 x float> %413, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %451 = fptrunc <1 x float> %450 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %451, ptr addrspace(3) %238, align 2, !dbg !104
  %452 = shufflevector <2 x float> %421, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %453 = fptrunc <1 x float> %452 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %453, ptr addrspace(3) %239, align 2, !dbg !104
  %454 = shufflevector <2 x float> %414, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %455 = fptrunc <1 x float> %454 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %455, ptr addrspace(3) %241, align 2, !dbg !104
  %456 = shufflevector <2 x float> %422, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %457 = fptrunc <1 x float> %456 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %457, ptr addrspace(3) %242, align 2, !dbg !104
  %458 = shufflevector <2 x float> %415, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %459 = fptrunc <1 x float> %458 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %459, ptr addrspace(3) %244, align 2, !dbg !104
  %460 = shufflevector <2 x float> %423, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %461 = fptrunc <1 x float> %460 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %461, ptr addrspace(3) %245, align 2, !dbg !104
  %462 = shufflevector <2 x float> %416, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %463 = fptrunc <1 x float> %462 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %463, ptr addrspace(3) %247, align 2, !dbg !104
  %464 = shufflevector <2 x float> %424, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %465 = fptrunc <1 x float> %464 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %465, ptr addrspace(3) %248, align 2, !dbg !104
  %466 = shufflevector <2 x float> %417, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %467 = fptrunc <1 x float> %466 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %467, ptr addrspace(3) %250, align 2, !dbg !104
  %468 = shufflevector <2 x float> %425, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %469 = fptrunc <1 x float> %468 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %469, ptr addrspace(3) %251, align 2, !dbg !104
  %470 = shufflevector <2 x float> %418, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %471 = fptrunc <1 x float> %470 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %471, ptr addrspace(3) %253, align 2, !dbg !104
  %472 = shufflevector <2 x float> %426, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %473 = fptrunc <1 x float> %472 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %473, ptr addrspace(3) %254, align 2, !dbg !104
  %474 = shufflevector <2 x float> %419, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %475 = fptrunc <1 x float> %474 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %475, ptr addrspace(3) %256, align 2, !dbg !104
  %476 = shufflevector <2 x float> %427, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %477 = fptrunc <1 x float> %476 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %477, ptr addrspace(3) %257, align 2, !dbg !104
  %478 = shufflevector <2 x float> %420, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %479 = fptrunc <1 x float> %478 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %479, ptr addrspace(3) %259, align 2, !dbg !104
  %480 = shufflevector <2 x float> %428, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %481 = fptrunc <1 x float> %480 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %481, ptr addrspace(3) %260, align 2, !dbg !104
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !104
  tail call void @llvm.amdgcn.s.barrier(), !dbg !104
  %482 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %269), !dbg !104
  %483 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %270), !dbg !104
  %484 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %271), !dbg !104
  %485 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %272), !dbg !104
  %486 = shufflevector <4 x bfloat> %482, <4 x bfloat> %483, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %487 = shufflevector <4 x bfloat> %484, <4 x bfloat> %485, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %488 = shufflevector <4 x bfloat> %314, <4 x bfloat> %316, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %489 = shufflevector <2 x float> %292, <2 x float> %293, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !105
  %490 = shufflevector <1 x float> %446, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !105
  %491 = fmul <4 x float> %489, %490, !dbg !105
  %492 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %488, <8 x bfloat> %486, <4 x float> %491, i32 0, i32 0, i32 0), !dbg !105
  %493 = shufflevector <2 x float> %290, <2 x float> %291, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !105
  %494 = shufflevector <1 x float> %447, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !105
  %495 = fmul <4 x float> %493, %494, !dbg !105
  %496 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %488, <8 x bfloat> %487, <4 x float> %495, i32 0, i32 0, i32 0), !dbg !105
  %497 = add i32 %286, 1, !dbg !72
  %498 = icmp slt i32 %497, 2, !dbg !72
  %499 = select i1 %498, i32 %497, i32 0, !dbg !72
  %500 = shl i32 %499, 9, !dbg !74
  %501 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 10240), i32 %500, !dbg !74
  %502 = getelementptr inbounds nuw i8, ptr addrspace(3) %501, i32 %174, !dbg !74
  store i32 %304, ptr addrspace(3) %502, align 4, !dbg !74
  %503 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %500, !dbg !76
  %504 = getelementptr inbounds nuw i8, ptr addrspace(3) %503, i32 %174, !dbg !76
  store i32 %308, ptr addrspace(3) %504, align 4, !dbg !76
  %505 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288), i32 %500, !dbg !78
  %506 = getelementptr inbounds nuw i8, ptr addrspace(3) %505, i32 %178, !dbg !78
  store i32 %312, ptr addrspace(3) %506, align 4, !dbg !78
  %507 = add nuw nsw i32 %289, 32, !dbg !72
  %508 = icmp slt i32 %507, %199, !dbg !72
  %509 = shufflevector <4 x float> %496, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !106
  %510 = shufflevector <4 x float> %496, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !106
  %511 = shufflevector <4 x float> %492, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !106
  %512 = shufflevector <4 x float> %492, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !106
  br i1 %508, label %279, label %._crit_edge, !dbg !72

._crit_edge:                                      ; preds = %279, %.._crit_edge_crit_edge
  %.pre-phi145 = phi i32 [ %.pre144, %.._crit_edge_crit_edge ], [ %210, %279 ], !dbg !78
  %.pre-phi143 = phi i32 [ %.pre142, %.._crit_edge_crit_edge ], [ %209, %279 ], !dbg !78
  %.pre-phi141 = phi i32 [ %.pre140, %.._crit_edge_crit_edge ], [ %206, %279 ], !dbg !74
  %.lcssa98 = phi float [ 1.000000e+00, %.._crit_edge_crit_edge ], [ %449, %279 ], !dbg !107
  %.lcssa97 = phi float [ 0xFFF0000000000000, %.._crit_edge_crit_edge ], [ %371, %279 ], !dbg !108
  %513 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 10240), %.._crit_edge_crit_edge ], [ %284, %279 ], !dbg !109
  %.lcssa95 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 11264), %.._crit_edge_crit_edge ], [ %501, %279 ], !dbg !110
  %514 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), %.._crit_edge_crit_edge ], [ %282, %279 ], !dbg !111
  %.lcssa93 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 9216), %.._crit_edge_crit_edge ], [ %503, %279 ], !dbg !112
  %515 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288), %.._crit_edge_crit_edge ], [ %280, %279 ], !dbg !113
  %.lcssa91 = phi ptr addrspace(3) [ getelementptr (i8, ptr addrspace(3) @global_smem, i32 13312), %.._crit_edge_crit_edge ], [ %505, %279 ], !dbg !114
  %516 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %509, %279 ], !dbg !106
  %517 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %510, %279 ], !dbg !106
  %518 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %511, %279 ], !dbg !106
  %519 = phi <2 x float> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %512, %279 ], !dbg !106
  %520 = or disjoint i32 %153, 31, !dbg !72
  %521 = icmp sgt i32 %520, 63, !dbg !72
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !74
  tail call void @llvm.amdgcn.s.barrier(), !dbg !74
  %522 = icmp eq i32 %203, 0, !dbg !74
  %523 = select i1 %522, i32 0, i32 272, !dbg !74
  %524 = xor i32 %523, %.pre-phi141, !dbg !74
  %525 = or disjoint i32 %524, %202, !dbg !74
  %526 = icmp eq i32 %97, 0, !dbg !78
  %527 = select i1 %526, i32 0, i32 264, !dbg !78
  %528 = xor i32 %527, %.pre-phi145, !dbg !78
  %529 = getelementptr inbounds nuw i8, ptr addrspace(3) %515, i32 %528, !dbg !78
  %530 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %529), !dbg !78
  %531 = or disjoint i32 %528, 512, !dbg !78
  %532 = getelementptr inbounds nuw i8, ptr addrspace(3) %515, i32 %531, !dbg !78
  %533 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %532), !dbg !78
  br i1 %154, label %534, label %556, !dbg !88

534:                                              ; preds = %._crit_edge
  %535 = getelementptr inbounds nuw i8, ptr addrspace(3) %514, i32 %525, !dbg !76
  %536 = load <8 x bfloat>, ptr addrspace(3) %535, align 16, !dbg !76
  %537 = getelementptr inbounds nuw i8, ptr addrspace(3) %513, i32 %525, !dbg !74
  %538 = load <8 x bfloat>, ptr addrspace(3) %537, align 16, !dbg !74
  %539 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !87
  %540 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !87
  %541 = shufflevector <8 x bfloat> %539, <8 x bfloat> %540, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !87
  %542 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !87
  %543 = shufflevector <8 x bfloat> %541, <8 x bfloat> %542, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !87
  %544 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %536, <8 x bfloat> %543, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !87
  %545 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !88
  %546 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !88
  %547 = shufflevector <8 x bfloat> %545, <8 x bfloat> %546, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !88
  %548 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !88
  %549 = shufflevector <8 x bfloat> %547, <8 x bfloat> %548, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !88
  %550 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %538, <8 x bfloat> %549, <16 x float> %544, i32 0, i32 0, i32 0), !dbg !88
  %551 = extractelement <16 x float> %550, i64 8, !dbg !88
  %552 = extractelement <16 x float> %550, i64 15, !dbg !88
  %553 = shufflevector <16 x float> %550, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !115
  %554 = shufflevector <16 x float> %550, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !115
  %555 = shufflevector <16 x float> %550, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !115
  br label %556, !dbg !88

556:                                              ; preds = %534, %._crit_edge
  %557 = phi float [ %551, %534 ], [ 0.000000e+00, %._crit_edge ], !dbg !115
  %558 = phi float [ %552, %534 ], [ 0.000000e+00, %._crit_edge ], !dbg !115
  %559 = phi <8 x float> [ %553, %534 ], [ zeroinitializer, %._crit_edge ], !dbg !115
  %560 = phi <4 x float> [ %554, %534 ], [ zeroinitializer, %._crit_edge ], !dbg !115
  %561 = phi <2 x float> [ %555, %534 ], [ zeroinitializer, %._crit_edge ], !dbg !115
  %562 = insertelement <8 x float> poison, float %166, i64 0, !dbg !86
  %563 = shufflevector <8 x float> %562, <8 x float> poison, <8 x i32> zeroinitializer, !dbg !86
  %564 = fmul <8 x float> %563, %559, !dbg !86
  %565 = fmul float %166, %557, !dbg !86
  %566 = insertelement <4 x float> poison, float %166, i64 0, !dbg !86
  %567 = shufflevector <4 x float> %566, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !86
  %568 = fmul <4 x float> %567, %560, !dbg !86
  %569 = insertelement <2 x float> poison, float %166, i64 0, !dbg !86
  %570 = shufflevector <2 x float> %569, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !86
  %571 = fmul <2 x float> %570, %561, !dbg !86
  %572 = fmul float %166, %558, !dbg !86
  %573 = extractelement <8 x float> %564, i64 0, !dbg !89
  %574 = extractelement <8 x float> %564, i64 1, !dbg !89
  %575 = tail call float @llvm.maxnum.f32(float %573, float %574), !dbg !90
  %576 = extractelement <8 x float> %564, i64 2, !dbg !89
  %577 = tail call float @llvm.maxnum.f32(float %575, float %576), !dbg !90
  %578 = extractelement <8 x float> %564, i64 3, !dbg !89
  %579 = extractelement <8 x float> %564, i64 4, !dbg !89
  %580 = tail call float @llvm.maxnum.f32(float %578, float %579), !dbg !90
  %581 = extractelement <8 x float> %564, i64 5, !dbg !89
  %582 = tail call float @llvm.maxnum.f32(float %580, float %581), !dbg !90
  %583 = extractelement <8 x float> %564, i64 6, !dbg !89
  %584 = extractelement <8 x float> %564, i64 7, !dbg !89
  %585 = tail call float @llvm.maxnum.f32(float %583, float %584), !dbg !90
  %586 = tail call float @llvm.maxnum.f32(float %585, float %565), !dbg !90
  %587 = extractelement <4 x float> %568, i64 0, !dbg !89
  %588 = extractelement <4 x float> %568, i64 1, !dbg !89
  %589 = tail call float @llvm.maxnum.f32(float %587, float %588), !dbg !90
  %590 = extractelement <4 x float> %568, i64 2, !dbg !89
  %591 = tail call float @llvm.maxnum.f32(float %589, float %590), !dbg !90
  %592 = extractelement <4 x float> %568, i64 3, !dbg !89
  %593 = extractelement <2 x float> %571, i64 0, !dbg !90
  %594 = tail call float @llvm.maxnum.f32(float %592, float %593), !dbg !90
  %595 = extractelement <2 x float> %571, i64 1, !dbg !90
  %596 = tail call float @llvm.maxnum.f32(float %594, float %595), !dbg !90
  %597 = tail call float @llvm.maxnum.f32(float %577, float %582), !dbg !90
  %598 = tail call float @llvm.maxnum.f32(float %597, float %586), !dbg !90
  %599 = tail call float @llvm.maxnum.f32(float %591, float %596), !dbg !90
  %600 = tail call float @llvm.maxnum.f32(float %599, float %572), !dbg !90
  %601 = tail call float @llvm.maxnum.f32(float %598, float %600), !dbg !90
  %602 = bitcast float %601 to i32, !dbg !93
  %603 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %602, i32 %602, i1 false, i1 false), !dbg !93
  %604 = extractvalue { i32, i32 } %603, 0, !dbg !93
  %605 = extractvalue { i32, i32 } %603, 1, !dbg !93
  %606 = bitcast i32 %604 to float, !dbg !93
  %607 = bitcast i32 %605 to float, !dbg !93
  %608 = tail call float @llvm.maxnum.f32(float %606, float %607), !dbg !90
  %609 = tail call float @llvm.maxnum.f32(float %.lcssa97, float %608), !dbg !95
  %610 = shufflevector <8 x float> %564, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !89
  %611 = insertelement <2 x float> poison, float %609, i64 0, !dbg !89
  %612 = shufflevector <2 x float> %611, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !89
  %613 = fsub <2 x float> %610, %612, !dbg !89
  %614 = shufflevector <8 x float> %564, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !89
  %615 = fsub <2 x float> %614, %612, !dbg !89
  %616 = shufflevector <8 x float> %564, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !89
  %617 = fsub <2 x float> %616, %612, !dbg !89
  %618 = shufflevector <8 x float> %564, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !89
  %619 = fsub <2 x float> %618, %612, !dbg !89
  %620 = fsub float %565, %609, !dbg !89
  %621 = shufflevector <4 x float> %568, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !89
  %622 = fsub <2 x float> %621, %612, !dbg !89
  %623 = shufflevector <4 x float> %568, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !89
  %624 = fsub <2 x float> %623, %612, !dbg !89
  %625 = fsub <2 x float> %571, %612, !dbg !89
  %626 = fsub float %572, %609, !dbg !89
  %627 = extractelement <2 x float> %613, i64 0, !dbg !96
  %628 = tail call float @llvm.amdgcn.exp2.f32(float %627), !dbg !96
  %629 = extractelement <2 x float> %613, i64 1, !dbg !96
  %630 = tail call float @llvm.amdgcn.exp2.f32(float %629), !dbg !96
  %631 = extractelement <2 x float> %615, i64 0, !dbg !96
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %631), !dbg !96
  %633 = extractelement <2 x float> %615, i64 1, !dbg !96
  %634 = tail call float @llvm.amdgcn.exp2.f32(float %633), !dbg !96
  %635 = extractelement <2 x float> %617, i64 0, !dbg !96
  %636 = tail call float @llvm.amdgcn.exp2.f32(float %635), !dbg !96
  %637 = extractelement <2 x float> %617, i64 1, !dbg !96
  %638 = tail call float @llvm.amdgcn.exp2.f32(float %637), !dbg !96
  %639 = extractelement <2 x float> %619, i64 0, !dbg !96
  %640 = tail call float @llvm.amdgcn.exp2.f32(float %639), !dbg !96
  %641 = extractelement <2 x float> %619, i64 1, !dbg !96
  %642 = tail call float @llvm.amdgcn.exp2.f32(float %641), !dbg !96
  %643 = tail call float @llvm.amdgcn.exp2.f32(float %620), !dbg !96
  %644 = extractelement <2 x float> %622, i64 0, !dbg !96
  %645 = tail call float @llvm.amdgcn.exp2.f32(float %644), !dbg !96
  %646 = extractelement <2 x float> %622, i64 1, !dbg !96
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %646), !dbg !96
  %648 = extractelement <2 x float> %624, i64 0, !dbg !96
  %649 = tail call float @llvm.amdgcn.exp2.f32(float %648), !dbg !96
  %650 = extractelement <2 x float> %624, i64 1, !dbg !96
  %651 = tail call float @llvm.amdgcn.exp2.f32(float %650), !dbg !96
  %652 = extractelement <2 x float> %625, i64 0, !dbg !96
  %653 = tail call float @llvm.amdgcn.exp2.f32(float %652), !dbg !96
  %654 = extractelement <2 x float> %625, i64 1, !dbg !96
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !96
  %656 = tail call float @llvm.amdgcn.exp2.f32(float %626), !dbg !96
  %657 = insertelement <2 x float> poison, float %628, i64 0, !dbg !97
  %658 = insertelement <2 x float> %657, float %630, i64 1, !dbg !97
  %659 = insertelement <2 x float> poison, float %632, i64 0, !dbg !97
  %660 = insertelement <2 x float> %659, float %634, i64 1, !dbg !97
  %661 = insertelement <2 x float> poison, float %636, i64 0, !dbg !97
  %662 = insertelement <2 x float> %661, float %638, i64 1, !dbg !97
  %663 = insertelement <2 x float> poison, float %640, i64 0, !dbg !97
  %664 = insertelement <2 x float> %663, float %642, i64 1, !dbg !97
  %665 = insertelement <2 x float> poison, float %643, i64 0, !dbg !97
  %666 = insertelement <2 x float> %665, float %645, i64 1, !dbg !97
  %667 = insertelement <2 x float> poison, float %647, i64 0, !dbg !97
  %668 = insertelement <2 x float> %667, float %649, i64 1, !dbg !97
  %669 = insertelement <2 x float> poison, float %651, i64 0, !dbg !97
  %670 = insertelement <2 x float> %669, float %653, i64 1, !dbg !97
  %671 = insertelement <2 x float> poison, float %655, i64 0, !dbg !97
  %672 = insertelement <2 x float> %671, float %656, i64 1, !dbg !97
  %673 = fadd <2 x float> %658, %660, !dbg !97
  %674 = fadd <2 x float> %662, %664, !dbg !97
  %675 = fadd <2 x float> %666, %668, !dbg !97
  %676 = fadd <2 x float> %670, %672, !dbg !97
  %677 = fadd <2 x float> %673, %674, !dbg !97
  %678 = fadd <2 x float> %675, %676, !dbg !97
  %679 = fadd <2 x float> %677, %678, !dbg !97
  %shift186 = shufflevector <2 x float> %679, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !99
  %foldExtExtBinop187 = fadd <2 x float> %679, %shift186, !dbg !99
  %bc195 = bitcast <2 x float> %foldExtExtBinop187 to <2 x i32>, !dbg !97
  %680 = extractelement <2 x i32> %bc195, i64 0, !dbg !97
  %681 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %680, i32 %680, i1 false, i1 false), !dbg !97
  %682 = fsub float %.lcssa97, %609, !dbg !100
  %683 = tail call float @llvm.amdgcn.exp2.f32(float %682), !dbg !101
  %684 = and i32 %169, 380, !dbg !102
  %685 = and i32 %57, 128, !dbg !102
  %686 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %684, !dbg !102
  %687 = getelementptr inbounds nuw i8, ptr addrspace(3) %686, i32 %685, !dbg !102
  %688 = insertelement <1 x float> poison, float %683, i64 0, !dbg !102
  store <1 x float> %688, ptr addrspace(3) %687, align 4, !dbg !102
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !102
  tail call void @llvm.amdgcn.s.barrier(), !dbg !102
  %689 = shl nuw nsw i32 %59, 2, !dbg !102
  %690 = shl nuw nsw i32 %685, 1, !dbg !102
  %691 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %690, !dbg !102
  %692 = getelementptr inbounds nuw i8, ptr addrspace(3) %691, i32 %689, !dbg !102
  %693 = getelementptr inbounds nuw i8, ptr addrspace(3) %692, i32 %171, !dbg !102
  %694 = load <1 x float>, ptr addrspace(3) %693, align 4, !dbg !102
  %695 = getelementptr inbounds nuw i8, ptr addrspace(3) %693, i32 128, !dbg !102
  %696 = load <1 x float>, ptr addrspace(3) %695, align 4, !dbg !102
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !104
  tail call void @llvm.amdgcn.s.barrier(), !dbg !104
  %697 = shl nuw nsw i32 %.pre-phi143, 1, !dbg !104
  %698 = select i1 %526, i32 0, i32 1056, !dbg !104
  %699 = or disjoint i32 %57, %697, !dbg !104
  %700 = xor i32 %699, %698, !dbg !104
  %701 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %700, !dbg !104
  %702 = shufflevector <2 x float> %657, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %703 = fptrunc <1 x float> %702 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %703, ptr addrspace(3) %701, align 2, !dbg !104
  %704 = getelementptr inbounds nuw i8, ptr addrspace(3) %701, i32 4096, !dbg !104
  %705 = shufflevector <2 x float> %665, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %706 = fptrunc <1 x float> %705 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %706, ptr addrspace(3) %704, align 2, !dbg !104
  %707 = xor i32 %700, 264, !dbg !104
  %708 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %707, !dbg !104
  %709 = shufflevector <2 x float> %658, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %710 = fptrunc <1 x float> %709 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %710, ptr addrspace(3) %708, align 2, !dbg !104
  %711 = getelementptr inbounds nuw i8, ptr addrspace(3) %708, i32 4096, !dbg !104
  %712 = shufflevector <2 x float> %666, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %713 = fptrunc <1 x float> %712 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %713, ptr addrspace(3) %711, align 2, !dbg !104
  %714 = xor i32 %700, 528, !dbg !104
  %715 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %714, !dbg !104
  %716 = shufflevector <2 x float> %659, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %717 = fptrunc <1 x float> %716 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %717, ptr addrspace(3) %715, align 2, !dbg !104
  %718 = getelementptr inbounds nuw i8, ptr addrspace(3) %715, i32 4096, !dbg !104
  %719 = shufflevector <2 x float> %667, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %720 = fptrunc <1 x float> %719 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %720, ptr addrspace(3) %718, align 2, !dbg !104
  %721 = xor i32 %700, 792, !dbg !104
  %722 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %721, !dbg !104
  %723 = shufflevector <2 x float> %660, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %724 = fptrunc <1 x float> %723 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %724, ptr addrspace(3) %722, align 2, !dbg !104
  %725 = getelementptr inbounds nuw i8, ptr addrspace(3) %722, i32 4096, !dbg !104
  %726 = shufflevector <2 x float> %668, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %727 = fptrunc <1 x float> %726 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %727, ptr addrspace(3) %725, align 2, !dbg !104
  %728 = xor i32 %700, 2112, !dbg !104
  %729 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %728, !dbg !104
  %730 = shufflevector <2 x float> %661, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %731 = fptrunc <1 x float> %730 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %731, ptr addrspace(3) %729, align 2, !dbg !104
  %732 = getelementptr inbounds nuw i8, ptr addrspace(3) %729, i32 4096, !dbg !104
  %733 = shufflevector <2 x float> %669, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %734 = fptrunc <1 x float> %733 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %734, ptr addrspace(3) %732, align 2, !dbg !104
  %735 = xor i32 %700, 2376, !dbg !104
  %736 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %735, !dbg !104
  %737 = shufflevector <2 x float> %662, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %738 = fptrunc <1 x float> %737 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %738, ptr addrspace(3) %736, align 2, !dbg !104
  %739 = getelementptr inbounds nuw i8, ptr addrspace(3) %736, i32 4096, !dbg !104
  %740 = shufflevector <2 x float> %670, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %741 = fptrunc <1 x float> %740 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %741, ptr addrspace(3) %739, align 2, !dbg !104
  %742 = xor i32 %700, 2640, !dbg !104
  %743 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %742, !dbg !104
  %744 = shufflevector <2 x float> %663, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %745 = fptrunc <1 x float> %744 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %745, ptr addrspace(3) %743, align 2, !dbg !104
  %746 = getelementptr inbounds nuw i8, ptr addrspace(3) %743, i32 4096, !dbg !104
  %747 = shufflevector <2 x float> %671, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %748 = fptrunc <1 x float> %747 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %748, ptr addrspace(3) %746, align 2, !dbg !104
  %749 = xor i32 %700, 2904, !dbg !104
  %750 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %749, !dbg !104
  %751 = shufflevector <2 x float> %664, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %752 = fptrunc <1 x float> %751 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %752, ptr addrspace(3) %750, align 2, !dbg !104
  %753 = getelementptr inbounds nuw i8, ptr addrspace(3) %750, i32 4096, !dbg !104
  %754 = shufflevector <2 x float> %672, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %755 = fptrunc <1 x float> %754 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %755, ptr addrspace(3) %753, align 2, !dbg !104
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !104
  tail call void @llvm.amdgcn.s.barrier(), !dbg !104
  %756 = and i32 %38, 60, !dbg !104
  %757 = shl nuw nsw i32 %756, 6, !dbg !104
  %758 = and i32 %67, 24, !dbg !104
  %759 = shl nuw nsw i32 %756, 1, !dbg !104
  %760 = shl nuw nsw i32 %41, 5, !dbg !104
  %761 = or disjoint i32 %757, %758, !dbg !104
  %762 = xor i32 %761, %759, !dbg !104
  %763 = xor i32 %762, %760, !dbg !104
  %764 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %763, !dbg !104
  %765 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %764), !dbg !104
  %766 = getelementptr inbounds nuw i8, ptr addrspace(3) %764, i32 4096, !dbg !104
  %767 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %766), !dbg !104
  %768 = getelementptr inbounds nuw i8, ptr addrspace(3) %764, i32 128, !dbg !104
  %769 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %768), !dbg !104
  %770 = getelementptr inbounds nuw i8, ptr addrspace(3) %764, i32 4224, !dbg !104
  %771 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %770), !dbg !104
  br i1 %154, label %772, label %795, !dbg !105

772:                                              ; preds = %556
  %773 = fmul float %.lcssa98, %683, !dbg !103
  %774 = extractvalue { i32, i32 } %681, 0, !dbg !97
  %775 = bitcast i32 %774 to float, !dbg !97
  %776 = extractvalue { i32, i32 } %681, 1, !dbg !97
  %777 = bitcast i32 %776 to float, !dbg !97
  %778 = fadd float %775, %777, !dbg !99
  %779 = fadd float %778, %773, !dbg !103
  %780 = shufflevector <4 x bfloat> %765, <4 x bfloat> %767, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %781 = shufflevector <4 x bfloat> %769, <4 x bfloat> %771, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %782 = shufflevector <4 x bfloat> %530, <4 x bfloat> %533, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %783 = shufflevector <2 x float> %519, <2 x float> %518, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !105
  %784 = shufflevector <1 x float> %694, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !105
  %785 = fmul <4 x float> %783, %784, !dbg !105
  %786 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %782, <8 x bfloat> %780, <4 x float> %785, i32 0, i32 0, i32 0), !dbg !105
  %787 = shufflevector <2 x float> %517, <2 x float> %516, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !105
  %788 = shufflevector <1 x float> %696, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !105
  %789 = fmul <4 x float> %787, %788, !dbg !105
  %790 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %782, <8 x bfloat> %781, <4 x float> %789, i32 0, i32 0, i32 0), !dbg !105
  %791 = shufflevector <4 x float> %790, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %792 = shufflevector <4 x float> %790, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %793 = shufflevector <4 x float> %786, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %794 = shufflevector <4 x float> %786, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  br label %795, !dbg !105

795:                                              ; preds = %772, %556
  %796 = phi float [ %609, %772 ], [ %.lcssa97, %556 ]
  %797 = phi float [ %779, %772 ], [ %.lcssa98, %556 ]
  %798 = phi <2 x float> [ %791, %772 ], [ %516, %556 ]
  %799 = phi <2 x float> [ %792, %772 ], [ %517, %556 ]
  %800 = phi <2 x float> [ %793, %772 ], [ %518, %556 ]
  %801 = phi <2 x float> [ %794, %772 ], [ %519, %556 ]
  %802 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa91, i32 %528, !dbg !78
  %803 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %802), !dbg !78
  %804 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa91, i32 %531, !dbg !78
  %805 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %804), !dbg !78
  br i1 %521, label %806, label %828, !dbg !88

806:                                              ; preds = %795
  %807 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa93, i32 %525, !dbg !76
  %808 = load <8 x bfloat>, ptr addrspace(3) %807, align 16, !dbg !76
  %809 = getelementptr inbounds nuw i8, ptr addrspace(3) %.lcssa95, i32 %525, !dbg !74
  %810 = load <8 x bfloat>, ptr addrspace(3) %809, align 16, !dbg !74
  %811 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !87
  %812 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !87
  %813 = shufflevector <8 x bfloat> %811, <8 x bfloat> %812, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !87
  %814 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !87
  %815 = shufflevector <8 x bfloat> %813, <8 x bfloat> %814, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !87
  %816 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %808, <8 x bfloat> %815, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !87
  %817 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !88
  %818 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !88
  %819 = shufflevector <8 x bfloat> %817, <8 x bfloat> %818, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>, !dbg !88
  %820 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !88
  %821 = shufflevector <8 x bfloat> %819, <8 x bfloat> %820, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>, !dbg !88
  %822 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %810, <8 x bfloat> %821, <16 x float> %816, i32 0, i32 0, i32 0), !dbg !88
  %823 = extractelement <16 x float> %822, i64 8, !dbg !88
  %824 = extractelement <16 x float> %822, i64 15, !dbg !88
  %825 = shufflevector <16 x float> %822, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !115
  %826 = shufflevector <16 x float> %822, <16 x float> poison, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, !dbg !115
  %827 = shufflevector <16 x float> %822, <16 x float> poison, <2 x i32> <i32 13, i32 14>, !dbg !115
  br label %828, !dbg !88

828:                                              ; preds = %806, %795
  %829 = phi float [ %823, %806 ], [ 0.000000e+00, %795 ], !dbg !115
  %830 = phi float [ %824, %806 ], [ 0.000000e+00, %795 ], !dbg !115
  %831 = phi <8 x float> [ %825, %806 ], [ zeroinitializer, %795 ], !dbg !115
  %832 = phi <4 x float> [ %826, %806 ], [ zeroinitializer, %795 ], !dbg !115
  %833 = phi <2 x float> [ %827, %806 ], [ zeroinitializer, %795 ], !dbg !115
  %834 = fmul <8 x float> %563, %831, !dbg !86
  %835 = fmul float %166, %829, !dbg !86
  %836 = fmul <4 x float> %567, %832, !dbg !86
  %837 = fmul <2 x float> %570, %833, !dbg !86
  %838 = fmul float %166, %830, !dbg !86
  %839 = extractelement <8 x float> %834, i64 0, !dbg !89
  %840 = extractelement <8 x float> %834, i64 1, !dbg !89
  %841 = tail call float @llvm.maxnum.f32(float %839, float %840), !dbg !90
  %842 = extractelement <8 x float> %834, i64 2, !dbg !89
  %843 = tail call float @llvm.maxnum.f32(float %841, float %842), !dbg !90
  %844 = extractelement <8 x float> %834, i64 3, !dbg !89
  %845 = extractelement <8 x float> %834, i64 4, !dbg !89
  %846 = tail call float @llvm.maxnum.f32(float %844, float %845), !dbg !90
  %847 = extractelement <8 x float> %834, i64 5, !dbg !89
  %848 = tail call float @llvm.maxnum.f32(float %846, float %847), !dbg !90
  %849 = extractelement <8 x float> %834, i64 6, !dbg !89
  %850 = extractelement <8 x float> %834, i64 7, !dbg !89
  %851 = tail call float @llvm.maxnum.f32(float %849, float %850), !dbg !90
  %852 = tail call float @llvm.maxnum.f32(float %851, float %835), !dbg !90
  %853 = extractelement <4 x float> %836, i64 0, !dbg !89
  %854 = extractelement <4 x float> %836, i64 1, !dbg !89
  %855 = tail call float @llvm.maxnum.f32(float %853, float %854), !dbg !90
  %856 = extractelement <4 x float> %836, i64 2, !dbg !89
  %857 = tail call float @llvm.maxnum.f32(float %855, float %856), !dbg !90
  %858 = extractelement <4 x float> %836, i64 3, !dbg !89
  %859 = extractelement <2 x float> %837, i64 0, !dbg !90
  %860 = tail call float @llvm.maxnum.f32(float %858, float %859), !dbg !90
  %861 = extractelement <2 x float> %837, i64 1, !dbg !90
  %862 = tail call float @llvm.maxnum.f32(float %860, float %861), !dbg !90
  %863 = tail call float @llvm.maxnum.f32(float %843, float %848), !dbg !90
  %864 = tail call float @llvm.maxnum.f32(float %863, float %852), !dbg !90
  %865 = tail call float @llvm.maxnum.f32(float %857, float %862), !dbg !90
  %866 = tail call float @llvm.maxnum.f32(float %865, float %838), !dbg !90
  %867 = tail call float @llvm.maxnum.f32(float %864, float %866), !dbg !90
  %868 = bitcast float %867 to i32, !dbg !93
  %869 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %868, i32 %868, i1 false, i1 false), !dbg !93
  %870 = extractvalue { i32, i32 } %869, 0, !dbg !93
  %871 = extractvalue { i32, i32 } %869, 1, !dbg !93
  %872 = bitcast i32 %870 to float, !dbg !93
  %873 = bitcast i32 %871 to float, !dbg !93
  %874 = tail call float @llvm.maxnum.f32(float %872, float %873), !dbg !90
  %875 = tail call float @llvm.maxnum.f32(float %796, float %874), !dbg !95
  %876 = shufflevector <8 x float> %834, <8 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !89
  %877 = insertelement <2 x float> poison, float %875, i64 0, !dbg !89
  %878 = shufflevector <2 x float> %877, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !89
  %879 = fsub <2 x float> %876, %878, !dbg !89
  %880 = shufflevector <8 x float> %834, <8 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !89
  %881 = fsub <2 x float> %880, %878, !dbg !89
  %882 = shufflevector <8 x float> %834, <8 x float> poison, <2 x i32> <i32 4, i32 5>, !dbg !89
  %883 = fsub <2 x float> %882, %878, !dbg !89
  %884 = shufflevector <8 x float> %834, <8 x float> poison, <2 x i32> <i32 6, i32 7>, !dbg !89
  %885 = fsub <2 x float> %884, %878, !dbg !89
  %886 = fsub float %835, %875, !dbg !89
  %887 = shufflevector <4 x float> %836, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !89
  %888 = fsub <2 x float> %887, %878, !dbg !89
  %889 = shufflevector <4 x float> %836, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !89
  %890 = fsub <2 x float> %889, %878, !dbg !89
  %891 = fsub <2 x float> %837, %878, !dbg !89
  %892 = fsub float %838, %875, !dbg !89
  %893 = extractelement <2 x float> %879, i64 0, !dbg !96
  %894 = tail call float @llvm.amdgcn.exp2.f32(float %893), !dbg !96
  %895 = extractelement <2 x float> %879, i64 1, !dbg !96
  %896 = tail call float @llvm.amdgcn.exp2.f32(float %895), !dbg !96
  %897 = extractelement <2 x float> %881, i64 0, !dbg !96
  %898 = tail call float @llvm.amdgcn.exp2.f32(float %897), !dbg !96
  %899 = extractelement <2 x float> %881, i64 1, !dbg !96
  %900 = tail call float @llvm.amdgcn.exp2.f32(float %899), !dbg !96
  %901 = extractelement <2 x float> %883, i64 0, !dbg !96
  %902 = tail call float @llvm.amdgcn.exp2.f32(float %901), !dbg !96
  %903 = extractelement <2 x float> %883, i64 1, !dbg !96
  %904 = tail call float @llvm.amdgcn.exp2.f32(float %903), !dbg !96
  %905 = extractelement <2 x float> %885, i64 0, !dbg !96
  %906 = tail call float @llvm.amdgcn.exp2.f32(float %905), !dbg !96
  %907 = extractelement <2 x float> %885, i64 1, !dbg !96
  %908 = tail call float @llvm.amdgcn.exp2.f32(float %907), !dbg !96
  %909 = tail call float @llvm.amdgcn.exp2.f32(float %886), !dbg !96
  %910 = extractelement <2 x float> %888, i64 0, !dbg !96
  %911 = tail call float @llvm.amdgcn.exp2.f32(float %910), !dbg !96
  %912 = extractelement <2 x float> %888, i64 1, !dbg !96
  %913 = tail call float @llvm.amdgcn.exp2.f32(float %912), !dbg !96
  %914 = extractelement <2 x float> %890, i64 0, !dbg !96
  %915 = tail call float @llvm.amdgcn.exp2.f32(float %914), !dbg !96
  %916 = extractelement <2 x float> %890, i64 1, !dbg !96
  %917 = tail call float @llvm.amdgcn.exp2.f32(float %916), !dbg !96
  %918 = extractelement <2 x float> %891, i64 0, !dbg !96
  %919 = tail call float @llvm.amdgcn.exp2.f32(float %918), !dbg !96
  %920 = extractelement <2 x float> %891, i64 1, !dbg !96
  %921 = tail call float @llvm.amdgcn.exp2.f32(float %920), !dbg !96
  %922 = tail call float @llvm.amdgcn.exp2.f32(float %892), !dbg !96
  %923 = insertelement <2 x float> poison, float %894, i64 0, !dbg !97
  %924 = insertelement <2 x float> %923, float %896, i64 1, !dbg !97
  %925 = insertelement <2 x float> poison, float %898, i64 0, !dbg !97
  %926 = insertelement <2 x float> %925, float %900, i64 1, !dbg !97
  %927 = insertelement <2 x float> poison, float %902, i64 0, !dbg !97
  %928 = insertelement <2 x float> %927, float %904, i64 1, !dbg !97
  %929 = insertelement <2 x float> poison, float %906, i64 0, !dbg !97
  %930 = insertelement <2 x float> %929, float %908, i64 1, !dbg !97
  %931 = insertelement <2 x float> poison, float %909, i64 0, !dbg !97
  %932 = insertelement <2 x float> %931, float %911, i64 1, !dbg !97
  %933 = insertelement <2 x float> poison, float %913, i64 0, !dbg !97
  %934 = insertelement <2 x float> %933, float %915, i64 1, !dbg !97
  %935 = insertelement <2 x float> poison, float %917, i64 0, !dbg !97
  %936 = insertelement <2 x float> %935, float %919, i64 1, !dbg !97
  %937 = insertelement <2 x float> poison, float %921, i64 0, !dbg !97
  %938 = insertelement <2 x float> %937, float %922, i64 1, !dbg !97
  %939 = fadd <2 x float> %924, %926, !dbg !97
  %940 = fadd <2 x float> %928, %930, !dbg !97
  %941 = fadd <2 x float> %932, %934, !dbg !97
  %942 = fadd <2 x float> %936, %938, !dbg !97
  %943 = fadd <2 x float> %939, %940, !dbg !97
  %944 = fadd <2 x float> %941, %942, !dbg !97
  %945 = fadd <2 x float> %943, %944, !dbg !97
  %shift189 = shufflevector <2 x float> %945, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !99
  %foldExtExtBinop190 = fadd <2 x float> %945, %shift189, !dbg !99
  %bc196 = bitcast <2 x float> %foldExtExtBinop190 to <2 x i32>, !dbg !97
  %946 = extractelement <2 x i32> %bc196, i64 0, !dbg !97
  %947 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %946, i32 %946, i1 false, i1 false), !dbg !97
  %948 = fsub float %796, %875, !dbg !100
  %949 = tail call float @llvm.amdgcn.exp2.f32(float %948), !dbg !101
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !102
  tail call void @llvm.amdgcn.s.barrier(), !dbg !102
  %950 = insertelement <1 x float> poison, float %949, i64 0, !dbg !102
  store <1 x float> %950, ptr addrspace(3) %687, align 4, !dbg !102
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !102
  tail call void @llvm.amdgcn.s.barrier(), !dbg !102
  %951 = load <1 x float>, ptr addrspace(3) %693, align 4, !dbg !102
  %952 = load <1 x float>, ptr addrspace(3) %695, align 4, !dbg !102
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !104
  tail call void @llvm.amdgcn.s.barrier(), !dbg !104
  %953 = shufflevector <2 x float> %923, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %954 = fptrunc <1 x float> %953 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %954, ptr addrspace(3) %701, align 2, !dbg !104
  %955 = shufflevector <2 x float> %931, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %956 = fptrunc <1 x float> %955 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %956, ptr addrspace(3) %704, align 2, !dbg !104
  %957 = shufflevector <2 x float> %924, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %958 = fptrunc <1 x float> %957 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %958, ptr addrspace(3) %708, align 2, !dbg !104
  %959 = shufflevector <2 x float> %932, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %960 = fptrunc <1 x float> %959 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %960, ptr addrspace(3) %711, align 2, !dbg !104
  %961 = shufflevector <2 x float> %925, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %962 = fptrunc <1 x float> %961 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %962, ptr addrspace(3) %715, align 2, !dbg !104
  %963 = shufflevector <2 x float> %933, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %964 = fptrunc <1 x float> %963 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %964, ptr addrspace(3) %718, align 2, !dbg !104
  %965 = shufflevector <2 x float> %926, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %966 = fptrunc <1 x float> %965 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %966, ptr addrspace(3) %722, align 2, !dbg !104
  %967 = shufflevector <2 x float> %934, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %968 = fptrunc <1 x float> %967 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %968, ptr addrspace(3) %725, align 2, !dbg !104
  %969 = shufflevector <2 x float> %927, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %970 = fptrunc <1 x float> %969 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %970, ptr addrspace(3) %729, align 2, !dbg !104
  %971 = shufflevector <2 x float> %935, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %972 = fptrunc <1 x float> %971 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %972, ptr addrspace(3) %732, align 2, !dbg !104
  %973 = shufflevector <2 x float> %928, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %974 = fptrunc <1 x float> %973 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %974, ptr addrspace(3) %736, align 2, !dbg !104
  %975 = shufflevector <2 x float> %936, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %976 = fptrunc <1 x float> %975 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %976, ptr addrspace(3) %739, align 2, !dbg !104
  %977 = shufflevector <2 x float> %929, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %978 = fptrunc <1 x float> %977 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %978, ptr addrspace(3) %743, align 2, !dbg !104
  %979 = shufflevector <2 x float> %937, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !104
  %980 = fptrunc <1 x float> %979 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %980, ptr addrspace(3) %746, align 2, !dbg !104
  %981 = shufflevector <2 x float> %930, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %982 = fptrunc <1 x float> %981 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %982, ptr addrspace(3) %750, align 2, !dbg !104
  %983 = shufflevector <2 x float> %938, <2 x float> poison, <1 x i32> <i32 1>, !dbg !104
  %984 = fptrunc <1 x float> %983 to <1 x bfloat>, !dbg !104
  store <1 x bfloat> %984, ptr addrspace(3) %753, align 2, !dbg !104
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !104
  tail call void @llvm.amdgcn.s.barrier(), !dbg !104
  %985 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %764), !dbg !104
  %986 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %766), !dbg !104
  %987 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %768), !dbg !104
  %988 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %770), !dbg !104
  br i1 %521, label %989, label %1012, !dbg !105

989:                                              ; preds = %828
  %990 = fmul float %797, %949, !dbg !103
  %991 = extractvalue { i32, i32 } %947, 0, !dbg !97
  %992 = bitcast i32 %991 to float, !dbg !97
  %993 = extractvalue { i32, i32 } %947, 1, !dbg !97
  %994 = bitcast i32 %993 to float, !dbg !97
  %995 = fadd float %992, %994, !dbg !99
  %996 = fadd float %995, %990, !dbg !103
  %997 = shufflevector <4 x bfloat> %985, <4 x bfloat> %986, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %998 = shufflevector <4 x bfloat> %987, <4 x bfloat> %988, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %999 = shufflevector <4 x bfloat> %803, <4 x bfloat> %805, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !105
  %1000 = shufflevector <2 x float> %801, <2 x float> %800, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !105
  %1001 = shufflevector <1 x float> %951, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !105
  %1002 = fmul <4 x float> %1000, %1001, !dbg !105
  %1003 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %999, <8 x bfloat> %997, <4 x float> %1002, i32 0, i32 0, i32 0), !dbg !105
  %1004 = shufflevector <2 x float> %799, <2 x float> %798, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !105
  %1005 = shufflevector <1 x float> %952, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !105
  %1006 = fmul <4 x float> %1004, %1005, !dbg !105
  %1007 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %999, <8 x bfloat> %998, <4 x float> %1006, i32 0, i32 0, i32 0), !dbg !105
  %1008 = shufflevector <4 x float> %1007, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !70
  %1009 = shufflevector <4 x float> %1007, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !70
  %1010 = shufflevector <4 x float> %1003, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !70
  %1011 = shufflevector <4 x float> %1003, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !70
  br label %1012, !dbg !105

1012:                                             ; preds = %828, %989, %37
  %1013 = phi float [ 0xFFF0000000000000, %37 ], [ %875, %989 ], [ %796, %828 ], !dbg !70
  %1014 = phi float [ 1.000000e+00, %37 ], [ %996, %989 ], [ %797, %828 ], !dbg !70
  %1015 = phi i32 [ 0, %37 ], [ %153, %989 ], [ %153, %828 ], !dbg !70
  %1016 = phi <2 x float> [ zeroinitializer, %37 ], [ %1008, %989 ], [ %798, %828 ], !dbg !70
  %1017 = phi <2 x float> [ zeroinitializer, %37 ], [ %1009, %989 ], [ %799, %828 ], !dbg !70
  %1018 = phi <2 x float> [ zeroinitializer, %37 ], [ %1010, %989 ], [ %800, %828 ], !dbg !70
  %1019 = phi <2 x float> [ zeroinitializer, %37 ], [ %1011, %989 ], [ %801, %828 ], !dbg !70
  %1020 = icmp sgt i32 %142, 0, !dbg !116
  br i1 %1020, label %1021, label %.loopexit, !dbg !117

1021:                                             ; preds = %1012
  %1022 = fmul float %28, 0x3FF7154760000000, !dbg !118
  %1023 = shl nsw i64 %73, 5, !dbg !120
  %1024 = shl nsw i64 %76, 5, !dbg !121
  %1025 = icmp slt i32 %1015, %144, !dbg !122
  br i1 %1025, label %.lr.ph121, label %.loopexit, !dbg !122

.lr.ph121:                                        ; preds = %1021
  %1026 = shl i32 %143, 5, !dbg !123
  %1027 = zext i32 %1026 to i64, !dbg !123
  %1028 = mul nsw i64 %1027, %76, !dbg !124
  %1029 = add i64 %105, %1028, !dbg !125
  %1030 = add i64 %107, %1029, !dbg !125
  %1031 = mul nsw i64 %1027, %73, !dbg !123
  %1032 = add i64 %96, %1031, !dbg !125
  %1033 = add i64 %103, %1032, !dbg !125
  %1034 = add i64 %101, %1032, !dbg !125
  %1035 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %1, i16 0, i64 2147483646, i32 159744)
  %1036 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 2147483646, i32 159744)
  %1037 = shl nuw nsw i32 %58, 2
  %1038 = and i32 %1037, 764
  %1039 = and i32 %57, 64
  %1040 = icmp eq i32 %1039, 0
  %1041 = select i1 %1040, i32 0, i32 272
  %1042 = xor i32 %1038, %1041
  %1043 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1042
  %1044 = shl nuw nsw i32 %38, 5
  %1045 = and i32 %1044, 736
  %1046 = and i32 %38, 8
  %1047 = icmp eq i32 %1046, 0
  %1048 = select i1 %1047, i32 0, i32 272
  %1049 = lshr exact i32 %97, 1
  %1050 = xor i32 %1048, %1049
  %1051 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1045
  %1052 = getelementptr inbounds nuw i8, ptr addrspace(3) %1051, i32 %1050
  %1053 = shufflevector <2 x bfloat> %122, <2 x bfloat> %123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1054 = shufflevector <2 x bfloat> %124, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1055 = shufflevector <8 x bfloat> %1053, <8 x bfloat> %1054, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1056 = shufflevector <2 x bfloat> %125, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1057 = shufflevector <8 x bfloat> %1055, <8 x bfloat> %1056, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1058 = shufflevector <2 x bfloat> %133, <2 x bfloat> %134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1059 = shufflevector <2 x bfloat> %135, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1060 = shufflevector <8 x bfloat> %1058, <8 x bfloat> %1059, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1061 = shufflevector <2 x bfloat> %136, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1062 = shufflevector <8 x bfloat> %1060, <8 x bfloat> %1061, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1063 = and i32 %1037, 380
  %1064 = and i32 %57, 128
  %1065 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1063
  %1066 = getelementptr inbounds nuw i8, ptr addrspace(3) %1065, i32 %1064
  %1067 = shl nuw nsw i32 %59, 2
  %1068 = shl nuw nsw i32 %1064, 1
  %1069 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1068
  %1070 = getelementptr inbounds nuw i8, ptr addrspace(3) %1069, i32 %1067
  %1071 = getelementptr inbounds nuw i8, ptr addrspace(3) %1070, i32 %1039
  %1072 = getelementptr inbounds nuw i8, ptr addrspace(3) %1071, i32 128
  %1073 = and i32 %38, 31
  %1074 = shl nuw nsw i32 %1073, 1
  %1075 = icmp eq i32 %97, 0
  %1076 = select i1 %1075, i32 0, i32 1056
  %1077 = or disjoint i32 %57, %1074
  %1078 = xor i32 %1077, %1076
  %1079 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1078
  %1080 = getelementptr inbounds nuw i8, ptr addrspace(3) %1079, i32 4096
  %1081 = xor i32 %1078, 264
  %1082 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1081
  %1083 = getelementptr inbounds nuw i8, ptr addrspace(3) %1082, i32 4096
  %1084 = xor i32 %1078, 528
  %1085 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1084
  %1086 = getelementptr inbounds nuw i8, ptr addrspace(3) %1085, i32 4096
  %1087 = xor i32 %1078, 792
  %1088 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1087
  %1089 = getelementptr inbounds nuw i8, ptr addrspace(3) %1088, i32 4096
  %1090 = xor i32 %1078, 2112
  %1091 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1090
  %1092 = getelementptr inbounds nuw i8, ptr addrspace(3) %1091, i32 4096
  %1093 = xor i32 %1078, 2376
  %1094 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1093
  %1095 = getelementptr inbounds nuw i8, ptr addrspace(3) %1094, i32 4096
  %1096 = xor i32 %1078, 2640
  %1097 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1096
  %1098 = getelementptr inbounds nuw i8, ptr addrspace(3) %1097, i32 4096
  %1099 = xor i32 %1078, 2904
  %1100 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1099
  %1101 = getelementptr inbounds nuw i8, ptr addrspace(3) %1100, i32 4096
  %1102 = and i32 %38, 60
  %1103 = shl nuw nsw i32 %1102, 6
  %1104 = and i32 %67, 24
  %1105 = shl nuw nsw i32 %1102, 1
  %1106 = shl nuw nsw i32 %41, 5
  %1107 = or disjoint i32 %1103, %1104
  %1108 = xor i32 %1107, %1105
  %1109 = xor i32 %1108, %1106
  %1110 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1109
  %1111 = getelementptr inbounds nuw i8, ptr addrspace(3) %1110, i32 4096
  %1112 = getelementptr inbounds nuw i8, ptr addrspace(3) %1110, i32 128
  %1113 = getelementptr inbounds nuw i8, ptr addrspace(3) %1110, i32 4224
  %1114 = select i1 %1040, i32 0, i32 264
  %1115 = xor i32 %1038, %1114
  %1116 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1115
  %1117 = shl nuw nsw i32 %1073, 3
  %1118 = select i1 %1075, i32 0, i32 264
  %1119 = xor i32 %1118, %1117
  %1120 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1119
  %1121 = getelementptr inbounds nuw i8, ptr addrspace(3) %1120, i32 512
  br label %1122, !dbg !122

1122:                                             ; preds = %.lr.ph121, %1122
  %.pn73.in118 = phi i64 [ %1030, %.lr.ph121 ], [ %1372, %1122 ]
  %.pn69.in116 = phi i64 [ %1033, %.lr.ph121 ], [ %1370, %1122 ]
  %.pn65.in114 = phi i64 [ %1034, %.lr.ph121 ], [ %1368, %1122 ]
  %1123 = phi float [ %1013, %.lr.ph121 ], [ %1236, %1122 ]
  %1124 = phi float [ %1014, %.lr.ph121 ], [ %1317, %1122 ]
  %1125 = phi i32 [ %1015, %.lr.ph121 ], [ %1141, %1122 ]
  %1126 = phi <2 x float> [ %1017, %.lr.ph121 ], [ %1375, %1122 ]
  %1127 = phi <2 x float> [ %1016, %.lr.ph121 ], [ %1374, %1122 ]
  %1128 = phi <2 x float> [ %1019, %.lr.ph121 ], [ %1377, %1122 ]
  %1129 = phi <2 x float> [ %1018, %.lr.ph121 ], [ %1376, %1122 ]
  %.pn73 = trunc i64 %.pn73.in118 to i32, !dbg !125
  %.pn69 = trunc i64 %.pn69.in116 to i32, !dbg !125
  %.pn65 = trunc i64 %.pn65.in114 to i32, !dbg !125
  %1130 = or disjoint i32 %1125, %64, !dbg !126
  %1131 = icmp slt i32 %1130, %33, !dbg !127
  %1132 = shl i32 %.pn65, 1, !dbg !129
  %1133 = select i1 %1131, i32 %1132, i32 -2147483648, !dbg !129
  %1134 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1035, i32 %1133, i32 0, i32 0), !dbg !129
  %1135 = shl i32 %.pn69, 1, !dbg !130
  %1136 = select i1 %1131, i32 %1135, i32 -2147483648, !dbg !130
  %1137 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1035, i32 %1136, i32 0, i32 0), !dbg !130
  %1138 = shl i32 %.pn73, 1, !dbg !132
  %1139 = select i1 %1131, i32 %1138, i32 -2147483648, !dbg !132
  %1140 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %1036, i32 %1139, i32 0, i32 0), !dbg !132
  %1141 = add nsw i32 %1125, 32, !dbg !134
  %1142 = icmp eq i32 %1141, %144, !dbg !134
  %1143 = and i1 %140, %1142, !dbg !135
  %1144 = or disjoint i32 %1125, %98, !dbg !136
  %1145 = or disjoint i32 %1144, 16, !dbg !136
  %1146 = icmp sge i32 %1144, %33, !dbg !137
  %1147 = icmp sge i32 %1145, %33, !dbg !137
  %.not74 = select i1 %1143, i1 %1146, i1 false, !dbg !138
  %.not75 = select i1 %1143, i1 %1147, i1 false, !dbg !130
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !130
  tail call void @llvm.amdgcn.s.barrier(), !dbg !130
  store i32 %1137, ptr addrspace(3) %1043, align 4, !dbg !130
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !130
  tail call void @llvm.amdgcn.s.barrier(), !dbg !130
  %1148 = load <8 x bfloat>, ptr addrspace(3) %1052, align 16, !dbg !130
  %1149 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %1148, <8 x bfloat> %1057, <16 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !139
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !129
  tail call void @llvm.amdgcn.s.barrier(), !dbg !129
  store i32 %1134, ptr addrspace(3) %1043, align 4, !dbg !129
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !129
  tail call void @llvm.amdgcn.s.barrier(), !dbg !129
  %1150 = load <8 x bfloat>, ptr addrspace(3) %1052, align 16, !dbg !129
  %1151 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %1150, <8 x bfloat> %1062, <16 x float> %1149, i32 0, i32 0, i32 0), !dbg !140
  %1152 = extractelement <16 x float> %1151, i64 0, !dbg !140
  %1153 = extractelement <16 x float> %1151, i64 1, !dbg !140
  %1154 = extractelement <16 x float> %1151, i64 2, !dbg !140
  %1155 = extractelement <16 x float> %1151, i64 3, !dbg !140
  %1156 = extractelement <16 x float> %1151, i64 4, !dbg !140
  %1157 = extractelement <16 x float> %1151, i64 5, !dbg !140
  %1158 = extractelement <16 x float> %1151, i64 6, !dbg !140
  %1159 = extractelement <16 x float> %1151, i64 7, !dbg !140
  %1160 = extractelement <16 x float> %1151, i64 8, !dbg !140
  %1161 = extractelement <16 x float> %1151, i64 9, !dbg !140
  %1162 = extractelement <16 x float> %1151, i64 10, !dbg !140
  %1163 = extractelement <16 x float> %1151, i64 11, !dbg !140
  %1164 = extractelement <16 x float> %1151, i64 12, !dbg !140
  %1165 = extractelement <16 x float> %1151, i64 13, !dbg !140
  %1166 = extractelement <16 x float> %1151, i64 14, !dbg !140
  %1167 = extractelement <16 x float> %1151, i64 15, !dbg !140
  %1168 = fmul float %1022, %1153, !dbg !141
  %1169 = fmul float %1022, %1152, !dbg !141
  %1170 = select i1 %.not74, float 0xFFF0000000000000, float %1168, !dbg !142
  %1171 = select i1 %.not74, float 0xFFF0000000000000, float %1169, !dbg !142
  %1172 = tail call float @llvm.maxnum.f32(float %1171, float %1170), !dbg !143
  %1173 = insertelement <2 x float> poison, float %1171, i64 0, !dbg !146
  %1174 = insertelement <2 x float> %1173, float %1170, i64 1, !dbg !146
  %1175 = fmul float %1022, %1155, !dbg !141
  %1176 = fmul float %1022, %1154, !dbg !141
  %1177 = select i1 %.not74, float 0xFFF0000000000000, float %1175, !dbg !142
  %1178 = select i1 %.not74, float 0xFFF0000000000000, float %1176, !dbg !142
  %1179 = tail call float @llvm.maxnum.f32(float %1172, float %1178), !dbg !143
  %1180 = insertelement <2 x float> poison, float %1178, i64 0, !dbg !146
  %1181 = insertelement <2 x float> %1180, float %1177, i64 1, !dbg !146
  %1182 = fmul float %1022, %1157, !dbg !141
  %1183 = fmul float %1022, %1156, !dbg !141
  %1184 = select i1 %.not74, float 0xFFF0000000000000, float %1182, !dbg !142
  %1185 = select i1 %.not74, float 0xFFF0000000000000, float %1183, !dbg !142
  %1186 = tail call float @llvm.maxnum.f32(float %1177, float %1185), !dbg !143
  %1187 = tail call float @llvm.maxnum.f32(float %1186, float %1184), !dbg !143
  %1188 = tail call float @llvm.maxnum.f32(float %1179, float %1187), !dbg !143
  %1189 = insertelement <2 x float> poison, float %1185, i64 0, !dbg !146
  %1190 = insertelement <2 x float> %1189, float %1184, i64 1, !dbg !146
  %1191 = fmul float %1022, %1159, !dbg !141
  %1192 = fmul float %1022, %1158, !dbg !141
  %1193 = select i1 %.not74, float 0xFFF0000000000000, float %1191, !dbg !142
  %1194 = select i1 %.not74, float 0xFFF0000000000000, float %1192, !dbg !142
  %1195 = tail call float @llvm.maxnum.f32(float %1194, float %1193), !dbg !143
  %1196 = insertelement <2 x float> poison, float %1194, i64 0, !dbg !146
  %1197 = insertelement <2 x float> %1196, float %1193, i64 1, !dbg !146
  %1198 = fmul float %1022, %1161, !dbg !141
  %1199 = fmul float %1022, %1160, !dbg !141
  %1200 = select i1 %.not75, float 0xFFF0000000000000, float %1198, !dbg !142
  %1201 = select i1 %.not75, float 0xFFF0000000000000, float %1199, !dbg !142
  %1202 = tail call float @llvm.maxnum.f32(float %1195, float %1201), !dbg !143
  %1203 = tail call float @llvm.maxnum.f32(float %1188, float %1202), !dbg !143
  %1204 = insertelement <2 x float> poison, float %1201, i64 0, !dbg !146
  %1205 = insertelement <2 x float> %1204, float %1200, i64 1, !dbg !146
  %1206 = fmul float %1022, %1163, !dbg !141
  %1207 = fmul float %1022, %1162, !dbg !141
  %1208 = select i1 %.not75, float 0xFFF0000000000000, float %1206, !dbg !142
  %1209 = select i1 %.not75, float 0xFFF0000000000000, float %1207, !dbg !142
  %1210 = tail call float @llvm.maxnum.f32(float %1200, float %1209), !dbg !143
  %1211 = tail call float @llvm.maxnum.f32(float %1210, float %1208), !dbg !143
  %1212 = insertelement <2 x float> poison, float %1209, i64 0, !dbg !146
  %1213 = insertelement <2 x float> %1212, float %1208, i64 1, !dbg !146
  %1214 = fmul float %1022, %1165, !dbg !141
  %1215 = fmul float %1022, %1164, !dbg !141
  %1216 = select i1 %.not75, float 0xFFF0000000000000, float %1214, !dbg !142
  %1217 = select i1 %.not75, float 0xFFF0000000000000, float %1215, !dbg !142
  %1218 = tail call float @llvm.maxnum.f32(float %1217, float %1216), !dbg !143
  %1219 = insertelement <2 x float> poison, float %1217, i64 0, !dbg !146
  %1220 = insertelement <2 x float> %1219, float %1216, i64 1, !dbg !146
  %1221 = fmul float %1022, %1167, !dbg !141
  %1222 = fmul float %1022, %1166, !dbg !141
  %1223 = select i1 %.not75, float 0xFFF0000000000000, float %1221, !dbg !142
  %1224 = select i1 %.not75, float 0xFFF0000000000000, float %1222, !dbg !142
  %1225 = tail call float @llvm.maxnum.f32(float %1218, float %1224), !dbg !143
  %1226 = tail call float @llvm.maxnum.f32(float %1211, float %1225), !dbg !143
  %1227 = tail call float @llvm.maxnum.f32(float %1226, float %1223), !dbg !143
  %1228 = tail call float @llvm.maxnum.f32(float %1203, float %1227), !dbg !143
  %1229 = bitcast float %1228 to i32, !dbg !144
  %1230 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %1229, i32 %1229, i1 false, i1 false), !dbg !144
  %1231 = extractvalue { i32, i32 } %1230, 0, !dbg !144
  %1232 = extractvalue { i32, i32 } %1230, 1, !dbg !144
  %1233 = bitcast i32 %1231 to float, !dbg !144
  %1234 = bitcast i32 %1232 to float, !dbg !144
  %1235 = tail call float @llvm.maxnum.f32(float %1233, float %1234), !dbg !143
  %1236 = tail call float @llvm.maxnum.f32(float %1123, float %1235), !dbg !147
  %1237 = insertelement <2 x float> poison, float %1236, i64 0, !dbg !146
  %1238 = shufflevector <2 x float> %1237, <2 x float> poison, <2 x i32> zeroinitializer, !dbg !146
  %1239 = fsub <2 x float> %1174, %1238, !dbg !146
  %1240 = fsub <2 x float> %1181, %1238, !dbg !146
  %1241 = fsub <2 x float> %1190, %1238, !dbg !146
  %1242 = fsub <2 x float> %1197, %1238, !dbg !146
  %1243 = fsub <2 x float> %1205, %1238, !dbg !146
  %1244 = fsub <2 x float> %1213, %1238, !dbg !146
  %1245 = fsub <2 x float> %1220, %1238, !dbg !146
  %1246 = insertelement <2 x float> poison, float %1224, i64 0, !dbg !146
  %1247 = insertelement <2 x float> %1246, float %1223, i64 1, !dbg !146
  %1248 = fsub <2 x float> %1247, %1238, !dbg !146
  %1249 = extractelement <2 x float> %1239, i64 0, !dbg !148
  %1250 = tail call float @llvm.amdgcn.exp2.f32(float %1249), !dbg !148
  %1251 = extractelement <2 x float> %1239, i64 1, !dbg !148
  %1252 = tail call float @llvm.amdgcn.exp2.f32(float %1251), !dbg !148
  %1253 = extractelement <2 x float> %1240, i64 0, !dbg !148
  %1254 = tail call float @llvm.amdgcn.exp2.f32(float %1253), !dbg !148
  %1255 = extractelement <2 x float> %1240, i64 1, !dbg !148
  %1256 = tail call float @llvm.amdgcn.exp2.f32(float %1255), !dbg !148
  %1257 = extractelement <2 x float> %1241, i64 0, !dbg !148
  %1258 = tail call float @llvm.amdgcn.exp2.f32(float %1257), !dbg !148
  %1259 = extractelement <2 x float> %1241, i64 1, !dbg !148
  %1260 = tail call float @llvm.amdgcn.exp2.f32(float %1259), !dbg !148
  %1261 = extractelement <2 x float> %1242, i64 0, !dbg !148
  %1262 = tail call float @llvm.amdgcn.exp2.f32(float %1261), !dbg !148
  %1263 = extractelement <2 x float> %1242, i64 1, !dbg !148
  %1264 = tail call float @llvm.amdgcn.exp2.f32(float %1263), !dbg !148
  %1265 = extractelement <2 x float> %1243, i64 0, !dbg !148
  %1266 = tail call float @llvm.amdgcn.exp2.f32(float %1265), !dbg !148
  %1267 = extractelement <2 x float> %1243, i64 1, !dbg !148
  %1268 = tail call float @llvm.amdgcn.exp2.f32(float %1267), !dbg !148
  %1269 = extractelement <2 x float> %1244, i64 0, !dbg !148
  %1270 = tail call float @llvm.amdgcn.exp2.f32(float %1269), !dbg !148
  %1271 = extractelement <2 x float> %1244, i64 1, !dbg !148
  %1272 = tail call float @llvm.amdgcn.exp2.f32(float %1271), !dbg !148
  %1273 = extractelement <2 x float> %1245, i64 0, !dbg !148
  %1274 = tail call float @llvm.amdgcn.exp2.f32(float %1273), !dbg !148
  %1275 = extractelement <2 x float> %1245, i64 1, !dbg !148
  %1276 = tail call float @llvm.amdgcn.exp2.f32(float %1275), !dbg !148
  %1277 = extractelement <2 x float> %1248, i64 0, !dbg !148
  %1278 = tail call float @llvm.amdgcn.exp2.f32(float %1277), !dbg !148
  %1279 = extractelement <2 x float> %1248, i64 1, !dbg !148
  %1280 = tail call float @llvm.amdgcn.exp2.f32(float %1279), !dbg !148
  %1281 = insertelement <2 x float> poison, float %1250, i64 0, !dbg !149
  %1282 = insertelement <2 x float> %1281, float %1252, i64 1, !dbg !149
  %1283 = insertelement <2 x float> poison, float %1254, i64 0, !dbg !149
  %1284 = insertelement <2 x float> %1283, float %1256, i64 1, !dbg !149
  %1285 = insertelement <2 x float> poison, float %1258, i64 0, !dbg !149
  %1286 = insertelement <2 x float> %1285, float %1260, i64 1, !dbg !149
  %1287 = insertelement <2 x float> poison, float %1262, i64 0, !dbg !149
  %1288 = insertelement <2 x float> %1287, float %1264, i64 1, !dbg !149
  %1289 = insertelement <2 x float> poison, float %1266, i64 0, !dbg !149
  %1290 = insertelement <2 x float> %1289, float %1268, i64 1, !dbg !149
  %1291 = insertelement <2 x float> poison, float %1270, i64 0, !dbg !149
  %1292 = insertelement <2 x float> %1291, float %1272, i64 1, !dbg !149
  %1293 = insertelement <2 x float> poison, float %1274, i64 0, !dbg !149
  %1294 = insertelement <2 x float> %1293, float %1276, i64 1, !dbg !149
  %1295 = insertelement <2 x float> poison, float %1278, i64 0, !dbg !149
  %1296 = insertelement <2 x float> %1295, float %1280, i64 1, !dbg !149
  %1297 = fadd <2 x float> %1282, %1284, !dbg !149
  %1298 = fadd <2 x float> %1286, %1288, !dbg !149
  %1299 = fadd <2 x float> %1290, %1292, !dbg !149
  %1300 = fadd <2 x float> %1294, %1296, !dbg !149
  %1301 = fadd <2 x float> %1297, %1298, !dbg !149
  %1302 = fadd <2 x float> %1299, %1300, !dbg !149
  %1303 = fadd <2 x float> %1301, %1302, !dbg !149
  %shift192 = shufflevector <2 x float> %1303, <2 x float> poison, <2 x i32> <i32 1, i32 poison>, !dbg !151
  %foldExtExtBinop193 = fadd <2 x float> %1303, %shift192, !dbg !151
  %bc197 = bitcast <2 x float> %foldExtExtBinop193 to <2 x i32>, !dbg !149
  %1304 = extractelement <2 x i32> %bc197, i64 0, !dbg !149
  %1305 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %1304, i32 %1304, i1 false, i1 false), !dbg !149
  %1306 = extractvalue { i32, i32 } %1305, 0, !dbg !149
  %1307 = extractvalue { i32, i32 } %1305, 1, !dbg !149
  %1308 = bitcast i32 %1306 to float, !dbg !149
  %1309 = bitcast i32 %1307 to float, !dbg !149
  %1310 = fadd float %1308, %1309, !dbg !151
  %1311 = fsub float %1123, %1236, !dbg !152
  %1312 = tail call float @llvm.amdgcn.exp2.f32(float %1311), !dbg !153
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !154
  tail call void @llvm.amdgcn.s.barrier(), !dbg !154
  %1313 = insertelement <1 x float> poison, float %1312, i64 0, !dbg !154
  store <1 x float> %1313, ptr addrspace(3) %1066, align 4, !dbg !154
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !154
  tail call void @llvm.amdgcn.s.barrier(), !dbg !154
  %1314 = load <1 x float>, ptr addrspace(3) %1071, align 4, !dbg !154
  %1315 = load <1 x float>, ptr addrspace(3) %1072, align 4, !dbg !154
  %1316 = fmul float %1124, %1312, !dbg !155
  %1317 = fadd float %1310, %1316, !dbg !155
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !156
  tail call void @llvm.amdgcn.s.barrier(), !dbg !156
  %1318 = shufflevector <2 x float> %1281, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !156
  %1319 = fptrunc <1 x float> %1318 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1319, ptr addrspace(3) %1079, align 2, !dbg !156
  %1320 = shufflevector <2 x float> %1289, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !156
  %1321 = fptrunc <1 x float> %1320 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1321, ptr addrspace(3) %1080, align 2, !dbg !156
  %1322 = shufflevector <2 x float> %1282, <2 x float> poison, <1 x i32> <i32 1>, !dbg !156
  %1323 = fptrunc <1 x float> %1322 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1323, ptr addrspace(3) %1082, align 2, !dbg !156
  %1324 = shufflevector <2 x float> %1290, <2 x float> poison, <1 x i32> <i32 1>, !dbg !156
  %1325 = fptrunc <1 x float> %1324 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1325, ptr addrspace(3) %1083, align 2, !dbg !156
  %1326 = shufflevector <2 x float> %1283, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !156
  %1327 = fptrunc <1 x float> %1326 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1327, ptr addrspace(3) %1085, align 2, !dbg !156
  %1328 = shufflevector <2 x float> %1291, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !156
  %1329 = fptrunc <1 x float> %1328 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1329, ptr addrspace(3) %1086, align 2, !dbg !156
  %1330 = shufflevector <2 x float> %1284, <2 x float> poison, <1 x i32> <i32 1>, !dbg !156
  %1331 = fptrunc <1 x float> %1330 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1331, ptr addrspace(3) %1088, align 2, !dbg !156
  %1332 = shufflevector <2 x float> %1292, <2 x float> poison, <1 x i32> <i32 1>, !dbg !156
  %1333 = fptrunc <1 x float> %1332 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1333, ptr addrspace(3) %1089, align 2, !dbg !156
  %1334 = shufflevector <2 x float> %1285, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !156
  %1335 = fptrunc <1 x float> %1334 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1335, ptr addrspace(3) %1091, align 2, !dbg !156
  %1336 = shufflevector <2 x float> %1293, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !156
  %1337 = fptrunc <1 x float> %1336 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1337, ptr addrspace(3) %1092, align 2, !dbg !156
  %1338 = shufflevector <2 x float> %1286, <2 x float> poison, <1 x i32> <i32 1>, !dbg !156
  %1339 = fptrunc <1 x float> %1338 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1339, ptr addrspace(3) %1094, align 2, !dbg !156
  %1340 = shufflevector <2 x float> %1294, <2 x float> poison, <1 x i32> <i32 1>, !dbg !156
  %1341 = fptrunc <1 x float> %1340 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1341, ptr addrspace(3) %1095, align 2, !dbg !156
  %1342 = shufflevector <2 x float> %1287, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !156
  %1343 = fptrunc <1 x float> %1342 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1343, ptr addrspace(3) %1097, align 2, !dbg !156
  %1344 = shufflevector <2 x float> %1295, <2 x float> poison, <1 x i32> zeroinitializer, !dbg !156
  %1345 = fptrunc <1 x float> %1344 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1345, ptr addrspace(3) %1098, align 2, !dbg !156
  %1346 = shufflevector <2 x float> %1288, <2 x float> poison, <1 x i32> <i32 1>, !dbg !156
  %1347 = fptrunc <1 x float> %1346 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1347, ptr addrspace(3) %1100, align 2, !dbg !156
  %1348 = shufflevector <2 x float> %1296, <2 x float> poison, <1 x i32> <i32 1>, !dbg !156
  %1349 = fptrunc <1 x float> %1348 to <1 x bfloat>, !dbg !156
  store <1 x bfloat> %1349, ptr addrspace(3) %1101, align 2, !dbg !156
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !156
  tail call void @llvm.amdgcn.s.barrier(), !dbg !156
  %1350 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %1110), !dbg !156
  %1351 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1111), !dbg !156
  %1352 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1112), !dbg !156
  %1353 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1113), !dbg !156
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !132
  tail call void @llvm.amdgcn.s.barrier(), !dbg !132
  store i32 %1140, ptr addrspace(3) %1116, align 4, !dbg !132
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !132
  tail call void @llvm.amdgcn.s.barrier(), !dbg !132
  %1354 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %1120), !dbg !132
  %1355 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) nonnull %1121), !dbg !132
  %1356 = shufflevector <4 x bfloat> %1350, <4 x bfloat> %1351, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !157
  %1357 = shufflevector <4 x bfloat> %1352, <4 x bfloat> %1353, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !157
  %1358 = shufflevector <4 x bfloat> %1354, <4 x bfloat> %1355, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !157
  %1359 = shufflevector <2 x float> %1128, <2 x float> %1129, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !157
  %1360 = shufflevector <1 x float> %1314, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !157
  %1361 = fmul <4 x float> %1359, %1360, !dbg !157
  %1362 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1358, <8 x bfloat> %1356, <4 x float> %1361, i32 0, i32 0, i32 0), !dbg !157
  %1363 = shufflevector <2 x float> %1126, <2 x float> %1127, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !157
  %1364 = shufflevector <1 x float> %1315, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !157
  %1365 = fmul <4 x float> %1363, %1364, !dbg !157
  %1366 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %1358, <8 x bfloat> %1357, <4 x float> %1365, i32 0, i32 0, i32 0), !dbg !157
  %sext = shl i64 %.pn65.in114, 32, !dbg !158
  %1367 = ashr exact i64 %sext, 32, !dbg !158
  %1368 = add nsw i64 %1367, %1023, !dbg !158
  %sext77 = shl i64 %.pn69.in116, 32, !dbg !159
  %1369 = ashr exact i64 %sext77, 32, !dbg !159
  %1370 = add nsw i64 %1369, %1023, !dbg !159
  %sext79 = shl i64 %.pn73.in118, 32, !dbg !160
  %1371 = ashr exact i64 %sext79, 32, !dbg !160
  %1372 = add nsw i64 %1371, %1024, !dbg !160
  %1373 = icmp slt i32 %1141, %144, !dbg !122
  %1374 = shufflevector <4 x float> %1366, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !117
  %1375 = shufflevector <4 x float> %1366, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !117
  %1376 = shufflevector <4 x float> %1362, <4 x float> poison, <2 x i32> <i32 2, i32 3>, !dbg !117
  %1377 = shufflevector <4 x float> %1362, <4 x float> poison, <2 x i32> <i32 0, i32 1>, !dbg !117
  br i1 %1373, label %1122, label %.loopexit, !dbg !122

.loopexit:                                        ; preds = %1122, %1021, %1012
  %1378 = phi float [ %1013, %1012 ], [ %1013, %1021 ], [ %1236, %1122 ], !dbg !117
  %1379 = phi float [ %1014, %1012 ], [ %1014, %1021 ], [ %1317, %1122 ], !dbg !117
  %1380 = phi <2 x float> [ %1016, %1012 ], [ %1016, %1021 ], [ %1374, %1122 ], !dbg !117
  %1381 = phi <2 x float> [ %1017, %1012 ], [ %1017, %1021 ], [ %1375, %1122 ], !dbg !117
  %1382 = phi <2 x float> [ %1018, %1012 ], [ %1018, %1021 ], [ %1376, %1122 ], !dbg !117
  %1383 = phi <2 x float> [ %1019, %1012 ], [ %1019, %1021 ], [ %1377, %1122 ], !dbg !117
  %1384 = fdiv float 1.000000e+00, %1379, !dbg !161
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !162
  tail call void @llvm.amdgcn.s.barrier(), !dbg !162
  %1385 = shl nuw nsw i32 %58, 2, !dbg !162
  %1386 = and i32 %1385, 380, !dbg !162
  %1387 = and i32 %57, 128, !dbg !162
  %1388 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1386, !dbg !162
  %1389 = getelementptr inbounds nuw i8, ptr addrspace(3) %1388, i32 %1387, !dbg !162
  %1390 = insertelement <1 x float> poison, float %1384, i64 0, !dbg !162
  store <1 x float> %1390, ptr addrspace(3) %1389, align 4, !dbg !162
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !162
  tail call void @llvm.amdgcn.s.barrier(), !dbg !162
  %1391 = shl nuw nsw i32 %59, 2, !dbg !162
  %1392 = and i32 %57, 64, !dbg !162
  %1393 = shl nuw nsw i32 %1387, 1, !dbg !162
  %1394 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1392, !dbg !162
  %1395 = getelementptr inbounds nuw i8, ptr addrspace(3) %1394, i32 %1393, !dbg !162
  %1396 = getelementptr inbounds nuw i8, ptr addrspace(3) %1395, i32 %1391, !dbg !162
  %1397 = load <1 x float>, ptr addrspace(3) %1396, align 4, !dbg !162
  %1398 = getelementptr inbounds nuw i8, ptr addrspace(3) %1396, i32 128, !dbg !162
  %1399 = load <1 x float>, ptr addrspace(3) %1398, align 4, !dbg !162
  %reass.sub = sub i32 %55, %32, !dbg !163
  %1400 = add i32 %reass.sub, 128, !dbg !163
  %1401 = tail call noundef float @llvm.log2.f32(float %1379), !dbg !164
  %1402 = fadd float %1378, %1401, !dbg !165
  %1403 = fmul float %1402, 0x3FE62E4300000000, !dbg !166
  %1404 = mul nuw i64 %87, %77, !dbg !167
  %1405 = mul nuw i64 %78, %82, !dbg !168
  %1406 = add i64 %1404, %1405, !dbg !167
  %1407 = icmp slt i32 %1400, 1, !dbg !169
  br i1 %1407, label %1428, label %1408, !dbg !170

1408:                                             ; preds = %.loopexit
  %1409 = sub nsw i32 0, %reass.sub, !dbg !171
  %1410 = icmp slt i32 %61, %1409, !dbg !172
  %1411 = trunc i64 %1406 to i32, !dbg !173
  %1412 = add i32 %63, %1411, !dbg !173
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !174
  tail call void @llvm.amdgcn.s.barrier(), !dbg !174
  %1413 = shl nuw nsw i32 %38, 2, !dbg !174
  %1414 = and i32 %1413, 124, !dbg !174
  %1415 = shl nuw nsw i32 %41, 7, !dbg !174
  %1416 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1414, !dbg !174
  %1417 = getelementptr inbounds nuw i8, ptr addrspace(3) %1416, i32 %1415, !dbg !174
  %1418 = insertelement <1 x float> poison, float %1403, i64 0, !dbg !174
  store <1 x float> %1418, ptr addrspace(3) %1417, align 4, !dbg !174
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !174
  tail call void @llvm.amdgcn.s.barrier(), !dbg !174
  %1419 = shl nuw nsw i32 %61, 2, !dbg !174
  %1420 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1419, !dbg !174
  %1421 = load float, ptr addrspace(3) %1420, align 4, !dbg !174
  %1422 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 2147483646, i32 159744), !dbg !174
  %1423 = and i32 %39, 128, !dbg !174
  %1424 = icmp eq i32 %1423, 0, !dbg !174
  %1425 = and i1 %1424, %1410, !dbg !174
  %1426 = shl i32 %1412, 2, !dbg !174
  %1427 = select i1 %1425, i32 %1426, i32 -2147483648, !dbg !174
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %1421, ptr addrspace(8) %1422, i32 %1427, i32 0, i32 0), !dbg !174
  br label %1445, !dbg !170

1428:                                             ; preds = %.loopexit
  %1429 = trunc i64 %1406 to i32, !dbg !175
  %1430 = add i32 %63, %1429, !dbg !175
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  %1431 = shl nuw nsw i32 %38, 2, !dbg !14
  %1432 = and i32 %1431, 124, !dbg !14
  %1433 = shl nuw nsw i32 %41, 7, !dbg !14
  %1434 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1432, !dbg !14
  %1435 = getelementptr inbounds nuw i8, ptr addrspace(3) %1434, i32 %1433, !dbg !14
  %1436 = insertelement <1 x float> poison, float %1403, i64 0, !dbg !14
  store <1 x float> %1436, ptr addrspace(3) %1435, align 4, !dbg !14
  tail call void @llvm.amdgcn.s.waitcnt(i32 49279), !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  %1437 = shl nuw nsw i32 %61, 2, !dbg !14
  %1438 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %1437, !dbg !14
  %1439 = load float, ptr addrspace(3) %1438, align 4, !dbg !14
  %1440 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 2147483646, i32 159744), !dbg !14
  %1441 = and i32 %39, 128, !dbg !14
  %1442 = icmp eq i32 %1441, 0, !dbg !14
  %1443 = shl i32 %1430, 2, !dbg !14
  %1444 = select i1 %1442, i32 %1443, i32 -2147483648, !dbg !14
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %1439, ptr addrspace(8) %1440, i32 %1444, i32 0, i32 0), !dbg !14
  br label %1445, !dbg !170

1445:                                             ; preds = %1408, %1428
  %1446 = shufflevector <1 x float> %1399, <1 x float> poison, <2 x i32> zeroinitializer, !dbg !162
  %1447 = fmul <2 x float> %1380, %1446, !dbg !162
  %1448 = fmul <2 x float> %1381, %1446, !dbg !162
  %1449 = shufflevector <1 x float> %1397, <1 x float> poison, <2 x i32> zeroinitializer, !dbg !162
  %1450 = fmul <2 x float> %1382, %1449, !dbg !162
  %1451 = fmul <2 x float> %1383, %1449, !dbg !162
  %1452 = shl nuw nsw i32 %41, 4, !dbg !31
  %1453 = or disjoint i32 %1452, %59, !dbg !31
  %1454 = or disjoint i32 %55, %1453, !dbg !30
  %1455 = or disjoint i32 %1454, 64, !dbg !30
  %1456 = icmp slt i32 %1455, %32, !dbg !61
  %1457 = icmp slt i32 %1454, %32, !dbg !61
  %1458 = lshr i32 %38, 2, !dbg !51
  %1459 = and i32 %1458, 12, !dbg !51
  %1460 = zext nneg i32 %1459 to i64, !dbg !51
  %1461 = zext i32 %1455 to i64, !dbg !176
  %1462 = zext i32 %1454 to i64, !dbg !176
  %1463 = zext i32 %19 to i64, !dbg !177
  %1464 = zext i32 %18 to i64, !dbg !178
  %1465 = zext i32 %17 to i64, !dbg !179
  %1466 = mul nuw i64 %87, %1465, !dbg !180
  %1467 = mul nuw i64 %1464, %82, !dbg !181
  %1468 = add i64 %1466, %1467, !dbg !180
  %1469 = select i1 %1407, i1 true, i1 %1457, !dbg !182
  %1470 = select i1 %1407, i1 true, i1 %1456, !dbg !182
  %1471 = fptrunc <2 x float> %1451 to <2 x bfloat>, !dbg !183
  %1472 = fptrunc <2 x float> %1450 to <2 x bfloat>, !dbg !183
  %1473 = fptrunc <2 x float> %1448 to <2 x bfloat>, !dbg !183
  %1474 = fptrunc <2 x float> %1447 to <2 x bfloat>, !dbg !183
  %1475 = mul nuw i64 %1462, %1463, !dbg !184
  %1476 = mul nuw i64 %1461, %1463, !dbg !184
  %1477 = add i64 %1468, %1460, !dbg !184
  %1478 = add i64 %1477, %1475, !dbg !184
  %1479 = add i64 %1477, %1476, !dbg !184
  %1480 = trunc i64 %1478 to i32, !dbg !184
  %1481 = trunc i64 %1479 to i32, !dbg !184
  %1482 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %3, i16 0, i64 2147483646, i32 159744), !dbg !185
  %1483 = shufflevector <2 x bfloat> %1471, <2 x bfloat> %1472, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !185
  %1484 = bitcast <4 x bfloat> %1483 to <2 x i32>, !dbg !185
  %1485 = shl i32 %1480, 1, !dbg !185
  %1486 = select i1 %1469, i32 %1485, i32 -2147483648, !dbg !185
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1484, ptr addrspace(8) %1482, i32 %1486, i32 0, i32 0), !dbg !185
  %1487 = shufflevector <2 x bfloat> %1473, <2 x bfloat> %1474, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !185
  %1488 = bitcast <4 x bfloat> %1487 to <2 x i32>, !dbg !185
  %1489 = shl i32 %1481, 1, !dbg !185
  %1490 = select i1 %1470, i32 %1489, i32 -2147483648, !dbg !185
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1488, ptr addrspace(8) %1482, i32 %1490, i32 0, i32 0), !dbg !185
  ret void, !dbg !186
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

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) captures(none)) #8

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #9

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1 immarg, i1 immarg) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #3

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #10

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #10

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.log2.f32(float) #3

attributes #0 = { nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "amdgpu-waves-per-eu"="0, 0" "denormal-fp-math-f32"="ieee" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #8 = { convergent mustprogress nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #9 = { convergent mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #10 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: write) }

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
!32 = !DILocation(line: 381, column: 14, scope: !6)
!33 = !DILocation(line: 382, column: 14, scope: !6)
!34 = !DILocation(line: 385, column: 19, scope: !6)
!35 = !DILocation(line: 401, column: 21, scope: !6)
!36 = !DILocation(line: 402, column: 21, scope: !6)
!37 = !DILocation(line: 403, column: 21, scope: !6)
!38 = !DILocation(line: 405, column: 21, scope: !6)
!39 = !DILocation(line: 406, column: 21, scope: !6)
!40 = !DILocation(line: 407, column: 21, scope: !6)
!41 = !DILocation(line: 426, column: 24, scope: !6)
!42 = !DILocation(line: 427, column: 24, scope: !6)
!43 = !DILocation(line: 19, column: 13, scope: !24, inlinedAt: !44)
!44 = !DILocation(line: 510, column: 16, scope: !24)
!45 = !DILocation(line: 19, column: 12, scope: !24, inlinedAt: !44)
!46 = !DILocation(line: 563, column: 22, scope: !6)
!47 = !DILocation(line: 573, column: 14, scope: !6)
!48 = !DILocation(line: 574, column: 14, scope: !6)
!49 = !DILocation(line: 575, column: 14, scope: !6)
!50 = !DILocation(line: 582, column: 9, scope: !6)
!51 = !DILocation(line: 586, column: 11, scope: !6)
!52 = !DILocation(line: 588, column: 14, scope: !6)
!53 = !DILocation(line: 597, column: 21, scope: !6)
!54 = !DILocation(line: 602, column: 9, scope: !6)
!55 = !DILocation(line: 606, column: 11, scope: !6)
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
!72 = !DILocation(line: 132, column: 5, scope: !24, inlinedAt: !73)
!73 = !DILocation(line: 746, column: 25, scope: !24)
!74 = !DILocation(line: 36, column: 18, scope: !24, inlinedAt: !75)
!75 = !DILocation(line: 140, column: 13, scope: !24, inlinedAt: !73)
!76 = !DILocation(line: 36, column: 18, scope: !24, inlinedAt: !77)
!77 = !DILocation(line: 142, column: 20, scope: !24, inlinedAt: !73)
!78 = !DILocation(line: 36, column: 18, scope: !24, inlinedAt: !79)
!79 = !DILocation(line: 150, column: 17, scope: !24, inlinedAt: !73)
!80 = !DILocation(line: 177, column: 20, scope: !24, inlinedAt: !73)
!81 = !DILocation(line: 261, column: 19, scope: !24, inlinedAt: !73)
!82 = !DILocation(line: 264, column: 19, scope: !24, inlinedAt: !73)
!83 = !DILocation(line: 261, column: 9, scope: !24, inlinedAt: !73)
!84 = !DILocation(line: 263, column: 13, scope: !24, inlinedAt: !73)
!85 = !DILocation(line: 264, column: 9, scope: !24, inlinedAt: !73)
!86 = !DILocation(line: 186, column: 18, scope: !24, inlinedAt: !73)
!87 = !DILocation(line: 181, column: 18, scope: !24, inlinedAt: !73)
!88 = !DILocation(line: 182, column: 14, scope: !24, inlinedAt: !73)
!89 = !DILocation(line: 212, column: 26, scope: !24, inlinedAt: !73)
!90 = !DILocation(line: 170, column: 12, scope: !91, inlinedAt: !93)
!91 = distinct !DILexicalBlockFile(scope: !6, file: !92, discriminator: 0)
!92 = !DIFile(filename: "standard.py", directory: "/root/triton/python/triton/language")
!93 = !DILocation(line: 191, column: 16, scope: !91, inlinedAt: !94)
!94 = !DILocation(line: 209, column: 32, scope: !24, inlinedAt: !73)
!95 = !DILocation(line: 209, column: 16, scope: !24, inlinedAt: !73)
!96 = !DILocation(line: 212, column: 13, scope: !24, inlinedAt: !73)
!97 = !DILocation(line: 293, column: 12, scope: !91, inlinedAt: !98)
!98 = !DILocation(line: 220, column: 16, scope: !24, inlinedAt: !73)
!99 = !DILocation(line: 263, column: 12, scope: !91, inlinedAt: !97)
!100 = !DILocation(line: 241, column: 30, scope: !24, inlinedAt: !73)
!101 = !DILocation(line: 241, column: 17, scope: !24, inlinedAt: !73)
!102 = !DILocation(line: 246, column: 15, scope: !24, inlinedAt: !73)
!103 = !DILocation(line: 248, column: 15, scope: !24, inlinedAt: !73)
!104 = !DILocation(line: 259, column: 26, scope: !24, inlinedAt: !73)
!105 = !DILocation(line: 259, column: 19, scope: !24, inlinedAt: !73)
!106 = !DILocation(line: 296, column: 93, scope: !6)
!107 = !DILocation(line: 296, column: 110, scope: !6)
!108 = !DILocation(line: 296, column: 127, scope: !6)
!109 = !DILocation(line: 296, column: 254, scope: !6)
!110 = !DILocation(line: 296, column: 281, scope: !6)
!111 = !DILocation(line: 296, column: 308, scope: !6)
!112 = !DILocation(line: 296, column: 335, scope: !6)
!113 = !DILocation(line: 296, column: 362, scope: !6)
!114 = !DILocation(line: 296, column: 389, scope: !6)
!115 = !DILocation(line: 746, column: 25, scope: !6)
!116 = !DILocation(line: 798, column: 8, scope: !6)
!117 = !DILocation(line: 798, column: 5, scope: !6)
!118 = !DILocation(line: 177, column: 20, scope: !24, inlinedAt: !119)
!119 = !DILocation(line: 811, column: 25, scope: !24)
!120 = !DILocation(line: 261, column: 19, scope: !24, inlinedAt: !119)
!121 = !DILocation(line: 264, column: 19, scope: !24, inlinedAt: !119)
!122 = !DILocation(line: 132, column: 5, scope: !24, inlinedAt: !119)
!123 = !DILocation(line: 803, column: 19, scope: !6)
!124 = !DILocation(line: 806, column: 19, scope: !6)
!125 = !DILocation(line: 0, scope: !6)
!126 = !DILocation(line: 136, column: 24, scope: !24, inlinedAt: !119)
!127 = !DILocation(line: 33, column: 16, scope: !24, inlinedAt: !128)
!128 = !DILocation(line: 140, column: 13, scope: !24, inlinedAt: !119)
!129 = !DILocation(line: 34, column: 18, scope: !24, inlinedAt: !128)
!130 = !DILocation(line: 34, column: 18, scope: !24, inlinedAt: !131)
!131 = !DILocation(line: 142, column: 20, scope: !24, inlinedAt: !119)
!132 = !DILocation(line: 31, column: 18, scope: !24, inlinedAt: !133)
!133 = !DILocation(line: 150, column: 17, scope: !24, inlinedAt: !119)
!134 = !DILocation(line: 168, column: 27, scope: !24, inlinedAt: !119)
!135 = !DILocation(line: 168, column: 26, scope: !24, inlinedAt: !119)
!136 = !DILocation(line: 169, column: 22, scope: !24, inlinedAt: !119)
!137 = !DILocation(line: 170, column: 28, scope: !24, inlinedAt: !119)
!138 = !DILocation(line: 171, column: 20, scope: !24, inlinedAt: !119)
!139 = !DILocation(line: 181, column: 18, scope: !24, inlinedAt: !119)
!140 = !DILocation(line: 182, column: 14, scope: !24, inlinedAt: !119)
!141 = !DILocation(line: 186, column: 18, scope: !24, inlinedAt: !119)
!142 = !DILocation(line: 198, column: 14, scope: !24, inlinedAt: !119)
!143 = !DILocation(line: 170, column: 12, scope: !91, inlinedAt: !144)
!144 = !DILocation(line: 191, column: 16, scope: !91, inlinedAt: !145)
!145 = !DILocation(line: 209, column: 32, scope: !24, inlinedAt: !119)
!146 = !DILocation(line: 212, column: 26, scope: !24, inlinedAt: !119)
!147 = !DILocation(line: 209, column: 16, scope: !24, inlinedAt: !119)
!148 = !DILocation(line: 212, column: 13, scope: !24, inlinedAt: !119)
!149 = !DILocation(line: 293, column: 12, scope: !91, inlinedAt: !150)
!150 = !DILocation(line: 220, column: 16, scope: !24, inlinedAt: !119)
!151 = !DILocation(line: 263, column: 12, scope: !91, inlinedAt: !149)
!152 = !DILocation(line: 241, column: 30, scope: !24, inlinedAt: !119)
!153 = !DILocation(line: 241, column: 17, scope: !24, inlinedAt: !119)
!154 = !DILocation(line: 246, column: 15, scope: !24, inlinedAt: !119)
!155 = !DILocation(line: 248, column: 15, scope: !24, inlinedAt: !119)
!156 = !DILocation(line: 259, column: 26, scope: !24, inlinedAt: !119)
!157 = !DILocation(line: 259, column: 19, scope: !24, inlinedAt: !119)
!158 = !DILocation(line: 261, column: 9, scope: !24, inlinedAt: !119)
!159 = !DILocation(line: 263, column: 13, scope: !24, inlinedAt: !119)
!160 = !DILocation(line: 264, column: 9, scope: !24, inlinedAt: !119)
!161 = !DILocation(line: 861, column: 15, scope: !6)
!162 = !DILocation(line: 862, column: 11, scope: !6)
!163 = !DILocation(line: 884, column: 21, scope: !6)
!164 = !DILocation(line: 888, column: 29, scope: !6)
!165 = !DILocation(line: 888, column: 23, scope: !6)
!166 = !DILocation(line: 890, column: 9, scope: !6)
!167 = !DILocation(line: 900, column: 13, scope: !6)
!168 = !DILocation(line: 901, column: 15, scope: !6)
!169 = !DILocation(line: 905, column: 12, scope: !6)
!170 = !DILocation(line: 905, column: 9, scope: !6)
!171 = !DILocation(line: 906, column: 44, scope: !6)
!172 = !DILocation(line: 907, column: 24, scope: !6)
!173 = !DILocation(line: 909, column: 17, scope: !6)
!174 = !DILocation(line: 908, column: 13, scope: !6)
!175 = !DILocation(line: 913, column: 17, scope: !6)
!176 = !DILocation(line: 585, column: 11, scope: !6)
!177 = !DILocation(line: 415, column: 21, scope: !6)
!178 = !DILocation(line: 414, column: 21, scope: !6)
!179 = !DILocation(line: 413, column: 21, scope: !6)
!180 = !DILocation(line: 918, column: 9, scope: !6)
!181 = !DILocation(line: 919, column: 11, scope: !6)
!182 = !DILocation(line: 925, column: 5, scope: !6)
!183 = !DILocation(line: 929, column: 10, scope: !6)
!184 = !DILocation(line: 930, column: 14, scope: !6)
!185 = !DILocation(line: 930, column: 5, scope: !6)
!186 = !DILocation(line: 297, column: 1, scope: !6)
