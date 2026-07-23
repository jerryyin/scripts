; Minimal repro (llvm-reduce'd from the MoE MLA-decode kernel) for the AMDGPU
; isReallyAClobber missed-optimization. COMPILE-ONLY (loops forever / loads null;
; never executed) — it exists to exercise the annotate-uniform / ISel decision.
;
; The uniform read-only `load <1 x i32>` from addrspace(1) sits in a loop whose only
; memory def is `llvm.amdgcn.tensor.load.to.lds` (writes argmem/inaccessiblemem, i.e.
; LDS). AA proves that intrinsic cannot modify the addrspace(1) location, yet upstream
; isReallyAClobber declares it a clobber WITHOUT asking AA -> the load is denied
; !amdgpu.noclobber -> selected as VMEM global_load instead of scalar s_load.
;
;   opt -passes=amdgpu-annotate-uniform -mcpu=gfx1250 -S   : !amdgpu.noclobber on the load?
;   llc -mcpu=gfx1250                                      : global_load_b32 (before) vs s_load_b32 (after)
;
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16() {
.lr.ph:
  tail call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %0

0:                                                ; preds = %0, %.lr.ph
  %1 = load <1 x i32>, ptr addrspace(1) null, align 4
  fence release
  br label %0
}

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #0

attributes #0 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
