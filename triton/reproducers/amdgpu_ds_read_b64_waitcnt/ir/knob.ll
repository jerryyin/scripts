; Minimal instruction-selection demo for the gfx950 LDS read-lowering knob.
;
; The runtime race lives in the attention kernel (see attn_fwd.ll); this file
; isolates ONLY the backend decision that distinguishes the racy build from the
; stable one: how a 2-element f32 LDS read is lowered.
;
;   @contig:  one contiguous <2 x float> LDS load   -> ds_read_b64
;   @strided: two stride-64 scalar f32 LDS loads     -> ds_read2st64_b32 offset1:1
;
; Both read two f32 lanes of the same shared buffer; only the access pattern
; (and therefore the emitted DS opcode) differs. This is exactly the difference
; between attn_fwd.ll (ds_read_b64, races) and attn_fwd_strided.ll
; (ds_read2st64_b32, stable) — there, the LDS *writes* are byte-identical and
; only the read pattern changes.
;
; Reproduce:
;   llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 knob.ll -o knob.s
;   grep -A2 '^contig:'  knob.s   # ds_read_b64
;   grep -A2 '^strided:' knob.s   # ds_read2st64_b32

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

; Contiguous 64-bit LDS read -> ds_read_b64
define amdgpu_kernel void @contig(ptr addrspace(1) %out, i32 %i) {
  %base = getelementptr float, ptr addrspace(3) null, i32 %i
  %v = load <2 x float>, ptr addrspace(3) %base, align 8
  store <2 x float> %v, ptr addrspace(1) %out
  ret void
}

; Same two f32 values, read stride-64 -> ds_read2st64_b32 offset1:1
define amdgpu_kernel void @strided(ptr addrspace(1) %out, i32 %i) {
  %b0 = getelementptr float, ptr addrspace(3) null, i32 %i
  %i1 = add i32 %i, 64
  %b1 = getelementptr float, ptr addrspace(3) null, i32 %i1
  %v0 = load float, ptr addrspace(3) %b0, align 4
  %v1 = load float, ptr addrspace(3) %b1, align 4
  %e0 = insertelement <2 x float> undef, float %v0, i32 0
  %e1 = insertelement <2 x float> %e0, float %v1, i32 1
  store <2 x float> %e1, ptr addrspace(1) %out
  ret void
}
