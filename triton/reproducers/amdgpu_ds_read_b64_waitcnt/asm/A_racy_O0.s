	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	repro                           ; -- Begin function repro
	.p2align	8
	.type	repro,@function
repro:                                  ; @repro
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.1:
	.file	1 "/root/scripts/triton/reproducers/amdgpu_ds_read_b64_waitcnt/ir" "micro-dot.ttgir"
	.loc	1 11 0 prologue_end             ; micro-dot.ttgir:11:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx4 s[4:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.2:
.LBB0_0:
	s_mov_b32 s10, s4
	s_mov_b64 s[4:5], s[2:3]
                                        ; kill: def $sgpr0 killed $sgpr10
                                        ; kill: def $sgpr2_sgpr3 killed $sgpr4_sgpr5
.Ltmp1:
	.loc	1 114 11 is_stmt 1              ; micro-dot.ttgir:114:11
	v_readfirstlane_b32 s7, v0
                                        ; implicit-def: $sgpr0_sgpr1
	.loc	1 129 5                         ; micro-dot.ttgir:129:5
	s_mov_b32 s2, s1
	s_mov_b32 s0, s7
	s_mov_b32 s1, s2
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	v_mov_b32_e32 v2, s0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v0
	s_mov_b32 s2, 32
	s_mov_b32 s0, 0xc0
                                        ; kill: def $sgpr0 killed $sgpr0 def $sgpr0_sgpr1
	s_mov_b32 s1, s2
	s_mov_b32 s2, s1
	v_mov_b32_e32 v1, v3
	v_and_b32_e64 v1, v1, s2
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_and_b32_e64 v4, v2, s0
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v1
	.loc	1 80 14                         ; micro-dot.ttgir:80:14
	v_mov_b32_e32 v3, v4
	s_mov_b32 s0, 63
	.loc	1 29 11                         ; micro-dot.ttgir:29:11
	v_and_or_b32 v1, v0, s0, v3
	s_mov_b32 s0, 31
	v_and_b32_e64 v7, v0, s0
                                        ; kill: def $vgpr5 killed $vgpr5 killed $vgpr4_vgpr5 killed $exec
	s_mov_b32 s3, 3
	v_lshrrev_b32_e64 v12, s3, v5
	s_mov_b32 s2, 1
	v_lshrrev_b32_e64 v8, s2, v3
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v12
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_mov_b32_e32 v2, v9
	v_or_b32_e64 v13, v2, s2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v25, v11
	s_mov_b32 s8, 2
	s_mov_b32 s0, s8
	s_mov_b32 s1, s3
	s_mov_b32 s6, s1
	v_or_b32_e64 v4, v25, s6
	v_mov_b32_e32 v26, v10
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	v_or_b32_e64 v18, v26, s0
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v4
	s_mov_b32 s6, 10
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $sgpr1
                                        ; kill: def $sgpr0 killed $sgpr0 def $sgpr0_sgpr1
	.loc	1 29 11                         ; micro-dot.ttgir:29:11
	s_mov_b32 s1, s6
	s_mov_b32 s0, s1
	v_mov_b32_e32 v4, s0
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v4
	v_mov_b32_e32 v4, v11
	v_or_b32_e64 v4, v2, v4
                                        ; kill: def $vgpr8 killed $vgpr8 killed $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v6, v10
	v_or_b32_e64 v14, v8, v6
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v4
	s_mov_b32 s0, 11
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_or_b32_e64 v6, v2, s0
	s_mov_b32 s0, 9
	v_or_b32_e64 v10, v2, s0
	s_mov_b32 s0, 8
	v_or_b32_e64 v20, v2, s0
	s_mov_b32 s0, 16
	v_or_b32_e64 v4, v2, s0
	s_mov_b32 s1, 27
	v_or_b32_e64 v2, v2, s1
	.loc	1 33 12                         ; micro-dot.ttgir:33:12
	v_mov_b32_e32 v9, v15
	v_cvt_f32_u32_e64 v11, v9
	v_mov_b32_e32 v9, v14
	v_cvt_f32_u32_e64 v16, v9
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v11
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr6
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21_vgpr22_vgpr23 killed $exec
	.loc	1 34 12                         ; micro-dot.ttgir:34:12
	v_mov_b32_e32 v21, v10
	v_mov_b32_e32 v22, v9
	v_mov_b32_e32 v23, s1
                                        ; implicit-def: $sgpr14
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr13
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr12
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr11
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr9
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr15
	v_mov_b32_e32 v28, v12
	v_mov_b32_e32 v29, s14
	v_mov_b32_e32 v30, s13
	v_mov_b32_e32 v31, s12
	v_mov_b32_e32 v32, s11
	v_mov_b32_e32 v33, s9
	v_mov_b32_e32 v34, s6
	v_mov_b32_e32 v35, s1
	v_mov_b32_e32 v9, v22
	v_mov_b32_e32 v10, v21
	v_mov_b32_e32 v11, v20
                                        ; kill: def $vgpr28 killed $vgpr28 killed $vgpr28_vgpr29_vgpr30_vgpr31_vgpr32_vgpr33_vgpr34_vgpr35 killed $exec
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr9
                                        ; kill: def $vgpr28 killed $vgpr28 def $vgpr28_vgpr29_vgpr30_vgpr31_vgpr32_vgpr33_vgpr34_vgpr35 killed $exec
	v_mov_b32_e32 v29, v13
	v_mov_b32_e32 v30, s6
	v_mov_b32_e32 v31, s1
	v_mov_b32_e32 v32, v11
	v_mov_b32_e32 v33, v10
	v_mov_b32_e32 v34, v9
	v_mov_b32_e32 v35, v6
	v_mov_b32_e32 v6, v35
	v_mov_b32_e32 v9, v34
	v_mov_b32_e32 v10, v33
	v_mov_b32_e32 v11, v32
	v_mov_b32_e32 v15, v29
	v_mov_b32_e32 v34, v28
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_mov_b32_e32 v13, v19
	v_mov_b32_e32 v14, v18
                                        ; kill: def $vgpr34 killed $vgpr34 def $vgpr34_vgpr35_vgpr36_vgpr37_vgpr38_vgpr39_vgpr40_vgpr41 killed $exec
	.loc	1 34 12                         ; micro-dot.ttgir:34:12
	v_mov_b32_e32 v35, v15
	v_mov_b32_e32 v36, v14
	v_mov_b32_e32 v37, v13
	v_mov_b32_e32 v38, v11
	v_mov_b32_e32 v39, v10
	v_mov_b32_e32 v40, v9
	v_mov_b32_e32 v41, v6
	v_mov_b32_e32 v6, v37
	v_cvt_f32_u32_e64 v10, v6
	v_mov_b32_e32 v6, v36
	v_cvt_f32_u32_e64 v11, v6
	v_mov_b32_e32 v6, v35
	v_cvt_f32_u32_e64 v14, v6
	v_mov_b32_e32 v6, v34
	v_cvt_f32_u32_e64 v32, v6
	v_mov_b32_e32 v6, v41
	v_cvt_f32_u32_e64 v9, v6
	v_mov_b32_e32 v6, v40
	v_cvt_f32_u32_e64 v15, v6
	v_mov_b32_e32 v6, v39
	v_cvt_f32_u32_e64 v18, v6
	v_mov_b32_e32 v6, v38
	v_cvt_f32_u32_e64 v36, v6
	v_cvt_f32_u32_e64 v4, v4
	v_cvt_f32_u32_e64 v2, v2
	s_mov_b32 s12, 0x3d000000
	s_mov_b32 s1, 0x3b800000
	.loc	1 37 17                         ; micro-dot.ttgir:37:17
	s_mov_b32 s14, s1
	s_mov_b32 s15, s12
	v_pk_mul_f32 v[20:21], v[16:17], s[14:15]
	s_nop 0
	v_mov_b32_e32 v16, v20
	v_pk_mov_b32 v[20:21], v[20:21], v[20:21] op_sel:[1,0]
	s_nop 0
	v_mov_b32_e32 v6, v21
	v_mov_b32_e32 v13, v20
	v_mov_b32_e32 v38, v16
	v_mov_b32_e32 v39, v16
	v_mov_b32_e32 v40, v16
	v_mov_b32_e32 v41, v16
	v_mov_b32_e32 v42, v16
	v_mov_b32_e32 v43, v16
	v_mov_b32_e32 v44, v13
	v_mov_b32_e32 v45, v6
	v_mov_b32_e32 v6, v45
	v_mov_b32_e32 v13, v44
	v_mov_b32_e32 v31, v43
	v_mov_b32_e32 v16, v42
	v_mov_b32_e32 v17, v41
	v_mov_b32_e32 v20, v40
	v_mov_b32_e32 v21, v39
	v_mov_b32_e32 v44, v38
                                        ; kill: def $vgpr36 killed $vgpr36 def $vgpr36_vgpr37_vgpr38_vgpr39 killed $exec
	.loc	1 38 17                         ; micro-dot.ttgir:38:17
	v_mov_b32_e32 v37, v18
	v_mov_b32_e32 v38, v15
	v_mov_b32_e32 v39, v9
	v_mov_b32_e32 v9, v39
	v_mov_b32_e32 v18, v38
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v9
	s_mov_b32 s20, s12
	s_mov_b32 s21, s12
	s_mov_b32 s22, s1
	s_mov_b32 s23, s12
	s_mov_b32 s1, s23
	s_mov_b32 s14, s22
                                        ; kill: def $sgpr14 killed $sgpr14 def $sgpr14_sgpr15
	s_mov_b32 s15, s1
	v_pk_mul_f32 v[28:29], v[18:19], s[14:15]
	s_nop 0
	v_mov_b32_e32 v9, v29
	v_mov_b32_e32 v24, v28
                                        ; kill: def $vgpr32 killed $vgpr32 def $vgpr32_vgpr33_vgpr34_vgpr35 killed $exec
	v_mov_b32_e32 v33, v14
	v_mov_b32_e32 v34, v11
	v_mov_b32_e32 v35, v10
	v_mov_b32_e32 v14, v35
	v_mov_b32_e32 v10, v34
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v14
	s_mov_b32 s16, s12
	s_mov_b32 s17, s12
	s_mov_b32 s18, s12
	s_mov_b32 s19, s12
	s_mov_b32 s1, s19
	s_mov_b32 s14, s18
                                        ; kill: def $sgpr14 killed $sgpr14 def $sgpr14_sgpr15
	s_mov_b32 s15, s1
	v_pk_mul_f32 v[18:19], v[10:11], s[14:15]
	s_nop 0
	v_mov_b32_e32 v10, v19
	v_mov_b32_e32 v11, v18
	v_mov_b64_e32 v[14:15], v[36:37]
	s_mov_b64 s[14:15], s[20:21]
	v_pk_mul_f32 v[14:15], v[14:15], s[14:15]
	s_nop 0
	v_mov_b32_e32 v27, v15
	v_mov_b32_e32 v30, v14
	v_mov_b64_e32 v[22:23], v[32:33]
	s_mov_b64 s[14:15], s[16:17]
	v_pk_mul_f32 v[22:23], v[22:23], s[14:15]
	s_nop 0
	v_mov_b32_e32 v32, v23
	v_mov_b32_e32 v36, v22
                                        ; kill: def $vgpr36 killed $vgpr36 def $vgpr36_vgpr37_vgpr38_vgpr39_vgpr40_vgpr41_vgpr42_vgpr43 killed $exec
	v_mov_b32_e32 v37, v32
	v_mov_b32_e32 v38, v11
	v_mov_b32_e32 v39, v10
	v_mov_b32_e32 v40, v30
	v_mov_b32_e32 v41, v27
	v_mov_b32_e32 v42, v24
	v_mov_b32_e32 v43, v9
	v_mul_f32_e64 v4, v4, s12
                                        ; implicit-def: $sgpr14_sgpr15
	.loc	1 27 20                         ; micro-dot.ttgir:27:20
	s_mov_b32 s1, s15
                                        ; kill: def $sgpr10 killed $sgpr10 def $sgpr10_sgpr11
	s_mov_b32 s11, s1
	s_mov_b32 s6, s10
	v_mov_b32_e32 v10, s6
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v2
	s_mov_b32 s10, 0x3fb8aa3b
                                        ; kill: def $sgpr10 killed $sgpr10 def $sgpr10_sgpr11
	s_mov_b32 s11, s12
	v_pk_mul_f32 v[10:11], v[10:11], s[10:11]
	s_nop 0
	v_mov_b32_e32 v2, v10
                                        ; kill: def $vgpr44 killed $vgpr44 def $vgpr44_vgpr45_vgpr46_vgpr47 killed $exec
	.loc	1 39 16                         ; micro-dot.ttgir:39:16
	v_mov_b32_e32 v45, v21
	v_mov_b32_e32 v46, v20
	v_mov_b32_e32 v47, v17
	v_mov_b32_e32 v17, v47
	v_mov_b32_e32 v20, v46
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v17
	v_pk_add_f32 v[20:21], v[18:19], v[20:21]
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17_vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v17, v31
	v_mov_b32_e32 v18, v13
	v_mov_b32_e32 v19, v6
	v_mov_b32_e32 v6, v19
	v_mov_b32_e32 v32, v18
                                        ; kill: def $vgpr32 killed $vgpr32 def $vgpr32_vgpr33 killed $exec
	v_mov_b32_e32 v33, v6
	v_pk_add_f32 v[32:33], v[28:29], v[32:33]
	v_mov_b64_e32 v[34:35], v[44:45]
	v_pk_add_f32 v[34:35], v[22:23], v[34:35]
                                        ; kill: def $vgpr16_vgpr17 killed $vgpr16_vgpr17 killed $vgpr16_vgpr17_vgpr18_vgpr19 killed $exec
	v_pk_add_f32 v[22:23], v[14:15], v[16:17]
	v_mov_b32_e32 v13, v42
	v_add_f32_e64 v4, v4, v13
	v_mov_b32_e32 v6, v11
	v_add_f32_e64 v6, v6, v13
	.loc	1 40 15                         ; micro-dot.ttgir:40:15
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v15, v41
	v_mov_b32_e32 v18, v40
	v_mov_b32_e32 v19, v39
	v_mov_b32_e32 v36, v38
	v_mov_b32_e32 v13, v45
	v_mov_b32_e32 v14, v44
	v_mov_b32_e32 v16, v43
	v_mov_b32_e32 v40, v42
                                        ; kill: def $vgpr40 killed $vgpr40 def $vgpr40_vgpr41_vgpr42_vgpr43 killed $exec
	v_mov_b32_e32 v41, v16
	v_mov_b32_e32 v42, v14
	v_mov_b32_e32 v43, v13
	v_mov_b32_e32 v13, v43
	v_mov_b32_e32 v16, v42
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v13
	v_pk_add_f32 v[16:17], v[16:17], v[32:33]
	s_nop 0
	v_mov_b32_e32 v13, v17
	v_mov_b32_e32 v14, v16
                                        ; kill: def $vgpr36 killed $vgpr36 def $vgpr36_vgpr37_vgpr38_vgpr39 killed $exec
	v_mov_b32_e32 v37, v19
	v_mov_b32_e32 v38, v18
	v_mov_b32_e32 v39, v15
	v_mov_b32_e32 v15, v39
	v_mov_b32_e32 v18, v38
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v15
	v_pk_add_f32 v[20:21], v[18:19], v[20:21]
	s_nop 0
	v_mov_b32_e32 v32, v21
	v_mov_b32_e32 v33, v20
	v_mov_b64_e32 v[18:19], v[40:41]
	v_pk_add_f32 v[18:19], v[18:19], v[22:23]
	s_nop 0
	v_mov_b32_e32 v15, v19
	v_mov_b32_e32 v31, v18
	v_mov_b64_e32 v[22:23], v[36:37]
	v_pk_add_f32 v[22:23], v[22:23], v[34:35]
	s_nop 0
	v_mov_b32_e32 v34, v23
	v_mov_b32_e32 v36, v22
                                        ; kill: def $vgpr36 killed $vgpr36 def $vgpr36_vgpr37_vgpr38_vgpr39_vgpr40_vgpr41_vgpr42_vgpr43 killed $exec
	v_mov_b32_e32 v37, v34
	v_mov_b32_e32 v38, v33
	v_mov_b32_e32 v39, v32
	v_mov_b32_e32 v40, v31
	v_mov_b32_e32 v41, v15
	v_mov_b32_e32 v42, v14
	v_mov_b32_e32 v43, v13
	v_add_f32_e64 v4, v2, v4
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_mov_b32_e32 v32, v12
	v_mov_b32_e32 v33, v12
	v_mov_b32_e32 v34, v12
	v_mov_b32_e32 v35, v12
	v_mov_b32_e32 v12, v32
	s_mov_b32 s6, 17
	v_or_b32_e64 v12, v12, s6
	v_mov_b32_e32 v13, v33
	s_mov_b32 s6, 18
	v_or_b32_e64 v13, v13, s6
	v_mov_b32_e32 v14, v34
	s_mov_b32 s6, 19
	v_or_b32_e64 v14, v14, s6
	v_mov_b32_e32 v15, v35
	s_mov_b32 s10, 24
	v_or_b32_e64 v15, v15, s10
	.loc	1 34 12                         ; micro-dot.ttgir:34:12
	v_cvt_f32_u32_e64 v31, v15
	v_cvt_f32_u32_e64 v14, v14
	v_cvt_f32_u32_e64 v15, v13
	v_cvt_f32_u32_e64 v12, v12
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	.loc	1 38 17                         ; micro-dot.ttgir:38:17
	v_mov_b32_e32 v13, v15
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr9
                                        ; kill: def $sgpr12 killed $sgpr12 def $sgpr12_sgpr13
	s_mov_b32 s13, s6
	v_pk_mul_f32 v[12:13], v[12:13], s[12:13] op_sel_hi:[1,0]
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v31
	v_pk_mul_f32 v[14:15], v[14:15], s[12:13] op_sel_hi:[1,0]
                                        ; kill: def $vgpr30 killed $vgpr30 def $vgpr30_vgpr31_vgpr32_vgpr33 killed $exec
	.loc	1 39 16                         ; micro-dot.ttgir:39:16
	v_mov_b32_e32 v31, v27
	v_mov_b32_e32 v32, v24
	v_mov_b32_e32 v33, v9
	v_mov_b32_e32 v9, v32
	v_mov_b32_e32 v30, v9
	v_mov_b32_e32 v31, v9
	v_mov_b32_e32 v32, v9
	v_mov_b32_e32 v33, v9
	v_mov_b32_e32 v9, v33
	v_mov_b32_e32 v34, v32
                                        ; kill: def $vgpr34 killed $vgpr34 def $vgpr34_vgpr35 killed $exec
	v_mov_b32_e32 v35, v9
	v_pk_add_f32 v[14:15], v[14:15], v[34:35]
                                        ; kill: def $vgpr30_vgpr31 killed $vgpr30_vgpr31 killed $vgpr30_vgpr31_vgpr32_vgpr33 killed $exec
	v_pk_add_f32 v[30:31], v[12:13], v[30:31]
	.loc	1 40 15                         ; micro-dot.ttgir:40:15
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v9, v35
	v_mov_b32_e32 v12, v34
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v9
	v_pk_add_f32 v[12:13], v[12:13], v[14:15]
	s_nop 0
	v_mov_b32_e32 v9, v13
	v_mov_b32_e32 v24, v12
	v_mov_b64_e32 v[14:15], v[32:33]
	v_pk_add_f32 v[14:15], v[14:15], v[30:31]
	s_nop 0
	v_mov_b32_e32 v27, v15
	v_mov_b32_e32 v32, v14
                                        ; kill: def $vgpr32 killed $vgpr32 def $vgpr32_vgpr33_vgpr34_vgpr35 killed $exec
	v_mov_b32_e32 v33, v27
	v_mov_b32_e32 v34, v24
	v_mov_b32_e32 v35, v9
	v_add_f32_e64 v2, v2, v6
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v6, v36
	v_mov_b32_e32 v9, v37
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v6, v6, v9
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v9, v38
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v6, v6, v9
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v9, v39
	v_mov_b32_e32 v24, v40
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v9, v9, v24
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v24, v41
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v30, v9, v24
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v9, v42
	v_mov_b32_e32 v24, v43
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v9, v9, v24
	v_max_f32_e64 v27, v9, v4
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v9, v32
	v_mov_b32_e32 v24, v33
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v9, v9, v24
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v24, v34
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v9, v9, v24
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v24, v35
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v6, v6, v30
	v_max_f32_e64 v6, v6, v27
	s_mov_b32 s6, 26
	s_mov_b32 s14, 25
                                        ; kill: def $sgpr14 killed $sgpr14 def $sgpr14_sgpr15
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	s_mov_b32 s15, s6
	s_mov_b32 s6, s15
	v_or_b32_e64 v25, v25, s6
	s_mov_b32 s6, s14
	v_or_b32_e64 v26, v26, s6
                                        ; kill: def $vgpr26 killed $vgpr26 def $vgpr26_vgpr27 killed $exec
	v_mov_b32_e32 v27, v25
	.loc	1 34 12                         ; micro-dot.ttgir:34:12
	v_mov_b32_e32 v25, v27
	v_cvt_f32_u32_e64 v25, v25
                                        ; kill: def $vgpr26 killed $vgpr26 killed $vgpr26_vgpr27 killed $exec
	v_cvt_f32_u32_e64 v26, v26
                                        ; kill: def $vgpr26 killed $vgpr26 def $vgpr26_vgpr27 killed $exec
	v_mov_b32_e32 v27, v25
	.loc	1 38 17                         ; micro-dot.ttgir:38:17
	v_pk_mul_f32 v[26:27], v[26:27], s[12:13] op_sel_hi:[1,0]
	.loc	1 39 16                         ; micro-dot.ttgir:39:16
	s_nop 0
	v_pk_add_f32 v[26:27], v[26:27], v[28:29] op_sel_hi:[1,0]
	.loc	1 40 15                         ; micro-dot.ttgir:40:15
	s_nop 0
	v_pk_add_f32 v[10:11], v[10:11], v[26:27] op_sel_hi:[0,1]
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_mov_b32_e32 v25, v10
	v_max_f32_e64 v24, v24, v25
	v_mov_b32_e32 v25, v11
	v_max_f32_e64 v24, v24, v25
	v_max_f32_e64 v9, v9, v24
	v_max_f32_e64 v9, v9, v2
	v_max_f32_e64 v9, v6, v9
	.loc	1 41 17                         ; micro-dot.ttgir:41:17
	v_mov_b32_e32 v6, v9
	s_nop 1
	v_permlane32_swap_b32_e64 v6, v9
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e64 v9, v9, v9
	v_max_f32_e64 v6, v6, v6
	v_max_f32_e64 v6, v6, v9
	s_mov_b32 s6, 0xff800000
	.loc	1 46 16                         ; micro-dot.ttgir:46:16
	v_max_f32_e64 v6, v6, s6
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_mov_b32_e32 v9, s1
	v_mov_b32_e32 v24, v6
	v_mov_b32_e32 v25, v9
	v_pk_add_f32 v[22:23], v[22:23], v[24:25] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21], v[20:21], v[24:25] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[18:19], v[18:19], v[24:25] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17], v[16:17], v[24:25] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e64 v4, v4, v6
	v_pk_add_f32 v[14:15], v[14:15], v[24:25] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[12:13], v[24:25] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[10:11], v[10:11], v[24:25] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e64 v2, v2, v6
	.loc	1 50 14                         ; micro-dot.ttgir:50:14
	v_mov_b32_e32 v6, v22
	v_exp_f32_e64 v32, v6
	v_mov_b32_e32 v6, v23
	v_exp_f32_e64 v30, v6
	v_mov_b32_e32 v6, v20
	v_exp_f32_e64 v28, v6
	v_mov_b32_e32 v6, v21
	v_exp_f32_e64 v26, v6
	v_mov_b32_e32 v6, v18
	v_exp_f32_e64 v24, v6
	v_mov_b32_e32 v6, v19
	v_exp_f32_e64 v22, v6
	v_mov_b32_e32 v6, v16
	v_exp_f32_e64 v20, v6
	v_mov_b32_e32 v6, v17
	v_exp_f32_e64 v17, v6
	v_exp_f32_e64 v31, v4
	v_mov_b32_e32 v4, v14
	v_exp_f32_e64 v29, v4
	v_mov_b32_e32 v4, v15
	v_exp_f32_e64 v27, v4
	v_mov_b32_e32 v4, v12
	v_exp_f32_e64 v25, v4
	v_mov_b32_e32 v4, v13
	v_exp_f32_e64 v23, v4
	v_mov_b32_e32 v4, v10
	v_exp_f32_e64 v21, v4
	v_mov_b32_e32 v4, v11
	v_exp_f32_e64 v19, v4
	v_exp_f32_e64 v16, v2
	.loc	1 51 16                         ; micro-dot.ttgir:51:16
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v10, v32
	v_mov_b32_e32 v11, v2
                                        ; kill: def $vgpr10 killed $vgpr10 killed $vgpr10_vgpr11 killed $exec
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v30
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v12, v28
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v42, v12
                                        ; kill: def $vgpr42 killed $vgpr42 def $vgpr42_vgpr43 killed $exec
	v_mov_b32_e32 v43, v26
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v12, v24
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v34, v12
                                        ; kill: def $vgpr34 killed $vgpr34 def $vgpr34_vgpr35 killed $exec
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v12, v20
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v40, v12
                                        ; kill: def $vgpr40 killed $vgpr40 def $vgpr40_vgpr41 killed $exec
	v_mov_b32_e32 v41, v17
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v12, v31
	v_mov_b32_e32 v13, v2
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v29
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v14, v27
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v38, v14
                                        ; kill: def $vgpr38 killed $vgpr38 def $vgpr38_vgpr39 killed $exec
	v_mov_b32_e32 v39, v25
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v14, v23
	v_mov_b32_e32 v15, v2
                                        ; kill: def $vgpr14 killed $vgpr14 killed $vgpr14_vgpr15 killed $exec
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v21
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v36, v19
	v_mov_b32_e32 v37, v2
                                        ; kill: def $vgpr36 killed $vgpr36 killed $vgpr36_vgpr37 killed $exec
                                        ; kill: def $vgpr36 killed $vgpr36 def $vgpr36_vgpr37 killed $exec
	v_mov_b32_e32 v37, v16
	v_pk_add_f32 v[10:11], v[10:11], v[42:43]
	v_pk_add_f32 v[34:35], v[34:35], v[40:41]
	v_pk_add_f32 v[12:13], v[12:13], v[38:39]
	v_pk_add_f32 v[14:15], v[14:15], v[36:37]
	v_pk_add_f32 v[10:11], v[10:11], v[34:35]
	v_pk_add_f32 v[12:13], v[12:13], v[14:15]
	s_nop 0
	v_pk_add_f32 v[10:11], v[10:11], v[12:13]
	.loc	1 53 14                         ; micro-dot.ttgir:53:14
	s_nop 0
	v_pk_add_f32 v[10:11], v[10:11], v[10:11] op_sel:[0,1] op_sel_hi:[1,0]
	.loc	1 51 16                         ; micro-dot.ttgir:51:16
	s_nop 0
	v_mov_b32_e32 v4, v10
	v_mov_b32_e32 v2, v4
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v4
	.loc	1 53 14                         ; micro-dot.ttgir:53:14
	v_add_f32_e64 v11, v2, v4
	.loc	1 63 5                          ; micro-dot.ttgir:63:5
	v_lshlrev_b32_e64 v4, s8, v1
	s_mov_b32 s1, 64
	s_and_b32 s11, s7, s1
	s_mov_b32 s9, 0
	s_mov_b32 s1, 0x2000
	s_add_i32 s6, s9, s1
	s_mov_b32 s1, 0
	s_cmp_eq_u32 s11, s1
	s_mov_b32 s12, 0x110
	s_cselect_b32 s12, s1, s12
	s_mov_b32 s13, 0x2fc
	v_mov_b32_e32 v2, s13
	v_bitop3_b32 v9, v4, s12, v2 bitop3:0x6c
	v_add_u32_e64 v2, s6, v9
	v_mov_b32_e32 v6, 0x3f803f80
	ds_write_b32 v2, v6
	.loc	1 64 5                          ; micro-dot.ttgir:64:5
	v_add_u32_e64 v10, s9, v9
	ds_write_b32 v10, v6
	s_mov_b32 s6, 0x108
	.loc	1 65 5                          ; micro-dot.ttgir:65:5
	s_cselect_b32 s12, s1, s6
	v_mov_b32_e32 v10, s13
	v_bitop3_b32 v4, v4, s12, v10 bitop3:0x6c
	s_mov_b32 s12, 0x800
	.loc	1 72 14                         ; micro-dot.ttgir:72:14
	s_add_i32 s14, s9, s12
	.loc	1 65 5                          ; micro-dot.ttgir:65:5
	v_add_u32_e64 v10, s14, v4
	ds_write_b32 v10, v6
	.loc	1 69 5                          ; micro-dot.ttgir:69:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s12, 0x2400
	s_add_i32 s12, s9, s12
	v_add_u32_e64 v10, s12, v9
	ds_write_b32 v10, v6
	s_mov_b32 s12, 0x400
	.loc	1 70 5                          ; micro-dot.ttgir:70:5
	s_add_i32 s12, s9, s12
	v_add_u32_e64 v9, s12, v9
	ds_write_b32 v9, v6
	s_mov_b32 s12, 0xc00
	.loc	1 71 5                          ; micro-dot.ttgir:71:5
	s_add_i32 s12, s9, s12
	v_add_u32_e64 v4, s12, v4
	ds_write_b32 v4, v6
	.loc	1 72 14                         ; micro-dot.ttgir:72:14
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e64 v4, s3, v7
	v_cmp_eq_u32_e64 s[12:13], v5, s1
	v_mov_b32_e32 v5, s6
	v_mov_b32_e32 v6, s1
	v_cndmask_b32_e64 v5, v5, v6, s[12:13]
	v_xad_u32 v5, v5, v4, s14
	ds_read_b64_tr_b16 v[12:13], v5
	ds_read_b64_tr_b16 v[14:15], v5 offset:512
	s_mov_b32 s14, 5
	.loc	1 75 18                         ; micro-dot.ttgir:75:18
	s_lshr_b32 s14, s7, s14
	s_mov_b32 s7, 4
	s_and_b32 s14, s14, s7
	s_lshl2_add_u32 s11, s11, s9
	s_add_i32 s11, s11, s14
	v_add_u32_e64 v10, s11, v4
	ds_write_b32 v10, v11
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s11, 15
	v_and_b32_e64 v4, v0, s11
	v_lshl_add_u32 v5, v4, 3, s9
	v_lshl_add_u32 v9, v3, 1, v5
	v_mov_b32_e32 v5, v9
	ds_read_b64 v[34:35], v5
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, v34
	v_mov_b32_e32 v5, v35
	s_mov_b32 s11, 0
	.loc	1 77 12                         ; micro-dot.ttgir:77:12
	v_mul_f32_e64 v6, v6, s11
	v_mul_f32_e64 v5, v5, s11
	.loc	1 80 14                         ; micro-dot.ttgir:80:14
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e64 v18, s2, v7
	s_mov_b32 s11, 0x420
	v_mov_b32_e32 v7, s11
	v_mov_b32_e32 v33, s1
	v_cndmask_b32_e64 v7, v7, v33, s[12:13]
	v_bitop3_b32 v7, v3, v7, v18 bitop3:0x36
	v_add_u32_e64 v18, s9, v7
                                        ; implicit-def: $sgpr11
	v_cvt_pk_bf16_f32 v32, v32, s11
	ds_write_b16 v18, v32
                                        ; implicit-def: $sgpr11
	v_cvt_pk_bf16_f32 v31, v31, s11
	ds_write_b16 v18, v31 offset:4096
	v_xor_b32_e64 v18, v7, s6
	v_add_u32_e64 v18, s9, v18
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v30, v30, s6
	ds_write_b16 v18, v30
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v29, v29, s6
	ds_write_b16 v18, v29 offset:4096
	s_mov_b32 s6, 0x210
	v_xor_b32_e64 v18, v7, s6
	v_add_u32_e64 v18, s9, v18
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v28, v28, s6
	ds_write_b16 v18, v28
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v27, v27, s6
	ds_write_b16 v18, v27 offset:4096
	s_mov_b32 s6, 0x318
	v_xor_b32_e64 v18, v7, s6
	v_add_u32_e64 v18, s9, v18
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v26, v26, s6
	ds_write_b16 v18, v26
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v25, v25, s6
	ds_write_b16 v18, v25 offset:4096
	s_mov_b32 s6, 0x840
	v_xor_b32_e64 v18, v7, s6
	v_add_u32_e64 v18, s9, v18
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v24, v24, s6
	ds_write_b16 v18, v24
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v23, v23, s6
	ds_write_b16 v18, v23 offset:4096
	s_mov_b32 s6, 0x948
	v_xor_b32_e64 v18, v7, s6
	v_add_u32_e64 v18, s9, v18
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v22, v22, s6
	ds_write_b16 v18, v22
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v21, v21, s6
	ds_write_b16 v18, v21 offset:4096
	s_mov_b32 s6, 0xa50
	v_xor_b32_e64 v18, v7, s6
	v_add_u32_e64 v18, s9, v18
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v20, v20, s6
	ds_write_b16 v18, v20
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v19, v19, s6
	ds_write_b16 v18, v19 offset:4096
	s_mov_b32 s6, 0xb58
	v_xor_b32_e64 v7, v7, s6
	v_add_u32_e64 v7, s9, v7
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v17, v17, s6
	ds_write_b16 v7, v17
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v16, v16, s6
	ds_write_b16 v7, v16 offset:4096
	.loc	1 81 14                         ; micro-dot.ttgir:81:14
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s6, 60
	v_and_b32_e64 v16, v0, s6
	s_mov_b32 s6, 6
	v_lshlrev_b32_e64 v7, s6, v16
	v_lshlrev_b32_e64 v17, s3, v0
	v_and_b32_e64 v17, v17, s10
	v_lshlrev_b32_e64 v16, s2, v16
	v_bitop3_b32 v7, v7, v16, v17 bitop3:0x36
	v_xad_u32 v7, v7, v8, s9
	ds_read_b64_tr_b16 v[20:21], v7
	ds_read_b64_tr_b16 v[22:23], v7 offset:4096
	ds_read_b64_tr_b16 v[16:17], v7 offset:128
	ds_read_b64_tr_b16 v[18:19], v7 offset:4224
	.loc	1 83 12                         ; micro-dot.ttgir:83:12
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v7, v23
	v_mov_b32_e32 v8, v22
	v_mov_b32_e32 v24, v21
                                        ; kill: def $vgpr20 killed $vgpr20 killed $vgpr20_vgpr21 killed $exec
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21_vgpr22_vgpr23 killed $exec
	v_mov_b32_e32 v21, v24
	v_mov_b32_e32 v22, v8
	v_mov_b32_e32 v23, v7
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v7, v19
	v_mov_b32_e32 v8, v18
	v_mov_b32_e32 v24, v17
                                        ; kill: def $vgpr16 killed $vgpr16 killed $vgpr16_vgpr17 killed $exec
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17_vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v17, v24
	v_mov_b32_e32 v18, v8
	v_mov_b32_e32 v19, v7
	v_mov_b32_e32 v7, v15
	v_mov_b32_e32 v8, v14
	v_mov_b32_e32 v24, v13
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13_vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v13, v24
	v_mov_b32_e32 v14, v8
	v_mov_b32_e32 v15, v7
	v_mov_b32_e32 v24, v6
	v_mov_b32_e32 v25, v6
	v_mov_b32_e32 v26, v6
	v_mov_b32_e32 v27, v6
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[22:25], v[12:15], v[20:23], v[24:27]
	s_nop 7
	v_mov_b32_e32 v7, v25
	v_mov_b32_e32 v8, v24
	v_mov_b32_e32 v26, v5
	v_mov_b32_e32 v27, v5
	v_mov_b32_e32 v28, v5
	v_mov_b32_e32 v29, v5
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[18:21], v[12:15], v[16:19], v[26:29]
	s_nop 7
	v_mov_b32_e32 v5, v21
	v_mov_b32_e32 v6, v20
	.loc	1 85 16                         ; micro-dot.ttgir:85:16
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v10, v11
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v12, v9
	ds_read_b64 v[16:17], v12
	.loc	1 88 12                         ; micro-dot.ttgir:88:12
	v_add_f32_e64 v12, v11, v11
	s_mov_b32 s9, 1.0
	.loc	1 90 12                         ; micro-dot.ttgir:90:12
	v_div_scale_f32 v11, s[10:11], v12, v12, s9
	v_rcp_f32_e64 v13, v11
	s_nop 0
	v_fma_f32 v14, -v11, v13, s9
	v_fmac_f32_e64 v13, v14, v13
	v_div_scale_f32 v15, vcc, s9, v12, s9
	v_mul_f32_e64 v14, v15, v13
	v_fma_f32 v26, -v11, v14, v15
	v_fmac_f32_e64 v14, v26, v13
	v_fma_f32 v11, -v11, v14, v15
	v_div_fmas_f32 v11, v11, v13, v14
	v_div_fixup_f32 v11, v11, v12, s9
	.loc	1 91 14                         ; micro-dot.ttgir:91:14
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v10, v11
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v9
	.loc	1 87 15                         ; micro-dot.ttgir:87:15
	v_mov_b64_e32 v[10:11], v[22:23]
	v_pk_mul_f32 v[10:11], v[10:11], v[16:17] op_sel_hi:[1,0]
	.loc	1 93 19                         ; micro-dot.ttgir:93:19
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[10:11], v[10:11], v[12:13] op_sel_hi:[1,0]
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	.loc	1 87 15                         ; micro-dot.ttgir:87:15
	v_mov_b32_e32 v9, v7
	v_pk_mul_f32 v[8:9], v[8:9], v[16:17] op_sel_hi:[1,0]
	.loc	1 93 19                         ; micro-dot.ttgir:93:19
	s_nop 0
	v_pk_mul_f32 v[14:15], v[8:9], v[12:13] op_sel_hi:[1,0]
	.loc	1 87 15                         ; micro-dot.ttgir:87:15
	v_mov_b64_e32 v[8:9], v[18:19]
	v_pk_mul_f32 v[8:9], v[8:9], v[16:17] op_sel:[0,1]
	.loc	1 93 19                         ; micro-dot.ttgir:93:19
	s_nop 0
	v_pk_mul_f32 v[8:9], v[8:9], v[12:13] op_sel:[0,1]
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	.loc	1 87 15                         ; micro-dot.ttgir:87:15
	v_mov_b32_e32 v7, v5
	v_pk_mul_f32 v[6:7], v[6:7], v[16:17] op_sel:[0,1]
	.loc	1 93 19                         ; micro-dot.ttgir:93:19
	s_nop 0
	v_pk_mul_f32 v[12:13], v[6:7], v[12:13] op_sel:[0,1]
	.loc	1 94 17                         ; micro-dot.ttgir:94:17
	v_mov_b32_e32 v6, v11
	v_mov_b32_e32 v5, v10
	v_cvt_pk_bf16_f32 v10, v5, v6
	v_mov_b32_e32 v6, v15
	v_mov_b32_e32 v5, v14
	v_cvt_pk_bf16_f32 v7, v5, v6
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v5, v8
	v_cvt_pk_bf16_f32 v6, v5, v6
	v_mov_b32_e32 v8, v13
	v_mov_b32_e32 v5, v12
	v_cvt_pk_bf16_f32 v8, v5, v8
	.loc	1 95 15                         ; micro-dot.ttgir:95:15
	ds_read_b32 v2, v2
	.loc	1 97 10                         ; micro-dot.ttgir:97:10
	v_lshrrev_b32_e64 v3, s8, v3
	v_or_b32_e64 v3, v3, v4
	.loc	1 98 10                         ; micro-dot.ttgir:98:10
	v_lshrrev_b32_e64 v4, s8, v0
	s_mov_b32 s8, 12
	v_and_b32_e64 v4, v4, s8
	.loc	1 107 14                        ; micro-dot.ttgir:107:14
	v_lshl_or_b32 v4, v3, s7, v4
	v_mov_b32_e32 v3, 0
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	.loc	1 110 14                        ; micro-dot.ttgir:110:14
	v_mov_b32_e32 v5, v3
	v_mov_b64_e32 v[12:13], s[4:5]
	v_lshl_add_u64 v[4:5], v[4:5], s2, v[12:13]
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	.loc	1 111 5                         ; micro-dot.ttgir:111:5
	v_mov_b32_e32 v11, v7
	global_store_dwordx2 v[4:5], v[10:11], off
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v8
	global_store_dwordx2 v[4:5], v[6:7], off offset:2048
	.loc	1 113 11                        ; micro-dot.ttgir:113:11
	v_lshlrev_b32_e64 v0, s6, v0
	.loc	1 114 11                        ; micro-dot.ttgir:114:11
	v_lshrrev_b32_e64 v1, s3, v1
	s_mov_b32 s3, 0x1c0
	.loc	1 123 15                        ; micro-dot.ttgir:123:15
	v_and_or_b32 v0, v0, s3, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	.loc	1 128 16                        ; micro-dot.ttgir:128:16
	v_mov_b32_e32 v1, v3
	v_mov_b64_e32 v[4:5], s[4:5]
	v_lshl_add_u64 v[4:5], v[0:1], s2, v[4:5]
	.loc	1 129 5                         ; micro-dot.ttgir:129:5
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v0, v4
	s_mov_b32 s2, 0x1000
	v_add_co_u32_e64 v0, s[2:3], s2, v0
	v_mov_b32_e32 v4, v5
	v_mov_b32_e32 v1, s1
	v_addc_co_u32_e64 v4, s[2:3], v1, v4, s[2:3]
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	global_store_short v[0:1], v3, off
	v_lshrrev_b32_e64 v2, s0, v2
	global_store_short v[0:1], v2, off offset:64
	.loc	1 130 5                         ; micro-dot.ttgir:130:5
	s_endpgm
.Ltmp2:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel repro
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 10
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 8
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 48
		.amdhsa_next_free_sgpr 24
		.amdhsa_accum_offset 48
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	repro, .Lfunc_end0-repro
	.cfi_endproc
                                        ; -- End function
	.set repro.num_vgpr, 48
	.set repro.num_agpr, 0
	.set repro.numbered_sgpr, 24
	.set repro.num_named_barrier, 0
	.set repro.private_seg_size, 0
	.set repro.uses_vcc, 1
	.set repro.uses_flat_scratch, 0
	.set repro.has_dyn_sized_stack, 0
	.set repro.has_recursion, 0
	.set repro.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4464
; TotalNumSgprs: 30
; NumVgprs: 48
; NumAgprs: 0
; TotalNumVgprs: 48
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 48
; AccumOffset: 48
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 10
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 11
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	0                               ; DW_CHILDREN_no
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"micro-dot.ttgir"               ; string offset=7 ; micro-dot.ttgir
.Linfo_string2:
	.asciz	"/root/scripts/triton/reproducers/amdgpu_ds_read_b64_waitcnt/ir" ; string offset=23 ; /root/scripts/triton/reproducers/amdgpu_ds_read_b64_waitcnt/ir
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name:           repro
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         repro.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     48
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
