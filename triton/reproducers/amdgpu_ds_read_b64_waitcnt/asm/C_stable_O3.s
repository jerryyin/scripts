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
	.file	1 "ir" "micro-dot.ttgir"
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
.Ltmp1:
	.loc	1 114 11 is_stmt 1              ; micro-dot.ttgir:114:11
	v_readfirstlane_b32 s5, v0
	.loc	1 129 5                         ; micro-dot.ttgir:129:5
	v_and_b32_e32 v1, 32, v0
	.loc	1 29 11                         ; micro-dot.ttgir:29:11
	v_and_b32_e32 v28, 31, v0
	.loc	1 129 5                         ; micro-dot.ttgir:129:5
	v_mov_b32_e32 v2, s5
	v_and_b32_e32 v26, 0xc0, v2
	.loc	1 29 11                         ; micro-dot.ttgir:29:11
	v_lshrrev_b32_e32 v18, 3, v1
	v_lshrrev_b32_e32 v29, 1, v26
	v_or_b32_e32 v2, 10, v18
	v_or_b32_e32 v5, v29, v28
	.loc	1 33 12                         ; micro-dot.ttgir:33:12
	v_cvt_f32_u32_e32 v3, v2
	v_cvt_f32_u32_e32 v2, v5
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_or_b32_e32 v10, 11, v18
	v_or_b32_e32 v8, 9, v18
	v_or_b32_e32 v11, 8, v18
	v_or_b32_e32 v12, 16, v18
	v_or_b32_e32 v13, 27, v18
	.loc	1 34 12                         ; micro-dot.ttgir:34:12
	v_cvt_f32_ubyte0_e32 v10, v10
	s_mov_b32 s1, 0x3d000000
	s_mov_b32 s0, 0x3b800000
	v_cvt_f32_ubyte0_e32 v9, v8
	v_cvt_f32_ubyte0_e32 v8, v11
	v_cvt_f32_ubyte0_e32 v14, v12
	v_cvt_f32_ubyte0_e32 v11, v13
	.loc	1 37 17                         ; micro-dot.ttgir:37:17
	v_pk_mul_f32 v[12:13], v[2:3], s[0:1]
	.loc	1 38 17                         ; micro-dot.ttgir:38:17
	v_mov_b32_e32 v3, v10
	v_pk_mul_f32 v[2:3], v[2:3], s[0:1]
	.loc	1 27 20                         ; micro-dot.ttgir:27:20
	v_mov_b32_e32 v10, s4
	s_mov_b32 s0, 0x3fb8aa3b
	.loc	1 38 17                         ; micro-dot.ttgir:38:17
	v_mul_f32_e32 v14, 0x3d000000, v14
	.loc	1 27 20                         ; micro-dot.ttgir:27:20
	v_pk_mul_f32 v[10:11], v[10:11], s[0:1]
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_or_b32_e32 v4, 3, v18
	v_or_b32_e32 v6, 2, v18
	.loc	1 39 16                         ; micro-dot.ttgir:39:16
	v_add_f32_e32 v14, v14, v2
	v_add_f32_e32 v19, v11, v2
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_or_b32_e32 v7, 1, v18
	.loc	1 34 12                         ; micro-dot.ttgir:34:12
	v_cvt_f32_u32_e32 v5, v4
	v_cvt_f32_u32_e32 v4, v6
	v_cvt_f32_ubyte0_e32 v6, v18
	.loc	1 40 15                         ; micro-dot.ttgir:40:15
	v_add_f32_e32 v20, v10, v14
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_or_b32_e32 v16, 19, v18
	v_or_b32_e32 v17, 24, v18
	v_or_b32_e32 v14, 17, v18
	v_or_b32_e32 v15, 18, v18
	.loc	1 40 15                         ; micro-dot.ttgir:40:15
	v_add_f32_e32 v21, v10, v19
	.loc	1 30 11                         ; micro-dot.ttgir:30:11
	v_or_b32_e32 v19, 26, v18
	v_or_b32_e32 v18, 25, v18
	.loc	1 34 12                         ; micro-dot.ttgir:34:12
	v_cvt_f32_u32_e32 v19, v19
	v_cvt_f32_u32_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v7, v7
	.loc	1 38 17                         ; micro-dot.ttgir:38:17
	s_mov_b32 s6, s1
	.loc	1 34 12                         ; micro-dot.ttgir:34:12
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v16, v16
	.loc	1 38 17                         ; micro-dot.ttgir:38:17
	v_pk_mul_f32 v[4:5], v[4:5], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], s[6:7] op_sel_hi:[1,0]
	.loc	1 39 16                         ; micro-dot.ttgir:39:16
	v_pk_add_f32 v[6:7], v[6:7], v[12:13] op_sel_hi:[1,0]
	v_pk_add_f32 v[4:5], v[4:5], v[12:13] op_sel_hi:[1,0]
	v_pk_add_f32 v[8:9], v[8:9], v[12:13] op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[2:3], v[12:13] op_sel:[0,1] op_sel_hi:[1,0]
	v_pk_add_f32 v[14:15], v[14:15], v[2:3] op_sel_hi:[1,0]
	v_pk_add_f32 v[16:17], v[16:17], v[2:3] op_sel_hi:[1,0]
	v_pk_add_f32 v[2:3], v[18:19], v[2:3] op_sel_hi:[1,0]
	.loc	1 40 15                         ; micro-dot.ttgir:40:15
	v_pk_add_f32 v[8:9], v[10:11], v[8:9] op_sel_hi:[0,1]
	v_pk_add_f32 v[4:5], v[10:11], v[4:5] op_sel_hi:[0,1]
	v_pk_add_f32 v[6:7], v[10:11], v[6:7] op_sel_hi:[0,1]
	v_pk_add_f32 v[16:17], v[10:11], v[16:17] op_sel_hi:[0,1]
	v_pk_add_f32 v[14:15], v[10:11], v[14:15] op_sel_hi:[0,1]
	v_pk_add_f32 v[2:3], v[10:11], v[2:3] op_sel_hi:[0,1]
	v_pk_add_f32 v[12:13], v[10:11], v[12:13] op_sel_hi:[0,1]
	.loc	1 43 13                         ; micro-dot.ttgir:43:13
	v_max_f32_e32 v22, v6, v7
	v_max3_f32 v23, v5, v8, v9
	v_max3_f32 v25, v14, v15, v16
	v_max3_f32 v10, v17, v2, v3
	v_max3_f32 v24, v12, v13, v20
	v_max3_f32 v22, v22, v4, v23
	v_max3_f32 v10, v25, v10, v21
	v_max3_f32 v10, v22, v24, v10
	.loc	1 41 17                         ; micro-dot.ttgir:41:17
	v_mov_b32_e32 v11, v10
	s_nop 1
	v_permlane32_swap_b32_e32 v10, v11
	s_mov_b32 s0, 0xff800000
	.loc	1 46 16                         ; micro-dot.ttgir:46:16
	v_max3_f32 v10, v10, v11, s0
	.loc	1 49 17                         ; micro-dot.ttgir:49:17
	v_pk_add_f32 v[6:7], v[6:7], v[10:11] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[10:11] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9], v[8:9], v[10:11] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[12:13], v[10:11] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e32 v18, v20, v10
	v_pk_add_f32 v[14:15], v[14:15], v[10:11] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17], v[16:17], v[10:11] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e32 v21, v21, v10
	.loc	1 50 14                         ; micro-dot.ttgir:50:14
	v_exp_f32_e32 v10, v6
	v_exp_f32_e32 v11, v7
	v_exp_f32_e32 v4, v4
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v8, v8
	v_exp_f32_e32 v9, v9
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	v_exp_f32_e32 v18, v18
	v_exp_f32_e32 v19, v14
	v_exp_f32_e32 v14, v15
	v_exp_f32_e32 v15, v16
	v_exp_f32_e32 v16, v17
	v_exp_f32_e32 v17, v2
	v_exp_f32_e32 v20, v3
	v_exp_f32_e32 v21, v21
	.loc	1 51 16                         ; micro-dot.ttgir:51:16
	v_pk_add_f32 v[2:3], v[10:11], v[4:5]
	v_pk_add_f32 v[6:7], v[8:9], v[12:13]
	v_pk_add_f32 v[22:23], v[18:19], v[14:15]
	v_pk_add_f32 v[24:25], v[16:17], v[20:21]
	v_pk_add_f32 v[2:3], v[2:3], v[6:7]
	v_pk_add_f32 v[6:7], v[22:23], v[24:25]
	.loc	1 29 11                         ; micro-dot.ttgir:29:11
	v_and_or_b32 v27, v0, 63, v26
	.loc	1 51 16                         ; micro-dot.ttgir:51:16
	v_pk_add_f32 v[2:3], v[2:3], v[6:7]
	.loc	1 63 5                          ; micro-dot.ttgir:63:5
	s_bfe_i32 s1, s5, 0x10006
	.loc	1 53 14                         ; micro-dot.ttgir:53:14
	v_pk_add_f32 v[2:3], v[2:3], v[2:3] op_sel:[0,1] op_sel_hi:[1,0]
	.loc	1 63 5                          ; micro-dot.ttgir:63:5
	s_and_b32 s4, s1, 0x110
	.loc	1 51 16                         ; micro-dot.ttgir:51:16
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v2, v3
	.loc	1 53 14                         ; micro-dot.ttgir:53:14
	v_add_f32_e32 v22, v2, v3
	.loc	1 63 5                          ; micro-dot.ttgir:63:5
	v_lshlrev_b32_e32 v2, 2, v27
	v_mov_b32_e32 v3, 0x2fc
	.loc	1 65 5                          ; micro-dot.ttgir:65:5
	s_and_b32 s1, s1, 0x108
	.loc	1 63 5                          ; micro-dot.ttgir:63:5
	v_bitop3_b32 v6, v2, s4, v3 bitop3:0x6c
	.loc	1 65 5                          ; micro-dot.ttgir:65:5
	v_bitop3_b32 v3, v2, s1, v3 bitop3:0x6c
	.loc	1 63 5                          ; micro-dot.ttgir:63:5
	v_add_u32_e32 v23, 0, v6
	v_mov_b32_e32 v6, 0x3f803f80
	.loc	1 65 5                          ; micro-dot.ttgir:65:5
	v_add_u32_e32 v3, 0, v3
	.loc	1 0 0 is_stmt 0                 ; micro-dot.ttgir:0
	ds_write2st64_b32 v23, v6, v6 offset1:32
	.loc	1 65 5                          ; micro-dot.ttgir:65:5
	ds_write_b32 v3, v6 offset:2048
	.loc	1 69 5 is_stmt 1                ; micro-dot.ttgir:69:5
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 0 0 is_stmt 0                 ; micro-dot.ttgir:0
	ds_write2st64_b32 v23, v6, v6 offset0:4 offset1:36
	.loc	1 71 5 is_stmt 1                ; micro-dot.ttgir:71:5
	ds_write_b32 v3, v6 offset:3072
	.loc	1 72 14                         ; micro-dot.ttgir:72:14
	v_mov_b32_e32 v6, 0x108
	v_cmp_eq_u32_e32 vcc, 0, v1
	v_lshlrev_b32_e32 v3, 3, v28
	.loc	1 75 18                         ; micro-dot.ttgir:75:18
	s_and_b32 s1, s5, 0x80
	.loc	1 72 14                         ; micro-dot.ttgir:72:14
	v_cndmask_b32_e64 v1, v6, 0, vcc
	.loc	1 63 5                          ; micro-dot.ttgir:63:5
	s_and_b32 s0, s5, 64
	.loc	1 72 14                         ; micro-dot.ttgir:72:14
	v_xad_u32 v1, v1, v3, 0
	.loc	1 75 18                         ; micro-dot.ttgir:75:18
	s_add_i32 s4, s1, 0
	s_lshl1_add_u32 s1, s1, 0
	.loc	1 72 14                         ; micro-dot.ttgir:72:14
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[30:31], v1 offset:2048
	ds_read_b64_tr_b16 v[32:33], v1 offset:2560
	.loc	1 75 18                         ; micro-dot.ttgir:75:18
	v_and_b32_e32 v1, 0x17c, v2
	v_and_b32_e32 v24, 15, v0
	s_add_i32 s0, s0, s1
	v_add_u32_e32 v1, s4, v1
	v_lshl_add_u32 v25, v24, 2, s0
	ds_write_b32 v1, v22
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[2:3], v25 offset1:32
	.loc	1 80 14                         ; micro-dot.ttgir:80:14
	v_mov_b32_e32 v7, 0x420
	v_cndmask_b32_e64 v7, v7, 0, vcc
	v_cvt_pk_bf16_f32 v10, v10, s0
	s_waitcnt lgkmcnt(0)
	.loc	1 77 12                         ; micro-dot.ttgir:77:12
	v_mul_f32_e32 v6, 0, v3
	.loc	1 80 14                         ; micro-dot.ttgir:80:14
	v_lshlrev_b32_e32 v3, 1, v28
	v_bitop3_b32 v3, v26, v7, v3 bitop3:0x36
	v_add_u32_e32 v7, 0, v3
	s_barrier
	ds_write_b16 v7, v10
	v_cvt_pk_bf16_f32 v10, v18, s0
	ds_write_b16 v7, v10 offset:4096
	v_xor_b32_e32 v7, 0x108, v3
	v_add_u32_e32 v7, 0, v7
	v_cvt_pk_bf16_f32 v10, v11, s0
	ds_write_b16 v7, v10
	v_cvt_pk_bf16_f32 v10, v19, s0
	ds_write_b16 v7, v10 offset:4096
	v_xor_b32_e32 v7, 0x210, v3
	v_add_u32_e32 v7, 0, v7
	v_cvt_pk_bf16_f32 v4, v4, s0
	ds_write_b16 v7, v4
	v_cvt_pk_bf16_f32 v4, v14, s0
	ds_write_b16 v7, v4 offset:4096
	v_xor_b32_e32 v4, 0x318, v3
	v_add_u32_e32 v4, 0, v4
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5
	v_cvt_pk_bf16_f32 v5, v15, s0
	ds_write_b16 v4, v5 offset:4096
	v_xor_b32_e32 v4, 0x840, v3
	v_add_u32_e32 v4, 0, v4
	v_cvt_pk_bf16_f32 v5, v8, s0
	ds_write_b16 v4, v5
	v_cvt_pk_bf16_f32 v5, v16, s0
	ds_write_b16 v4, v5 offset:4096
	v_xor_b32_e32 v4, 0x948, v3
	v_add_u32_e32 v4, 0, v4
	v_cvt_pk_bf16_f32 v5, v9, s0
	ds_write_b16 v4, v5
	v_cvt_pk_bf16_f32 v5, v17, s0
	ds_write_b16 v4, v5 offset:4096
	v_xor_b32_e32 v4, 0xa50, v3
	v_add_u32_e32 v4, 0, v4
	v_cvt_pk_bf16_f32 v5, v12, s0
	ds_write_b16 v4, v5
	v_cvt_pk_bf16_f32 v5, v20, s0
	v_xor_b32_e32 v3, 0xb58, v3
	ds_write_b16 v4, v5 offset:4096
	v_add_u32_e32 v3, 0, v3
	v_cvt_pk_bf16_f32 v4, v13, s0
	ds_write_b16 v3, v4
	v_cvt_pk_bf16_f32 v4, v21, s0
	ds_write_b16 v3, v4 offset:4096
	.loc	1 81 14                         ; micro-dot.ttgir:81:14
	v_and_b32_e32 v3, 60, v0
	v_lshlrev_b32_e32 v5, 3, v0
	v_lshlrev_b32_e32 v4, 6, v3
	v_and_b32_e32 v5, 24, v5
	v_lshlrev_b32_e32 v3, 1, v3
	v_bitop3_b32 v3, v4, v3, v5 bitop3:0x36
	v_xad_u32 v3, v3, v29, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[8:9], v3
	ds_read_b64_tr_b16 v[10:11], v3 offset:4096
	ds_read_b64_tr_b16 v[18:19], v3 offset:4224
	ds_read_b64_tr_b16 v[16:17], v3 offset:128
	.loc	1 77 12                         ; micro-dot.ttgir:77:12
	v_mul_f32_e32 v2, 0, v2
	.loc	1 83 12                         ; micro-dot.ttgir:83:12
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	.loc	1 85 16                         ; micro-dot.ttgir:85:16
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 83 12                         ; micro-dot.ttgir:83:12
	v_mfma_f32_16x16x32_bf16 v[2:5], v[30:33], v[8:11], v[2:5]
	.loc	1 88 12                         ; micro-dot.ttgir:88:12
	v_add_f32_e32 v8, v22, v22
	.loc	1 90 12                         ; micro-dot.ttgir:90:12
	v_div_scale_f32 v9, s[0:1], v8, v8, 1.0
	v_rcp_f32_e32 v12, v9
	.loc	1 85 16                         ; micro-dot.ttgir:85:16
	ds_write_b32 v1, v22
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 90 12                         ; micro-dot.ttgir:90:12
	v_fma_f32 v13, -v9, v12, 1.0
	v_fmac_f32_e32 v12, v13, v12
	v_div_scale_f32 v13, vcc, 1.0, v8, 1.0
	v_mul_f32_e32 v14, v13, v12
	v_fma_f32 v15, -v9, v14, v13
	v_fmac_f32_e32 v14, v15, v12
	v_fma_f32 v9, -v9, v14, v13
	v_div_fmas_f32 v9, v9, v12, v14
	.loc	1 85 16                         ; micro-dot.ttgir:85:16
	ds_read2_b32 v[10:11], v25 offset1:32
	.loc	1 90 12                         ; micro-dot.ttgir:90:12
	v_div_fixup_f32 v8, v9, v8, 1.0
	.loc	1 83 12                         ; micro-dot.ttgir:83:12
	v_mov_b32_e32 v7, v6
	.loc	1 91 14                         ; micro-dot.ttgir:91:14
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v1, v8
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[12:13], v25 offset1:32
	.loc	1 83 12                         ; micro-dot.ttgir:83:12
	v_mov_b32_e32 v8, v6
	v_mov_b32_e32 v9, v6
	.loc	1 87 15                         ; micro-dot.ttgir:87:15
	v_pk_mul_f32 v[2:3], v[2:3], v[10:11] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[10:11] op_sel_hi:[1,0]
	.loc	1 83 12                         ; micro-dot.ttgir:83:12
	v_mfma_f32_16x16x32_bf16 v[6:9], v[30:33], v[16:19], v[6:9]
	.loc	1 87 15                         ; micro-dot.ttgir:87:15
	v_mov_b32_e32 v10, v11
	.loc	1 93 19                         ; micro-dot.ttgir:93:19
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[2:3], v[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[12:13] op_sel_hi:[1,0]
	v_mov_b32_e32 v12, v13
	.loc	1 94 17                         ; micro-dot.ttgir:94:17
	v_cvt_pk_bf16_f32 v2, v2, v3
	.loc	1 87 15                         ; micro-dot.ttgir:87:15
	s_nop 1
	v_pk_mul_f32 v[6:7], v[6:7], v[10:11] op_sel_hi:[1,0]
	.loc	1 94 17                         ; micro-dot.ttgir:94:17
	v_cvt_pk_bf16_f32 v3, v4, v5
	.loc	1 93 19                         ; micro-dot.ttgir:93:19
	v_pk_mul_f32 v[6:7], v[6:7], v[12:13] op_sel_hi:[1,0]
	.loc	1 97 10                         ; micro-dot.ttgir:97:10
	v_lshrrev_b32_e32 v1, 2, v26
	.loc	1 94 17                         ; micro-dot.ttgir:94:17
	v_cvt_pk_bf16_f32 v4, v6, v7
	.loc	1 110 14                        ; micro-dot.ttgir:110:14
	v_lshrrev_b32_e32 v6, 1, v0
	.loc	1 87 15                         ; micro-dot.ttgir:87:15
	v_pk_mul_f32 v[8:9], v[8:9], v[10:11] op_sel_hi:[1,0]
	.loc	1 97 10                         ; micro-dot.ttgir:97:10
	v_or_b32_e32 v1, v1, v24
	.loc	1 110 14                        ; micro-dot.ttgir:110:14
	v_and_b32_e32 v6, 24, v6
	.loc	1 93 19                         ; micro-dot.ttgir:93:19
	v_pk_mul_f32 v[8:9], v[8:9], v[12:13] op_sel_hi:[1,0]
	.loc	1 110 14                        ; micro-dot.ttgir:110:14
	v_lshl_or_b32 v1, v1, 5, v6
	.loc	1 94 17                         ; micro-dot.ttgir:94:17
	v_cvt_pk_bf16_f32 v5, v8, v9
	.loc	1 111 5                         ; micro-dot.ttgir:111:5
	global_store_dwordx2 v1, v[2:3], s[2:3]
	global_store_dwordx2 v1, v[4:5], s[2:3] offset:2048
	.loc	1 113 11                        ; micro-dot.ttgir:113:11
	v_lshlrev_b32_e32 v0, 6, v0
	.loc	1 114 11                        ; micro-dot.ttgir:114:11
	v_lshrrev_b32_e32 v1, 3, v27
	s_movk_i32 s0, 0x1c0
	.loc	1 123 15                        ; micro-dot.ttgir:123:15
	v_and_or_b32 v0, v0, s0, v1
	.loc	1 129 5                         ; micro-dot.ttgir:129:5
	ds_read_b32 v2, v23 offset:8192
	.loc	1 128 16                        ; micro-dot.ttgir:128:16
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	.loc	1 129 5                         ; micro-dot.ttgir:129:5
	v_add_co_u32_e32 v0, vcc, 0x1000, v0
	s_nop 1
	v_addc_co_u32_e32 v1, vcc, 0, v1, vcc
	s_waitcnt lgkmcnt(0)
	global_store_short v[0:1], v2, off
	global_store_short_d16_hi v[0:1], v2, off offset:64
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
		.amdhsa_next_free_vgpr 34
		.amdhsa_next_free_sgpr 10
		.amdhsa_accum_offset 36
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
	.set repro.num_vgpr, 34
	.set repro.num_agpr, 0
	.set repro.numbered_sgpr, 10
	.set repro.num_named_barrier, 0
	.set repro.private_seg_size, 0
	.set repro.uses_vcc, 1
	.set repro.uses_flat_scratch, 0
	.set repro.has_dyn_sized_stack, 0
	.set repro.has_recursion, 0
	.set repro.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2084
; TotalNumSgprs: 16
; NumVgprs: 34
; NumAgprs: 0
; TotalNumVgprs: 34
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 34
; AccumOffset: 36
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 10
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 8
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
	.asciz	"ir"                            ; string offset=23 ; ir
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
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         repro.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     34
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
