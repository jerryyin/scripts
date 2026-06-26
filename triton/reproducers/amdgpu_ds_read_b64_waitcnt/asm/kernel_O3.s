	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0 ; -- Begin function _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
	.p2align	8
	.type	_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0,@function
_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0: ; @_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.28:
	.file	1 "/root/aiter/aiter/ops/triton/_triton_kernels/attention" "mha.py"
	.loc	1 297 0 prologue_end            ; mha.py:297:0
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.29:
.LBB0_0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_dword s46, s[4:5], 0x84
	s_load_dwordx2 s[6:7], s[4:5], 0x94
	s_mov_b64 s[36:37], s[10:11]
	s_load_dwordx8 s[20:27], s[4:5], 0x20
	s_load_dword s30, s[4:5], 0x48
	v_and_b32_e32 v54, 0x3ff, v0
	s_mov_b64 s[28:29], s[14:15]
.Ltmp1:
	.loc	1 367 19 is_stmt 1              ; mha.py:367:19
	s_waitcnt lgkmcnt(0)
	s_add_i32 s0, s6, 0x7f
	.loc	1 367 18 is_stmt 0              ; mha.py:367:18
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 25
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 7
	.loc	1 376 15 is_stmt 1              ; mha.py:376:15
	s_abs_i32 s11, s0
	v_cvt_f32_u32_e32 v1, s11
	.loc	1 376 16 is_stmt 0              ; mha.py:376:16
	s_ashr_i32 s1, s16, 31
	s_lshr_b32 s1, s1, 27
	s_add_i32 s1, s16, s1
	.loc	1 374 18 is_stmt 1              ; mha.py:374:18
	s_and_b32 s3, s1, 0xffffffe0
	.loc	1 376 15                        ; mha.py:376:15
	v_rcp_f32_e32 v1, v1
	.loc	1 374 18                        ; mha.py:374:18
	s_sub_i32 s3, s16, s3
.Ltmp2:
	.file	2 "/root/aiter/aiter/ops/triton/utils/_triton" "pid_preprocessing.py"
	.loc	2 41 17                         ; pid_preprocessing.py:41:17 @[ mha.py:375:18 ]
	s_bfe_i32 s10, s3, 0x80000
	s_bfe_u32 s10, s10, 0x3000c
	s_add_i32 s10, s3, s10
.Ltmp3:
	.loc	1 376 15                        ; mha.py:376:15
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
.Ltmp4:
	.loc	2 41 17                         ; pid_preprocessing.py:41:17 @[ mha.py:375:18 ]
	s_bfe_i32 s10, s10, 0x80000
.Ltmp5:
	.loc	1 376 15                        ; mha.py:376:15
	v_cvt_u32_f32_e32 v1, v1
.Ltmp6:
	.loc	2 41 17                         ; pid_preprocessing.py:41:17 @[ mha.py:375:18 ]
	s_sext_i32_i16 s10, s10
	s_ashr_i32 s10, s10, 3
	.loc	2 47 15                         ; pid_preprocessing.py:47:15 @[ mha.py:375:18 ]
	s_mulk_i32 s10, 0xffe1
	s_lshl2_add_u32 s41, s3, s10
.Ltmp7:
	.loc	1 376 15                        ; mha.py:376:15
	s_sub_i32 s3, 0, s11
	v_readfirstlane_b32 s10, v1
	s_mul_i32 s3, s3, s10
	.loc	1 376 16 is_stmt 0              ; mha.py:376:16
	s_ashr_i32 s2, s1, 5
	.loc	1 376 15                        ; mha.py:376:15
	s_mul_hi_u32 s3, s10, s3
	s_abs_i32 s2, s2
	s_add_i32 s10, s10, s3
	s_mul_hi_u32 s3, s2, s10
	s_mul_i32 s3, s3, s11
	.loc	1 912 13 is_stmt 1              ; mha.py:912:13
	v_readfirstlane_b32 s40, v54
	.loc	1 376 15                        ; mha.py:376:15
	s_sub_i32 s2, s2, s3
	s_ashr_i32 s15, s27, 31
	s_ashr_i32 s31, s30, 31
	.loc	1 912 13                        ; mha.py:912:13
	s_lshr_b32 s42, s40, 6
	s_bfe_u32 s33, s40, 0x20006
	.loc	1 376 15                        ; mha.py:376:15
	s_ashr_i32 s1, s1, 31
	s_sub_i32 s3, s2, s11
	s_cmp_ge_u32 s2, s11
	s_cselect_b32 s2, s3, s2
	s_sub_i32 s3, s2, s11
	s_cmp_ge_u32 s2, s11
	s_cselect_b32 s2, s3, s2
	.loc	1 377 22                        ; mha.py:377:22
	s_lshl_b32 s3, s0, 5
	.loc	1 377 14 is_stmt 0              ; mha.py:377:14
	s_abs_i32 s3, s3
	v_cvt_f32_u32_e32 v1, s3
	.loc	1 376 15 is_stmt 1              ; mha.py:376:15
	s_xor_b32 s2, s2, s1
	s_sub_i32 s2, s2, s1
	.loc	1 377 14                        ; mha.py:377:14
	s_sub_i32 s1, 0, s3
	v_rcp_f32_e32 v1, v1
	s_xor_b32 s0, s16, s0
	s_ashr_i32 s11, s0, 31
	s_abs_i32 s0, s16
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_load_dword s10, s[4:5], 0x9c
	.loc	1 380 34                        ; mha.py:380:34
	v_and_b32_e32 v59, 63, v0
	.loc	1 382 14                        ; mha.py:382:14
	v_lshlrev_b32_e32 v58, 3, v54
	.loc	1 377 14                        ; mha.py:377:14
	v_readfirstlane_b32 s14, v1
	s_mul_i32 s1, s1, s14
	s_mul_hi_u32 s1, s14, s1
	s_add_i32 s14, s14, s1
	s_mul_hi_u32 s1, s0, s14
	s_mul_i32 s14, s1, s3
	s_sub_i32 s0, s0, s14
	s_add_i32 s14, s1, 1
	s_sub_i32 s16, s0, s3
	s_cmp_ge_u32 s0, s3
	s_cselect_b32 s1, s14, s1
	s_cselect_b32 s0, s16, s0
	s_add_i32 s14, s1, 1
	s_cmp_ge_u32 s0, s3
	s_cselect_b32 s3, s14, s1
	s_load_dwordx2 s[0:1], s[4:5], 0x40
	.loc	1 377 13 is_stmt 0              ; mha.py:377:13
	s_waitcnt lgkmcnt(0)
	s_abs_i32 s10, s10
	v_cvt_f32_u32_e32 v1, s10
	s_sub_i32 s14, 0, s10
	.loc	1 377 14                        ; mha.py:377:14
	s_xor_b32 s3, s3, s11
	s_sub_i32 s3, s3, s11
	.loc	1 377 13                        ; mha.py:377:13
	v_rcp_f32_e32 v1, v1
	s_ashr_i32 s11, s3, 31
	s_abs_i32 s3, s3
	.loc	1 382 14 is_stmt 1              ; mha.py:382:14
	v_lshlrev_b32_e32 v12, 1, v54
	.loc	1 377 13                        ; mha.py:377:13
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	.loc	1 586 11                        ; mha.py:586:11
	v_mov_b32_e32 v11, 0
	.loc	1 380 34                        ; mha.py:380:34
	v_and_b32_e32 v55, 15, v0
	.loc	1 606 11                        ; mha.py:606:11
	v_and_b32_e32 v60, 32, v0
	.loc	1 377 13                        ; mha.py:377:13
	v_readfirstlane_b32 s16, v1
	s_mul_i32 s14, s14, s16
	s_mul_hi_u32 s14, s16, s14
	s_add_i32 s16, s16, s14
	s_mul_hi_u32 s14, s3, s16
	s_mul_i32 s14, s14, s10
	s_sub_i32 s3, s3, s14
	s_sub_i32 s14, s3, s10
	s_cmp_ge_u32 s3, s10
	s_cselect_b32 s3, s14, s3
	s_sub_i32 s14, s3, s10
	s_cmp_ge_u32 s3, s10
	s_cselect_b32 s3, s14, s3
	s_xor_b32 s3, s3, s11
	.loc	1 380 14                        ; mha.py:380:14
	s_lshl_b32 s44, s2, 7
.Ltmp8:
	.loc	1 19 13                         ; mha.py:19:13 @[ mha.py:510:16 ]
	s_add_i32 s2, s7, 31
.Ltmp9:
	.loc	1 377 13                        ; mha.py:377:13
	s_sub_i32 s43, s3, s11
.Ltmp10:
	.loc	1 19 12                         ; mha.py:19:12 @[ mha.py:510:16 ]
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_add_i32 s2, s2, s3
	s_ashr_i32 s48, s2, 5
.Ltmp11:
	.loc	1 563 22                        ; mha.py:563:22
	s_bfe_u32 s2, s41, 0x3001c
	s_add_i32 s2, s41, s2
	s_sext_i32_i16 s10, s2
	s_lshr_b32 s2, s10, 3
	.loc	1 380 34                        ; mha.py:380:34
	s_lshl_b32 s47, s33, 6
	.loc	1 574 14                        ; mha.py:574:14
	s_bfe_i64 s[2:3], s[2:3], 0x100000
	s_ashr_i32 s2, s10, 3
	.loc	1 380 34                        ; mha.py:380:34
	v_or_b32_e32 v56, s47, v59
	.loc	1 574 14                        ; mha.py:574:14
	s_mul_hi_u32 s10, s26, s2
	s_mul_i32 s14, s26, s3
	.loc	1 380 34                        ; mha.py:380:34
	v_lshrrev_b32_e32 v2, 1, v56
	.loc	1 573 14                        ; mha.py:573:14
	s_mul_i32 s11, s23, s41
	.loc	1 574 14                        ; mha.py:574:14
	s_add_i32 s10, s10, s14
	s_mul_i32 s14, s26, s2
	.loc	1 575 14                        ; mha.py:575:14
	s_mul_hi_u32 s16, s1, s2
	s_mul_i32 s3, s1, s3
	s_mul_i32 s1, s1, s2
	.loc	1 582 9                         ; mha.py:582:9
	s_mul_i32 s2, s43, s22
	.loc	1 380 14                        ; mha.py:380:14
	v_or_b32_e32 v2, s44, v2
	.loc	1 575 14                        ; mha.py:575:14
	s_add_i32 s16, s16, s3
	.loc	1 582 9                         ; mha.py:582:9
	s_add_i32 s2, s2, s11
	.loc	1 602 9                         ; mha.py:602:9
	s_mul_i32 s11, s43, s25
	.loc	1 588 14                        ; mha.py:588:14
	v_mul_lo_u32 v3, v2, s24
	.loc	1 382 14                        ; mha.py:382:14
	v_and_b32_e32 v1, 8, v58
	.loc	1 602 9                         ; mha.py:602:9
	s_mul_hi_u32 s3, s43, s25
	s_add_u32 s34, s11, s14
	.loc	1 588 14                        ; mha.py:588:14
	v_add_u32_e32 v3, s2, v3
	.loc	1 602 9                         ; mha.py:602:9
	s_addc_u32 s35, s3, s10
	.loc	1 622 9                         ; mha.py:622:9
	s_mul_hi_u32 s2, s43, s0
	s_mul_i32 s0, s43, s0
	.loc	1 597 21                        ; mha.py:597:21
	v_add_lshl_u32 v6, v1, v3, 1
	.loc	1 622 9                         ; mha.py:622:9
	s_add_u32 s24, s0, s1
	.loc	1 689 16                        ; mha.py:689:16
	v_add_u32_e32 v3, 32, v6
	v_bfrev_b32_e32 v1, 1
	.loc	1 679 18                        ; mha.py:679:18
	v_cmp_gt_i32_e32 vcc, s6, v2
	.loc	1 622 9                         ; mha.py:622:9
	s_addc_u32 s25, s2, s16
	.loc	1 689 16                        ; mha.py:689:16
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_cndmask_b32_e32 v2, v1, v3, vcc
	.loc	1 692 9                         ; mha.py:692:9
	v_cndmask_b32_e32 v6, v1, v6, vcc
	.loc	1 689 16                        ; mha.py:689:16
	buffer_load_dwordx4 v[2:5], v2, s[8:11], 0 offen sc0 nt
	v_bfe_u32 v0, v0, 5, 1
	.loc	1 692 9                         ; mha.py:692:9
	buffer_load_dwordx4 v[6:9], v6, s[8:11], 0 offen sc0 nt
	.loc	1 382 14                        ; mha.py:382:14
	v_and_b32_e32 v10, 14, v12
	.loc	1 614 15                        ; mha.py:614:15
	v_mov_b32_e32 v13, v11
	.loc	1 689 16                        ; mha.py:689:16
	v_and_or_b32 v0, v12, 62, v0
	.loc	1 385 19                        ; mha.py:385:19
	v_or_b32_e32 v12, 16, v10
	.loc	1 381 14                        ; mha.py:381:14
	v_lshrrev_b32_e32 v14, 3, v56
	.loc	1 608 14                        ; mha.py:608:14
	v_mad_i64_i32 v[32:33], s[0:1], v14, s27, v[10:11]
	.loc	1 617 21                        ; mha.py:617:21
	v_mad_i64_i32 v[34:35], s[0:1], v14, s27, v[12:13]
	.loc	1 701 8                         ; mha.py:701:8
	s_cmp_lt_i32 s7, 32
	s_cselect_b64 s[0:1], -1, 0
	.loc	1 701 5 is_stmt 0               ; mha.py:701:5
	s_and_b32 s2, s7, 31
	s_cselect_b64 s[2:3], -1, 0
	.loc	1 689 16 is_stmt 1              ; mha.py:689:16
	v_lshlrev_b32_e32 v0, 2, v0
	.loc	1 701 5                         ; mha.py:701:5
	s_or_b64 s[22:23], s[0:1], s[2:3]
	s_mov_b32 s45, 0
	s_mov_b32 s14, s27
	.loc	1 689 16                        ; mha.py:689:16
	s_waitcnt vmcnt(1)
	ds_bpermute_b32 v16, v0, v2
	ds_bpermute_b32 v17, v0, v3
	ds_bpermute_b32 v18, v0, v4
	ds_bpermute_b32 v19, v0, v5
	.loc	1 692 9                         ; mha.py:692:9
	s_waitcnt vmcnt(0)
	ds_bpermute_b32 v20, v0, v6
	ds_bpermute_b32 v21, v0, v7
	ds_bpermute_b32 v22, v0, v8
	ds_bpermute_b32 v23, v0, v9
	.loc	1 728 21                        ; mha.py:728:21
	v_cndmask_b32_e64 v0, 0, 1, s[22:23]
	s_nop 0
	v_readfirstlane_b32 s0, v0
	s_min_i32 s50, s0, s48
	.loc	1 729 21                        ; mha.py:729:21
	s_sub_i32 s49, s48, s50
	.loc	1 744 8                         ; mha.py:744:8
	s_cmp_lt_i32 s49, 1
	.loc	1 628 14                        ; mha.py:628:14
	v_mad_i64_i32 v[36:37], s[0:1], v14, s30, v[10:11]
	.loc	1 744 5                         ; mha.py:744:5
	s_cbranch_scc1 .LBB0_3
; %bb.1:
	.loc	1 745 33                        ; mha.py:745:33
	s_lshl_b32 s45, s49, 5
.Ltmp12:
	.loc	1 261 19                        ; mha.py:261:19 @[ mha.py:746:25 ]
	s_lshl_b64 s[26:27], s[14:15], 5
	.loc	1 264 19                        ; mha.py:264:19 @[ mha.py:746:25 ]
	s_lshl_b64 s[38:39], s[30:31], 5
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cmp_gt_i32 s45, 0
	s_cselect_b64 s[0:1], -1, 0
.Ltmp13:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_bfe_i32 s2, s47, 0x10006
	v_lshlrev_b32_e32 v0, 1, v56
	s_and_b32 s2, s2, 0x88
	v_mov_b32_e32 v8, 0x17e
	v_bitop3_b32 v8, v0, s2, v8 bitop3:0x6c
	v_sub_u32_e32 v0, v8, v0
	v_ashrrev_i32_e32 v0, 1, v0
	v_add_u32_e32 v0, v0, v59
.Ltmp14:
	.loc	1 608 14                        ; mha.py:608:14
	v_lshl_add_u64 v[6:7], v[32:33], 0, s[34:35]
.Ltmp15:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_lshlrev_b32_e32 v43, 2, v0
	ds_bpermute_b32 v11, v43, v6
	v_cmp_gt_i32_e64 s[2:3], s45, 0
.Ltmp16:
	.loc	1 617 21                        ; mha.py:617:21
	v_lshl_add_u64 v[4:5], v[34:35], 0, s[34:35]
.Ltmp17:
	.loc	1 261 9                         ; mha.py:261:9 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[46:47], v[6:7], 0, s[26:27]
.Ltmp18:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_lshrrev_b64 v[8:9], v0, s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v9, 1, v11
.Ltmp19:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	ds_bpermute_b32 v11, v43, v4
.Ltmp20:
	.loc	1 263 13 is_stmt 1              ; mha.py:263:13 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[48:49], v[4:5], 0, s[26:27]
.Ltmp21:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	ds_bpermute_b32 v4, v43, v46
	v_and_b32_e32 v8, 1, v8
.Ltmp22:
	.loc	1 628 14                        ; mha.py:628:14
	v_lshl_add_u64 v[2:3], v[36:37], 0, s[24:25]
.Ltmp23:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_lshl_b32 s55, s33, 8
	v_cmp_eq_u32_e32 vcc, 1, v8
	v_cmp_gt_i32_e64 s[2:3], s45, 32
.Ltmp24:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_lshlrev_b32_e32 v10, 1, v2
.Ltmp25:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_and_b32 s9, s37, 0xffff
	s_mov_b32 s8, s36
	s_add_i32 m0, s55, 0x3840
	v_cndmask_b32_e32 v8, v1, v9, vcc
.Ltmp26:
	.loc	1 264 9 is_stmt 1               ; mha.py:264:9 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[44:45], v[2:3], 0, s[38:39]
.Ltmp27:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_lshrrev_b64 v[2:3], v0, s[2:3]
	buffer_load_dword v8, s[8:11], 0 offen lds
.Ltmp28:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v8, 1, v11
.Ltmp29:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v4
.Ltmp30:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	ds_bpermute_b32 v4, v43, v48
	s_add_i32 m0, s55, 0x2c40
	v_cndmask_b32_e32 v8, v1, v8, vcc
	; asyncmark
	buffer_load_dword v8, s[8:11], 0 offen lds
.Ltmp31:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_and_b32 s17, s13, 0xffff
	s_add_i32 m0, s55, 0x2000
.Ltmp32:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_and_b32_e32 v2, 1, v2
.Ltmp33:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_mov_b32 s16, s12
	s_mov_b32 s18, s10
	s_mov_b32 s19, s11
	v_cndmask_b32_e64 v8, v1, v10, s[0:1]
.Ltmp34:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cmp_gt_i32 s45, 32
.Ltmp35:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_cmp_eq_u32_e64 s[2:3], 1, v2
.Ltmp36:
	; asyncmark
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	buffer_load_dword v8, s[16:19], 0 offen lds
.Ltmp37:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cselect_b64 vcc, -1, 0
.Ltmp38:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s55, 0x3c40
	v_cndmask_b32_e64 v2, v1, v3, s[2:3]
	; asyncmark
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dword v2, s[8:11], 0 offen lds
.Ltmp39:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_lshlrev_b32_e32 v2, 1, v4
.Ltmp40:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_lshlrev_b32_e32 v8, 1, v44
.Ltmp41:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s55, 0x3040
	v_cndmask_b32_e64 v2, v1, v2, s[2:3]
	; asyncmark
	buffer_load_dword v2, s[8:11], 0 offen lds
.Ltmp42:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s55, 0x2420
	v_cndmask_b32_e32 v1, v1, v8, vcc
	; asyncmark
	buffer_load_dword v1, s[16:19], 0 offen lds
.Ltmp43:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_sub_i32 s56, s45, 64
	.loc	1 177 20                        ; mha.py:177:20 @[ mha.py:746:25 ]
	v_mov_b32_e32 v1, 0x3fb8aa3b
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cmp_gt_i32 s56, 0
	v_lshlrev_b32_e32 v2, 5, v54
	v_lshlrev_b32_e32 v39, 3, v59
	; asyncmark
	; wait_asyncmark(3)
.Ltmp44:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt vmcnt(3) lgkmcnt(0)
	s_barrier
.Ltmp45:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cbranch_scc1 .LBB0_4
; %bb.2:                                ; %.._crit_edge_crit_edge
.Ltmp46:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_lshrrev_b32_e32 v42, 1, v60
	s_mov_b64 s[2:3], 0
	s_branch .LBB0_5
.Ltmp47:
.LBB0_3:
	.loc	1 0 18 is_stmt 0                ; mha.py:0:18
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v38, 0xff800000
	v_mov_b32_e32 v57, 1.0
	v_mov_b32_e32 v30, v31
	v_mov_b32_e32 v29, v31
	v_mov_b32_e32 v28, v31
	v_mov_b32_e32 v27, v31
	v_mov_b32_e32 v26, v31
	v_mov_b32_e32 v25, v31
	v_mov_b32_e32 v24, v31
	.loc	1 744 5 is_stmt 1               ; mha.py:744:5
	s_branch .LBB0_19
.LBB0_4:
	.loc	1 0 5 is_stmt 0                 ; mha.py:0:5
	s_mov_b64 s[2:3], -1
                                        ; implicit-def: $vgpr42
.LBB0_5:                                ; %Flow353
	v_mul_f32_e32 v40, s46, v1
	v_and_b32_e32 v1, 8, v54
	s_and_b32 s51, s47, 64
	v_and_b32_e32 v61, 0x2e0, v2
	s_mov_b32 s61, 1
	s_andn2_b64 vcc, exec, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], 0, v1
	s_cbranch_vccnz .LBB0_8
; %bb.6:                                ; %.lr.ph
	v_mov_b32_e32 v1, 0x110
	v_cndmask_b32_e64 v1, v1, 0, s[2:3]
	v_lshrrev_b32_e32 v42, 1, v60
	s_lshl_b32 s8, s42, 1
	v_bitop3_b32 v62, v1, v61, v42 bitop3:0xde
	v_and_b32_e32 v1, 31, v54
	s_and_b32 s62, s8, 4
	s_lshl2_add_u32 s8, s51, 0
	v_mov_b32_e32 v4, 0x420
	v_cmp_eq_u32_e32 vcc, 0, v60
	v_lshl_add_u32 v2, v1, 3, s8
	v_lshlrev_b32_e32 v1, 1, v1
	v_cndmask_b32_e64 v4, v4, 0, vcc
	v_bitop3_b32 v4, s47, v4, v1 bitop3:0x36
	v_and_b32_e32 v1, 60, v54
	v_lshlrev_b32_e32 v12, 6, v1
	v_and_b32_e32 v13, 24, v58
	v_lshlrev_b32_e32 v1, 1, v1
	s_lshl_b32 s8, s33, 5
	v_bitop3_b32 v1, v12, v1, v13 bitop3:0x36
	v_xor_b32_e32 v12, s8, v1
	v_lshrrev_b64 v[0:1], v0, exec
	v_mov_b32_e32 v26, 0
	s_lshl_b32 s63, s33, 7
	v_lshl_add_u32 v3, v55, 3, 0
	v_xor_b32_e32 v5, 8, v4
	v_xor_b32_e32 v6, 16, v4
	v_xor_b32_e32 v7, 24, v4
	v_xor_b32_e32 v8, 64, v4
	v_xor_b32_e32 v9, 0x48, v4
	v_xor_b32_e32 v10, 0x50, v4
	v_xor_b32_e32 v11, 0x58, v4
	v_and_b32_e32 v0, 1, v0
	s_mov_b32 s60, 0
.Ltmp48:
	.loc	1 186 18 is_stmt 1              ; mha.py:186:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v41, v40
	v_mov_b32_e32 v50, v40
	v_mov_b32_e32 v51, v40
	s_movk_i32 s57, 0x3840
	s_movk_i32 s52, 0x3c40
	s_movk_i32 s58, 0x2c40
	s_movk_i32 s53, 0x3040
	s_movk_i32 s59, 0x2000
	s_movk_i32 s54, 0x2420
	v_mov_b32_e32 v52, v40
	v_mov_b32_e32 v53, v40
	v_mov_b32_e32 v64, 0xff800000
	v_mov_b32_e32 v57, 1.0
	v_cmp_eq_u32_e32 vcc, 1, v0
	v_bfrev_b32_e32 v63, 1
	s_mov_b32 s8, s36
	s_mov_b32 s16, s12
	s_mov_b32 s18, s10
	s_mov_b32 s19, s11
	v_add_u32_e32 v65, s62, v2
	v_add_u32_e32 v66, s63, v3
	v_add_u32_e32 v67, 0, v4
	v_add_u32_e32 v68, 0, v5
	v_add_u32_e32 v69, 0, v6
	v_add_u32_e32 v70, 0, v7
	v_add_u32_e32 v71, 0, v8
	v_add_u32_e32 v72, 0, v9
	v_add_u32_e32 v73, 0, v10
	v_add_u32_e32 v74, 0, v11
	v_add_u32_e32 v75, 0, v12
	v_mov_b32_e32 v27, v26
	v_mov_b32_e32 v24, v26
	v_mov_b32_e32 v25, v26
	v_mov_b32_e32 v30, v26
	v_mov_b32_e32 v31, v26
	v_mov_b32_e32 v28, v26
	v_mov_b32_e32 v29, v26
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 261 9                         ; mha.py:261:9 @[ mha.py:746:25 ]
	v_ashrrev_i32_e32 v47, 31, v46
	.loc	1 263 13                        ; mha.py:263:13 @[ mha.py:746:25 ]
	v_ashrrev_i32_e32 v49, 31, v48
	.loc	1 261 9                         ; mha.py:261:9 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[46:47], v[46:47], 0, s[26:27]
	.loc	1 263 13                        ; mha.py:263:13 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[48:49], v[48:49], 0, s[26:27]
.Ltmp49:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	ds_bpermute_b32 v2, v43, v46
	s_mov_b32 s62, s57
	s_mov_b32 s57, s52
	s_mov_b32 s52, s58
	s_mov_b32 s58, s53
	s_mov_b32 s53, s59
	s_mov_b32 s59, s54
.Ltmp50:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_add_i32 s54, s61, 1
.Ltmp51:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	ds_bpermute_b32 v3, v43, v48
.Ltmp52:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cmp_lt_i32 s54, 3
	s_cselect_b32 s61, s54, 0
.Ltmp53:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v0, s62, v62
	s_lshl_b32 s54, s61, 10
.Ltmp54:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_lshl_b32 s62, s61, 5
.Ltmp55:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 s63, s54, s55
.Ltmp56:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_and_b32 s62, s62, 0xfffffe0
.Ltmp57:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v2, 1, v2
.Ltmp58:
	.loc	1 264 9 is_stmt 1               ; mha.py:264:9 @[ mha.py:746:25 ]
	v_ashrrev_i32_e32 v45, 31, v44
.Ltmp59:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v1, s52, v62
.Ltmp60:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v38, s53, v39
.Ltmp61:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 s52, s54, 0x3840
.Ltmp62:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_add_i32 s53, s54, 0x2c40
.Ltmp63:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s63, 0x3840
.Ltmp64:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_add_i32 s54, s54, s62
.Ltmp65:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v3
.Ltmp66:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_cndmask_b32_e32 v2, v63, v2, vcc
.Ltmp67:
	.loc	1 264 9 is_stmt 1               ; mha.py:264:9 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[44:45], v[44:45], 0, s[38:39]
.Ltmp68:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_cndmask_b32_e32 v3, v63, v3, vcc
.Ltmp69:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	buffer_load_dword v2, s[8:11], 0 offen lds
.Ltmp70:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s63, 0x2c40
.Ltmp71:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_addk_i32 s54, 0x2000
	v_lshlrev_b32_e32 v45, 1, v44
.Ltmp72:
	; asyncmark
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	buffer_load_dword v3, s[8:11], 0 offen lds
.Ltmp73:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s54, s55
.Ltmp74:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	ds_read_b128 v[76:79], v0
.Ltmp75:
	; asyncmark
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	buffer_load_dword v45, s[16:19], 0 offen lds
.Ltmp76:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	ds_read_b128 v[80:83], v1
.Ltmp77:
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[80:83], v[16:19], 0
	; asyncmark
.Ltmp78:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	ds_read_b64_tr_b16 v[88:89], v38
	ds_read_b64_tr_b16 v[90:91], v38 offset:512
.Ltmp79:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_add_i32 s60, s60, 32
	s_cmp_lt_i32 s60, s56
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[76:79], v[20:23], v[0:15]
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	s_nop 11
	v_pk_mul_f32 v[4:5], v[52:53], v[4:5]
	v_pk_mul_f32 v[2:3], v[52:53], v[2:3]
	v_pk_mul_f32 v[0:1], v[40:41], v[0:1]
	v_mul_f32_e32 v45, v40, v8
	v_mov_b32_e32 v76, v11
	v_mov_b32_e32 v77, v12
	v_mov_b32_e32 v8, v9
	v_mov_b32_e32 v9, v10
	v_mov_b32_e32 v10, v13
	v_mov_b32_e32 v11, v14
	v_pk_mul_f32 v[12:13], v[52:53], v[76:77]
	v_pk_mul_f32 v[8:9], v[50:51], v[8:9]
.Ltmp80:
	.file	3 "/root/triton/python/triton/language" "standard.py"
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e32 v14, v0, v1
	v_max3_f32 v38, v3, v4, v5
.Ltmp81:
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[10:11], v[52:53], v[10:11]
	v_pk_mul_f32 v[6:7], v[52:53], v[6:7]
	v_mul_f32_e32 v15, v40, v15
.Ltmp82:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max3_f32 v49, v8, v9, v12
	v_max3_f32 v14, v14, v2, v38
	v_max3_f32 v38, v13, v10, v11
	v_max3_f32 v47, v6, v7, v45
	v_max3_f32 v38, v49, v38, v15
	v_max3_f32 v14, v14, v47, v38
.Ltmp83:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v38, v14
	s_nop 1
	v_permlane32_swap_b32_e32 v14, v38
.Ltmp84:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:746:25 ]
	v_max3_f32 v38, v64, v14, v38
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_sub_f32_e32 v14, v45, v38
	v_pk_add_f32 v[8:9], v[8:9], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[12:13], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[10:11], v[10:11], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e32 v45, v15, v38
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:746:25 ]
	v_sub_f32_e32 v47, v64, v38
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_pk_add_f32 v[0:1], v[0:1], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[6:7], v[6:7], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:746:25 ]
	v_exp_f32_e32 v15, v8
	v_exp_f32_e32 v8, v9
	v_exp_f32_e32 v9, v12
	v_exp_f32_e32 v12, v13
	v_exp_f32_e32 v13, v10
	v_exp_f32_e32 v10, v11
	v_exp_f32_e32 v11, v45
	.loc	1 241 17 is_stmt 1              ; mha.py:241:17 @[ mha.py:746:25 ]
	v_exp_f32_e32 v45, v47
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_pk_add_f32 v[2:3], v[2:3], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:746:25 ]
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v4, v4
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v14, v14
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v2, v2
	v_exp_f32_e32 v3, v3
	.loc	1 246 15 is_stmt 1              ; mha.py:246:15 @[ mha.py:746:25 ]
	ds_write_b32 v65, v45
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[84:85], v66
.Ltmp85:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[78:79], v[4:5], v[6:7]
	v_pk_add_f32 v[80:81], v[14:15], v[8:9]
	v_pk_add_f32 v[82:83], v[12:13], v[10:11]
.Ltmp86:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_cvt_pk_bf16_f32 v47, v0, s0
	v_cvt_pk_bf16_f32 v8, v8, s0
	v_cvt_pk_bf16_f32 v9, v9, s0
	v_cvt_pk_bf16_f32 v12, v12, s0
	v_cvt_pk_bf16_f32 v13, v13, s0
	v_cvt_pk_bf16_f32 v6, v6, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
.Ltmp87:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[76:77], v[0:1], v[2:3]
.Ltmp88:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_mul_f32_e32 v45, v57, v45
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cvt_pk_bf16_f32 v14, v14, s0
	v_cvt_pk_bf16_f32 v49, v1, s0
	v_cvt_pk_bf16_f32 v15, v15, s0
	v_cvt_pk_bf16_f32 v57, v2, s0
	v_cvt_pk_bf16_f32 v86, v3, s0
	v_cvt_pk_bf16_f32 v4, v4, s0
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v67, v47
	ds_write_b16 v67, v14 offset:4096
	ds_write_b16 v68, v49 offset:256
	ds_write_b16 v68, v15 offset:4352
	ds_write_b16 v69, v57 offset:512
	ds_write_b16 v69, v8 offset:4608
	ds_write_b16 v70, v86 offset:768
	ds_write_b16 v70, v9 offset:4864
	ds_write_b16 v71, v4 offset:2048
	ds_write_b16 v71, v12 offset:6144
	ds_write_b16 v72, v5 offset:2304
	ds_write_b16 v72, v13 offset:6400
	ds_write_b16 v73, v6 offset:2560
	ds_write_b16 v73, v10 offset:6656
	ds_write_b16 v74, v7 offset:2816
	ds_write_b16 v74, v11 offset:6912
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[6:7], v75
	ds_read_b64_tr_b16 v[8:9], v75 offset:4096
	ds_read_b64_tr_b16 v[12:13], v75 offset:4224
	ds_read_b64_tr_b16 v[10:11], v75 offset:128
.Ltmp89:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[0:1], v[76:77], v[78:79]
	v_pk_add_f32 v[2:3], v[80:81], v[82:83]
	v_mov_b32_e32 v64, v38
	v_pk_add_f32 v[0:1], v[0:1], v[2:3]
.Ltmp90:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[2:3], v[26:27], v[84:85] op_sel_hi:[1,0]
.Ltmp91:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_pk_add_f32 v[4:5], v[0:1], v[0:1] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp92:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[0:1], v[24:25], v[84:85] op_sel_hi:[1,0]
.Ltmp93:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v5, v4
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v5
.Ltmp94:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[88:91], v[6:9], v[0:3]
	; wait_asyncmark(3)
.Ltmp95:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt vmcnt(3) lgkmcnt(0)
	s_barrier
.Ltmp96:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_nop 0
	v_pk_mul_f32 v[2:3], v[30:31], v[84:85] op_sel:[0,1]
	v_pk_mul_f32 v[0:1], v[28:29], v[84:85] op_sel:[0,1]
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[28:31], v[88:91], v[10:13], v[0:3]
.Ltmp97:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	s_nop 2
	v_add_f32_e32 v0, v4, v5
.Ltmp98:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_add_f32_e32 v57, v0, v45
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cbranch_scc1 .LBB0_7
	s_branch .LBB0_9
.LBB0_8:
	.loc	1 0 5 is_stmt 0                 ; mha.py:0:5
	v_mov_b32_e32 v31, 0
	s_movk_i32 s57, 0x3840
	s_movk_i32 s52, 0x3c40
	s_movk_i32 s58, 0x2c40
	s_movk_i32 s53, 0x3040
	s_movk_i32 s59, 0x2000
	s_movk_i32 s54, 0x2420
	v_mov_b32_e32 v38, 0xff800000
	v_mov_b32_e32 v57, 1.0
	v_mov_b32_e32 v30, v31
	v_mov_b32_e32 v29, v31
	v_mov_b32_e32 v28, v31
	v_mov_b32_e32 v27, v31
	v_mov_b32_e32 v26, v31
	v_mov_b32_e32 v25, v31
	v_mov_b32_e32 v24, v31
.LBB0_9:                                ; %._crit_edge
.Ltmp99:
	.loc	1 36 18 is_stmt 1               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v0, 0x110
	v_cndmask_b32_e64 v0, v0, 0, s[2:3]
	v_bitop3_b32 v46, v0, v61, v42 bitop3:0xde
.Ltmp100:
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	v_cndmask_b32_e64 v0, 0, 1, s[0:1]
	v_cmp_ne_u32_e64 s[2:3], 1, v0
	s_andn2_b64 vcc, exec, s[0:1]
.Ltmp101:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v41, s59, v39
.Ltmp102:
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	s_cbranch_vccnz .LBB0_11
; %bb.10:
.Ltmp103:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v0, s58, v46
	ds_read_b128 v[48:51], v0
.Ltmp104:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v42, s57, v46
	ds_read_b128 v[42:45], v42
.Ltmp105:
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[48:51], v[16:19], 0
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[42:45], v[20:23], v[0:15]
	s_nop 11
	v_mov_b32_e32 v42, v13
	v_mov_b32_e32 v43, v14
	v_mov_b32_e32 v44, v9
	v_mov_b32_e32 v45, v10
	v_mov_b32_e32 v10, v11
	v_mov_b32_e32 v11, v12
	s_branch .LBB0_12
.Ltmp106:
.LBB0_11:
	.loc	1 0 14 is_stmt 0                ; mha.py:0:14
	v_mov_b32_e32 v42, 0
	.loc	1 746 25 is_stmt 1              ; mha.py:746:25
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v43, v42
	v_mov_b32_e32 v44, v42
	v_mov_b32_e32 v45, v42
	v_mov_b32_e32 v10, v42
	v_mov_b32_e32 v11, v42
	v_mov_b32_e32 v0, v42
	v_mov_b32_e32 v1, v42
	v_mov_b32_e32 v2, v42
	v_mov_b32_e32 v3, v42
	v_mov_b32_e32 v4, v42
	v_mov_b32_e32 v5, v42
	v_mov_b32_e32 v6, v42
	v_mov_b32_e32 v7, v42
.LBB0_12:
.Ltmp107:
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[4:5], v[40:41], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[40:41], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[40:41], v[0:1] op_sel_hi:[0,1]
	v_mul_f32_e32 v1, v40, v8
	v_pk_mul_f32 v[8:9], v[40:41], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[40:41], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[40:41], v[42:43] op_sel_hi:[0,1]
	.loc	1 0 0 is_stmt 0                 ; mha.py:0 @[ mha.py:746:25 ]
	ds_read_b64_tr_b16 v[68:69], v41
	ds_read_b64_tr_b16 v[70:71], v41 offset:512
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[6:7], v[40:41], v[6:7] op_sel_hi:[0,1]
	v_mul_f32_e32 v41, v40, v15
.Ltmp108:
	.loc	3 170 12 is_stmt 1              ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e32 v0, v12, v13
	v_max3_f32 v14, v3, v4, v5
	v_max3_f32 v44, v10, v11, v8
	v_max3_f32 v45, v9, v42, v43
	v_max3_f32 v15, v6, v7, v1
	v_max3_f32 v0, v0, v2, v14
	v_max3_f32 v14, v44, v45, v41
	v_max3_f32 v0, v0, v15, v14
.Ltmp109:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v14, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v14
.Ltmp110:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:746:25 ]
	v_max3_f32 v0, v38, v0, v14
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_pk_add_f32 v[12:13], v[12:13], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[2:3], v[2:3], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[6:7], v[6:7], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e32 v1, v1, v0
	v_pk_add_f32 v[10:11], v[10:11], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9], v[8:9], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15], v[42:43], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e32 v41, v41, v0
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:746:25 ]
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	v_exp_f32_e32 v42, v2
	v_exp_f32_e32 v43, v3
	v_exp_f32_e32 v44, v4
	v_exp_f32_e32 v45, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v48, v1
	v_exp_f32_e32 v49, v10
	v_exp_f32_e32 v10, v11
	v_exp_f32_e32 v11, v8
	v_exp_f32_e32 v8, v9
	v_exp_f32_e32 v9, v14
	v_exp_f32_e32 v14, v15
	v_exp_f32_e32 v15, v41
.Ltmp111:
	.loc	3 293 12 is_stmt 1              ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[2:3], v[12:13], v[42:43]
	v_pk_add_f32 v[4:5], v[44:45], v[6:7]
	v_pk_add_f32 v[50:51], v[48:49], v[10:11]
	v_pk_add_f32 v[52:53], v[8:9], v[14:15]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
	v_pk_add_f32 v[4:5], v[50:51], v[52:53]
.Ltmp112:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	v_and_b32_e32 v41, 31, v54
.Ltmp113:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
.Ltmp114:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	s_lshl_b32 s0, s42, 1
.Ltmp115:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_pk_add_f32 v[4:5], v[2:3], v[2:3] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp116:
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:746:25 ]
	v_sub_f32_e32 v2, v38, v0
	.loc	1 241 17 is_stmt 0              ; mha.py:241:17 @[ mha.py:746:25 ]
	v_exp_f32_e32 v5, v2
	.loc	1 246 15 is_stmt 1              ; mha.py:246:15 @[ mha.py:746:25 ]
	s_lshl2_add_u32 s1, s51, 0
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v47, 0x420
	v_cmp_eq_u32_e32 vcc, 0, v60
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	s_and_b32 s0, s0, 4
	v_lshl_add_u32 v2, v41, 3, s1
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_lshlrev_b32_e32 v41, 1, v41
	v_cndmask_b32_e64 v47, v47, 0, vcc
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	v_add_u32_e32 v50, s0, v2
	s_lshl_b32 s0, s33, 7
	v_lshl_add_u32 v2, v55, 3, 0
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_bitop3_b32 v41, s47, v47, v41 bitop3:0x36
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	v_add_u32_e32 v51, s0, v2
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_add_u32_e32 v52, 0, v41
	v_cvt_pk_bf16_f32 v12, v12, s0
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	ds_write_b32 v50, v5
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[2:3], v51
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v52, v12
	v_cvt_pk_bf16_f32 v12, v48, s0
	ds_write_b16 v52, v12 offset:4096
	v_xor_b32_e32 v12, 8, v41
	v_add_u32_e32 v53, 0, v12
	v_cvt_pk_bf16_f32 v12, v13, s0
	ds_write_b16 v53, v12 offset:256
	v_cvt_pk_bf16_f32 v12, v49, s0
	ds_write_b16 v53, v12 offset:4352
	v_xor_b32_e32 v12, 16, v41
	v_add_u32_e32 v61, 0, v12
	v_cvt_pk_bf16_f32 v10, v10, s0
	ds_write_b16 v61, v10 offset:4608
	v_xor_b32_e32 v10, 24, v41
	v_cvt_pk_bf16_f32 v12, v42, s0
	v_add_u32_e32 v62, 0, v10
	v_cvt_pk_bf16_f32 v10, v43, s0
	ds_write_b16 v61, v12 offset:512
	ds_write_b16 v62, v10 offset:768
	v_cvt_pk_bf16_f32 v10, v11, s0
	ds_write_b16 v62, v10 offset:4864
	v_xor_b32_e32 v10, 64, v41
	v_add_u32_e32 v63, 0, v10
	v_cvt_pk_bf16_f32 v8, v8, s0
	ds_write_b16 v63, v8 offset:6144
	v_xor_b32_e32 v8, 0x48, v41
	v_cvt_pk_bf16_f32 v10, v44, s0
	v_add_u32_e32 v64, 0, v8
	v_cvt_pk_bf16_f32 v8, v45, s0
	ds_write_b16 v63, v10 offset:2048
	ds_write_b16 v64, v8 offset:2304
	v_cvt_pk_bf16_f32 v8, v9, s0
	ds_write_b16 v64, v8 offset:6400
	v_xor_b32_e32 v8, 0x50, v41
	v_add_u32_e32 v65, 0, v8
	v_cvt_pk_bf16_f32 v6, v6, s0
	ds_write_b16 v65, v6 offset:2560
	v_cvt_pk_bf16_f32 v6, v14, s0
	ds_write_b16 v65, v6 offset:6656
	v_xor_b32_e32 v6, 0x58, v41
	v_add_u32_e32 v66, 0, v6
	v_cvt_pk_bf16_f32 v6, v7, s0
	ds_write_b16 v66, v6 offset:2816
	v_cvt_pk_bf16_f32 v6, v15, s0
	ds_write_b16 v66, v6 offset:6912
	v_and_b32_e32 v6, 60, v54
	v_lshlrev_b32_e32 v7, 6, v6
	v_and_b32_e32 v8, 24, v58
	v_lshlrev_b32_e32 v6, 1, v6
	s_lshl_b32 s0, s33, 5
	v_bitop3_b32 v6, v7, v6, v8 bitop3:0x36
.Ltmp117:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v1, v4
.Ltmp118:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_xor_b32_e32 v6, s0, v6
.Ltmp119:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	s_nop 0
	v_permlane32_swap_b32_e32 v4, v1
.Ltmp120:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_and_b64 vcc, exec, s[2:3]
	v_add_u32_e32 v67, 0, v6
	.loc	1 259 26 is_stmt 0              ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_cbranch_vccnz .LBB0_14
; %bb.13:
	.loc	1 0 0                           ; mha.py:0 @[ mha.py:746:25 ]
	ds_read_b64_tr_b16 v[10:11], v67
	ds_read_b64_tr_b16 v[12:13], v67 offset:4096
	ds_read_b64_tr_b16 v[44:45], v67 offset:4224
	ds_read_b64_tr_b16 v[42:43], v67 offset:128
	.loc	1 248 15 is_stmt 1              ; mha.py:248:15 @[ mha.py:746:25 ]
	v_mul_f32_e32 v8, v57, v5
.Ltmp121:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_add_f32_e32 v1, v4, v1
.Ltmp122:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[6:7], v[26:27], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[24:25], v[2:3] op_sel_hi:[1,0]
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_add_f32_e32 v57, v1, v8
	v_mov_b32_e32 v38, v0
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[68:71], v[10:13], v[4:7]
	s_nop 2
	v_mul_f32_e64 v4, v30, v3
	v_mul_f32_e64 v5, v31, v3
	v_pk_mul_f32 v[2:3], v[28:29], v[2:3] op_sel:[0,1]
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[28:31], v[68:71], v[42:45], v[2:5]
.LBB0_14:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_or_b32 s2, s45, 31
	s_cmp_gt_i32 s2, 63
	.loc	1 0 0 is_stmt 0                 ; mha.py:0 @[ mha.py:746:25 ]
	v_mov_b32_e32 v41, v40
	v_mov_b32_e32 v42, v40
	v_mov_b32_e32 v43, v40
	v_mov_b32_e32 v44, v40
	v_mov_b32_e32 v45, v40
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s2, 64
.Ltmp123:
	.loc	1 36 18 is_stmt 1               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v39, s54, v39
.Ltmp124:
	; wait_asyncmark(0)
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
.Ltmp125:
	.loc	1 182 14 is_stmt 1              ; mha.py:182:14 @[ mha.py:746:25 ]
	s_cbranch_scc1 .LBB0_16
; %bb.15:
.Ltmp126:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v0, s53, v46
	ds_read_b128 v[68:71], v0
.Ltmp127:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v46, s52, v46
	ds_read_b128 v[46:49], v46
.Ltmp128:
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[68:71], v[16:19], 0
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[46:49], v[20:23], v[0:15]
	s_nop 11
	v_mov_b32_e32 v46, v13
	v_mov_b32_e32 v47, v14
	v_mov_b32_e32 v48, v9
	v_mov_b32_e32 v49, v10
	v_mov_b32_e32 v10, v11
	v_mov_b32_e32 v11, v12
	s_branch .LBB0_17
.Ltmp129:
.LBB0_16:
	.loc	1 0 14 is_stmt 0                ; mha.py:0:14
	v_mov_b32_e32 v46, 0
	.loc	1 746 25 is_stmt 1              ; mha.py:746:25
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v47, v46
	v_mov_b32_e32 v48, v46
	v_mov_b32_e32 v49, v46
	v_mov_b32_e32 v10, v46
	v_mov_b32_e32 v11, v46
	v_mov_b32_e32 v0, v46
	v_mov_b32_e32 v1, v46
	v_mov_b32_e32 v2, v46
	v_mov_b32_e32 v3, v46
	v_mov_b32_e32 v4, v46
	v_mov_b32_e32 v5, v46
	v_mov_b32_e32 v6, v46
	v_mov_b32_e32 v7, v46
.LBB0_17:
.Ltmp130:
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v12, v40
	v_mov_b32_e32 v13, v40
	v_pk_mul_f32 v[6:7], v[12:13], v[6:7]
	v_pk_mul_f32 v[4:5], v[12:13], v[4:5]
	v_pk_mul_f32 v[2:3], v[12:13], v[2:3]
	v_pk_mul_f32 v[72:73], v[40:41], v[0:1]
	v_mul_f32_e32 v1, v40, v8
	v_pk_mul_f32 v[8:9], v[12:13], v[10:11]
	v_pk_mul_f32 v[10:11], v[42:43], v[48:49]
	v_pk_mul_f32 v[12:13], v[44:45], v[46:47]
	.loc	1 0 0 is_stmt 0                 ; mha.py:0 @[ mha.py:746:25 ]
	ds_read_b64_tr_b16 v[68:69], v39
	ds_read_b64_tr_b16 v[70:71], v39 offset:512
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_mul_f32_e32 v39, v40, v15
.Ltmp131:
	.loc	3 170 12 is_stmt 1              ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e32 v0, v72, v73
	v_max3_f32 v14, v3, v4, v5
	v_max3_f32 v40, v10, v11, v8
	v_max3_f32 v41, v9, v12, v13
	v_max3_f32 v15, v6, v7, v1
	v_max3_f32 v0, v0, v2, v14
	v_max3_f32 v14, v40, v41, v39
	v_max3_f32 v0, v0, v15, v14
.Ltmp132:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v14, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v14
.Ltmp133:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:746:25 ]
	v_max3_f32 v0, v38, v0, v14
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_pk_add_f32 v[14:15], v[72:73], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[2:3], v[2:3], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[6:7], v[6:7], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e32 v1, v1, v0
	v_pk_add_f32 v[10:11], v[10:11], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9], v[8:9], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[12:13], v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e32 v39, v39, v0
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:746:25 ]
	v_exp_f32_e32 v14, v14
	v_exp_f32_e32 v15, v15
	v_exp_f32_e32 v40, v2
	v_exp_f32_e32 v41, v3
	v_exp_f32_e32 v42, v4
	v_exp_f32_e32 v43, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v44, v1
	v_exp_f32_e32 v45, v10
	v_exp_f32_e32 v10, v11
	v_exp_f32_e32 v11, v8
	v_exp_f32_e32 v8, v9
	v_exp_f32_e32 v9, v12
	v_exp_f32_e32 v12, v13
	v_exp_f32_e32 v13, v39
	.loc	1 241 30 is_stmt 1              ; mha.py:241:30 @[ mha.py:746:25 ]
	v_sub_f32_e32 v1, v38, v0
	.loc	1 241 17 is_stmt 0              ; mha.py:241:17 @[ mha.py:746:25 ]
	v_exp_f32_e32 v1, v1
.Ltmp134:
	.loc	3 293 12 is_stmt 1              ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[2:3], v[14:15], v[40:41]
	v_pk_add_f32 v[4:5], v[42:43], v[6:7]
	v_pk_add_f32 v[46:47], v[44:45], v[10:11]
	v_pk_add_f32 v[48:49], v[8:9], v[12:13]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
	v_pk_add_f32 v[4:5], v[46:47], v[48:49]
.Ltmp135:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_cvt_pk_bf16_f32 v14, v14, s0
.Ltmp136:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
.Ltmp137:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	ds_write_b32 v50, v1
.Ltmp138:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_pk_add_f32 v[4:5], v[2:3], v[2:3] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp139:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[2:3], v51
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v52, v14
	v_cvt_pk_bf16_f32 v14, v44, s0
	ds_write_b16 v52, v14 offset:4096
	v_cvt_pk_bf16_f32 v14, v15, s0
	ds_write_b16 v53, v14 offset:256
	v_cvt_pk_bf16_f32 v14, v45, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	ds_write_b16 v53, v14 offset:4352
	v_cvt_pk_bf16_f32 v14, v40, s0
	ds_write_b16 v61, v10 offset:4608
	v_cvt_pk_bf16_f32 v10, v41, s0
	ds_write_b16 v61, v14 offset:512
	ds_write_b16 v62, v10 offset:768
	v_cvt_pk_bf16_f32 v10, v11, s0
	v_cvt_pk_bf16_f32 v8, v8, s0
	ds_write_b16 v62, v10 offset:4864
	v_cvt_pk_bf16_f32 v10, v42, s0
	ds_write_b16 v63, v8 offset:6144
	v_cvt_pk_bf16_f32 v8, v43, s0
	v_cvt_pk_bf16_f32 v6, v6, s0
	ds_write_b16 v63, v10 offset:2048
	ds_write_b16 v64, v8 offset:2304
	v_cvt_pk_bf16_f32 v8, v9, s0
	ds_write_b16 v65, v6 offset:2560
	v_cvt_pk_bf16_f32 v6, v12, s0
.Ltmp140:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v5, v4
.Ltmp141:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	ds_write_b16 v64, v8 offset:6400
	ds_write_b16 v65, v6 offset:6656
	v_cvt_pk_bf16_f32 v6, v7, s0
.Ltmp142:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_permlane32_swap_b32_e32 v4, v5
.Ltmp143:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	ds_write_b16 v66, v6 offset:2816
	v_cvt_pk_bf16_f32 v6, v13, s0
	.loc	1 259 19 is_stmt 0              ; mha.py:259:19 @[ mha.py:746:25 ]
	s_andn2_b64 vcc, exec, s[0:1]
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	ds_write_b16 v66, v6 offset:6912
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_cbranch_vccnz .LBB0_19
; %bb.18:
	.loc	1 0 0                           ; mha.py:0 @[ mha.py:746:25 ]
	ds_read_b64_tr_b16 v[10:11], v67
	ds_read_b64_tr_b16 v[12:13], v67 offset:4096
	ds_read_b64_tr_b16 v[40:41], v67 offset:4224
	ds_read_b64_tr_b16 v[38:39], v67 offset:128
.Ltmp144:
	.loc	3 263 12 is_stmt 1              ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_add_f32_e32 v8, v4, v5
.Ltmp145:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[6:7], v[26:27], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[24:25], v[2:3] op_sel_hi:[1,0]
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_mul_f32_e32 v1, v57, v1
	v_add_f32_e32 v57, v8, v1
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[68:71], v[10:13], v[4:7]
	s_nop 2
	v_mul_f32_e64 v4, v30, v3
	v_mul_f32_e64 v5, v31, v3
	v_pk_mul_f32 v[2:3], v[28:29], v[2:3] op_sel:[0,1]
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[28:31], v[68:71], v[38:41], v[2:5]
	v_mov_b32_e32 v38, v0
.Ltmp146:
.LBB0_19:                               ; %Flow355
	.loc	1 0 19 is_stmt 0                ; mha.py:0:19
	s_load_dwordx2 s[8:9], s[4:5], 0x58
	s_load_dwordx2 s[10:11], s[4:5], 0x7c
	s_load_dword s18, s[4:5], 0x60
	.loc	1 798 8 is_stmt 1               ; mha.py:798:8
	s_cmp_lt_i32 s50, 1
	.loc	1 798 5 is_stmt 0               ; mha.py:798:5
	s_cbranch_scc1 .LBB0_23
; %bb.20:
	.loc	1 731 17 is_stmt 1              ; mha.py:731:17
	s_lshl_b32 s19, s48, 5
.Ltmp147:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	s_cmp_ge_i32 s45, s19
	s_cbranch_scc1 .LBB0_23
.Ltmp148:
; %bb.21:                               ; %.lr.ph121
	.loc	1 803 19                        ; mha.py:803:19
	s_lshl_b32 s2, s49, 5
	.loc	1 806 19                        ; mha.py:806:19
	s_mul_i32 s0, s2, s31
	s_mul_hi_u32 s1, s2, s30
.Ltmp149:
	.loc	1 261 19                        ; mha.py:261:19 @[ mha.py:811:25 ]
	s_lshl_b64 s[4:5], s[14:15], 5
	.loc	1 264 19                        ; mha.py:264:19 @[ mha.py:811:25 ]
	s_lshl_b64 s[16:17], s[30:31], 5
.Ltmp150:
	.loc	1 806 19                        ; mha.py:806:19
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s2, s30
	.loc	1 0 0 is_stmt 0                 ; mha.py:0
	s_add_u32 s0, s24, s0
	s_addc_u32 s1, s25, s1
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[0:1]
	.loc	1 803 19 is_stmt 1              ; mha.py:803:19
	s_mul_i32 s0, s2, s15
	s_mul_hi_u32 s1, s2, s14
	s_add_i32 s1, s1, s0
	s_mul_i32 s2, s2, s14
	.loc	1 0 0 is_stmt 0                 ; mha.py:0
	s_add_u32 s0, s34, s2
	s_addc_u32 s1, s35, s1
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[0:1]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[0:1]
	s_bfe_i32 s1, s47, 0x10006
	s_movk_i32 s2, 0x110
	v_bfe_i32 v5, v54, 3, 1
	v_lshrrev_b32_e32 v6, 1, v60
	v_lshlrev_b32_e32 v1, 2, v56
	s_and_b32 s3, s1, 0x110
	v_mov_b32_e32 v2, 0x2fc
	v_bitop3_b32 v5, v5, v6, s2 bitop3:0x6c
	v_and_b32_e32 v6, 31, v54
	v_mov_b32_e32 v9, 0x420
	v_cmp_eq_u32_e32 vcc, 0, v60
	v_and_b32_e32 v33, 60, v54
	s_and_b32 s1, s1, 0x108
.Ltmp151:
	.loc	1 177 20 is_stmt 1              ; mha.py:177:20 @[ mha.py:811:25 ]
	v_mov_b32_e32 v0, 0x3fb8aa3b
	s_and_b32 s0, s47, 64
	v_bitop3_b32 v3, v1, s3, v2 bitop3:0x6c
	v_lshlrev_b32_e32 v4, 5, v54
	v_lshlrev_b32_e32 v7, 3, v6
	s_lshl_b32 s2, s42, 1
	v_lshlrev_b32_e32 v6, 1, v6
	v_cndmask_b32_e64 v9, v9, 0, vcc
	v_lshlrev_b32_e32 v35, 6, v33
	v_and_b32_e32 v37, 24, v58
	v_lshlrev_b32_e32 v33, 1, v33
	v_bitop3_b32 v1, v1, s1, v2 bitop3:0x6c
	v_mov_b32_e32 v2, 0x108
	v_mul_f32_e32 v39, s46, v0
.Ltmp152:
	.loc	1 606 11                        ; mha.py:606:11
	v_lshrrev_b32_e32 v0, 3, v60
	v_and_b32_e32 v4, 0x2e0, v4
	s_and_b32 s2, s2, 4
	s_lshl2_add_u32 s0, s0, 0
	v_bitop3_b32 v6, s47, v9, v6 bitop3:0x36
	s_lshl_b32 s3, s33, 5
	v_bitop3_b32 v33, v35, v33, v37 bitop3:0x36
	v_cndmask_b32_e64 v2, v2, 0, vcc
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_add_u32_e32 v4, 0, v4
	s_add_i32 s0, s0, s2
	s_lshl_b32 s2, s33, 7
	v_lshl_add_u32 v8, v55, 3, 0
	v_xor_b32_e32 v9, 8, v6
	v_xor_b32_e32 v10, 16, v6
	v_xor_b32_e32 v11, 24, v6
	v_xor_b32_e32 v12, 64, v6
	v_xor_b32_e32 v13, 0x48, v6
	v_xor_b32_e32 v14, 0x50, v6
	v_xor_b32_e32 v15, 0x58, v6
	v_xor_b32_e32 v33, s3, v33
	v_xor_b32_e32 v2, v2, v7
.Ltmp153:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	v_add_u32_e32 v40, s45, v0
	v_add_u32_e32 v0, s47, v59
	s_and_b32 s37, s37, 0xffff
	s_and_b32 s13, s13, 0xffff
	s_sub_i32 s24, s19, 32
	v_lshrrev_b32_e32 v41, 3, v0
	v_bfrev_b32_e32 v42, 1
	v_add_u32_e32 v43, 0, v3
	v_add_u32_e32 v44, v4, v5
	v_mov_b32_e32 v45, 0xff800000
	v_add_u32_e32 v46, s0, v7
	v_add_u32_e32 v47, s2, v8
	v_add_u32_e32 v48, 0, v6
	v_add_u32_e32 v49, 0, v9
	v_add_u32_e32 v50, 0, v10
	v_add_u32_e32 v51, 0, v11
	v_add_u32_e32 v52, 0, v12
	v_add_u32_e32 v53, 0, v13
	v_add_u32_e32 v58, 0, v14
	v_add_u32_e32 v59, 0, v15
	v_add_u32_e32 v60, 0, v33
	v_add_u32_e32 v61, 0, v1
	v_add_u32_e32 v62, 0, v2
	s_mov_b32 s14, s38
	s_mov_b32 s15, s39
	v_mov_b32_e32 v63, v38
.LBB0_22:                               ; =>This Inner Loop Header: Depth=1
.Ltmp154:
	.loc	1 33 16                         ; mha.py:33:16 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_add_u32_e32 v2, s45, v41
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e32 v3, 1, v32
	.loc	1 33 16                         ; mha.py:33:16 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_cmp_gt_i32_e64 s[0:1], s7, v2
.Ltmp155:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e32 v0, 1, v34
.Ltmp156:
	.loc	1 31 18                         ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e32 v1, 1, v36
.Ltmp157:
	.loc	1 170 28                        ; mha.py:170:28 @[ mha.py:811:25 ]
	v_add_u32_e32 v4, 16, v40
.Ltmp158:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_cndmask_b32_e64 v2, v42, v3, s[0:1]
.Ltmp159:
	.loc	1 170 28                        ; mha.py:170:28 @[ mha.py:811:25 ]
	v_cmp_le_i32_e64 s[2:3], s7, v4
.Ltmp160:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	v_cndmask_b32_e64 v0, v42, v0, s[0:1]
.Ltmp161:
	.loc	1 31 18                         ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	v_cndmask_b32_e64 v1, v42, v1, s[0:1]
.Ltmp162:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	buffer_load_dword v3, v2, s[36:39], 0 offen
.Ltmp163:
	.loc	1 34 18 is_stmt 0               ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	buffer_load_dword v4, v0, s[36:39], 0 offen
.Ltmp164:
	.loc	1 31 18 is_stmt 1               ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	buffer_load_dword v74, v1, s[12:15], 0 offen
.Ltmp165:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp166:
	.loc	1 168 27                        ; mha.py:168:27 @[ mha.py:811:25 ]
	s_add_i32 s25, s45, 32
	s_cmp_eq_u32 s24, s45
	s_cselect_b64 s[0:1], -1, 0
	.loc	1 170 28                        ; mha.py:170:28 @[ mha.py:811:25 ]
	v_cmp_le_i32_e32 vcc, s7, v40
	.loc	1 168 26                        ; mha.py:168:26 @[ mha.py:811:25 ]
	s_and_b64 s[26:27], s[22:23], s[0:1]
	.loc	1 261 9                         ; mha.py:261:9 @[ mha.py:811:25 ]
	v_ashrrev_i32_e32 v33, 31, v32
	.loc	1 263 13                        ; mha.py:263:13 @[ mha.py:811:25 ]
	v_ashrrev_i32_e32 v35, 31, v34
.Ltmp167:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	s_and_b64 s[0:1], s[26:27], s[2:3]
.Ltmp168:
	.loc	1 171 20                        ; mha.py:171:20 @[ mha.py:811:25 ]
	s_and_b64 vcc, s[26:27], vcc
	.loc	1 264 9                         ; mha.py:264:9 @[ mha.py:811:25 ]
	v_ashrrev_i32_e32 v37, 31, v36
	.loc	1 261 9                         ; mha.py:261:9 @[ mha.py:811:25 ]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[4:5]
	.loc	1 263 13                        ; mha.py:263:13 @[ mha.py:811:25 ]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[4:5]
	.loc	1 264 9                         ; mha.py:264:9 @[ mha.py:811:25 ]
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[16:17]
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	v_add_u32_e32 v40, 32, v40
	s_mov_b32 s45, s25
	s_cmp_lt_i32 s25, s19
.Ltmp169:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	s_waitcnt vmcnt(1)
	ds_write_b32 v43, v4
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[64:67], v44
.Ltmp170:
	.loc	1 34 18 is_stmt 0               ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v43, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp171:
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:811:25 ]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[64:67], v[16:19], 0
.Ltmp172:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	ds_read_b128 v[64:67], v44
.Ltmp173:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:811:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:811:25 ]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[64:67], v[20:23], v[0:15]
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	s_nop 11
	v_mul_f32_e32 v9, v39, v9
	v_mul_f32_e32 v1, v39, v1
	v_mul_f32_e32 v0, v39, v0
	v_mul_f32_e32 v3, v39, v3
	v_mul_f32_e32 v5, v39, v5
	v_mul_f32_e32 v4, v39, v4
	v_mul_f32_e32 v11, v39, v11
	v_mul_f32_e32 v10, v39, v10
	v_mul_f32_e32 v13, v39, v13
	v_mul_f32_e32 v12, v39, v12
	v_mul_f32_e32 v14, v39, v14
	v_mul_f32_e32 v2, v39, v2
	v_mul_f32_e32 v7, v39, v7
	v_mul_f32_e32 v6, v39, v6
	v_mul_f32_e32 v8, v39, v8
	v_mul_f32_e32 v15, v39, v15
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_cndmask_b32_e32 v1, v1, v45, vcc
	v_cndmask_b32_e32 v0, v0, v45, vcc
	v_cndmask_b32_e32 v3, v3, v45, vcc
	v_cndmask_b32_e32 v5, v5, v45, vcc
	v_cndmask_b32_e32 v4, v4, v45, vcc
	v_cndmask_b32_e64 v9, v9, v45, s[0:1]
	v_cndmask_b32_e64 v11, v11, v45, s[0:1]
	v_cndmask_b32_e64 v10, v10, v45, s[0:1]
	v_cndmask_b32_e64 v13, v13, v45, s[0:1]
	v_cndmask_b32_e64 v12, v12, v45, s[0:1]
	v_cndmask_b32_e64 v14, v14, v45, s[0:1]
	v_cndmask_b32_e32 v2, v2, v45, vcc
	v_cndmask_b32_e32 v7, v7, v45, vcc
	v_cndmask_b32_e32 v6, v6, v45, vcc
	v_cndmask_b32_e64 v8, v8, v45, s[0:1]
	v_cndmask_b32_e64 v15, v15, v45, s[0:1]
.Ltmp174:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e32 v33, v0, v1
	v_max3_f32 v35, v3, v4, v5
	v_max3_f32 v38, v9, v10, v11
	v_max3_f32 v64, v12, v13, v14
	v_max3_f32 v37, v6, v7, v8
	v_max3_f32 v33, v33, v2, v35
	v_max3_f32 v35, v38, v64, v15
	v_max3_f32 v33, v33, v37, v35
.Ltmp175:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ]
	v_mov_b32_e32 v35, v33
	s_nop 1
	v_permlane32_swap_b32_e32 v33, v35
.Ltmp176:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:811:25 ]
	v_max3_f32 v38, v63, v33, v35
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	v_pk_add_f32 v[0:1], v[0:1], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[2:3], v[2:3], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[6:7], v[6:7], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9], v[8:9], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[10:11], v[10:11], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[12:13], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15], v[14:15], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:811:25 ]
	v_sub_f32_e32 v33, v63, v38
	.loc	1 212 13                        ; mha.py:212:13 @[ mha.py:811:25 ]
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v2, v2
	v_exp_f32_e32 v3, v3
	v_exp_f32_e32 v4, v4
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v8, v8
	v_exp_f32_e32 v9, v9
	v_exp_f32_e32 v10, v10
	v_exp_f32_e32 v11, v11
	v_exp_f32_e32 v12, v12
	v_exp_f32_e32 v13, v13
	v_exp_f32_e32 v14, v14
	v_exp_f32_e32 v15, v15
	.loc	1 241 17                        ; mha.py:241:17 @[ mha.py:811:25 ]
	v_exp_f32_e32 v33, v33
.Ltmp177:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_pk_add_f32 v[64:65], v[0:1], v[2:3]
	v_pk_add_f32 v[66:67], v[4:5], v[6:7]
	v_pk_add_f32 v[68:69], v[8:9], v[10:11]
	v_pk_add_f32 v[70:71], v[12:13], v[14:15]
.Ltmp178:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:811:25 ]
	ds_write_b32 v46, v33
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[72:73], v47
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:811:25 ]
	v_cvt_pk_bf16_f32 v0, v0, s0
	v_cvt_pk_bf16_f32 v8, v8, s0
	v_cvt_pk_bf16_f32 v9, v9, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_cvt_pk_bf16_f32 v12, v12, s0
	v_cvt_pk_bf16_f32 v13, v13, s0
	v_cvt_pk_bf16_f32 v6, v6, s0
	v_cvt_pk_bf16_f32 v7, v7, s0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_cvt_pk_bf16_f32 v4, v4, s0
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_cvt_pk_bf16_f32 v14, v14, s0
	v_cvt_pk_bf16_f32 v15, v15, s0
	ds_write_b16 v48, v0
	ds_write_b16 v48, v8 offset:4096
	ds_write_b16 v49, v1 offset:256
	ds_write_b16 v49, v9 offset:4352
	ds_write_b16 v50, v2 offset:512
	ds_write_b16 v50, v10 offset:4608
	ds_write_b16 v51, v3 offset:768
	ds_write_b16 v51, v11 offset:4864
	ds_write_b16 v52, v4 offset:2048
	ds_write_b16 v52, v12 offset:6144
	ds_write_b16 v53, v5 offset:2304
	ds_write_b16 v53, v13 offset:6400
	ds_write_b16 v58, v6 offset:2560
	ds_write_b16 v58, v14 offset:6656
	ds_write_b16 v59, v7 offset:2816
	ds_write_b16 v59, v15 offset:6912
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[6:7], v60
	ds_read_b64_tr_b16 v[8:9], v60 offset:4096
	ds_read_b64_tr_b16 v[12:13], v60 offset:4224
	ds_read_b64_tr_b16 v[10:11], v60 offset:128
.Ltmp179:
	.loc	1 31 18                         ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b32 v61, v74
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[74:75], v62
	ds_read_b64_tr_b16 v[76:77], v62 offset:512
.Ltmp180:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_pk_add_f32 v[0:1], v[64:65], v[66:67]
	v_pk_add_f32 v[2:3], v[68:69], v[70:71]
.Ltmp181:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:811:25 ]
	v_mul_f32_e32 v33, v57, v33
.Ltmp182:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_pk_add_f32 v[0:1], v[0:1], v[2:3]
.Ltmp183:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:811:25 ]
	v_pk_mul_f32 v[2:3], v[26:27], v[72:73] op_sel_hi:[1,0]
.Ltmp184:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ] ]
	v_pk_add_f32 v[4:5], v[0:1], v[0:1] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp185:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:811:25 ]
	v_pk_mul_f32 v[0:1], v[24:25], v[72:73] op_sel_hi:[1,0]
.Ltmp186:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_mov_b32_e32 v5, v4
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v5
.Ltmp187:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:811:25 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[74:77], v[6:9], v[0:3]
	v_mov_b32_e32 v63, v38
	s_nop 1
	v_pk_mul_f32 v[2:3], v[30:31], v[72:73] op_sel:[0,1]
	v_pk_mul_f32 v[0:1], v[28:29], v[72:73] op_sel:[0,1]
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[28:31], v[74:77], v[10:13], v[0:3]
.Ltmp188:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ] ]
	s_nop 2
	v_add_f32_e32 v0, v4, v5
.Ltmp189:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:811:25 ]
	v_add_f32_e32 v57, v0, v33
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	s_cbranch_scc1 .LBB0_22
.Ltmp190:
.LBB0_23:                               ; %.loopexit
	.loc	1 861 15                        ; mha.py:861:15
	v_div_scale_f32 v0, s[0:1], v57, v57, 1.0
	v_rcp_f32_e32 v1, v0
	v_div_scale_f32 v3, vcc, 1.0, v57, 1.0
	.loc	1 862 11                        ; mha.py:862:11
	s_lshl_b32 s0, s33, 8
	s_lshl_b32 s1, s42, 1
	.loc	1 861 15                        ; mha.py:861:15
	v_fma_f32 v4, -v0, v1, 1.0
	v_fmac_f32_e32 v1, v4, v1
	v_mul_f32_e32 v4, v3, v1
	v_fma_f32 v5, -v0, v4, v3
	v_fmac_f32_e32 v4, v5, v1
	v_fma_f32 v0, -v0, v4, v3
	.loc	1 862 11                        ; mha.py:862:11
	s_and_b32 s0, s0, 0x100
	s_and_b32 s1, s1, 4
	.loc	1 861 15                        ; mha.py:861:15
	v_div_fmas_f32 v0, v0, v1, v4
	.loc	1 862 11                        ; mha.py:862:11
	v_and_b32_e32 v3, 31, v54
	s_or_b32 s1, s1, s0
	.loc	1 861 15                        ; mha.py:861:15
	v_div_fixup_f32 v0, v0, v57, 1.0
	.loc	1 862 11                        ; mha.py:862:11
	v_lshl_add_u32 v1, v3, 3, s1
	s_mov_b32 s0, 0x800000
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v1, v0
	.loc	1 888 29                        ; mha.py:888:29
	v_mov_b32_e32 v1, 0x42000000
	v_cmp_gt_f32_e32 vcc, s0, v57
	.loc	1 862 11                        ; mha.py:862:11
	s_lshl_b32 s5, s33, 7
	v_lshl_add_u32 v0, v55, 3, s5
	.loc	1 888 29                        ; mha.py:888:29
	v_cndmask_b32_e32 v4, 0, v1, vcc
	v_cndmask_b32_e64 v1, 0, 32, vcc
	v_ldexp_f32 v1, v57, v1
	v_log_f32_e32 v5, v1
	.loc	1 862 11                        ; mha.py:862:11
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[0:1], v0
	.loc	1 884 21                        ; mha.py:884:21
	s_sub_i32 s4, s44, s6
	.loc	1 900 13                        ; mha.py:900:13
	s_mul_i32 s1, s43, s10
	.loc	1 901 15                        ; mha.py:901:15
	s_mul_i32 s2, s11, s41
	.loc	1 884 21                        ; mha.py:884:21
	s_add_i32 s0, s4, 0x80
	.loc	1 900 13                        ; mha.py:900:13
	s_add_u32 s7, s1, s2
	.loc	1 380 34                        ; mha.py:380:34
	v_and_b32_e32 v2, 0x7f, v56
	.loc	1 888 29                        ; mha.py:888:29
	v_sub_f32_e32 v4, v5, v4
	.loc	1 905 12                        ; mha.py:905:12
	s_cmp_lt_i32 s0, 1
	.loc	1 380 14                        ; mha.py:380:14
	v_or_b32_e32 v7, s44, v2
	.loc	1 888 23                        ; mha.py:888:23
	v_add_f32_e32 v4, v38, v4
	.loc	1 905 12                        ; mha.py:905:12
	s_cselect_b64 s[2:3], -1, 0
	.loc	1 890 9                         ; mha.py:890:9
	v_mul_f32_e32 v4, 0x3f317218, v4
	s_mov_b64 s[0:1], -1
	.loc	1 905 9                         ; mha.py:905:9
	s_and_b64 vcc, exec, s[2:3]
	v_lshl_add_u32 v6, v3, 2, s5
	v_lshl_add_u32 v5, v2, 2, 0
	v_add_lshl_u32 v3, v7, s7, 2
	s_cbranch_vccnz .LBB0_25
; %bb.24:
	.loc	1 906 44                        ; mha.py:906:44
	s_sub_i32 s0, 0, s4
	.loc	1 908 13                        ; mha.py:908:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v6, v4
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v7, v5
	s_and_b32 s13, s21, 0xffff
	s_bitcmp0_b32 s40, 7
	.loc	1 907 24                        ; mha.py:907:24
	v_cmp_gt_i32_e32 vcc, s0, v2
	.loc	1 908 13                        ; mha.py:908:13
	s_cselect_b64 s[0:1], -1, 0
	v_bfrev_b32_e32 v2, 1
	s_and_b64 vcc, s[0:1], vcc
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_mov_b32 s12, s20
	v_cndmask_b32_e32 v2, v2, v3, vcc
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v7, v2, s[12:15], 0 offen
.LBB0_25:                               ; %Flow
	.loc	1 0 13 is_stmt 0                ; mha.py:0:13
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_27
; %bb.26:
	.loc	1 912 13 is_stmt 1              ; mha.py:912:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v6, v4
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v4, v5
	s_and_b32 s21, s21, 0xffff
	s_bitcmp0_b32 s40, 7
	v_bfrev_b32_e32 v2, 1
	s_cselect_b64 vcc, -1, 0
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v2, v2, v3, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v4, v2, s[20:23], 0 offen
.LBB0_27:
	.loc	1 380 34                        ; mha.py:380:34
	v_lshl_or_b32 v8, s33, 4, v55
	.loc	1 586 11                        ; mha.py:586:11
	v_lshrrev_b32_e32 v10, 2, v54
	.loc	1 918 9                         ; mha.py:918:9
	s_mul_i32 s4, s43, s8
	.loc	1 919 11                        ; mha.py:919:11
	s_mul_i32 s5, s9, s41
	.loc	1 862 11                        ; mha.py:862:11
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[30:31], v[0:1] op_sel:[0,1]
	v_pk_mul_f32 v[4:5], v[28:29], v[0:1] op_sel:[0,1]
	v_pk_mul_f32 v[6:7], v[26:27], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[24:25], v[0:1] op_sel_hi:[1,0]
	.loc	1 380 14                        ; mha.py:380:14
	v_or_b32_e32 v8, s44, v8
	.loc	1 586 11                        ; mha.py:586:11
	v_and_b32_e32 v10, 12, v10
	.loc	1 918 9                         ; mha.py:918:9
	s_add_i32 s4, s4, s5
	.loc	1 679 18                        ; mha.py:679:18
	v_cmp_gt_i32_e64 s[0:1], s6, v8
	.loc	1 929 10                        ; mha.py:929:10
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_cvt_pk_bf16_f32 v1, v6, v7
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_cvt_pk_bf16_f32 v5, v2, v3
	.loc	1 930 14                        ; mha.py:930:14
	v_mul_lo_u32 v2, v8, s18
	v_add_u32_e32 v6, s4, v10
	.loc	1 380 14                        ; mha.py:380:14
	v_or_b32_e32 v9, 64, v8
	.loc	1 930 5                         ; mha.py:930:5
	v_add_lshl_u32 v2, v6, v2, 1
	v_bfrev_b32_e32 v7, 1
	.loc	1 925 5                         ; mha.py:925:5
	s_or_b64 s[0:1], s[2:3], s[0:1]
	.loc	1 679 18                        ; mha.py:679:18
	v_cmp_gt_i32_e32 vcc, s6, v9
	.loc	1 930 14                        ; mha.py:930:14
	v_mul_lo_u32 v3, v9, s18
	.loc	1 930 5 is_stmt 0               ; mha.py:930:5
	s_and_b32 s29, s29, 0xffff
	s_mov_b32 s31, 0x27000
	s_mov_b32 s30, 0x7ffffffe
	v_cndmask_b32_e64 v2, v7, v2, s[0:1]
	buffer_store_dwordx2 v[0:1], v2, s[28:31], 0 offen
	v_add_lshl_u32 v0, v6, v3, 1
	.loc	1 925 5 is_stmt 1               ; mha.py:925:5
	s_or_b64 vcc, s[2:3], vcc
	.loc	1 930 5                         ; mha.py:930:5
	v_cndmask_b32_e32 v0, v7, v0, vcc
	buffer_store_dwordx2 v[4:5], v0, s[28:31], 0 offen
	.loc	1 297 1                         ; mha.py:297:1
	s_endpgm
.Ltmp191:
.Lfunc_end0:
	.size	_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0, .Lfunc_end0-_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 432
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 8
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 169
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 92
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
                                        ; -- End function
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_vgpr, 92
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_agpr, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.numbered_sgpr, 64
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_named_barrier, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.private_seg_size, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.uses_vcc, 1
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.uses_flat_scratch, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_dyn_sized_stack, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_recursion, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7832
; TotalNumSgprs: 70
; NumVgprs: 92
; NumAgprs: 0
; TotalNumVgprs: 92
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 169
; AccumOffset: 92
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 22
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
	.byte	1                               ; DW_CHILDREN_yes
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
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	8                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	9                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
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
	.byte	1                               ; Abbrev [1] 0xb:0x12c DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x106 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	375                             ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x4e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	510                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x5b:0x65 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	746                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x68:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	140                             ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x74:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	142                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x80:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	150                             ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x8c:0x19 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	209                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x98:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges7                 ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.byte	191                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0xa5:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges8                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	220                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xb1:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges9                 ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0xc0:0x75 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges10                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	811                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xcd:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges11                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	140                             ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xd9:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges12                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	142                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xe5:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges13                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	150                             ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0xf1:0x29 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp174                        ; DW_AT_low_pc
	.long	.Ltmp176-.Ltmp174               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	209                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	9                               ; Abbrev [9] 0x105:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp174                        ; DW_AT_low_pc
	.long	.Ltmp175-.Ltmp174               ; DW_AT_high_pc
	.byte	3                               ; DW_AT_call_file
	.byte	191                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x11a:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges14                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	220                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x126:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges15                ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges8:
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges9:
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges10:
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges11:
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges12:
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges13:
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges14:
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	.Ltmp187-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges15:
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp188-.Lfunc_begin0
	.quad	.Ltmp189-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"mha.py"                        ; string offset=7 ; mha.py
.Linfo_string2:
	.asciz	"/root/aiter/aiter/ops/triton/_triton_kernels/attention" ; string offset=14 ; /root/aiter/aiter/ops/triton/_triton_kernels/attention
.Linfo_string3:
	.asciz	"_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0" ; string offset=69 ; _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .offset:         108
        .size:           4
        .value_kind:     by_value
      - .offset:         112
        .size:           4
        .value_kind:     by_value
      - .offset:         116
        .size:           4
        .value_kind:     by_value
      - .offset:         120
        .size:           4
        .value_kind:     by_value
      - .offset:         124
        .size:           4
        .value_kind:     by_value
      - .offset:         128
        .size:           4
        .value_kind:     by_value
      - .offset:         132
        .size:           4
        .value_kind:     by_value
      - .offset:         136
        .size:           4
        .value_kind:     by_value
      - .offset:         140
        .size:           4
        .value_kind:     by_value
      - .offset:         144
        .size:           4
        .value_kind:     by_value
      - .offset:         148
        .size:           4
        .value_kind:     by_value
      - .offset:         152
        .size:           4
        .value_kind:     by_value
      - .offset:         156
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         168
        .size:           8
        .value_kind:     global_buffer
      - .offset:         176
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         180
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         184
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         188
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         190
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         192
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         194
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         196
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         198
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         216
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         224
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         232
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         240
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         256
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         264
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         272
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         280
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         288
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         296
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
      - .offset:         376
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 432
    .max_flat_workgroup_size: 256
    .name:           _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
    .private_segment_fixed_size: 0
    .sgpr_count:     70
    .sgpr_spill_count: 0
    .symbol:         _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     92
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
