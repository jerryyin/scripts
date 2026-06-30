	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0 ; -- Begin function _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
	.p2align	8
	.type	_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0,@function
_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0: ; @_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.27:
	.file	1 "/root/aiter/aiter/ops/triton/_triton_kernels/attention" "mha.py"
	.loc	1 297 0 prologue_end            ; mha.py:297:0
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.28:
.LBB0_0:
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
	v_rcp_iflag_f32_e32 v1, v1
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
	v_rcp_iflag_f32_e32 v1, v1
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
	v_rcp_iflag_f32_e32 v1, v1
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
	s_lshl_b32 s43, s2, 7
.Ltmp8:
	.loc	1 19 13                         ; mha.py:19:13 @[ mha.py:510:16 ]
	s_add_i32 s2, s7, 31
.Ltmp9:
	.loc	1 377 13                        ; mha.py:377:13
	s_sub_i32 s42, s3, s11
.Ltmp10:
	.loc	1 19 12                         ; mha.py:19:12 @[ mha.py:510:16 ]
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_add_i32 s2, s2, s3
	s_ashr_i32 s47, s2, 5
.Ltmp11:
	.loc	1 563 22                        ; mha.py:563:22
	s_bfe_u32 s2, s41, 0x3001c
	s_add_i32 s2, s41, s2
	s_sext_i32_i16 s10, s2
	s_lshr_b32 s2, s10, 3
	.loc	1 380 34                        ; mha.py:380:34
	s_lshl_b32 s44, s33, 6
	.loc	1 574 14                        ; mha.py:574:14
	s_bfe_i64 s[2:3], s[2:3], 0x100000
	s_ashr_i32 s2, s10, 3
	.loc	1 380 34                        ; mha.py:380:34
	v_or_b32_e32 v56, s44, v59
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
	s_mul_i32 s2, s42, s22
	.loc	1 380 14                        ; mha.py:380:14
	v_or_b32_e32 v2, s43, v2
	.loc	1 575 14                        ; mha.py:575:14
	s_add_i32 s16, s16, s3
	.loc	1 582 9                         ; mha.py:582:9
	s_add_i32 s2, s2, s11
	.loc	1 602 9                         ; mha.py:602:9
	s_mul_i32 s11, s42, s25
	.loc	1 588 14                        ; mha.py:588:14
	v_mul_lo_u32 v3, v2, s24
	.loc	1 382 14                        ; mha.py:382:14
	v_and_b32_e32 v1, 8, v58
	.loc	1 602 9                         ; mha.py:602:9
	s_mul_hi_u32 s3, s42, s25
	s_add_u32 s34, s11, s14
	.loc	1 588 14                        ; mha.py:588:14
	v_add_u32_e32 v3, s2, v3
	.loc	1 602 9                         ; mha.py:602:9
	s_addc_u32 s35, s3, s10
	.loc	1 622 9                         ; mha.py:622:9
	s_mul_hi_u32 s2, s42, s0
	s_mul_i32 s0, s42, s0
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
	s_min_i32 s49, s0, s47
	.loc	1 729 21                        ; mha.py:729:21
	s_sub_i32 s48, s47, s49
	.loc	1 744 8                         ; mha.py:744:8
	s_cmp_lt_i32 s48, 1
	.loc	1 628 14                        ; mha.py:628:14
	v_mad_i64_i32 v[36:37], s[0:1], v14, s30, v[10:11]
	.loc	1 744 5                         ; mha.py:744:5
	s_cbranch_scc1 .LBB0_4
; %bb.1:
	.loc	1 745 33                        ; mha.py:745:33
	s_lshl_b32 s45, s48, 5
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
	s_bfe_i32 s2, s40, 0x10006
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
	v_lshlrev_b32_e32 v61, 2, v0
	ds_bpermute_b32 v11, v61, v6
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
	ds_bpermute_b32 v11, v61, v4
.Ltmp20:
	.loc	1 263 13 is_stmt 1              ; mha.py:263:13 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[48:49], v[4:5], 0, s[26:27]
.Ltmp21:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	ds_bpermute_b32 v4, v61, v46
	s_lshl_b32 s53, s33, 8
	v_and_b32_e32 v8, 1, v8
.Ltmp22:
	.loc	1 628 14                        ; mha.py:628:14
	v_lshl_add_u64 v[2:3], v[36:37], 0, s[24:25]
.Ltmp23:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 s50, s53, 0
	v_cmp_eq_u32_e32 vcc, 1, v8
	v_cmp_gt_i32_e64 s[2:3], s45, 32
.Ltmp24:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_lshlrev_b32_e32 v10, 1, v2
.Ltmp25:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_and_b32 s9, s37, 0xffff
	s_mov_b32 s8, s36
	s_add_i32 m0, s50, 0x3840
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
	ds_bpermute_b32 v4, v61, v48
	s_add_i32 m0, s50, 0x2c40
	v_cndmask_b32_e32 v8, v1, v8, vcc
.Ltmp31:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	; asyncmark
.Ltmp32:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	buffer_load_dword v8, s[8:11], 0 offen lds
.Ltmp33:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_and_b32 s17, s13, 0xffff
	s_add_i32 m0, s50, 0x2000
.Ltmp34:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_and_b32_e32 v2, 1, v2
.Ltmp35:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_mov_b32 s16, s12
	s_mov_b32 s18, s10
	s_mov_b32 s19, s11
	v_cndmask_b32_e64 v8, v1, v10, s[0:1]
.Ltmp36:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cmp_gt_i32 s45, 32
.Ltmp37:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_cmp_eq_u32_e64 s[2:3], 1, v2
.Ltmp38:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	; asyncmark
.Ltmp39:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	buffer_load_dword v8, s[16:19], 0 offen lds
.Ltmp40:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cselect_b64 vcc, -1, 0
.Ltmp41:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s50, 0x3c40
	v_cndmask_b32_e64 v2, v1, v3, s[2:3]
.Ltmp42:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	; asyncmark
.Ltmp43:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dword v2, s[8:11], 0 offen lds
.Ltmp44:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_lshlrev_b32_e32 v2, 1, v4
.Ltmp45:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_lshlrev_b32_e32 v8, 1, v44
.Ltmp46:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s50, 0x3040
	v_cndmask_b32_e64 v2, v1, v2, s[2:3]
.Ltmp47:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	; asyncmark
.Ltmp48:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	buffer_load_dword v2, s[8:11], 0 offen lds
.Ltmp49:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s50, 0x2420
	v_cndmask_b32_e32 v1, v1, v8, vcc
.Ltmp50:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	; asyncmark
.Ltmp51:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	buffer_load_dword v1, s[16:19], 0 offen lds
.Ltmp52:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_sub_i32 s55, s45, 64
	.loc	1 177 20                        ; mha.py:177:20 @[ mha.py:746:25 ]
	v_mov_b32_e32 v1, 0x3fb8aa3b
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cmp_gt_i32 s55, 0
	v_and_b32_e32 v43, 8, v54
	v_lshlrev_b32_e32 v39, 3, v59
.Ltmp53:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	; asyncmark
.Ltmp54:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	; wait_asyncmark(3)
	s_waitcnt vmcnt(3) lgkmcnt(0)
	s_barrier
.Ltmp55:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cbranch_scc1 .LBB0_5
; %bb.2:                                ; %.._crit_edge_crit_edge
.Ltmp56:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_lshlrev_b32_e32 v2, 5, v54
	v_and_b32_e32 v62, 0x2e0, v2
	v_lshrrev_b32_e32 v42, 1, v60
	v_mul_f32_e32 v40, s46, v1
	s_mov_b32 s3, 1
	s_cbranch_execz .LBB0_6
.Ltmp57:
; %bb.3:
	.loc	1 0 18 is_stmt 0                ; mha.py:0:18
	v_mov_b32_e32 v31, 0
	s_add_i32 s54, 0, 0x3840
	s_add_i32 s50, 0, 0x3c40
	s_add_i32 s56, 0, 0x2c40
	s_add_i32 s51, 0, 0x3040
	s_add_i32 s57, 0, 0x2000
	s_add_i32 s52, 0, 0x2420
	v_mov_b32_e32 v38, 0xff800000
	v_mov_b32_e32 v57, 1.0
	v_mov_b32_e32 v30, v31
	v_mov_b32_e32 v29, v31
	v_mov_b32_e32 v28, v31
	v_mov_b32_e32 v27, v31
	v_mov_b32_e32 v26, v31
	v_mov_b32_e32 v25, v31
	v_mov_b32_e32 v24, v31
	s_branch .LBB0_8
.LBB0_4:
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
	s_branch .LBB0_18
.LBB0_5:
                                        ; implicit-def: $vgpr42
                                        ; implicit-def: $vgpr62
	.loc	1 0 5 is_stmt 0                 ; mha.py:0:5
	v_mul_f32_e32 v40, s46, v1
	s_mov_b32 s3, 1
.LBB0_6:                                ; %.lr.ph
	v_lshlrev_b32_e32 v1, 5, v54
	v_and_b32_e32 v62, 0x2e0, v1
	v_bfe_i32 v1, v54, 3, 1
	v_lshrrev_b32_e32 v42, 1, v60
	s_movk_i32 s8, 0x110
	v_bitop3_b32 v1, v1, v42, s8 bitop3:0x6c
	v_or_b32_e32 v63, v1, v62
	v_and_b32_e32 v1, 31, v54
	v_mov_b32_e32 v4, 0x420
	v_cmp_eq_u32_e32 vcc, 0, v60
	v_lshl_add_u32 v2, v1, 2, 0
	v_lshlrev_b32_e32 v1, 1, v1
	v_cndmask_b32_e64 v4, v4, 0, vcc
	v_bitop3_b32 v4, s44, v4, v1 bitop3:0x36
	v_and_b32_e32 v1, 60, v54
	v_lshlrev_b32_e32 v12, 6, v1
	v_and_b32_e32 v13, 24, v58
	v_lshlrev_b32_e32 v1, 1, v1
	s_lshl_b32 s8, s33, 5
	v_bitop3_b32 v1, v12, v1, v13 bitop3:0x36
	v_xor_b32_e32 v12, s8, v1
	v_lshrrev_b64 v[0:1], v0, exec
.Ltmp58:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_add_i32 s58, s44, 0
	s_lshl_b32 s59, s33, 7
	v_lshlrev_b32_e32 v3, 2, v55
	v_xor_b32_e32 v5, 0x108, v4
	v_xor_b32_e32 v6, 0x210, v4
	v_xor_b32_e32 v7, 0x318, v4
	v_xor_b32_e32 v8, 0x840, v4
	v_xor_b32_e32 v9, 0x948, v4
	v_xor_b32_e32 v10, 0xa50, v4
	v_xor_b32_e32 v11, 0xb58, v4
	v_mov_b32_e32 v26, 0
	v_and_b32_e32 v0, 1, v0
	s_mov_b32 s2, 0
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v41, v40
	v_mov_b32_e32 v50, v40
	v_mov_b32_e32 v51, v40
	s_add_i32 s54, 0, 0x3840
	s_add_i32 s50, 0, 0x3c40
	s_add_i32 s56, 0, 0x2c40
	s_add_i32 s51, 0, 0x3040
	s_add_i32 s57, 0, 0x2000
	s_add_i32 s52, 0, 0x2420
	v_mov_b32_e32 v52, v40
	v_mov_b32_e32 v53, v40
	v_mov_b32_e32 v65, 0xff800000
	v_mov_b32_e32 v57, 1.0
	v_cmp_eq_u32_e32 vcc, 1, v0
	v_bfrev_b32_e32 v64, 1
	s_mov_b32 s8, s36
	s_mov_b32 s16, s12
	s_mov_b32 s18, s10
	s_mov_b32 s19, s11
	v_add_u32_e32 v66, s59, v2
	v_add_u32_e32 v67, s58, v3
	v_add_u32_e32 v68, 0, v4
	v_add_u32_e32 v69, 0, v5
	v_add_u32_e32 v70, 0, v6
	v_add_u32_e32 v71, 0, v7
	v_add_u32_e32 v72, 0, v8
	v_add_u32_e32 v73, 0, v9
	v_add_u32_e32 v74, 0, v10
	v_add_u32_e32 v75, 0, v11
	v_add_u32_e32 v76, 0, v12
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
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_add_i32 s3, s3, 1
	.loc	1 263 13                        ; mha.py:263:13 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[48:49], v[48:49], 0, s[26:27]
.Ltmp59:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	ds_bpermute_b32 v2, v61, v46
.Ltmp60:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cmp_lt_i32 s3, 3
.Ltmp61:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	ds_bpermute_b32 v3, v61, v48
	s_mov_b32 s58, s54
	s_mov_b32 s54, s50
	s_mov_b32 s50, s56
.Ltmp62:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cselect_b32 s3, s3, 0
	s_mov_b32 s56, s51
	s_mov_b32 s51, s57
.Ltmp63:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v1, s50, v63
.Ltmp64:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_lshl_b32 s50, s3, 10
	s_mov_b32 s57, s52
.Ltmp65:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v38, s51, v39
	s_lshl_b32 s51, s3, 5
.Ltmp66:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 s52, s50, 0
	v_add_u32_e32 v0, s58, v63
.Ltmp67:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_and_b32 s58, s51, 0xfffffe0
.Ltmp68:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 s59, s52, s53
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v2, 1, v2
.Ltmp69:
	.loc	1 264 9 is_stmt 1               ; mha.py:264:9 @[ mha.py:746:25 ]
	v_ashrrev_i32_e32 v45, 31, v44
.Ltmp70:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 s50, s52, 0x3840
.Ltmp71:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_add_i32 s51, s52, 0x2c40
.Ltmp72:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_add_i32 s52, s52, s58
.Ltmp73:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s59, 0x3840
.Ltmp74:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v3
.Ltmp75:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_cndmask_b32_e32 v2, v64, v2, vcc
.Ltmp76:
	.loc	1 264 9 is_stmt 1               ; mha.py:264:9 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[44:45], v[44:45], 0, s[38:39]
.Ltmp77:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_cndmask_b32_e32 v3, v64, v3, vcc
.Ltmp78:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	buffer_load_dword v2, s[8:11], 0 offen lds
.Ltmp79:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s59, 0x2c40
.Ltmp80:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_addk_i32 s52, 0x2000
	v_lshlrev_b32_e32 v45, 1, v44
.Ltmp81:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	; asyncmark
.Ltmp82:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	buffer_load_dword v3, s[8:11], 0 offen lds
.Ltmp83:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_add_i32 m0, s52, s53
.Ltmp84:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	ds_read_b128 v[78:81], v0
.Ltmp85:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	; asyncmark
.Ltmp86:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	buffer_load_dword v45, s[16:19], 0 offen lds
.Ltmp87:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	ds_read_b128 v[82:85], v1
.Ltmp88:
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[82:85], v[16:19], 0
.Ltmp89:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	; asyncmark
	ds_read_b64_tr_b16 v[88:89], v38
	ds_read_b64_tr_b16 v[90:91], v38 offset:512
.Ltmp90:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_add_i32 s2, s2, 32
	s_cmp_lt_i32 s2, s55
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[78:81], v[20:23], v[0:15]
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	s_nop 11
	v_pk_mul_f32 v[4:5], v[52:53], v[4:5]
	v_pk_mul_f32 v[2:3], v[52:53], v[2:3]
	v_pk_mul_f32 v[0:1], v[40:41], v[0:1]
	v_mul_f32_e32 v45, v40, v8
	v_mov_b32_e32 v78, v11
	v_mov_b32_e32 v79, v12
	v_mov_b32_e32 v8, v9
	v_mov_b32_e32 v9, v10
	v_mov_b32_e32 v10, v13
	v_mov_b32_e32 v11, v14
	v_pk_mul_f32 v[12:13], v[52:53], v[78:79]
	v_pk_mul_f32 v[8:9], v[50:51], v[8:9]
.Ltmp91:
	.file	3 "/root/triton/python/triton/language" "standard.py"
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e32 v14, v0, v1
	v_max3_f32 v38, v3, v4, v5
.Ltmp92:
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[10:11], v[52:53], v[10:11]
	v_pk_mul_f32 v[6:7], v[52:53], v[6:7]
	v_mul_f32_e32 v15, v40, v15
.Ltmp93:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max3_f32 v49, v8, v9, v12
	v_max3_f32 v14, v14, v2, v38
	v_max3_f32 v38, v13, v10, v11
	v_max3_f32 v47, v6, v7, v45
	v_max3_f32 v38, v49, v38, v15
	v_max3_f32 v14, v14, v47, v38
.Ltmp94:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v38, v14
	s_nop 1
	v_permlane32_swap_b32_e32 v14, v38
.Ltmp95:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:746:25 ]
	v_max3_f32 v38, v65, v14, v38
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_sub_f32_e32 v14, v45, v38
	v_pk_add_f32 v[8:9], v[8:9], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[12:13], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[10:11], v[10:11], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e32 v45, v15, v38
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:746:25 ]
	v_sub_f32_e32 v47, v65, v38
	.loc	1 212 13                        ; mha.py:212:13 @[ mha.py:746:25 ]
	v_exp_f32_e32 v15, v8
	v_exp_f32_e32 v8, v9
	v_exp_f32_e32 v9, v12
	v_exp_f32_e32 v12, v13
	v_exp_f32_e32 v13, v10
	v_exp_f32_e32 v10, v11
	v_exp_f32_e32 v11, v45
	.loc	1 241 17                        ; mha.py:241:17 @[ mha.py:746:25 ]
	v_exp_f32_e32 v45, v47
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_pk_add_f32 v[0:1], v[0:1], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[2:3], v[2:3], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[6:7], v[6:7], v[38:39] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:746:25 ]
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v2, v2
	v_exp_f32_e32 v3, v3
	v_exp_f32_e32 v4, v4
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v14, v14
	.loc	1 246 15 is_stmt 1              ; mha.py:246:15 @[ mha.py:746:25 ]
	ds_write_b32 v66, v45
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b32 v[86:87], v67 offset1:1
.Ltmp96:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[78:79], v[0:1], v[2:3]
	v_pk_add_f32 v[80:81], v[4:5], v[6:7]
	v_pk_add_f32 v[82:83], v[14:15], v[8:9]
	v_pk_add_f32 v[84:85], v[12:13], v[10:11]
.Ltmp97:
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
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_mul_f32_e32 v45, v57, v45
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cvt_pk_bf16_f32 v14, v14, s0
	v_cvt_pk_bf16_f32 v49, v1, s0
	v_cvt_pk_bf16_f32 v15, v15, s0
	v_cvt_pk_bf16_f32 v57, v2, s0
	v_cvt_pk_bf16_f32 v77, v3, s0
	v_cvt_pk_bf16_f32 v4, v4, s0
	v_cvt_pk_bf16_f32 v5, v5, s0
.Ltmp98:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[0:1], v[78:79], v[80:81]
	v_pk_add_f32 v[2:3], v[82:83], v[84:85]
.Ltmp99:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	ds_write_b16 v68, v47
	ds_write_b16 v68, v14 offset:4096
	ds_write_b16 v69, v49
	ds_write_b16 v69, v15 offset:4096
	ds_write_b16 v70, v57
	ds_write_b16 v70, v8 offset:4096
	ds_write_b16 v71, v77
	ds_write_b16 v71, v9 offset:4096
	ds_write_b16 v72, v4
	ds_write_b16 v72, v12 offset:4096
	ds_write_b16 v73, v5
	ds_write_b16 v73, v13 offset:4096
	ds_write_b16 v74, v6
	ds_write_b16 v74, v10 offset:4096
	ds_write_b16 v75, v7
	ds_write_b16 v75, v11 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[6:7], v76
	ds_read_b64_tr_b16 v[8:9], v76 offset:4096
	ds_read_b64_tr_b16 v[12:13], v76 offset:4224
	ds_read_b64_tr_b16 v[10:11], v76 offset:128
.Ltmp100:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[0:1], v[0:1], v[2:3]
.Ltmp101:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[2:3], v[26:27], v[86:87] op_sel_hi:[1,0]
.Ltmp102:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_pk_add_f32 v[4:5], v[0:1], v[0:1] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp103:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[0:1], v[24:25], v[86:87] op_sel_hi:[1,0]
.Ltmp104:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v5, v4
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v5
.Ltmp105:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[88:91], v[6:9], v[0:3]
	v_mov_b32_e32 v65, v38
.Ltmp106:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	; wait_asyncmark(3)
	s_waitcnt vmcnt(3) lgkmcnt(0)
.Ltmp107:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_mov_b32_e32 v0, v87
	v_pk_mul_f32 v[2:3], v[30:31], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[28:29], v[0:1] op_sel_hi:[1,0]
.Ltmp108:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_barrier
.Ltmp109:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[28:31], v[88:91], v[10:13], v[0:3]
.Ltmp110:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	s_nop 2
	v_add_f32_e32 v0, v4, v5
.Ltmp111:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_add_f32_e32 v57, v0, v45
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cbranch_scc1 .LBB0_7
.LBB0_8:                                ; %._crit_edge
.Ltmp112:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v0, 0x110
	v_cmp_eq_u32_e32 vcc, 0, v43
.Ltmp113:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v41, s57, v39
.Ltmp114:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_nop 0
	v_cndmask_b32_e64 v0, v0, 0, vcc
	v_bitop3_b32 v46, v0, v62, v42 bitop3:0xde
.Ltmp115:
	.loc	1 182 14 is_stmt 1              ; mha.py:182:14 @[ mha.py:746:25 ]
	v_cndmask_b32_e64 v0, 0, 1, s[0:1]
	v_cmp_ne_u32_e64 s[2:3], 1, v0
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_10
; %bb.9:
.Ltmp116:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v0, s56, v46
	ds_read_b128 v[48:51], v0
.Ltmp117:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v42, s54, v46
	ds_read_b128 v[42:45], v42
.Ltmp118:
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
	s_branch .LBB0_11
.Ltmp119:
.LBB0_10:
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
.LBB0_11:
.Ltmp120:
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
.Ltmp121:
	.loc	3 170 12 is_stmt 1              ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e32 v0, v12, v13
	v_max3_f32 v14, v3, v4, v5
	v_max3_f32 v44, v10, v11, v8
	v_max3_f32 v45, v9, v42, v43
	v_max3_f32 v15, v6, v7, v1
	v_max3_f32 v0, v0, v2, v14
	v_max3_f32 v14, v44, v45, v41
	v_max3_f32 v0, v0, v15, v14
.Ltmp122:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v14, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v14
.Ltmp123:
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
.Ltmp124:
	.loc	3 293 12 is_stmt 1              ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[2:3], v[12:13], v[42:43]
	v_pk_add_f32 v[4:5], v[44:45], v[6:7]
	v_pk_add_f32 v[50:51], v[48:49], v[10:11]
	v_pk_add_f32 v[52:53], v[8:9], v[14:15]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
	v_pk_add_f32 v[4:5], v[50:51], v[52:53]
.Ltmp125:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	v_and_b32_e32 v41, 31, v54
.Ltmp126:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
.Ltmp127:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v47, 0x420
.Ltmp128:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_pk_add_f32 v[4:5], v[2:3], v[2:3] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp129:
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:746:25 ]
	v_sub_f32_e32 v2, v38, v0
	.loc	1 241 17 is_stmt 0              ; mha.py:241:17 @[ mha.py:746:25 ]
	v_exp_f32_e32 v5, v2
	.loc	1 259 26 is_stmt 1              ; mha.py:259:26 @[ mha.py:746:25 ]
	v_cmp_eq_u32_e32 vcc, 0, v60
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	s_lshl_b32 s0, s33, 7
	v_lshl_add_u32 v2, v41, 2, 0
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_lshlrev_b32_e32 v41, 1, v41
	v_cndmask_b32_e64 v47, v47, 0, vcc
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	v_add_u32_e32 v50, s0, v2
	v_lshl_add_u32 v2, v55, 2, 0
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_bitop3_b32 v41, s44, v47, v41 bitop3:0x36
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	v_add_u32_e32 v51, s44, v2
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_add_u32_e32 v52, 0, v41
	v_cvt_pk_bf16_f32 v12, v12, s0
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	ds_write_b32 v50, v5
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b32 v[2:3], v51 offset1:1
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v52, v12
	v_cvt_pk_bf16_f32 v12, v48, s0
	ds_write_b16 v52, v12 offset:4096
	v_xor_b32_e32 v12, 0x108, v41
	v_add_u32_e32 v53, 0, v12
	v_cvt_pk_bf16_f32 v12, v13, s0
	ds_write_b16 v53, v12
	v_cvt_pk_bf16_f32 v12, v49, s0
	ds_write_b16 v53, v12 offset:4096
	v_xor_b32_e32 v12, 0x210, v41
	v_add_u32_e32 v61, 0, v12
	v_cvt_pk_bf16_f32 v10, v10, s0
	ds_write_b16 v61, v10 offset:4096
	v_xor_b32_e32 v10, 0x318, v41
	v_cvt_pk_bf16_f32 v12, v42, s0
	v_add_u32_e32 v62, 0, v10
	v_cvt_pk_bf16_f32 v10, v43, s0
	ds_write_b16 v61, v12
	ds_write_b16 v62, v10
	v_cvt_pk_bf16_f32 v10, v11, s0
	ds_write_b16 v62, v10 offset:4096
	v_xor_b32_e32 v10, 0x840, v41
	v_add_u32_e32 v63, 0, v10
	v_cvt_pk_bf16_f32 v8, v8, s0
	ds_write_b16 v63, v8 offset:4096
	v_xor_b32_e32 v8, 0x948, v41
	v_cvt_pk_bf16_f32 v10, v44, s0
	v_add_u32_e32 v64, 0, v8
	v_cvt_pk_bf16_f32 v8, v45, s0
	ds_write_b16 v63, v10
	ds_write_b16 v64, v8
	v_cvt_pk_bf16_f32 v8, v9, s0
	ds_write_b16 v64, v8 offset:4096
	v_xor_b32_e32 v8, 0xa50, v41
	v_add_u32_e32 v65, 0, v8
	v_cvt_pk_bf16_f32 v6, v6, s0
	ds_write_b16 v65, v6
	v_cvt_pk_bf16_f32 v6, v14, s0
	ds_write_b16 v65, v6 offset:4096
	v_xor_b32_e32 v6, 0xb58, v41
	v_add_u32_e32 v66, 0, v6
	v_cvt_pk_bf16_f32 v6, v7, s0
	ds_write_b16 v66, v6
	v_cvt_pk_bf16_f32 v6, v15, s0
	ds_write_b16 v66, v6 offset:4096
	v_and_b32_e32 v6, 60, v54
	v_lshlrev_b32_e32 v7, 6, v6
	v_and_b32_e32 v8, 24, v58
	v_lshlrev_b32_e32 v6, 1, v6
	s_lshl_b32 s0, s33, 5
	v_bitop3_b32 v6, v7, v6, v8 bitop3:0x36
.Ltmp130:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v1, v4
.Ltmp131:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_xor_b32_e32 v6, s0, v6
.Ltmp132:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	s_nop 0
	v_permlane32_swap_b32_e32 v4, v1
.Ltmp133:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_and_b64 vcc, exec, s[2:3]
	v_add_u32_e32 v67, 0, v6
	.loc	1 259 26 is_stmt 0              ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_cbranch_vccnz .LBB0_13
; %bb.12:
	.loc	1 0 0                           ; mha.py:0 @[ mha.py:746:25 ]
	ds_read_b64_tr_b16 v[8:9], v67
	ds_read_b64_tr_b16 v[10:11], v67 offset:4096
	ds_read_b64_tr_b16 v[14:15], v67 offset:4224
	ds_read_b64_tr_b16 v[12:13], v67 offset:128
	.loc	1 248 15 is_stmt 1              ; mha.py:248:15 @[ mha.py:746:25 ]
	v_mul_f32_e32 v5, v57, v5
.Ltmp134:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_add_f32_e32 v1, v4, v1
.Ltmp135:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_add_f32_e32 v57, v1, v5
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[6:7], v[26:27], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[24:25], v[2:3] op_sel_hi:[1,0]
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v38, v0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[68:71], v[8:11], v[4:7]
	s_nop 2
	v_mul_f32_e64 v4, v30, v2
	v_mul_f32_e64 v5, v31, v2
	v_pk_mul_f32 v[2:3], v[28:29], v[2:3] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[28:31], v[68:71], v[12:15], v[2:5]
.LBB0_13:
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
.Ltmp136:
	.loc	1 36 18 is_stmt 1               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v39, s52, v39
.Ltmp137:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
.Ltmp138:
	.loc	1 182 14 is_stmt 1              ; mha.py:182:14 @[ mha.py:746:25 ]
	s_cbranch_scc1 .LBB0_15
; %bb.14:
.Ltmp139:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v0, s51, v46
	ds_read_b128 v[68:71], v0
.Ltmp140:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_add_u32_e32 v46, s50, v46
	ds_read_b128 v[46:49], v46
.Ltmp141:
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
	s_branch .LBB0_16
.Ltmp142:
.LBB0_15:
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
.LBB0_16:
.Ltmp143:
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
.Ltmp144:
	.loc	3 170 12 is_stmt 1              ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e32 v0, v72, v73
	v_max3_f32 v14, v3, v4, v5
	v_max3_f32 v40, v10, v11, v8
	v_max3_f32 v41, v9, v12, v13
	v_max3_f32 v15, v6, v7, v1
	v_max3_f32 v0, v0, v2, v14
	v_max3_f32 v14, v40, v41, v39
	v_max3_f32 v0, v0, v15, v14
.Ltmp145:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v14, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v14
.Ltmp146:
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
.Ltmp147:
	.loc	3 293 12 is_stmt 1              ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[2:3], v[14:15], v[40:41]
	v_pk_add_f32 v[4:5], v[42:43], v[6:7]
	v_pk_add_f32 v[46:47], v[44:45], v[10:11]
	v_pk_add_f32 v[48:49], v[8:9], v[12:13]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
	v_pk_add_f32 v[4:5], v[46:47], v[48:49]
.Ltmp148:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	v_cvt_pk_bf16_f32 v14, v14, s0
.Ltmp149:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
.Ltmp150:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	ds_write_b32 v50, v1
.Ltmp151:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_pk_add_f32 v[4:5], v[2:3], v[2:3] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp152:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b32 v[2:3], v51 offset1:1
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v52, v14
	v_cvt_pk_bf16_f32 v14, v44, s0
	ds_write_b16 v52, v14 offset:4096
	v_cvt_pk_bf16_f32 v14, v15, s0
	ds_write_b16 v53, v14
	v_cvt_pk_bf16_f32 v14, v45, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	ds_write_b16 v53, v14 offset:4096
	v_cvt_pk_bf16_f32 v14, v40, s0
	ds_write_b16 v61, v10 offset:4096
	v_cvt_pk_bf16_f32 v10, v41, s0
	ds_write_b16 v61, v14
	ds_write_b16 v62, v10
	v_cvt_pk_bf16_f32 v10, v11, s0
	v_cvt_pk_bf16_f32 v8, v8, s0
	ds_write_b16 v62, v10 offset:4096
	v_cvt_pk_bf16_f32 v10, v42, s0
	ds_write_b16 v63, v8 offset:4096
	v_cvt_pk_bf16_f32 v8, v43, s0
	v_cvt_pk_bf16_f32 v6, v6, s0
	ds_write_b16 v63, v10
	ds_write_b16 v64, v8
	v_cvt_pk_bf16_f32 v8, v9, s0
	ds_write_b16 v65, v6
	v_cvt_pk_bf16_f32 v6, v12, s0
.Ltmp153:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v5, v4
.Ltmp154:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	ds_write_b16 v64, v8 offset:4096
	ds_write_b16 v65, v6 offset:4096
	v_cvt_pk_bf16_f32 v6, v7, s0
.Ltmp155:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_permlane32_swap_b32_e32 v4, v5
.Ltmp156:
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	ds_write_b16 v66, v6
	v_cvt_pk_bf16_f32 v6, v13, s0
	.loc	1 259 19 is_stmt 0              ; mha.py:259:19 @[ mha.py:746:25 ]
	s_andn2_b64 vcc, exec, s[0:1]
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	ds_write_b16 v66, v6 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	s_cbranch_vccnz .LBB0_18
; %bb.17:
	.loc	1 0 0                           ; mha.py:0 @[ mha.py:746:25 ]
	ds_read_b64_tr_b16 v[8:9], v67
	ds_read_b64_tr_b16 v[10:11], v67 offset:4096
	ds_read_b64_tr_b16 v[14:15], v67 offset:4224
	ds_read_b64_tr_b16 v[12:13], v67 offset:128
	.loc	1 248 15 is_stmt 1              ; mha.py:248:15 @[ mha.py:746:25 ]
	v_mul_f32_e32 v1, v57, v1
.Ltmp157:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_add_f32_e32 v4, v4, v5
.Ltmp158:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_add_f32_e32 v57, v4, v1
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[6:7], v[26:27], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[24:25], v[2:3] op_sel_hi:[1,0]
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v38, v0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[68:71], v[8:11], v[4:7]
	s_nop 2
	v_mul_f32_e64 v4, v30, v2
	v_mul_f32_e64 v5, v31, v2
	v_pk_mul_f32 v[2:3], v[28:29], v[2:3] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x32_bf16 v[28:31], v[68:71], v[12:15], v[2:5]
.Ltmp159:
.LBB0_18:                               ; %Flow343
	.loc	1 0 19 is_stmt 0                ; mha.py:0:19
	s_load_dwordx2 s[8:9], s[4:5], 0x58
	s_load_dwordx2 s[10:11], s[4:5], 0x7c
	s_load_dword s18, s[4:5], 0x60
	.loc	1 798 8 is_stmt 1               ; mha.py:798:8
	s_cmp_lt_i32 s49, 1
	.loc	1 798 5 is_stmt 0               ; mha.py:798:5
	s_cbranch_scc1 .LBB0_22
; %bb.19:
	.loc	1 731 17 is_stmt 1              ; mha.py:731:17
	s_lshl_b32 s19, s47, 5
.Ltmp160:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	s_cmp_ge_i32 s45, s19
	s_cbranch_scc1 .LBB0_22
.Ltmp161:
; %bb.20:                               ; %.lr.ph123
	.loc	1 803 19                        ; mha.py:803:19
	s_lshl_b32 s3, s48, 5
	.loc	1 806 19                        ; mha.py:806:19
	s_mul_i32 s0, s3, s31
	s_mul_hi_u32 s1, s3, s30
.Ltmp162:
	.loc	1 261 19                        ; mha.py:261:19 @[ mha.py:811:25 ]
	s_lshl_b64 s[4:5], s[14:15], 5
	.loc	1 264 19                        ; mha.py:264:19 @[ mha.py:811:25 ]
	s_lshl_b64 s[16:17], s[30:31], 5
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	s_add_i32 s2, s44, 0
.Ltmp163:
	.loc	1 806 19                        ; mha.py:806:19
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s3, s30
	.loc	1 0 0 is_stmt 0                 ; mha.py:0
	s_add_u32 s0, s24, s0
	s_addc_u32 s1, s25, s1
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[0:1]
	.loc	1 803 19 is_stmt 1              ; mha.py:803:19
	s_mul_i32 s0, s3, s15
	s_mul_hi_u32 s1, s3, s14
	s_add_i32 s1, s1, s0
	s_mul_i32 s3, s3, s14
	.loc	1 0 0 is_stmt 0                 ; mha.py:0
	s_add_u32 s0, s34, s3
	s_addc_u32 s1, s35, s1
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[0:1]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[0:1]
	s_bfe_i32 s0, s40, 0x10006
	s_movk_i32 s1, 0x110
	v_bfe_i32 v5, v54, 3, 1
	v_lshrrev_b32_e32 v6, 1, v60
	v_lshlrev_b32_e32 v1, 2, v56
	s_and_b32 s3, s0, 0x110
	v_mov_b32_e32 v2, 0x2fc
	v_bitop3_b32 v5, v5, v6, s1 bitop3:0x6c
	v_and_b32_e32 v6, 31, v54
	v_mov_b32_e32 v10, 0x420
	v_cmp_eq_u32_e32 vcc, 0, v60
	v_and_b32_e32 v35, 60, v54
	s_and_b32 s0, s0, 0x108
.Ltmp164:
	.loc	1 177 20 is_stmt 1              ; mha.py:177:20 @[ mha.py:811:25 ]
	v_mov_b32_e32 v0, 0x3fb8aa3b
	v_bitop3_b32 v3, v1, s3, v2 bitop3:0x6c
	v_lshlrev_b32_e32 v4, 5, v54
	v_lshl_add_u32 v7, v6, 2, 0
	v_lshlrev_b32_e32 v9, 1, v6
	v_cndmask_b32_e64 v10, v10, 0, vcc
	v_lshlrev_b32_e32 v37, 6, v35
	v_and_b32_e32 v40, 24, v58
	v_lshlrev_b32_e32 v35, 1, v35
	v_bitop3_b32 v1, v1, s0, v2 bitop3:0x6c
	v_lshlrev_b32_e32 v2, 3, v6
	v_mov_b32_e32 v6, 0x108
	v_mul_f32_e32 v39, s46, v0
.Ltmp165:
	.loc	1 606 11                        ; mha.py:606:11
	v_lshrrev_b32_e32 v0, 3, v60
	v_and_b32_e32 v4, 0x2e0, v4
	v_bitop3_b32 v9, s44, v10, v9 bitop3:0x36
	s_lshl_b32 s3, s33, 5
	v_bitop3_b32 v35, v37, v35, v40 bitop3:0x36
	v_cndmask_b32_e64 v6, v6, 0, vcc
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_add_u32_e32 v4, 0, v4
	s_lshl_b32 s1, s33, 7
	v_lshlrev_b32_e32 v8, 2, v55
	v_xor_b32_e32 v10, 0x108, v9
	v_xor_b32_e32 v11, 0x210, v9
	v_xor_b32_e32 v12, 0x318, v9
	v_xor_b32_e32 v13, 0x840, v9
	v_xor_b32_e32 v14, 0x948, v9
	v_xor_b32_e32 v15, 0xa50, v9
	v_xor_b32_e32 v33, 0xb58, v9
	v_xor_b32_e32 v35, s3, v35
	v_xor_b32_e32 v2, v6, v2
.Ltmp166:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	v_add_u32_e32 v40, s45, v0
	v_add_u32_e32 v0, s44, v59
	s_and_b32 s37, s37, 0xffff
	s_and_b32 s13, s13, 0xffff
	s_sub_i32 s24, s19, 32
	v_lshrrev_b32_e32 v41, 3, v0
	v_bfrev_b32_e32 v42, 1
	v_add_u32_e32 v43, 0, v3
	v_add_u32_e32 v44, v4, v5
	v_mov_b32_e32 v45, 0xff800000
	v_add_u32_e32 v46, s1, v7
	v_add_u32_e32 v47, s2, v8
	v_add_u32_e32 v48, 0, v9
	v_add_u32_e32 v49, 0, v10
	v_add_u32_e32 v50, 0, v11
	v_add_u32_e32 v51, 0, v12
	v_add_u32_e32 v52, 0, v13
	v_add_u32_e32 v53, 0, v14
	v_add_u32_e32 v58, 0, v15
	v_add_u32_e32 v59, 0, v33
	v_add_u32_e32 v60, 0, v35
	v_add_u32_e32 v61, 0, v1
	v_add_u32_e32 v62, 0, v2
	s_mov_b32 s14, s38
	s_mov_b32 s15, s39
	v_mov_b32_e32 v63, v38
.LBB0_21:                               ; =>This Inner Loop Header: Depth=1
.Ltmp167:
	.loc	1 33 16                         ; mha.py:33:16 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_add_u32_e32 v2, s45, v41
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e32 v3, 1, v32
	.loc	1 33 16                         ; mha.py:33:16 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_cmp_gt_i32_e64 s[0:1], s7, v2
.Ltmp168:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e32 v0, 1, v34
.Ltmp169:
	.loc	1 31 18                         ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e32 v1, 1, v36
.Ltmp170:
	.loc	1 170 28                        ; mha.py:170:28 @[ mha.py:811:25 ]
	v_add_u32_e32 v4, 16, v40
.Ltmp171:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_cndmask_b32_e64 v2, v42, v3, s[0:1]
.Ltmp172:
	.loc	1 170 28                        ; mha.py:170:28 @[ mha.py:811:25 ]
	v_cmp_le_i32_e64 s[2:3], s7, v4
.Ltmp173:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	v_cndmask_b32_e64 v0, v42, v0, s[0:1]
.Ltmp174:
	.loc	1 31 18                         ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	v_cndmask_b32_e64 v1, v42, v1, s[0:1]
.Ltmp175:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	buffer_load_dword v3, v2, s[36:39], 0 offen
.Ltmp176:
	.loc	1 34 18 is_stmt 0               ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	buffer_load_dword v4, v0, s[36:39], 0 offen
.Ltmp177:
	.loc	1 31 18 is_stmt 1               ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	buffer_load_dword v74, v1, s[12:15], 0 offen
.Ltmp178:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp179:
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
.Ltmp180:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	s_and_b64 s[0:1], s[26:27], s[2:3]
.Ltmp181:
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
.Ltmp182:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	s_waitcnt vmcnt(1)
	ds_write_b32 v43, v4
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[64:67], v44
.Ltmp183:
	.loc	1 34 18 is_stmt 0               ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v43, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp184:
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:811:25 ]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[64:67], v[16:19], 0
.Ltmp185:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	ds_read_b128 v[64:67], v44
.Ltmp186:
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
.Ltmp187:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e32 v33, v0, v1
	v_max3_f32 v35, v3, v4, v5
	v_max3_f32 v38, v9, v10, v11
	v_max3_f32 v64, v12, v13, v14
	v_max3_f32 v37, v6, v7, v8
	v_max3_f32 v33, v33, v2, v35
	v_max3_f32 v35, v38, v64, v15
	v_max3_f32 v33, v33, v37, v35
.Ltmp188:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ]
	v_mov_b32_e32 v35, v33
	s_nop 1
	v_permlane32_swap_b32_e32 v33, v35
.Ltmp189:
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
.Ltmp190:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_pk_add_f32 v[64:65], v[0:1], v[2:3]
	v_pk_add_f32 v[66:67], v[4:5], v[6:7]
	v_pk_add_f32 v[68:69], v[8:9], v[10:11]
	v_pk_add_f32 v[70:71], v[12:13], v[14:15]
.Ltmp191:
	.loc	1 246 15                        ; mha.py:246:15 @[ mha.py:811:25 ]
	ds_write_b32 v46, v33
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b32 v[72:73], v47 offset1:1
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
	ds_write_b16 v49, v1
	ds_write_b16 v49, v9 offset:4096
	ds_write_b16 v50, v2
	ds_write_b16 v50, v10 offset:4096
	ds_write_b16 v51, v3
	ds_write_b16 v51, v11 offset:4096
	ds_write_b16 v52, v4
	ds_write_b16 v52, v12 offset:4096
	ds_write_b16 v53, v5
	ds_write_b16 v53, v13 offset:4096
	ds_write_b16 v58, v6
	ds_write_b16 v58, v14 offset:4096
	ds_write_b16 v59, v7
	ds_write_b16 v59, v15 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[6:7], v60
	ds_read_b64_tr_b16 v[8:9], v60 offset:4096
	ds_read_b64_tr_b16 v[12:13], v60 offset:4224
	ds_read_b64_tr_b16 v[10:11], v60 offset:128
.Ltmp192:
	.loc	1 31 18                         ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b32 v61, v74
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[74:75], v62
	ds_read_b64_tr_b16 v[76:77], v62 offset:512
.Ltmp193:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_pk_add_f32 v[0:1], v[64:65], v[66:67]
	v_pk_add_f32 v[2:3], v[68:69], v[70:71]
.Ltmp194:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:811:25 ]
	v_mul_f32_e32 v33, v57, v33
.Ltmp195:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_pk_add_f32 v[0:1], v[0:1], v[2:3]
.Ltmp196:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:811:25 ]
	v_pk_mul_f32 v[2:3], v[26:27], v[72:73] op_sel_hi:[1,0]
.Ltmp197:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ] ]
	v_pk_add_f32 v[4:5], v[0:1], v[0:1] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp198:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:811:25 ]
	v_pk_mul_f32 v[0:1], v[24:25], v[72:73] op_sel_hi:[1,0]
.Ltmp199:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_mov_b32_e32 v5, v4
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v5
.Ltmp200:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:811:25 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[24:27], v[74:77], v[6:9], v[0:3]
	v_mov_b32_e32 v63, v38
	s_nop 1
	v_mov_b32_e32 v0, v73
	v_pk_mul_f32 v[2:3], v[30:31], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[28:29], v[0:1] op_sel_hi:[1,0]
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[28:31], v[74:77], v[10:13], v[0:3]
.Ltmp201:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ] ]
	s_nop 2
	v_add_f32_e32 v0, v4, v5
.Ltmp202:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:811:25 ]
	v_add_f32_e32 v57, v0, v33
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	s_cbranch_scc1 .LBB0_21
.Ltmp203:
.LBB0_22:                               ; %.loopexit
	.loc	1 861 15                        ; mha.py:861:15
	v_div_scale_f32 v0, s[0:1], v57, v57, 1.0
	v_rcp_f32_e32 v1, v0
	.loc	1 862 11                        ; mha.py:862:11
	s_lshl_b32 s0, s33, 7
	s_add_i32 s44, s44, 0
	s_waitcnt lgkmcnt(0)
	.loc	1 861 15                        ; mha.py:861:15
	v_fma_f32 v2, -v0, v1, 1.0
	v_fmac_f32_e32 v1, v2, v1
	v_div_scale_f32 v2, vcc, 1.0, v57, 1.0
	v_mul_f32_e32 v5, v2, v1
	v_fma_f32 v6, -v0, v5, v2
	v_fmac_f32_e32 v5, v6, v1
	v_fma_f32 v0, -v0, v5, v2
	v_div_fmas_f32 v0, v0, v1, v5
	.loc	1 862 11                        ; mha.py:862:11
	v_lshlrev_b32_e32 v1, 2, v54
	v_and_b32_e32 v1, 0x7c, v1
	v_add_u32_e32 v1, 0, v1
	v_add_u32_e32 v2, s0, v1
	s_mov_b32 s0, 0x800000
	.loc	1 888 29                        ; mha.py:888:29
	v_mov_b32_e32 v1, 0x42000000
	v_cmp_gt_f32_e32 vcc, s0, v57
	.loc	1 861 15                        ; mha.py:861:15
	v_div_fixup_f32 v0, v0, v57, 1.0
	.loc	1 862 11                        ; mha.py:862:11
	s_barrier
	.loc	1 888 29                        ; mha.py:888:29
	v_cndmask_b32_e32 v5, 0, v1, vcc
	v_cndmask_b32_e64 v1, 0, 32, vcc
	v_ldexp_f32 v1, v57, v1
	v_log_f32_e32 v6, v1
	.loc	1 862 11                        ; mha.py:862:11
	ds_write_b32 v2, v0
	v_lshl_add_u32 v0, v55, 2, s44
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b32 v[0:1], v0 offset1:1
	.loc	1 884 21                        ; mha.py:884:21
	s_sub_i32 s4, s43, s6
	.loc	1 900 13                        ; mha.py:900:13
	s_mul_i32 s1, s42, s10
	.loc	1 901 15                        ; mha.py:901:15
	s_mul_i32 s2, s11, s41
	.loc	1 884 21                        ; mha.py:884:21
	s_add_i32 s0, s4, 0x80
	.loc	1 900 13                        ; mha.py:900:13
	s_add_u32 s5, s1, s2
	.loc	1 380 34                        ; mha.py:380:34
	v_and_b32_e32 v3, 0x7f, v56
	.loc	1 888 29                        ; mha.py:888:29
	v_sub_f32_e32 v5, v6, v5
	.loc	1 905 12                        ; mha.py:905:12
	s_cmp_lt_i32 s0, 1
	.loc	1 380 14                        ; mha.py:380:14
	v_or_b32_e32 v4, s43, v3
	.loc	1 888 23                        ; mha.py:888:23
	v_add_f32_e32 v5, v38, v5
	.loc	1 905 12                        ; mha.py:905:12
	s_cselect_b64 s[2:3], -1, 0
	.loc	1 890 9                         ; mha.py:890:9
	v_mul_f32_e32 v5, 0x3f317218, v5
	s_mov_b64 s[0:1], -1
	.loc	1 905 9                         ; mha.py:905:9
	s_and_b64 vcc, exec, s[2:3]
	v_lshl_add_u32 v6, v3, 2, 0
	v_add_lshl_u32 v4, v4, s5, 2
	s_cbranch_vccnz .LBB0_24
; %bb.23:
	.loc	1 906 44                        ; mha.py:906:44
	s_sub_i32 s0, 0, s4
	.loc	1 908 13                        ; mha.py:908:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v2, v5
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v7, v6
	s_and_b32 s13, s21, 0xffff
	s_bitcmp0_b32 s40, 7
	.loc	1 907 24                        ; mha.py:907:24
	v_cmp_gt_i32_e32 vcc, s0, v3
	.loc	1 908 13                        ; mha.py:908:13
	s_cselect_b64 s[0:1], -1, 0
	v_bfrev_b32_e32 v3, 1
	s_and_b64 vcc, s[0:1], vcc
	s_mov_b32 s15, 0x27000
	s_mov_b32 s14, 0x7ffffffe
	s_mov_b32 s12, s20
	v_cndmask_b32_e32 v3, v3, v4, vcc
	s_mov_b64 s[0:1], 0
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v7, v3, s[12:15], 0 offen
.LBB0_24:                               ; %Flow
	.loc	1 0 13 is_stmt 0                ; mha.py:0:13
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_26
; %bb.25:
	.loc	1 912 13 is_stmt 1              ; mha.py:912:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v2, v5
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v3, v6
	s_and_b32 s21, s21, 0xffff
	s_bitcmp0_b32 s40, 7
	v_bfrev_b32_e32 v2, 1
	s_cselect_b64 vcc, -1, 0
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_cndmask_b32_e32 v2, v2, v4, vcc
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v3, v2, s[20:23], 0 offen
.LBB0_26:
	.loc	1 862 11                        ; mha.py:862:11
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, v1
	.loc	1 380 34                        ; mha.py:380:34
	v_lshl_or_b32 v8, s33, 4, v55
	.loc	1 586 11                        ; mha.py:586:11
	v_lshrrev_b32_e32 v10, 2, v54
	.loc	1 918 9                         ; mha.py:918:9
	s_mul_i32 s4, s42, s8
	.loc	1 919 11                        ; mha.py:919:11
	s_mul_i32 s5, s9, s41
	.loc	1 862 11                        ; mha.py:862:11
	v_pk_mul_f32 v[4:5], v[30:31], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[28:29], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[26:27], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[24:25], v[0:1] op_sel_hi:[1,0]
	.loc	1 380 14                        ; mha.py:380:14
	v_or_b32_e32 v8, s43, v8
	.loc	1 586 11                        ; mha.py:586:11
	v_and_b32_e32 v10, 12, v10
	.loc	1 918 9                         ; mha.py:918:9
	s_add_i32 s4, s4, s5
	.loc	1 679 18                        ; mha.py:679:18
	v_cmp_gt_i32_e64 s[0:1], s6, v8
	.loc	1 929 10                        ; mha.py:929:10
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_cvt_pk_bf16_f32 v1, v6, v7
	v_cvt_pk_bf16_f32 v2, v2, v3
	v_cvt_pk_bf16_f32 v3, v4, v5
	.loc	1 930 14                        ; mha.py:930:14
	v_mul_lo_u32 v4, v8, s18
	v_add_u32_e32 v6, s4, v10
	.loc	1 380 14                        ; mha.py:380:14
	v_or_b32_e32 v9, 64, v8
	.loc	1 930 5                         ; mha.py:930:5
	v_add_lshl_u32 v4, v6, v4, 1
	v_bfrev_b32_e32 v7, 1
	.loc	1 925 5                         ; mha.py:925:5
	s_or_b64 s[0:1], s[2:3], s[0:1]
	.loc	1 679 18                        ; mha.py:679:18
	v_cmp_gt_i32_e32 vcc, s6, v9
	.loc	1 930 14                        ; mha.py:930:14
	v_mul_lo_u32 v5, v9, s18
	.loc	1 930 5 is_stmt 0               ; mha.py:930:5
	s_and_b32 s29, s29, 0xffff
	s_mov_b32 s31, 0x27000
	s_mov_b32 s30, 0x7ffffffe
	v_cndmask_b32_e64 v4, v7, v4, s[0:1]
	buffer_store_dwordx2 v[0:1], v4, s[28:31], 0 offen
	v_add_lshl_u32 v0, v6, v5, 1
	.loc	1 925 5 is_stmt 1               ; mha.py:925:5
	s_or_b64 vcc, s[2:3], vcc
	.loc	1 930 5                         ; mha.py:930:5
	v_cndmask_b32_e32 v0, v7, v0, vcc
	buffer_store_dwordx2 v[2:3], v0, s[28:31], 0 offen
	.loc	1 297 1                         ; mha.py:297:1
	s_endpgm
.Ltmp204:
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
.Lfunc_end0:
	.size	_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0, .Lfunc_end0-_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
	.cfi_endproc
                                        ; -- End function
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_vgpr, 92
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_agpr, 0
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.numbered_sgpr, 60
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_named_barrier, 0
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.private_seg_size, 0
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.uses_vcc, 1
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.uses_flat_scratch, 0
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_dyn_sized_stack, 0
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_recursion, 0
	.set _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7904
; TotalNumSgprs: 66
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
	.quad	.Ltmp187                        ; DW_AT_low_pc
	.long	.Ltmp189-.Ltmp187               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	209                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	9                               ; Abbrev [9] 0x105:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp187                        ; DW_AT_low_pc
	.long	.Ltmp188-.Ltmp187               ; DW_AT_high_pc
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
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp159-.Lfunc_begin0
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
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges8:
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp156-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges9:
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp157-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges10:
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges11:
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	.Ltmp184-.Lfunc_begin0
	.quad	.Ltmp185-.Lfunc_begin0
	.quad	.Ltmp186-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges12:
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
	.quad	.Ltmp180-.Lfunc_begin0
	.quad	.Ltmp181-.Lfunc_begin0
	.quad	.Ltmp182-.Lfunc_begin0
	.quad	.Ltmp183-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges13:
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges14:
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp195-.Lfunc_begin0
	.quad	.Ltmp196-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges15:
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
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
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     92
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
