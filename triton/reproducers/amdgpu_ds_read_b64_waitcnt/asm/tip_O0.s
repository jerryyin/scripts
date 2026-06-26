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
; %bb.29:
	.file	1 "/root/aiter/aiter/ops/triton/_triton_kernels/attention" "mha.py"
	.loc	1 297 0 prologue_end            ; mha.py:297:0
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.30:
.LBB0_0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
                                        ; implicit-def: $vgpr118 : SGPR spill to VGPR lane
	v_writelane_b32 v118, s16, 0
	s_mov_b64 s[34:35], s[14:15]
	v_readlane_b32 s15, v118, 0
	s_mov_b64 s[30:31], s[12:13]
	s_mov_b64 s[26:27], s[10:11]
	s_mov_b64 s[36:37], s[4:5]
	v_writelane_b32 v118, s36, 1
	s_nop 1
	v_writelane_b32 v118, s37, 2
	s_load_dwordx2 s[0:1], s[36:37], 0xa8
	s_waitcnt lgkmcnt(0)
	s_load_dwordx2 s[0:1], s[36:37], 0xa0
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x90
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x8c
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x88
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x78
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x74
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x70
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x6c
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x68
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x64
	s_load_dword s7, s[36:37], 0x84
	s_load_dword s6, s[36:37], 0x60
	s_load_dword s1, s[36:37], 0x98
	s_load_dword s38, s[36:37], 0x80
	s_load_dword s40, s[36:37], 0x7c
	s_load_dwordx2 s[2:3], s[36:37], 0x20
	s_load_dword s18, s[36:37], 0x28
	s_load_dword s24, s[36:37], 0x2c
	s_load_dword s19, s[36:37], 0x30
	s_load_dword s23, s[36:37], 0x34
	s_load_dword s28, s[36:37], 0x38
	s_load_dword s12, s[36:37], 0x3c
	s_load_dword s14, s[36:37], 0x40
	s_load_dword s16, s[36:37], 0x44
	s_load_dword s10, s[36:37], 0x48
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x4c
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x50
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x54
	s_waitcnt lgkmcnt(0)
	s_load_dword s0, s[36:37], 0x58
	s_load_dword s4, s[36:37], 0x5c
	s_load_dword s21, s[36:37], 0x9c
	s_load_dword s5, s[36:37], 0x94
	v_writelane_b32 v118, s34, 3
	s_nop 1
	v_writelane_b32 v118, s35, 4
	v_writelane_b32 v118, s30, 5
	s_nop 1
	v_writelane_b32 v118, s31, 6
	v_writelane_b32 v118, s26, 7
	s_nop 1
	v_writelane_b32 v118, s27, 8
	s_mov_b32 s11, s1
	v_writelane_b32 v118, s11, 9
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s11, s5
	v_writelane_b32 v118, s11, 10
	v_writelane_b32 v118, s7, 11
	v_writelane_b32 v118, s6, 12
	v_writelane_b32 v118, s4, 13
	v_writelane_b32 v118, s0, 14
	v_writelane_b32 v118, s2, 15
	s_nop 1
	v_writelane_b32 v118, s3, 16
	s_mov_b32 s0, 0x3ff
	v_and_b32_e64 v7, v0, s0
.Ltmp1:
	.loc	1 912 13 is_stmt 1              ; mha.py:912:13
	v_accvgpr_write_b32 a0, v7              ;  Reload Reuse
	v_mov_b32_e32 v0, v7
	v_accvgpr_write_b32 a1, v0              ;  Reload Reuse
	v_readfirstlane_b32 s0, v7
	s_mov_b32 s2, s0
	v_writelane_b32 v118, s2, 17
	s_mov_b32 s13, 6
	s_lshr_b32 s2, s0, s13
	v_writelane_b32 v118, s2, 18
	s_bfe_u32 s6, s0, 0x20006
	.loc	1 367 19                        ; mha.py:367:19
	v_writelane_b32 v118, s6, 19
	s_mov_b32 s22, 0x7f
	s_add_i32 s2, s5, s22
	s_mov_b32 s0, 0
	.loc	1 367 18 is_stmt 0              ; mha.py:367:18
	v_writelane_b32 v118, s0, 20
	s_cmp_lt_i32 s2, s0
	s_mov_b32 s17, -1
	v_writelane_b32 v118, s17, 21
	s_cselect_b32 s3, s17, s0
	s_add_i32 s2, s2, s3
	s_xor_b32 s11, s2, s3
	s_mov_b32 s2, 0x80
	s_add_i32 s2, s0, s2
	s_xor_b32 s20, s2, s0
	s_sub_i32 s2, s0, s20
	v_cvt_f32_u32_e32 v0, s20
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s4, v0
	s_mul_i32 s4, s2, s4
	v_readfirstlane_b32 s2, v0
	s_mul_hi_u32 s4, s2, s4
	v_readfirstlane_b32 s2, v0
	s_add_i32 s2, s2, s4
	s_mul_hi_u32 s4, s11, s2
	s_mov_b32 s7, 1
	v_writelane_b32 v118, s7, 22
	s_add_i32 s2, s4, s7
	s_mul_i32 s25, s4, s20
	s_sub_i32 s25, s11, s25
	s_sub_i32 s11, s25, s20
	s_cmp_ge_u32 s25, s20
	s_cselect_b32 s11, s11, s25
	s_cselect_b32 s4, s2, s4
	s_add_i32 s2, s4, s7
	s_cmp_ge_u32 s11, s20
	s_cselect_b32 s2, s2, s4
	s_xor_b32 s3, s3, s0
	s_xor_b32 s2, s2, s3
	s_sub_i32 s11, s2, s3
	.loc	1 376 16 is_stmt 1              ; mha.py:376:16
	s_cmp_lt_i32 s15, s0
	s_cselect_b32 s20, s17, s0
	s_add_i32 s2, s15, s20
	s_xor_b32 s30, s2, s20
	s_mov_b32 s4, 32
	v_writelane_b32 v118, s4, 23
	s_add_i32 s2, s0, s4
	s_xor_b32 s37, s2, s0
	v_writelane_b32 v118, s37, 24
	s_sub_i32 s2, s0, s37
	v_cvt_f32_u32_e32 v0, s37
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s3, v0
	s_mul_i32 s3, s2, s3
	v_readfirstlane_b32 s2, v0
	s_mul_hi_u32 s3, s2, s3
	v_readfirstlane_b32 s2, v0
	s_add_i32 s2, s2, s3
	s_mul_hi_u32 s25, s30, s2
	s_add_i32 s3, s25, s7
	s_mul_i32 s26, s25, s37
	s_sub_i32 s27, s30, s26
	s_sub_i32 s26, s27, s37
	s_cmp_ge_u32 s27, s37
	s_cselect_b32 s26, s26, s27
	s_cselect_b32 s25, s3, s25
	s_add_i32 s3, s25, s7
	s_cmp_ge_u32 s26, s37
	s_cselect_b32 s3, s3, s25
	s_xor_b32 s25, s20, s0
	s_xor_b32 s3, s3, s25
	s_sub_i32 s3, s3, s25
	.loc	1 374 18                        ; mha.py:374:18
	s_mul_i32 s25, s3, s4
	s_sub_i32 s15, s15, s25
.Ltmp2:
	.file	2 "/root/aiter/aiter/ops/triton/utils/_triton" "pid_preprocessing.py"
	.loc	2 41 17                         ; pid_preprocessing.py:41:17 @[ mha.py:375:18 ]
	s_mov_b32 s25, s15
	s_bfe_i32 s25, s25, 0x80000
	s_sext_i32_i16 s25, s25
	v_cvt_f32_i32_e64 v1, s25
	s_mov_b32 s34, 0x41000000
	v_rcp_f32_e64 v2, s34
	s_nop 0
	v_mul_f32_e64 v0, v1, v2
	v_trunc_f32_e64 v0, v0
	v_fma_f32 v1, -v0, s34, v1
	v_cmp_ge_f32_e64 s[26:27], |v1|, s34
	s_mov_b32 s33, 8
	s_xor_b32 s25, s25, s33
	s_mov_b32 s29, 30
	s_ashr_i32 s25, s25, s29
	s_or_b32 s25, s25, s7
	s_and_b64 s[26:27], s[26:27], exec
	s_cselect_b32 s26, s25, s0
	v_cvt_i32_f32_e64 v0, v0
	s_nop 0
	v_readfirstlane_b32 s25, v0
	s_add_i32 s25, s25, s26
	s_sext_i32_i8 s25, s25
	s_sext_i32_i8 s25, s25
	.loc	2 47 15                         ; pid_preprocessing.py:47:15 @[ mha.py:375:18 ]
	s_mov_b32 s26, 0xffffffe1
	s_mul_i32 s25, s25, s26
	s_lshl2_add_u32 s25, s15, s25
.Ltmp3:
	.loc	1 376 15                        ; mha.py:376:15
	s_cmp_lt_i32 s11, s0
	s_cselect_b32 s26, s17, s0
	s_add_i32 s15, s11, s26
	s_xor_b32 s27, s15, s26
	s_sub_i32 s15, s0, s27
	v_cvt_f32_u32_e32 v0, s27
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s26, v0
	s_mul_i32 s26, s15, s26
	v_readfirstlane_b32 s15, v0
	s_mul_hi_u32 s26, s15, s26
	v_readfirstlane_b32 s15, v0
	s_add_i32 s26, s15, s26
	s_cmp_lt_i32 s3, s0
	s_cselect_b32 s15, s17, s0
	s_add_i32 s3, s3, s15
	s_xor_b32 s3, s3, s15
	s_mul_hi_u32 s26, s3, s26
	s_mul_i32 s26, s26, s27
	s_sub_i32 s26, s3, s26
	s_sub_i32 s3, s26, s27
	s_cmp_ge_u32 s26, s27
	s_cselect_b32 s26, s3, s26
	s_sub_i32 s3, s26, s27
	s_cmp_ge_u32 s26, s27
	s_cselect_b32 s3, s3, s26
	s_xor_b32 s3, s3, s15
	s_sub_i32 s15, s3, s15
	.loc	1 377 22                        ; mha.py:377:22
	s_mov_b32 s3, 5
	s_lshl_b32 s11, s11, s3
	.loc	1 377 14 is_stmt 0              ; mha.py:377:14
	s_cmp_lt_i32 s11, s0
	s_cselect_b32 s26, s17, s0
	s_add_i32 s11, s11, s26
	s_xor_b32 s31, s11, s26
	s_sub_i32 s11, s0, s31
	v_cvt_f32_u32_e32 v0, s31
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s27, v0
	s_mul_i32 s27, s11, s27
	v_readfirstlane_b32 s11, v0
	s_mul_hi_u32 s27, s11, s27
	v_readfirstlane_b32 s11, v0
	s_add_i32 s11, s11, s27
	s_mul_hi_u32 s27, s30, s11
	s_add_i32 s11, s27, s7
	s_mul_i32 s35, s27, s31
	s_sub_i32 s35, s30, s35
	s_sub_i32 s30, s35, s31
	s_cmp_ge_u32 s35, s31
	s_cselect_b32 s30, s30, s35
	s_cselect_b32 s27, s11, s27
	s_add_i32 s11, s27, s7
	s_cmp_ge_u32 s30, s31
	s_cselect_b32 s11, s11, s27
	s_xor_b32 s20, s20, s26
	s_xor_b32 s11, s11, s20
	s_sub_i32 s11, s11, s20
	.loc	1 377 13                        ; mha.py:377:13
	s_cmp_lt_i32 s11, s0
	s_cselect_b32 s20, s17, s0
	s_add_i32 s11, s11, s20
	s_xor_b32 s11, s11, s20
	s_cmp_lt_i32 s21, s0
	s_cselect_b32 s26, s17, s0
	s_add_i32 s21, s21, s26
	s_xor_b32 s26, s21, s26
	s_sub_i32 s21, s0, s26
	v_cvt_f32_u32_e32 v0, s26
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s27, v0
	s_mul_i32 s27, s21, s27
	v_readfirstlane_b32 s21, v0
	s_mul_hi_u32 s27, s21, s27
	v_readfirstlane_b32 s21, v0
	s_add_i32 s21, s21, s27
	s_mul_hi_u32 s21, s11, s21
	s_mul_i32 s21, s21, s26
	s_sub_i32 s21, s11, s21
	s_sub_i32 s11, s21, s26
	s_cmp_ge_u32 s21, s26
	s_cselect_b32 s21, s11, s21
	s_sub_i32 s11, s21, s26
	s_cmp_ge_u32 s21, s26
	s_cselect_b32 s11, s11, s21
	s_xor_b32 s11, s11, s20
	s_sub_i32 s11, s11, s20
	.loc	1 380 14 is_stmt 1              ; mha.py:380:14
	s_mov_b32 s20, 7
	s_lshl_b32 s20, s15, s20
	.loc	1 380 34 is_stmt 0              ; mha.py:380:34
	v_writelane_b32 v118, s20, 25
	s_lshl_b32 s21, s6, s13
	v_writelane_b32 v118, s21, 26
	s_mov_b32 s6, 15
	v_and_b32_e64 v0, v7, s6
	.loc	1 382 14 is_stmt 1              ; mha.py:382:14
	v_accvgpr_write_b32 a2, v0              ;  Reload Reuse
	v_lshlrev_b32_e64 v0, s7, v7
	s_mov_b32 s6, 14
	v_and_b32_e64 v10, v0, s6
	s_mov_b32 s13, 3
	v_lshlrev_b32_e64 v1, s13, v7
	v_mov_b32_e32 v3, v1
	v_accvgpr_write_b32 a3, v3              ;  Reload Reuse
	v_and_b32_e64 v1, v1, s33
	s_mov_b32 s6, 16
	.loc	1 385 19                        ; mha.py:385:19
	v_or_b32_e64 v12, v10, s6
	v_or_b32_e64 v3, v1, s6
	s_mov_b32 s15, 0
	.loc	1 402 21                        ; mha.py:402:21
	v_writelane_b32 v118, s15, 27
	s_mov_b32 s30, s28
	s_mov_b32 s31, s15
	.loc	1 403 21                        ; mha.py:403:21
	s_ashr_i32 s6, s12, 31
	s_mov_b32 s26, s12
	s_mov_b32 s27, s6
	.loc	1 406 21                        ; mha.py:406:21
	v_writelane_b32 v118, s26, 28
	s_nop 1
	v_writelane_b32 v118, s27, 29
	s_mov_b32 s26, s16
	s_mov_b32 s27, s15
	.loc	1 407 21                        ; mha.py:407:21
	s_ashr_i32 s6, s10, 31
	s_mov_b32 s42, s10
	s_mov_b32 s43, s6
	.loc	1 426 24                        ; mha.py:426:24
	v_writelane_b32 v118, s42, 30
	s_nop 1
	v_writelane_b32 v118, s43, 31
                                        ; kill: def $sgpr40 killed $sgpr40 def $sgpr40_sgpr41
	s_mov_b32 s41, s15
	.loc	1 427 24                        ; mha.py:427:24
	v_writelane_b32 v118, s40, 32
	s_nop 1
	v_writelane_b32 v118, s41, 33
                                        ; kill: def $sgpr38 killed $sgpr38 def $sgpr38_sgpr39
	s_mov_b32 s39, s15
.Ltmp4:
	.loc	1 19 13                         ; mha.py:19:13 @[ mha.py:510:16 ]
	v_writelane_b32 v118, s38, 34
	s_nop 1
	v_writelane_b32 v118, s39, 35
	s_mov_b32 s6, 31
	s_add_i32 s35, s1, s6
	.loc	1 19 12 is_stmt 0               ; mha.py:19:12 @[ mha.py:510:16 ]
	s_cmp_lt_i32 s35, s0
	s_cselect_b32 s17, s17, s0
	s_add_i32 s35, s35, s17
	s_xor_b32 s36, s35, s17
	s_mul_hi_u32 s35, s36, s2
	s_add_i32 s2, s35, s7
	s_mul_i32 s38, s35, s37
	s_sub_i32 s38, s36, s38
	s_sub_i32 s36, s38, s37
	s_cmp_ge_u32 s38, s37
	s_cselect_b32 s36, s36, s38
	s_cselect_b32 s35, s2, s35
	s_add_i32 s2, s35, s7
	s_cmp_ge_u32 s36, s37
	s_cselect_b32 s2, s2, s35
	s_xor_b32 s17, s17, s0
	s_xor_b32 s2, s2, s17
	s_sub_i32 s2, s2, s17
.Ltmp5:
	.loc	1 563 22 is_stmt 1              ; mha.py:563:22
	s_mov_b32 s17, s25
	s_sext_i32_i16 s17, s17
	v_cvt_f32_i32_e64 v4, s17
	v_mul_f32_e64 v2, v4, v2
	v_trunc_f32_e64 v2, v2
	v_fma_f32 v4, -v2, s34, v4
	v_cmp_ge_f32_e64 s[34:35], |v4|, s34
	s_xor_b32 s17, s17, s33
	s_ashr_i32 s17, s17, s29
	s_or_b32 s17, s17, s7
	s_and_b64 s[34:35], s[34:35], exec
	s_cselect_b32 s29, s17, s0
	v_cvt_i32_f32_e64 v2, v2
	s_nop 0
	v_readfirstlane_b32 s17, v2
	s_add_i32 s17, s17, s29
	s_sext_i32_i16 s17, s17
	.loc	1 573 14                        ; mha.py:573:14
	s_mov_b32 s34, s25
	s_mov_b32 s35, s15
	v_writelane_b32 v118, s34, 36
	s_nop 1
	v_writelane_b32 v118, s35, 37
	s_mul_i32 s24, s24, s25
	.loc	1 574 14                        ; mha.py:574:14
	s_sext_i32_i16 s17, s17
	v_writelane_b32 v118, s17, 38
	s_mov_b32 s25, 31
	s_ashr_i32 s25, s17, s25
	s_mov_b32 s34, s17
	s_mov_b32 s35, s25
	s_lshr_b64 s[34:35], s[34:35], s4
	s_mov_b32 s25, s34
	s_mul_i32 s33, s28, s25
	s_mul_hi_u32 s29, s28, s17
	s_add_i32 s29, s29, s33
	s_lshr_b64 s[30:31], s[30:31], s4
                                        ; kill: def $sgpr30 killed $sgpr30 killed $sgpr30_sgpr31
	s_mul_i32 s30, s30, s17
	s_add_i32 s30, s29, s30
                                        ; implicit-def: $sgpr29
                                        ; implicit-def: $sgpr31
                                        ; kill: def $sgpr30 killed $sgpr30 def $sgpr30_sgpr31
	s_mov_b32 s31, s29
	s_lshl_b64 s[30:31], s[30:31], s4
	s_mul_i32 s28, s28, s17
                                        ; kill: def $sgpr28 killed $sgpr28 def $sgpr28_sgpr29
	s_mov_b32 s29, s15
	s_or_b64 s[28:29], s[28:29], s[30:31]
	.loc	1 575 14                        ; mha.py:575:14
	s_mul_i32 s30, s16, s25
	s_mul_hi_u32 s25, s16, s17
	s_add_i32 s25, s25, s30
	s_lshr_b64 s[26:27], s[26:27], s4
                                        ; kill: def $sgpr26 killed $sgpr26 killed $sgpr26_sgpr27
	s_mul_i32 s26, s26, s17
	s_add_i32 s26, s25, s26
                                        ; implicit-def: $sgpr25
                                        ; implicit-def: $sgpr27
                                        ; kill: def $sgpr26 killed $sgpr26 def $sgpr26_sgpr27
	s_mov_b32 s27, s25
	s_lshl_b64 s[26:27], s[26:27], s4
	s_mul_i32 s16, s16, s17
                                        ; kill: def $sgpr16 killed $sgpr16 def $sgpr16_sgpr17
	s_mov_b32 s17, s15
	s_or_b64 s[16:17], s[16:17], s[26:27]
	.loc	1 582 9                         ; mha.py:582:9
	s_mov_b32 s26, s11
	s_mov_b32 s27, s15
	v_writelane_b32 v118, s26, 39
	s_nop 1
	v_writelane_b32 v118, s27, 40
	s_mul_i32 s18, s11, s18
	s_add_i32 s18, s18, s24
	.loc	1 586 11                        ; mha.py:586:11
	v_mov_b32_e32 v5, 0
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v5
	.loc	1 602 9                         ; mha.py:602:9
	s_mul_i32 s24, s11, s23
                                        ; kill: def $sgpr24 killed $sgpr24 def $sgpr24_sgpr25
	s_mov_b32 s25, s15
	s_mul_hi_u32 s26, s11, s23
                                        ; implicit-def: $sgpr23
                                        ; implicit-def: $sgpr27
                                        ; kill: def $sgpr26 killed $sgpr26 def $sgpr26_sgpr27
	s_mov_b32 s27, s23
	s_lshl_b64 s[26:27], s[26:27], s4
	s_or_b64 s[26:27], s[24:25], s[26:27]
	s_mov_b32 s24, s26
	s_mov_b32 s23, s27
	s_mov_b32 s26, s28
	s_mov_b32 s25, s29
	s_add_u32 s24, s24, s26
	s_addc_u32 s23, s23, s25
                                        ; kill: def $sgpr24 killed $sgpr24 def $sgpr24_sgpr25
	s_mov_b32 s25, s23
	.loc	1 606 11                        ; mha.py:606:11
	v_writelane_b32 v118, s24, 41
	s_nop 1
	v_writelane_b32 v118, s25, 42
	v_and_b32_e64 v8, v7, s4
	v_mov_b32_e32 v2, v8
	.loc	1 380 34                        ; mha.py:380:34
	v_accvgpr_write_b32 a4, v2              ;  Reload Reuse
	s_mov_b32 s23, 63
	v_and_b32_e64 v9, v7, s23
	v_mov_b32_e32 v2, v9
	v_accvgpr_write_b32 a5, v2              ;  Reload Reuse
	v_or_b32_e64 v2, s21, v9
	v_mov_b32_e32 v4, v2
	v_accvgpr_write_b32 a6, v4              ;  Reload Reuse
	v_lshrrev_b32_e64 v4, s7, v2
	v_mov_b32_e32 v6, s22
	v_bitop3_b32 v6, s21, v6, v9 bitop3:0xc8
	v_mov_b32_e32 v9, v6
	.loc	1 380 14 is_stmt 0              ; mha.py:380:14
	v_accvgpr_write_b32 a7, v9              ;  Reload Reuse
	v_or_b32_e64 v4, v4, s20
	v_or_b32_e64 v6, v6, s20
	.loc	1 381 14 is_stmt 1              ; mha.py:381:14
	v_accvgpr_write_b32 a8, v6              ;  Reload Reuse
	v_lshrrev_b32_e64 v6, s13, v2
	v_mov_b32_e32 v2, v6
	.loc	1 588 14                        ; mha.py:588:14
	v_accvgpr_write_b32 a9, v2              ;  Reload Reuse
	v_mul_lo_u32 v2, v4, s19
	v_add_u32_e64 v2, v2, s18
	.loc	1 606 11                        ; mha.py:606:11
	v_lshrrev_b32_e64 v8, s13, v8
	.loc	1 608 14                        ; mha.py:608:14
	v_accvgpr_write_b32 a10, v8             ;  Reload Reuse
	v_mad_i64_i32 v[14:15], s[12:13], v6, s12, 0
	v_mov_b32_e32 v8, v14
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v5
	v_mov_b32_e32 v13, v9
	v_mov_b32_e32 v14, v15
                                        ; implicit-def: $sgpr12
                                        ; implicit-def: $sgpr13
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, s12
	v_lshlrev_b64 v[14:15], s4, v[14:15]
	v_mov_b32_e32 v16, v15
	v_or_b32_e64 v13, v13, v16
                                        ; kill: def $vgpr8 killed $vgpr8 killed $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v14
	v_or_b32_e64 v8, v8, v9
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v13
	v_lshl_add_u64 v[14:15], v[8:9], 0, v[10:11]
	.loc	1 614 15                        ; mha.py:614:15
	v_accvgpr_write_b32 a11, v15            ;  Reload Reuse
	v_accvgpr_write_b32 a12, v14            ;  Reload Reuse
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v5
	.loc	1 617 21                        ; mha.py:617:21
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[12:13]
	.loc	1 622 9                         ; mha.py:622:9
	v_accvgpr_write_b32 a13, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a14, v8             ;  Reload Reuse
	s_mul_i32 s12, s11, s14
                                        ; kill: def $sgpr12 killed $sgpr12 def $sgpr12_sgpr13
	s_mov_b32 s13, s15
	s_mul_hi_u32 s14, s11, s14
                                        ; implicit-def: $sgpr11
                                        ; implicit-def: $sgpr15
                                        ; kill: def $sgpr14 killed $sgpr14 def $sgpr14_sgpr15
	s_mov_b32 s15, s11
	s_lshl_b64 s[14:15], s[14:15], s4
	s_or_b64 s[14:15], s[12:13], s[14:15]
	s_mov_b32 s12, s14
	s_mov_b32 s11, s15
	s_mov_b32 s14, s16
	s_mov_b32 s13, s17
	s_add_u32 s12, s12, s14
	s_addc_u32 s11, s11, s13
                                        ; kill: def $sgpr12 killed $sgpr12 def $sgpr12_sgpr13
	s_mov_b32 s13, s11
	.loc	1 628 14                        ; mha.py:628:14
	v_writelane_b32 v118, s12, 43
	s_nop 1
	v_writelane_b32 v118, s13, 44
	v_mad_i64_i32 v[8:9], s[10:11], v6, s10, 0
	v_mov_b32_e32 v12, v8
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v5
	v_mov_b32_e32 v5, v13
	v_mov_b32_e32 v8, v9
                                        ; implicit-def: $sgpr10
                                        ; implicit-def: $sgpr11
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, s10
	v_lshlrev_b64 v[8:9], s4, v[8:9]
	v_mov_b32_e32 v6, v9
	v_or_b32_e64 v5, v5, v6
	v_mov_b32_e32 v6, v12
                                        ; kill: def $vgpr8 killed $vgpr8 killed $vgpr8_vgpr9 killed $exec
	v_or_b32_e64 v8, v6, v8
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v5
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	.loc	1 679 18                        ; mha.py:679:18
	v_accvgpr_write_b32 a15, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a16, v8             ;  Reload Reuse
	v_cmp_lt_i32_e64 s[12:13], v4, s5
	.loc	1 689 16                        ; mha.py:689:16
	s_lshr_b64 s[10:11], s[8:9], s4
	s_mov_b32 s5, s10
	s_mov_b32 s10, 0xffff
	s_and_b32 s15, s5, s10
                                        ; kill: def $sgpr8 killed $sgpr8 killed $sgpr8_sgpr9
	s_mov_b32 s5, 0x27000
	s_mov_b32 s14, 0x7ffffffe
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9_sgpr10_sgpr11
	s_mov_b32 s9, s15
	s_mov_b32 s10, s14
	s_mov_b32 s11, s5
	v_add_lshl_u32 v4, v3, v2, s7
	s_mov_b32 s5, 0x80000000
	v_mov_b32_e32 v3, s5
	v_cndmask_b32_e64 v3, v3, v4, s[12:13]
	buffer_load_dwordx4 v[8:11], v3, s[8:11], s0 offen sc0 nt
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v6, v8
	v_mov_b32_e32 v5, v9
	v_mov_b32_e32 v4, v10
	v_mov_b32_e32 v3, v11
	v_bfe_u32 v7, v7, 5, 1
	s_mov_b32 s14, 62
	v_and_or_b32 v0, v0, s14, v7
	s_mov_b32 s14, 2
	v_lshlrev_b32_e64 v0, s14, v0
	v_accvgpr_write_b32 a17, v0             ;  Reload Reuse
	ds_bpermute_b32 v6, v0, v6
	ds_bpermute_b32 v5, v0, v5
	ds_bpermute_b32 v4, v0, v4
	ds_bpermute_b32 v3, v0, v3
	s_waitcnt lgkmcnt(3)
	v_accvgpr_write_b32 a18, v6             ;  Reload Reuse
	s_waitcnt lgkmcnt(2)
	v_accvgpr_write_b32 a19, v5             ;  Reload Reuse
	s_waitcnt lgkmcnt(1)
	v_accvgpr_write_b32 a20, v4             ;  Reload Reuse
	.loc	1 692 9                         ; mha.py:692:9
	s_waitcnt lgkmcnt(0)
	v_accvgpr_write_b32 a21, v3             ;  Reload Reuse
	v_add_lshl_u32 v2, v1, v2, s7
	v_mov_b32_e32 v1, s5
	v_cndmask_b32_e64 v1, v1, v2, s[12:13]
	buffer_load_dwordx4 v[4:7], v1, s[8:11], s0 offen sc0 nt
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v3, v4
	v_mov_b32_e32 v2, v5
	v_mov_b32_e32 v1, v6
	v_mov_b32_e32 v4, v7
	ds_bpermute_b32 v3, v0, v3
	ds_bpermute_b32 v2, v0, v2
	ds_bpermute_b32 v1, v0, v1
	ds_bpermute_b32 v0, v0, v4
	s_waitcnt lgkmcnt(3)
	v_accvgpr_write_b32 a22, v3             ;  Reload Reuse
	s_waitcnt lgkmcnt(2)
	v_accvgpr_write_b32 a23, v2             ;  Reload Reuse
	s_waitcnt lgkmcnt(1)
	v_accvgpr_write_b32 a24, v1             ;  Reload Reuse
	.loc	1 701 8                         ; mha.py:701:8
	s_waitcnt lgkmcnt(0)
	v_accvgpr_write_b32 a25, v0             ;  Reload Reuse
	s_cmp_lt_i32 s1, s4
	s_cselect_b64 s[4:5], -1, 0
	.loc	1 701 5 is_stmt 0               ; mha.py:701:5
	s_and_b32 s1, s1, s6
	s_cmp_lg_u32 s1, s0
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[4:5], s[4:5], s[6:7]
	.loc	1 728 21 is_stmt 1              ; mha.py:728:21
	v_writelane_b32 v118, s4, 45
	s_nop 1
	v_writelane_b32 v118, s5, 46
	v_cndmask_b32_e64 v0, 0, 1, s[4:5]
	s_nop 0
	v_readfirstlane_b32 s1, v0
	s_min_i32 s1, s1, s2
	.loc	1 729 21                        ; mha.py:729:21
	s_nop 0
	v_writelane_b32 v118, s1, 47
	s_sub_i32 s1, s2, s1
	.loc	1 731 17                        ; mha.py:731:17
	v_writelane_b32 v118, s1, 48
	s_lshl_b32 s2, s2, s3
	v_writelane_b32 v118, s2, 49
	s_mov_b32 s2, 0
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v0, s2
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v0, 0xff800000
	v_mov_b32_e32 v1, 1.0
	.loc	1 744 8                         ; mha.py:744:8
	s_cmp_gt_i32 s1, s0
	s_cselect_b64 s[2:3], -1, 0
	.loc	1 744 5 is_stmt 0               ; mha.py:744:5
	s_mov_b64 s[4:5], -1
	s_xor_b64 s[2:3], s[2:3], s[4:5]
	s_and_b64 vcc, exec, s[2:3]
	v_mov_b64_e32 v[4:5], v[2:3]
	v_accvgpr_write_b32 a26, v5             ;  Reload Reuse
	v_accvgpr_write_b32 a27, v4             ;  Reload Reuse
	v_mov_b64_e32 v[4:5], v[2:3]
	v_accvgpr_write_b32 a28, v5             ;  Reload Reuse
	v_accvgpr_write_b32 a29, v4             ;  Reload Reuse
	v_mov_b64_e32 v[4:5], v[2:3]
	v_accvgpr_write_b32 a30, v5             ;  Reload Reuse
	v_accvgpr_write_b32 a31, v4             ;  Reload Reuse
	v_accvgpr_write_b32 a32, v3             ;  Reload Reuse
	v_accvgpr_write_b32 a33, v2             ;  Reload Reuse
	v_writelane_b32 v118, s0, 50
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a34, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_accvgpr_write_b32 a35, v1             ;  Reload Reuse
	v_accvgpr_write_b32 a36, v0             ;  Reload Reuse
                                        ; implicit-def: $vgpr118 : SGPR spill to VGPR lane
	s_cbranch_vccnz .LBB0_8
; %bb.1:
	.loc	1 628 14 is_stmt 1              ; mha.py:628:14
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s6, v117, 5
	v_readlane_b32 s7, v117, 6
	v_readlane_b32 s5, v117, 19
	v_readlane_b32 s10, v117, 7
	v_readlane_b32 s11, v117, 8
	v_readlane_b32 s0, v117, 26
	v_readlane_b32 s8, v117, 30
	v_readlane_b32 s9, v117, 31
	v_readlane_b32 s12, v117, 28
	v_readlane_b32 s13, v117, 29
	v_readlane_b32 s3, v117, 11
	v_readlane_b32 s2, v117, 48
	v_readlane_b32 s14, v117, 41
	v_readlane_b32 s15, v117, 42
	v_readlane_b32 s16, v117, 43
	v_readlane_b32 s17, v117, 44
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_accvgpr_read_b32 v0, a1               ;  Reload Reuse
	v_accvgpr_read_b32 v5, a5               ;  Reload Reuse
	v_accvgpr_read_b32 v2, a6               ;  Reload Reuse
	v_accvgpr_read_b32 v11, a11             ;  Reload Reuse
	v_accvgpr_read_b32 v10, a12             ;  Reload Reuse
	v_accvgpr_read_b32 v9, a13              ;  Reload Reuse
	v_accvgpr_read_b32 v8, a14              ;  Reload Reuse
	v_accvgpr_read_b32 v7, a15              ;  Reload Reuse
	v_accvgpr_read_b32 v6, a16              ;  Reload Reuse
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[16:17]
	v_mov_b32_e32 v1, v6
	.loc	1 617 21                        ; mha.py:617:21
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[14:15]
	v_mov_b32_e32 v3, v8
	.loc	1 608 14                        ; mha.py:608:14
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[14:15]
	v_mov_b32_e32 v4, v10
	s_mov_b32 s1, 5
	.loc	1 745 33                        ; mha.py:745:33
	s_lshl_b32 s2, s2, s1
	v_writelane_b32 v117, s2, 51
	s_mov_b32 s4, 0x3fb8aa3b
.Ltmp6:
	.loc	1 177 20                        ; mha.py:177:20 @[ mha.py:746:25 ]
	v_mov_b32_e32 v12, s4
	v_mul_f32_e64 v12, s3, v12
	.loc	1 261 19                        ; mha.py:261:19 @[ mha.py:746:25 ]
	v_accvgpr_write_b32 a38, v12            ;  Reload Reuse
	s_lshl_b64 s[24:25], s[12:13], s1
	.loc	1 264 19                        ; mha.py:264:19 @[ mha.py:746:25 ]
	v_writelane_b32 v117, s24, 52
	s_nop 1
	v_writelane_b32 v117, s25, 53
	s_lshl_b64 s[22:23], s[8:9], s1
	v_writelane_b32 v117, s22, 54
	s_nop 1
	v_writelane_b32 v117, s23, 55
	s_mov_b32 s3, 0
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	v_writelane_b32 v117, s3, 56
	s_cmp_gt_i32 s2, s3
	s_cselect_b64 s[26:27], -1, 0
	s_mov_b64 s[8:9], s[26:27]
	v_writelane_b32 v117, s8, 57
	s_nop 1
	v_writelane_b32 v117, s9, 58
	s_mov_b32 s13, 1
.Ltmp7:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_writelane_b32 v117, s13, 59
	v_lshlrev_b32_e64 v12, s13, v2
	s_mov_b32 s4, 64
	s_and_b32 s0, s0, s4
	v_writelane_b32 v117, s0, 60
	s_cmp_eq_u32 s0, s3
	s_mov_b32 s0, 0x88
	s_cselect_b32 s0, s3, s0
	s_mov_b32 s8, 0x17e
	v_mov_b32_e32 v2, s8
	v_bitop3_b32 v2, v12, s0, v2 bitop3:0x6c
	v_sub_u32_e64 v2, v2, v12
	v_ashrrev_i32_e64 v2, s13, v2
	s_mov_b32 s20, 32
	s_lshr_b64 s[8:9], s[10:11], s20
	s_mov_b32 s0, s8
	s_mov_b32 s9, 0xffff
	s_and_b32 s0, s0, s9
	s_mov_b32 s16, s10
	s_mov_b32 s10, s16
	s_mov_b32 s11, s0
	s_mov_b32 s15, 0x27000
	s_mov_b32 s21, 0x7ffffffe
                                        ; kill: def $sgpr16 killed $sgpr16 def $sgpr16_sgpr17_sgpr18_sgpr19
	s_mov_b32 s17, s0
	s_mov_b32 s18, s21
	s_mov_b32 s19, s15
	s_mov_b32 s28, s21
	s_mov_b32 s29, s15
	s_mov_b64 s[30:31], s[28:29]
	v_writelane_b32 v117, s30, 61
	s_nop 1
	v_writelane_b32 v117, s31, 62
	v_writelane_b32 v117, s10, 63
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a34, v117           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_writelane_b32 v118, s11, 0
	s_mov_b32 s0, 8
	s_lshl_b32 s14, s5, s0
	v_writelane_b32 v118, s14, 1
	s_mov_b32 s5, 0
	s_mov_b32 s8, 0x3840
	s_add_i32 s8, s5, s8
	s_add_i32 s8, s8, s14
	v_add_u32_e64 v5, v2, v5
	s_mov_b32 s10, 2
	v_lshlrev_b32_e64 v2, s10, v5
	v_mov_b32_e32 v12, v2
	v_accvgpr_write_b32 a39, v12            ;  Reload Reuse
	ds_bpermute_b32 v4, v2, v4
	v_mov_b32_e32 v12, s3
	v_cmp_gt_i32_e64 s[10:11], s2, v12
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v12, v5
	v_mov_b32_e32 v13, v14
	v_accvgpr_write_b32 a40, v13            ;  Reload Reuse
	v_accvgpr_write_b32 a41, v12            ;  Reload Reuse
	v_lshrrev_b64 v[12:13], v5, s[10:11]
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	v_and_b32_e64 v12, 1, v12
	v_cmp_eq_u32_e64 s[10:11], v12, 1
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e64 v12, s13, v4
	s_mov_b32 s12, 0x80000000
	v_mov_b32_e32 v4, s12
	v_cndmask_b32_e64 v4, v4, v12, s[10:11]
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dword v4, s[16:19], s3 offen lds
.Ltmp8:
	; asyncmark
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_mov_b32 s8, 0x2c40
	s_add_i32 s8, s5, s8
	s_add_i32 s8, s8, s14
	ds_bpermute_b32 v3, v2, v3
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e64 v4, s13, v3
	v_mov_b32_e32 v3, s12
	v_cndmask_b32_e64 v3, v3, v4, s[10:11]
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dword v3, s[16:19], s3 offen lds
.Ltmp9:
	; asyncmark
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_lshr_b64 s[10:11], s[6:7], s20
	s_mov_b32 s8, s10
	s_and_b32 s30, s8, s9
	s_mov_b32 s8, s6
	s_mov_b32 s6, s8
	s_mov_b32 s7, s30
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9_sgpr10_sgpr11
	s_mov_b32 s9, s30
	s_mov_b32 s10, s21
	s_mov_b32 s11, s15
	v_writelane_b32 v118, s28, 2
	s_nop 1
	v_writelane_b32 v118, s29, 3
	v_writelane_b32 v118, s6, 4
	s_nop 1
	v_writelane_b32 v118, s7, 5
	s_mov_b32 s6, 0x2000
	s_add_i32 s6, s5, s6
	s_add_i32 s6, s6, s14
	v_lshlrev_b32_e64 v3, s13, v1
	v_mov_b32_e32 v1, s12
	v_cndmask_b32_e64 v1, v1, v3, s[26:27]
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dword v1, s[8:11], s3 offen lds
.Ltmp10:
	; asyncmark
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_cmp_gt_i32 s2, s20
	s_cselect_b64 s[6:7], -1, 0
	.loc	1 261 9                         ; mha.py:261:9 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[24:25]
	v_mov_b64_e32 v[12:13], v[10:11]
	v_accvgpr_write_b32 a42, v13            ;  Reload Reuse
	v_accvgpr_write_b32 a43, v12            ;  Reload Reuse
	v_mov_b32_e32 v4, v10
	.loc	1 263 13                        ; mha.py:263:13 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[24:25]
	v_mov_b64_e32 v[10:11], v[8:9]
	v_accvgpr_write_b32 a44, v11            ;  Reload Reuse
	v_accvgpr_write_b32 a45, v10            ;  Reload Reuse
	v_mov_b32_e32 v3, v8
	.loc	1 264 9                         ; mha.py:264:9 @[ mha.py:746:25 ]
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[22:23]
	v_mov_b64_e32 v[8:9], v[6:7]
	v_accvgpr_write_b32 a46, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a47, v8             ;  Reload Reuse
	v_mov_b32_e32 v1, v6
.Ltmp11:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s15, 0x3c40
	s_add_i32 s15, s5, s15
	s_add_i32 s15, s15, s14
	ds_bpermute_b32 v4, v2, v4
	v_mov_b32_e32 v6, s20
	v_cmp_gt_i32_e64 s[20:21], s2, v6
	s_nop 1
	v_lshrrev_b64 v[6:7], v5, s[20:21]
	v_mov_b32_e32 v5, v6
	v_and_b32_e64 v5, 1, v5
	v_cmp_eq_u32_e64 s[20:21], v5, 1
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e64 v5, s13, v4
	v_mov_b32_e32 v4, s12
	v_cndmask_b32_e64 v4, v4, v5, s[20:21]
	s_mov_b32 m0, s15
	s_nop 0
	buffer_load_dword v4, s[16:19], s3 offen lds
.Ltmp12:
	; asyncmark
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_mov_b32 s15, 0x3040
	s_add_i32 s15, s5, s15
	s_add_i32 s15, s15, s14
	ds_bpermute_b32 v2, v2, v3
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e64 v3, s13, v2
	v_mov_b32_e32 v2, s12
	v_cndmask_b32_e64 v2, v2, v3, s[20:21]
	s_mov_b32 m0, s15
	s_nop 0
	buffer_load_dword v2, s[16:19], s3 offen lds
.Ltmp13:
	; asyncmark
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_mov_b32 s15, 0x2420
	s_add_i32 s5, s5, s15
	s_add_i32 s5, s5, s14
	v_lshlrev_b32_e64 v2, s13, v1
	v_mov_b32_e32 v1, s12
	v_cndmask_b32_e64 v1, v1, v2, s[6:7]
	s_mov_b32 m0, s5
	s_nop 0
	buffer_load_dword v1, s[8:11], s3 offen lds
.Ltmp14:
	; asyncmark
	; wait_asyncmark(3)
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt vmcnt(3) lgkmcnt(0)
	s_barrier
.Ltmp15:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_sub_i32 s2, s2, s4
	v_writelane_b32 v118, s2, 6
	s_cmp_le_i32 s2, s3
	s_cselect_b64 s[2:3], -1, 0
	v_lshlrev_b32_e64 v1, s1, v0
	s_mov_b32 s1, 0x2e0
	v_and_b32_e64 v1, v1, s1
	v_accvgpr_write_b32 a48, v1             ;  Reload Reuse
	v_and_b32_e64 v0, v0, s0
	s_mov_b64 s[0:1], -1
	v_accvgpr_write_b32 a49, v0             ;  Reload Reuse
	s_xor_b64 s[2:3], s[2:3], s[0:1]
	s_and_b64 vcc, exec, s[2:3]
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr2
	v_writelane_b32 v118, s0, 7
	s_nop 1
	v_writelane_b32 v118, s1, 8
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_cbranch_vccnz .LBB0_3
.Ltmp16:
; %bb.2:                                ; %.._crit_edge_crit_edge
	.loc	1 746 25                        ; mha.py:746:25
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_accvgpr_read_b32 v1, a5               ;  Reload Reuse
	v_accvgpr_read_b32 v0, a4               ;  Reload Reuse
	s_mov_b32 s0, 0
	s_mov_b32 s1, 0x3840
	s_add_i32 s2, s0, s1
	s_mov_b32 s1, 0x2c40
	s_add_i32 s4, s0, s1
	s_mov_b32 s1, 0x2000
	s_add_i32 s6, s0, s1
	s_mov_b32 s1, 0x3c40
	s_add_i32 s3, s0, s1
	s_mov_b32 s1, 0x3040
	s_add_i32 s5, s0, s1
	s_mov_b32 s1, 0x2420
	s_add_i32 s7, s0, s1
	s_mov_b32 s0, 1
.Ltmp17:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_lshrrev_b32_e64 v0, s0, v0
.Ltmp18:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_mov_b32 s0, 3
	v_lshlrev_b32_e64 v2, s0, v1
                                        ; implicit-def: $sgpr0_sgpr1
	s_mov_b32 s0, s1
	v_mov_b32_e32 v3, s0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	s_mov_b64 s[0:1], 0
	v_accvgpr_write_b32 a50, v1             ;  Reload Reuse
	v_accvgpr_write_b32 a51, v0             ;  Reload Reuse
	v_writelane_b32 v118, s7, 9
	v_writelane_b32 v118, s6, 10
	v_writelane_b32 v118, s5, 11
	v_writelane_b32 v118, s4, 12
	v_writelane_b32 v118, s3, 13
	v_writelane_b32 v118, s2, 14
.Ltmp19:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	v_writelane_b32 v118, s0, 7
	s_nop 1
	v_writelane_b32 v118, s1, 8
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
.LBB0_3:                                ; %Flow8
	.loc	1 0 5 is_stmt 0                 ; mha.py:0:5
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s6, v118, 7
	v_readlane_b32 s7, v118, 8
	v_readlane_b32 s5, v118, 9
	v_readlane_b32 s4, v118, 10
	v_readlane_b32 s3, v118, 11
	v_readlane_b32 s2, v118, 12
	v_readlane_b32 s1, v118, 13
	v_readlane_b32 s0, v118, 14
	v_accvgpr_read_b32 v11, a50             ;  Reload Reuse
	v_accvgpr_read_b32 v10, a51             ;  Reload Reuse
	s_mov_b32 s8, 0
	v_mov_b32_e32 v2, s8
	v_mov_b32_e32 v0, s8
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v0, 1.0
	v_mov_b32_e32 v1, 0xff800000
	s_mov_b64 s[8:9], -1
	s_xor_b64 s[6:7], s[6:7], s[8:9]
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b64_e32 v[8:9], v[2:3]
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b64_e32 v[4:5], v[2:3]
                                        ; kill: def $vgpr2_vgpr3 killed $vgpr2_vgpr3 killed $exec
                                        ; kill: def $vgpr1 killed $vgpr1 killed $exec
                                        ; kill: def $vgpr0 killed $vgpr0 killed $exec
	v_accvgpr_write_b32 a52, v11            ;  Reload Reuse
	v_accvgpr_write_b32 a53, v10            ;  Reload Reuse
	v_accvgpr_write_b32 a54, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a55, v8             ;  Reload Reuse
	v_accvgpr_write_b32 a56, v7             ;  Reload Reuse
	v_accvgpr_write_b32 a57, v6             ;  Reload Reuse
	v_accvgpr_write_b32 a58, v5             ;  Reload Reuse
	v_accvgpr_write_b32 a59, v4             ;  Reload Reuse
	v_accvgpr_write_b32 a60, v3             ;  Reload Reuse
	v_accvgpr_write_b32 a61, v2             ;  Reload Reuse
	v_writelane_b32 v118, s5, 15
	v_writelane_b32 v118, s4, 16
	v_writelane_b32 v118, s3, 17
	v_writelane_b32 v118, s2, 18
	v_writelane_b32 v118, s1, 19
	v_writelane_b32 v118, s0, 20
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_accvgpr_write_b32 a62, v1             ;  Reload Reuse
	v_accvgpr_write_b32 a63, v0             ;  Reload Reuse
	s_cbranch_vccnz .LBB0_5
; %bb.4:                                ; %.lr.ph
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s10, v117, 19
	v_readlane_b32 s8, v117, 26
	v_readlane_b32 s13, v117, 60
	v_readlane_b32 s12, v117, 18
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_accvgpr_read_b32 v11, a42             ;  Reload Reuse
	v_accvgpr_read_b32 v10, a43             ;  Reload Reuse
	v_accvgpr_read_b32 v13, a44             ;  Reload Reuse
	v_accvgpr_read_b32 v12, a45             ;  Reload Reuse
	v_accvgpr_read_b32 v15, a46             ;  Reload Reuse
	v_accvgpr_read_b32 v14, a47             ;  Reload Reuse
	v_accvgpr_read_b32 v0, a38              ;  Reload Reuse
	v_accvgpr_read_b32 v3, a3               ;  Reload Reuse
	v_accvgpr_read_b32 v1, a1               ;  Reload Reuse
	v_accvgpr_read_b32 v2, a4               ;  Reload Reuse
	v_accvgpr_read_b32 v5, a2               ;  Reload Reuse
	v_accvgpr_read_b32 v4, a25              ;  Reload Reuse
	v_accvgpr_read_b32 v16, a24             ;  Reload Reuse
	v_accvgpr_read_b32 v17, a23             ;  Reload Reuse
	v_accvgpr_read_b32 v6, a22              ;  Reload Reuse
	v_accvgpr_read_b32 v7, a21              ;  Reload Reuse
	v_accvgpr_read_b32 v8, a20              ;  Reload Reuse
	v_accvgpr_read_b32 v9, a19              ;  Reload Reuse
	v_accvgpr_read_b32 v18, a18             ;  Reload Reuse
	v_accvgpr_read_b32 v20, a48             ;  Reload Reuse
	v_accvgpr_read_b32 v21, a5              ;  Reload Reuse
	v_accvgpr_read_b32 v19, a49             ;  Reload Reuse
	s_mov_b32 s9, 0
	v_writelane_b32 v118, s9, 21
	s_mov_b32 s0, 0x3840
	s_add_i32 s2, s9, s0
	s_mov_b32 s0, 0x2c40
	s_add_i32 s4, s9, s0
	s_mov_b32 s0, 0x2000
	s_add_i32 s6, s9, s0
	s_mov_b32 s0, 0x3c40
	s_add_i32 s3, s9, s0
	s_mov_b32 s0, 0x3040
	s_add_i32 s5, s9, s0
	s_mov_b32 s0, 0x2420
	s_add_i32 s7, s9, s0
	s_mov_b32 s0, 0
	v_cmp_eq_u32_e64 s[14:15], v19, s0
	s_mov_b32 s1, 0x110
	v_mov_b32_e32 v19, s1
	v_mov_b32_e32 v22, s0
	v_cndmask_b32_e64 v19, v19, v22, s[14:15]
	s_mov_b32 s11, 3
	v_lshlrev_b32_e64 v22, s11, v21
	s_mov_b32 s1, 1
	v_lshlrev_b32_e64 v24, s1, v2
                                        ; kill: def $vgpr24 killed $vgpr24 def $vgpr24_vgpr25 killed $exec
	v_mov_b32_e32 v25, v22
	v_lshrrev_b32_e64 v21, s11, v21
	v_lshrrev_b32_e64 v22, s1, v2
                                        ; kill: def $vgpr22 killed $vgpr22 def $vgpr22_vgpr23 killed $exec
	v_mov_b32_e32 v23, v21
	v_mov_b32_e32 v24, v25
	v_accvgpr_write_b32 a64, v24            ;  Reload Reuse
	v_mov_b32_e32 v21, v22
	v_mov_b32_e32 v22, v21
	v_mov_b32_e32 v23, v24
	v_accvgpr_write_b32 a65, v23            ;  Reload Reuse
	v_accvgpr_write_b32 a66, v22            ;  Reload Reuse
	v_bitop3_b32 v19, v19, v20, v21 bitop3:0xde
	v_accvgpr_write_b32 a67, v19            ;  Reload Reuse
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19_vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v19, v9
	v_mov_b32_e32 v20, v8
	v_mov_b32_e32 v21, v7
	v_accvgpr_write_b32 a68, v21            ;  Reload Reuse
	v_accvgpr_write_b32 a69, v20            ;  Reload Reuse
	v_accvgpr_write_b32 a70, v19            ;  Reload Reuse
	v_accvgpr_write_b32 a71, v18            ;  Reload Reuse
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7_vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v7, v17
	v_mov_b32_e32 v8, v16
	v_mov_b32_e32 v9, v4
	v_accvgpr_write_b32 a72, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a73, v8             ;  Reload Reuse
	v_accvgpr_write_b32 a74, v7             ;  Reload Reuse
	v_accvgpr_write_b32 a75, v6             ;  Reload Reuse
	s_mov_b32 s14, 31
	v_and_b32_e64 v4, v1, s14
	s_lshl_b32 s12, s12, s1
	s_mov_b32 s14, 4
	s_and_b32 s12, s12, s14
	v_mov_b32_e32 v6, s9
	v_lshl_add_u32 v7, v4, s11, v6
	s_mov_b32 s14, 2
	v_mov_b32_e32 v6, s14
	v_lshl_add_u32 v6, s13, v6, v7
	v_add_u32_e64 v6, v6, s12
	v_accvgpr_write_b32 a76, v6             ;  Reload Reuse
	v_mov_b32_e32 v6, s9
	v_lshl_add_u32 v6, v5, s11, v6
	s_mov_b32 s11, 7
	v_mov_b32_e32 v5, s11
	v_lshl_add_u32 v5, s10, v5, v6
	v_accvgpr_write_b32 a77, v5             ;  Reload Reuse
	v_lshlrev_b32_e64 v4, s1, v4
	v_cmp_eq_u32_e64 s[12:13], v2, s0
	s_mov_b32 s11, 0x420
	v_mov_b32_e32 v2, s11
	v_mov_b32_e32 v5, s0
	v_cndmask_b32_e64 v2, v2, v5, s[12:13]
	v_bitop3_b32 v2, s8, v2, v4 bitop3:0x36
	v_accvgpr_write_b32 a78, v2             ;  Reload Reuse
	v_add_u32_e64 v4, s9, v2
	v_mov_b32_e32 v5, v4
	v_accvgpr_write_b32 a79, v5             ;  Reload Reuse
	s_mov_b32 s8, 0x1000
	v_writelane_b32 v118, s8, 22
	v_add_u32_e64 v4, v4, s8
	v_accvgpr_write_b32 a80, v4             ;  Reload Reuse
	s_mov_b32 s11, 0x108
	v_xor_b32_e64 v4, v2, s11
	v_add_u32_e64 v4, s9, v4
	v_mov_b32_e32 v5, v4
	v_accvgpr_write_b32 a81, v5             ;  Reload Reuse
	v_add_u32_e64 v4, v4, s8
	v_accvgpr_write_b32 a82, v4             ;  Reload Reuse
	s_mov_b32 s11, 0x210
	v_xor_b32_e64 v4, v2, s11
	v_add_u32_e64 v4, s9, v4
	v_mov_b32_e32 v5, v4
	v_accvgpr_write_b32 a83, v5             ;  Reload Reuse
	v_add_u32_e64 v4, v4, s8
	v_accvgpr_write_b32 a84, v4             ;  Reload Reuse
	s_mov_b32 s11, 0x318
	v_xor_b32_e64 v4, v2, s11
	v_add_u32_e64 v4, s9, v4
	v_mov_b32_e32 v5, v4
	v_accvgpr_write_b32 a85, v5             ;  Reload Reuse
	v_add_u32_e64 v4, v4, s8
	v_accvgpr_write_b32 a86, v4             ;  Reload Reuse
	s_mov_b32 s11, 0x840
	v_xor_b32_e64 v4, v2, s11
	v_add_u32_e64 v4, s9, v4
	v_mov_b32_e32 v5, v4
	v_accvgpr_write_b32 a87, v5             ;  Reload Reuse
	v_add_u32_e64 v4, v4, s8
	v_accvgpr_write_b32 a88, v4             ;  Reload Reuse
	s_mov_b32 s11, 0x948
	v_xor_b32_e64 v4, v2, s11
	v_add_u32_e64 v4, s9, v4
	v_mov_b32_e32 v5, v4
	v_accvgpr_write_b32 a89, v5             ;  Reload Reuse
	v_add_u32_e64 v4, v4, s8
	v_accvgpr_write_b32 a90, v4             ;  Reload Reuse
	s_mov_b32 s11, 0xa50
	v_xor_b32_e64 v4, v2, s11
	v_add_u32_e64 v4, s9, v4
	v_mov_b32_e32 v5, v4
	v_accvgpr_write_b32 a91, v5             ;  Reload Reuse
	v_add_u32_e64 v4, v4, s8
	v_accvgpr_write_b32 a92, v4             ;  Reload Reuse
	s_mov_b32 s11, 0xb58
	v_xor_b32_e64 v2, v2, s11
	v_add_u32_e64 v2, s9, v2
	v_mov_b32_e32 v4, v2
	v_accvgpr_write_b32 a93, v4             ;  Reload Reuse
	v_add_u32_e64 v2, v2, s8
	v_accvgpr_write_b32 a94, v2             ;  Reload Reuse
	s_mov_b32 s11, 60
	v_and_b32_e64 v2, v1, s11
	s_mov_b32 s11, 6
	v_lshlrev_b32_e64 v1, s11, v2
	s_mov_b32 s11, 24
	v_and_b32_e64 v3, v3, s11
	v_lshlrev_b32_e64 v2, s1, v2
	s_mov_b32 s11, 5
	s_lshl_b32 s10, s10, s11
	v_bitop3_b32 v1, v1, v2, v3 bitop3:0x36
	v_xor_b32_e64 v1, v1, s10
	v_add_u32_e64 v1, s9, v1
	v_mov_b32_e32 v2, v1
	v_accvgpr_write_b32 a95, v2             ;  Reload Reuse
	v_add_u32_e64 v2, v1, s8
	v_accvgpr_write_b32 a96, v2             ;  Reload Reuse
	s_mov_b32 s8, 0x80
	v_add_u32_e64 v2, v1, s8
	v_accvgpr_write_b32 a97, v2             ;  Reload Reuse
	s_mov_b32 s8, 0x1080
	v_add_u32_e64 v1, v1, s8
	.loc	1 186 18 is_stmt 1              ; mha.py:186:18 @[ mha.py:746:25 ]
	v_accvgpr_write_b32 a98, v1             ;  Reload Reuse
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_accvgpr_write_b32 a103, v5            ;  Reload Reuse
	v_accvgpr_write_b32 a104, v4            ;  Reload Reuse
	v_accvgpr_write_b32 a105, v3            ;  Reload Reuse
	v_accvgpr_write_b32 a106, v2            ;  Reload Reuse
	v_accvgpr_write_b32 a99, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a100, v8            ;  Reload Reuse
	v_accvgpr_write_b32 a101, v7            ;  Reload Reuse
	v_accvgpr_write_b32 a102, v6            ;  Reload Reuse
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_accvgpr_write_b32 a107, v5            ;  Reload Reuse
	v_accvgpr_write_b32 a108, v4            ;  Reload Reuse
	v_accvgpr_write_b32 a109, v3            ;  Reload Reuse
	v_accvgpr_write_b32 a110, v2            ;  Reload Reuse
                                        ; implicit-def: $sgpr8_sgpr9
	s_mov_b32 s8, s9
	v_mov_b32_e32 v2, s8
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v0, v2
	v_mov_b32_e32 v1, v2
	v_accvgpr_write_b32 a111, v1            ;  Reload Reuse
	v_accvgpr_write_b32 a112, v0            ;  Reload Reuse
	s_mov_b32 s8, 0
	v_mov_b32_e32 v0, s8
	v_mov_b32_e32 v2, s8
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v8, 1.0
	v_mov_b32_e32 v9, 0xff800000
                                        ; kill: def $vgpr14_vgpr15 killed $vgpr14_vgpr15 killed $exec
                                        ; kill: def $vgpr12_vgpr13 killed $vgpr12_vgpr13 killed $exec
                                        ; kill: def $vgpr10_vgpr11 killed $vgpr10_vgpr11 killed $exec
                                        ; kill: def $vgpr9 killed $vgpr9 killed $exec
                                        ; kill: def $vgpr8 killed $vgpr8 killed $exec
	v_mov_b64_e32 v[6:7], v[0:1]
	v_mov_b64_e32 v[4:5], v[0:1]
	v_mov_b64_e32 v[2:3], v[0:1]
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	v_writelane_b32 v118, s7, 23
	v_writelane_b32 v118, s6, 24
	v_writelane_b32 v118, s5, 25
	v_writelane_b32 v118, s4, 26
	v_writelane_b32 v118, s3, 27
	v_writelane_b32 v118, s2, 28
	v_writelane_b32 v118, s1, 29
	v_accvgpr_write_b32 a113, v15           ;  Reload Reuse
	v_accvgpr_write_b32 a114, v14           ;  Reload Reuse
	v_accvgpr_write_b32 a115, v13           ;  Reload Reuse
	v_accvgpr_write_b32 a116, v12           ;  Reload Reuse
	v_accvgpr_write_b32 a117, v11           ;  Reload Reuse
	v_accvgpr_write_b32 a118, v10           ;  Reload Reuse
	v_accvgpr_write_b32 a119, v9            ;  Reload Reuse
	v_accvgpr_write_b32 a120, v8            ;  Reload Reuse
	v_writelane_b32 v118, s0, 30
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_accvgpr_write_b32 a121, v7            ;  Reload Reuse
	v_accvgpr_write_b32 a122, v6            ;  Reload Reuse
	v_accvgpr_write_b32 a123, v5            ;  Reload Reuse
	v_accvgpr_write_b32 a124, v4            ;  Reload Reuse
	v_accvgpr_write_b32 a125, v3            ;  Reload Reuse
	v_accvgpr_write_b32 a126, v2            ;  Reload Reuse
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	v_accvgpr_write_b32 a127, v1            ;  Reload Reuse
	scratch_store_dword off, v0, off offset:4 ; 4-byte Folded Spill
	s_branch .LBB0_6
.LBB0_5:                                ; %Flow9
	.loc	1 0 5 is_stmt 0                 ; mha.py:0:5
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 15
	v_readlane_b32 s1, v118, 16
	v_readlane_b32 s2, v118, 17
	v_readlane_b32 s3, v118, 18
	v_readlane_b32 s4, v118, 19
	v_readlane_b32 s5, v118, 20
	v_accvgpr_read_b32 v1, a52              ;  Reload Reuse
	v_accvgpr_read_b32 v0, a53              ;  Reload Reuse
	v_accvgpr_read_b32 v3, a54              ;  Reload Reuse
	v_accvgpr_read_b32 v2, a55              ;  Reload Reuse
	v_accvgpr_read_b32 v5, a56              ;  Reload Reuse
	v_accvgpr_read_b32 v4, a57              ;  Reload Reuse
	v_accvgpr_read_b32 v7, a58              ;  Reload Reuse
	v_accvgpr_read_b32 v6, a59              ;  Reload Reuse
	v_accvgpr_read_b32 v9, a60              ;  Reload Reuse
	v_accvgpr_read_b32 v8, a61              ;  Reload Reuse
	v_accvgpr_read_b32 v10, a62             ;  Reload Reuse
	v_accvgpr_read_b32 v11, a63             ;  Reload Reuse
	scratch_store_dword off, v11, off offset:56 ; 4-byte Folded Spill
	scratch_store_dword off, v10, off offset:52 ; 4-byte Folded Spill
	v_writelane_b32 v118, s5, 31
	v_writelane_b32 v118, s4, 32
	v_writelane_b32 v118, s3, 33
	v_writelane_b32 v118, s2, 34
	v_writelane_b32 v118, s1, 35
	v_writelane_b32 v118, s0, 36
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	scratch_store_dwordx2 off, v[8:9], off offset:44 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:36 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:28 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:20 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], off offset:12 ; 8-byte Folded Spill
	s_branch .LBB0_9
.LBB0_6:                                ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s6, v118, 30
	v_readlane_b32 s2, v118, 29
	v_readlane_b32 s1, v118, 28
	v_readlane_b32 s3, v118, 27
	v_readlane_b32 s0, v118, 26
	v_readlane_b32 s4, v118, 25
	v_readlane_b32 s10, v118, 24
	v_readlane_b32 s5, v118, 23
	v_readlane_b32 s8, v118, 6
	v_readlane_b32 s14, v118, 2
	v_readlane_b32 s15, v118, 3
	v_readlane_b32 s12, v118, 4
	v_readlane_b32 s13, v118, 5
	v_readlane_b32 s16, v118, 1
	v_readlane_b32 s22, v117, 61
	v_readlane_b32 s23, v117, 62
	v_readlane_b32 s20, v117, 63
	v_readlane_b32 s21, v118, 0
	v_readlane_b32 s18, v117, 54
	v_readlane_b32 s19, v117, 55
	v_readlane_b32 s24, v117, 52
	v_readlane_b32 s25, v117, 53
	v_accvgpr_read_b32 v11, a127            ;  Reload Reuse
	scratch_load_dword v10, off, off offset:4 ; 4-byte Folded Reload
	v_accvgpr_read_b32 v33, a125            ;  Reload Reuse
	v_accvgpr_read_b32 v32, a126            ;  Reload Reuse
	v_accvgpr_read_b32 v19, a123            ;  Reload Reuse
	v_accvgpr_read_b32 v18, a124            ;  Reload Reuse
	v_accvgpr_read_b32 v17, a121            ;  Reload Reuse
	v_accvgpr_read_b32 v16, a122            ;  Reload Reuse
	v_accvgpr_read_b32 v58, a120            ;  Reload Reuse
	v_accvgpr_read_b32 v59, a119            ;  Reload Reuse
	v_accvgpr_read_b32 v9, a117             ;  Reload Reuse
	v_accvgpr_read_b32 v8, a118             ;  Reload Reuse
	v_accvgpr_read_b32 v23, a115            ;  Reload Reuse
	v_accvgpr_read_b32 v22, a116            ;  Reload Reuse
	v_accvgpr_read_b32 v25, a113            ;  Reload Reuse
	v_accvgpr_read_b32 v24, a114            ;  Reload Reuse
	v_accvgpr_read_b32 v14, a98             ;  Reload Reuse
	v_accvgpr_read_b32 v12, a97             ;  Reload Reuse
	v_accvgpr_read_b32 v2, a96              ;  Reload Reuse
	v_accvgpr_read_b32 v0, a95              ;  Reload Reuse
	v_accvgpr_read_b32 v1, a94              ;  Reload Reuse
	v_accvgpr_read_b32 v13, a93             ;  Reload Reuse
	v_accvgpr_read_b32 v28, a92             ;  Reload Reuse
	v_accvgpr_read_b32 v30, a91             ;  Reload Reuse
	v_accvgpr_read_b32 v34, a90             ;  Reload Reuse
	v_accvgpr_read_b32 v36, a89             ;  Reload Reuse
	v_accvgpr_read_b32 v38, a88             ;  Reload Reuse
	v_accvgpr_read_b32 v40, a87             ;  Reload Reuse
	v_accvgpr_read_b32 v42, a86             ;  Reload Reuse
	v_accvgpr_read_b32 v44, a85             ;  Reload Reuse
	v_accvgpr_read_b32 v46, a84             ;  Reload Reuse
	v_accvgpr_read_b32 v48, a83             ;  Reload Reuse
	v_accvgpr_read_b32 v50, a82             ;  Reload Reuse
	v_accvgpr_read_b32 v52, a81             ;  Reload Reuse
	v_accvgpr_read_b32 v54, a80             ;  Reload Reuse
	v_accvgpr_read_b32 v56, a79             ;  Reload Reuse
	v_accvgpr_read_b32 v26, a77             ;  Reload Reuse
	v_accvgpr_read_b32 v27, a76             ;  Reload Reuse
	v_accvgpr_read_b32 v61, a111            ;  Reload Reuse
	v_accvgpr_read_b32 v60, a112            ;  Reload Reuse
	v_accvgpr_read_b32 v3, a38              ;  Reload Reuse
	v_accvgpr_read_b32 v95, a107            ;  Reload Reuse
	v_accvgpr_read_b32 v94, a108            ;  Reload Reuse
	v_accvgpr_read_b32 v93, a109            ;  Reload Reuse
	v_accvgpr_read_b32 v92, a110            ;  Reload Reuse
	v_accvgpr_read_b32 v111, a103           ;  Reload Reuse
	v_accvgpr_read_b32 v110, a104           ;  Reload Reuse
	v_accvgpr_read_b32 v109, a105           ;  Reload Reuse
	v_accvgpr_read_b32 v108, a106           ;  Reload Reuse
	v_accvgpr_read_b32 v115, a99            ;  Reload Reuse
	v_accvgpr_read_b32 v114, a100           ;  Reload Reuse
	v_accvgpr_read_b32 v113, a101           ;  Reload Reuse
	v_accvgpr_read_b32 v112, a102           ;  Reload Reuse
	v_accvgpr_read_b32 v69, a72             ;  Reload Reuse
	v_accvgpr_read_b32 v68, a73             ;  Reload Reuse
	v_accvgpr_read_b32 v67, a74             ;  Reload Reuse
	v_accvgpr_read_b32 v66, a75             ;  Reload Reuse
	v_accvgpr_read_b32 v99, a68             ;  Reload Reuse
	v_accvgpr_read_b32 v98, a69             ;  Reload Reuse
	v_accvgpr_read_b32 v97, a70             ;  Reload Reuse
	v_accvgpr_read_b32 v96, a71             ;  Reload Reuse
	v_accvgpr_read_b32 v4, a64              ;  Reload Reuse
	v_accvgpr_read_b32 v6, a67              ;  Reload Reuse
	v_accvgpr_read_b32 v7, a39              ;  Reload Reuse
	v_accvgpr_read_b32 v63, a40             ;  Reload Reuse
	v_accvgpr_read_b32 v62, a41             ;  Reload Reuse
	s_mov_b32 s9, 32
	.loc	1 261 9 is_stmt 1               ; mha.py:261:9 @[ mha.py:746:25 ]
	v_lshlrev_b64 v[8:9], s9, v[8:9]
	v_ashrrev_i64 v[8:9], s9, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[24:25]
	v_mov_b64_e32 v[20:21], v[8:9]
	v_mov_b32_e32 v9, v8
	.loc	1 263 13                        ; mha.py:263:13 @[ mha.py:746:25 ]
	v_lshlrev_b64 v[22:23], s9, v[22:23]
	v_ashrrev_i64 v[22:23], s9, v[22:23]
	v_lshl_add_u64 v[64:65], v[22:23], 0, s[24:25]
	v_mov_b64_e32 v[22:23], v[64:65]
	v_mov_b32_e32 v8, v64
	.loc	1 264 9                         ; mha.py:264:9 @[ mha.py:746:25 ]
	v_lshlrev_b64 v[24:25], s9, v[24:25]
	v_ashrrev_i64 v[24:25], s9, v[24:25]
	v_lshl_add_u64 v[64:65], v[24:25], 0, s[18:19]
	v_mov_b64_e32 v[24:25], v[64:65]
	v_mov_b32_e32 v5, v64
	s_mov_b32 s17, 1
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_add_i32 s2, s2, s17
	s_mov_b32 s7, 3
	s_cmp_lt_i32 s2, s7
	s_mov_b32 s11, 0
	s_cselect_b32 s7, s2, s11
	s_mov_b32 s2, 9
.Ltmp20:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_lshl_b32 s19, s7, s2
	s_mov_b32 s18, 0
	s_mov_b32 s2, 0x3840
	s_add_i32 s2, s18, s2
	s_lshl_b32 s19, s19, s17
	s_add_i32 s2, s2, s19
	s_add_i32 s24, s2, s16
	ds_bpermute_b32 v9, v7, v9
	s_mov_b64 s[26:27], exec
	v_mov_b32_e32 v15, v62
	v_lshrrev_b64 v[62:63], v15, s[26:27]
	v_mov_b32_e32 v15, v62
	v_and_b32_e64 v15, 1, v15
	v_cmp_eq_u32_e64 s[26:27], v15, 1
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e64 v15, s17, v9
	s_mov_b32 s25, 0x80000000
	v_mov_b32_e32 v9, s25
	v_cndmask_b32_e64 v9, v9, v15, s[26:27]
	s_mov_b32 s30, s21
	s_mov_b32 s28, s23
	s_mov_b32 s29, s22
                                        ; kill: def $sgpr20 killed $sgpr20 def $sgpr20_sgpr21_sgpr22_sgpr23
	s_mov_b32 s21, s30
	s_mov_b32 s22, s29
	s_mov_b32 s23, s28
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dword v9, s[20:23], s11 offen lds
	; asyncmark
	v_add_u32_e64 v9, s1, v6
	ds_read_b128 v[62:65], v9
.Ltmp21:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_mov_b32 s1, 0x2c40
	s_add_i32 s1, s18, s1
	s_add_i32 s1, s1, s19
	s_add_i32 s24, s1, s16
	ds_bpermute_b32 v7, v7, v8
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e64 v8, s17, v7
	v_mov_b32_e32 v7, s25
	v_cndmask_b32_e64 v7, v7, v8, s[26:27]
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dword v7, s[20:23], s11 offen lds
	; asyncmark
	v_add_u32_e64 v6, s0, v6
	ds_read_b128 v[70:73], v6
	s_mov_b32 s0, 4
.Ltmp22:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	s_lshl_b32 s0, s7, s0
	s_mov_b32 s20, 0x7fffff0
	s_and_b32 s0, s0, s20
	s_mov_b32 s20, 0x2000
	s_add_i32 s18, s18, s20
	s_add_i32 s18, s18, s19
	s_lshl1_add_u32 s0, s0, s18
	s_add_i32 s16, s0, s16
	v_lshlrev_b32_e64 v5, s17, v5
	s_mov_b32 s19, s13
	s_mov_b32 s17, s15
	s_mov_b32 s18, s14
                                        ; kill: def $sgpr12 killed $sgpr12 def $sgpr12_sgpr13_sgpr14_sgpr15
	s_mov_b32 s13, s19
	s_mov_b32 s14, s18
	s_mov_b32 s15, s17
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dword v5, s[12:15], s11 offen lds
	; asyncmark
	v_add_u32_e64 v6, s10, v4
	ds_read_b64_tr_b16 v[4:5], v6
	ds_read_b64_tr_b16 v[6:7], v6 offset:512
	s_mov_b32 s10, 0
.Ltmp23:
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:746:25 ]
	v_writelane_b32 v118, s10, 37
	v_mov_b32_e32 v76, s10
	v_mov_b32_e32 v53, s10
	v_mov_b32_e32 v51, s10
	v_mov_b32_e32 v49, s10
	v_mov_b32_e32 v47, s10
	v_mov_b32_e32 v45, s10
	v_mov_b32_e32 v43, s10
	v_mov_b32_e32 v41, s10
	v_mov_b32_e32 v39, s10
	v_mov_b32_e32 v37, s10
	v_mov_b32_e32 v35, s10
	v_mov_b32_e32 v31, s10
	v_mov_b32_e32 v29, s10
	v_mov_b32_e32 v15, s10
	v_mov_b32_e32 v9, s10
	v_mov_b32_e32 v8, s10
                                        ; kill: def $vgpr76 killed $vgpr76 def $vgpr76_vgpr77_vgpr78_vgpr79_vgpr80_vgpr81_vgpr82_vgpr83_vgpr84_vgpr85_vgpr86_vgpr87_vgpr88_vgpr89_vgpr90_vgpr91 killed $exec
	v_mov_b32_e32 v77, v53
	v_mov_b32_e32 v78, v51
	v_mov_b32_e32 v79, v49
	v_mov_b32_e32 v80, v47
	v_mov_b32_e32 v81, v45
	v_mov_b32_e32 v82, v43
	v_mov_b32_e32 v83, v41
	v_mov_b32_e32 v84, v39
	v_mov_b32_e32 v85, v37
	v_mov_b32_e32 v86, v35
	v_mov_b32_e32 v87, v31
	v_mov_b32_e32 v88, v29
	v_mov_b32_e32 v89, v15
	v_mov_b32_e32 v90, v9
	v_mov_b32_e32 v91, v8
	scratch_store_dwordx4 off, v[76:79], off offset:228 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[80:83], off offset:244 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[84:87], off offset:260 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[88:91], off offset:276 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_32x32x16_bf16 v[76:91], v[70:73], v[96:99], v[76:91]
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	v_mfma_f32_32x32x16_bf16 v[76:91], v[62:65], v[66:69], v[76:91]
	s_nop 11
	scratch_store_dwordx4 off, v[76:79], off offset:164 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[80:83], off offset:180 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[84:87], off offset:196 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[88:91], off offset:212 ; 16-byte Folded Spill
	v_mov_b32_e32 v15, v83
	v_mov_b32_e32 v29, v82
	v_mov_b32_e32 v31, v81
	v_mov_b32_e32 v35, v80
	v_mov_b32_e32 v37, v79
	v_mov_b32_e32 v39, v78
	v_mov_b32_e32 v41, v77
	v_mov_b32_e32 v64, v76
	v_mov_b32_e32 v8, v84
	v_mov_b32_e32 v9, v91
                                        ; kill: def $vgpr64 killed $vgpr64 def $vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71 killed $exec
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v65, v41
	v_mov_b32_e32 v66, v39
	v_mov_b32_e32 v67, v37
	v_mov_b32_e32 v68, v35
	v_mov_b32_e32 v69, v31
	v_mov_b32_e32 v70, v29
	v_mov_b32_e32 v71, v15
	scratch_store_dwordx4 off, v[64:67], off offset:132 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[68:71], off offset:148 ; 16-byte Folded Spill
	v_mov_b32_e32 v31, v67
	v_mov_b32_e32 v35, v66
	v_mov_b32_e32 v37, v65
	v_mov_b32_e32 v62, v64
	v_mov_b32_e32 v15, v71
	v_mov_b32_e32 v29, v70
	v_mov_b32_e32 v39, v69
	v_mov_b32_e32 v104, v68
                                        ; kill: def $vgpr104 killed $vgpr104 def $vgpr104_vgpr105_vgpr106_vgpr107 killed $exec
	v_mov_b32_e32 v105, v39
	v_mov_b32_e32 v106, v29
	v_mov_b32_e32 v107, v15
	v_mov_b32_e32 v15, v107
	v_mov_b32_e32 v66, v106
                                        ; kill: def $vgpr66 killed $vgpr66 def $vgpr66_vgpr67 killed $exec
	v_mov_b32_e32 v67, v15
	v_mov_b32_e32 v15, v115
	v_mov_b32_e32 v29, v114
	v_mov_b32_e32 v39, v113
	v_mov_b32_e32 v100, v112
                                        ; kill: def $vgpr100 killed $vgpr100 def $vgpr100_vgpr101_vgpr102_vgpr103 killed $exec
	v_mov_b32_e32 v101, v39
	v_mov_b32_e32 v102, v29
	v_mov_b32_e32 v103, v15
	v_mov_b32_e32 v15, v103
	v_mov_b32_e32 v64, v102
                                        ; kill: def $vgpr64 killed $vgpr64 def $vgpr64_vgpr65 killed $exec
	v_mov_b32_e32 v65, v15
	v_pk_mul_f32 v[68:69], v[64:65], v[66:67]
	s_nop 0
	v_mov_b32_e32 v15, v69
	v_mov_b32_e32 v29, v68
                                        ; kill: def $vgpr62 killed $vgpr62 def $vgpr62_vgpr63_vgpr64_vgpr65 killed $exec
	v_mov_b32_e32 v63, v37
	v_mov_b32_e32 v64, v35
	v_mov_b32_e32 v65, v31
	v_mov_b32_e32 v31, v65
	v_mov_b32_e32 v70, v64
                                        ; kill: def $vgpr70 killed $vgpr70 def $vgpr70_vgpr71 killed $exec
	v_mov_b32_e32 v71, v31
	v_mov_b32_e32 v31, v111
	v_mov_b32_e32 v35, v110
	v_mov_b32_e32 v37, v109
	v_mov_b32_e32 v96, v108
                                        ; kill: def $vgpr96 killed $vgpr96 def $vgpr96_vgpr97_vgpr98_vgpr99 killed $exec
	v_mov_b32_e32 v97, v37
	v_mov_b32_e32 v98, v35
	v_mov_b32_e32 v99, v31
	v_mov_b32_e32 v31, v99
	v_mov_b32_e32 v66, v98
                                        ; kill: def $vgpr66 killed $vgpr66 def $vgpr66_vgpr67 killed $exec
	v_mov_b32_e32 v67, v31
	v_pk_mul_f32 v[72:73], v[66:67], v[70:71]
	s_nop 0
	v_mov_b32_e32 v37, v73
	v_mov_b32_e32 v39, v72
	v_mov_b64_e32 v[70:71], v[104:105]
	v_mov_b64_e32 v[66:67], v[100:101]
	v_pk_mul_f32 v[70:71], v[66:67], v[70:71]
	s_nop 0
	v_mov_b32_e32 v31, v71
	v_mov_b32_e32 v35, v70
	v_mov_b64_e32 v[64:65], v[62:63]
	v_mov_b64_e32 v[62:63], v[96:97]
	v_pk_mul_f32 v[74:75], v[62:63], v[64:65]
	s_nop 0
	v_mov_b32_e32 v41, v75
	v_mov_b32_e32 v96, v74
                                        ; kill: def $vgpr96 killed $vgpr96 def $vgpr96_vgpr97_vgpr98_vgpr99_vgpr100_vgpr101_vgpr102_vgpr103 killed $exec
	v_mov_b32_e32 v97, v41
	v_mov_b32_e32 v98, v39
	v_mov_b32_e32 v99, v37
	v_mov_b32_e32 v100, v35
	v_mov_b32_e32 v101, v31
	v_mov_b32_e32 v102, v29
	v_mov_b32_e32 v103, v15
	scratch_store_dwordx4 off, v[96:99], off offset:100 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[100:103], off offset:116 ; 16-byte Folded Spill
	v_mul_f32_e64 v8, v3, v8
	v_mov_b32_e32 v31, v86
	v_mov_b32_e32 v66, v85
	v_mov_b32_e32 v15, v88
	v_mov_b32_e32 v64, v87
                                        ; kill: def $vgpr64 killed $vgpr64 def $vgpr64_vgpr65 killed $exec
	v_mov_b32_e32 v65, v15
	v_mov_b32_e32 v15, v95
	v_mov_b32_e32 v62, v94
                                        ; kill: def $vgpr62 killed $vgpr62 def $vgpr62_vgpr63 killed $exec
	v_mov_b32_e32 v63, v15
	v_pk_mul_f32 v[64:65], v[62:63], v[64:65]
	s_nop 0
	v_mov_b32_e32 v15, v65
	v_mov_b32_e32 v29, v64
                                        ; kill: def $vgpr66 killed $vgpr66 def $vgpr66_vgpr67 killed $exec
	v_mov_b32_e32 v67, v31
	v_mov_b64_e32 v[62:63], v[92:93]
	v_pk_mul_f32 v[66:67], v[62:63], v[66:67]
	s_nop 0
	v_mov_b32_e32 v31, v67
	v_mov_b32_e32 v92, v66
                                        ; kill: def $vgpr92 killed $vgpr92 def $vgpr92_vgpr93_vgpr94_vgpr95 killed $exec
	v_mov_b32_e32 v93, v31
	v_mov_b32_e32 v94, v29
	v_mov_b32_e32 v95, v15
	v_mul_f32_e64 v3, v3, v9
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v9, v96
	v_mov_b32_e32 v15, v97
.Ltmp24:
	.file	3 "/root/triton/python/triton/language" "standard.py"
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v9, v9, v15
.Ltmp25:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v15, v98
.Ltmp26:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v9, v9, v15
.Ltmp27:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v15, v99
	v_mov_b32_e32 v29, v100
.Ltmp28:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v15, v15, v29
.Ltmp29:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v29, v101
.Ltmp30:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v35, v15, v29
.Ltmp31:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v15, v102
	v_mov_b32_e32 v29, v103
.Ltmp32:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v15, v15, v29
	v_max_f32_e64 v31, v15, v8
.Ltmp33:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v15, v92
	v_mov_b32_e32 v29, v93
.Ltmp34:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v15, v15, v29
.Ltmp35:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v29, v94
.Ltmp36:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v15, v15, v29
.Ltmp37:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v29, v95
.Ltmp38:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v9, v9, v35
	v_max_f32_e64 v9, v9, v31
.Ltmp39:
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v31, v90
	v_mov_b32_e32 v62, v89
                                        ; kill: def $vgpr62 killed $vgpr62 def $vgpr62_vgpr63 killed $exec
	v_mov_b32_e32 v63, v31
	v_pk_mul_f32 v[62:63], v[60:61], v[62:63]
.Ltmp40:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	s_nop 0
	v_mov_b32_e32 v31, v62
	v_max_f32_e64 v29, v29, v31
	v_mov_b32_e32 v31, v63
	v_max_f32_e64 v29, v29, v31
	v_max_f32_e64 v15, v15, v29
	v_max_f32_e64 v15, v15, v3
	v_max_f32_e64 v15, v9, v15
.Ltmp41:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v9, v15
	s_nop 1
	v_permlane32_swap_b32_e64 v9, v15
.Ltmp42:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v15, v15, v15
	v_max_f32_e64 v9, v9, v9
	v_max_f32_e64 v15, v9, v15
.Ltmp43:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:746:25 ]
	v_max_f32_e64 v9, v59, v59
	v_max_f32_e64 v60, v9, v15
	v_mov_b32_e32 v9, v60
                                        ; implicit-def: $sgpr10_sgpr11
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	s_mov_b32 s10, s11
	v_writelane_b32 v118, s10, 38
	v_mov_b32_e32 v15, s10
	v_mov_b32_e32 v76, v60
	v_mov_b32_e32 v77, v15
	v_pk_add_f32 v[74:75], v[74:75], v[76:77] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[72:73], v[72:73], v[76:77] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[70:71], v[70:71], v[76:77] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69], v[68:69], v[76:77] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e64 v8, v8, v60
	v_pk_add_f32 v[66:67], v[66:67], v[76:77] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[64:65], v[64:65], v[76:77] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63], v[62:63], v[76:77] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e64 v3, v3, v60
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:746:25 ]
	v_mov_b32_e32 v15, v74
	v_exp_f32_e64 v57, v15
	v_mov_b32_e32 v15, v75
	v_exp_f32_e64 v53, v15
	v_mov_b32_e32 v15, v72
	v_exp_f32_e64 v49, v15
	v_mov_b32_e32 v15, v73
	v_exp_f32_e64 v45, v15
	v_mov_b32_e32 v15, v70
	v_exp_f32_e64 v41, v15
	v_mov_b32_e32 v15, v71
	v_exp_f32_e64 v37, v15
	v_mov_b32_e32 v15, v68
	v_exp_f32_e64 v31, v15
	v_mov_b32_e32 v15, v69
	v_exp_f32_e64 v15, v15
	v_exp_f32_e64 v55, v8
	v_mov_b32_e32 v8, v66
	v_exp_f32_e64 v51, v8
	v_mov_b32_e32 v8, v67
	v_exp_f32_e64 v47, v8
	v_mov_b32_e32 v8, v64
	v_exp_f32_e64 v43, v8
	v_mov_b32_e32 v8, v65
	v_exp_f32_e64 v39, v8
	v_mov_b32_e32 v8, v62
	v_exp_f32_e64 v35, v8
	v_mov_b32_e32 v8, v63
	v_exp_f32_e64 v29, v8
	v_exp_f32_e64 v3, v3
.Ltmp44:
	.loc	3 293 12 is_stmt 1              ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v62, v57
	v_mov_b32_e32 v63, v8
                                        ; kill: def $vgpr62 killed $vgpr62 killed $vgpr62_vgpr63 killed $exec
                                        ; kill: def $vgpr62 killed $vgpr62 def $vgpr62_vgpr63 killed $exec
	v_mov_b32_e32 v63, v53
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v64, v49
	v_mov_b32_e32 v65, v8
	v_mov_b32_e32 v76, v64
                                        ; kill: def $vgpr76 killed $vgpr76 def $vgpr76_vgpr77 killed $exec
	v_mov_b32_e32 v77, v45
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v64, v41
	v_mov_b32_e32 v65, v8
	v_mov_b32_e32 v68, v64
                                        ; kill: def $vgpr68 killed $vgpr68 def $vgpr68_vgpr69 killed $exec
	v_mov_b32_e32 v69, v37
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v64, v31
	v_mov_b32_e32 v65, v8
	v_mov_b32_e32 v74, v64
                                        ; kill: def $vgpr74 killed $vgpr74 def $vgpr74_vgpr75 killed $exec
	v_mov_b32_e32 v75, v15
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v64, v55
	v_mov_b32_e32 v65, v8
                                        ; kill: def $vgpr64 killed $vgpr64 killed $vgpr64_vgpr65 killed $exec
                                        ; kill: def $vgpr64 killed $vgpr64 def $vgpr64_vgpr65 killed $exec
	v_mov_b32_e32 v65, v51
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v66, v47
	v_mov_b32_e32 v67, v8
	v_mov_b32_e32 v72, v66
                                        ; kill: def $vgpr72 killed $vgpr72 def $vgpr72_vgpr73 killed $exec
	v_mov_b32_e32 v73, v43
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v66, v39
	v_mov_b32_e32 v67, v8
                                        ; kill: def $vgpr66 killed $vgpr66 killed $vgpr66_vgpr67 killed $exec
                                        ; kill: def $vgpr66 killed $vgpr66 def $vgpr66_vgpr67 killed $exec
	v_mov_b32_e32 v67, v35
	v_mov_b32_e32 v8, s10
	v_mov_b32_e32 v70, v29
	v_mov_b32_e32 v71, v8
                                        ; kill: def $vgpr70 killed $vgpr70 killed $vgpr70_vgpr71 killed $exec
                                        ; kill: def $vgpr70 killed $vgpr70 def $vgpr70_vgpr71 killed $exec
	v_mov_b32_e32 v71, v3
	v_pk_add_f32 v[62:63], v[62:63], v[76:77]
	v_pk_add_f32 v[68:69], v[68:69], v[74:75]
	v_pk_add_f32 v[64:65], v[64:65], v[72:73]
	v_pk_add_f32 v[66:67], v[66:67], v[70:71]
	v_pk_add_f32 v[62:63], v[62:63], v[68:69]
	v_pk_add_f32 v[64:65], v[64:65], v[66:67]
	s_nop 0
	v_pk_add_f32 v[62:63], v[62:63], v[64:65]
.Ltmp45:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	s_nop 0
	v_pk_add_f32 v[62:63], v[62:63], v[62:63] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp46:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	s_nop 0
	v_mov_b32_e32 v61, v62
	v_mov_b32_e32 v8, v61
	s_nop 1
	v_permlane32_swap_b32_e64 v8, v61
.Ltmp47:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_add_f32_e64 v8, v8, v61
.Ltmp48:
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:746:25 ]
	v_sub_f32_e64 v59, v59, v60
	.loc	1 241 17 is_stmt 0              ; mha.py:241:17 @[ mha.py:746:25 ]
	v_exp_f32_e64 v59, v59
	.loc	1 246 15 is_stmt 1              ; mha.py:246:15 @[ mha.py:746:25 ]
	ds_write_b32 v27, v59
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[26:27], v26
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_mul_f32_e64 v58, v58, v59
	v_add_f32_e64 v8, v8, v58
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v57, v57, s10
	ds_write_b16 v56, v57
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v55, v55, s10
	ds_write_b16 v54, v55
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v53, v53, s10
	ds_write_b16 v52, v53
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v51, v51, s10
	ds_write_b16 v50, v51
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v49, v49, s10
	ds_write_b16 v48, v49
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v47, v47, s10
	ds_write_b16 v46, v47
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v45, v45, s10
	ds_write_b16 v44, v45
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v43, v43, s10
	ds_write_b16 v42, v43
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v41, v41, s10
	ds_write_b16 v40, v41
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v39, v39, s10
	ds_write_b16 v38, v39
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v37, v37, s10
	ds_write_b16 v36, v37
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v35, v35, s10
	ds_write_b16 v34, v35
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v31, v31, s10
	ds_write_b16 v30, v31
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v29, v29, s10
	ds_write_b16 v28, v29
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v15, v15, s10
	ds_write_b16 v13, v15
                                        ; implicit-def: $sgpr10
	v_cvt_pk_bf16_f32 v3, v3, s10
	ds_write_b16 v1, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[0:1], v0
	ds_read_b64_tr_b16 v[2:3], v2
	ds_read_b64_tr_b16 v[12:13], v12
	ds_read_b64_tr_b16 v[14:15], v14
	.loc	1 259 19 is_stmt 0              ; mha.py:259:19 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v28, v3
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v1
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1_vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v1, v30
	v_mov_b32_e32 v2, v29
	v_mov_b32_e32 v3, v28
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v28, v15
	v_mov_b32_e32 v29, v14
	v_mov_b32_e32 v30, v13
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13_vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v13, v30
	v_mov_b32_e32 v14, v29
	v_mov_b32_e32 v15, v28
	v_mov_b32_e32 v28, v7
	v_mov_b32_e32 v29, v6
	v_mov_b32_e32 v30, v5
                                        ; kill: def $vgpr4 killed $vgpr4 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v30
	v_mov_b32_e32 v6, v29
	v_mov_b32_e32 v7, v28
	v_mov_b32_e32 v28, v26
	v_mov_b32_e32 v34, v28
	v_mov_b32_e32 v35, v28
	v_mov_b32_e32 v36, v28
	v_mov_b32_e32 v37, v28
	v_mov_b32_e32 v29, v37
	v_mov_b32_e32 v30, v36
                                        ; kill: def $vgpr34_vgpr35 killed $vgpr34_vgpr35 killed $vgpr34_vgpr35_vgpr36_vgpr37 killed $exec
	v_pk_mul_f32 v[34:35], v[32:33], v[34:35]
	s_nop 0
	v_mov_b32_e32 v32, v35
	v_mov_b32_e32 v28, v34
                                        ; kill: def $vgpr30 killed $vgpr30 def $vgpr30_vgpr31 killed $exec
	v_mov_b32_e32 v31, v29
	s_waitcnt vmcnt(15)
	v_pk_mul_f32 v[30:31], v[10:11], v[30:31]
	s_nop 0
	v_mov_b32_e32 v10, v31
	v_mov_b32_e32 v11, v30
                                        ; kill: def $vgpr28 killed $vgpr28 def $vgpr28_vgpr29_vgpr30_vgpr31 killed $exec
	v_mov_b32_e32 v29, v32
	v_mov_b32_e32 v30, v11
	v_mov_b32_e32 v31, v10
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[28:31], v[4:7], v[0:3], v[28:31]
	s_nop 7
	v_mov_b32_e32 v10, v29
	v_mov_b32_e32 v0, v28
	v_mov_b32_e32 v1, v31
	v_mov_b32_e32 v2, v30
	v_mov_b32_e32 v3, v27
	v_mov_b32_e32 v30, v3
	v_mov_b32_e32 v31, v3
	v_mov_b32_e32 v32, v3
	v_mov_b32_e32 v33, v3
	v_mov_b32_e32 v3, v33
	v_mov_b32_e32 v28, v32
	v_mov_b64_e32 v[26:27], v[30:31]
	v_pk_mul_f32 v[16:17], v[16:17], v[26:27]
	s_nop 0
	v_mov_b32_e32 v26, v17
                                        ; kill: def $vgpr16 killed $vgpr16 killed $vgpr16_vgpr17 killed $exec
                                        ; kill: def $vgpr28 killed $vgpr28 def $vgpr28_vgpr29 killed $exec
	v_mov_b32_e32 v29, v3
	v_pk_mul_f32 v[18:19], v[18:19], v[28:29]
	s_nop 0
	v_mov_b32_e32 v3, v19
	v_mov_b32_e32 v11, v18
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17_vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v17, v26
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v3
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[12:15], v[4:7], v[12:15], v[16:19]
	s_nop 7
	v_mov_b32_e32 v3, v13
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v5, v15
	v_mov_b32_e32 v6, v14
	; wait_asyncmark(3)
.Ltmp49:
	.loc	1 36 18 is_stmt 1               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp50:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:746:25 ]
	s_add_i32 s6, s6, s9
	s_cmp_ge_i32 s6, s8
	s_cselect_b64 s[8:9], -1, 0
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v10
	s_mov_b64 s[10:11], -1
	s_xor_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
                                        ; kill: def $vgpr24_vgpr25 killed $vgpr24_vgpr25 killed $exec
                                        ; kill: def $vgpr22_vgpr23 killed $vgpr22_vgpr23 killed $exec
                                        ; kill: def $vgpr20_vgpr21 killed $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v19, v9
	v_mov_b32_e32 v18, v8
	v_mov_b64_e32 v[16:17], v[4:5]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[12:13], v[0:1]
	v_mov_b64_e32 v[10:11], v[2:3]
	s_mov_b32 s8, s0
	v_writelane_b32 v118, s8, 23
	s_mov_b32 s8, s5
	v_writelane_b32 v118, s8, 24
	s_mov_b32 s8, s1
	v_writelane_b32 v118, s8, 25
	s_mov_b32 s8, s4
	v_writelane_b32 v118, s8, 26
	s_mov_b32 s8, s2
	v_writelane_b32 v118, s8, 27
	s_mov_b32 s8, s3
	v_writelane_b32 v118, s8, 28
	v_writelane_b32 v118, s7, 29
	v_accvgpr_write_b32 a113, v25           ;  Reload Reuse
	v_accvgpr_write_b32 a114, v24           ;  Reload Reuse
	v_accvgpr_write_b32 a115, v23           ;  Reload Reuse
	v_accvgpr_write_b32 a116, v22           ;  Reload Reuse
	v_accvgpr_write_b32 a117, v21           ;  Reload Reuse
	v_accvgpr_write_b32 a118, v20           ;  Reload Reuse
	v_accvgpr_write_b32 a119, v19           ;  Reload Reuse
	v_accvgpr_write_b32 a120, v18           ;  Reload Reuse
	v_writelane_b32 v118, s6, 30
	v_accvgpr_write_b32 a121, v17           ;  Reload Reuse
	v_accvgpr_write_b32 a122, v16           ;  Reload Reuse
	v_accvgpr_write_b32 a123, v15           ;  Reload Reuse
	v_accvgpr_write_b32 a124, v14           ;  Reload Reuse
	v_accvgpr_write_b32 a125, v13           ;  Reload Reuse
	v_accvgpr_write_b32 a126, v12           ;  Reload Reuse
	v_accvgpr_write_b32 a127, v11           ;  Reload Reuse
	scratch_store_dword off, v10, off offset:4 ; 4-byte Folded Spill
	v_writelane_b32 v118, s5, 39
	v_writelane_b32 v118, s4, 40
	v_writelane_b32 v118, s3, 41
	v_writelane_b32 v118, s2, 42
	v_writelane_b32 v118, s1, 43
	v_writelane_b32 v118, s0, 44
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	scratch_store_dword off, v9, off offset:96 ; 4-byte Folded Spill
	scratch_store_dword off, v8, off offset:92 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:84 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:76 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:68 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], off offset:60 ; 8-byte Folded Spill
	s_cbranch_vccnz .LBB0_6
; %bb.7:                                ; %Flow6
	.loc	1 0 5 is_stmt 0                 ; mha.py:0:5
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s4, v118, 39
	v_readlane_b32 s2, v118, 40
	v_readlane_b32 s0, v118, 41
	v_readlane_b32 s1, v118, 42
	v_readlane_b32 s3, v118, 43
	v_readlane_b32 s5, v118, 44
	v_accvgpr_read_b32 v11, a65             ;  Reload Reuse
	v_accvgpr_read_b32 v10, a66             ;  Reload Reuse
	scratch_load_dwordx2 v[8:9], off, off offset:60 ; 8-byte Folded Reload
	scratch_load_dword v1, off, off offset:96 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:92 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:84 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:76 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:68 ; 8-byte Folded Reload
                                        ; kill: def $vgpr8_vgpr9 killed $vgpr8_vgpr9 killed $exec
                                        ; kill: def $vgpr6_vgpr7 killed $vgpr6_vgpr7 killed $exec
                                        ; kill: def $vgpr4_vgpr5 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr2_vgpr3 killed $vgpr2_vgpr3 killed $exec
                                        ; kill: def $vgpr1 killed $vgpr1 killed $exec
                                        ; kill: def $vgpr0 killed $vgpr0 killed $exec
	v_accvgpr_write_b32 a52, v11            ;  Reload Reuse
	v_accvgpr_write_b32 a53, v10            ;  Reload Reuse
	s_waitcnt vmcnt(5)
	v_accvgpr_write_b32 a54, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a55, v8             ;  Reload Reuse
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a56, v7             ;  Reload Reuse
	v_accvgpr_write_b32 a57, v6             ;  Reload Reuse
	v_accvgpr_write_b32 a58, v5             ;  Reload Reuse
	v_accvgpr_write_b32 a59, v4             ;  Reload Reuse
	v_accvgpr_write_b32 a60, v3             ;  Reload Reuse
	v_accvgpr_write_b32 a61, v2             ;  Reload Reuse
	v_writelane_b32 v118, s5, 15
	v_writelane_b32 v118, s4, 16
	v_writelane_b32 v118, s3, 17
	v_writelane_b32 v118, s2, 18
	v_writelane_b32 v118, s1, 19
	v_writelane_b32 v118, s0, 20
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_accvgpr_write_b32 a62, v1             ;  Reload Reuse
	v_accvgpr_write_b32 a63, v0             ;  Reload Reuse
	s_branch .LBB0_5
.LBB0_8:                                ; %Flow10
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v117, 50
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_accvgpr_read_b32 v1, a26              ;  Reload Reuse
	v_accvgpr_read_b32 v0, a27              ;  Reload Reuse
	v_accvgpr_read_b32 v3, a28              ;  Reload Reuse
	v_accvgpr_read_b32 v2, a29              ;  Reload Reuse
	v_accvgpr_read_b32 v5, a30              ;  Reload Reuse
	v_accvgpr_read_b32 v4, a31              ;  Reload Reuse
	v_accvgpr_read_b32 v7, a32              ;  Reload Reuse
	v_accvgpr_read_b32 v6, a33              ;  Reload Reuse
	v_accvgpr_read_b32 v8, a35              ;  Reload Reuse
	v_accvgpr_read_b32 v9, a36              ;  Reload Reuse
	scratch_store_dword off, v9, off offset:328 ; 4-byte Folded Spill
	scratch_store_dword off, v8, off offset:324 ; 4-byte Folded Spill
	v_writelane_b32 v118, s0, 45
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	scratch_store_dwordx2 off, v[6:7], off offset:316 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:308 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:300 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], off offset:292 ; 8-byte Folded Spill
	s_branch .LBB0_18
.LBB0_9:                                ; %._crit_edge
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:746:25 ]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v117, 57
	v_readlane_b32 s1, v117, 58
	v_readlane_b32 s2, v118, 35
	v_readlane_b32 s3, v117, 51
	scratch_load_dwordx2 v[0:1], off, off offset:12 ; 8-byte Folded Reload
	v_accvgpr_read_b32 v3, a48              ;  Reload Reuse
	v_accvgpr_read_b32 v2, a49              ;  Reload Reuse
	s_mov_b32 s4, 31
	s_or_b32 s3, s3, s4
	s_mov_b32 s4, 63
	s_cmp_gt_i32 s3, s4
	s_cselect_b64 s[4:5], -1, 0
.Ltmp51:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_writelane_b32 v118, s4, 46
	s_nop 1
	v_writelane_b32 v118, s5, 47
	s_mov_b32 s3, 0
	v_cmp_eq_u32_e64 s[4:5], v2, s3
	s_mov_b32 s6, 0x110
	v_mov_b32_e32 v2, s6
	v_mov_b32_e32 v4, s3
	v_cndmask_b32_e64 v2, v2, v4, s[4:5]
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v4, v0
	v_bitop3_b32 v2, v2, v3, v4 bitop3:0xde
.Ltmp52:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	scratch_store_dword off, v2, off offset:452 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, v1
	scratch_store_dword off, v0, off offset:448 ; 4-byte Folded Spill
	v_add_u32_e64 v1, s2, v0
	ds_read_b64_tr_b16 v[2:3], v1
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[2:3], off offset:440 ; 8-byte Folded Spill
	s_mov_b32 s3, 0x200
	v_or_b32_e64 v0, v0, s3
	v_mov_b32_e32 v1, v0
	scratch_store_dword off, v1, off offset:436 ; 4-byte Folded Spill
	v_add_u32_e64 v0, s2, v0
	ds_read_b64_tr_b16 v[0:1], v0
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], off offset:428 ; 8-byte Folded Spill
	s_mov_b32 s2, 0
	v_writelane_b32 v118, s2, 48
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_mov_b32_e32 v0, s2
	v_mov_b32_e32 v2, s2
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v8, s2
	v_mov_b32_e32 v7, s2
	v_mov_b32_e32 v6, s2
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3_vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v4, v7
	v_mov_b32_e32 v5, v6
	v_mov_b32_e32 v6, s2
	v_mov_b32_e32 v20, s2
	v_mov_b32_e32 v19, s2
	v_mov_b32_e32 v18, s2
	v_mov_b32_e32 v17, s2
	v_mov_b32_e32 v16, s2
	v_mov_b32_e32 v15, s2
	v_mov_b32_e32 v14, s2
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v7, v20
	v_mov_b32_e32 v8, v19
	v_mov_b32_e32 v9, v18
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v11, v16
	v_mov_b32_e32 v12, v15
	v_mov_b32_e32 v13, v14
	scratch_store_dwordx4 off, v[6:9], off offset:396 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[10:13], off offset:412 ; 16-byte Folded Spill
	v_mov_b32_e32 v14, s2
.Ltmp53:
	.loc	1 182 14 is_stmt 1              ; mha.py:182:14 @[ mha.py:746:25 ]
	s_mov_b64 s[2:3], -1
	s_xor_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 vcc, exec, s[0:1]
.Ltmp54:
	.loc	1 746 25                        ; mha.py:746:25
	v_mov_b32_e32 v15, v14
                                        ; kill: def $vgpr14 killed $vgpr14 killed $exec
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	scratch_store_dword off, v15, off offset:392 ; 4-byte Folded Spill
	scratch_store_dword off, v14, off offset:388 ; 4-byte Folded Spill
	scratch_store_dwordx4 off, v[6:9], off offset:356 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[10:13], off offset:372 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[2:5], off offset:340 ; 16-byte Folded Spill
.Ltmp55:
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	scratch_store_dwordx2 off, v[0:1], off offset:332 ; 8-byte Folded Spill
	s_cbranch_vccnz .LBB0_11
; %bb.10:
.Ltmp56:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 31
	v_readlane_b32 s1, v118, 33
	v_accvgpr_read_b32 v8, a25              ;  Reload Reuse
	v_accvgpr_read_b32 v9, a24              ;  Reload Reuse
	v_accvgpr_read_b32 v10, a23             ;  Reload Reuse
	v_accvgpr_read_b32 v4, a22              ;  Reload Reuse
	v_accvgpr_read_b32 v5, a21              ;  Reload Reuse
	v_accvgpr_read_b32 v6, a20              ;  Reload Reuse
	v_accvgpr_read_b32 v7, a19              ;  Reload Reuse
	v_accvgpr_read_b32 v34, a18             ;  Reload Reuse
	scratch_load_dword v0, off, off offset:452 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e64 v1, s1, v0
	ds_read_b128 v[12:15], v1
.Ltmp57:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_add_u32_e64 v0, s0, v0
	ds_read_b128 v[0:3], v0
.Ltmp58:
                                        ; kill: def $vgpr34 killed $vgpr34 def $vgpr34_vgpr35_vgpr36_vgpr37 killed $exec
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v35, v7
	v_mov_b32_e32 v36, v6
	v_mov_b32_e32 v37, v5
	s_mov_b32 s0, 0
	v_writelane_b32 v118, s0, 49
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_mov_b32_e32 v18, s0
	v_mov_b32_e32 v46, s0
	v_mov_b32_e32 v45, s0
	v_mov_b32_e32 v44, s0
	v_mov_b32_e32 v43, s0
	v_mov_b32_e32 v42, s0
	v_mov_b32_e32 v41, s0
	v_mov_b32_e32 v40, s0
	v_mov_b32_e32 v39, s0
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v17, s0
	v_mov_b32_e32 v16, s0
	v_mov_b32_e32 v11, s0
	v_mov_b32_e32 v7, s0
	v_mov_b32_e32 v6, s0
	v_mov_b32_e32 v5, s0
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31_vgpr32_vgpr33 killed $exec
	v_mov_b32_e32 v19, v46
	v_mov_b32_e32 v20, v45
	v_mov_b32_e32 v21, v44
	v_mov_b32_e32 v22, v43
	v_mov_b32_e32 v23, v42
	v_mov_b32_e32 v24, v41
	v_mov_b32_e32 v25, v40
	v_mov_b32_e32 v26, v39
	v_mov_b32_e32 v27, v38
	v_mov_b32_e32 v28, v17
	v_mov_b32_e32 v29, v16
	v_mov_b32_e32 v30, v11
	v_mov_b32_e32 v31, v7
	v_mov_b32_e32 v32, v6
	v_mov_b32_e32 v33, v5
	scratch_store_dwordx4 off, v[18:21], off offset:552 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[22:25], off offset:568 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[26:29], off offset:584 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[30:33], off offset:600 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_f32_32x32x16_bf16 v[18:33], v[12:15], v[34:37], v[18:33]
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v8
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_32x32x16_bf16 v[18:33], v[0:3], v[4:7], v[18:33]
	s_nop 11
	scratch_store_dwordx4 off, v[18:21], off offset:488 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[22:25], off offset:504 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[26:29], off offset:520 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[30:33], off offset:536 ; 16-byte Folded Spill
	v_mov_b32_e32 v0, v25
	v_mov_b32_e32 v1, v24
	v_mov_b32_e32 v2, v23
	v_mov_b32_e32 v3, v22
	v_mov_b32_e32 v4, v21
	v_mov_b32_e32 v5, v20
	v_mov_b32_e32 v16, v19
	v_mov_b32_e32 v6, v18
	v_mov_b32_e32 v15, v26
	v_mov_b32_e32 v14, v33
.Ltmp59:
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13 killed $exec
	.loc	1 746 25                        ; mha.py:746:25
	v_mov_b32_e32 v7, v16
	v_mov_b32_e32 v8, v5
	v_mov_b32_e32 v9, v4
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v1
	v_mov_b32_e32 v13, v0
	scratch_store_dwordx4 off, v[6:9], off offset:456 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[10:13], off offset:472 ; 16-byte Folded Spill
	v_mov_b32_e32 v0, v30
	v_mov_b32_e32 v1, v29
	v_mov_b32_e32 v16, v28
	v_mov_b32_e32 v2, v27
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3_vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v3, v16
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v16, v32
	v_mov_b32_e32 v0, v31
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v16
                                        ; kill: def $vgpr15 killed $vgpr15 killed $exec
                                        ; kill: def $vgpr14 killed $vgpr14 killed $exec
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	scratch_store_dword off, v15, off offset:392 ; 4-byte Folded Spill
	scratch_store_dword off, v14, off offset:388 ; 4-byte Folded Spill
	scratch_store_dwordx4 off, v[6:9], off offset:356 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[10:13], off offset:372 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[2:5], off offset:340 ; 16-byte Folded Spill
.Ltmp60:
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	scratch_store_dwordx2 off, v[0:1], off offset:332 ; 8-byte Folded Spill
.Ltmp61:
.LBB0_11:
	.loc	1 746 25                        ; mha.py:746:25
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v117, 57
	v_readlane_b32 s1, v117, 58
	v_readlane_b32 s4, v117, 19
	v_readlane_b32 s2, v117, 26
	v_readlane_b32 s8, v117, 60
	v_readlane_b32 s3, v117, 18
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	scratch_load_dwordx2 v[0:1], off, off offset:20 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:28 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:36 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:44 ; 8-byte Folded Reload
	scratch_load_dword v8, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v9, off, off offset:52 ; 4-byte Folded Reload
	v_accvgpr_read_b32 v12, a3              ;  Reload Reuse
	v_accvgpr_read_b32 v10, a1              ;  Reload Reuse
	v_accvgpr_read_b32 v11, a4              ;  Reload Reuse
	v_accvgpr_read_b32 v30, a2              ;  Reload Reuse
	v_accvgpr_read_b32 v13, a38             ;  Reload Reuse
	scratch_load_dwordx2 v[26:27], off, off offset:332 ; 8-byte Folded Reload
	scratch_load_dword v14, off, off offset:392 ; 4-byte Folded Reload
	scratch_load_dword v15, off, off offset:388 ; 4-byte Folded Reload
	scratch_load_dwordx4 v[56:59], off, off offset:356 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[60:63], off, off offset:372 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[32:35], off, off offset:340 ; 16-byte Folded Reload
	s_waitcnt vmcnt(2)
	scratch_store_dwordx4 off, v[56:59], off offset:896 ; 16-byte Folded Spill
	s_waitcnt vmcnt(2)
	scratch_store_dwordx4 off, v[60:63], off offset:912 ; 16-byte Folded Spill
.Ltmp62:
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v42, v13
	v_mov_b32_e32 v43, v13
	v_mov_b32_e32 v44, v13
	v_mov_b32_e32 v45, v13
	v_mov_b32_e32 v46, v13
	v_mov_b32_e32 v47, v13
	v_mov_b32_e32 v48, v13
	v_mov_b32_e32 v49, v13
	scratch_store_dwordx4 off, v[42:45], off offset:864 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[46:49], off offset:880 ; 16-byte Folded Spill
	v_mov_b32_e32 v18, v45
	v_mov_b32_e32 v19, v44
	v_mov_b32_e32 v22, v43
	v_mov_b32_e32 v40, v42
	v_mov_b32_e32 v16, v49
	v_mov_b32_e32 v17, v48
	v_mov_b32_e32 v23, v47
	v_mov_b32_e32 v48, v46
	v_mov_b32_e32 v20, v63
	v_mov_b32_e32 v21, v62
	v_mov_b32_e32 v24, v61
	v_mov_b32_e32 v52, v60
                                        ; kill: def $vgpr52 killed $vgpr52 def $vgpr52_vgpr53_vgpr54_vgpr55 killed $exec
	v_mov_b32_e32 v53, v24
	v_mov_b32_e32 v54, v21
	v_mov_b32_e32 v55, v20
	v_mov_b32_e32 v24, v55
	v_mov_b32_e32 v20, v54
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v24
                                        ; kill: def $vgpr48 killed $vgpr48 def $vgpr48_vgpr49_vgpr50_vgpr51 killed $exec
	v_mov_b32_e32 v49, v23
	v_mov_b32_e32 v50, v17
	v_mov_b32_e32 v51, v16
	v_mov_b32_e32 v23, v51
	v_mov_b32_e32 v16, v50
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v23
	v_pk_mul_f32 v[16:17], v[16:17], v[20:21]
	s_nop 0
	v_mov_b32_e32 v24, v17
	v_mov_b32_e32 v25, v16
	v_mov_b32_e32 v20, v59
	v_mov_b32_e32 v21, v58
	v_mov_b32_e32 v23, v57
	v_mov_b32_e32 v44, v56
                                        ; kill: def $vgpr44 killed $vgpr44 def $vgpr44_vgpr45_vgpr46_vgpr47 killed $exec
	v_mov_b32_e32 v45, v23
	v_mov_b32_e32 v46, v21
	v_mov_b32_e32 v47, v20
	v_mov_b32_e32 v23, v47
	v_mov_b32_e32 v20, v46
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v23
                                        ; kill: def $vgpr40 killed $vgpr40 def $vgpr40_vgpr41_vgpr42_vgpr43 killed $exec
	v_mov_b32_e32 v41, v22
	v_mov_b32_e32 v42, v19
	v_mov_b32_e32 v43, v18
	v_mov_b32_e32 v22, v43
	v_mov_b32_e32 v18, v42
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v22
	v_pk_mul_f32 v[20:21], v[18:19], v[20:21]
	s_nop 0
	v_mov_b32_e32 v31, v21
	v_mov_b32_e32 v36, v20
	v_mov_b64_e32 v[22:23], v[52:53]
	v_mov_b64_e32 v[18:19], v[48:49]
	v_pk_mul_f32 v[18:19], v[18:19], v[22:23]
	s_nop 0
	v_mov_b32_e32 v28, v19
	v_mov_b32_e32 v29, v18
	v_mov_b64_e32 v[38:39], v[44:45]
	v_mov_b64_e32 v[22:23], v[40:41]
	v_pk_mul_f32 v[22:23], v[22:23], v[38:39]
	s_nop 0
	v_mov_b32_e32 v37, v23
	v_mov_b32_e32 v40, v22
                                        ; kill: def $vgpr40 killed $vgpr40 def $vgpr40_vgpr41_vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47 killed $exec
	v_mov_b32_e32 v41, v37
	v_mov_b32_e32 v42, v36
	v_mov_b32_e32 v43, v31
	v_mov_b32_e32 v44, v29
	v_mov_b32_e32 v45, v28
	v_mov_b32_e32 v46, v25
	v_mov_b32_e32 v47, v24
	scratch_store_dwordx4 off, v[40:43], off offset:832 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[44:47], off offset:848 ; 16-byte Folded Spill
	v_mul_f32_e64 v14, v13, v14
	v_mov_b32_e32 v36, v13
	v_mov_b32_e32 v37, v13
	v_mov_b32_e32 v38, v13
	v_mov_b32_e32 v39, v13
	scratch_store_dwordx4 off, v[36:39], off offset:816 ; 16-byte Folded Spill
	v_mov_b32_e32 v31, v39
	v_mov_b32_e32 v24, v38
	s_waitcnt vmcnt(7)
	v_mov_b32_e32 v25, v35
	v_mov_b32_e32 v28, v34
                                        ; kill: def $vgpr28 killed $vgpr28 def $vgpr28_vgpr29 killed $exec
	v_mov_b32_e32 v29, v25
                                        ; kill: def $vgpr24 killed $vgpr24 def $vgpr24_vgpr25 killed $exec
	v_mov_b32_e32 v25, v31
	v_pk_mul_f32 v[28:29], v[24:25], v[28:29]
	s_nop 0
	v_mov_b32_e32 v24, v29
	v_mov_b32_e32 v25, v28
	v_mov_b64_e32 v[34:35], v[32:33]
	v_mov_b64_e32 v[32:33], v[36:37]
	v_pk_mul_f32 v[32:33], v[32:33], v[34:35]
	s_nop 0
	v_mov_b32_e32 v31, v33
	v_mov_b32_e32 v36, v32
                                        ; kill: def $vgpr36 killed $vgpr36 def $vgpr36_vgpr37_vgpr38_vgpr39 killed $exec
	v_mov_b32_e32 v37, v31
	v_mov_b32_e32 v38, v25
	v_mov_b32_e32 v39, v24
                                        ; implicit-def: $sgpr6_sgpr7
	s_mov_b32 s5, s7
	v_writelane_b32 v118, s5, 50
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v24, v13
	v_mov_b32_e32 v25, v31
	v_mov_b32_e32 v31, v24
	v_mov_b32_e32 v34, v31
	v_mov_b32_e32 v35, v31
.Ltmp63:
	.loc	1 746 25                        ; mha.py:746:25
	scratch_store_dwordx2 off, v[34:35], off offset:808 ; 8-byte Folded Spill
.Ltmp64:
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_pk_mul_f32 v[24:25], v[24:25], v[26:27] op_sel_hi:[0,1]
	v_mul_f32_e64 v13, v13, v15
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v15, v40
	v_mov_b32_e32 v26, v41
.Ltmp65:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v15, v15, v26
.Ltmp66:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v26, v42
.Ltmp67:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v15, v15, v26
.Ltmp68:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v26, v43
	v_mov_b32_e32 v27, v44
.Ltmp69:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v26, v26, v27
.Ltmp70:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v27, v45
.Ltmp71:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v34, v26, v27
.Ltmp72:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v26, v46
	v_mov_b32_e32 v27, v47
.Ltmp73:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v26, v26, v27
	v_max_f32_e64 v31, v26, v14
.Ltmp74:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v26, v36
	v_mov_b32_e32 v27, v37
.Ltmp75:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v26, v26, v27
.Ltmp76:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v27, v38
.Ltmp77:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v26, v26, v27
.Ltmp78:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v27, v39
.Ltmp79:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_mov_b32_e32 v35, v24
	v_max_f32_e64 v27, v27, v35
	v_mov_b32_e32 v35, v25
	v_max_f32_e64 v27, v27, v35
	v_max_f32_e64 v15, v15, v34
	v_max_f32_e64 v15, v15, v31
	v_max_f32_e64 v26, v26, v27
	v_max_f32_e64 v26, v26, v13
	v_max_f32_e64 v26, v15, v26
.Ltmp80:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v15, v26
	s_nop 1
	v_permlane32_swap_b32_e64 v15, v26
.Ltmp81:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v26, v26, v26
	v_max_f32_e64 v15, v15, v15
	v_max_f32_e64 v26, v15, v26
.Ltmp82:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:746:25 ]
	v_max_f32_e64 v15, v9, v9
	v_max_f32_e64 v15, v15, v26
	v_mov_b32_e32 v26, v15
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	scratch_store_dword off, v26, off offset:804 ; 4-byte Folded Spill
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v26, v15
	v_mov_b32_e32 v27, v31
	v_pk_add_f32 v[22:23], v[22:23], v[26:27] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21], v[20:21], v[26:27] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[18:19], v[18:19], v[26:27] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[38:39], v[16:17], v[26:27] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e64 v16, v14, v15
	v_pk_add_f32 v[36:37], v[32:33], v[26:27] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[34:35], v[28:29], v[26:27] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33], v[24:25], v[26:27] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e64 v13, v13, v15
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:746:25 ]
	v_mov_b32_e32 v14, v22
	v_exp_f32_e64 v29, v14
	v_mov_b32_e32 v14, v23
	v_exp_f32_e64 v27, v14
	v_mov_b32_e32 v14, v20
	v_exp_f32_e64 v25, v14
	v_mov_b32_e32 v14, v21
	v_exp_f32_e64 v23, v14
	v_mov_b32_e32 v14, v18
	v_exp_f32_e64 v21, v14
	v_mov_b32_e32 v14, v19
	v_exp_f32_e64 v19, v14
	v_mov_b32_e32 v14, v38
	v_exp_f32_e64 v17, v14
	v_mov_b32_e32 v14, v39
	v_exp_f32_e64 v14, v14
	v_exp_f32_e64 v28, v16
	v_mov_b32_e32 v16, v36
	v_exp_f32_e64 v26, v16
	v_mov_b32_e32 v16, v37
	v_exp_f32_e64 v24, v16
	v_mov_b32_e32 v16, v34
	v_exp_f32_e64 v22, v16
	v_mov_b32_e32 v16, v35
	v_exp_f32_e64 v20, v16
	v_mov_b32_e32 v16, v32
	v_exp_f32_e64 v18, v16
	v_mov_b32_e32 v16, v33
	v_exp_f32_e64 v16, v16
	v_exp_f32_e64 v13, v13
.Ltmp83:
	.loc	3 293 12 is_stmt 1              ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v32, v29
	v_mov_b32_e32 v33, v31
                                        ; kill: def $vgpr32 killed $vgpr32 killed $vgpr32_vgpr33 killed $exec
                                        ; kill: def $vgpr32 killed $vgpr32 def $vgpr32_vgpr33 killed $exec
	v_mov_b32_e32 v33, v27
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v34, v25
	v_mov_b32_e32 v35, v31
	v_mov_b32_e32 v46, v34
                                        ; kill: def $vgpr46 killed $vgpr46 def $vgpr46_vgpr47 killed $exec
	v_mov_b32_e32 v47, v23
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v34, v21
	v_mov_b32_e32 v35, v31
	v_mov_b32_e32 v38, v34
                                        ; kill: def $vgpr38 killed $vgpr38 def $vgpr38_vgpr39 killed $exec
	v_mov_b32_e32 v39, v19
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v34, v17
	v_mov_b32_e32 v35, v31
	v_mov_b32_e32 v44, v34
                                        ; kill: def $vgpr44 killed $vgpr44 def $vgpr44_vgpr45 killed $exec
	v_mov_b32_e32 v45, v14
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v34, v28
	v_mov_b32_e32 v35, v31
                                        ; kill: def $vgpr34 killed $vgpr34 killed $vgpr34_vgpr35 killed $exec
                                        ; kill: def $vgpr34 killed $vgpr34 def $vgpr34_vgpr35 killed $exec
	v_mov_b32_e32 v35, v26
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v37, v31
	v_mov_b32_e32 v42, v36
                                        ; kill: def $vgpr42 killed $vgpr42 def $vgpr42_vgpr43 killed $exec
	v_mov_b32_e32 v43, v22
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v36, v20
	v_mov_b32_e32 v37, v31
                                        ; kill: def $vgpr36 killed $vgpr36 killed $vgpr36_vgpr37 killed $exec
                                        ; kill: def $vgpr36 killed $vgpr36 def $vgpr36_vgpr37 killed $exec
	v_mov_b32_e32 v37, v18
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v40, v16
	v_mov_b32_e32 v41, v31
                                        ; kill: def $vgpr40 killed $vgpr40 killed $vgpr40_vgpr41 killed $exec
                                        ; kill: def $vgpr40 killed $vgpr40 def $vgpr40_vgpr41 killed $exec
	v_mov_b32_e32 v41, v13
	v_pk_add_f32 v[32:33], v[32:33], v[46:47]
	v_pk_add_f32 v[38:39], v[38:39], v[44:45]
	v_pk_add_f32 v[34:35], v[34:35], v[42:43]
	v_pk_add_f32 v[36:37], v[36:37], v[40:41]
	v_pk_add_f32 v[32:33], v[32:33], v[38:39]
	v_pk_add_f32 v[34:35], v[34:35], v[36:37]
	s_nop 0
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
.Ltmp84:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	s_nop 0
	v_pk_add_f32 v[32:33], v[32:33], v[32:33] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp85:
                                        ; kill: def $vgpr32 killed $vgpr32 killed $vgpr32_vgpr33 killed $exec
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	s_nop 0
	v_mov_b32_e32 v31, v32
	s_nop 1
	v_permlane32_swap_b32_e64 v31, v32
	scratch_store_dword off, v32, off offset:800 ; 4-byte Folded Spill
.Ltmp86:
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:746:25 ]
	scratch_store_dword off, v31, off offset:796 ; 4-byte Folded Spill
	v_sub_f32_e64 v15, v9, v15
	.loc	1 241 17 is_stmt 0              ; mha.py:241:17 @[ mha.py:746:25 ]
	v_exp_f32_e64 v32, v15
	s_nop 0
	v_mov_b32_e32 v15, v32
	.loc	1 246 15 is_stmt 1              ; mha.py:246:15 @[ mha.py:746:25 ]
	scratch_store_dword off, v15, off offset:792 ; 4-byte Folded Spill
	s_mov_b32 s5, 31
	v_and_b32_e64 v15, v10, s5
	s_mov_b32 s5, 1
	s_lshl_b32 s3, s3, s5
	s_mov_b32 s6, 4
	s_and_b32 s7, s3, s6
	s_mov_b32 s3, 0
	v_writelane_b32 v118, s3, 51
	s_mov_b32 s6, 3
	v_mov_b32_e32 v31, s3
	v_lshl_add_u32 v33, v15, s6, v31
	s_mov_b32 s9, 2
	v_mov_b32_e32 v31, s9
	v_lshl_add_u32 v31, s8, v31, v33
	v_add_u32_e64 v31, v31, s7
	v_mov_b32_e32 v33, v31
	scratch_store_dword off, v33, off offset:788 ; 4-byte Folded Spill
	ds_write_b32 v31, v32
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v31, s3
	v_lshl_add_u32 v31, v30, s6, v31
	s_mov_b32 s6, 7
	v_mov_b32_e32 v30, s6
	v_lshl_add_u32 v30, s4, v30, v31
	v_mov_b32_e32 v31, v30
	scratch_store_dword off, v31, off offset:784 ; 4-byte Folded Spill
	ds_read_b64 v[30:31], v30
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[30:31], off offset:776 ; 8-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e64 v15, s5, v15
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s[6:7], v11, s8
	s_mov_b32 s9, 0x420
	v_mov_b32_e32 v11, s9
	v_mov_b32_e32 v30, s8
	v_cndmask_b32_e64 v11, v11, v30, s[6:7]
	v_bitop3_b32 v11, s2, v11, v15 bitop3:0x36
	scratch_store_dword off, v11, off offset:772 ; 4-byte Folded Spill
	v_add_u32_e64 v15, s3, v11
	v_mov_b32_e32 v30, v15
	scratch_store_dword off, v30, off offset:768 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v29, v29, s2
	ds_write_b16 v15, v29
	s_mov_b32 s2, 0x1000
	v_writelane_b32 v118, s2, 52
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_add_u32_e64 v29, v15, s2
	scratch_store_dword off, v29, off offset:764 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v28, v28, s6
	ds_write_b16 v15, v28 offset:4096
	s_mov_b32 s6, 0x108
	v_xor_b32_e64 v15, v11, s6
	v_add_u32_e64 v15, s3, v15
	v_mov_b32_e32 v28, v15
	scratch_store_dword off, v28, off offset:760 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v27, v27, s6
	ds_write_b16 v15, v27
	v_add_u32_e64 v27, v15, s2
	scratch_store_dword off, v27, off offset:756 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v26, v26, s6
	ds_write_b16 v15, v26 offset:4096
	s_mov_b32 s6, 0x210
	v_xor_b32_e64 v15, v11, s6
	v_add_u32_e64 v15, s3, v15
	v_mov_b32_e32 v26, v15
	scratch_store_dword off, v26, off offset:752 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v25, v25, s6
	ds_write_b16 v15, v25
	v_add_u32_e64 v25, v15, s2
	scratch_store_dword off, v25, off offset:748 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v24, v24, s6
	ds_write_b16 v15, v24 offset:4096
	s_mov_b32 s6, 0x318
	v_xor_b32_e64 v15, v11, s6
	v_add_u32_e64 v15, s3, v15
	v_mov_b32_e32 v24, v15
	scratch_store_dword off, v24, off offset:744 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v23, v23, s6
	ds_write_b16 v15, v23
	v_add_u32_e64 v23, v15, s2
	scratch_store_dword off, v23, off offset:740 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v22, v22, s6
	ds_write_b16 v15, v22 offset:4096
	s_mov_b32 s6, 0x840
	v_xor_b32_e64 v15, v11, s6
	v_add_u32_e64 v15, s3, v15
	v_mov_b32_e32 v22, v15
	scratch_store_dword off, v22, off offset:736 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v21, v21, s6
	ds_write_b16 v15, v21
	v_add_u32_e64 v21, v15, s2
	scratch_store_dword off, v21, off offset:732 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v20, v20, s6
	ds_write_b16 v15, v20 offset:4096
	s_mov_b32 s6, 0x948
	v_xor_b32_e64 v15, v11, s6
	v_add_u32_e64 v15, s3, v15
	v_mov_b32_e32 v20, v15
	scratch_store_dword off, v20, off offset:728 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v19, v19, s6
	ds_write_b16 v15, v19
	v_add_u32_e64 v19, v15, s2
	scratch_store_dword off, v19, off offset:724 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v18, v18, s6
	ds_write_b16 v15, v18 offset:4096
	s_mov_b32 s6, 0xa50
	v_xor_b32_e64 v15, v11, s6
	v_add_u32_e64 v15, s3, v15
	v_mov_b32_e32 v18, v15
	scratch_store_dword off, v18, off offset:720 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v17, v17, s6
	ds_write_b16 v15, v17
	v_add_u32_e64 v17, v15, s2
	scratch_store_dword off, v17, off offset:716 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v16, v16, s6
	ds_write_b16 v15, v16 offset:4096
	s_mov_b32 s6, 0xb58
	v_xor_b32_e64 v11, v11, s6
	v_add_u32_e64 v11, s3, v11
	v_mov_b32_e32 v15, v11
	scratch_store_dword off, v15, off offset:712 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v14, v14, s6
	ds_write_b16 v11, v14
	v_add_u32_e64 v14, v11, s2
	scratch_store_dword off, v14, off offset:708 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr6
	v_cvt_pk_bf16_f32 v13, v13, s6
	ds_write_b16 v11, v13 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s6, 60
	v_and_b32_e64 v11, v10, s6
	s_mov_b32 s6, 6
	v_lshlrev_b32_e64 v10, s6, v11
	s_mov_b32 s6, 24
	v_and_b32_e64 v12, v12, s6
	v_lshlrev_b32_e64 v11, s5, v11
	s_mov_b32 s5, 5
	s_lshl_b32 s4, s4, s5
	v_bitop3_b32 v10, v10, v11, v12 bitop3:0x36
	v_xor_b32_e64 v10, v10, s4
	v_add_u32_e64 v10, s3, v10
	scratch_store_dword off, v10, off offset:704 ; 4-byte Folded Spill
	v_mov_b32_e32 v11, v10
	scratch_store_dword off, v11, off offset:700 ; 4-byte Folded Spill
	ds_read_b64_tr_b16 v[12:13], v10
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[12:13], off offset:692 ; 8-byte Folded Spill
	v_add_u32_e64 v11, v10, s2
	scratch_store_dword off, v11, off offset:688 ; 4-byte Folded Spill
	ds_read_b64_tr_b16 v[12:13], v10 offset:4096
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[12:13], off offset:680 ; 8-byte Folded Spill
	s_mov_b32 s2, 0x80
	v_add_u32_e64 v11, v10, s2
	scratch_store_dword off, v11, off offset:676 ; 4-byte Folded Spill
	ds_read_b64_tr_b16 v[12:13], v10 offset:128
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[12:13], off offset:668 ; 8-byte Folded Spill
	s_mov_b32 s2, 0x1080
	v_add_u32_e64 v11, v10, s2
	scratch_store_dword off, v11, off offset:664 ; 4-byte Folded Spill
	ds_read_b64_tr_b16 v[10:11], v10 offset:4224
	.loc	1 259 19 is_stmt 0              ; mha.py:259:19 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[10:11], off offset:656 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], -1
	s_xor_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 vcc, exec, s[0:1]
	scratch_store_dword off, v9, off offset:652 ; 4-byte Folded Spill
	scratch_store_dword off, v8, off offset:648 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:640 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:632 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:624 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], off offset:616 ; 8-byte Folded Spill
	s_cbranch_vccnz .LBB0_13
; %bb.12:
	.loc	1 248 15 is_stmt 1              ; mha.py:248:15 @[ mha.py:746:25 ]
	scratch_load_dword v9, off, off offset:804 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[18:19], off, off offset:44 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[16:17], off, off offset:36 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[20:21], off, off offset:776 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[10:11], off, off offset:28 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[26:27], off, off offset:20 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:440 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:428 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[12:13], off, off offset:668 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[14:15], off, off offset:656 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, off offset:692 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:680 ; 8-byte Folded Reload
	scratch_load_dword v8, off, off offset:796 ; 4-byte Folded Reload
	scratch_load_dword v23, off, off offset:800 ; 4-byte Folded Reload
	scratch_load_dword v22, off, off offset:56 ; 4-byte Folded Reload
	scratch_load_dword v24, off, off offset:792 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mul_f32_e64 v22, v22, v24
.Ltmp87:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_add_f32_e64 v8, v8, v23
.Ltmp88:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_add_f32_e64 v8, v8, v22
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_mov_b32_e32 v22, v3
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1_vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v1, v24
	v_mov_b32_e32 v2, v23
	v_mov_b32_e32 v3, v22
	v_mov_b32_e32 v22, v15
	v_mov_b32_e32 v23, v14
	v_mov_b32_e32 v24, v13
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13_vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v13, v24
	v_mov_b32_e32 v14, v23
	v_mov_b32_e32 v15, v22
	v_mov_b32_e32 v22, v7
	v_mov_b32_e32 v23, v6
	v_mov_b32_e32 v24, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v24
	v_mov_b32_e32 v6, v23
	v_mov_b32_e32 v7, v22
	v_mov_b32_e32 v22, v20
	v_mov_b32_e32 v28, v22
	v_mov_b32_e32 v29, v22
	v_mov_b32_e32 v30, v22
	v_mov_b32_e32 v31, v22
	v_mov_b32_e32 v23, v31
	v_mov_b32_e32 v24, v30
                                        ; kill: def $vgpr28_vgpr29 killed $vgpr28_vgpr29 killed $vgpr28_vgpr29_vgpr30_vgpr31 killed $exec
	v_pk_mul_f32 v[28:29], v[26:27], v[28:29]
	s_nop 0
	v_mov_b32_e32 v26, v29
	v_mov_b32_e32 v22, v28
                                        ; kill: def $vgpr24 killed $vgpr24 def $vgpr24_vgpr25 killed $exec
	v_mov_b32_e32 v25, v23
	v_pk_mul_f32 v[24:25], v[10:11], v[24:25]
	s_nop 0
	v_mov_b32_e32 v10, v25
	v_mov_b32_e32 v11, v24
                                        ; kill: def $vgpr22 killed $vgpr22 def $vgpr22_vgpr23_vgpr24_vgpr25 killed $exec
	v_mov_b32_e32 v23, v26
	v_mov_b32_e32 v24, v11
	v_mov_b32_e32 v25, v10
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[22:25], v[4:7], v[0:3], v[22:25]
	s_nop 7
	v_mov_b32_e32 v10, v23
	v_mov_b32_e32 v0, v22
	v_mov_b32_e32 v1, v25
	v_mov_b32_e32 v2, v24
	v_mov_b32_e32 v3, v21
	v_mov_b32_e32 v24, v3
	v_mov_b32_e32 v25, v3
	v_mov_b32_e32 v26, v3
	v_mov_b32_e32 v27, v3
	v_mov_b32_e32 v3, v27
	v_mov_b32_e32 v22, v26
	v_mov_b64_e32 v[20:21], v[24:25]
	v_pk_mul_f32 v[16:17], v[16:17], v[20:21]
	s_nop 0
	v_mov_b32_e32 v20, v17
                                        ; kill: def $vgpr16 killed $vgpr16 killed $vgpr16_vgpr17 killed $exec
                                        ; kill: def $vgpr22 killed $vgpr22 def $vgpr22_vgpr23 killed $exec
	v_mov_b32_e32 v23, v3
	v_pk_mul_f32 v[18:19], v[18:19], v[22:23]
	s_nop 0
	v_mov_b32_e32 v3, v19
	v_mov_b32_e32 v11, v18
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17_vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v17, v20
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v3
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[12:15], v[4:7], v[12:15], v[16:19]
	s_nop 7
	v_mov_b32_e32 v3, v13
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v5, v15
	v_mov_b32_e32 v6, v14
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v10
                                        ; kill: def $vgpr9 killed $vgpr9 killed $exec
                                        ; kill: def $vgpr8 killed $vgpr8 killed $exec
                                        ; kill: def $vgpr6_vgpr7 killed $vgpr6_vgpr7 killed $exec
                                        ; kill: def $vgpr4_vgpr5 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr2_vgpr3 killed $vgpr2_vgpr3 killed $exec
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	scratch_store_dword off, v9, off offset:652 ; 4-byte Folded Spill
	scratch_store_dword off, v8, off offset:648 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:640 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:632 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:624 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], off offset:616 ; 8-byte Folded Spill
.LBB0_13:
	.loc	1 0 19 is_stmt 0                ; mha.py:0:19
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 46
	v_readlane_b32 s1, v118, 47
	v_readlane_b32 s2, v118, 36
	scratch_load_dword v0, off, off offset:436 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:448 ; 4-byte Folded Reload
	scratch_load_dword v2, off, off offset:652 ; 4-byte Folded Reload
	scratch_load_dword v3, off, off offset:648 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:640 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:632 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, off offset:624 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[10:11], off, off offset:616 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[10:11], off offset:1072 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[8:9], off offset:1064 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:1056 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1048 ; 8-byte Folded Spill
	scratch_store_dword off, v3, off offset:1044 ; 4-byte Folded Spill
.Ltmp89:
	.loc	1 36 18 is_stmt 1               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	scratch_store_dword off, v2, off offset:1040 ; 4-byte Folded Spill
	; wait_asyncmark(0)
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp90:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:150:17 @[ mha.py:746:25 ] ]
	v_add_u32_e64 v1, s2, v1
	ds_read_b64_tr_b16 v[2:3], v1
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[2:3], off offset:1032 ; 8-byte Folded Spill
	v_add_u32_e64 v0, s2, v0
	ds_read_b64_tr_b16 v[0:1], v0
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], off offset:1024 ; 8-byte Folded Spill
	s_mov_b32 s2, 0
	v_writelane_b32 v118, s2, 53
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_mov_b32_e32 v0, s2
	v_mov_b32_e32 v2, s2
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v8, s2
	v_mov_b32_e32 v7, s2
	v_mov_b32_e32 v6, s2
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3_vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v4, v7
	v_mov_b32_e32 v5, v6
	v_mov_b32_e32 v6, s2
	v_mov_b32_e32 v20, s2
	v_mov_b32_e32 v19, s2
	v_mov_b32_e32 v18, s2
	v_mov_b32_e32 v17, s2
	v_mov_b32_e32 v16, s2
	v_mov_b32_e32 v15, s2
	v_mov_b32_e32 v14, s2
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v7, v20
	v_mov_b32_e32 v8, v19
	v_mov_b32_e32 v9, v18
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v11, v16
	v_mov_b32_e32 v12, v15
	v_mov_b32_e32 v13, v14
	scratch_store_dwordx4 off, v[6:9], off offset:992 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[10:13], off offset:1008 ; 16-byte Folded Spill
	v_mov_b32_e32 v14, s2
.Ltmp91:
	.loc	1 182 14 is_stmt 1              ; mha.py:182:14 @[ mha.py:746:25 ]
	s_mov_b64 s[2:3], -1
	s_xor_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 vcc, exec, s[0:1]
.Ltmp92:
	.loc	1 746 25                        ; mha.py:746:25
	v_mov_b32_e32 v15, v14
                                        ; kill: def $vgpr14 killed $vgpr14 killed $exec
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	scratch_store_dword off, v15, off offset:988 ; 4-byte Folded Spill
	scratch_store_dword off, v14, off offset:984 ; 4-byte Folded Spill
	scratch_store_dwordx4 off, v[6:9], off offset:952 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[10:13], off offset:968 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[2:5], off offset:936 ; 16-byte Folded Spill
.Ltmp93:
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	scratch_store_dwordx2 off, v[0:1], off offset:928 ; 8-byte Folded Spill
	s_cbranch_vccnz .LBB0_15
; %bb.14:
.Ltmp94:
	.loc	1 36 18                         ; mha.py:36:18 @[ mha.py:142:20 @[ mha.py:746:25 ] ]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 32
	v_readlane_b32 s1, v118, 34
	v_accvgpr_read_b32 v8, a25              ;  Reload Reuse
	v_accvgpr_read_b32 v9, a24              ;  Reload Reuse
	v_accvgpr_read_b32 v10, a23             ;  Reload Reuse
	v_accvgpr_read_b32 v4, a22              ;  Reload Reuse
	v_accvgpr_read_b32 v5, a21              ;  Reload Reuse
	v_accvgpr_read_b32 v6, a20              ;  Reload Reuse
	v_accvgpr_read_b32 v7, a19              ;  Reload Reuse
	v_accvgpr_read_b32 v34, a18             ;  Reload Reuse
	scratch_load_dword v0, off, off offset:452 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e64 v1, s1, v0
	ds_read_b128 v[12:15], v1
.Ltmp95:
	.loc	1 36 18 is_stmt 0               ; mha.py:36:18 @[ mha.py:140:13 @[ mha.py:746:25 ] ]
	v_add_u32_e64 v0, s0, v0
	ds_read_b128 v[0:3], v0
.Ltmp96:
                                        ; kill: def $vgpr34 killed $vgpr34 def $vgpr34_vgpr35_vgpr36_vgpr37 killed $exec
	.loc	1 181 18 is_stmt 1              ; mha.py:181:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v35, v7
	v_mov_b32_e32 v36, v6
	v_mov_b32_e32 v37, v5
	s_mov_b32 s0, 0
	v_writelane_b32 v118, s0, 54
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_mov_b32_e32 v18, s0
	v_mov_b32_e32 v46, s0
	v_mov_b32_e32 v45, s0
	v_mov_b32_e32 v44, s0
	v_mov_b32_e32 v43, s0
	v_mov_b32_e32 v42, s0
	v_mov_b32_e32 v41, s0
	v_mov_b32_e32 v40, s0
	v_mov_b32_e32 v39, s0
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v17, s0
	v_mov_b32_e32 v16, s0
	v_mov_b32_e32 v11, s0
	v_mov_b32_e32 v7, s0
	v_mov_b32_e32 v6, s0
	v_mov_b32_e32 v5, s0
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31_vgpr32_vgpr33 killed $exec
	v_mov_b32_e32 v19, v46
	v_mov_b32_e32 v20, v45
	v_mov_b32_e32 v21, v44
	v_mov_b32_e32 v22, v43
	v_mov_b32_e32 v23, v42
	v_mov_b32_e32 v24, v41
	v_mov_b32_e32 v25, v40
	v_mov_b32_e32 v26, v39
	v_mov_b32_e32 v27, v38
	v_mov_b32_e32 v28, v17
	v_mov_b32_e32 v29, v16
	v_mov_b32_e32 v30, v11
	v_mov_b32_e32 v31, v7
	v_mov_b32_e32 v32, v6
	v_mov_b32_e32 v33, v5
	scratch_store_dwordx4 off, v[18:21], off offset:1176 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[22:25], off offset:1192 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[26:29], off offset:1208 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[30:33], off offset:1224 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_f32_32x32x16_bf16 v[18:33], v[12:15], v[34:37], v[18:33]
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v8
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_32x32x16_bf16 v[18:33], v[0:3], v[4:7], v[18:33]
	s_nop 11
	scratch_store_dwordx4 off, v[18:21], off offset:1112 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[22:25], off offset:1128 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[26:29], off offset:1144 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[30:33], off offset:1160 ; 16-byte Folded Spill
	v_mov_b32_e32 v0, v25
	v_mov_b32_e32 v1, v24
	v_mov_b32_e32 v2, v23
	v_mov_b32_e32 v3, v22
	v_mov_b32_e32 v4, v21
	v_mov_b32_e32 v5, v20
	v_mov_b32_e32 v16, v19
	v_mov_b32_e32 v6, v18
	v_mov_b32_e32 v15, v26
	v_mov_b32_e32 v14, v33
.Ltmp97:
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13 killed $exec
	.loc	1 746 25                        ; mha.py:746:25
	v_mov_b32_e32 v7, v16
	v_mov_b32_e32 v8, v5
	v_mov_b32_e32 v9, v4
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v1
	v_mov_b32_e32 v13, v0
	scratch_store_dwordx4 off, v[6:9], off offset:1080 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[10:13], off offset:1096 ; 16-byte Folded Spill
	v_mov_b32_e32 v0, v30
	v_mov_b32_e32 v1, v29
	v_mov_b32_e32 v16, v28
	v_mov_b32_e32 v2, v27
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3_vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v3, v16
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v16, v32
	v_mov_b32_e32 v0, v31
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v16
                                        ; kill: def $vgpr15 killed $vgpr15 killed $exec
                                        ; kill: def $vgpr14 killed $vgpr14 killed $exec
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	scratch_store_dword off, v15, off offset:988 ; 4-byte Folded Spill
	scratch_store_dword off, v14, off offset:984 ; 4-byte Folded Spill
	scratch_store_dwordx4 off, v[6:9], off offset:952 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[10:13], off offset:968 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[2:5], off offset:936 ; 16-byte Folded Spill
.Ltmp98:
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:746:25 ]
	scratch_store_dwordx2 off, v[0:1], off offset:928 ; 8-byte Folded Spill
.Ltmp99:
.LBB0_15:
	.loc	1 746 25                        ; mha.py:746:25
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 46
	v_readlane_b32 s1, v118, 47
	scratch_load_dword v0, off, off offset:1040 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:1044 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:1048 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:1056 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:1064 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, off offset:1072 ; 8-byte Folded Reload
	scratch_load_dword v10, off, off offset:664 ; 4-byte Folded Reload
	scratch_load_dword v11, off, off offset:676 ; 4-byte Folded Reload
	scratch_load_dword v12, off, off offset:688 ; 4-byte Folded Reload
	scratch_load_dword v13, off, off offset:700 ; 4-byte Folded Reload
	scratch_load_dword v14, off, off offset:708 ; 4-byte Folded Reload
	scratch_load_dword v16, off, off offset:712 ; 4-byte Folded Reload
	scratch_load_dword v18, off, off offset:716 ; 4-byte Folded Reload
	scratch_load_dword v20, off, off offset:720 ; 4-byte Folded Reload
	scratch_load_dword v22, off, off offset:724 ; 4-byte Folded Reload
	scratch_load_dword v24, off, off offset:728 ; 4-byte Folded Reload
	scratch_load_dword v26, off, off offset:732 ; 4-byte Folded Reload
	scratch_load_dword v28, off, off offset:736 ; 4-byte Folded Reload
	scratch_load_dword v30, off, off offset:740 ; 4-byte Folded Reload
	scratch_load_dword v32, off, off offset:744 ; 4-byte Folded Reload
	scratch_load_dword v34, off, off offset:748 ; 4-byte Folded Reload
	scratch_load_dword v36, off, off offset:752 ; 4-byte Folded Reload
	scratch_load_dword v38, off, off offset:756 ; 4-byte Folded Reload
	scratch_load_dword v40, off, off offset:760 ; 4-byte Folded Reload
	scratch_load_dword v42, off, off offset:764 ; 4-byte Folded Reload
	scratch_load_dword v44, off, off offset:768 ; 4-byte Folded Reload
	scratch_load_dword v46, off, off offset:784 ; 4-byte Folded Reload
	scratch_load_dword v47, off, off offset:788 ; 4-byte Folded Reload
	v_accvgpr_read_b32 v15, a38             ;  Reload Reuse
	scratch_load_dwordx2 v[48:49], off, off offset:808 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[50:51], off, off offset:928 ; 8-byte Folded Reload
	scratch_load_dwordx4 v[76:79], off, off offset:816 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[84:87], off, off offset:864 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[88:91], off, off offset:880 ; 16-byte Folded Reload
	scratch_load_dword v17, off, off offset:988 ; 4-byte Folded Reload
	scratch_load_dword v19, off, off offset:984 ; 4-byte Folded Reload
	scratch_load_dwordx4 v[92:95], off, off offset:952 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[96:99], off, off offset:968 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[64:67], off, off offset:936 ; 16-byte Folded Reload
	s_waitcnt vmcnt(2)
	scratch_store_dwordx4 off, v[92:95], off offset:1368 ; 16-byte Folded Spill
	s_waitcnt vmcnt(2)
	scratch_store_dwordx4 off, v[96:99], off offset:1384 ; 16-byte Folded Spill
.Ltmp100:
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:746:25 ]
	v_mov_b32_e32 v21, v99
	v_mov_b32_e32 v23, v98
	v_mov_b32_e32 v25, v97
	v_mov_b32_e32 v80, v96
                                        ; kill: def $vgpr80 killed $vgpr80 def $vgpr80_vgpr81_vgpr82_vgpr83 killed $exec
	v_mov_b32_e32 v81, v25
	v_mov_b32_e32 v82, v23
	v_mov_b32_e32 v83, v21
	v_mov_b32_e32 v21, v83
	v_mov_b32_e32 v54, v82
                                        ; kill: def $vgpr54 killed $vgpr54 def $vgpr54_vgpr55 killed $exec
	v_mov_b32_e32 v55, v21
	v_mov_b32_e32 v21, v91
	v_mov_b32_e32 v23, v90
	v_mov_b32_e32 v25, v89
	v_mov_b32_e32 v72, v88
                                        ; kill: def $vgpr72 killed $vgpr72 def $vgpr72_vgpr73_vgpr74_vgpr75 killed $exec
	v_mov_b32_e32 v73, v25
	v_mov_b32_e32 v74, v23
	v_mov_b32_e32 v75, v21
	v_mov_b32_e32 v21, v75
	v_mov_b32_e32 v52, v74
                                        ; kill: def $vgpr52 killed $vgpr52 def $vgpr52_vgpr53 killed $exec
	v_mov_b32_e32 v53, v21
	v_pk_mul_f32 v[56:57], v[52:53], v[54:55]
	s_nop 0
	v_mov_b32_e32 v21, v57
	v_mov_b32_e32 v23, v56
	v_mov_b32_e32 v25, v95
	v_mov_b32_e32 v27, v94
	v_mov_b32_e32 v29, v93
	v_mov_b32_e32 v52, v92
                                        ; kill: def $vgpr52 killed $vgpr52 def $vgpr52_vgpr53_vgpr54_vgpr55 killed $exec
	v_mov_b32_e32 v53, v29
	v_mov_b32_e32 v54, v27
	v_mov_b32_e32 v55, v25
	v_mov_b32_e32 v25, v55
	v_mov_b32_e32 v60, v54
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v25
	v_mov_b32_e32 v25, v87
	v_mov_b32_e32 v27, v86
	v_mov_b32_e32 v29, v85
	v_mov_b32_e32 v68, v84
                                        ; kill: def $vgpr68 killed $vgpr68 def $vgpr68_vgpr69_vgpr70_vgpr71 killed $exec
	v_mov_b32_e32 v69, v29
	v_mov_b32_e32 v70, v27
	v_mov_b32_e32 v71, v25
	v_mov_b32_e32 v25, v71
	v_mov_b32_e32 v58, v70
                                        ; kill: def $vgpr58 killed $vgpr58 def $vgpr58_vgpr59 killed $exec
	v_mov_b32_e32 v59, v25
	v_pk_mul_f32 v[60:61], v[58:59], v[60:61]
	s_nop 0
	v_mov_b32_e32 v29, v61
	v_mov_b32_e32 v31, v60
	v_mov_b64_e32 v[62:63], v[80:81]
	v_mov_b64_e32 v[58:59], v[72:73]
	v_pk_mul_f32 v[58:59], v[58:59], v[62:63]
	s_nop 0
	v_mov_b32_e32 v25, v59
	v_mov_b32_e32 v27, v58
	v_mov_b64_e32 v[54:55], v[52:53]
	v_mov_b64_e32 v[52:53], v[68:69]
	v_pk_mul_f32 v[62:63], v[52:53], v[54:55]
	s_nop 0
	v_mov_b32_e32 v33, v63
	v_mov_b32_e32 v68, v62
                                        ; kill: def $vgpr68 killed $vgpr68 def $vgpr68_vgpr69_vgpr70_vgpr71_vgpr72_vgpr73_vgpr74_vgpr75 killed $exec
	v_mov_b32_e32 v69, v33
	v_mov_b32_e32 v70, v31
	v_mov_b32_e32 v71, v29
	v_mov_b32_e32 v72, v27
	v_mov_b32_e32 v73, v25
	v_mov_b32_e32 v74, v23
	v_mov_b32_e32 v75, v21
	scratch_store_dwordx4 off, v[68:71], off offset:1336 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[72:75], off offset:1352 ; 16-byte Folded Spill
	v_mul_f32_e64 v17, v15, v17
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v21, v67
	v_mov_b32_e32 v54, v66
                                        ; kill: def $vgpr54 killed $vgpr54 def $vgpr54_vgpr55 killed $exec
	v_mov_b32_e32 v55, v21
	v_mov_b32_e32 v21, v79
	v_mov_b32_e32 v52, v78
                                        ; kill: def $vgpr52 killed $vgpr52 def $vgpr52_vgpr53 killed $exec
	v_mov_b32_e32 v53, v21
	v_pk_mul_f32 v[52:53], v[52:53], v[54:55]
	s_nop 0
	v_mov_b32_e32 v21, v53
	v_mov_b32_e32 v23, v52
                                        ; kill: def $vgpr64_vgpr65 killed $vgpr64_vgpr65 killed $vgpr64_vgpr65_vgpr66_vgpr67 killed $exec
	v_mov_b64_e32 v[54:55], v[76:77]
	v_pk_mul_f32 v[54:55], v[54:55], v[64:65]
	s_nop 0
	v_mov_b32_e32 v25, v55
	v_mov_b32_e32 v64, v54
                                        ; kill: def $vgpr64 killed $vgpr64 def $vgpr64_vgpr65_vgpr66_vgpr67 killed $exec
	v_mov_b32_e32 v65, v25
	v_mov_b32_e32 v66, v23
	v_mov_b32_e32 v67, v21
	v_pk_mul_f32 v[50:51], v[48:49], v[50:51]
	v_mul_f32_e64 v15, v15, v19
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v19, v68
	v_mov_b32_e32 v21, v69
.Ltmp101:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v19, v19, v21
.Ltmp102:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v21, v70
.Ltmp103:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v19, v19, v21
.Ltmp104:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v21, v71
	v_mov_b32_e32 v23, v72
.Ltmp105:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v21, v21, v23
.Ltmp106:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v23, v73
.Ltmp107:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v27, v21, v23
.Ltmp108:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v21, v74
	v_mov_b32_e32 v23, v75
.Ltmp109:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v21, v21, v23
	v_max_f32_e64 v25, v21, v17
.Ltmp110:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v21, v64
	v_mov_b32_e32 v23, v65
.Ltmp111:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v21, v21, v23
.Ltmp112:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v23, v66
.Ltmp113:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v21, v21, v23
.Ltmp114:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	v_mov_b32_e32 v23, v67
.Ltmp115:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_mov_b32_e32 v29, v50
	v_max_f32_e64 v23, v23, v29
	v_mov_b32_e32 v29, v51
	v_max_f32_e64 v23, v23, v29
	v_max_f32_e64 v19, v19, v27
	v_max_f32_e64 v19, v19, v25
	v_max_f32_e64 v21, v21, v23
	v_max_f32_e64 v21, v21, v15
	v_max_f32_e64 v21, v19, v21
.Ltmp116:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v19, v21
	s_nop 1
	v_permlane32_swap_b32_e64 v19, v21
.Ltmp117:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:746:25 ] ] ]
	v_max_f32_e64 v21, v21, v21
	v_max_f32_e64 v19, v19, v19
	v_max_f32_e64 v21, v19, v21
.Ltmp118:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:746:25 ]
	v_max_f32_e64 v19, v0, v0
	v_max_f32_e64 v48, v19, v21
	v_mov_b32_e32 v19, v48
	scratch_store_dword off, v19, off offset:1332 ; 4-byte Folded Spill
                                        ; implicit-def: $sgpr2_sgpr3
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:746:25 ]
	s_mov_b32 s2, s3
	v_writelane_b32 v118, s2, 55
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_mov_b32_e32 v19, s2
	v_mov_b32_e32 v64, v48
	v_mov_b32_e32 v65, v19
	v_pk_add_f32 v[62:63], v[62:63], v[64:65] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[60:61], v[60:61], v[64:65] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[58:59], v[58:59], v[64:65] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[56:57], v[56:57], v[64:65] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e64 v19, v17, v48
	v_pk_add_f32 v[54:55], v[54:55], v[64:65] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53], v[52:53], v[64:65] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[50:51], v[50:51], v[64:65] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_sub_f32_e64 v15, v15, v48
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:746:25 ]
	v_mov_b32_e32 v17, v62
	v_exp_f32_e64 v45, v17
	v_mov_b32_e32 v17, v63
	v_exp_f32_e64 v41, v17
	v_mov_b32_e32 v17, v60
	v_exp_f32_e64 v37, v17
	v_mov_b32_e32 v17, v61
	v_exp_f32_e64 v33, v17
	v_mov_b32_e32 v17, v58
	v_exp_f32_e64 v29, v17
	v_mov_b32_e32 v17, v59
	v_exp_f32_e64 v25, v17
	v_mov_b32_e32 v17, v56
	v_exp_f32_e64 v21, v17
	v_mov_b32_e32 v17, v57
	v_exp_f32_e64 v17, v17
	v_exp_f32_e64 v43, v19
	v_mov_b32_e32 v19, v54
	v_exp_f32_e64 v39, v19
	v_mov_b32_e32 v19, v55
	v_exp_f32_e64 v35, v19
	v_mov_b32_e32 v19, v52
	v_exp_f32_e64 v31, v19
	v_mov_b32_e32 v19, v53
	v_exp_f32_e64 v27, v19
	v_mov_b32_e32 v19, v50
	v_exp_f32_e64 v23, v19
	v_mov_b32_e32 v19, v51
	v_exp_f32_e64 v19, v19
	v_exp_f32_e64 v15, v15
.Ltmp119:
	.loc	3 293 12 is_stmt 1              ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	v_mov_b32_e32 v49, s2
	v_mov_b32_e32 v50, v45
	v_mov_b32_e32 v51, v49
                                        ; kill: def $vgpr50 killed $vgpr50 killed $vgpr50_vgpr51 killed $exec
                                        ; kill: def $vgpr50 killed $vgpr50 def $vgpr50_vgpr51 killed $exec
	v_mov_b32_e32 v51, v41
	v_mov_b32_e32 v49, s2
	v_mov_b32_e32 v52, v37
	v_mov_b32_e32 v53, v49
	v_mov_b32_e32 v64, v52
                                        ; kill: def $vgpr64 killed $vgpr64 def $vgpr64_vgpr65 killed $exec
	v_mov_b32_e32 v65, v33
	v_mov_b32_e32 v49, s2
	v_mov_b32_e32 v52, v29
	v_mov_b32_e32 v53, v49
	v_mov_b32_e32 v56, v52
                                        ; kill: def $vgpr56 killed $vgpr56 def $vgpr56_vgpr57 killed $exec
	v_mov_b32_e32 v57, v25
	v_mov_b32_e32 v49, s2
	v_mov_b32_e32 v52, v21
	v_mov_b32_e32 v53, v49
	v_mov_b32_e32 v62, v52
                                        ; kill: def $vgpr62 killed $vgpr62 def $vgpr62_vgpr63 killed $exec
	v_mov_b32_e32 v63, v17
	v_mov_b32_e32 v49, s2
	v_mov_b32_e32 v52, v43
	v_mov_b32_e32 v53, v49
                                        ; kill: def $vgpr52 killed $vgpr52 killed $vgpr52_vgpr53 killed $exec
                                        ; kill: def $vgpr52 killed $vgpr52 def $vgpr52_vgpr53 killed $exec
	v_mov_b32_e32 v53, v39
	v_mov_b32_e32 v49, s2
	v_mov_b32_e32 v54, v35
	v_mov_b32_e32 v55, v49
	v_mov_b32_e32 v60, v54
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v31
	v_mov_b32_e32 v49, s2
	v_mov_b32_e32 v54, v27
	v_mov_b32_e32 v55, v49
                                        ; kill: def $vgpr54 killed $vgpr54 killed $vgpr54_vgpr55 killed $exec
                                        ; kill: def $vgpr54 killed $vgpr54 def $vgpr54_vgpr55 killed $exec
	v_mov_b32_e32 v55, v23
	v_mov_b32_e32 v49, s2
	v_mov_b32_e32 v58, v19
	v_mov_b32_e32 v59, v49
                                        ; kill: def $vgpr58 killed $vgpr58 killed $vgpr58_vgpr59 killed $exec
                                        ; kill: def $vgpr58 killed $vgpr58 def $vgpr58_vgpr59 killed $exec
	v_mov_b32_e32 v59, v15
	v_pk_add_f32 v[50:51], v[50:51], v[64:65]
	v_pk_add_f32 v[56:57], v[56:57], v[62:63]
	v_pk_add_f32 v[52:53], v[52:53], v[60:61]
	v_pk_add_f32 v[54:55], v[54:55], v[58:59]
	v_pk_add_f32 v[50:51], v[50:51], v[56:57]
	v_pk_add_f32 v[52:53], v[52:53], v[54:55]
	s_nop 0
	v_pk_add_f32 v[50:51], v[50:51], v[52:53]
.Ltmp120:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	s_nop 0
	v_pk_add_f32 v[50:51], v[50:51], v[50:51] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp121:
                                        ; kill: def $vgpr50 killed $vgpr50 killed $vgpr50_vgpr51 killed $exec
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ]
	s_nop 0
	v_mov_b32_e32 v49, v50
	s_nop 1
	v_permlane32_swap_b32_e64 v49, v50
	scratch_store_dword off, v50, off offset:1328 ; 4-byte Folded Spill
.Ltmp122:
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:746:25 ]
	scratch_store_dword off, v49, off offset:1324 ; 4-byte Folded Spill
	v_sub_f32_e64 v48, v0, v48
	.loc	1 241 17 is_stmt 0              ; mha.py:241:17 @[ mha.py:746:25 ]
	v_exp_f32_e64 v48, v48
	s_nop 0
	v_mov_b32_e32 v49, v48
	.loc	1 246 15 is_stmt 1              ; mha.py:246:15 @[ mha.py:746:25 ]
	scratch_store_dword off, v49, off offset:1320 ; 4-byte Folded Spill
	ds_write_b32 v47, v48
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[46:47], v46
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[46:47], off offset:1312 ; 8-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	s_barrier
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v45, v45, s2
	ds_write_b16 v44, v45
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v43, v43, s2
	ds_write_b16 v42, v43
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v41, v41, s2
	ds_write_b16 v40, v41
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v39, v39, s2
	ds_write_b16 v38, v39
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v37, v37, s2
	ds_write_b16 v36, v37
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v35, v35, s2
	ds_write_b16 v34, v35
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v33, v33, s2
	ds_write_b16 v32, v33
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v31, v31, s2
	ds_write_b16 v30, v31
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v29, v29, s2
	ds_write_b16 v28, v29
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v27, v27, s2
	ds_write_b16 v26, v27
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v25, v25, s2
	ds_write_b16 v24, v25
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v23, v23, s2
	ds_write_b16 v22, v23
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v21, v21, s2
	ds_write_b16 v20, v21
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v19, v19, s2
	ds_write_b16 v18, v19
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v17, v17, s2
	ds_write_b16 v16, v17
                                        ; implicit-def: $sgpr2
	v_cvt_pk_bf16_f32 v15, v15, s2
	ds_write_b16 v14, v15
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[14:15], v13
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[14:15], off offset:1304 ; 8-byte Folded Spill
	ds_read_b64_tr_b16 v[12:13], v12
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[12:13], off offset:1296 ; 8-byte Folded Spill
	ds_read_b64_tr_b16 v[12:13], v11
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[12:13], off offset:1288 ; 8-byte Folded Spill
	ds_read_b64_tr_b16 v[10:11], v10
	.loc	1 259 19 is_stmt 0              ; mha.py:259:19 @[ mha.py:746:25 ]
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[10:11], off offset:1280 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], -1
	s_xor_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 vcc, exec, s[0:1]
                                        ; kill: def $vgpr8_vgpr9 killed $vgpr8_vgpr9 killed $exec
                                        ; kill: def $vgpr6_vgpr7 killed $vgpr6_vgpr7 killed $exec
                                        ; kill: def $vgpr4_vgpr5 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr2_vgpr3 killed $vgpr2_vgpr3 killed $exec
                                        ; kill: def $vgpr1 killed $vgpr1 killed $exec
                                        ; kill: def $vgpr0 killed $vgpr0 killed $exec
	scratch_store_dwordx2 off, v[8:9], off offset:1272 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:1264 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1256 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:1248 ; 8-byte Folded Spill
	scratch_store_dword off, v1, off offset:1244 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:1240 ; 4-byte Folded Spill
	s_cbranch_vccnz .LBB0_17
; %bb.16:
	.loc	1 248 15 is_stmt 1              ; mha.py:248:15 @[ mha.py:746:25 ]
	scratch_load_dword v0, off, off offset:1332 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[18:19], off, off offset:1048 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[16:17], off, off offset:1056 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[20:21], off, off offset:1312 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[10:11], off, off offset:1064 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[26:27], off, off offset:1072 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:1032 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:1024 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[12:13], off, off offset:1288 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[14:15], off, off offset:1280 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:1304 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, off offset:1296 ; 8-byte Folded Reload
	scratch_load_dword v1, off, off offset:1324 ; 4-byte Folded Reload
	scratch_load_dword v23, off, off offset:1328 ; 4-byte Folded Reload
	scratch_load_dword v22, off, off offset:1044 ; 4-byte Folded Reload
	scratch_load_dword v24, off, off offset:1320 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mul_f32_e64 v22, v22, v24
.Ltmp123:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:746:25 ] ] ]
	v_add_f32_e64 v1, v1, v23
.Ltmp124:
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:746:25 ]
	v_add_f32_e64 v1, v1, v22
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	v_mov_b32_e32 v22, v9
	v_mov_b32_e32 v23, v8
	v_mov_b32_e32 v24, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7_vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v7, v24
	v_mov_b32_e32 v8, v23
	v_mov_b32_e32 v9, v22
	v_mov_b32_e32 v22, v15
	v_mov_b32_e32 v23, v14
	v_mov_b32_e32 v24, v13
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13_vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v13, v24
	v_mov_b32_e32 v14, v23
	v_mov_b32_e32 v15, v22
	v_mov_b32_e32 v22, v5
	v_mov_b32_e32 v23, v4
	v_mov_b32_e32 v24, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3_vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v3, v24
	v_mov_b32_e32 v4, v23
	v_mov_b32_e32 v5, v22
	v_mov_b32_e32 v22, v20
	v_mov_b32_e32 v28, v22
	v_mov_b32_e32 v29, v22
	v_mov_b32_e32 v30, v22
	v_mov_b32_e32 v31, v22
	v_mov_b32_e32 v23, v31
	v_mov_b32_e32 v24, v30
                                        ; kill: def $vgpr28_vgpr29 killed $vgpr28_vgpr29 killed $vgpr28_vgpr29_vgpr30_vgpr31 killed $exec
	v_pk_mul_f32 v[28:29], v[26:27], v[28:29]
	s_nop 0
	v_mov_b32_e32 v26, v29
	v_mov_b32_e32 v22, v28
                                        ; kill: def $vgpr24 killed $vgpr24 def $vgpr24_vgpr25 killed $exec
	v_mov_b32_e32 v25, v23
	v_pk_mul_f32 v[24:25], v[10:11], v[24:25]
	s_nop 0
	v_mov_b32_e32 v10, v25
	v_mov_b32_e32 v11, v24
                                        ; kill: def $vgpr22 killed $vgpr22 def $vgpr22_vgpr23_vgpr24_vgpr25 killed $exec
	v_mov_b32_e32 v23, v26
	v_mov_b32_e32 v24, v11
	v_mov_b32_e32 v25, v10
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[22:25], v[2:5], v[6:9], v[22:25]
	s_nop 7
	v_mov_b32_e32 v10, v23
	v_mov_b32_e32 v8, v22
	v_mov_b32_e32 v9, v25
	v_mov_b32_e32 v6, v24
	v_mov_b32_e32 v7, v21
	v_mov_b32_e32 v24, v7
	v_mov_b32_e32 v25, v7
	v_mov_b32_e32 v26, v7
	v_mov_b32_e32 v27, v7
	v_mov_b32_e32 v7, v27
	v_mov_b32_e32 v22, v26
	v_mov_b64_e32 v[20:21], v[24:25]
	v_pk_mul_f32 v[16:17], v[16:17], v[20:21]
	s_nop 0
	v_mov_b32_e32 v20, v17
                                        ; kill: def $vgpr16 killed $vgpr16 killed $vgpr16_vgpr17 killed $exec
                                        ; kill: def $vgpr22 killed $vgpr22 def $vgpr22_vgpr23 killed $exec
	v_mov_b32_e32 v23, v7
	v_pk_mul_f32 v[18:19], v[18:19], v[22:23]
	s_nop 0
	v_mov_b32_e32 v7, v19
	v_mov_b32_e32 v11, v18
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17_vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v17, v20
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v7
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[12:15], v[2:5], v[12:15], v[16:19]
	s_nop 7
	v_mov_b32_e32 v7, v13
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v5, v15
	v_mov_b32_e32 v2, v14
.Ltmp125:
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	.loc	1 744 5                         ; mha.py:744:5
	v_mov_b32_e32 v3, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v9
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v10
	scratch_store_dwordx2 off, v[8:9], off offset:1272 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:1264 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1256 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:1248 ; 8-byte Folded Spill
	scratch_store_dword off, v1, off offset:1244 ; 4-byte Folded Spill
.Ltmp126:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:746:25 ]
	scratch_store_dword off, v0, off offset:1240 ; 4-byte Folded Spill
.Ltmp127:
.LBB0_17:                               ; %Flow4
	.loc	1 0 19 is_stmt 0                ; mha.py:0:19
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 51
	scratch_load_dword v0, off, off offset:1240 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, off offset:1272 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:1264 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:1256 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:1248 ; 8-byte Folded Reload
	scratch_load_dword v1, off, off offset:1244 ; 4-byte Folded Reload
	s_waitcnt vmcnt(4)
	v_accvgpr_write_b32 a26, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a27, v8             ;  Reload Reuse
	s_waitcnt vmcnt(3)
	v_accvgpr_write_b32 a28, v7             ;  Reload Reuse
	v_accvgpr_write_b32 a29, v6             ;  Reload Reuse
	s_waitcnt vmcnt(2)
	v_accvgpr_write_b32 a30, v5             ;  Reload Reuse
	v_accvgpr_write_b32 a31, v4             ;  Reload Reuse
	s_waitcnt vmcnt(1)
	v_accvgpr_write_b32 a32, v3             ;  Reload Reuse
	v_accvgpr_write_b32 a33, v2             ;  Reload Reuse
	v_writelane_b32 v118, s0, 50
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a34, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a35, v1             ;  Reload Reuse
	v_accvgpr_write_b32 a36, v0             ;  Reload Reuse
	s_branch .LBB0_8
.LBB0_18:
	.loc	1 798 8 is_stmt 1               ; mha.py:798:8
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 47
	scratch_load_dwordx2 v[0:1], off, off offset:292 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:300 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:308 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:316 ; 8-byte Folded Reload
	scratch_load_dword v8, off, off offset:324 ; 4-byte Folded Reload
	scratch_load_dword v9, off, off offset:328 ; 4-byte Folded Reload
	s_mov_b32 s1, 0
	s_cmp_gt_i32 s0, s1
	s_cselect_b64 s[0:1], -1, 0
	.loc	1 798 5 is_stmt 0               ; mha.py:798:5
	s_mov_b64 s[2:3], -1
	s_xor_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 vcc, exec, s[0:1]
                                        ; kill: def $vgpr9 killed $vgpr9 killed $exec
                                        ; kill: def $vgpr8 killed $vgpr8 killed $exec
                                        ; kill: def $vgpr6_vgpr7 killed $vgpr6_vgpr7 killed $exec
                                        ; kill: def $vgpr4_vgpr5 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr2_vgpr3 killed $vgpr2_vgpr3 killed $exec
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v9, off offset:1436 ; 4-byte Folded Spill
	scratch_store_dword off, v8, off offset:1432 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:1424 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1416 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:1408 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], off offset:1400 ; 8-byte Folded Spill
                                        ; implicit-def: $vgpr118 : SGPR spill to VGPR lane
	s_cbranch_vccnz .LBB0_24
; %bb.19:
	.loc	1 0 5                           ; mha.py:0:5
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 45
	v_readlane_b32 s1, v117, 49
	v_readlane_b32 s2, v117, 30
	v_readlane_b32 s3, v117, 31
	v_readlane_b32 s6, v117, 28
	v_readlane_b32 s7, v117, 29
	v_readlane_b32 s4, v117, 11
	scratch_load_dword v0, off, off offset:328 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:324 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:316 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:308 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:300 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, off offset:292 ; 8-byte Folded Reload
	s_mov_b32 s5, 0x3fb8aa3b
.Ltmp128:
	.loc	1 177 20 is_stmt 1              ; mha.py:177:20 @[ mha.py:811:25 ]
	v_mov_b32_e32 v10, s5
	v_mul_f32_e64 v10, s4, v10
	scratch_store_dword off, v10, off offset:1480 ; 4-byte Folded Spill
	s_mov_b32 s4, 5
	.loc	1 261 19                        ; mha.py:261:19 @[ mha.py:811:25 ]
	s_lshl_b64 s[6:7], s[6:7], s4
	.loc	1 264 19                        ; mha.py:264:19 @[ mha.py:811:25 ]
	v_writelane_b32 v118, s6, 56
	s_nop 1
	v_writelane_b32 v118, s7, 57
	s_lshl_b64 s[2:3], s[2:3], s4
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	v_writelane_b32 v118, s2, 58
	s_nop 1
	v_writelane_b32 v118, s3, 59
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v118           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_cmp_lt_i32 s0, s1
	s_cselect_b64 s[0:1], -1, 0
	s_mov_b64 s[2:3], -1
	s_xor_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 vcc, exec, s[0:1]
	s_waitcnt vmcnt(1)
	scratch_store_dwordx2 off, v[8:9], off offset:1472 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:1464 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1456 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:1448 ; 8-byte Folded Spill
	scratch_store_dword off, v1, off offset:1444 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:1440 ; 4-byte Folded Spill
	s_cbranch_vccnz .LBB0_21
.Ltmp129:
; %bb.20:                               ; %.lr.ph121
	.loc	1 803 19                        ; mha.py:803:19
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v116, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v117, 45
	v_readlane_b32 s5, v116, 26
	v_readlane_b32 s2, v116, 19
	v_readlane_b32 s4, v116, 18
	v_readlane_b32 s6, v116, 5
	v_readlane_b32 s7, v116, 6
	v_readlane_b32 s12, v116, 7
	v_readlane_b32 s13, v116, 8
	v_readlane_b32 s16, v116, 41
	v_readlane_b32 s17, v116, 42
	v_readlane_b32 s18, v116, 28
	v_readlane_b32 s19, v116, 29
	v_readlane_b32 s20, v116, 43
	v_readlane_b32 s21, v116, 44
	v_readlane_b32 s24, v116, 30
	v_readlane_b32 s25, v116, 31
	v_readlane_b32 s1, v116, 48
	s_or_saveexec_b64 s[44:45], -1
	scratch_load_dword v118, off, off       ; 4-byte Folded Reload
	s_mov_b64 exec, s[44:45]
	scratch_load_dwordx2 v[0:1], off, off offset:300 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:292 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:316 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:308 ; 8-byte Folded Reload
	scratch_load_dword v8, off, off offset:324 ; 4-byte Folded Reload
	scratch_load_dword v9, off, off offset:328 ; 4-byte Folded Reload
	v_accvgpr_read_b32 v20, a3              ;  Reload Reuse
	v_accvgpr_read_b32 v18, a1              ;  Reload Reuse
	v_accvgpr_read_b32 v19, a4              ;  Reload Reuse
	v_accvgpr_read_b32 v22, a2              ;  Reload Reuse
	v_accvgpr_read_b32 v17, a25             ;  Reload Reuse
	v_accvgpr_read_b32 v21, a24             ;  Reload Reuse
	v_accvgpr_read_b32 v23, a23             ;  Reload Reuse
	v_accvgpr_read_b32 v24, a22             ;  Reload Reuse
	v_accvgpr_read_b32 v25, a21             ;  Reload Reuse
	v_accvgpr_read_b32 v30, a20             ;  Reload Reuse
	v_accvgpr_read_b32 v31, a19             ;  Reload Reuse
	v_accvgpr_read_b32 v26, a18             ;  Reload Reuse
	v_accvgpr_read_b32 v16, a6              ;  Reload Reuse
	v_accvgpr_read_b32 v11, a11             ;  Reload Reuse
	v_accvgpr_read_b32 v10, a12             ;  Reload Reuse
	v_accvgpr_read_b32 v13, a13             ;  Reload Reuse
	v_accvgpr_read_b32 v12, a14             ;  Reload Reuse
	v_accvgpr_read_b32 v15, a15             ;  Reload Reuse
	v_accvgpr_read_b32 v14, a16             ;  Reload Reuse
	s_mov_b32 s10, 5
	s_lshl_b32 s8, s1, s10
	s_mov_b32 s3, 0
	s_mov_b32 s14, s8
	s_mov_b32 s15, s3
	s_mov_b32 s1, 32
	.loc	1 806 19                        ; mha.py:806:19
	s_lshr_b64 s[22:23], s[24:25], s1
	s_mov_b32 s9, s22
	s_mul_i32 s22, s8, s9
	s_mov_b32 s9, s24
	s_mul_hi_u32 s11, s8, s9
	s_add_i32 s11, s11, s22
	s_lshr_b64 s[14:15], s[14:15], s1
                                        ; kill: def $sgpr14 killed $sgpr14 killed $sgpr14_sgpr15
	s_mul_i32 s15, s14, s9
	s_add_i32 s22, s11, s15
                                        ; implicit-def: $sgpr11
                                        ; implicit-def: $sgpr15
                                        ; kill: def $sgpr22 killed $sgpr22 def $sgpr22_sgpr23
	s_mov_b32 s23, s11
	s_lshl_b64 s[24:25], s[22:23], s1
	s_mul_i32 s22, s8, s9
                                        ; kill: def $sgpr22 killed $sgpr22 def $sgpr22_sgpr23
	s_mov_b32 s23, s3
	s_or_b64 s[22:23], s[22:23], s[24:25]
	.loc	1 0 0 is_stmt 0                 ; mha.py:0
	s_mov_b32 s15, s20
	s_mov_b32 s9, s21
	s_mov_b32 s20, s22
	s_mov_b32 s11, s23
	s_add_u32 s20, s15, s20
	s_addc_u32 s9, s9, s11
                                        ; kill: def $sgpr20 killed $sgpr20 def $sgpr20_sgpr21
	s_mov_b32 s21, s9
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[20:21]
	.loc	1 803 19 is_stmt 1              ; mha.py:803:19
	s_lshr_b64 s[20:21], s[18:19], s1
	s_mov_b32 s9, s20
	s_mul_i32 s15, s8, s9
	s_mov_b32 s9, s18
	s_mul_hi_u32 s11, s8, s9
	s_add_i32 s11, s11, s15
	s_mul_i32 s14, s14, s9
	s_add_i32 s14, s11, s14
                                        ; implicit-def: $sgpr11
                                        ; implicit-def: $sgpr15
                                        ; kill: def $sgpr14 killed $sgpr14 def $sgpr14_sgpr15
	s_mov_b32 s15, s11
	s_lshl_b64 s[14:15], s[14:15], s1
	s_mul_i32 s8, s8, s9
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s3
	s_or_b64 s[14:15], s[8:9], s[14:15]
	.loc	1 0 0 is_stmt 0                 ; mha.py:0
	s_mov_b32 s8, s16
	s_mov_b32 s3, s17
	s_mov_b32 s11, s14
	s_mov_b32 s9, s15
	s_add_u32 s8, s8, s11
	s_addc_u32 s3, s3, s9
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s3
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[8:9]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[8:9]
	s_lshr_b64 s[8:9], s[12:13], s1
                                        ; kill: def $sgpr8 killed $sgpr8 killed $sgpr8_sgpr9
	s_mov_b32 s3, 0xffff
	s_and_b32 s8, s8, s3
                                        ; kill: def $sgpr12 killed $sgpr12 def $sgpr12_sgpr13
	s_mov_b32 s13, s8
	s_mov_b32 s11, 0x27000
	s_mov_b32 s8, 0x7ffffffe
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s11
	s_mov_b64 s[14:15], s[8:9]
	v_writelane_b32 v117, s14, 60
	s_nop 1
	v_writelane_b32 v117, s15, 61
	v_writelane_b32 v117, s12, 62
	s_nop 1
	v_writelane_b32 v117, s13, 63
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_write_b32 a37, v117           ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_lshr_b64 s[12:13], s[6:7], s1
	s_mov_b32 s1, s12
	s_and_b32 s1, s1, s3
                                        ; kill: def $sgpr6 killed $sgpr6 def $sgpr6_sgpr7
	s_mov_b32 s7, s1
	s_waitcnt vmcnt(6)
	v_writelane_b32 v118, s8, 0
	s_nop 1
	v_writelane_b32 v118, s9, 1
	v_writelane_b32 v118, s6, 2
	s_nop 1
	v_writelane_b32 v118, s7, 3
	s_mov_b32 s1, 2
	v_lshlrev_b32_e64 v16, s1, v16
	s_mov_b32 s1, 64
	s_and_b32 s3, s5, s1
	s_mov_b32 s1, 0
	v_writelane_b32 v118, s1, 4
	s_mov_b32 s9, 1
	s_lshl_b32 s4, s4, s9
	s_mov_b32 s6, 4
	s_and_b32 s6, s4, s6
	s_lshl2_add_u32 s4, s3, s1
	s_add_i32 s6, s4, s6
	s_lshl_b32 s8, s2, s10
	s_mov_b32 s4, 0
	v_writelane_b32 v118, s4, 5
	s_cmp_eq_u32 s3, s4
	s_mov_b32 s3, 0x110
	s_cselect_b32 s11, s4, s3
	s_mov_b32 s7, 0x2fc
	v_mov_b32_e32 v27, s7
	v_bitop3_b32 v27, v16, s11, v27 bitop3:0x6c
	v_add_u32_e64 v27, s1, v27
	scratch_store_dword off, v27, off offset:1688 ; 4-byte Folded Spill
	v_lshlrev_b32_e64 v27, s10, v18
	s_mov_b32 s10, 0x2e0
	v_and_b32_e64 v27, v27, s10
	s_mov_b32 s10, 8
	v_and_b32_e64 v28, v18, s10
	v_cmp_eq_u32_e64 s[10:11], v28, s4
	v_mov_b32_e32 v28, s3
	v_mov_b32_e32 v29, s4
	v_cndmask_b32_e64 v28, v28, v29, s[10:11]
	v_lshrrev_b32_e64 v29, s9, v19
	v_xor_b32_e64 v28, v28, v29
	v_add3_u32 v27, s1, v27, v28
	scratch_store_dword off, v27, off offset:1684 ; 4-byte Folded Spill
                                        ; kill: def $vgpr26 killed $vgpr26 def $vgpr26_vgpr27_vgpr28_vgpr29 killed $exec
	v_mov_b32_e32 v27, v31
	v_mov_b32_e32 v28, v30
	v_mov_b32_e32 v29, v25
	scratch_store_dwordx4 off, v[26:29], off offset:1668 ; 16-byte Folded Spill
                                        ; kill: def $vgpr24 killed $vgpr24 def $vgpr24_vgpr25_vgpr26_vgpr27 killed $exec
	v_mov_b32_e32 v25, v23
	s_nop 0
	v_mov_b32_e32 v26, v21
	v_mov_b32_e32 v27, v17
	scratch_store_dwordx4 off, v[24:27], off offset:1652 ; 16-byte Folded Spill
	s_mov_b32 s3, 31
	v_and_b32_e64 v21, v18, s3
	s_mov_b32 s3, 3
	v_lshlrev_b32_e64 v17, s3, v21
	v_add_u32_e64 v23, s6, v17
	scratch_store_dword off, v23, off offset:1648 ; 4-byte Folded Spill
	v_mov_b32_e32 v23, s1
	v_lshl_add_u32 v23, v22, s3, v23
	s_mov_b32 s3, 7
	v_mov_b32_e32 v22, s3
	v_lshl_add_u32 v22, s2, v22, v23
	scratch_store_dword off, v22, off offset:1644 ; 4-byte Folded Spill
	v_lshlrev_b32_e64 v21, s9, v21
	v_cmp_eq_u32_e64 s[2:3], v19, s4
	s_mov_b32 s6, 0x420
	v_mov_b32_e32 v19, s6
	v_mov_b32_e32 v22, s4
	v_cndmask_b32_e64 v19, v19, v22, s[2:3]
	v_bitop3_b32 v19, s5, v19, v21 bitop3:0x36
	scratch_store_dword off, v19, off offset:1640 ; 4-byte Folded Spill
	v_add_u32_e64 v21, s1, v19
	v_mov_b32_e32 v22, v21
	scratch_store_dword off, v22, off offset:1636 ; 4-byte Folded Spill
	s_mov_b32 s6, 0x1000
	v_writelane_b32 v118, s6, 6
	v_add_u32_e64 v21, v21, s6
	scratch_store_dword off, v21, off offset:1632 ; 4-byte Folded Spill
	s_mov_b32 s5, 0x108
	v_xor_b32_e64 v21, v19, s5
	v_add_u32_e64 v21, s1, v21
	v_mov_b32_e32 v22, v21
	scratch_store_dword off, v22, off offset:1628 ; 4-byte Folded Spill
	v_add_u32_e64 v21, v21, s6
	scratch_store_dword off, v21, off offset:1624 ; 4-byte Folded Spill
	s_mov_b32 s10, 0x210
	v_xor_b32_e64 v21, v19, s10
	v_add_u32_e64 v21, s1, v21
	v_mov_b32_e32 v22, v21
	scratch_store_dword off, v22, off offset:1620 ; 4-byte Folded Spill
	v_add_u32_e64 v21, v21, s6
	scratch_store_dword off, v21, off offset:1616 ; 4-byte Folded Spill
	s_mov_b32 s10, 0x318
	v_xor_b32_e64 v21, v19, s10
	v_add_u32_e64 v21, s1, v21
	v_mov_b32_e32 v22, v21
	scratch_store_dword off, v22, off offset:1612 ; 4-byte Folded Spill
	v_add_u32_e64 v21, v21, s6
	scratch_store_dword off, v21, off offset:1608 ; 4-byte Folded Spill
	s_mov_b32 s10, 0x840
	v_xor_b32_e64 v21, v19, s10
	v_add_u32_e64 v21, s1, v21
	v_mov_b32_e32 v22, v21
	scratch_store_dword off, v22, off offset:1604 ; 4-byte Folded Spill
	v_add_u32_e64 v21, v21, s6
	scratch_store_dword off, v21, off offset:1600 ; 4-byte Folded Spill
	s_mov_b32 s10, 0x948
	v_xor_b32_e64 v21, v19, s10
	v_add_u32_e64 v21, s1, v21
	v_mov_b32_e32 v22, v21
	scratch_store_dword off, v22, off offset:1596 ; 4-byte Folded Spill
	v_add_u32_e64 v21, v21, s6
	scratch_store_dword off, v21, off offset:1592 ; 4-byte Folded Spill
	s_mov_b32 s10, 0xa50
	v_xor_b32_e64 v21, v19, s10
	v_add_u32_e64 v21, s1, v21
	v_mov_b32_e32 v22, v21
	scratch_store_dword off, v22, off offset:1588 ; 4-byte Folded Spill
	v_add_u32_e64 v21, v21, s6
	scratch_store_dword off, v21, off offset:1584 ; 4-byte Folded Spill
	s_mov_b32 s10, 0xb58
	v_xor_b32_e64 v19, v19, s10
	v_add_u32_e64 v19, s1, v19
	v_mov_b32_e32 v21, v19
	scratch_store_dword off, v21, off offset:1580 ; 4-byte Folded Spill
	v_add_u32_e64 v19, v19, s6
	scratch_store_dword off, v19, off offset:1576 ; 4-byte Folded Spill
	s_mov_b32 s10, 60
	v_and_b32_e64 v19, v18, s10
	s_mov_b32 s10, 6
	v_lshlrev_b32_e64 v18, s10, v19
	s_mov_b32 s10, 24
	v_and_b32_e64 v20, v20, s10
	v_lshlrev_b32_e64 v19, s9, v19
	v_bitop3_b32 v18, v18, v19, v20 bitop3:0x36
	v_xor_b32_e64 v18, v18, s8
	v_add_u32_e64 v18, s1, v18
	v_mov_b32_e32 v19, v18
	scratch_store_dword off, v19, off offset:1572 ; 4-byte Folded Spill
	v_add_u32_e64 v19, v18, s6
	scratch_store_dword off, v19, off offset:1568 ; 4-byte Folded Spill
	s_mov_b32 s6, 0x80
	v_add_u32_e64 v19, v18, s6
	scratch_store_dword off, v19, off offset:1564 ; 4-byte Folded Spill
	s_mov_b32 s6, 0x1080
	v_add_u32_e64 v18, v18, s6
	scratch_store_dword off, v18, off offset:1560 ; 4-byte Folded Spill
	s_cselect_b32 s6, s4, s5
	v_mov_b32_e32 v18, s7
	v_bitop3_b32 v16, v16, s6, v18 bitop3:0x6c
	v_add_u32_e64 v16, s1, v16
	scratch_store_dword off, v16, off offset:1556 ; 4-byte Folded Spill
	v_mov_b32_e32 v16, s5
	v_mov_b32_e32 v18, s4
	v_cndmask_b32_e64 v16, v16, v18, s[2:3]
	v_xad_u32 v16, v16, v17, s1
	v_mov_b32_e32 v17, v16
	scratch_store_dword off, v17, off offset:1552 ; 4-byte Folded Spill
	s_mov_b32 s1, 0x200
	v_add_u32_e64 v16, v16, s1
	scratch_store_dword off, v16, off offset:1548 ; 4-byte Folded Spill
                                        ; kill: def $vgpr14_vgpr15 killed $vgpr14_vgpr15 killed $exec
                                        ; kill: def $vgpr12_vgpr13 killed $vgpr12_vgpr13 killed $exec
                                        ; kill: def $vgpr10_vgpr11 killed $vgpr10_vgpr11 killed $exec
                                        ; kill: def $vgpr9 killed $vgpr9 killed $exec
                                        ; kill: def $vgpr8 killed $vgpr8 killed $exec
                                        ; kill: def $vgpr6_vgpr7 killed $vgpr6_vgpr7 killed $exec
                                        ; kill: def $vgpr4_vgpr5 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr2_vgpr3 killed $vgpr2_vgpr3 killed $exec
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	scratch_store_dwordx2 off, v[14:15], off offset:1540 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[12:13], off offset:1532 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[10:11], off offset:1524 ; 8-byte Folded Spill
	s_waitcnt vmcnt(33)
	scratch_store_dword off, v9, off offset:1520 ; 4-byte Folded Spill
	scratch_store_dword off, v8, off offset:1516 ; 4-byte Folded Spill
	v_writelane_b32 v118, s0, 7
	s_or_saveexec_b64 s[44:45], -1
	scratch_store_dword off, v118, off      ; 4-byte Folded Spill
	s_mov_b64 exec, s[44:45]
	scratch_store_dwordx2 off, v[6:7], off offset:1508 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1500 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:1492 ; 8-byte Folded Spill
.Ltmp130:
	.loc	1 132 5 is_stmt 1               ; mha.py:132:5 @[ mha.py:811:25 ]
	scratch_store_dwordx2 off, v[0:1], off offset:1484 ; 8-byte Folded Spill
	s_branch .LBB0_22
.Ltmp131:
.LBB0_21:                               ; %Flow3
	.loc	1 0 5 is_stmt 0                 ; mha.py:0:5
	scratch_load_dword v9, off, off offset:1440 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, off offset:1472 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:1464 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:1456 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:1448 ; 8-byte Folded Reload
	scratch_load_dword v8, off, off offset:1444 ; 4-byte Folded Reload
                                        ; kill: def $vgpr9 killed $vgpr9 killed $exec
                                        ; kill: def $vgpr8 killed $vgpr8 killed $exec
                                        ; kill: def $vgpr6_vgpr7 killed $vgpr6_vgpr7 killed $exec
                                        ; kill: def $vgpr4_vgpr5 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr2_vgpr3 killed $vgpr2_vgpr3 killed $exec
                                        ; kill: def $vgpr0_vgpr1 killed $vgpr0_vgpr1 killed $exec
	s_waitcnt vmcnt(5)
	scratch_store_dword off, v9, off offset:1436 ; 4-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dword off, v8, off offset:1432 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:1424 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1416 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:1408 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], off offset:1400 ; 8-byte Folded Spill
	s_branch .LBB0_24
.LBB0_22:                               ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v116, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a37            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	scratch_load_dword v118, off, off       ; 4-byte Folded Reload
	s_mov_b64 exec, s[44:45]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s10, v118, 7
	v_readlane_b32 s1, v116, 49
	v_readlane_b32 s2, v117, 58
	v_readlane_b32 s3, v117, 59
	v_readlane_b32 s6, v117, 56
	v_readlane_b32 s7, v117, 57
	v_readlane_b32 s5, v116, 9
	v_readlane_b32 s8, v116, 45
	v_readlane_b32 s9, v116, 46
	v_readlane_b32 s14, v118, 0
	v_readlane_b32 s15, v118, 1
	v_readlane_b32 s12, v118, 2
	v_readlane_b32 s13, v118, 3
	v_readlane_b32 s22, v117, 60
	v_readlane_b32 s23, v117, 61
	v_readlane_b32 s20, v117, 62
	v_readlane_b32 s21, v117, 63
	scratch_load_dwordx2 v[10:11], off, off offset:1484 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[32:33], off, off offset:1492 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[24:25], off, off offset:1500 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[22:23], off, off offset:1508 ; 8-byte Folded Reload
	scratch_load_dword v58, off, off offset:1516 ; 4-byte Folded Reload
	scratch_load_dword v59, off, off offset:1520 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[16:17], off, off offset:1524 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[14:15], off, off offset:1532 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[12:13], off, off offset:1540 ; 8-byte Folded Reload
	scratch_load_dword v6, off, off offset:1548 ; 4-byte Folded Reload
	scratch_load_dword v4, off, off offset:1552 ; 4-byte Folded Reload
	scratch_load_dword v5, off, off offset:1556 ; 4-byte Folded Reload
	scratch_load_dword v20, off, off offset:1560 ; 4-byte Folded Reload
	scratch_load_dword v18, off, off offset:1564 ; 4-byte Folded Reload
	scratch_load_dword v2, off, off offset:1568 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:1572 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:1576 ; 4-byte Folded Reload
	scratch_load_dword v19, off, off offset:1580 ; 4-byte Folded Reload
	scratch_load_dword v28, off, off offset:1584 ; 4-byte Folded Reload
	scratch_load_dword v30, off, off offset:1588 ; 4-byte Folded Reload
	scratch_load_dword v34, off, off offset:1592 ; 4-byte Folded Reload
	scratch_load_dword v36, off, off offset:1596 ; 4-byte Folded Reload
	scratch_load_dword v38, off, off offset:1600 ; 4-byte Folded Reload
	scratch_load_dword v40, off, off offset:1604 ; 4-byte Folded Reload
	scratch_load_dword v42, off, off offset:1608 ; 4-byte Folded Reload
	scratch_load_dword v44, off, off offset:1612 ; 4-byte Folded Reload
	scratch_load_dword v46, off, off offset:1616 ; 4-byte Folded Reload
	scratch_load_dword v48, off, off offset:1620 ; 4-byte Folded Reload
	scratch_load_dword v50, off, off offset:1624 ; 4-byte Folded Reload
	scratch_load_dword v52, off, off offset:1628 ; 4-byte Folded Reload
	scratch_load_dword v54, off, off offset:1632 ; 4-byte Folded Reload
	scratch_load_dword v56, off, off offset:1636 ; 4-byte Folded Reload
	scratch_load_dword v26, off, off offset:1644 ; 4-byte Folded Reload
	scratch_load_dword v27, off, off offset:1648 ; 4-byte Folded Reload
	scratch_load_dword v29, off, off offset:1480 ; 4-byte Folded Reload
	scratch_load_dwordx4 v[80:83], off, off offset:1652 ; 16-byte Folded Reload
	scratch_load_dword v3, off, off offset:1684 ; 4-byte Folded Reload
	scratch_load_dword v8, off, off offset:1688 ; 4-byte Folded Reload
	scratch_load_dwordx4 v[84:87], off, off offset:1668 ; 16-byte Folded Reload
	v_accvgpr_read_b32 v31, a10             ;  Reload Reuse
	v_accvgpr_read_b32 v35, a9              ;  Reload Reuse
	s_waitcnt vmcnt(30)
	v_mov_b32_e32 v7, v12
	v_mov_b32_e32 v21, v14
	v_mov_b32_e32 v9, v16
.Ltmp132:
	.loc	1 136 24 is_stmt 1              ; mha.py:136:24 @[ mha.py:811:25 ]
	v_or_b32_e64 v35, s10, v35
.Ltmp133:
	.loc	1 33 16                         ; mha.py:33:16 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_cmp_lt_i32_e64 s[16:17], v35, s5
	s_mov_b32 s11, 1
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e64 v35, s11, v9
	s_mov_b32 s4, 0x80000000
	v_mov_b32_e32 v9, s4
	v_cndmask_b32_e64 v9, v9, v35, s[16:17]
	s_mov_b32 s19, s21
	s_mov_b32 s0, s23
	s_mov_b32 s18, s22
                                        ; kill: def $sgpr20 killed $sgpr20 def $sgpr20_sgpr21_sgpr22_sgpr23
	s_mov_b32 s21, s19
	s_mov_b32 s22, s18
	s_mov_b32 s23, s0
	s_mov_b32 s0, 0
	buffer_load_dword v9, v9, s[20:23], s0 offen
.Ltmp134:
	.loc	1 34 18 is_stmt 0               ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e64 v35, s11, v21
	v_mov_b32_e32 v21, s4
	v_cndmask_b32_e64 v21, v21, v35, s[16:17]
	buffer_load_dword v21, v21, s[20:23], s0 offen
.Ltmp135:
	.loc	1 31 18 is_stmt 1               ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	v_lshlrev_b32_e64 v35, s11, v7
	v_mov_b32_e32 v7, s4
	v_cndmask_b32_e64 v7, v7, v35, s[16:17]
	s_mov_b32 s16, s13
	s_mov_b32 s4, s15
	s_mov_b32 s11, s14
                                        ; kill: def $sgpr12 killed $sgpr12 def $sgpr12_sgpr13_sgpr14_sgpr15
	s_mov_b32 s13, s16
	s_mov_b32 s14, s11
	s_mov_b32 s15, s4
	buffer_load_dword v7, v7, s[12:15], s0 offen
	s_mov_b32 s4, 32
.Ltmp136:
	.loc	1 168 27                        ; mha.py:168:27 @[ mha.py:811:25 ]
	s_add_i32 s0, s10, s4
	s_cmp_eq_u32 s0, s1
	s_cselect_b64 s[12:13], -1, 0
	.loc	1 168 26 is_stmt 0              ; mha.py:168:26 @[ mha.py:811:25 ]
	s_and_b64 s[8:9], s[8:9], s[12:13]
	.loc	1 169 22 is_stmt 1              ; mha.py:169:22 @[ mha.py:811:25 ]
	v_or_b32_e64 v35, s10, v31
	s_mov_b32 s10, 16
	v_or_b32_e64 v31, v35, s10
	.loc	1 170 28                        ; mha.py:170:28 @[ mha.py:811:25 ]
	v_cmp_ge_i32_e64 s[12:13], v35, s5
	v_cmp_ge_i32_e64 s[10:11], v31, s5
	.loc	1 171 20                        ; mha.py:171:20 @[ mha.py:811:25 ]
	s_nop 0
	v_cndmask_b32_e64 v35, 0, 1, s[12:13]
	s_mov_b32 s5, 0
	v_mov_b32_e32 v31, s5
	v_cndmask_b32_e64 v31, v31, v35, s[8:9]
	v_and_b32_e64 v31, 1, v31
	v_cmp_eq_u32_e64 s[12:13], v31, 1
.Ltmp137:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:142:20 @[ mha.py:811:25 ] ]
	s_nop 1
	v_writelane_b32 v118, s12, 8
	s_nop 1
	v_writelane_b32 v118, s13, 9
	v_cndmask_b32_e64 v35, 0, 1, s[10:11]
	v_mov_b32_e32 v31, s5
	v_cndmask_b32_e64 v31, v31, v35, s[8:9]
	v_and_b32_e64 v31, 1, v31
	v_cmp_eq_u32_e64 s[8:9], v31, 1
	s_nop 1
	v_writelane_b32 v118, s8, 10
	s_nop 1
	v_writelane_b32 v118, s9, 11
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b32 v8, v21
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[76:79], v3
	s_mov_b32 s5, 0
.Ltmp138:
	.loc	1 181 18                        ; mha.py:181:18 @[ mha.py:811:25 ]
	v_writelane_b32 v118, s5, 12
	v_mov_b32_e32 v60, s5
	v_mov_b32_e32 v88, s5
	v_mov_b32_e32 v57, s5
	v_mov_b32_e32 v55, s5
	v_mov_b32_e32 v53, s5
	v_mov_b32_e32 v51, s5
	v_mov_b32_e32 v49, s5
	v_mov_b32_e32 v47, s5
	v_mov_b32_e32 v45, s5
	v_mov_b32_e32 v43, s5
	v_mov_b32_e32 v41, s5
	v_mov_b32_e32 v39, s5
	v_mov_b32_e32 v37, s5
	v_mov_b32_e32 v35, s5
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v21, s5
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61_vgpr62_vgpr63_vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71_vgpr72_vgpr73_vgpr74_vgpr75 killed $exec
	v_mov_b32_e32 v61, v88
	v_mov_b32_e32 v62, v57
	v_mov_b32_e32 v63, v55
	v_mov_b32_e32 v64, v53
	v_mov_b32_e32 v65, v51
	v_mov_b32_e32 v66, v49
	v_mov_b32_e32 v67, v47
	v_mov_b32_e32 v68, v45
	v_mov_b32_e32 v69, v43
	v_mov_b32_e32 v70, v41
	v_mov_b32_e32 v71, v39
	v_mov_b32_e32 v72, v37
	v_mov_b32_e32 v73, v35
	v_mov_b32_e32 v74, v31
	v_mov_b32_e32 v75, v21
	scratch_store_dwordx4 off, v[60:63], off offset:1804 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[64:67], off offset:1820 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[68:71], off offset:1836 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[72:75], off offset:1852 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_32x32x16_bf16 v[60:75], v[76:79], v[84:87], v[60:75]
.Ltmp139:
	.loc	1 34 18                         ; mha.py:34:18 @[ mha.py:140:13 @[ mha.py:811:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v8, v9
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[76:79], v3
.Ltmp140:
	.loc	1 182 14                        ; mha.py:182:14 @[ mha.py:811:25 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[60:75], v[76:79], v[80:83], v[60:75]
	s_nop 11
	scratch_store_dwordx4 off, v[60:63], off offset:1740 ; 16-byte Folded Spill
	s_nop 0
	scratch_store_dwordx4 off, v[64:67], off offset:1756 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[68:71], off offset:1772 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[72:75], off offset:1788 ; 16-byte Folded Spill
	v_mov_b32_e32 v8, v60
	v_mov_b32_e32 v55, v61
	v_mov_b32_e32 v53, v62
	v_mov_b32_e32 v49, v63
	v_mov_b32_e32 v51, v64
	v_mov_b32_e32 v47, v65
	v_mov_b32_e32 v41, v66
	v_mov_b32_e32 v45, v67
	v_mov_b32_e32 v43, v68
	v_mov_b32_e32 v9, v69
	v_mov_b32_e32 v39, v70
	v_mov_b32_e32 v37, v71
	v_mov_b32_e32 v21, v72
	v_mov_b32_e32 v35, v73
	v_mov_b32_e32 v31, v74
	v_mov_b32_e32 v3, v75
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	v_mul_f32_e64 v55, v29, v55
	v_mul_f32_e64 v8, v29, v8
	s_mov_b32 s10, 0xff800000
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_writelane_b32 v118, s10, 13
	v_mov_b32_e32 v57, s10
	v_cndmask_b32_e64 v55, v55, v57, s[12:13]
	v_mov_b32_e32 v57, s10
	v_cndmask_b32_e64 v60, v8, v57, s[12:13]
.Ltmp141:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v8, v60, v55
.Ltmp142:
                                        ; implicit-def: $sgpr14_sgpr15
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	s_mov_b32 s5, s15
	v_writelane_b32 v118, s5, 14
	v_mov_b32_e32 v57, s5
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v57
	v_mov_b32_e32 v76, v60
                                        ; kill: def $vgpr76 killed $vgpr76 def $vgpr76_vgpr77 killed $exec
	v_mov_b32_e32 v77, v55
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	v_mul_f32_e64 v49, v29, v49
	v_mul_f32_e64 v53, v29, v53
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_mov_b32_e32 v55, s10
	v_cndmask_b32_e64 v49, v49, v55, s[12:13]
	v_mov_b32_e32 v55, s10
	v_cndmask_b32_e64 v60, v53, v55, s[12:13]
.Ltmp143:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v8, v8, v60
.Ltmp144:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	v_mov_b32_e32 v53, s5
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v53
	v_mov_b32_e32 v74, v60
                                        ; kill: def $vgpr74 killed $vgpr74 def $vgpr74_vgpr75 killed $exec
	v_mov_b32_e32 v75, v49
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	v_mul_f32_e64 v47, v29, v47
	v_mul_f32_e64 v51, v29, v51
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_mov_b32_e32 v53, s10
	v_cndmask_b32_e64 v47, v47, v53, s[12:13]
	v_mov_b32_e32 v53, s10
	v_cndmask_b32_e64 v60, v51, v53, s[12:13]
.Ltmp145:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v49, v49, v60
	v_max_f32_e64 v49, v49, v47
	v_max_f32_e64 v8, v8, v49
.Ltmp146:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	v_mov_b32_e32 v49, s5
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v49
	v_mov_b32_e32 v72, v60
                                        ; kill: def $vgpr72 killed $vgpr72 def $vgpr72_vgpr73 killed $exec
	v_mov_b32_e32 v73, v47
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	v_mul_f32_e64 v45, v29, v45
	v_mul_f32_e64 v41, v29, v41
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_mov_b32_e32 v47, s10
	v_cndmask_b32_e64 v45, v45, v47, s[12:13]
	v_mov_b32_e32 v47, s10
	v_cndmask_b32_e64 v60, v41, v47, s[12:13]
.Ltmp147:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v41, v60, v45
.Ltmp148:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	v_mov_b32_e32 v47, s5
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v47
	v_mov_b32_e32 v70, v60
                                        ; kill: def $vgpr70 killed $vgpr70 def $vgpr70_vgpr71 killed $exec
	v_mov_b32_e32 v71, v45
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	v_mul_f32_e64 v9, v29, v9
	v_mul_f32_e64 v43, v29, v43
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_mov_b32_e32 v45, s10
	v_cndmask_b32_e64 v9, v9, v45, s[8:9]
	v_mov_b32_e32 v45, s10
	v_cndmask_b32_e64 v60, v43, v45, s[8:9]
.Ltmp149:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v41, v41, v60
	v_max_f32_e64 v8, v8, v41
.Ltmp150:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	v_mov_b32_e32 v41, s5
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v41
	v_mov_b32_e32 v68, v60
                                        ; kill: def $vgpr68 killed $vgpr68 def $vgpr68_vgpr69 killed $exec
	v_mov_b32_e32 v69, v9
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	v_mul_f32_e64 v37, v29, v37
	v_mul_f32_e64 v39, v29, v39
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_mov_b32_e32 v41, s10
	v_cndmask_b32_e64 v37, v37, v41, s[8:9]
	v_mov_b32_e32 v41, s10
	v_cndmask_b32_e64 v60, v39, v41, s[8:9]
.Ltmp151:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v9, v9, v60
	v_max_f32_e64 v9, v9, v37
.Ltmp152:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	v_mov_b32_e32 v39, s5
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v39
	v_mov_b32_e32 v66, v60
                                        ; kill: def $vgpr66 killed $vgpr66 def $vgpr66_vgpr67 killed $exec
	v_mov_b32_e32 v67, v37
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	v_mul_f32_e64 v35, v29, v35
	v_mul_f32_e64 v21, v29, v21
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_mov_b32_e32 v37, s10
	v_cndmask_b32_e64 v35, v35, v37, s[8:9]
	v_mov_b32_e32 v37, s10
	v_cndmask_b32_e64 v60, v21, v37, s[8:9]
.Ltmp153:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v21, v60, v35
.Ltmp154:
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	v_mov_b32_e32 v37, s5
                                        ; kill: def $vgpr60 killed $vgpr60 def $vgpr60_vgpr61 killed $exec
	v_mov_b32_e32 v61, v37
	v_mov_b32_e32 v64, v60
                                        ; kill: def $vgpr64 killed $vgpr64 def $vgpr64_vgpr65 killed $exec
	v_mov_b32_e32 v65, v35
	.loc	1 186 18                        ; mha.py:186:18 @[ mha.py:811:25 ]
	v_mul_f32_e64 v3, v29, v3
	v_mul_f32_e64 v29, v29, v31
	.loc	1 198 14                        ; mha.py:198:14 @[ mha.py:811:25 ]
	v_mov_b32_e32 v31, s10
	v_cndmask_b32_e64 v3, v3, v31, s[8:9]
	v_mov_b32_e32 v31, s10
	v_cndmask_b32_e64 v62, v29, v31, s[8:9]
.Ltmp155:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v21, v21, v62
	v_max_f32_e64 v9, v9, v21
	v_max_f32_e64 v9, v9, v3
	v_max_f32_e64 v9, v8, v9
.Ltmp156:
	.loc	3 191 16                        ; standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ]
	v_mov_b32_e32 v8, v9
	s_nop 1
	v_permlane32_swap_b32_e64 v8, v9
.Ltmp157:
	.loc	3 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mha.py:209:32 @[ mha.py:811:25 ] ] ]
	v_max_f32_e64 v9, v9, v9
	v_max_f32_e64 v8, v8, v8
	v_max_f32_e64 v9, v8, v9
.Ltmp158:
	.loc	1 209 16                        ; mha.py:209:16 @[ mha.py:811:25 ]
	v_max_f32_e64 v8, v59, v59
	v_max_f32_e64 v60, v8, v9
	v_mov_b32_e32 v9, v60
	.loc	1 212 26                        ; mha.py:212:26 @[ mha.py:811:25 ]
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v78, v60
	v_mov_b32_e32 v79, v8
	scratch_store_dwordx2 off, v[78:79], off offset:1732 ; 8-byte Folded Spill
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_add_f32 v[76:77], v[76:77], v[80:81] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_add_f32 v[74:75], v[74:75], v[80:81] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_add_f32 v[72:73], v[72:73], v[80:81] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_add_f32 v[70:71], v[70:71], v[80:81] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_add_f32 v[68:69], v[68:69], v[80:81] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_add_f32 v[66:67], v[66:67], v[80:81] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_add_f32 v[64:65], v[64:65], v[80:81] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mov_b32_e32 v8, s5
                                        ; kill: def $vgpr62 killed $vgpr62 def $vgpr62_vgpr63 killed $exec
	v_mov_b32_e32 v63, v8
                                        ; kill: def $vgpr62 killed $vgpr62 killed $vgpr62_vgpr63 killed $exec
                                        ; kill: def $vgpr62 killed $vgpr62 def $vgpr62_vgpr63 killed $exec
	v_mov_b32_e32 v63, v3
	v_pk_add_f32 v[62:63], v[62:63], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 212 13 is_stmt 0              ; mha.py:212:13 @[ mha.py:811:25 ]
	v_mov_b32_e32 v3, v76
	v_exp_f32_e64 v57, v3
	v_mov_b32_e32 v3, v77
	v_exp_f32_e64 v53, v3
	v_mov_b32_e32 v3, v74
	v_exp_f32_e64 v49, v3
	v_mov_b32_e32 v3, v75
	v_exp_f32_e64 v45, v3
	v_mov_b32_e32 v3, v72
	v_exp_f32_e64 v41, v3
	v_mov_b32_e32 v3, v73
	v_exp_f32_e64 v37, v3
	v_mov_b32_e32 v3, v70
	v_exp_f32_e64 v31, v3
	v_mov_b32_e32 v3, v71
	v_exp_f32_e64 v21, v3
	v_mov_b32_e32 v3, v68
	v_exp_f32_e64 v55, v3
	v_mov_b32_e32 v3, v69
	v_exp_f32_e64 v51, v3
	v_mov_b32_e32 v3, v66
	v_exp_f32_e64 v47, v3
	v_mov_b32_e32 v3, v67
	v_exp_f32_e64 v43, v3
	v_mov_b32_e32 v3, v64
	v_exp_f32_e64 v39, v3
	v_mov_b32_e32 v3, v65
	v_exp_f32_e64 v35, v3
	v_mov_b32_e32 v3, v62
	v_exp_f32_e64 v29, v3
	v_mov_b32_e32 v3, v63
	v_exp_f32_e64 v3, v3
.Ltmp159:
	.loc	3 293 12 is_stmt 1              ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v62, v57
	v_mov_b32_e32 v63, v8
                                        ; kill: def $vgpr62 killed $vgpr62 killed $vgpr62_vgpr63 killed $exec
                                        ; kill: def $vgpr62 killed $vgpr62 def $vgpr62_vgpr63 killed $exec
	v_mov_b32_e32 v63, v53
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v64, v49
	v_mov_b32_e32 v65, v8
	v_mov_b32_e32 v76, v64
                                        ; kill: def $vgpr76 killed $vgpr76 def $vgpr76_vgpr77 killed $exec
	v_mov_b32_e32 v77, v45
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v64, v41
	v_mov_b32_e32 v65, v8
	v_mov_b32_e32 v68, v64
                                        ; kill: def $vgpr68 killed $vgpr68 def $vgpr68_vgpr69 killed $exec
	v_mov_b32_e32 v69, v37
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v64, v31
	v_mov_b32_e32 v65, v8
	v_mov_b32_e32 v74, v64
                                        ; kill: def $vgpr74 killed $vgpr74 def $vgpr74_vgpr75 killed $exec
	v_mov_b32_e32 v75, v21
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v64, v55
	v_mov_b32_e32 v65, v8
                                        ; kill: def $vgpr64 killed $vgpr64 killed $vgpr64_vgpr65 killed $exec
                                        ; kill: def $vgpr64 killed $vgpr64 def $vgpr64_vgpr65 killed $exec
	v_mov_b32_e32 v65, v51
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v66, v47
	v_mov_b32_e32 v67, v8
	v_mov_b32_e32 v72, v66
                                        ; kill: def $vgpr72 killed $vgpr72 def $vgpr72_vgpr73 killed $exec
	v_mov_b32_e32 v73, v43
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v66, v39
	v_mov_b32_e32 v67, v8
                                        ; kill: def $vgpr66 killed $vgpr66 killed $vgpr66_vgpr67 killed $exec
                                        ; kill: def $vgpr66 killed $vgpr66 def $vgpr66_vgpr67 killed $exec
	v_mov_b32_e32 v67, v35
	v_mov_b32_e32 v8, s5
	v_mov_b32_e32 v70, v29
	v_mov_b32_e32 v71, v8
                                        ; kill: def $vgpr70 killed $vgpr70 killed $vgpr70_vgpr71 killed $exec
                                        ; kill: def $vgpr70 killed $vgpr70 def $vgpr70_vgpr71 killed $exec
	v_mov_b32_e32 v71, v3
	v_pk_add_f32 v[62:63], v[62:63], v[76:77]
	v_pk_add_f32 v[68:69], v[68:69], v[74:75]
	v_pk_add_f32 v[64:65], v[64:65], v[72:73]
	v_pk_add_f32 v[66:67], v[66:67], v[70:71]
	v_pk_add_f32 v[62:63], v[62:63], v[68:69]
	v_pk_add_f32 v[64:65], v[64:65], v[66:67]
	s_nop 0
	v_pk_add_f32 v[62:63], v[62:63], v[64:65]
.Ltmp160:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ] ]
	s_nop 0
	v_pk_add_f32 v[62:63], v[62:63], v[62:63] op_sel:[0,1] op_sel_hi:[1,0]
.Ltmp161:
	.loc	3 293 12                        ; standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ]
	s_nop 0
	v_mov_b32_e32 v61, v62
	v_mov_b32_e32 v8, v61
	s_nop 1
	v_permlane32_swap_b32_e64 v8, v61
.Ltmp162:
	.loc	3 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mha.py:220:16 @[ mha.py:811:25 ] ] ]
	v_add_f32_e64 v8, v8, v61
.Ltmp163:
	.loc	1 241 30                        ; mha.py:241:30 @[ mha.py:811:25 ]
	v_sub_f32_e64 v59, v59, v60
	.loc	1 241 17 is_stmt 0              ; mha.py:241:17 @[ mha.py:811:25 ]
	v_exp_f32_e64 v59, v59
	.loc	1 246 15 is_stmt 1              ; mha.py:246:15 @[ mha.py:811:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v27, v59
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[26:27], v26
	.loc	1 248 15                        ; mha.py:248:15 @[ mha.py:811:25 ]
	v_mul_f32_e64 v58, v58, v59
	v_add_f32_e64 v8, v8, v58
	.loc	1 259 26                        ; mha.py:259:26 @[ mha.py:811:25 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v57, v57, s5
	ds_write_b16 v56, v57
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v55, v55, s5
	ds_write_b16 v54, v55
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v53, v53, s5
	ds_write_b16 v52, v53
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v51, v51, s5
	ds_write_b16 v50, v51
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v49, v49, s5
	ds_write_b16 v48, v49
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v47, v47, s5
	ds_write_b16 v46, v47
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v45, v45, s5
	ds_write_b16 v44, v45
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v43, v43, s5
	ds_write_b16 v42, v43
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v41, v41, s5
	ds_write_b16 v40, v41
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v39, v39, s5
	ds_write_b16 v38, v39
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v37, v37, s5
	ds_write_b16 v36, v37
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v35, v35, s5
	ds_write_b16 v34, v35
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v31, v31, s5
	ds_write_b16 v30, v31
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v29, v29, s5
	ds_write_b16 v28, v29
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v21, v21, s5
	ds_write_b16 v19, v21
                                        ; implicit-def: $sgpr5
	v_cvt_pk_bf16_f32 v3, v3, s5
	ds_write_b16 v1, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[0:1], v0
	ds_read_b64_tr_b16 v[2:3], v2
	ds_read_b64_tr_b16 v[18:19], v18
	ds_read_b64_tr_b16 v[20:21], v20
.Ltmp164:
	.loc	1 31 18                         ; mha.py:31:18 @[ mha.py:150:17 @[ mha.py:811:25 ] ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(9)
	ds_write_b32 v5, v7
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[4:5], v4
	ds_read_b64_tr_b16 v[6:7], v6
.Ltmp165:
	.loc	1 259 19                        ; mha.py:259:19 @[ mha.py:811:25 ]
	v_mov_b32_e32 v28, v3
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v1
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1_vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v1, v30
	v_mov_b32_e32 v2, v29
	v_mov_b32_e32 v3, v28
	v_mov_b32_e32 v28, v21
	v_mov_b32_e32 v29, v20
	v_mov_b32_e32 v30, v19
                                        ; kill: def $vgpr18 killed $vgpr18 killed $vgpr18_vgpr19 killed $exec
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19_vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v19, v30
	v_mov_b32_e32 v20, v29
	v_mov_b32_e32 v21, v28
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v28, v7
	v_mov_b32_e32 v29, v6
	v_mov_b32_e32 v30, v5
                                        ; kill: def $vgpr4 killed $vgpr4 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v30
	v_mov_b32_e32 v6, v29
	v_mov_b32_e32 v7, v28
	v_mov_b32_e32 v28, v26
	v_mov_b32_e32 v34, v28
	v_mov_b32_e32 v35, v28
	v_mov_b32_e32 v36, v28
	v_mov_b32_e32 v37, v28
	v_mov_b32_e32 v29, v37
	v_mov_b32_e32 v30, v36
                                        ; kill: def $vgpr34_vgpr35 killed $vgpr34_vgpr35 killed $vgpr34_vgpr35_vgpr36_vgpr37 killed $exec
	v_pk_mul_f32 v[34:35], v[32:33], v[34:35]
	s_nop 0
	v_mov_b32_e32 v32, v35
	v_mov_b32_e32 v28, v34
                                        ; kill: def $vgpr30 killed $vgpr30 def $vgpr30_vgpr31 killed $exec
	v_mov_b32_e32 v31, v29
	v_pk_mul_f32 v[30:31], v[10:11], v[30:31]
	s_nop 0
	v_mov_b32_e32 v10, v31
	v_mov_b32_e32 v11, v30
                                        ; kill: def $vgpr28 killed $vgpr28 def $vgpr28_vgpr29_vgpr30_vgpr31 killed $exec
	v_mov_b32_e32 v29, v32
	v_mov_b32_e32 v30, v11
	v_mov_b32_e32 v31, v10
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[28:31], v[4:7], v[0:3], v[28:31]
	s_nop 7
	v_mov_b32_e32 v10, v29
	v_mov_b32_e32 v0, v28
	v_mov_b32_e32 v1, v31
	v_mov_b32_e32 v2, v30
	v_mov_b32_e32 v3, v27
	v_mov_b32_e32 v30, v3
	v_mov_b32_e32 v31, v3
	v_mov_b32_e32 v32, v3
	v_mov_b32_e32 v33, v3
	v_mov_b32_e32 v3, v33
	v_mov_b32_e32 v28, v32
	v_mov_b64_e32 v[26:27], v[30:31]
	v_pk_mul_f32 v[22:23], v[22:23], v[26:27]
	s_nop 0
	v_mov_b32_e32 v26, v23
                                        ; kill: def $vgpr22 killed $vgpr22 killed $vgpr22_vgpr23 killed $exec
                                        ; kill: def $vgpr28 killed $vgpr28 def $vgpr28_vgpr29 killed $exec
	v_mov_b32_e32 v29, v3
	v_pk_mul_f32 v[24:25], v[24:25], v[28:29]
	s_nop 0
	v_mov_b32_e32 v3, v25
	v_mov_b32_e32 v11, v24
                                        ; kill: def $vgpr22 killed $vgpr22 def $vgpr22_vgpr23_vgpr24_vgpr25 killed $exec
	v_mov_b32_e32 v23, v26
	v_mov_b32_e32 v24, v11
	v_mov_b32_e32 v25, v3
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[18:21], v[4:7], v[18:21], v[22:25]
	s_nop 7
	v_mov_b32_e32 v3, v19
	v_mov_b32_e32 v4, v18
	v_mov_b32_e32 v5, v21
	v_mov_b32_e32 v6, v20
	.loc	1 261 9                         ; mha.py:261:9 @[ mha.py:811:25 ]
	v_lshlrev_b64 v[16:17], s4, v[16:17]
	v_ashrrev_i64 v[16:17], s4, v[16:17]
	v_lshl_add_u64 v[20:21], v[16:17], 0, s[6:7]
	.loc	1 263 13                        ; mha.py:263:13 @[ mha.py:811:25 ]
	v_lshlrev_b64 v[14:15], s4, v[14:15]
	v_ashrrev_i64 v[14:15], s4, v[14:15]
	v_lshl_add_u64 v[22:23], v[14:15], 0, s[6:7]
	.loc	1 264 9                         ; mha.py:264:9 @[ mha.py:811:25 ]
	v_lshlrev_b64 v[12:13], s4, v[12:13]
	v_ashrrev_i64 v[12:13], s4, v[12:13]
	v_lshl_add_u64 v[24:25], v[12:13], 0, s[2:3]
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	s_cmp_ge_i32 s0, s1
	s_cselect_b64 s[2:3], -1, 0
.Ltmp166:
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	.loc	1 798 5                         ; mha.py:798:5
	v_mov_b32_e32 v7, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v10
.Ltmp167:
	.loc	1 132 5                         ; mha.py:132:5 @[ mha.py:811:25 ]
	s_mov_b64 s[4:5], -1
	s_xor_b64 s[2:3], s[2:3], s[4:5]
	s_and_b64 vcc, exec, s[2:3]
                                        ; kill: def $vgpr24_vgpr25 killed $vgpr24_vgpr25 killed $exec
                                        ; kill: def $vgpr22_vgpr23 killed $vgpr22_vgpr23 killed $exec
                                        ; kill: def $vgpr20_vgpr21 killed $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v19, v9
	v_mov_b32_e32 v18, v8
	v_mov_b64_e32 v[16:17], v[4:5]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[12:13], v[0:1]
	v_mov_b64_e32 v[10:11], v[2:3]
	scratch_store_dwordx2 off, v[24:25], off offset:1540 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[22:23], off offset:1532 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[20:21], off offset:1524 ; 8-byte Folded Spill
	scratch_store_dword off, v19, off offset:1520 ; 4-byte Folded Spill
	scratch_store_dword off, v18, off offset:1516 ; 4-byte Folded Spill
	v_writelane_b32 v118, s0, 7
	s_or_saveexec_b64 s[44:45], -1
	scratch_store_dword off, v118, off      ; 4-byte Folded Spill
	s_mov_b64 exec, s[44:45]
	scratch_store_dwordx2 off, v[16:17], off offset:1508 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[14:15], off offset:1500 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[12:13], off offset:1492 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[10:11], off offset:1484 ; 8-byte Folded Spill
	scratch_store_dword off, v9, off offset:1728 ; 4-byte Folded Spill
	scratch_store_dword off, v8, off offset:1724 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:1716 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1708 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:1700 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], off offset:1692 ; 8-byte Folded Spill
	s_cbranch_vccnz .LBB0_22
.Ltmp168:
; %bb.23:                               ; %Flow2
	.loc	1 798 5                         ; mha.py:798:5
	scratch_load_dwordx2 v[8:9], off, off offset:1692 ; 8-byte Folded Reload
	scratch_load_dword v0, off, off offset:1728 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:1724 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, off offset:1716 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:1708 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:1700 ; 8-byte Folded Reload
	s_waitcnt vmcnt(5)
	scratch_store_dwordx2 off, v[8:9], off offset:1472 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx2 off, v[6:7], off offset:1464 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1456 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:1448 ; 8-byte Folded Spill
	scratch_store_dword off, v1, off offset:1444 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:1440 ; 4-byte Folded Spill
	s_branch .LBB0_21
.LBB0_24:                               ; %.loopexit
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s10, v117, 34
	v_readlane_b32 s11, v117, 35
	v_readlane_b32 s12, v117, 36
	v_readlane_b32 s13, v117, 37
	v_readlane_b32 s6, v117, 39
	v_readlane_b32 s7, v117, 40
	v_readlane_b32 s14, v117, 32
	v_readlane_b32 s15, v117, 33
	v_readlane_b32 s0, v117, 25
	v_readlane_b32 s2, v117, 10
	v_readlane_b32 s3, v117, 19
	v_readlane_b32 s4, v117, 18
	s_or_saveexec_b64 s[44:45], -1
	scratch_load_dword v118, off, off       ; 4-byte Folded Reload
	s_mov_b64 exec, s[44:45]
	v_accvgpr_read_b32 v2, a2               ;  Reload Reuse
	v_accvgpr_read_b32 v3, a1               ;  Reload Reuse
	scratch_load_dword v0, off, off offset:1436 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:1432 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:1424 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:1416 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, off offset:1408 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[10:11], off, off offset:1400 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[10:11], off offset:1908 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[8:9], off offset:1900 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[6:7], off offset:1892 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], off offset:1884 ; 8-byte Folded Spill
	s_mov_b32 s5, 1.0
	.loc	1 861 15                        ; mha.py:861:15
	v_div_scale_f32 v4, s[8:9], v1, v1, s5
	v_rcp_f32_e64 v5, v4
	s_nop 0
	v_fma_f32 v6, -v4, v5, s5
	v_fmac_f32_e64 v5, v6, v5
	v_div_scale_f32 v7, vcc, s5, v1, s5
	v_mul_f32_e64 v6, v7, v5
	v_fma_f32 v8, -v4, v6, v7
	v_fmac_f32_e64 v6, v8, v5
	v_fma_f32 v4, -v4, v6, v7
	v_div_fmas_f32 v4, v4, v5, v6
	v_div_fixup_f32 v4, v4, v1, s5
	.loc	1 862 11                        ; mha.py:862:11
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s1, 31
	v_and_b32_e64 v3, v3, s1
	v_mov_b32_e32 v5, v3
	scratch_store_dword off, v5, off offset:1880 ; 4-byte Folded Spill
	s_mov_b32 s1, 8
	s_lshl_b32 s1, s3, s1
	s_mov_b32 s8, 0x100
	s_and_b32 s16, s1, s8
	s_mov_b32 s1, 1
	s_lshl_b32 s4, s4, s1
	s_mov_b32 s8, 4
	s_and_b32 s9, s4, s8
	s_mov_b32 s8, 0
	s_mov_b32 s4, 3
	v_mov_b32_e32 v5, s8
	v_lshl_add_u32 v3, v3, s4, v5
	v_add_u32_e64 v3, v3, s16
	v_add_u32_e64 v3, v3, s9
	ds_write_b32 v3, v4
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s9, 7
	s_lshl_b32 s3, s3, s9
	v_writelane_b32 v118, s3, 15
	v_mov_b32_e32 v3, s8
	v_lshl_add_u32 v2, v2, s4, v3
	v_add_u32_e64 v2, v2, s3
	ds_read_b64 v[2:3], v2
	.loc	1 884 21                        ; mha.py:884:21
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[2:3], off offset:1872 ; 8-byte Folded Spill
	s_sub_i32 s0, s0, s2
	v_writelane_b32 v118, s0, 16
	s_mov_b32 s2, 0x80
	s_add_i32 s0, s0, s2
	s_mov_b32 s2, 0x800000
	.loc	1 888 29                        ; mha.py:888:29
	v_cmp_lt_f32_e64 s[2:3], v1, s2
	s_mov_b32 s4, 0x42000000
	s_mov_b32 s8, 0
	v_mov_b32_e32 v2, s8
	v_mov_b32_e32 v3, s4
	v_cndmask_b32_e64 v2, v2, v3, s[2:3]
	s_mov_b32 s4, 0x4f800000
	v_mov_b32_e32 v3, s5
	v_mov_b32_e32 v4, s4
	v_cndmask_b32_e64 v3, v3, v4, s[2:3]
	v_mul_f32_e64 v1, v1, v3
	v_log_f32_e64 v1, v1
	s_nop 0
	v_sub_f32_e64 v1, v1, v2
	.loc	1 888 23 is_stmt 0              ; mha.py:888:23
	v_add_f32_e64 v0, v0, v1
	s_mov_b32 s2, 0x3f317218
	.loc	1 890 9 is_stmt 1               ; mha.py:890:9
	v_mul_f32_e64 v0, v0, s2
	.loc	1 900 13                        ; mha.py:900:13
	scratch_store_dword off, v0, off offset:1868 ; 4-byte Folded Spill
	s_mov_b32 s3, s14
	s_mov_b32 s2, s6
	s_mul_hi_u32 s4, s2, s3
	s_mov_b32 s9, 32
	s_lshr_b64 s[14:15], s[14:15], s9
	s_mov_b32 s5, s14
	s_mul_i32 s5, s2, s5
	s_add_i32 s4, s4, s5
	s_lshr_b64 s[6:7], s[6:7], s9
	s_mov_b32 s5, s6
	s_mul_i32 s5, s5, s3
	s_add_i32 s4, s4, s5
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr5
                                        ; kill: def $sgpr4 killed $sgpr4 def $sgpr4_sgpr5
	s_mov_b32 s5, s6
	s_lshl_b64 s[4:5], s[4:5], s9
	s_mul_i32 s2, s2, s3
	s_mov_b32 s8, 0
                                        ; kill: def $sgpr2 killed $sgpr2 def $sgpr2_sgpr3
	s_mov_b32 s3, s8
	s_or_b64 s[4:5], s[2:3], s[4:5]
	.loc	1 901 15                        ; mha.py:901:15
	s_mov_b32 s3, s12
	s_mov_b32 s2, s10
	s_mul_hi_u32 s6, s2, s3
	s_lshr_b64 s[12:13], s[12:13], s9
	s_mov_b32 s7, s12
	s_mul_i32 s7, s2, s7
	s_add_i32 s6, s6, s7
	s_lshr_b64 s[10:11], s[10:11], s9
	s_mov_b32 s7, s10
	s_mul_i32 s7, s7, s3
	s_add_i32 s6, s6, s7
                                        ; implicit-def: $sgpr10
                                        ; implicit-def: $sgpr7
                                        ; kill: def $sgpr6 killed $sgpr6 def $sgpr6_sgpr7
	s_mov_b32 s7, s10
	s_lshl_b64 s[6:7], s[6:7], s9
	s_mul_i32 s2, s2, s3
                                        ; kill: def $sgpr2 killed $sgpr2 def $sgpr2_sgpr3
	s_mov_b32 s3, s8
	s_or_b64 s[6:7], s[2:3], s[6:7]
	.loc	1 900 13                        ; mha.py:900:13
	s_mov_b32 s2, s4
	s_mov_b32 s3, s5
	s_mov_b32 s5, s6
	s_mov_b32 s4, s7
	s_add_u32 s2, s2, s5
	s_addc_u32 s4, s3, s4
                                        ; kill: def $sgpr2 killed $sgpr2 def $sgpr2_sgpr3
	s_mov_b32 s3, s4
	.loc	1 905 12                        ; mha.py:905:12
	v_writelane_b32 v118, s2, 17
	s_nop 1
	v_writelane_b32 v118, s3, 18
	s_cmp_lt_i32 s0, s1
	s_cselect_b64 s[2:3], -1, 0
	s_mov_b64 s[0:1], s[2:3]
	v_writelane_b32 v118, s0, 19
	s_nop 1
	v_writelane_b32 v118, s1, 20
	s_mov_b64 s[0:1], -1
	s_xor_b64 s[2:3], s[2:3], s[0:1]
	.loc	1 905 9 is_stmt 0               ; mha.py:905:9
	s_xor_b64 s[2:3], s[2:3], s[0:1]
	s_and_b64 vcc, exec, s[2:3]
	v_writelane_b32 v118, s0, 21
	s_nop 1
	v_writelane_b32 v118, s1, 22
	s_or_saveexec_b64 s[44:45], -1
	scratch_store_dword off, v118, off      ; 4-byte Folded Spill
	s_mov_b64 exec, s[44:45]
	s_cbranch_vccnz .LBB0_26
; %bb.25:
	.loc	1 0 9                           ; mha.py:0:9
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	scratch_load_dword v118, off, off       ; 4-byte Folded Reload
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s6, v117, 17
	v_readlane_b32 s0, v117, 15
	v_readlane_b32 s1, v117, 16
	s_waitcnt vmcnt(0)
	v_readlane_b32 s3, v118, 15
	v_readlane_b32 s8, v118, 17
	v_readlane_b32 s9, v118, 18
	v_readlane_b32 s2, v118, 16
	v_accvgpr_read_b32 v1, a8               ;  Reload Reuse
	v_accvgpr_read_b32 v0, a7               ;  Reload Reuse
	scratch_load_dword v3, off, off offset:1868 ; 4-byte Folded Reload
	scratch_load_dword v2, off, off offset:1880 ; 4-byte Folded Reload
	s_mov_b32 s4, 0
	.loc	1 906 44 is_stmt 1              ; mha.py:906:44
	s_sub_i32 s2, s4, s2
	.loc	1 907 24                        ; mha.py:907:24
	v_cmp_lt_i32_e64 s[10:11], v0, s2
	.loc	1 909 17                        ; mha.py:909:17
	s_mov_b32 s5, s8
	.loc	1 908 13                        ; mha.py:908:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s2, 0
	s_mov_b32 s8, 2
	v_mov_b32_e32 v4, s2
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v2, v2, s8, v4
	v_add_u32_e64 v2, v2, s3
	ds_write_b32 v2, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v2, s2
	v_lshl_add_u32 v0, v0, s8, v2
	ds_read_b32 v0, v0
	s_mov_b32 s2, 32
	s_lshr_b64 s[2:3], s[0:1], s2
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	s_mov_b32 s3, 0xffff
	s_and_b32 s12, s2, s3
	s_mov_b32 s7, 0x27000
	s_mov_b32 s9, 0x7ffffffe
                                        ; kill: def $sgpr0 killed $sgpr0 def $sgpr0_sgpr1_sgpr2_sgpr3
	s_mov_b32 s1, s12
	s_mov_b32 s2, s9
	s_mov_b32 s3, s7
	s_mov_b32 s7, 0x80
	s_and_b32 s6, s6, s7
	s_cmp_eq_u32 s6, s4
	s_cselect_b64 s[6:7], -1, 0
	s_and_b64 s[6:7], s[6:7], s[10:11]
	v_mov_b32_e32 v2, s8
	v_add_lshl_u32 v2, v1, s5, v2
	s_mov_b32 s5, 0x80000000
	v_mov_b32_e32 v1, s5
	v_cndmask_b32_e64 v1, v1, v2, s[6:7]
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v0, v1, s[0:3], s4 offen
	s_mov_b64 s[0:1], 0
	.loc	1 905 9                         ; mha.py:905:9
	v_writelane_b32 v118, s0, 21
	s_nop 1
	v_writelane_b32 v118, s1, 22
	s_or_saveexec_b64 s[44:45], -1
	scratch_store_dword off, v118, off      ; 4-byte Folded Spill
	s_mov_b64 exec, s[44:45]
.LBB0_26:                               ; %Flow
	.loc	1 0 9 is_stmt 0                 ; mha.py:0:9
	s_or_saveexec_b64 s[44:45], -1
	scratch_load_dword v118, off, off       ; 4-byte Folded Reload
	s_mov_b64 exec, s[44:45]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v118, 21
	v_readlane_b32 s1, v118, 22
	s_mov_b64 s[2:3], -1
	s_xor_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_28
; %bb.27:
	.loc	1 913 17 is_stmt 1              ; mha.py:913:17
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v117, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	scratch_load_dword v118, off, off       ; 4-byte Folded Reload
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s4, v117, 17
	v_readlane_b32 s0, v117, 15
	v_readlane_b32 s1, v117, 16
	s_waitcnt vmcnt(0)
	v_readlane_b32 s3, v118, 15
	v_readlane_b32 s6, v118, 17
	v_readlane_b32 s7, v118, 18
	v_accvgpr_read_b32 v1, a8               ;  Reload Reuse
	v_accvgpr_read_b32 v0, a7               ;  Reload Reuse
	scratch_load_dword v3, off, off offset:1868 ; 4-byte Folded Reload
	scratch_load_dword v2, off, off offset:1880 ; 4-byte Folded Reload
	s_mov_b32 s5, s6
	.loc	1 912 13                        ; mha.py:912:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s2, 0
	s_mov_b32 s8, 2
	v_mov_b32_e32 v4, s2
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v2, v2, s8, v4
	v_add_u32_e64 v2, v2, s3
	ds_write_b32 v2, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v2, s2
	v_lshl_add_u32 v0, v0, s8, v2
	ds_read_b32 v0, v0
	s_mov_b32 s2, 32
	s_lshr_b64 s[2:3], s[0:1], s2
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	s_mov_b32 s3, 0xffff
	s_and_b32 s9, s2, s3
	s_mov_b32 s6, 0x27000
	s_mov_b32 s7, 0x7ffffffe
                                        ; kill: def $sgpr0 killed $sgpr0 def $sgpr0_sgpr1_sgpr2_sgpr3
	s_mov_b32 s1, s9
	s_mov_b32 s2, s7
	s_mov_b32 s3, s6
	s_mov_b32 s6, 0x80
	s_and_b32 s6, s4, s6
	s_mov_b32 s4, 0
	s_cmp_eq_u32 s6, s4
	s_cselect_b64 s[6:7], -1, 0
	v_mov_b32_e32 v2, s8
	v_add_lshl_u32 v2, v1, s5, v2
	s_mov_b32 s5, 0x80000000
	v_mov_b32_e32 v1, s5
	v_cndmask_b32_e64 v1, v1, v2, s[6:7]
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v0, v1, s[0:3], s4 offen
.LBB0_28:
	.loc	1 862 11                        ; mha.py:862:11
	s_or_saveexec_b64 s[44:45], -1
	scratch_load_dword v117, off, off       ; 4-byte Folded Reload
	s_mov_b64 exec, s[44:45]
	s_or_saveexec_b64 s[44:45], -1
	v_accvgpr_read_b32 v118, a34            ;  Reload Reuse
	s_mov_b64 exec, s[44:45]
	v_readlane_b32 s0, v118, 3
	v_readlane_b32 s1, v118, 4
	v_readlane_b32 s3, v118, 12
	s_waitcnt vmcnt(0)
	v_readlane_b32 s6, v117, 19
	v_readlane_b32 s7, v117, 20
	v_readlane_b32 s4, v118, 13
	v_readlane_b32 s20, v118, 36
	v_readlane_b32 s21, v118, 37
	v_readlane_b32 s10, v118, 14
	v_readlane_b32 s14, v118, 39
	v_readlane_b32 s15, v118, 40
	v_readlane_b32 s2, v118, 10
	v_readlane_b32 s5, v118, 25
	v_readlane_b32 s8, v118, 19
	v_accvgpr_read_b32 v0, a1               ;  Reload Reuse
	v_accvgpr_read_b32 v2, a2               ;  Reload Reuse
	scratch_load_dwordx2 v[4:5], off, off offset:1908 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, off offset:1872 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, off offset:1900 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[12:13], off, off offset:1892 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[10:11], off, off offset:1884 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[10:11], v[10:11], v[8:9] op_sel:[0,1]
	v_pk_mul_f32 v[12:13], v[12:13], v[8:9] op_sel:[0,1]
	v_pk_mul_f32 v[6:7], v[6:7], v[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[4:5], v[8:9] op_sel_hi:[1,0]
	.loc	1 380 34                        ; mha.py:380:34
	s_mov_b32 s9, 4
	v_mov_b32_e32 v1, s9
	v_lshl_or_b32 v1, s8, v1, v2
	.loc	1 380 14 is_stmt 0              ; mha.py:380:14
	v_or_b32_e64 v5, s5, v1
	s_mov_b32 s5, 64
	v_or_b32_e64 v2, v5, s5
	.loc	1 679 18 is_stmt 1              ; mha.py:679:18
	v_cmp_lt_i32_e64 s[12:13], v2, s2
	v_cmp_lt_i32_e64 s[8:9], v5, s2
	.loc	1 586 11                        ; mha.py:586:11
	s_mov_b32 s2, 2
	v_lshrrev_b32_e64 v0, s2, v0
	s_mov_b32 s2, 12
	v_and_b32_e64 v8, v0, s2
	v_mov_b32_e32 v1, 0
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v1
	.loc	1 414 21                        ; mha.py:414:21
	s_mov_b32 s16, 0
	s_mov_b32 s18, s4
	s_mov_b32 s19, s16
	.loc	1 413 21                        ; mha.py:413:21
	s_mov_b32 s22, s10
	s_mov_b32 s23, s16
	.loc	1 918 9                         ; mha.py:918:9
	s_mov_b32 s5, s14
	s_mul_hi_u32 s11, s5, s10
	s_mov_b32 s2, 32
	s_lshr_b64 s[22:23], s[22:23], s2
	s_mov_b32 s17, s22
	s_mul_i32 s17, s5, s17
	s_add_i32 s11, s11, s17
	s_lshr_b64 s[14:15], s[14:15], s2
                                        ; kill: def $sgpr14 killed $sgpr14 killed $sgpr14_sgpr15
	s_mul_i32 s14, s14, s10
	s_add_i32 s14, s11, s14
                                        ; implicit-def: $sgpr11
                                        ; implicit-def: $sgpr15
                                        ; kill: def $sgpr14 killed $sgpr14 def $sgpr14_sgpr15
	s_mov_b32 s15, s11
	s_lshl_b64 s[14:15], s[14:15], s2
	s_mul_i32 s10, s5, s10
                                        ; kill: def $sgpr10 killed $sgpr10 def $sgpr10_sgpr11
	s_mov_b32 s11, s16
	s_or_b64 s[10:11], s[10:11], s[14:15]
	.loc	1 919 11                        ; mha.py:919:11
	s_mov_b32 s5, s20
	s_mul_hi_u32 s14, s4, s5
	s_lshr_b64 s[20:21], s[20:21], s2
	s_mov_b32 s15, s20
	s_mul_i32 s15, s4, s15
	s_add_i32 s14, s14, s15
	s_lshr_b64 s[18:19], s[18:19], s2
	s_mov_b32 s15, s18
	s_mul_i32 s15, s15, s5
	s_add_i32 s14, s14, s15
                                        ; implicit-def: $sgpr17
                                        ; implicit-def: $sgpr15
                                        ; kill: def $sgpr14 killed $sgpr14 def $sgpr14_sgpr15
	s_mov_b32 s15, s17
	s_lshl_b64 s[14:15], s[14:15], s2
	s_mul_i32 s4, s4, s5
                                        ; kill: def $sgpr4 killed $sgpr4 def $sgpr4_sgpr5
	s_mov_b32 s5, s16
	s_or_b64 s[14:15], s[4:5], s[14:15]
	.loc	1 918 9                         ; mha.py:918:9
	s_mov_b32 s4, s10
	s_mov_b32 s5, s11
	s_mov_b32 s11, s14
	s_mov_b32 s10, s15
	s_add_u32 s4, s4, s11
	s_addc_u32 s10, s5, s10
                                        ; kill: def $sgpr4 killed $sgpr4 def $sgpr4_sgpr5
	s_mov_b32 s5, s10
	.loc	1 925 5                         ; mha.py:925:5
	v_cndmask_b32_e64 v0, 0, 1, s[8:9]
	s_mov_b32 s8, 1
	v_mov_b32_e32 v3, s8
	v_cndmask_b32_e64 v0, v0, v3, s[6:7]
	v_and_b32_e64 v0, 1, v0
	v_cmp_eq_u32_e64 s[10:11], v0, 1
	v_cndmask_b32_e64 v0, 0, 1, s[12:13]
	v_mov_b32_e32 v3, s8
	v_cndmask_b32_e64 v0, v0, v3, s[6:7]
	v_and_b32_e64 v0, 1, v0
	v_cmp_eq_u32_e64 s[6:7], v0, 1
	.loc	1 929 10                        ; mha.py:929:10
	v_mov_b32_e32 v3, v15
	v_mov_b32_e32 v0, v14
	v_cvt_pk_bf16_f32 v4, v0, v3
	v_mov_b32_e32 v3, v7
	v_mov_b32_e32 v0, v6
	v_cvt_pk_bf16_f32 v6, v0, v3
	v_mov_b32_e32 v3, v13
	v_mov_b32_e32 v0, v12
	v_cvt_pk_bf16_f32 v0, v0, v3
	v_mov_b32_e32 v7, v11
	v_mov_b32_e32 v3, v10
	v_cvt_pk_bf16_f32 v3, v3, v7
	.loc	1 930 14                        ; mha.py:930:14
	v_mad_u64_u32 v[10:11], s[8:9], v5, s3, 0
	v_mov_b32_e32 v12, v10
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v1
	v_mov_b32_e32 v5, v13
	v_mov_b32_e32 v10, v11
                                        ; implicit-def: $sgpr8
                                        ; implicit-def: $sgpr9
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, s8
	v_lshlrev_b64 v[10:11], s2, v[10:11]
	v_mov_b32_e32 v7, v11
	v_or_b32_e64 v5, v5, v7
	v_mov_b32_e32 v7, v12
                                        ; kill: def $vgpr10 killed $vgpr10 killed $vgpr10_vgpr11 killed $exec
	v_or_b32_e64 v10, v7, v10
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v5
	v_mad_u64_u32 v[12:13], s[8:9], v2, s3, 0
	v_mov_b32_e32 v14, v12
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v1
	v_mov_b32_e32 v1, v15
	v_mov_b32_e32 v12, v13
                                        ; implicit-def: $sgpr3
                                        ; implicit-def: $sgpr8
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, s3
	v_lshlrev_b64 v[12:13], s2, v[12:13]
	v_mov_b32_e32 v2, v13
	v_or_b32_e64 v1, v1, v2
	v_mov_b32_e32 v2, v14
	v_mov_b32_e32 v5, v12
	v_or_b32_e64 v12, v2, v5
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v1
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[8:9]
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[12:13]
	v_mov_b32_e32 v1, v10
	v_mov_b32_e32 v2, v8
	.loc	1 930 5 is_stmt 0               ; mha.py:930:5
	s_lshr_b64 s[2:3], s[0:1], s2
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	s_mov_b32 s3, 0xffff
	s_and_b32 s8, s2, s3
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	s_mov_b32 s4, 0x27000
	s_mov_b32 s5, 0x7ffffffe
                                        ; kill: def $sgpr0 killed $sgpr0 def $sgpr0_sgpr1_sgpr2_sgpr3
	s_mov_b32 s1, s8
	s_mov_b32 s2, s5
	s_mov_b32 s3, s4
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v6
	s_mov_b32 s8, 1
	v_lshlrev_b32_e64 v6, s8, v1
	s_mov_b32 s5, 0x80000000
	v_mov_b32_e32 v1, s5
	v_cndmask_b32_e64 v1, v1, v6, s[10:11]
	s_mov_b32 s4, 0
	buffer_store_dwordx2 v[4:5], v1, s[0:3], s4 offen
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	v_lshlrev_b32_e64 v3, s8, v2
	v_mov_b32_e32 v2, s5
	v_cndmask_b32_e64 v2, v2, v3, s[6:7]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], s4 offen
	.loc	1 297 1 is_stmt 1               ; mha.py:297:1
	s_endpgm
.Ltmp169:
.Lfunc_end0:
	.size	_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0, .Lfunc_end0-_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 1920
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 248
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 120
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
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_vgpr, 119
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_agpr, 128
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.numbered_sgpr, 46
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.num_named_barrier, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.private_seg_size, 1920
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.uses_vcc, 1
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.uses_flat_scratch, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_dyn_sized_stack, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_recursion, 0
	.set .L_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 28788
; TotalNumSgprs: 52
; NumVgprs: 119
; NumAgprs: 128
; TotalNumVgprs: 248
; ScratchSize: 1920
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 30
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 248
; AccumOffset: 120
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 29
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
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
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
	.byte	9                               ; Abbreviation Code
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x134 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x10e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	375                             ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x56:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp4                          ; DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	510                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x6b:0x65 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	746                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x78:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	140                             ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x84:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	142                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x90:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	150                             ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x9c:0x19 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	209                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xa8:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.byte	191                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0xb5:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	220                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0xc1:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges7                 ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0xd0:0x6d DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges8                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	811                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xdd:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges9                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	140                             ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xe9:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges10                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	142                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xf5:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges11                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	150                             ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x101:0x19 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges12                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	209                             ; DW_AT_call_line
	.byte	32                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x10d:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges13                ; DW_AT_ranges
	.byte	3                               ; DW_AT_call_file
	.byte	191                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	9                               ; Abbrev [9] 0x11a:0x22 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp159                        ; DW_AT_low_pc
	.long	.Ltmp163-.Ltmp159               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	220                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	8                               ; Abbrev [8] 0x12e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges14                ; DW_AT_ranges
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
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp125-.Lfunc_begin0
	.quad	.Ltmp126-.Lfunc_begin0
	.quad	.Ltmp127-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
	.quad	.Ltmp124-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges8:
	.quad	.Ltmp128-.Lfunc_begin0
	.quad	.Ltmp129-.Lfunc_begin0
	.quad	.Ltmp130-.Lfunc_begin0
	.quad	.Ltmp131-.Lfunc_begin0
	.quad	.Ltmp132-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges9:
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp139-.Lfunc_begin0
	.quad	.Ltmp140-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges10:
	.quad	.Ltmp134-.Lfunc_begin0
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp137-.Lfunc_begin0
	.quad	.Ltmp138-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges11:
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Ltmp136-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges12:
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
	.quad	.Ltmp147-.Lfunc_begin0
	.quad	.Ltmp148-.Lfunc_begin0
	.quad	.Ltmp149-.Lfunc_begin0
	.quad	.Ltmp150-.Lfunc_begin0
	.quad	.Ltmp151-.Lfunc_begin0
	.quad	.Ltmp152-.Lfunc_begin0
	.quad	.Ltmp153-.Lfunc_begin0
	.quad	.Ltmp154-.Lfunc_begin0
	.quad	.Ltmp155-.Lfunc_begin0
	.quad	.Ltmp158-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges13:
	.quad	.Ltmp141-.Lfunc_begin0
	.quad	.Ltmp142-.Lfunc_begin0
	.quad	.Ltmp143-.Lfunc_begin0
	.quad	.Ltmp144-.Lfunc_begin0
	.quad	.Ltmp145-.Lfunc_begin0
	.quad	.Ltmp146-.Lfunc_begin0
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
.Ldebug_ranges14:
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
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
  - .agpr_count:     128
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
    .private_segment_fixed_size: 1920
    .sgpr_count:     52
    .sgpr_spill_count: 171
    .symbol:         _attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     248
    .vgpr_spill_count: 752
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
