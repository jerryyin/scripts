	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 5
	.text
	.globl	_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16 ; -- Begin function _mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16
	.p2align	8
	.type	_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16,@function
_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16: ; @_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16
.Lfunc_begin0:
	.file	1 "/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/attention" "mla.py"
	.loc	1 1500 0                        ; mla.py:1500:0
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp0:
	.loc	1 1291 17 prologue_end          ; mla.py:1291:17 @[ mla.py:1332:9 @[ mla.py:2029:9 ] ]
	s_load_b512 s[4:19], s[0:1], 0x0 nv
.Ltmp1:
	.loc	1 1591 27                       ; mla.py:1591:27
	s_mov_b32 s2, ttmp9
	s_ashr_i32 s3, ttmp9, 31
	s_lshr_b32 s33, ttmp7, 16
	.loc	1 1598 23                       ; mla.py:1598:23
	s_lshl_b64 s[20:21], s[2:3], 2
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[16:17], s[16:17], s[20:21]
	.loc	1 1598 15 is_stmt 0             ; mla.py:1598:15
	s_load_b32 s25, s[16:17], 0x0
.Ltmp2:
	.loc	1 1444 13 is_stmt 1             ; mla.py:1444:13 @[ mla.py:1602:25 ]
	s_wait_kmcnt 0x0
	s_add_co_i32 s16, s25, 0xff
	.loc	1 1444 12 is_stmt 0             ; mla.py:1444:12 @[ mla.py:1602:25 ]
	s_ashr_i32 s17, s16, 31
	s_lshr_b32 s17, s17, 24
	s_add_co_i32 s16, s16, s17
	s_ashr_i32 s24, s16, 8
.Ltmp3:
	.loc	1 1604 8 is_stmt 1              ; mla.py:1604:8
	s_mul_i32 s34, s24, s33
	s_lshl_b32 s16, s34, 6
	s_cmp_lt_i32 s16, s25
	s_cbranch_scc0 .LBB0_48
; %bb.1:
.Ltmp4:
	.loc	1 1291 17                       ; mla.py:1291:17 @[ mla.py:1332:9 @[ mla.py:2029:9 ] ]
	s_load_b128 s[28:31], s[0:1], 0x70 nv
	s_bfe_u32 s27, ttmp8, 0x50019
	s_and_b32 s77, ttmp7, 0xffff
.Ltmp5:
	.loc	1 1648 21                       ; mla.py:1648:21
	s_and_b32 s74, s27, 1
	.loc	1 1666 9                        ; mla.py:1666:9
	s_lshl_b32 s76, s77, 4
	.loc	1 1679 9                        ; mla.py:1679:9
	v_dual_mov_b32 v2, 0 :: v_dual_bitop2_b32 v1, 31, v0 bitop3:0x40
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	v_mov_b32_e32 v8, 0
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v19 /*v275*/, v2 :: v_dual_lshlrev_b32 v18 /*v274*/, 4, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v9, 0 :: v_dual_lshlrev_b32 v12, 3, v1
	.loc	1 1591 27                       ; mla.py:1591:27
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[16:17], s[28:29], s[20:21]
	.loc	1 1589 25                       ; mla.py:1589:25
	s_sub_co_i32 s26, 1, s30
	.loc	1 1591 19                       ; mla.py:1591:19
	s_load_b32 s28, s[16:17], 0x0
.Ltmp6:
	.loc	1 1291 17                       ; mla.py:1291:17 @[ mla.py:1332:9 @[ mla.py:2029:9 ] ]
	s_load_b128 s[20:23], s[0:1], 0x40 nv
.Ltmp7:
	.loc	1 1589 25                       ; mla.py:1589:25
	s_mul_i32 s26, s26, ttmp9
	.loc	1 1666 9                        ; mla.py:1666:9
	s_wait_xcnt 0x0
	s_or_b32 s16, s74, s76
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mov_b32 s17, 0
	.loc	1 1664 27                       ; mla.py:1664:27
	s_wait_kmcnt 0x0
	s_add_co_i32 s28, s28, s26
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mul_u64 s[36:37], s[22:23], s[16:17]
	.loc	1 1671 9                        ; mla.py:1671:9
	s_ashr_i32 s29, s28, 31
	.loc	1 1674 25                       ; mla.py:1674:25
	s_cmp_lt_i32 s26, s30
	.loc	1 1671 9                        ; mla.py:1671:9
	s_mul_u64 s[20:21], s[20:21], s[28:29]
	.loc	1 1674 25                       ; mla.py:1674:25
	s_cselect_b32 s75, -1, 0
	.loc	1 1675 25                       ; mla.py:1675:25
	s_cmp_lt_i32 s16, 16
	s_cselect_b32 s17, -1, 0
	.loc	1 1679 9                        ; mla.py:1679:9
	s_lshl_b64 s[20:21], s[20:21], 1
	s_lshl_b64 s[36:37], s[36:37], 1
	s_add_nc_u64 s[10:11], s[10:11], s[20:21]
	.loc	1 1680 14                       ; mla.py:1680:14
	s_and_b32 s17, s17, s75
	.loc	1 1679 9                        ; mla.py:1679:9
	s_add_nc_u64 s[20:21], s[10:11], s[36:37]
	.loc	1 1678 19                       ; mla.py:1678:19
	v_cndmask_b32_e64 v62, 0, 1, s17
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	.loc	1 1679 9                        ; mla.py:1679:9
	v_add_nc_u64_e32 v[10:11], s[20:21], v[18:19] /*v[274:275]*/
	.loc	1 1678 19                       ; mla.py:1678:19
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_vccnz .LBB0_3
; %bb.2:
	global_load_b128 v[6:9], v[10:11], off
.LBB0_3:
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v5, 0
	s_cbranch_vccnz .LBB0_5
; %bb.4:
	global_load_b128 v[2:5], v[10:11], off offset:512
.LBB0_5:
	.loc	1 1666 9                        ; mla.py:1666:9
	s_or_b32 s20, s16, 2
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mov_b32 s21, 0
	.loc	1 1679 9                        ; mla.py:1679:9
	v_dual_mov_b32 v67, 0 :: v_dual_lshlrev_b32 v66, 1, v12
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mul_u64 s[20:21], s[22:23], s[20:21]
	.loc	1 1678 19                       ; mla.py:1678:19
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	.loc	1 1679 9                        ; mla.py:1679:9
	s_lshl_b64 s[20:21], s[20:21], 1
	s_wait_xcnt 0x0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v11, 0
	s_add_nc_u64 s[20:21], s[10:11], s[20:21]
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	v_add_nc_u64_e32 v[18:19], s[20:21], v[66:67]
	.loc	1 1678 19                       ; mla.py:1678:19
	s_cbranch_vccnz .LBB0_7
; %bb.6:
	global_load_b128 v[10:13], v[18:19], off
.LBB0_7:
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v16, 0
	v_mov_b32_e32 v17, 0
	s_cbranch_vccnz .LBB0_9
; %bb.8:
	global_load_b128 v[14:17], v[18:19], off offset:512
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v67, v14
.LBB0_9:
	.loc	1 1666 9                        ; mla.py:1666:9
	s_or_b32 s20, s16, 4
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mov_b32 s21, 0
	.loc	1 1679 9                        ; mla.py:1679:9
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, v66
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mul_u64 s[20:21], s[22:23], s[20:21]
	.loc	1 1678 19                       ; mla.py:1678:19
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	.loc	1 1679 9                        ; mla.py:1679:9
	s_lshl_b64 s[20:21], s[20:21], 1
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	s_add_nc_u64 s[20:21], s[10:11], s[20:21]
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	v_add_nc_u64_e32 v[26:27], s[20:21], v[68:69]
	.loc	1 1678 19                       ; mla.py:1678:19
	s_cbranch_vccnz .LBB0_11
; %bb.10:
	global_load_b128 v[18:21], v[26:27], off
.LBB0_11:
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	v_mov_b32_e32 v25, 0
	s_cbranch_vccnz .LBB0_13
; %bb.12:
	global_load_b128 v[22:25], v[26:27], off offset:512
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v69, v22
.LBB0_13:
	.loc	1 1666 9                        ; mla.py:1666:9
	s_or_b32 s20, s16, 6
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mov_b32 s21, 0
	.loc	1 1679 9                        ; mla.py:1679:9
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v70, v66
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mul_u64 s[20:21], s[22:23], s[20:21]
	.loc	1 1678 19                       ; mla.py:1678:19
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	.loc	1 1679 9                        ; mla.py:1679:9
	s_lshl_b64 s[20:21], s[20:21], 1
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	s_add_nc_u64 s[20:21], s[10:11], s[20:21]
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_add_nc_u64_e32 v[34:35], s[20:21], v[70:71]
	.loc	1 1678 19                       ; mla.py:1678:19
	s_cbranch_vccnz .LBB0_15
; %bb.14:
	global_load_b128 v[26:29], v[34:35], off
.LBB0_15:
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v32, 0
	v_mov_b32_e32 v33, 0
	s_cbranch_vccnz .LBB0_17
; %bb.16:
	global_load_b128 v[30:33], v[34:35], off offset:512
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v71, v30
.LBB0_17:
	.loc	1 1666 9                        ; mla.py:1666:9
	s_or_b32 s20, s16, 8
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mov_b32 s21, 0
	.loc	1 1679 9                        ; mla.py:1679:9
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v72, v66
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mul_u64 s[20:21], s[22:23], s[20:21]
	.loc	1 1678 19                       ; mla.py:1678:19
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	.loc	1 1679 9                        ; mla.py:1679:9
	s_lshl_b64 s[20:21], s[20:21], 1
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	s_add_nc_u64 s[20:21], s[10:11], s[20:21]
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0
	v_add_nc_u64_e32 v[42:43], s[20:21], v[72:73]
	.loc	1 1678 19                       ; mla.py:1678:19
	s_cbranch_vccnz .LBB0_19
; %bb.18:
	global_load_b128 v[34:37], v[42:43], off
.LBB0_19:
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v40, 0
	v_mov_b32_e32 v41, 0
	s_cbranch_vccnz .LBB0_21
; %bb.20:
	global_load_b128 v[38:41], v[42:43], off offset:512
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v73, v38
.LBB0_21:
	.loc	1 1666 9                        ; mla.py:1666:9
	s_or_b32 s20, s16, 10
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mov_b32 s21, 0
	.loc	1 1679 9                        ; mla.py:1679:9
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, v66
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mul_u64 s[20:21], s[22:23], s[20:21]
	.loc	1 1678 19                       ; mla.py:1678:19
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	.loc	1 1679 9                        ; mla.py:1679:9
	s_lshl_b64 s[20:21], s[20:21], 1
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	s_add_nc_u64 s[20:21], s[10:11], s[20:21]
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_add_nc_u64_e32 v[50:51], s[20:21], v[74:75]
	.loc	1 1678 19                       ; mla.py:1678:19
	s_cbranch_vccnz .LBB0_23
; %bb.22:
	global_load_b128 v[42:45], v[50:51], off
.LBB0_23:
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	v_mov_b32_e32 v49, 0
	s_cbranch_vccnz .LBB0_25
; %bb.24:
	global_load_b128 v[46:49], v[50:51], off offset:512
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v75, v46
.LBB0_25:
	.loc	1 1666 9                        ; mla.py:1666:9
	s_or_b32 s20, s16, 12
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mov_b32 s21, 0
	.loc	1 1679 9                        ; mla.py:1679:9
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, v66
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mul_u64 s[20:21], s[22:23], s[20:21]
	.loc	1 1678 19                       ; mla.py:1678:19
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	.loc	1 1679 9                        ; mla.py:1679:9
	s_lshl_b64 s[20:21], s[20:21], 1
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	s_add_nc_u64 s[20:21], s[10:11], s[20:21]
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_add_nc_u64_e32 v[58:59], s[20:21], v[76:77]
	.loc	1 1678 19                       ; mla.py:1678:19
	s_cbranch_vccnz .LBB0_27
; %bb.26:
	global_load_b128 v[50:53], v[58:59], off
.LBB0_27:
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v56, 0
	v_mov_b32_e32 v57, 0
	s_cbranch_vccnz .LBB0_29
; %bb.28:
	global_load_b128 v[54:57], v[58:59], off offset:512
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v77, v54
.LBB0_29:
	.loc	1 1666 9                        ; mla.py:1666:9
	s_or_b32 s16, s16, 14
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mov_b32 s17, 0
	.loc	1 1679 9                        ; mla.py:1679:9
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v78, v66
	.loc	1 1672 11                       ; mla.py:1672:11
	s_mul_u64 s[16:17], s[22:23], s[16:17]
	.loc	1 1678 19                       ; mla.py:1678:19
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	.loc	1 1679 9                        ; mla.py:1679:9
	s_lshl_b64 s[16:17], s[16:17], 1
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	s_add_nc_u64 s[16:17], s[10:11], s[16:17]
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	v_add_nc_u64_e32 v[80:81], s[16:17], v[78:79]
	.loc	1 0 0 is_stmt 0                 ; mla.py:0
	s_lshl_b32 s16, s27, 5
	.loc	1 1678 19 is_stmt 1             ; mla.py:1678:19
	s_cbranch_vccnz .LBB0_31
; %bb.30:
	global_load_b128 v[58:61], v[80:81], off
.LBB0_31:
	v_cmp_ne_u32_e32 vcc_lo, 1, v62
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v64, 0
	v_mov_b32_e32 v65, 0
	.loc	1 0 0 is_stmt 0                 ; mla.py:0
	s_and_b32 s17, s16, 32
	.loc	1 1678 19                       ; mla.py:1678:19
	s_cbranch_vccnz .LBB0_33
; %bb.32:
	global_load_b128 v[62:65], v[80:81], off offset:512
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v79, v62
.LBB0_33:
	.loc	1 1683 5 is_stmt 1              ; mla.py:1683:5
	s_lshl_b32 s20, s17, 5
	s_lshr_b32 s17, s17, 1
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	.loc	1 1684 14                       ; mla.py:1684:14
	v_and_b32_e32 v21 /*v277*/, 15, v0
	s_set_vgpr_msb 0x4014                   ;  msbs: dst=0 src0=0 src1=1 src2=1
	.loc	1 1683 5                        ; mla.py:1683:5
	v_add3_u32 v66, s17, s20, v18 /*v274*/
	.loc	1 1678 19                       ; mla.py:1678:19
	v_dual_mov_b32 v22, v69 :: v_dual_mov_b32 v14, v67
	v_dual_mov_b32 v54, v77 :: v_dual_mov_b32 v46, v75
	v_dual_mov_b32 v38, v73 :: v_dual_mov_b32 v30, v71
	.loc	1 1683 5                        ; mla.py:1683:5
	v_dual_mov_b32 v62, v79 :: v_dual_lshlrev_b32 v142, 4, v21 /*v277*/
	s_set_vgpr_msb 0x1440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	s_wait_loadcnt 0x0
	ds_store_b128 v66, v[6:9]
	ds_store_b128 v66, v[2:5] offset:512
	ds_store_b128 v66, v[10:13] offset:2080
	ds_store_b128 v66, v[14:17] offset:2592
	ds_store_b128 v66, v[18:21] offset:4160
	ds_store_b128 v66, v[22:25] offset:4672
	ds_store_b128 v66, v[26:29] offset:6240
	ds_store_b128 v66, v[30:33] offset:6752
	ds_store_b128 v66, v[34:37] offset:8320
	ds_store_b128 v66, v[38:41] offset:8832
	ds_store_b128 v66, v[42:45] offset:10400
	ds_store_b128 v66, v[46:49] offset:10912
	ds_store_b128 v66, v[50:53] offset:12480
	ds_store_b128 v66, v[54:57] offset:12992
	ds_store_b128 v66, v[58:61] offset:14560
	ds_store_b128 v66, v[62:65] offset:15072
	.loc	1 1684 14                       ; mla.py:1684:14
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v19 /*v275*/, 16, v0
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_lshl_add_u32 v2, v21 /*v277*/, 10, 0
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
	.loc	1 1648 21                       ; mla.py:1648:21
	v_or_b32_e32 v20 /*v276*/, s16, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 1657 21                       ; mla.py:1657:21
	v_dual_lshlrev_b32 v130, 3, v0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v136, 0
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	.loc	1 1684 14                       ; mla.py:1684:14
	v_add3_u32 v6, v2, v19 /*v275*/, v142
	.loc	1 1654 21                       ; mla.py:1654:21
	v_dual_lshrrev_b32 v1, 3, v20 /*v276*/ :: v_dual_mov_b32 v137, 0
	.loc	1 1691 9                        ; mla.py:1691:9
	v_and_or_b32 v140, v1, 7, s76
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 1704 9                        ; mla.py:1704:9
	v_dual_mov_b32 v130, 0 :: v_dual_bitop2_b32 v1, 56, v130 bitop3:0x40
	.loc	1 1700 25                       ; mla.py:1700:25
	v_cmp_gt_i32_e32 vcc_lo, 16, v140
	v_lshlrev_b32_e32 v138, 1, v1
	.loc	1 1705 14                       ; mla.py:1705:14
	s_and_b32 s16, vcc_lo, s75
	.loc	1 1684 14                       ; mla.py:1684:14
	s_barrier_wait -1
	ds_load_b128 v[122:125], v6
	ds_load_b128 v[126:129], v6 offset:32
	ds_load_b128 v[114:117], v6 offset:64
	ds_load_b128 v[118:121], v6 offset:96
	ds_load_b128 v[106:109], v6 offset:128
	ds_load_b128 v[110:113], v6 offset:160
	ds_load_b128 v[98:101], v6 offset:192
	ds_load_b128 v[102:105], v6 offset:224
	ds_load_b128 v[90:93], v6 offset:256
	ds_load_b128 v[94:97], v6 offset:288
	ds_load_b128 v[82:85], v6 offset:320
	ds_load_b128 v[86:89], v6 offset:352
	ds_load_b128 v[74:77], v6 offset:384
	ds_load_b128 v[78:81], v6 offset:416
	ds_load_b128 v[66:69], v6 offset:448
	ds_load_b128 v[70:73], v6 offset:480
	ds_load_b128 v[58:61], v6 offset:512
	ds_load_b128 v[62:65], v6 offset:544
	ds_load_b128 v[50:53], v6 offset:576
	ds_load_b128 v[54:57], v6 offset:608
	ds_load_b128 v[42:45], v6 offset:640
	ds_load_b128 v[46:49], v6 offset:672
	ds_load_b128 v[34:37], v6 offset:704
	ds_load_b128 v[38:41], v6 offset:736
	ds_load_b128 v[26:29], v6 offset:768
	ds_load_b128 v[30:33], v6 offset:800
	ds_load_b128 v[18:21], v6 offset:832
	ds_load_b128 v[22:25], v6 offset:864
	ds_load_b128 v[10:13], v6 offset:896
	ds_load_b128 v[14:17], v6 offset:928
	ds_load_b128 v[2:5], v6 offset:960
	ds_load_b128 v[6:9], v6 offset:992
	.loc	1 1703 19                       ; mla.py:1703:19
	s_and_saveexec_b32 s17, s16
	s_cbranch_execz .LBB0_35
; %bb.34:
	.loc	1 1697 11                       ; mla.py:1697:11
	v_mov_b32_e32 v141, 0
	v_mul_u64_e32 v[132:133], s[22:23], v[140:141]
	.loc	1 1704 9                        ; mla.py:1704:9
	v_mov_b32_e32 v139, v141
	v_lshl_add_u64 v[132:133], v[132:133], 1, s[10:11]
	v_add_nc_u64_e32 v[132:133], v[132:133], v[138:139]
	.loc	1 1703 19                       ; mla.py:1703:19
	global_load_b128 v[134:137], v[132:133], off offset:1024
.LBB0_35:
	.loc	1 0 19 is_stmt 0                ; mla.py:0:19
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s17
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_mov_b32_e32 v133, 0
	.loc	1 1703 19                       ; mla.py:1703:19
	s_and_saveexec_b32 s17, s16
	s_cbranch_execz .LBB0_37
; %bb.36:
	.loc	1 1697 11 is_stmt 1             ; mla.py:1697:11
	v_dual_mov_b32 v131, 0 :: v_dual_bitop2_b32 v130, 8, v140 bitop3:0x54
	v_mul_u64_e32 v[132:133], s[22:23], v[130:131]
	.loc	1 1704 9                        ; mla.py:1704:9
	v_mov_b32_e32 v139, v131
	v_lshl_add_u64 v[132:133], v[132:133], 1, s[10:11]
	v_add_nc_u64_e32 v[130:131], v[132:133], v[138:139]
	.loc	1 1703 19                       ; mla.py:1703:19
	global_load_b128 v[130:133], v[130:131], off offset:1024
.LBB0_37:
	.loc	1 0 19 is_stmt 0                ; mla.py:0:19
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s17
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	.loc	1 1708 5 is_stmt 1              ; mla.py:1708:5
	v_dual_lshlrev_b32 v1, 4, v20 /*v276*/ :: v_dual_lshlrev_b32 v138, 1, v20 /*v276*/
	.loc	1 1815 9                        ; mla.py:1815:9
	s_sub_co_i32 s10, s26, s30
.Ltmp8:
	.loc	1 742 24                        ; mla.py:742:24 @[ mla.py:1822:23 ]
	s_add_co_i32 s16, s34, s24
.Ltmp9:
	.loc	1 1815 9                        ; mla.py:1815:9
	s_add_co_i32 s29, s10, s25
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 1708 5                        ; mla.py:1708:5
	v_and_b32_e32 v139, 0x3f0, v1
	v_and_b32_e32 v138, 0x70, v138
	.loc	1 1815 9                        ; mla.py:1815:9
	s_add_co_i32 s29, s29, 1
	.loc	1 1626 51                       ; mla.py:1626:51
	s_mul_u64 s[2:3], s[18:19], s[2:3]
	.loc	1 1820 26                       ; mla.py:1820:26
	s_min_i32 s10, s29, s25
.Ltmp10:
	.loc	1 852 38                        ; mla.py:852:38 @[ mla.py:1863:33 ]
	s_ashr_i32 s35, s34, 31
.Ltmp11:
	.loc	1 1708 5                        ; mla.py:1708:5
	v_add3_u32 v138, 0, v139, v138
.Ltmp12:
	.loc	1 740 22                        ; mla.py:740:22 @[ mla.py:1822:23 ]
	s_add_co_i32 s10, s10, 63
.Ltmp13:
	.loc	1 1626 32                       ; mla.py:1626:32
	s_lshl_b64 s[2:3], s[2:3], 2
.Ltmp14:
	.loc	1 740 21                        ; mla.py:740:21 @[ mla.py:1822:23 ]
	s_ashr_i32 s11, s10, 31
.Ltmp15:
	.loc	1 1708 5                        ; mla.py:1708:5
	s_wait_loadcnt 0x0
	ds_store_b128 v138, v[134:137] offset:16624
	ds_store_b128 v138, v[130:133] offset:17776
	.loc	1 1709 14                       ; mla.py:1709:14
	s_wait_dscnt 0x0
	s_barrier_signal -1
.Ltmp16:
	.loc	1 740 21                        ; mla.py:740:21 @[ mla.py:1822:23 ]
	s_lshr_b32 s11, s11, 26
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp17:
	.loc	1 1709 14                       ; mla.py:1709:14
	v_lshl_add_u32 v130, v21 /*v277*/, 7, 0
.Ltmp18:
	.loc	1 740 21                        ; mla.py:740:21 @[ mla.py:1822:23 ]
	s_add_co_i32 s10, s10, s11
.Ltmp19:
	.loc	1 846 32                        ; mla.py:846:32 @[ mla.py:1866:38 ]
	s_not_b32 s11, s34
.Ltmp20:
	.loc	1 740 21                        ; mla.py:740:21 @[ mla.py:1822:23 ]
	s_ashr_i32 s10, s10, 6
.Ltmp21:
	.loc	1 1626 32                       ; mla.py:1626:32
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]
.Ltmp22:
	.loc	1 742 20                        ; mla.py:742:20 @[ mla.py:1822:23 ]
	s_min_i32 s24, s16, s10
.Ltmp23:
	.loc	1 1709 14                       ; mla.py:1709:14
	v_add3_u32 v134, v19 /*v275*/, v130, v142
.Ltmp24:
	.loc	1 846 32                        ; mla.py:846:32 @[ mla.py:1866:38 ]
	s_add_co_i32 s30, s24, s11
.Ltmp25:
	.loc	1 852 38                        ; mla.py:852:38 @[ mla.py:1863:33 ]
	s_lshl_b64 s[10:11], s[34:35], 2
.Ltmp26:
	.loc	1 846 18                        ; mla.py:846:18 @[ mla.py:1866:38 ]
	s_min_i32 s16, s30, 1
.Ltmp27:
	.loc	1 852 38                        ; mla.py:852:38 @[ mla.py:1863:33 ]
	s_add_nc_u64 s[10:11], s[2:3], s[10:11]
.Ltmp28:
	.loc	1 847 38                        ; mla.py:847:38 @[ mla.py:1866:38 ]
	s_ashr_i32 s17, s16, 31
.Ltmp29:
	.loc	1 1291 17                       ; mla.py:1291:17 @[ mla.py:1332:9 @[ mla.py:2029:9 ] ]
	s_load_b32 s14, s[0:1], 0x64 nv
.Ltmp30:
	.loc	1 847 38                        ; mla.py:847:38 @[ mla.py:1866:38 ]
	s_lshl_b64 s[2:3], s[16:17], 2
.Ltmp31:
	.loc	1 1708 5                        ; mla.py:1708:5
	s_mov_b32 s23, 0
.Ltmp32:
	.loc	1 847 38                        ; mla.py:847:38 @[ mla.py:1866:38 ]
	s_wait_xcnt 0x0
	s_add_nc_u64 s[0:1], s[10:11], s[2:3]
.Ltmp33:
	.loc	1 705 22                        ; mla.py:705:22 @[ mla.py:1822:23 ]
	s_add_nc_u64 s[68:69], s[12:13], 0x10000
.Ltmp34:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_mov_b32 s15, s23
	s_mov_b32 s2, s23
	s_mov_b32 s16, 0x10000
.Ltmp35:
	.loc	1 704 27                        ; mla.py:704:27 @[ mla.py:1822:23 ]
	s_bitset1_b32 s69, 31
.Ltmp36:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1871:5 ]
	s_mov_b32 s71, s23
	s_set_vgpr_msb 0x144                    ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_or_b32_e32 v23 /*v279*/, 0x200, v18 /*v274*/
	v_or_b32_e32 v24 /*v280*/, 0x400, v18 /*v274*/
	v_or_b32_e32 v25 /*v281*/, 0x600, v18 /*v274*/
	v_or_b32_e32 v26 /*v282*/, 0x800, v18 /*v274*/
	v_or_b32_e32 v27 /*v283*/, 0xa00, v18 /*v274*/
	v_or_b32_e32 v28 /*v284*/, 0xc00, v18 /*v274*/
	v_or_b32_e32 v29 /*v285*/, 0xe00, v18 /*v274*/
	v_or_b32_e32 v30 /*v286*/, 0x1000, v18 /*v274*/
	v_or_b32_e32 v31 /*v287*/, 0x1200, v18 /*v274*/
	v_or_b32_e32 v32 /*v288*/, 0x1400, v18 /*v274*/
.Ltmp37:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_wait_kmcnt 0x0
	s_lshl_b64 s[18:19], s[14:15], 1
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp38:
	.loc	1 1709 14                       ; mla.py:1709:14
	s_barrier_wait -1
	ds_load_b128 v[138:141], v134 offset:16624
	ds_load_b128 v[142:145], v134 offset:16656
	ds_load_b128 v[130:133], v134 offset:16688
	ds_load_b128 v[134:137], v134 offset:16720
.Ltmp39:
	.loc	1 852 30                        ; mla.py:852:30 @[ mla.py:1863:33 ]
	s_clause 0x1
	s_load_b32 s17, s[10:11], 0x0
	s_load_b32 s25, s[0:1], 0x0
.Ltmp40:
	.loc	1 694 28                        ; mla.py:694:28 @[ mla.py:1822:23 ]
	s_wait_xcnt 0x0
	s_or_b32 s1, s13, 0x80000000
	s_mov_b32 s0, 1
.Ltmp41:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_mov_b32 s13, s1
.Ltmp42:
	.loc	1 694 28                        ; mla.py:694:28 @[ mla.py:1822:23 ]
	s_ashr_i32 s1, s14, 31
.Ltmp43:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_mov_b32 s20, s0
.Ltmp44:
	.loc	1 694 28                        ; mla.py:694:28 @[ mla.py:1822:23 ]
	s_and_b32 s35, s1, 0xffff
.Ltmp45:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_and_b32 s3, s1, 0x1fffe
	s_mov_b32 s21, s14
	s_or_b64 s[72:73], s[2:3], s[18:19]
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_or_b32_e32 v33 /*v289*/, 0x1600, v18 /*v274*/
	v_or_b32_e32 v34 /*v290*/, 0x1800, v18 /*v274*/
	v_or_b32_e32 v35 /*v291*/, 0x1a00, v18 /*v274*/
	v_or_b32_e32 v36 /*v292*/, 0x1c00, v18 /*v274*/
	v_or_b32_e32 v37 /*v293*/, 0x1e00, v18 /*v274*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_and_b32_e32 v38 /*v294*/, 0x270, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_lshlrev_b32_e32 v0, 5, v0
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_lshlrev_b32_e32 v39 /*v295*/, 3, v19 /*v275*/
.Ltmp46:
	.loc	1 1035 18                       ; mla.py:1035:18 @[ mla.py:1869:19 ]
	s_wait_kmcnt 0x0
	s_add_co_i32 s26, s17, s77
.Ltmp47:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_sub_co_i32 s1, s31, s26
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_ashr_i32 s27, s26, 31
	s_max_i32 s1, s1, 0
	s_cmp_gt_i32 s26, -1
	s_mul_u64 s[52:53], s[72:73], s[26:27]
	s_cselect_b32 s15, s1, 0
	s_lshl_b32 s17, s74, 30
.Ltmp48:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1871:5 ]
	s_lshl_b32 s19, s74, 27
.Ltmp49:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_lshl_b32 s18, s15, 16
	s_lshr_b32 s15, s15, 16
	s_lshl_b32 s22, s74, 15
	s_add_nc_u64 s[2:3], s[52:53], s[12:13]
	s_sub_co_i32 s17, 0x80000000, s17
.Ltmp50:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1871:5 ]
	s_sub_co_i32 s26, 0x10000000, s19
.Ltmp51:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_or_b32 s19, s15, 2.0
	s_mov_b64 s[42:43], s[22:23]
	s_mov_b32 s1, s22
	s_add_nc_u64 s[2:3], s[2:3], s[22:23]
	s_mov_b64 s[40:41], s[20:21]
	s_mov_b64 s[38:39], s[18:19]
	s_mov_b64 s[36:37], s[16:17]
	s_mov_b32 s42, s35
.Ltmp52:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1871:5 ]
	s_lshl_b32 s70, s74, 12
	s_bitset1_b32 s15, 27
	s_mov_b64 s[50:51], s[22:23]
	s_mov_b64 s[48:49], s[20:21]
	s_mov_b64 s[46:47], s[18:19]
	s_mov_b64 s[44:45], s[16:17]
	s_mov_b32 s45, s26
	s_mov_b32 s46, s18
	s_mov_b32 s47, s15
	s_mov_b32 s49, s14
	s_mov_b32 s50, s35
.Ltmp53:
	.loc	1 1873 36                       ; mla.py:1873:36
	s_add_co_i32 s15, s24, -1
.Ltmp54:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1870:5 ]
	s_barrier_wait -1
	tensor_load_to_lds s[0:3], s[36:43]
.Ltmp55:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1871:5 ]
	s_add_nc_u64 s[2:3], s[52:53], s[68:69]
	s_add_co_i32 s1, s70, 0x20000
	s_add_nc_u64 s[2:3], s[2:3], s[70:71]
.Ltmp56:
	.loc	1 1873 5                        ; mla.py:1873:5
	s_cmp_lt_i32 s34, s15
.Ltmp57:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1871:5 ]
	tensor_load_to_lds s[0:3], s[44:51]
	s_mov_b32 s0, -1
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp58:
	.loc	1 1873 5                        ; mla.py:1873:5
	s_cbranch_scc1 .LBB0_40
; %bb.38:                               ; %.._crit_edge_crit_edge
	.loc	1 0 5 is_stmt 0                 ; mla.py:0:5
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
.Ltmp59:
	.loc	1 1008 16 is_stmt 1             ; mla.py:1008:16 @[ mla.py:1951:19 ]
	v_or_b32_e32 v44 /*v300*/, 0x200, v18 /*v274*/
	v_or_b32_e32 v45 /*v301*/, 0x400, v18 /*v274*/
	v_or_b32_e32 v46 /*v302*/, 0x600, v18 /*v274*/
	v_or_b32_e32 v47 /*v303*/, 0x800, v18 /*v274*/
	v_or_b32_e32 v48 /*v304*/, 0xa00, v18 /*v274*/
	v_or_b32_e32 v49 /*v305*/, 0xc00, v18 /*v274*/
	v_or_b32_e32 v50 /*v306*/, 0xe00, v18 /*v274*/
	v_or_b32_e32 v51 /*v307*/, 0x1000, v18 /*v274*/
	v_or_b32_e32 v52 /*v308*/, 0x1200, v18 /*v274*/
	v_or_b32_e32 v53 /*v309*/, 0x1400, v18 /*v274*/
	v_or_b32_e32 v54 /*v310*/, 0x1600, v18 /*v274*/
	v_or_b32_e32 v55 /*v311*/, 0x1800, v18 /*v274*/
	v_or_b32_e32 v56 /*v312*/, 0x1a00, v18 /*v274*/
	v_or_b32_e32 v42 /*v298*/, 0x1c00, v18 /*v274*/
	v_or_b32_e32 v43 /*v299*/, 0x1e00, v18 /*v274*/
	s_set_vgpr_msb 0x4440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp60:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	v_and_b32_e32 v40 /*v296*/, 0x270, v1
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v1, 0x100, v0
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_lshlrev_b32_e32 v41 /*v297*/, 3, v19 /*v275*/
	s_set_vgpr_msb 0x4400                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_41
.Ltmp61:
.LBB0_39:
	.loc	1 0 16 is_stmt 0                ; mla.py:0:16
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v0, 0xff800000
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v22 /*v278*/, 1.0 :: v_dual_mov_b32 v1 /*v257*/, v153
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v152, v153 :: v_dual_mov_b32 v151, v153
	v_dual_mov_b32 v150, v153 :: v_dual_mov_b32 v149, v153
	v_dual_mov_b32 v148, v153 :: v_dual_mov_b32 v147, v153
	v_dual_mov_b32 v146, v153 :: v_dual_mov_b32 v161, v153
	v_dual_mov_b32 v160, v153 :: v_dual_mov_b32 v159, v153
	v_dual_mov_b32 v158, v153 :: v_dual_mov_b32 v157, v153
	v_dual_mov_b32 v156, v153 :: v_dual_mov_b32 v155, v153
	v_dual_mov_b32 v154, v153 :: v_dual_mov_b32 v169, v153
	v_dual_mov_b32 v168, v153 :: v_dual_mov_b32 v167, v153
	v_dual_mov_b32 v166, v153 :: v_dual_mov_b32 v165, v153
	v_dual_mov_b32 v164, v153 :: v_dual_mov_b32 v163, v153
	v_dual_mov_b32 v162, v153 :: v_dual_mov_b32 v177, v153
	v_dual_mov_b32 v176, v153 :: v_dual_mov_b32 v175, v153
	v_dual_mov_b32 v174, v153 :: v_dual_mov_b32 v173, v153
	v_dual_mov_b32 v172, v153 :: v_dual_mov_b32 v171, v153
	v_dual_mov_b32 v170, v153 :: v_dual_mov_b32 v185, v153
	v_dual_mov_b32 v184, v153 :: v_dual_mov_b32 v183, v153
	v_dual_mov_b32 v182, v153 :: v_dual_mov_b32 v181, v153
	v_dual_mov_b32 v180, v153 :: v_dual_mov_b32 v179, v153
	v_dual_mov_b32 v178, v153 :: v_dual_mov_b32 v193, v153
	v_dual_mov_b32 v192, v153 :: v_dual_mov_b32 v191, v153
	v_dual_mov_b32 v190, v153 :: v_dual_mov_b32 v189, v153
	v_dual_mov_b32 v188, v153 :: v_dual_mov_b32 v187, v153
	v_dual_mov_b32 v186, v153 :: v_dual_mov_b32 v201, v153
	v_dual_mov_b32 v200, v153 :: v_dual_mov_b32 v199, v153
	v_dual_mov_b32 v198, v153 :: v_dual_mov_b32 v197, v153
	v_dual_mov_b32 v196, v153 :: v_dual_mov_b32 v195, v153
	v_dual_mov_b32 v194, v153 :: v_dual_mov_b32 v209, v153
	v_dual_mov_b32 v208, v153 :: v_dual_mov_b32 v207, v153
	v_dual_mov_b32 v206, v153 :: v_dual_mov_b32 v205, v153
	v_dual_mov_b32 v204, v153 :: v_dual_mov_b32 v203, v153
	v_dual_mov_b32 v202, v153 :: v_dual_mov_b32 v217, v153
	v_dual_mov_b32 v216, v153 :: v_dual_mov_b32 v215, v153
	v_dual_mov_b32 v214, v153 :: v_dual_mov_b32 v213, v153
	v_dual_mov_b32 v212, v153 :: v_dual_mov_b32 v211, v153
	v_dual_mov_b32 v210, v153 :: v_dual_mov_b32 v225, v153
	v_dual_mov_b32 v224, v153 :: v_dual_mov_b32 v223, v153
	v_dual_mov_b32 v222, v153 :: v_dual_mov_b32 v221, v153
	v_dual_mov_b32 v220, v153 :: v_dual_mov_b32 v219, v153
	v_dual_mov_b32 v218, v153 :: v_dual_mov_b32 v233, v153
	v_dual_mov_b32 v232, v153 :: v_dual_mov_b32 v231, v153
	v_dual_mov_b32 v230, v153 :: v_dual_mov_b32 v229, v153
	v_dual_mov_b32 v228, v153 :: v_dual_mov_b32 v227, v153
	v_dual_mov_b32 v226, v153 :: v_dual_mov_b32 v241, v153
	v_dual_mov_b32 v240, v153 :: v_dual_mov_b32 v239, v153
	v_dual_mov_b32 v238, v153 :: v_dual_mov_b32 v237, v153
	v_dual_mov_b32 v236, v153 :: v_dual_mov_b32 v235, v153
	v_dual_mov_b32 v234, v153 :: v_dual_mov_b32 v249, v153
	v_dual_mov_b32 v248, v153 :: v_dual_mov_b32 v247, v153
	v_dual_mov_b32 v246, v153 :: v_dual_mov_b32 v245, v153
	v_dual_mov_b32 v244, v153 :: v_dual_mov_b32 v243, v153
	v_dual_mov_b32 v242, v153 :: v_dual_mov_b32 v255, v153
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v0 /*v256*/, v153 :: v_dual_mov_b32 v9 /*v265*/, v153
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_mov_b32 v254, v153 :: v_dual_mov_b32 v253, v153
	v_dual_mov_b32 v252, v153 :: v_dual_mov_b32 v251, v153
	v_mov_b32_e32 v250, v153
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v8 /*v264*/, v153 :: v_dual_mov_b32 v7 /*v263*/, v153
	v_dual_mov_b32 v6 /*v262*/, v153 :: v_dual_mov_b32 v5 /*v261*/, v153
	v_dual_mov_b32 v4 /*v260*/, v153 :: v_dual_mov_b32 v3 /*v259*/, v153
	v_dual_mov_b32 v2 /*v258*/, v153 :: v_dual_mov_b32 v17 /*v273*/, v153
	v_dual_mov_b32 v16 /*v272*/, v153 :: v_dual_mov_b32 v15 /*v271*/, v153
	v_dual_mov_b32 v14 /*v270*/, v153 :: v_dual_mov_b32 v13 /*v269*/, v153
	v_dual_mov_b32 v12 /*v268*/, v153 :: v_dual_mov_b32 v11 /*v267*/, v153
	v_mov_b32_e32 v10 /*v266*/, v153
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_branch .LBB0_44
.LBB0_40:
                                        ; implicit-def: $vgpr300
                                        ; implicit-def: $vgpr301
                                        ; implicit-def: $vgpr302
                                        ; implicit-def: $vgpr303
                                        ; implicit-def: $vgpr304
                                        ; implicit-def: $vgpr305
                                        ; implicit-def: $vgpr306
                                        ; implicit-def: $vgpr307
                                        ; implicit-def: $vgpr308
                                        ; implicit-def: $vgpr309
                                        ; implicit-def: $vgpr310
                                        ; implicit-def: $vgpr311
                                        ; implicit-def: $vgpr312
                                        ; implicit-def: $vgpr298
                                        ; implicit-def: $vgpr299
                                        ; implicit-def: $vgpr296
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr297
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB0_39
.LBB0_41:                               ; %.lr.ph
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v22 /*v278*/, 1.0 :: v_dual_mov_b32 v10 /*v266*/, 0
	.loc	1 1873 5 is_stmt 1              ; mla.py:1873:5
	s_mov_b32 s27, 0
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_and_b32_e32 v1, 0x100, v0
	s_set_vgpr_msb 0x50                     ;  msbs: dst=1 src0=0 src1=0 src2=1
	v_add3_u32 v40 /*v296*/, 0x20000, 0, v18 /*v274*/
	s_set_vgpr_msb 0x5001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v0, 0xff800000 :: v_dual_mov_b32 v251, v10 /*v266*/
	s_mov_b32 s20, 0x10000
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v11 /*v267*/, v10 /*v266*/ :: v_dual_mov_b32 v12 /*v268*/, v10 /*v266*/
	v_dual_mov_b32 v13 /*v269*/, v10 /*v266*/ :: v_dual_mov_b32 v14 /*v270*/, v10 /*v266*/
	v_dual_mov_b32 v15 /*v271*/, v10 /*v266*/ :: v_dual_mov_b32 v16 /*v272*/, v10 /*v266*/
	v_dual_mov_b32 v17 /*v273*/, v10 /*v266*/ :: v_dual_mov_b32 v2 /*v258*/, v10 /*v266*/
	v_dual_mov_b32 v3 /*v259*/, v10 /*v266*/ :: v_dual_mov_b32 v4 /*v260*/, v10 /*v266*/
	v_dual_mov_b32 v5 /*v261*/, v10 /*v266*/ :: v_dual_mov_b32 v6 /*v262*/, v10 /*v266*/
	v_dual_mov_b32 v7 /*v263*/, v10 /*v266*/ :: v_dual_mov_b32 v8 /*v264*/, v10 /*v266*/
	v_dual_mov_b32 v9 /*v265*/, v10 /*v266*/ :: v_dual_mov_b32 v0 /*v256*/, v10 /*v266*/
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v250, v10 /*v266*/ :: v_dual_mov_b32 v252, v10 /*v266*/
	v_dual_mov_b32 v253, v10 /*v266*/ :: v_dual_mov_b32 v254, v10 /*v266*/
	v_dual_mov_b32 v255, v10 /*v266*/ :: v_dual_mov_b32 v242, v10 /*v266*/
	s_set_vgpr_msb 0x141                    ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_mov_b32_e32 v1 /*v257*/, v10 /*v266*/
	s_set_vgpr_msb 0x4101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_dual_mov_b32 v243, v10 /*v266*/ :: v_dual_mov_b32 v244, v10 /*v266*/
	v_dual_mov_b32 v245, v10 /*v266*/ :: v_dual_mov_b32 v246, v10 /*v266*/
	v_dual_mov_b32 v247, v10 /*v266*/ :: v_dual_mov_b32 v248, v10 /*v266*/
	v_dual_mov_b32 v249, v10 /*v266*/ :: v_dual_mov_b32 v234, v10 /*v266*/
	v_dual_mov_b32 v235, v10 /*v266*/ :: v_dual_mov_b32 v236, v10 /*v266*/
	v_dual_mov_b32 v237, v10 /*v266*/ :: v_dual_mov_b32 v238, v10 /*v266*/
	v_dual_mov_b32 v239, v10 /*v266*/ :: v_dual_mov_b32 v240, v10 /*v266*/
	v_dual_mov_b32 v241, v10 /*v266*/ :: v_dual_mov_b32 v226, v10 /*v266*/
	v_dual_mov_b32 v227, v10 /*v266*/ :: v_dual_mov_b32 v228, v10 /*v266*/
	v_dual_mov_b32 v229, v10 /*v266*/ :: v_dual_mov_b32 v230, v10 /*v266*/
	v_dual_mov_b32 v231, v10 /*v266*/ :: v_dual_mov_b32 v232, v10 /*v266*/
	v_dual_mov_b32 v233, v10 /*v266*/ :: v_dual_mov_b32 v218, v10 /*v266*/
	v_dual_mov_b32 v219, v10 /*v266*/ :: v_dual_mov_b32 v220, v10 /*v266*/
	v_dual_mov_b32 v221, v10 /*v266*/ :: v_dual_mov_b32 v222, v10 /*v266*/
	v_dual_mov_b32 v223, v10 /*v266*/ :: v_dual_mov_b32 v224, v10 /*v266*/
	v_dual_mov_b32 v225, v10 /*v266*/ :: v_dual_mov_b32 v210, v10 /*v266*/
	v_dual_mov_b32 v211, v10 /*v266*/ :: v_dual_mov_b32 v212, v10 /*v266*/
	v_dual_mov_b32 v213, v10 /*v266*/ :: v_dual_mov_b32 v214, v10 /*v266*/
	v_dual_mov_b32 v215, v10 /*v266*/ :: v_dual_mov_b32 v216, v10 /*v266*/
	v_dual_mov_b32 v217, v10 /*v266*/ :: v_dual_mov_b32 v202, v10 /*v266*/
	v_dual_mov_b32 v203, v10 /*v266*/ :: v_dual_mov_b32 v204, v10 /*v266*/
	v_dual_mov_b32 v205, v10 /*v266*/ :: v_dual_mov_b32 v206, v10 /*v266*/
	v_dual_mov_b32 v207, v10 /*v266*/ :: v_dual_mov_b32 v208, v10 /*v266*/
	v_dual_mov_b32 v209, v10 /*v266*/ :: v_dual_mov_b32 v194, v10 /*v266*/
	v_dual_mov_b32 v195, v10 /*v266*/ :: v_dual_mov_b32 v196, v10 /*v266*/
	v_dual_mov_b32 v197, v10 /*v266*/ :: v_dual_mov_b32 v198, v10 /*v266*/
	v_dual_mov_b32 v199, v10 /*v266*/ :: v_dual_mov_b32 v200, v10 /*v266*/
	v_dual_mov_b32 v201, v10 /*v266*/ :: v_dual_mov_b32 v186, v10 /*v266*/
	v_dual_mov_b32 v187, v10 /*v266*/ :: v_dual_mov_b32 v188, v10 /*v266*/
	v_dual_mov_b32 v189, v10 /*v266*/ :: v_dual_mov_b32 v190, v10 /*v266*/
	v_dual_mov_b32 v191, v10 /*v266*/ :: v_dual_mov_b32 v192, v10 /*v266*/
	v_dual_mov_b32 v193, v10 /*v266*/ :: v_dual_mov_b32 v178, v10 /*v266*/
	v_dual_mov_b32 v179, v10 /*v266*/ :: v_dual_mov_b32 v180, v10 /*v266*/
	v_dual_mov_b32 v181, v10 /*v266*/ :: v_dual_mov_b32 v182, v10 /*v266*/
	v_dual_mov_b32 v183, v10 /*v266*/ :: v_dual_mov_b32 v184, v10 /*v266*/
	v_dual_mov_b32 v185, v10 /*v266*/ :: v_dual_mov_b32 v170, v10 /*v266*/
	v_dual_mov_b32 v171, v10 /*v266*/ :: v_dual_mov_b32 v172, v10 /*v266*/
	v_dual_mov_b32 v173, v10 /*v266*/ :: v_dual_mov_b32 v174, v10 /*v266*/
	v_dual_mov_b32 v175, v10 /*v266*/ :: v_dual_mov_b32 v176, v10 /*v266*/
	v_dual_mov_b32 v177, v10 /*v266*/ :: v_dual_mov_b32 v162, v10 /*v266*/
	v_dual_mov_b32 v163, v10 /*v266*/ :: v_dual_mov_b32 v164, v10 /*v266*/
	v_dual_mov_b32 v165, v10 /*v266*/ :: v_dual_mov_b32 v166, v10 /*v266*/
	v_dual_mov_b32 v167, v10 /*v266*/ :: v_dual_mov_b32 v168, v10 /*v266*/
	v_dual_mov_b32 v169, v10 /*v266*/ :: v_dual_mov_b32 v154, v10 /*v266*/
	v_dual_mov_b32 v155, v10 /*v266*/ :: v_dual_mov_b32 v156, v10 /*v266*/
	v_dual_mov_b32 v157, v10 /*v266*/ :: v_dual_mov_b32 v158, v10 /*v266*/
	v_dual_mov_b32 v159, v10 /*v266*/ :: v_dual_mov_b32 v160, v10 /*v266*/
	v_dual_mov_b32 v161, v10 /*v266*/ :: v_dual_mov_b32 v146, v10 /*v266*/
	v_dual_mov_b32 v147, v10 /*v266*/ :: v_dual_mov_b32 v148, v10 /*v266*/
	v_dual_mov_b32 v149, v10 /*v266*/ :: v_dual_mov_b32 v150, v10 /*v266*/
	v_dual_mov_b32 v151, v10 /*v266*/ :: v_dual_mov_b32 v152, v10 /*v266*/
	v_mov_b32_e32 v153, v10 /*v266*/
	s_mov_b32 s24, 1
	s_mov_b64 s[42:43], s[26:27]
	s_mov_b64 s[50:51], s[26:27]
	s_mov_b64 s[36:37], s[20:21]
	s_mov_b64 s[44:45], s[20:21]
	s_mov_b32 s18, 2
	s_mov_b64 s[40:41], s[24:25]
	s_mov_b64 s[38:39], s[22:23]
	s_mov_b32 s37, s17
	s_mov_b64 s[48:49], s[24:25]
	s_mov_b64 s[46:47], s[22:23]
	s_mov_b32 s45, s26
	s_mov_b32 s0, 0x3d76384f
	s_mov_b32 s19, 0x76543210
	s_mov_b32 s20, s27
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_42:                               ; =>This Inner Loop Header: Depth=1
.Ltmp62:
	.loc	1 846 18                        ; mla.py:846:18 @[ mla.py:1875:42 ]
	s_min_i32 s3, s18, s30
.Ltmp63:
	.loc	1 910 13                        ; mla.py:910:13 @[ mla.py:1008:16 @[ mla.py:1889:23 ] ]
	s_lshl_b32 s21, s20, 16
	s_set_vgpr_msb 0x80                     ;  msbs: dst=2 src0=0 src1=0 src2=0
.Ltmp64:
	.loc	1 847 30                        ; mla.py:847:30 @[ mla.py:1875:42 ]
	v_mov_b32_e32 v4 /*v516*/, s3
	s_set_vgpr_msb 0x8044                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_dual_mov_b32 v41 /*v297*/, v0 :: v_dual_add_nc_u32 v126 /*v382*/, s21, v18 /*v274*/
.Ltmp65:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	v_nop
	v_nop
	v_add_nc_u32_e32 v42 /*v298*/, s21, v38 /*v294*/
.Ltmp66:
	.loc	1 1035 18                       ; mla.py:1035:18 @[ mla.py:1894:23 ]
	s_add_co_i32 s16, s25, s77
	s_mov_b32 s2, s20
.Ltmp67:
	.loc	1 799 20                        ; mla.py:799:20 @[ mla.py:1893:26 ]
	s_xor_b32 s20, s20, 1
.Ltmp68:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1895:9 ]
	s_ashr_i32 s17, s16, 31
	s_set_vgpr_msb 0x4491                   ;  msbs: dst=2 src0=1 src1=0 src2=1
.Ltmp69:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	v_add3_u32 v5 /*v517*/, v42 /*v298*/, v1, v39 /*v295*/
.Ltmp70:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1895:9 ]
	s_sub_co_i32 s26, s31, s16
.Ltmp71:
	.loc	1 848 16                        ; mla.py:848:16 @[ mla.py:1875:42 ]
	s_add_co_i32 s18, s18, 1
	s_set_vgpr_msb 0x9110                   ;  msbs: dst=0 src0=0 src1=0 src2=1
.Ltmp72:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1904:22 ]
	v_lshl_add_u32 v0, s2, 13, v40 /*v296*/
.Ltmp73:
	.loc	1 1043 13                       ; mla.py:1043:13 @[ mla.py:1895:9 ]
	s_lshl_b32 s25, s20, 16
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1895:9 ]
	s_mul_u64 s[2:3], s[72:73], s[16:17]
	s_max_i32 s17, s26, 0
	s_cmp_gt_i32 s16, -1
	s_set_vgpr_msb 0x1042                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp74:
	.loc	1 847 30                        ; mla.py:847:30 @[ mla.py:1875:42 ]
	global_load_b32 v253 /*v509*/, v4 /*v516*/, s[10:11] scale_offset
.Ltmp75:
	.loc	1 1007 9                        ; mla.py:1007:9 @[ mla.py:1889:23 ]
	s_wait_tensorcnt 0x1
	s_barrier_signal -1
.Ltmp76:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1895:9 ]
	s_cselect_b32 s16, s17, 0
	s_mov_b64 s[58:59], s[42:43]
	s_add_nc_u64 s[26:27], s[2:3], s[12:13]
	s_lshl_b32 s21, s16, 16
	s_lshr_b32 s16, s16, 16
	s_mov_b64 s[56:57], s[40:41]
	s_mov_b64 s[54:55], s[38:39]
	s_mov_b64 s[52:53], s[36:37]
	s_mov_b32 s57, s14
	s_mov_b32 s58, s35
	s_add_co_i32 s25, s22, s25
	s_add_nc_u64 s[26:27], s[26:27], s[22:23]
	s_or_b32 s55, s16, 2.0
	s_mov_b32 s54, s21
.Ltmp77:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1896:9 ]
	s_mov_b64 s[66:67], s[50:51]
	.loc	1 1056 40                       ; mla.py:1056:40 @[ mla.py:1896:9 ]
	s_lshl_b32 s17, s20, 13
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1896:9 ]
	s_mov_b64 s[64:65], s[48:49]
	s_mov_b64 s[62:63], s[46:47]
	s_mov_b64 s[60:61], s[44:45]
	s_mov_b32 s65, s14
	s_mov_b32 s66, s35
	s_or_b32 s63, s16, 0x8000000
	s_mov_b32 s62, s21
.Ltmp78:
	.loc	1 1873 5                        ; mla.py:1873:5
	s_add_co_i32 s34, s34, 1
	s_set_vgpr_msb 0x4251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp79:
	.loc	1 1007 9                        ; mla.py:1007:9 @[ mla.py:1889:23 ]
	s_barrier_wait -1
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:512
	ds_load_b128 v[50:53] /*v[306:309]*/, v126 /*v382*/ offset:16384
	ds_load_b128 v[54:57] /*v[310:313]*/, v126 /*v382*/ offset:16896
	ds_load_b128 v[58:61] /*v[314:317]*/, v126 /*v382*/ offset:32768
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:1536
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:17408
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:17920
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:33792
.Ltmp80:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[122:129], 0
.Ltmp81:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[62:65] /*v[318:321]*/, v126 /*v382*/ offset:33280
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:49152
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:49664
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:1024
.Ltmp82:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[50:57] /*v[306:313]*/, v[122:129], 0
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[122:129], 0
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[122:129], 0
.Ltmp83:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:34304
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:50176
.Ltmp84:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[114:121], v[66:73] /*v[322:329]*/
.Ltmp85:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:50688
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:2048
.Ltmp86:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[114:121], v[74:81] /*v[330:337]*/
.Ltmp87:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:2560
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:18432
.Ltmp88:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[114:121], v[50:57] /*v[306:313]*/
.Ltmp89:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:18944
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:34816
.Ltmp90:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[114:121], v[58:65] /*v[314:321]*/
.Ltmp91:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:35328
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:51200
.Ltmp92:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[106:113], v[66:73] /*v[322:329]*/
.Ltmp93:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:51712
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:3072
.Ltmp94:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[106:113], v[74:81] /*v[330:337]*/
.Ltmp95:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:3584
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:19456
.Ltmp96:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[106:113], v[50:57] /*v[306:313]*/
.Ltmp97:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:19968
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:35840
.Ltmp98:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[106:113], v[58:65] /*v[314:321]*/
.Ltmp99:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:36352
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:52224
.Ltmp100:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[98:105], v[66:73] /*v[322:329]*/
.Ltmp101:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:52736
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:4096
.Ltmp102:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[98:105], v[74:81] /*v[330:337]*/
.Ltmp103:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:4608
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:20480
.Ltmp104:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[98:105], v[50:57] /*v[306:313]*/
.Ltmp105:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:20992
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:36864
.Ltmp106:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[98:105], v[58:65] /*v[314:321]*/
.Ltmp107:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:37376
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:53248
.Ltmp108:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[90:97], v[66:73] /*v[322:329]*/
.Ltmp109:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:53760
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:5120
.Ltmp110:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[90:97], v[74:81] /*v[330:337]*/
.Ltmp111:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:5632
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:21504
.Ltmp112:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[90:97], v[50:57] /*v[306:313]*/
.Ltmp113:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:22016
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:37888
.Ltmp114:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[90:97], v[58:65] /*v[314:321]*/
.Ltmp115:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:38400
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:54272
.Ltmp116:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[82:89], v[66:73] /*v[322:329]*/
.Ltmp117:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:54784
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:6144
.Ltmp118:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[82:89], v[74:81] /*v[330:337]*/
.Ltmp119:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:6656
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:22528
.Ltmp120:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[82:89], v[50:57] /*v[306:313]*/
.Ltmp121:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:23040
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:38912
.Ltmp122:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[82:89], v[58:65] /*v[314:321]*/
.Ltmp123:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:39424
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:55296
.Ltmp124:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[74:81], v[66:73] /*v[322:329]*/
.Ltmp125:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:55808
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:7168
.Ltmp126:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[74:81], v[74:81] /*v[330:337]*/
.Ltmp127:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:7680
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:23552
.Ltmp128:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[74:81], v[50:57] /*v[306:313]*/
.Ltmp129:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:24064
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:39936
.Ltmp130:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[74:81], v[58:65] /*v[314:321]*/
.Ltmp131:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:40448
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:56320
.Ltmp132:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[66:73], v[66:73] /*v[322:329]*/
.Ltmp133:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:56832
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:8192
.Ltmp134:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[66:73], v[74:81] /*v[330:337]*/
.Ltmp135:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:8704
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:24576
.Ltmp136:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[66:73], v[50:57] /*v[306:313]*/
.Ltmp137:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:25088
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:40960
.Ltmp138:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[66:73], v[58:65] /*v[314:321]*/
.Ltmp139:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:41472
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:57344
.Ltmp140:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[58:65], v[66:73] /*v[322:329]*/
.Ltmp141:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:57856
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:9216
.Ltmp142:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[58:65], v[74:81] /*v[330:337]*/
.Ltmp143:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:9728
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:25600
.Ltmp144:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[58:65], v[50:57] /*v[306:313]*/
.Ltmp145:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:26112
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:41984
.Ltmp146:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[58:65], v[58:65] /*v[314:321]*/
.Ltmp147:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:42496
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:58368
.Ltmp148:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[50:57], v[66:73] /*v[322:329]*/
.Ltmp149:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:58880
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:10240
.Ltmp150:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[50:57], v[74:81] /*v[330:337]*/
.Ltmp151:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:10752
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:26624
.Ltmp152:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[50:57], v[50:57] /*v[306:313]*/
.Ltmp153:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:27136
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:43008
.Ltmp154:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[50:57], v[58:65] /*v[314:321]*/
.Ltmp155:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:43520
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:59392
.Ltmp156:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[42:49], v[66:73] /*v[322:329]*/
.Ltmp157:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:59904
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:11264
.Ltmp158:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[42:49], v[74:81] /*v[330:337]*/
.Ltmp159:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:11776
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:27648
.Ltmp160:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[42:49], v[50:57] /*v[306:313]*/
.Ltmp161:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:28160
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:44032
.Ltmp162:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[42:49], v[58:65] /*v[314:321]*/
.Ltmp163:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:44544
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:60416
.Ltmp164:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[34:41], v[66:73] /*v[322:329]*/
.Ltmp165:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:60928
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:12288
.Ltmp166:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[34:41], v[74:81] /*v[330:337]*/
.Ltmp167:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:12800
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:28672
.Ltmp168:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[34:41], v[50:57] /*v[306:313]*/
.Ltmp169:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:29184
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:45056
.Ltmp170:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[34:41], v[58:65] /*v[314:321]*/
.Ltmp171:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:45568
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:61440
.Ltmp172:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[26:33], v[66:73] /*v[322:329]*/
.Ltmp173:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:61952
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:13312
.Ltmp174:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[26:33], v[74:81] /*v[330:337]*/
.Ltmp175:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:13824
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:29696
.Ltmp176:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[26:33], v[50:57] /*v[306:313]*/
.Ltmp177:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:30208
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:46080
.Ltmp178:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[26:33], v[58:65] /*v[314:321]*/
.Ltmp179:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:46592
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:62464
.Ltmp180:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[18:25], v[66:73] /*v[322:329]*/
.Ltmp181:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:62976
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:14336
.Ltmp182:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[18:25], v[74:81] /*v[330:337]*/
.Ltmp183:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:14848
	ds_load_b128 v[90:93] /*v[346:349]*/, v126 /*v382*/ offset:30720
.Ltmp184:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[10:17], v[66:73] /*v[322:329]*/
.Ltmp185:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[82:85] /*v[338:341]*/, v126 /*v382*/ offset:31744
	ds_load_b128 v[86:89] /*v[342:345]*/, v126 /*v382*/ offset:32256
	ds_load_b128 v[106:109] /*v[362:365]*/, v126 /*v382*/ offset:48128
	ds_load_b128 v[110:113] /*v[366:369]*/, v126 /*v382*/ offset:48640
	ds_load_b128 v[114:117] /*v[370:373]*/, v126 /*v382*/ offset:63488
	ds_load_b128 v[118:121] /*v[374:377]*/, v126 /*v382*/ offset:64000
	ds_load_b128 v[122:125] /*v[378:381]*/, v126 /*v382*/ offset:64512
.Ltmp186:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[18:25], v[50:57] /*v[306:313]*/
.Ltmp187:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[94:97] /*v[350:353]*/, v126 /*v382*/ offset:31232
	ds_load_b128 v[98:101] /*v[354:357]*/, v126 /*v382*/ offset:47104
.Ltmp188:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[18:25], v[58:65] /*v[314:321]*/
.Ltmp189:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1889:23 ]
	ds_load_b128 v[102:105] /*v[358:361]*/, v126 /*v382*/ offset:47616
	ds_load_b128 v[42:45] /*v[298:301]*/, v126 /*v382*/ offset:15360
	ds_load_b128 v[46:49] /*v[302:305]*/, v126 /*v382*/ offset:15872
	ds_load_b128 v[126:129] /*v[382:385]*/, v126 /*v382*/ offset:65024
.Ltmp190:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1895:9 ]
	s_wait_dscnt 0x0
	s_barrier_signal -1
.Ltmp191:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[10:17], v[74:81] /*v[330:337]*/
.Ltmp192:
	.loc	1 1040 9                        ; mla.py:1040:9 @[ mla.py:1895:9 ]
	s_barrier_wait -1
	tensor_load_to_lds s[24:27], s[52:59]
.Ltmp193:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1896:9 ]
	s_add_nc_u64 s[26:27], s[2:3], s[68:69]
	s_add_co_i32 s25, s1, s17
	s_add_nc_u64 s[26:27], s[26:27], s[70:71]
.Ltmp194:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[10:17], v[50:57] /*v[306:313]*/
.Ltmp195:
	.loc	1 1873 5                        ; mla.py:1873:5
	s_cmp_lt_i32 s34, s15
.Ltmp196:
	.loc	1 1055 9                        ; mla.py:1055:9 @[ mla.py:1896:9 ]
	tensor_load_to_lds s[24:27], s[60:67]
.Ltmp197:
	.loc	1 1030 9                        ; mla.py:1030:9 @[ mla.py:1904:22 ]
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
.Ltmp198:
	.loc	1 847 30                        ; mla.py:847:30 @[ mla.py:1875:42 ]
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s25, v253 /*v509*/
.Ltmp199:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[114:121] /*v[370:377]*/, v[10:17], v[58:65] /*v[314:321]*/
.Ltmp200:
	.loc	1 1030 9                        ; mla.py:1030:9 @[ mla.py:1904:22 ]
	s_barrier_wait -1
.Ltmp201:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[2:9], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5140                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp202:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1904:22 ]
	ds_load_b128 v[42:45] /*v[298:301]*/, v0
	s_set_vgpr_msb 0x4051                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp203:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[82:89] /*v[338:345]*/, v[2:9], v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5140                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp204:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1904:22 ]
	ds_load_b128 v[46:49] /*v[302:305]*/, v0 offset:512
	ds_load_b128 v[82:85] /*v[338:341]*/, v0 offset:2048
	ds_load_b128 v[86:89] /*v[342:345]*/, v0 offset:2560
	s_set_vgpr_msb 0x4051                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp205:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[106:113] /*v[362:369]*/, v[2:9], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5140                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp206:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1904:22 ]
	ds_load_b128 v[90:93] /*v[346:349]*/, v0 offset:1024
	ds_load_b128 v[94:97] /*v[350:353]*/, v0 offset:1536
	ds_load_b128 v[98:101] /*v[354:357]*/, v0 offset:3072
	ds_load_b128 v[102:105] /*v[358:361]*/, v0 offset:3584
	ds_load_b128 v[106:109] /*v[362:365]*/, v0 offset:4096
	ds_load_b128 v[110:113] /*v[366:369]*/, v0 offset:4608
	ds_load_b128 v[114:117] /*v[370:373]*/, v0 offset:5120
	s_set_vgpr_msb 0x4051                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp207:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1891:13 ]
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[122:129] /*v[378:385]*/, v[2:9], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5140                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp208:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1904:22 ]
	ds_load_b128 v[118:121] /*v[374:377]*/, v0 offset:5632
	ds_load_b128 v[122:125] /*v[378:381]*/, v0 offset:6144
	ds_load_b128 v[126:129] /*v[382:385]*/, v0 offset:6656
	ds_load_b128 v[130:133] /*v[386:389]*/, v0 offset:7168
	ds_load_b128 v[134:137] /*v[390:393]*/, v0 offset:7680
.Ltmp209:
	.loc	1 1012 9                        ; mla.py:1012:9 @[ mla.py:1937:29 ]
	s_wait_tensorcnt 0x2
	s_set_vgpr_msb 0x4051                   ;  msbs: dst=1 src0=1 src1=0 src2=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.Ltmp210:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1906:13 ]
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[138:145], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5142                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp211:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[42:45] /*v[298:301]*/, v5 /*v517*/
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v5 /*v517*/ offset:1024
	s_set_vgpr_msb 0x4251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp212:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1906:13 ]
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[82:89] /*v[338:345]*/, v[138:145], v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5142                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp213:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[46:49] /*v[302:305]*/, v5 /*v517*/ offset:16384
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v5 /*v517*/ offset:17408
	ds_load_tr16_b128 v[82:85] /*v[338:341]*/, v5 /*v517*/ offset:32768
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v5 /*v517*/ offset:33792
	ds_load_tr16_b128 v[86:89] /*v[342:345]*/, v5 /*v517*/ offset:49152
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v5 /*v517*/ offset:50176
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v5 /*v517*/ offset:51200
	s_set_vgpr_msb 0x4251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp214:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1906:13 ]
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[106:113] /*v[362:369]*/, v[138:145], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5142                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp215:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[106:109] /*v[362:365]*/, v5 /*v517*/ offset:2048
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v5 /*v517*/ offset:3072
	ds_load_tr16_b128 v[110:113] /*v[366:369]*/, v5 /*v517*/ offset:18432
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v5 /*v517*/ offset:19456
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v5 /*v517*/ offset:20480
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v5 /*v517*/ offset:4096
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v5 /*v517*/ offset:5120
	s_set_vgpr_msb 0x4251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp216:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1906:13 ]
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[122:129] /*v[378:385]*/, v[138:145], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5142                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp217:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v5 /*v517*/ offset:21504
	ds_load_tr16_b128 v[122:125] /*v[378:381]*/, v5 /*v517*/ offset:6144
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v5 /*v517*/ offset:7168
	ds_load_tr16_b128 v[126:129] /*v[382:385]*/, v5 /*v517*/ offset:22528
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v5 /*v517*/ offset:23552
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v5 /*v517*/ offset:8192
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v5 /*v517*/ offset:9216
	s_set_vgpr_msb 0x4251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp218:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1906:13 ]
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[90:97] /*v[346:353]*/, v[130:137], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5142                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp219:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v5 /*v517*/ offset:24576
	ds_load_tr16_b128 v[206:209] /*v[462:465]*/, v5 /*v517*/ offset:25600
	ds_load_tr16_b128 v[90:93] /*v[346:349]*/, v5 /*v517*/ offset:10240
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v5 /*v517*/ offset:11264
	ds_load_tr16_b128 v[94:97] /*v[350:353]*/, v5 /*v517*/ offset:26624
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v5 /*v517*/ offset:27648
	ds_load_tr16_b128 v[218:221] /*v[474:477]*/, v5 /*v517*/ offset:12288
.Ltmp220:
	.loc	1 1907 13                       ; mla.py:1907:13
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x4251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
	v_pk_mul_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[0:1] op_sel_hi:[1,0]
.Ltmp221:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1906:13 ]
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[98:105] /*v[354:361]*/, v[130:137], v[74:81] /*v[330:337]*/
.Ltmp222:
	.loc	1 1907 13                       ; mla.py:1907:13
	v_pk_mul_f32 v[68:69] /*v[324:325]*/, v[68:69] /*v[324:325]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71] /*v[326:327]*/, v[70:71] /*v[326:327]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73] /*v[328:329]*/, v[72:73] /*v[328:329]*/, s[0:1] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x5105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp223:
	.file	2 "/root/triton/python/triton/language" "standard.py"
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1909:23 ] ] ]
	v_max_num_f32_e32 v0, v66 /*v322*/, v67 /*v323*/
	s_set_vgpr_msb 0x551                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp224:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1906:13 ]
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[114:121] /*v[370:377]*/, v[130:137], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5142                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp225:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[98:101] /*v[354:357]*/, v5 /*v517*/ offset:13312
	ds_load_tr16_b128 v[222:225] /*v[478:481]*/, v5 /*v517*/ offset:28672
	ds_load_tr16_b128 v[102:105] /*v[358:361]*/, v5 /*v517*/ offset:29696
	ds_load_tr16_b128 v[226:229] /*v[482:485]*/, v5 /*v517*/ offset:14336
	ds_load_tr16_b128 v[114:117] /*v[370:373]*/, v5 /*v517*/ offset:15360
	ds_load_tr16_b128 v[230:233] /*v[486:489]*/, v5 /*v517*/ offset:30720
	ds_load_tr16_b128 v[118:121] /*v[374:377]*/, v5 /*v517*/ offset:31744
	s_set_vgpr_msb 0x4251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp226:
	.loc	1 1907 13                       ; mla.py:1907:13
	v_pk_mul_f32 v[74:75] /*v[330:331]*/, v[74:75] /*v[330:331]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77] /*v[332:333]*/, v[76:77] /*v[332:333]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79] /*v[334:335]*/, v[78:79] /*v[334:335]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, s[0:1] op_sel_hi:[1,0]
.Ltmp227:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1906:13 ]
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[130:137] /*v[386:393]*/, v[130:137], v[58:65] /*v[314:321]*/
.Ltmp228:
	.loc	1 1907 13                       ; mla.py:1907:13
	v_pk_mul_f32 v[50:51] /*v[306:307]*/, v[50:51] /*v[306:307]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53] /*v[308:309]*/, v[52:53] /*v[308:309]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55] /*v[310:311]*/, v[54:55] /*v[310:311]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57] /*v[312:313]*/, v[56:57] /*v[312:313]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59] /*v[314:315]*/, v[58:59] /*v[314:315]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61] /*v[316:317]*/, v[60:61] /*v[316:317]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63] /*v[318:319]*/, v[62:63] /*v[318:319]*/, s[0:1] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x5155                   ;  msbs: dst=1 src0=1 src1=1 src2=1
.Ltmp229:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1909:23 ] ] ]
	v_max3_num_f32 v130 /*v386*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v132 /*v388*/, v75 /*v331*/, v76 /*v332*/, v77 /*v333*/
	v_max3_num_f32 v133 /*v389*/, v78 /*v334*/, v79 /*v335*/, v80 /*v336*/
	v_max3_num_f32 v134 /*v390*/, v81 /*v337*/, v50 /*v306*/, v51 /*v307*/
.Ltmp230:
	.loc	1 1907 13                       ; mla.py:1907:13
	v_pk_mul_f32 v[64:65] /*v[320:321]*/, v[64:65] /*v[320:321]*/, s[0:1] op_sel_hi:[1,0]
.Ltmp231:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1909:23 ] ] ]
	v_max3_num_f32 v131 /*v387*/, v72 /*v328*/, v73 /*v329*/, v74 /*v330*/
	v_max3_num_f32 v135 /*v391*/, v52 /*v308*/, v53 /*v309*/, v54 /*v310*/
	v_max3_num_f32 v136 /*v392*/, v55 /*v311*/, v56 /*v312*/, v57 /*v313*/
	v_max3_num_f32 v137 /*v393*/, v58 /*v314*/, v59 /*v315*/, v60 /*v316*/
	v_max3_num_f32 v154 /*v410*/, v61 /*v317*/, v62 /*v318*/, v63 /*v319*/
	s_set_vgpr_msb 0x5514                   ;  msbs: dst=0 src0=0 src1=1 src2=1
	v_max3_num_f32 v0, v0, v68 /*v324*/, v130 /*v386*/
	s_set_vgpr_msb 0x1455                   ;  msbs: dst=1 src0=1 src1=1 src2=1
	v_max3_num_f32 v130 /*v386*/, v132 /*v388*/, v133 /*v389*/, v134 /*v390*/
	v_max3_num_f32 v132 /*v388*/, v135 /*v391*/, v136 /*v392*/, v137 /*v393*/
	v_max3_num_f32 v133 /*v389*/, v154 /*v410*/, v64 /*v320*/, v65 /*v321*/
	s_set_vgpr_msb 0x5514                   ;  msbs: dst=0 src0=0 src1=1 src2=1
	v_max3_num_f32 v0, v0, v131 /*v387*/, v130 /*v386*/
	v_max3_num_f32 v0, v0, v132 /*v388*/, v133 /*v389*/
	s_set_vgpr_msb 0x1440                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp232:
	.loc	2 191 16                        ; standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1909:23 ] ]
	v_permlanex16_b32 v130 /*v386*/, v0, s19, 0xfedcba98 op_sel:[1,0]
	s_set_vgpr_msb 0x4011                   ;  msbs: dst=0 src0=1 src1=0 src2=1
.Ltmp233:
	.loc	1 1101 16                       ; mla.py:1101:16 @[ mla.py:1909:23 ]
	v_max3_num_f32 v0, v41 /*v297*/, v0, v130 /*v386*/
	s_set_vgpr_msb 0x1141                   ;  msbs: dst=1 src0=1 src1=0 src2=0
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[68:69] /*v[324:325]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[70:71] /*v[326:327]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[72:73] /*v[328:329]*/, v[72:73] /*v[328:329]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[74:75] /*v[330:331]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[76:77] /*v[332:333]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[78:79] /*v[334:335]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1104 25                       ; mla.py:1104:25 @[ mla.py:1909:23 ]
	v_sub_f32_e32 v41 /*v297*/, v41 /*v297*/, v0
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v130 /*v386*/, v66 /*v322*/
	.loc	1 1103 21 is_stmt 0             ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[50:51] /*v[306:307]*/, v[50:51] /*v[306:307]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v131 /*v387*/, v67 /*v323*/
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[52:53] /*v[308:309]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v132 /*v388*/, v68 /*v324*/
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[54:55] /*v[310:311]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v133 /*v389*/, v69 /*v325*/
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[56:57] /*v[312:313]*/, v[56:57] /*v[312:313]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v134 /*v390*/, v70 /*v326*/
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[62:63] /*v[318:319]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v135 /*v391*/, v71 /*v327*/
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[64:65] /*v[320:321]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v136 /*v392*/, v72 /*v328*/
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[58:59] /*v[314:315]*/, v[58:59] /*v[314:315]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v137 /*v393*/, v73 /*v329*/
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1909:23 ]
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[60:61] /*v[316:317]*/, v[0:1] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v234 /*v490*/, v74 /*v330*/
	v_exp_f32_e32 v235 /*v491*/, v75 /*v331*/
	v_exp_f32_e32 v236 /*v492*/, v76 /*v332*/
	v_exp_f32_e32 v237 /*v493*/, v77 /*v333*/
	v_exp_f32_e32 v238 /*v494*/, v78 /*v334*/
	v_exp_f32_e32 v239 /*v495*/, v79 /*v335*/
	v_exp_f32_e32 v240 /*v496*/, v80 /*v336*/
	v_exp_f32_e32 v241 /*v497*/, v81 /*v337*/
	.loc	1 1104 17 is_stmt 1             ; mla.py:1104:17 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v252 /*v508*/, v41 /*v297*/
.Ltmp234:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1913:25 ]
	v_nop
	s_set_vgpr_msb 0x4145                   ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_pk_mul_f32 v[10:11] /*v[266:267]*/, v[10:11] /*v[266:267]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
.Ltmp235:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v242 /*v498*/, v50 /*v306*/
.Ltmp236:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1913:25 ]
	v_pk_mul_f32 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
.Ltmp237:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v243 /*v499*/, v51 /*v307*/
.Ltmp238:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1913:25 ]
	v_pk_mul_f32 v[14:15] /*v[270:271]*/, v[14:15] /*v[270:271]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
.Ltmp239:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v244 /*v500*/, v52 /*v308*/
.Ltmp240:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1913:25 ]
	v_pk_mul_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
.Ltmp241:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v245 /*v501*/, v53 /*v309*/
.Ltmp242:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1938:19 ]
	v_perm_b32 v50 /*v306*/, v131 /*v387*/, v130 /*v386*/, 0x7060302
.Ltmp243:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v246 /*v502*/, v54 /*v310*/
.Ltmp244:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1938:19 ]
	v_perm_b32 v51 /*v307*/, v133 /*v389*/, v132 /*v388*/, 0x7060302
.Ltmp245:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v247 /*v503*/, v55 /*v311*/
.Ltmp246:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1938:19 ]
	v_perm_b32 v52 /*v308*/, v135 /*v391*/, v134 /*v390*/, 0x7060302
.Ltmp247:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v248 /*v504*/, v56 /*v312*/
.Ltmp248:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1938:19 ]
	v_perm_b32 v53 /*v309*/, v137 /*v393*/, v136 /*v392*/, 0x7060302
.Ltmp249:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v249 /*v505*/, v57 /*v313*/
.Ltmp250:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1938:19 ]
	v_perm_b32 v54 /*v310*/, v235 /*v491*/, v234 /*v490*/, 0x7060302
	v_perm_b32 v55 /*v311*/, v237 /*v493*/, v236 /*v492*/, 0x7060302
	v_perm_b32 v56 /*v312*/, v239 /*v495*/, v238 /*v494*/, 0x7060302
	v_perm_b32 v57 /*v313*/, v241 /*v497*/, v240 /*v496*/, 0x7060302
.Ltmp251:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1913:25 ]
	v_pk_mul_f32 v[2:3] /*v[258:259]*/, v[2:3] /*v[258:259]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5] /*v[260:261]*/, v[4:5] /*v[260:261]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7] /*v[262:263]*/, v[6:7] /*v[262:263]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9] /*v[264:265]*/, v[8:9] /*v[264:265]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4504                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_pk_mul_f32 v[250:251], v[250:251], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[252:253], v[252:253], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[254:255], v[254:255], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x445                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_pk_mul_f32 v[0:1] /*v[256:257]*/, v[0:1] /*v[256:257]*/, v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4504                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_pk_mul_f32 v[242:243], v[242:243], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[244:245], v[244:245], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[246:247], v[246:247], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249], v[248:249], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235], v[234:235], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237], v[236:237], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[238:239], v[238:239], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[240:241], v[240:241], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[226:227], v[226:227], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[228:229], v[228:229], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[230:231], v[230:231], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[232:233], v[232:233], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[218:219], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[220:221], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225], v[224:225], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211], v[210:211], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213], v[212:213], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215], v[214:215], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[216:217], v[216:217], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203], v[202:203], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[204:205], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[206:207], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209], v[208:209], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195], v[194:195], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197], v[196:197], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199], v[198:199], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201], v[200:201], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[186:187], v[186:187], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189], v[188:189], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191], v[190:191], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193], v[192:193], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179], v[178:179], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181], v[180:181], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183], v[182:183], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[184:185], v[184:185], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171], v[170:171], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173], v[172:173], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175], v[174:175], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177], v[176:177], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[164:165], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167], v[166:167], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[168:169], v[168:169], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157], v[156:157], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161], v[160:161], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[148:149], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[152:153], v[152:153], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x455                    ;  msbs: dst=1 src0=1 src1=1 src2=1
.Ltmp252:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x22
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[42:49] /*v[298:305]*/, v[50:57] /*v[306:313]*/, v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5581                   ;  msbs: dst=2 src0=1 src1=0 src2=0
.Ltmp253:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v0 /*v512*/, v62 /*v318*/
	v_exp_f32_e32 v1 /*v513*/, v63 /*v319*/
	s_set_vgpr_msb 0x8155                   ;  msbs: dst=1 src0=1 src1=1 src2=1
.Ltmp254:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x21
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[138:145] /*v[394:401]*/, v[50:57] /*v[306:313]*/, v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5581                   ;  msbs: dst=2 src0=1 src1=0 src2=0
.Ltmp255:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v2 /*v514*/, v64 /*v320*/
	v_exp_f32_e32 v3 /*v515*/, v65 /*v321*/
	s_set_vgpr_msb 0x8105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp256:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x19
	v_wmma_f32_16x16x32_bf16 v[250:257], v[106:113] /*v[362:369]*/, v[50:57] /*v[306:313]*/, v[250:257]
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
.Ltmp257:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v250 /*v506*/, v58 /*v314*/
	v_exp_f32_e32 v251 /*v507*/, v59 /*v315*/
	s_set_vgpr_msb 0x4105                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp258:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[242:249], v[162:169] /*v[418:425]*/, v[50:57] /*v[306:313]*/, v[242:249]
	s_set_vgpr_msb 0x541                    ;  msbs: dst=1 src0=1 src1=0 src2=0
.Ltmp259:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1909:23 ]
	v_exp_f32_e32 v254 /*v510*/, v60 /*v316*/
	v_exp_f32_e32 v255 /*v511*/, v61 /*v317*/
	s_set_vgpr_msb 0x4142                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp260:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[58:61] /*v[314:317]*/, v5 /*v517*/ offset:36864
	ds_load_tr16_b128 v[66:69] /*v[322:325]*/, v5 /*v517*/ offset:37888
	s_set_vgpr_msb 0x4205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp261:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[234:241], v[170:177] /*v[426:433]*/, v[50:57] /*v[306:313]*/, v[234:241]
	s_set_vgpr_msb 0x545                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1938:19 ]
	v_perm_b32 v42 /*v298*/, v243 /*v499*/, v242 /*v498*/, 0x7060302
	v_perm_b32 v43 /*v299*/, v245 /*v501*/, v244 /*v500*/, 0x7060302
	v_perm_b32 v44 /*v300*/, v247 /*v503*/, v246 /*v502*/, 0x7060302
	v_perm_b32 v45 /*v301*/, v249 /*v505*/, v248 /*v504*/, 0x7060302
	s_set_vgpr_msb 0x4505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[226:233], v[178:185] /*v[434:441]*/, v[50:57] /*v[306:313]*/, v[226:233]
	s_set_vgpr_msb 0x545                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1938:19 ]
	v_perm_b32 v46 /*v302*/, v251 /*v507*/, v250 /*v506*/, 0x7060302
	v_perm_b32 v47 /*v303*/, v255 /*v511*/, v254 /*v510*/, 0x7060302
	s_set_vgpr_msb 0x454a                   ;  msbs: dst=1 src0=2 src1=2 src2=0
	v_perm_b32 v48 /*v304*/, v1 /*v513*/, v0 /*v512*/, 0x7060302
	v_perm_b32 v49 /*v305*/, v3 /*v515*/, v2 /*v514*/, 0x7060302
	s_set_vgpr_msb 0x4a05                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[218:225], v[122:129] /*v[378:385]*/, v[50:57] /*v[306:313]*/, v[218:225]
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[210:217], v[186:193] /*v[442:449]*/, v[50:57] /*v[306:313]*/, v[210:217]
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[202:209], v[194:201] /*v[450:457]*/, v[50:57] /*v[306:313]*/, v[202:209]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[194:201], v[202:209] /*v[458:465]*/, v[50:57] /*v[306:313]*/, v[194:201]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[186:193], v[90:97] /*v[346:353]*/, v[50:57] /*v[306:313]*/, v[186:193]
.Ltmp262:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x545                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[250:251] /*v[506:507]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x454a                   ;  msbs: dst=1 src0=2 src1=2 src2=0
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[0:1] /*v[512:513]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x4a05                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp263:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[178:185], v[210:217] /*v[466:473]*/, v[50:57] /*v[306:313]*/, v[178:185]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[170:177], v[218:225] /*v[474:481]*/, v[50:57] /*v[306:313]*/, v[170:177]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[162:169], v[98:105] /*v[354:361]*/, v[50:57] /*v[306:313]*/, v[162:169]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[154:161], v[226:233] /*v[482:489]*/, v[50:57] /*v[306:313]*/, v[154:161]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[146:153], v[114:121] /*v[370:377]*/, v[50:57] /*v[306:313]*/, v[146:153]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp264:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v5 /*v517*/ offset:34816
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v5 /*v517*/ offset:35840
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v5 /*v517*/ offset:52224
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v5 /*v517*/ offset:53248
	s_set_vgpr_msb 0x4205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp265:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[242:249], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[242:249]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp266:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[70:73] /*v[326:329]*/, v5 /*v517*/ offset:54272
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v5 /*v517*/ offset:55296
	s_set_vgpr_msb 0x4205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp267:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[234:241], v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[234:241]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp268:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v5 /*v517*/ offset:38912
	ds_load_tr16_b128 v[58:61] /*v[314:317]*/, v5 /*v517*/ offset:39936
	s_set_vgpr_msb 0x4205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp269:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[226:233], v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[226:233]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp270:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v5 /*v517*/ offset:56320
	ds_load_tr16_b128 v[70:73] /*v[326:329]*/, v5 /*v517*/ offset:57344
	s_set_vgpr_msb 0x4205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp271:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[218:225], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[218:225]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp272:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[66:69] /*v[322:325]*/, v5 /*v517*/ offset:40960
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v5 /*v517*/ offset:41984
	s_set_vgpr_msb 0x4205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp273:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[210:217], v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[210:217]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp274:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v5 /*v517*/ offset:58368
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v5 /*v517*/ offset:59392
	s_set_vgpr_msb 0x4205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp275:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[202:209], v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[202:209]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp276:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[58:61] /*v[314:317]*/, v5 /*v517*/ offset:43008
	ds_load_tr16_b128 v[66:69] /*v[322:325]*/, v5 /*v517*/ offset:44032
	s_set_vgpr_msb 0x4205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp277:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[194:201], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[194:201]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp278:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[70:73] /*v[326:329]*/, v5 /*v517*/ offset:60416
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v5 /*v517*/ offset:61440
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v5 /*v517*/ offset:45056
	ds_load_tr16_b128 v[74:77] /*v[330:333]*/, v5 /*v517*/ offset:46080
	ds_load_tr16_b128 v[78:81] /*v[334:337]*/, v5 /*v517*/ offset:62464
	s_set_vgpr_msb 0x4255                   ;  msbs: dst=1 src0=1 src1=1 src2=1
.Ltmp279:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[82:89] /*v[338:345]*/, v[42:49] /*v[298:305]*/, v[10:17] /*v[266:273]*/
.Ltmp280:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ]
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[130:131] /*v[386:387]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[134:135] /*v[390:391]*/, v[136:137] /*v[392:393]*/
	s_set_vgpr_msb 0x5505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp281:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[186:193], v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[186:193]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp282:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[58:61] /*v[314:317]*/, v5 /*v517*/ offset:47104
	s_set_vgpr_msb 0x4245                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp283:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ]
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[234:235] /*v[490:491]*/, v[236:237] /*v[492:493]*/
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[238:239] /*v[494:495]*/, v[240:241] /*v[496:497]*/
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[82:83] /*v[338:339]*/, v[84:85] /*v[340:341]*/
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[86:87] /*v[342:343]*/, v[88:89] /*v[344:345]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[242:243] /*v[498:499]*/, v[244:245] /*v[500:501]*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[246:247] /*v[502:503]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x4505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp284:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[178:185], v[66:73] /*v[322:329]*/, v[42:49] /*v[298:305]*/, v[178:185]
.Ltmp285:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x545                    ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[62:63] /*v[318:319]*/, v[64:65] /*v[320:321]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[90:91] /*v[346:347]*/, v[92:93] /*v[348:349]*/
	s_set_vgpr_msb 0x4505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp286:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[170:177], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[170:177]
	s_set_vgpr_msb 0x542                    ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp287:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v5 /*v517*/ offset:63488
	s_set_vgpr_msb 0x4245                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp288:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ]
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[82:83] /*v[338:339]*/, v[84:85] /*v[340:341]*/
	s_set_vgpr_msb 0x4542                   ;  msbs: dst=1 src0=2 src1=0 src2=0
.Ltmp289:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1937:29 ]
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v5 /*v517*/ offset:48128
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v5 /*v517*/ offset:64512
	s_set_vgpr_msb 0x4255                   ;  msbs: dst=1 src0=1 src1=1 src2=1
.Ltmp290:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ]
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[70:71] /*v[326:327]*/, v[66:67] /*v[322:323]*/
.Ltmp291:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[146:153] /*v[402:409]*/, v[42:49] /*v[298:305]*/, v[2:9] /*v[258:265]*/
.Ltmp292:
	.loc	2 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ] ]
	v_add_f32_e32 v41 /*v297*/, v66 /*v322*/, v67 /*v323*/
.Ltmp293:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ]
	v_permlanex16_b32 v66 /*v322*/, v41 /*v297*/, s19, 0xfedcba98 op_sel:[1,0]
.Ltmp294:
	.loc	1 1112 13                       ; mla.py:1112:13 @[ mla.py:1913:25 ]
	v_mul_f32_e32 v22 /*v278*/, v22 /*v278*/, v252 /*v508*/
	s_set_vgpr_msb 0x5505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp295:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	v_wmma_f32_16x16x32_bf16 v[250:257], v[154:161] /*v[410:417]*/, v[42:49] /*v[298:305]*/, v[250:257]
	s_set_vgpr_msb 0x545                    ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp296:
	.loc	2 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1913:25 ] ] ]
	v_add_f32_e32 v41 /*v297*/, v41 /*v297*/, v66 /*v322*/
.Ltmp297:
	.loc	1 1112 13                       ; mla.py:1112:13 @[ mla.py:1913:25 ]
	v_add_f32_e32 v22 /*v278*/, v22 /*v278*/, v41 /*v297*/
	s_set_vgpr_msb 0x4505                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp298:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1938:19 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[162:169], v[74:81] /*v[330:337]*/, v[42:49] /*v[298:305]*/, v[162:169]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65] /*v[314:321]*/, v[42:49] /*v[298:305]*/, v[154:161]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[146:153], v[50:57] /*v[306:313]*/, v[42:49] /*v[298:305]*/, v[146:153]
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp299:
	.loc	1 1873 5                        ; mla.py:1873:5
	s_cbranch_scc1 .LBB0_42
; %bb.43:                               ; %._crit_edge.loopexit
	.loc	1 0 5 is_stmt 0                 ; mla.py:0:5
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_dual_mov_b32 v44 /*v300*/, v23 /*v279*/ :: v_dual_mov_b32 v45 /*v301*/, v24 /*v280*/
	v_dual_mov_b32 v46 /*v302*/, v25 /*v281*/ :: v_dual_mov_b32 v47 /*v303*/, v26 /*v282*/
	v_dual_mov_b32 v48 /*v304*/, v27 /*v283*/ :: v_dual_mov_b32 v49 /*v305*/, v28 /*v284*/
	v_dual_mov_b32 v50 /*v306*/, v29 /*v285*/ :: v_dual_mov_b32 v51 /*v307*/, v30 /*v286*/
	v_dual_mov_b32 v52 /*v308*/, v31 /*v287*/ :: v_dual_mov_b32 v53 /*v309*/, v32 /*v288*/
	v_dual_mov_b32 v54 /*v310*/, v33 /*v289*/ :: v_dual_mov_b32 v55 /*v311*/, v34 /*v290*/
	v_dual_mov_b32 v56 /*v312*/, v35 /*v291*/ :: v_dual_mov_b32 v42 /*v298*/, v36 /*v292*/
	v_dual_mov_b32 v43 /*v299*/, v37 /*v293*/ :: v_dual_mov_b32 v40 /*v296*/, v38 /*v294*/
	v_mov_b32_e32 v41 /*v297*/, v39 /*v295*/
	s_mov_b32 s23, s20
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.LBB0_44:                               ; %._crit_edge
.Ltmp300:
	.loc	1 1007 9 is_stmt 1              ; mla.py:1007:9 @[ mla.py:1951:19 ]
	s_wait_tensorcnt 0x1
	s_barrier_signal -1
.Ltmp301:
	.loc	1 910 13                        ; mla.py:910:13 @[ mla.py:1008:16 @[ mla.py:1951:19 ] ]
	s_lshl_b32 s1, s23, 16
.Ltmp302:
	.loc	1 955 13                        ; mla.py:955:13 @[ mla.py:1031:16 @[ mla.py:1960:18 ] ]
	s_lshl_b32 s2, s23, 13
	s_set_vgpr_msb 0x44                     ;  msbs: dst=1 src0=0 src1=1 src2=0
.Ltmp303:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	v_dual_add_nc_u32 v23 /*v279*/, s1, v18 /*v274*/ :: v_dual_add_nc_u32 v28 /*v284*/, s1, v44 /*v300*/
.Ltmp304:
	.loc	1 955 13                        ; mla.py:955:13 @[ mla.py:1031:16 @[ mla.py:1960:18 ] ]
	s_add_co_i32 s2, s2, 0x20000
.Ltmp305:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	v_dual_add_nc_u32 v38 /*v294*/, s1, v47 /*v303*/ :: v_dual_add_nc_u32 v39 /*v295*/, s1, v48 /*v304*/
	v_dual_add_nc_u32 v60 /*v316*/, s1, v49 /*v305*/ :: v_dual_add_nc_u32 v61 /*v317*/, s1, v50 /*v306*/
	v_dual_add_nc_u32 v62 /*v318*/, s1, v51 /*v307*/ :: v_dual_add_nc_u32 v63 /*v319*/, s1, v52 /*v308*/
	s_set_vgpr_msb 0x4484                   ;  msbs: dst=2 src0=0 src1=1 src2=0
.Ltmp306:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1960:18 ]
	v_dual_add_nc_u32 v120 /*v632*/, s2, v48 /*v304*/ :: v_dual_add_nc_u32 v121 /*v633*/, s2, v49 /*v305*/
	v_dual_add_nc_u32 v122 /*v634*/, s2, v50 /*v306*/ :: v_dual_add_nc_u32 v123 /*v635*/, s2, v51 /*v307*/
	s_set_vgpr_msb 0x8444                   ;  msbs: dst=1 src0=0 src1=1 src2=0
.Ltmp307:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	v_dual_add_nc_u32 v36 /*v292*/, s1, v45 /*v301*/ :: v_dual_add_nc_u32 v37 /*v293*/, s1, v46 /*v302*/
	v_dual_add_nc_u32 v64 /*v320*/, s1, v53 /*v309*/ :: v_dual_add_nc_u32 v65 /*v321*/, s1, v54 /*v310*/
	v_dual_add_nc_u32 v66 /*v322*/, s1, v55 /*v311*/ :: v_dual_add_nc_u32 v67 /*v323*/, s1, v56 /*v312*/
	s_set_vgpr_msb 0x4484                   ;  msbs: dst=2 src0=0 src1=1 src2=0
.Ltmp308:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1960:18 ]
	v_dual_add_nc_u32 v124 /*v636*/, s2, v52 /*v308*/ :: v_dual_add_nc_u32 v125 /*v637*/, s2, v53 /*v309*/
	v_dual_add_nc_u32 v128 /*v640*/, s2, v54 /*v310*/ :: v_dual_add_nc_u32 v132 /*v644*/, s2, v55 /*v311*/
	v_add_nc_u32_e32 v136 /*v648*/, s2, v56 /*v312*/
	s_set_vgpr_msb 0x8444                   ;  msbs: dst=1 src0=0 src1=1 src2=0
.Ltmp309:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	v_dual_add_nc_u32 v68 /*v324*/, s1, v42 /*v298*/ :: v_dual_add_nc_u32 v69 /*v325*/, s1, v43 /*v299*/
	s_set_vgpr_msb 0x4484                   ;  msbs: dst=2 src0=0 src1=1 src2=0
.Ltmp310:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1960:18 ]
	v_dual_add_nc_u32 v116 /*v628*/, s2, v44 /*v300*/ :: v_dual_add_nc_u32 v117 /*v629*/, s2, v45 /*v301*/
	v_dual_add_nc_u32 v118 /*v630*/, s2, v46 /*v302*/ :: v_dual_add_nc_u32 v119 /*v631*/, s2, v47 /*v303*/
	s_set_vgpr_msb 0x8445                   ;  msbs: dst=1 src0=1 src1=1 src2=0
	v_add_nc_u32_e32 v18 /*v274*/, s2, v18 /*v274*/
	s_mov_b32 s0, 0x3d76384f
.Ltmp311:
	.loc	1 1007 9                        ; mla.py:1007:9 @[ mla.py:1951:19 ]
	s_barrier_wait -1
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v23 /*v279*/
	ds_load_b128 v[32:35] /*v[288:291]*/, v23 /*v279*/ offset:8192
	ds_load_b128 v[28:31] /*v[284:287]*/, v28 /*v284*/
	ds_load_b128 v[48:51] /*v[304:307]*/, v23 /*v279*/ offset:32256
	s_set_vgpr_msb 0x4581                   ;  msbs: dst=2 src0=1 src1=0 src2=0
	ds_load_b128 v[20:23] /*v[532:535]*/, v23 /*v279*/ offset:49152
	ds_load_b128 v[24:27] /*v[536:539]*/, v23 /*v279*/ offset:49664
	ds_load_b128 v[28:31] /*v[540:543]*/, v23 /*v279*/ offset:50176
	ds_load_b128 v[32:35] /*v[544:547]*/, v23 /*v279*/ offset:50688
	ds_load_b128 v[36:39] /*v[548:551]*/, v23 /*v279*/ offset:51200
	ds_load_b128 v[40:43] /*v[552:555]*/, v23 /*v279*/ offset:51712
	s_set_vgpr_msb 0x8151                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp312:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[122:129], 0
.Ltmp313:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v36 /*v292*/
	ds_load_b128 v[28:31] /*v[284:287]*/, v37 /*v293*/
.Ltmp314:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[114:121], v[52:59] /*v[308:315]*/
.Ltmp315:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v38 /*v294*/
	ds_load_b128 v[28:31] /*v[284:287]*/, v39 /*v295*/
.Ltmp316:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[106:113], v[52:59] /*v[308:315]*/
.Ltmp317:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v60 /*v316*/
	ds_load_b128 v[28:31] /*v[284:287]*/, v61 /*v317*/
.Ltmp318:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[98:105], v[52:59] /*v[308:315]*/
.Ltmp319:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v62 /*v318*/
	ds_load_b128 v[28:31] /*v[284:287]*/, v63 /*v319*/
.Ltmp320:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[90:97], v[52:59] /*v[308:315]*/
.Ltmp321:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v64 /*v320*/
	ds_load_b128 v[28:31] /*v[284:287]*/, v65 /*v321*/
.Ltmp322:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[82:89], v[52:59] /*v[308:315]*/
.Ltmp323:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v66 /*v322*/
	ds_load_b128 v[28:31] /*v[284:287]*/, v67 /*v323*/
.Ltmp324:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[74:81], v[52:59] /*v[308:315]*/
.Ltmp325:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v68 /*v324*/
	ds_load_b128 v[28:31] /*v[284:287]*/, v69 /*v325*/
.Ltmp326:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[66:73], v[52:59] /*v[308:315]*/
.Ltmp327:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[36:39] /*v[292:295]*/, v23 /*v279*/ offset:8704
	ds_load_b128 v[24:27] /*v[280:283]*/, v23 /*v279*/ offset:9216
.Ltmp328:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[32:39] /*v[288:295]*/, v[58:65], v[52:59] /*v[308:315]*/
.Ltmp329:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[28:31] /*v[284:287]*/, v23 /*v279*/ offset:9728
	ds_load_b128 v[32:35] /*v[288:291]*/, v23 /*v279*/ offset:10240
	ds_load_b128 v[36:39] /*v[292:295]*/, v23 /*v279*/ offset:10752
	ds_load_b128 v[60:63] /*v[316:319]*/, v23 /*v279*/ offset:11264
	ds_load_b128 v[64:67] /*v[320:323]*/, v23 /*v279*/ offset:11776
.Ltmp330:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[50:57], v[52:59] /*v[308:315]*/
.Ltmp331:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v23 /*v279*/ offset:12288
	ds_load_b128 v[28:31] /*v[284:287]*/, v23 /*v279*/ offset:12800
	ds_load_b128 v[68:71] /*v[324:327]*/, v23 /*v279*/ offset:13312
	ds_load_b128 v[72:75] /*v[328:331]*/, v23 /*v279*/ offset:13824
	ds_load_b128 v[76:79] /*v[332:335]*/, v23 /*v279*/ offset:14336
	ds_load_b128 v[80:83] /*v[336:339]*/, v23 /*v279*/ offset:14848
	ds_load_b128 v[84:87] /*v[340:343]*/, v23 /*v279*/ offset:15360
.Ltmp332:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[32:39] /*v[288:295]*/, v[42:49], v[52:59] /*v[308:315]*/
.Ltmp333:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[88:91] /*v[344:347]*/, v23 /*v279*/ offset:15872
	ds_load_b128 v[32:35] /*v[288:291]*/, v23 /*v279*/ offset:16384
	ds_load_b128 v[36:39] /*v[292:295]*/, v23 /*v279*/ offset:16896
	ds_load_b128 v[92:95] /*v[348:351]*/, v23 /*v279*/ offset:17408
	ds_load_b128 v[96:99] /*v[352:355]*/, v23 /*v279*/ offset:17920
	ds_load_b128 v[100:103] /*v[356:359]*/, v23 /*v279*/ offset:18432
	ds_load_b128 v[104:107] /*v[360:363]*/, v23 /*v279*/ offset:18944
	ds_load_b128 v[108:111] /*v[364:367]*/, v23 /*v279*/ offset:19456
.Ltmp334:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[32:39] /*v[288:295]*/, v[122:129], 0
.Ltmp335:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[32:35] /*v[288:291]*/, v23 /*v279*/ offset:37888
	ds_load_b128 v[36:39] /*v[292:295]*/, v23 /*v279*/ offset:38400
	ds_load_b128 v[212:215] /*v[468:471]*/, v23 /*v279*/ offset:38912
	ds_load_b128 v[216:219] /*v[472:475]*/, v23 /*v279*/ offset:39424
	ds_load_b128 v[220:223] /*v[476:479]*/, v23 /*v279*/ offset:39936
	ds_load_b128 v[224:227] /*v[480:483]*/, v23 /*v279*/ offset:40448
	ds_load_b128 v[228:231] /*v[484:487]*/, v23 /*v279*/ offset:40960
	ds_load_b128 v[232:235] /*v[488:491]*/, v23 /*v279*/ offset:41472
.Ltmp336:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[92:99] /*v[348:355]*/, v[114:121], v[204:211] /*v[460:467]*/
.Ltmp337:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[92:95] /*v[348:351]*/, v23 /*v279*/ offset:41984
	ds_load_b128 v[96:99] /*v[352:355]*/, v23 /*v279*/ offset:42496
	ds_load_b128 v[236:239] /*v[492:495]*/, v23 /*v279*/ offset:43008
	ds_load_b128 v[240:243] /*v[496:499]*/, v23 /*v279*/ offset:43520
	ds_load_b128 v[244:247] /*v[500:503]*/, v23 /*v279*/ offset:44032
	ds_load_b128 v[248:251] /*v[504:507]*/, v23 /*v279*/ offset:44544
.Ltmp338:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[60:67] /*v[316:323]*/, v[34:41], v[52:59] /*v[308:315]*/
.Ltmp339:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[112:115] /*v[368:371]*/, v23 /*v279*/ offset:19968
	ds_load_b128 v[60:63] /*v[316:319]*/, v23 /*v279*/ offset:20480
	ds_load_b128 v[64:67] /*v[320:323]*/, v23 /*v279*/ offset:20992
	ds_load_b128 v[116:119] /*v[372:375]*/, v23 /*v279*/ offset:21504
	ds_load_b128 v[120:123] /*v[376:379]*/, v23 /*v279*/ offset:22016
	ds_load_b128 v[124:127] /*v[380:383]*/, v23 /*v279*/ offset:22528
.Ltmp340:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[100:107] /*v[356:363]*/, v[106:113], v[204:211] /*v[460:467]*/
.Ltmp341:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[100:103] /*v[356:359]*/, v23 /*v279*/ offset:45056
	ds_load_b128 v[104:107] /*v[360:363]*/, v23 /*v279*/ offset:45568
	ds_load_b128 v[252:255] /*v[508:511]*/, v23 /*v279*/ offset:46080
	s_set_vgpr_msb 0x5181                   ;  msbs: dst=2 src0=1 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v23 /*v279*/ offset:46592
	ds_load_b128 v[4:7] /*v[516:519]*/, v23 /*v279*/ offset:47104
	ds_load_b128 v[8:11] /*v[520:523]*/, v23 /*v279*/ offset:47616
	ds_load_b128 v[12:15] /*v[524:527]*/, v23 /*v279*/ offset:48128
	ds_load_b128 v[16:19] /*v[528:531]*/, v23 /*v279*/ offset:48640
	s_set_vgpr_msb 0x8151                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp342:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[108:115] /*v[364:371]*/, v[98:105], v[204:211] /*v[460:467]*/
.Ltmp343:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[108:111] /*v[364:367]*/, v23 /*v279*/ offset:52224
	ds_load_b128 v[112:115] /*v[368:371]*/, v23 /*v279*/ offset:52736
	s_set_vgpr_msb 0x5181                   ;  msbs: dst=2 src0=1 src1=0 src2=0
	ds_load_b128 v[44:47] /*v[556:559]*/, v23 /*v279*/ offset:53248
	ds_load_b128 v[48:51] /*v[560:563]*/, v23 /*v279*/ offset:53760
	ds_load_b128 v[52:55] /*v[564:567]*/, v23 /*v279*/ offset:54272
	ds_load_b128 v[56:59] /*v[568:571]*/, v23 /*v279*/ offset:54784
	ds_load_b128 v[60:63] /*v[572:575]*/, v23 /*v279*/ offset:55296
	ds_load_b128 v[64:67] /*v[576:579]*/, v23 /*v279*/ offset:55808
	s_set_vgpr_msb 0x8151                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp344:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[60:67] /*v[316:323]*/, v[90:97], v[204:211] /*v[460:467]*/
.Ltmp345:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[60:63] /*v[316:319]*/, v23 /*v279*/ offset:56320
	ds_load_b128 v[64:67] /*v[320:323]*/, v23 /*v279*/ offset:56832
	s_set_vgpr_msb 0x5181                   ;  msbs: dst=2 src0=1 src1=0 src2=0
	ds_load_b128 v[68:71] /*v[580:583]*/, v23 /*v279*/ offset:57344
	ds_load_b128 v[72:75] /*v[584:587]*/, v23 /*v279*/ offset:57856
	ds_load_b128 v[76:79] /*v[588:591]*/, v23 /*v279*/ offset:58368
	ds_load_b128 v[80:83] /*v[592:595]*/, v23 /*v279*/ offset:58880
	s_set_vgpr_msb 0x8151                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp346:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[26:33], v[52:59] /*v[308:315]*/
.Ltmp347:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[128:131] /*v[384:387]*/, v23 /*v279*/ offset:23040
	ds_load_b128 v[24:27] /*v[280:283]*/, v23 /*v279*/ offset:23552
	ds_load_b128 v[28:31] /*v[284:287]*/, v23 /*v279*/ offset:24064
	ds_load_b128 v[132:135] /*v[388:391]*/, v23 /*v279*/ offset:24576
	ds_load_b128 v[136:139] /*v[392:395]*/, v23 /*v279*/ offset:25088
	ds_load_b128 v[140:143] /*v[396:399]*/, v23 /*v279*/ offset:25600
	ds_load_b128 v[144:147] /*v[400:403]*/, v23 /*v279*/ offset:26112
	ds_load_b128 v[148:151] /*v[404:407]*/, v23 /*v279*/ offset:26624
.Ltmp348:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x1f
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[116:123] /*v[372:379]*/, v[82:89], v[204:211] /*v[460:467]*/
.Ltmp349:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[116:119] /*v[372:375]*/, v23 /*v279*/ offset:59392
	ds_load_b128 v[120:123] /*v[376:379]*/, v23 /*v279*/ offset:59904
	s_set_vgpr_msb 0x5181                   ;  msbs: dst=2 src0=1 src1=0 src2=0
	ds_load_b128 v[84:87] /*v[596:599]*/, v23 /*v279*/ offset:60416
	ds_load_b128 v[88:91] /*v[600:603]*/, v23 /*v279*/ offset:60928
	ds_load_b128 v[92:95] /*v[604:607]*/, v23 /*v279*/ offset:61440
	ds_load_b128 v[96:99] /*v[608:611]*/, v23 /*v279*/ offset:61952
	ds_load_b128 v[100:103] /*v[612:615]*/, v23 /*v279*/ offset:62464
	ds_load_b128 v[104:107] /*v[616:619]*/, v23 /*v279*/ offset:62976
	s_set_vgpr_msb 0x8151                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp350:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[124:131] /*v[380:387]*/, v[74:81], v[204:211] /*v[460:467]*/
.Ltmp351:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[124:127] /*v[380:383]*/, v23 /*v279*/ offset:63488
	ds_load_b128 v[128:131] /*v[384:387]*/, v23 /*v279*/ offset:64000
	s_set_vgpr_msb 0x5181                   ;  msbs: dst=2 src0=1 src1=0 src2=0
	ds_load_b128 v[108:111] /*v[620:623]*/, v23 /*v279*/ offset:64512
	ds_load_b128 v[112:115] /*v[624:627]*/, v23 /*v279*/ offset:65024
	s_set_vgpr_msb 0x8151                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp352:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[24:31] /*v[280:287]*/, v[66:73], v[204:211] /*v[460:467]*/
.Ltmp353:
	.loc	1 1965 55                       ; mla.py:1965:55
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5144                   ;  msbs: dst=1 src0=0 src1=1 src2=0
	v_lshrrev_b32_e32 v24 /*v280*/, 1, v19 /*v275*/
	s_set_vgpr_msb 0x4490                   ;  msbs: dst=2 src0=0 src1=0 src2=1
	.loc	1 1965 18 is_stmt 0             ; mla.py:1965:18
	v_lshl_or_b32 v140 /*v652*/, s15, 6, v24 /*v280*/
	s_set_vgpr_msb 0x9051                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp354:
	.loc	1 1080 20 is_stmt 1             ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[132:139] /*v[388:395]*/, v[58:65], v[204:211] /*v[460:467]*/
	s_set_vgpr_msb 0x5108                   ;  msbs: dst=0 src0=0 src1=2 src2=0
.Ltmp355:
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v140 /*v652*/
	s_set_vgpr_msb 0x851                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp356:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[68:75] /*v[324:331]*/, v[18:25], v[52:59] /*v[308:315]*/
.Ltmp357:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[152:155] /*v[408:411]*/, v23 /*v279*/ offset:27136
	ds_load_b128 v[68:71] /*v[324:327]*/, v23 /*v279*/ offset:27648
	ds_load_b128 v[72:75] /*v[328:331]*/, v23 /*v279*/ offset:28160
	ds_load_b128 v[156:159] /*v[412:415]*/, v23 /*v279*/ offset:28672
	ds_load_b128 v[160:163] /*v[416:419]*/, v23 /*v279*/ offset:29184
	ds_load_b128 v[164:167] /*v[420:423]*/, v23 /*v279*/ offset:29696
.Ltmp358:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[140:147] /*v[396:403]*/, v[50:57], v[204:211] /*v[460:467]*/
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[76:83] /*v[332:339]*/, v[10:17], v[52:59] /*v[308:315]*/
.Ltmp359:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[168:171] /*v[424:427]*/, v23 /*v279*/ offset:30208
	ds_load_b128 v[76:79] /*v[332:335]*/, v23 /*v279*/ offset:30720
	ds_load_b128 v[80:83] /*v[336:339]*/, v23 /*v279*/ offset:31232
	ds_load_b128 v[44:47] /*v[300:303]*/, v23 /*v279*/ offset:31744
	ds_load_b128 v[172:175] /*v[428:431]*/, v23 /*v279*/ offset:32768
	ds_load_b128 v[176:179] /*v[432:435]*/, v23 /*v279*/ offset:33280
	ds_load_b128 v[180:183] /*v[436:439]*/, v23 /*v279*/ offset:33792
	ds_load_b128 v[184:187] /*v[440:443]*/, v23 /*v279*/ offset:34304
.Ltmp360:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[148:155] /*v[404:411]*/, v[42:49], v[204:211] /*v[460:467]*/
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[84:91] /*v[340:347]*/, v[2:9], v[52:59] /*v[308:315]*/
.Ltmp361:
	.loc	1 1008 16                       ; mla.py:1008:16 @[ mla.py:1951:19 ]
	ds_load_b128 v[84:87] /*v[340:343]*/, v23 /*v279*/ offset:34816
	ds_load_b128 v[88:91] /*v[344:347]*/, v23 /*v279*/ offset:35328
	ds_load_b128 v[188:191] /*v[444:447]*/, v23 /*v279*/ offset:35840
	ds_load_b128 v[192:195] /*v[448:451]*/, v23 /*v279*/ offset:36352
	ds_load_b128 v[196:199] /*v[452:455]*/, v23 /*v279*/ offset:36864
	ds_load_b128 v[200:203] /*v[456:459]*/, v23 /*v279*/ offset:37376
.Ltmp362:
	.loc	1 1030 9                        ; mla.py:1030:9 @[ mla.py:1960:18 ]
	s_wait_tensorcnt 0x0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.Ltmp363:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[68:75] /*v[324:331]*/, v[34:41], v[204:211] /*v[460:467]*/
	s_set_vgpr_msb 0x5145                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp364:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1960:18 ]
	v_dual_add_nc_u32 v23 /*v279*/, s2, v42 /*v298*/ :: v_dual_add_nc_u32 v42 /*v298*/, s2, v43 /*v299*/
	ds_load_b128 v[68:71] /*v[324:327]*/, v23 /*v279*/
	ds_load_b128 v[72:75] /*v[328:331]*/, v42 /*v298*/
	s_set_vgpr_msb 0x4551                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp365:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[156:163] /*v[412:419]*/, v[26:33], v[204:211] /*v[460:467]*/
.Ltmp366:
	.loc	1 1031 16                       ; mla.py:1031:16 @[ mla.py:1960:18 ]
	ds_load_b128 v[24:27] /*v[280:283]*/, v18 /*v274*/
	s_set_vgpr_msb 0x5142                   ;  msbs: dst=1 src0=2 src1=0 src2=0
	ds_load_b128 v[28:31] /*v[284:287]*/, v116 /*v628*/
	ds_load_b128 v[132:135] /*v[388:391]*/, v117 /*v629*/
	ds_load_b128 v[136:139] /*v[392:395]*/, v118 /*v630*/
	ds_load_b128 v[140:143] /*v[396:399]*/, v119 /*v631*/
	ds_load_b128 v[144:147] /*v[400:403]*/, v120 /*v632*/
	ds_load_b128 v[148:151] /*v[404:407]*/, v121 /*v633*/
	ds_load_b128 v[152:155] /*v[408:411]*/, v122 /*v634*/
	s_set_vgpr_msb 0x4282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[116:119] /*v[628:631]*/, v123 /*v635*/
	ds_load_b128 v[120:123] /*v[632:635]*/, v124 /*v636*/
	ds_load_b128 v[124:127] /*v[636:639]*/, v125 /*v637*/
	ds_load_b128 v[128:131] /*v[640:643]*/, v128 /*v640*/
	ds_load_b128 v[132:135] /*v[644:647]*/, v132 /*v644*/
	ds_load_b128 v[136:139] /*v[648:651]*/, v136 /*v648*/
.Ltmp367:
	.loc	1 1012 9                        ; mla.py:1012:9 @[ mla.py:1993:25 ]
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x8251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.Ltmp368:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[164:171] /*v[420:427]*/, v[18:25], v[204:211] /*v[460:467]*/
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[76:83] /*v[332:339]*/, v[10:17], v[204:211] /*v[460:467]*/
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[44:51] /*v[300:307]*/, v[2:9], v[204:211] /*v[460:467]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[172:179] /*v[428:435]*/, v[122:129], 0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[180:187] /*v[436:443]*/, v[114:121], v[42:49] /*v[298:305]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[84:91] /*v[340:347]*/, v[106:113], v[42:49] /*v[298:305]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[188:195] /*v[444:451]*/, v[98:105], v[42:49] /*v[298:305]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[196:203] /*v[452:459]*/, v[90:97], v[42:49] /*v[298:305]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[32:39] /*v[288:295]*/, v[82:89], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5152                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[20:27] /*v[532:539]*/, v[122:129], 0
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[28:35] /*v[540:547]*/, v[114:121], v[32:39] /*v[288:295]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[36:43] /*v[548:555]*/, v[106:113], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[108:115] /*v[364:371]*/, v[98:105], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5152                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[44:51] /*v[556:563]*/, v[90:97], v[32:39] /*v[288:295]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[52:59] /*v[564:571]*/, v[82:89], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[212:219] /*v[468:475]*/, v[74:81], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5152                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[60:67] /*v[572:579]*/, v[74:81], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[220:227] /*v[476:483]*/, v[66:73], v[42:49] /*v[298:305]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[60:67] /*v[316:323]*/, v[66:73], v[32:39] /*v[288:295]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[228:235] /*v[484:491]*/, v[58:65], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5152                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[68:75] /*v[580:587]*/, v[58:65], v[32:39] /*v[288:295]*/
.Ltmp369:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5208                   ;  msbs: dst=0 src0=0 src1=2 src2=0
	v_or_b32_e32 v58, 49, v140 /*v652*/
	v_or_b32_e32 v59, 50, v140 /*v652*/
	s_set_vgpr_msb 0x851                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp370:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[92:99] /*v[348:355]*/, v[50:57], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5108                   ;  msbs: dst=0 src0=0 src1=2 src2=0
.Ltmp371:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_or_b32_e32 v60, 51, v140 /*v652*/
	v_or_b32_e32 v61, 52, v140 /*v652*/
	v_or_b32_e32 v62, 53, v140 /*v652*/
	v_or_b32_e32 v63, 54, v140 /*v652*/
	s_set_vgpr_msb 0x852                    ;  msbs: dst=1 src0=2 src1=0 src2=1
.Ltmp372:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[76:83] /*v[588:595]*/, v[50:57], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5208                   ;  msbs: dst=0 src0=0 src1=2 src2=0
.Ltmp373:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_or_b32_e32 v64, 55, v140 /*v652*/
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v50, 33, v140 /*v652*/
	v_or_b32_e32 v51, 34, v140 /*v652*/
	s_set_vgpr_msb 0x851                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp374:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[236:243] /*v[492:499]*/, v[42:49], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5108                   ;  msbs: dst=0 src0=0 src1=2 src2=0
.Ltmp375:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_or_b32_e32 v52, 35, v140 /*v652*/
	v_or_b32_e32 v53, 36, v140 /*v652*/
	v_or_b32_e32 v54, 37, v140 /*v652*/
	v_or_b32_e32 v55, 38, v140 /*v652*/
	s_set_vgpr_msb 0x851                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp376:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[116:123] /*v[372:379]*/, v[42:49], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5108                   ;  msbs: dst=0 src0=0 src1=2 src2=0
.Ltmp377:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_or_b32_e32 v56, 39, v140 /*v652*/
	v_or_b32_e32 v57, 48, v140 /*v652*/
	s_set_vgpr_msb 0x851                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp378:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[244:251] /*v[500:507]*/, v[34:41], v[42:49] /*v[298:305]*/
.Ltmp379:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_nop
	s_set_vgpr_msb 0x5108                   ;  msbs: dst=0 src0=0 src1=2 src2=0
	v_or_b32_e32 v42, 17, v140 /*v652*/
	v_or_b32_e32 v43, 18, v140 /*v652*/
	v_or_b32_e32 v44, 19, v140 /*v652*/
	v_or_b32_e32 v45, 20, v140 /*v652*/
	s_set_vgpr_msb 0x852                    ;  msbs: dst=1 src0=2 src1=0 src2=1
.Ltmp380:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[84:91] /*v[596:603]*/, v[34:41], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5208                   ;  msbs: dst=0 src0=0 src1=2 src2=0
.Ltmp381:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_or_b32_e32 v46, 21, v140 /*v652*/
	v_or_b32_e32 v47, 22, v140 /*v652*/
	v_or_b32_e32 v48, 23, v140 /*v652*/
	v_or_b32_e32 v49, 32, v140 /*v652*/
	s_set_vgpr_msb 0x851                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp382:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[100:107] /*v[356:363]*/, v[26:33], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5108                   ;  msbs: dst=0 src0=0 src1=2 src2=0
.Ltmp383:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_or_b32_e32 v34, 1, v140 /*v652*/
	v_or_b32_e32 v35, 2, v140 /*v652*/
	v_or_b32_e32 v36, 3, v140 /*v652*/
	v_or_b32_e32 v37, 4, v140 /*v652*/
	s_set_vgpr_msb 0x852                    ;  msbs: dst=1 src0=2 src1=0 src2=1
.Ltmp384:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[92:99] /*v[604:611]*/, v[26:33], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5208                   ;  msbs: dst=0 src0=0 src1=2 src2=0
.Ltmp385:
	.loc	1 1965 18                       ; mla.py:1965:18
	v_or_b32_e32 v38, 5, v140 /*v652*/
	v_or_b32_e32 v39, 6, v140 /*v652*/
	v_or_b32_e32 v40, 7, v140 /*v652*/
	v_or_b32_e32 v41, 16, v140 /*v652*/
	s_set_vgpr_msb 0x851                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp386:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[252:259] /*v[508:515]*/, v[18:25], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5152                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[100:107] /*v[612:619]*/, v[18:25], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp387:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1962:9 ]
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[24:31] /*v[280:287]*/, v[138:145], v[52:59] /*v[308:315]*/
	s_set_vgpr_msb 0x5152                   ;  msbs: dst=1 src0=2 src1=0 src2=1
.Ltmp388:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[4:11] /*v[516:523]*/, v[10:17], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5251                   ;  msbs: dst=1 src0=1 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[124:131] /*v[380:387]*/, v[10:17], v[32:39] /*v[288:295]*/
.Ltmp389:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1962:9 ]
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[308:315]*/, v[132:139] /*v[388:395]*/, v[130:137], v[52:59] /*v[308:315]*/
	s_set_vgpr_msb 0x5152                   ;  msbs: dst=1 src0=2 src1=0 src2=1
.Ltmp390:
	.loc	1 1080 20                       ; mla.py:1080:20 @[ mla.py:1953:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[12:19] /*v[524:531]*/, v[2:9], v[42:49] /*v[298:305]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[108:115] /*v[620:627]*/, v[2:9], v[32:39] /*v[288:295]*/
.Ltmp391:
	.loc	1 1963 9                        ; mla.py:1963:9
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5201                   ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_pk_mul_f32 v[2:3], v[52:53] /*v[308:309]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[54:55] /*v[310:311]*/, s[0:1] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x151                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp392:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1962:9 ]
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[140:147] /*v[396:403]*/, v[138:145], v[204:211] /*v[460:467]*/
	s_set_vgpr_msb 0x5101                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp393:
	.loc	1 1963 9                        ; mla.py:1963:9
	v_pk_mul_f32 v[6:7], v[56:57] /*v[312:313]*/, s[0:1] op_sel_hi:[1,0]
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v2, 0xff800000, v2, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v34
	.loc	1 1963 9                        ; mla.py:1963:9
	v_pk_mul_f32 v[8:9], v[58:59] /*v[314:315]*/, s[0:1] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x151                    ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp394:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1962:9 ]
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[148:155] /*v[404:411]*/, v[130:137], v[204:211] /*v[460:467]*/
	s_set_vgpr_msb 0x5100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp395:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v3, 0xff800000, v3, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v35
.Ltmp396:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max_num_f32_e32 v34, v2, v3
.Ltmp397:
	.loc	1 1963 9                        ; mla.py:1963:9
	v_nop
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_pk_mul_f32 v[10:11], v[204:205] /*v[460:461]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[206:207] /*v[462:463]*/, s[0:1] op_sel_hi:[1,0]
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v4, 0xff800000, v4, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v36
	.loc	1 1963 9                        ; mla.py:1963:9
	v_pk_mul_f32 v[14:15], v[208:209] /*v[464:465]*/, s[0:1] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x152                    ;  msbs: dst=1 src0=2 src1=0 src2=1
.Ltmp398:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1962:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[116:123] /*v[628:635]*/, v[138:145], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5201                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp399:
	.loc	1 1963 9                        ; mla.py:1963:9
	v_pk_mul_f32 v[16:17], v[210:211] /*v[466:467]*/, s[0:1] op_sel_hi:[1,0]
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v5, 0xff800000, v5, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v37
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v6, 0xff800000, v6, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v38
	s_set_vgpr_msb 0x152                    ;  msbs: dst=1 src0=2 src1=0 src2=1
.Ltmp400:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1962:9 ]
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[124:131] /*v[636:643]*/, v[130:137], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5201                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp401:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v7, 0xff800000, v7, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v39
	.loc	1 1963 9                        ; mla.py:1963:9
	v_nop
	v_nop
	v_pk_mul_f32 v[18:19], v[42:43] /*v[298:299]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[44:45] /*v[300:301]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[46:47] /*v[302:303]*/, s[0:1] op_sel_hi:[1,0]
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v8, 0xff800000, v8, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v40
	s_set_vgpr_msb 0x152                    ;  msbs: dst=1 src0=2 src1=0 src2=1
.Ltmp402:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1962:9 ]
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[132:139] /*v[644:651]*/, v[138:145], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5201                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp403:
	.loc	1 1963 9                        ; mla.py:1963:9
	v_pk_mul_f32 v[24:25], v[48:49] /*v[304:305]*/, s[0:1] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp404:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v35, v5, v6, v7
.Ltmp405:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v9, 0xff800000, v9, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v41
	s_set_vgpr_msb 0x51                     ;  msbs: dst=1 src0=1 src1=0 src2=1
.Ltmp406:
	.loc	1 1097 20                       ; mla.py:1097:20 @[ mla.py:1962:9 ]
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[68:75] /*v[324:331]*/, v[130:137], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp407:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v34, v34, v4, v35
.Ltmp408:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v10, 0xff800000, v10, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v42
	.loc	1 1963 9                        ; mla.py:1963:9
	v_nop
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_pk_mul_f32 v[26:27], v[32:33] /*v[288:289]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[34:35] /*v[290:291]*/, s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[36:37] /*v[292:293]*/, s[0:1] op_sel_hi:[1,0]
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v11, 0xff800000, v11, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v43
	.loc	1 1963 9                        ; mla.py:1963:9
	v_pk_mul_f32 v[32:33], v[38:39] /*v[294:295]*/, s[0:1] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp409:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v36, v8, v9, v10
.Ltmp410:
	.loc	2 191 16                        ; standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ]
	s_mov_b32 s0, 0x76543210
.Ltmp411:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v12, 0xff800000, v12, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v44
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v13, 0xff800000, v13, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v45
.Ltmp412:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v37, v11, v12, v13
.Ltmp413:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v14, 0xff800000, v14, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v46
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v15, 0xff800000, v15, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v47
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v16, 0xff800000, v16, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v48
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v17, 0xff800000, v17, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v49
.Ltmp414:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v38, v14, v15, v16
.Ltmp415:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v18, 0xff800000, v18, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v50
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v19, 0xff800000, v19, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v51
.Ltmp416:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v39, v17, v18, v19
.Ltmp417:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v20, 0xff800000, v20, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v52
.Ltmp418:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v35, v37, v38, v39
.Ltmp419:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v21, 0xff800000, v21, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v53
.Ltmp420:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v34, v34, v36, v35
.Ltmp421:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v22, 0xff800000, v22, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v54
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v23, 0xff800000, v23, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v55
.Ltmp422:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v40, v20, v21, v22
.Ltmp423:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v24, 0xff800000, v24, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v56
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v25, 0xff800000, v25, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v57
.Ltmp424:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v41, v23, v24, v25
.Ltmp425:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v26, 0xff800000, v26, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v58
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v27, 0xff800000, v27, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v59
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v28, 0xff800000, v28, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v60
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v29, 0xff800000, v29, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v61
.Ltmp426:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v42, v26, v27, v28
.Ltmp427:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v30, 0xff800000, v30, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v62
.Ltmp428:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v37, v40, v41, v42
.Ltmp429:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v31, 0xff800000, v31, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v63
.Ltmp430:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v43, v29, v30, v31
.Ltmp431:
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v32, 0xff800000, v32, vcc_lo
	.loc	1 1968 16                       ; mla.py:1968:16
	v_cmp_gt_i32_e32 vcc_lo, s29, v64
	.loc	1 1969 9                        ; mla.py:1969:9
	v_cndmask_b32_e32 v33, 0xff800000, v33, vcc_lo
.Ltmp432:
	.loc	2 170 12                        ; standard.py:170:12 @[ standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ] ]
	v_max3_num_f32 v38, v43, v32, v33
	v_max3_num_f32 v34, v34, v37, v38
.Ltmp433:
	.loc	2 191 16                        ; standard.py:191:16 @[ mla.py:1101:30 @[ mla.py:1970:19 ] ]
	v_permlanex16_b32 v35, v34, s0, 0xfedcba98 op_sel:[1,0]
.Ltmp434:
	.loc	1 1101 16                       ; mla.py:1101:16 @[ mla.py:1970:19 ]
	v_max3_num_f32 v128, v0, v34, v35
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1970:19 ]
	v_pk_add_f32 v[2:3], v[2:3], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5], v[4:5], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[6:7], v[6:7], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9], v[8:9], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[10:11], v[10:11], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	.loc	1 1103 13 is_stmt 0             ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v140, v2
.Ltmp435:
	.loc	1 1013 16 is_stmt 1             ; mla.py:1013:16 @[ mla.py:1993:25 ]
	v_nop
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_add_nc_u32_e32 v2, s1, v40 /*v296*/
	s_set_vgpr_msb 0x410                    ;  msbs: dst=0 src0=0 src1=0 src2=1
.Ltmp436:
	.loc	1 1103 21                       ; mla.py:1103:21 @[ mla.py:1970:19 ]
	v_pk_add_f32 v[12:13], v[12:13], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15], v[14:15], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17], v[16:17], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[18:19], v[18:19], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21], v[20:21], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[22:23], v[22:23], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25], v[24:25], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[26:27], v[26:27], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[28:29], v[28:29], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[30:31], v[30:31], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33], v[32:33], v[128:129] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
.Ltmp437:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	v_add3_u32 v129, v2, v1, v41 /*v297*/
	s_set_vgpr_msb 0x1040                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp438:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v24 /*v280*/, v8
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 1104 25                       ; mla.py:1104:25 @[ mla.py:1970:19 ]
	v_sub_f32_e32 v0, v0, v128
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v25 /*v281*/, v9
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp439:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[48:51], v129 offset:34816
	ds_load_tr16_b128 v[56:59], v129 offset:3072
	ds_load_tr16_b128 v[60:63], v129 offset:19456
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp440:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v26 /*v282*/, v10
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp441:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[64:67], v129 offset:35840
	ds_load_tr16_b128 v[68:71], v129 offset:52224
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp442:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v27 /*v283*/, v11
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp443:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[80:83], v129 offset:38912
	ds_load_tr16_b128 v[88:91], v129 offset:7168
	ds_load_tr16_b128 v[92:95], v129 offset:23552
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp444:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v28 /*v284*/, v12
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp445:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v135, v25 /*v281*/, v24 /*v280*/, 0x7060302
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp446:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v29 /*v285*/, v13
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp447:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[96:99], v129 offset:39936
	ds_load_tr16_b128 v[100:103], v129 offset:56320
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp448:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v30 /*v286*/, v14
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp449:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[112:115], v129 offset:43008
	ds_load_tr16_b128 v[120:123], v129 offset:11264
	ds_load_tr16_b128 v[124:127], v129 offset:27648
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp450:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v31 /*v287*/, v15
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp451:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[8:11], v129
	ds_load_tr16_b128 v[12:15], v129 offset:16384
.Ltmp452:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v141, v3
.Ltmp453:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[40:43], v129 offset:2048
	ds_load_tr16_b128 v[72:75], v129 offset:6144
.Ltmp454:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v142, v4
.Ltmp455:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[44:47], v129 offset:18432
	ds_load_tr16_b128 v[76:79], v129 offset:22528
.Ltmp456:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v143, v5
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp457:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v136, v27 /*v283*/, v26 /*v282*/, 0x7060302
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp458:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v144, v6
.Ltmp459:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v132, v141, v140, 0x7060302
.Ltmp460:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v145, v7
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp461:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v137, v29 /*v285*/, v28 /*v284*/, 0x7060302
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp462:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v32 /*v288*/, v16
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp463:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v138, v31 /*v287*/, v30 /*v286*/, 0x7060302
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp464:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v33 /*v289*/, v17
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp465:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v133, v143, v142, 0x7060302
.Ltmp466:
	.loc	1 1104 17                       ; mla.py:1104:17 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v130, v0
.Ltmp467:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[52:55], v129 offset:51200
	ds_load_tr16_b128 v[84:87], v129 offset:55296
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp468:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v34 /*v290*/, v18
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp469:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v134, v145, v144, 0x7060302
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp470:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v35 /*v291*/, v19
	s_set_vgpr_msb 0x4005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp471:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v139, v33 /*v289*/, v32 /*v288*/, 0x7060302
	s_set_vgpr_msb 0x540                    ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp472:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v36 /*v292*/, v20
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp473:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[0:1], v[10:11] /*v[266:267]*/, v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp474:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v37 /*v293*/, v21
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp475:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[2:3], v[12:13] /*v[268:269]*/, v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp476:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v38 /*v294*/, v22
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp477:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[4:5], v[14:15] /*v[270:271]*/, v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp478:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v39 /*v295*/, v23
	s_set_vgpr_msb 0x4001                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp479:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[6:7], v[16:17] /*v[272:273]*/, v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x140                    ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp480:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v40 /*v296*/, v24
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp481:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[132:139], v[0:7]
.Ltmp482:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[104:107], v129 offset:10240
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp483:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v41 /*v297*/, v25
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp484:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[108:111], v129 offset:26624
	ds_load_tr16_b128 v[116:119], v129 offset:59392
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp485:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v42 /*v298*/, v26
	v_exp_f32_e32 v43 /*v299*/, v27
	s_set_vgpr_msb 0x4045                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp486:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v10 /*v266*/, v35 /*v291*/, v34 /*v290*/, 0x7060302
	s_set_vgpr_msb 0x4540                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp487:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v44 /*v300*/, v28
	s_set_vgpr_msb 0x4045                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp488:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v11 /*v267*/, v37 /*v293*/, v36 /*v292*/, 0x7060302
	s_set_vgpr_msb 0x4540                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp489:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v45 /*v301*/, v29
	s_set_vgpr_msb 0x4045                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp490:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v12 /*v268*/, v39 /*v295*/, v38 /*v294*/, 0x7060302
	s_set_vgpr_msb 0x4540                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp491:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v46 /*v302*/, v30
	s_set_vgpr_msb 0x4045                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp492:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v13 /*v269*/, v41 /*v297*/, v40 /*v296*/, 0x7060302
	s_set_vgpr_msb 0x4540                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp493:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v47 /*v303*/, v31
	s_set_vgpr_msb 0x4000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp494:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[16:19], v129 offset:32768
	ds_load_tr16_b128 v[20:23], v129 offset:49152
	ds_load_tr16_b128 v[24:27], v129 offset:1024
	ds_load_tr16_b128 v[28:31], v129 offset:17408
	s_set_vgpr_msb 64                       ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp495:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v48 /*v304*/, v32
	s_set_vgpr_msb 0x4045                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp496:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v14 /*v270*/, v43 /*v299*/, v42 /*v298*/, 0x7060302
	s_set_vgpr_msb 0x4540                   ;  msbs: dst=1 src0=0 src1=0 src2=0
.Ltmp497:
	.loc	1 1103 13                       ; mla.py:1103:13 @[ mla.py:1970:19 ]
	v_exp_f32_e32 v49 /*v305*/, v33
	s_set_vgpr_msb 0x4045                   ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp498:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v15 /*v271*/, v45 /*v301*/, v44 /*v300*/, 0x7060302
	s_set_vgpr_msb 0x4500                   ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp499:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[32:35], v129 offset:33792
	ds_load_tr16_b128 v[36:39], v129 offset:50176
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp500:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[8:9], v[2:3] /*v[258:259]*/, v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp501:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v16 /*v272*/, v47 /*v303*/, v46 /*v302*/, 0x7060302
	s_set_vgpr_msb 0x4501                   ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp502:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[10:11], v[4:5] /*v[260:261]*/, v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[6:7] /*v[262:263]*/, v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[8:9] /*v[264:265]*/, v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x145                    ;  msbs: dst=1 src0=1 src1=1 src2=0
.Ltmp503:
	.loc	1 1164 17                       ; mla.py:1164:17 @[ mla.py:1994:15 ]
	v_perm_b32 v17 /*v273*/, v49 /*v305*/, v48 /*v304*/, 0x7060302
	s_set_vgpr_msb 0x4504                   ;  msbs: dst=0 src0=0 src1=1 src2=0
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[0:7], v[16:23], v[10:17] /*v[266:273]*/, v[0:7]
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp504:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ]
	v_pk_add_f32 v[140:141], v[140:141], v[142:143]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	v_pk_add_f32 v[142:143], v[144:145], v[24:25] /*v[280:281]*/
	s_set_vgpr_msb 0x405                    ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_pk_add_f32 v[144:145], v[26:27] /*v[282:283]*/, v[28:29] /*v[284:285]*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp505:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[8:15], v[24:31], v[132:139], v[8:15]
.Ltmp506:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[16:17], v[250:251], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[252:253], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[254:255], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
	v_pk_mul_f32 v[22:23], v[0:1] /*v[256:257]*/, v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp507:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[40:47], v[132:139], v[16:23]
.Ltmp508:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[40:43], v129 offset:4096
	ds_load_tr16_b128 v[44:47], v129 offset:20480
.Ltmp509:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[24:25], v[242:243], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[244:245], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[246:247], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp510:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[48:55], v[10:17] /*v[266:273]*/, v[16:23]
.Ltmp511:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[52:55], v129 offset:53248
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp512:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[30:31], v[248:249], v[130:131] op_sel_hi:[1,0]
.Ltmp513:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63], v[132:139], v[24:31]
.Ltmp514:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[48:51], v129 offset:36864
	ds_load_tr16_b128 v[56:59], v129 offset:5120
	ds_load_tr16_b128 v[60:63], v129 offset:21504
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp515:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[8:15], v[32:39], v[10:17] /*v[266:273]*/, v[8:15]
.Ltmp516:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_pk_mul_f32 v[32:33], v[234:235], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[236:237], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp517:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[64:71], v[10:17] /*v[266:273]*/, v[24:31]
.Ltmp518:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[64:67], v129 offset:37888
	ds_load_tr16_b128 v[68:71], v129 offset:54272
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp519:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[36:37], v[238:239], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[240:241], v[130:131] op_sel_hi:[1,0]
.Ltmp520:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[32:39], v[40:47], v[132:139], v[32:39]
.Ltmp521:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_mul_f32 v[40:41], v[226:227], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[228:229], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp522:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[32:39], v[48:55], v[10:17] /*v[266:273]*/, v[32:39]
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp523:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[44:45], v[230:231], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[232:233], v[130:131] op_sel_hi:[1,0]
.Ltmp524:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[40:47], v[56:63], v[132:139], v[40:47]
.Ltmp525:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_nop
	v_pk_mul_f32 v[48:49], v[218:219], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[220:221], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[222:223], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[224:225], v[130:131] op_sel_hi:[1,0]
.Ltmp526:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[72:79], v[132:139], v[48:55]
.Ltmp527:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[72:75], v129 offset:8192
	ds_load_tr16_b128 v[76:79], v129 offset:24576
.Ltmp528:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[56:57], v[210:211], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[212:213], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[214:215], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp529:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[80:87], v[10:17] /*v[266:273]*/, v[48:55]
.Ltmp530:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[84:87], v129 offset:57344
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp531:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[62:63], v[216:217], v[130:131] op_sel_hi:[1,0]
.Ltmp532:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[88:95], v[132:139], v[56:63]
.Ltmp533:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[80:83], v129 offset:40960
	ds_load_tr16_b128 v[88:91], v129 offset:9216
	ds_load_tr16_b128 v[92:95], v129 offset:25600
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp534:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[40:47], v[64:71], v[10:17] /*v[266:273]*/, v[40:47]
.Ltmp535:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_pk_mul_f32 v[64:65], v[202:203], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[204:205], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp536:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[96:103], v[10:17] /*v[266:273]*/, v[56:63]
.Ltmp537:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[96:99], v129 offset:41984
	ds_load_tr16_b128 v[100:103], v129 offset:58368
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp538:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[68:69], v[206:207], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[208:209], v[130:131] op_sel_hi:[1,0]
.Ltmp539:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[64:71], v[72:79], v[132:139], v[64:71]
.Ltmp540:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_mul_f32 v[72:73], v[194:195], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[196:197], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp541:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[64:71], v[80:87], v[10:17] /*v[266:273]*/, v[64:71]
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp542:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[76:77], v[198:199], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[200:201], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
.Ltmp543:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ]
	v_pk_add_f32 v[194:195], v[46:47] /*v[302:303]*/, v[48:49] /*v[304:305]*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp544:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[72:79], v[88:95], v[132:139], v[72:79]
.Ltmp545:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[80:81], v[186:187], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[188:189], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[190:191], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[192:193], v[130:131] op_sel_hi:[1,0]
.Ltmp546:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[104:111], v[132:139], v[80:87]
.Ltmp547:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[104:107], v129 offset:12288
	ds_load_tr16_b128 v[108:111], v129 offset:28672
.Ltmp548:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[88:89], v[178:179], v[130:131] op_sel_hi:[1,0]
.Ltmp549:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[186:189], v129 offset:44032
	ds_load_tr16_b128 v[190:193], v129 offset:60416
.Ltmp550:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[90:91], v[180:181], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[182:183], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp551:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[112:119], v[10:17] /*v[266:273]*/, v[80:87]
.Ltmp552:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[116:119], v129 offset:61440
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp553:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[94:95], v[184:185], v[130:131] op_sel_hi:[1,0]
.Ltmp554:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[178:181], v129 offset:47104
	ds_load_tr16_b128 v[182:185], v129 offset:63488
.Ltmp555:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[120:127], v[132:139], v[88:95]
.Ltmp556:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[112:115], v129 offset:45056
	ds_load_tr16_b128 v[120:123], v129 offset:13312
	ds_load_tr16_b128 v[124:127], v129 offset:29696
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp557:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[72:79], v[96:103], v[10:17] /*v[266:273]*/, v[72:79]
.Ltmp558:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_pk_mul_f32 v[96:97], v[170:171], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[172:173], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp559:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[88:95], v[186:193], v[10:17] /*v[266:273]*/, v[88:95]
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp560:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[100:101], v[174:175], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[176:177], v[130:131] op_sel_hi:[1,0]
.Ltmp561:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[170:173], v129 offset:46080
	ds_load_tr16_b128 v[174:177], v129 offset:62464
.Ltmp562:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[104:111], v[132:139], v[96:103]
.Ltmp563:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ]
	v_nop
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_pk_add_f32 v[186:187], v[30:31] /*v[286:287]*/, v[32:33] /*v[288:289]*/
	v_pk_add_f32 v[188:189], v[34:35] /*v[290:291]*/, v[36:37] /*v[292:293]*/
	v_pk_add_f32 v[190:191], v[38:39] /*v[294:295]*/, v[40:41] /*v[296:297]*/
	v_pk_add_f32 v[192:193], v[42:43] /*v[298:299]*/, v[44:45] /*v[300:301]*/
	s_set_vgpr_msb 0x504                    ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp564:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[96:103], v[112:119], v[10:17] /*v[266:273]*/, v[96:103]
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp565:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[104:105], v[162:163], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[164:165], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[166:167], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[168:169], v[130:131] op_sel_hi:[1,0]
.Ltmp566:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[162:165], v129 offset:14336
	ds_load_tr16_b128 v[166:169], v129 offset:30720
.Ltmp567:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127], v[132:139], v[104:111]
.Ltmp568:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[112:113], v[154:155], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[156:157], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[158:159], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[160:161], v[130:131] op_sel_hi:[1,0]
.Ltmp569:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[154:157], v129 offset:15360
	ds_load_tr16_b128 v[158:161], v129 offset:31744
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp570:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[104:111], v[170:177], v[10:17] /*v[266:273]*/, v[104:111]
.Ltmp571:
	.loc	1 1013 16                       ; mla.py:1013:16 @[ mla.py:1993:25 ]
	ds_load_tr16_b128 v[170:173], v129 offset:48128
	ds_load_tr16_b128 v[174:177], v129 offset:64512
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp572:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ]
	v_pk_add_f32 v[120:121], v[140:141], v[142:143]
	v_pk_add_f32 v[122:123], v[144:145], v[186:187]
	v_pk_add_f32 v[124:125], v[188:189], v[190:191]
	v_pk_add_f32 v[126:127], v[192:193], v[194:195]
.Ltmp573:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[112:119], v[162:169], v[132:139], v[112:119]
.Ltmp574:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ]
	v_pk_add_f32 v[140:141], v[120:121], v[122:123]
.Ltmp575:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[120:121], v[146:147], v[130:131] op_sel_hi:[1,0]
.Ltmp576:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ]
	v_pk_add_f32 v[142:143], v[124:125], v[126:127]
.Ltmp577:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[122:123], v[148:149], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp578:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[178:185], v[10:17] /*v[266:273]*/, v[112:119]
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp579:
	.loc	1 1111 17                       ; mla.py:1111:17 @[ mla.py:1974:21 ]
	v_pk_mul_f32 v[124:125], v[150:151], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[152:153], v[130:131] op_sel_hi:[1,0]
.Ltmp580:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ]
	v_pk_add_f32 v[140:141], v[140:141], v[142:143]
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp581:
	.loc	1 1792 9                        ; mla.py:1792:9
	v_or_b32_e32 v129, s76, v21 /*v277*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
.Ltmp582:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[120:127], v[154:161], v[132:139], v[120:127]
.Ltmp583:
	.loc	2 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ] ]
	v_add_f32_e32 v131, v140, v141
.Ltmp584:
	.loc	1 1797 23                       ; mla.py:1797:23
	v_cmp_gt_i32_e32 vcc_lo, 16, v129
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp585:
	.loc	1 1166 15                       ; mla.py:1166:15 @[ mla.py:1994:15 ]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[120:127], v[170:177], v[10:17] /*v[266:273]*/, v[120:127]
.Ltmp586:
	.loc	2 293 12                        ; standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ]
	v_nop
	v_permlanex16_b32 v132, v131, s0, 0xfedcba98 op_sel:[1,0]
.Ltmp587:
	.loc	1 1304 16                       ; mla.py:1304:16 @[ mla.py:2029:9 ]
	s_and_b32 s0, vcc_lo, s75
	.loc	1 1326 13                       ; mla.py:1326:13 @[ mla.py:2029:9 ]
	s_and_saveexec_b32 s1, s0
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_execz .LBB0_46
; %bb.45:                               ; %.critedge
	.loc	1 1313 15                       ; mla.py:1313:15 @[ mla.py:2029:9 ]
	v_lshlrev_b32_e32 v133, 11, v129
	.loc	1 1307 13                       ; mla.py:1307:13 @[ mla.py:2029:9 ]
	s_lshl_b32 s2, s28, 15
	.loc	1 1315 15                       ; mla.py:1315:15 @[ mla.py:2029:9 ]
	s_lshl_b32 s3, s33, 9
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
	.loc	1 1299 28                       ; mla.py:1299:28 @[ mla.py:2029:9 ]
	v_lshrrev_b32_e32 v134, 1, v20 /*v276*/
	.loc	1 1307 13                       ; mla.py:1307:13 @[ mla.py:2029:9 ]
	v_add3_u32 v133, s2, s3, v133
	.loc	1 1299 28                       ; mla.py:1299:28 @[ mla.py:2029:9 ]
	v_and_or_b32 v134, v134, 24, v133
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	.loc	1 1327 17                       ; mla.py:1327:17 @[ mla.py:2029:9 ]
	v_ashrrev_i32_e32 v135, 31, v134
	v_lshl_add_u64 v[134:135], v[134:135], 2, s[4:5]
	.loc	1 1326 13                       ; mla.py:1326:13 @[ mla.py:2029:9 ]
	s_clause 0x1f
	global_store_b128 v[134:135], v[0:3], off
	global_store_b128 v[134:135], v[4:7], off offset:16
	global_store_b128 v[134:135], v[8:11], off offset:128
	global_store_b128 v[134:135], v[12:15], off offset:144
	global_store_b128 v[134:135], v[16:19], off offset:256
	global_store_b128 v[134:135], v[20:23], off offset:272
	global_store_b128 v[134:135], v[24:27], off offset:384
	global_store_b128 v[134:135], v[28:31], off offset:400
	global_store_b128 v[134:135], v[32:35], off offset:512
	global_store_b128 v[134:135], v[36:39], off offset:528
	global_store_b128 v[134:135], v[40:43], off offset:640
	global_store_b128 v[134:135], v[44:47], off offset:656
	global_store_b128 v[134:135], v[48:51], off offset:768
	global_store_b128 v[134:135], v[52:55], off offset:784
	global_store_b128 v[134:135], v[56:59], off offset:896
	global_store_b128 v[134:135], v[60:63], off offset:912
	global_store_b128 v[134:135], v[64:67], off offset:1024
	global_store_b128 v[134:135], v[68:71], off offset:1040
	global_store_b128 v[134:135], v[72:75], off offset:1152
	global_store_b128 v[134:135], v[76:79], off offset:1168
	global_store_b128 v[134:135], v[80:83], off offset:1280
	global_store_b128 v[134:135], v[84:87], off offset:1296
	global_store_b128 v[134:135], v[88:91], off offset:1408
	global_store_b128 v[134:135], v[92:95], off offset:1424
	global_store_b128 v[134:135], v[96:99], off offset:1536
	global_store_b128 v[134:135], v[100:103], off offset:1552
	global_store_b128 v[134:135], v[104:107], off offset:1664
	global_store_b128 v[134:135], v[108:111], off offset:1680
	global_store_b128 v[134:135], v[112:115], off offset:1792
	global_store_b128 v[134:135], v[116:119], off offset:1808
	global_store_b128 v[134:135], v[120:123], off offset:1920
	global_store_b128 v[134:135], v[124:127], off offset:1936
.LBB0_46:                               ; %.critedge60
	.loc	1 0 13 is_stmt 0                ; mla.py:0:13
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_set_vgpr_msb 4                        ;  msbs: dst=0 src0=0 src1=1 src2=0
.Ltmp588:
	.loc	1 1286 17 is_stmt 1             ; mla.py:1286:17 @[ mla.py:1332:9 @[ mla.py:2029:9 ] ]
	v_or_b32_e32 v0, s74, v19 /*v275*/
	s_set_vgpr_msb 0x400                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s1, s0
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB0_48
.Ltmp589:
; %bb.47:
	.loc	2 263 12                        ; standard.py:263:12 @[ standard.py:293:12 @[ mla.py:1109:16 @[ mla.py:1974:21 ] ] ]
	v_dual_add_f32 v0, v131, v132 :: v_dual_lshlrev_b32 v1, 2, v129
	s_set_vgpr_msb 1                        ;  msbs: dst=0 src0=1 src1=0 src2=0
.Ltmp590:
	.loc	1 1112 13                       ; mla.py:1112:13 @[ mla.py:1974:21 ]
	v_mul_f32_e32 v2, v22 /*v278*/, v130
	s_lshl_b32 s0, s28, 6
	v_add3_u32 v1, s0, s33, v1
	s_set_vgpr_msb 0x100                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_f32_e32 v0, v2, v0
.Ltmp591:
	.loc	1 1286 17                       ; mla.py:1286:17 @[ mla.py:1332:9 @[ mla.py:2029:9 ] ]
	s_clause 0x1
	global_store_b32 v1, v128, s[6:7] scale_offset
	global_store_b32 v1, v0, s[8:9] scale_offset
.Ltmp592:
.LBB0_48:                               ; %common.ret
	.loc	1 0 0 is_stmt 0                 ; mla.py:0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Ltmp593:
.Lfunc_end0:
	.size	_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16, .Lfunc_end0-_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 144
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 653
		.amdhsa_next_free_sgpr 78
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
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
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.num_vgpr, 653
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.num_agpr, 0
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.numbered_sgpr, 78
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.num_named_barrier, 0
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.private_seg_size, 0
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.uses_vcc, 1
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.uses_flat_scratch, 0
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.has_dyn_sized_stack, 0
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.has_recursion, 0
	.set .L_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15272
; TotalNumSgprs: 80
; NumVgprs: 653
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 40
; NumSGPRsForWavesPerEU: 80
; NumVGPRsForWavesPerEU: 653
; NamedBarCnt: 0
; Occupancy: 1
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
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
	.byte	5                               ; Abbreviation Code
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
	.byte	6                               ; Abbreviation Code
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
	.byte	7                               ; Abbreviation Code
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x287 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x261 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	2029                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1332                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x5c:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1602                            ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x71:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1822                            ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x7e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1863                            ; DW_AT_call_line
	.byte	33                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x8b:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges4                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1866                            ; DW_AT_call_line
	.byte	38                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x98:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges5                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1870                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xa5:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges6                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1871                            ; DW_AT_call_line
	.byte	5                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xb2:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp46                         ; DW_AT_low_pc
	.long	.Ltmp47-.Ltmp46                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1869                            ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xc7:0x23 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges7                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1951                            ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xd4:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp301                        ; DW_AT_low_pc
	.long	.Ltmp302-.Ltmp301               ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0xea:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges8                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1993                            ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xf7:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges9                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1875                            ; DW_AT_call_line
	.byte	42                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x104:0x23 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges10                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1889                            ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x111:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp63                         ; DW_AT_low_pc
	.long	.Ltmp64-.Ltmp63                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1008                            ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x127:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges11                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1937                            ; DW_AT_call_line
	.byte	29                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x134:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp66                         ; DW_AT_low_pc
	.long	.Ltmp67-.Ltmp66                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1894                            ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x149:0x15 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp67                         ; DW_AT_low_pc
	.long	.Ltmp68-.Ltmp67                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	1893                            ; DW_AT_call_line
	.byte	26                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x15e:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges12                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1895                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x16b:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges13                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1904                            ; DW_AT_call_line
	.byte	22                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x178:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges14                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1896                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x185:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges15                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1891                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x192:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges16                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1906                            ; DW_AT_call_line
	.byte	13                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x19f:0x28 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges17                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1909                            ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x1ac:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges18                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1101                            ; DW_AT_call_line
	.byte	30                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x1b9:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges19                ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	191                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x1c7:0x29 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges20                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1913                            ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x1d4:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges21                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1109                            ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x1e1:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges22                ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x1f0:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges23                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1938                            ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x1fd:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges24                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1960                            ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x20a:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges25                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1031                            ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x218:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges26                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1953                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x225:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges27                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1962                            ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x232:0x28 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges28                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1970                            ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x23f:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges29                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1101                            ; DW_AT_call_line
	.byte	30                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x24c:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges30                ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	191                             ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x25a:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges31                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1994                            ; DW_AT_call_line
	.byte	15                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x267:0x29 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges32                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1974                            ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x274:0x1b DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges33                ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	1109                            ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x281:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges34                ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp0-.Lfunc_begin0
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp587-.Lfunc_begin0
	.quad	.Ltmp589-.Lfunc_begin0
	.quad	.Ltmp591-.Lfunc_begin0
	.quad	.Ltmp592-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp0-.Lfunc_begin0
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp588-.Lfunc_begin0
	.quad	.Ltmp589-.Lfunc_begin0
	.quad	.Ltmp591-.Lfunc_begin0
	.quad	.Ltmp592-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
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
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp300-.Lfunc_begin0
	.quad	.Ltmp302-.Lfunc_begin0
	.quad	.Ltmp303-.Lfunc_begin0
	.quad	.Ltmp304-.Lfunc_begin0
	.quad	.Ltmp305-.Lfunc_begin0
	.quad	.Ltmp306-.Lfunc_begin0
	.quad	.Ltmp307-.Lfunc_begin0
	.quad	.Ltmp308-.Lfunc_begin0
	.quad	.Ltmp309-.Lfunc_begin0
	.quad	.Ltmp310-.Lfunc_begin0
	.quad	.Ltmp311-.Lfunc_begin0
	.quad	.Ltmp312-.Lfunc_begin0
	.quad	.Ltmp313-.Lfunc_begin0
	.quad	.Ltmp314-.Lfunc_begin0
	.quad	.Ltmp315-.Lfunc_begin0
	.quad	.Ltmp316-.Lfunc_begin0
	.quad	.Ltmp317-.Lfunc_begin0
	.quad	.Ltmp318-.Lfunc_begin0
	.quad	.Ltmp319-.Lfunc_begin0
	.quad	.Ltmp320-.Lfunc_begin0
	.quad	.Ltmp321-.Lfunc_begin0
	.quad	.Ltmp322-.Lfunc_begin0
	.quad	.Ltmp323-.Lfunc_begin0
	.quad	.Ltmp324-.Lfunc_begin0
	.quad	.Ltmp325-.Lfunc_begin0
	.quad	.Ltmp326-.Lfunc_begin0
	.quad	.Ltmp327-.Lfunc_begin0
	.quad	.Ltmp328-.Lfunc_begin0
	.quad	.Ltmp329-.Lfunc_begin0
	.quad	.Ltmp330-.Lfunc_begin0
	.quad	.Ltmp331-.Lfunc_begin0
	.quad	.Ltmp332-.Lfunc_begin0
	.quad	.Ltmp333-.Lfunc_begin0
	.quad	.Ltmp334-.Lfunc_begin0
	.quad	.Ltmp335-.Lfunc_begin0
	.quad	.Ltmp336-.Lfunc_begin0
	.quad	.Ltmp337-.Lfunc_begin0
	.quad	.Ltmp338-.Lfunc_begin0
	.quad	.Ltmp339-.Lfunc_begin0
	.quad	.Ltmp340-.Lfunc_begin0
	.quad	.Ltmp341-.Lfunc_begin0
	.quad	.Ltmp342-.Lfunc_begin0
	.quad	.Ltmp343-.Lfunc_begin0
	.quad	.Ltmp344-.Lfunc_begin0
	.quad	.Ltmp345-.Lfunc_begin0
	.quad	.Ltmp346-.Lfunc_begin0
	.quad	.Ltmp347-.Lfunc_begin0
	.quad	.Ltmp348-.Lfunc_begin0
	.quad	.Ltmp349-.Lfunc_begin0
	.quad	.Ltmp350-.Lfunc_begin0
	.quad	.Ltmp351-.Lfunc_begin0
	.quad	.Ltmp352-.Lfunc_begin0
	.quad	.Ltmp357-.Lfunc_begin0
	.quad	.Ltmp358-.Lfunc_begin0
	.quad	.Ltmp359-.Lfunc_begin0
	.quad	.Ltmp360-.Lfunc_begin0
	.quad	.Ltmp361-.Lfunc_begin0
	.quad	.Ltmp362-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges8:
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp367-.Lfunc_begin0
	.quad	.Ltmp368-.Lfunc_begin0
	.quad	.Ltmp435-.Lfunc_begin0
	.quad	.Ltmp436-.Lfunc_begin0
	.quad	.Ltmp437-.Lfunc_begin0
	.quad	.Ltmp438-.Lfunc_begin0
	.quad	.Ltmp439-.Lfunc_begin0
	.quad	.Ltmp440-.Lfunc_begin0
	.quad	.Ltmp441-.Lfunc_begin0
	.quad	.Ltmp442-.Lfunc_begin0
	.quad	.Ltmp443-.Lfunc_begin0
	.quad	.Ltmp444-.Lfunc_begin0
	.quad	.Ltmp447-.Lfunc_begin0
	.quad	.Ltmp448-.Lfunc_begin0
	.quad	.Ltmp449-.Lfunc_begin0
	.quad	.Ltmp450-.Lfunc_begin0
	.quad	.Ltmp451-.Lfunc_begin0
	.quad	.Ltmp452-.Lfunc_begin0
	.quad	.Ltmp453-.Lfunc_begin0
	.quad	.Ltmp454-.Lfunc_begin0
	.quad	.Ltmp455-.Lfunc_begin0
	.quad	.Ltmp456-.Lfunc_begin0
	.quad	.Ltmp467-.Lfunc_begin0
	.quad	.Ltmp468-.Lfunc_begin0
	.quad	.Ltmp482-.Lfunc_begin0
	.quad	.Ltmp483-.Lfunc_begin0
	.quad	.Ltmp484-.Lfunc_begin0
	.quad	.Ltmp485-.Lfunc_begin0
	.quad	.Ltmp494-.Lfunc_begin0
	.quad	.Ltmp495-.Lfunc_begin0
	.quad	.Ltmp499-.Lfunc_begin0
	.quad	.Ltmp500-.Lfunc_begin0
	.quad	.Ltmp508-.Lfunc_begin0
	.quad	.Ltmp509-.Lfunc_begin0
	.quad	.Ltmp511-.Lfunc_begin0
	.quad	.Ltmp512-.Lfunc_begin0
	.quad	.Ltmp514-.Lfunc_begin0
	.quad	.Ltmp515-.Lfunc_begin0
	.quad	.Ltmp518-.Lfunc_begin0
	.quad	.Ltmp519-.Lfunc_begin0
	.quad	.Ltmp527-.Lfunc_begin0
	.quad	.Ltmp528-.Lfunc_begin0
	.quad	.Ltmp530-.Lfunc_begin0
	.quad	.Ltmp531-.Lfunc_begin0
	.quad	.Ltmp533-.Lfunc_begin0
	.quad	.Ltmp534-.Lfunc_begin0
	.quad	.Ltmp537-.Lfunc_begin0
	.quad	.Ltmp538-.Lfunc_begin0
	.quad	.Ltmp547-.Lfunc_begin0
	.quad	.Ltmp548-.Lfunc_begin0
	.quad	.Ltmp549-.Lfunc_begin0
	.quad	.Ltmp550-.Lfunc_begin0
	.quad	.Ltmp552-.Lfunc_begin0
	.quad	.Ltmp553-.Lfunc_begin0
	.quad	.Ltmp554-.Lfunc_begin0
	.quad	.Ltmp555-.Lfunc_begin0
	.quad	.Ltmp556-.Lfunc_begin0
	.quad	.Ltmp557-.Lfunc_begin0
	.quad	.Ltmp561-.Lfunc_begin0
	.quad	.Ltmp562-.Lfunc_begin0
	.quad	.Ltmp566-.Lfunc_begin0
	.quad	.Ltmp567-.Lfunc_begin0
	.quad	.Ltmp569-.Lfunc_begin0
	.quad	.Ltmp570-.Lfunc_begin0
	.quad	.Ltmp571-.Lfunc_begin0
	.quad	.Ltmp572-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges9:
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges10:
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
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
	.quad	.Ltmp95-.Lfunc_begin0
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
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
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
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
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
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges11:
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp216-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp218-.Lfunc_begin0
	.quad	.Ltmp219-.Lfunc_begin0
	.quad	.Ltmp220-.Lfunc_begin0
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp226-.Lfunc_begin0
	.quad	.Ltmp260-.Lfunc_begin0
	.quad	.Ltmp261-.Lfunc_begin0
	.quad	.Ltmp264-.Lfunc_begin0
	.quad	.Ltmp265-.Lfunc_begin0
	.quad	.Ltmp266-.Lfunc_begin0
	.quad	.Ltmp267-.Lfunc_begin0
	.quad	.Ltmp268-.Lfunc_begin0
	.quad	.Ltmp269-.Lfunc_begin0
	.quad	.Ltmp270-.Lfunc_begin0
	.quad	.Ltmp271-.Lfunc_begin0
	.quad	.Ltmp272-.Lfunc_begin0
	.quad	.Ltmp273-.Lfunc_begin0
	.quad	.Ltmp274-.Lfunc_begin0
	.quad	.Ltmp275-.Lfunc_begin0
	.quad	.Ltmp276-.Lfunc_begin0
	.quad	.Ltmp277-.Lfunc_begin0
	.quad	.Ltmp278-.Lfunc_begin0
	.quad	.Ltmp279-.Lfunc_begin0
	.quad	.Ltmp282-.Lfunc_begin0
	.quad	.Ltmp283-.Lfunc_begin0
	.quad	.Ltmp287-.Lfunc_begin0
	.quad	.Ltmp288-.Lfunc_begin0
	.quad	.Ltmp289-.Lfunc_begin0
	.quad	.Ltmp290-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges12:
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp190-.Lfunc_begin0
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges13:
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	.Ltmp198-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	.Ltmp209-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges14:
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp193-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp196-.Lfunc_begin0
	.quad	.Ltmp197-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges15:
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
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
	.quad	.Ltmp95-.Lfunc_begin0
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
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	.Ltmp121-.Lfunc_begin0
	.quad	.Ltmp122-.Lfunc_begin0
	.quad	.Ltmp123-.Lfunc_begin0
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
	.quad	.Ltmp159-.Lfunc_begin0
	.quad	.Ltmp160-.Lfunc_begin0
	.quad	.Ltmp161-.Lfunc_begin0
	.quad	.Ltmp162-.Lfunc_begin0
	.quad	.Ltmp163-.Lfunc_begin0
	.quad	.Ltmp164-.Lfunc_begin0
	.quad	.Ltmp165-.Lfunc_begin0
	.quad	.Ltmp166-.Lfunc_begin0
	.quad	.Ltmp167-.Lfunc_begin0
	.quad	.Ltmp168-.Lfunc_begin0
	.quad	.Ltmp169-.Lfunc_begin0
	.quad	.Ltmp170-.Lfunc_begin0
	.quad	.Ltmp171-.Lfunc_begin0
	.quad	.Ltmp172-.Lfunc_begin0
	.quad	.Ltmp173-.Lfunc_begin0
	.quad	.Ltmp174-.Lfunc_begin0
	.quad	.Ltmp175-.Lfunc_begin0
	.quad	.Ltmp176-.Lfunc_begin0
	.quad	.Ltmp177-.Lfunc_begin0
	.quad	.Ltmp178-.Lfunc_begin0
	.quad	.Ltmp179-.Lfunc_begin0
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
	.quad	.Ltmp191-.Lfunc_begin0
	.quad	.Ltmp192-.Lfunc_begin0
	.quad	.Ltmp194-.Lfunc_begin0
	.quad	.Ltmp195-.Lfunc_begin0
	.quad	.Ltmp199-.Lfunc_begin0
	.quad	.Ltmp200-.Lfunc_begin0
	.quad	.Ltmp201-.Lfunc_begin0
	.quad	.Ltmp202-.Lfunc_begin0
	.quad	.Ltmp203-.Lfunc_begin0
	.quad	.Ltmp204-.Lfunc_begin0
	.quad	.Ltmp205-.Lfunc_begin0
	.quad	.Ltmp206-.Lfunc_begin0
	.quad	.Ltmp207-.Lfunc_begin0
	.quad	.Ltmp208-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges16:
	.quad	.Ltmp210-.Lfunc_begin0
	.quad	.Ltmp211-.Lfunc_begin0
	.quad	.Ltmp212-.Lfunc_begin0
	.quad	.Ltmp213-.Lfunc_begin0
	.quad	.Ltmp214-.Lfunc_begin0
	.quad	.Ltmp215-.Lfunc_begin0
	.quad	.Ltmp216-.Lfunc_begin0
	.quad	.Ltmp217-.Lfunc_begin0
	.quad	.Ltmp218-.Lfunc_begin0
	.quad	.Ltmp219-.Lfunc_begin0
	.quad	.Ltmp221-.Lfunc_begin0
	.quad	.Ltmp222-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp225-.Lfunc_begin0
	.quad	.Ltmp227-.Lfunc_begin0
	.quad	.Ltmp228-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges17:
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	.Ltmp234-.Lfunc_begin0
	.quad	.Ltmp235-.Lfunc_begin0
	.quad	.Ltmp236-.Lfunc_begin0
	.quad	.Ltmp237-.Lfunc_begin0
	.quad	.Ltmp238-.Lfunc_begin0
	.quad	.Ltmp239-.Lfunc_begin0
	.quad	.Ltmp240-.Lfunc_begin0
	.quad	.Ltmp241-.Lfunc_begin0
	.quad	.Ltmp242-.Lfunc_begin0
	.quad	.Ltmp243-.Lfunc_begin0
	.quad	.Ltmp244-.Lfunc_begin0
	.quad	.Ltmp245-.Lfunc_begin0
	.quad	.Ltmp246-.Lfunc_begin0
	.quad	.Ltmp247-.Lfunc_begin0
	.quad	.Ltmp248-.Lfunc_begin0
	.quad	.Ltmp249-.Lfunc_begin0
	.quad	.Ltmp250-.Lfunc_begin0
	.quad	.Ltmp253-.Lfunc_begin0
	.quad	.Ltmp254-.Lfunc_begin0
	.quad	.Ltmp255-.Lfunc_begin0
	.quad	.Ltmp256-.Lfunc_begin0
	.quad	.Ltmp257-.Lfunc_begin0
	.quad	.Ltmp258-.Lfunc_begin0
	.quad	.Ltmp259-.Lfunc_begin0
	.quad	.Ltmp260-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges18:
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	.Ltmp233-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges19:
	.quad	.Ltmp223-.Lfunc_begin0
	.quad	.Ltmp224-.Lfunc_begin0
	.quad	.Ltmp229-.Lfunc_begin0
	.quad	.Ltmp230-.Lfunc_begin0
	.quad	.Ltmp231-.Lfunc_begin0
	.quad	.Ltmp232-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges20:
	.quad	.Ltmp234-.Lfunc_begin0
	.quad	.Ltmp235-.Lfunc_begin0
	.quad	.Ltmp236-.Lfunc_begin0
	.quad	.Ltmp237-.Lfunc_begin0
	.quad	.Ltmp238-.Lfunc_begin0
	.quad	.Ltmp239-.Lfunc_begin0
	.quad	.Ltmp240-.Lfunc_begin0
	.quad	.Ltmp241-.Lfunc_begin0
	.quad	.Ltmp251-.Lfunc_begin0
	.quad	.Ltmp252-.Lfunc_begin0
	.quad	.Ltmp262-.Lfunc_begin0
	.quad	.Ltmp263-.Lfunc_begin0
	.quad	.Ltmp280-.Lfunc_begin0
	.quad	.Ltmp281-.Lfunc_begin0
	.quad	.Ltmp283-.Lfunc_begin0
	.quad	.Ltmp284-.Lfunc_begin0
	.quad	.Ltmp285-.Lfunc_begin0
	.quad	.Ltmp286-.Lfunc_begin0
	.quad	.Ltmp288-.Lfunc_begin0
	.quad	.Ltmp289-.Lfunc_begin0
	.quad	.Ltmp290-.Lfunc_begin0
	.quad	.Ltmp291-.Lfunc_begin0
	.quad	.Ltmp292-.Lfunc_begin0
	.quad	.Ltmp295-.Lfunc_begin0
	.quad	.Ltmp296-.Lfunc_begin0
	.quad	.Ltmp298-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges21:
	.quad	.Ltmp262-.Lfunc_begin0
	.quad	.Ltmp263-.Lfunc_begin0
	.quad	.Ltmp280-.Lfunc_begin0
	.quad	.Ltmp281-.Lfunc_begin0
	.quad	.Ltmp283-.Lfunc_begin0
	.quad	.Ltmp284-.Lfunc_begin0
	.quad	.Ltmp285-.Lfunc_begin0
	.quad	.Ltmp286-.Lfunc_begin0
	.quad	.Ltmp288-.Lfunc_begin0
	.quad	.Ltmp289-.Lfunc_begin0
	.quad	.Ltmp290-.Lfunc_begin0
	.quad	.Ltmp291-.Lfunc_begin0
	.quad	.Ltmp292-.Lfunc_begin0
	.quad	.Ltmp294-.Lfunc_begin0
	.quad	.Ltmp296-.Lfunc_begin0
	.quad	.Ltmp297-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges22:
	.quad	.Ltmp292-.Lfunc_begin0
	.quad	.Ltmp293-.Lfunc_begin0
	.quad	.Ltmp296-.Lfunc_begin0
	.quad	.Ltmp297-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges23:
	.quad	.Ltmp242-.Lfunc_begin0
	.quad	.Ltmp243-.Lfunc_begin0
	.quad	.Ltmp244-.Lfunc_begin0
	.quad	.Ltmp245-.Lfunc_begin0
	.quad	.Ltmp246-.Lfunc_begin0
	.quad	.Ltmp247-.Lfunc_begin0
	.quad	.Ltmp248-.Lfunc_begin0
	.quad	.Ltmp249-.Lfunc_begin0
	.quad	.Ltmp250-.Lfunc_begin0
	.quad	.Ltmp251-.Lfunc_begin0
	.quad	.Ltmp252-.Lfunc_begin0
	.quad	.Ltmp253-.Lfunc_begin0
	.quad	.Ltmp254-.Lfunc_begin0
	.quad	.Ltmp255-.Lfunc_begin0
	.quad	.Ltmp256-.Lfunc_begin0
	.quad	.Ltmp257-.Lfunc_begin0
	.quad	.Ltmp258-.Lfunc_begin0
	.quad	.Ltmp259-.Lfunc_begin0
	.quad	.Ltmp261-.Lfunc_begin0
	.quad	.Ltmp262-.Lfunc_begin0
	.quad	.Ltmp263-.Lfunc_begin0
	.quad	.Ltmp264-.Lfunc_begin0
	.quad	.Ltmp265-.Lfunc_begin0
	.quad	.Ltmp266-.Lfunc_begin0
	.quad	.Ltmp267-.Lfunc_begin0
	.quad	.Ltmp268-.Lfunc_begin0
	.quad	.Ltmp269-.Lfunc_begin0
	.quad	.Ltmp270-.Lfunc_begin0
	.quad	.Ltmp271-.Lfunc_begin0
	.quad	.Ltmp272-.Lfunc_begin0
	.quad	.Ltmp273-.Lfunc_begin0
	.quad	.Ltmp274-.Lfunc_begin0
	.quad	.Ltmp275-.Lfunc_begin0
	.quad	.Ltmp276-.Lfunc_begin0
	.quad	.Ltmp277-.Lfunc_begin0
	.quad	.Ltmp278-.Lfunc_begin0
	.quad	.Ltmp279-.Lfunc_begin0
	.quad	.Ltmp280-.Lfunc_begin0
	.quad	.Ltmp281-.Lfunc_begin0
	.quad	.Ltmp282-.Lfunc_begin0
	.quad	.Ltmp284-.Lfunc_begin0
	.quad	.Ltmp285-.Lfunc_begin0
	.quad	.Ltmp286-.Lfunc_begin0
	.quad	.Ltmp287-.Lfunc_begin0
	.quad	.Ltmp291-.Lfunc_begin0
	.quad	.Ltmp292-.Lfunc_begin0
	.quad	.Ltmp295-.Lfunc_begin0
	.quad	.Ltmp296-.Lfunc_begin0
	.quad	.Ltmp298-.Lfunc_begin0
	.quad	.Ltmp299-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges24:
	.quad	.Ltmp302-.Lfunc_begin0
	.quad	.Ltmp303-.Lfunc_begin0
	.quad	.Ltmp304-.Lfunc_begin0
	.quad	.Ltmp305-.Lfunc_begin0
	.quad	.Ltmp306-.Lfunc_begin0
	.quad	.Ltmp307-.Lfunc_begin0
	.quad	.Ltmp308-.Lfunc_begin0
	.quad	.Ltmp309-.Lfunc_begin0
	.quad	.Ltmp310-.Lfunc_begin0
	.quad	.Ltmp311-.Lfunc_begin0
	.quad	.Ltmp362-.Lfunc_begin0
	.quad	.Ltmp363-.Lfunc_begin0
	.quad	.Ltmp364-.Lfunc_begin0
	.quad	.Ltmp365-.Lfunc_begin0
	.quad	.Ltmp366-.Lfunc_begin0
	.quad	.Ltmp367-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges25:
	.quad	.Ltmp302-.Lfunc_begin0
	.quad	.Ltmp303-.Lfunc_begin0
	.quad	.Ltmp304-.Lfunc_begin0
	.quad	.Ltmp305-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges26:
	.quad	.Ltmp312-.Lfunc_begin0
	.quad	.Ltmp313-.Lfunc_begin0
	.quad	.Ltmp314-.Lfunc_begin0
	.quad	.Ltmp315-.Lfunc_begin0
	.quad	.Ltmp316-.Lfunc_begin0
	.quad	.Ltmp317-.Lfunc_begin0
	.quad	.Ltmp318-.Lfunc_begin0
	.quad	.Ltmp319-.Lfunc_begin0
	.quad	.Ltmp320-.Lfunc_begin0
	.quad	.Ltmp321-.Lfunc_begin0
	.quad	.Ltmp322-.Lfunc_begin0
	.quad	.Ltmp323-.Lfunc_begin0
	.quad	.Ltmp324-.Lfunc_begin0
	.quad	.Ltmp325-.Lfunc_begin0
	.quad	.Ltmp326-.Lfunc_begin0
	.quad	.Ltmp327-.Lfunc_begin0
	.quad	.Ltmp328-.Lfunc_begin0
	.quad	.Ltmp329-.Lfunc_begin0
	.quad	.Ltmp330-.Lfunc_begin0
	.quad	.Ltmp331-.Lfunc_begin0
	.quad	.Ltmp332-.Lfunc_begin0
	.quad	.Ltmp333-.Lfunc_begin0
	.quad	.Ltmp334-.Lfunc_begin0
	.quad	.Ltmp335-.Lfunc_begin0
	.quad	.Ltmp336-.Lfunc_begin0
	.quad	.Ltmp337-.Lfunc_begin0
	.quad	.Ltmp338-.Lfunc_begin0
	.quad	.Ltmp339-.Lfunc_begin0
	.quad	.Ltmp340-.Lfunc_begin0
	.quad	.Ltmp341-.Lfunc_begin0
	.quad	.Ltmp342-.Lfunc_begin0
	.quad	.Ltmp343-.Lfunc_begin0
	.quad	.Ltmp344-.Lfunc_begin0
	.quad	.Ltmp345-.Lfunc_begin0
	.quad	.Ltmp346-.Lfunc_begin0
	.quad	.Ltmp347-.Lfunc_begin0
	.quad	.Ltmp348-.Lfunc_begin0
	.quad	.Ltmp349-.Lfunc_begin0
	.quad	.Ltmp350-.Lfunc_begin0
	.quad	.Ltmp351-.Lfunc_begin0
	.quad	.Ltmp352-.Lfunc_begin0
	.quad	.Ltmp353-.Lfunc_begin0
	.quad	.Ltmp354-.Lfunc_begin0
	.quad	.Ltmp355-.Lfunc_begin0
	.quad	.Ltmp356-.Lfunc_begin0
	.quad	.Ltmp357-.Lfunc_begin0
	.quad	.Ltmp358-.Lfunc_begin0
	.quad	.Ltmp359-.Lfunc_begin0
	.quad	.Ltmp360-.Lfunc_begin0
	.quad	.Ltmp361-.Lfunc_begin0
	.quad	.Ltmp363-.Lfunc_begin0
	.quad	.Ltmp364-.Lfunc_begin0
	.quad	.Ltmp365-.Lfunc_begin0
	.quad	.Ltmp366-.Lfunc_begin0
	.quad	.Ltmp368-.Lfunc_begin0
	.quad	.Ltmp369-.Lfunc_begin0
	.quad	.Ltmp370-.Lfunc_begin0
	.quad	.Ltmp371-.Lfunc_begin0
	.quad	.Ltmp372-.Lfunc_begin0
	.quad	.Ltmp373-.Lfunc_begin0
	.quad	.Ltmp374-.Lfunc_begin0
	.quad	.Ltmp375-.Lfunc_begin0
	.quad	.Ltmp376-.Lfunc_begin0
	.quad	.Ltmp377-.Lfunc_begin0
	.quad	.Ltmp378-.Lfunc_begin0
	.quad	.Ltmp379-.Lfunc_begin0
	.quad	.Ltmp380-.Lfunc_begin0
	.quad	.Ltmp381-.Lfunc_begin0
	.quad	.Ltmp382-.Lfunc_begin0
	.quad	.Ltmp383-.Lfunc_begin0
	.quad	.Ltmp384-.Lfunc_begin0
	.quad	.Ltmp385-.Lfunc_begin0
	.quad	.Ltmp386-.Lfunc_begin0
	.quad	.Ltmp387-.Lfunc_begin0
	.quad	.Ltmp388-.Lfunc_begin0
	.quad	.Ltmp389-.Lfunc_begin0
	.quad	.Ltmp390-.Lfunc_begin0
	.quad	.Ltmp391-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges27:
	.quad	.Ltmp387-.Lfunc_begin0
	.quad	.Ltmp388-.Lfunc_begin0
	.quad	.Ltmp389-.Lfunc_begin0
	.quad	.Ltmp390-.Lfunc_begin0
	.quad	.Ltmp392-.Lfunc_begin0
	.quad	.Ltmp393-.Lfunc_begin0
	.quad	.Ltmp394-.Lfunc_begin0
	.quad	.Ltmp395-.Lfunc_begin0
	.quad	.Ltmp398-.Lfunc_begin0
	.quad	.Ltmp399-.Lfunc_begin0
	.quad	.Ltmp400-.Lfunc_begin0
	.quad	.Ltmp401-.Lfunc_begin0
	.quad	.Ltmp402-.Lfunc_begin0
	.quad	.Ltmp403-.Lfunc_begin0
	.quad	.Ltmp406-.Lfunc_begin0
	.quad	.Ltmp407-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges28:
	.quad	.Ltmp396-.Lfunc_begin0
	.quad	.Ltmp397-.Lfunc_begin0
	.quad	.Ltmp404-.Lfunc_begin0
	.quad	.Ltmp405-.Lfunc_begin0
	.quad	.Ltmp407-.Lfunc_begin0
	.quad	.Ltmp408-.Lfunc_begin0
	.quad	.Ltmp409-.Lfunc_begin0
	.quad	.Ltmp411-.Lfunc_begin0
	.quad	.Ltmp412-.Lfunc_begin0
	.quad	.Ltmp413-.Lfunc_begin0
	.quad	.Ltmp414-.Lfunc_begin0
	.quad	.Ltmp415-.Lfunc_begin0
	.quad	.Ltmp416-.Lfunc_begin0
	.quad	.Ltmp417-.Lfunc_begin0
	.quad	.Ltmp418-.Lfunc_begin0
	.quad	.Ltmp419-.Lfunc_begin0
	.quad	.Ltmp420-.Lfunc_begin0
	.quad	.Ltmp421-.Lfunc_begin0
	.quad	.Ltmp422-.Lfunc_begin0
	.quad	.Ltmp423-.Lfunc_begin0
	.quad	.Ltmp424-.Lfunc_begin0
	.quad	.Ltmp425-.Lfunc_begin0
	.quad	.Ltmp426-.Lfunc_begin0
	.quad	.Ltmp427-.Lfunc_begin0
	.quad	.Ltmp428-.Lfunc_begin0
	.quad	.Ltmp429-.Lfunc_begin0
	.quad	.Ltmp430-.Lfunc_begin0
	.quad	.Ltmp431-.Lfunc_begin0
	.quad	.Ltmp432-.Lfunc_begin0
	.quad	.Ltmp435-.Lfunc_begin0
	.quad	.Ltmp436-.Lfunc_begin0
	.quad	.Ltmp437-.Lfunc_begin0
	.quad	.Ltmp438-.Lfunc_begin0
	.quad	.Ltmp439-.Lfunc_begin0
	.quad	.Ltmp440-.Lfunc_begin0
	.quad	.Ltmp441-.Lfunc_begin0
	.quad	.Ltmp442-.Lfunc_begin0
	.quad	.Ltmp443-.Lfunc_begin0
	.quad	.Ltmp444-.Lfunc_begin0
	.quad	.Ltmp445-.Lfunc_begin0
	.quad	.Ltmp446-.Lfunc_begin0
	.quad	.Ltmp447-.Lfunc_begin0
	.quad	.Ltmp448-.Lfunc_begin0
	.quad	.Ltmp449-.Lfunc_begin0
	.quad	.Ltmp450-.Lfunc_begin0
	.quad	.Ltmp451-.Lfunc_begin0
	.quad	.Ltmp452-.Lfunc_begin0
	.quad	.Ltmp453-.Lfunc_begin0
	.quad	.Ltmp454-.Lfunc_begin0
	.quad	.Ltmp455-.Lfunc_begin0
	.quad	.Ltmp456-.Lfunc_begin0
	.quad	.Ltmp457-.Lfunc_begin0
	.quad	.Ltmp458-.Lfunc_begin0
	.quad	.Ltmp459-.Lfunc_begin0
	.quad	.Ltmp460-.Lfunc_begin0
	.quad	.Ltmp461-.Lfunc_begin0
	.quad	.Ltmp462-.Lfunc_begin0
	.quad	.Ltmp463-.Lfunc_begin0
	.quad	.Ltmp464-.Lfunc_begin0
	.quad	.Ltmp465-.Lfunc_begin0
	.quad	.Ltmp466-.Lfunc_begin0
	.quad	.Ltmp467-.Lfunc_begin0
	.quad	.Ltmp468-.Lfunc_begin0
	.quad	.Ltmp469-.Lfunc_begin0
	.quad	.Ltmp470-.Lfunc_begin0
	.quad	.Ltmp471-.Lfunc_begin0
	.quad	.Ltmp472-.Lfunc_begin0
	.quad	.Ltmp473-.Lfunc_begin0
	.quad	.Ltmp474-.Lfunc_begin0
	.quad	.Ltmp475-.Lfunc_begin0
	.quad	.Ltmp476-.Lfunc_begin0
	.quad	.Ltmp477-.Lfunc_begin0
	.quad	.Ltmp478-.Lfunc_begin0
	.quad	.Ltmp479-.Lfunc_begin0
	.quad	.Ltmp480-.Lfunc_begin0
	.quad	.Ltmp481-.Lfunc_begin0
	.quad	.Ltmp483-.Lfunc_begin0
	.quad	.Ltmp484-.Lfunc_begin0
	.quad	.Ltmp485-.Lfunc_begin0
	.quad	.Ltmp486-.Lfunc_begin0
	.quad	.Ltmp487-.Lfunc_begin0
	.quad	.Ltmp488-.Lfunc_begin0
	.quad	.Ltmp489-.Lfunc_begin0
	.quad	.Ltmp490-.Lfunc_begin0
	.quad	.Ltmp491-.Lfunc_begin0
	.quad	.Ltmp492-.Lfunc_begin0
	.quad	.Ltmp493-.Lfunc_begin0
	.quad	.Ltmp494-.Lfunc_begin0
	.quad	.Ltmp495-.Lfunc_begin0
	.quad	.Ltmp496-.Lfunc_begin0
	.quad	.Ltmp497-.Lfunc_begin0
	.quad	.Ltmp498-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges29:
	.quad	.Ltmp396-.Lfunc_begin0
	.quad	.Ltmp397-.Lfunc_begin0
	.quad	.Ltmp404-.Lfunc_begin0
	.quad	.Ltmp405-.Lfunc_begin0
	.quad	.Ltmp407-.Lfunc_begin0
	.quad	.Ltmp408-.Lfunc_begin0
	.quad	.Ltmp409-.Lfunc_begin0
	.quad	.Ltmp411-.Lfunc_begin0
	.quad	.Ltmp412-.Lfunc_begin0
	.quad	.Ltmp413-.Lfunc_begin0
	.quad	.Ltmp414-.Lfunc_begin0
	.quad	.Ltmp415-.Lfunc_begin0
	.quad	.Ltmp416-.Lfunc_begin0
	.quad	.Ltmp417-.Lfunc_begin0
	.quad	.Ltmp418-.Lfunc_begin0
	.quad	.Ltmp419-.Lfunc_begin0
	.quad	.Ltmp420-.Lfunc_begin0
	.quad	.Ltmp421-.Lfunc_begin0
	.quad	.Ltmp422-.Lfunc_begin0
	.quad	.Ltmp423-.Lfunc_begin0
	.quad	.Ltmp424-.Lfunc_begin0
	.quad	.Ltmp425-.Lfunc_begin0
	.quad	.Ltmp426-.Lfunc_begin0
	.quad	.Ltmp427-.Lfunc_begin0
	.quad	.Ltmp428-.Lfunc_begin0
	.quad	.Ltmp429-.Lfunc_begin0
	.quad	.Ltmp430-.Lfunc_begin0
	.quad	.Ltmp431-.Lfunc_begin0
	.quad	.Ltmp432-.Lfunc_begin0
	.quad	.Ltmp434-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges30:
	.quad	.Ltmp396-.Lfunc_begin0
	.quad	.Ltmp397-.Lfunc_begin0
	.quad	.Ltmp404-.Lfunc_begin0
	.quad	.Ltmp405-.Lfunc_begin0
	.quad	.Ltmp407-.Lfunc_begin0
	.quad	.Ltmp408-.Lfunc_begin0
	.quad	.Ltmp409-.Lfunc_begin0
	.quad	.Ltmp410-.Lfunc_begin0
	.quad	.Ltmp412-.Lfunc_begin0
	.quad	.Ltmp413-.Lfunc_begin0
	.quad	.Ltmp414-.Lfunc_begin0
	.quad	.Ltmp415-.Lfunc_begin0
	.quad	.Ltmp416-.Lfunc_begin0
	.quad	.Ltmp417-.Lfunc_begin0
	.quad	.Ltmp418-.Lfunc_begin0
	.quad	.Ltmp419-.Lfunc_begin0
	.quad	.Ltmp420-.Lfunc_begin0
	.quad	.Ltmp421-.Lfunc_begin0
	.quad	.Ltmp422-.Lfunc_begin0
	.quad	.Ltmp423-.Lfunc_begin0
	.quad	.Ltmp424-.Lfunc_begin0
	.quad	.Ltmp425-.Lfunc_begin0
	.quad	.Ltmp426-.Lfunc_begin0
	.quad	.Ltmp427-.Lfunc_begin0
	.quad	.Ltmp428-.Lfunc_begin0
	.quad	.Ltmp429-.Lfunc_begin0
	.quad	.Ltmp430-.Lfunc_begin0
	.quad	.Ltmp431-.Lfunc_begin0
	.quad	.Ltmp432-.Lfunc_begin0
	.quad	.Ltmp433-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges31:
	.quad	.Ltmp445-.Lfunc_begin0
	.quad	.Ltmp446-.Lfunc_begin0
	.quad	.Ltmp457-.Lfunc_begin0
	.quad	.Ltmp458-.Lfunc_begin0
	.quad	.Ltmp459-.Lfunc_begin0
	.quad	.Ltmp460-.Lfunc_begin0
	.quad	.Ltmp461-.Lfunc_begin0
	.quad	.Ltmp462-.Lfunc_begin0
	.quad	.Ltmp463-.Lfunc_begin0
	.quad	.Ltmp464-.Lfunc_begin0
	.quad	.Ltmp465-.Lfunc_begin0
	.quad	.Ltmp466-.Lfunc_begin0
	.quad	.Ltmp469-.Lfunc_begin0
	.quad	.Ltmp470-.Lfunc_begin0
	.quad	.Ltmp471-.Lfunc_begin0
	.quad	.Ltmp472-.Lfunc_begin0
	.quad	.Ltmp481-.Lfunc_begin0
	.quad	.Ltmp482-.Lfunc_begin0
	.quad	.Ltmp486-.Lfunc_begin0
	.quad	.Ltmp487-.Lfunc_begin0
	.quad	.Ltmp488-.Lfunc_begin0
	.quad	.Ltmp489-.Lfunc_begin0
	.quad	.Ltmp490-.Lfunc_begin0
	.quad	.Ltmp491-.Lfunc_begin0
	.quad	.Ltmp492-.Lfunc_begin0
	.quad	.Ltmp493-.Lfunc_begin0
	.quad	.Ltmp496-.Lfunc_begin0
	.quad	.Ltmp497-.Lfunc_begin0
	.quad	.Ltmp498-.Lfunc_begin0
	.quad	.Ltmp499-.Lfunc_begin0
	.quad	.Ltmp501-.Lfunc_begin0
	.quad	.Ltmp502-.Lfunc_begin0
	.quad	.Ltmp503-.Lfunc_begin0
	.quad	.Ltmp504-.Lfunc_begin0
	.quad	.Ltmp505-.Lfunc_begin0
	.quad	.Ltmp506-.Lfunc_begin0
	.quad	.Ltmp507-.Lfunc_begin0
	.quad	.Ltmp508-.Lfunc_begin0
	.quad	.Ltmp510-.Lfunc_begin0
	.quad	.Ltmp511-.Lfunc_begin0
	.quad	.Ltmp513-.Lfunc_begin0
	.quad	.Ltmp514-.Lfunc_begin0
	.quad	.Ltmp515-.Lfunc_begin0
	.quad	.Ltmp516-.Lfunc_begin0
	.quad	.Ltmp517-.Lfunc_begin0
	.quad	.Ltmp518-.Lfunc_begin0
	.quad	.Ltmp520-.Lfunc_begin0
	.quad	.Ltmp521-.Lfunc_begin0
	.quad	.Ltmp522-.Lfunc_begin0
	.quad	.Ltmp523-.Lfunc_begin0
	.quad	.Ltmp524-.Lfunc_begin0
	.quad	.Ltmp525-.Lfunc_begin0
	.quad	.Ltmp526-.Lfunc_begin0
	.quad	.Ltmp527-.Lfunc_begin0
	.quad	.Ltmp529-.Lfunc_begin0
	.quad	.Ltmp530-.Lfunc_begin0
	.quad	.Ltmp532-.Lfunc_begin0
	.quad	.Ltmp533-.Lfunc_begin0
	.quad	.Ltmp534-.Lfunc_begin0
	.quad	.Ltmp535-.Lfunc_begin0
	.quad	.Ltmp536-.Lfunc_begin0
	.quad	.Ltmp537-.Lfunc_begin0
	.quad	.Ltmp539-.Lfunc_begin0
	.quad	.Ltmp540-.Lfunc_begin0
	.quad	.Ltmp541-.Lfunc_begin0
	.quad	.Ltmp542-.Lfunc_begin0
	.quad	.Ltmp544-.Lfunc_begin0
	.quad	.Ltmp545-.Lfunc_begin0
	.quad	.Ltmp546-.Lfunc_begin0
	.quad	.Ltmp547-.Lfunc_begin0
	.quad	.Ltmp551-.Lfunc_begin0
	.quad	.Ltmp552-.Lfunc_begin0
	.quad	.Ltmp555-.Lfunc_begin0
	.quad	.Ltmp556-.Lfunc_begin0
	.quad	.Ltmp557-.Lfunc_begin0
	.quad	.Ltmp558-.Lfunc_begin0
	.quad	.Ltmp559-.Lfunc_begin0
	.quad	.Ltmp560-.Lfunc_begin0
	.quad	.Ltmp562-.Lfunc_begin0
	.quad	.Ltmp563-.Lfunc_begin0
	.quad	.Ltmp564-.Lfunc_begin0
	.quad	.Ltmp565-.Lfunc_begin0
	.quad	.Ltmp567-.Lfunc_begin0
	.quad	.Ltmp568-.Lfunc_begin0
	.quad	.Ltmp570-.Lfunc_begin0
	.quad	.Ltmp571-.Lfunc_begin0
	.quad	.Ltmp573-.Lfunc_begin0
	.quad	.Ltmp574-.Lfunc_begin0
	.quad	.Ltmp578-.Lfunc_begin0
	.quad	.Ltmp579-.Lfunc_begin0
	.quad	.Ltmp582-.Lfunc_begin0
	.quad	.Ltmp583-.Lfunc_begin0
	.quad	.Ltmp585-.Lfunc_begin0
	.quad	.Ltmp586-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges32:
	.quad	.Ltmp473-.Lfunc_begin0
	.quad	.Ltmp474-.Lfunc_begin0
	.quad	.Ltmp475-.Lfunc_begin0
	.quad	.Ltmp476-.Lfunc_begin0
	.quad	.Ltmp477-.Lfunc_begin0
	.quad	.Ltmp478-.Lfunc_begin0
	.quad	.Ltmp479-.Lfunc_begin0
	.quad	.Ltmp480-.Lfunc_begin0
	.quad	.Ltmp500-.Lfunc_begin0
	.quad	.Ltmp501-.Lfunc_begin0
	.quad	.Ltmp502-.Lfunc_begin0
	.quad	.Ltmp503-.Lfunc_begin0
	.quad	.Ltmp504-.Lfunc_begin0
	.quad	.Ltmp505-.Lfunc_begin0
	.quad	.Ltmp506-.Lfunc_begin0
	.quad	.Ltmp507-.Lfunc_begin0
	.quad	.Ltmp509-.Lfunc_begin0
	.quad	.Ltmp510-.Lfunc_begin0
	.quad	.Ltmp512-.Lfunc_begin0
	.quad	.Ltmp513-.Lfunc_begin0
	.quad	.Ltmp516-.Lfunc_begin0
	.quad	.Ltmp517-.Lfunc_begin0
	.quad	.Ltmp519-.Lfunc_begin0
	.quad	.Ltmp520-.Lfunc_begin0
	.quad	.Ltmp521-.Lfunc_begin0
	.quad	.Ltmp522-.Lfunc_begin0
	.quad	.Ltmp523-.Lfunc_begin0
	.quad	.Ltmp524-.Lfunc_begin0
	.quad	.Ltmp525-.Lfunc_begin0
	.quad	.Ltmp526-.Lfunc_begin0
	.quad	.Ltmp528-.Lfunc_begin0
	.quad	.Ltmp529-.Lfunc_begin0
	.quad	.Ltmp531-.Lfunc_begin0
	.quad	.Ltmp532-.Lfunc_begin0
	.quad	.Ltmp535-.Lfunc_begin0
	.quad	.Ltmp536-.Lfunc_begin0
	.quad	.Ltmp538-.Lfunc_begin0
	.quad	.Ltmp539-.Lfunc_begin0
	.quad	.Ltmp540-.Lfunc_begin0
	.quad	.Ltmp541-.Lfunc_begin0
	.quad	.Ltmp542-.Lfunc_begin0
	.quad	.Ltmp544-.Lfunc_begin0
	.quad	.Ltmp545-.Lfunc_begin0
	.quad	.Ltmp546-.Lfunc_begin0
	.quad	.Ltmp548-.Lfunc_begin0
	.quad	.Ltmp549-.Lfunc_begin0
	.quad	.Ltmp550-.Lfunc_begin0
	.quad	.Ltmp551-.Lfunc_begin0
	.quad	.Ltmp553-.Lfunc_begin0
	.quad	.Ltmp554-.Lfunc_begin0
	.quad	.Ltmp558-.Lfunc_begin0
	.quad	.Ltmp559-.Lfunc_begin0
	.quad	.Ltmp560-.Lfunc_begin0
	.quad	.Ltmp561-.Lfunc_begin0
	.quad	.Ltmp563-.Lfunc_begin0
	.quad	.Ltmp564-.Lfunc_begin0
	.quad	.Ltmp565-.Lfunc_begin0
	.quad	.Ltmp566-.Lfunc_begin0
	.quad	.Ltmp568-.Lfunc_begin0
	.quad	.Ltmp569-.Lfunc_begin0
	.quad	.Ltmp572-.Lfunc_begin0
	.quad	.Ltmp573-.Lfunc_begin0
	.quad	.Ltmp574-.Lfunc_begin0
	.quad	.Ltmp578-.Lfunc_begin0
	.quad	.Ltmp579-.Lfunc_begin0
	.quad	.Ltmp581-.Lfunc_begin0
	.quad	.Ltmp583-.Lfunc_begin0
	.quad	.Ltmp584-.Lfunc_begin0
	.quad	.Ltmp586-.Lfunc_begin0
	.quad	.Ltmp587-.Lfunc_begin0
	.quad	.Ltmp589-.Lfunc_begin0
	.quad	.Ltmp591-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges33:
	.quad	.Ltmp504-.Lfunc_begin0
	.quad	.Ltmp505-.Lfunc_begin0
	.quad	.Ltmp543-.Lfunc_begin0
	.quad	.Ltmp544-.Lfunc_begin0
	.quad	.Ltmp563-.Lfunc_begin0
	.quad	.Ltmp564-.Lfunc_begin0
	.quad	.Ltmp572-.Lfunc_begin0
	.quad	.Ltmp573-.Lfunc_begin0
	.quad	.Ltmp574-.Lfunc_begin0
	.quad	.Ltmp575-.Lfunc_begin0
	.quad	.Ltmp576-.Lfunc_begin0
	.quad	.Ltmp577-.Lfunc_begin0
	.quad	.Ltmp580-.Lfunc_begin0
	.quad	.Ltmp581-.Lfunc_begin0
	.quad	.Ltmp583-.Lfunc_begin0
	.quad	.Ltmp584-.Lfunc_begin0
	.quad	.Ltmp586-.Lfunc_begin0
	.quad	.Ltmp587-.Lfunc_begin0
	.quad	.Ltmp589-.Lfunc_begin0
	.quad	.Ltmp590-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges34:
	.quad	.Ltmp583-.Lfunc_begin0
	.quad	.Ltmp584-.Lfunc_begin0
	.quad	.Ltmp589-.Lfunc_begin0
	.quad	.Ltmp590-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"mla.py"                        ; string offset=7 ; mla.py
.Linfo_string2:
	.asciz	"/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/attention" ; string offset=14 ; /root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/attention
.Linfo_string3:
	.asciz	"_mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16" ; string offset=76 ; _mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
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
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           8
        .value_kind:     by_value
      - .offset:         64
        .size:           8
        .value_kind:     by_value
      - .offset:         72
        .size:           8
        .value_kind:     by_value
      - .offset:         80
        .size:           8
        .value_kind:     by_value
      - .offset:         88
        .size:           8
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
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .offset:         120
        .size:           4
        .value_kind:     by_value
      - .offset:         124
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         136
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .max_flat_workgroup_size: 64
    .name:           _mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16
    .private_segment_fixed_size: 0
    .sgpr_count:     80
    .sgpr_spill_count: 0
    .symbol:         _mla_decode_fwd_kernel_num_query_heads_16_num_kv_heads_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_SEGMENTS_PER_SEQ_4_num_warps_2_num_stages_2_ALL_DECODE_1_SHUFFLED_KV_CACHE_1_QUERY_DTYPE_BF16_KV_CACHE_DTYPE_BF16.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     653
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
