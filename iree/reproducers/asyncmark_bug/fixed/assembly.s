	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_dispatch_0_matmul_4096x4096x4096_f32 ; -- Begin function matmul_dispatch_0_matmul_4096x4096x4096_f32
	.p2align	8
	.type	matmul_dispatch_0_matmul_4096x4096x4096_f32,@function
matmul_dispatch_0_matmul_4096x4096x4096_f32: ; @matmul_dispatch_0_matmul_4096x4096x4096_f32
; %bb.3:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx4 s[4:7], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.4:
.LBB0_0:
	s_mov_b64 s[12:13], s[2:3]
	s_lshl_b32 s2, s8, 7
	v_lshrrev_b32_e32 v1, 6, v0
	v_and_b32_e32 v2, 63, v0
	s_and_b32 s9, s2, 0xf80
	s_lshl_b32 s2, s8, 2
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v9, 5, v1
	s_and_b32 s2, s2, 0xf80
	v_bfe_u32 v3, v0, 2, 4
	v_or_b32_e32 v4, 0x100, v2
	v_or_b32_e32 v11, s2, v9
	v_lshlrev_b32_e32 v13, 4, v0
	v_lshlrev_b32_e32 v14, 11, v1
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v2, 0x7c, v2
	v_or_b32_e32 v12, v11, v3
	v_and_b32_e32 v13, 48, v13
	v_readfirstlane_b32 s23, v14
	v_or_b32_e32 v15, 0x400, v14
	s_and_b32 s13, s13, 0xffff
	s_mov_b32 s15, 0x27000
	s_brev_b32 s14, 32
	v_bfe_u32 v6, v0, 5, 1
	v_lshlrev_b32_e32 v8, 2, v1
	v_lshl_or_b32 v12, v12, 14, v13
	s_mov_b32 m0, s23
	v_or_b32_e32 v11, v5, v11
	v_readfirstlane_b32 s3, v15
	v_or_b32_e32 v2, s9, v2
	v_or_b32_e32 v15, 0x6000, v14
	buffer_load_dwordx4 v12, s[12:15], 0 offen lds
	v_lshl_or_b32 v11, v11, 14, v13
	s_mov_b32 m0, s3
	v_or_b32_e32 v13, v8, v6
	v_lshlrev_b32_e32 v2, 2, v2
	v_readfirstlane_b32 s3, v15
	s_mov_b64 s[0:1], s[6:7]
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, s14
	s_mov_b32 s7, s15
	buffer_load_dwordx4 v11, s[12:15], 0 offen lds
	v_lshl_or_b32 v13, v13, 14, v2
	s_mov_b32 m0, s3
	v_lshrrev_b32_e32 v7, 7, v4
	buffer_load_dwordx4 v13, s[4:7], 0 offen lds
	v_or_b32_e32 v13, 0x6400, v14
	v_or_b32_e32 v10, 16, v8
	v_or_b32_e32 v8, v7, v8
	v_readfirstlane_b32 s3, v13
	v_lshl_or_b32 v8, v8, 14, v2
	s_mov_b32 m0, s3
	v_or_b32_e32 v7, v7, v10
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	v_or_b32_e32 v8, 64, v12
	v_or_b32_e32 v12, 0x2000, v14
	v_bfe_u32 v68, v0, 4, 2
	v_readfirstlane_b32 s3, v12
	s_mov_b32 m0, s3
	v_and_b32_e32 v70, 0x4f, v0
	buffer_load_dwordx4 v8, s[12:15], 0 offen lds
	v_or_b32_e32 v8, 64, v11
	v_or_b32_e32 v11, 0x2400, v14
	v_lshlrev_b32_e32 v1, 16, v1
	v_readfirstlane_b32 s3, v11
	s_mov_b32 m0, s3
	v_or_b32_e32 v11, 0x8000, v14
	buffer_load_dwordx4 v8, s[12:15], 0 offen lds
	v_or_b32_e32 v8, v10, v6
	v_lshl_or_b32 v8, v8, 14, v2
	v_readfirstlane_b32 s3, v11
	v_lshl_or_b32 v2, v7, 14, v2
	v_or_b32_e32 v7, 0x8400, v14
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v7
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	s_mov_b32 m0, s3
	s_mov_b32 s3, 0x80000
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v69, 64, v2
	v_and_or_b32 v2, v0, 15, v69
	v_lshlrev_b32_e32 v72, 4, v2
	v_and_b32_e32 v2, 3, v0
	v_lshlrev_b32_e32 v73, 4, v2
	v_or3_b32 v2, v5, s2, v9
	v_mov_b32_e32 v5, 0x80
	v_lshl_or_b32 v74, v2, 14, v5
	v_or3_b32 v2, v3, s2, v9
	v_lshl_or_b32 v75, v2, 14, v5
	v_lshlrev_b32_e32 v2, 7, v4
	s_and_b32 s2, s8, 31
	v_and_b32_e32 v2, 0xc000, v2
	s_lshl_b32 s2, s2, 9
	v_and_b32_e32 v0, 31, v0
	v_or3_b32 v2, v1, v2, s2
	v_lshlrev_b32_e32 v0, 4, v0
	v_or3_b32 v76, v2, v0, s3
	v_lshlrev_b32_e32 v2, 14, v6
	v_or3_b32 v1, v1, v2, s2
	v_mov_b32_e32 v28, 0
	s_mov_b32 s20, 2
	s_mov_b32 s22, 0
	s_movk_i32 s21, 0x6000
	v_lshlrev_b32_e32 v71, 7, v68
	v_or3_b32 v77, v1, v0, s3
	s_mov_b64 s[16:17], 0
	s_mov_b64 s[2:3], 0x800
	s_mov_b64 s[18:19], 0x800
	s_mov_b32 s24, -4
	s_mov_b64 s[10:11], 0
	v_mov_b32_e32 v29, v28
	v_mov_b32_e32 v30, v28
	v_mov_b32_e32 v31, v28
	v_mov_b32_e32 v48, v28
	v_mov_b32_e32 v49, v28
	v_mov_b32_e32 v50, v28
	v_mov_b32_e32 v51, v28
	v_mov_b32_e32 v56, v28
	v_mov_b32_e32 v57, v28
	v_mov_b32_e32 v58, v28
	v_mov_b32_e32 v59, v28
	v_mov_b32_e32 v40, v28
	v_mov_b32_e32 v41, v28
	v_mov_b32_e32 v42, v28
	v_mov_b32_e32 v43, v28
	v_mov_b32_e32 v60, v28
	v_mov_b32_e32 v61, v28
	v_mov_b32_e32 v62, v28
	v_mov_b32_e32 v63, v28
	v_mov_b32_e32 v16, v28
	v_mov_b32_e32 v17, v28
	v_mov_b32_e32 v18, v28
	v_mov_b32_e32 v19, v28
	v_mov_b32_e32 v52, v28
	v_mov_b32_e32 v53, v28
	v_mov_b32_e32 v54, v28
	v_mov_b32_e32 v55, v28
	v_mov_b32_e32 v44, v28
	v_mov_b32_e32 v45, v28
	v_mov_b32_e32 v46, v28
	v_mov_b32_e32 v47, v28
	v_mov_b32_e32 v12, v28
	v_mov_b32_e32 v13, v28
	v_mov_b32_e32 v14, v28
	v_mov_b32_e32 v15, v28
	v_mov_b32_e32 v8, v28
	v_mov_b32_e32 v9, v28
	v_mov_b32_e32 v10, v28
	v_mov_b32_e32 v11, v28
	v_mov_b32_e32 v32, v28
	v_mov_b32_e32 v33, v28
	v_mov_b32_e32 v34, v28
	v_mov_b32_e32 v35, v28
	v_mov_b32_e32 v36, v28
	v_mov_b32_e32 v37, v28
	v_mov_b32_e32 v38, v28
	v_mov_b32_e32 v39, v28
	v_mov_b32_e32 v4, v28
	v_mov_b32_e32 v5, v28
	v_mov_b32_e32 v6, v28
	v_mov_b32_e32 v7, v28
	v_mov_b32_e32 v24, v28
	v_mov_b32_e32 v25, v28
	v_mov_b32_e32 v26, v28
	v_mov_b32_e32 v27, v28
	v_mov_b32_e32 v0, v28
	v_mov_b32_e32 v1, v28
	v_mov_b32_e32 v2, v28
	v_mov_b32_e32 v3, v28
	v_mov_b32_e32 v20, v28
	v_mov_b32_e32 v21, v28
	v_mov_b32_e32 v22, v28
	v_mov_b32_e32 v23, v28
	.p2align	5, , 4
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s25, s20, 0xff
	s_mulk_i32 s25, 0xab
	s_lshr_b32 s25, s25, 9
	s_mul_i32 s25, s25, 3
	s_sub_i32 s25, s20, s25
	s_and_b32 s25, s25, 0xff
	s_lshl_b32 s26, s25, 13
	s_or_b32 s26, s23, s26
	s_mov_b32 m0, s26
	v_add_u32_e32 v64, v75, v73
	s_barrier
	buffer_load_dwordx4 v64, s[12:15], 0 offen lds
	v_add_u32_e32 v64, v74, v73
	s_add_i32 m0, s26, 0x400
	s_lshl2_add_u32 s16, s16, s22
	buffer_load_dwordx4 v64, s[12:15], 0 offen lds
	s_add_i32 m0, s26, 0x6000
	s_lshl2_add_u32 s10, s10, s21
	buffer_load_dwordx4 v77, s[4:7], 0 offen lds
	s_add_i32 m0, s26, 0x6400
	v_lshl_add_u32 v64, v68, 2, s16
	buffer_load_dwordx4 v76, s[4:7], 0 offen lds
	v_lshl_add_u32 v86, v70, 2, s10
	v_lshl_add_u32 v64, v72, 2, v64
	v_lshl_add_u32 v104, v71, 2, v86
	v_add_u32_e32 v65, 0x400, v64
	v_add_u32_e32 v66, 0x800, v64
	v_add_u32_e32 v67, 0xc00, v64
	s_waitcnt vmcnt(8)
	s_barrier
	ds_read2_b32 v[78:79], v64 offset1:4
	ds_read2_b32 v[80:81], v65 offset1:4
	ds_read2_b32 v[82:83], v66 offset1:4
	ds_read2_b32 v[84:85], v67 offset1:4
	ds_read2_b32 v[86:87], v104 offset1:16
	ds_read2_b32 v[88:89], v104 offset0:32 offset1:48
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x4_f32 v[28:31], v78, v86, v[28:31]
	v_add_u32_e32 v92, 0x800, v104
	ds_read2_b32 v[90:91], v92 offset1:16
	ds_read2_b32 v[92:93], v92 offset0:32 offset1:48
	ds_read2_b32 v[94:95], v64 offset0:8 offset1:12
	ds_read2_b32 v[96:97], v65 offset0:8 offset1:12
	ds_read2_b32 v[98:99], v66 offset0:8 offset1:12
	ds_read2_b32 v[64:65], v67 offset0:8 offset1:12
	v_add_u32_e32 v66, 0x1000, v104
	ds_read2_b32 v[100:101], v66 offset1:16
	ds_read2_b32 v[102:103], v66 offset0:32 offset1:48
	v_add_u32_e32 v66, 0x1800, v104
	ds_read2_b32 v[104:105], v66 offset1:16
	ds_read2_b32 v[66:67], v66 offset0:32 offset1:48
	s_mov_b64 s[16:17], s[2:3]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x4_f32 v[56:59], v78, v88, v[56:59]
	s_add_i32 s24, s24, 4
	s_add_i32 s20, s20, 1
	s_lshl_b32 s2, s25, 11
	s_mov_b64 s[10:11], s[18:19]
	v_add_u32_e32 v74, 64, v74
	v_add_u32_e32 v75, 64, v75
	s_mov_b64 s[18:19], s[2:3]
	v_mfma_f32_16x16x4_f32 v[40:43], v78, v89, v[40:43]
	v_add_u32_e32 v77, 0x40000, v77
	s_cmpk_lt_u32 s24, 0x3f4
	v_add_u32_e32 v76, 0x40000, v76
	v_mfma_f32_16x16x4_f32 v[52:55], v80, v88, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v80, v89, v[44:47]
	v_mfma_f32_16x16x4_f32 v[32:35], v82, v88, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v82, v89, v[36:39]
	v_mfma_f32_16x16x4_f32 v[48:51], v78, v87, v[48:51]
	v_mfma_f32_16x16x4_f32 v[60:63], v80, v86, v[60:63]
	v_mfma_f32_16x16x4_f32 v[16:19], v80, v87, v[16:19]
	v_mfma_f32_16x16x4_f32 v[12:15], v82, v86, v[12:15]
	v_mfma_f32_16x16x4_f32 v[8:11], v82, v87, v[8:11]
	v_mfma_f32_16x16x4_f32 v[4:7], v84, v86, v[4:7]
	v_mfma_f32_16x16x4_f32 v[24:27], v84, v87, v[24:27]
	v_mfma_f32_16x16x4_f32 v[0:3], v84, v88, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v84, v89, v[20:23]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x4_f32 v[28:31], v79, v90, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v79, v91, v[48:51]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x4_f32 v[56:59], v79, v92, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v79, v93, v[40:43]
	v_mfma_f32_16x16x4_f32 v[60:63], v81, v90, v[60:63]
	v_mfma_f32_16x16x4_f32 v[16:19], v81, v91, v[16:19]
	v_mfma_f32_16x16x4_f32 v[52:55], v81, v92, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v81, v93, v[44:47]
	v_mfma_f32_16x16x4_f32 v[12:15], v83, v90, v[12:15]
	v_mfma_f32_16x16x4_f32 v[8:11], v83, v91, v[8:11]
	v_mfma_f32_16x16x4_f32 v[32:35], v83, v92, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v83, v93, v[36:39]
	v_mfma_f32_16x16x4_f32 v[4:7], v85, v90, v[4:7]
	v_mfma_f32_16x16x4_f32 v[24:27], v85, v91, v[24:27]
	v_mfma_f32_16x16x4_f32 v[0:3], v85, v92, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v85, v93, v[20:23]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x4_f32 v[28:31], v94, v100, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v94, v101, v[48:51]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x4_f32 v[56:59], v94, v102, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v94, v103, v[40:43]
	v_mfma_f32_16x16x4_f32 v[60:63], v96, v100, v[60:63]
	v_mfma_f32_16x16x4_f32 v[16:19], v96, v101, v[16:19]
	v_mfma_f32_16x16x4_f32 v[52:55], v96, v102, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v96, v103, v[44:47]
	v_mfma_f32_16x16x4_f32 v[12:15], v98, v100, v[12:15]
	v_mfma_f32_16x16x4_f32 v[8:11], v98, v101, v[8:11]
	v_mfma_f32_16x16x4_f32 v[32:35], v98, v102, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v98, v103, v[36:39]
	v_mfma_f32_16x16x4_f32 v[4:7], v64, v100, v[4:7]
	v_mfma_f32_16x16x4_f32 v[24:27], v64, v101, v[24:27]
	v_mfma_f32_16x16x4_f32 v[0:3], v64, v102, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v64, v103, v[20:23]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x4_f32 v[28:31], v95, v104, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v95, v105, v[48:51]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x4_f32 v[56:59], v95, v66, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v95, v67, v[40:43]
	v_mfma_f32_16x16x4_f32 v[60:63], v97, v104, v[60:63]
	v_mfma_f32_16x16x4_f32 v[16:19], v97, v105, v[16:19]
	v_mfma_f32_16x16x4_f32 v[52:55], v97, v66, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v97, v67, v[44:47]
	v_mfma_f32_16x16x4_f32 v[12:15], v99, v104, v[12:15]
	v_mfma_f32_16x16x4_f32 v[8:11], v99, v105, v[8:11]
	v_mfma_f32_16x16x4_f32 v[32:35], v99, v66, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v99, v67, v[36:39]
	v_mfma_f32_16x16x4_f32 v[4:7], v65, v104, v[4:7]
	v_mfma_f32_16x16x4_f32 v[24:27], v65, v105, v[24:27]
	v_mfma_f32_16x16x4_f32 v[0:3], v65, v66, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v65, v67, v[20:23]
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	v_or_b32_e32 v64, v72, v68
	s_lshl2_add_u32 s4, s16, s22
	v_lshlrev_b32_e32 v116, 2, v64
	v_add_u32_e32 v64, s4, v116
	v_add_u32_e32 v65, 0x400, v64
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read2_b32 v[72:73], v64 offset1:4
	ds_read2_b32 v[74:75], v64 offset0:8 offset1:12
	ds_read2_b32 v[76:77], v65 offset1:4
	ds_read2_b32 v[78:79], v65 offset0:8 offset1:12
	v_add_u32_e32 v65, 0x800, v64
	v_add_u32_e32 v64, 0xc00, v64
	ds_read2_b32 v[84:85], v64 offset1:4
	ds_read2_b32 v[86:87], v64 offset0:8 offset1:12
	v_or_b32_e32 v64, v71, v70
	s_lshl2_add_u32 s4, s10, s21
	v_lshlrev_b32_e32 v71, 2, v64
	v_add_u32_e32 v100, s4, v71
	ds_read2_b32 v[88:89], v100 offset1:16
	ds_read2_b32 v[90:91], v100 offset0:32 offset1:48
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x4_f32 v[28:31], v72, v88, v[28:31]
	v_add_u32_e32 v94, 0x800, v100
	ds_read2_b32 v[92:93], v94 offset1:16
	ds_read2_b32 v[94:95], v94 offset0:32 offset1:48
	ds_read2_b32 v[80:81], v65 offset1:4
	ds_read2_b32 v[82:83], v65 offset0:8 offset1:12
	s_and_b32 s4, s8, 0x3e0
	v_or_b32_e32 v68, s4, v68
	v_lshlrev_b32_e32 v69, 14, v69
	v_or_b32_e32 v70, s9, v70
	v_mfma_f32_16x16x4_f32 v[48:51], v72, v89, v[48:51]
	v_lshl_add_u32 v68, v68, 16, v69
	v_lshl_or_b32 v68, v70, 2, v68
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	s_brev_b32 s2, 32
	v_add_u32_e32 v69, 0x4000, v68
	v_add_u32_e32 v70, 0x8000, v68
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x4_f32 v[56:59], v72, v90, v[56:59]
	v_add_u32_e32 v132, 0x4c000, v68
	v_add_u32_e32 v133, 0x80000, v68
	v_add_u32_e32 v134, 0x84000, v68
	v_add_u32_e32 v135, 0x88000, v68
	v_add_u32_e32 v136, 0x8c000, v68
	v_add_u32_e32 v137, 0xc0000, v68
	v_mfma_f32_16x16x4_f32 v[40:43], v72, v91, v[40:43]
	v_add_u32_e32 v72, 0x40000, v68
	v_mfma_f32_16x16x4_f32 v[60:63], v76, v88, v[60:63]
	v_mfma_f32_16x16x4_f32 v[64:67], v76, v89, v[16:19]
	v_add_u32_e32 v16, 0x1000, v100
	ds_read2_b32 v[96:97], v16 offset1:16
	ds_read2_b32 v[98:99], v16 offset0:32 offset1:48
	v_add_u32_e32 v16, 0x1800, v100
	ds_read2_b32 v[100:101], v16 offset1:16
	ds_read2_b32 v[102:103], v16 offset0:32 offset1:48
	ds_read2_b32 v[104:105], v116 offset1:4
	ds_read2_b32 v[106:107], v116 offset0:8 offset1:12
	v_add_u32_e32 v18, 0x6000, v71
	ds_read2_b32 v[118:119], v18 offset1:16
	v_mfma_f32_16x16x4_f32 v[52:55], v76, v90, v[52:55]
	ds_read2_b32 v[120:121], v18 offset0:32 offset1:48
	v_add_u32_e32 v16, 0x400, v116
	ds_read2_b32 v[108:109], v16 offset1:4
	ds_read2_b32 v[110:111], v16 offset0:8 offset1:12
	v_add_u32_e32 v18, 0x6800, v71
	ds_read2_b32 v[122:123], v18 offset1:16
	ds_read2_b32 v[124:125], v18 offset0:32 offset1:48
	v_add_u32_e32 v16, 0x800, v116
	ds_read2_b32 v[112:113], v16 offset1:4
	ds_read2_b32 v[114:115], v16 offset0:8 offset1:12
	v_mfma_f32_16x16x4_f32 v[44:47], v76, v91, v[44:47]
	v_add_u32_e32 v18, 0x7000, v71
	ds_read2_b32 v[126:127], v18 offset1:16
	ds_read2_b32 v[128:129], v18 offset0:32 offset1:48
	v_add_u32_e32 v16, 0xc00, v116
	ds_read2_b32 v[116:117], v16 offset1:4
	ds_read2_b32 v[16:17], v16 offset0:8 offset1:12
	v_add_u32_e32 v18, 0x7800, v71
	ds_read2_b32 v[130:131], v18 offset1:16
	ds_read2_b32 v[18:19], v18 offset0:32 offset1:48
	v_add_u32_e32 v71, 0xc000, v68
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x4_f32 v[28:31], v73, v92, v[28:31]
	v_add_u32_e32 v76, 0x44000, v68
	v_mfma_f32_16x16x4_f32 v[48:51], v73, v93, v[48:51]
	v_mfma_f32_16x16x4_f32 v[56:59], v73, v94, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v73, v95, v[40:43]
	v_mfma_f32_16x16x4_f32 v[12:15], v80, v88, v[12:15]
	v_mfma_f32_16x16x4_f32 v[60:63], v77, v92, v[60:63]
	v_mfma_f32_16x16x4_f32 v[8:11], v80, v89, v[8:11]
	v_mfma_f32_16x16x4_f32 v[64:67], v77, v93, v[64:67]
	v_mfma_f32_16x16x4_f32 v[32:35], v80, v90, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v80, v91, v[36:39]
	v_mfma_f32_16x16x4_f32 v[52:55], v77, v94, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v77, v95, v[44:47]
	v_mfma_f32_16x16x4_f32 v[28:31], v74, v96, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v74, v97, v[48:51]
	v_mfma_f32_16x16x4_f32 v[56:59], v74, v98, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v74, v99, v[40:43]
	v_mfma_f32_16x16x4_f32 v[4:7], v84, v88, v[4:7]
	v_mfma_f32_16x16x4_f32 v[12:15], v81, v92, v[12:15]
	v_mfma_f32_16x16x4_f32 v[60:63], v78, v96, v[60:63]
	v_mfma_f32_16x16x4_f32 v[24:27], v84, v89, v[24:27]
	v_add_u32_e32 v89, 0x48000, v68
	v_mfma_f32_16x16x4_f32 v[8:11], v81, v93, v[8:11]
	v_mfma_f32_16x16x4_f32 v[64:67], v78, v97, v[64:67]
	v_mfma_f32_16x16x4_f32 v[0:3], v84, v90, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v84, v91, v[20:23]
	v_mfma_f32_16x16x4_f32 v[32:35], v81, v94, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v81, v95, v[36:39]
	v_mfma_f32_16x16x4_f32 v[52:55], v78, v98, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v78, v99, v[44:47]
	v_mfma_f32_16x16x4_f32 v[28:31], v75, v100, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v75, v101, v[48:51]
	v_mfma_f32_16x16x4_f32 v[56:59], v75, v102, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v75, v103, v[40:43]
	v_mfma_f32_16x16x4_f32 v[4:7], v85, v92, v[4:7]
	v_mfma_f32_16x16x4_f32 v[12:15], v82, v96, v[12:15]
	v_mfma_f32_16x16x4_f32 v[60:63], v79, v100, v[60:63]
	v_mfma_f32_16x16x4_f32 v[24:27], v85, v93, v[24:27]
	v_mfma_f32_16x16x4_f32 v[8:11], v82, v97, v[8:11]
	v_mfma_f32_16x16x4_f32 v[64:67], v79, v101, v[64:67]
	v_mfma_f32_16x16x4_f32 v[0:3], v85, v94, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v85, v95, v[20:23]
	v_mfma_f32_16x16x4_f32 v[32:35], v82, v98, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v82, v99, v[36:39]
	v_mfma_f32_16x16x4_f32 v[52:55], v79, v102, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v79, v103, v[44:47]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x4_f32 v[28:31], v104, v118, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v104, v119, v[48:51]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x4_f32 v[56:59], v104, v120, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v104, v121, v[40:43]
	v_mfma_f32_16x16x4_f32 v[4:7], v86, v96, v[4:7]
	v_mfma_f32_16x16x4_f32 v[12:15], v83, v100, v[12:15]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x4_f32 v[60:63], v108, v118, v[60:63]
	v_mfma_f32_16x16x4_f32 v[24:27], v86, v97, v[24:27]
	v_mfma_f32_16x16x4_f32 v[8:11], v83, v101, v[8:11]
	v_mfma_f32_16x16x4_f32 v[64:67], v108, v119, v[64:67]
	v_mfma_f32_16x16x4_f32 v[0:3], v86, v98, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v86, v99, v[20:23]
	v_mfma_f32_16x16x4_f32 v[32:35], v83, v102, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v83, v103, v[36:39]
	v_mfma_f32_16x16x4_f32 v[52:55], v108, v120, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v108, v121, v[44:47]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x4_f32 v[28:31], v105, v122, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v105, v123, v[48:51]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x4_f32 v[56:59], v105, v124, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v105, v125, v[40:43]
	v_mfma_f32_16x16x4_f32 v[4:7], v87, v100, v[4:7]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x4_f32 v[12:15], v112, v118, v[12:15]
	v_mfma_f32_16x16x4_f32 v[60:63], v109, v122, v[60:63]
	v_mfma_f32_16x16x4_f32 v[24:27], v87, v101, v[24:27]
	v_mfma_f32_16x16x4_f32 v[8:11], v112, v119, v[8:11]
	v_mfma_f32_16x16x4_f32 v[64:67], v109, v123, v[64:67]
	v_mfma_f32_16x16x4_f32 v[0:3], v87, v102, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v87, v103, v[20:23]
	v_mfma_f32_16x16x4_f32 v[32:35], v112, v120, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v112, v121, v[36:39]
	v_mfma_f32_16x16x4_f32 v[52:55], v109, v124, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v109, v125, v[44:47]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x4_f32 v[28:31], v106, v126, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v106, v127, v[48:51]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x4_f32 v[56:59], v106, v128, v[56:59]
	v_mfma_f32_16x16x4_f32 v[40:43], v106, v129, v[40:43]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x4_f32 v[4:7], v116, v118, v[4:7]
	v_mfma_f32_16x16x4_f32 v[12:15], v113, v122, v[12:15]
	v_mfma_f32_16x16x4_f32 v[60:63], v110, v126, v[60:63]
	v_mfma_f32_16x16x4_f32 v[24:27], v116, v119, v[24:27]
	v_mfma_f32_16x16x4_f32 v[8:11], v113, v123, v[8:11]
	v_mfma_f32_16x16x4_f32 v[64:67], v110, v127, v[64:67]
	v_mfma_f32_16x16x4_f32 v[0:3], v116, v120, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v116, v121, v[20:23]
	v_mfma_f32_16x16x4_f32 v[32:35], v113, v124, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v113, v125, v[36:39]
	v_mfma_f32_16x16x4_f32 v[52:55], v110, v128, v[52:55]
	v_mfma_f32_16x16x4_f32 v[44:47], v110, v129, v[44:47]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x4_f32 v[28:31], v107, v130, v[28:31]
	v_mfma_f32_16x16x4_f32 v[48:51], v107, v131, v[48:51]
	s_nop 8
	buffer_store_dword v28, v68, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x4_f32 v[56:59], v107, v18, v[56:59]
	buffer_store_dword v48, v68, s[0:3], 0 offen offset:64
	v_mfma_f32_16x16x4_f32 v[40:43], v107, v19, v[40:43]
	s_nop 7
	buffer_store_dword v56, v68, s[0:3], 0 offen offset:128
	v_mfma_f32_16x16x4_f32 v[4:7], v117, v122, v[4:7]
	buffer_store_dword v40, v68, s[0:3], 0 offen offset:192
	buffer_store_dword v29, v69, s[0:3], 0 offen
	buffer_store_dword v49, v69, s[0:3], 0 offen offset:64
	buffer_store_dword v57, v69, s[0:3], 0 offen offset:128
	buffer_store_dword v41, v69, s[0:3], 0 offen offset:192
	buffer_store_dword v30, v70, s[0:3], 0 offen
	buffer_store_dword v50, v70, s[0:3], 0 offen offset:64
	buffer_store_dword v58, v70, s[0:3], 0 offen offset:128
	v_mfma_f32_16x16x4_f32 v[12:15], v114, v126, v[12:15]
	v_mfma_f32_16x16x4_f32 v[60:63], v111, v130, v[60:63]
	v_mfma_f32_16x16x4_f32 v[24:27], v117, v123, v[24:27]
	v_mfma_f32_16x16x4_f32 v[8:11], v114, v127, v[8:11]
	v_mfma_f32_16x16x4_f32 v[64:67], v111, v131, v[64:67]
	buffer_store_dword v42, v70, s[0:3], 0 offen offset:192
	buffer_store_dword v31, v71, s[0:3], 0 offen
	buffer_store_dword v51, v71, s[0:3], 0 offen offset:64
	buffer_store_dword v59, v71, s[0:3], 0 offen offset:128
	buffer_store_dword v43, v71, s[0:3], 0 offen offset:192
	s_nop 1
	buffer_store_dword v60, v72, s[0:3], 0 offen
	s_nop 1
	buffer_store_dword v64, v72, s[0:3], 0 offen offset:64
	v_mfma_f32_16x16x4_f32 v[0:3], v117, v124, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v117, v125, v[20:23]
	v_mfma_f32_16x16x4_f32 v[32:35], v114, v128, v[32:35]
	v_mfma_f32_16x16x4_f32 v[36:39], v114, v129, v[36:39]
	v_mfma_f32_16x16x4_f32 v[28:31], v111, v18, v[52:55]
	v_mfma_f32_16x16x4_f32 v[40:43], v111, v19, v[44:47]
	s_nop 8
	buffer_store_dword v28, v72, s[0:3], 0 offen offset:128
	v_mfma_f32_16x16x4_f32 v[4:7], v16, v126, v[4:7]
	buffer_store_dword v40, v72, s[0:3], 0 offen offset:192
	buffer_store_dword v61, v76, s[0:3], 0 offen
	buffer_store_dword v65, v76, s[0:3], 0 offen offset:64
	buffer_store_dword v29, v76, s[0:3], 0 offen offset:128
	buffer_store_dword v41, v76, s[0:3], 0 offen offset:192
	buffer_store_dword v62, v89, s[0:3], 0 offen
	buffer_store_dword v66, v89, s[0:3], 0 offen offset:64
	buffer_store_dword v30, v89, s[0:3], 0 offen offset:128
	v_mfma_f32_16x16x4_f32 v[12:15], v115, v130, v[12:15]
	v_mfma_f32_16x16x4_f32 v[24:27], v16, v127, v[24:27]
	v_mfma_f32_16x16x4_f32 v[8:11], v115, v131, v[8:11]
	buffer_store_dword v42, v89, s[0:3], 0 offen offset:192
	buffer_store_dword v63, v132, s[0:3], 0 offen
	buffer_store_dword v67, v132, s[0:3], 0 offen offset:64
	buffer_store_dword v31, v132, s[0:3], 0 offen offset:128
	buffer_store_dword v43, v132, s[0:3], 0 offen offset:192
	s_nop 2
	buffer_store_dword v12, v133, s[0:3], 0 offen
	s_nop 0
	buffer_store_dword v8, v133, s[0:3], 0 offen offset:64
	v_mfma_f32_16x16x4_f32 v[0:3], v16, v128, v[0:3]
	v_mfma_f32_16x16x4_f32 v[20:23], v16, v129, v[20:23]
	v_mfma_f32_16x16x4_f32 v[28:31], v115, v18, v[32:35]
	v_mfma_f32_16x16x4_f32 v[32:35], v115, v19, v[36:39]
	s_nop 8
	buffer_store_dword v28, v133, s[0:3], 0 offen offset:128
	v_mfma_f32_16x16x4_f32 v[4:7], v17, v130, v[4:7]
	buffer_store_dword v32, v133, s[0:3], 0 offen offset:192
	buffer_store_dword v13, v134, s[0:3], 0 offen
	buffer_store_dword v9, v134, s[0:3], 0 offen offset:64
	buffer_store_dword v29, v134, s[0:3], 0 offen offset:128
	buffer_store_dword v33, v134, s[0:3], 0 offen offset:192
	buffer_store_dword v14, v135, s[0:3], 0 offen
	buffer_store_dword v10, v135, s[0:3], 0 offen offset:64
	buffer_store_dword v30, v135, s[0:3], 0 offen offset:128
	v_mfma_f32_16x16x4_f32 v[24:27], v17, v131, v[24:27]
	buffer_store_dword v34, v135, s[0:3], 0 offen offset:192
	buffer_store_dword v15, v136, s[0:3], 0 offen
	buffer_store_dword v11, v136, s[0:3], 0 offen offset:64
	buffer_store_dword v31, v136, s[0:3], 0 offen offset:128
	buffer_store_dword v35, v136, s[0:3], 0 offen offset:192
	buffer_store_dword v4, v137, s[0:3], 0 offen
	s_nop 3
	buffer_store_dword v24, v137, s[0:3], 0 offen offset:64
	v_mfma_f32_16x16x4_f32 v[0:3], v17, v18, v[0:3]
	v_mfma_f32_16x16x4_f32 v[8:11], v17, v19, v[20:23]
	s_nop 8
	buffer_store_dword v0, v137, s[0:3], 0 offen offset:128
	v_add_u32_e32 v0, 0xc4000, v68
	buffer_store_dword v8, v137, s[0:3], 0 offen offset:192
	buffer_store_dword v5, v0, s[0:3], 0 offen
	buffer_store_dword v25, v0, s[0:3], 0 offen offset:64
	buffer_store_dword v1, v0, s[0:3], 0 offen offset:128
	buffer_store_dword v9, v0, s[0:3], 0 offen offset:192
	v_add_u32_e32 v0, 0xc8000, v68
	buffer_store_dword v6, v0, s[0:3], 0 offen
	buffer_store_dword v26, v0, s[0:3], 0 offen offset:64
	buffer_store_dword v2, v0, s[0:3], 0 offen offset:128
	buffer_store_dword v10, v0, s[0:3], 0 offen offset:192
	v_add_u32_e32 v0, 0xcc000, v68
	buffer_store_dword v7, v0, s[0:3], 0 offen
	buffer_store_dword v27, v0, s[0:3], 0 offen offset:64
	buffer_store_dword v3, v0, s[0:3], 0 offen offset:128
	buffer_store_dword v11, v0, s[0:3], 0 offen offset:192
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_dispatch_0_matmul_4096x4096x4096_f32
		.amdhsa_group_segment_fixed_size 49152
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 6
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 138
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 140
		.amdhsa_reserve_vcc 0
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
	.size	matmul_dispatch_0_matmul_4096x4096x4096_f32, .Lfunc_end0-matmul_dispatch_0_matmul_4096x4096x4096_f32
                                        ; -- End function
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.num_vgpr, 138
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.num_agpr, 0
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.numbered_sgpr, 27
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.num_named_barrier, 0
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.private_seg_size, 0
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.uses_vcc, 0
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.uses_flat_scratch, 0
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.has_dyn_sized_stack, 0
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.has_recursion, 0
	.set matmul_dispatch_0_matmul_4096x4096x4096_f32.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4300
; TotalNumSgprs: 33
; NumVgprs: 138
; NumAgprs: 0
; TotalNumVgprs: 138
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 49152 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 17
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 138
; AccumOffset: 140
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 34
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
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.2 25385 0dda3adf56766e0aac0d03173ced3759e1ffecbc)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 49152
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           matmul_dispatch_0_matmul_4096x4096x4096_f32
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         matmul_dispatch_0_matmul_4096x4096x4096_f32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     138
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
