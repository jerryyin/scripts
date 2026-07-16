	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	bug                             ; -- Begin function bug
	.p2align	8
	.type	bug,@function
bug:                                    ; @bug
; %bb.0:                                ; %entry
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[12:14], s[4:5], 0x0 nv
	v_and_b32_e32 v2, 0x3ff, v0
	s_mov_b32 s0, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	s_mov_b32 s1, s0
	s_mov_b32 s2, s0
	v_readfirstlane_b32 s10, v2
	s_mov_b32 s3, s0
	s_wait_xcnt 0x0
	s_mov_b32 s4, s0
	s_mov_b32 s6, s0
	s_mov_b32 s7, s0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v2, s[12:13] scale_offset
	s_load_b32 s5, s[12:13], 0x0
	s_wait_xcnt 0x0
	s_mov_b32 s12, s0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v4, s5 :: v_dual_mov_b32 v7, s5
	s_mov_b32 s5, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s8, v4
	v_readfirstlane_b32 s11, v7
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s9, v1
.LBB0_1:                                ; %loop
                                        ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[0:7]
	s_add_co_i32 s12, s12, 1
	s_cmp_lt_i32 s12, s14
	s_cbranch_scc1 .LBB0_1
; %bb.2:                                ; %exit
	s_endpgm
.Lfunc_end0:
	.size	bug, .Lfunc_end0-bug
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bug
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
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
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 15
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-bug)<<4)&4080)>>4
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
	.set .Lbug.num_vgpr, 8
	.set .Lbug.num_agpr, 0
	.set .Lbug.numbered_sgpr, 15
	.set .Lbug.num_named_barrier, 0
	.set .Lbug.private_seg_size, 0
	.set .Lbug.uses_vcc, 0
	.set .Lbug.uses_flat_scratch, 0
	.set .Lbug.has_dyn_sized_stack, 0
	.set .Lbug.has_recursion, 0
	.set .Lbug.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 164
; TotalNumSgprs: 15
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 15
; NumVGPRsForWavesPerEU: 8
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.globl	safe                            ; -- Begin function safe
	.p2align	8
	.type	safe,@function
safe:                                   ; @safe
; %bb.0:                                ; %entry
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[8:10], s[4:5], 0x0 nv
	v_and_b32_e32 v2, 0x3ff, v0
	v_mov_b32_e32 v5, 0
	s_mov_b32 s0, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s2, s0
	s_mov_b32 s3, s0
	s_wait_xcnt 0x0
	s_mov_b32 s4, s0
	s_mov_b32 s5, s0
	s_mov_b32 s6, s0
	s_mov_b32 s7, s0
	s_mov_b32 s11, s0
	s_wait_kmcnt 0x0
	global_load_b32 v1, v2, s[8:9] scale_offset
	s_load_b32 s1, s[8:9], 0x0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v4, s1 :: v_dual_mov_b32 v7, s1
	s_mov_b32 s1, s0
	s_branch .LBB1_2
.LBB1_1:                                ; %cont
                                        ;   in Loop: Header=BB1_2 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_add_co_i32 s11, s11, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s11, s10
	s_cbranch_scc0 .LBB1_4
.LBB1_2:                                ; %loop
                                        ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s12, v4
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s13, v1
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v7
	tensor_load_to_lds s[12:15], s[0:7]
	s_mov_b32 s12, exec_lo
	v_cmpx_gt_i32_e32 s11, v2
	s_cbranch_execz .LBB1_1
; %bb.3:                                ; %side
                                        ;   in Loop: Header=BB1_2 Depth=1
	v_mov_b32_e32 v0, s11
	global_store_b32 v5, v0, s[8:9]
	s_branch .LBB1_1
.LBB1_4:                                ; %exit
	s_endpgm
.Lfunc_end1:
	.size	safe, .Lfunc_end1-safe
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel safe
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
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
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 16
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-safe)<<4)&4080)>>4
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
	.set .Lsafe.num_vgpr, 8
	.set .Lsafe.num_agpr, 0
	.set .Lsafe.numbered_sgpr, 16
	.set .Lsafe.num_named_barrier, 0
	.set .Lsafe.private_seg_size, 0
	.set .Lsafe.uses_vcc, 0
	.set .Lsafe.uses_flat_scratch, 0
	.set .Lsafe.has_dyn_sized_stack, 0
	.set .Lsafe.has_recursion, 0
	.set .Lsafe.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 204
; TotalNumSgprs: 16
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 8
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           p
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           n
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         96
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         104
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         112
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         120
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         128
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         216
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           bug
    .private_segment_fixed_size: 0
    .sgpr_count:     15
    .sgpr_spill_count: 0
    .symbol:         bug.kd
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .name:           p
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           n
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         96
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         104
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         112
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         120
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         128
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         216
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           safe
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         safe.kd
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
