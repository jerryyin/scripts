#!/usr/bin/env python3
"""
LDS Access Pattern Analysis for ds_load_tr16_b128 Instruction
==============================================================

This script documents the complete analysis of how Triton generates LDS access
patterns for transposed loads using ds_load_tr16_b128 on AMD gfx1250 GPUs.

The analysis was derived from:
1. Running the flash_attention_tool.py to generate LLVM IR
2. Extracting the address computation from the LLVM IR
3. Simulating the thread-to-address mapping
4. Analyzing bank conflicts for different padding strategies

Source LLVM IR Location:
    /root/ir_compare_controlled_ns2/triton/attn_fwd_B8_H8_N512_D128_full_BM128_BN64_NS2.llir

Key LLVM IR Snippets Analyzed:
==============================

1. Thread ID extraction:
   ```llvm
   %21 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !33
   ```

2. Address computation (lines 282-293):
   ```llvm
   %265 = shl nuw nsw i32 %21, 8        ; %21 << 8 = thread_id * 256
   %266 = and i32 %265, 1792            ; (thread_id * 256) & 1792
   %267 = shl nuw nsw i32 %21, 1        ; thread_id * 2
   %268 = and i32 %267, 16              ; (thread_id * 2) & 16
   %60 = and i32 %21, 16                ; thread_id & 16
   %269 = shl nuw nsw i32 %60, 7        ; (thread_id & 16) << 7
   %270 = or disjoint i32 %268, %269   
   %271 = or disjoint i32 %270, %266   ; Main lane-dependent offset
   %272 = lshr exact i32 %271, 3        ; %271 >> 3
   %273 = and i32 %272, 480             ; Padding adjustment
   ```

3. Final address used for ds_load_tr16_b128:
   ```llvm
   %951 = getelementptr inbounds nuw i8, ptr addrspace(3) %275, i32 %271
   %952 = getelementptr inbounds nuw i8, ptr addrspace(3) %951, i32 %273
   %953 = tail call <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3) %952)
   ```

4. Additional constant offsets for subsequent loads:
   ```llvm
   ; Offsets observed: 0, 4608, 9216, 13824, 32, 4640, 9248, 13856, ...
   %954 = getelementptr inbounds nuw i8, ptr addrspace(3) %952, i32 4608
   %955 = tail call <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3) nonnull %954)
   ```

Padding Configuration (from LowerLoops.cpp):
============================================
The padding is computed in getPaddedEncoding():
- maxVecSize = 128 / typeWidthInBit = 128 / 16 = 8 elements for fp16
- padAmount = loadTransposed ? 2 * maxVecSize : maxVecSize = 16 elements for transposed
- Row stride with padding = (HEAD_DIM + padAmount) * sizeof(fp16) = (128 + 16) * 2 = 288 bytes

TTGIR Encoding:
===============
The padding is encoded in TTGIR as:
    #shared1 = #ttg.padded_shared<[128:+16] {order = [1, 0], shape = [64, 128]}>

Meaning of [128:+16]:
    - interval = 128 elements
    - padding  = 16 elements  
    - Every 128 elements, 16 padding elements are inserted (on the right)

Physical LDS Layout:
    Bytes 0-255:   Row 0 DATA (128 fp16 elements)
    Bytes 256-287: Row 0 PADDING (16 fp16 elements) <- unused space
    Bytes 288-543: Row 1 DATA
    Bytes 544-575: Row 1 PADDING
    ...

The emitPadding() function in Utility.cpp translates this to address offsets:
    padOffset = (smemOffset / intervalScaled) * paddingScaled
              = (smemOffset / 256) * 32   (for fp16)

This maps to LLVM IR variable %273 which adds the padding adjustment.

Hardware Specs (gfx1250):
========================
- ds_load_tr16_b128: 8 lanes cooperate, each loads 128 bits (8 fp16 elements)
- LDS has 64 banks, each 4 bytes wide
- Wavefront size: 32 threads (but ds_load_tr works on groups of 8)
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class LDSConfig:
    """LDS configuration parameters."""
    num_banks: int = 64      # AMD gfx1250 has 64 banks
    bank_width: int = 4      # Each bank is 4 bytes wide
    

@dataclass
class TensorConfig:
    """Tensor and padding configuration."""
    head_dim: int = 128      # HEAD_DIM dimension
    block_n: int = 64        # BLOCK_N dimension
    elem_size: int = 2       # fp16 = 2 bytes
    pad_elements: int = 16   # Padding elements (2x * maxVecSize for transposed)
    
    @property
    def row_stride_bytes(self) -> int:
        """Padded row stride in bytes."""
        return (self.head_dim + self.pad_elements) * self.elem_size


def compute_llvm_ir_offset(thread_id: int) -> Tuple[int, int, int]:
    """
    Replicate the LLVM IR address computation exactly.
    
    This mirrors the instructions:
        %265 = shl nuw nsw i32 %21, 8
        %266 = and i32 %265, 1792
        %267 = shl nuw nsw i32 %21, 1
        %268 = and i32 %267, 16
        %60 = and i32 %21, 16
        %269 = shl nuw nsw i32 %60, 7
        %270 = or disjoint i32 %268, %269
        %271 = or disjoint i32 %270, %266
        %272 = lshr exact i32 %271, 3
        %273 = and i32 %272, 480
    
    Returns:
        (v271, v273, total_offset) where total_offset = v271 + v273
    """
    # %60 = %21 & 16  (bit 4 of thread_id)
    v60 = thread_id & 16
    
    # %265 = %21 << 8  (thread_id * 256)
    v265 = thread_id << 8
    
    # %266 = %265 & 1792  (1792 = 0b11100000000 = 7 * 256)
    # This extracts bits 8-10 of thread_id, scaled by 256
    # Equivalent to (thread_id & 7) * 256
    v266 = v265 & 1792
    
    # %267 = %21 << 1  (thread_id * 2)
    v267 = thread_id << 1
    
    # %268 = %267 & 16  (= (thread_id & 8) * 2)
    # This is bit 3 of thread_id, shifted left by 1
    v268 = v267 & 16
    
    # %269 = %60 << 7  (= (thread_id & 16) * 128)
    # This is bit 4 of thread_id, shifted left by 7
    v269 = v60 << 7
    
    # %270 = %268 | %269
    v270 = v268 | v269
    
    # %271 = %270 | %266  (main lane-dependent offset)
    v271 = v270 | v266
    
    # %272 = %271 >> 3
    v272 = v271 >> 3
    
    # %273 = %272 & 480  (480 = 0b111100000)
    # This is the padding adjustment
    v273 = v272 & 480
    
    total_offset = v271 + v273
    return v271, v273, total_offset


def get_banks_accessed(byte_offset: int, access_size_bytes: int, num_banks: int) -> List[int]:
    """
    Compute which LDS banks are accessed for a given byte offset.
    
    AMD LDS bank assignment: bank = (byte_offset / 4) % num_banks
    Each bank is 4 bytes wide.
    
    Args:
        byte_offset: Starting byte offset in LDS
        access_size_bytes: Number of bytes accessed (16 for ds_load_tr16_b128)
        num_banks: Number of LDS banks (64 for gfx1250)
    
    Returns:
        List of bank indices accessed
    """
    num_slots = access_size_bytes // 4  # 4 bytes per bank slot
    return [(byte_offset // 4 + i) % num_banks for i in range(num_slots)]


def analyze_thread_offsets(num_threads: int = 32) -> Dict[int, Tuple[int, int, int]]:
    """
    Compute offsets for all threads in a warp.
    
    Returns:
        Dict mapping thread_id -> (v271, v273, total_offset)
    """
    results = {}
    for tid in range(num_threads):
        results[tid] = compute_llvm_ir_offset(tid)
    return results


def analyze_bank_conflicts(
    row_stride_bytes: int,
    num_threads: int = 8,
    access_size_bytes: int = 16,
    num_banks: int = 64,
    verbose: bool = True,
    show_details: bool = True
) -> Tuple[Dict[int, List[int]], int]:
    """
    Analyze bank conflicts for a given row stride and thread count.
    
    Thread-to-offset mapping:
    - For 8 threads: simple consecutive rows (thread i -> row i)
    - For 32 threads: 2x2 tile arrangement
        - Threads 0-7:   rows 0-7,  cols 0-7   (tile top-left)
        - Threads 8-15:  rows 0-7,  cols 8-15  (tile top-right, +16 bytes)
        - Threads 16-23: rows 8-15, cols 0-7   (tile bottom-left)
        - Threads 24-31: rows 8-15, cols 8-15  (tile bottom-right, +16 bytes)
    
    Args:
        row_stride_bytes: Bytes between consecutive rows (includes padding)
        num_threads: Number of threads to analyze (8 or 32)
        access_size_bytes: Bytes loaded per thread (16 for ds_load_tr16_b128)
        num_banks: Number of LDS banks
        verbose: Print output
        show_details: Show per-thread access details (only for 8 threads)
    
    Returns:
        Tuple of (conflicts dict, max_conflict_degree)
    """
    # Compute thread-to-offset mapping
    all_offsets = []
    for tid in range(num_threads):
        if num_threads <= 8:
            # Simple: thread i loads from row i
            offset = tid * row_stride_bytes
        else:
            # 2x2 tile arrangement for 32 threads
            group = tid // 8
            lane_in_group = tid % 8
            
            # Row offset: groups 0-1 use rows 0-7, groups 2-3 use rows 8-15
            row = lane_in_group + (group // 2) * 8
            
            # Column offset: groups 0,2 use cols 0-7, groups 1,3 use cols 8-15
            col_offset_bytes = (group % 2) * 16  # 16 bytes = 8 fp16 elements
            
            offset = row * row_stride_bytes + col_offset_bytes
        
        all_offsets.append((tid, offset))
    
    # Analyze bank conflicts
    banks_accessed = {}
    for tid, offset in all_offsets:
        banks = get_banks_accessed(offset, access_size_bytes, num_banks)
        for b in banks:
            if b not in banks_accessed:
                banks_accessed[b] = []
            banks_accessed[b].append(tid)
    
    # Find conflicts
    conflicts = {b: threads for b, threads in banks_accessed.items() if len(threads) > 1}
    max_conflict = max(len(threads) for threads in banks_accessed.values()) if banks_accessed else 0
    
    if verbose:
        # Show per-thread details only for 8 threads
        if show_details and num_threads <= 8:
            for tid, offset in all_offsets:
                banks = get_banks_accessed(offset, access_size_bytes, num_banks)
                print(f"  Thread {tid}: addr={offset:5d} bytes, banks {banks}")
            print()
        
        # Summary line
        conflict_str = f"{len(conflicts)}/{len(banks_accessed)} banks"
        if max_conflict == 1:
            status = "NO CONFLICTS"
        elif max_conflict == 2:
            # Find sample conflicting thread pairs
            pairs = set()
            for threads in conflicts.values():
                if len(threads) == 2:
                    pairs.add(tuple(sorted(threads)))
            sample_pairs = sorted(pairs)[:4]
            pair_str = ", ".join(f"{p[0]}&{p[1]}" for p in sample_pairs)
            status = f"2-way ({pair_str}, ...)"
        else:
            status = f"{max_conflict}-way (severe!)"
        
        print(f"  {num_threads} threads: {conflict_str} with conflicts, {status}")
    
    return conflicts, max_conflict


def analyze_padding_strategies(head_dim: int = 128, elem_size: int = 2, num_banks: int = 64):
    """
    Compare different padding strategies for bank conflict avoidance.
    """
    print("\n" + "="*70)
    print("PADDING STRATEGY COMPARISON (8 threads)")
    print("="*70)
    print(f"HEAD_DIM = {head_dim}, element size = {elem_size} bytes, {num_banks} banks")
    print()
    
    # maxVecSize for fp16 = 128 bits / 16 bits = 8 elements
    max_vec_size = 128 // (elem_size * 8)
    
    strategies = [
        (0, "No padding"),
        (max_vec_size, f"1x maxVecSize ({max_vec_size} elements)"),
        (2 * max_vec_size, f"2x maxVecSize ({2*max_vec_size} elements) [Triton's choice]"),
    ]
    
    results = []
    for pad_elements, name in strategies:
        row_stride = (head_dim + pad_elements) * elem_size
        conflicts, max_conflict = analyze_bank_conflicts(
            row_stride, num_threads=8, verbose=False, num_banks=num_banks)
        
        status = "NO CONFLICT" if max_conflict == 1 else f"{max_conflict}-way conflict"
        results.append((name, pad_elements, row_stride, status))
        
        print(f"  {name}:")
        print(f"    Row stride: {row_stride} bytes")
        print(f"    Bank offset per row: {(row_stride // 4) % num_banks}")
        print(f"    Status: {status}")
        print()
    
    return results


def analyze_triton_ir_offsets():
    """
    Analyze the actual offsets from Triton's LLVM IR.
    
    The constant offsets added after the base address computation:
    0, 4608, 9216, 13824, 32, 4640, 9248, 13856, 64, 4672, 9280, 13888, ...
    
    These represent loading different tiles within the K/V tensors.
    """
    print("\n" + "="*70)
    print("TRITON LLVM IR CONSTANT OFFSETS ANALYSIS")
    print("="*70)
    
    # Offsets observed in the LLVM IR for ds_load_tr16_b128 calls
    # These are added to the thread-dependent base address
    offsets = [
        0, 4608, 9216, 13824,      # First group (column 0-7)
        32, 4640, 9248, 13856,     # Second group (column 16-23)
        64, 4672, 9280, 13888,     # Third group (column 32-39)
        96, 4704, 9312, 13920,     # Fourth group (column 48-55)
        128, 4736, 9344, 13952,    # Fifth group (column 64-71)
        160, 4768, 9376, 13984,    # Sixth group (column 80-87)
        192, 4800, 9408, 14016,    # Seventh group (column 96-103)
        224, 4832, 9440, 14048,    # Eighth group (column 104-111)
    ]
    
    # With row stride of 288 bytes (HEAD_DIM=128, pad=16, fp16)
    row_stride = 288
    
    print(f"Row stride = {row_stride} bytes (HEAD_DIM=128 + pad=16, fp16)")
    print()
    
    # Group offsets by their row pattern
    print("Offset groups (each group loads from same column range, different rows):")
    for i in range(0, len(offsets), 4):
        group = offsets[i:i+4]
        cols = [off % row_stride // 2 for off in group]  # Column in elements
        rows = [off // row_stride for off in group]       # Row index
        print(f"  Group {i//4}: offsets={group}")
        print(f"           rows={rows}, starting col={cols[0]}")


def print_ttgir_to_llvmir_mapping():
    """
    Show how TTGIR padded_shared maps to LLVM IR access patterns.
    """
    print("\n" + "="*70)
    print("TTGIR TO LLVM IR MAPPING")
    print("="*70)
    print("""
TTGIR Encoding (from .ttgir file):
    #shared1 = #ttg.padded_shared<[128:+16] {order = [1, 0], shape = [64, 128]}>

    Breakdown:
    - [128:+16]     : Every 128 elements, add 16 padding elements (right-side)
    - order = [1,0] : Column-major storage order
    - shape = [64, 128] : BLOCK_N=64 rows, HEAD_DIM=128 columns

Physical Memory Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Bytes 0-255:   Row 0 DATA (128 fp16)                            │
    │ Bytes 256-287: Row 0 PADDING (16 fp16) ← for bank conflict      │
    │ Bytes 288-543: Row 1 DATA (128 fp16)                            │
    │ Bytes 544-575: Row 1 PADDING                                    │
    │ ...                                                             │
    └─────────────────────────────────────────────────────────────────┘

LLVM IR Address Computation:
    %271 = <base offset from thread ID>      ; Which row to access
    %273 = <padding adjustment>              ; Skip over padding regions
    
    final_addr = base + %271 + %273

The %273 padding adjustment implements the emitPadding() formula:
    padOffset = (smemOffset / 256) * 32   ; For fp16 with [128:+16]
""")

    # Show concrete examples
    print("Concrete Examples (fp16, [128:+16] padding):")
    print("-" * 50)
    for row in range(4):
        logical_elem = row * 128
        raw_bytes = logical_elem * 2
        pad_offset = (raw_bytes // 256) * 32
        physical = raw_bytes + pad_offset
        print(f"  Row {row}: logical elem {logical_elem:4d} -> "
              f"raw {raw_bytes:4d}B + pad {pad_offset:3d}B = physical {physical:4d}B")
    print()


def print_hardware_demo_explanation():
    """
    Explain what the ds_load_tr_demo.hpp showed from actual hardware.
    """
    print("\n" + "="*70)
    print("HARDWARE BEHAVIOR FROM ds_load_tr_demo.hpp")
    print("="*70)
    print("""
The demo ran on actual gfx1250 hardware and showed:

INPUT (LDS layout, 16x16 matrix, row-major):
  Row 0:  [  1,  2,  3,  4,  5,  6,  7,  8, ...]
  Row 1:  [ 17, 18, 19, 20, 21, 22, 23, 24, ...]
  Row 2:  [ 33, 34, 35, 36, 37, 38, 39, 40, ...]
  ...

DIRECT LOAD (each lane loads 8 consecutive elements from its row):
  Lane 0: {  1,  2,  3,  4,  5,  6,  7,  8}  <- Row 0, cols 0-7
  Lane 1: { 17, 18, 19, 20, 21, 22, 23, 24}  <- Row 1, cols 0-7
  Lane 2: { 33, 34, 35, 36, 37, 38, 39, 40}  <- Row 2, cols 0-7
  ...

AFTER ds_load_tr16_b128 (data shuffled/transposed):
  Lane 0: {  1, 17, 33, 49, 65, 81, 97, 113}  <- Column 0, rows 0-7
  Lane 1: {  2, 18, 34, 50, 66, 82, 98, 114}  <- Column 1, rows 0-7
  Lane 2: {  3, 19, 35, 51, 67, 83, 99, 115}  <- Column 2, rows 0-7
  ...

KEY INSIGHT:
- 8 lanes cooperate (not 2, not 16)
- Each lane provides an LDS address to the START of its row
- The instruction loads 8 elements per lane and shuffles across lanes
- Result: each lane receives a COLUMN instead of a row (transpose!)
""")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LDS access patterns for ds_load_tr16_b128"
    )
    parser.add_argument("--num-banks", type=int, default=64,
                        help="Number of LDS banks (default: 64 for gfx1250)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="HEAD_DIM dimension (default: 128)")
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses")
    args = parser.parse_args()
    
    print("="*70)
    print("LDS ACCESS PATTERN ANALYSIS FOR ds_load_tr16_b128")
    print("="*70)
    
    # 1. Show TTGIR to LLVM IR mapping
    print_ttgir_to_llvmir_mapping()
    
    # 2. Show hardware demo explanation
    print_hardware_demo_explanation()
    
    # 3. Analyze thread offsets from LLVM IR
    print("\n" + "="*70)
    print("LLVM IR THREAD OFFSET COMPUTATION")
    print("="*70)
    print("\nReplicating LLVM IR address computation for threads 0-31:")
    print("-" * 60)
    
    offsets = analyze_thread_offsets(32)
    
    # Show grouped by cooperative group (8 threads)
    for group in range(4):
        print(f"\nGroup {group} (threads {group*8}-{group*8+7}):")
        group_offsets = []
        for tid in range(group*8, group*8+8):
            v271, v273, total = offsets[tid]
            group_offsets.append(total)
            if tid < group*8 + 3 or tid >= group*8 + 6:  # Show first 3 and last 2
                print(f"  Thread {tid:2d}: v271={v271:5d}, v273={v273:3d}, "
                      f"total={total:5d} bytes")
            elif tid == group*8 + 3:
                print("  ...")
        
        strides = [group_offsets[i+1] - group_offsets[i] for i in range(7)]
        print(f"  Stride between threads: {strides[0]} bytes (constant)")
    
    # 4. Padding strategy comparison
    analyze_padding_strategies(args.head_dim, num_banks=args.num_banks)
    
    # 5. Bank conflict analysis for all padding strategies
    print("\n" + "="*70)
    print("BANK CONFLICT ANALYSIS BY PADDING STRATEGY")
    print("="*70)
    
    configs = [
        (256, "No padding (128*2)"),
        (272, "1x padding (128+8)*2"),
        (288, "2x padding (128+16)*2 [Triton]"),
    ]
    
    for row_stride, name in configs:
        print(f"\n--- {name}, row stride = {row_stride} bytes ---")
        # 8 threads (one cooperative group)
        analyze_bank_conflicts(row_stride, num_threads=8, num_banks=args.num_banks, 
                               verbose=True, show_details=True)
        # 32 threads (full wave)
        analyze_bank_conflicts(row_stride, num_threads=32, num_banks=args.num_banks,
                               verbose=True, show_details=False)
    
    # 7. Analyze LLVM IR constant offsets
    analyze_triton_ir_offsets()
    
    # 8. Summary
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
1. ds_load_tr16_b128 COOPERATIVITY:
   - 8 threads cooperate to load one 8x8 tile
   - Each thread loads 16 bytes (128 bits) from a DIFFERENT row
   - Data is shuffled so each thread gets a column (transpose)

2. BANK CONFLICT ANALYSIS - 8 THREADS (one cooperative group):
   - No padding (256-byte rows): 8-way conflict
   - 1x padding (272-byte rows): NO conflicts
   - 2x padding (288-byte rows): NO conflicts [Triton's choice]

3. BANK CONFLICT ANALYSIS - 32 THREADS (full wave):
   - With 288-byte row stride: 2-way conflicts
   - Threads 0 & 16, 1 & 17, etc. share banks (8 rows apart wraps around)
   - This is acceptable: 2-way = 50% bandwidth, much better than 8-way or 32-way

4. WHY 2-WAY CONFLICT OCCURS IN FULL WAVE:
   - Row stride = 288 bytes = 72 bank slots
   - 72 % 64 = 8 (each row 8 banks apart)
   - 8 rows * 8 banks = 64 banks → wraps around to same banks
   - Threads loading from rows 0-7 and rows 8-15 hit same bank sets

5. OPTIMAL PADDING FORMULA:
   For zero conflicts within 8-thread cooperative group:
   - Bank offset per row = (row_stride / 4) % num_banks
   - Need: bank_offset >= 4 (so 8 threads * 4 banks = 32 slots don't overlap)
   - Minimum padding: 8 elements (1x maxVecSize) for fp16 on 64-bank LDS
""")


if __name__ == "__main__":
    main()
