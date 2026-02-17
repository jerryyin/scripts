#!/usr/bin/env python3
"""
Comprehensive LDS padding sweep for gfx1250 WMMA dot operands.

Sweeps across all three axes of padding consideration:
  Axis 1: Transposed (ds_load_tr*) vs non-transposed (ds_load_b*) access patterns
  Axis 2: Actual instruction width per dtype (128-bit vs 64-bit for transposed)
  Axis 3: Actual vectorization width (kWidth) for non-transposed

For each (dtype, kWidth, tile_shape) combination, reports bank conflicts
for a range of padding values, separately for each load path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lds_bank_conflict_analyzer import (
    LDSConfig,
    analyze_bank_conflicts,
    ds_load_tr16_b128_pattern,
    wmma16_kcontig_pattern,
)


# ============================================================================
# Hardware and sweep configuration
# ============================================================================

GFX1250_NUM_BANKS = 64
GFX1250_BYTES_PER_BANK = 4

DTYPES = {
    # One representative per bit-width. bf16 behaves identically to fp16,
    # and bf8/i8 behave identically to fp8 -- same element size and same
    # transposed instruction.
    #
    # name: (element_bytes, element_bits, tr_inst_bits, kDim, kBase)
    #   kDim:  WMMA v3 instruction K dimension
    #   kBase: elements per thread per instruction = (nonKDim * kDim) / warpSize
    #          This is the hardware minimum each thread must hold for one invocation.
    #
    # From AccelerateAMDMatmul.cpp:
    #   kWidth = 8           when kDimTensor >= kDim  (WMMA v3 always 8)
    #   kWidth = kDimTensor/2 when kDimTensor < kDim   (small-K tile fallback)
    "f32":  (4, 32, None,   4,  2),  # wmma_f32_16x16x4_f32    kBase=16*4/32=2
    "fp16": (2, 16, 128,   32, 16),  # wmma_f32_16x16x32_f16   kBase=16*32/32=16
    "fp8":  (1,  8,  64,   64, 32),  # wmma_f32_16x16x64_fp8   kBase=16*64/32=32
}

TILE_SHAPES = [
    # Small tiles
    (16,   8),
    (16,  16),
    (16,  32),
    (32,  32),
    # Medium tiles
    (16,  64),
    (32,  64),
    (64,  64),
    # Large tiles (flash attention scale)
    (16, 128),
    (32, 128),
    (64, 128),
    (128, 128),
    (128,  64),
    # Extreme
    (128,  32),
    (256,  64),
    (256, 128),
]

PADDING_SWEEP = [0, 1, 2, 4, 8, 16, 32]


# ============================================================================
# Helpers
# ============================================================================

def valid_kwidths(kDim, kBase):
    """
    Return the set of kWidth values that can actually occur for a WMMA v3
    instruction with the given kDim and kBase.

    From AccelerateAMDMatmul.cpp:
        if kDimTensor < kDim:  kWidth = kDimTensor / 2
        else:                  kWidth = 8   (always 8 for WMMA v3)

    kDimTensor (the tile's K dimension) must be >= 2 (K=1 is rejected) and
    is typically a power of 2.

    kBase = (nonKDim * kDim) / warpSize = elements per thread per instruction.
    This is the hardware-derived minimum: each thread must hold kBase elements
    to feed one WMMA invocation.

    For the small-K case (kDimTensor < kDim), kWidth = kDimTensor/2.
    We only include kDimTensor values that make kWidth >= kBase, since
    kWidth < kBase would mean a thread can't even fill one instruction.
    The minimum sensible kDimTensor is therefore 2 * kBase.

    Note: for WMMA v3 the normal kWidth=8 can be < kBase (e.g., fp16 has
    kBase=16 but kWidth=8). This is valid because the register repeat
    mechanism issues multiple loads to fill the instruction.
    """
    result = set()
    result.add(8)  # normal case: kDimTensor >= kDim
    # Small-K case: kDimTensor in [2*kBase, 4*kBase, ..., kDim/2]
    # giving kWidth = kBase, 2*kBase, ...
    kdt = 2 * kBase
    while kdt < kDim:
        result.add(kdt // 2)  # kWidth = kDimTensor / 2
        kdt *= 2
    return sorted(result)

def get_max_conflict(row_width, pad, element_bytes, pattern):
    """Return max bank conflict for one (row_width, padding, pattern) config."""
    config = LDSConfig(
        num_banks=GFX1250_NUM_BANKS,
        bytes_per_bank=GFX1250_BYTES_PER_BANK,
        row_width_elements=row_width,
        padding_elements=pad,
        element_bytes=element_bytes,
    )
    return analyze_bank_conflicts(config, pattern, verbose=False)


def overhead_pct(pad, cols):
    """Padding overhead as a percentage of row width."""
    return pad / cols * 100 if cols > 0 else 0


def padding_tag(pad, current_pad, proposed_pad):
    """Label a padding value as CUR, PRO, C+P, or empty."""
    if pad == current_pad and pad == proposed_pad:
        return " C+P"
    if pad == current_pad:
        return " CUR"
    if pad == proposed_pad:
        return " PRO"
    return ""


def relevant_pads(current_pad):
    """Filter PADDING_SWEEP to values worth testing for a given current pad."""
    max_useful = max(current_pad * 2, 32)
    return sorted(set(p for p in PADDING_SWEEP if p <= max_useful))


def conflict_marker(conflict):
    """Return a visual quality marker for a conflict count."""
    if conflict <= 1:
        return "  "
    if conflict <= 2:
        return "  "
    if conflict <= 4:
        return " !"
    return " X"


def print_sweep_table(pattern, elem_bytes, pads, current_pad, proposed_pad,
                      tile_shapes, min_cols=1, indent=4):
    """
    Print a conflict table: rows = tile col widths, columns = padding values.

    Each cell shows the max bank-conflict degree. The header shows padding
    overhead (%) for a reference column width (the most common cols value).

    Args:
        pattern:      AccessPattern to analyze
        elem_bytes:   bytes per element
        pads:         list of padding values to sweep
        current_pad:  current padding value (for labeling)
        proposed_pad: proposed padding value (for labeling)
        tile_shapes:  list of (rows, cols) tile shapes to test
        min_cols:     skip tiles with cols < min_cols
        indent:       number of leading spaces
    """
    prefix = " " * indent
    col_w = 9  # width per padding column

    # Header line 1: pad values with tags
    hdr = f"{prefix}{'cols':>5s} |"
    for p in pads:
        tag = padding_tag(p, current_pad, proposed_pad)
        label = f"p={p}{tag}"
        hdr += f" {label:>{col_w}s}"
    print(hdr)

    # Header line 2: overhead % for the most common cols value
    col_counts = {}
    for _r, c in tile_shapes:
        if c >= min_cols:
            col_counts[c] = col_counts.get(c, 0) + 1
    ref_cols = max(col_counts, key=col_counts.get) if col_counts else 128
    oh_line = f"{prefix}{'oh%':>5s} |"
    for p in pads:
        if p > ref_cols:
            oh_line += f" {'':>{col_w}s}"
        else:
            oh = overhead_pct(p, ref_cols)
            oh_line += f" {oh:>{col_w - 1}.1f}%"
    print(oh_line)
    print(f"{prefix}{'-----':>5s}-+" + "-" * (len(pads) * (col_w + 1)))

    # Data rows
    for _rows, cols in tile_shapes:
        if cols < min_cols:
            continue
        line = f"{prefix}{cols:5d} |"
        for p in pads:
            if p > cols:
                line += f" {'N/A':>{col_w}s}"
            else:
                c = get_max_conflict(cols, p, elem_bytes, pattern)
                mk = conflict_marker(c)
                cell = f"{c:2d}-way{mk}"
                line += f" {cell:>{col_w}s}"
        print(line)


# ============================================================================
# Part 1: Transposed load path sweep
# ============================================================================

def sweep_transposed():
    """
    Sweep the transposed load path (ds_load_tr*).

    Used for dot operands where loadTransposed=True (K-contiguous shared
    memory order). The transposed instruction has a FIXED access pattern
    per dtype -- kWidth does NOT affect it.
    """
    print()
    print("#" * 100)
    print("# PART 1: TRANSPOSED LOAD PATH (ds_load_tr*)")
    print("#")
    print("# Used when loadTransposed = (order[0] != (1 - opIdx)) is True.")
    print("# The transposed instruction has a FIXED access pattern per dtype.")
    print("# kWidth does NOT affect the transposed access pattern.")
    print("#" * 100)
    print()

    for dtype_name, (elem_bytes, elem_bits, tr_inst_bits, _kDim, _kBase) in DTYPES.items():
        if tr_inst_bits is None:
            print(f"  {dtype_name}: No transposed load instruction. Skipping.")
            print()
            continue

        tr_elems = tr_inst_bits // elem_bits
        current_pad = 128 // elem_bits
        proposed_pad = tr_elems

        pattern = ds_load_tr16_b128_pattern()
        pads = relevant_pads(current_pad)

        print(f"  {dtype_name} ({elem_bits}-bit): ds_load_tr{elem_bits}_b{tr_inst_bits}")
        print(f"    {tr_elems} elems/lane, current pad={current_pad}, "
              f"proposed pad={proposed_pad}")
        print()

        # ds_load_tr* operates on a full 16x16 sub-tile (32 lanes: two
        # half-warps of 16 lanes each covering 8 columns). Tiles narrower
        # than 16 columns can't use the transposed instruction.
        print_sweep_table(pattern, elem_bytes, pads, current_pad, proposed_pad,
                          TILE_SHAPES, min_cols=16, indent=4)
        print()


# ============================================================================
# Part 2: Non-transposed load path sweep
# ============================================================================

def sweep_non_transposed():
    """
    Sweep the non-transposed load path (ds_load_b*).

    Used for dot operands where loadTransposed=False. The access pattern
    depends on kWidth (vectorization width along K dimension).
    """
    print()
    print("#" * 100)
    print("# PART 2: NON-TRANSPOSED LOAD PATH (ds_load_b*)")
    print("#")
    print("# Used when loadTransposed = False.")
    print("# The access pattern depends on kWidth (vectorization width).")
    print("# Axis 3: padding should match kWidth, not 128/elemBits.")
    print("#" * 100)
    print()

    for dtype_name, (elem_bytes, elem_bits, _tr_inst_bits, kDim, kBase) in DTYPES.items():
        current_pad = 128 // elem_bits
        kwidths = valid_kwidths(kDim, kBase)

        print(f"  {dtype_name} ({elem_bits}-bit): non-transposed ds_load_b*")
        print(f"    current pad={current_pad}, WMMA v3 kDim={kDim}, kBase={kBase}")
        print(f"    valid kWidths: {kwidths}  (8=normal, others from small kDimTensor)")
        print()

        for kw in kwidths:
            proposed_pad = min(kw, 128 // elem_bits)
            pattern = wmma16_kcontig_pattern(kWidth=kw)
            pads = relevant_pads(current_pad)

            tag = " (normal)" if kw == 8 else f" (kDimTensor={kw*2})"
            print(f"    kWidth={kw}{tag}, proposed pad={proposed_pad}")

            print_sweep_table(pattern, elem_bytes, pads, current_pad,
                              proposed_pad, TILE_SHAPES, min_cols=kw * 2,
                              indent=6)
            print()


# ============================================================================
# Part 3: Dual-path proposal summary
# ============================================================================

def sweep_summary(ref_cols=128):
    """
    Print a compact summary comparing current vs proposed padding for
    both transposed and non-transposed paths at a reference column width.
    """
    print()
    print("#" * 100)
    print("# PART 3: DUAL-PATH PROPOSAL SUMMARY")
    print("#")
    print("# Transposed path:     padAmount = instBitWidth / elemBits")
    print("# Non-transposed path: padAmount = min(kWidth, 128 / elemBits)")
    print("#" * 100)
    print()

    print(f"  {'Path':<15s} {'dtype':>5s} {'kW':>3s} {'current':>8s} {'proposed':>9s} "
          f"{'conflict':>9s} {'overhead_cur':>12s} {'overhead_pro':>12s}")
    print(f"  {'-'*14:<15s} {'-----':>5s} {'---':>3s} {'--------':>8s} {'---------':>9s} "
          f"{'---------':>9s} {'------------':>12s} {'------------':>12s}")

    for dtype_name, (elem_bytes, elem_bits, tr_inst_bits, kDim, kBase) in DTYPES.items():
        current_pad = 128 // elem_bits

        # Transposed path
        if tr_inst_bits is not None:
            tr_elems = tr_inst_bits // elem_bits
            proposed = tr_elems
            pattern = ds_load_tr16_b128_pattern()
            cur_c = get_max_conflict(ref_cols, current_pad, elem_bytes, pattern)
            pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern)
            delta = f" ({cur_c}\u2192{pro_c})" if cur_c != pro_c else ""
            print(f"  {'transposed':<15s} {dtype_name:>5s} {'*':>3s} {current_pad:8d} {proposed:9d} "
                  f"{pro_c:2d}-way{delta:>4s} "
                  f"{overhead_pct(current_pad, ref_cols):10.1f}% "
                  f"{overhead_pct(proposed, ref_cols):10.1f}%")

        # Non-transposed path
        for kw in valid_kwidths(kDim, kBase):
            proposed = min(kw, 128 // elem_bits)
            pattern = wmma16_kcontig_pattern(kWidth=kw)
            cur_c = get_max_conflict(ref_cols, current_pad, elem_bytes, pattern)
            pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern)
            delta = f" ({cur_c}\u2192{pro_c})" if cur_c != pro_c else ""
            print(f"  {'non-transposed':<15s} {dtype_name:>5s} {kw:3d} {current_pad:8d} {proposed:9d} "
                  f"{pro_c:2d}-way{delta:>4s} "
                  f"{overhead_pct(current_pad, ref_cols):10.1f}% "
                  f"{overhead_pct(proposed, ref_cols):10.1f}%")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 100)
    print("gfx1250 LDS PADDING COMPREHENSIVE SWEEP")
    print("=" * 100)
    print()
    print("Hardware: 64 banks, 4 bytes/bank, warp_size=32, WMMA nonKDim=16")

    sweep_transposed()
    sweep_non_transposed()
    sweep_summary()


if __name__ == "__main__":
    main()
