#!/usr/bin/env python3
"""
Comprehensive LDS padding sweep for dot operands.

For each (dtype, col_width) combination, reports bank conflicts for a range
of padding values, separately for each load path.

Covers:
  - gfx1250 WMMA paths (transposed, non-transposed, transposed-scalar)
  - CDNA MFMA f32 non-transposed paths (16x16x4 and 32x32x2 geometries)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lds_bank_conflict_analyzer import (
    LDSConfig,
    analyze_bank_conflicts,
    ds_load_tr16_b128_pattern,
    ds_load_tr8_b64_pattern,
    ds_load_2addr_b64_pattern,
    wmma16_kcontig_pattern,
    wmma16_transposed_scalar_pattern,
    mfma16_kcontig_pattern,
    mfma32_kcontig_pattern,
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
    # name: (element_bytes, element_bits, tr_inst_bits)
    "f32":  (4, 32, None),  # wmma_f32_16x16x4_f32,   no transposed load
    "fp16": (2, 16, 128),   # wmma_f32_16x16x32_f16,  ds_load_tr16_b128
    "fp8":  (1,  8,  64),   # wmma_f32_16x16x64_fp8,  ds_load_tr8_b64
}

# Column widths (inner/contiguous tile dimension) to sweep.
# Row count is irrelevant for bank conflict analysis — conflicts depend
# only on the row stride (cols + padding) and the per-lane access pattern.
COL_WIDTHS = [8, 16, 32, 64, 128, 256]

PADDING_SWEEP = [0, 1, 2, 4, 8, 16, 32]

# kWidth for WMMA v3 is hardcoded to 8 for ALL dtypes.
# Reference: AccelerateAMDMatmul.cpp:1628-1629
#     kWidth = wmmaVersion == 3 ? 8 : kBase;
#
# kWidth=8 means each thread reads 8 consecutive K elements per LDS load.
# For the non-transposed padding formula, the actual LDS load granularity
# is min(kWidth, 128/elemBits):
#   f32:  min(8, 4) = 4  (8 f32 elems = 256 bits, needs two ds_load_b128)
#   fp16: min(8, 8) = 8  (8 fp16 elems = 128 bits = one ds_load_b128)
#   fp8:  min(8, 16)= 8  (8 fp8 elems = 64 bits, one ds_load_b64)
WMMA_V3_KWIDTH = 8


# ============================================================================
# Helpers
# ============================================================================

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


def conflict_marker(conflict):
    """Return a visual quality marker for a conflict count."""
    if conflict <= 2:
        return "  "
    if conflict <= 4:
        return " !"
    return " X"


def print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                      col_widths, min_cols=1, indent=4):
    """
    Print a conflict table: rows = column widths, columns = padding values.

    Each cell shows the max bank-conflict degree. The header shows padding
    overhead (%) for a reference column width.

    Args:
        pattern:      AccessPattern to analyze
        elem_bytes:   bytes per element
        pads:         list of padding values to sweep
        proposed_pad: proposed padding value (for labeling)
        col_widths:   list of column widths to test
        min_cols:     skip cols < min_cols
        indent:       number of leading spaces
    """
    prefix = " " * indent
    col_w = 9

    # Header: pad values with proposed tag
    hdr = f"{prefix}{'cols':>5s} |"
    for p in pads:
        tag = " PRO" if p == proposed_pad else ""
        label = f"p={p}{tag}"
        hdr += f" {label:>{col_w}s}"
    print(hdr)

    # Overhead % row (for the largest column width as reference)
    ref_cols = max(c for c in col_widths if c >= min_cols)
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
    for cols in col_widths:
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
    print("#" * 80)
    print("# PART 1: TRANSPOSED LOAD PATH (ds_load_tr*)")
    print("#")
    print("# Used when loadTransposed = (order[0] != (1 - opIdx)) is True.")
    print("# The transposed instruction has a FIXED access pattern per dtype.")
    print("# kWidth does NOT affect the transposed access pattern.")
    print("#" * 80)
    print()

    for dtype_name, (elem_bytes, elem_bits, tr_inst_bits) in DTYPES.items():
        if tr_inst_bits is None:
            print(f"  {dtype_name}: No transposed load instruction. Skipping.")
            print()
            continue

        tr_elems = tr_inst_bits // elem_bits
        proposed_pad = 2 * tr_elems  # 2 × instBitWidth/elemBits (matches Utility.cpp)

        if elem_bits == 8:
            pattern = ds_load_tr8_b64_pattern()
        else:
            pattern = ds_load_tr16_b128_pattern()
        pads = PADDING_SWEEP

        print(f"  {dtype_name} ({elem_bits}-bit): ds_load_tr{elem_bits}_b{tr_inst_bits}")
        print(f"    {tr_elems} elems/lane, proposed pad={proposed_pad}")
        print()

        # ds_load_tr* operates on a full 16x16 sub-tile. Tiles narrower
        # than 16 columns can't use the transposed instruction.
        print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                          COL_WIDTHS, min_cols=16, indent=4)
        print()


# ============================================================================
# Part 2: Non-transposed load path sweep
# ============================================================================

def sweep_non_transposed():
    """
    Sweep the non-transposed load path (ds_load_b* / ds_load_2addr_b64).

    Used for dot operands where loadTransposed=False. The access pattern
    depends on kWidth (vectorization width along K dimension).
    """
    print()
    print("#" * 80)
    print("# PART 2: NON-TRANSPOSED LOAD PATH (ds_load_b* / ds_load_2addr_b64)")
    print("#")
    print("# Used when loadTransposed = False.")
    print("# kWidth=8 for all WMMA v3 dtypes (AccelerateAMDMatmul.cpp:1629).")
    print("# Effective load width = min(kWidth, 128/elemBits).")
    print("#")
    print("# fp8 uses ds_load_2addr_b64 (dual-addr 64-bit load, 8 bytes/lane).")
    print("# fp16 uses ds_load_b128 (128-bit load, 16 bytes/lane).")
    print("# f32  uses 2× ds_load_b128 (two 128-bit loads, 16+16 bytes/lane).")
    print("#")
    print("# Per sub-load the lane→bank mapping is identical across all dtypes")
    print("# (same lane_bits), so bank conflict analysis is per-sub-load.")
    print("#" * 80)
    print()

    for dtype_name, (elem_bytes, elem_bits, _tr_inst_bits) in DTYPES.items():
        kw = WMMA_V3_KWIDTH
        effective_kw = min(kw, 128 // elem_bits)

        if elem_bits == 8:
            pattern = ds_load_2addr_b64_pattern(kWidth=effective_kw)
            inst_name = "ds_load_2addr_b64"
        else:
            pattern = wmma16_kcontig_pattern(kWidth=effective_kw)
            inst_name = "ds_load_b128" if effective_kw * elem_bytes <= 16 else "2x ds_load_b128"

        proposed_pad = 128 // elem_bits
        pads = PADDING_SWEEP

        print(f"  {dtype_name} ({elem_bits}-bit): non-transposed {inst_name}")
        print(f"    kWidth={kw}, effective load width={effective_kw}, "
              f"proposed pad={proposed_pad}")
        print()

        print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                          COL_WIDTHS, min_cols=effective_kw * 2, indent=4)
        print()


# ============================================================================
# Part 3: Transposed scalar fallback (f32)
# ============================================================================

def sweep_transposed_scalar():
    """
    Sweep the transposed scalar fallback path.

    Used when loadTransposed=True but TransLocalLoadOpConversion fails.
    This happens when:
      - f32: no ds_load_tr32 instruction exists (bitWidth check fails)
      - fp16/fp8: layout doesn't match transposed instruction requirements
        (e.g. minInterval < tileSize, divideLeft fails, tile too narrow)

    In all cases, the generic LocalLoadOpConversion is used.
    largestVectorisation finds vec=1 for the strided K dimension,
    emitting scalar ds_load_b{elemBits} per element per lane.

    Access pattern (from wmmaDotOperandToLinearLayout):
        - 32 lanes, depth=2, kWidth=8
        - Lanes 0-15:  K_row = 0,       col = lane
        - Lanes 16-31: K_row = kWidth,   col = lane - 16
        - Each lane issues kWidth scalar loads (one per K element)

    Sub-dword note: for fp16 (2B) and fp8 (1B), multiple elements share
    a 4-byte LDS dword.  Accesses to the same dword are broadcasts, not
    conflicts.  The analyzer accounts for this.
    """
    print()
    print("#" * 80)
    print("# PART 3: TRANSPOSED SCALAR FALLBACK")
    print("#")
    print("# Used when loadTransposed=True but TransLocalLoadOpConversion fails:")
    print("#   f32:  always (no ds_load_tr32 instruction)")
    print("#   fp16: layout mismatch (minInterval < tileSize, divideLeft, ...)")
    print("#   fp8:  layout mismatch (same conditions as fp16)")
    print("#")
    print("# Falls back to scalar ds_load_b{elemBits}, vec=1.")
    print("# kWidth=8 scalar loads per thread per tile.")
    print("#")
    print("# Sub-dword elements (fp16/fp8): same-dword accesses are LDS")
    print("# broadcasts, not conflicts.  The analyzer accounts for this.")
    print("#" * 80)
    print()

    for dtype_name, (elem_bytes, elem_bits, tr_inst_bits) in DTYPES.items():
        kw = WMMA_V3_KWIDTH
        # The padding is set by getPaddedEncodingForDotOp BEFORE the
        # TransLocalLoadOpConversion runs.  For types with a transposed
        # instruction, pad = instBitWidth/elemBits.  For f32 (no
        # transposed instruction), pad = 128/elemBits (fallback).
        # The scalar fallback reuses whichever padding was already set.
        if tr_inst_bits is not None:
            proposed_pad = tr_inst_bits // elem_bits
        else:
            proposed_pad = 128 // elem_bits

        pattern = wmma16_transposed_scalar_pattern(kWidth=kw)
        pads = PADDING_SWEEP

        print(f"  {dtype_name} ({elem_bits}-bit): transposed scalar ds_load_b{elem_bits}")
        print(f"    kWidth={kw}, 1 elem/load, {kw} loads/thread, proposed pad={proposed_pad}")
        print(f"    cols = nonK tile dimension (M for opA, N for opB)")
        if elem_bytes < 4:
            elems_per_dword = 4 // elem_bytes
            print(f"    ({elems_per_dword} elems/dword → same-dword = broadcast)")
        print()

        print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                          COL_WIDTHS, min_cols=16, indent=4)
        print()


# ============================================================================
# Part 4: MFMA f32 non-transposed (CDNA / MI350)
# ============================================================================

# f32 MFMA geometries: (name, nonKDim, kWidth, kTileSize, pattern_fn)
MFMA_F32_GEOMETRIES = [
    ("mfma_f32_16x16x4f32", 16, 4, 16, mfma16_kcontig_pattern),
    ("mfma_f32_32x32x2f32", 32, 2,  4, mfma32_kcontig_pattern),
]

# BLOCK_K values relevant for f32 MFMA (must be >= kTileSize)
MFMA_F32_COL_WIDTHS = [4, 8, 16, 32, 64, 128]


def sweep_mfma_f32_non_transposed():
    """
    Sweep the non-transposed load path for f32 MFMA dot operands.

    f32 MFMA uses standard ds_load_b* (not transposed), with vectorization
    width = kWidth (kWidth * 4 bytes <= 128 bits for both geometries).

    From mfmaDotToLinearLayout (LinearLayoutConversions.cpp):
        regs  = identity1D(kWidth, register, K)
        lanes = identity1D(nonKDim, lane, nonK)
              * identity1D(warpSize/nonKDim, lane, K)

    warpSize = 64, lanes_per_cycle = 16.

    Optimal padding: pad = kWidth (i.e. min(kWidth, 128/32) = kWidth).
    This ensures gcd(stride_dwords, 64) = kWidth, making 16 lanes × kWidth
    dwords/lane fit into 64 banks without collision.
    """
    print()
    print("#" * 80)
    print("# PART 4: MFMA f32 NON-TRANSPOSED (CDNA / MI350)")
    print("#")
    print("# warpSize=64, lanes_per_cycle=16, element=f32 (4 bytes = 1 dword).")
    print("# Vectorized load width = kWidth dwords per lane.")
    print("# padAmount = min(kWidth, 128/32) = kWidth.")
    print("#" * 80)
    print()

    elem_bytes = 4
    for name, nonKDim, kWidth, kTileSize, pattern_fn in MFMA_F32_GEOMETRIES:
        pattern = pattern_fn(kWidth=kWidth)
        proposed_pad = kWidth   # min(kWidth, 128/32) = kWidth for f32
        pads = PADDING_SWEEP

        print(f"  {name}: non-transposed ds_load_b{kWidth * 32}")
        print(f"    nonKDim={nonKDim}, kWidth={kWidth}, kTileSize={kTileSize}")
        print(f"    {kWidth} dwords/lane/load, proposed pad={proposed_pad}")
        print()

        valid_cols = [c for c in MFMA_F32_COL_WIDTHS if c >= kTileSize]
        print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                          valid_cols, min_cols=kTileSize, indent=4)
        print()


# ============================================================================
# Part 5: Summary
# ============================================================================

def sweep_summary(ref_cols=128):
    """
    Print a compact summary of proposed padding for all paths.
    """
    print()
    print("#" * 80)
    print("# PART 5: PROPOSAL SUMMARY (ref cols=128)")
    print("#")
    print("# WMMA (gfx1250):")
    print("#   Transposed (tr*):  padAmount = 2 * instBitWidth / elemBits")
    print("#   Transposed scalar: padAmount = 2 * instBitWidth / elemBits (fallback)")
    print("#   Non-transposed:    padAmount = 128 / elemBits")
    print("#")
    print("# MFMA f32 (CDNA):")
    print("#   Non-transposed:    padAmount = kWidth = min(kWidth, 128 / 32)")
    print("#" * 80)
    print()

    print(f"  {'Path':<22s} {'dtype':>5s} {'proposed':>9s} "
          f"{'conflict':>9s} {'overhead':>9s}")
    print(f"  {'-'*21:<22s} {'-----':>5s} {'---------':>9s} "
          f"{'---------':>9s} {'---------':>9s}")

    for dtype_name, (elem_bytes, elem_bits, tr_inst_bits) in DTYPES.items():
        # Transposed path (ds_load_tr*)
        if tr_inst_bits is not None:
            tr_elems = tr_inst_bits // elem_bits
            proposed = 2 * tr_elems  # matches Utility.cpp: 2 * instBitWidth/elemBits
            if elem_bits == 8:
                pattern = ds_load_tr8_b64_pattern()
            else:
                pattern = ds_load_tr16_b128_pattern()
            pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern)
            print(f"  {'wmma transposed':<22s} {dtype_name:>5s} {proposed:9d} "
                  f"{pro_c:2d}-way    "
                  f"{overhead_pct(proposed, ref_cols):7.1f}%")

        # Transposed scalar fallback (all dtypes)
        kw = WMMA_V3_KWIDTH
        proposed = 2 * tr_inst_bits // elem_bits if tr_inst_bits is not None else 128 // elem_bits
        pattern = wmma16_transposed_scalar_pattern(kWidth=kw)
        pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern)
        print(f"  {'wmma tr-scalar':<22s} {dtype_name:>5s} {proposed:9d} "
              f"{pro_c:2d}-way    "
              f"{overhead_pct(proposed, ref_cols):7.1f}%")

        # Non-transposed path (WMMA)
        kw = WMMA_V3_KWIDTH
        effective_kw = min(kw, 128 // elem_bits)
        proposed = 128 // elem_bits
        if elem_bits == 8:
            pattern = ds_load_2addr_b64_pattern(kWidth=effective_kw)
        else:
            pattern = wmma16_kcontig_pattern(kWidth=effective_kw)
        pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern)
        print(f"  {'wmma non-transposed':<22s} {dtype_name:>5s} {proposed:9d} "
              f"{pro_c:2d}-way    "
              f"{overhead_pct(proposed, ref_cols):7.1f}%")
        print()

    # MFMA f32 non-transposed
    elem_bytes = 4
    for name, nonKDim, kWidth, kTileSize, pattern_fn in MFMA_F32_GEOMETRIES:
        pattern = pattern_fn(kWidth=kWidth)
        proposed = kWidth
        pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern)
        short_name = f"{nonKDim}x{nonKDim}" if nonKDim != 16 else "16x16"
        print(f"  {'mfma ' + short_name + ' non-tr':<22s} {'f32':>5s} {proposed:9d} "
              f"{pro_c:2d}-way    "
              f"{overhead_pct(proposed, ref_cols):7.1f}%")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("LDS PADDING COMPREHENSIVE SWEEP")
    print("=" * 80)
    print()
    print("Hardware: 64 banks, 4 bytes/bank")
    print("WMMA (gfx1250):  warp_size=32, nonKDim=16")
    print("MFMA (CDNA/MI350): warp_size=64, nonKDim=16 or 32")

    sweep_transposed()
    sweep_non_transposed()
    sweep_transposed_scalar()
    sweep_mfma_f32_non_transposed()
    sweep_summary()


if __name__ == "__main__":
    main()
