#!/usr/bin/env python3
"""
Comprehensive LDS padding sweep for gfx1250 WMMA dot operands.

For each (dtype, col_width) combination, reports bank conflicts for a range
of padding values, separately for each load path.
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
        proposed_pad = tr_elems

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
    Sweep the non-transposed load path (ds_load_b*).

    Used for dot operands where loadTransposed=False. The access pattern
    depends on kWidth (vectorization width along K dimension).
    """
    print()
    print("#" * 80)
    print("# PART 2: NON-TRANSPOSED LOAD PATH (ds_load_b*)")
    print("#")
    print("# Used when loadTransposed = False.")
    print("# kWidth=8 for all WMMA v3 dtypes (AccelerateAMDMatmul.cpp:1629).")
    print("# Effective load width = min(kWidth, 128/elemBits).")
    print("#" * 80)
    print()

    for dtype_name, (elem_bytes, elem_bits, _tr_inst_bits) in DTYPES.items():
        kw = WMMA_V3_KWIDTH
        proposed_pad = min(kw, 128 // elem_bits)
        pattern = wmma16_kcontig_pattern(kWidth=kw)
        pads = PADDING_SWEEP

        print(f"  {dtype_name} ({elem_bits}-bit): non-transposed ds_load_b*")
        print(f"    kWidth={kw}, effective load width={proposed_pad}, "
              f"proposed pad={proposed_pad}")
        print()

        print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                          COL_WIDTHS, min_cols=kw * 2, indent=4)
        print()


# ============================================================================
# Part 3: Summary
# ============================================================================

def sweep_summary(ref_cols=128):
    """
    Print a compact summary of proposed padding for both paths.
    """
    print()
    print("#" * 80)
    print("# PART 3: PROPOSAL SUMMARY (ref cols=128)")
    print("#")
    print("# Transposed:     padAmount = instBitWidth / elemBits")
    print("# Non-transposed: padAmount = min(kWidth, 128 / elemBits)")
    print("#" * 80)
    print()

    print(f"  {'Path':<15s} {'dtype':>5s} {'proposed':>9s} "
          f"{'conflict':>9s} {'overhead':>9s}")
    print(f"  {'-'*14:<15s} {'-----':>5s} {'---------':>9s} "
          f"{'---------':>9s} {'---------':>9s}")

    for dtype_name, (elem_bytes, elem_bits, tr_inst_bits) in DTYPES.items():
        # Transposed path
        if tr_inst_bits is not None:
            tr_elems = tr_inst_bits // elem_bits
            proposed = tr_elems
            pattern = ds_load_tr16_b128_pattern()
            pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern)
            print(f"  {'transposed':<15s} {dtype_name:>5s} {proposed:9d} "
                  f"{pro_c:2d}-way    "
                  f"{overhead_pct(proposed, ref_cols):7.1f}%")

        # Non-transposed path
        kw = WMMA_V3_KWIDTH
        proposed = min(kw, 128 // elem_bits)
        pattern = wmma16_kcontig_pattern(kWidth=kw)
        pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern)
        print(f"  {'non-transposed':<15s} {dtype_name:>5s} {proposed:9d} "
              f"{pro_c:2d}-way    "
              f"{overhead_pct(proposed, ref_cols):7.1f}%")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("gfx1250 LDS PADDING COMPREHENSIVE SWEEP")
    print("=" * 80)
    print()
    print("Hardware: 64 banks, 4 bytes/bank, warp_size=32, WMMA nonKDim=16")

    sweep_transposed()
    sweep_non_transposed()
    sweep_summary()


if __name__ == "__main__":
    main()
