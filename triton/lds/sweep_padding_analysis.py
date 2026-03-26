#!/usr/bin/env python3
"""
Comprehensive LDS padding sweep for dot operands.

For each (dtype, col_width) combination, reports bank conflicts for a range
of padding values, separately for each load path.

Covers:
  - MI450/gfx1250 WMMA paths (64 banks, transposed / non-transposed / scalar)
  - MI350/CDNA3 MFMA paths (32 banks, f32/bf16/fp8, non-transposed + transposed scalar)
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
    wmma_kcontig_pattern,
    wmma_transposed_scalar_pattern,
    mfma16_kcontig_pattern,
    mfma32_kcontig_pattern,
    mfma16_transposed_scalar_pattern,
    mfma32_transposed_scalar_pattern,
)


# ============================================================================
# Hardware and sweep configuration
# ============================================================================

MI450_NUM_BANKS = 64   # gfx1250 WMMA — padding layout
MI350_NUM_BANKS = 32   # gfx942  MFMA — swizzle layout
BYTES_PER_BANK = 4     # same on both architectures

DTYPES = {
    # One representative per bit-width. bf16 behaves identically to fp16,
    # and bf8/i8 behave identically to fp8 -- same element size and same
    # transposed instruction.
    #
    # name: (element_bytes, element_bits, tr_inst_bits)
    #
    # kBase reference (from WmmaGroup.cpp WMMA v3 intrinsics):
    #   f32:  wmma_f32_16x16x4_f32   -> kDim=4,   kBase=2
    #   fp16: wmma_f32_16x16x32_f16  -> kDim=32,  kBase=16
    #   fp8:  wmma_f32_16x16x64_fp8  -> kDim=64,  kBase=32
    #   iu8:  wmma_i32_16x16x64_iu8  -> kDim=64,  kBase=32
    #
    # kWidth for WMMA v3: always 8 (AccelerateAMDMatmul.cpp:1628-1629).
    # kBase only matters if the compiler switches to kWidth = min(8, kBase).
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

def get_max_conflict(row_width, pad, element_bytes, pattern,
                     pad_interval=0, num_banks=MI450_NUM_BANKS):
    """Return max bank conflict for one (row_width, padding, pattern) config."""
    config = LDSConfig(
        num_banks=num_banks,
        bytes_per_bank=BYTES_PER_BANK,
        row_width_elements=row_width,
        padding_elements=pad,
        element_bytes=element_bytes,
        pad_interval=pad_interval,
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
                      col_widths, min_cols=1, indent=4,
                      use_bank_wrap=False, num_banks=MI450_NUM_BANKS):
    """
    Print a conflict table: rows = column widths, columns = padding values.

    Each cell shows the max bank-conflict degree. The header shows padding
    overhead (%) for a reference column width.

    Args:
        pattern:        AccessPattern to analyze
        elem_bytes:     bytes per element
        pads:           list of padding values to sweep
        proposed_pad:   proposed padding value (for labeling)
        col_widths:     list of column widths to test
        min_cols:       skip cols < min_cols
        indent:         number of leading spaces
        use_bank_wrap:  If True, set pad_interval to the bank-wrap boundary
                        (numBanks * bankBytes / elemBytes).  The analyzer's
                        effective_pad_interval clamps to max(row_width, value).
        num_banks:      LDS bank count (64 for MI450, 32 for MI350).
    """
    pi = bank_wrap_interval(elem_bytes, num_banks) if use_bank_wrap else 0
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
                c = get_max_conflict(cols, p, elem_bytes, pattern,
                                     pad_interval=pi, num_banks=num_banks)
                mk = conflict_marker(c)
                cell = f"{c:2d}-way{mk}"
                line += f" {cell:>{col_w}s}"
        print(line)


# ============================================================================
# Part 1: Transposed load path sweep
# ============================================================================

def sweep_transposed(use_bank_wrap=False, num_banks=MI450_NUM_BANKS):
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
                          COL_WIDTHS, min_cols=16, indent=4,
                          use_bank_wrap=use_bank_wrap, num_banks=num_banks)
        print()


# ============================================================================
# Part 2: Non-transposed load path sweep
# ============================================================================

def sweep_non_transposed(use_bank_wrap=False, num_banks=MI450_NUM_BANKS):
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
            pattern = wmma_kcontig_pattern(kWidth=effective_kw, element_bytes=elem_bytes)
            load_bytes = effective_kw * elem_bytes
            if load_bytes > 16:
                inst_name = "2x ds_load_b128"
            elif load_bytes >= 16:
                inst_name = "ds_load_b128"
            else:
                inst_name = "ds_load_b64"

        proposed_pad = 128 // elem_bits
        pads = PADDING_SWEEP

        print(f"  {dtype_name} ({elem_bits}-bit): non-transposed {inst_name}")
        print(f"    kWidth={kw}, effective load width={effective_kw}, "
              f"proposed pad={proposed_pad}")
        print()

        print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                          COL_WIDTHS, min_cols=effective_kw * 2, indent=4,
                          use_bank_wrap=use_bank_wrap, num_banks=num_banks)
        print()

        # fp8 K-contiguous: when elements are fully contiguous in LDS
        # (e.g., flash attention V operand), the compiler emits a plain
        # ds_load_b64 instead of the interleaved ds_load_2addr_b64.
        # 8 fp8 elems × 1 byte = 8 bytes = 64 bits → ds_load_b64,
        # lds_group_size=32 (expert-confirmed: all 32 threads at once).
        if elem_bits == 8:
            pattern_b64 = wmma_kcontig_pattern(kWidth=effective_kw,
                                               element_bytes=elem_bytes)
            print(f"  {dtype_name} ({elem_bits}-bit): non-transposed ds_load_b64 "
                  f"(K-contiguous, e.g. flash attention V)")
            print(f"    kWidth={kw}, effective load width={effective_kw}, "
                  f"lds_group_size=32, proposed pad={proposed_pad}")
            print()

            print_sweep_table(pattern_b64, elem_bytes, pads, proposed_pad,
                              COL_WIDTHS, min_cols=effective_kw * 2, indent=4,
                              use_bank_wrap=use_bank_wrap, num_banks=num_banks)
            print()


# ============================================================================
# Part 3: Transposed scalar fallback (f32)
# ============================================================================

def sweep_transposed_scalar(use_bank_wrap=False, num_banks=MI450_NUM_BANKS):
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

        pattern = wmma_transposed_scalar_pattern(kWidth=kw)
        pads = PADDING_SWEEP

        print(f"  {dtype_name} ({elem_bits}-bit): transposed scalar ds_load_b{elem_bits}")
        print(f"    kWidth={kw}, 1 elem/load, {kw} loads/thread, proposed pad={proposed_pad}")
        print(f"    cols = nonK tile dimension (M for opA, N for opB)")
        if elem_bytes < 4:
            elems_per_dword = 4 // elem_bytes
            print(f"    ({elems_per_dword} elems/dword → same-dword = broadcast)")
        print()

        print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                          COL_WIDTHS, min_cols=16, indent=4,
                          use_bank_wrap=use_bank_wrap, num_banks=num_banks)
        print()


# ============================================================================
# Part 4: MFMA non-transposed + transposed (CDNA / MI350)
# ============================================================================

# (instr_name, nonKDim, kWidth, kDim)
# kDim = minimum BLOCK_K (the instruction's K dimension)
MFMA_GEOMETRIES = {
    "f32": [
        ("mfma_f32_16x16x4_f32",   16, 4,   4),
        ("mfma_f32_32x32x2_f32",   32, 2,   2),
    ],
    "bf16": [
        ("mfma_f32_16x16x32_f16",  16, 8,  32),
        ("mfma_f32_32x32x16_f16",  32, 4,  16),
    ],
    "fp8": [
        ("mfma_f32_16x16x64_fp8",  16, 8,  64),
        ("mfma_f32_32x32x32_fp8",  32, 8,  32),
    ],
}

MFMA_DTYPES = {
    "f32":  (4, 32),
    "bf16": (2, 16),
    "fp8":  (1,  8),
}

MFMA_COL_WIDTHS = [8, 16, 32, 64, 128, 256]


def sweep_mfma_non_transposed(use_bank_wrap=False, num_banks=MI350_NUM_BANKS):
    """
    Sweep the non-transposed (K-contiguous) load path for MFMA dot operands.

    Each lane reads kWidth consecutive K elements via ds_load_b*.
    Covers f32, bf16, and fp8 for both 16x16 and 32x32 geometries.
    Defaults to MI350 (32 banks).
    """
    print()
    print("#" * 80)
    print(f"# PART 4A: MFMA NON-TRANSPOSED / K-CONTIGUOUS (CDNA / MI350, {num_banks} banks)")
    print("#")
    print("# warpSize=64, lanes_per_cycle=16.")
    print("# Each lane reads kWidth consecutive K elements via ds_load_b*.")
    print("#" * 80)
    print()

    for dtype_name, (elem_bytes, elem_bits) in MFMA_DTYPES.items():
        for instr_name, nonKDim, kWidth, kDim in MFMA_GEOMETRIES[dtype_name]:
            pattern = mfma16_kcontig_pattern(kWidth=kWidth) if nonKDim == 16 \
                else mfma32_kcontig_pattern(kWidth=kWidth)
            proposed_pad = min(kWidth, 128 // elem_bits)
            load_bits = kWidth * elem_bits
            pads = PADDING_SWEEP

            print(f"  {dtype_name} {instr_name}: non-transposed ds_load_b{load_bits}")
            print(f"    nonKDim={nonKDim}, kWidth={kWidth}, kDim={kDim}, "
                  f"proposed pad={proposed_pad}")
            print()

            valid_cols = [c for c in MFMA_COL_WIDTHS if c >= kDim]
            print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                              valid_cols, min_cols=kDim, indent=4,
                              use_bank_wrap=use_bank_wrap, num_banks=num_banks)
            print()


def sweep_mfma_transposed_scalar(use_bank_wrap=False, num_banks=MI350_NUM_BANKS):
    """
    Sweep the transposed scalar fallback path for MFMA dot operands.

    MI350/CDNA has no ds_load_tr* instruction — when loadTransposed=True,
    the compiler falls back to scalar ds_load_b32 loads (one element per
    lane per cycle, kWidth loads per thread total).

    The lane mapping is the same as K-contiguous but with rows/cols swapped:
    nonK is now contiguous (columns), K is strided (rows).
    The row_width here is the nonK tile dimension (M for opA, N for opB).
    """
    print()
    print("#" * 80)
    print(f"# PART 4B: MFMA TRANSPOSED SCALAR (CDNA / MI350, {num_banks} banks)")
    print("#")
    print("# No ds_load_tr* on MI350 — scalar ds_load_b32 fallback.")
    print("# nonK is contiguous (columns), K is strided (rows).")
    print("# cols = nonK tile dimension (M for opA, N for opB).")
    print("#" * 80)
    print()

    for dtype_name, (elem_bytes, elem_bits) in MFMA_DTYPES.items():
        for instr_name, nonKDim, kWidth, kDim in MFMA_GEOMETRIES[dtype_name]:
            pattern = mfma16_transposed_scalar_pattern(kWidth=kWidth) if nonKDim == 16 \
                else mfma32_transposed_scalar_pattern(kWidth=kWidth)
            proposed_pad = min(kWidth, 128 // elem_bits)
            pads = PADDING_SWEEP

            print(f"  {dtype_name} {instr_name}: transposed scalar ds_load_b{elem_bits}")
            print(f"    nonKDim={nonKDim}, kWidth={kWidth}, kDim={kDim}, 1 elem/load, "
                  f"{kWidth} loads/thread, proposed pad={proposed_pad}")
            if elem_bytes < 4:
                elems_per_dword = 4 // elem_bytes
                print(f"    ({elems_per_dword} elems/dword → same-dword = broadcast)")
            print()

            print_sweep_table(pattern, elem_bytes, pads, proposed_pad,
                              MFMA_COL_WIDTHS, min_cols=nonKDim, indent=4,
                              use_bank_wrap=use_bank_wrap, num_banks=num_banks)
            print()


# ============================================================================
# Part 5: Summary
# ============================================================================

def bank_wrap_interval(elem_bytes, num_banks=MI450_NUM_BANKS):
    """Bank-wrap interval: numBanks * bankBytes / elemBytes."""
    return num_banks * BYTES_PER_BANK // elem_bytes


def sweep_summary(ref_cols=128, use_bank_wrap=False,
                  wmma_banks=MI450_NUM_BANKS, mfma_banks=MI350_NUM_BANKS):
    """
    Print a compact summary of proposed padding for all paths.
    """
    print()
    print("#" * 80)
    print(f"# PART 5: PROPOSAL SUMMARY (ref cols={ref_cols})")
    print("#")
    print(f"# WMMA (MI450, {wmma_banks} banks):")
    print("#   Transposed (tr*):  padAmount = 2 * instBitWidth / elemBits")
    print("#   Transposed scalar: padAmount = 2 * instBitWidth / elemBits (fallback)")
    print("#   Non-transposed:    padAmount = 128 / elemBits")
    print(f"# MFMA (MI350, {mfma_banks} banks):")
    print("#   Non-transposed:    padAmount = kWidth")
    print("#" * 80)
    print()

    print(f"  {'Path':<22s} {'dtype':>5s} {'proposed':>9s} "
          f"{'conflict':>9s} {'overhead':>9s}")
    print(f"  {'-'*21:<22s} {'-----':>5s} {'---------':>9s} "
          f"{'---------':>9s} {'---------':>9s}")

    for dtype_name, (elem_bytes, elem_bits, tr_inst_bits) in DTYPES.items():
        pi = bank_wrap_interval(elem_bytes, wmma_banks) if use_bank_wrap else 0

        # Transposed path (ds_load_tr*)
        if tr_inst_bits is not None:
            tr_elems = tr_inst_bits // elem_bits
            proposed = 2 * tr_elems
            if elem_bits == 8:
                pattern = ds_load_tr8_b64_pattern()
            else:
                pattern = ds_load_tr16_b128_pattern()
            pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern,
                                     pad_interval=pi, num_banks=wmma_banks)
            print(f"  {'wmma transposed':<22s} {dtype_name:>5s} {proposed:9d} "
                  f"{pro_c:2d}-way    "
                  f"{overhead_pct(proposed, ref_cols):7.1f}%")

        # Transposed scalar fallback (all dtypes)
        kw = WMMA_V3_KWIDTH
        proposed = 2 * tr_inst_bits // elem_bits if tr_inst_bits is not None else 128 // elem_bits
        pattern = wmma_transposed_scalar_pattern(kWidth=kw)
        pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern,
                                 pad_interval=pi, num_banks=wmma_banks)
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
            pattern = wmma_kcontig_pattern(kWidth=effective_kw, element_bytes=elem_bytes)
        pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern,
                                 pad_interval=pi, num_banks=wmma_banks)
        print(f"  {'wmma non-transposed':<22s} {dtype_name:>5s} {proposed:9d} "
              f"{pro_c:2d}-way    "
              f"{overhead_pct(proposed, ref_cols):7.1f}%")

        # fp8 K-contiguous ds_load_b64 (flash attention V operand path)
        if elem_bits == 8:
            pattern_b64 = wmma_kcontig_pattern(kWidth=effective_kw,
                                               element_bytes=elem_bytes)
            pro_c_b64 = get_max_conflict(ref_cols, proposed, elem_bytes,
                                         pattern_b64, pad_interval=pi,
                                         num_banks=wmma_banks)
            print(f"  {'wmma non-tr (b64)':<22s} {dtype_name:>5s} {proposed:9d} "
                  f"{pro_c_b64:2d}-way    "
                  f"{overhead_pct(proposed, ref_cols):7.1f}%")

        print()

    # MFMA summary (all dtypes, both directions)
    for dtype_name, (elem_bytes, elem_bits) in MFMA_DTYPES.items():
        pi_mfma = bank_wrap_interval(elem_bytes, mfma_banks) if use_bank_wrap else 0
        for instr_name, nonKDim, kWidth, kDim in MFMA_GEOMETRIES[dtype_name]:
            proposed = min(kWidth, 128 // elem_bits)

            # Non-transposed (K-contiguous)
            pattern_kc = mfma16_kcontig_pattern(kWidth=kWidth) if nonKDim == 16 \
                else mfma32_kcontig_pattern(kWidth=kWidth)
            pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern_kc,
                                     pad_interval=pi_mfma, num_banks=mfma_banks)
            print(f"  {'mfma non-transposed':<22s} {dtype_name:>5s} {proposed:9d} "
                  f"{pro_c:2d}-way    "
                  f"{overhead_pct(proposed, ref_cols):7.1f}%")
            print(f"    ({instr_name}, nonKDim={nonKDim})")

            # Transposed scalar
            pattern_ts = mfma16_transposed_scalar_pattern(kWidth=kWidth) if nonKDim == 16 \
                else mfma32_transposed_scalar_pattern(kWidth=kWidth)
            pro_c = get_max_conflict(ref_cols, proposed, elem_bytes, pattern_ts,
                                     pad_interval=pi_mfma, num_banks=mfma_banks)
            print(f"  {'mfma tr-scalar':<22s} {dtype_name:>5s} {proposed:9d} "
                  f"{pro_c:2d}-way    "
                  f"{overhead_pct(proposed, ref_cols):7.1f}%")
            print(f"    ({instr_name}, nonKDim={nonKDim})")

        print()


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Comprehensive LDS padding sweep for dot operands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python sweep_padding_analysis.py                        # per-row padding (default)
  python sweep_padding_analysis.py --pad-mode bank-wrap   # bank-wrap padInterval
  python sweep_padding_analysis.py --num-banks 32         # override all parts to 32 banks
""")
    parser.add_argument(
        '--pad-mode', choices=['per-row', 'bank-wrap'], default='per-row',
        help='Padding strategy: per-row (default) inserts padding after every '
             'row. bank-wrap uses padInterval = max(cols, numBanks*4/elemBytes) '
             'so small tiles pad at the bank-wrap boundary instead.')
    parser.add_argument(
        '--num-banks', type=int, default=None,
        help='Override LDS bank count for ALL parts. '
             'Default: 64 for WMMA (MI450), 32 for MFMA (MI350).')
    args = parser.parse_args()

    use_bw = args.pad_mode == 'bank-wrap'
    wmma_banks = args.num_banks if args.num_banks is not None else MI450_NUM_BANKS
    mfma_banks = args.num_banks if args.num_banks is not None else MI350_NUM_BANKS

    print("=" * 80)
    print("LDS PADDING COMPREHENSIVE SWEEP")
    print("=" * 80)
    print()
    print(f"WMMA (MI450): {wmma_banks} banks, 4 bytes/bank, warp_size=32")
    print(f"MFMA (MI350): {mfma_banks} banks, 4 bytes/bank, warp_size=64")
    if args.num_banks is not None:
        print(f"  (--num-banks {args.num_banks} overrides both)")
    print(f"Pad mode: {args.pad_mode}")
    if use_bw:
        print("  padInterval = max(cols, numBanks * bankBytes / elemBytes)")

    sweep_transposed(use_bw, wmma_banks)
    sweep_non_transposed(use_bw, wmma_banks)
    sweep_transposed_scalar(use_bw, wmma_banks)
    sweep_mfma_non_transposed(use_bw, mfma_banks)
    sweep_mfma_transposed_scalar(use_bw, mfma_banks)
    sweep_summary(use_bank_wrap=use_bw, wmma_banks=wmma_banks, mfma_banks=mfma_banks)


if __name__ == "__main__":
    main()
