#!/usr/bin/env python3
"""
Cross-wave bank conflict model for validating the LDS bank conflict analyzer
against the AM simulator's DS_READ_BANK_CONFLICTS_SUM counter.

Key findings:
1. gfx1250 LDS banks are DUAL-PORTED (2 requests per cycle per bank)
2. AM counter = sum over all ds_load wave-executions of stall_cycles
3. stall_cycles = max(0, ceil(total_ways / 2) - 1)
4. Both ds_load_b128 and ds_load_tr16 follow the same conflict rules
5. A operand: cross-wave multiplier G comes from M-group aliasing
6. B operand: cross-wave multiplier G comes from N-group aliasing

Evidence (3 data points):
- Isolated test (1 wave, 2-way intra): counter = 0 -> dual ports absorb 2-way
- GEMM BN=128 (b128: G_A=2, tr16: G_B=1): 2048 -> only A contributes
- GEMM BN=256 (b128: G_A=2, tr16: G_B=2): 4096 -> A + B both contribute

See ../lds_analytical_model.md for full derivation.

Usage:
    python cross_wave_model.py
"""

import os
import sys
import math

# Import from the parent lds/ directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lds_bank_conflict_analyzer import (
    LDSConfig, AccessPattern, wmma16_kcontig_pattern, ds_load_tr16_b128_pattern,
    compute_lane_accesses, find_bank_conflicts
)


def predict_stall_cycles(intra_wave_ways, num_aliasing_groups, ports_per_bank=2):
    """Predict extra stall cycles per ds_load execution on dual-port LDS."""
    total_ways = intra_wave_ways * num_aliasing_groups
    return max(0, math.ceil(total_ways / ports_per_bank) - 1)


def compute_n_group_aliasing(block_n, num_n_groups, element_bytes=2, num_banks=64, bytes_per_bank=4):
    """Compute how many N-groups alias to the same bank."""
    cols_per_group = block_n // num_n_groups
    byte_sep = cols_per_group * element_bytes
    bank_row_bytes = num_banks * bytes_per_bank

    banks = [(ng * byte_sep // bytes_per_bank) % num_banks for ng in range(num_n_groups)]
    from collections import Counter
    return max(Counter(banks).values()), banks


def compute_cross_wave_detail(config, pattern, m_group_separation):
    """
    Verify cross-wave conflict by checking if rows separated by m_group_separation
    hit the same banks.
    """
    stride = config.row_stride_bytes
    bank_bytes = config.num_banks * config.bytes_per_bank
    offset = (m_group_separation * stride) % bank_bytes
    return offset == 0, stride, bank_bytes, offset


def main():
    print("=" * 80)
    print("ANALYTICAL MODEL: Mapping LDS Bank Conflict Analyzer -> AM Counter")
    print("=" * 80)

    # =========================================================================
    # Kernel parameters (fp16 GEMM from the case study)
    # =========================================================================
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    NUM_WARPS = 8
    K_TOTAL = 1024
    NUM_ITER = K_TOTAL // BLOCK_K
    WMMA_M = 16

    NUM_M_GROUPS = 2
    ROWS_PER_M_GROUP = BLOCK_M // NUM_M_GROUPS
    PORTS_PER_BANK = 2

    # Instruction counts from AM dumpPerDrawPerf.csv (fp16 run)
    DS_B128_MAIN = 16 * 120
    DS_B128_EPILOGUE = 16 * 8
    DS_B128_TOTAL = DS_B128_MAIN + DS_B128_EPILOGUE

    DS_TR16_MAIN = 8 * 120
    DS_TR16_EPILOGUE = 8 * 8
    DS_TR16_TOTAL = DS_TR16_MAIN + DS_TR16_EPILOGUE

    AM_COUNTER_ACTUAL = 2048

    # =========================================================================
    # LDS layouts from TTGIR: #ttg.padded_shared
    # =========================================================================
    configs = {
        "padding=8": {
            "A": LDSConfig.with_padding(8, row_width=64, element_bytes=2, num_banks=64),
            "B": LDSConfig.with_padding(8, row_width=128, element_bytes=2, num_banks=64),
        },
        "no_padding": {
            "A": LDSConfig.with_padding(0, row_width=64, element_bytes=2, num_banks=64),
            "B": LDSConfig.with_padding(0, row_width=128, element_bytes=2, num_banks=64),
        },
    }

    pattern_a = wmma16_kcontig_pattern(kWidth=8)
    pattern_b = ds_load_tr16_b128_pattern()

    # =========================================================================
    # Step 1: Intra-wave analysis (what the analyzer computes)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 1: Intra-wave conflict (analyzer output)")
    print("=" * 80)

    for label, cfg in configs.items():
        accesses_a = compute_lane_accesses(cfg["A"], pattern_a)
        _, intra_a = find_bank_conflicts(accesses_a, num_banks=64)

        accesses_b = compute_lane_accesses(cfg["B"], pattern_b)
        _, intra_b = find_bank_conflicts(accesses_b, num_banks=64)

        print(f"\n  {label}:")
        print(f"    A tile (DS_LOAD_B128): stride={cfg['A'].row_stride_bytes}B -> {intra_a}-way")
        print(f"    B tile (DS_LOAD_TR16): stride={cfg['B'].row_stride_bytes}B -> {intra_b}-way")

    # =========================================================================
    # Step 2: Cross-wave structure
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 2: Cross-wave structure")
    print("=" * 80)

    print("""
  Wave tiling (ctaLayout warp=[[0,1],[0,2],[1,0]]):
    8 warps = 2 M-groups x 4 N-groups
    Waves 0-3 (M_group 0): load A rows 0-63
    Waves 4-7 (M_group 1): load A rows 64-127

  Within each M-group: all 4 waves load identical A tile rows -> BROADCAST
  Between M-groups: different rows, but may hit same banks -> CONFLICT""")

    for label, cfg in configs.items():
        same_bank, stride, bank_bytes, offset = compute_cross_wave_detail(
            cfg["A"], pattern_a, ROWS_PER_M_GROUP
        )
        print(f"\n  {label}: stride={stride}B")
        print(f"    {ROWS_PER_M_GROUP} rows x {stride} B/row = {ROWS_PER_M_GROUP * stride} B")
        print(f"    {ROWS_PER_M_GROUP * stride} mod {bank_bytes} = {offset}")
        if same_bank:
            print(f"    -> Row N and row N+{ROWS_PER_M_GROUP} hit SAME banks -> cross-wave conflict")
        else:
            print(f"    -> Row N and row N+{ROWS_PER_M_GROUP} hit DIFFERENT banks -> no cross-wave conflict!")

    # =========================================================================
    # Step 3: Dual-port model + counter prediction
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 3: Dual-port LDS bank model")
    print("=" * 80)

    print("""
  gfx1250 LDS banks have 2 ports (service 2 requests/cycle/bank)

  Formula:
    stall_cycles = max(0, ceil(total_ways / ports_per_bank) - 1)

  For A operand (ds_load_b128):
    total_ways = intra_wave_ways x G_A  (G_A = aliasing M-groups)

  For B operand (ds_load_tr16):
    total_ways = intra_wave_ways x G_B  (G_B = aliasing N-groups)

  DS_READ_BANK_CONFLICTS_SUM = sum of stall_cycles over ALL ds_load executions""")

    print(f"\n  {'Scenario':<43} | Intra | Groups | Total | Stall | x Execs | Predicted")
    print(f"  {'-'*43}-+-------+--------+-------+-------+---------+----------")

    for label, cfg in configs.items():
        accesses_a = compute_lane_accesses(cfg["A"], pattern_a)
        _, intra_a = find_bank_conflicts(accesses_a, num_banks=64)

        stall = predict_stall_cycles(intra_a, NUM_M_GROUPS, PORTS_PER_BANK)
        predicted = DS_B128_TOTAL * stall
        marker = " <-- AM actual" if label == "padding=8" else ""

        print(f"  A {label:<40} | {intra_a:>5} | {NUM_M_GROUPS:>6} | {intra_a * NUM_M_GROUPS:>5} | {stall:>5} | {DS_B128_TOTAL:>7} | {predicted:>8}{marker}")

    # =========================================================================
    # Step 4: Validation (BLOCK_N=128 and BLOCK_N=256)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 4: Validation against AM")
    print("=" * 80)

    validation_cases = [
        {
            "name": "BLOCK_N=128 (baseline)",
            "block_n": 128,
            "b_row_width": 128,
            "ds_b128_total": 2048,
            "ds_tr16_total": 1024,
            "am_actual": 2048,
        },
        {
            "name": "BLOCK_N=256 (experiment)",
            "block_n": 256,
            "b_row_width": 256,
            "ds_b128_total": 2048,
            "ds_tr16_total": 2048,
            "am_actual": 4096,
        },
    ]

    cfg_padded_a = configs["padding=8"]["A"]
    accesses_a = compute_lane_accesses(cfg_padded_a, pattern_a)
    _, intra_a = find_bank_conflicts(accesses_a, num_banks=64)

    for case in validation_cases:
        print(f"\n  --- {case['name']} ---")

        # A operand
        stall_a = predict_stall_cycles(intra_a, NUM_M_GROUPS, PORTS_PER_BANK)
        conflicts_a = case["ds_b128_total"] * stall_a

        # B operand
        cfg_b = LDSConfig.with_padding(8, row_width=case["b_row_width"],
                                        element_bytes=2, num_banks=64)
        accesses_b = compute_lane_accesses(cfg_b, pattern_b)
        _, intra_b = find_bank_conflicts(accesses_b, num_banks=64)

        num_n_groups = 4
        g_b, banks_b = compute_n_group_aliasing(case["block_n"], num_n_groups)
        stall_b = predict_stall_cycles(intra_b, g_b, PORTS_PER_BANK)
        conflicts_b = case["ds_tr16_total"] * stall_b

        total_predicted = conflicts_a + conflicts_b

        print(f"    A (ds_load_b128): W={intra_a}, G_A={NUM_M_GROUPS} -> stall={stall_a}")
        print(f"      {case['ds_b128_total']} execs x {stall_a} = {conflicts_a}")
        print(f"    B (ds_load_tr16): W={intra_b}, G_B={g_b} (banks: {banks_b})")
        print(f"      {case['ds_tr16_total']} execs x {stall_b} = {conflicts_b}")
        print(f"    Predicted: {conflicts_a} + {conflicts_b} = {total_predicted}")
        print(f"    AM actual: {case['am_actual']}")
        print(f"    MATCH: {'YES' if total_predicted == case['am_actual'] else 'NO'}")

    # =========================================================================
    # Step 5: What-if analysis
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("STEP 5: What-if analysis (impact of padding)")
    print("=" * 80)

    scenarios = [
        ("No padding (baseline)",  0),
        ("Padding = 2 elements",   2),
        ("Padding = 4 elements",   4),
        ("Padding = 8 elements (current fork)", 8),
        ("Padding = 16 elements", 16),
        ("Padding = 32 elements", 32),
    ]

    print(f"\n  {'Padding':<40} | Stride | Intra | Total | Stall | Predicted | vs pad=8")
    print(f"  {'-'*40}-+--------+-------+-------+-------+-----------+-----------")

    # Compute pad=8 baseline for ratio column
    cfg_pad8 = LDSConfig.with_padding(8, row_width=64, element_bytes=2, num_banks=64)
    acc_pad8 = compute_lane_accesses(cfg_pad8, pattern_a)
    _, intra_pad8 = find_bank_conflicts(acc_pad8, num_banks=64)
    stall_pad8 = predict_stall_cycles(intra_pad8, NUM_M_GROUPS, PORTS_PER_BANK)
    current_predicted = DS_B128_TOTAL * stall_pad8
    for label, pad in scenarios:
        cfg = LDSConfig.with_padding(pad, row_width=64, element_bytes=2, num_banks=64)
        acc = compute_lane_accesses(cfg, pattern_a)
        _, intra = find_bank_conflicts(acc, num_banks=64)

        same_bank, _, _, _ = compute_cross_wave_detail(cfg, pattern_a, ROWS_PER_M_GROUP)
        if same_bank:
            total = intra * NUM_M_GROUPS
        else:
            total = intra

        stall = predict_stall_cycles(intra, NUM_M_GROUPS if same_bank else 1, PORTS_PER_BANK)
        pred = DS_B128_TOTAL * stall

        ratio = f"{pred / current_predicted:.1f}x" if current_predicted > 0 else "N/A"
        print(f"  {label:<40} | {cfg.row_stride_bytes:>4}B  | {intra:>5} | {total:>5} | {stall:>5} | {pred:>9} | {ratio:>9}")

    # =========================================================================
    # Summary equation
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("COMPLETE ANALYTICAL MODEL")
    print("=" * 80)
    print("""
  DS_READ_BANK_CONFLICTS_SUM =

    sum  max(0, ceil((W x G) / P) - 1)
    over all ds_load wave-executions (b128 AND tr16)

  Where:
    W = intra-wave conflict from lds_bank_conflict_analyzer.py
    G = number of aliasing wave-groups (G_A for A/M-groups, G_B for B/N-groups)
    P = 2 (gfx1250 dual-ported LDS banks)

  Both ds_load_b128 and ds_load_tr16 follow the same rules.
  G depends on the operand's tiling dimension and bank aliasing pattern.
""")


if __name__ == "__main__":
    main()
