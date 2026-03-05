#!/usr/bin/env python3
"""
Cross-validation of lds_bank_conflict_analyzer.py swizzle logic against the
upstream plot_layout.py / tikzplot.tex (ROCm/triton-internal, branch triton-mlir).

The upstream TikZ math is re-implemented here in Python as the reference, then
bank assignments are compared element-by-element against our LDSConfig for
every (K, vec, layout) combination the upstream supports.

Run:
    python3 -m tests.test_swizzle_vs_upstream          # from lds/ directory
    python3 tests/test_swizzle_vs_upstream.py           # also works

Upstream assumptions (hardcoded in tikzplot.tex):
    - fp16 (2 bytes/element)
    - 32 LDS banks (LDSK = 64 elements = 128 bytes)
    - vec (kpack) in {4, 8}
    - lds_layout in {none, swizzle}
"""

import math
import os
import sys

# Allow running from lds/ or from lds/tests/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from lds_bank_conflict_analyzer import (
    LDSConfig, SwizzleMode, AccessPattern,
    compute_lane_accesses, find_bank_conflicts,
)


# ---------------------------------------------------------------------------
# Reference: upstream tikzplot.tex LDS swizzle math (Python transliteration)
# ---------------------------------------------------------------------------

def upstream_swizzle_bank(row: int, col: int, K: int, vec: int,
                          lds_layout: str) -> int:
    """Bank index for (row, col) as the upstream tikzplot.tex computes it."""
    LDSK = 64
    num_banks = 32
    elem_bytes = 2

    numVecK = K // vec
    swizzleK = max(LDSK, K)

    perPhase = math.ceil(LDSK / K)
    numVecSwizzleK = swizzleK // vec

    maxPhase = 1 if lds_layout == "none" else int(min(16 / perPhase, 64 / vec))

    vecCoordM = row
    vecCoordK = col // vec
    vecId = vecCoordM * numVecK + vecCoordK
    rawPhase = vecId // numVecSwizzleK

    if LDSK < K:
        vecLDSM = vecCoordM * K // LDSK + (vecCoordK * vec) // LDSK
        vecLDSK = vecCoordK % (LDSK // vec)
    else:
        vecLDSM = vecCoordM // perPhase
        vecLDSK = vecCoordK + (vecCoordM % perPhase) * numVecK

    phase = int(rawPhase % maxPhase)
    vecLDSKSwizzled = vecLDSK ^ phase

    col_in_lds = vecLDSKSwizzled * vec + (col % vec)
    byte_addr = vecLDSM * LDSK * elem_bytes + col_in_lds * elem_bytes
    return (byte_addr // 4) % num_banks


# ---------------------------------------------------------------------------
# Our LDSConfig-based bank computation
# ---------------------------------------------------------------------------

def local_bank(row: int, col: int, K: int, vec: int,
               lds_layout: str) -> int:
    """Bank index using LDSConfig.logical_to_byte_addr()."""
    num_banks = 32
    LDSK = 64
    elem_bytes = 2

    if lds_layout == "none":
        config = LDSConfig(
            num_banks=num_banks, row_width_elements=K,
            element_bytes=elem_bytes, mode=SwizzleMode.NONE,
        )
    elif lds_layout == "swizzle":
        perPhase = math.ceil(LDSK / K)
        maxPhase = int(min(16 / perPhase, 64 / vec))
        config = LDSConfig.with_swizzle(
            vec=vec, per_phase=perPhase, max_phase=maxPhase,
            row_width=K, element_bytes=elem_bytes, num_banks=num_banks,
        )
    else:
        raise ValueError(f"unsupported layout: {lds_layout}")

    byte_addr = config.logical_to_byte_addr(row, col)
    return (byte_addr // 4) % num_banks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_upstream_bank_match():
    """Element-by-element bank comparison for all (K, vec, layout) combos."""
    M = 16
    K_values = [16, 32, 64, 128, 256]
    vec_values = [4, 8]
    layouts = ["none", "swizzle"]

    total = 0
    passed = 0
    failed_cases = []

    for K in K_values:
        for vec in vec_values:
            if vec > K:
                continue
            for layout in layouts:
                mismatches = []
                for row in range(M):
                    for col in range(K):
                        ref = upstream_swizzle_bank(row, col, K, vec, layout)
                        ours = local_bank(row, col, K, vec, layout)
                        if ref != ours:
                            mismatches.append((row, col, ref, ours))
                total += 1
                if not mismatches:
                    passed += 1
                    print(f"  PASS  K={K:3d} vec={vec} layout={layout:8s}")
                else:
                    failed_cases.append((K, vec, layout, mismatches))
                    print(f"  FAIL  K={K:3d} vec={vec} layout={layout:8s}  "
                          f"({len(mismatches)} mismatches)")
                    for r, c, ref_b, our_b in mismatches[:5]:
                        print(f"        (row={r}, col={c}): "
                              f"upstream bank={ref_b}, ours={our_b}")
                    if len(mismatches) > 5:
                        print(f"        ... and {len(mismatches) - 5} more")

    print()
    print(f"Results: {passed}/{total} passed")
    return len(failed_cases) == 0


def test_triton_td_examples():
    """Verify against canonical examples from TritonGPUAttrDefs.td."""
    cases = [
        ("vec=1, perPhase=1, maxPhase=4",
         dict(vec=1, per_phase=1, max_phase=4, row_width=4, element_bytes=1),
         4, [[0,1,2,3], [5,4,7,6], [10,11,8,9], [15,14,13,12]]),
        ("vec=1, perPhase=2, maxPhase=4",
         dict(vec=1, per_phase=2, max_phase=4, row_width=4, element_bytes=1),
         4, [[0,1,2,3], [4,5,6,7], [9,8,11,10], [13,12,15,14]]),
        ("vec=1, perPhase=1, maxPhase=2",
         dict(vec=1, per_phase=1, max_phase=2, row_width=4, element_bytes=1),
         4, [[0,1,2,3], [5,4,7,6], [8,9,10,11], [13,12,15,14]]),
        ("vec=1, perPhase=2, maxPhase=2",
         dict(vec=1, per_phase=2, max_phase=2, row_width=4, element_bytes=1),
         4, [[0,1,2,3], [4,5,6,7], [9,8,11,10], [13,12,15,14]]),
        ("vec=2, perPhase=1, maxPhase=4",
         dict(vec=2, per_phase=1, max_phase=4, row_width=8, element_bytes=1),
         8, [[0,1,2,3,4,5,6,7], [10,11,8,9,14,15,12,13],
             [20,21,22,23,16,17,18,19], [30,31,28,29,26,27,24,25]]),
    ]

    all_ok = True
    for label, kwargs, ncols, expected_rows in cases:
        cfg = LDSConfig.with_swizzle(num_banks=32, **kwargs)
        ok = True
        for row_idx, expected in enumerate(expected_rows):
            actual = [cfg.logical_to_byte_addr(row_idx, c) for c in range(ncols)]
            if actual != expected:
                print(f"  FAIL  {label}, row {row_idx}: "
                      f"expected {expected}, got {actual}")
                ok = False
        if ok:
            print(f"  PASS  {label}")
        else:
            all_ok = False
    return all_ok


def test_swizzle_params():
    """Print upstream-derived swizzle parameters for reference."""
    print(f"  {'K':>4s}  {'vec':>3s}  {'perPhase':>8s}  {'maxPhase':>8s}")
    print(f"  {'----':>4s}  {'---':>3s}  {'--------':>8s}  {'--------':>8s}")
    LDSK = 64
    for K in [16, 32, 64, 128, 256]:
        for vec in [4, 8]:
            if vec > K:
                continue
            perPhase = math.ceil(LDSK / K)
            maxPhase = int(min(16 / perPhase, 64 / vec))
            print(f"  {K:4d}  {vec:3d}  {perPhase:8d}  {maxPhase:8d}")


# ---------------------------------------------------------------------------
# Tests: raw bank mapping (no conflict avoidance)
# ---------------------------------------------------------------------------

def test_raw_bank_mapping():
    """Verify bank = (byte_addr / 4) % num_banks for the NONE layout.

    With no swizzle and no padding, every row maps to the same bank pattern.
    This is the baseline that causes worst-case conflicts.
    """
    all_ok = True

    for elem_bytes, label in [(2, "fp16"), (4, "f32"), (1, "fp8")]:
        for num_banks in [32, 64]:
            cfg = LDSConfig(num_banks=num_banks, row_width_elements=64,
                            element_bytes=elem_bytes, mode=SwizzleMode.NONE)
            ok = True
            for row in range(4):
                for col in range(64):
                    addr = cfg.logical_to_byte_addr(row, col)
                    expected = row * 64 * elem_bytes + col * elem_bytes
                    if addr != expected:
                        print(f"  FAIL  raw {label} {num_banks}banks "
                              f"(row={row}, col={col}): "
                              f"expected addr={expected}, got {addr}")
                        ok = False
                        break
                if not ok:
                    break

            # Verify all rows have the same bank pattern (the defining
            # property of "no conflict avoidance")
            if ok:
                row0_banks = [(cfg.logical_to_byte_addr(0, c) // 4) % num_banks
                              for c in range(64)]
                for row in range(1, 4):
                    row_banks = [(cfg.logical_to_byte_addr(row, c) // 4) % num_banks
                                 for c in range(64)]
                    if row_banks != row0_banks:
                        # Rows only differ if stride != multiple of bank row
                        stride = cfg.row_stride_bytes
                        bank_row_bytes = num_banks * 4
                        if stride % bank_row_bytes == 0 and row_banks != row0_banks:
                            print(f"  FAIL  raw {label} {num_banks}banks: "
                                  f"row {row} banks differ despite aligned stride")
                            ok = False

            if ok:
                print(f"  PASS  raw bank mapping: {label}, {num_banks} banks")
            all_ok = all_ok and ok

    return all_ok


# ---------------------------------------------------------------------------
# Tests: padding layout
# ---------------------------------------------------------------------------

def test_padding_layout():
    """Verify padding widens the row stride and shifts bank alignment.

    Key properties:
    1. byte_addr = row * (row_width + pad) * elem_bytes + col * elem_bytes
    2. Padding elements are NOT part of the data but change the stride
    3. Different padding values produce different bank patterns per row
    """
    all_ok = True

    test_cases = [
        # (row_width, padding, elem_bytes, num_banks, label)
        (128, 8, 2, 64, "fp16 K=128 pad=8 64banks"),
        (128, 16, 2, 64, "fp16 K=128 pad=16 64banks"),
        (64, 4, 2, 32, "fp16 K=64 pad=4 32banks"),
        (128, 8, 1, 64, "fp8 K=128 pad=8 64banks"),
        (16, 8, 4, 32, "f32 K=16 pad=8 32banks"),
    ]

    for row_width, padding, elem_bytes, num_banks, label in test_cases:
        cfg = LDSConfig.with_padding(padding, row_width, elem_bytes, num_banks)
        ok = True

        # Verify stride
        expected_stride = (row_width + padding) * elem_bytes
        if cfg.row_stride_bytes != expected_stride:
            print(f"  FAIL  {label}: stride expected {expected_stride}, "
                  f"got {cfg.row_stride_bytes}")
            ok = False

        # Verify addresses match the linear formula
        for row in range(4):
            for col in range(row_width):
                addr = cfg.logical_to_byte_addr(row, col)
                expected = row * expected_stride + col * elem_bytes
                if addr != expected:
                    print(f"  FAIL  {label} (row={row}, col={col}): "
                          f"expected {expected}, got {addr}")
                    ok = False
                    break
            if not ok:
                break

        # Verify padding actually changes bank pattern vs no-padding
        if ok and padding > 0:
            cfg_nopad = LDSConfig(num_banks=num_banks,
                                  row_width_elements=row_width,
                                  element_bytes=elem_bytes,
                                  mode=SwizzleMode.NONE)
            # Row 1 should differ (unless padding happens to be a multiple of
            # the full bank row, which would be a degenerate case)
            pad_banks = [(cfg.logical_to_byte_addr(1, c) // 4) % num_banks
                         for c in range(min(row_width, 16))]
            nopad_banks = [(cfg_nopad.logical_to_byte_addr(1, c) // 4) % num_banks
                           for c in range(min(row_width, 16))]
            if pad_banks == nopad_banks:
                stride_bytes = expected_stride
                bank_row_bytes = num_banks * 4
                if stride_bytes % bank_row_bytes != 0:
                    print(f"  FAIL  {label}: padding didn't change bank pattern "
                          f"(stride={stride_bytes}, bank_row={bank_row_bytes})")
                    ok = False

        if ok:
            print(f"  PASS  {label}")
        all_ok = all_ok and ok

    return all_ok


# ---------------------------------------------------------------------------
# Tests: end-to-end conflict analysis
# ---------------------------------------------------------------------------

def test_conflict_analysis_e2e():
    """End-to-end test: verify known conflict counts for specific configs.

    These are regression values from manual analysis.
    """
    all_ok = True

    cases = [
        # (config, pattern_name, kwidth, expected_max_conflict)
        ("K=128 fp16 no-pad 64banks → 16-way",
         LDSConfig(num_banks=64, row_width_elements=128, element_bytes=2,
                   mode=SwizzleMode.NONE),
         "wmma_kcontig", 8, 16),

        ("K=128 fp16 pad=8 64banks → 2-way",
         LDSConfig.with_padding(8, 128, 2, 64),
         "wmma_kcontig", 8, 2),

        ("K=128 fp16 swizzle(8,1,16) 64banks → 2-way",
         LDSConfig.with_swizzle(8, 1, 16, 128, 2, 64),
         "wmma_kcontig", 8, 2),

        ("K=128 fp16 no-pad 64banks ds_load_tr → 16-way",
         LDSConfig(num_banks=64, row_width_elements=128, element_bytes=2,
                   mode=SwizzleMode.NONE),
         "ds_load_tr16_b128", 8, 16),

        ("K=128 fp16 pad=8 64banks ds_load_tr → 2-way",
         LDSConfig.with_padding(8, 128, 2, 64),
         "ds_load_tr16_b128", 8, 2),
    ]

    from lds_bank_conflict_analyzer import get_pattern

    for label, config, pattern_name, kwidth, expected in cases:
        pattern = get_pattern(pattern_name, kwidth)
        accesses = compute_lane_accesses(config, pattern)
        _, max_conflict = find_bank_conflicts(
            accesses, config.bytes_per_bank, config.num_banks)
        if max_conflict == expected:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}: expected {expected}-way, got {max_conflict}-way")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Cross-Validation: LDS Bank Conflict Analyzer")
    print("=" * 70)

    results = []

    print("\n--- TritonGPUAttrDefs.td Examples ---")
    results.append(test_triton_td_examples())

    print("\n--- Swizzle Parameter Derivation ---")
    test_swizzle_params()

    print("\n--- Element-by-element Bank Comparison (vs upstream tikzplot.tex) ---")
    results.append(test_upstream_bank_match())

    print("\n--- Raw Bank Mapping (NONE layout) ---")
    results.append(test_raw_bank_mapping())

    print("\n--- Padding Layout ---")
    results.append(test_padding_layout())

    print("\n--- End-to-end Conflict Analysis ---")
    results.append(test_conflict_analysis_e2e())

    if all(results):
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
