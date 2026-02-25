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

from lds_bank_conflict_analyzer import LDSConfig, SwizzleMode


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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Cross-Validation: LDS Bank Conflict Analyzer vs Upstream tikzplot.tex")
    print("=" * 70)

    print("\n--- TritonGPUAttrDefs.td Examples ---")
    td_ok = test_triton_td_examples()

    print("\n--- Swizzle Parameter Derivation ---")
    test_swizzle_params()

    print("\n--- Element-by-element Bank Comparison ---")
    bank_ok = test_upstream_bank_match()

    if td_ok and bank_ok:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
