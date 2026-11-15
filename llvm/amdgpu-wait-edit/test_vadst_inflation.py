#!/usr/bin/env python3
"""CPU tests for vadst_inflation.py. Run: python3 -m pytest -q test_vadst_inflation.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vadst_inflation as vi  # noqa: E402


def test_known_answers():
    vi.selftest()  # a planted inflation of 2 and a negative control of 0; raises on a miss


def test_counts_only_valu_instructions_that_write_a_vgpr(tmp_path):
    asm = tmp_path / "k.s"
    asm.write_text("\tv_add_u32 v1, v2, v3\n\tv_cmp_eq_u32_e32 vcc_lo, v1, v2\n"
                   "\ts_wait_alu depctr_va_vdst(2)\n\tglobal_load_b32 v4, v[0:1], off\n")
    [site] = vi.analyze(str(asm))
    assert (site.n, site.counted, site.vgpr_writers, site.inflation) == (2, 2, 1, 1)
