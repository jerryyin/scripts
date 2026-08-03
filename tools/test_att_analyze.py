#!/usr/bin/env python3
"""Non-GPU unit tests for the ATT CSV reducer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import att_analyze


HEADER = "Codeobj,Vaddr,Instruction,Hitcount,Latency,Stall,Idle,Source\n"


def write_csv(body: str, header: str = HEADER) -> Path:
    path = Path(tempfile.mkdtemp()) / "stats.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_parse_and_explicit_loop_selection() -> None:
    path = write_csv(
        "7,4096,v_wmma_scale_f32,20,200,40,10,kernel.py:10\n"
        "7,4100,s_waitcnt,20,100,80,5,kernel.py:11\n"
        "7,4104,s_endpgm,1,10,0,0,kernel.py:12\n"
    )
    records = att_analyze.parse_att_csv(str(path))
    categories, hitcount, coverage = att_analyze.analyze(records, 20)
    assert hitcount == 20
    assert categories["mfma"]["latency"] == 200
    assert categories["s_waitcnt"]["stall"] == 80
    assert 96.0 < coverage < 97.0


def test_malformed_row_is_not_silently_dropped() -> None:
    path = write_csv("7,not-an-address,v_add_f32,20,200,0,0,kernel.py:10\n")
    try:
        att_analyze.parse_att_csv(str(path))
    except ValueError as error:
        assert "malformed" in str(error)
        return
    raise AssertionError("malformed ATT row was silently accepted")


def test_missing_required_column_is_rejected() -> None:
    path = write_csv(
        "7,4096,v_add_f32,20,200,0,kernel.py:10\n",
        header="Codeobj,Vaddr,Instruction,Hitcount,Latency,Stall,Source\n",
    )
    try:
        att_analyze.parse_att_csv(str(path))
    except ValueError as error:
        assert "idle" in str(error)
        return
    raise AssertionError("ATT CSV with a missing column was accepted")


def test_code_object_selection_is_explicit_and_unambiguous() -> None:
    path = write_csv(
        "7,4096,v_add_f32,20,200,0,0,kernel.py:10\n"
        "8,4100,v_add_f32,20,100,0,0,kernel.py:11\n"
    )
    records = att_analyze.parse_att_csv(str(path))
    try:
        att_analyze.select_code_object(records)
    except ValueError as error:
        assert "multiple Codeobj" in str(error)
    else:
        raise AssertionError("multiple code-object load IDs were silently combined")
    selected, code_object_id = att_analyze.select_code_object(records, 8)
    assert code_object_id == 8
    assert len(selected) == 1


def test_zero_stall_remains_zero_in_json_summary() -> None:
    categories = {"alu": {"count": 1, "latency": 10, "stall": 0, "idle": 0}}
    summary = att_analyze.json_summary(
        "stats.csv", "sample", categories, 5, 100.0, 7
    )
    assert summary["totals"]["stall"] == 0
    assert summary["totals"]["stall_rate_pct"] == 0.0
    assert summary["totals"]["latency_per_hitcount"] == 2.0


def test_gfx12_clause_and_wait_xcnt_are_not_other() -> None:
    path = write_csv(
        "7,4096,s_clause 0x3,20,200,40,10,kernel.py:10\n"
        "7,4100,s_wait_xcnt null,20,100,80,5,kernel.py:11\n"
    )
    records = att_analyze.parse_att_csv(str(path))
    categories, hitcount, _coverage = att_analyze.analyze(records, 20)
    assert hitcount == 20
    assert categories["s_clause"]["latency"] == 200
    assert categories["s_waitcnt"]["stall"] == 80
    assert "other" not in categories


def test_gfx12_suffixed_alu_families_are_not_other() -> None:
    path = write_csv(
        "7,4096,v_mad_nc_i64_i32,20,200,40,10,kernel.py:10\n"
        "7,4100,v_dual_lshlrev_b32,20,100,20,5,kernel.py:11\n"
        "7,4104,v_pk_fma_f32,20,80,10,2,kernel.py:12\n"
        "7,4108,v_cvt_scale_pk8_f32_fp8,20,60,0,0,kernel.py:13\n"
        "7,4112,s_cselect_b32,20,40,0,0,kernel.py:14\n"
    )
    records = att_analyze.parse_att_csv(str(path))
    categories, hitcount, _coverage = att_analyze.analyze(records, 20)
    assert hitcount == 20
    assert categories["alu"]["latency"] == 480
    assert "other" not in categories


def test_memory_and_lane_families_are_explicit() -> None:
    path = write_csv(
        "7,4096,s_load_dword,20,100,90,0,kernel.py:10\n"
        "7,4100,flat_load_dword,20,80,70,0,kernel.py:11\n"
        "7,4104,v_readlane_b32,20,40,0,0,kernel.py:12\n"
    )
    records = att_analyze.parse_att_csv(str(path))
    categories, hitcount, _coverage = att_analyze.analyze(records, 20)
    assert hitcount == 20
    assert categories["scalar_load"]["latency"] == 100
    assert categories["flat_load"]["latency"] == 80
    assert categories["v_perm"]["latency"] == 40
    assert "other" not in categories


def main() -> None:
    tests = sorted(
        (
            value
            for name, value in globals().items()
            if name.startswith("test_") and callable(value)
        ),
        key=lambda test: test.__name__,
    )
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
