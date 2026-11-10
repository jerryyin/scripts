#!/usr/bin/env python3
"""CPU tests for step_breakdown.py on a hand-made analyzer output."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import step_breakdown as sb


def fixture(root: Path) -> tuple[Path, Path]:
    analysis, dump = root / "analysis", root / "dump"
    analysis.mkdir(); dump.mkdir()
    rows = ["rank,role,category,median_ms"]
    for rank, role, attn in ((0, "busy", 3.0), (1, "idle", 2.0)):
        rows += [f"{rank},{role},attn.sparse_mla_core,{attn}", f"{rank},{role},petit.stage1(dispatch_wait+gate_up),1.0",
                 f"{rank},{role},mystery,0.5", f"{rank},{role},aux.shared_expert,2.0"]
    (analysis / "per_category.csv").write_text("\n".join(rows) + "\n")
    (analysis / "summary.csv").write_text("rank,graph_ms,period_ms,host_gap_ms,harvest_ms,n\n"
                                          "0,4.5,4.6,0.1,1.0,100\n1,3.5,4.6,1.1,1.0,100\n")
    seq = {"type": "seq", "stride": 1, "names": ["0:attn.core", "0:attn.core", "0:petit.s1", "0:shared.b"],
           "streams": ["main", "main", "main", "aux"]}
    (dump / "step-timing-r0-p1.jsonl").write_text(json.dumps(seq) + "\n")
    return analysis, dump


def test_table_correction_and_diff() -> None:
    with tempfile.TemporaryDirectory() as d:
        analysis, dump = fixture(Path(d))
        t = sb.table(str(analysis), str(dump), "x", clean_ms=4.2)
        assert t["main_markers_per_step"] == 3 and abs(t["delta_us_per_marker"] - 1000 * (4.0 - 4.2) / 3) < 1e-9
        core = t["comp"]["attn: sparse-MLA core"]
        assert (core["busy"], core["idle"], core["markers"]) == (3.0, 2.0, 2)
        assert abs(core["corr"] - (3.0 + 2 * 0.2 / 3)) < 1e-9
        assert "other: mystery" in t["comp"] and t["aux"]["aux.shared_expert"]["0"] == 2.0
        md = sb.table_markdown(t)
        assert "| **sum of components** | **4.500** | **3.500** |" in md
        last = sb.diff(t, t, "A", "B").splitlines()[-2]
        assert last.startswith("| **sum** | **4.500** | **4.500** | **+0.000** |")


def test_without_clean_step_there_is_no_correction() -> None:
    with tempfile.TemporaryDirectory() as d:
        analysis, dump = fixture(Path(d))
        t = sb.table(str(analysis), str(dump), "x")
        assert t["delta_us_per_marker"] is None and t["comp"]["attn: sparse-MLA core"]["corr"] is None
        json.dumps(t, allow_nan=False)
        assert "no clean step given" in sb.table_markdown(t)


def test_overhead_row() -> None:
    with tempfile.TemporaryDirectory() as d:
        analysis, _ = fixture(Path(d))
        metrics = Path(d) / "metrics.json"
        metrics.write_text(json.dumps({"tpot_ms_mean": 5.0}))
        row = sb.overhead([f"p B=1={metrics},{analysis},4.0"]).splitlines()[2]
        assert row == "| p B=1 | 5.00 | 4.00 | +25.0 | 4.000 | 4.600 | 0.600 | 100 |"


if __name__ == "__main__":
    test_table_correction_and_diff()
    test_without_clean_step_there_is_no_correction()
    test_overhead_row()
    print("ok")
