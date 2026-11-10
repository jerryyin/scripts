#!/usr/bin/env python3
"""Unit tests for range_compare.py (standard library only)."""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import range_compare as rc


def write(name: str, text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(text)
    return path


def test_compare_verdicts() -> None:
    assert rc.compare((10, 9, 11), (5, 4, 6))["verdict"] == "higher"
    assert rc.compare((5, 4, 6), (10, 9, 11))["verdict"] == "lower"
    c = rc.compare((10, 8, 12), (11, 9, 13))
    assert c["verdict"] == "overlap"
    assert abs(c["ratio"] - 10 / 11) < 1e-12
    assert abs(c["lo"] - 8 / 13) < 1e-12 and abs(c["hi"] - 12 / 9) < 1e-12


def test_summarize_groups_and_nested_fields() -> None:
    rows = [{"B": 1, "s": {"x": 1.0}}, {"B": 1, "s": {"x": 3.0}}, {"B": 1, "s": {"x": 2.0}},
            {"B": 2, "s": {"x": 5.0}}]
    out = rc.summarize(rows, ["B"], ["s.x"])
    assert out[0] == {"B": 1, "s.x_n": 3, "s.x_median": 2.0, "s.x_min": 1.0, "s.x_max": 3.0}
    assert out[1]["s.x_median"] == 5.0 and out[1]["s.x_n"] == 1


def test_compare_csv_tables_with_scale_and_single_values() -> None:
    a = write("a.csv", "L,B,tok_median,tok_min,tok_max\n8192,1,100,99,101\n8192,4,400,390,410\n")
    b = write("b.json", json.dumps({"summary": [{"L": 8192, "B": 1, "tok": 50}]}))
    _, ta = rc.load_table(f"A={a}", ["L", "B"], None)
    _, tb = rc.load_table(f"B={b}", ["L", "B"], "summary")
    rows = rc.compare_tables({"A": ta, "B": tb}, "tok", [("A", "B")], {"B": 2.0})
    assert rows[0]["key"] == (8192, 1)
    assert rows[0]["values"]["B"] == (100.0, 100.0, 100.0)
    assert rows[0]["ratios"]["A/B"]["verdict"] == "overlap"
    assert rows[1]["ratios"]["A/B"] is None


def test_pool_alias_where() -> None:
    a = write("a.csv", "batch_size,v_median,v_min,v_max\n4,10,9,11\n")
    b = write("b.csv", "batch_size,v_median,v_min,v_max\n4,14,13,20\n")
    n = write("n.csv", "L,B,v_median,v_min,v_max\n8192,4,1,1,1\n50000,4,5,5,5\n")
    _, pooled = rc.load_table(f"P={a}+{b}", ["B"], None, "v", {"batch_size": "B"})
    assert rc.stat(pooled[(4,)], "v") == (12.0, 9.0, 20.0)
    _, nat = rc.load_table(f"N={n}", ["B"], None, "v", {}, rc.parse_where(["L=50000"]))
    assert rc.stat(nat[(4,)], "v") == (5.0, 5.0, 5.0)


def test_summarize_cli_where_regex_and_one_group() -> None:
    j = write("b.json", json.dumps({"results": [
        {"phase": "warmup-1", "summary": {"rps": 9.0}}, {"phase": "measured-1", "summary": {"rps": 1.0}},
        {"phase": "measured-2", "summary": {"rps": 3.0}}, {"phase": "measured-3", "summary": {"rps": 2.0}}]}))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc.main(["summarize", str(j), "--rows-field", "results", "--key", "", "--metric", "summary.rps",
                 "--where", "phase~^measured", "--format", "json"])
    assert json.loads(buf.getvalue()) == [{"summary.rps_n": 3, "summary.rps_median": 2.0,
                                           "summary.rps_min": 1.0, "summary.rps_max": 3.0}]


def test_key_divide_aligns_per_rank_batch() -> None:
    a = write("a.csv", "B,v_median,v_min,v_max\n8,80,80,80\n")
    b = write("b.csv", "B,v_median,v_min,v_max\n16,120,120,120\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc.main(["compare", "--key", "B", "--metric", "v", "--table", f"E4={a}", "--table", f"E8={b}",
                 "--key-divide", "E4=B/4", "--key-divide", "E8=B/8", "--scale", "E4=0.25", "--scale", "E8=0.125",
                 "--ratio", "E4/E8", "--format", "json"])
    row = json.loads(buf.getvalue())[0]
    assert row["key"] == [2] and abs(row["ratios"]["E4/E8"]["ratio"] - 20 / 15) < 1e-12


def test_cli_markdown() -> None:
    a = write("a.csv", "B,v_median,v_min,v_max\n1,10,9,11\n")
    b = write("b.csv", "B,v_median,v_min,v_max\n1,5,4,6\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc.main(["compare", "--key", "B", "--metric", "v", "--table", f"A={a}", "--table", f"B={b}",
                 "--ratio", "A/B", "--bounds"])
    line = buf.getvalue().splitlines()[2]
    assert line == "| 1 | 10.0 [9.0,11.0] | 5.0 [4.0,6.0] | 2.00 [1.50,2.75] higher |"


def test_duplicate_keys_are_refused_unless_pooled() -> None:
    a = write("a.csv", "B,v\n1,10\n1,30\n")
    b = write("b.csv", "B,v\n1,20\n")
    try:
        rc.load_table(f"A={a}", ["B"], None, "v")
    except ValueError as exc:
        assert "several rows" in str(exc)
    else:
        raise AssertionError("duplicate keys must be refused")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc.main(["compare", "--key", "B", "--metric", "v", "--table", f"A={a}", "--table", f"B={b}",
                 "--ratio", "A/B", "--pool", "--bounds"])
    # pooled A: median of 10 and 30 = 20, range [10, 30]; against B = 20
    assert buf.getvalue().splitlines()[2] == "| 1 | 20.0 [10.0,30.0] | 20.0 | 1.00 [0.50,1.50] overlap |"


def test_where_is_one_filter_with_clear_errors() -> None:
    rows = [{"L": 8192, "B": 1}, {"L": 50000, "B": 1}, {"B": 2}]
    where = rc.parse_where(["L=50000"])
    assert [r for r in rows if rc.keep(r, where)] == rows[1:]
    assert [r for r in rows if rc.keep(r, rc.parse_where(["L~^8"]))] == [rows[0], rows[2]]
    for bad in (["bad"], ["v~("]):
        try:
            rc.parse_where(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    a = write("w.csv", "B,v\n1,10\n")
    for argv in (["summarize", str(a), "--key", "B", "--metric", "v", "--where", "bad"],
                 ["compare", "--key", "B", "--metric", "v", "--table", f"A={a}", "--where", "v~("]):
        try:
            rc.main(argv)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(argv)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
