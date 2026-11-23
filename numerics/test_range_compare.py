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


# Fixture and outputs for --metric: PINNED_* were produced by range_compare.py at toolkit commit
# 381de41a29794b22e1fa3e5c22f77a4084eaf7a7 (compare took one --metric then) on these two tables.
METRIC_A = "B,v_median,v_min,v_max,w_median,w_min,w_max\n1,10,9,11,3,2,4\n2,20,19,21,6,5,7\n"
METRIC_B = "B,v_median,v_min,v_max,w\n1,5,4,6,3\n2,30,25,35,4\n"
PINNED_SINGLE_MD = '| B | A med [min,max] | B med [min,max] | A/B |\n|---|---|---|---|\n| 1 | 10.0 [9.0,11.0] | 5.0 [4.0,6.0] | 2.00 [1.50,2.75] higher |\n| 2 | 20.0 [19.0,21.0] | 30.0 [25.0,35.0] | 0.67 [0.54,0.84] lower |\n'
PINNED_SINGLE_JSON = '[\n {\n  "key": [\n   1\n  ],\n  "values": {\n   "A": [\n    10.0,\n    9.0,\n    11.0\n   ],\n   "B": [\n    5.0,\n    4.0,\n    6.0\n   ]\n  },\n  "ratios": {\n   "A/B": {\n    "ratio": 2.0,\n    "lo": 1.5,\n    "hi": 2.75,\n    "verdict": "higher"\n   }\n  }\n },\n {\n  "key": [\n   2\n  ],\n  "values": {\n   "A": [\n    20.0,\n    19.0,\n    21.0\n   ],\n   "B": [\n    30.0,\n    25.0,\n    35.0\n   ]\n  },\n  "ratios": {\n   "A/B": {\n    "ratio": 0.6666666666666666,\n    "lo": 0.5428571428571428,\n    "hi": 0.84,\n    "verdict": "lower"\n   }\n  }\n }\n]\n'


def run_compare(*metrics: str, fmt: str = "md", bounds: bool = True) -> str:
    a, b = write("a.csv", METRIC_A), write("b.csv", METRIC_B)
    argv = ["compare", "--key", "B", "--table", f"A={a}", "--table", f"B={b}", "--ratio", "A/B", "--format", fmt]
    for m in metrics:
        argv += ["--metric", m]
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc.main(argv + (["--bounds"] if bounds else []))
    return buf.getvalue()


def test_single_metric_is_byte_identical_to_381de41() -> None:
    assert run_compare("v") == PINNED_SINGLE_MD
    assert run_compare("v", fmt="json", bounds=False) == PINNED_SINGLE_JSON


def test_repeated_metric_prints_every_table_in_order() -> None:
    out = run_compare("w", "v")
    w_table = ("| B | A med [min,max] | B med [min,max] | A/B |\n|---|---|---|---|\n"
               "| 1 | 3.0 [2.0,4.0] | 3.0 | 1.00 [0.67,1.33] overlap |\n"
               "| 2 | 6.0 [5.0,7.0] | 4.0 | 1.50 [1.25,1.75] higher |\n")
    assert out == "### w\n\n" + w_table + "\n### v\n\n" + PINNED_SINGLE_MD
    data = json.loads(run_compare("w", "v", fmt="json", bounds=False))
    assert list(data) == ["w", "v"]
    assert data["v"] == json.loads(PINNED_SINGLE_JSON)
    assert [r["ratios"]["A/B"]["verdict"] for r in data["w"]] == ["overlap", "higher"]


def test_repeated_metric_is_refused_twice() -> None:
    try:
        run_compare("v", "v")
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("a metric given twice must be refused")



def test_consensus_agreement_disagreement_and_one_overlap() -> None:
    hi, lo, ov = ({"verdict": v} for v in ("higher", "lower", "overlap"))
    assert rc.consensus([hi, hi, hi]) == {"verdict": "higher", "n": 3}
    assert rc.consensus([lo, None, lo]) == {"verdict": "lower", "n": 2}
    assert rc.consensus([hi, lo]) == {"verdict": "no verdict", "n": 2}
    assert rc.consensus([hi, hi, ov]) == {"verdict": "no verdict", "n": 3}
    assert rc.consensus([ov, ov]) == {"verdict": "no verdict", "n": 2}
    assert rc.consensus([None, None]) == {"verdict": None, "n": 0}


# Three tables over B=1..4: C agrees with A at B=1 (both above B), overlaps B at B=2, sits on the
# other side of B at B=3 (disagreement) and lacks B=4 (left out of the consensus).
CONS_A = "B,v_median,v_min,v_max\n1,10,9,11\n2,10,9,11\n3,10,9,11\n4,10,9,11\n"
CONS_B = "B,v_median,v_min,v_max\n1,5,4,6\n2,5,4,6\n3,5,4,6\n4,5,4,6\n"
CONS_C = "B,v_median,v_min,v_max,w\n1,8,7,8.5,1\n2,5.5,5,6,1\n3,3,2,3.5,1\n"


def run_consensus(*extra: str, fmt: str = "md") -> str:
    a, b, c = write("a.csv", CONS_A), write("b.csv", CONS_B), write("c.csv", CONS_C)
    argv = ["compare", "--key", "B", "--metric", "v", "--table", f"A={a}", "--table", f"B={b}",
            "--table", f"C={c}", "--ratio", "A/B", "--ratio", "C/B", "--format", fmt, *extra]
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc.main(argv)
    return buf.getvalue()


def test_cli_consensus_column_markdown_and_json() -> None:
    lines = run_consensus("--consensus", "all=A/B,C/B", "--consensus", "AB=A/B").splitlines()
    assert lines[0] == "| B | A med [min,max] | B med [min,max] | C med [min,max] | A/B | C/B | all | AB |"
    assert [line.rsplit("|", 3)[1:3] for line in lines[2:]] == [
        [" higher ", " higher "], [" no verdict ", " higher "], [" no verdict ", " higher "], [" higher ", " higher "]]
    data = json.loads(run_consensus("--consensus", "all=A/B,C/B", fmt="json"))
    assert [r["consensus"]["all"] for r in data] == [
        {"verdict": "higher", "n": 2}, {"verdict": "no verdict", "n": 2}, {"verdict": "no verdict", "n": 2},
        {"verdict": "higher", "n": 1}]
    only_c = json.loads(run_consensus("--consensus", "c=C/B", fmt="json"))
    assert only_c[3]["consensus"]["c"] == {"verdict": None, "n": 0}
    assert run_consensus("--consensus", "c=C/B").splitlines()[5].endswith("| - | - |")


def test_consensus_with_repeated_metric() -> None:
    a, b = write("a.csv", METRIC_A), write("b.csv", METRIC_B)
    argv = ["compare", "--key", "B", "--table", f"A={a}", "--table", f"B={b}", "--ratio", "A/B",
            "--ratio", "B/A", "--metric", "w", "--metric", "v", "--consensus", "both=A/B,B/A"]
    for fmt in ("md", "json"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc.main(argv + ["--format", fmt])
        out = buf.getvalue()
        if fmt == "md":
            assert out.count("| both |") == 2 and out.startswith("### w\n")
            assert out.rstrip().endswith("0.67 lower | 1.50 higher | no verdict |")
        else:
            data = json.loads(out)
            assert [r["consensus"]["both"]["verdict"] for r in data["w"]] == ["no verdict", "no verdict"]


def test_consensus_option_errors() -> None:
    for spec in ("bad", "x=", "x=A/C", "A/B=A/B", "=A/B"):
        try:
            run_consensus("--consensus", spec)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(spec)
    try:
        run_consensus("--consensus", "x=A/B", "--consensus", "x=C/B")
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("a consensus name given twice must be refused")


def test_without_consensus_output_is_byte_identical_to_40f9d01() -> None:
    # Pinned at 40f9d01958ce65942a78ef66a5a9e3075068873c, which had no --consensus.
    assert run_compare("v") == PINNED_SINGLE_MD
    assert run_compare("v", fmt="json", bounds=False) == PINNED_SINGLE_JSON
    assert "consensus" not in run_consensus(fmt="json") and run_consensus().count("|") == 7 * 6


# ---------------------------------------------------------------- intervals of a ratio of medians

WON = [110.0, 111.0, 112.0, 113.0, 114.0]
REF = [100.0, 101.0, 102.0, 103.0, 104.0]


def test_percentile_and_bootstrap_are_fixed_by_seed() -> None:
    assert rc.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5 and rc.percentile([5.0], 0.975) == 5.0
    assert rc.bootstrap_ratio(WON, REF) == rc.bootstrap_ratio(WON, REF)
    assert rc.bootstrap_ratio_paired({1: 1.0}, {2: 1.0}) is None


def test_bootstrap_bounds_at_small_n_and_the_minimum() -> None:
    # n = 3: a resampled median is the side's smallest value with probability 7/27 (two or three of three
    # draws), so min(X)/max(REF) holds 49/729 = 6.7 % of the ratios and is the 2.5th percentile exactly;
    # symmetrically max(X)/min(REF) is the 97.5th
    x, ref = [110.0, 111.0, 112.0], [100.0, 101.0, 102.0]
    assert rc.bootstrap_ratio(x, ref, min_n=3) == (110 / 102, 112 / 100)
    # n = 1: the interval is the point ratio, which is why the default minimum refuses it
    assert rc.bootstrap_ratio([101.0], [100.0], min_n=1) == (1.01, 1.01)
    for a, b in (([101.0], [100.0]), (x, ref), (WON[:4], REF)):
        try:
            rc.bootstrap_ratio(a, b)
        except ValueError as exc:
            assert "below the minimum 5" in str(exc)
        else:
            raise AssertionError(f"n = {len(a)}/{len(b)} must be refused")
    try:
        rc.bootstrap_ratio(WON, [0.0] + REF[1:])
    except ValueError as exc:
        assert "0 value" in str(exc)
    else:
        raise AssertionError("a 0 reference value must be refused")


def test_interval_verdict_flags_and_rerun() -> None:
    assert [rc.interval_verdict(*ci) for ci in ((1.01, 1.2), (0.8, 0.99), (0.9, 1.1), (1.0, 1.1))] == [
        "higher", "lower", "no verdict", "no verdict"]
    assert rc.combine_flags("higher", [False, False]) == "higher"
    assert rc.combine_flags("higher", [False, None]) == "incomplete"
    assert rc.combine_flags("higher", [None, True]) == "flagged"
    assert rc.final_word("won", "lost") == "won" and rc.final_word("flagged", None) == "flagged"
    assert rc.final_word("flagged", "lost") == "lost" and rc.final_word("flagged", "flagged") == "no verdict"


def test_load_reps_keys_repetitions_and_refuses_a_repeated_index() -> None:
    j = write("b.json", json.dumps({"results": [
        {"phase": "warmup-1", "c": 1, "s": {"v": 9.0}}, {"phase": "measured-1", "c": 1, "s": {"v": 1.0}},
        {"phase": "measured-2", "c": 1, "s": {"v": 3.0}}, {"phase": "measured-1", "c": 2, "s": {"v": 5.0}}]}))
    label, reps = rc.load_reps(f"A={j}", ["c"], "results", "s.v", "phase", rc.parse_where(["phase~^measured-"]))
    assert label == "A" and reps == {(1,): {"measured-1": 1.0, "measured-2": 3.0}, (2,): {"measured-1": 5.0}}
    try:  # without the key, measured-1 of c=1 and of c=2 are one point
        rc.load_reps(f"A={j}", [], "results", "s.v", "phase", rc.parse_where(["phase~^measured-1"]))
    except ValueError as exc:
        assert "twice" in str(exc)
    else:
        raise AssertionError("a repeated repetition index must be refused")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
