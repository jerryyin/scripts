#!/usr/bin/env python3
"""CPU tests for decide.py, on small per-repetition tables and flag inputs (run with pytest or directly)."""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import decide as dc


def write(name: str, text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(text)
    return path


# Fixtures hold only the rows each case needs: five repetitions per layout at one or two points.

WON = [110.0, 111.0, 112.0, 113.0, 114.0]
REF = [100.0, 101.0, 102.0, 103.0, 104.0]
LOST = [90.0, 91.0, 92.0, 93.0, 94.0]
TIE = [96.0, 99.0, 102.0, 105.0, 108.0]


def bench(points: dict[int, list[float]], warm: float | None = 1.0) -> Path:
    """A benchmark.json-shaped table: results rows with phase, concurrency and summary.tok."""
    rows = []
    for c, vals in points.items():
        if warm is not None:
            rows.append({"phase": "warmup-1", "concurrency": c, "summary": {"tok": warm}})
        rows += [{"phase": f"measured-{i}", "concurrency": c, "summary": {"tok": v}} for i, v in enumerate(vals, 1)]
    return write("benchmark.json", json.dumps({"results": rows}))


def split(shares: dict[int, float | tuple], label: str = "x", ref: str = "ref") -> Path:
    """An --accept-split JSON: per concurrency a share v (as accept_part_ms.median = 100 v over
    latency_gap_ms.median = 100, accept_share.median = v) or (part_ms, gap_ms, per-repetition share median)."""
    def row(c, v):
        part, gap, per_rep = v if isinstance(v, tuple) else (100.0 * v, 100.0, v)
        m = lambda x: {"median": x, "min": x, "max": x, "n": 5}  # noqa: E731
        return {"label": label, "reference": ref, "concurrency": c, "repetitions": 5, "latency_gap_ms": m(gap),
                "accept_part_ms": m(part), "accept_share": m(per_rep)}
    return write("split.json", json.dumps({"reference": ref, "summary": [row(c, v) for c, v in shares.items()]}))


def accept_test(x_wilson: list[float], ref_wilson: list[float]) -> Path:
    side = lambda w: {"looping_prompts": 1, "prompts": 5, "looping_fraction": 0.2, "looping_wilson95": w}  # noqa: E731
    return write("test.jsonl", json.dumps({"pair": "ref vs x", "accept_test": {"ref": side(ref_wilson),
                                                                                 "x": side(x_wilson)}}) + "\n")


OVERLAP_W = ([0.1, 0.6], [0.2, 0.7])


def run_decide(x: dict[int, list[float]], ref: dict[int, list[float]], *extra: str, shares=None, wilson=OVERLAP_W,
               canon: tuple[dict, dict] | None = ({1: TIE[:3], 2: TIE[:3]}, {1: REF[:3], 2: REF[:3]})) -> list[dict]:
    args = ["--key", "concurrency", "--metric", "summary.tok", "--rows-field", "results",
            "--where", "phase~^measured-", "--table", f"X={bench(x)}", "--table", f"REF={bench(ref)}",
            "--ratio", "X/REF", "--label", "X=x", "--label", "REF=ref", "--format", "json",
            "--accept-split", str(split(shares or {c: 0.0 for c in x}))]
    for c in x:
        args += ["--accept-test", f"{c}={accept_test(*wilson)}"]
    if canon:
        args += ["--canonical-table", f"X={bench(canon[0])}", "--canonical-table", f"REF={bench(canon[1])}"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        dc.main(args + list(extra))
    return json.loads(buf.getvalue())


def test_decide_below_minimum_repetitions_is_incomplete() -> None:
    r = run_decide({1: [101.0]}, {1: [100.0]})[0]
    run = r["runs"][0]
    assert run["interval"] is None and run["verdict"] == "no verdict" and r["final"] == "incomplete"
    assert "below the minimum 5" in run["reason"]
    r = run_decide({1: WON[:3]}, {1: REF[:3]}, "--min-repetitions", "3")[0]
    assert r["runs"][0]["interval"] == [110 / 102, 112 / 100] and r["final"] == "won"


def test_decide_refuses_a_zero_reference_value() -> None:
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            run_decide({1: WON}, {1: [0.0] + REF[1:]})
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("a 0 value in the REF table must be refused")


def test_decide_won_lost_no_verdict() -> None:
    rows = run_decide({1: WON, 2: LOST}, {1: REF, 2: REF})
    by_c = {r["key"][0]: r for r in rows}
    assert by_c[1]["runs"][0]["verdict"] == "won" and by_c[1]["final"] == "won"
    assert by_c[2]["runs"][0]["verdict"] == "lost" and by_c[2]["final"] == "lost"
    assert abs(by_c[1]["runs"][0]["ratio"] - 112 / 102) < 1e-12
    assert by_c[1]["runs"][0]["paired_interval"] is not None  # same measured-N indices
    tie = run_decide({1: TIE}, {1: REF})[0]
    assert tie["runs"][0]["verdict"] == "no verdict" and tie["final"] == "no verdict"
    lo, hi = tie["runs"][0]["interval"]
    assert lo < 1 < hi


def test_decide_acceptance_share_is_a_ratio_of_medians() -> None:
    r = run_decide({1: WON, 2: WON}, {1: REF, 2: REF}, shares={1: 0.11, 2: -0.10})
    by_c = {x["key"][0]: x for x in r}
    assert by_c[1]["runs"][0]["flags"]["acceptance"] == {
        "share": 0.11, "undefined": False, "accept_part_ms_median": 11.0, "latency_gap_ms_median": 100.0,
        "per_rep_share_median": 0.11, "repetitions": 5, "flag": True}
    assert by_c[1]["final"] == "flagged"
    assert by_c[2]["runs"][0]["flags"]["acceptance"]["flag"] is False and by_c[2]["final"] == "won"


def test_decide_acceptance_ratio_of_medians_overrides_per_repetition_median() -> None:
    # the two share definitions disagree: the median acceptance part is 0 while one repetition with a near-zero gap
    # makes the median of per-repetition shares large
    r = run_decide({1: WON}, {1: REF}, shares={1: (0.0, 80.0, 0.2)})[0]
    a = r["runs"][0]["flags"]["acceptance"]
    assert a["share"] == 0.0 and a["per_rep_share_median"] == 0.2 and a["flag"] is False and r["final"] == "won"
    # the reverse disagreement: per-repetition median small, ratio of medians large -> flagged
    r = run_decide({1: WON}, {1: REF}, shares={1: (30.0, 100.0, 0.02)})[0]
    a = r["runs"][0]["flags"]["acceptance"]
    assert a["share"] == 0.3 and a["flag"] is True and r["final"] == "flagged"


def test_decide_acceptance_boundary_at_the_limit() -> None:
    # |share| = 0.10 exactly is not flagged (strictly greater), on both signs; just above is
    for part, flag in ((10.0, False), (-10.0, False), (10.001, True), (-10.001, True)):
        r = run_decide({1: WON}, {1: REF}, shares={1: (part, 100.0, 0.0)})[0]
        assert r["runs"][0]["flags"]["acceptance"]["flag"] is flag, part
        assert r["final"] == ("flagged" if flag else "won")
    # a negative gap: the sign of the share is the sign of the quotient, the flag reads its magnitude
    r = run_decide({1: LOST}, {1: REF}, shares={1: (5.0, -40.0, 0.0)}, canon=({1: TIE[:3]}, {1: REF[:3]}))[0]
    assert r["runs"][0]["flags"]["acceptance"]["share"] == -0.125 and r["final"] == "flagged"
    # --accept-share-limit moves the boundary
    r = run_decide({1: WON}, {1: REF}, "--accept-share-limit", "0.2", shares={1: (15.0, 100.0, 0.0)})[0]
    assert r["runs"][0]["flags"]["acceptance"]["flag"] is False


def test_decide_acceptance_zero_gap_is_undefined_and_flagged() -> None:
    r = run_decide({1: WON}, {1: REF}, shares={1: (0.0, 0.0, 0.05)})[0]
    a = r["runs"][0]["flags"]["acceptance"]
    assert a["share"] is None and a["undefined"] is True and a["flag"] is True and r["final"] == "flagged"
    md = dc.fmt_flag(a, "acceptance")
    assert md == "undefined (0.0 / 0.0 ms) FLAG" and dc.fmt_info(a) == "+0.050"


def test_decide_acceptance_missing_fields_are_incomplete() -> None:
    p = write("split.json", json.dumps({"summary": [{"label": "x", "reference": "ref", "concurrency": 1,
                                                       "accept_share": {"median": 0.0}}]}))
    r = run_decide({1: WON}, {1: REF}, "--accept-split", str(p))[0]
    a = r["runs"][0]["flags"]["acceptance"]
    assert a["flag"] is None and "latency_gap_ms.median" in a["reason"] and r["final"] == "incomplete"


def test_decide_looping_flag_on_disjoint_wilson_intervals() -> None:
    r = run_decide({1: WON}, {1: REF}, wilson=([0.6, 0.9], [0.1, 0.5]))[0]
    loop = r["runs"][0]["flags"]["looping"]
    assert loop["disjoint"] and loop["flag"] and r["final"] == "flagged"
    assert loop["sides"]["x"]["wilson95"] == [0.6, 0.9] and loop["sides"]["ref"]["fraction"] == 0.2
    touching = run_decide({1: WON}, {1: REF}, wilson=([0.5, 0.9], [0.1, 0.5]))[0]
    assert touching["runs"][0]["flags"]["looping"]["flag"] is False and touching["final"] == "won"
    # a row that counted records (no per-prompt counts) is not read
    old = write("old.jsonl", json.dumps({"accept_test": {k: {"looping": 1, "outputs": 5, "looping_wilson95": [0.1, 0.6]}
                                                          for k in ("x", "ref")}}) + "\n")
    r = run_decide({1: WON}, {1: REF}, "--accept-test", f"1={old}")[0]
    assert r["runs"][0]["flags"]["looping"]["flag"] is None and r["final"] == "incomplete"


def test_decide_canonical_cross_check() -> None:
    # canonical ranges say lost, the deciding run says won: flagged
    r = run_decide({1: WON}, {1: REF}, canon=({1: LOST[:3]}, {1: REF[:3]}))[0]
    c = r["runs"][0]["flags"]["canonical"]
    assert c["verdict"] == "lost" and c["flag"] and r["final"] == "flagged"
    # canonical no verdict against a won run is not a flag
    r = run_decide({1: WON}, {1: REF}, canon=({1: TIE[:3]}, {1: REF[:3]}))[0]
    assert r["runs"][0]["flags"]["canonical"]["verdict"] == "no verdict" and r["final"] == "won"
    # canonical won against a lost run: flagged
    r = run_decide({1: LOST}, {1: REF}, canon=({1: WON[:3]}, {1: REF[:3]}))[0]
    assert r["runs"][0]["flags"]["canonical"]["verdict"] == "won" and r["final"] == "flagged"
    # no canonical table: the point is incomplete, never silently decided
    r = run_decide({1: WON}, {1: REF}, canon=None)[0]
    assert r["runs"][0]["flags"]["canonical"]["flag"] is None and r["final"] == "incomplete"


def test_decide_rerun_rule() -> None:
    clean = split({1: 0.0})
    flagged = split({1: 0.5})
    tests = ["--rerun-accept-test", f"1={accept_test(*OVERLAP_W)}"]
    # first run flagged, second clean: the second run's verdict
    r = run_decide({1: WON}, {1: REF}, "--rerun-table", f"X={bench({1: LOST})}", "--rerun-table",
                   f"REF={bench({1: REF})}", "--rerun-accept-split", str(clean), *tests, shares={1: 0.5},
                   canon=({1: TIE[:3]}, {1: REF[:3]}))[0]
    assert [x["word"] for x in r["runs"]] == ["flagged", "lost"] and r["final"] == "lost"
    # both runs flagged: no verdict
    r = run_decide({1: WON}, {1: REF}, "--rerun-table", f"X={bench({1: WON})}", "--rerun-table",
                   f"REF={bench({1: REF})}", "--rerun-accept-split", str(flagged), *tests, shares={1: 0.5})[0]
    assert [x["word"] for x in r["runs"]] == ["flagged", "flagged"] and r["final"] == "no verdict"
    # an unflagged first run stands; a re-run is reported beside it
    r = run_decide({1: WON}, {1: REF}, "--rerun-table", f"X={bench({1: TIE})}", "--rerun-table",
                   f"REF={bench({1: REF})}", "--rerun-accept-split", str(clean), *tests)[0]
    assert [x["word"] for x in r["runs"]] == ["won", "no verdict"] and r["final"] == "won"


def test_decide_markdown_and_errors() -> None:
    args = ["--key", "concurrency", "--metric", "summary.tok", "--rows-field", "results",
            "--where", "phase~^measured-", "--table", f"X={bench({1: WON})}", "--table", f"REF={bench({1: REF})}",
            "--ratio", "X/REF"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        dc.main(args)
    lines = buf.getvalue().splitlines()
    assert lines[0].startswith("| concurrency | ratio | run |") and len(lines) == 3
    assert "| won |" in lines[2] and lines[2].endswith("| incomplete | incomplete |")
    assert ("| acceptance share (median part / median gap) | per-rep share median (info, not used) |"
            in lines[0])
    buf = io.StringIO()
    with redirect_stdout(buf):
        dc.main(args + ["--label", "X=x", "--label", "REF=ref", "--accept-split",
                        str(split({1: (0.0, 80.0, 0.2)}))])
    assert "| +0.000 (0.0 / 80.0 ms) ok | +0.200 |" in buf.getvalue().splitlines()[2]
    for bad in (["--ratio", "X/NOPE"], ["--metric", "summary.other"]):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                dc.main(args + bad)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"{bad} must be refused")
    dup = write("dup.json", json.dumps({"results": [{"phase": "measured-1", "concurrency": 1, "summary": {"tok": 1}},
                                                    {"phase": "measured-1", "concurrency": 1, "summary": {"tok": 2}}]}))
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            dc.main(args[:-4] + ["--table", f"X={dup}", "--table", f"REF={bench({1: REF})}", "--ratio", "X/REF"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("a repeated repetition index must be refused")



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
