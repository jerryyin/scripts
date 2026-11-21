#!/usr/bin/env python3
"""Non-GPU unit tests for petit_layer.py (arguments, token list, limits, test-source extraction)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import petit_layer

RANGE_COMPARE = Path(__file__).resolve().parents[2] / "numerics" / "range_compare.py"


def test_parse_requires_the_profile_and_tokens() -> None:
    a = petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "any_profile", "--tokens", "1,6"])
    assert a.profile == "any_profile" and a.tokens == [1, 6] and a.mode == "both" and a.reps == 3
    assert a.routing == "test" and a.seed == 0 and a.active_ranks is None
    p = petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "1024",
                                "--active-ranks", "1"])
    assert p.tokens == [1024] and p.active_ranks == 1
    for missing in (["--profile", "x"], ["--tokens", "1"]):
        with pytest.raises(SystemExit):
            petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", *missing])


@pytest.mark.parametrize("argv", [
    ["--tokens", "1025"],        # above Petit's per-rank capacity
    ["--tokens", "0"],
    ["--reps", "2"],
    ["--warmup", "0"],
    ["--mode", "neither"],
    ["--active-ranks", "0"],
])
def test_parse_rejects(argv) -> None:
    with pytest.raises(SystemExit):
        petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "1", *argv])


def test_check_tokens_cites_capacity() -> None:
    with pytest.raises(ValueError, match="max_tokens_per_rank=1024"):
        petit_layer.check_tokens([6, 2048])


def test_capacity_raises_token_limit_and_sets_plan() -> None:
    base = ["--ts-src", "/ts", "--out", "o.json", "--profile", "x"]
    p = petit_layer.parse_args([*base, "--tokens", "1024,2048,3072,4096", "--capacity", "4096"])
    assert p.tokens == [1024, 2048, 3072, 4096] and p.capacity == 4096
    assert petit_layer.parse_args([*base, "--tokens", "1024"]).capacity == 1024
    for bad in (["--tokens", "4097", "--capacity", "4096"], ["--tokens", "2048"],
                ["--tokens", "1", "--capacity", "1000"], ["--tokens", "1", "--capacity", "0"]):
        with pytest.raises(SystemExit):
            petit_layer.parse_args([*base, *bad])
    with pytest.raises(ValueError, match="max_tokens_per_rank=4096"):
        petit_layer.check_tokens([4097], 4096)
    assert petit_layer.apply_capacity({"a": 1}, 1024) == {"a": 1}  # default: plan unchanged
    assert petit_layer.apply_capacity({"a": 1}, 4096) == {"a": 1, "gluon_petit_max_tokens_per_rank": 4096}


SAMPLE = '''
    for case in cases:
        def refresh(step):
            x.copy_(step)

        def run():
            return kernel(x)

        refresh(0)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=8e-3)
'''


def test_extract_closures_and_tolerance() -> None:
    text = petit_layer.extract_closures(SAMPLE)
    assert text.startswith("def refresh(step):") and "def run():" in text and "refresh(0)" not in text
    assert petit_layer.test_tolerance(SAMPLE) == {"atol": 2e-6, "rtol": 8e-3}


@pytest.mark.skipif(not os.environ.get("TS_SRC"), reason="needs $TS_SRC (TokenSpeed checkout)")
def test_extract_from_pinned_test() -> None:
    source = (Path(os.environ["TS_SRC"]) / petit_layer.TEST_REL).read_text()
    text = petit_layer.extract_closures(source)
    compile(text, "closures", "exec")
    assert "def refresh(step):" in text and "kernel_moe_apply(" in text
    assert petit_layer.test_tolerance(source) == {"atol": 2e-6, "rtol": 8e-3}


def _timing(v):
    one = {"median": v, "min": v, "max": v, "mean": v, "n": 3}
    return {"max_over_ranks_ms": one, "per_rank_ms": [one, one]}


def test_summary_table() -> None:
    record = {"results": [{
        "tokens_per_gpu": 6, "pass": True, "max_abs_err": 0.0, "max_rel_err": 0.0,
        "graph_equals_eager": True,
        "per_rank": [{"first_call": {"compiled": True}}, {"first_call": {"compiled": False}}],
        "timing_ms": {"eager": _timing(0.25), "graph": _timing(0.125)}}]}
    table = petit_layer.summary_table(record).splitlines()
    assert table[0].startswith("| tokens/GPU | candidate | launched | pass |") and "graph per-rank medians" in table[0]
    # results without candidate/launched (Petit's own choice, older JSON) read as "petit" and ""
    assert table[2] == ("| 6 | petit |  | PASS | 0 | 0 | True | 1/2 | 0.2500 [0.2500, 0.2500] | 0.2500 / 0.2500 "
                        "| 0.1250 [0.1250, 0.1250] | 0.1250 / 0.1250 |")


def _record(reps=(0.20, 0.21, 0.22)):
    one = lambda v: {"per_rank_median_ms": [v, v - 0.01, v, v], "max_over_ranks_median_ms": v,
                     "max_over_ranks_min_ms": v - 0.005, "max_over_ranks_max_ms": v + 0.01}
    timing = {"max_over_ranks_ms": {}, "per_rank_ms": [], "per_repetition": [one(v) for v in reps]}
    return {"args": {"routing": "test", "seed": 0, "iters": 50}, "world_size": 4,
            "profile": {"name": "dsv41_flash"},
            "tokenspeed": {"commit": "e" * 40, "test": {"sha256": "a" * 64}},
            # results written before --active-ranks existed have no active_ranks field
            "results": [{"tokens_per_gpu": 6, "pass": True, "timing_ms": {"graph": timing, "eager": timing}}]}


def test_rows_one_per_repetition_and_range_compare_reads_them(tmp_path) -> None:
    src = tmp_path / "petit.json"
    src.write_text(json.dumps(_record()))
    out = tmp_path / "rows.csv"
    assert petit_layer.main(["--rows", str(out), str(src)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6 and {r["mode"] for r in rows} == {"graph", "eager"}
    assert [r["repetition"] for r in rows if r["mode"] == "graph"] == ["0", "1", "2"]
    assert rows[0]["active_ranks"] == "4" and rows[0]["tokens_per_gpu"] == "6"
    assert rows[0]["candidate"] == "petit" and rows[0]["launched"] == "" and rows[0]["layers"] == "1"
    summary = subprocess.run(
        [sys.executable, str(RANGE_COMPARE), "summarize", str(out), "--key", "profile,mode,tokens_per_gpu",
         "--metric", "max_over_ranks_median_ms", "--where", "mode=graph"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    head = summary[0].split(",")
    row = dict(zip(head, summary[1].split(",")))
    assert row["max_over_ranks_median_ms_n"] == "3"
    assert float(row["max_over_ranks_median_ms_median"]) == 0.21
    assert (float(row["max_over_ranks_median_ms_min"]), float(row["max_over_ranks_median_ms_max"])) == (0.20, 0.22)


def test_parse_candidates_layers_weights_and_skew() -> None:
    base = ["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "256"]
    a = petit_layer.parse_args(base)
    assert a.candidates is None and a.layers == 1 and a.weights == "test" and a.skew == 0.5
    f = petit_layer.parse_args(base + ["--candidates", "cta192/m32,cta56/m64w8", "--layers", "8",
                                       "--weights", "random", "--routing", "skew", "--skew", "1.0"])
    assert f.candidates == "cta192/m32,cta56/m64w8" and f.layers == 8 and f.routing == "skew"
    for bad in (["--candidates", "cta57/m32"], ["--candidates", "cta56/m48"], ["--layers", "0"],
                ["--weights", "dense"]):
        with pytest.raises(SystemExit):
            petit_layer.parse_args(base + bad)


def test_candidate_names_and_expected_launch() -> None:
    assert petit_layer.candidate_names(None, 384) is None
    flash = petit_layer.candidate_names("all", 384)
    assert len(flash) == 12 and flash[0] == "cta56/m32" and "cta192/m64w8" in flash
    # with 128 experts cta192 is cta128 (kProducerBlocks = min(blocks, experts)), so 'all' drops it
    dspark = petit_layer.candidate_names("all", 128)
    assert len(dspark) == 9 and not any(n.startswith("cta192") for n in dspark)
    with pytest.raises(ValueError, match="unknown candidates"):
        petit_layer.candidate_names("cta192/m32", 128)
    assert petit_layer.expected_launch("cta192/m64w4", 384) == (192, True, False)
    assert petit_layer.expected_launch("cta192/m32", 128) == (128, False, False)


def test_fit_line_and_split_chunk(tmp_path) -> None:
    rows = tmp_path / "rows.csv"
    fields = ["profile", "routing", "mode", "active_ranks", "candidate", "tokens_per_gpu", "repetition",
              "max_over_ranks_median_ms"]
    with rows.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n in (128, 256, 512, 1024):
            for k, d in enumerate((-0.01, 0.0, 0.01)):  # t = 0.3 + 0.0004 n, median exact
                w.writerow({"profile": "dsv41_flash", "routing": "test", "mode": "eager", "active_ranks": 1,
                            "candidate": "petit", "tokens_per_gpu": n, "repetition": k,
                            "max_over_ranks_median_ms": 0.3 + 0.0004 * n + d})
    text = petit_layer.fit_table(list(csv.DictReader(rows.open())), layers=40, chunks=(4096, 8192, 1536))
    assert "fit t = 0.3000 ms + 0.4000 us x rows" in text
    assert "| 1024 | 0.7096 [0.6996, 0.7196] | 3 | 0.7096 | +0.0000 | 42% | 28.38 |" in text
    assert "| 8192 | 8 | 227.07 | 8.00x | 27.72 |" in text  # 40 x 8 x 0.7096
    assert "| 1536 | 2 | 48.58 | 1.71x |" in text  # 40 x (0.7096 + 0.3 + 0.0004 x 512)
    assert petit_layer.main(["--fit", str(rows), "--fit-chunks", "4096", "--where", "routing=test"]) == 0
