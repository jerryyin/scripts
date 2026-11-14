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


def test_default_tokens_from_serving() -> None:
    # 1 + decode tokens per rank (requests per rank 1..32 under DP4 x 6) + prefill chunks
    assert petit_layer.default_tokens() == [1, 6, 12, 24, 48, 96, 192, 256, 512, 1024]
    # draft MoE: one DSpark block (5 rows per request) per decode step, never prefill tokens
    assert petit_layer.default_tokens("dsv41_flash_dspark") == [1, 5, 10, 20, 40, 80, 160]


def test_parse_defaults() -> None:
    a = petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json"])
    assert a.profile == "dsv41_flash" and a.mode == "both" and a.reps == 3
    assert a.tokens == petit_layer.default_tokens()
    assert a.routing == "test" and a.seed == 0 and a.active_ranks is None
    d = petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "dsv41_flash_dspark"])
    assert d.tokens == petit_layer.default_tokens("dsv41_flash_dspark")
    p = petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--tokens", "1024", "--active-ranks", "1"])
    assert p.tokens == [1024] and p.active_ranks == 1


@pytest.mark.parametrize("argv", [
    ["--tokens", "1025"],        # above Petit's per-rank capacity
    ["--tokens", "0"],
    ["--reps", "2"],
    ["--warmup", "0"],
    ["--profile", "dsv4"],       # EP8-only profile
    ["--mode", "neither"],
    ["--active-ranks", "0"],
])
def test_parse_rejects(argv) -> None:
    with pytest.raises(SystemExit):
        petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", *argv])


def test_check_tokens_cites_capacity() -> None:
    with pytest.raises(ValueError, match="max_tokens_per_rank=1024"):
        petit_layer.check_tokens([6, 2048])


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
    assert table[0].startswith("| tokens/GPU | pass |") and "graph per-rank medians" in table[0]
    assert table[2] == ("| 6 | PASS | 0 | 0 | True | 1/2 | 0.2500 [0.2500, 0.2500] | 0.2500 / 0.2500 "
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
    summary = subprocess.run(
        [sys.executable, str(RANGE_COMPARE), "summarize", str(out), "--key", "profile,mode,tokens_per_gpu",
         "--metric", "max_over_ranks_median_ms", "--where", "mode=graph"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    head = summary[0].split(",")
    row = dict(zip(head, summary[1].split(",")))
    assert row["max_over_ranks_median_ms_n"] == "3"
    assert float(row["max_over_ranks_median_ms_median"]) == 0.21
    assert (float(row["max_over_ranks_median_ms_min"]), float(row["max_over_ranks_median_ms_max"])) == (0.20, 0.22)
