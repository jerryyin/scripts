#!/usr/bin/env python3
"""Non-GPU unit tests for petit_layer.py (arguments, token list, limits, test-source extraction)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import petit_layer


def test_default_tokens_from_serving() -> None:
    # 1 + decode tokens per rank (requests per rank 1..32 under DP4 x 6) + prefill chunks
    assert petit_layer.default_tokens() == [1, 6, 12, 24, 48, 96, 192, 256, 512, 1024]


def test_parse_defaults() -> None:
    a = petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json"])
    assert a.profile == "dsv41_flash" and a.mode == "both" and a.reps == 3
    assert a.tokens == petit_layer.default_tokens()
    assert a.routing == "test" and a.seed == 0


@pytest.mark.parametrize("argv", [
    ["--tokens", "1025"],        # above Petit's per-rank capacity
    ["--tokens", "0"],
    ["--reps", "2"],
    ["--warmup", "0"],
    ["--profile", "dsv4"],       # EP8-only profile
    ["--mode", "neither"],
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
