#!/usr/bin/env python3
"""Non-GPU unit tests for the pure logic of dist_timing.py."""

from __future__ import annotations

import pytest

import dist_timing


def test_parse_int_list_accepts_commas_and_spaces() -> None:
    assert dist_timing.parse_int_list("1,6, 12 24") == [1, 6, 12, 24]


@pytest.mark.parametrize("text", ["", " , ", "1,-2", "1,x"])
def test_parse_int_list_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        dist_timing.parse_int_list(text)


def test_summarize() -> None:
    s = dist_timing.summarize([3.0, 1.0, 2.0, 10.0])
    assert s == {"median": 2.5, "min": 1.0, "max": 10.0, "mean": 4.0, "n": 4}
    with pytest.raises(ValueError):
        dist_timing.summarize([])


def test_reduce_repetition_takes_max_over_ranks_per_call() -> None:
    # rank 0 is slow on call 0, rank 1 on call 2: per-call max is [5, 2, 6]
    r = dist_timing.reduce_repetition([[5.0, 1.0, 1.0], [1.0, 2.0, 6.0]])
    assert r["per_rank_median_ms"] == [1.0, 2.0]
    assert r["max_over_ranks_median_ms"] == 5.0
    assert (r["max_over_ranks_min_ms"], r["max_over_ranks_max_ms"]) == (2.0, 6.0)


def test_reduce_repetition_rejects_unequal_call_counts() -> None:
    with pytest.raises(ValueError):
        dist_timing.reduce_repetition([[1.0, 2.0], [1.0]])


def test_aggregate_over_repetitions() -> None:
    reps = [dist_timing.reduce_repetition(c) for c in
            ([[1.0], [2.0]], [[3.0], [1.0]], [[2.0], [2.0]])]
    agg = dist_timing.aggregate(reps)
    assert agg["repetitions"] == 3
    assert agg["max_over_ranks_ms"]["median"] == 2.0
    assert (agg["max_over_ranks_ms"]["min"], agg["max_over_ranks_ms"]["max"]) == (2.0, 3.0)
    assert agg["per_rank_ms"][0]["median"] == 2.0 and agg["per_rank_ms"][1]["max"] == 2.0


def test_count_files(tmp_path) -> None:
    assert dist_timing.count_files(tmp_path / "missing") == 0
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "k.hsaco").write_text("x")
    (tmp_path / "b.json").write_text("{}")
    assert dist_timing.count_files(tmp_path) == 2


def test_appeared() -> None:
    before = {"0": [{"pid": 9, "mem_bytes": 1}], "1": []}
    after = {"0": [{"pid": 9, "mem_bytes": 1}, {"pid": 12, "mem_bytes": 5}], "1": [{"pid": 13, "mem_bytes": 5}]}
    assert dist_timing.appeared(before, after) == {"0": [12], "1": [13]}
