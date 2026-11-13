#!/usr/bin/env python3
"""CPU tests for batch_log.py dp-spread, on synthetic logs from log_fixtures.py (run with pytest or directly)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batch_log  # noqa: E402
from log_fixtures import batch, finish, ts  # noqa: E402

T = 1_800_000_000.0  # 2027-01-15T08:00:00Z, a whole second


def spread_log() -> list[str]:
    lines = ["\x1b[2m"]  # ANSI noise is stripped by the parser
    # smoke window [T, T+10): 4 requests over ranks 0..3
    lines += [batch(T + 1 + 0.25 * r, "Prefill", r, new_seq=1, new_token=50) for r in range(4)]
    # rep1 [T+10, T+20): rank 0 gets 3, rank 1 gets 5, rank 3 gets 2 sequences; decode lines are ignored
    lines += [batch(T + 10.0, "Prefill", 0, new_seq=3), batch(T + 12.5, "Prefill", 1, new_seq=5),
              batch(T + 19.999, "Prefill", 3, new_seq=2), batch(T + 15, "Decode", 0, running_req=8)]
    # exactly at the end bound: belongs to rep2 [T+20, open end)
    lines += [batch(T + 20.0, "Prefill", 2, new_seq=7), finish(T + 21, "a", 100, 0)]
    lines += [batch(T - 5, "Prefill", 0, new_seq=9)]  # before every window
    return lines


def test_dp_spread_counts_new_sequences_per_rank_and_window() -> None:
    events = batch_log.parse_lines(spread_log())
    windows = [("smoke", T, T + 10), ("rep1", T + 10, T + 20), ("rep2", T + 20, None), ("unset", None, T + 30)]
    counts = batch_log.dp_spread(events, windows)
    assert counts == {"smoke": {0: 1, 1: 1, 2: 1, 3: 1}, "rep1": {0: 3, 1: 5, 3: 2}, "rep2": {2: 7}, "unset": {}}
    assert batch_log.render_dp_spread(counts).splitlines() == [
        "smoke rank0=1 rank1=1 rank2=1 rank3=1 total=4", "rep1 rank0=3 rank1=5 rank3=2 total=10",
        "rep2 rank2=7 total=7", "unset  total=0"]


def test_parse_window_bounds() -> None:
    start = ts(T)[:19] + "Z"
    assert batch_log.parse_window(f"rep1={start},{ts(T + 10)[:19]}Z") == ("rep1", T, T + 10)
    assert batch_log.parse_window(f"tail={start}") == ("tail", T, None)
    assert batch_log.parse_window(f"naive={start[:-1]}") == ("naive", T, None)  # no offset: UTC


def test_dp_spread_cli() -> None:
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.stderr.log"
        log.write_text("".join(spread_log()))
        iso = lambda t: ts(t)[:19] + "Z"  # noqa: E731
        out = subprocess.run([sys.executable, str(HERE / "batch_log.py"), "dp-spread", str(log),
                              "--window", f"smoke={iso(T)},{iso(T + 10)}", "--window", f"rep1={iso(T + 10)},{iso(T + 20)}"],
                             capture_output=True, text=True, check=True).stdout
        assert out.splitlines() == ["smoke rank0=1 rank1=1 rank2=1 rank3=1 total=4", "rep1 rank0=3 rank1=5 rank3=2 total=10"]


def prefill_log() -> list[str]:
    """Two prefill points: p1 = 2 requests of 1000 tokens on 2 DP ranks, chunk size 512 (valid);
    p2 = 1 request of 1000 tokens with a prefix-cache hit (cache_tokens=200, invalid)."""
    lines = []
    for r in (0, 1):  # p1: per rank 512 + 488 tokens, chunk 0.1 s, final step 0.05 s
        lines += [batch(T + 1.0 + 0.01 * r, "Prefill", r, new_seq=1, new_token=512),
                  batch(T + 1.1 + 0.01 * r, "Prefill", r, new_seq=0, new_token=488)]
    lines += [finish(T + 1.16, "a", 1000, 0, out=1), finish(T + 1.15, "b", 1000, 0, out=1)]
    lines += [batch(T + 5.0, "Prefill", 0, new_seq=1, new_token=800), batch(T + 5.2, "Decode", 0),
              finish(T + 5.25, "c", 1000, 200, out=1)]
    return lines


def test_prefill_point_ranks_validity_chunk_cost_and_tail() -> None:
    events = batch_log.parse_lines(prefill_log())
    p1 = batch_log.prefill_point(events, 1000, 2, T, T + 4)
    assert p1["valid"] and p1["requeststats_cache_tokens"] == [0] and p1["prefill_tokens_total"] == 2000
    r0 = p1["ranks"]["0"]
    assert (r0["chunks"], r0["prefill_tokens"], r0["requests_by_tokens"]) == (2, 1000, 1.0)
    assert round(r0["chunk_time_median_s"], 6) == 0.1 and r0["chunk_time_source"] == "prefill->prefill line gaps"
    assert round(r0["prefill_span_s"], 6) == 0.2 and round(p1["server_tok_per_s_all_ranks"]) == 10000  # 2000 tokens over the longest rank span
    p2 = batch_log.prefill_point(events, 1000, 1, T + 4, T + 10)
    assert not p2["valid"] and p2["requeststats_cache_tokens"] == [200] and not p2["prefill_tokens_ok"]
    assert p2["ranks"]["0"]["chunk_time_source"] == "prefill->any next line gaps"
    cc = batch_log.chunk_cost([p1, p2])
    assert [(c["chunk_tokens"], c["completes_request"], c["n"], round(c["median_ms"], 3)) for c in cc] == [
        (512, "intermediate", 2, 100.0)]
    tl = batch_log.tail({"p1": p1, "p2": p2})
    assert [(r["point"], r["tail_tokens"], round(r["final_step_ms"], 3)) for r in tl["rows"]] == [("p1", [488], 50.0)]


def test_prefill_cli_with_matrix_benchmark_json() -> None:
    import json
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.stderr.log"
        log.write_text("".join(prefill_log()))
        iso = lambda t: ts(t)[:19] + "Z"  # noqa: E731
        bj = Path(d) / "benchmark.json"
        bj.write_text(json.dumps({"prompt_tokens": 1000, "results": [
            {"phase": "measured-1-c2", "concurrency": 2, "output_dir": d, "started_at": iso(T), "finished_at": iso(T + 4)},
            {"phase": "measured-1-c1", "concurrency": 1, "output_dir": d, "started_at": iso(T + 4), "finished_at": iso(T + 10)}]}))
        out = Path(d) / "prefill.json"
        p = subprocess.run([sys.executable, str(HERE / "batch_log.py"), "prefill", str(log), "--benchmark-json", str(bj),
                            "--chunk-cost", "--tail", "--out-json", str(out)], capture_output=True, text=True)
        assert p.returncode == 3, p.stderr  # one invalid point
        res = json.loads(out.read_text())
        assert {k: v["valid"] for k, v in res["points"].items()} == {"measured-1-c2": True, "measured-1-c1": False}
        assert "| measured-1-c2 | 1000 | 2 | True | [0] |" in p.stdout


def test_prefill_validity_from_prefill_lines_without_requeststats() -> None:
    """Ported from r0002-f2's prefill-steps test: a server without RequestStats lines (TokenSpeed e8fd096e)
    is checked from the Prefill lines' #cached-token; batching = steps, present_max and the step sequence."""
    import json
    lines = [batch(T + 1.0, "Prefill", 0, new_seq=2, new_token=8192, running_req=2, queue_req=6),
             batch(T + 1.5, "Prefill", 0, new_seq=2, new_token=7808, cached_token=0, running_req=2, queue_req=4),
             batch(T + 2.0, "Decode", 0, running_req=8),
             batch(T + 12.0, "Prefill", 0, new_seq=1, new_token=4000, cached_token=96, running_req=1)]
    events = batch_log.parse_lines(lines)
    c8 = batch_log.prefill_point(events, 8000, 2, T, T + 10)
    assert (c8["valid"], c8["validity_source"], c8["cache_tokens"], c8["requeststats_n"]) == (True, "prefill-lines", [0], 0)
    assert (c8["steps"], c8["present_max"], c8["steps_seq"]) == (2, 8, ["2/8192", "2/7808"])
    c1 = batch_log.prefill_point(events, 4000, 1, T + 10, T + 20)
    assert (c1["valid"], c1["validity_source"], c1["cache_tokens"], c1["prefill_lines_cached_token_sum"]) == (
        False, "prefill-lines", [96], 96)
    assert (c1["prefill_tokens_ok"], c1["present_max"]) == (True, 1)
    empty = batch_log.prefill_point(events, 8000, 1, T + 30, T + 40)
    assert (empty["valid"], empty["validity_source"], empty["steps"]) == (False, "none", 0)
    # RequestStats present -> they decide (a cached prefix in RequestStats invalidates despite clean lines)
    rs = batch_log.prefill_point(batch_log.parse_lines(lines + [finish(T + 3, "a", 8000, 0, out=1), finish(T + 3.1, "b", 8000, 0, out=1)]),
                                 8000, 2, T, T + 10)
    assert (rs["valid"], rs["validity_source"]) == (True, "requeststats")
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.stderr.log"
        log.write_text("".join(lines))
        iso = lambda t: ts(t)[:19] + "Z"  # noqa: E731
        out = Path(d) / "p.json"
        p = subprocess.run([sys.executable, str(HERE / "batch_log.py"), "prefill", str(log), "--window", f"c8={iso(T)},{iso(T + 10)}",
                            "--prompt-tokens", "8000", "--requests", "2", "--out-json", str(out)], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        assert "| c8 | 8000 | 2 | True | [0] | prefill-lines | 2 | 8 | 2/8192 2/7808 |" in p.stdout
        assert json.loads(out.read_text())["points"]["c8"]["validity_source"] == "prefill-lines"


def test_prefill_cli_matrix_phases_repeated_across_concurrencies() -> None:
    """The matrix names phases measured-1.. for every concurrency; each result must stay its own point."""
    import json
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.stderr.log"
        log.write_text("".join(prefill_log()))
        iso = lambda t: ts(t)[:19] + "Z"  # noqa: E731
        bj = Path(d) / "benchmark.json"
        bj.write_text(json.dumps({"prompt_tokens": 1000, "results": [
            {"phase": "measured-1", "concurrency": 2, "output_dir": d, "started_at": iso(T), "finished_at": iso(T + 4)},
            {"phase": "measured-1", "concurrency": 1, "output_dir": d, "started_at": iso(T + 4), "finished_at": iso(T + 10)}]}))
        out = Path(d) / "prefill.json"
        p = subprocess.run([sys.executable, str(HERE / "batch_log.py"), "prefill", str(log), "--benchmark-json", str(bj),
                            "--out-json", str(out)], capture_output=True, text=True)
        assert p.returncode == 3, p.stderr
        res = json.loads(out.read_text())["points"]
        assert {k: (v["requests"], v["valid"]) for k, v in res.items()} == {"measured-1/c2": (2, True), "measured-1/c1": (1, False)}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
