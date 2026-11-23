#!/usr/bin/env python3
"""CPU tests for the greedy-correctness tools, on small synthetic inputs (run with pytest or directly)."""

from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from log_fixtures import batch, evalscope_db, finish, ts  # noqa: E402,F401


def load(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, HERE / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_greedy_prompts_are_deterministic() -> None:
    gp = load("greedy_prompts.py")

    class Chars:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    text = ("A line of prose. Another sentence here! And one more? " * 3 + "\n") * 3000
    a = gp.build(text, Chars(), "m", seed=0, count=4, min_tokens=50, max_tokens=200, max_new=16)
    b = gp.build(text, Chars(), "m", seed=0, count=4, min_tokens=50, max_tokens=200, max_new=16)
    assert a == b and len(a) == 12
    assert [r["endpoint"] for r in a].count("/v1/chat/completions") == 8
    first = a[0]["body"]["prompt"]
    assert first.endswith((".", "!", "?")) and len(first) <= 50
    assert all(r["body"]["temperature"] == 0 and r["body"]["max_tokens"] == 16 for r in a)
    assert gp.build(text, Chars(), "m", seed=1, count=4, min_tokens=50, max_tokens=200)[0] != a[0]
    try:
        gp.build(text[:1000], Chars(), "m")
    except ValueError:
        pass
    else:
        raise AssertionError("a corpus shorter than MIN_CORPUS_CHARS must be refused")
    try:
        gp.build("no sentence ends here\n" * 8000, Chars(), "m", count=2, min_tokens=50, max_tokens=60)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a slice without a sentence end must not be used")


def test_compare_outputs_pairs_and_cache_free() -> None:
    co = load("compare_outputs.py")
    rec = lambda i, ids, cache, acc=None: json.dumps({"idx": i, "endpoint": "/v1/completions", "text": str(ids),
                                                       "http_status": 200, "reencoded_ids": ids, "acc_len": acc,
                                                       "request_stats": {"cache_tokens": cache}})
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.jsonl", Path(d) / "b.jsonl"
        a.write_text("\n".join([rec(0, [1, 2, 3], "0", 3.0), rec(1, [1, 2], "0", 2.0), rec(2, [5, 6], "256")]) + "\n")
        b.write_text("\n".join([rec(0, [1, 2, 3], "0"), rec(1, [1, 9], "0"), rec(2, [7, 6], "0")]) + "\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            co.main(["--pair", f"A={a},B={b}", "--cache-free"])
        s = json.loads(buf.getvalue())
        assert (s["exact"], s["token_identical"], s["differ_at_token_0"], s["min_first_diff"]) == (1, 1, 1, 0)
        assert s["total_common_prefix"] == 3 + 1 + 0 and s["acceptance"]["A"]["mean"] == 2.5
        assert s["cache_free"]["idx"] == [0, 1] and s["cache_free"]["exact"] == 1
        assert s["max_first_diff"] == 1
        assert s["first_diff_buckets"] == {"0": 1, "1-4": 1, "5-31": 0, "32-127": 0, "128+": 0}
    # Bucket edges: every token-differing prompt is counted once; token-identical ones are not counted.
    edges = [0, 1, 4, 5, 31, 32, 127, 128, 5000]
    rows = [{"idx": i, "exact": False, "first_diff_token": d, "common_prefix": d} for i, d in enumerate(edges)]
    rows.append({"idx": len(edges), "exact": True, "first_diff_token": None, "common_prefix": 7})
    e = co.summarize(rows, "edges")
    assert e["first_diff_buckets"] == {"0": 1, "1-4": 2, "5-31": 2, "32-127": 2, "128+": 2}
    assert sum(e["first_diff_buckets"].values()) == e["prompts"] - e["token_identical"] == len(edges)
    assert (e["min_first_diff"], e["max_first_diff"]) == (0, 5000)
    none = co.summarize(rows[-1:], "identical")
    assert none["max_first_diff"] is None and set(none["first_diff_buckets"].values()) == {0}


def test_probe_summary_counts() -> None:
    pb = load("probe.py")
    ok = {"pair": "X vs Y", "http_status": 200}
    rows = [{**ok, "idx": 0, "probe_top1_is": "X", "probe_top1_prob": 0.9, "gap_lower_bound_nonTop1_candidate": 2.2},
            {**ok, "idx": 1, "probe_top1_is": "Y", "probe_top1_prob": 0.3, "gap_lower_bound_nonTop1_candidate": 0.4},
            {**ok, "idx": 2, "probe_top1_is": "neither", "probe_top1_prob": 0.2,
             "gap_lower_bound_nonTop1_candidate": 1.5, "regenerated_prefix_tail_matches": True},
            {**ok, "idx": 3, "probe_top1_is": "X", "probe_top1_prob": 0.9,
             "gap_lower_bound_nonTop1_candidate": 9.0, "regenerated_prefix_tail_matches": False},
            {**ok, "idx": 4, "http_status": 503, "probe_top1_is": "neither", "probe_top1_prob": None,
             "gap_lower_bound_nonTop1_candidate": None},
            {"label": "X", "idx": 0, "j": 0, "match": True, "gap_lb_if_not_top1": None},
            {"label": "X", "idx": 0, "j": 1, "match": False, "gap_lb_if_not_top1": 1.2}]
    s = pb.summarize(rows)
    p = s["X vs Y"]
    assert (p["valid"], p["top1_is_X"], p["top1_is_Y"], p["top1_neither"]) == (3, 1, 1, 1)
    assert (p["X_certified_idx"], p["Y_certified_idx"], p["excluded_idx"]) == ([2], [0, 2], [3, 4])
    assert s["teacher-forced X"] == {"positions": 2, "prompts": 1, "emitted_not_top1": 1,
                                     "emitted_not_top1_certified_gap_gt": 1}


def test_compare_outputs_pairs_by_prompt_key_with_loops() -> None:
    co = load("compare_outputs.py")
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.jsonl", Path(d) / "b.jsonl"
        loop = [7, 8] * 40
        a.write_text("".join(json.dumps(r) + "\n" for r in [
            {"idx": 0, "concurrency": 1, "prompt_sha256": "p", "text": "xy", "reencoded_ids": loop, "acc_len": 5.8},
            {"idx": 0, "concurrency": 2, "prompt_sha256": "q", "text": "z", "reencoded_ids": list(range(80)),
             "acc_len": 3.0}]))
        b.write_text("".join(json.dumps(r) + "\n" for r in [
            {"idx": 3, "concurrency": 1, "prompt_sha256": "p", "text": "xw", "reencoded_ids": loop[:5] + [9] * 75,
             "acc_len": 4.0}]))
        A = co.load(str(a), ("prompt_sha256",), [("concurrency", "=", "1")])
        B = co.load(str(b), ("prompt_sha256",), [co.parse_where("concurrency=1")])
        assert len(co.load(str(a), ("prompt_sha256",), [co.parse_where("text~^x")])) == 1
        rows = co.compare(A, B)
    assert len(rows) == 1 and rows[0]["first_diff_token"] == 5 and not rows[0]["exact"]
    assert rows[0]["loop_period"] == (2, 1) and rows[0]["acc_len"] == (5.8, 4.0)
    s = co.summarize(rows, "a vs b")
    assert (s["unmatched"], s["looping"], s["same_loop_period"]) == (0, [1, 1], 0)

def test_compare_outputs_accept_test_server_log_and_paired_sign_test() -> None:
    co = load("compare_outputs.py")
    loop = [7, 8] * 40
    with tempfile.TemporaryDirectory() as d:
        a, b, log = Path(d) / "a.jsonl", Path(d) / "b.jsonl", Path(d) / "server.log"
        rec = lambda i, ids, acc: {"idx": i, "concurrency": 1, "prompt_sha256": f"p{i}", "text": str(ids[:3]),
                                   "reencoded_ids": ids, "acc_len": acc, "request_id": f"r{i}"}
        a.write_text("".join(json.dumps(rec(i, loop, 5.5)) + "\n" for i in range(6)))
        rows_b = [rec(0, loop, 5.5)] + [rec(i, list(range(80)), 4.0) for i in range(1, 6)]
        b.write_text("".join(json.dumps(r) + "\n" for r in rows_b))
        log.write_text("".join(f"Req: r{i} Finish! Accept_num_tokens_avg: {3.5 + 0.1 * i}\n" for i in range(1, 6)))
        out = io.StringIO()
        with redirect_stdout(out):
            co.main(["--pair", f"x={a},y={b}", "--key", "prompt_sha256", "--accept-server-log", f"y={log}",
                     "--accept-test", "--bootstrap", "200"])
    s = json.loads(out.getvalue().strip().splitlines()[-1])
    t = s["accept_test"]
    assert s["server_acceptance_matched"] == {"y": [5, 6]}
    assert t["y"]["acc_source"] == ["evalscope", "server"] and abs(t["y"]["min"] - 3.6) < 1e-9
    assert (t["x"]["looping_prompts"], t["y"]["looping_prompts"], t["y"]["prompts"]) == (6, 1, 6)
    p = t["paired"]
    assert (p["n"], p["positive"], p["negative"], p["zero"]) == (6, 0, 5, 1)
    assert abs(p["sign_test_p"] - 2 / 32) < 1e-12 and p["bootstrap95_mean_diff"][1] < 0
    assert t["paired_token_identical"]["n"] == 1 and t["paired_token_identical"]["mean_diff"] == 0
    assert t["paired_both_looping"]["n"] == 1 and t["paired_both_nonlooping"]["n"] == 0
    assert t["unpaired"]["mean_diff"] < 0


def test_wilson_interval_values() -> None:
    co = load("compare_outputs.py")
    # Wilson 95%: centre (k + z^2/2) / (n + z^2), half-width z sqrt(k (n - k) / n + z^2 / 4) / (n + z^2), z = 1.96
    for (k, n), want in (((0, 10), [0.0, 0.2775]), ((5, 10), [0.2366, 0.7634]), ((10, 10), [0.7225, 1.0])):
        lo, hi = co.wilson(k, n)
        assert abs(lo - want[0]) < 5e-5 and abs(hi - want[1]) < 5e-5, (k, n, lo, hi)
    assert co.wilson(0, 0) is None


def test_looping_counts_each_prompt_once() -> None:
    """Repeated greedy runs of one prompt are one observation: 2 prompts x 5 repetitions, p loops in 3 of its
    5 records and q in 1, so 1 of 2 prompts loops (not 4 of 10 records). A tie (2 of 4) does not loop; a
    failed record and one without ids or loop period are left out."""
    co = load("compare_outputs.py")
    loop, flat = [7, 8] * 40, list(range(80))
    def rec(prompt, rep, ids, **extra):
        return {"phase": f"measured-{rep}", "prompt_sha256": prompt, "text": "t", "reencoded_ids": ids,
                "acc_len": 3.0, **extra}
    rows = ([rec("p", i, loop if i < 3 else flat) for i in range(5)]
            + [rec("q", i, loop if i < 1 else flat) for i in range(5)]
            + [rec("t", i, loop if i < 2 else flat) for i in range(4)]
            + [rec("t", 5, loop, success=False), rec("t", 6, None)])
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.jsonl"
        a.write_text("".join(json.dumps(r) + "\n" for r in rows))
        A = co.load(str(a), ("prompt_sha256",))
    s = co.side_stats(A)
    assert (s["looping_prompts"], s["prompts"]) == (1, 3)
    assert (s["looping_records"], s["records"], s["unclassifiable_records"]) == (6, 16, 2)
    assert s["looping_wilson95"] == co.wilson(1, 3) and abs(s["looping_fraction"] - 1 / 3) < 1e-12


def test_records_only_in_b_are_unmatched() -> None:
    co = load("compare_outputs.py")
    rec = lambda i: json.dumps({"idx": i, "text": "x", "reencoded_ids": [1, 2]})  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.jsonl", Path(d) / "b.jsonl"
        a.write_text(rec(0) + "\n" + rec(1) + "\n")
        b.write_text(rec(0) + "\n" + rec(2) + "\n" + rec(3) + "\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            co.main(["--pair", f"A={a},B={b}"])
    s = json.loads(buf.getvalue())
    assert s["prompts"] == 2 and s["unmatched"] == 3 and s["unmatched_b_only"] == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
