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

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
