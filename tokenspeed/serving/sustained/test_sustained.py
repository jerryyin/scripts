#!/usr/bin/env python3
"""CPU tests for the sustained-load analyses, on small synthetic inputs (run with pytest or directly)."""

from __future__ import annotations

import base64
import importlib.util
import pickle
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


def test_sustained_client_steady_and_conversations() -> None:
    sus = load("analyze_sustained.py")
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        req = lambda salt, turn: json.dumps({"messages": [{"role": "user", "content": f"[session {salt}] hi"}] * turn})
        evalscope_db(run / "collected" / "b.db", [
            (req("s1", 1), 0.0, 50.0, 1, 50.0, 1.0, 1000, 11, None),     # completes before the steady window
            (req("s1", 2), 60.0, 150.0, 1, 90.0, 2.0, 2000, 11, None),   # in [120, 900]
            (req("s2", 1), 100.0, 200.0, 1, 100.0, 3.0, 1000, 21, None),  # in
            (req("s2", 2), 210.0, 950.0, 0, None, None, None, None, None)])
        inst = run / "instances.json"
        inst.write_text(json.dumps([{"salt": "[session s1]", "turns": 2}, {"salt": "[session s2]", "turns": 2}]))
        res, _ = sus.client_metrics(str(run), (120, 900), str(inst))
        st = res["steady"]
        assert st["requests_ok"] == 2 and st["window_s"] == 780.0
        assert st["output_tokens"] == 32 and st["total_tok_per_s"] == round(3032 / 780, 2)
        assert res["conversations"] == {"started": 2, "completed_all_turns": 1, "unsalted_requests": 0, "unknown_salts": 0}

def test_prefix_reuse_classifies_turns() -> None:
    pr = load("prefix_reuse.py")
    cid = lambda i: f"chatcmpl-X-0000000{i}-0000-0000-0000-000000000000"
    blob = lambda i: base64.b64encode(pickle.dumps([{"id": cid(i)}])).decode()
    msgs = lambda salt, n: json.dumps({"messages": [{"role": "user", "content": f"[session {salt}] q"}] * n})
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "b.db"
        evalscope_db(db, [(msgs("a", 1), 0, 1, 1, 1, 0.5, 0, 0, blob(1)), (msgs("a", 2), 0, 1, 1, 1, 0.5, 0, 0, blob(2)),
                          (msgs("a", 3), 0, 1, 1, 1, 4.0, 0, 0, blob(3))])
        log = Path(d) / "s.log"
        log.write_text(finish(10, cid(1), 10000, 0, commit=10) + finish(20, cid(2), 15000, 9984, commit=20)
                       + finish(30, cid(3), 20000, 0, commit=30))
        res, recs = pr.analyze(str(log), str(db), dp=4, seeds=20, steady=(5, 25))
        assert [r["kind"] for r in recs] == ["first", "hit_prev", "miss_zero"]
        f = res["full"]
        assert (f["later_turns"], f["later_hit_prev"], f["uncached_tokens"]) == (2, 1, 10000 + 5016 + 20000)
        assert f["expected_uncached_same_rank"] == 10000 + 5000 + 5000
        assert res["steady"]["requests"] == 2 and res["uniform_placement_uncached"]["seeds"] == 20

def test_steady_window_parsing() -> None:
    db = load("evalscope_db.py")
    assert db.parse_steady("120:900") == (120.0, 900.0) and db.parse_steady("") is None
    for bad in ("900:120", "a:b", "5"):
        try:
            db.parse_steady(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert db.steady_bounds(10.0, (5.0, 50.0), end=40.0) == (15.0, 40.0)
    assert db.salt_of(b'{"x": "[session k1] hi"}') == "k1" and db.salt_of("none") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
