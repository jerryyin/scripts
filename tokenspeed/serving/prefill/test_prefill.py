#!/usr/bin/env python3
"""CPU tests for prefill_aggregate.py, on small synthetic inputs (run with pytest or directly)."""

from __future__ import annotations

import csv
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


def test_prefill_chunk_cost_and_tail() -> None:
    pa = load("prefill_aggregate.py")
    T = 1_800_000_000.0
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d)
        for rep, tail_ms in ((1, 140.0), (2, 150.0)):
            p = raw / f"L2048_B1_rep{rep}"
            p.mkdir()
            (p / "metrics.json").write_text(json.dumps({"L": 2048, "B": 1, "valid": True, "max_tokens": 1}))
            gaps = [{"chunk_tokens": 1024, "new_seq": 1, "dt_s": 0.05, "next_kind": "Prefill"},
                    {"chunk_tokens": 1024, "new_seq": 1, "dt_s": 0.07, "next_kind": "Decode"}]
            (p / "metrics-chunks.json").write_text(json.dumps({"0": {"chunk_gaps": gaps}}))
            (p / "requests.json").write_text(json.dumps({"requests": [{"t_send": T}]}))
            (p / "server-log-slice.txt").write_text(
                batch(T + 0.01, "Prefill", 0, new_seq=1, new_token=1024)
                + batch(T + 0.06, "Prefill", 0, new_seq=1, new_token=1024 if rep == 1 else 512)
                + finish(T + 0.06 + tail_ms / 1e3, "r", 2048, 0, out=1))
        cc = pa.chunk_cost(str(raw))
        assert [(c["completes_request"], c["n"], round(c["median_ms"], 3)) for c in cc] == [("intermediate", 2, 50.0)]
        rows, summary, fits = pa.tail(str(raw))
        assert [(s["tail_tokens"], round(s["median_ms"], 3)) for s in summary] == [(512, 150.0), (1024, 140.0)]
        assert abs(fits["B1"]["per_token_ms"] - (-10.0 / 512)) < 1e-6

def test_prefill_summary_table() -> None:
    pa = load("prefill_aggregate.py")
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "raw", Path(d) / "out"
        for rep, (tok_s, valid) in enumerate(((1000.0, True), (3000.0, True), (9000.0, False)), 1):
            p = raw / f"L1024_B4_rep{rep}"
            p.mkdir(parents=True)
            server = {"server_tok_per_s_all_ranks": tok_s, "ranks": {"0": {
                "chunk_time_median_s": 0.05, "chunks": 4, "requests_by_tokens": 4.0, "prefill_tok_per_s": tok_s}}}
            (p / "metrics.json").write_text(json.dumps({"L": 1024, "B": 4, "valid": valid, "batch_prompt_tok_per_s": tok_s,
                                                        "ttft_s": {"mean": 1.0, "max": 2.0}, "server": server}))
        buf = io.StringIO()
        with redirect_stdout(buf):
            pa.main(["--raw", str(raw), "--out", str(out)])
        row = next(csv.DictReader(open(out / "summary.csv")))
        assert (row["n_valid"], row["n_total"], float(row["batch_prompt_tok_per_s_median"])) == ("2", "3", 2000.0)
        assert (float(row["batch_prompt_tok_per_s_min"]), float(row["chunk_time_median_ms_median"])) == (1000.0, 50.0)
        assert "| 1024 | 4 | 2/3 |" in buf.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
