#!/usr/bin/env python3
"""CPU tests for e2e_decompose.py, on small synthetic inputs (run with pytest or directly)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from log_fixtures import batch, evalscope_db, finish, ts  # noqa: E402,F401


def load(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, HERE / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2e_decompose_residency_phases_and_markdown() -> None:
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    lines = []
    for i in range(4):  # 4 prefill chunks of 1000 tokens, 0.1 s each
        lines.append(batch(T + 0.1 * i, "Prefill", 0, new_seq=1, new_token=1000, running_req=2, queue_req=3))
    for i in range(6):  # decode at full batch 2, one line per 0.2 s
        lines.append(batch(T + 0.4 + 0.2 * i, "Decode", 0, running_req=2, queue_req=1 if i == 0 else 0, gen=50.0, acc=3.0))
    lines += [finish(T + 1.6, "a", 2000, 0, out=11, acc=3.0), finish(T + 1.7, "b", 2000, 0, out=11, acc=2.0)]
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.log"
        log.write_text("".join(lines))
        r = e2e.decompose(str(log), 2000)
        x = r["ranks"]["0"]
        assert (x["max_running"], x["max_queue_on_decode_lines"], x["full_batch"]) == (2, 1, 2)
        assert x["full_batch_line_gap_s"]["n"] == 5 and abs(x["full_batch_line_gap_s"]["median"] - 0.2) < 1e-6
        assert r["requeststats"]["acc_len"]["n_below_3"] == 1
        r["label"] = "p1"
        row = e2e.markdown([r], log_interval=10).splitlines()[2]
        assert row.startswith("| p1 | 1.70 | 2 | 0.40 = 4 x 100.0 ") and "| 20.0 |" in row
        bj = Path(d) / "benchmark.json"
        bj.write_text(json.dumps({"results": [
            {"phase": "w", "output_dir": "/o/w", "started_at": ts(T - 1), "finished_at": ts(T + 0.35)},
            {"phase": "m", "output_dir": "/o/m", "started_at": ts(T + 0.35), "finished_at": ts(T + 5)}]}))
        windows = e2e.phase_windows(str(bj))
        paths = e2e.write_slices([str(log)], windows, str(Path(d) / "slices"))
        assert [len(Path(p).read_text().splitlines()) for p in paths] == [4, 8]
        bj.write_text(json.dumps({"results": [{"phase": "w", "output_dir": "/o/w"}]}))
        try:
            e2e.phase_windows(str(bj))
        except SystemExit as exc:
            assert "started_at" in str(exc)
        else:
            raise AssertionError("phases without times must be refused")


def test_predict_terms_and_whatifs() -> None:
    e2e = load("e2e_decompose.py")
    rank = lambda p, d, i, gap: {"prefill_time_s": p, "decode_time_s": d, "idle_s": i,
                                 "decode_line_interval_median_s": gap}
    runs = {"a": [{"server_span_s": 12.0, "ranks": {"0": rank(8.0, 3.0, 1.0, 0.02)}, "requeststats": {"acc_len_mean": None}}],
            "b": [{"server_span_s": 9.0, "ranks": {"0": rank(6.0, 3.0, 0.0, 0.04)}, "requeststats": {"acc_len_mean": 2.0}}]}
    with tempfile.TemporaryDirectory() as d:
        csvs = {}
        for name, rate in (("a", 400.0), ("b", 500.0)):
            p = Path(d) / f"{name}-benchmark.json"
            p.write_text(json.dumps(prefill_benchmark(1000, {2: rate, 4: 1.0})))
            csvs[name] = str(p)
        res = e2e.predict(runs, csvs, [("a-fast", "a", 1.0)], {"b": 4.0}, requests=4, prompt_tokens=1000,
                          output_tokens=11, concurrency=2, interval=10)
    # 2 waves x 10 steps; a: step 2 ms, core 40 ms, tail 2.96 s; b: step 4 ms / acceptance 2, tail 2.96 s
    assert res["workload"]["decode_steps"] == 20
    p = res["predictions"][0]
    assert (p["prefill_s"], p["decode_s"], [round(x, 9) for x in p["residuals_s"]]) == (10.0, 0.02, [1.98])
    w = {x["name"]: round(x["span_s"], 6) for x in res["whatif"]}
    assert w["a as measured (model check)"] == 12.0
    assert w["a with b's end-to-end prefill time"] == 6.0 + 3.0 + 1.0
    assert w["b with a's end-to-end decode time"] == 6.0 + 3.0 + 0.0
    assert w["b with a's prefill-ladder rate"] == 4000 / 400 + 3.0
    assert w["a, decode at a-fast (1 ms per output token)"] == round(8.0 + 0.02 + (3.0 - 0.04) + 1.0, 6)
    assert w["b, acceptance 4 at the measured step time"] == round(6.0 + 20 / 4 * 0.004 + (3.0 - 0.04), 6)


def prefill_benchmark(prompt_tokens, rates):
    """The fields of a run_evalscope_matrix.py --prefill-summary benchmark.json that --prefill reads."""
    stat = lambda v: {"median": v, "min": v, "max": v}  # noqa: E731
    return {"prompt_tokens": prompt_tokens, "results": [],
            "prefill_summary": [{"concurrency": c, "repetitions": 3, "prompt_tokens_per_request": [prompt_tokens],
                                 "ttft_mean_s": stat(1.0), "ttft_max_s": stat(1.0), "last_first_token_s": stat(1.0),
                                 "prefill_tok_per_s": stat(r), "send_spread_s": stat(0.0)} for c, r in rates.items()]}


def test_prefill_rate_reads_matrix_prefill_summary_and_refuses_mismatches() -> None:
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "benchmark.json"
        p.write_text(json.dumps(prefill_benchmark(8000, {1: 9000.0, 4: 20000.0})))
        assert e2e.prefill_rate(str(p), 8000, 4) == 20000.0
        for args, word in (((8000, 2), "concurrency 2"), ((50000, 1), "prompt_tokens")):
            try:
                e2e.prefill_rate(str(p), *args)
            except SystemExit as exc:
                assert word in str(exc)
            else:
                raise AssertionError(f"{args} must be refused")
        p.write_text(json.dumps({"prompt_tokens": 8000, "results": []}))
        try:
            e2e.prefill_rate(str(p), 8000, 1)
        except SystemExit as exc:
            assert "--prefill-summary" in str(exc)
        else:
            raise AssertionError("a benchmark.json without prefill_summary must be refused")


def test_tpot_from_number_or_retained_kv_repetition_table() -> None:
    e2e = load("e2e_decompose.py")
    assert e2e.tpot_ms("12.5") == 12.5
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "repetitions.csv"  # columns as report_retained_kv.py --repetitions-only writes them
        p.write_text("point,batch_size,per_rank_batch,repetitions,decode_context_min,decode_context_max,"
                     "decode_tpot_ms_median,decode_tpot_ms_min,decode_tpot_ms_max\n"
                     "b64-c50000,64,16,3,50000,50500,7.25,7.0,7.5\n"
                     "b8-c50000,8,2,3,50000,50500,,,\n")
        assert e2e.tpot_ms(f"{p}@b64-c50000") == 7.25
        for point, word in (("b8-c50000", "no decode_tpot_ms_median"), ("b1", "no point")):
            try:
                e2e.tpot_ms(f"{p}@{point}")
            except SystemExit as exc:
                assert word in str(exc)
            else:
                raise AssertionError(point)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
