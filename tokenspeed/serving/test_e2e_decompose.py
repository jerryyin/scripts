#!/usr/bin/env python3
"""CPU tests for e2e_decompose.py, on small synthetic inputs (run with pytest or directly)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from log_fixtures import agentic_c1, batch, evalscope_db, finish, ts  # noqa: E402,F401


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


def test_benchmark_json_repeated_phase_slices_per_concurrency() -> None:
    """A matrix benchmark.json repeats a phase at each concurrency: one slice and one row per window."""
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    lines = []
    for w, (t0, n) in enumerate(((T, 1), (T + 10, 2))):  # window w: n requests of 2 chunks of 1000, 3 decode lines
        for i in range(2 * n):
            lines.append(batch(t0 + 0.1 * i, "Prefill", 0, new_seq=1, new_token=1000, running_req=n))
        for i in range(3):
            lines.append(batch(t0 + 0.5 + 0.2 * i, "Decode", 0, running_req=n, gen=50.0, acc=3.0))
        lines += [finish(t0 + 0.9, f"r{w}{k}", 2000, 0, out=11, acc=3.0) for k in range(n)]
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.log"
        log.write_text("".join(lines))
        bj = Path(d) / "benchmark.json"
        bj.write_text(json.dumps({"results": [
            {"phase": "measured-1", "concurrency": c, "output_dir": f"{d}/o{c}",
             "started_at": ts(t0 - 0.5), "finished_at": ts(t0 + 5)} for c, t0 in ((1, T), (2, T + 10))]}))
        assert [w[0] for w in e2e.phase_windows(str(bj))] == ["measured-1", "measured-1"]
        slices, out = Path(d) / "slices", Path(d) / "out.json"
        proc = subprocess.run([sys.executable, str(HERE / "e2e_decompose.py"), "--benchmark-json", str(bj),
                               "--server-log", str(log), "--slices-dir", str(slices), "--out-json", str(out),
                               "--prompt-tokens", "2000", "--markdown"], capture_output=True, text=True, check=True)
        assert sorted(p.name for p in slices.iterdir()) == ["measured-1_c1.log", "measured-1_c2.log"]
        assert [len((slices / f).read_text().splitlines()) for f in ("measured-1_c1.log", "measured-1_c2.log")] == [6, 9]
        assert [r["label"] for r in json.loads(out.read_text())] == ["measured-1/c1", "measured-1/c2"]
        rows = [x for x in proc.stdout.splitlines() if x.startswith("| measured-1")]
        assert [x.split(" | ")[0] for x in rows] == ["| measured-1/c1", "| measured-1/c2"]
        assert [x.split(" | ")[2] for x in rows] == ["1", "2"]  # requests per rank: each row its own window
        # a phase that does not repeat keeps its name next to repeated ones; a same-concurrency repeat is refused
        res = json.loads(bj.read_text())["results"]
        bj.write_text(json.dumps({"results": [dict(res[0], phase="warmup-1")] + res}))
        assert [w[0] for w in e2e.phase_windows(str(bj), per_concurrency=True)] == ["warmup-1", "measured-1/c1", "measured-1/c2"]
        bj.write_text(json.dumps({"results": [res[0], res[0]]}))
        try:
            e2e.phase_windows(str(bj), per_concurrency=True)
        except SystemExit as exc:
            assert "measured-1/c1" in str(exc)
        else:
            raise AssertionError("two windows of one phase and concurrency must be refused")


def test_markdown_without_request_prompt_length() -> None:
    # No finished request of >= 1000 prompt tokens: requests per rank is unknown (None), printed "-".
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    lines = [batch(T + 0.1 * i, "Prefill", 0, new_seq=1, new_token=1000, running_req=2) for i in range(4)]
    lines += [batch(T + 0.4 + 0.2 * i, "Decode", 0, running_req=2, gen=50.0, acc=3.0) for i in range(6)]
    lines += [finish(T + 1.6, "a", 500, 0, out=11, acc=3.0), finish(T + 1.7, "b", 500, 0, out=11, acc=2.0)]
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.log"
        log.write_text("".join(lines))
        r = e2e.decompose(str(log))
        assert r["request_prompt_tokens"] is None and r["ranks"]["0"]["requests_by_prefill_tokens"] is None
        r["label"] = "p1"
        row = e2e.markdown([r], log_interval=10).splitlines()[2]
        assert row.startswith("| p1 | 1.40 | - | 0.40 = 4 x 100.0 "), row


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


def test_window_finish_line_splits_decode_from_idle() -> None:
    """Agentic window mode on a single-rank log: a Finish line splits decode from idle."""
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    dec = lambda t, seq: batch(t, "Decode", 0, running_req=1, gen=400.0, acc=4.0).replace(
        "\n", f", avg_seq_len: {seq}\n")
    lines = [batch(T, "Prefill", 0, new_seq=1, new_token=300, cached_token=300000, running_req=1)]
    t = T + 0.01
    lines.append(dec(t, 300300))  # dispatched right after the prefill: the next gap holds the prefill
    t += 0.6; lines.append(dec(t, 300400))
    for i in range(10):  # 40 rounds of 10 ms per line
        t += 0.4; lines.append(dec(t, 300500 + i))
    t += 2.0; lines.append(dec(t, 300600))  # stalled interval
    lines.append(f"[ts] {ts(t + 0.1)}  INFO ts_control::output: Req: r Finish! Accept_num_tokens_avg: 4.0\n")
    lines.append(batch(t + 5.0, "Prefill", 0, new_seq=1, new_token=200, cached_token=300600, running_req=1))
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.log"
        log.write_text("".join(lines))
        r = e2e.decompose(str(log), window=(T - 1, T + 30))
        y = r["ranks"]["0"]
        assert abs(y["idle_between_waves_s"] - 4.9) < 1e-4  # Finish at t+0.1 to the next Prefill at t+5.0


def test_step_table_bins_long_gaps_and_post_prefill() -> None:
    """--step-table: clean Decode gaps give the step by batch/context, the gap right after a Prefill
    line is kept out of the bins, and a stalled clean gap is listed with its excess."""
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    dec = lambda t, seq: batch(t, "Decode", 0, running_req=1, gen=400.0, acc=4.0).replace(
        "\n", f", avg_seq_len: {seq}\n")
    lines = [batch(T, "Prefill", 0, new_seq=1, new_token=300, cached_token=300000, running_req=1)]
    t = T + 0.01
    lines.append(dec(t, 300300))  # dispatched right after the prefill: the next gap holds the prefill
    t += 0.6; lines.append(dec(t, 300400))
    for i in range(10):  # 40 rounds of 10 ms per line
        t += 0.4; lines.append(dec(t, 300500 + i))
    t += 2.0; lines.append(dec(t, 300600))  # stalled interval
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.log"
        log.write_text("".join(lines))
        x = e2e.step_table(str(log), 40, (T - 1, T + 30))["ranks"]["0"]
        row = x["clean_steps"]["bs1/256k-512k"]
        assert abs(row["step_ms_median"] - 10.0) < 1e-3 and row["accept_len_mean"] == 4.0
        assert abs(row["ms_per_token_per_request"] - 2.5) < 1e-3
        assert x["post_prefill_gaps"] == 1
        assert x["long_clean_gaps"]["n"] == 1 and abs(x["long_clean_gaps"]["excess_s"] - 1.6) < 1e-4


def test_window_finish_split_needs_every_running_request_finished() -> None:
    """Two requests running, one Finish line between two Decode lines: the other request keeps
    decoding, so the gap stays decode (no idle); after the second Finish the wait is idle."""
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    lines = [batch(T, "Prefill", 0, new_seq=2, new_token=2000, running_req=2),
             batch(T + 1.0, "Decode", 0, running_req=2, gen=400.0),
             f"[ts] {ts(T + 1.5)}  INFO ts_control::output: Req: a Finish! Accept_num_tokens_avg: 3.0\n",
             batch(T + 2.0, "Decode", 0, running_req=1, gen=200.0),
             f"[ts] {ts(T + 2.5)}  INFO ts_control::output: Req: b Finish! Accept_num_tokens_avg: 3.0\n",
             batch(T + 6.0, "Prefill", 0, new_seq=1, new_token=1000, running_req=1)]
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.log"
        log.write_text("".join(lines))
        y = e2e.decompose(str(log), window=(T, T + 6.0))["ranks"]["0"]
        assert abs(y["idle_between_waves_s"] - 3.5) < 1e-4  # only (T+2.5, T+6.0]


def test_aiperf_window_is_the_profiling_phase() -> None:
    e2e = load("e2e_decompose.py")
    recs = [{"metadata": {"benchmark_phase": "warmup", "request_start_ns": 1_000_000_000, "request_end_ns": 2_000_000_000}},
            {"metadata": {"benchmark_phase": "profiling", "request_start_ns": 5_000_000_000, "request_end_ns": 7_500_000_000}},
            {"metadata": {"benchmark_phase": "profiling", "request_start_ns": 6_000_000_000, "request_end_ns": 9_250_000_000}}]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "profile_export.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        assert e2e.aiperf_window(str(p)) == (5.0, 9.25)
        # as InferenceX counts records: no benchmark_phase is profiling, an errored record is dropped
        recs += [{"metadata": {"request_start_ns": 4_000_000_000, "request_end_ns": 9_000_000_000}},
                 {"error": {"code": 500}, "metadata": {"benchmark_phase": "profiling", "request_start_ns": 6_000_000_000,
                                                       "request_end_ns": 12_000_000_000}}]
        p.write_text("".join(json.dumps(r) + "\n" for r in recs) + "\n")
        assert e2e.aiperf_window(str(p)) == (4.0, 9.25)


def test_agentic_c1_window_places_short_turns_and_client_gaps() -> None:
    """AgentX at C=1 on one rank (Finish lines with Accept_num_tokens_avg only): with the AIPerf window, a
    short turn's Prefill -> next Prefill gap and a long turn's last Decode -> next Prefill gap end decode at
    the turn's Finish line, so prefill is the chunks, decode the rounds and the client gaps idle; the only
    loss is the last turn's rounds after its last Decode line (no RequestStats token counts for a tail)."""
    e2e = load("e2e_decompose.py")
    lines, recs, truth = agentic_c1(1_800_000_000.0)
    with tempfile.TemporaryDirectory() as d:
        log, pe = Path(d) / "server.log", Path(d) / "profile_export.jsonl"
        log.write_text("".join(lines))
        pe.write_text("".join(json.dumps(r) + "\n" for r in recs))
        win = e2e.aiperf_window(str(pe))
        assert win == truth["window"]
        y = e2e.decompose(str(log), window=win)["ranks"]["0"]
        tail = 10 * truth["decode_step_s"]  # last turn: 250 rounds, Decode lines every 40
        assert abs(y["prefill_time_s"] - truth["prefill_s"]) < 1e-4
        assert abs(y["decode_time_s"] - (truth["decode_s"] - tail)) < 1e-4
        assert abs(y["idle_between_waves_s"] - 33.5) < 1e-4  # the client gaps after every Finish
        assert y["prefill_stalls"]["n_gaps_gt_2x_median"] == 0 and y["chunk_time_s"]["n"] == 4
        fs = y["finish_split_gaps"]  # 6 turns before the last: 3 short (from their Prefill), 3 from a Decode line
        assert (fs["n"], fs["from_prefill"]) == (6, 3) and abs(fs["idle_s"] - 33.5) < 1e-4
        assert abs(fs["prefill_s"] - 3 * 0.125) < 1e-4 and abs(fs["decode_s"] - (20 + 35 + 10 + 3 * 10) * 0.01066) < 1e-4
        assert e2e.decompose(str(log))["ranks"]["0"]["finish_split_gaps"] is None  # start/end rules: no split
        st = e2e.step_table(str(log), 40, win)["ranks"]["0"]
        assert {k: (v["n"], round(v["step_ms_median"], 3)) for k, v in st["clean_steps"].items()} == {
            "bs1/256k-512k": (9, 10.66), "bs1/512k-1024k": (22, 10.66)}
        pool = st["clean_all"]
        assert pool["n"] == 31 and abs(pool["step_ms_median"] - 10.66) < 1e-3 and pool["accept_len_mean"] == 3.5
        assert abs(pool["ms_per_token_per_request"] - 10.66 / 3.5) < 1e-3
        md = e2e.step_markdown(e2e.step_table(str(log), 40, win)).splitlines()
        assert md[-1].startswith("| 0 | all | 31 | 10.66 [10.66, 10.66] | 3.50 | 3.05 |"), md[-1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
