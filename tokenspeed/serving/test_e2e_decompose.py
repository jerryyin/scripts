#!/usr/bin/env python3
"""CPU tests for e2e_decompose.py, on small synthetic inputs (run with pytest or directly)."""

from __future__ import annotations

import importlib.util
import json
import math
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
        bj.write_text(json.dumps({"results": [  # a matrix ladder: one phase, two concurrencies
            {"phase": "m", "concurrency": 1, "output_dir": "/o/m1", "started_at": ts(T - 1), "finished_at": ts(T + 0.35)},
            {"phase": "m", "concurrency": 2, "output_dir": "/o/m2", "started_at": ts(T + 0.35), "finished_at": ts(T + 5)}]}))
        windows = e2e.phase_windows(str(bj))
        assert [w[0] for w in windows] == ["m-c1", "m-c2"]
        paths = e2e.write_slices([str(log)], windows, str(Path(d) / "ladder"))
        assert [Path(p).name for p in paths] == ["m-c1.log", "m-c2.log"]
        assert [len(Path(p).read_text().splitlines()) for p in paths] == [4, 8]
        bj.write_text(json.dumps({"results": [{"phase": "w", "output_dir": "/o/w"}]}))
        try:
            e2e.phase_windows(str(bj))
        except SystemExit as exc:
            assert "started_at" in str(exc)
        else:
            raise AssertionError("phases without times must be refused")


def test_prefill_to_decode_gap_reports_stall_lines() -> None:
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    lines = []
    for r, stall in ((0, 0.0), (1, 19.0)):  # rank 1: 19 s between its last chunk and its first Decode line
        for i in range(4):
            lines.append(batch(T + 0.05 * i, "Prefill", r, new_seq=1, new_token=1000, running_req=1))
        for i in range(5):
            lines.append(batch(T + 0.2 + stall + 0.5 * i, "Decode", r, running_req=1, gen=80.0))
    lines += [finish(T + 22.0, "a", 4000, 0, out=11), finish(T + 22.1, "b", 4000, 0, out=11)]
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "server.log"
        log.write_text("".join(sorted(lines, key=lambda s: s.split()[1])))
        r = e2e.decompose(str(log), 4000)
        g0, g1 = r["ranks"]["0"]["prefill_to_decode_gap_s"], r["ranks"]["1"]["prefill_to_decode_gap_s"]
        assert g0["n"] == 1 and g0["stalls"] == [] and abs(g0["max"] - 0.05) < 1e-6
        assert abs(g1["max"] - 19.05) < 1e-6 and len(g1["stalls"]) == 1
        assert g1["max_prefill_line_utc"] == ts(T + 0.15) and g1["stalls"][0]["decode_line_utc"] == ts(T + 19.2)



def test_prefill_to_decode_gap_per_ladder_point_slice() -> None:
    """--benchmark-json slices a matrix ladder per (phase, concurrency); each slice carries its own record."""
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    lines = []
    for t, stall in ((T, 0.0), (T + 100.0, 19.0)):  # c1 at T without a stall, c2 at T + 100 s with a 19 s stall
        for i in range(4):
            lines.append(batch(t + 0.05 * i, "Prefill", 0, new_seq=1, new_token=1000, running_req=1))
        for i in range(5):
            lines.append(batch(t + 0.2 + stall + 0.5 * i, "Decode", 0, running_req=1, gen=80.0))
        lines.append(finish(t + stall + 3.0, f"r{t}", 4000, 0, out=11))
    with tempfile.TemporaryDirectory() as d:
        log, bj, out = Path(d) / "server.log", Path(d) / "benchmark.json", Path(d) / "out.json"
        log.write_text("".join(lines))
        bj.write_text(json.dumps({"results": [
            {"phase": "measured-1", "concurrency": c, "output_dir": str(Path(d) / f"c{c}"),
             "started_at": ts(t - 1), "finished_at": ts(t + 50)} for c, t in ((1, T), (2, T + 100.0))]}))
        argv = sys.argv
        sys.argv = ["e2e_decompose.py", "--benchmark-json", str(bj), "--server-log", str(log),
                    "--slices-dir", str(Path(d) / "slices"), "--out-json", str(out)]
        try:
            e2e.main()
        finally:
            sys.argv = argv
        runs = {r["label"]: r["ranks"]["0"]["prefill_to_decode_gap_s"] for r in json.loads(out.read_text())}
        assert sorted(runs) == ["measured-1-c1", "measured-1-c2"]
        assert runs["measured-1-c1"]["n"] == 1 and runs["measured-1-c1"]["stalls"] == []
        c2 = runs["measured-1-c2"]
        assert c2["n"] == 1 and abs(c2["max"] - 19.05) < 1e-6
        assert [x["decode_line_utc"] for x in c2["stalls"]] == [ts(T + 100.0 + 19.2)]

def test_fmt_none() -> None:
    e2e = load("e2e_decompose.py")
    assert (e2e.fmt(None, ".1f"), e2e.fmt(2.25, ".1f")) == ("-", "2.2")


def test_chunk_model_fit_and_waves() -> None:
    e2e = load("e2e_decompose.py")
    rank = lambda sizes, gap, acc: {"prefill_time_s": 1.0, "decode_time_s": 1.0,  # noqa: E731
                                    "chunk_time_by_size_median_s": sizes,
                                    "full_batch_line_gap_s": {"median": gap}, "full_batch_accept_len_line_mean": acc}
    runs = [{"label": "measured-1-c1", "ranks": {"0": rank({"1000": 0.03}, 0.4, 5.0)}},
            {"label": "measured-2-c1", "ranks": {"0": rank({"2000": 0.05}, 0.4, 5.0)}},
            {"label": "warmup-1-c1", "ranks": {"0": rank({"2000": 9.0}, 9.0, 1.0)}}]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "runs.json"
        path.write_text(json.dumps(runs))
        a, b, pts = e2e.fit_chunk_line(e2e.measured_runs([str(path)]))
        assert (round(a * 1e3, 6), round(b * 1e6, 6), pts) == (10.0, 20.0, {1000: 0.03, 2000: 0.05})
        # 2 prompts of 2500 tokens, 1000-token chunks, one group: chunks end at 30,60,90,120,(rem 1000)150 ms;
        # prompt 1 ends in chunk 3 (90 ms), prompt 2 in chunk 5 (150 ms)
        end, own = e2e.chunk_wave(0.01, 20e-6, 1000, 1, 2, 2500)
        assert (round(end, 6), round(own, 6)) == (0.15, 0.12)
        # two groups: one prompt each, 3 chunks: 2 x 30 + t(500) = 20 ms -> 80 ms
        assert [round(x, 6) for x in e2e.chunk_wave(0.01, 20e-6, 1000, 2, 2, 2500)] == [0.08, 0.08]
        res = e2e.chunk_model([("M", str(path))], [("S", "10:40")], [("base", "M:1000:1:M"), ("slow", "S:1000:1:M")],
                              [1], 2500, 11, 40)
        base, slow = res["rows"]
        assert round(base["decode_s"], 6) == round(10 / 5.0 * 0.01, 6)
        assert slow["tok_per_s_ratio_to_first"] < 1 and base["tok_per_s_ratio_to_first"] == 1
        assert "| base | M | 1000 | 1 | 1 |" in e2e.chunk_model_markdown(res)
        # spread: one point per size here, so the envelope is the line itself; fixed share 10 / 50 at 2000 tokens
        sp = res["models"]["M"]["spread"]
        assert [round(x, 6) for x in sp["a_ms_range"]] == [10.0, 10.0] and sp["fixed_share_at"] == 2000
        assert round(res["models"]["M"]["fixed_share"], 6) == 0.2
    # two repetitions per size: 1000 -> 30/40 ms, 2000 -> 50 ms; lines (10 ms, 20 us) and (30 ms, 10 us)
    sp = e2e.fit_chunk_line_spread([
        {"label": "measured-1-c1", "ranks": {"0": rank({"1000": 0.03, "2000": 0.05}, 0.4, 5.0)}},
        {"label": "measured-2-c1", "ranks": {"0": rank({"1000": 0.04}, 0.4, 5.0)}},
        {"label": "measured-2-c2", "ranks": {"0": rank({"2000": 0.05}, 0.4, 5.0)}}])
    assert [round(x, 6) for x in sp["a_ms_range"]] == [10.0, 30.0]
    assert [round(x, 6) for x in sp["b_us_per_token_range"]] == [10.0, 20.0]
    assert [round(x, 6) for x in sp["fixed_share_range"]] == [0.2, 0.6] and sp["combinations"] == 4
    assert e2e.run_repetition({"label": "measured"}) == 1 and e2e.run_repetition({"label": "measured-3-c16"}) == 3


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


def _gap_bench(ttft, tpot, out=11, phases=None):
    """A matrix benchmark.json with one warmup row and one measured row per repetition at C=1."""
    def summ(t, p):
        return {"Avg TTFT (ms)": t, "Avg TPOT (ms)": p, "Avg Latency (s)": (t + (out - 1) * p) / 1e3,
                "Avg Output Tokens": out, "Output Throughput (tok/s)": 100.0}
    res = [{"phase": "warmup-1", "concurrency": 1, "summary": summ(1.0, 1.0)}]
    phases = phases or [f"measured-{i + 1}" for i in range(len(ttft))]
    for ph, t, p in zip(phases, ttft, tpot):
        res.append({"phase": ph, "concurrency": 1, "summary": summ(t, p)})
    return {"prompt_tokens": 100, "output_tokens": out, "results": res}


def _split(e2e, d, a, b):
    (d / "a.json").write_text(json.dumps(a))
    (d / "b.json").write_text(json.dumps(b))
    return e2e.accept_split([("B", str(d / "b.json")), ("A", str(d / "a.json"))], None, None)


def test_accept_split_ttft_and_decode_parts_without_a_decomposition() -> None:
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        res = _split(e2e, Path(d), _gap_bench([300, 310, 320], [12, 12, 13]), _gap_bench([100, 100, 100], [10, 10, 10]))
    row = res["summary"][0]
    assert row["repetitions"] == 3 and row["split_incomplete"] == 3 and row["accept_part_ms"] is None
    assert row["ttft_part_ms"] == {"median": 210, "min": 200, "max": 220, "n": 3}
    assert row["decode_part_ms"] == {"median": 20, "min": 20, "max": 30, "n": 3}  # O - 1 = 10 x dTPOT
    assert row["latency_gap_ms"]["median"] == 230 and len(res["pairs"]) == 4  # the warmup pair is listed too


def test_accept_split_is_a_median_of_per_repetition_differences() -> None:
    """TTFT A = 1, 2, 10 and B = 0, 9, 10 at repetitions 1..3: the per-repetition differences 1, -7, 0 have
    median 0 (the difference of the medians, 2 - 9 = -7, mixes repetitions). measured-10 pairs with
    measured-10 whatever the order in the file."""
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        res = _split(e2e, Path(d), _gap_bench([10, 1, 2], [2.0] * 3, phases=["measured-10", "measured-1", "measured-2"]),
                     _gap_bench([0, 9, 10], [2.0] * 3, phases=["measured-1", "measured-2", "measured-10"]))
        assert res["summary"][0]["ttft_part_ms"] == {"median": 0, "min": -7, "max": 1, "n": 3}
        # differing output lengths give no parts
        diff = _split(e2e, Path(d), _gap_bench([5], [2.0], out=500), _gap_bench([1], [2.0], out=11))
    pair = diff["pairs"][1]
    assert pair["split_incomplete"] == "output lengths differ: 500 vs 11" and "decode_part_ms" not in pair


def test_accept_split_cli_without_a_decomposition_writes_json() -> None:
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        a, b, out = Path(d) / "a.json", Path(d) / "b.json", Path(d) / "gap.json"
        a.write_text(json.dumps(_gap_bench([300], [12])))
        b.write_text(json.dumps(_gap_bench([100], [10])))
        argv = sys.argv
        sys.argv = ["e2e_decompose.py", "--accept-split", f"TP4={b}", "--accept-split", f"EP4={a}", "--out-json", str(out)]
        try:
            e2e.main()
        finally:
            sys.argv = argv
        r = json.loads(out.read_text())
    assert r["reference"] == "TP4" and r["summary"][0]["ttft_part_ms"]["median"] == 200


def _sample(t, busy, cu, sdma=0):
    """One gpu_admission.py sample: 4 GPUs, one own KFD process per GPU (KFD PID 100+g, own PID 200+g)."""
    return {"t": t, "utc": ts(t), "busy": {str(g): busy[g] for g in range(4)},
            "kfd": {str(100 + g): {str(g): {"cu_occupancy": cu[g], "queues": 6, "sdma_us": sdma, "vram": 285 << 30}}
                    for g in range(4)},
            "own": {str(200 + g): {"comm": "tokenspeed::sch", "drm_vram": 289 << 30, "kfd_open": True, "pgid": 1}
                    for g in range(4)} | {"300": {"comm": "foreign", "drm_vram": 10 << 30, "kfd_open": True, "pgid": 2}}}


def test_samples_attach_gpu_activity_to_stall() -> None:
    """--samples: during a 19 s stall of rank 1, GPU 1 is idle and the other three spin at cu_occupancy 32."""
    e2e = load("e2e_decompose.py")
    T = 1_800_000_000.0
    lines = []
    for i in range(4):
        lines.append(batch(T + 0.05 * i, "Prefill", 1, new_seq=1, new_token=1000, running_req=1))
    for i in range(5):
        lines.append(batch(T + 0.2 + 19.0 + 0.5 * i, "Decode", 1, running_req=1, gen=80.0))
    lines.append(finish(T + 22.0, "a", 4000, 0, out=11))
    samples = []
    for k in range(-6, 48):  # every 0.5 s from T - 3 s to T + 23.5 s
        t = T + 0.5 * k
        if T + 0.15 < t < T + 19.2:  # stall: rank 1's GPU idle, its peers in a spinning exchange kernel
            samples.append(_sample(t, [100, 0, 100, 100], [32, 0, 32, 32], sdma=5))
        else:
            samples.append(_sample(t, [100, 100, 100, 100], [40, 60, 30, 50], sdma=5))
    samples.insert(10, samples[10])  # a duplicate (two files covering the same time) is dropped
    with tempfile.TemporaryDirectory() as d:
        log, sj, out = Path(d) / "server.log", Path(d) / "samples.jsonl", Path(d) / "out.json"
        log.write_text("".join(lines))
        sj.write_text("".join(json.dumps(s) + "\n" for s in samples))
        argv = sys.argv
        sys.argv = ["e2e_decompose.py", "--log", str(log), "--prompt-tokens", "4000", "--samples", str(sj),
                    "--out-json", str(out)]
        try:
            e2e.main()
        finally:
            sys.argv = argv
        st = json.loads(out.read_text())[0]["ranks"]["1"]["prefill_to_decode_gap_s"]["stalls"][0]
        ga = st["gpu_activity"]
        assert ga["own_kfd_pids"] == ["100", "101", "102", "103"]
        assert ga["before_prefill"]["state"] == "all-busy" and ga["after_decode"]["state"] == "all-busy"
        stall = ga["stall"]
        assert stall["state"] == "mixed" and stall["idle_gpus"] == ["1"] and stall["busy_gpus"] == ["0", "2", "3"]
        assert stall["gpus"]["1"]["cu_occupancy_max"] == 0 and stall["gpus"]["1"]["busy_mean"] == 0
        assert stall["gpus"]["0"]["cu_occupancy_mean"] == 32 and stall["gpus"]["0"]["sdma_us_delta"] == 0
        assert stall["n_samples"] == len([s for s in samples[:10] + samples[11:]
                                          if T + 0.15 + 1.0 <= s["t"] <= T + 19.2 - 1.0 - 0.5])
        assert ga["before_prefill"]["n_samples"] == 4  # T - 1.85 .. T + 0.15
        tl = ga["stall_timeline"]
        assert [x["gpus"]["1"] for x in tl] == [[True, 60], [False, 0], [True, 60]]
        assert tl[1]["utc"] == ts(T + 0.5) and tl[2]["utc"] == ts(T + 19.5) and tl[1]["gpus"]["0"] == [True, 32]


def test_point_table_busiest_rank_upper_median_round_and_p2d_max() -> None:
    e2e = load("e2e_decompose.py")

    def rank(tp, td, chunks, cmed, gap, p2d):
        return {"prefill_time_s": tp, "decode_time_s": td, "prefill_chunks": chunks,
                "chunk_time_s": {"median": cmed}, "full_batch_line_gap_s": {"median": gap},
                "prefill_to_decode_gap_s": {"max": p2d}}

    runs = [{"label": "measured-1-c4", "ranks": {
        "0": rank(1.0, 2.0, 13, 0.1, 0.8, 0.4), "1": rank(1.5, 2.0, 14, 0.11, 0.6, 2.5),
        "2": rank(1.0, 1.0, 12, 0.1, None, None), "3": rank(1.0, 1.0, 12, 0.1, 0.7, 0.3)}},
        {"label": "load", "ranks": {}}]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "runs.json"
        p.write_text(json.dumps(runs))
        rows = e2e.point_table([str(p)], 40)
    assert len(rows) == 1  # a run without ranks is skipped
    x = rows[0]
    assert (x["busiest_rank"], x["prefill_chunks"]) == ("1", 14)
    assert abs(x["chunks_x_median_s"] - 14 * 0.11) < 1e-12
    # ranks with a gap: 0.6, 0.7, 0.8 s / 40 -> upper median (element 3 // 2) = 0.7 / 40
    assert set(x["decode_round_s_by_rank"]) == {"0", "1", "3"}
    assert abs(x["decode_round_s_upper_median"] - 0.7 / 40) < 1e-12
    assert (x["prefill_to_decode_gap_max_s"], x["prefill_to_decode_gap_max_rank"]) == (2.5, "1")
    text = "\n".join(e2e.point_table_lines(rows))
    assert "| 1 | 14 x 110.0 = 1.54 | 17.5 (0:20.0, 1:15.0, 3:17.5) | 2.50 (1) |" in text


def _accept_bench(path, rows):
    """benchmark.json with one result per (phase, tok/s, TTFT ms, TPOT ms, EvalScope summary acceptance), O = 101."""
    results = [{"phase": ph, "concurrency": 1, "summary": {
        "Avg Output Tokens": 101.0, "Output Throughput (tok/s)": tps, "Avg TTFT (ms)": ttft,
        "Avg TPOT (ms)": tpot, "Avg Latency (s)": (ttft + 100 * tpot) / 1e3, "Avg Decoded Tok/Iter": acc}}
        for ph, tps, ttft, tpot, acc in rows]
    path.write_text(json.dumps({"results": results}))


def _accept_decomp(path, labels_gap):
    runs = [{"label": lab, "ranks": {"0": {"prefill_time_s": 1.0, "decode_time_s": 1.0, "prefill_chunks": 1,
                                           "chunk_time_s": {"median": 0.1},
                                           "full_batch_line_gap_s": {"median": gap},
                                           "prefill_to_decode_gap_s": {"max": 0.0}}}}
            for lab, gap in labels_gap]
    path.write_text(json.dumps(runs))


def _accept_requests(path, rows):
    """Per-request EvalScope records (phase, request id, acc_len) and the server's Accept lines (rid, value)."""
    path.with_suffix(".req").write_text("".join(json.dumps({"phase": ph, "concurrency": 1, "request_id": rid,
                                                             "acc_len": acc}) + "\n" for ph, rid, acc, _ in rows))
    path.with_suffix(".log").write_text("".join(
        f"[ts] {ts(1.8e9)}  INFO ts_control::output: Req: {rid} Finish! Accept_num_tokens_avg: {srv}\n"
        for _, rid, _, srv in rows if srv is not None))


def test_accept_split_parts_sum_to_the_gap() -> None:
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # REF: step 10 ms, A 5 -> 2 ms/token; X: step 12 ms, A 4 -> 3 ms/token, plus 0.1 ms/token other
        _accept_bench(d / "ref.json", [("measured-1", 300.0, 700.0, 2.0, 5.0), ("measured-2", 300.0, 700.0, 2.0, 5.0)])
        _accept_bench(d / "x.json", [("measured-1", 200.0, 800.0, 3.1, 4.0), ("measured-2", 250.0, 800.0, 2.4, 5.0)])
        _accept_decomp(d / "ref-d.json", [("measured-1", 0.4), ("measured-2", 0.4)])  # 0.4 s / 40 rounds = 10 ms
        _accept_decomp(d / "x-d.json", [("measured-1", 0.48), ("measured-2", 0.48)])
        _accept_requests(d / "ref", [("measured-1", "q1", 5.1, 5.0), ("measured-2", "q2", 5.0, 5.0)])
        _accept_requests(d / "x", [("measured-1", "r1", 4.2, 4.0), ("measured-2", "r2", 5.0, 5.0)])
        res = e2e.accept_split([("REF", str(d / "ref.json")), ("X", str(d / "x.json"))],
                               {"REF": str(d / "ref-d.json"), "X": str(d / "x-d.json")}, 40,
                               {k: [str(d / f"{k.lower()}.log")] for k in ("REF", "X")},
                               {k: [str(d / f"{k.lower()}.req")] for k in ("REF", "X")})
    p1 = next(p for p in res["pairs"] if p["phase"] == "measured-1")
    assert p1["acceptance_source"] == "server" and "split_incomplete" not in p1
    assert abs(p1["ttft_part_ms"] - 100.0) < 1e-9
    assert abs(p1["step_part_ms"] - 100 * 2 / 5) < 1e-9          # D (s_X - s_R) / A_R = 40 ms
    assert abs(p1["accept_part_ms"] - 100 * 12 * (1 / 4 - 1 / 5)) < 1e-9  # 60 ms
    assert abs(p1["other_part_ms"] - 100 * 0.1) < 1e-9
    parts = sum(p1[k] for k in ("ttft_part_ms", "step_part_ms", "accept_part_ms", "other_part_ms"))
    assert abs(parts - p1["latency_gap_ms"]) < 1e-9
    assert abs(p1["log_accept"] + p1["log_step"] - (math.log(4 / 5) - math.log(1.2))) < 1e-12
    x1 = next(p for p in res["points"] if p["label"] == "X" and p["phase"] == "measured-1")
    assert x1["acceptance_source"] == "server" and x1["acceptance"] == 4.0
    assert abs(x1["acceptance_evalscope_vs_server_max_abs_diff"] - 0.2) < 1e-9
    row = res["summary"][0]
    assert row["repetitions"] == 2 and row["split_incomplete"] == 0 and row["accept_part_ms"]["min"] == 0.0
    assert "acceptance share" in e2e.accept_split_markdown(res)
    ps = {r["label"]: r for r in res["point_summary"]}
    assert set(ps) == {"REF", "X"} and all(r["tok_per_s"]["n"] == r["repetitions"] for r in ps.values())
    assert "| step/A ms |" in e2e.accept_split_markdown(res)


def test_accept_split_uses_the_harmonic_mean_of_per_request_acceptance() -> None:
    """Decode time follows mean(1/A_i): REF requests at A = 3, 3 and X requests at A = 2, 4 have the same
    arithmetic mean 3, but X's TPOT = s x mean(1/A_i) = 20 x 0.375 = 7.5 ms against REF's 20 / 3 ms. With
    D = 499 and the same 20 ms step, X decodes 499 x 20 x (0.375 - 1/3) = 415.8 ms longer, all of it acceptance."""
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, tpot, accs in (("ref", 20 / 3, (3.0, 3.0)), ("x", 7.5, (2.0, 4.0))):
            res = {"phase": "measured-1", "concurrency": 1, "summary": {
                "Avg Output Tokens": 500.0, "Output Throughput (tok/s)": 100.0, "Avg TTFT (ms)": 50.0,
                "Avg TPOT (ms)": tpot, "Avg Latency (s)": (50 + 499 * tpot) / 1e3, "Avg Decoded Tok/Iter": 3.0}}
            (d / f"{name}.json").write_text(json.dumps({"results": [res]}))
            _accept_decomp(d / f"{name}-d.json", [("measured-1", 0.8)])  # 0.8 s / 40 rounds = 20 ms
            _accept_requests(d / name, [("measured-1", f"{name}{i}", a, a) for i, a in enumerate(accs)])
        res = e2e.accept_split([("REF", str(d / "ref.json")), ("X", str(d / "x.json"))],
                               {"REF": str(d / "ref-d.json"), "X": str(d / "x-d.json")}, 40,
                               {k: [str(d / f"{k.lower()}.log")] for k in ("REF", "X")},
                               {k: [str(d / f"{k.lower()}.req")] for k in ("REF", "X")})
    p = res["pairs"][0]
    assert abs(p["latency_gap_ms"] - 499 * (7.5 - 20 / 3)) < 1e-9
    assert abs(p["accept_part_ms"] - 499 * 20 * (0.375 - 1 / 3)) < 1e-9 and abs(p["accept_part_ms"] - 415.83) < 0.01
    assert abs(p["step_part_ms"]) < 1e-9 and abs(p["other_part_ms"]) < 1e-9 and p["ttft_part_ms"] == 0
    assert {x["label"]: x["acceptance"] for x in res["points"]} == {"REF": 3.0, "X": 8 / 3}


def test_accept_split_pair_needs_one_acceptance_source() -> None:
    """REF's requests all match server lines, X's do not: the pair has no acceptance split (the server counts
    verify steps, EvalScope stream chunks), and the summary counts it as incomplete."""
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _accept_bench(d / "ref.json", [("measured-1", 300.0, 700.0, 2.0, 5.0)])
        _accept_bench(d / "x.json", [("measured-1", 200.0, 800.0, 3.1, 4.0)])
        _accept_decomp(d / "ref-d.json", [("measured-1", 0.4)])
        _accept_decomp(d / "x-d.json", [("measured-1", 0.48)])
        _accept_requests(d / "ref", [("measured-1", "q1", 5.0, 5.0)])
        _accept_requests(d / "x", [("measured-1", "r1", 4.0, None)])
        res = e2e.accept_split([("REF", str(d / "ref.json")), ("X", str(d / "x.json"))],
                               {"REF": str(d / "ref-d.json"), "X": str(d / "x-d.json")}, 40,
                               {k: [str(d / f"{k.lower()}.log")] for k in ("REF", "X")},
                               {k: [str(d / f"{k.lower()}.req")] for k in ("REF", "X")})
        without = e2e.accept_split([("REF", str(d / "ref.json")), ("X", str(d / "x.json"))],
                                   {"REF": str(d / "ref-d.json"), "X": str(d / "x-d.json")}, 40)
    p = res["pairs"][0]
    assert p["split_incomplete"] == "acceptance sources differ: evalscope requests vs server"
    assert "accept_part_ms" not in p and res["summary"][0]["split_incomplete"] == 1
    assert res["summary"][0]["accept_part_ms"] is None and abs(p["ttft_part_ms"] - 100) < 1e-9
    # without per-request values the EvalScope summary (an arithmetic mean) is not used for the split
    assert without["pairs"][0]["split_incomplete"].startswith("no per-request acceptance")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
