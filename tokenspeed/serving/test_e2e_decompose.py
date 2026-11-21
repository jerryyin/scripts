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
    rank = lambda sizes, gap, acc: {"prefill_time_s": 1.0, "chunk_time_by_size_median_s": sizes,  # noqa: E731
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")


def _gap_bench(ttft, tpot, lat, out=11, phases=None, top_out=None):
    """A matrix benchmark.json with one warmup row and one measured row per repetition at C=1."""
    def summ(t, p, l):
        return {"Avg TTFT (ms)": t, "Avg TPOT (ms)": p, "Avg Latency (s)": l, "Avg Output Tokens": out}
    res = [{"phase": "warmup-1", "concurrency": 1, "summary": summ(1.0, 1.0, 1.0)}]
    phases = phases or [f"measured-{i + 1}" for i in range(len(ttft))]
    for ph, t, p, l in zip(phases, ttft, tpot, lat):
        res.append({"phase": ph, "concurrency": 1, "summary": summ(t, p, l)})
    return {"prompt_tokens": 100, "output_tokens": out if top_out is None else top_out, "results": res}


def test_gap_split_ttft_and_decode_parts() -> None:
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.json", Path(d) / "b.json"
        a.write_text(json.dumps(_gap_bench([300, 310, 320], [12, 12, 13], [0.43, 0.44, 0.45])))
        b.write_text(json.dumps(_gap_bench([100, 100, 100], [10, 10, 10], [0.2, 0.2, 0.2])))
        r = e2e.gap_split([("A", str(a)), ("B", str(b))])
        row = r["rows"][0]
        assert r["output_tokens"] == 11
        assert abs(row["ttft_part_ms"] - 210) < 1e-9 and abs(row["decode_part_ms"] - 20) < 1e-9
        assert row["larger_part"] == "ttft" and row["n_a"] == 3
        assert row["ttft_part_ms_paired_range"] == [200, 220]
        assert row["decode_part_ms_paired_range"] == [20, 30]
        assert abs(row["residual_ms"] - (240 - 230)) < 1e-9
        assert "| 1 |" in e2e.gap_split_markdown(r)


def test_gap_split_uses_file_output_length_and_pairs_by_repetition() -> None:
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.json", Path(d) / "b.json"
        # O = 501 from the file: decode part = 500 x dTPOT. Phases listed out of order, and
        # measured-10 must pair with measured-10 (not by lexical sort).
        a.write_text(json.dumps(_gap_bench([5, 7, 6], [2.0, 2.2, 2.1], [1.0, 1.0, 1.0], out=501,
                                           phases=["measured-2", "measured-10", "measured-1"])))
        b.write_text(json.dumps(_gap_bench([1, 1, 4], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0], out=501,
                                           phases=["measured-1", "measured-2", "measured-10"])))
        row = e2e.gap_split([("A", str(a)), ("B", str(b))])["rows"][0]
        assert row["paired_repetitions"] == [1, 2, 10]
        # pairs: rep1 6-1=5, rep2 5-1=4, rep10 7-4=3
        assert row["ttft_part_ms_paired_range"] == [3, 5]
        assert abs(row["decode_part_ms"] - 500 * 0.1) < 1e-9
        # a single unsuffixed "measured" phase (--repetitions 1) is repetition 1
        a.write_text(json.dumps(_gap_bench([5], [2.0], [1.0], phases=["measured"])))
        b.write_text(json.dumps(_gap_bench([1], [2.0], [1.0], phases=["measured"])))
        assert e2e.gap_split([("A", str(a)), ("B", str(b))])["rows"][0]["ttft_part_ms"] == 4
        # differing output lengths are refused, not split with a constant
        a.write_text(json.dumps(_gap_bench([5], [2.0], [1.0], out=500)))
        b.write_text(json.dumps(_gap_bench([1], [2.0], [1.0], out=11)))
        try:
            e2e.gap_split([("A", str(a)), ("B", str(b))])
            raise AssertionError("expected SystemExit")
        except SystemExit:
            pass


def test_gap_split_cli_writes_json() -> None:
    e2e = load("e2e_decompose.py")
    with tempfile.TemporaryDirectory() as d:
        a, b, out = Path(d) / "a.json", Path(d) / "b.json", Path(d) / "gap.json"
        a.write_text(json.dumps(_gap_bench([300], [12], [0.43])))
        b.write_text(json.dumps(_gap_bench([100], [10], [0.2])))
        argv = sys.argv
        sys.argv = ["e2e_decompose.py", "--gap-split", f"EP4={a}", "--gap-split", f"TP4={b}", "--out-json", str(out)]
        try:
            e2e.main()
        finally:
            sys.argv = argv
        r = json.loads(out.read_text())
        assert r["a"] == "EP4" and r["b"] == "TP4" and r["rows"][0]["ttft_part_ms"] == 200


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


def test_samples_unobserved_gpu_is_not_called_idle() -> None:
    """A GPU with no own KFD process in the samples (the sampler could not read it) is unobserved."""
    e2e = load("e2e_decompose.py")
    s = [_sample(10.0 + 0.5 * k, [100, 100, 100, 0], [32, 32, 32, 0]) for k in range(4)]
    for x in s:
        del x["kfd"]["103"]
    act = e2e.window_activity(s, 10.0, 12.0, e2e.own_kfd_pids(s))
    st = e2e.window_state(act)
    assert st["per_gpu"]["3"] == "unobserved" and st["state"] == "all-busy" and st["idle_gpus"] == []
