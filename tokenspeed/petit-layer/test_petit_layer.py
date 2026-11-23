#!/usr/bin/env python3
"""Non-GPU unit tests for petit_layer.py (arguments, token list, limits, test-source extraction)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import petit_layer

RANGE_COMPARE = Path(__file__).resolve().parents[2] / "numerics" / "range_compare.py"


def test_parse_requires_the_profile_and_tokens() -> None:
    a = petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "any_profile", "--tokens", "1,6"])
    assert a.profile == "any_profile" and a.tokens == [1, 6] and a.mode == "both" and a.reps == 3
    assert a.routing == "test" and a.seed == 0 and a.active_ranks is None
    p = petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "1024",
                                "--active-ranks", "1"])
    assert p.tokens == [1024] and p.active_ranks == 1
    for missing in (["--profile", "x"], ["--tokens", "1"]):
        with pytest.raises(SystemExit):
            petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", *missing])


@pytest.mark.parametrize("argv", [
    ["--tokens", "1025"],        # above Petit's per-rank capacity
    ["--tokens", "-1"],
    ["--marker-us", "5"],        # needs --stages
    ["--reps", "2"],
    ["--warmup", "0"],
    ["--mode", "neither"],
    ["--active-ranks", "0"],
])
def test_parse_rejects(argv) -> None:
    with pytest.raises(SystemExit):
        petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "1", *argv])


def test_check_tokens_cites_capacity() -> None:
    with pytest.raises(ValueError, match="max_tokens_per_rank=1024"):
        petit_layer.check_tokens([6, 2048])


def test_capacity_raises_token_limit_and_sets_plan() -> None:
    base = ["--ts-src", "/ts", "--out", "o.json", "--profile", "x"]
    p = petit_layer.parse_args([*base, "--tokens", "1024,2048,3072,4096", "--capacity", "4096"])
    assert p.tokens == [1024, 2048, 3072, 4096] and p.capacity == 4096
    assert petit_layer.parse_args([*base, "--tokens", "1024"]).capacity == 1024
    for bad in (["--tokens", "4097", "--capacity", "4096"], ["--tokens", "2048"],
                ["--tokens", "1", "--capacity", "1000"], ["--tokens", "1", "--capacity", "0"]):
        with pytest.raises(SystemExit):
            petit_layer.parse_args([*base, *bad])
    with pytest.raises(ValueError, match="max_tokens_per_rank=4096"):
        petit_layer.check_tokens([4097], 4096)
    assert petit_layer.apply_capacity({"a": 1}, 1024) == {"a": 1}  # default: plan unchanged
    assert petit_layer.apply_capacity({"a": 1}, 4096) == {"a": 1, "gluon_petit_max_tokens_per_rank": 4096}


SAMPLE = '''
    for case in cases:
        def refresh(step):
            x.copy_(step)

        def run():
            return kernel(x)

        refresh(0)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=8e-3)
'''


def test_extract_closures_and_tolerance() -> None:
    text = petit_layer.extract_closures(SAMPLE)
    assert text.startswith("def refresh(step):") and "def run():" in text and "refresh(0)" not in text
    assert petit_layer.test_tolerance(SAMPLE) == {"atol": 2e-6, "rtol": 8e-3}


@pytest.mark.skipif(not os.environ.get("TS_SRC"), reason="needs $TS_SRC (TokenSpeed checkout)")
def test_extract_from_pinned_test() -> None:
    source = (Path(os.environ["TS_SRC"]) / petit_layer.TEST_REL).read_text()
    text = petit_layer.extract_closures(source)
    compile(text, "closures", "exec")
    assert "def refresh(step):" in text and "kernel_moe_apply(" in text
    assert petit_layer.test_tolerance(source) == {"atol": 2e-6, "rtol": 8e-3}


def _timing(v):
    one = {"median": v, "min": v, "max": v, "mean": v, "n": 3}
    return {"max_over_ranks_ms": one, "per_rank_ms": [one, one]}


def test_summary_table() -> None:
    record = {"results": [{
        "tokens_per_gpu": 6, "pass": True, "max_abs_err": 0.0, "max_rel_err": 0.0,
        "graph_equals_eager": True,
        "per_rank": [{"first_call": {"compiled": True}}, {"first_call": {"compiled": False}}],
        "timing_ms": {"eager": _timing(0.25), "graph": _timing(0.125)}}]}
    table = petit_layer.summary_table(record).splitlines()
    assert table[0].startswith("| tokens/GPU | candidate | launched | pass |") and "graph per-rank medians" in table[0]
    # results without candidate/launched (Petit's own choice, older JSON) read as "petit" and ""
    assert table[2] == ("| 6 | petit |  | PASS | 0 | 0 | True | 1/2 | 0.2500 [0.2500, 0.2500] | 0.2500 / 0.2500 "
                        "| 0.1250 [0.1250, 0.1250] | 0.1250 / 0.1250 |")


def _record(reps=(0.20, 0.21, 0.22)):
    one = lambda v: {"per_rank_median_ms": [v, v - 0.01, v, v], "max_over_ranks_median_ms": v,
                     "max_over_ranks_min_ms": v - 0.005, "max_over_ranks_max_ms": v + 0.01}
    timing = {"max_over_ranks_ms": {}, "per_rank_ms": [], "per_repetition": [one(v) for v in reps]}
    return {"args": {"routing": "test", "seed": 0, "iters": 50}, "world_size": 4,
            "profile": {"name": "dsv41_flash"},
            "tokenspeed": {"commit": "e" * 40, "test": {"sha256": "a" * 64}},
            # results written before --active-ranks existed have no active_ranks field
            "results": [{"tokens_per_gpu": 6, "pass": True, "timing_ms": {"graph": timing, "eager": timing}}]}


def test_rows_one_per_repetition_and_range_compare_reads_them(tmp_path) -> None:
    src = tmp_path / "petit.json"
    src.write_text(json.dumps(_record()))
    out = tmp_path / "rows.csv"
    assert petit_layer.main(["--rows", str(out), str(src)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6 and {r["mode"] for r in rows} == {"graph", "eager"}
    assert [r["repetition"] for r in rows if r["mode"] == "graph"] == ["0", "1", "2"]
    assert rows[0]["active_ranks"] == "4" and rows[0]["tokens_per_gpu"] == "6"
    assert rows[0]["candidate"] == "petit" and rows[0]["launched"] == "" and rows[0]["layers"] == "1"
    summary = subprocess.run(
        [sys.executable, str(RANGE_COMPARE), "summarize", str(out), "--key", "profile,mode,tokens_per_gpu",
         "--metric", "max_over_ranks_median_ms", "--where", "mode=graph"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    head = summary[0].split(",")
    row = dict(zip(head, summary[1].split(",")))
    assert row["max_over_ranks_median_ms_n"] == "3"
    assert float(row["max_over_ranks_median_ms_median"]) == 0.21
    assert (float(row["max_over_ranks_median_ms_min"]), float(row["max_over_ranks_median_ms_max"])) == (0.20, 0.22)


def test_parse_candidates_layers_weights_and_skew() -> None:
    base = ["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "256"]
    a = petit_layer.parse_args(base)
    assert a.candidates is None and a.layers == 1 and a.weights == "test" and a.skew == 0.5
    f = petit_layer.parse_args(base + ["--candidates", "cta192/m32,cta56/m64w8", "--layers", "8",
                                       "--weights", "random", "--routing", "skew", "--skew", "1.0"])
    assert f.candidates == "cta192/m32,cta56/m64w8" and f.layers == 8 and f.routing == "skew"
    for bad in (["--candidates", "cta57/m32"], ["--candidates", "cta56/m48"], ["--layers", "0"],
                ["--weights", "dense"]):
        with pytest.raises(SystemExit):
            petit_layer.parse_args(base + bad)


def test_candidate_names_and_expected_launch() -> None:
    assert petit_layer.candidate_names(None, 384) is None
    flash = petit_layer.candidate_names("all", 384)
    assert len(flash) == 12 and flash[0] == "cta56/m32" and "cta192/m64w8" in flash
    # with 128 experts cta192 is cta128 (kProducerBlocks = min(blocks, experts)), so 'all' drops it
    dspark = petit_layer.candidate_names("all", 128)
    assert len(dspark) == 9 and not any(n.startswith("cta192") for n in dspark)
    with pytest.raises(ValueError, match="unknown candidates"):
        petit_layer.candidate_names("cta192/m32", 128)
    assert petit_layer.expected_launch("cta192/m64w4", 384) == (192, True, False)
    assert petit_layer.expected_launch("cta192/m32", 128) == (128, False, False)


def test_fit_line_and_split_chunk(tmp_path) -> None:
    rows = tmp_path / "rows.csv"
    fields = ["profile", "routing", "mode", "active_ranks", "candidate", "tokens_per_gpu", "repetition",
              "max_over_ranks_median_ms"]
    with rows.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n in (128, 256, 512, 1024):
            for k, d in enumerate((-0.01, 0.0, 0.01)):  # t = 0.3 + 0.0004 n, median exact
                w.writerow({"profile": "dsv41_flash", "routing": "test", "mode": "eager", "active_ranks": 1,
                            "candidate": "petit", "tokens_per_gpu": n, "repetition": k,
                            "max_over_ranks_median_ms": 0.3 + 0.0004 * n + d})
    text = petit_layer.fit_table(list(csv.DictReader(rows.open())), layers=40, chunks=(4096, 8192, 1536))
    assert "fit t = 0.3000 ms + 0.4000 us x rows" in text
    assert "| 1024 | 0.7096 [0.6996, 0.7196] | 3 | 0.7096 | +0.0000 | 42% | 28.38 |" in text
    assert "| 8192 | 8 | 227.07 | 8.00x | 27.72 |" in text  # 40 x 8 x 0.7096
    assert "| 1536 | 2 | 48.58 | 1.71x |" in text  # 40 x (0.7096 + 0.3 + 0.0004 x 512)
    assert petit_layer.main(["--fit", str(rows), "--fit-chunks", "4096", "--where", "routing=test"]) == 0


# --- --stages, --tokens 0, distinct_local_experts ---

def test_parse_stages_zero_tokens_and_marker_cost() -> None:
    base = ["--ts-src", "/ts", "--out", "o.json", "--profile", "x"]
    a = petit_layer.parse_args(base + ["--tokens", "0,1,2,6"])
    assert a.tokens == [0, 1, 2, 6] and a.stages is False and a.marker_us is None
    b = petit_layer.parse_args(base + ["--tokens", "0", "--stages", "--marker-us", "5.011"])
    assert b.stages and b.marker_us == 5.011
    with pytest.raises(SystemExit):
        petit_layer.parse_args(base + ["--tokens", "1", "--stages", "--marker-us", "-1"])
    assert petit_layer.check_tokens([0, 1024]) == [0, 1024]


def test_distinct_local_experts() -> None:
    load = [0] * 16
    for e in (0, 0, 3, 5, 12, 15, 15):
        load[e] += 1
    assert petit_layer.distinct_local_experts(load, 4) == [2, 1, 0, 2]
    assert petit_layer.distinct_local_experts([0] * 384, 4) == [0, 0, 0, 0]


def _points(layer, t0, d=(0.010, 0.050, 0.020, 0.016, 0.005)):
    t = [t0]
    for v in d:
        t.append(t[-1] + v)
    return [f"{layer}:{m}" for m in petit_layer.STAGE_MARKS], t


def test_stage_split_per_layer_and_mean() -> None:
    n0, t0 = _points(0, 1.0)
    n1, t1 = _points(1, 2.0, (0.012, 0.054, 0.024, 0.018, 0.007))
    per = petit_layer.stage_split(n0 + n1, t0 + t1, 2)
    assert per[0] == pytest.approx({"prep": 0.010, "s1": 0.050, "s2": 0.020, "comb": 0.016, "post": 0.005})
    mean = petit_layer.mean_stages(per)
    assert mean == pytest.approx({"prep": 0.011, "s1": 0.052, "s2": 0.022, "comb": 0.017, "post": 0.006})
    with pytest.raises(ValueError, match="not recorded"):
        petit_layer.stage_split(n0[:-1], t0[:-1], 1)
    with pytest.raises(ValueError, match="twice"):
        petit_layer.stage_split(n0 + n0, t0 + t0, 1)
    # names without a layer prefix (layer 0)
    bare = [m for m in petit_layer.STAGE_MARKS]
    assert petit_layer.stage_split(bare, t0, 1) == per[:1]


def test_stage_closure_arithmetic() -> None:
    raw = {"prep": 0.016, "s1": 0.056, "s2": 0.026, "comb": 0.022, "post": 0.006}
    # 6 timing points: marked 0.140 vs unmarked 0.104 -> 6 us each from this run
    c = petit_layer.stage_closure(raw, 0.104, 0.140)
    assert c["marker_cost_run_ms"] == pytest.approx(0.006) and c["marker_cost_ms"] == pytest.approx(0.006)
    assert c["marker_cost_source"] == "run"
    assert c["stage_net_ms"] == pytest.approx({"prep": 0.010, "s1": 0.050, "s2": 0.020, "comb": 0.016, "post": 0.0})
    assert c["stage_sum_raw_ms"] == pytest.approx(0.126) and c["stage_sum_net_ms"] == pytest.approx(0.096)
    assert c["closure_gap_ms"] == pytest.approx(0.008) and c["closure_ratio"] == pytest.approx(0.096 / 0.104)
    g = petit_layer.stage_closure(raw, 0.104, 0.140, marker_ms=0.005011)
    assert g["marker_cost_source"] == "--marker-us" and g["stage_net_ms"]["s1"] == pytest.approx(0.050989)


def _rep(values):
    return {"per_rank_median_ms": list(values), "max_over_ranks_median_ms": max(values),
            "max_over_ranks_min_ms": max(values), "max_over_ranks_max_ms": max(values)}


def test_stage_rows(tmp_path) -> None:
    def result(tokens, active, s1, s2, comb, prep, experts):
        raw = {"prep": prep + 0.006, "s1": s1 + 0.006, "s2": s2 + 0.006, "comb": comb + 0.006, "post": 0.006}
        stages = {s: _rep([v, v - 0.001, v, v]) for s, v in raw.items()}
        unmarked, marked = _rep([0.1] * 4), _rep([0.136] * 4)
        clos = [petit_layer.repetition_closure(stages, unmarked, marked, None)] * 3
        timing = {"max_over_ranks_ms": {}, "per_rank_ms": [], "per_repetition": [unmarked] * 3}
        return {"tokens_per_gpu": tokens, "active_ranks": active, "pass": True,
                "routing": {"distinct_local_experts": experts},
                "timing_ms": {"graph": timing},
                "stages_ms": {"graph": {"closure_per_repetition": clos}}}
    record = {"args": {"routing": "test", "seed": 0, "iters": 50, "layers": 40}, "world_size": 4,
              "profile": {"name": "dsv41_flash"},
              "tokenspeed": {"commit": "e" * 40, "test": {"sha256": "a" * 64}},
              "results": [result(0, 4, 0.040, 0.004, 0.017, 0.0, [0, 0, 0, 0]),
                          result(6, 1, 0.058, 0.018, 0.017, 0.011, [9, 8, 10, 9])]}
    src = tmp_path / "petit.json"
    src.write_text(json.dumps(record))
    out = tmp_path / "rows.csv"
    assert petit_layer.main(["--rows", str(out), str(src)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6
    b1 = [r for r in rows if r["tokens_per_gpu"] == "6"][0]
    assert b1["distinct_local_experts"] == "9 8 10 9" and b1["distinct_local_experts_max"] == "10"
    assert float(b1["marker_cost_ms"]) == pytest.approx(0.006)
    assert float(b1["s1_net_ms"]) == pytest.approx(0.058) and float(b1["post_net_ms"]) == pytest.approx(0.0)
    assert float(b1["stage_sum_net_ms"]) == pytest.approx(0.104)
    # a record without --stages or distinct_local_experts leaves the new columns empty
    old = tmp_path / "old.json"
    old.write_text(json.dumps(_record()))
    assert petit_layer.main(["--rows", str(tmp_path / "old.csv"), str(old)]) == 0
    r0 = next(csv.DictReader((tmp_path / "old.csv").open()))
    assert r0["s1_ms"] == "" and r0["distinct_local_experts"] == ""


def test_marked_launch_records_around_the_launch_and_passes_warmup() -> None:
    calls = []

    class Kernel:
        warmup = "warmup-fn"

        def __getitem__(self, grid):
            return lambda *a, **k: calls.append(("launch", grid, a))
    w = petit_layer._MarkedLaunch(Kernel(), "petit.comb", "petit.comb_end")
    assert w.warmup == "warmup-fn"
    w[(128,)](1, 2)   # hook off: launch only
    petit_layer.Stages.mark = lambda name: calls.append(("mark", name))
    try:
        w[(128,)](3)
    finally:
        petit_layer.Stages.mark = None
    assert calls == [("launch", (128,), (1, 2)), ("mark", "petit.comb"), ("launch", (128,), (3,)),
                     ("mark", "petit.comb_end")]


def test_step_timing_source_from_the_owner_diff() -> None:
    src = petit_layer.step_timing_source(petit_layer.STEP_TIMING_DIFF.read_text())
    assert src.startswith('"""TEST-ONLY decode-step component timing')
    for fn in ("def mark(", "def capture_begin(", "def capture_end(", "def before_replay(", "def after_replay(",
               "def _harvest(", "external=True"):
        assert fn in src
    compile(src, "step_timing.py", "exec")
    with pytest.raises(RuntimeError):
        petit_layer.step_timing_source("diff --git a/x b/x\n+x\n")


def test_load_step_timing_keeps_records_in_memory(tmp_path) -> None:
    pytest.importorskip("torch")
    before = os.environ.get("TS_STEP_TIMING")
    st = petit_layer.load_step_timing(str(tmp_path / "st"))
    assert st.ENABLED and st._EVERY == 1 and os.environ.get("TS_STEP_TIMING") == before
    st._write({"type": "seq", "names": []})
    assert st.RECORDS == [{"type": "seq", "names": []}] and not (tmp_path / "st").exists()
    assert len(st.SOURCE_SHA256) == 64


def test_stage_table_per_rank_and_max() -> None:
    def cell(scale):
        net = {n: scale * (i + 1) * 1e-3 for i, n in enumerate(petit_layer.STAGES)}
        return {"stage_net_ms": net, "stage_sum_net_ms": sum(net.values()), "unmarked_call_ms": 0.016,
                "closure_ratio": sum(net.values()) / 0.016}
    reps = [{"per_rank": [cell(1.0), cell(2.0)], "max_over_ranks": cell(2.0)},
            {"per_rank": [cell(1.0), cell(2.0)], "max_over_ranks": cell(2.0)}]
    record = {"results": [{"tokens_per_gpu": 6, "active_ranks": 1, "candidate": "petit",
                           "routing": {"distinct_local_experts": [9, 10]},
                           "stages_ms": {"graph": {"closure_per_repetition": reps}}}]}
    lines = petit_layer.stage_table(record).splitlines()
    assert len(lines) == 2 + 3  # header, rule, ranks 0 and 1, max
    assert lines[2].startswith("| graph | 6 | 1 | petit | 9 10 | 0 | 1.0 [1.0, 1.0] | 2.0 [2.0, 2.0] |")
    assert "| max | 2.0 [2.0, 2.0] | 4.0 [4.0, 4.0] |" in lines[4]
    assert lines[4].endswith("| 30.0 [30.0, 30.0] | 16.0 [16.0, 16.0] | 1.875 |")
    assert petit_layer.stage_table({"results": [{"tokens_per_gpu": 1, "stages_ms": {}}]}).count("\n") == 1


def test_petit_switches_recorded_and_labelled():
    env = {"TOKENSPEED_PETIT_FUSE_PREP": "1", "TOKENSPEED_PETIT_STAGE1_N128": "0", "HIP_VISIBLE_DEVICES": "0"}
    switches = petit_layer.petit_switches(env)
    assert switches == {"TOKENSPEED_PETIT_FUSE_PREP": "1", "TOKENSPEED_PETIT_STAGE1_N128": "0"}
    assert petit_layer.switches_label(switches) == "FUSE_PREP=1;STAGE1_N128=0"
    assert petit_layer.switches_label({}) == "off"
    assert "petit_switches" in petit_layer.ROW_FIELDS


def test_project_step_against_switch_off():
    def row(profile, act, tok, sw, ms):
        return {"profile": profile, "mode": "graph", "active_ranks": act, "tokens_per_gpu": tok,
                "candidate": "petit", "petit_switches": sw, "max_over_ranks_median_ms": ms}
    rows = [row("dsv41_flash", "1", "6", "off", 0.0912), row("dsv41_flash", "1", "6", "off", 0.0914),
            row("dsv41_flash", "1", "6", "off", 0.0910), row("dsv41_flash", "1", "6", "FUSE_PREP=1", 0.0862),
            row("dsv41_flash_dspark", "4", "6", "off", 0.1123), row("dsv41_flash_dspark", "4", "6", "FUSE_PREP=1", 0.1083)]
    res = petit_layer.project_step(rows)
    flash = [r for r in res if r["profile"] == "dsv41_flash"][0]
    assert flash["delta_us"] == -5.0 and flash["draft_delta_us"] == -4.0
    assert flash["step_delta_ms"] == round((40 * -5.0 + 3 * -4.0) / 1000, 3)
    assert [r for r in res if r["profile"] == "dsv41_flash_dspark"][0]["delta_us"] == -4.0


def test_post_mode_geometry_and_ops():
    assert petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "1"]).post == "none"
    assert petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "1",
                                   "--post", "copy-add"]).post == "copy-add"
    with pytest.raises(SystemExit):
        petit_layer.parse_args(["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "1", "--post", "x"])
    assert petit_layer.post_geometry("add", [6, 0, 0, 0], 0) == (6, 0)
    assert petit_layer.post_geometry("add", [6, 0, 0, 0], 2) == (0, 0)
    assert petit_layer.post_geometry("copy-add", [2, 2, 2, 0], 2) == (6, 4)
    assert petit_layer.post_geometry("copy-add", [2, 2, 2, 0], 3) == (6, 6)  # 0-row rank keeps its shared rows
    with pytest.raises(ValueError):
        petit_layer.post_geometry("none", [1], 0)
    assert "post" in petit_layer.ROW_FIELDS
    torch = pytest.importorskip("torch")
    routed = torch.randn(2, 8).to(torch.bfloat16)
    fn = lambda: routed  # noqa: E731
    assert petit_layer.with_post(fn, "none", None, None, 0, 2) is fn
    shared, out = torch.randn(2, 8).to(torch.bfloat16), torch.empty(2, 8, dtype=torch.bfloat16)
    assert torch.equal(petit_layer.with_post(fn, "add", shared, out, 0, 2)(), routed + shared)
    shared6, out6 = torch.randn(6, 8).to(torch.bfloat16), torch.empty(6, 8, dtype=torch.bfloat16)
    got = petit_layer.with_post(fn, "copy-add", shared6, out6, 4, 2)()
    assert torch.equal(got, shared6[4:6] + routed) and torch.equal(out6[:4], shared6[:4])
    zeros = torch.zeros(6, 8, dtype=torch.bfloat16)  # what the GPU path passes: post is exact
    assert torch.equal(petit_layer.with_post(fn, "copy-add", zeros, out6, 2, 2)(), routed)
    empty = lambda: routed[:0]  # noqa: E731
    assert petit_layer.with_post(empty, "copy-add", shared6, out6, 6, 0)().shape == (0, 8)
    assert torch.equal(out6, shared6)


def test_rows_carry_post_mode():
    record = {"args": {"post": "add"}, "profile": {"name": "p"}, "world_size": 4, "tokenspeed": {"test": {"sha256": "s"}},
              "results": [{"tokens_per_gpu": 1, "pass": True, "timing_ms": {"graph": {"per_repetition": [
                  {"max_over_ranks_median_ms": 1.0, "max_over_ranks_min_ms": 1.0, "max_over_ranks_max_ms": 1.0,
                   "per_rank_median_ms": [1.0]}]}}}]}
    rows = petit_layer.record_rows(record, "x.json")
    assert rows[0]["post"] == "add"
    record["args"] = {}
    assert petit_layer.record_rows(record, "x.json")[0]["post"] == "none"


def test_project_step_keeps_post_modes_apart():
    def row(post, sw, ms):
        return {"profile": "dsv41_flash", "mode": "graph", "active_ranks": "1", "tokens_per_gpu": "6",
                "candidate": "petit", "petit_switches": sw, "post": post, "max_over_ranks_median_ms": ms}
    rows = [row("none", "off", 0.090), row("none", "C3=1", 0.088), row("add", "off", 0.095), row("add", "C3=1", 0.089)]
    res = {r["post"]: r for r in petit_layer.project_step(rows)}
    assert res["none"]["delta_us"] == -2.0 and res["add"]["delta_us"] == -6.0
    assert res["add"]["baseline_us"] == 95.0


def test_stage2_tile_forcing_and_label():
    a = petit_layer.parse_args(["--ts-src", "/ts", "--profile", "dsv41_flash", "--tokens", "6", "--out", "x.json"])
    assert a.stage2_tile == "petit"
    a = petit_layer.parse_args(["--ts-src", "/ts", "--profile", "dsv41_flash", "--tokens", "6", "--out", "x.json",
                                "--stage2-tile", "n128"])
    assert a.stage2_tile == "n128"
    with pytest.raises(SystemExit):
        petit_layer.parse_args(["--ts-src", "/ts", "--profile", "dsv41_flash", "--tokens", "6", "--out", "x.json",
                                "--stage2-tile", "n64"])
    kw = {"reduce_routes": True}
    assert petit_layer.stage2_kwargs(kw, None, False) == kw  # Petit's own choice passes through
    assert petit_layer.stage2_kwargs({"stage2_n128": True}, None, True) == {"stage2_n128": True}
    assert petit_layer.stage2_kwargs({"stage2_n128": True, "n128": True}, "n256", True) == {"n128": True}
    assert petit_layer.stage2_kwargs(kw, "n128", True) == {"reduce_routes": True, "stage2_n128": True}
    assert kw == {"reduce_routes": True}  # not mutated
    with pytest.raises(RuntimeError, match="stage2_n128"):
        petit_layer.stage2_kwargs(kw, "n128", False)  # a TokenSpeed without stage2_n128
    assert petit_layer.expected_launch("cta56/m32", 384) == (56, False, False)
    assert petit_layer.expected_launch("cta56/m32", 384, "n256") == (56, False, False)
    assert petit_layer.expected_launch("cta56/m32", 384, "n128") == (56, False, False, "s2n128")
    assert petit_layer.run_label({}) == "off"
    assert petit_layer.run_label({}, "n128") == "stage2_tile=n128"
    sel = {"TOKENSPEED_PETIT_FUSE_PREP": "1"}
    assert petit_layer.run_label(sel, "petit") == "FUSE_PREP=1"
    assert petit_layer.run_label(sel, "n256") == "FUSE_PREP=1;stage2_tile=n256"
    assert "stage2_tile" in petit_layer.ROW_FIELDS


def test_rows_and_project_keep_forced_stage2_apart():
    def record(tile, ms):
        return {"profile": {"name": "dsv41_flash"}, "world_size": 4, "petit_switches": {},
                "tokenspeed": {"commit": "c", "test": {"sha256": "t"}},
                "args": {"routing": "test", "seed": 0, "layers": 40, "iters": 50, "stage2_tile": tile},
                "results": [{"tokens_per_gpu": 6, "active_ranks": 1, "candidate": "petit", "pass": True,
                             "timing_ms": {"graph": {"per_repetition": [
                                 {"max_over_ranks_median_ms": ms, "max_over_ranks_min_ms": ms,
                                  "max_over_ranks_max_ms": ms, "per_rank_median_ms": [ms]}] * 3}}}]}
    rows = petit_layer.record_rows(record("petit", 0.0900)) + petit_layer.record_rows(record("n128", 0.0880))
    assert {r["petit_switches"] for r in rows} == {"off", "stage2_tile=n128"}
    assert {r["stage2_tile"] for r in rows} == {"petit", "n128"}
    res = [r for r in petit_layer.project_step(rows) if r["profile"] == "dsv41_flash"]
    assert len(res) == 1 and res[0]["delta_us"] == -2.0


def test_fused_post_modes_and_label():
    import contextlib
    assert petit_layer.post_geometry("fused-add", [6, 0, 0, 0], 0) == (6, 0)
    assert petit_layer.post_geometry("fused-copy-add", [2, 2, 2, 0], 2) == (6, 4)
    assert petit_layer.run_label({}, "petit", "fused-add") == "post=fused"
    assert petit_layer.run_label({}, "petit", "add") == "off"
    assert petit_layer.run_label({"TOKENSPEED_PETIT_FUSE_PREP": "1"}, "n128", "fused-copy-add") == \
        "FUSE_PREP=1;stage2_tile=n128;post=fused"
    torch = pytest.importorskip("torch")
    routed = torch.randn(2, 8).to(torch.bfloat16)
    scopes = []

    @contextlib.contextmanager
    def combine_post(residual, out=None, ready=None):  # what Petit's combine does with the scope
        scopes.append((residual, out))
        yield
        out.copy_((residual.float() + routed.float()).to(torch.bfloat16))

    with pytest.raises(RuntimeError, match="petit_combine_post"):
        petit_layer.with_post(lambda: routed, "fused-add", None, None, 0, 2)
    shared, out = torch.randn(2, 8).to(torch.bfloat16), torch.empty(2, 8, dtype=torch.bfloat16)
    got = petit_layer.with_post(lambda: routed, "fused-add", shared, out, 0, 2, combine_post)()
    assert got is out and scopes[-1][0] is shared and scopes[-1][1] is out
    assert torch.equal(got, (shared.float() + routed.float()).to(torch.bfloat16))
    shared6, out6 = torch.randn(6, 8).to(torch.bfloat16), torch.empty(6, 8, dtype=torch.bfloat16)
    got = petit_layer.with_post(lambda: routed, "fused-copy-add", shared6, out6, 4, 2, combine_post)()
    assert got.data_ptr() == out6[4:6].data_ptr() and torch.equal(out6[:4], shared6[:4])
    assert scopes[-1][0].data_ptr() == scopes[-1][1].data_ptr()  # residual aliases out (in place)


def test_rows_label_fused_post_against_the_separate_post():
    def record(profile, active, post, ms):
        return {"profile": {"name": profile}, "world_size": 4, "petit_switches": {},
                "tokenspeed": {"commit": "c", "test": {"sha256": "t"}},
                "args": {"routing": "test", "seed": 0, "layers": 40, "iters": 50, "post": post},
                "results": [{"tokens_per_gpu": 6, "active_ranks": active, "candidate": "petit", "pass": True,
                             "timing_ms": {"graph": {"per_repetition": [
                                 {"max_over_ranks_median_ms": ms, "max_over_ranks_min_ms": ms,
                                  "max_over_ranks_max_ms": ms, "per_rank_median_ms": [ms]}] * 3}}}]}
    rows = petit_layer.record_rows(record("dsv41_flash", 1, "add", 0.0954)) + \
        petit_layer.record_rows(record("dsv41_flash", 1, "fused-add", 0.0940))
    assert {r["post"] for r in rows} == {"add"}
    assert {r["petit_switches"] for r in rows} == {"off", "post=fused"}
    # the draft MoE run the same way: its fused-post change enters the step at the draft's post op
    rows += petit_layer.record_rows(record("dsv41_flash_dspark", 4, "add", 0.1120)) + \
        petit_layer.record_rows(record("dsv41_flash_dspark", 4, "fused-add", 0.1100))
    res = [r for r in petit_layer.project_step(rows) if r["profile"] == "dsv41_flash"]
    assert len(res) == 1 and res[0]["delta_us"] == -1.4
    assert res[0]["draft_delta_us"] == -2.0
    assert res[0]["step_delta_ms"] == round((40 * -1.4 + 3 * -2.0) / 1000, 3)


def test_stage_marks_wrap_the_c3_combine_when_present(monkeypatch):
    import types
    for with_c3 in (False, True):
        mod = types.ModuleType(f"fake_mega_moe_{with_c3}")

        class Adapter:
            pass
        Adapter.__module__ = mod.__name__
        mod.MegaMoESolutionAdapter = Adapter
        for name in ("MegaMoEStage1", "MegaMoEStage2", "MegaMoECombine") + (("MegaMoECombineResidual",) if with_c3 else ()):
            setattr(mod, name, object())
        pkg = types.ModuleType("lib.moe.rocm.mega_moe")
        pkg.MegaMoESolutionAdapter = Adapter
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
        for parent in ("lib", "lib.moe", "lib.moe.rocm"):
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
        monkeypatch.setitem(sys.modules, "lib.moe.rocm.mega_moe", pkg)
        monkeypatch.setattr(sys.modules["lib.moe.rocm"], "mega_moe", pkg, raising=False)
        monkeypatch.setattr(sys.modules["lib.moe"], "rocm", sys.modules["lib.moe.rocm"], raising=False)
        monkeypatch.setattr(sys.modules["lib"], "moe", sys.modules["lib.moe"], raising=False)
        petit_layer.install_stage_marks()
        assert isinstance(mod.MegaMoECombine, petit_layer._MarkedLaunch)
        assert hasattr(mod, "MegaMoECombineResidual") == with_c3
        if with_c3:
            assert isinstance(mod.MegaMoECombineResidual, petit_layer._MarkedLaunch)


# --- --hops / --hop-table (TOKENSPEED_PETIT_TRACE): synthetic traces, no GPU ---

import numpy as np  # noqa: E402

import petit_hops  # noqa: E402

THETA0, THETA1, X_TICKS, TURN = 1000.0, 1100.0, 150.0, 2.0  # rank 1's clock offset before/after; one-way hop


def _theta(t0):
    """Rank 1 minus rank 0 ticks at rank-0 time t0 (linear between the two ping medians)."""
    tb, ta = 10000 + 31.5 * 10, 90000 + 31.5 * 10
    return THETA0 + (t0 - tb) / (ta - tb) * (THETA1 - THETA0)


def _synthetic(ws=2, slots=(5, 6)):
    rings = [np.zeros((8, 3, 256, 16), dtype="<u8") for _ in range(ws)]
    pings = [np.zeros((2, 8, 64, 2), dtype="<u8") for _ in range(ws)]
    for phase, start in ((0, 10000), (1, 90000)):
        for i in range(64):
            send = start + i * 10
            for r in range(1, ws):
                seen_r = send + X_TICKS + _theta(send + X_TICKS)
                pings[0][phase, r, i] = (send, send + 2 * X_TICKS + TURN)
                pings[r][phase, r, i] = (round(seen_r), round(seen_r + TURN))
    for call, slot in enumerate(slots):
        t = 20000 + call * 10000
        for r in range(ws):
            lag = 50 * r  # rank r enters 0.5 us later
            def put(kernel, row, site, t0):
                rings[r][slot, kernel, row, site] = round(t0 + (_theta(t0) if r else 0))
            for row in range(4):
                put(0, row, 0, t + lag)
                put(0, row, 14, t + lag + 1)
                put(0, row, 12, t + lag + 2000)
            for site, d in ((1, 10), (2, 110), (3, 160), (4, 260), (5, 300)):
                put(0, 0, site, t + lag + d)
            put(0, 1, 2, t + lag + 310); put(0, 1, 3, t + lag + 410); put(0, 1, 5, t + lag + 700)
            put(0, 2, 2, t + lag + 310); put(0, 2, 3, t + lag + 420)
            put(0, 3, 8, t + lag + 320); put(0, 3, 9, t + lag + 800)
            put(0, 3, 10, t + lag + 1800); put(0, 3, 11, t + lag + 1900)
            put(1, 0, 0, t + lag + 2300); put(1, 0, 2, t + lag + 2400); put(1, 0, 3, t + lag + 2900)
            for site, d in ((0, 3000), (1, 3010), (2, 3100), (4, 3300), (5, 3400)):
                put(2, 0, site, t + lag + d)
    return [ring.tobytes() + ping.tobytes() for ring, ping in zip(rings, pings)]


def test_hops_decode_layout_matches_tokenspeed_trace_offsets():
    buf = _synthetic()[0]
    assert len(buf) == petit_hops.TRACE_BYTES == 8 * 3 * 256 * 16 * 8 + 2 * 8 * 64 * 2 * 8
    ring, ping = petit_hops.decode(buf)
    words = np.frombuffer(buf, dtype="<u8")
    # TraceOffset(slot, kernel, row, site) = base + (((slot*3 + kernel)*256 + row)*16 + site)*8
    assert ring[5, 0, 0, 3] == words[((5 * 3 + 0) * 256 + 0) * 16 + 3] != 0
    assert ring[6, 2, 0, 5] == words[((6 * 3 + 2) * 256 + 0) * 16 + 5] != 0
    # TracePingStampOffset(phase, peer, it, which) after the ring
    assert ping[1, 1, 7, 1] == words[8 * 3 * 256 * 16 + ((1 * 8 + 1) * 64 + 7) * 2 + 1]
    with pytest.raises(ValueError):
        petit_hops.decode(buf[:-8])


def test_hops_ping_gives_x_and_offset():
    bufs = _synthetic()
    _, p0 = petit_hops.decode(bufs[0])
    _, p1 = petit_hops.decode(bufs[1])
    f = petit_hops.ping_fit(p0, p1, 1, 0)
    assert f["n"] == 64 and all(abs(x - X_TICKS) <= 1 for x in f["x_ticks"])
    assert abs(f["theta"] - _theta(f["t0"] + X_TICKS)) <= 1


def test_hops_offset_interpolation():
    off = {"before": (0.0, 1000.0), "after": (100000.0, 1100.0), "x_ticks": []}
    out = petit_hops.to_rank0(np.array([50000 + 1050.0, 0.0]), off)
    assert abs(out[0] - 50000) < 0.1 and np.isnan(out[1])
    same = petit_hops.to_rank0(np.array([7.0]), {"before": (0.0, 0.0), "after": (1.0, 0.0)})
    assert same[0] == 7.0


def test_hops_chain_reconstruction():
    h = petit_hops.analyze(_synthetic(), [2, 2])
    assert h["calls"] == 2 and h["slots"] == [5, 6] and h["ranks"] == 2
    med = {k: [None if a is None else round(a["median"], 2) for a in v] for k, v in h["chain_us"].items()}
    assert med["stamp_cost"] == [0.01, 0.01, 0.01]
    assert med["hop1_admit"][:2] == [1.0, 1.0] and med["s13_counts"][0] == 0.5
    assert med["hop2_countdone"][0] == 1.0 and med["s15_plan"][0] == 0.4
    assert med["hop3_planready"][0] == 1.05 and med["push_chain"][0] == 2.9
    assert med["hop4_payload"][0] == 4.8 and med["tile"][0] == 10.0 and med["epilogue"][0] == 1.0
    assert med["s1_to_s2"][0] == 3.0 and med["w2_tile"][0] == 5.0
    assert med["combine_barrier"][0] == 0.9 and med["hop5_epoch"][0] == 2.0 and med["reduce"][0] == 1.0
    assert abs(med["launch_skew"][2] - 0.5) <= 0.02  # rank 1's later entry, after clock alignment
    assert abs(med["push_to_payload"][2] - 3.9) <= 0.02  # rank 0's PlanReady seen -> rank 0's payload seen
    assert abs(med["call"][2] - 34.5) <= 0.02
    assert abs(h["x_us"]["median"] - 1.5) <= 0.01
    # site 3: the owner at 1.6 us and two producers at 4.1, 4.2 us
    assert h["sites_us_from_call_start"]["0"]["stage1"]["3"]["median"] == pytest.approx(4.1)
    json.dumps(h)  # the hops section is JSON as written
    table = petit_hops.hop_table(h)
    assert table.count("\n") == len(petit_hops.CHAIN) + 2 and "| hop2_countdone | 1.00 [1.00, 1.00]" in table


def test_hops_refuses_without_the_switch(monkeypatch):
    env = {"TOKENSPEED_PETIT_TRACE": "0"}
    with pytest.raises(SystemExit):
        petit_layer.hop_switch(env)
    env = {}
    petit_layer.hop_switch(env)
    assert env["TOKENSPEED_PETIT_TRACE"] == "1"
    with pytest.raises(SystemExit):
        petit_layer.hop_module(object())  # a TokenSpeed without the trace
    traced = type("M", (), {"kPetitTrace": True, "MegaMoETracePingRun": staticmethod(lambda *a: 0)})
    petit_layer.hop_module(traced)
    base = ["--ts-src", "/ts", "--out", "o.json", "--profile", "x", "--tokens", "1", "--hops"]
    with pytest.raises(SystemExit):
        petit_layer.parse_args(base)  # --mode both
    with pytest.raises(SystemExit):
        petit_layer.parse_args(base + ["--mode", "graph", "--candidates", "all"])
    a = petit_layer.parse_args(base + ["--mode", "graph", "--layers", "40"])
    assert a.hops and a.hop_clock_mhz == 100.0


def test_hop_table_cli(tmp_path, capsys):
    h = petit_hops.analyze(_synthetic(), [2, 2])
    record = {"profile": {"name": "dsv41_flash"}, "petit_switches": {"TOKENSPEED_PETIT_TRACE": "1"},
              "results": [{"tokens_per_gpu": 6, "active_ranks": 1, "hops": h}, {"tokens_per_gpu": 0}]}
    path = tmp_path / "h.json"
    path.write_text(json.dumps(record))
    assert petit_layer.main(["--hop-table", str(path)]) == 0
    out = capsys.readouterr().out
    assert "tokens/GPU=6" in out and "| tile | 10.00 [10.00, 10.00]" in out and out.count("###") == 1


def test_hops_drop_stale_stamps_of_an_older_call():
    ring = np.zeros((8, 3, 256, 16), dtype="<u8")
    ring[2, 0, :4, 0] = 900_000 + np.arange(4)       # the slot's latest call: CTAs 0-3 enter at ~900000
    ring[2, 0, :4, 12] = 902_000
    ring[2, 0, 7, 9] = 100_000                       # CTA 7's tile stamp from the call 8 calls earlier
    ring[2, 0, 3, 9] = 900_500                       # CTA 3's tile stamp from the latest call
    ring[2, 2, 0, 4] = 903_000
    out = petit_hops.drop_stale(ring)
    assert out[2, 0, 7, 9] == 0 and out[2, 0, 3, 9] == 900_500 and out[2, 2, 0, 4] == 903_000
    assert (out[2, 0, :4, 0] == ring[2, 0, :4, 0]).all() and ring[2, 0, 7, 9] == 100_000  # a copy


def test_hop_table_site_gap_row():
    """The site 7 -> 9 gap per rank from the site medians, blank where a rank has no tile."""
    import petit_hops
    sites = {"0": {"stage1": {"7": {"median": 13.0}, "9": {"median": 15.5}}},
             "1": {"stage1": {"7": {"median": 12.0}}}}
    hops = {"ranks": 2, "sites_us_from_call_start": sites}
    assert petit_hops.site_gap(hops, "stage1", 7, 9) == [2.5, None]
    hops.update({"chain_us": {k: [None, None, None] for k in petit_hops.CHAIN}, "x_us": None, "x_us_by_peer": [None]})
    row = [l for l in petit_hops.hop_table(hops).splitlines() if l.startswith("| site7_to_9 |")]
    assert row and row[0].startswith("| site7_to_9 | 2.50 |  | 2.50 |")


def test_hop_summary_columns():
    """--hop-summary puts each run's max-over-ranks medians side by side, with the site gap and X."""
    import petit_hops
    chain = {k: [None, None, {"median": 1.0, "min": 1.0, "max": 1.0}] for k in petit_hops.CHAIN}
    sites = {"0": {"stage1": {"7": {"median": 13.0}, "9": {"median": 15.5}}}}
    h = {"ranks": 2, "chain_us": chain, "x_us": {"median": 0.5}, "sites_us_from_call_start": sites}
    out = petit_hops.hop_summary([("a 6/1", h)]).splitlines()
    assert out[0] == "| interval | a 6/1 |"
    assert "| call | 1.00 |" in out and "| site7_to_9 | 2.50 |" in out and "| X | 0.50 |" in out
