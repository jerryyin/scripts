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


# --- --stages, --tokens 0, distinct_local_experts (r0002-p2) ---

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


def test_stage_rows_and_decide(tmp_path) -> None:
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
    d = {x["candidate"]: x for x in petit_layer.decide(rows)}
    assert d["C1"]["build"] is True and d["C1"]["value_ms"] == pytest.approx(0.011)
    assert d["C2"]["build"] is False and d["C2"]["value_ms"] == pytest.approx(0.004)
    w13 = 10 * petit_layer.W13_EXPERT_BYTES / 6e12 * 1e3
    assert d["C5"]["value_ms"] == pytest.approx(0.018 - w13) and d["C5"]["build"] is (0.018 - w13 > 0.005)
    assert d["C6/C7"]["value_ms"] == pytest.approx(0.057 - 0.018) and d["C6/C7"]["build"] is True
    assert petit_layer.decide(rows, node_us=15)[3]["build"] is False   # 0.057 - 0.045 < 0.020
    assert petit_layer.main(["--decide", str(out), "--where", "mode=graph"]) == 0
    # a record without --stages or distinct_local_experts leaves the new columns empty
    old = tmp_path / "old.json"
    old.write_text(json.dumps(_record()))
    assert petit_layer.main(["--rows", str(tmp_path / "old.csv"), str(old)]) == 0
    r0 = next(csv.DictReader((tmp_path / "old.csv").open()))
    assert r0["s1_ms"] == "" and r0["distinct_local_experts"] == ""
    assert petit_layer.decide([dict(r0)])[0]["build"] is None


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
        net = {n: scale * (i + 1) * 1e-3 for i, n in enumerate(petit_layer.STAGE_NAMES)}
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
