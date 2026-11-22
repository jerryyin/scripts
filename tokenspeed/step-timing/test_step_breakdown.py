#!/usr/bin/env python3
"""CPU tests for step_breakdown.py on a hand-made analyzer output."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import step_breakdown as sb


def fixture(root: Path) -> tuple[Path, Path]:
    analysis, dump = root / "analysis", root / "dump"
    analysis.mkdir(); dump.mkdir()
    rows = ["rank,role,category,median_ms"]
    for rank, role, attn in ((0, "busy", 3.0), (1, "idle", 2.0)):
        rows += [f"{rank},{role},attn.sparse_mla_core,{attn}", f"{rank},{role},petit.stage1(dispatch_wait+gate_up),1.0",
                 f"{rank},{role},mystery,0.5", f"{rank},{role},aux.shared_expert,2.0"]
    (analysis / "per_category.csv").write_text("\n".join(rows) + "\n")
    (analysis / "summary.csv").write_text("rank,graph_ms,period_ms,host_gap_ms,harvest_ms,n\n"
                                          "0,4.5,4.6,0.1,1.0,100\n1,3.5,4.6,1.1,1.0,100\n")
    seq = {"type": "seq", "stride": 1, "names": ["0:attn.core", "0:attn.core", "0:petit.s1", "0:shared.b"],
           "streams": ["main", "main", "main", "aux"]}
    (dump / "step-timing-r0-p1.jsonl").write_text(json.dumps(seq) + "\n")
    return analysis, dump


def test_table_correction_and_diff() -> None:
    with tempfile.TemporaryDirectory() as d:
        analysis, dump = fixture(Path(d))
        t = sb.table(str(analysis), str(dump), "x", clean_ms=4.2)
        assert t["main_markers_per_step"] == 3 and abs(t["delta_us_per_marker"] - 1000 * (4.0 - 4.2) / 3) < 1e-9
        core = t["comp"]["attn: sparse-MLA core"]
        assert (core["busy"], core["idle"], core["markers"]) == (3.0, 2.0, 2)
        assert abs(core["corr"] - (3.0 + 2 * 0.2 / 3)) < 1e-9
        assert "other: mystery" in t["comp"] and t["aux"]["aux.shared_expert"]["0"] == 2.0
        md = sb.table_markdown(t)
        assert "| **sum of components** | **4.500** | **3.500** |" in md
        last = sb.diff(t, t, "A", "B").splitlines()[-2]
        assert last.startswith("| **sum** | **4.500** | **4.500** | **+0.000** |")


def test_without_clean_step_there_is_no_correction() -> None:
    with tempfile.TemporaryDirectory() as d:
        analysis, dump = fixture(Path(d))
        t = sb.table(str(analysis), str(dump), "x")
        assert t["delta_us_per_marker"] is None and t["comp"]["attn: sparse-MLA core"]["corr"] is None
        json.dumps(t, allow_nan=False)
        assert "no clean step given" in sb.table_markdown(t)


def test_overhead_row() -> None:
    with tempfile.TemporaryDirectory() as d:
        analysis, _ = fixture(Path(d))
        metrics = Path(d) / "metrics.json"
        metrics.write_text(json.dumps({"tpot_ms_mean": 5.0}))
        row = sb.overhead([f"p B=1={metrics},{analysis},4.0"]).splitlines()[2]
        assert row == "| p B=1 | 5.00 | 4.00 | +25.0 | 4.000 | 4.600 | 0.600 | 100 |"


def test_share_on_self_test_dumps() -> None:
    """Petit share from the analyzer's synthetic dumps: 43 target layers and 1 draft stage instrumented."""
    import analyze_step_timing as ast
    with tempfile.TemporaryDirectory() as d:
        dump, out = Path(d) / "dump", Path(d) / "analysis"
        ast.synth(str(dump))
        ast.analyze(str(dump), str(out), skip=20)
        busy = ast.TRUTH
        per_target = sum(busy[n][0] for n in ("moe.experts", "petit.s1", "petit.s2", "petit.comb", "petit.post"))
        rows = sb.share(str(out), clean_ms=10.0, target_layers=43, draft_layers=3, fields={"batch_size": "4"})
        r = [x for x in rows if x["rank"] == 2][0]
        assert r["batch_size"] == "4" and r["role"] == "busy"
        assert (r["instrumented_target_layers"], r["instrumented_draft_layers"]) == (43, 1)
        assert abs(r["target_petit_ms_per_layer"] - per_target) < 0.03 * per_target
        assert abs(r["draft_petit_ms_per_layer"] - 0.5 * per_target) < 0.03 * per_target
        want = 43 * per_target + 3 * 0.5 * per_target
        assert abs(r["petit_ms_per_step"] - want) < 0.03 * want
        assert abs(r["petit_share_of_clean"] - want / 10.0) < 0.03 * want / 10.0
        # corr subtracts (graph - clean) / main markers per Petit segment (5 per instrumented layer)
        delta = (r["graph_ms"] - 10.0) / int(next(x for x in __import__("csv").DictReader(open(out / "summary.csv"))
                                                if x["rank"] == "2")["main_markers"])
        assert abs(r["petit_ms_per_step_corr"] - (r["petit_ms_per_step"] - 5 * delta * (43 + 3))) < 1e-6
        csv_path = Path(d) / "rows.csv"
        sb.append_csv(str(csv_path), rows)
        sb.append_csv(str(csv_path), rows)
        assert len(csv_path.read_text().splitlines()) == 1 + 2 * len(rows)


def test_draft_markers_counted_apart() -> None:
    assert sb.marker_name("0:moe.experts") == "moe.experts"
    assert sb.marker_name("1000:moe.experts") == "draft:moe.experts"
    assert sb.marker_name("-1:draft.end") == "draft.end"


def test_tp4ep4_petit_markers() -> None:
    # One tp4ep4-petit MoE layer at TokenSpeed 649a6353: ffn, moe.split, moe.router, moe.experts, the four
    # Petit markers, shared.b/e (main stream), moe.join (row placement), moe.comm (combine all-reduce), ffn.end.
    import analyze_step_timing as ast
    names = ["0:ffn", "0:moe.split", "0:moe.router", "0:moe.experts", "0:petit.s1", "0:petit.s2", "0:petit.comb",
             "0:petit.post", "0:shared.b", "0:shared.e", "0:moe.join", "0:moe.comm", "0:ffn.end"]
    t = [0.0, 0.1, 0.11, 0.2, 0.3, 0.6, 0.8, 0.9, 0.95, 1.45, 1.5, 1.6, 2.4]
    cats, _, aux, _ = ast.step_components({"names": names, "streams": ["main"] * len(names)}, t, 2.5)
    assert abs(cats["moe.row_split(tp4ep4-petit)"] - 0.01) < 1e-9
    assert abs(cats["ffn.add+post_comm(allreduce)"] - 0.1) < 1e-9  # moe.join -> moe.comm: row placement
    assert abs(cats["ffn.post_comm(moe allreduce)"] - 0.8) < 1e-9  # moe.comm -> ffn.end: the all-reduce
    assert abs(aux["aux.shared_expert"] - 0.5) < 1e-9
    comps, _ = sb.load_map(str(Path(sb.HERE) / "components.json"))
    owner = {c: comp["name"] for comp in comps for c in comp["categories"]}
    assert sb.project_group(owner["moe.row_split(tp4ep4-petit)"]) == "row_split"
    assert sb.project_group(owner["ffn.post_comm(moe allreduce)"]) == "moe_allreduce"
    # groups of the components that existed before 649a6353 keep their meaning
    assert sb.project_group(owner["ffn.add+post_comm(allreduce)"]) == "other"
    assert sb.project_group(owner["other:shared.b"]) == "shared_expert"
    assert sb.project_group(owner["moe.router_gate_topk"]) == "router"
    assert sb.project_group("other: draft:ffn.post_comm(moe allreduce)") == "draft"


def test_c3_fused_post_markers() -> None:
    # One DP4-Petit MoE layer with C3 (TOKENSPEED_PETIT_FUSE_POST=1) at TokenSpeed 20ffdf4f: the shared expert is
    # issued first on the aux stream, moe.experts follows it, and the combine is petit.cwait (wait for the
    # shared rows) + petit.combr (combine with the shared add) instead of petit.comb.
    import analyze_step_timing as ast
    names = ["0:ffn", "0:moe.router", "0:shared.b", "0:shared.e", "0:moe.experts", "0:petit.s1", "0:petit.s2",
             "0:petit.cwait", "0:petit.combr", "0:petit.post", "0:moe.join", "0:ffn.end"]
    streams = ["main", "main", "s9", "s9", "main", "main", "main", "main", "main", "main", "main", "main"]
    t = [0.0, 0.1, 0.12, 0.42, 0.15, 0.2, 0.5, 0.7, 0.75, 0.95, 0.96, 0.97]
    cats, buckets, aux, _ = ast.step_components({"names": names, "streams": streams}, t, 1.0)
    assert abs(cats["moe.router_gate_topk"] - 0.05) < 1e-9
    assert abs(cats["moe.prep(quant/non-petit experts)"] - 0.05) < 1e-9
    assert abs(cats["petit.stage2(down)"] - 0.2) < 1e-9  # C5b's tile stays inside petit.s2
    assert abs(cats["petit.combine_wait_shared(c3)"] - 0.05) < 1e-9
    assert abs(cats["petit.combine+shared_add(c3)"] - 0.2) < 1e-9
    assert "petit.combine" not in cats
    assert abs(aux["aux.shared_expert"] - 0.3) < 1e-9
    assert abs(buckets["moe_combine"] - 0.25) < 1e-9
    # both C3 segments are part of the Petit MoE of the layer (share) and of the petit group (project)
    for cat in ("petit.combine_wait_shared(c3)", "petit.combine+shared_add(c3)"):
        assert cat in sb.PETIT_CATEGORIES
    assert {"petit.cwait", "petit.combr"} <= set(sb.PETIT_MARKERS)
    comps, _ = sb.load_map(str(Path(sb.HERE) / "components.json"))
    owner = {c: comp["name"] for comp in comps for c in comp["categories"]}
    for cat in ("petit.combine_wait_shared(c3)", "petit.combine+shared_add(c3)"):
        assert sb.project_group(owner[cat]) == "petit"
        assert sb.project_group(owner["draft:" + cat]) == "draft"
    markers = {m for comp in comps for m in comp["markers"]}
    assert {"petit.cwait", "petit.combr", "draft:petit.cwait", "draft:petit.combr"} <= markers


if __name__ == "__main__":
    test_table_correction_and_diff()
    test_without_clean_step_there_is_no_correction()
    test_overhead_row()
    test_share_on_self_test_dumps()
    test_draft_markers_counted_apart()
    test_tp4ep4_petit_markers()
    test_c3_fused_post_markers()
    print("ok")


def test_stage_window_selected_forwards() -> None:
    import analyze_step_timing as ast
    with tempfile.TemporaryDirectory() as d:
        attempt = Path(d) / "attempt-001"
        stage = attempt / "stages" / "c0-l50000-b32"
        stage.mkdir(parents=True)
        st = {"started_unix": 100.0, "prefill_request_seconds": 10.0, "finished_unix": 200.0,
              "selected_decode_indices_by_rank": {"0": [5, 6, 7], "1": [8, 9, 10], "2": []}}
        (stage / "stage.json").write_text(json.dumps(st))
        for rank, base in ((0, 120.0), (1, 120.5)):
            (attempt / f"forwards-rank{rank}.jsonl").write_text(
                "".join(json.dumps({"index": i, "wall_unix": base + (i % 5)}) + "\n" for i in range(12)))
        assert ast.stage_window(str(stage / "stage.json")) == (110.0, 200.0)
        lo, hi = ast.stage_window(str(stage / "stage.json"), selected=True)
        # rank 0: 5..7 -> walls 120..122; rank 1: 8..10 -> 123.5, 124.5, 120.5 (wall order need not follow
        # index; first/last are the selected first and last forwards); period = max over ranks
        assert lo == 120.0 and abs(hi - (122.0 + 1.0)) < 1e-9, (lo, hi)
        # attention DP 1: one index list (selected_decode_indices) read against rank 0's forwards
        st.pop("selected_decode_indices_by_rank"); st["selected_decode_indices"] = [5, 6, 7]
        (stage / "stage.json").write_text(json.dumps(st))
        lo, hi = ast.stage_window(str(stage / "stage.json"), selected=True)
        assert lo == 120.0 and abs(hi - 123.0) < 1e-9, (lo, hi)


def test_project_whole_step() -> None:
    t = {"stride": 10, "clean_ms": 10.0, "role": {"0": "busy"}, "graph": {"0": 11.0},
         "comp": {"embed/prep (+graph.pre)": {"vals": {"0": 0.1}},
                  "attn: sparse-MLA core": {"vals": {"0": 0.4}},
                  "Petit Stage1 (dispatch wait + gate/up)": {"vals": {"0": 0.2}},
                  "layer tail (ffn.end->next layer)": {"vals": {"0": 9.0}},
                  "DSpark draft: Petit MoE (moe.experts..petit.post of the instrumented stages)": {"vals": {"0": 0.1}},
                  "DSpark draft: layer tail (ffn.end->next marker; holds the uninstrumented stages)": {"vals": {"0": 0.5}},
                  "lm_head+sampling+outputs": {"vals": {"0": 0.3}}}}
    rows = {r["component"]: r for r in sb.project(t, 40, 3, {"batch_size": "4"})}
    assert rows["attn: sparse-MLA core"]["projected_ms"] == 4.0 and rows["attn: sparse-MLA core"]["group"] == "attention"
    assert rows["Petit Stage1 (dispatch wait + gate/up)"]["projected_ms"] == 2.0
    assert abs(rows["DSpark draft: Petit MoE (moe.experts..petit.post of the instrumented stages)"]["projected_ms"] - 0.3) < 1e-12
    assert rows["embed/prep (+graph.pre)"]["projected_ms"] == 0.1 and rows["lm_head+sampling+outputs"]["factor"] == 1.0
    assert "layer tail (ffn.end->next layer)" not in rows
    assert sb.project_group("norms/mHC pre-attn") == "norms" and sb.project_group("MoE router/top-k") == "router"
    assert abs(rows["remainder (graph - projected)"]["projected_ms"] - (11.0 - 0.1 - 4.0 - 2.0 - 0.3 - 0.3)) < 1e-9
    assert rows["remainder (graph - projected)"]["batch_size"] == "4"


def test_share_and_project_need_the_models_layer_counts() -> None:
    for argv in (["share", "dir", "--clean-ms", "1"], ["share", "dir", "--clean-ms", "1", "--target-layers", "40"],
                 ["project", "t.json", "--draft-layers", "3"]):
        try:
            sb.main(argv)
        except SystemExit as exit_:
            assert exit_.code == 2, argv
        else:
            raise AssertionError(f"{argv} ran without the layer counts")


def test_project_groups_sum() -> None:
    t = {"stride": 40, "clean_ms": 10.0, "role": {"0": "busy"}, "graph": {"0": 10.0},
         "comp": {"attn: sparse-MLA core": {"vals": {"0": 0.1}}, "attn: qkv_a projection": {"vals": {"0": 0.05}},
                  "Petit Combine": {"vals": {"0": 0.02}},
                  "other: draft:aux.shared_expert": {"vals": {"0": 0.5}}}}
    g = {r["group"]: r for r in sb.project_groups(sb.project(t, 40, 3))}
    assert abs(g["attention"]["projected_ms"] - 6.0) < 1e-9 and g["attention"]["components"] == 2
    assert abs(g["petit"]["projected_ms"] - 0.8) < 1e-9
    assert abs(g["remainder"]["projected_ms"] - 3.2) < 1e-9
    assert abs(sum(r["share_of_graph"] for r in g.values()) - 1.0) < 1e-9


def test_share_marker_us() -> None:
    import analyze_step_timing as ast
    with tempfile.TemporaryDirectory() as d:
        dump, out = Path(d) / "dump", Path(d) / "analysis"
        ast.synth(str(dump))
        ast.analyze(str(dump), str(out), skip=20)
        r = [x for x in sb.share(str(out), clean_ms=10.0, target_layers=43, draft_layers=3, marker_us=5.0)
             if x["rank"] == 2][0]
        # 5 Petit markers per instrumented layer, 5 us each, projected to 43 + 3 layers
        assert abs(r["petit_ms_per_step_marker"] - (r["petit_ms_per_step"] - 5 * 0.005 * (43 + 3))) < 1e-6
        assert abs(r["petit_share_marker"] - r["petit_ms_per_step_marker"] / 10.0) < 1e-12
        assert "marker_us" not in sb.share(str(out), clean_ms=10.0, target_layers=43, draft_layers=3)[0]


def test_project_marker_closure() -> None:
    # 4 timed layers of 1.0 ms each, 10 markers each at 5 us; untimed layers 0.95 ms (= 1.0 - 10 x 5 us)
    # and 3 step-level markers: the marker-free projection must close on the clean step exactly
    us, timed_ms, m = 5.0, 1.0, 10
    clean = 40 * (timed_ms - m * us / 1000)
    graph = clean + (4 * m + 3) * us / 1000
    t = {"stride": 10, "clean_ms": clean, "role": {"0": "busy"}, "graph": {"0": graph}, "main_markers_per_step": 4 * m + 3,
         "comp": {"attn: sparse-MLA core": {"vals": {"0": 4 * timed_ms}, "markers": 4 * m},
                  "layer tail (ffn.end->next layer)": {"vals": {"0": graph - 4 * timed_ms}, "markers": 3}}}
    plain = sb.project(t, 40, 3)
    rows = sb.project(t, 40, 3, marker_us=us)
    assert [r["projected_ms"] for r in rows] == [r["projected_ms"] for r in plain]
    assert "markers" not in plain[0] and all(set(r) == set(rows[0]) for r in rows)
    rem = rows[-1]
    assert abs(rem["overshoot_ms"] - (40 * timed_ms - graph)) < 1e-9
    assert abs(rem["overshoot_marker_vs_clean_ms"]) < 1e-9
    assert abs(rem["closing_marker_us_clean"] - us) < 1e-9 and abs(rem["closing_marker_us_graph"] - us) < 1e-9
    g = {r["group"]: r for r in sb.project_groups(rows)}
    assert g["attention"]["markers"] == 40 and abs(g["attention"]["projected_ms_marker"] - clean) < 1e-9
