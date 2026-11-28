#!/usr/bin/env python3
"""Component tables from analyze_step_timing.py output (instrumented, test-only numbers).

  step_breakdown.py table ANALYSIS_DIR DUMP_DIR --label L [--clean-ms X] [--json OUT]
      per-rank component times (sums of analyzer categories, components.json), idle-minus-busy,
      timing markers per component and, with --clean-ms (the uninstrumented step time of the same
      point), a corrected column: busy-rank time minus markers x (graph span - clean) / markers.
  step_breakdown.py diff A.json B.json --labels A,B
      two `table --json` outputs side by side: raw busy-rank and corrected time per component.
  step_breakdown.py overhead --point LABEL=METRICS_JSON,ANALYSIS_DIR,CLEAN_TPOT_MS [...]
      instrumentation overhead per point: decode TPOT measured while timing (fixed-batch metrics.json)
      against the clean TPOT, with the graph span, step period and host gap from the analyzer.
  step_breakdown.py share ANALYSIS_DIR --clean-ms X --target-layers N --draft-layers M
                    [--field batch_size=128 --field repetition=0 ...] [--append-csv ROWS.csv]
      the Gluon Petit MoE's share of a decode step, one row per rank: the Petit MoE time of the
      instrumented layers (segments moe.experts, petit.s1, petit.s2, petit.comb, petit.post: the whole
      MoELayer.forward -> kernel_moe_apply call, from activation quantization to the combined output;
      router/top-k and the shared expert excluded) per instrumented target layer and per instrumented
      draft stage, projected to all MoE layers (target x --target-layers + draft x --draft-layers), over
      the clean (uninstrumented) step; 'corr' subtracts the per-marker cost (graph - clean) / markers
      from each Petit segment. Rows append to a CSV for numerics/range_compare.py summarize.

The correction is approximate: the shares, not the absolute corrected times, are the evidence.
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFT_LAYER_OFFSET = 1000  # analyze_step_timing.DRAFT_LAYER_OFFSET
# Analyzer categories of the segments from moe.experts to the next marker after the Petit call
# (shared.b): activation quantize + route copies, stage 1 (dispatch wait + gate/up), stage 2 (down),
# combine, and the return to Python. Without kernel-side markers moe.experts alone spans the call.
# With TOKENSPEED_PETIT_FUSE_POST=1 the combine is petit.cwait (wait for the
# shared expert's rows) + petit.combr (combine with the shared add) instead of petit.comb, so the Petit
# MoE there also holds the shared add (and any join wait) that the unfused path does after the call.
PETIT_CATEGORIES = ("moe.prep(quant/non-petit experts)", "petit.stage1(dispatch_wait+gate_up)",
                    "petit.stage2(down)", "petit.combine", "petit.combine_wait_shared(c3)",
                    "petit.combine+shared_add(c3)", "moe.shared_join_wait")
PETIT_MARKERS = ("moe.experts", "petit.s1", "petit.s2", "petit.comb", "petit.cwait", "petit.combr", "petit.post")


def marker_name(name):
    """'L:name' -> name, prefixed 'draft:' for a draft-model layer (L >= DRAFT_LAYER_OFFSET)."""
    layer, n = name.split(":", 1)
    return ("draft:" + n) if int(layer) >= DRAFT_LAYER_OFFSET else n


def load_map(path):
    m = json.load(open(path))
    return [dict(c) for c in m["components"]], m["aux"]


def first_seq(dump_dir):
    for f in sorted(glob.glob(os.path.join(dump_dir, "*.jsonl"))):
        for line in open(f):
            if line.lstrip().startswith("{") and '"seq"' in line[:60]:
                rec = json.loads(line)
                if rec.get("type") == "seq":
                    return rec
    raise SystemExit(f"no seq record in {dump_dir}")


def table(analysis_dir, dump_dir, label, clean_ms=None, components=None):
    """Component table of one point; without clean_ms the per-marker correction (corr) is None."""
    comps, aux = load_map(components or os.path.join(HERE, "components.json"))
    cat = collections.defaultdict(dict)
    role = {}
    for r in csv.DictReader(open(os.path.join(analysis_dir, "per_category.csv"))):
        cat[int(r["rank"])][r["category"]] = float(r["median_ms"])
        role[int(r["rank"])] = r["role"]
    summ = {int(r["rank"]): r for r in csv.DictReader(open(os.path.join(analysis_dir, "summary.csv")))}
    ranks = sorted(cat)
    seq = first_seq(dump_dir)
    names = [marker_name(n) for n, s in zip(seq["names"], seq["streams"]) if s == "main"]
    count = collections.Counter(names)
    known = {c for comp in comps for c in comp["categories"]}
    for c in sorted({c for r in ranks for c in cat[r] if c not in known and not c.startswith("aux.")}):
        comps.append({"name": f"other: {c}", "categories": [c], "markers": []})
    graph = statistics.median(float(summ[r]["graph_ms"]) for r in ranks)
    delta = (graph - clean_ms) / len(names) if clean_ms is not None else None
    busy = [r for r in ranks if role[r] == "busy"]
    idle = [r for r in ranks if role[r] != "busy"]
    ref = busy[0] if busy else ranks[0]
    out = {"label": label, "ref_rank": ref, "role": {str(r): role[r] for r in ranks}, "stride": seq.get("stride"),
           "markers_per_step": len(seq["names"]), "main_markers_per_step": len(names), "graph_median_ms": graph,
           "clean_ms": clean_ms, "delta_us_per_marker": delta * 1e3 if delta is not None else None, "comp": {},
           "graph": {str(r): float(summ[r]["graph_ms"]) for r in ranks},
           "period": {str(r): float(summ[r]["period_ms"]) for r in ranks},
           "host_gap": {str(r): float(summ[r]["host_gap_ms"]) for r in ranks},
           "harvest": {str(r): float(summ[r]["harvest_ms"]) for r in ranks},
           # graph-replay span vs serving step (analyzer summary.csv; None for pre-1be88db0 dumps)
           **{k: {str(r): _num_or_none(summ[r].get(k)) for r in ranks}
              for k in ("gpu_step_ms", "outside_graph_ms", "host_step_ms", "launch_ms")},
           "n": {str(r): int(summ[r]["n"]) for r in ranks},
           "aux": {a: {str(r): cat[r].get(a, 0.0) for r in ranks} for a in aux}}
    for comp in comps:
        vals = {r: sum(cat[r].get(c, 0.0) for c in comp["categories"]) for r in ranks}
        if all(v == 0 for v in vals.values()):
            continue
        markers = sum(count[m] for m in comp["markers"])
        out["comp"][comp["name"]] = {
            "vals": {str(r): vals[r] for r in ranks},
            "busy": statistics.median(vals[r] for r in busy) if busy else None,
            "idle": statistics.median(vals[r] for r in idle) if idle else None,
            "markers": markers,
            "corr": vals[ref] - markers * delta if delta is not None else None}
    return out


def _num_or_none(x):
    """A summary.csv cell as a float, None when missing or nan (older analyzer output)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def table_markdown(t):
    ranks = sorted(t["role"], key=int)
    lines = [f"### {t['label']} (instrumented, test-only)", "",
             f"- markers per step: {t['markers_per_step']} ({t['main_markers_per_step']} main-stream); stride "
             f"{t['stride']}; median graph span {t['graph_median_ms']:.3f} ms; "
             + (f"clean step {t['clean_ms']} ms -> {t['delta_us_per_marker']:.2f} us per main-stream marker"
                if t["clean_ms"] is not None else "no clean step given: no correction"),
             "- 'corr' = reference-rank time minus markers x that per-marker cost: approximate, not a measurement", "",
             "| component | " + " | ".join(f"r{r} ({t['role'][r]})" for r in ranks) + " | idle-busy | markers | corr |",
             "|" + "---|" * (len(ranks) + 4)]
    tot = collections.Counter()
    totc = 0.0
    num = lambda x, spec: "-" if x is None else format(x, spec)
    for name, c in t["comp"].items():
        for r in ranks:
            tot[r] += c["vals"][r]
        totc = None if c["corr"] is None or totc is None else totc + c["corr"]
        ib = c["idle"] - c["busy"] if c["idle"] is not None and c["busy"] is not None else None
        lines.append(f"| {name} | " + " | ".join(f"{c['vals'][r]:.3f}" for r in ranks) +
                     f" | {num(ib, '+.3f')} | {c['markers']} | {num(c['corr'], '.3f')} |")
    lines.append("| **sum of components** | " + " | ".join(f"**{tot[r]:.3f}**" for r in ranks) +
                 f" | | {t['main_markers_per_step']} | {num(totc, '.3f')} |")
    for key, title in (("graph", "GPU graph span (median)"), ("period", "step period (median)"),
                       ("host_gap", "host gap = period - graph"), ("harvest", "harvest ms (host)"),
                       ("gpu_step_ms", "GPU step: replay start to next replay start"),
                       ("outside_graph_ms", "GPU outside the graph (non-graph kernels + idle)"),
                       ("host_step_ms", "host step: between replay calls"), ("launch_ms", "replay call (host)")):
        if key not in t or all(t[key][r] is None for r in ranks):
            continue
        lines.append(f"| {title} | " + " | ".join(num(t[key][r], ".3f") for r in ranks) + " | | | |")
    lines.append("| sum vs graph | " + " | ".join(f"{(tot[r] / t['graph'][r] - 1) * 100:+.2f}%" for r in ranks) + " | | | |")
    for a, v in t["aux"].items():
        lines.append(f"| {a} (overlapped, not in sum) | " + " | ".join(f"{v[r]:.3f}" for r in ranks) + " | | | |")
    lines.append("| sampled steps n | " + " | ".join(str(t["n"][r]) for r in ranks) + " | | | |")
    return "\n".join(lines) + "\n"


def diff(a, b, la, lb):
    def get(t, name, kind):
        c = t["comp"].get(name)
        if not c:
            return 0.0
        if kind == "corr":
            if c["corr"] is None:
                raise SystemExit("diff needs both tables made with --clean-ms")
            return c["corr"]
        return c["busy"] if c["busy"] is not None else c["vals"][str(t["ref_rank"])]
    names = list(a["comp"]) + [n for n in b["comp"] if n not in a["comp"]]
    lines = [f"| component | {la} raw | {lb} raw | {lb}-{la} raw | {la} corr | {lb} corr | {lb}-{la} corr |",
             "|---|---|---|---|---|---|---|"]
    s = [0.0] * 4
    for n in names:
        v = [get(a, n, "raw"), get(b, n, "raw"), get(a, n, "corr"), get(b, n, "corr")]
        s = [x + y for x, y in zip(s, v)]
        lines.append(f"| {n} | {v[0]:.3f} | {v[1]:.3f} | {v[1] - v[0]:+.3f} | {v[2]:.3f} | {v[3]:.3f} | {v[3] - v[2]:+.3f} |")
    lines.append(f"| **sum** | **{s[0]:.3f}** | **{s[1]:.3f}** | **{s[1] - s[0]:+.3f}** | **{s[2]:.3f}** | "
                 f"**{s[3]:.3f}** | **{s[3] - s[2]:+.3f}** |")
    ga, gb = a["graph"][str(a["ref_rank"])], b["graph"][str(b["ref_rank"])]
    lines.append(f"| graph span (reference rank) | {ga:.3f} | {gb:.3f} | {gb - ga:+.3f} | | | |")
    return "\n".join(lines) + "\n"


def overhead(points):
    lines = ["| point (instrumented) | TPOT while timing ms | clean TPOT ms | overhead % | graph span ms | "
             "step period ms | host gap ms | steps per rank |", "|---|---|---|---|---|---|---|---|"]
    for spec in points:
        # LABEL=METRICS,ANALYSIS,CLEAN_MS; the label may itself contain "=" (for example "B=1")
        head, _, rest = spec.partition(",")
        label, _, metrics = head.rpartition("=")
        analysis, clean = rest.split(",")
        tpot = json.load(open(metrics))["tpot_ms_mean"]
        s = list(csv.DictReader(open(os.path.join(analysis, "summary.csv"))))
        med = lambda k: statistics.median(float(x[k]) for x in s)
        lines.append(f"| {label} | {tpot:.2f} | {float(clean):.2f} | {(tpot / float(clean) - 1) * 100:+.1f} | "
                     f"{med('graph_ms'):.3f} | {med('period_ms'):.3f} | {med('host_gap_ms'):.3f} | "
                     f"{min(int(x['n']) for x in s)} |")
    return "\n".join(lines) + "\n"


def share(analysis_dir, clean_ms, target_layers=40, draft_layers=3, fields=None, marker_us=None):
    """Rows (one per rank) of the Petit MoE's share of the step; see the module docstring."""
    summ = {int(r["rank"]): r for r in csv.DictReader(open(os.path.join(analysis_dir, "summary.csv")))}
    rows = []
    for rank in sorted(summ):
        per = collections.defaultdict(float)
        layers = {"target": set(), "draft": set()}
        markers = {"target": 0, "draft": 0}
        for r in csv.DictReader(open(os.path.join(analysis_dir, f"per_layer_r{rank}.csv"))):
            layer = int(r["layer"])
            kind = "draft" if layer >= DRAFT_LAYER_OFFSET else "target"
            cat = r["category"][len("draft:"):] if kind == "draft" else r["category"]
            if cat in PETIT_CATEGORIES:
                per[kind] += float(r["median_ms"])
                layers[kind].add(layer)
                markers[kind] += 1
        graph = float(summ[rank]["graph_ms"])
        main_markers = int(summ[rank].get("main_markers") or 0)
        delta = (graph - clean_ms) / main_markers if main_markers else None
        row = dict(fields or {})
        row.update(rank=rank, role=summ[rank]["role"], n=int(summ[rank]["n"]), graph_ms=graph, clean_ms=clean_ms,
                   target_layers=target_layers, draft_layers=draft_layers,
                   instrumented_target_layers=len(layers["target"]), instrumented_draft_layers=len(layers["draft"]))
        proj = proj_corr = 0.0
        for kind, total in (("target", target_layers), ("draft", draft_layers)):
            k = len(layers[kind])
            if not k or not total:
                row[f"{kind}_petit_ms_per_layer"] = None
                continue
            per_layer = per[kind] / k
            row[f"{kind}_petit_ms_per_layer"] = per_layer
            proj += per_layer * total
            if delta is not None:
                proj_corr += (per[kind] - markers[kind] * delta) / k * total
        row["petit_ms_per_step"] = proj
        row["petit_share_of_clean"] = proj / clean_ms
        row["petit_share_of_graph"] = proj / graph
        row["petit_ms_per_step_corr"] = proj_corr if delta is not None else None
        row["petit_share_corr"] = proj_corr / clean_ms if delta is not None else None
        if marker_us is not None:
            # a fixed per-marker cost (e.g. profiling/graph_event_smoke.py per_event_node_overhead_us) removed
            # from each Petit marker of the instrumented layers before projecting
            proj_m = 0.0
            for kind, total in (("target", target_layers), ("draft", draft_layers)):
                k = len(layers[kind])
                if k and total:
                    proj_m += (per[kind] - markers[kind] * marker_us / 1000.0) / k * total
            row["marker_us"] = marker_us
            row["petit_ms_per_step_marker"] = proj_m
            row["petit_share_marker"] = proj_m / clean_ms
        rows.append(row)
    return rows


STEP_LEVEL = ("embed/prep", "final norm", "lm_head", "DSpark draft: embed/prep", "DSpark draft: tail")
UNTIMED_TAILS = ("layer tail", "DSpark draft: layer tail")


def project_group(name):
    """Coarse group of a table component (attention, router, petit, shared, norms, head, draft, row_split,
    moe_allreduce, other). row_split and moe_allreduce come from the markers moe.split
    and moe.comm; in dumps without them their time stays where it was (router / other)."""
    if name.startswith("DSpark draft") or name.startswith("other: draft:"):
        return "draft"
    for prefix, group in (("MoE row split", "row_split"), ("MoE output all-reduce", "moe_allreduce")):
        if name.startswith(prefix):
            return group
    for key, group in (("norm", "norms"), ("attn", "attention"), ("Petit", "petit"), ("MoE prep", "petit"),
                       ("router", "router"), ("shared", "shared_expert"), ("lm_head", "head"), ("embed", "embed")):
        if key in name:
            return group
    return "other"


def project(t, target_layers=40, draft_layers=3, fields=None, marker_us=None):
    """Rows (rank x component) of a whole decode step from one `table` result (its --json): each per-layer
    component of the instrumented target layers x target_layers / timed target layers, each per-stage
    draft component x draft_layers / timed draft stages, step-level ones x 1. The layer tails of timed
    layers hold the untimed layers, so they are left out and `remainder` = graph span - sum of projected
    components (the true per-layer tails, the draft stages' own tails and the projection error). Overlapped
    aux-stream draft categories (`other: draft:aux.*`) are left out like the target's aux rows.

    marker_us (e.g. profiling/graph_event_smoke.py per_event_node_overhead_us) adds, to every row,
    projected_ms_marker = (raw - markers x marker_us) x factor, and on the remainder row the closure of
    the projection: projected_sum_ms, graph_ms, clean_ms, overshoot_ms (sum - graph),
    projected_sum_marker_ms, overshoot_marker_vs_clean_ms (marker-free sum - clean step) and the
    per-marker cost that would close it, closing_marker_us_clean = (sum - clean) / sum(markers x factor)
    and closing_marker_us_graph = (sum - graph) / (sum(markers x factor) - main markers per step)."""
    stride = int(t["stride"])
    timed = {"target": -(-target_layers // stride), "draft": -(-draft_layers // stride)}
    rows = []
    for rank, graph in sorted(t["graph"].items(), key=lambda kv: int(kv[0])):
        total = total_m = marked = 0.0
        for name, c in t["comp"].items():
            if any(name.startswith(x) for x in UNTIMED_TAILS) or ":aux." in name:
                continue  # untimed layers' tails; overlapped aux-stream draft parts (not on the step's path)
            raw = c["vals"].get(rank)
            if raw is None:
                continue
            if any(name.startswith(x) for x in STEP_LEVEL):
                factor = 1.0
            elif name.startswith("DSpark draft") or name.startswith("other: draft:"):
                factor = draft_layers / timed["draft"]
            else:
                factor = target_layers / timed["target"]
            row = dict(fields or {})
            row.update(rank=int(rank), role=t["role"][rank], component=name, group=project_group(name), raw_ms=raw,
                       factor=factor, projected_ms=raw * factor, share_of_graph=raw * factor / graph,
                       share_of_clean=raw * factor / t["clean_ms"] if t.get("clean_ms") else None)
            if marker_us is not None:
                m = c.get("markers") or 0
                row.update(markers=m, projected_ms_marker=(raw - m * marker_us / 1000.0) * factor)
                total_m += row["projected_ms_marker"]
                marked += m * factor
            rows.append(row)
            total += raw * factor
        row = dict(fields or {})
        row.update(rank=int(rank), role=t["role"][rank], component="remainder (graph - projected)", group="remainder",
                   raw_ms=None, factor=None, projected_ms=graph - total, share_of_graph=(graph - total) / graph,
                   share_of_clean=(graph - total) / t["clean_ms"] if t.get("clean_ms") else None)
        if marker_us is not None:
            clean, main = t.get("clean_ms"), t.get("main_markers_per_step")
            row.update(markers=None, projected_ms_marker=None, marker_us=marker_us, projected_sum_ms=total,
                       graph_ms=graph, clean_ms=clean, overshoot_ms=total - graph, projected_sum_marker_ms=total_m,
                       overshoot_marker_vs_clean_ms=total_m - clean if clean else None,
                       closing_marker_us_clean=(total - clean) / marked * 1000.0 if clean and marked else None,
                       closing_marker_us_graph=(total - graph) / (marked - main) * 1000.0
                       if main is not None and marked > main else None)
            for r in rows:
                if r["rank"] == int(rank) and "marker_us" not in r:
                    r.update({k: None for k in row if k not in r})
                    r["marker_us"] = marker_us
        rows.append(row)
    return rows


def project_groups(rows):
    """project() rows summed per (rank, group): one row per group with the summed projected_ms and shares."""
    out = {}
    for r in rows:
        key = (r["rank"], r["group"])
        if key not in out:
            o = {k: v for k, v in r.items() if k not in ("component", "raw_ms", "factor")}
            o.update(components=0, projected_ms=0.0, share_of_graph=0.0,
                     share_of_clean=0.0 if r["share_of_clean"] is not None else None)
            if r.get("markers") is not None:
                o.update(markers=0, projected_ms_marker=0.0)
            out[key] = o
        o = out[key]
        o["components"] += 1
        o["projected_ms"] += r["projected_ms"]
        if r.get("markers") is not None:
            o["markers"] += r["markers"]
            o["projected_ms_marker"] += r["projected_ms_marker"]
        o["share_of_graph"] += r["share_of_graph"]
        if o["share_of_clean"] is not None:
            o["share_of_clean"] += r["share_of_clean"]
    return list(out.values())


def append_csv(path, rows):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    cols = list(rows[0])
    if not new:
        with open(path, newline="") as fh:
            cols = next(csv.reader(fh))
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        if new:
            w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("analysis_dir"); t.add_argument("dump_dir")
    t.add_argument("--label", required=True)
    t.add_argument("--clean-ms", type=float, help="uninstrumented step ms; enables the per-marker correction")
    t.add_argument("--components", help="component map (default: components.json beside this script)")
    t.add_argument("--json")
    d = sub.add_parser("diff")
    d.add_argument("a"); d.add_argument("b"); d.add_argument("--labels", default="A,B")
    o = sub.add_parser("overhead")
    o.add_argument("--point", action="append", required=True)
    s = sub.add_parser("share")
    s.add_argument("analysis_dir")
    s.add_argument("--clean-ms", type=float, required=True, help="uninstrumented decode step ms of the same point")
    s.add_argument("--target-layers", type=int, required=True, help="target MoE layers per step (the model's)")
    s.add_argument("--draft-layers", type=int, required=True, help="draft MoE layers per step (0 without a draft)")
    s.add_argument("--field", action="append", default=[], help="NAME=VALUE column added to every row")
    s.add_argument("--append-csv", help="append the rows to this CSV (header written when new)")
    s.add_argument("--marker-us", type=float, help="also give the share with this fixed cost per Petit marker removed "
                   "(columns petit_ms_per_step_marker, petit_share_marker)")
    pj = sub.add_parser("project", help="whole-step component rows projected from a `table --json` result")
    pj.add_argument("table_json")
    pj.add_argument("--target-layers", type=int, required=True)
    pj.add_argument("--draft-layers", type=int, required=True)
    pj.add_argument("--field", action="append", default=[], help="NAME=VALUE column added to every row")
    pj.add_argument("--append-csv", help="append the rows to this CSV (header written when new)")
    pj.add_argument("--groups", action="store_true", help="one row per (rank, group), summed over its components")
    pj.add_argument("--marker-us", type=float, help="also project each component net of this fixed cost per marker "
                    "(projected_ms_marker) and give the projection's closure on the remainder row (overshoot_ms, "
                    "overshoot_marker_vs_clean_ms, closing_marker_us_clean/_graph)")
    a = ap.parse_args(argv)
    if a.cmd == "table":
        res = table(a.analysis_dir, a.dump_dir, a.label, a.clean_ms, a.components)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
        sys.stdout.write(table_markdown(res))
    elif a.cmd == "share":
        rows = share(a.analysis_dir, a.clean_ms, a.target_layers, a.draft_layers,
                     dict(f.split("=", 1) for f in a.field), a.marker_us)
        if a.append_csv:
            append_csv(a.append_csv, rows)
        json.dump(rows, sys.stdout, indent=1)
        sys.stdout.write("\n")
    elif a.cmd == "project":
        rows = project(json.load(open(a.table_json)), a.target_layers, a.draft_layers,
                       dict(f.split("=", 1) for f in a.field), a.marker_us)
        if a.groups:
            rows = project_groups(rows)
        if a.append_csv:
            append_csv(a.append_csv, rows)
        json.dump(rows, sys.stdout, indent=1)
        sys.stdout.write("\n")
    elif a.cmd == "diff":
        la, lb = a.labels.split(",")
        sys.stdout.write(diff(json.load(open(a.a)), json.load(open(a.b)), la, lb))
    else:
        sys.stdout.write(overhead(a.point))
    return 0


if __name__ == "__main__":
    sys.exit(main())
