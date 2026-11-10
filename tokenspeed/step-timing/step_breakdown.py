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

The correction is approximate: the shares, not the absolute corrected times, are the evidence.
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


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
    names = [n.split(":", 1)[1] for n, s in zip(seq["names"], seq["streams"]) if s == "main"]
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
                       ("host_gap", "host gap = period - graph"), ("harvest", "harvest ms (host)")):
        lines.append(f"| {title} | " + " | ".join(f"{t[key][r]:.3f}" for r in ranks) + " | | | |")
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
    a = ap.parse_args(argv)
    if a.cmd == "table":
        res = table(a.analysis_dir, a.dump_dir, a.label, a.clean_ms, a.components)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
        sys.stdout.write(table_markdown(res))
    elif a.cmd == "diff":
        la, lb = a.labels.split(",")
        sys.stdout.write(diff(json.load(open(a.a)), json.load(open(a.b)), la, lb))
    else:
        sys.stdout.write(overhead(a.point))
    return 0


if __name__ == "__main__":
    sys.exit(main())
