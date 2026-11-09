#!/usr/bin/env python3
"""Per-decode-step GPU time breakdown from torch-profiler Chrome traces
(TokenSpeed /start_profile output, one <id>-DP<d>-TP<t>.trace.json.gz per rank)
or rocprofv3 kernel-trace CSV (--kernel-trace --output-format csv).

stdlib only. Usage:
  analyze_trace.py --categories categories.json --out OUTDIR [--step-mode launch|gap]
                   [--gap-us 300] [--skip-steps 2] TRACE [TRACE ...]
Each TRACE file is one rank (rank label = DPx-TPy tag from file name, else index).
rocprofv3 CSV: one file may hold several agents/processes; each becomes a rank.

Step boundaries:
  launch: host-side hipGraphLaunch/cudaGraphLaunch (cat cuda_runtime) starts a step;
          a step = [launch_i, launch_{i+1}); kernels are assigned by GPU start ts.
          Requires CPU activity in the trace (default activities CPU+GPU).
  gap:    split the GPU kernel timeline where the GPU is idle > --gap-us (works
          for rocprofv3 and for traces without runtime events).
Outputs: steps.csv (incl. per-category us per step), top_kernels.csv, cross_rank.csv,
collective_wait.csv, report.md
"""
import argparse, csv, gzip, json, os, re, statistics, sys
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_RE = re.compile(r"(hip|cuda)GraphLaunch", re.I)


def load_categories(path):
    d = json.load(open(path))
    pats = [(c["category"], re.compile(c["regex"], re.I)) for c in d["categories"]]
    return pats, d.get("default", "other")


def categorize(name, pats, default, cache):
    c = cache.get(name)
    if c is None:
        c = default
        for cat, rx in pats:
            if rx.search(name):
                c = cat
                break
        cache[name] = c
    return c


def rank_label(path, idx):
    m = re.search(r"(DP\d+-)?TP\d+", os.path.basename(path))
    return m.group(0) if m else f"r{idx}"


def load_chrome(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    evs = d["traceEvents"] if isinstance(d, dict) else d
    kernels, launches = [], []
    for e in evs:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in GPU_CATS:
            kernels.append((float(e["ts"]), float(e.get("dur", 0)), e.get("name", "?"),
                            (e.get("args") or {}).get("stream", 0)))
        elif LAUNCH_RE.search(e.get("name", "")):
            launches.append(float(e["ts"]))
    return {"main": (kernels, sorted(launches))}


def load_rocprof_csv(path):
    ranks = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Kernel_Name") or row.get("KernelName") or row.get("Name")
            s = row.get("Start_Timestamp") or row.get("BeginNs")
            t = row.get("End_Timestamp") or row.get("EndNs")
            if name is None or s is None or t is None:
                continue
            key = "agent" + (row.get("Agent_Id") or row.get("gpu-id") or row.get("Process_Id") or "0")
            s, t = int(s) / 1000.0, int(t) / 1000.0
            ranks[key].append((s, t - s, name, row.get("Queue_Id", 0)))
    return {k: (v, []) for k, v in ranks.items()}


def busy_union(iv):
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def split_steps(kernels, launches, mode, gap_us):
    kernels = sorted(kernels)
    if not kernels:
        return []
    steps = []
    if mode == "launch" and len(launches) >= 2:
        bounds = launches + [float("inf")]
        j = 0
        cur = []
        for k in kernels:
            while j + 1 < len(bounds) and k[0] >= bounds[j + 1]:
                if cur:
                    steps.append((bounds[j], cur))
                cur = []
                j += 1
            if k[0] >= bounds[0]:
                cur.append(k)
        if cur:
            steps.append((bounds[j], cur))
        return steps
    cur, last_end = [kernels[0]], kernels[0][0] + kernels[0][1]
    for k in kernels[1:]:
        if k[0] - last_end > gap_us:
            steps.append((cur[0][0], cur))
            cur = []
        cur.append(k)
        last_end = max(last_end, k[0] + k[1])
    steps.append((cur[0][0], cur))
    return steps


def med(xs):
    return statistics.median(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--categories", required=True,
                    help='ordered kernel-name map {"categories": [{"category", "regex"}], "default"}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-mode", choices=["launch", "gap"], default="launch")
    ap.add_argument("--gap-us", type=float, default=300.0)
    ap.add_argument("--skip-steps", type=int, default=1, help="drop first/last N steps (partial)")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    pats, default = load_categories(a.categories)
    cats = [c for c, _ in pats] + [default]
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    per_rank = {}  # rank -> list of step dicts
    kern_tot = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for i, p in enumerate(a.traces):
        data = load_rocprof_csv(p) if p.endswith(".csv") else load_chrome(p)
        for sub, (kernels, launches) in data.items():
            rk = rank_label(p, i) if sub == "main" else f"{rank_label(p, i)}:{sub}"
            mode = a.step_mode if launches else "gap"
            steps = split_steps(kernels, launches, mode, a.gap_us)
            if a.skip_steps and len(steps) > 2 * a.skip_steps:
                steps = steps[a.skip_steps:-a.skip_steps]
            out = []
            for idx, (t0, ks) in enumerate(steps):
                gs = min(k[0] for k in ks)
                ge = max(k[0] + k[1] for k in ks)
                busy = busy_union([(k[0], k[0] + k[1]) for k in ks])
                cat_t = defaultdict(float)
                coll = []
                for k in ks:
                    c = categorize(k[2], pats, default, cache)
                    cat_t[c] += k[1]
                    kern_tot[rk][k[2]][0] += k[1]
                    kern_tot[rk][k[2]][1] += 1
                    if c in ("collective_comm", "moe_dispatch", "moe_combine"):
                        coll.append((c, k[2], k[1]))
                out.append(dict(step=idx, start_us=t0, gpu_span_us=ge - gs, busy_us=busy,
                                n_kernels=len(ks), cat=cat_t, coll=coll, gpu_start=gs, gpu_end=ge,
                                mode=mode))
            # wall time per step = distance between consecutive step starts
            for j in range(len(out)):
                if j + 1 < len(out):
                    out[j]["wall_us"] = out[j + 1]["start_us"] - out[j]["start_us"]
                else:
                    out[j]["wall_us"] = out[j]["gpu_end"] - out[j]["start_us"]
                out[j]["idle_us"] = max(0.0, out[j]["wall_us"] - out[j]["busy_us"])
            per_rank[rk] = out
    ranks = sorted(per_rank)
    with open(os.path.join(a.out, "steps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "step", "mode", "wall_us", "gpu_span_us", "busy_us", "idle_us", "n_kernels"] + cats)
        for rk in ranks:
            for s in per_rank[rk]:
                w.writerow([rk, s["step"], s["mode"], f"{s['wall_us']:.1f}", f"{s['gpu_span_us']:.1f}", f"{s['busy_us']:.1f}",
                            f"{s['idle_us']:.1f}", s["n_kernels"]] + [f"{s['cat'].get(c, 0.0):.1f}" for c in cats])
    with open(os.path.join(a.out, "top_kernels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "kernel", "category", "total_us", "count", "us_per_step"])
        for rk in ranks:
            n = max(1, len(per_rank[rk]))
            for name, (t, c) in sorted(kern_tot[rk].items(), key=lambda x: -x[1][0])[: a.top]:
                w.writerow([rk, name, categorize(name, pats, default, cache), f"{t:.1f}", c, f"{t / n:.1f}"])
    # cross-rank medians
    rows = []
    for c in ["wall_us", "busy_us", "idle_us", "n_kernels"] + cats:
        row = [c]
        for rk in ranks:
            st = per_rank[rk]
            vals = [s[c] if c in s else s["cat"].get(c, 0.0) for s in st]
            row.append(med(vals))
        rows.append(row)
    with open(os.path.join(a.out, "cross_rank.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric_median_per_step"] + ranks)
        for r in rows:
            w.writerow([r[0]] + [f"{v:.1f}" for v in r[1:]])
    # collective wait: align step k across ranks; per comm category, min across ranks ~ transfer
    wait_rows = []
    nsteps = min((len(per_rank[r]) for r in ranks), default=0)
    for c in ("collective_comm", "moe_dispatch", "moe_combine"):
        per_r = {rk: [] for rk in ranks}
        mins = []
        for k in range(nsteps):
            vals = {rk: per_rank[rk][k]["cat"].get(c, 0.0) for rk in ranks}
            m = min(vals.values())
            mins.append(m)
            for rk in ranks:
                per_r[rk].append(vals[rk] - m)
        wait_rows.append([c, f"{med(mins):.1f}"] + [f"{med(per_r[rk]):.1f}" for rk in ranks])
    with open(os.path.join(a.out, "collective_wait.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "median_min_across_ranks_us(~transfer)"] + [f"excess_{rk}(~wait)" for rk in ranks])
        w.writerows(wait_rows)
    # markdown
    L = ["# Decode-step breakdown", "", f"traces: {len(a.traces)}; ranks: {', '.join(ranks)}; steps analysed per rank: "
         + ", ".join(f"{rk}={len(per_rank[rk])}({per_rank[rk][0]['mode'] if per_rank[rk] else '-'})" for rk in ranks), "",
         "## Median per step (us)", "", "| metric | " + " | ".join(ranks) + " |", "|---|" + "---|" * len(ranks)]
    for r in rows:
        if r[0] in cats and all(v == 0 for v in r[1:]):
            continue
        L.append(f"| {r[0]} | " + " | ".join(f"{v:.1f}" for v in r[1:]) + " |")
    L += ["", "Sanity: busy/wall ratio per rank: " + ", ".join(
        f"{rk}={(med([s['busy_us'] for s in per_rank[rk]]) / max(1e-9, med([s['wall_us'] for s in per_rank[rk]]))):.2f}" for rk in ranks),
          "", "## Communication time split (step k aligned across ranks)", "",
          "| category | min across ranks (~transfer) | " + " | ".join(f"excess {rk} (~wait)" for rk in ranks) + " |",
          "|---|---|" + "---|" * len(ranks)]
    for r in wait_rows:
        L.append("| " + " | ".join(r) + " |")
    L += ["", f"## Top {a.top} kernels per rank", ""]
    for rk in ranks:
        n = max(1, len(per_rank[rk]))
        L += [f"### {rk}", "", "| kernel | category | us/step | count/step |", "|---|---|---|---|"]
        for name, (t, c) in sorted(kern_tot[rk].items(), key=lambda x: -x[1][0])[: a.top]:
            L.append(f"| `{name[:90]}` | {categorize(name, pats, default, cache)} | {t / n:.1f} | {c / n:.1f} |")
        L.append("")
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
