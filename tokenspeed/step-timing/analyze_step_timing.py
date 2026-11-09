#!/usr/bin/env python3
"""Turn TS_STEP_TIMING per-rank dumps (step-timing-r<rank>-p<pid>.jsonl) into a
per-component breakdown per decode step. CPU-only, stdlib only.

usage: analyze_step_timing.py --dir <dump dir> --out <out dir> [--skip 20] [--bs N]
       analyze_step_timing.py --self-test --out <dir>

Main-stream segments: marker k -> next main-stream marker, labelled by marker k.
Aux-stream (overlapped) pairs: attn.kvidx.b->e (compressor/KV write/indexer) and
shared.b->e (shared expert) are reported separately; they are hidden unless the
main stream waits at the join (attn.q segment / moe.shared_join segment).
"""
import argparse, collections, glob, json, math, os, random, statistics, sys

SEG = {  # start-marker name -> (category, bucket)
    "model.begin": ("embed+prep", "other"),
    "engram": ("engram", "engram"),
    "layer": ("mhc+norm(pre-attn)", "norm/mhc"),
    "attn": ("attn.qkv_a", "attention"),
    "attn.fork": ("attn.main_fork_gap", "attention"),
    "attn.kvidx.b": ("attn.compressor+indexer(serial)", "attention"),
    "attn.kvidx.e": ("attn.after_kvidx", "attention"),
    "attn.q": ("attn.q_proj+rope+join_wait", "attention"),
    "attn.core": ("attn.sparse_mla_core", "attention"),
    "attn.wo": ("attn.out_proj(+attn allreduce)", "attention"),
    "hc_ffn": ("mhc+norm(pre-ffn)", "norm/mhc"),
    "ffn": ("ffn.pre_comm", "collectives"),
    "moe.router": ("moe.router_gate_topk", "moe_router"),
    "moe.experts": ("moe.prep(quant/non-petit experts)", "moe_experts"),
    "petit.s1": ("petit.stage1(dispatch_wait+gate_up)", "moe_dispatch+gate_up"),
    "petit.s2": ("petit.stage2(down)", "moe_down"),
    "petit.comb": ("petit.combine", "moe_combine"),
    "petit.post": ("moe.shared_join_wait", "moe_experts"),
    "moe.join": ("ffn.add+post_comm(allreduce)", "collectives"),
    "ffn.end": ("layer_tail", "norm/mhc"),
    "final": ("final_norm", "other"),
    "model.end": ("lm_head+sampling+outputs", "head+sample"),
}
AUX = {("attn.kvidx.b", "attn.kvidx.e"): "aux.attn.compressor+indexer",
       ("shared.b", "shared.e"): "aux.shared_expert"}


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    f = math.floor(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def split(name):
    l, n = name.split(":", 1)
    return int(l), n


def step_components(seq, t, graph_ms):
    """Return {category: ms}, {bucket: ms}, {aux: ms}, per-layer {(layer,cat): ms}."""
    cats, buckets, aux, per_layer = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    main = [i for i, s in enumerate(seq["streams"]) if s == "main"]
    if main:
        cats["graph.pre(before model.begin)"] += t[main[0]]
        buckets["other"] += t[main[0]]
    for a, b in zip(main, main[1:] + [None]):
        lay, n = split(seq["names"][a])
        dt = (t[b] if b is not None else graph_ms) - t[a]
        cat, bk = SEG.get(n, ("other:" + n, "other"))
        if n.startswith("petit.") or n in ("moe.experts",):
            pass
        cats[cat] += dt; buckets[bk] += dt
        if lay >= 0:
            per_layer[(lay, cat)] += dt
    open_ = {}
    for i, nm in enumerate(seq["names"]):
        lay, n = split(nm)
        for (b, e), label in AUX.items():
            if n == b:
                open_[(lay, label)] = t[i]
            elif n == e and (lay, label) in open_:
                aux[label] += t[i] - open_.pop((lay, label))
    return cats, buckets, aux, per_layer


def load(dirpath):
    ranks = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "step-timing-r*-p*.jsonl"))):
        seqs, steps = {}, []
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") == "seq":
                seqs[o["sid"]] = o
            elif o.get("type") == "step":
                steps.append(o)
        base = os.path.basename(f)
        rank = int(base.split("-r")[1].split("-p")[0])
        ranks.setdefault(rank, []).append((f, seqs, steps))
    return ranks


def analyze(dirpath, out, skip=20, want_bs=None, expect_layers=None, wall_from=None, wall_to=None, busy_ranks=None):
    os.makedirs(out, exist_ok=True)
    ranks = load(dirpath)
    rows, bucket_rows, summary = [], [], []
    report = [f"# step-timing analysis of {dirpath}", ""]
    per_rank = {}
    for rank, files in sorted(ranks.items()):
        recs = []
        for f, seqs, steps in files:
            prev = None
            for s in steps:
                seq = seqs.get(s["sid"])
                if seq is None or len(s["t"]) != len(seq["names"]):
                    continue
                period = None
                if prev is not None and s["replay"] > prev["replay"]:
                    period = (s["wall"] - prev["wall"]) * 1e3 / (s["replay"] - prev["replay"])
                prev = s
                recs.append((seq, s, period))
        if wall_from is not None:
            recs = [r for r in recs if r[1]["wall"] >= wall_from]
        if wall_to is not None:
            recs = [r for r in recs if r[1]["wall"] <= wall_to]
        recs = recs[skip:]
        if want_bs is not None:
            recs = [r for r in recs if r[1].get("bs") == want_bs or r[1].get("padded_bs") == want_bs]
        if not recs:
            report.append(f"rank {rank}: no usable records"); continue
        busy = [r for r in recs if (r[1].get("input_num_tokens") or 0) > 0]
        role = "busy" if len(busy) * 2 >= len(recs) else "idle"
        if busy_ranks is not None:
            role = "busy" if rank in busy_ranks else "idle"
        C, B, A, L = (collections.defaultdict(list) for _ in range(4))
        graph, periods, harvest = [], [], []
        layers = set()
        for seq, s, period in recs:
            cats, buckets, aux, pl = step_components(seq, s["t"], s["graph_ms"])
            for k, v in cats.items(): C[k].append(v)
            for k, v in buckets.items(): B[k].append(v)
            for k, v in aux.items(): A[k].append(v)
            for (lay, c), v in pl.items(): L[(lay, c)].append(v); layers.add(lay)
            graph.append(s["graph_ms"]); harvest.append(s.get("harvest_ms", 0.0))
            if period is not None: periods.append(period)
        n_layers = len(layers)
        per_rank[rank] = dict(role=role, n=len(recs), C=C, B=B, A=A, graph=graph, periods=periods)
        gmed = statistics.median(graph)
        pmed = statistics.median(periods) if periods else float("nan")
        bsum = sum(statistics.median(v) for v in B.values())
        report += [f"## rank {rank} ({role}; {len(recs)} sampled steps; instrumented layers={n_layers}"
                   + (f" of {expect_layers}" if expect_layers else "") + ")",
                   f"- GPU graph span median {gmed:.3f} ms; step period median {pmed:.3f} ms; host gap (period - graph) {pmed - gmed:.3f} ms; harvest cost median {statistics.median(harvest):.3f} ms",
                   f"- sum of bucket medians {bsum:.3f} ms vs graph median {gmed:.3f} ms ({100*(bsum-gmed)/gmed:+.1f} %)", "",
                   "| bucket | median ms/step | p10 | p90 | % of graph |", "|---|---|---|---|---|"]
        for k, v in sorted(B.items(), key=lambda kv: -statistics.median(kv[1])):
            m = statistics.median(v)
            report.append(f"| {k} | {m:.3f} | {pct(v,.1):.3f} | {pct(v,.9):.3f} | {100*m/gmed:.1f} |")
            bucket_rows.append(dict(rank=rank, role=role, bucket=k, median_ms=m, p10=pct(v, .1), p90=pct(v, .9), n=len(v)))
        report += ["", "| category | median ms/step | p10 | p90 |", "|---|---|---|---|"]
        for k, v in sorted(C.items(), key=lambda kv: -statistics.median(kv[1])):
            m = statistics.median(v)
            report.append(f"| {k} | {m:.3f} | {pct(v,.1):.3f} | {pct(v,.9):.3f} |")
            rows.append(dict(rank=rank, role=role, category=k, median_ms=m, p10=pct(v, .1), p90=pct(v, .9), n=len(v)))
        for k, v in sorted(A.items()):
            m = statistics.median(v)
            report.append(f"| {k} (overlapped, not in graph sum) | {m:.3f} | {pct(v,.1):.3f} | {pct(v,.9):.3f} |")
            rows.append(dict(rank=rank, role=role, category=k, median_ms=m, p10=pct(v, .1), p90=pct(v, .9), n=len(v)))
        with open(os.path.join(out, f"per_layer_r{rank}.csv"), "w") as f:
            f.write("layer,category,median_ms\n")
            for (lay, c), v in sorted(L.items()):
                f.write(f"{lay},{c},{statistics.median(v):.5f}\n")
        report.append("")
        summary.append(dict(rank=rank, role=role, n=len(recs), graph_ms=gmed, period_ms=pmed,
                            host_gap_ms=pmed - gmed, bucket_sum_ms=bsum, layers=n_layers,
                            harvest_ms=statistics.median(harvest)))
    # busy vs idle comparison
    busy = [r for r, d in per_rank.items() if d["role"] == "busy"]
    idle = [r for r, d in per_rank.items() if d["role"] == "idle"]
    if busy and idle:
        report += ["## busy vs idle (median of rank medians)", "", "| bucket | busy | idle | idle-busy |", "|---|---|---|---|"]
        keys = sorted(set().union(*[per_rank[r]["B"].keys() for r in per_rank]))
        for k in keys:
            bm = statistics.median([statistics.median(per_rank[r]["B"][k]) for r in busy if per_rank[r]["B"].get(k)] or [0.0])
            im = statistics.median([statistics.median(per_rank[r]["B"][k]) for r in idle if per_rank[r]["B"].get(k)] or [0.0])
            report.append(f"| {k} | {bm:.3f} | {im:.3f} | {im-bm:+.3f} |")
        report.append("")
    for name, data, cols in (("per_category.csv", rows, ["rank", "role", "category", "median_ms", "p10", "p90", "n"]),
                             ("per_bucket.csv", bucket_rows, ["rank", "role", "bucket", "median_ms", "p10", "p90", "n"]),
                             ("summary.csv", summary, ["rank", "role", "n", "graph_ms", "period_ms", "host_gap_ms", "bucket_sum_ms", "layers", "harvest_ms"])):
        with open(os.path.join(out, name), "w") as f:
            f.write(",".join(cols) + "\n")
            for r in data:
                f.write(",".join(f"{r[c]:.5f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    open(os.path.join(out, "report.md"), "w").write("\n".join(report) + "\n")
    return summary, rows, bucket_rows


# ---------------------------------------------------------------- self-test
TRUTH = {  # per-layer main-stream segment durations (ms) for busy / idle ranks
    "layer": (0.010, 0.010), "attn": (0.020, 0.018), "attn.fork": (0.0, 0.0), "attn.q": (0.015, 0.012),
    "attn.core": (0.030, 0.025), "attn.wo": (0.012, 0.011), "hc_ffn": (0.008, 0.008), "ffn": (0.001, 0.001),
    "moe.router": (0.009, 0.009), "moe.experts": (0.004, 0.004), "petit.s1": (0.060, 0.110),
    "petit.s2": (0.025, 0.010), "petit.comb": (0.020, 0.045), "petit.post": (0.002, 0.002),
    "moe.join": (0.003, 0.003), "ffn.end": (0.004, 0.004)}
LAYER_ORDER = ["layer", "attn", "attn.fork", "attn.kvidx.b", "attn.kvidx.e", "attn.q", "attn.core", "attn.wo",
               "hc_ffn", "ffn", "moe.router", "moe.experts", "shared.b", "shared.e", "petit.s1", "petit.s2",
               "petit.comb", "petit.post", "moe.join", "ffn.end"]


def synth(dirpath, nlayers=43, steps=200, engram_layers=(0, 21)):
    os.makedirs(dirpath, exist_ok=True)
    rnd = random.Random(7)
    for rank in range(4):
        busy = rank == 2
        names, streams = ["-1:model.begin"], ["main"]
        for l in range(nlayers):
            if l in engram_layers:
                names.append(f"{l}:engram"); streams.append("main")
            for n in LAYER_ORDER:
                names.append(f"{l}:{n}")
                streams.append("s9" if n in ("attn.kvidx.b", "attn.kvidx.e", "shared.b", "shared.e") else "main")
        names += ["-1:final", "-1:model.end"]; streams += ["main", "main"]
        lines = [json.dumps({"type": "seq", "sid": 0, "bs": 1, "names": names, "streams": streams, "dropped": 0})]
        wall = 1000.0
        for k in range(steps):
            t, x = [], 0.05  # graph.pre
            fork_t = None
            for nm, st in zip(names, streams):
                lay, n = split(nm)
                if st != "main":
                    if n.endswith(".b"):
                        t.append(fork_t + 0.001)
                    else:
                        t.append(t[-1] + (0.040 if n == "attn.kvidx.e" else 0.030))
                    continue
                t.append(x)
                if n == "attn.fork":
                    fork_t = x
                if n == "moe.experts":
                    fork_t = x
                d = {"model.begin": 0.1, "engram": 0.05, "final": 0.02, "model.end": 0.3}.get(n)
                if d is None:
                    d = TRUTH[n][0 if busy else 1]
                x += d * (1 + rnd.uniform(-0.02, 0.02))
            graph_ms = x
            replay = 4 * (k + 1)
            wall += 4 * 15.3e-3
            lines.append(json.dumps({"type": "step", "sid": 0, "replay": replay, "wall": wall, "bs": 1 if busy else 0,
                                     "padded_bs": 1, "input_num_tokens": 1 if busy else 0,
                                     "global_num_tokens": [0, 0, 1, 0], "t": t, "graph_ms": graph_ms, "harvest_ms": 1.0}))
        open(os.path.join(dirpath, f"step-timing-r{rank}-p{1000+rank}.jsonl"), "w").write("\n".join(lines) + "\n")
    return nlayers


def self_test(out):
    d = os.path.join(out, "synthetic-dumps")
    nl = synth(d)
    summary, rows, brows = analyze(d, os.path.join(out, "synthetic-analysis"), skip=20, expect_layers=nl)
    ok = True
    def get(rank, cat):
        return [r["median_ms"] for r in rows if r["rank"] == rank and r["category"] == cat][0]
    checks = [(2, "petit.stage1(dispatch_wait+gate_up)", 43 * TRUTH["petit.s1"][0]),
              (0, "petit.stage1(dispatch_wait+gate_up)", 43 * TRUTH["petit.s1"][1]),
              (2, "petit.combine", 43 * TRUTH["petit.comb"][0]),
              (1, "attn.sparse_mla_core", 43 * TRUTH["attn.core"][1]),
              (2, "moe.router_gate_topk", 43 * TRUTH["moe.router"][0]),
              (2, "engram", 2 * 0.05),
              (2, "aux.attn.compressor+indexer", 43 * 0.040),
              (2, "aux.shared_expert", 43 * 0.030),
              (3, "lm_head+sampling+outputs", 0.3)]
    lines = []
    for rank, cat, want in checks:
        got = get(rank, cat)
        good = abs(got - want) <= 0.03 * want + 1e-6
        ok &= good
        lines.append(f"{'PASS' if good else 'FAIL'} rank{rank} {cat}: got {got:.4f} want {want:.4f}")
    for s in summary:
        good = abs(s["bucket_sum_ms"] - s["graph_ms"]) <= 0.02 * s["graph_ms"] and s["role"] == ("busy" if s["rank"] == 2 else "idle")
        ok &= good
        lines.append(f"{'PASS' if good else 'FAIL'} rank{s['rank']} role={s['role']} bucket_sum {s['bucket_sum_ms']:.3f} vs graph {s['graph_ms']:.3f}; period {s['period_ms']:.3f}")
    lines.append("SELF-TEST " + ("PASS" if ok else "FAIL"))
    open(os.path.join(out, "self-test.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=20); ap.add_argument("--bs", type=int)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--wall-from", type=float, help="epoch seconds; keep records at/after (point window start)")
    ap.add_argument("--wall-to", type=float, help="epoch seconds; keep records at/before (point window end)")
    ap.add_argument("--busy-ranks", help="comma list overriding the busy/idle role (e.g. from server-log placement)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test(a.out))
    analyze(a.dir, a.out, a.skip, a.bs, a.layers, a.wall_from, a.wall_to,
            None if a.busy_ranks is None else {int(x) for x in a.busy_ranks.split(',') if x})
    print(open(os.path.join(a.out, "report.md")).read())
