#!/usr/bin/env python3
"""Aggregate raw/L*_B*_rep*/metrics.json into results.json/summary.csv (median/min/max over
valid reps per (L,B)) and print a markdown table.

  prefill_aggregate.py --raw raw --out . [--chunk-cost] [--tail]

--chunk-cost  pool every same-rank Prefill->Prefill line gap of the valid max_tokens-1 points by
              (chunk tokens, new sequences in the chunk, whether the chunk completes a request,
              B=1 or B>=4): chunk-cost.csv with median, p10, p90 and tok/s.
--tail        time of each point's final prefill step (last Prefill line to last Finish line;
              ranks step together) by tail-chunk size: tail.csv, and a least-squares line
              final_step_ms = fixed + per_token x tail_tokens over the group medians.
"""
import argparse, csv, glob, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_log import parse_lines  # noqa: E402

KEYS = {"ttft_mean_s": lambda m: m["ttft_s"]["mean"], "ttft_max_s": lambda m: m["ttft_s"]["max"],
        "batch_prompt_tok_per_s": lambda m: m["batch_prompt_tok_per_s"],
        "server_tok_per_s": lambda m: m["server"]["server_tok_per_s_all_ranks"],
        "chunk_time_median_ms": lambda m: 1e3 * statistics.median(
            [r["chunk_time_median_s"] for r in m["server"]["ranks"].values() if r["chunk_time_median_s"] is not None] or [float("nan")]),
        "chunks_max_rank": lambda m: max(r["chunks"] for r in m["server"]["ranks"].values()),
        "requests_max_rank": lambda m: max(r["requests_by_tokens"] for r in m["server"]["ranks"].values()),
        "rank_prefill_tok_per_s": lambda m: statistics.median(
            [r["prefill_tok_per_s"] for r in m["server"]["ranks"].values() if r["prefill_tok_per_s"]])}


def quantile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q; i = int(k)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (k - i)


def point_dirs(raw):
    for d in sorted(glob.glob(os.path.join(raw, "L*_B*_rep*"))):
        mf = os.path.join(d, "metrics.json")
        if os.path.exists(mf):
            m = json.load(open(mf))
            if m["valid"] and m["max_tokens"] == 1:
                yield d, m


def chunk_cost(raw):
    """Per-chunk time pooled over valid max_tokens-1 points (requests assumed sequential per rank)."""
    pool = {}
    for d, m in point_dirs(raw):
        for rank in json.load(open(os.path.join(d, "metrics-chunks.json"))).values():
            done = 0
            for g in rank["chunk_gaps"]:
                before, done = done, done + g["chunk_tokens"]
                if g["next_kind"] not in ("Prefill", "Mix"):
                    continue
                completes = done // m["L"] > before // m["L"]
                key = (g["chunk_tokens"], g["new_seq"], "final" if completes else "intermediate",
                       "B1" if m["B"] == 1 else "B>=4")
                pool.setdefault(key, []).append(g["dt_s"])
    return [{"chunk_tokens": k[0], "new_seq": k[1], "completes_request": k[2], "batch_class": k[3], "n": len(v),
             "median_ms": 1e3 * statistics.median(v), "p10_ms": 1e3 * quantile(v, .1), "p90_ms": 1e3 * quantile(v, .9),
             "tok_per_s": k[0] / statistics.median(v)} for k, v in sorted(pool.items())]


def fit(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def tail(raw):
    """Final prefill step per point and by tail size, with a line through the group medians."""
    rows = []
    for d, m in point_dirs(raw):
        req = json.load(open(os.path.join(d, "requests.json")))
        t0 = min(r["t_send"] for r in req["requests"])
        ev = [e for e in parse_lines(open(os.path.join(d, "server-log-slice.txt"), errors="replace"))
              if e["ts"] and e["ts"] >= t0 - 1]
        last = {}
        for e in ev:
            if e["kind"] in ("Prefill", "Mix"):
                last[e["dp_rank"]] = e
        fin = [e["ts"] for e in ev if e["kind"] == "Finish"]
        if not last or not fin:
            continue
        sizes = sorted({e["new_token"] for e in last.values()})
        rows.append({"point": os.path.basename(d), "L": m["L"], "B": m["B"], "tail_tokens": sizes,
                     "final_step_ms": 1e3 * (max(fin) - max(e["ts"] for e in last.values()))})
    groups = {}
    for r in rows:
        if len(r["tail_tokens"]) == 1:
            groups.setdefault((r["tail_tokens"][0], "B1" if r["B"] == 1 else "B>=4"), []).append(r["final_step_ms"])
    summary = [{"tail_tokens": k[0], "batch_class": k[1], "n": len(v), "median_ms": statistics.median(v),
                "min_ms": min(v), "max_ms": max(v)} for k, v in sorted(groups.items())]
    fits = {}
    for bc in ("B1", "B>=4"):
        pts = [(s["tail_tokens"], s["median_ms"]) for s in summary if s["batch_class"] == bc]
        if len({x for x, _ in pts}) > 1:
            a, b = fit(pts)
            fits[bc] = {"fixed_ms": a, "per_token_ms": b}
    return rows, summary, fits


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-cost", action="store_true"); ap.add_argument("--tail", action="store_true")
    a = ap.parse_args(argv)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.raw, "L*_B*_rep*", "metrics.json"))):
        m = json.load(open(p)); rep = int(p.split("_rep")[1].split(os.sep)[0])
        row = {"L": m["L"], "B": m["B"], "rep": rep, "valid": m["valid"], "path": p}
        for k, f in KEYS.items():
            try:
                row[k] = f(m)
            except Exception:  # noqa
                row[k] = None
        rows.append(row)
    groups = {}
    for r in rows:
        groups.setdefault((r["L"], r["B"]), []).append(r)
    summ = []
    for (L, B), rs in sorted(groups.items()):
        v = [r for r in rs if r["valid"]]
        s = {"L": L, "B": B, "n_valid": len(v), "n_total": len(rs)}
        for k in KEYS:
            xs = [r[k] for r in v if r[k] is not None]
            s[k + "_median"] = statistics.median(xs) if xs else None
            s[k + "_min"] = min(xs) if xs else None; s[k + "_max"] = max(xs) if xs else None
        summ.append(s)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"rows": rows, "summary": summ}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    if summ:
        with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    print("| L | B | valid | TTFT mean s (min-max) | TTFT max s | batch prompt tok/s | server tok/s | chunk ms | chunks/rank | req/rank |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
    for s in summ:
        print(f"| {s['L']} | {s['B']} | {s['n_valid']}/{s['n_total']} | {f(s['ttft_mean_s_median'])} ({f(s['ttft_mean_s_min'])}-{f(s['ttft_mean_s_max'])}) | "
              f"{f(s['ttft_max_s_median'])} | {f(s['batch_prompt_tok_per_s_median'],0)} | {f(s['server_tok_per_s_median'],0)} | "
              f"{f(s['chunk_time_median_ms_median'],2)} | {f(s['chunks_max_rank_median'],0)} | {f(s['requests_max_rank_median'],2)} |")
    if a.chunk_cost:
        cc = chunk_cost(a.raw)
        if cc:
            write_csv(os.path.join(a.out, "chunk-cost.csv"), cc)
        print("\n| chunk tokens | new seq | completes a request | batch | n | median ms | p10 ms | p90 ms | tok/s |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c in cc:
            print(f"| {c['chunk_tokens']} | {c['new_seq']} | {c['completes_request']} | {c['batch_class']} | {c['n']} | "
                  f"{c['median_ms']:.2f} | {c['p10_ms']:.2f} | {c['p90_ms']:.2f} | {c['tok_per_s']:.0f} |")
    if a.tail:
        rows, summary, fits = tail(a.raw)
        if summary:
            write_csv(os.path.join(a.out, "tail.csv"), summary)
        json.dump({"rows": rows, "summary": summary, "fit_group_medians": fits},
                  open(os.path.join(a.out, "tail.json"), "w"), indent=1)
        print("\n| tail tokens | batch | n | final step median ms | min | max |\n|---|---|---|---|---|---|")
        for t in summary:
            print(f"| {t['tail_tokens']} | {t['batch_class']} | {t['n']} | {t['median_ms']:.1f} | {t['min_ms']:.1f} | {t['max_ms']:.1f} |")
        for bc, v in fits.items():
            print(f"fit {bc}: final_step_ms = {v['fixed_ms']:.1f} + {v['per_token_ms'] * 1e3:.2f} ms per 1000 tokens")


if __name__ == "__main__":
    main()
