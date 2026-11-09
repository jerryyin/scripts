#!/usr/bin/env python3
"""Aggregate raw/L*_B*_rep*/metrics.json into results.json/summary.csv (median/min/max over
valid reps per (L,B)) and print a markdown table.  usage: prefill_aggregate.py --raw raw --out ."""
import argparse, csv, glob, json, os, statistics

KEYS = {"ttft_mean_s": lambda m: m["ttft_s"]["mean"], "ttft_max_s": lambda m: m["ttft_s"]["max"],
        "batch_prompt_tok_per_s": lambda m: m["batch_prompt_tok_per_s"],
        "server_tok_per_s": lambda m: m["server"]["server_tok_per_s_all_ranks"],
        "chunk_time_median_ms": lambda m: 1e3 * statistics.median(
            [r["chunk_time_median_s"] for r in m["server"]["ranks"].values() if r["chunk_time_median_s"] is not None] or [float("nan")]),
        "chunks_max_rank": lambda m: max(r["chunks"] for r in m["server"]["ranks"].values()),
        "requests_max_rank": lambda m: max(r["requests_by_tokens"] for r in m["server"]["ranks"].values())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
