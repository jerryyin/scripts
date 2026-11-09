#!/usr/bin/env python3
"""Aggregate raw/<L*_B*_rep*>/metrics.json into results.csv, results.json (rows + summary),
summary.csv, and a Markdown table (stdout, and --markdown FILE).

  fbdw_aggregate.py --raw raw --out . [--dp 4] [--node-note "..."] [--markdown summary-table.md]

--dp is the number of data-parallel ranks (for the idle-rank column); by default it is inferred
as one more than the highest rank seen in any point. The table gains an avg_accept_len column
when any point reports one (speculative decoding).
"""
import argparse, csv, glob, json, os, re, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--node-note", default="")
    ap.add_argument("--no-foreign-gpu-process", default="true")
    ap.add_argument("--dp", type=int, help="data-parallel ranks; default: inferred from the points")
    ap.add_argument("--markdown", help="also write the summary table to this file")
    a = ap.parse_args()
    dp = a.dp or 1 + max((int(r) for mf in glob.glob(os.path.join(a.raw, "L*_B*_rep*", "metrics.json"))
                          for r in json.load(open(mf)).get("per_rank", {})), default=0)
    rows = []
    for d in sorted(glob.glob(os.path.join(a.raw, "L*_B*_rep*"))):
        m = re.fullmatch(r"L(\d+)_B(\d+)_rep(\d+)", os.path.basename(d))
        if not m:
            continue  # tagged runs (smoke/warmup) are not ladder rows
        L, B, rep = map(int, m.groups())
        mf = os.path.join(d, "metrics.json")
        if not os.path.exists(mf):
            rows.append({"L": L, "B": B, "rep": rep, "valid": False, "invalid_reason": "no metrics.json"}); continue
        x = json.load(open(mf))
        pr = x.get("per_rank", {})
        ranks = sorted(pr, key=int)
        row = {"L": L, "B": B, "rep": rep,
               "active_ranks": ";".join(ranks),
               "idle_ranks": ";".join(str(r) for r in range(dp) if str(r) not in pr),
               "split_from_prefill": ";".join(f"{r}:{pr[r]['split_from_prefill']:.3f}" for r in ranks),
               "batch_per_rank": ";".join(f"{r}:{pr[r]['expected_batch']}" for r in ranks),
               "decode_running_req_values": ";".join(f"{r}:{'/'.join(map(str, pr[r]['running_req_values']))}" for r in ranks),
               "qualifying_rounds": ";".join(f"{r}:{pr[r]['qualifying_rounds']}" for r in ranks),
               "rate_per_rank_tok_s": ";".join(f"{r}:{pr[r].get('rate_tok_s', float('nan')):.1f}" for r in ranks),
               "system_tok_s": x.get("system_tok_s"), "tpot_ms_mean": x.get("tpot_ms_mean"),
               "tpot_ms_max": x.get("tpot_ms_max"), "client_over_server": x.get("client_over_server"),
               "client_over_server_per_chunk_retok": x.get("client_over_server_system_tok_s_per_chunk_retok"),
               "client_chunks_over_server": x.get("client_over_server_system_chunks_s"),
               "avg_seq_len_min": x.get("avg_seq_len_min"), "avg_seq_len_max": x.get("avg_seq_len_max"),
               "avg_accept_len": x.get("avg_accept_len_mean"),
               "prefill_tokens_total": x.get("prefill_tokens_total"),
               "prefill_tokens_expected": x.get("prefill_tokens_expected"),
               "requeststats_prompt_tokens": "/".join(map(str, x.get("requeststats_prompt_tokens", []))),
               "fixed_batch": x.get("fixed_batch"), "enough_rounds": x.get("enough_rounds"),
               "no_foreign_gpu_process": a.no_foreign_gpu_process == "true",
               "node_note": a.node_note, "stop_reason": x["meta"]["stop_reason"],
               "idle_confirmed": x["meta"]["idle_confirmed"],
               "flush_cache_status": (x["meta"].get("flush_cache") or {}).get("status")}
        reasons = []
        if not row["fixed_batch"]: reasons.append("fixed_batch=false")
        if not row["enough_rounds"]: reasons.append("enough_rounds=false")
        if row["prefill_tokens_total"] != row["prefill_tokens_expected"]: reasons.append("prefill token sum != B*L")
        if not row["no_foreign_gpu_process"]: reasons.append("foreign GPU process")
        row["valid"] = not reasons; row["invalid_reason"] = "; ".join(reasons)
        rows.append(row)
    rows.sort(key=lambda r: (r["L"], r["B"], r["rep"]))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summ = []
    for L, B in sorted({(r["L"], r["B"]) for r in rows}):
        rs = [r for r in rows if r["L"] == L and r["B"] == B and r.get("valid")]
        s = {"L": L, "B": B, "reps_valid": len(rs), "reps_total": len([r for r in rows if r["L"] == L and r["B"] == B])}
        for k in ("system_tok_s", "tpot_ms_mean", "tpot_ms_max", "client_over_server", "client_over_server_per_chunk_retok",
                  "avg_accept_len"):
            v = [r[k] for r in rs if r.get(k) is not None]
            if v:
                s[k + "_median"] = statistics.median(v); s[k + "_min"] = min(v); s[k + "_max"] = max(v)
        if rs:
            s["batch_per_rank"] = sorted({r["batch_per_rank"] for r in rs})
            s["avg_seq_len_range"] = [min(r["avg_seq_len_min"] for r in rs), max(r["avg_seq_len_max"] for r in rs)]
        summ.append(s)
    json.dump({"rows": rows, "summary": summ}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        ks = []
        for s in summ:
            for k in s:
                if k not in ks: ks.append(k)
        w = csv.DictWriter(f, fieldnames=ks); w.writeheader(); w.writerows(summ)
    table = markdown_table(summ)
    print(table, end="")
    if a.markdown:
        with open(a.markdown, "w") as fh:
            fh.write(table)


def markdown_table(summ):
    spec = any("avg_accept_len_median" in s for s in summ)
    rng = lambda s, k, p: f"{s[k + '_median']:.{p}f} [{s[k + '_min']:.{p}f}, {s[k + '_max']:.{p}f}]"
    lines = ["| L | B | reps ok | sys tok/s med [min,max] | TPOT mean ms med [min,max] | TPOT max ms med | "
             "client/server med | batch/rank |" + (" avg_accept_len med [min,max] |" if spec else ""),
             "|---|---|---|---|---|---|---|---|" + ("---|" if spec else "")]
    for s in summ:
        if "system_tok_s_median" in s:
            row = (f"| {s['L']} | {s['B']} | {s['reps_valid']}/{s['reps_total']} | {rng(s, 'system_tok_s', 1)} | "
                   f"{rng(s, 'tpot_ms_mean', 2)} | {s['tpot_ms_max_median']:.2f} | "
                   f"{s.get('client_over_server_median', float('nan')):.3f} | {' '.join(s['batch_per_rank'])} |")
            if spec:
                row += f" {rng(s, 'avg_accept_len', 2)} |" if "avg_accept_len_median" in s else " - |"
        else:
            row = f"| {s['L']} | {s['B']} | 0/{s['reps_total']} | - | - | - | - | - |" + (" - |" if spec else "")
        lines.append(row)
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    main()
