#!/usr/bin/env python3
"""Decompose one serving run (e.g. evalscope perf 128 x 50000/500 at c64) from a
TokenSpeed server log slice into per-DP-rank prefill / decode / idle time.

Works for DP>1 logs (#dp-rank) and TP single-rank logs (no #dp-rank -> rank 0),
reads avg_accept_len when present (speculative configs).

Line semantics (tokenspeed-engine/crates/ts-control/src/batch_log.rs):
  * a line is emitted when a round is DISPATCHED (log_dispatch), so a line's
    timestamp is the start of its round;
  * every round with extends logs "Prefill batch." (all extends) or "Mix batch.";
  * a decode round logs "Decode batch." only when the per-rank round counter
    (which counts ALL rounds) is a multiple of --decode-log-interval
    (server_args.py default 40); its "gen throughput" = tokens committed by
    decode rounds since the previous Decode line / wall time since that line.

Attribution rule (per rank, consecutive lines a -> b of that rank, gap = b.ts - a.ts):
  * a = Prefill/Mix and b = Prefill/Mix: whole gap is prefill (one chunk round);
  * a = Prefill/Mix and b = Decode or end: prefill gets min(gap, c_med) where c_med
    is the rank's median Prefill->Prefill gap for that chunk size, the rest is
    decode (unlogged decode rounds follow the chunk);
  * a = Decode, b = Decode: whole gap is decode (any prefill round would have logged a line);
  * a = Decode, b = Prefill/Mix: decode gets min(gap, median Decode->Decode gap of the
    rank) (at most one log interval of unlogged decode rounds), the rest is
    'idle_between_waves' (rank empty, waiting for the client's next requests);
  * rank tail after its last line: decode until the rank's estimated end
    = last line + remaining decode tokens / last gen throughput, capped at run end;
    remaining tokens = rank decode tokens - tokens accounted by Decode lines.
  * idle = run span - prefill - decode (start lag + end lag).
Error: per prefill->decode boundary at most |actual chunk - c_med| (chunk times
are tight, see chunk_time stats); the tail estimate is bounded by
decode_log_interval rounds. Mixed batches (if any) are counted as prefill time.
"""
import argparse, collections, hashlib, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_log import parse_lines  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs); k = (len(xs) - 1) * q; i = int(k)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (k - i)


def decompose(log, req_prompt=None, start_min_tokens=512, out_tokens=None):
    ev = parse_lines(open(log, errors="replace"))
    fin = [e for e in ev if e["kind"] == "Finish"]
    if req_prompt is None:  # most common prompt length among finished requests >= 1000 tokens
        c = collections.Counter(int(e["stats"].get("prompt_tokens", 0)) for e in fin
                                if int(e["stats"].get("prompt_tokens", 0)) >= 1000)
        req_prompt = c.most_common(1)[0][0] if c else None
    run_fin = [e for e in fin if req_prompt and int(e["stats"].get("prompt_tokens", 0)) == req_prompt]
    batch = [e for e in ev if e["kind"] in ("Prefill", "Mix", "Decode") and e["ts"] is not None]
    starts = [e["ts"] for e in batch if e["kind"] in ("Prefill", "Mix") and e.get("new_token", 0) >= start_min_tokens]
    t0 = min(starts)
    t1 = max(e["ts"] for e in run_fin) if run_fin else max(e["ts"] for e in batch)
    if out_tokens is None:
        out_tokens = statistics.median([int(e["stats"]["output_tokens"]) for e in run_fin]) if run_fin else 0
    allbatch = batch
    batch = [e for e in batch if t0 <= e["ts"] <= t1]
    ranks = sorted({e["dp_rank"] for e in batch})
    res = {"log": log, "log_sha256": sha256(log), "request_prompt_tokens": req_prompt,
           "requests_finished_in_log": len(run_fin), "output_tokens_per_request": out_tokens,
           "run_start_utc_epoch": t0, "run_end_utc_epoch": t1, "server_span_s": t1 - t0,
           "run_start_rule": f"first Prefill/Mix line with #new-token >= {start_min_tokens}",
           "run_end_rule": f"last 'Finish! RequestStats' line with prompt_tokens == {req_prompt}",
           "ranks": {}}
    if run_fin:
        st = [e["stats"] for e in run_fin]
        res["requeststats"] = {
            "n": len(st), "prompt_tokens": dict(collections.Counter(int(s["prompt_tokens"]) for s in st)),
            "output_tokens": dict(collections.Counter(int(s["output_tokens"]) for s in st)),
            "cache_tokens": dict(collections.Counter(int(s["cache_tokens"]) for s in st)),
            "prefill_ms_mean": statistics.mean(float(s.get("prefill_ms", 0)) for s in st),
            "prefill_ms_median": statistics.median(float(s.get("prefill_ms", 0)) for s in st),
            "decode_tps_mean": statistics.mean(float(s.get("decode_tps", 0)) for s in st),
            "acc_len_mean": (statistics.mean(float(s["acc_len"]) for s in st if isinstance(s.get("acc_len"), float))
                             if any(isinstance(s.get("acc_len"), float) for s in st) else None)}
    for r in ranks:
        L = [e for e in batch if e["dp_rank"] == r]
        Lall = [e for e in allbatch if e["dp_rank"] == r]
        pf = [e for e in L if e["kind"] in ("Prefill", "Mix")]
        dec = [e for e in L if e["kind"] == "Decode"]
        ptok = sum(e.get("new_token", 0) - (e.get("running", 0) - e.get("new_seq", 0) if e["kind"] == "Mix" else 0) for e in pf)
        nreq = round(ptok / req_prompt, 2) if req_prompt else None
        # chunk time by chunk size from Prefill->Prefill gaps
        bysize = collections.defaultdict(list); gaps = []
        for a, b in zip(L, L[1:]):
            g = b["ts"] - a["ts"]; gaps.append((a, b, g))
            if a["kind"] in ("Prefill", "Mix") and b["kind"] in ("Prefill", "Mix"):
                bysize[a.get("new_token", 0)].append(g)
        allc = [g for v in bysize.values() for g in v]
        cmed_all = statistics.median(allc) if allc else 0.0
        def cmed(n):
            v = bysize.get(n)
            return statistics.median(v) if v and len(v) >= 3 else cmed_all
        dd = [g for a, b, g in gaps if a["kind"] == "Decode" and b["kind"] == "Decode"]
        dmed = statistics.median(dd) if dd else None
        tp = td = gap_idle = 0.0; nb = 0; stall_n = 0; stall_s = 0.0
        for a, b, g in gaps:
            if a["kind"] in ("Prefill", "Mix"):
                if b["kind"] in ("Prefill", "Mix"):
                    tp += g
                    c = cmed(a.get("new_token", 0))
                    if g > 2 * c:
                        stall_n += 1; stall_s += g - c
                else:
                    c = min(g, cmed(a.get("new_token", 0))); tp += c; td += g - c; nb += 1
            elif b["kind"] == "Decode" or dmed is None:
                td += g
            else:  # Decode -> Prefill/Mix: at most one log interval of decode, rest waiting for arrivals
                c = min(g, dmed); td += c; gap_idle += g - c
        # decode tokens accounted by Decode lines (gap measured from the previous Decode line of the rank, any time)
        prev = None; acc_tok = 0.0; unref = 0
        for e in Lall:
            if e["kind"] != "Decode":
                continue
            if t0 <= e["ts"] <= t1:
                if prev is not None:  # gen throughput x time since previous Decode line = tokens (exact up to rounding)
                    acc_tok += e.get("gen_tput", 0.0) * (e["ts"] - prev["ts"])
                else:
                    unref += 1  # logger start time unknown: tokens not counted
            prev = e
        dec_tok_expected = (nreq or 0) * max(out_tokens - 1, 0)
        last = L[-1]
        if last["kind"] in ("Prefill", "Mix"):
            c = cmed(last.get("new_token", 0)); tp += c; tail_start = last["ts"] + c
        else:
            tail_start = last["ts"]
        last_tput = dec[-1].get("gen_tput") if dec else None
        remaining = max(dec_tok_expected - acc_tok, 0.0)
        tail = remaining / last_tput if last_tput else 0.0
        rank_end_uncapped = tail_start + tail
        rank_end = min(t1, rank_end_uncapped)
        td += max(rank_end - tail_start, 0.0)
        span = t1 - t0
        idle = span - tp - td
        bs = [e.get("running", 0) for e in dec]
        dts = [g for a, b, g in gaps if a["kind"] == "Decode"]
        # time-weighted decode batch size (decode lines hold until next line)
        tw = collections.Counter()
        for a, b, g in gaps:
            if a["kind"] == "Decode":
                tw[a.get("running", 0)] += g
        acc = [e.get("avg_accept_len") for e in dec if e.get("avg_accept_len") is not None]
        res["ranks"][str(r)] = {
            "first_line_lag_s": L[0]["ts"] - t0, "last_line_s": last["ts"] - t0,
            "estimated_rank_end_s": rank_end - t0, "end_lag_s": t1 - rank_end,
            "prefill_chunks": len(pf), "prefill_tokens": ptok, "requests_by_prefill_tokens": nreq,
            "chunk_tokens_hist": dict(collections.Counter(e.get("new_token", 0) for e in pf)),
            "chunk_time_s": {"median": cmed_all, "p10": pct(allc, .1), "p90": pct(allc, .9), "n": len(allc)},
            "chunk_time_by_size_median_s": {str(k): statistics.median(v) for k, v in sorted(bysize.items())},
            "prefill_time_s": tp, "prefill_time_chunk_model_s": len(pf) * cmed_all,
            "prefill_overhead_vs_chunk_model_s": tp - len(pf) * cmed_all, "prefill_tok_per_s_while_prefilling": ptok / tp if tp else None,
            "prefill_to_decode_boundaries": nb,
            "decode_time_s": td, "decode_tokens_expected": dec_tok_expected,
            "decode_tokens_logged": acc_tok, "decode_tok_per_s_while_decoding": dec_tok_expected / td if td else None,
            "decode_lines": len(dec), "decode_batch_size_lines": dict(collections.Counter(bs)),
            "decode_batch_size_time_weighted_s": {str(k): v for k, v in sorted(tw.items())},
            "avg_accept_len_line_mean": statistics.mean(acc) if acc else None,
            "idle_s": idle, "idle_between_waves_s": gap_idle, "idle_start_end_s": idle - gap_idle, "span_s": span,
            "decode_line_interval_median_s": dmed, "decode_lines_without_reference": unref,
            "estimated_rank_end_uncapped_s": rank_end_uncapped - t0, "tail_after_last_line_s": tail,
            "prefill_stalls": {"n_gaps_gt_2x_median": stall_n, "excess_s": stall_s},
            "fractions": {"prefill": tp / span, "decode": td / span, "idle": idle / span}}
    R = res["ranks"].values()
    res["sum"] = {"prefill_tokens": sum(x["prefill_tokens"] for x in R),
                  "requests_by_prefill_tokens": [x["requests_by_prefill_tokens"] for x in R],
                  "max_rank_prefill_plus_decode_s": max(x["prefill_time_s"] + x["decode_time_s"] for x in R),
                  "mean_prefill_time_s": statistics.mean(x["prefill_time_s"] for x in R),
                  "mean_decode_time_s": statistics.mean(x["decode_time_s"] for x in R),
                  "mean_idle_s": statistics.mean(x["idle_s"] for x in R)}
    res["sum"]["max_rank_prefill_plus_decode_plus_between_wave_idle_s"] = max(
        x["prefill_time_s"] + x["decode_time_s"] + x["idle_between_waves_s"] for x in R)
    return res


def evalscope(d):
    for root, _, names in os.walk(d):
        if "benchmark_summary.json" in names:
            p = os.path.join(root, "benchmark_summary.json")
            j = json.load(open(p)); j["_path"] = p; j["_sha256"] = sha256(p)
            return j
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, action="append", help="server log slice (repeatable)")
    ap.add_argument("--evalscope", action="append", default=[], help="evalscope perf output dir per --log")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--prompt-tokens", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    runs = []
    for i, lg in enumerate(a.log):
        r = decompose(lg, a.prompt_tokens)
        r["label"] = a.label[i] if i < len(a.label) else os.path.basename(lg)
        if i < len(a.evalscope):
            es = evalscope(a.evalscope[i]); r["evalscope"] = es
            if es:
                n = es["Total Requests"]
                pred = r["sum"]["max_rank_prefill_plus_decode_plus_between_wave_idle_s"]
                busiest = max(r["ranks"].values(), key=lambda x: x["prefill_time_s"] + x["decode_time_s"] + x["idle_between_waves_s"])
                model = busiest["prefill_time_chunk_model_s"] + busiest["decode_time_s"] + busiest["idle_between_waves_s"]
                r["reconciliation"] = {
                    "busiest_rank_model_s (chunks x median chunk + decode + between-wave idle)": model,
                    "span_minus_model_s (prefill stalls/overheads)": r["server_span_s"] - model,
                    "req_per_s_model": n / model,
                    "evalscope_test_duration_s": es["Test Duration (s)"], "server_span_s": r["server_span_s"],
                    "client_minus_server_s": es["Test Duration (s)"] - r["server_span_s"],
                    "predicted_busiest_rank_prefill_plus_decode_plus_between_wave_idle_s": pred,
                    "predicted_minus_server_span_s": pred - r["server_span_s"],
                    "req_per_s_evalscope": es["Req Throughput (req/s)"],
                    "req_per_s_server_span": n / r["server_span_s"],
                    "req_per_s_predicted_busiest_rank": n / pred}
        runs.append(r)
    json.dump(runs, open(a.out_json, "w"), indent=1, default=str)
    for r in runs:
        print("==", r["label"], "span %.2f s" % r["server_span_s"], json.dumps(r.get("reconciliation"), default=str))
        for k, x in r["ranks"].items():
            print(f" rank{k}: req={x['requests_by_prefill_tokens']} chunks={x['prefill_chunks']} "
                  f"prefill={x['prefill_time_s']:.2f}s ({x['prefill_tok_per_s_while_prefilling']:.0f} tok/s, chunk med {x['chunk_time_s']['median']*1e3:.1f} ms) "
                  f"decode={x['decode_time_s']:.2f}s ({x['decode_tok_per_s_while_decoding']:.1f} tok/s, logged {x['decode_tokens_logged']:.0f}/{x['decode_tokens_expected']}) "
                  f"idle={x['idle_s']:.2f}s (between waves {x['idle_between_waves_s']:.2f}) stalls={x['prefill_stalls']['n_gaps_gt_2x_median']}/{x['prefill_stalls']['excess_s']:.2f}s tail={x['tail_after_last_line_s']:.2f} lag0={x['first_line_lag_s']:.2f} end={x['estimated_rank_end_s']:.2f} bs={x['decode_batch_size_lines']}")


if __name__ == "__main__":
    main()
