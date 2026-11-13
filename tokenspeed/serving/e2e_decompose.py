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
Per rank it also reports residency (max running requests, max queued requests on Decode lines,
i.e. waiting while the rank decodes) and the gap between
consecutive Decode lines at the rank's full batch with no prefill between them (one log interval
of decode rounds; --decode-log-interval converts it to a per-round time).

With --benchmark-json (amd-tracking run_evalscope_matrix.py output) the whole server log is
cut into one slice per phase first, from the phases' recorded start and finish times.

--predict models a closed-loop benchmark of R requests of L prompt and O output tokens at
concurrency C (R/C waves) from the fixed-batch measurements, and swaps measured terms between
configurations (what-ifs). Inputs are this tool's JSON of the measured runs (--runs CFG=JSON),
prefill ladders (--prefill NAME=BENCHMARK_JSON: an amd-tracking run_evalscope_matrix.py
--prefill-summary benchmark.json; its prompt_tokens must be L and its prefill_summary row of
concurrency C gives prefill_tok_per_s median) and decode times per output token (--tpot
NAME=CFG:MS, a number, or NAME=CFG:CSV@POINT, the decode_tpot_ms_median of row POINT of the
table written by amd-tracking report_retained_kv.py RUN --repetitions-only CSV):
  prediction(NAME)   = R*L / prefill tok/s(CFG) + waves*(O-1)*MS
Per configuration the medians over its runs of the busiest rank's prefill P, decode D, idle I and
the Decode-line step are taken; core = waves*(O-1)*step/acceptance (acceptance = RequestStats mean,
1 without speculation) and tail = D - core (ramp and drain). Every what-if keeps the terms it does
not swap:
  X as measured                      P_X + D_X + I_X
  X with Y's end-to-end prefill      P_Y + D_X + I_X        (and decode: P_X + D_Y + I_X)
  X with prefill table T's rate      R*L / tok/s(T) + D_X + I_X
  X, decode at NAME                  P_X + waves*(O-1)*MS + tail_X + I_X
  X, acceptance A                    P_X + waves*(O-1)/A*step_X + tail_X + I_X   (--acceptance X=A)

Error: per prefill->decode boundary at most |actual chunk - c_med| (chunk times
are tight, see chunk_time stats); the tail estimate is bounded by
decode_log_interval rounds. Mixed batches (if any) are counted as prefill time.
"""
import argparse, collections, csv, datetime as dt, hashlib, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_log import parse_lines, parse_ts  # noqa: E402


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
        accs = [float(s["acc_len"]) for s in st if isinstance(s.get("acc_len"), float)]
        if accs:
            res["requeststats"]["acc_len"] = {"median": statistics.median(accs), "min": min(accs), "max": max(accs),
                                              "n_below_3": sum(1 for x in accs if x < 3)}
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
        full = max(bs) if bs else 0
        # consecutive Decode lines at the full batch with no Prefill/Mix line between them
        fb = [(g, b) for a, b, g in gaps if a["kind"] == b["kind"] == "Decode"
              and a.get("running") == full and b.get("running") == full]
        fb_g = [g for g, _ in fb]
        fb_acc = [b["avg_accept_len"] for _, b in fb if b.get("avg_accept_len") is not None]
        res["ranks"][str(r)] = {
            "max_running": max((e.get("running", 0) for e in L), default=0),
            "max_queue_on_decode_lines": max((e.get("queue", 0) for e in dec), default=0),
            "full_batch": full,
            "full_batch_line_gap_s": {"n": len(fb_g), "median": statistics.median(fb_g) if fb_g else None,
                                      "min": min(fb_g, default=None), "max": max(fb_g, default=None)},
            "full_batch_accept_len_line_mean": statistics.mean(fb_acc) if fb_acc else None,
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


def phase_windows(benchmark_json):
    """(phase, output_dir, start, end) epoch windows of an amd-tracking EvalScope matrix run, from
    each result's started_at/finished_at."""
    bj = json.load(open(benchmark_json))
    missing = [r.get("phase") for r in bj["results"] if "started_at" not in r or "finished_at" not in r]
    if missing:
        raise SystemExit(f"{benchmark_json}: phases {missing} have no started_at/finished_at "
                         "(recorded by amd-tracking's run_evalscope_matrix.py)")
    return [(r["phase"], r["output_dir"], parse_iso(r["started_at"]), parse_iso(r["finished_at"]))
            for r in bj["results"]]


def parse_iso(text):
    text = text.rstrip("Z")
    head, _, frac = text.partition(".")
    value = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return value + (float("0." + frac) if frac else 0.0)


def write_slices(logs, windows, out_dir):
    """Write the lines of the server logs that fall in each window to out_dir/PHASE.log."""
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for log in logs:
        for raw in open(log, errors="replace"):
            t = parse_ts(raw)[0]
            if t is not None:
                lines.append((t, raw if raw.endswith("\n") else raw + "\n"))
    lines.sort(key=lambda x: x[0])
    paths = []
    for phase, _, a, b in windows:
        path = os.path.join(out_dir, f"{phase}.log")
        with open(path, "w") as fh:
            fh.writelines(text for t, text in lines if a <= t and (b is None or t < b))
        paths.append(path)
    return paths


def markdown(runs, log_interval=None):
    head = ("| run | span s | requests per rank | busiest rank prefill s = chunks x median ms (share) | decode s (share) "
            "| Decode-line batch | full-batch round ms | acceptance (lines; RequestStats) | idle s | prefill->decode "
            "boundaries |")
    lines = [head, "|" + "---|" * 10]
    for r in runs:
        R = r["ranks"]
        k = max(R, key=lambda q: R[q]["prefill_time_s"] + R[q]["decode_time_s"])
        b = R[k]; span = r["server_span_s"]
        g = b["full_batch_line_gap_s"]["median"]
        rnd = f"{g / log_interval * 1e3:.1f}" if g and log_interval else "-"
        acc = [R[q]["avg_accept_len_line_mean"] for q in sorted(R)]
        rs = (r.get("requeststats") or {}).get("acc_len_mean")
        acc_text = ", ".join("-" if x is None else f"{x:.2f}" for x in acc) + ("" if rs is None else f"; {rs:.2f}")
        split = "/".join(f"{R[q]['requests_by_prefill_tokens']:g}" for q in sorted(R))
        lines.append(
            f"| {r['label']} | {span:.2f} | {split} | "
            f"{b['prefill_time_s']:.2f} = {b['prefill_chunks']} x {b['chunk_time_s']['median'] * 1e3:.1f} "
            f"({b['prefill_time_s'] / span * 100:.1f}%) | {b['decode_time_s']:.2f} ({b['decode_time_s'] / span * 100:.1f}%) | "
            f"{b['full_batch']} | {rnd} | {acc_text} | {b['idle_s']:.2f} | {b['prefill_to_decode_boundaries']} |")
    return "\n".join(lines) + "\n"


def prefill_rate(path, prompt_tokens, batch):
    """Median prefill tok/s at concurrency `batch` from a run_evalscope_matrix.py --prefill-summary
    benchmark.json whose prompt_tokens is `prompt_tokens`."""
    bj = json.load(open(path))
    if "prefill_summary" not in bj:
        raise SystemExit(f"{path}: no prefill_summary (run run_evalscope_matrix.py with --prefill-summary)")
    if bj.get("prompt_tokens") != prompt_tokens:
        raise SystemExit(f"{path}: prompt_tokens {bj.get('prompt_tokens')} != {prompt_tokens}")
    for r in bj["prefill_summary"]:
        if int(r["concurrency"]) == batch:
            rate = (r.get("prefill_tok_per_s") or {}).get("median")
            if rate is None:
                raise SystemExit(f"{path}: concurrency {batch} has no prefill_tok_per_s median")
            return float(rate)
    raise SystemExit(f"{path}: no prefill_summary row of concurrency {batch}")


def tpot_ms(value):
    """MS as a number, or CSV@POINT: decode_tpot_ms_median of row POINT of a report_retained_kv.py
    --repetitions-only table."""
    path, sep, point = value.rpartition("@")
    if not sep:
        return float(value)
    for r in csv.DictReader(open(path)):
        if r["point"] == point:
            if not r.get("decode_tpot_ms_median"):
                raise SystemExit(f"{path}: point {point} has no decode_tpot_ms_median")
            return float(r["decode_tpot_ms_median"])
    raise SystemExit(f"{path}: no point {point}")


def run_terms(runs, interval, steps):
    """Medians over runs of the busiest rank's prefill, decode and idle time and Decode-line step."""
    if any(not r["ranks"] for r in runs):
        raise SystemExit("a run has no ranks")
    b = [max(r["ranks"].values(), key=lambda x: x["prefill_time_s"] + x["decode_time_s"]) for r in runs]
    if any(x["decode_line_interval_median_s"] is None for x in b):
        raise SystemExit("a run's busiest rank has no consecutive Decode lines")
    accs = [(r.get("requeststats") or {}).get("acc_len_mean") for r in runs]
    t = {"span": statistics.median(r["server_span_s"] for r in runs),
         "prefill": statistics.median(x["prefill_time_s"] for x in b),
         "decode": statistics.median(x["decode_time_s"] for x in b),
         "idle": statistics.median(x["idle_s"] for x in b),
         "step_ms": statistics.median(x["decode_line_interval_median_s"] / interval * 1e3 for x in b),
         "acceptance": statistics.median(accs) if all(accs) else None}
    t["tail"] = t["decode"] - steps / (t["acceptance"] or 1) * t["step_ms"] / 1e3
    return t


def predict(runs, prefill, tpots, acceptance, requests, prompt_tokens, output_tokens, concurrency,
            interval):
    """Prediction rows and what-ifs (see the module docstring); returns a JSON-able dict."""
    if requests % concurrency:
        raise SystemExit(f"--requests {requests} is not a whole number of waves of {concurrency}")
    steps = requests // concurrency * (output_tokens - 1)
    work = requests * prompt_tokens
    rate = {name: prefill_rate(path, prompt_tokens, concurrency) for name, path in prefill.items()}
    terms = {cfg: run_terms(rs, interval, steps) for cfg, rs in runs.items()}
    preds = []
    for name, cfg, ms in tpots:
        if cfg not in rate:
            raise SystemExit(f"--tpot {name}: no --prefill table named {cfg}")
        pre, dec = work / rate[cfg], steps * ms / 1e3
        preds.append({"name": name, "config": cfg, "prefill_s": pre, "prefill_tok_s": rate[cfg],
                      "decode_s": dec, "span_s": pre + dec, "req_s": requests / (pre + dec),
                      "residuals_s": [r["server_span_s"] - pre - dec for r in runs.get(cfg, [])]})
    whatif = []
    for x, t in terms.items():
        whatif.append((f"{x} as measured (model check)", t["prefill"] + t["decode"] + t["idle"]))
    for x, t in terms.items():
        for y, u in terms.items():
            if x != y:
                whatif.append((f"{x} with {y}'s end-to-end prefill time", u["prefill"] + t["decode"] + t["idle"]))
                whatif.append((f"{x} with {y}'s end-to-end decode time", t["prefill"] + u["decode"] + t["idle"]))
        for name, r in rate.items():
            if name != x:
                whatif.append((f"{x} with {name}'s prefill-ladder rate", work / r + t["decode"] + t["idle"]))
    for name, cfg, ms in tpots:
        if cfg in terms:
            t = terms[cfg]
            whatif.append((f"{cfg}, decode at {name} ({ms:g} ms per output token)",
                           t["prefill"] + steps * ms / 1e3 + t["tail"] + t["idle"]))
    for cfg, a in acceptance.items():
        if cfg not in terms:
            raise SystemExit(f"--acceptance {cfg}: no --runs named {cfg}")
        t = terms[cfg]
        whatif.append((f"{cfg}, acceptance {a:g} at the measured step time",
                       t["prefill"] + steps / a * t["step_ms"] / 1e3 + t["tail"] + t["idle"]))
    return {"workload": {"requests": requests, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                         "concurrency": concurrency, "decode_steps": steps},
            "terms": terms, "predictions": preds,
            "whatif": [{"name": n, "span_s": v, "req_s": requests / v} for n, v in whatif]}


def predict_markdown(res):
    lines = ["| prediction | prefill term s | decode term s | predicted span s / req/s | residuals (measured - predicted) s |",
             "|---|---|---|---|---|"]
    for p in res["predictions"]:
        lines.append(f"| {p['name']} | {p['prefill_s']:.2f} ({p['prefill_tok_s']:.0f} tok/s) | {p['decode_s']:.2f} | "
                     f"{p['span_s']:.1f} / {p['req_s']:.3f} | {' / '.join(f'{x:+.1f}' for x in p['residuals_s'])} |")
    lines += ["", "| what-if (ESTIMATE) | span s | req/s |", "|---|---|---|"]
    lines += [f"| {w['name']} | {w['span_s']:.2f} | {w['req_s']:.3f} |" for w in res["whatif"]]
    return "\n".join(lines) + "\n"


def split_spec(spec, flag):
    name, sep, value = spec.partition("=")
    if not sep or not name or not value:
        raise SystemExit(f"{flag} {spec!r}: expected NAME=VALUE")
    return name, value


def main_predict(a):
    runs = {}
    for spec in a.runs:
        cfg, path = split_spec(spec, "--runs")
        runs.setdefault(cfg, []).extend(json.load(open(path)))
    prefill = dict(split_spec(x, "--prefill") for x in a.prefill)
    tpots = []
    for spec in a.tpot:
        name, rest = split_spec(spec, "--tpot")
        cfg, sep, ms = rest.partition(":")
        if not sep:
            raise SystemExit(f"--tpot {spec!r}: expected NAME=CFG:MS or NAME=CFG:CSV@POINT")
        tpots.append((name, cfg, tpot_ms(ms)))
    acceptance = {k: float(v) for k, v in (split_spec(x, "--acceptance") for x in a.acceptance)}
    res = predict(runs, prefill, tpots, acceptance, a.requests, a.prompt_tokens, a.output_tokens,
                  a.concurrency, a.decode_log_interval)
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1)
    print(predict_markdown(res), end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", default=[], help="server log slice (repeatable)")
    ap.add_argument("--evalscope", action="append", default=[], help="evalscope perf output dir per --log")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--benchmark-json", help="amd-tracking EvalScope matrix benchmark.json: slice the --server-log "
                    "per phase into --slices-dir and decompose every phase")
    ap.add_argument("--server-log", action="append", default=[], help="whole server log(s) for --benchmark-json")
    ap.add_argument("--slices-dir")
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="request prompt length (default: the most common one >= 1000); with --predict, L")
    ap.add_argument("--decode-log-interval", type=int, help="rounds per Decode line, for per-round times")
    ap.add_argument("--markdown", action="store_true", help="also print one table row per run")
    ap.add_argument("--out-json")
    pr = ap.add_argument_group("--predict")
    pr.add_argument("--predict", action="store_true", help="prediction and what-ifs instead of decomposition")
    pr.add_argument("--runs", action="append", default=[], help="CFG=JSON written by this tool (repeatable)")
    pr.add_argument("--prefill", action="append", default=[], help="NAME=run_evalscope_matrix.py --prefill-summary benchmark.json")
    pr.add_argument("--tpot", action="append", default=[], help="NAME=CFG:MS or NAME=CFG:CSV@POINT (report_retained_kv.py --repetitions-only table): decode time per output token")
    pr.add_argument("--acceptance", action="append", default=[], help="CFG=A: what-if acceptance")
    pr.add_argument("--requests", type=int)
    pr.add_argument("--output-tokens", type=int)
    pr.add_argument("--concurrency", type=int)
    a = ap.parse_args()
    if a.predict:
        missing = [f for f in ("runs", "prefill", "requests", "prompt_tokens", "output_tokens", "concurrency",
                               "decode_log_interval") if not getattr(a, f)]
        if missing:
            ap.error("--predict needs " + ", ".join("--" + f.replace("_", "-") for f in missing))
        main_predict(a)
        return
    if not a.out_json:
        ap.error("--out-json is required")
    if a.benchmark_json:
        if not a.server_log or not a.slices_dir:
            ap.error("--benchmark-json needs --server-log and --slices-dir")
        windows = phase_windows(a.benchmark_json)
        a.log = write_slices(a.server_log, windows, a.slices_dir)
        a.label = [w[0] for w in windows]
        a.evalscope = [w[1] for w in windows]
    if not a.log:
        ap.error("give --log or --benchmark-json")
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
    if a.markdown:
        print(markdown(runs, a.decode_log_interval), end="")


if __name__ == "__main__":
    main()
