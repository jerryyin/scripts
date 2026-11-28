#!/usr/bin/env python3
"""Decompose one serving run (e.g. an evalscope perf run of R requests of L/O tokens at concurrency C)
from a TokenSpeed server log slice into per-DP-rank prefill / decode / idle time.

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
  * with a given window (--window-from-aiperf) on a single-rank log (Finish lines carry no DP rank),
    a gap a -> b holding at least as many Finish lines as line a's running requests ends them all at
    the last of them (lf): prefill gets min(lf - a, c_med) when a is Prefill/Mix, decode the rest of
    [a, lf], and (lf, b] is 'idle_between_waves'; c_med then comes only from Prefill->Prefill gaps
    without a Finish line (chunks of one request). Such gaps are counted per rank in finish_split_gaps
    (n; from_prefill, those starting at a Prefill/Mix line, e.g. a turn shorter than one Decode-line
    interval; and the prefill, decode and idle seconds they were given).
Per rank it also reports residency (max running requests, max queued requests on Decode lines,
i.e. waiting while the rank decodes) and the gap between
consecutive Decode lines at the rank's full batch with no prefill between them (one log interval
of decode rounds; --decode-log-interval converts it to a per-round time).

With --benchmark-json (amd-tracking run_evalscope_matrix.py output) the whole server log is
cut into one slice per phase first, from the phases' recorded start and finish times; a phase
the matrix repeats across concurrencies is named PHASE/cB (slice PHASE_cB.log), one per window.

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
import argparse, bisect, collections, csv, datetime as dt, hashlib, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_log import parse_lines, parse_ts, quantile as pct  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def decompose(log, req_prompt=None, start_min_tokens=512, out_tokens=None, window=None):
    """window=(t0, t1) epoch seconds overrides the run start/end rules (e.g. an AIPerf profile window)."""
    ev = parse_lines(open(log, errors="replace"))
    fin = [e for e in ev if e["kind"] == "Finish"]
    if req_prompt is None:  # most common prompt length among finished requests >= 1000 tokens
        c = collections.Counter(int(e["stats"].get("prompt_tokens", 0)) for e in fin
                                if int(e["stats"].get("prompt_tokens", 0)) >= 1000)
        req_prompt = c.most_common(1)[0][0] if c else None
    run_fin = [e for e in fin if req_prompt and int(e["stats"].get("prompt_tokens", 0)) == req_prompt]
    batch = [e for e in ev if e["kind"] in ("Prefill", "Mix", "Decode") and e["ts"] is not None]
    starts = [e["ts"] for e in batch if e["kind"] in ("Prefill", "Mix") and e.get("new_token", 0) >= start_min_tokens]
    if window:
        t0, t1 = window
    else:
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
           "run_start_rule": "given window" if window else f"first Prefill/Mix line with #new-token >= {start_min_tokens}",
           "run_end_rule": "given window" if window else f"last 'Finish! RequestStats' line with prompt_tokens == {req_prompt}",
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
        # window mode (agentic runs of variable-length requests): a Finish line between two lines of a
        # single-rank log ends a request there, so the gap is not one chunk (or one log interval of decode)
        fts = sorted(e["ts"] for e in fin if e["ts"] is not None) if (window and len(ranks) == 1) else []
        def last_finish(x, y):
            i = bisect.bisect_left(fts, y) - 1
            return fts[i] if i >= 0 and fts[i] > x else None
        def all_finished(a, b):  # last Finish in (a, b) when every request running at a finished there
            n = bisect.bisect_left(fts, b["ts"]) - bisect.bisect_right(fts, a["ts"])
            return last_finish(a["ts"], b["ts"]) if n and n >= max(a.get("running", 1), 1) else None
        for a, b in zip(L, L[1:]):
            g = b["ts"] - a["ts"]; gaps.append((a, b, g))
            if a["kind"] in ("Prefill", "Mix") and b["kind"] in ("Prefill", "Mix") and last_finish(a["ts"], b["ts"]) is None:
                bysize[a.get("new_token", 0)].append(g)
        allc = [g for v in bysize.values() for g in v]
        cmed_all = statistics.median(allc) if allc else 0.0
        def cmed(n):
            v = bysize.get(n)
            return statistics.median(v) if v and len(v) >= 3 else cmed_all
        dd = [g for a, b, g in gaps if a["kind"] == "Decode" and b["kind"] == "Decode"]
        dmed = statistics.median(dd) if dd else None
        tp = td = gap_idle = 0.0; nb = 0; stall_n = 0; stall_s = 0.0
        fsplit = {"n": 0, "from_prefill": 0, "prefill_s": 0.0, "decode_s": 0.0, "idle_s": 0.0}
        for a, b, g in gaps:
            lf = all_finished(a, b) if fts else None
            if lf is not None:  # requests ended at lf: [a, lf] is its chunk and/or decode, (lf, b] waits for arrivals
                c = min(lf - a["ts"], cmed(a.get("new_token", 0))) if a["kind"] in ("Prefill", "Mix") else 0.0
                tp += c; td += lf - a["ts"] - c; gap_idle += b["ts"] - lf
                fsplit["n"] += 1; fsplit["from_prefill"] += a["kind"] in ("Prefill", "Mix")
                fsplit["prefill_s"] += c; fsplit["decode_s"] += lf - a["ts"] - c; fsplit["idle_s"] += b["ts"] - lf
                if a["kind"] in ("Prefill", "Mix") and b["kind"] not in ("Prefill", "Mix"):
                    nb += 1
                continue
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
            "finish_split_gaps": fsplit if fts else None,
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
    if window:  # the window's Finish lines (and those up to 1 s after it)
        wfin = [e for e in fin if e["ts"] and window[0] <= e["ts"] <= window[1] + 1.0]
        accs = [float(e["stats"]["acc_len"]) for e in wfin if isinstance(e["stats"].get("acc_len"), float)
                and e["stats"]["acc_len"] > 0]
        res["window_finish_lines"] = {"n": len(wfin), "acc_len_mean_nonzero": statistics.mean(accs) if accs else None,
                                      "n_acc_len_zero": sum(1 for e in wfin if e["stats"].get("acc_len") == 0)}
    return res


CTX_EDGES = (0, 32768, 65536, 131072, 262144, 524288, 1048576 + 64)


def ctx_bucket(n):
    for lo, hi in zip(CTX_EDGES, CTX_EDGES[1:]):
        if lo <= n < hi:
            return f"{lo // 1024}k-{hi // 1024}k"
    return "other"


def aiperf_window(path, phase="profiling"):
    """(t0, t1) epoch seconds of an AIPerf profile_export.jsonl phase: first request_start_ns to last request_end_ns.

    Records are counted as InferenceX's agentic aggregate counts them (infx/results/agentic/artifacts.py
    load_records_with_accounting, request_metrics.py compute_throughput_stats): a record without
    metadata.benchmark_phase is profiling, a record with an error is dropped, so the window is the one its
    throughput is divided by."""
    st, en = [], []
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line); md = rec.get("metadata", {})
        if rec.get("error") or md.get("benchmark_phase", "profiling") != phase:
            continue
        if md.get("request_start_ns") and md.get("request_end_ns"):
            st.append(md["request_start_ns"]); en.append(md["request_end_ns"])
    if not st:
        raise SystemExit(f"{path}: no {phase} records with request_start_ns/request_end_ns")
    return min(st) / 1e9, max(en) / 1e9


def step_table(log, interval, window=None, long_factor=1.5):
    """Decode round time from consecutive Decode lines of one rank (one line per `interval` rounds).

    A gap is "clean" when no Prefill/Mix line of ANY rank lies between the two lines (with attention DP and
    expert-parallel MoE, another rank's prefill round holds every rank's decode round) and the running batch is
    the same at both lines; clean gaps are binned by (running batch, context bucket of avg_seq_len) and give
    step_ms = gap / interval. A gap that holds other ranks' Prefill/Mix lines is "prefill-held": its excess over
    interval x the clean median step of its bin is decode stall caused by prefill. Also per rank: Prefill/Mix
    rounds dispatched while decode requests were resident (#running-req > #new-seq) and their chunk time
    (gap to the rank's next line when that is a Prefill/Mix line, else the median chunk gap). clean_all pools
    a rank's clean gaps over its bins (n, median, p10, p90, min, max step ms, accepted tokens per round)."""
    ev = [e for e in parse_lines(open(log, errors="replace")) if e["kind"] in ("Prefill", "Mix", "Decode") and e["ts"]]
    if window:
        ev = [e for e in ev if window[0] <= e["ts"] <= window[1]]
    pts = sorted(e["ts"] for e in ev if e["kind"] in ("Prefill", "Mix"))
    def n_prefill_between(a, b):
        return bisect.bisect_left(pts, b) - bisect.bisect_right(pts, a)
    out = {"log": log, "decode_log_interval": interval, "window": window, "ranks": {}}
    clean_all = collections.defaultdict(list); held = []
    for r in sorted({e["dp_rank"] for e in ev}):
        L = [e for e in ev if e["dp_rank"] == r]
        bins = collections.defaultdict(list); acc = collections.defaultdict(list)
        clean_at = collections.defaultdict(list); held_r = []
        pp = [g for a, b, g in ((a, b, b["ts"] - a["ts"]) for a, b in zip(L, L[1:]))
              if a["kind"] in ("Prefill", "Mix") and b["kind"] in ("Prefill", "Mix")]
        cmed = statistics.median(pp) if pp else 0.0
        pf_res_n = 0; pf_res_s = 0.0; pf_res_tok = 0
        post_prefill = 0
        for i, (a, b) in enumerate(zip(L, L[1:])):
            g = b["ts"] - a["ts"]
            if a["kind"] in ("Prefill", "Mix") and a.get("running", 0) > a.get("new_seq", 0):
                pf_res_n += 1; pf_res_tok += a.get("new_token", 0)
                pf_res_s += g if b["kind"] in ("Prefill", "Mix") else min(g, cmed)
            if a["kind"] != "Decode" or b["kind"] != "Decode":
                continue
            k = (b.get("running", 0), ctx_bucket(b.get("avg_seq_len", 0)))
            # lines are logged at dispatch and the overlap schedule dispatches a round before the previous
            # one finishes, so the first Decode gap after a Prefill/Mix line (of any rank) can hold that
            # prefill's execution: kept out of the clean bins
            prev_t = L[i - 1]["ts"] if i > 0 else a["ts"] - 1e9
            if i > 0 and L[i - 1]["kind"] in ("Prefill", "Mix"):
                post_prefill += 1; continue
            if n_prefill_between(prev_t, a["ts"]) > 0 and n_prefill_between(a["ts"], b["ts"]) == 0:
                post_prefill += 1; continue
            if n_prefill_between(a["ts"], b["ts"]) == 0 and a.get("running") == b.get("running"):
                bins[k].append(g / interval * 1e3); clean_at[k].append((b["ts"], g))
                if b.get("avg_accept_len") is not None:
                    acc[k].append(b["avg_accept_len"])
                clean_all[k].append(g / interval * 1e3)
            elif n_prefill_between(a["ts"], b["ts"]) > 0:
                held_r.append((k, g, n_prefill_between(a["ts"], b["ts"])))
        rows = {}
        for k in sorted(bins):
            v = bins[k]; am = statistics.mean(acc[k]) if acc[k] else None
            rows[f"bs{k[0]}/{k[1]}"] = {"n": len(v), "step_ms_median": statistics.median(v), "p10": pct(v, .1),
                                         "p90": pct(v, .9), "accept_len_mean": am,
                                         "ms_per_token_per_request": statistics.median(v) / am if am else None}
        # clean gaps (no prefill line anywhere) far above their bin's median: decode stalls with no
        # logged prefill cause (host stalls, compiles, collectives waiting on another rank)
        long_gaps = []
        for k, v in clean_at.items():
            m = statistics.median(bins[k])
            for t, g in v:
                if g / interval * 1e3 > long_factor * m:
                    long_gaps.append({"end_utc": dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S.%f")[:-3],
                                      "gap_s": g, "excess_s": g - interval * m / 1e3, "bin": f"bs{k[0]}/{k[1]}"})
        long_gaps.sort(key=lambda x: x["end_utc"])
        # every clean gap of the rank pooled over its bins (the rank's overall decode round time)
        pool = [x for v in bins.values() for x in v]; pacc = [x for v in acc.values() for x in v]
        pam = statistics.mean(pacc) if pacc else None
        pooled = {"n": len(pool), "step_ms_median": statistics.median(pool) if pool else None,
                  "p10": pct(pool, .1), "p90": pct(pool, .9), "min": min(pool, default=None),
                  "max": max(pool, default=None), "accept_len_mean": pam,
                  "ms_per_token_per_request": statistics.median(pool) / pam if pool and pam else None}
        out["ranks"][str(r)] = {"clean_steps": rows, "clean_all": pooled, "prefill_held_gaps": len(held_r), "post_prefill_gaps": post_prefill,
                                "long_clean_gaps": {"factor": long_factor, "n": len(long_gaps),
                                                    "excess_s": sum(x["excess_s"] for x in long_gaps), "gaps": long_gaps},
                                "prefill_rounds_with_decode_resident": {"n": pf_res_n, "tokens": pf_res_tok,
                                                                        "time_s": pf_res_s}}
        held += [(r,) + h for h in held_r]
    med = {k: statistics.median(v) for k, v in clean_all.items()}
    allmed = statistics.median([x for v in clean_all.values() for x in v]) if clean_all else None
    ex = 0.0
    for r, k, g, n in held:
        ex += max(g - interval * med.get(k, allmed or 0) / 1e3, 0.0)
    out["prefill_held"] = {"gaps": len(held), "excess_decode_time_s_summed_over_ranks": ex,
                           "time_s_summed_over_ranks": sum(g for _, _, g, _ in held)}
    return out


def step_markdown(st):
    lines = ["| rank | bin (running batch / context) | n gaps | step ms median [p10, p90] | accept/step | ms per token per request |",
             "|---|---|---|---|---|---|"]
    for r, x in st["ranks"].items():
        steps = dict(x["clean_steps"])
        if len(steps) > 1 and x.get("clean_all", {}).get("n"):
            steps["all"] = x["clean_all"]
        for k, v in steps.items():
            a = v["accept_len_mean"]; t = v["ms_per_token_per_request"]
            lines.append(f"| {r} | {k} | {v['n']} | {v['step_ms_median']:.2f} [{v['p10']:.2f}, {v['p90']:.2f}] | "
                         f"{a:.2f} | {t:.2f} |" if a else f"| {r} | {k} | {v['n']} | {v['step_ms_median']:.2f} | - | - |")
    return "\n".join(lines) + "\n"


def evalscope(d):
    for root, _, names in os.walk(d):
        if "benchmark_summary.json" in names:
            p = os.path.join(root, "benchmark_summary.json")
            j = json.load(open(p)); j["_path"] = p; j["_sha256"] = sha256(p)
            return j
    return None


def phase_windows(benchmark_json, per_concurrency=False):
    """(phase, output_dir, start, end) epoch windows of an amd-tracking EvalScope matrix run, from
    each result's started_at/finished_at. A matrix repeats a phase name across concurrencies
    (measured-1 at c1, c2, ...); with per_concurrency, a phase name that repeats becomes PHASE/cB
    so each window keeps its own slice, and a phase that does not repeat keeps its name (batch_log.py
    prefill, which reads the default, suffixes every phase once any repeats)."""
    bj = json.load(open(benchmark_json))
    missing = [r.get("phase") for r in bj["results"] if "started_at" not in r or "finished_at" not in r]
    if missing:
        raise SystemExit(f"{benchmark_json}: phases {missing} have no started_at/finished_at "
                         "(recorded by amd-tracking's run_evalscope_matrix.py)")
    names = [r["phase"] for r in bj["results"]]
    if per_concurrency:
        count = collections.Counter(names)
        names = [f'{r["phase"]}/c{int(r["concurrency"])}' if count[r["phase"]] > 1 and "concurrency" in r
                 else r["phase"] for r in bj["results"]]
        dup = sorted(n for n, k in collections.Counter(names).items() if k > 1)
        if dup:
            raise SystemExit(f"{benchmark_json}: windows {dup} repeat with the same phase and concurrency")
    return [(n, r["output_dir"], parse_iso(r["started_at"]), parse_iso(r["finished_at"]))
            for n, r in zip(names, bj["results"])]


def parse_iso(text):
    text = text.rstrip("Z")
    head, _, frac = text.partition(".")
    value = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return value + (float("0." + frac) if frac else 0.0)


def write_slices(logs, windows, out_dir):
    """Write the lines of the server logs that fall in each window to out_dir/PHASE.log (PHASE/cB to
    PHASE_cB.log)."""
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
        path = os.path.join(out_dir, phase.replace("/", "_") + ".log")
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
        nreq = [R[q]["requests_by_prefill_tokens"] for q in sorted(R)]
        split = "/".join("-" if x is None else f"{x:g}" for x in nreq)
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
    ap.add_argument("--window-from-aiperf", action="append", default=[],
                    help="AIPerf profile_export.jsonl per --log: decompose only its profiling phase (first request start "
                         "to last request end) instead of the start/end rules (variable-length agentic runs)")
    ap.add_argument("--step-table", action="store_true",
                    help="also give decode round time by running batch and context (needs --decode-log-interval)")
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
        windows = phase_windows(a.benchmark_json, per_concurrency=True)
        a.log = write_slices(a.server_log, windows, a.slices_dir)
        a.label = [w[0] for w in windows]
        a.evalscope = [w[1] for w in windows]
    if not a.log:
        ap.error("give --log or --benchmark-json")
    runs = []
    if a.step_table and not a.decode_log_interval:
        ap.error("--step-table needs --decode-log-interval")
    for i, lg in enumerate(a.log):
        win = aiperf_window(a.window_from_aiperf[i]) if i < len(a.window_from_aiperf) else None
        r = decompose(lg, a.prompt_tokens, window=win)
        if a.step_table:
            r["step_table"] = step_table(lg, a.decode_log_interval, win)
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
                  f"prefill={x['prefill_time_s']:.2f}s ({x['prefill_tok_per_s_while_prefilling'] or 0:.0f} tok/s, chunk med {x['chunk_time_s']['median']*1e3:.1f} ms over {x['chunk_time_s']['n']} gaps) "
                  f"decode={x['decode_time_s']:.2f}s ({x['decode_tok_per_s_while_decoding'] or 0:.1f} tok/s, logged {x['decode_tokens_logged']:.0f}/{x['decode_tokens_expected']}) "
                  f"idle={x['idle_s']:.2f}s (between waves {x['idle_between_waves_s']:.2f}) stalls={x['prefill_stalls']['n_gaps_gt_2x_median']}/{x['prefill_stalls']['excess_s']:.2f}s "
                  + (f"finish-split={x['finish_split_gaps']['n']} ({x['finish_split_gaps']['from_prefill']} from prefill) "
                     if x.get('finish_split_gaps') else "") + f"tail={x['tail_after_last_line_s']:.2f} lag0={x['first_line_lag_s']:.2f} end={x['estimated_rank_end_s']:.2f} bs={x['decode_batch_size_lines']}")
    if a.markdown:
        print(markdown(runs, a.decode_log_interval), end="")
    for r in runs:
        if "step_table" in r:
            print("== step table", r["label"], json.dumps(r["step_table"]["prefill_held"]))
            for k, x in r["step_table"]["ranks"].items():
                print(f" rank{k}: long clean gaps={x['long_clean_gaps']['n']} excess={x['long_clean_gaps']['excess_s']:.2f}s "
                      + " ".join(f"{g['end_utc']}/{g['gap_s']:.2f}s" for g in x['long_clean_gaps']['gaps']))
                print(f" rank{k}: prefill-held gaps={x['prefill_held_gaps']} "
                      f"prefill rounds with decode resident={x['prefill_rounds_with_decode_resident']}")
            print(step_markdown(r["step_table"]), end="")


if __name__ == "__main__":
    main()
