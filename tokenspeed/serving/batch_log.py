#!/usr/bin/env python3
"""Parse a TokenSpeed server log: batch lines and request-finish lines.

Shared by the sustained, end-to-end and correctness tools in this directory.

parse_line(text) returns None for any other line, else a dict with "ts" (epoch seconds or None),
"subsec", "tz_explicit", "line" and "kind":
  "Prefill" | "Mix" | "Decode"  from `ts_control::batch_log` lines, with the fields of FIELD_RES
                                that are present ("dp_rank" defaults to 0 for single-rank logs);
  "Finish"                      from `Req: RID Finish! RequestStats(...)`, with "rid" and "stats"
                                (numbers as int/float, quoted strings unquoted, None as None).
LogTail follows a growing log from its current end.
dp_spread(events, windows) counts new sequences per attention-DP rank ("#new-seq" of "Prefill batch."
lines) inside each named time window [start, end).

usage: batch_log.py LOG   prints the count of each kind (a quick check that a log parses)
       batch_log.py dp-spread LOG --window NAME=START[,END] [--window ...] [--json]
           new sequences per DP rank in each window; START/END are ISO 8601 times (UTC when no offset
           is given, e.g. 2026-11-12T23:05:00Z); a window without START counts nothing, one without END
           runs to the end of the log. Prints "NAME rank0=K rank1=K ... total=K" per window.
       batch_log.py prefill LOG [LOG ...] (--benchmark-json BJ | --window NAME=START,END ...
                    --prompt-tokens L --requests B) [--chunk-cost] [--tail] [--out-json PATH]
           prefill points (one wave of B simultaneous requests of L prompt tokens, one output token)
           read from the server log. Windows come from an amd-tracking run_evalscope_matrix.py
           benchmark.json (each result's started_at/finished_at; L = prompt_tokens, B = concurrency;
           points are named PHASE/cB when the matrix repeats a phase name across concurrencies)
           or from --window. Per point and attention-DP rank: Prefill/Mix chunks, prefill tokens,
           requests (tokens / L), chunk time (median of same-rank Prefill->Prefill line gaps, i.e.
           round-dispatch to next round-dispatch), p10/p90, by chunk size, prefill span and tok/s.
           A point is valid when the ranks' prefill tokens sum to B*L and no prompt token came from
           the prefix cache. The cache check uses the RequestStats lines when the window has any (B of
           them, each with prompt_tokens == L and cache_tokens == 0); a server that logs no RequestStats
           (e.g. TokenSpeed e8fd096e, whose Finish! lines carry Accept_num_tokens_avg instead) is
           checked from the "#cached-token" of every Prefill/Mix batch line in the window (all present
           and 0). "validity_source" names the source used ("requeststats", "prefill-lines" or "none").
           Per point it also gives the batching: steps (Prefill/Mix lines), present_max (most requests
           the scheduler held at once, max of #running-req + #queue-req over the steps) and the per-step
           sequence "new_seq/new_token" in log order (e.g. 1/8000 2/8192 2/8192 1/7616).
           --chunk-cost pools the chunk gaps of valid points by (chunk tokens, new sequences,
           whether the chunk completes a request, B=1 or B>=4); --tail gives each valid point's
           final prefill step (last Prefill line to last Finish line) by tail-chunk size and a
           least-squares line final_step_ms = fixed + per_token x tail_tokens over group medians.
"""
import collections, datetime as dt, re, statistics, sys, time

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS_PATTERNS = [
    # ISO 8601 (Rust tracing default), e.g. 2026-11-06T22:30:01.123456Z
    (re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:?\d{2})?"), "iso"),
    # python logging, e.g. 2026-11-06 17:22:37,194
    (re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.](\d+))?"), "py"),
]
NUM = r"([-+]?\d+(?:\.\d+)?)"
FIELD_RES = {
    "dp_rank": re.compile(r"#dp-rank: (\d+)"),
    "new_seq": re.compile(r"#new-seq: (\d+)"),
    "new_token": re.compile(r"#new-token: (\d+)"),
    "cached_token": re.compile(r"#cached-token: (\d+)"),
    "running": re.compile(r"#running-req: (\d+)"),
    "queue": re.compile(r"#queue-req: (\d+)"),
    "avg_seq_len": re.compile(r"avg_seq_len: " + NUM),
    "gen_tput": re.compile(r"gen throughput \(token/s\): " + NUM),
    "avg_accept_len": re.compile(r"avg_accept_len: " + NUM),
    "accept_rate": re.compile(r"accept_rate: " + NUM),
}
KIND_RE = re.compile(r"\b(Prefill|Mix|Decode) batch\.")
FINISH_RE = re.compile(r"Req: (\S+) Finish! RequestStats\((.*)\)")
STAT_RE = re.compile(r"(\w+)=('[^']*'|[^,()]+)")


def parse_ts(line):
    """Return (epoch_seconds_or_None, has_subsecond, tz_explicit)."""
    for pat, kind in TS_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        if kind == "iso":
            s, tz = m.group(1), m.group(2)
            sub = "." in s
            fmt = "%Y-%m-%dT%H:%M:%S.%f" if sub else "%Y-%m-%dT%H:%M:%S"
            d = dt.datetime.strptime(s, fmt)
            if tz:
                tzs = "+00:00" if tz == "Z" else tz
                d = d.replace(tzinfo=dt.datetime.strptime(tzs.replace(":", ""), "%z").tzinfo)
                return d.timestamp(), sub, True
            return time.mktime(d.timetuple()) + d.microsecond / 1e6, sub, False
        s, frac = m.group(1), m.group(2)
        d = dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        t = time.mktime(d.timetuple())  # local time
        if frac:
            t += int(frac) / (10 ** len(frac))
        return t, bool(frac), False
    return None, False, False


def stat_value(text):
    v = text.strip()
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    if v == "None":
        return None
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def parse_line(raw):
    line = ANSI.sub("", raw.rstrip("\n"))
    m = KIND_RE.search(line)
    fm = FINISH_RE.search(line)
    if not m and not fm:
        return None
    ts, sub, tzx = parse_ts(line)
    ev = {"ts": ts, "subsec": sub, "tz_explicit": tzx, "line": line}
    if m:
        ev["kind"] = m.group(1)
        for k, r in FIELD_RES.items():
            mm = r.search(line)
            if mm:
                v = mm.group(1)
                ev[k] = float(v) if "." in v else int(v)
        ev.setdefault("dp_rank", 0)
    else:
        ev["kind"] = "Finish"
        ev["rid"] = fm.group(1)
        ev["stats"] = {k: stat_value(v) for k, v in STAT_RE.findall(fm.group(2))}
    return ev


def parse_lines(lines):
    return [e for e in (parse_line(l) for l in lines) if e is not None]


class LogTail:
    """Lines appended to PATH after construction; poll() returns all complete lines so far."""

    def __init__(self, path):
        self.path = path; self.f = open(path, "r", errors="replace")
        self.f.seek(0, 2); self.start_offset = self.f.tell(); self.lines = []; self.buf = ""

    def poll(self):
        data = self.f.read()
        if data:
            data = self.buf + data
            parts = data.split("\n"); self.buf = parts.pop()
            self.lines.extend(p + "\n" for p in parts)
        return self.lines


def dp_spread(events, windows):
    """{window name: {dp_rank: new sequences}} over Prefill events with a timestamp; windows are
    (name, start_epoch_or_None, end_epoch_or_None), a line counts when start <= ts < end."""
    counts = {name: {} for name, _, _ in windows}
    for e in events:
        if e["kind"] != "Prefill" or e["ts"] is None or "new_seq" not in e:
            continue
        for name, start, end in windows:
            if start is not None and e["ts"] >= start and (end is None or e["ts"] < end):
                counts[name][e["dp_rank"]] = counts[name].get(e["dp_rank"], 0) + e["new_seq"]
    return counts


def parse_window(text):
    """NAME=START[,END] -> (name, start_epoch_or_None, end_epoch_or_None); times are ISO 8601, UTC unless an
    offset is given."""
    name, _, span = text.partition("=")
    if not name or not _:
        raise ValueError(f"expected NAME=START[,END], got {text!r}")
    bounds = []
    for part in (span.split(",") + [""])[:2]:
        part = part.strip()
        if not part:
            bounds.append(None)
            continue
        d = dt.datetime.fromisoformat(part.replace("Z", "+00:00"))
        bounds.append((d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)).timestamp())
    return name, bounds[0], bounds[1]


def render_dp_spread(counts):
    return "\n".join(f"{name} " + " ".join(f"rank{r}={c[r]}" for r in sorted(c)) + f" total={sum(c.values())}"
                     for name, c in counts.items())


def dp_spread_main(argv):
    import argparse, json
    ap = argparse.ArgumentParser(prog="batch_log.py dp-spread",
                                 description="new sequences per attention-DP rank per time window, from Prefill lines")
    ap.add_argument("log")
    ap.add_argument("--window", action="append", required=True, metavar="NAME=START[,END]")
    ap.add_argument("--json", action="store_true", help="print {window: {rank: count}} as JSON instead")
    a = ap.parse_args(argv)
    windows = [parse_window(w) for w in a.window]
    with open(a.log, errors="replace") as fh:
        counts = dp_spread(parse_lines(fh), windows)
    print(json.dumps({n: {str(r): k for r, k in sorted(c.items())} for n, c in counts.items()}) if a.json
          else render_dp_spread(counts))
    return 0


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q; i = int(k)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (k - i)


def prefill_point(events, prompt_tokens, requests, start=None, end=None):
    """Server-side metrics of one prefill point from parsed events in [start, end) (see `prefill` in
    the module docstring). Chunk time = gap from a rank's Prefill/Mix line to its next line (lines are
    written when a round is dispatched); without a following Prefill/Mix line, the median of gaps to
    any next line, else the median RequestStats prefill_ms (an upper bound: includes queueing)."""
    ev = sorted((e for e in events if e["ts"] is not None and (start is None or e["ts"] >= start)
                 and (end is None or e["ts"] < end)), key=lambda e: e["ts"])
    pf = [e for e in ev if e["kind"] in ("Prefill", "Mix")]
    fin = [e for e in ev if e["kind"] == "Finish"]
    ranks = {}
    for r in sorted({e["dp_rank"] for e in pf}):
        P = [e for e in pf if e["dp_rank"] == r]
        allr = [e for e in ev if e.get("dp_rank") == r and e["kind"] != "Finish"]
        gaps = [{"chunk_tokens": a.get("new_token", 0), "new_seq": a.get("new_seq", 0),
                 "dt_s": b["ts"] - a["ts"], "next_kind": b["kind"]}
                for a, b in zip(allr, allr[1:]) if a["kind"] in ("Prefill", "Mix")]
        pp = [g["dt_s"] for g in gaps if g["next_kind"] in ("Prefill", "Mix")]
        if pp:
            med, src = statistics.median(pp), "prefill->prefill line gaps"
        elif gaps:
            med, src = statistics.median(g["dt_s"] for g in gaps), "prefill->any next line gaps"
        else:
            pm = [float(e["stats"].get("prefill_ms") or 0) / 1e3 for e in fin]
            med, src = (statistics.median(pm) if pm else None), "RequestStats prefill_ms median (upper bound)"
        tok = sum(e.get("new_token", 0) for e in P)
        span = P[-1]["ts"] - P[0]["ts"] + (med or 0)
        by = {}
        for g in gaps:
            if g["next_kind"] in ("Prefill", "Mix"):
                by.setdefault(g["chunk_tokens"], []).append(g["dt_s"])
        ranks[str(r)] = {
            "chunks": len(P), "prefill_tokens": tok, "requests_by_tokens": tok / prompt_tokens,
            "first_prefill_line_s": P[0]["ts"], "last_prefill_line_s": P[-1]["ts"],
            "last_chunk_tokens": P[-1].get("new_token", 0),
            "chunk_time_median_s": med, "chunk_time_source": src,
            "chunk_time_p10_s": quantile(pp, .1), "chunk_time_p90_s": quantile(pp, .9),
            "chunk_time_by_tokens": {str(k): {"n": len(v), "median_s": statistics.median(v)}
                                     for k, v in sorted(by.items())},
            "prefill_span_s": span, "prefill_tok_per_s": tok / span if span else None,
            "chunk_gaps": gaps}
    stats = [e["stats"] for e in fin]
    total = sum(x["prefill_tokens"] for x in ranks.values())
    longest = max((x["prefill_span_s"] for x in ranks.values()), default=0)
    rs_ok = len(stats) == requests and all(
        stat_int(s, "prompt_tokens") == prompt_tokens and stat_int(s, "cache_tokens") == 0 for s in stats)
    cached = [e.get("cached_token") for e in pf]
    lines_ok = bool(pf) and all(c == 0 for c in cached)
    if stats:
        source, cache_ok = "requeststats", rs_ok
    elif pf and all(c is not None for c in cached):
        source, cache_ok = "prefill-lines", lines_ok
    else:
        source, cache_ok = "none", False
    return {"prompt_tokens": prompt_tokens, "requests": requests, "ranks": ranks,
            "prefill_tokens_total": total, "prefill_tokens_expected": requests * prompt_tokens,
            "prefill_tokens_ok": total == requests * prompt_tokens,
            "requeststats_n": len(stats),
            "requeststats_prompt_tokens": sorted({stat_int(s, "prompt_tokens") for s in stats}),
            "requeststats_cache_tokens": sorted({stat_int(s, "cache_tokens") for s in stats}),
            "requeststats_ok": rs_ok,
            "prefill_lines_cached_tokens": sorted({-1 if c is None else c for c in cached}),
            "prefill_lines_cached_token_sum": sum(c or 0 for c in cached),
            "prefill_lines_ok": lines_ok,
            "validity_source": source,
            "cache_tokens": (sorted({stat_int(s, "cache_tokens") for s in stats}) if source == "requeststats"
                             else sorted({-1 if c is None else c for c in cached})),
            "steps": len(pf),
            "present_max": max((e.get("running", 0) + e.get("queue", 0) for e in pf), default=0),
            "steps_seq": [f'{e.get("new_seq", 0)}/{e.get("new_token", 0)}' for e in pf],
            "last_finish_s": max((e["ts"] for e in fin), default=None),
            "server_tok_per_s_all_ranks": total / longest if longest > 0 else None,
            "valid": total == requests * prompt_tokens and cache_ok}


def stat_int(stats, key):
    v = stats.get(key)
    return int(v) if isinstance(v, (int, float)) else -1


def chunk_cost(points):
    """Chunk gaps of valid points pooled by (chunk tokens, new seq, completes a request, batch class);
    requests are taken as prefilled in order on each rank."""
    pool = {}
    for p in points:
        if not p["valid"]:
            continue
        for rank in p["ranks"].values():
            done = 0
            for g in rank["chunk_gaps"]:
                before, done = done, done + g["chunk_tokens"]
                if g["next_kind"] not in ("Prefill", "Mix"):
                    continue
                completes = done // p["prompt_tokens"] > before // p["prompt_tokens"]
                key = (g["chunk_tokens"], g["new_seq"], "final" if completes else "intermediate",
                       "B1" if p["requests"] == 1 else "B>=4")
                pool.setdefault(key, []).append(g["dt_s"])
    return [{"chunk_tokens": k[0], "new_seq": k[1], "completes_request": k[2], "batch_class": k[3], "n": len(v),
             "median_ms": 1e3 * statistics.median(v), "p10_ms": 1e3 * quantile(v, .1),
             "p90_ms": 1e3 * quantile(v, .9), "tok_per_s": k[0] / statistics.median(v)}
            for k, v in sorted(pool.items())]


def tail(points):
    """Final prefill step of each valid point (last Prefill line of any rank to the last Finish line;
    ranks step together) by tail-chunk size, and a line through the group medians per batch class."""
    rows = []
    for name, p in points.items():
        if not p["valid"] or not p["ranks"] or p["last_finish_s"] is None:
            continue
        sizes = sorted({r["last_chunk_tokens"] for r in p["ranks"].values()})
        last = max(r["last_prefill_line_s"] for r in p["ranks"].values())
        rows.append({"point": name, "L": p["prompt_tokens"], "B": p["requests"], "tail_tokens": sizes,
                     "final_step_ms": 1e3 * (p["last_finish_s"] - last)})
    groups = {}
    for r in rows:
        if len(r["tail_tokens"]) == 1:
            groups.setdefault((r["tail_tokens"][0], "B1" if r["B"] == 1 else "B>=4"), []).append(r["final_step_ms"])
    summary = [{"tail_tokens": k[0], "batch_class": k[1], "n": len(v), "median_ms": statistics.median(v),
                "min_ms": min(v), "max_ms": max(v)} for k, v in sorted(groups.items())]
    fits = {}
    for bc in ("B1", "B>=4"):
        pts = [(x["tail_tokens"], x["median_ms"]) for x in summary if x["batch_class"] == bc]
        if len({x for x, _ in pts}) > 1:
            mx, my = statistics.mean(x for x, _ in pts), statistics.mean(y for _, y in pts)
            b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
            fits[bc] = {"fixed_ms": my - b * mx, "per_token_ms": b}
    return {"rows": rows, "summary": summary, "fit_group_medians": fits}


def prefill_main(argv):
    import argparse, json, os
    ap = argparse.ArgumentParser(prog="batch_log.py prefill", description="prefill points from the server log")
    ap.add_argument("log", nargs="+")
    ap.add_argument("--benchmark-json", help="amd-tracking run_evalscope_matrix.py benchmark.json (phase windows, "
                    "prompt_tokens, concurrency)")
    ap.add_argument("--window", action="append", default=[], metavar="NAME=START,END")
    ap.add_argument("--prompt-tokens", type=int); ap.add_argument("--requests", type=int)
    ap.add_argument("--chunk-cost", action="store_true"); ap.add_argument("--tail", action="store_true")
    ap.add_argument("--out-json")
    a = ap.parse_args(argv)
    if bool(a.benchmark_json) == bool(a.window):
        ap.error("give --benchmark-json or --window")
    if a.window and not (a.prompt_tokens and a.requests):
        ap.error("--window needs --prompt-tokens and --requests")
    events = []
    for log in a.log:
        with open(log, errors="replace") as fh:
            events += parse_lines(fh)
    specs = []
    if a.benchmark_json:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from e2e_decompose import phase_windows
        bj = json.load(open(a.benchmark_json))
        # one point per result; the matrix reuses phase names across concurrencies (measured-1 at c1, c2, ...),
        # so names get "/c<concurrency>" when a phase name repeats
        names = [r["phase"] for r in bj["results"]]
        dup = len(set(names)) < len(names)
        specs = [(f'{ph}/c{int(r["concurrency"])}' if dup else ph, s, e, int(bj["prompt_tokens"]), int(r["concurrency"]))
                 for r, (ph, _, s, e) in zip(bj["results"], phase_windows(a.benchmark_json))]
    for w in a.window:
        name, s, e = parse_window(w)
        if s is None or e is None:
            ap.error(f"--window {w!r}: prefill windows need START and END")
        specs.append((name, s, e, a.prompt_tokens, a.requests))
    points = {name: prefill_point(events, L, B, s, e) for name, s, e, L, B in specs}
    res = {"points": points}
    print("| point | L | B | valid | cache_tokens | validity source | steps | present max | step new_seq/new_token "
          "| ranks: requests / chunks / chunk ms median | server tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, p in points.items():
        ranks = "; ".join(f"r{k} {v['requests_by_tokens']:g}/{v['chunks']}/"
                          + ("-" if v["chunk_time_median_s"] is None else f"{1e3 * v['chunk_time_median_s']:.2f}")
                          for k, v in p["ranks"].items())
        tps = p["server_tok_per_s_all_ranks"]
        print(f"| {name} | {p['prompt_tokens']} | {p['requests']} | {p['valid']} | {p['cache_tokens']} | "
              f"{p['validity_source']} | {p['steps']} | {p['present_max']} | {' '.join(p['steps_seq'])} | "
              f"{ranks} | {'-' if tps is None else f'{tps:.0f}'} |")
    if a.chunk_cost:
        res["chunk_cost"] = chunk_cost(points.values())
        print("\n| chunk tokens | new seq | completes a request | batch | n | median ms | p10 ms | p90 ms | tok/s |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c in res["chunk_cost"]:
            print(f"| {c['chunk_tokens']} | {c['new_seq']} | {c['completes_request']} | {c['batch_class']} | {c['n']} | "
                  f"{c['median_ms']:.2f} | {c['p10_ms']:.2f} | {c['p90_ms']:.2f} | {c['tok_per_s']:.0f} |")
    if a.tail:
        res["tail"] = tail(points)
        print("\n| tail tokens | batch | n | final step median ms | min | max |\n|---|---|---|---|---|---|")
        for t in res["tail"]["summary"]:
            print(f"| {t['tail_tokens']} | {t['batch_class']} | {t['n']} | {t['median_ms']:.1f} | {t['min_ms']:.1f} | {t['max_ms']:.1f} |")
        for bc, v in res["tail"]["fit_group_medians"].items():
            print(f"fit {bc}: final_step_ms = {v['fixed_ms']:.1f} + {v['per_token_ms'] * 1e3:.2f} ms per 1000 tokens")
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1)
    return 0 if all(p["valid"] for p in points.values()) else 3


if __name__ == "__main__":
    if sys.argv[1:2] == ["dp-spread"]:
        sys.exit(dp_spread_main(sys.argv[2:]))
    if sys.argv[1:2] == ["prefill"]:
        sys.exit(prefill_main(sys.argv[2:]))
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 else 2)
    with open(sys.argv[1], errors="replace") as fh:
        print(dict(collections.Counter(e["kind"] for e in parse_lines(fh))))
