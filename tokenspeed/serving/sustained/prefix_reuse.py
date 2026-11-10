#!/usr/bin/env python3
"""Prefix-cache reuse of multi-turn conversations under data-parallel placement.

Joins the server's `Finish! RequestStats` lines (prompt and cached tokens per request) with the
EvalScope database (conversation salt and turn per request) by response id, and classifies every
later turn: `hit_prev` (its cached prefix covers the previous turn's prompt, within --page-slack
tokens), `hit_older`, `miss_partial` or `miss_zero`. It compares the uncached tokens with two
references: every turn reusing its own previous prompt (same-rank placement), and each turn placed
on a uniformly random rank with no eviction (Monte Carlo, --seeds draws).

usage: prefix_reuse.py --log SLICE --db benchmark_data.db --dp 4 [--steady 120:900] [--csv OUT]
                       [--json OUT]

--steady A:B adds totals for requests committed in [first commit + A, first commit + B) (the anchor
is the first commit; evalscope_db.py names each analysis's anchor). The database is read with
evalscope_db.py, without unpickling.
"""
import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from batch_log import parse_line  # noqa: E402
from evalscope_db import parse_steady, read_results, steady_bounds  # noqa: E402


def read_log(path):
    """RequestStats by request id, and per-rank Prefill/Mix totals and peak KV page use."""
    reqs = {}
    ranks = defaultdict(lambda: dict(lines=0, new_seq=0, new_token=0, cached_token=0, max_used_frac=0.0))
    with open(path, errors="replace") as fh:
        for line in fh:
            e = parse_line(line)
            if not e:
                continue
            if e["kind"] == "Finish":
                s = e["stats"]
                if "prompt_tokens" in s:
                    reqs[e["rid"]] = dict(status=s.get("status"), prompt=int(s["prompt_tokens"]),
                                          cache=int(s.get("cache_tokens") or 0), out=int(s.get("output_tokens") or 0),
                                          prefill_ms=s.get("prefill_ms"), commit_ts=s.get("commit_ts"),
                                          finish_ts=s.get("finish_ts"))
            elif e["kind"] in ("Prefill", "Mix"):
                r = ranks[e["dp_rank"]]
                r["lines"] += 1
                for k in ("new_seq", "new_token", "cached_token"):
                    r[k] += e.get(k, 0)
            pages = re.search(r"#pages\(active/cached/total\): (\d+)/(\d+)/(\d+)", e["line"]) if e["kind"] == "Decode" else None
            if pages:
                a, c, t = map(int, pages.groups())
                r = ranks[e["dp_rank"]]
                r["max_used_frac"] = max(r["max_used_frac"], (a + c) / t if t else 0.0)
    return reqs, {str(k): v for k, v in sorted(ranks.items())}


def read_db(path):
    return [dict(cid=r["response_id"], salt=r["salt"], user_msgs=r["user_turns"], start=r["start_time"],
                 success=r["success"], ttft=r["first_chunk_latency"]) for r in read_results(path)]


def classify(convs, page_slack):
    """Annotate each turn (sorted per conversation) with kind and expected uncached under same-rank reuse."""
    out = []
    for salt, turns in convs.items():
        turns.sort(key=lambda t: t["user_msgs"])
        prev = None
        for i, t in enumerate(turns, 1):
            rec = dict(salt=salt, turn=i, prompt=t["prompt"], cache=t["cache"], uncached=t["prompt"] - t["cache"],
                       out=t["out"], prefill_ms=t["prefill_ms"], commit_ts=t["commit_ts"],
                       finish_ts=t["finish_ts"], ttft_s=t["ttft"])
            if prev is None:
                rec.update(kind="first", expected_same_rank=t["prompt"])
            else:
                rec["expected_same_rank"] = max(t["prompt"] - prev["prompt"], 0)
                if t["cache"] >= prev["prompt"] - page_slack:
                    rec["kind"] = "hit_prev"
                elif t["cache"] == 0:
                    rec["kind"] = "miss_zero"
                else:
                    older = [o["prompt"] for o in turns[:i - 2]]
                    near = any(o - page_slack <= t["cache"] <= o + page_slack for o in older)
                    rec["kind"] = "hit_older" if near else "miss_partial"
            out.append(rec)
            prev = t
    return out


def simulate_uniform(turns_by_conv, dp, seeds):
    """Uncached tokens if each turn lands on a uniformly random rank and nothing is evicted."""
    totals = []
    for seed in range(seeds):
        rng = random.Random(seed)
        unc = 0
        for turns in turns_by_conv:
            cached = {}
            for t in turns:
                r = rng.randrange(dp)
                unc += t["prompt"] - min(cached.get(r, 0), t["prompt"])
                cached[r] = max(cached.get(r, 0), t["prompt"])
        totals.append(unc)
    return dict(mean=statistics.mean(totals), min=min(totals), max=max(totals), seeds=seeds)


def totals(recs):
    later = [r for r in recs if r["kind"] != "first"]
    missed = [r for r in later if r["kind"] != "hit_prev"]
    p = sum(r["prompt"] for r in recs)
    c = sum(r["cache"] for r in recs)
    exp = sum(r["expected_same_rank"] for r in recs)
    kinds = defaultdict(int)
    for r in recs:
        kinds[r["kind"]] += 1
    return dict(requests=len(recs), prompt_tokens=p, cached_tokens=c, uncached_tokens=p - c,
                hit_ratio=c / p if p else None, expected_uncached_same_rank=exp, excess_vs_same_rank=p - c - exp,
                later_turns=len(later), later_hit_prev=len(later) - len(missed), later_missed=len(missed),
                later_hit_prev_fraction=(len(later) - len(missed)) / len(later) if later else None,
                kinds=dict(kinds), missed_uncached=sum(r["uncached"] for r in missed),
                missed_excess=sum(r["uncached"] - r["expected_same_rank"] for r in missed),
                ttft_s_mean_by_kind={k: statistics.mean(r["ttft_s"] for r in recs if r["kind"] == k
                                                        and r["ttft_s"] is not None)
                                     for k in kinds if any(r["kind"] == k and r["ttft_s"] is not None for r in recs)})


def analyze(log, db, dp, page_slack=3072, seeds=200, steady=None):
    reqs, ranks = read_log(log)
    rows = read_db(db)
    convs = defaultdict(list)
    unmatched = 0
    for r in rows:
        s = reqs.get(r["cid"])
        if s is None:
            unmatched += 1
            continue
        convs[r["salt"]].append({**r, **s})
    recs = classify(convs, page_slack)
    res = dict(dp=dp, conversations=len(convs), db_rows=len(rows), unmatched_db_rows=unmatched,
               server_requeststats=len(reqs), full=totals(recs), ranks=ranks)
    if dp > 1:
        by_conv = defaultdict(list)
        for r in recs:
            by_conv[r["salt"]].append(r)
        res["uniform_placement_uncached"] = simulate_uniform(by_conv.values(), dp, seeds)
    if steady:
        s0, s1 = steady_bounds(min(r["commit_ts"] for r in recs), steady)
        res["steady"] = dict(definition=f"commit_ts in [first commit + {steady[0]:g} s, + {steady[1]:g} s)",
                             **totals([r for r in recs if s0 <= r["commit_ts"] < s1]))
    return res, recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="server log slice of the run")
    ap.add_argument("--db", required=True, help="EvalScope benchmark_data.db of the run")
    ap.add_argument("--dp", type=int, required=True, help="data-parallel ranks")
    ap.add_argument("--page-slack", type=int, default=3072, help="tokens a hit may fall short of the previous prompt")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--steady", help="A:B seconds after the first commit (anchor: first commit)")
    ap.add_argument("--csv", help="write one row per request")
    ap.add_argument("--json", help="write the summary")
    a = ap.parse_args(argv)
    steady = parse_steady(a.steady)
    res, recs = analyze(a.log, a.db, a.dp, a.page_slack, a.seeds, steady)
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(sorted(recs, key=lambda r: (r["salt"], r["turn"])))
    text = json.dumps(res, indent=1)
    if a.json:
        with open(a.json, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
