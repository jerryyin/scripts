#!/usr/bin/env python3
"""Compare two greedy-output files written by greedy_client.py, prompt by prompt.

usage: compare_outputs.py A.jsonl B.jsonl [--label-a A] [--label-b B] [--out report.md] [--cache-free]
       compare_outputs.py --pair LABEL_A=A.jsonl,LABEL_B=B.jsonl [--pair ...] [--cache-free]
       [--key FIELD[,FIELD...]] [--where FIELD=VALUE ...] [--rows]

--key pairs records by other fields than idx, e.g. --key prompt_sha256 for amd-tracking
evalscope_degeneracy.py --requests records (the same prompt in two serving runs; a key seen k times in a
file pairs its i-th occurrences). --where keeps records whose FIELD equals VALUE (as text), e.g.
--where concurrency=1. Each row also gives both sides' acceptance (acc_len) and loop period (loop_period
of the record, else computed from the re-encoded ids: the smallest p <= 16 with t[i] == t[i-p] over the
last 64 tokens, amd-tracking evalscope_degeneracy.loop_period). --rows prints every row as a JSON line.

For each prompt index: exact match of the generated text (reasoning + content for chat), the first
token where the re-encoded outputs differ, the common token prefix, finish reasons, completion
token counts and prefix-cache hits. Prints a summary line per pair (exact, token-identical, differing
at token 0, median, minimum and maximum first differing token, first_diff_buckets: how many
token-differing prompts first differ at token 0, 1-4, 5-31, 32-127, 128+, each such prompt counted
once, total common prefix, mean / min / max acceptance length of each side) and, with --out,
writes a Markdown table. --cache-free adds the same counts over the prompts with no prefix-cache
hit on either side. Exits 0 whether or not the outputs match; deciding what a mismatch means is
the caller's job.
"""
import argparse
import json
import statistics
import sys


LOOP_WINDOW, MAX_LOOP_PERIOD = 64, 16


def loop_period(tokens, window=LOOP_WINDOW, max_period=MAX_LOOP_PERIOD):
    """Smallest p in 1..max_period with t[i] == t[i-p] for each of the last `window` positions."""
    for period in range(1, max_period + 1):
        if len(tokens) < window + period:
            return None
        if all(tokens[i] == tokens[i - period] for i in range(len(tokens) - window, len(tokens))):
            return period
    return None


def load(path, key=("idx",), where=()):
    out, seen = {}, {}
    with open(path) as fh:
        for r in map(json.loads, (line for line in fh if line.strip())):
            if any(str(r.get(f)) != v for f, v in where):
                continue
            k = tuple(r.get(f) for f in key)
            n = seen.get(k, 0); seen[k] = n + 1
            out[k[0] if key == ("idx",) else (*k, n)] = r
    return out


def period_of(r):
    if "loop_period" in r:
        return r["loop_period"]
    ids = r.get("reencoded_ids")
    return loop_period(ids) if ids else None


def generated(r):
    return (r.get("reasoning_content") or "") + "\x00" + (r.get("text") or "")


def common_prefix(x, y):
    n = 0
    for u, v in zip(x, y):
        if u != v:
            break
        n += 1
    return n


def compare(a, b):
    rows = []
    for i in (sorted(a) if all(isinstance(k, int) for k in a) else sorted(a, key=str)):
        ra, rb = a[i], b.get(i, {})
        ta, tb = ra.get("reencoded_ids") or [], rb.get("reencoded_ids") or []
        prefix = common_prefix(ta, tb)
        rows.append({
            "idx": i,
            "endpoint": "chat" if "chat" in ra.get("endpoint", "") else "completions",
            "exact": bool(rb) and generated(ra) == generated(rb)
            and ra.get("http_status", 200) == 200 and rb.get("http_status", 200) == 200,
            "first_diff_token": None if ta == tb else prefix,
            "common_prefix": prefix, "len_a": len(ta), "len_b": len(tb),
            "finish": (ra.get("finish_reason"), rb.get("finish_reason")),
            "completion_tokens": (ra.get("completion_tokens"), rb.get("completion_tokens")),
            "cache_tokens": ((ra.get("request_stats") or {}).get("cache_tokens"),
                             (rb.get("request_stats") or {}).get("cache_tokens")),
            "acc_len": (ra.get("acc_len"), rb.get("acc_len")),
            "loop_period": (period_of(ra), period_of(rb) if rb else None),
            "matched": bool(rb),
        })
    return rows


def acceptance(outputs):
    v = [r["acc_len"] for r in outputs.values() if r.get("acc_len") is not None]
    return {"n": len(v), "mean": statistics.mean(v), "min": min(v), "max": max(v)} if v else None


# Ranges of the first differing token (inclusive bounds); contiguous from 0, so every token-differing
# prompt falls in exactly one and the counts sum to prompts - token_identical.
FIRST_DIFF_BUCKETS = (("0", 0, 0), ("1-4", 1, 4), ("5-31", 5, 31), ("32-127", 32, 127), ("128+", 128, float("inf")))


def summarize(rows, label):
    diffs = [r["first_diff_token"] for r in rows if r["first_diff_token"] is not None]
    return {
        "pair": label,
        "exact": sum(r["exact"] for r in rows), "prompts": len(rows),
        "token_identical": sum(r["first_diff_token"] is None for r in rows),
        "differ_at_token_0": sum(d == 0 for d in diffs),
        "median_first_diff": statistics.median(diffs) if diffs else None,
        "min_first_diff": min(diffs) if diffs else None,
        "max_first_diff": max(diffs) if diffs else None,
        "first_diff_buckets": {
            name: sum(lo <= d <= hi for d in diffs)
            for name, lo, hi in FIRST_DIFF_BUCKETS
        },
        "total_common_prefix": sum(r["common_prefix"] for r in rows),
        "mismatching_idx": [r["idx"] for r in rows if not r["exact"]],
        "unmatched": sum(not r.get("matched", True) for r in rows),
        "looping": [sum(r.get("loop_period", (None, None))[k] is not None for r in rows) for k in (0, 1)],
        "same_loop_period": sum(len(set(r.get("loop_period", (None, None)))) == 1 for r in rows),
    }


def fmt_acc(value):
    return "-" if value is None else f"{value:.2f}"


def cache_free(rows):
    return [r for r in rows if not any(int(c or 0) for c in r["cache_tokens"])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?"); ap.add_argument("b", nargs="?")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--pair", action="append", default=[], help="LABEL_A=A.jsonl,LABEL_B=B.jsonl (repeatable)")
    ap.add_argument("--cache-free", action="store_true")
    ap.add_argument("--key", default="idx", help="comma-separated fields that pair records (default idx)")
    ap.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--rows", action="store_true", help="print every row as a JSON line")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    pairs = [tuple(tuple(side.split("=", 1)) for side in p.split(",")) for p in args.pair]
    if args.a and args.b:
        pairs.insert(0, ((args.label_a, args.a), (args.label_b, args.b)))
    if not pairs:
        ap.error("give A.jsonl B.jsonl or --pair")
    if args.out and len(pairs) > 1:
        ap.error("--out writes one pair's table")
    key = tuple(args.key.split(","))
    where = [tuple(w.split("=", 1)) for w in args.where]
    for (la, pa), (lb, pb) in pairs:
        A, B = load(pa, key, where), load(pb, key, where)
        rows = compare(A, B)
        summary = summarize(rows, f"{la} vs {lb}")
        if args.rows:
            for r in rows:
                print(json.dumps({"pair": summary["pair"], **{k: (list(v) if isinstance(v, tuple) else v)
                                                              for k, v in r.items()},
                                  "idx": list(r["idx"]) if isinstance(r["idx"], tuple) else r["idx"],
                                  **{f: A[r["idx"]].get(f) for f in ("phase", "concurrency")
                                     if f in A[r["idx"]]}}))
        summary["acceptance"] = {la: acceptance(A), lb: acceptance(B)}
        if args.cache_free:
            free = cache_free(rows)
            summary["cache_free"] = {**summarize(free, summary["pair"]), "idx": [r["idx"] for r in free]}
        print(json.dumps(summary))
    if args.out:
        lines = [f"# {summary['pair']}: exact {summary['exact']}/{summary['prompts']}, token-identical "
                 f"{summary['token_identical']}/{summary['prompts']}\n",
                 "| idx | endpoint | exact | first diff token | common prefix | len a | len b | finish a/b | completion tokens a/b | prefix-cache tokens a/b | acceptance a/b | loop period a/b |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            fd = "-" if r["first_diff_token"] is None else r["first_diff_token"]
            lines.append(f"| {r['idx']} | {r['endpoint']} | {'yes' if r['exact'] else 'NO'} | {fd} | {r['common_prefix']} | "
                         f"{r['len_a']} | {r['len_b']} | {r['finish'][0]}/{r['finish'][1]} | "
                         f"{r['completion_tokens'][0]}/{r['completion_tokens'][1]} | {r['cache_tokens'][0]}/{r['cache_tokens'][1]} | "
                         f"{fmt_acc(r['acc_len'][0])}/{fmt_acc(r['acc_len'][1])} | {r['loop_period'][0]}/{r['loop_period'][1]} |")
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
