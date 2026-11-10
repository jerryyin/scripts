#!/usr/bin/env python3
"""Compare two greedy-output files written by greedy_client.py, prompt by prompt.

usage: compare_outputs.py A.jsonl B.jsonl [--label-a A] [--label-b B] [--out report.md] [--cache-free]
       compare_outputs.py --pair LABEL_A=A.jsonl,LABEL_B=B.jsonl [--pair ...] [--cache-free]

For each prompt index: exact match of the generated text (reasoning + content for chat), the first
token where the re-encoded outputs differ, the common token prefix, finish reasons, completion
token counts and prefix-cache hits. Prints a summary line per pair (exact, token-identical, differing
at token 0, median and minimum first differing token, total common prefix, mean / min / max
acceptance length of each side) and, with --out, writes a Markdown table. --cache-free adds the
same counts over the prompts with no prefix-cache hit on either side. Exits 0 whether or not the
outputs match; deciding what a mismatch means is the caller's job.
"""
import argparse
import json
import statistics
import sys


def load(path):
    with open(path) as fh:
        return {r["idx"]: r for r in map(json.loads, fh)}


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
    for i in sorted(a):
        ra, rb = a[i], b.get(i, {})
        ta, tb = ra.get("reencoded_ids") or [], rb.get("reencoded_ids") or []
        prefix = common_prefix(ta, tb)
        rows.append({
            "idx": i,
            "endpoint": "chat" if "chat" in ra.get("endpoint", "") else "completions",
            "exact": generated(ra) == generated(rb) and ra.get("http_status") == 200 and rb.get("http_status") == 200,
            "first_diff_token": None if ta == tb else prefix,
            "common_prefix": prefix, "len_a": len(ta), "len_b": len(tb),
            "finish": (ra.get("finish_reason"), rb.get("finish_reason")),
            "completion_tokens": (ra.get("completion_tokens"), rb.get("completion_tokens")),
            "cache_tokens": ((ra.get("request_stats") or {}).get("cache_tokens"),
                             (rb.get("request_stats") or {}).get("cache_tokens")),
        })
    return rows


def acceptance(outputs):
    v = [r["acc_len"] for r in outputs.values() if r.get("acc_len") is not None]
    return {"n": len(v), "mean": statistics.mean(v), "min": min(v), "max": max(v)} if v else None


def summarize(rows, label):
    diffs = [r["first_diff_token"] for r in rows if r["first_diff_token"] is not None]
    return {
        "pair": label,
        "exact": sum(r["exact"] for r in rows), "prompts": len(rows),
        "token_identical": sum(r["first_diff_token"] is None for r in rows),
        "differ_at_token_0": sum(d == 0 for d in diffs),
        "median_first_diff": statistics.median(diffs) if diffs else None,
        "min_first_diff": min(diffs) if diffs else None,
        "total_common_prefix": sum(r["common_prefix"] for r in rows),
        "mismatching_idx": [r["idx"] for r in rows if not r["exact"]],
    }


def cache_free(rows):
    return [r for r in rows if not any(int(c or 0) for c in r["cache_tokens"])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?"); ap.add_argument("b", nargs="?")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--pair", action="append", default=[], help="LABEL_A=A.jsonl,LABEL_B=B.jsonl (repeatable)")
    ap.add_argument("--cache-free", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    pairs = [tuple(tuple(side.split("=", 1)) for side in p.split(",")) for p in args.pair]
    if args.a and args.b:
        pairs.insert(0, ((args.label_a, args.a), (args.label_b, args.b)))
    if not pairs:
        ap.error("give A.jsonl B.jsonl or --pair")
    if args.out and len(pairs) > 1:
        ap.error("--out writes one pair's table")
    for (la, pa), (lb, pb) in pairs:
        A, B = load(pa), load(pb)
        rows = compare(A, B)
        summary = summarize(rows, f"{la} vs {lb}")
        summary["acceptance"] = {la: acceptance(A), lb: acceptance(B)}
        if args.cache_free:
            free = cache_free(rows)
            summary["cache_free"] = {**summarize(free, summary["pair"]), "idx": [r["idx"] for r in free]}
        print(json.dumps(summary))
    if args.out:
        lines = [f"# {summary['pair']}: exact {summary['exact']}/{summary['prompts']}, token-identical "
                 f"{summary['token_identical']}/{summary['prompts']}\n",
                 "| idx | endpoint | exact | first diff token | common prefix | len a | len b | finish a/b | completion tokens a/b | prefix-cache tokens a/b |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            fd = "-" if r["first_diff_token"] is None else r["first_diff_token"]
            lines.append(f"| {r['idx']} | {r['endpoint']} | {'yes' if r['exact'] else 'NO'} | {fd} | {r['common_prefix']} | "
                         f"{r['len_a']} | {r['len_b']} | {r['finish'][0]}/{r['finish'][1]} | "
                         f"{r['completion_tokens'][0]}/{r['completion_tokens'][1]} | {r['cache_tokens'][0]}/{r['cache_tokens'][1]} |")
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
