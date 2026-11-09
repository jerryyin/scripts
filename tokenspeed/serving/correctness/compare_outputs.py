#!/usr/bin/env python3
"""Compare two greedy-output files written by greedy_client.py, prompt by prompt.

usage: compare_outputs.py A.jsonl B.jsonl [--label-a A] [--label-b B] [--out report.md]

For each prompt index: exact match of the generated text (reasoning + content for chat), the first
token where the re-encoded outputs differ, the common token prefix, finish reasons, completion
token counts and prefix-cache hits. Prints a summary line and, with --out, writes a Markdown table.
Exits 0 whether or not the outputs match; deciding what a mismatch means is the caller's job.
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    rows = compare(load(args.a), load(args.b))
    diffs = [r["first_diff_token"] for r in rows if r["first_diff_token"] is not None]
    summary = {
        "pair": f"{args.label_a} vs {args.label_b}",
        "exact": sum(r["exact"] for r in rows), "prompts": len(rows),
        "token_identical": sum(r["first_diff_token"] is None for r in rows),
        "differ_at_token_0": sum(d == 0 for d in diffs),
        "median_first_diff": statistics.median(diffs) if diffs else None,
        "mismatching_idx": [r["idx"] for r in rows if not r["exact"]],
    }
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
