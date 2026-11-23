#!/usr/bin/env python3
"""Compare two greedy-output files written by greedy_client.py, prompt by prompt.

usage: compare_outputs.py A.jsonl B.jsonl [--label-a A] [--label-b B] [--out report.md] [--cache-free]
       compare_outputs.py --pair LABEL_A=A.jsonl,LABEL_B=B.jsonl [--pair ...] [--cache-free]
       [--key FIELD[,FIELD...]] [--where FIELD=VALUE ...] [--rows]

--key pairs records by other fields than idx, e.g. --key prompt_sha256 for amd-tracking
evalscope_degeneracy.py --requests records (the same prompt in two serving runs; a key seen k times in a
file pairs its i-th occurrences). --where is numerics/range_compare.py's filter: FIELD=VALUE keeps records
whose FIELD equals VALUE (as text), e.g. --where concurrency=1, FIELD~REGEX those whose FIELD matches REGEX
(re.search), e.g. --where 'phase~^measured'; a record without the FIELD is kept. Each row also gives both
sides' acceptance (acc_len) and loop period (loop_period
of the record, else computed from the re-encoded ids: the smallest p <= 16 with t[i] == t[i-p] over the
last 64 tokens, amd-tracking evalscope_degeneracy.loop_period). --rows prints every row as a JSON line.

For each prompt index: exact match of the generated text (reasoning + content for chat), the first
token where the re-encoded outputs differ, the common token prefix, finish reasons, completion
token counts and prefix-cache hits. Prints a summary line per pair (exact, token-identical, differing
at token 0, median, minimum and maximum first differing token, first_diff_buckets: how many
token-differing prompts first differ at token 0, 1-4, 5-31, 32-127, 128+, each such prompt counted
once, total common prefix, unmatched: records of either side with no partner, mean / min / max
acceptance length of each side) and, with --out, writes a Markdown table. --cache-free adds the same
counts over the prompts with no prefix-cache hit on either side. Exits 0 whether or not the outputs match; deciding what a mismatch means is
the caller's job.

--accept-server-log LABEL=LOG (repeatable) replaces each record's acc_len of side LABEL with the server's own
per-request acceptance (batch_log kind "Accept": `Req: RID Finish! Accept_num_tokens_avg: X`, matched by
request_id = rid), keeping EvalScope's value as acc_len_evalscope; records with no server line keep EvalScope's.
--accept-test adds "accept_test" to each pair's summary: per side n / mean / median / min / max acceptance
over records; looping per prompt (a prompt with several records, e.g. repetitions paired by --key, loops when
more than half of its classifiable records loop; a tie does not; failed records and records with neither ids
nor a loop period are left out and counted as unclassifiable_records): looping_prompts of prompts with the
Wilson 95% interval, and the mean acceptance of looping and non-looping outputs; the
unpaired difference of means (B - A) with a bootstrap 95% interval (each side resampled on its own); and over
the matched keys the paired per-prompt difference B - A: mean, median, the sign test (positive / negative / zero
counts, exact two-sided binomial p over the non-zero ones) and a bootstrap 95% interval of the mean, also
restricted to token-identical pairs (same output: a difference there is draft/target agreement, not the
continuation), both-looping pairs and both-non-looping pairs. --bootstrap N (default 10000) and --seed (default
0) fix the resampling.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

SERVING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVING)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(SERVING)), "numerics"))
from batch_log import acceptance_by_rid, percentile  # noqa: E402
from range_compare import keep, parse_where  # noqa: E402


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
    """{key: record} of a JSONL file; where is range_compare.parse_where's output (a record without the field is
    kept)."""
    out, seen = {}, {}
    with open(path) as fh:
        for r in map(json.loads, (line for line in fh if line.strip())):
            if not keep(r, where):
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


def summarize(rows, label, b_only=0):
    """Counts over compare()'s rows (one per record of A); b_only is the number of B records with no A record,
    counted in unmatched with A's records that have no B record."""
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
        "unmatched": sum(not r.get("matched", True) for r in rows) + b_only, "unmatched_b_only": b_only,
        "looping": [sum(r.get("loop_period", (None, None))[k] is not None for r in rows) for k in (0, 1)],
        "same_loop_period": sum(len(set(r.get("loop_period", (None, None)))) == 1 for r in rows),
    }


def fmt_acc(value):
    return "-" if value is None else f"{value:.2f}"


def cache_free(rows):
    return [r for r in rows if not any(int(c or 0) for c in r["cache_tokens"])]


def use_server_acceptance(records, server):
    """Replace acc_len by the server value (by request_id); return how many records matched."""
    n = 0
    for r in records.values():
        v = server.get(r.get("request_id"))
        if v is not None:
            r["acc_len_evalscope"], r["acc_len"], r["acc_source"] = r.get("acc_len"), v, "server"
            n += 1
        else:
            r["acc_source"] = "evalscope"
    return n


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def sign_test_p(pos, neg):
    """Exact two-sided binomial p (p = 1/2) of pos successes in pos + neg trials."""
    n = pos + neg
    if not n:
        return None
    tail = sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_mean(values, n_boot, rng):
    if len(values) < 2:
        return None
    m = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_boot)]
    return [percentile(m, 0.025), percentile(m, 0.975)]


def boot_diff(a, b, n_boot, rng):
    if len(a) < 2 or len(b) < 2:
        return None
    m = [statistics.mean(rng.choices(b, k=len(b))) - statistics.mean(rng.choices(a, k=len(a)))
         for _ in range(n_boot)]
    return [percentile(m, 0.025), percentile(m, 0.975)]


def paired_stats(diffs, n_boot, rng):
    if not diffs:
        return {"n": 0}
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"n": len(diffs), "mean_diff": statistics.mean(diffs), "median_diff": statistics.median(diffs),
            "min_diff": min(diffs), "max_diff": max(diffs), "positive": pos, "negative": neg,
            "zero": len(diffs) - pos - neg, "sign_test_p": sign_test_p(pos, neg),
            "bootstrap95_mean_diff": boot_mean(diffs, n_boot, rng)}


def classifiable(r):
    """True when looping can be judged for the record: the request succeeded and it has a loop period or
    re-encoded ids."""
    if r.get("success") is False or r.get("http_status", 200) != 200:
        return False
    return r.get("loop_period") is not None or bool(r.get("reencoded_ids"))


def prompt_of(key):
    """The prompt of a load() key: a --key tuple without its occurrence index; an idx key is its own prompt."""
    return key[:-1] if isinstance(key, tuple) else key


def side_stats(records):
    """Acceptance over the records ({load() key: record}) and looping per prompt. Repeated greedy runs of one
    prompt are not independent, so each prompt counts once: looping when its output loops in more than half
    of its classifiable records (a tie is not looping); records that cannot be classified (failed, or no ids
    and no loop period) are left out, and a prompt with none is not counted. The Wilson interval is over
    prompts."""
    acc = [r["acc_len"] for r in records.values() if r.get("acc_len") is not None]
    loops = {k: period_of(r) is not None for k, r in records.items() if classifiable(r)}
    votes = {}
    for k, flag in loops.items():
        votes.setdefault(prompt_of(k), []).append(flag)
    looping = sum(2 * sum(v) > len(v) for v in votes.values())
    def by(flag):
        return [records[k]["acc_len"] for k, l in loops.items() if l == flag and records[k].get("acc_len") is not None]
    def mean(v):
        return statistics.mean(v) if v else None
    return {"n": len(acc), "mean": mean(acc), "median": statistics.median(acc) if acc else None,
            "min": min(acc) if acc else None, "max": max(acc) if acc else None,
            "acc_source": sorted({r.get("acc_source", "evalscope") for r in records.values()}),
            "looping_prompts": looping, "prompts": len(votes),
            "looping_fraction": looping / len(votes) if votes else None,
            "looping_wilson95": wilson(looping, len(votes)),
            "looping_records": sum(loops.values()), "records": len(records),
            "unclassifiable_records": len(records) - len(loops),
            "acc_looping_mean": mean(by(True)), "acc_looping_n": len(by(True)),
            "acc_nonlooping_mean": mean(by(False)), "acc_nonlooping_n": len(by(False))}


def accept_test(A, B, rows, la, lb, n_boot=10000, seed=0):
    """Acceptance distributions of both sides and the unpaired and paired (per key) differences B - A."""
    rng = random.Random(seed)
    a_acc = [r["acc_len"] for r in A.values() if r.get("acc_len") is not None]
    b_acc = [r["acc_len"] for r in B.values() if r.get("acc_len") is not None]
    matched = [r for r in rows if r.get("matched") and None not in r["acc_len"]]
    def d(rs):
        return [r["acc_len"][1] - r["acc_len"][0] for r in rs]
    def lp(r, k):
        return r["loop_period"][k] is not None
    return {la: side_stats(A), lb: side_stats(B),
            "unpaired": {"mean_diff": (statistics.mean(b_acc) - statistics.mean(a_acc)) if a_acc and b_acc else None,
                         "bootstrap95_mean_diff": boot_diff(a_acc, b_acc, n_boot, rng)},
            "paired": paired_stats(d(matched), n_boot, rng),
            "paired_token_identical": paired_stats(d([r for r in matched if r["first_diff_token"] is None]), n_boot, rng),
            "paired_both_looping": paired_stats(d([r for r in matched if lp(r, 0) and lp(r, 1)]), n_boot, rng),
            "paired_both_nonlooping": paired_stats(d([r for r in matched if not lp(r, 0) and not lp(r, 1)]), n_boot, rng),
            "bootstrap": n_boot, "seed": seed}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?"); ap.add_argument("b", nargs="?")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--pair", action="append", default=[], help="LABEL_A=A.jsonl,LABEL_B=B.jsonl (repeatable)")
    ap.add_argument("--cache-free", action="store_true")
    ap.add_argument("--key", default="idx", help="comma-separated fields that pair records (default idx)")
    ap.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE|FIELD~REGEX")
    ap.add_argument("--rows", action="store_true", help="print every row as a JSON line")
    ap.add_argument("--out")
    ap.add_argument("--accept-server-log", action="append", default=[], metavar="LABEL=LOG",
                    help="use the server's per-request acceptance for side LABEL (repeatable)")
    ap.add_argument("--accept-test", action="store_true", help="acceptance distributions and paired tests per pair")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pairs = [tuple(tuple(side.split("=", 1)) for side in p.split(",")) for p in args.pair]
    if args.a and args.b:
        pairs.insert(0, ((args.label_a, args.a), (args.label_b, args.b)))
    if not pairs:
        ap.error("give A.jsonl B.jsonl or --pair")
    if args.out and len(pairs) > 1:
        ap.error("--out writes one pair's table")
    key = tuple(args.key.split(","))
    try:
        where = parse_where(args.where)
    except ValueError as exc:
        ap.error(str(exc))
    logs = {}
    for spec in args.accept_server_log:
        label, path = spec.split("=", 1)
        logs.setdefault(label, {}).update(acceptance_by_rid([path]))
    for (la, pa), (lb, pb) in pairs:
        A, B = load(pa, key, where), load(pb, key, where)
        server_matched = {lab: [use_server_acceptance(R, logs[lab]), len(R)]
                          for lab, R in ((la, A), (lb, B)) if lab in logs}
        rows = compare(A, B)
        summary = summarize(rows, f"{la} vs {lb}", sum(k not in A for k in B))
        if args.rows:
            for r in rows:
                print(json.dumps({"pair": summary["pair"], **{k: (list(v) if isinstance(v, tuple) else v)
                                                              for k, v in r.items()},
                                  "idx": list(r["idx"]) if isinstance(r["idx"], tuple) else r["idx"],
                                  **{f: A[r["idx"]].get(f) for f in ("phase", "concurrency")
                                     if f in A[r["idx"]]}}))
        summary["acceptance"] = {la: acceptance(A), lb: acceptance(B)}
        if server_matched:
            summary["server_acceptance_matched"] = server_matched
        if args.accept_test:
            summary["accept_test"] = accept_test(A, B, rows, la, lb, args.bootstrap, args.seed)
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
