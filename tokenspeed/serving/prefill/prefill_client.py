#!/usr/bin/env python3
"""One prefill point against a TokenSpeed `ts serve` gateway (smg).

/flush_cache, then B simultaneous streaming /v1/completions with distinct natural-corpus
prompts of exactly L tokens (natural_prompts.make_prompts_natural, from
../decode), temperature 0, max_tokens 1 (default), ignore_eos.
Tails the server log during the point, waits until no new batch/Finish line arrives for
--idle-quiet s, and writes into --out:
  requests.json          per request: t_send, t_first (first streamed chunk), t_end,
                         finish_reason, usage (prompt_tokens), text, prompt offset info
  server-log-slice.txt   log lines appended during the point
  metrics.json           client TTFT stats, batch prompt throughput, server per-rank
                         chunk counts/tokens/times, verification flags
Clocks: client time.time() (epoch) and Rust log lines (explicit UTC 'Z') share the host clock.
"""
import argparse, asyncio, json, os, statistics, sys, time
import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "decode"), os.path.join(HERE, "..")]
from batch_log import LogTail, parse_lines            # noqa: E402
from fbdw_client import one_request, get_tok           # noqa: E402
from natural_prompts import make_prompts_natural        # noqa: E402



def pct(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q; i = int(k)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (k - i) if xs else None


def server_metrics(events, B, L, t_first_send):
    ev = [e for e in events if e["ts"] is not None and e["ts"] >= t_first_send - 1.0]
    pf = [e for e in ev if e["kind"] in ("Prefill", "Mix")]
    fin = [e for e in ev if e["kind"] == "Finish"]
    ranks = {}
    for r in sorted({e["dp_rank"] for e in pf}):
        P = sorted((e for e in pf if e["dp_rank"] == r), key=lambda e: e["ts"])
        allr = sorted((e for e in ev if e.get("dp_rank") == r and e["kind"] != "Finish"), key=lambda e: e["ts"])
        gaps = []  # per-chunk time = next line of this rank - this Prefill line (line = round dispatch)
        for a, b in zip(allr, allr[1:]):
            if a["kind"] in ("Prefill", "Mix"):
                gaps.append({"chunk_tokens": a.get("new_token", 0), "new_seq": a.get("new_seq", 0),
                             "dt_s": b["ts"] - a["ts"], "next_kind": b["kind"]})
        pp = [g["dt_s"] for g in gaps if g["next_kind"] in ("Prefill", "Mix")]
        med = statistics.median(pp) if pp else None; med_src = "prefill->prefill line gaps"
        if med is None and gaps:
            med = statistics.median(g["dt_s"] for g in gaps); med_src = "prefill->any next line gaps"
        if med is None:  # one chunk and no later line on this rank: RequestStats prefill_ms (includes queueing)
            pm = [float(e["stats"].get("prefill_ms", 0)) / 1e3 for e in fin]
            med = statistics.median(pm) if pm else None; med_src = "RequestStats prefill_ms median (upper bound)"
        tok = sum(e.get("new_token", 0) for e in P)
        span = (P[-1]["ts"] - P[0]["ts"] + (med or 0)) if P else 0
        by = {}
        for g in gaps:
            if g["next_kind"] in ("Prefill", "Mix"):
                by.setdefault(g["chunk_tokens"], []).append(g["dt_s"])
        ranks[str(r)] = {
            "chunks": len(P), "prefill_tokens": tok, "requests_by_tokens": tok / L,
            "first_prefill_line_rel_s": P[0]["ts"] - t_first_send if P else None,
            "last_prefill_line_rel_s": P[-1]["ts"] - t_first_send if P else None,
            "chunk_time_median_s": med, "chunk_time_source": med_src, "chunk_time_p10_s": pct(pp, .1), "chunk_time_p90_s": pct(pp, .9),
            "chunk_time_by_tokens": {str(k): {"n": len(v), "median_s": statistics.median(v),
                                              "tok_per_s": k / statistics.median(v) if statistics.median(v) > 0 else None}
                                     for k, v in sorted(by.items())},
            "prefill_span_s": span, "prefill_tok_per_s": tok / span if span else None,
            "chunk_gaps": gaps}
    tot = sum(x["prefill_tokens"] for x in ranks.values())
    st = [e["stats"] for e in fin]
    return {"ranks": ranks, "prefill_tokens_total": tot, "prefill_tokens_expected": B * L,
            "prefill_tokens_ok": tot == B * L,
            "requeststats_n": len(st),
            "requeststats_prompt_tokens": sorted({int(s.get("prompt_tokens", -1)) for s in st}),
            "requeststats_cache_tokens": sorted({int(s.get("cache_tokens", -1)) for s in st}),
            "requeststats_prefill_ms": [float(s.get("prefill_ms", 0)) for s in st],
            "requeststats_ok": len(st) == B and all(int(s.get("prompt_tokens", -1)) == L and int(s.get("cache_tokens", 0)) == 0 for s in st),
            "server_tok_per_s_all_ranks": (tot / max(x["prefill_span_s"] for x in ranks.values())) if ranks and max(x["prefill_span_s"] for x in ranks.values()) > 0 else None}


async def run(a):
    tok = get_tok(a.tokenizer)
    nspec = len(tok.encode("x")) - len(tok.encode("x", add_special_tokens=False))  # as fbdw_client.py
    prompts, pinfo = make_prompts_natural(tok, a.batch, a.context, a.seed, a.corpus, nspec, a.prompt_token_adjust)
    os.makedirs(a.out, exist_ok=True)
    tail = LogTail(a.server_log)
    timeout = aiohttp.ClientTimeout(total=a.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        try:
            async with s.post(a.url + "/flush_cache") as r:
                flush = {"status": r.status, "body": (await r.text())[:300]}
        except Exception as e:  # noqa
            flush = {"error": repr(e)}
        await asyncio.sleep(a.settle)
        recs = [{"i": i, "t_send": None, "t_first": None, "t_end": None, "chunks": [], "texts": []} for i in range(a.batch)]
        extra = json.loads(a.extra) if a.extra else {}
        await asyncio.gather(*(one_request(s, a.url, a.model, p, a.max_tokens, rec, extra)
                               for p, rec in zip(prompts, recs)))
    # wait for the log to go quiet (Finish lines are written at completion)
    last_n, last_change = -1, time.time()
    while time.time() - last_change < a.idle_quiet:
        n = len(tail.poll())
        if n != last_n:
            last_n, last_change = n, time.time()
        await asyncio.sleep(0.2)
    lines = tail.poll()
    open(os.path.join(a.out, "server-log-slice.txt"), "w").writelines(lines)
    for rec, (off, sh, k) in zip(recs, pinfo["offsets_shift_k"]):
        rec["text"] = "".join(rec.pop("texts")); rec["prompt_offset"] = [off, sh, k]
        rec["n_chunks"] = len(rec.pop("chunks"))
    json.dump({"flush": flush, "prompt_info": {k: v for k, v in pinfo.items() if k != "offsets_shift_k"},
               "requests": recs}, open(os.path.join(a.out, "requests.json"), "w"), indent=1)
    t0 = min(r["t_send"] for r in recs)
    ok = [r for r in recs if r.get("t_first") is not None and not r.get("error")]
    ttft = [r["t_first"] - r["t_send"] for r in ok]
    t_last_first = max((r["t_first"] for r in ok), default=None)
    usage_pt = sorted({(r.get("usage") or {}).get("prompt_tokens") for r in ok})
    ev = parse_lines(lines)
    sm = server_metrics(ev, a.batch, a.context, t0)
    m = {"L": a.context, "B": a.batch, "seed": a.seed, "max_tokens": a.max_tokens, "url": a.url,
         "n_ok": len(ok), "n_err": a.batch - len(ok),
         "send_spread_s": max(r["t_send"] for r in recs) - t0,
         "ttft_s": {"mean": statistics.mean(ttft) if ttft else None, "median": statistics.median(ttft) if ttft else None,
                    "min": min(ttft, default=None), "max": max(ttft, default=None), "all": ttft},
         "batch_prompt_tok_per_s": (a.batch * a.context / (t_last_first - t0)) if t_last_first else None,
         "last_first_token_s": (t_last_first - t0) if t_last_first else None,
         "usage_prompt_tokens": usage_pt,
         "finish_reasons": sorted({str(r.get("finish_reason")) for r in ok}),
         "server": sm,
         "valid": len(ok) == a.batch and sm["prefill_tokens_ok"] and sm["requeststats_ok"]}
    for r in m["server"]["ranks"].values():
        r.pop("chunk_gaps")  # kept in metrics-chunks.json
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    json.dump(server_metrics(ev, a.batch, a.context, t0)["ranks"], open(os.path.join(a.out, "metrics-chunks.json"), "w"), indent=1)
    print(json.dumps({k: m[k] for k in ("L", "B", "n_ok", "valid", "batch_prompt_tok_per_s", "last_first_token_s")}
                     | {"ttft_mean": m["ttft_s"]["mean"], "server_tok_per_s": sm["server_tok_per_s_all_ranks"],
                        "split": {k: v["requests_by_tokens"] for k, v in sm["ranks"].items()}}))
    return 0 if m["valid"] else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True); ap.add_argument("--server-log", required=True)
    ap.add_argument("--batch", type=int, required=True); ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=1); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--tokenizer", default=os.environ.get("MODEL"), required="MODEL" not in os.environ, help="model directory; defaults to $MODEL")
    ap.add_argument("--corpus", required=True, help="corpus directory built by ../decode/natural_prompts.py"); ap.add_argument("--prompt-token-adjust", type=int, default=0)
    ap.add_argument("--idle-quiet", type=float, default=5.0); ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=1800); ap.add_argument("--extra", default="")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
