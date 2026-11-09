#!/usr/bin/env python3
"""HTTP fixed-batch decode window: one measurement point (B requests, L-token prompts).

  fbdw_client.py --url http://127.0.0.1:8131 --server-log server.log --batch 4 \
      --context 8192 --max-tokens 8192 --seed 1 --out raw/L8192_B4_rep1

Steps: POST /flush_cache; send B simultaneous streaming /v1/completions with distinct
seeded random token-ID prompts of exactly L tokens (temperature 0, ignore_eos); tail the
server log; when every active rank has >= --min-rounds qualifying decode rounds after the
window opened (see fbdw_analyze.py), abort all streams; wait for the server to go idle;
save raw client data, the server-log slice and metrics.json.
Configuration-agnostic (any DP size, speculative or not).
"""
import argparse, asyncio, json, os, re, sys, time
import aiohttp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fbdw_analyze as fa  # noqa: E402
from batch_log import LogTail, parse_lines  # noqa: E402


_WORDS = {}
def make_prompts(B, L, seed, tok_path, adjust=0):
    """Distinct seeded random prompts of exactly L tokens (as counted by the model
    tokenizer with its default special tokens, plus --prompt-token-adjust).
    The gateway accepts only string prompts, so each prompt is a run of
    space-prefixed lowercase words that are single tokens of the vocabulary; the
    pre-tokenizer splits at each space, so the prompt is exactly one token per word."""
    tok = get_tok(tok_path)
    if tok_path not in _WORDS:
        ws = []
        for t, i in tok.get_vocab().items():
            w = tok.convert_tokens_to_string([t])
            if re.fullmatch(r" [a-z]{3,12}", w) and tok.encode(w, add_special_tokens=False) == [i]:
                ws.append(w)
        _WORDS[tok_path] = sorted(ws)
    ws = _WORDS[tok_path]
    nspec = len(tok.encode("x")) - len(tok.encode("x", add_special_tokens=False))
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(B):
        k = L - nspec - adjust
        idx = rng.integers(0, len(ws), size=k)
        text = "".join(ws[j] for j in idx)
        n = len(tok.encode(text))
        if n + adjust != L:
            raise RuntimeError(f"prompt token count {n}+{adjust} != {L}")
        ps.append(text)
    assert len({p[:400] for p in ps}) == B
    return ps, {"word_vocab": len(ws), "special_tokens_added": nspec, "adjust": adjust}


_TOK = {}
def get_tok(tok_path):
    if tok_path not in _TOK:
        from transformers import AutoTokenizer
        _TOK[tok_path] = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    return _TOK[tok_path]


async def one_request(session, url, model, prompt, max_tokens, rec, extra):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True}
    body.update(extra)
    rec["t_send"] = time.time()
    try:
        async with session.post(url + "/v1/completions", json=body) as resp:
            rec["http_status"] = resp.status
            if resp.status != 200:
                rec["error"] = (await resp.text())[:2000]; rec["t_end"] = time.time(); return
            async for raw in resp.content:
                t = time.time()
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    rec["done"] = True; break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    rec.setdefault("bad_chunks", []).append(payload[:200]); continue
                ch = (obj.get("choices") or [{}])[0]
                txt = ch.get("text") or ""
                if rec["t_first"] is None:
                    rec["t_first"] = t
                rec["chunks"].append([t, len(txt)]); rec["texts"].append(txt)
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
        rec["t_end"] = time.time(); rec["ended_naturally"] = True
    except asyncio.CancelledError:
        rec["t_end"] = time.time(); rec["aborted"] = True; raise
    except Exception as e:  # noqa
        rec["t_end"] = time.time(); rec["error"] = repr(e)


async def run(a):
    os.makedirs(a.out, exist_ok=True)
    meta = {"B": a.batch, "L": a.context, "max_tokens": a.max_tokens, "seed": a.seed,
            "url": a.url, "model": a.model, "min_rounds": a.min_rounds,
            "log_interval": a.log_interval, "argv": sys.argv, "t_point_start": time.time()}
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        try:
            async with s.post(a.url + "/flush_cache") as r:
                meta["flush_cache"] = {"status": r.status, "body": (await r.text())[:500]}
        except Exception as e:  # noqa
            meta["flush_cache"] = {"status": None, "error": repr(e)}
        await asyncio.sleep(0.5)
        tail = LogTail(a.server_log)
        if a.prompt_source == "natural":
            import natural_prompts as npm
            _t = get_tok(a.tokenizer)
            _nspec = len(_t.encode("x")) - len(_t.encode("x", add_special_tokens=False))
            prompts, meta["prompt_info"] = npm.make_prompts_natural(_t, a.batch, a.context, a.seed, a.corpus_dir,
                                                                    _nspec, a.prompt_token_adjust)
        elif a.prompt_source == "file":  # prompt i = line (offset + i) of a JSONL file of strings
            with open(a.prompts_file) as _f:
                _lines = [json.loads(x) for x in _f if x.strip()]
            prompts = _lines[a.prompt_offset:a.prompt_offset + a.batch]
            if len(prompts) != a.batch:
                raise RuntimeError(f"prompts file has {len(_lines)} lines; need {a.prompt_offset}+{a.batch}")
            _n = [len(get_tok(a.tokenizer).encode(x)) + a.prompt_token_adjust for x in prompts]
            if set(_n) != {a.context}:
                raise RuntimeError(f"prompt token counts {sorted(set(_n))} != {a.context}")
            meta["prompt_info"] = {"source": "file", "prompts_file": a.prompts_file, "offset": a.prompt_offset,
                                   "lines": [a.prompt_offset, a.prompt_offset + a.batch - 1], "adjust": a.prompt_token_adjust}
        else:
            prompts, meta["prompt_info"] = make_prompts(a.batch, a.context, a.seed, a.tokenizer, a.prompt_token_adjust)
        recs = [{"i": i, "t_send": None, "t_first": None, "t_end": None, "chunks": [], "texts": []}
                for i in range(a.batch)]
        extra = json.loads(a.extra_body) if a.extra_body else {}
        tasks = [asyncio.create_task(one_request(s, a.url, a.model, p, a.max_tokens, r, extra))
                 for p, r in zip(prompts, recs)]
        t0 = time.time(); stop_reason = None; clock_offset = None; m = None
        while True:
            await asyncio.sleep(a.poll)
            lines = tail.poll()
            ev = parse_lines(lines)
            if clock_offset is None:
                # log clock -> client wall clock: offset rounded to 15 min (timezone)
                ts = [e["ts"] for e in ev if e["ts"] is not None]
                if ts:
                    clock_offset = round((time.time() - ts[-1]) / 900.0) * 900.0
            firsts = [r["t_first"] for r in recs]
            ended = [r["i"] for r in recs if r["t_end"] is not None]
            if ended:
                stop_reason = f"streams ended before abort: {ended[:10]}"; break
            if all(f is not None for f in firsts) and clock_offset is not None:
                m = fa.analyze(ev, a.batch, a.context, max(firsts), None, a.min_rounds,
                               a.log_interval, clock_offset)
                if m["enough_rounds"]:
                    stop_reason = "enough_rounds"; break
            if time.time() - t0 > a.max_seconds:
                stop_reason = "timeout"; break
        t_abort = time.time()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    meta["t_abort"] = t_abort; meta["stop_reason"] = stop_reason
    # wait for idle: no new Decode/Prefill lines for --idle-quiet seconds
    last_n, last_change = None, time.time(); t_idle_start = time.time()
    while time.time() - t_idle_start < a.idle_timeout:
        time.sleep(0.25)
        lines = tail.poll()
        n = sum(1 for l in lines if "batch." in l)
        if n != last_n:
            last_n, last_change = n, time.time()
        elif time.time() - last_change >= a.idle_quiet:
            break
    meta["idle_confirmed"] = time.time() - last_change >= a.idle_quiet
    meta["idle_wait_s"] = time.time() - t_abort
    try:
        import urllib.request
        with urllib.request.urlopen(a.url + "/get_loads", timeout=5) as r:
            meta["get_loads_after"] = r.read().decode()[:2000]
    except Exception as e:  # noqa
        meta["get_loads_after"] = repr(e)
    lines = tail.poll()
    with open(os.path.join(a.out, "server-log-slice.txt"), "w") as f:
        f.writelines(lines)
    ev = parse_lines(lines)
    firsts = [r["t_first"] for r in recs]
    meta["t_all_first"] = max(firsts) if all(f is not None for f in firsts) else None
    meta["clock_offset"] = clock_offset or 0.0
    meta["client_ended_early"] = [r["i"] for r in recs if r.get("ended_naturally") or r.get("error")]
    meta["errors"] = [{"i": r["i"], "status": r.get("http_status"), "error": r.get("error")}
                      for r in recs if r.get("error")][:20]
    m = fa.analyze(ev, a.batch, a.context, meta["t_all_first"], t_abort, a.min_rounds,
                   a.log_interval, meta["clock_offset"], meta["client_ended_early"])
    # post-abort decode lines with running>0 (for the idle check)
    post = [e for e in ev if e["kind"] == "Decode" and e["ts"] is not None
            and e["ts"] + meta["clock_offset"] > t_abort]
    meta["decode_lines_after_abort"] = len(post)
    meta["last_decode_after_abort"] = post[-1]["line"][-300:] if post else None
    fin = [e for e in ev if e["kind"] == "Finish"]
    pt = [e["stats"].get("prompt_tokens") for e in fin]
    m["requeststats_prompt_tokens"] = sorted(set(x for x in pt if x is not None))
    m["requeststats_count"] = len(fin)
    m["requeststats_preempt_count_max"] = max((e["stats"].get("preempt_count", 0) or 0 for e in fin), default=None)
    # client cross-check over the common server window
    m["client"] = client_crosscheck(recs, m, meta["clock_offset"], a.tokenizer)
    if m.get("system_tok_s") and m["client"].get("system_tok_s"):
        m["client_over_server"] = m["client"]["system_tok_s"] / m["system_tok_s"]
        for k in ("system_tok_s_per_chunk_retok", "system_chunks_s"):
            if m["client"].get(k):
                m["client_over_server_" + k] = m["client"][k] / m["system_tok_s"]
    m["meta"] = {k: meta[k] for k in ("stop_reason", "idle_confirmed", "idle_wait_s", "flush_cache",
                                      "decode_lines_after_abort", "t_abort", "t_all_first", "clock_offset")}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(a.out, "requests.json"), "w") as f:
        json.dump(recs, f)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    return m


def client_crosscheck(recs, m, off, tok_path):
    if not m.get("common_window_log"):
        return {"note": "no server window"}
    ws, we = (x + off for x in m["common_window_log"])
    if we <= ws:
        return {"note": "empty common window", "window": [ws, we]}
    tok = get_tok(tok_path)
    rates, rates_pc, rates_ch = [], [], []
    for r in recs:
        ch = r["chunks"]
        idx = [i for i, c in enumerate(ch) if ws <= c[0] <= we]
        if len(idx) < 2:
            rates.append(None); rates_pc.append(None); rates_ch.append(None); continue
        i0, i1 = idx[0], idx[-1]
        dtv = ch[i1][0] - ch[i0][0]
        # (a) method: retokenize the cumulative text at both window edges
        n0 = len(tok.encode("".join(r["texts"][: i0 + 1]), add_special_tokens=False))
        n1 = len(tok.encode("".join(r["texts"][: i1 + 1]), add_special_tokens=False))
        rates.append((n1 - n0) / dtv)
        # (b) retokenize each chunk on its own and sum (robust to cross-token merges)
        rates_pc.append(sum(max(1, len(tok.encode(r["texts"][i], add_special_tokens=False)))
                            for i in range(i0 + 1, i1 + 1)) / dtv)
        # (c) count of non-empty chunks (a lower bound when the gateway coalesces)
        rates_ch.append(sum(1 for i in range(i0 + 1, i1 + 1) if ch[i][1] > 0) / dtv)
    full = lambda v: sum(v) if all(x is not None for x in v) else None
    ok = [x for x in rates if x is not None]
    return {"window_client_clock": [ws, we], "requests_with_rate": len(ok),
            "system_tok_s": full(rates), "system_tok_s_per_chunk_retok": full(rates_pc),
            "system_chunks_s": full(rates_ch),
            "per_request_tok_s_min": min(ok) if ok else None,
            "per_request_tok_s_max": max(ok) if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8131")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("MODEL"), required="MODEL" not in os.environ, help="model directory; defaults to $MODEL")
    ap.add_argument("--min-rounds", type=int, default=100)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--prompt-token-adjust", type=int, default=0,
                    help="tokens the server adds beyond the tokenizer's default special tokens")
    ap.add_argument("--poll", type=float, default=0.2)
    ap.add_argument("--max-seconds", type=float, default=900)
    ap.add_argument("--idle-quiet", type=float, default=3.0)
    ap.add_argument("--idle-timeout", type=float, default=120)
    ap.add_argument("--extra-body", default="")
    ap.add_argument("--prompt-source", choices=["random", "natural", "file"], default="random",
                    help="random single-token words (default), natural prose (natural_prompts.py) or a prompt file")
    ap.add_argument("--corpus-dir", help="corpus directory built by natural_prompts.py (required for --prompt-source natural)")
    ap.add_argument("--prompts-file", help="JSONL of prompt strings (required for --prompt-source file)")
    ap.add_argument("--prompt-offset", type=int, default=0, help="prompt i of the batch = line (offset + i) of --prompts-file")
    a = ap.parse_args()
    if a.prompt_source == "natural" and not a.corpus_dir:
        ap.error("--prompt-source natural needs --corpus-dir")
    if a.prompt_source == "file" and not a.prompts_file:
        ap.error("--prompt-source file needs --prompts-file")
    m = asyncio.run(run(a))
    summ = {k: m.get(k) for k in ("B", "L", "active_ranks", "system_tok_s", "tpot_ms_mean", "tpot_ms_max",
                                   "enough_rounds", "fixed_batch", "client_over_server",
                                   "prefill_tokens_total", "prefill_tokens_expected",
                                   "requeststats_prompt_tokens", "avg_accept_len_mean")}
    summ["rounds"] = {r: v["qualifying_rounds"] for r, v in m["per_rank"].items()}
    summ["batch"] = {r: v["expected_batch"] for r, v in m["per_rank"].items()}
    summ["stop"] = m["meta"]["stop_reason"]; summ["idle"] = m["meta"]["idle_confirmed"]
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
