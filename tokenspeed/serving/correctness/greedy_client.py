#!/usr/bin/env python3
"""Send the greedy prompt set (prompts.jsonl) to a ts serve gateway and record outputs.

usage: greedy_client.py --url http://127.0.0.1:8131 --prompts prompts.jsonl --mode seq|conc \
           --out outputs-A.jsonl [--server-log server.log] [--tokenizer <model dir>]
Bodies are sent exactly as stored in prompts.jsonl. Per response we record: text (completions:
choices[0].text; chat: message.content and message.reasoning_content), finish_reason,
usage.completion_tokens, any token ids / logprobs the API returned, the tokenizer re-encoding
of the generated text, the raw response JSON, and (if --server-log) the RequestStats acc_len
of the request whose id matches the response id.
"""
import argparse, asyncio, json, os, sys, time
import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_log import parse_lines  # noqa: E402


def stats_from_log(path):
    """Request id -> RequestStats fields, from the server log's Finish lines."""
    if not path:
        return {}
    with open(path, errors="replace") as f:
        return {e["rid"]: e["stats"] for e in parse_lines(f) if e["kind"] == "Finish"}


async def one(sess, url, row):
    t0 = time.time()
    async with sess.post(url + row["endpoint"], json=row["body"]) as r:
        status = r.status
        txt = await r.text()
    t1 = time.time()
    try:
        js = json.loads(txt)
    except Exception:
        js = {"_raw": txt}
    rec = {"idx": row["idx"], "endpoint": row["endpoint"], "http_status": status,
           "t_start": t0, "latency_s": t1 - t0, "id": js.get("id"), "raw": js}
    ch = (js.get("choices") or [{}])[0]
    if row["endpoint"] == "/v1/completions":
        rec["text"] = ch.get("text")
    else:
        msg = ch.get("message") or {}
        rec["text"] = msg.get("content")
        rec["reasoning_content"] = msg.get("reasoning_content")
    rec["finish_reason"] = ch.get("finish_reason")
    rec["completion_tokens"] = (js.get("usage") or {}).get("completion_tokens")
    for k in ("token_ids", "output_token_ids", "logprobs"):
        if ch.get(k) is not None:
            rec["api_" + k] = ch.get(k)
    return rec


async def run(a, rows):
    to = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=to) as sess:
        if a.mode == "seq":
            res = [await one(sess, a.url, r) for r in rows]
        else:
            res = await asyncio.gather(*[one(sess, a.url, r) for r in rows])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--mode", choices=["seq", "conc"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log", default="")
    ap.add_argument("--tokenizer", required=True, help="model directory (tokenizer and chat template)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts)]
    t0 = time.time()
    res = asyncio.run(run(a, rows))
    wall = time.time() - t0
    time.sleep(2)
    st = stats_from_log(a.server_log)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    except Exception as e:
        print("tokenizer unavailable:", e)
    accs = []
    with open(a.out, "w") as f:
        for r in sorted(res, key=lambda x: x["idx"]):
            s = None
            if r.get("id"):
                for k, v in st.items():
                    if k == r["id"] or k.startswith(r["id"]):
                        s = v; break
            r["request_stats"] = s
            r["acc_len"] = None
            if s and s.get("acc_len") is not None:
                r["acc_len"] = float(s["acc_len"]); accs.append(r["acc_len"])
            if tok is not None:
                gen = (r.get("reasoning_content") or "") + (r.get("text") or "") if r["endpoint"] != "/v1/completions" else (r.get("text") or "")
                r["reencoded_ids"] = tok.encode(gen, add_special_tokens=False)
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    ok = sum(1 for r in res if r["http_status"] == 200 and r.get("text") is not None)
    summ = {"mode": a.mode, "n": len(res), "ok": ok, "wall_s": wall,
            "acc_len_n": len(accs), "acc_len_mean": (sum(accs) / len(accs)) if accs else None,
            "finish_reasons": sorted(set(str(r["finish_reason"]) for r in res))}
    print(json.dumps(summ))
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
