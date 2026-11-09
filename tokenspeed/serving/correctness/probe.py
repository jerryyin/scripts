#!/usr/bin/env python3
"""Probe the server's own top-1 token where two greedy outputs first diverge.

usage: probe.py --url URL --tokenizer MODEL_DIR --prompts prompts.jsonl --a A.jsonl --b B.jsonl --out PREFIX

For each prompt whose re-encoded outputs X (from --a) and Y (from --b) differ, this sends
input_ids = ids(rendered prompt) + common prefix to /generate (greedy, one new token,
return_logprob) after POST /flush_cache, and reports the server's top-1 token t1 with logprob l1.
For a candidate Z that is not t1, logprob(Z) <= log(1 - exp(l1)), so the gap l1 - logprob(Z) is at
least l1 - log1p(-exp(l1)): a lower bound on how far the other side's token was from the model's
choice. Writes PREFIX.raw.jsonl (every request and response) and PREFIX.json (one row per prompt).
Note (TokenSpeed 0dd8d477): /v1/completions drops logprobs and /generate ignores top_logprobs_num,
so only top-1 is available; hence the bound rather than both logprobs.
"""
import argparse, json, math, os, sys, time, urllib.error, urllib.request
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_outputs import common_prefix, load  # noqa: E402


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            st, txt = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        st, txt = e.code, e.read().decode()
    return st, txt, time.time() - t


def rendered(tok, r):
    if "chat" in r["endpoint"]:
        return tok.apply_chat_template(r["body"]["messages"], add_generation_prompt=True, tokenize=False)
    return r["body"]["prompt"]


def probe_one(args, tok, row, rx, ry, raw):
    """Probe one prompt whose outputs differ; returns its result row."""
    pair = f"{args.label_a} vs {args.label_b}"
    a, b = rx.get("reencoded_ids") or [], ry.get("reencoded_ids") or []
    k = common_prefix(a, b)
    c = a[:k]; ctext = tok.decode(c)
    reenc_ok = tok.encode(ctext, add_special_tokens=False) == c
    p = rendered(tok, row); ids_p = tok.encode(p, add_special_tokens=False)
    text_boundary_ok = tok.encode(p + ctext, add_special_tokens=False) == ids_p + c
    reasoning_len = None
    if "chat" in row["endpoint"]:
        reasoning_len = len(tok.encode(rx.get("reasoning_content") or "", add_special_tokens=False))
    fs = post(args.url, "/flush_cache", {})
    body = {"input_ids": ids_p + c, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "top_logprobs_num": 5}
    st, txt, _ = post(args.url, "/generate", body)
    try:
        js = json.loads(txt)
    except ValueError:
        js = {"_raw": txt}
    raw.write(json.dumps({"pair": pair, "idx": row["idx"], "flush_cache": {"status": fs[0], "body": fs[1][:300]},
                          "request": body, "http_status": st, "response": js}, ensure_ascii=False) + "\n")
    raw.flush()
    mi = js.get("meta_info") or {}; otl = mi.get("output_token_logprobs") or [[None, None]]
    l1, t1 = otl[0][0], (int(otl[0][1]) if otl[0][1] is not None else None)
    tx = a[k] if k < len(a) else None; ty = b[k] if k < len(b) else None
    p1 = math.exp(l1) if l1 is not None else None
    bound = (l1 - math.log1p(-p1)) if (p1 is not None and p1 < 1) else float("inf")
    dec = lambda t: [t, tok.decode([t]) if t is not None else None]
    return {"pair": pair, "idx": row["idx"], "endpoint": row["endpoint"], "k_common_prefix": k,
            "len_X": len(a), "len_Y": len(b), "reencode_roundtrip_ok": reenc_ok, "text_boundary_ok": text_boundary_ok,
            "prompt_ids_len": len(ids_p), "reasoning_tokens_X": reasoning_len,
            "div_in_reasoning": (None if reasoning_len is None else k < reasoning_len),
            "http_status": st, "probe_top1": dec(t1), "probe_top1_logprob": l1, "probe_top1_prob": p1,
            "X_token": dec(tx), "Y_token": dec(ty),
            "probe_top1_is": ("X" if t1 == tx else "Y" if t1 == ty else "neither"),
            "X_logprob": l1 if t1 == tx else None, "Y_logprob": l1 if t1 == ty else None,
            "gap_lower_bound_nonTop1_candidate": bound,
            "context_tail": tok.decode((ids_p + c)[-12:])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = load(args.prompts)
    fx, fy = load(args.a), load(args.b)
    res = []
    with open(args.out + ".raw.jsonl", "w") as raw:
        for idx in sorted(fx):
            if (fx[idx].get("reencoded_ids") or []) == (fy[idx].get("reencoded_ids") or []):
                continue
            rec = probe_one(args, tok, rows[idx], fx[idx], fy[idx], raw)
            res.append(rec)
            print(json.dumps({k: rec[k] for k in ("pair", "idx", "k_common_prefix", "probe_top1_is", "probe_top1_prob",
                                                  "gap_lower_bound_nonTop1_candidate", "text_boundary_ok",
                                                  "reencode_roundtrip_ok")}), flush=True)
    with open(args.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
