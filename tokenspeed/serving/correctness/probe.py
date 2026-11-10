#!/usr/bin/env python3
"""Probe the server's own top-1 token where two greedy outputs first diverge.

usage: probe.py --url URL --tokenizer MODEL_DIR --prompts prompts.jsonl --a A.jsonl --b B.jsonl --out PREFIX
                [--mode prefill|decode-path]
       probe.py --mode teacher-forced --url URL --tokenizer MODEL_DIR --prompts prompts.jsonl --a A.jsonl
                --label-a A --positions 24 --out PREFIX
       probe.py --summarize PREFIX.json [PREFIX.json ...]

For each prompt whose re-encoded outputs X (from --a) and Y (from --b) differ, this sends
input_ids = ids(rendered prompt) + common prefix to /generate (greedy, one new token,
return_logprob) after POST /flush_cache, and reports the server's top-1 token t1 with logprob l1.
For a candidate Z that is not t1, logprob(Z) <= log(1 - exp(l1)), so the gap l1 - logprob(Z) is at
least l1 - log1p(-exp(l1)): a lower bound on how far the other side's token was from the model's
choice. Writes PREFIX.raw.jsonl (every request and response) and PREFIX.json (one row per prompt).
Note (TokenSpeed 0dd8d477): /v1/completions drops logprobs and /generate ignores top_logprobs_num,
so only top-1 is available; hence the bound rather than both logprobs.

--mode decode-path makes the server reach the divergence through its decode path, as the original
passes did: it sends the prompt plus the common prefix minus its last m = min(k, 8) tokens, asks for
m + 1 tokens, checks that the m regenerated tokens equal the prefix tail, and scores the last one.
--mode teacher-forced scores one output file instead of a pair: for each of its first --positions
generated tokens it asks for the top-1 after the prompt plus the tokens before it, and records
whether the emitted token was the top-1 and, if not, the same lower bound on its gap.
--summarize counts, per pair, how often the top-1 is X, Y or neither and how many X and Y tokens
are certified more than 1 nat below the top-1 (rows whose request failed, or decode-path rows whose
prefix did not reproduce, are excluded and listed), and per teacher-forced label how many emitted tokens were not the top-1.
"""
import argparse, json, math, os, statistics, sys, time, urllib.error, urllib.request

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
    p1, bound = gap_bound(l1) if st == 200 and l1 is not None else (None, None)
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


def gap_bound(l1):
    p1 = math.exp(l1)
    return p1, (l1 - math.log1p(-p1)) if p1 < 1 else float("inf")


def decode_path_one(args, tok, row, rx, ry, raw):
    """Reach the divergence through the decode path; score the token at the first difference."""
    pair = f"{args.label_a} vs {args.label_b}"
    a, b = rx.get("reencoded_ids") or [], ry.get("reencoded_ids") or []
    k = common_prefix(a, b); m = min(k, 8)
    ids_p = tok.encode(rendered(tok, row), add_special_tokens=False)
    fs = post(args.url, "/flush_cache", {})
    body = {"input_ids": ids_p + a[:k - m], "sampling_params": {"max_new_tokens": m + 1, "temperature": 0},
            "return_logprob": True}
    st, txt, _ = post(args.url, "/generate", body)
    js = json.loads(txt) if txt.lstrip().startswith("{") else {"_raw": txt}
    raw.write(json.dumps({"pair": pair, "idx": row["idx"], "flush": fs[0], "request": body, "http_status": st,
                          "response": js}, ensure_ascii=False) + "\n")
    ok = st == 200
    out = (js.get("output_ids") or []) if ok else []
    otl = ((js.get("meta_info") or {}).get("output_token_logprobs") or []) if ok else []
    t1 = out[m] if len(out) > m else None; l1 = otl[m][0] if len(otl) > m else None
    p1, bound = gap_bound(l1) if l1 is not None else (None, None)
    tx = a[k] if k < len(a) else None; ty = b[k] if k < len(b) else None
    dec = lambda t: [t, tok.decode([t]) if t is not None else None]
    return {"pair": pair, "idx": row["idx"], "k_common_prefix": k, "m": m, "http_status": st,
            "regenerated_prefix_tail_matches": out[:m] == a[k - m:k],
            "probe_top1": dec(t1), "probe_top1_logprob": l1, "probe_top1_prob": p1, "X_token": dec(tx), "Y_token": dec(ty),
            "probe_top1_is": ("X" if t1 == tx else "Y" if t1 == ty else "neither"),
            "gap_lower_bound_nonTop1_candidate": bound}


def teacher_forced(args, tok, rows, outputs, raw):
    res = []
    for idx in sorted(outputs):
        ids_p = tok.encode(rendered(tok, rows[idx]), add_special_tokens=False)
        g = outputs[idx].get("reencoded_ids") or []
        for j in range(min(args.positions, len(g))):
            fs = post(args.url, "/flush_cache", {})
            body = {"input_ids": ids_p + g[:j], "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    "return_logprob": True}
            st, txt, _ = post(args.url, "/generate", body)
            if st != 200:
                raise RuntimeError(f"teacher-forced idx {idx} position {j}: HTTP {st}: {txt[:300]}")
            js = json.loads(txt)
            l1, t1 = js["meta_info"]["output_token_logprobs"][0]
            t1 = int(t1); _, bound = gap_bound(l1)
            raw.write(json.dumps({"label": args.label_a, "idx": idx, "j": j, "flush": fs[0], "http_status": st,
                                  "output_token_logprobs": js["meta_info"]["output_token_logprobs"]}) + "\n")
            res.append({"label": args.label_a, "idx": idx, "j": j, "emitted": g[j], "top1": t1, "top1_logprob": l1,
                        "match": g[j] == t1, "gap_lb_if_not_top1": None if g[j] == t1 else bound})
    return res


def summarize(records, threshold=1.0):
    out = {}
    for pair in sorted({r["pair"] for r in records if "pair" in r}):
        rows = [r for r in records if r.get("pair") == pair]
        # A row counts when its request succeeded with a top-1 and, on the decode path, the
        # regenerated prefix matched.
        valid = [r for r in rows if r.get("regenerated_prefix_tail_matches", True)
                 and r.get("http_status") == 200 and r.get("gap_lower_bound_nonTop1_candidate") is not None]
        lb = lambda r: r["gap_lower_bound_nonTop1_candidate"]
        is_ = lambda r: r["probe_top1_is"]
        probs = [r["probe_top1_prob"] for r in valid]
        out[pair] = {"divergences": len(rows), "valid": len(valid),
                     "top1_is_X": sum(is_(r) == "X" for r in valid), "top1_is_Y": sum(is_(r) == "Y" for r in valid),
                     "top1_neither": sum(is_(r) == "neither" for r in valid),
                     "X_certified_gap_gt": sum(1 for r in valid if is_(r) != "X" and lb(r) > threshold),
                     "Y_certified_gap_gt": sum(1 for r in valid if is_(r) != "Y" and lb(r) > threshold),
                     "X_certified_idx": [r["idx"] for r in valid if is_(r) != "X" and lb(r) > threshold],
                     "Y_certified_idx": [r["idx"] for r in valid if is_(r) != "Y" and lb(r) > threshold],
                     "largest_X_bound": max((max(0.0, lb(r)) for r in valid if is_(r) != "X"), default=0.0),
                     "largest_Y_bound": max((max(0.0, lb(r)) for r in valid if is_(r) != "Y"), default=0.0),
                     "top1_prob_median": statistics.median(probs) if probs else None,
                     "top1_prob_le_half": sum(1 for p in probs if p <= 0.5),
                     "excluded_idx": [r["idx"] for r in rows if r not in valid]}
    for label in sorted({r["label"] for r in records if "j" in r}):
        rows = [r for r in records if r.get("label") == label and "j" in r]
        miss = [r for r in rows if not r["match"]]
        out[f"teacher-forced {label}"] = {
            "positions": len(rows), "prompts": len({r["idx"] for r in rows}), "emitted_not_top1": len(miss),
            "emitted_not_top1_certified_gap_gt": sum(1 for r in miss if r["gap_lb_if_not_top1"] > threshold)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", nargs="+", metavar="PROBE_JSON")
    ap.add_argument("--threshold", type=float, default=1.0, help="certified-gap threshold in nats")
    ap.add_argument("--mode", choices=["prefill", "decode-path", "teacher-forced"], default="prefill")
    ap.add_argument("--positions", type=int, default=24, help="teacher-forced: generated tokens scored per output")
    ap.add_argument("--url"); ap.add_argument("--tokenizer")
    ap.add_argument("--prompts"); ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.summarize:
        records = [r for p in args.summarize for r in json.load(open(p))]
        print(json.dumps(summarize(records, args.threshold), indent=1))
        return 0
    need = ["url", "tokenizer", "prompts", "a", "out"] + (["b"] if args.mode != "teacher-forced" else [])
    missing = [n for n in need if getattr(args, n) is None]
    if missing:
        ap.error("missing --" + ", --".join(missing))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = load(args.prompts)
    fx = load(args.a)
    if args.mode == "teacher-forced":
        with open(args.out + ".raw.jsonl", "w") as raw:
            res = teacher_forced(args, tok, rows, fx, raw)
        with open(args.out + ".json", "w") as fh:
            json.dump(res, fh)
        return 0
    fy = load(args.b)
    one = probe_one if args.mode == "prefill" else decode_path_one
    res = []
    with open(args.out + ".raw.jsonl", "w") as raw:
        for idx in sorted(fx):
            if (fx[idx].get("reencoded_ids") or []) == (fy[idx].get("reencoded_ids") or []):
                continue
            rec = one(args, tok, rows[idx], fx[idx], fy[idx], raw)
            res.append(rec)
            print(json.dumps({k: rec.get(k) for k in ("pair", "idx", "k_common_prefix", "probe_top1_is", "probe_top1_prob",
                                                      "gap_lower_bound_nonTop1_candidate", "text_boundary_ok",
                                                      "regenerated_prefix_tail_matches")}), flush=True)
    with open(args.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
