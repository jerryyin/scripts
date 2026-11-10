#!/usr/bin/env python3
"""Build a deterministic greedy prompt set (prompts.jsonl) for greedy_client.py.

  greedy_prompts.py --corpus TEXT --tokenizer MODEL_DIR --model NAME --out prompts.jsonl
                    [--seed 0] [--count 24] [--min-tokens 200] [--max-tokens 4000] [--max-new 256]

--count /v1/completions prompts are corpus slices with target lengths spread evenly from
--min-tokens to --max-tokens, each starting at a line start and cut back to its last sentence end
(after 60% of the slice); then 8 short /v1/chat/completions questions, the last one asking for a
summary of a fixed corpus paragraph. Every body is greedy (temperature 0, top_p 1, seed 0) with
max_tokens --max-new and no ignore_eos. The source is a UTF-8 text file of at least
MIN_CORPUS_CHARS characters (for a token bank from amd-tracking's prepare_token_corpus.py, decode
it to text first).

The same corpus, tokenizer, model name and seed give byte-identical output; the SHA-256 of the file
is printed. One JSON object per line: idx, endpoint, prompt_tokens_est (target_tokens and
corpus_char_start for completions), body.
"""
import argparse
import hashlib
import json
import random
import re
import sys

from transformers import AutoTokenizer

CHAT_QUESTIONS = [
    "Explain in a few sentences what a KV cache is in transformer inference and why it speeds up decoding.",
    "Write a small Python function `is_palindrome(s)` that ignores case and non-alphanumeric characters, with a short docstring.",
    "What is the difference between data parallelism and expert parallelism when serving a mixture-of-experts model? Answer briefly.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, and state its time complexity.",
    "Explain speculative decoding to a software engineer in one short paragraph.",
    "List three common causes of non-deterministic outputs in GPU inference servers and one mitigation for each.",
    "Write a bash one-liner that counts the number of lines in all .py files under the current directory.",
]
SENT_END = re.compile(r"[.!?](?=\s)")
# A completion slice starts at a line start at least SLICE_SPAN characters before the corpus end,
# reads CHARS_PER_TOKEN characters per target token, and is cut back to its last sentence end
# after KEEP_FRACTION of its text; a start without such an end is redrawn, up to MAX_DRAWS times.
SLICE_SPAN = 40000
CHARS_PER_TOKEN = 8
KEEP_FRACTION = 0.6
MAX_DRAWS = 50
# The summary question quotes the first paragraph starting after this character offset.
SUMMARY_OFFSET = 100000
MIN_CORPUS_CHARS = SUMMARY_OFFSET + SLICE_SPAN


def build(text, tokenizer, model, seed=0, count=24, min_tokens=200, max_tokens=4000, max_new=256):
    common = {"temperature": 0, "top_p": 1, "max_tokens": max_new, "seed": 0, "stream": False}
    rng = random.Random(seed)
    targets = [round(min_tokens + i * (max_tokens - min_tokens) / (count - 1)) for i in range(count)]
    if len(text) < MIN_CORPUS_CHARS:
        raise ValueError(f"corpus has {len(text)} characters; needs at least {MIN_CORPUS_CHARS}")
    rows = []
    for i, target in enumerate(targets):
        for _ in range(MAX_DRAWS):
            start = text.find("\n", rng.randrange(0, len(text) - SLICE_SPAN)) + 1
            ids = tokenizer.encode(text[start:start + target * CHARS_PER_TOKEN],
                                   add_special_tokens=False)[:target]
            piece = tokenizer.decode(ids)
            ends = [m.end() for m in SENT_END.finditer(piece) if m.end() >= KEEP_FRACTION * len(piece)]
            if not ends:
                continue
            piece = piece[:ends[-1]]
            if piece.strip():
                break
        else:
            raise RuntimeError(f"prompt {i}: no sentence-ended slice of {target} tokens in {MAX_DRAWS} draws")
        rows.append({"idx": i, "endpoint": "/v1/completions", "target_tokens": target,
                     "prompt_tokens_est": len(tokenizer.encode(piece, add_special_tokens=False)),
                     "corpus_char_start": start, "body": dict(model=model, prompt=piece, **common)})
    start = text.find("\n\n", SUMMARY_OFFSET) + 2
    para = text[start:start + 1500]
    ends = [m.end() for m in SENT_END.finditer(para)]
    para = para[:ends[-1]] if ends else para
    for j, q in enumerate(CHAT_QUESTIONS + ["Summarise the following text in three sentences:\n\n" + para]):
        rows.append({"idx": count + j, "endpoint": "/v1/chat/completions",
                     "prompt_tokens_est": len(tokenizer.encode(q, add_special_tokens=False)),
                     "body": dict(model=model, messages=[{"role": "user", "content": q}], **common)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="UTF-8 text file")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model", required=True, help="model name put in every request body")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--min-tokens", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--max-new", type=int, default=256)
    a = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    with open(a.corpus, encoding="utf-8") as fh:
        text = fh.read()
    rows = build(text, tok, a.model, a.seed, a.count, a.min_tokens, a.max_tokens, a.max_new)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"{len(rows)} prompts -> {a.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
