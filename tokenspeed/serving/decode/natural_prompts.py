#!/usr/bin/env python3
"""Natural-prose prompt source for fbdw_client.py (--prompt-source natural).

Reproduces the natural-prose corpus of the tokenspeed-dsv41-ep4-decode campaign's fixed-batch rows.
For new prompts use amd-tracking's real-token bank (prepare_token_corpus.py) and its prompt fitting
(run_evalscope_matrix.build_token_prompts): this file's window fitting is an older variant of it,
kept because changing it would change the recorded prompt bytes.

Corpus (fixed, sorted path order):
  1. git-tracked *.md/*.rst/*.txt of the --git-root checkout, excluding CMakeLists.txt and
     requirements*.txt (not prose);
  2. *.md under each --extra-root directory.
Fenced code blocks (``` ... ```) are removed from Markdown; files with >1% CJK characters
are skipped (English prose only). Files are joined with "\n\n",
tokenized once (no special tokens) with the model tokenizer.

Build:  natural_prompts.py build --out corpus-dir --git-root DIR [--extra-root DIR ...]
        -> corpus-manifest.json + corpus.txt + corpus-ids.npy (record the manifest with your results)
Prompt for request i of a point with seed S: rng = numpy.default_rng(S); offsets =
rng.choice(N - L - 256, size=B, replace=False); window ids[off:off+k], decoded, with k
adjusted (k += L - n, then +-1 offset shifts) until tok.encode(text) has exactly L tokens.
"""
import argparse, hashlib, json, os, re, subprocess, sys
import numpy as np

# Corpus sources, set by build(): a git checkout whose tracked *.md/*.rst/*.txt are used, plus
# extra directories whose *.md files are added (--git-root, --extra-root).
TS = None
EXTRA = []
FENCE = re.compile(r"^```.*?^```[^\n]*\n?", re.S | re.M)


def corpus_files():
    out = subprocess.run(["git", "-C", TS, "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    fs = sorted(os.path.join(TS, f) for f in out if re.search(r"\.(md|rst|txt)$", f, re.I)
                and not re.search(r"(CMakeLists|requirements)", f, re.I))
    for d in EXTRA:
        ex = []
        for root, _, names in os.walk(d):
            ex += [os.path.join(root, n) for n in names if n.lower().endswith(".md")]
        fs += sorted(ex)
    return fs


def build(out, tok_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    os.makedirs(out, exist_ok=True)
    files, parts = [], []
    for f in corpus_files():
        raw = open(f, "rb").read()
        t = raw.decode("utf-8", errors="replace")
        if f.lower().endswith(".md"):
            t = FENCE.sub("", t)
        t = t.strip()
        if not t or len(re.findall(r"[\u4e00-\u9fff]", t)) > 0.01 * len(t):
            continue  # empty, or not English prose (>1% CJK characters)
        files.append({"path": f, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "text_chars_used": len(t)})
        parts.append(t)
    text = "\n\n".join(parts)
    ids = np.array(tok.encode(text, add_special_tokens=False), dtype=np.int32)
    open(os.path.join(out, "corpus.txt"), "w").write(text)
    np.save(os.path.join(out, "corpus-ids.npy"), ids)
    man = {"files": files, "n_files": len(files), "join": "\\n\\n", "code_fences_removed_from_md": True,
           "corpus_txt_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "corpus_ids_npy_sha256": hashlib.sha256(open(os.path.join(out, "corpus-ids.npy"), "rb").read()).hexdigest(),
           "token_count": int(len(ids)), "tokenizer": tok_path,
           "tokenspeed_rev": subprocess.run(["git", "-C", TS, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "note": "target was >=2M tokens; all permitted sources together give token_count (fewer)",
           "prompt_rule": __doc__}
    json.dump(man, open(os.path.join(out, "corpus-manifest.json"), "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("n_files", "token_count", "corpus_txt_sha256")}))


_IDS = {}
def make_prompts_natural(tok, B, L, seed, corpus_dir, nspec=0, adjust=0):
    if corpus_dir not in _IDS:
        _IDS[corpus_dir] = np.load(os.path.join(corpus_dir, "corpus-ids.npy"))
    ids = _IDS[corpus_dir]
    target = L - nspec - adjust
    rng = np.random.default_rng(seed)
    offs = rng.choice(len(ids) - L - 256, size=B, replace=False)
    ps, info = [], []
    for off in offs:
        off = int(off); found = None
        for shift in range(0, 16):
            k = target
            for _ in range(12):
                text = tok.decode(ids[off + shift: off + shift + k].tolist())
                n = len(tok.encode(text))
                if n + adjust == L:
                    found = (text, shift, k); break
                k += (L - adjust) - n
            if found:
                break
        if not found:
            raise RuntimeError(f"cannot build exact {L}-token natural window at offset {off}")
        ps.append(found[0]); info.append([off, found[1], found[2]])
    assert len(set(ps)) == B
    return ps, {"source": "natural", "corpus_dir": corpus_dir, "corpus_tokens": int(len(ids)),
                "offsets_shift_k": info, "special_tokens_added": nspec, "adjust": adjust}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("MODEL"), required="MODEL" not in os.environ, help="model directory; defaults to $MODEL")
    ap.add_argument("--git-root", required=True, help="git checkout whose tracked *.md/*.rst/*.txt form the corpus")
    ap.add_argument("--extra-root", action="append", default=[], help="directory whose *.md files are added (repeatable)")
    a = ap.parse_args()
    TS = a.git_root
    EXTRA = a.extra_root
    build(a.out, a.tokenizer)
