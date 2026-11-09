#!/usr/bin/env python3
"""Build salted, 65536-fitting multi-turn workloads for evalscope `--dataset swe_smith`
from TokenSpeed CI's agentic_dataset.json (lightseekorg/agentic-dataset @ 2779acd4...).

Deterministic (seed). Output format == agentic_dataset.json (metadata + conversations,
each conversation = list of turns {messages: [delta user msgs], prompt_tokens}), so it is
loaded unchanged by evalscope 1.11.1 swe_smith._load_from_json.

Per instance i of segment S:
  * source conversation = perm[i % 71] (seeded permutation, re-permuted every 71 instances)
  * first user message prefixed with a unique salt "[session S-iiiii-xxxxxxxx]\n" so no two
    instances (in any segment/file) share a token prefix beyond the chat-template header.
  * turn 1 truncated (whole trailing messages dropped, last kept message token-truncated, like
    evalscope's _collect_until) to ~first_turn_ds_tokens DeepSeek tokens; turns 2..n unchanged.
  * turns = min(src_turns, T_fit) where T_fit is the largest turn count such that
    prompt(T) + max_tokens <= max_model_len - margin with the DeepSeek V4.1 tokenizer +
    chat template, budgeting a FULL max_tokens-token assistant reply per previous turn
    (verified exactly with apply_chat_template).
"""
import argparse, hashlib, json, math, os, random, sys, time
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True)
ap.add_argument('--model', default=os.environ.get('MODEL'), required='MODEL' not in os.environ, help='model directory; defaults to $MODEL')
ap.add_argument('--out-dir', required=True)
ap.add_argument('--seed', type=int, default=20261107)
ap.add_argument('--max-model-len', type=int, default=65536)
ap.add_argument('--max-tokens', type=int, default=500)
ap.add_argument('--margin', type=int, default=1000)
ap.add_argument('--first-turn-ds-tokens', type=int, default=42000,
                help='truncate turn 1 (salt+template incl.) to about this many DeepSeek tokens')
ap.add_argument('--segments', default='c32:1200,c64:1700', help='name:count[,name:count] measured instances per file')
ap.add_argument('--warmup', type=int, default=2, help='extra warmup instances appended at the end of each file')
ap.add_argument('--stats-only', action='store_true')
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
src = json.load(open(a.source, encoding='utf-8'))
convs = src['conversations']
budget = a.max_model_len - a.margin - a.max_tokens   # max allowed prompt tokens

def tmpl_len(msgs):
    r = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(r, 'keys'): r = r['input_ids']
    return len(r)

def bare(text):
    return len(tok.encode(text, add_special_tokens=False))

# assistant placeholder of exactly max_tokens tokens (worst case: model reply fills max_tokens)
_ids = tok.encode(convs[0][1]['messages'][0]['content'] * 4, add_special_tokens=False)[:a.max_tokens]
PLACE = tok.decode(_ids)
assert abs(bare(PLACE) - a.max_tokens) <= 2, bare(PLACE)
SALT_PROBE = '[session c64-01699-ffffffff]\n'

def history(conv, T, salt=''):
    msgs = []
    for t in range(T):
        delta = [dict(m) for m in conv[t]['messages']]
        if t == 0 and salt:
            delta[0]['content'] = salt + delta[0]['content']
        msgs.extend(delta)
        if t < T - 1:
            msgs.append({'role': 'assistant', 'content': PLACE})
    return msgs

def truncate_first(conv):
    msgs = conv[0]['messages']
    ovh = tmpl_len([{'role': m['role'], 'content': SALT_PROBE + m['content'] if i == 0 else m['content']} for i, m in enumerate(msgs)]) \
        - sum(bare(m['content']) for m in msgs)
    target = a.first_turn_ds_tokens - ovh
    out, acc = [], 0
    for m in msgs:
        ids = tok.encode(m['content'], add_special_tokens=False)
        if acc + len(ids) <= target:
            out.append(dict(m)); acc += len(ids)
        else:
            rem = target - acc
            if rem > 0:
                out.append({'role': m['role'], 'content': tok.decode(ids[:rem])})
            break
    new = [dict(conv[0])] + [dict(t) for t in conv[1:]]
    new[0]['messages'] = out
    return new, len(msgs), len(out)

t0 = time.time()
orig_first = [tmpl_len(history(c, 1, SALT_PROBE)) for c in convs]
trunc_info = []
for ci in range(len(convs)):
    convs[ci], n_before, n_after = truncate_first(convs[ci])
    trunc_info.append((n_before, n_after))
stats = []
for ci, conv in enumerate(convs):
    n = len(conv)
    first_ds = tmpl_len(history(conv, 1, SALT_PROBE))
    incr_bare = [sum(bare(m['content']) for m in conv[t]['messages']) for t in range(1, n)]
    full_ds = tmpl_len(history(conv, n, SALT_PROBE))
    # per-turn template overhead (assistant wrapper + user wrapper), averaged
    ovh = (full_ds - first_ds - sum(incr_bare) - (n - 1) * a.max_tokens) / max(1, n - 1)
    est = [first_ds]
    for t in range(1, n):
        est.append(est[-1] + incr_bare[t - 1] + a.max_tokens + ovh)
    tfit = max([T for T in range(1, n + 1) if est[T - 1] + 3 <= budget] or [0])
    # exact verification at tfit (shrink until exact count fits)
    while tfit >= 1:
        exact = tmpl_len(history(conv, tfit, SALT_PROBE))
        if exact <= budget: break
        tfit -= 1
    stats.append(dict(idx=ci, src_turns=n, ds_first_untruncated=orig_first[ci], first_turn_msgs_before=trunc_info[ci][0], first_turn_msgs_after=trunc_info[ci][1], kimi_first=conv[0]['prompt_tokens'], kimi_last=conv[-1]['prompt_tokens'],
                      ds_first=first_ds, ds_incr_bare_mean=sum(incr_bare) / max(1, len(incr_bare)),
                      ds_incr_bare_min=min(incr_bare) if incr_bare else 0, ds_incr_bare_max=max(incr_bare) if incr_bare else 0,
                      tmpl_overhead_per_turn=ovh,
                      ds_full_ctx_all_turns=full_ds, ds_max_req_all_turns=full_ds + a.max_tokens,
                      fits_all_turns=(full_ds + a.max_tokens <= a.max_model_len - a.margin),
                      t_fit=tfit, ds_ctx_at_tfit=exact if tfit else None, ds_max_req_at_tfit=(exact + a.max_tokens) if tfit else None))
print(f'tokenized 71 conversations in {time.time()-t0:.1f}s', file=sys.stderr)
os.makedirs(a.out_dir, exist_ok=True)
json.dump(dict(budget_prompt=budget, max_model_len=a.max_model_len, max_tokens=a.max_tokens, margin=a.margin,
               placeholder_tokens=bare(PLACE), conversations=stats), open(os.path.join(a.out_dir, 'source-stats.json'), 'w'), indent=1)
if a.stats_only: sys.exit(0)

usable = [s for s in stats if s['t_fit'] >= 2]
print(f'usable sources: {len(usable)}/71', file=sys.stderr)
rng = random.Random(a.seed)
summary = {}
for seg in a.segments.split(','):
    name, cnt = seg.split(':'); cnt = int(cnt)
    out_convs, meta_inst = [], []
    order = []
    total = cnt + a.warmup
    while len(order) < total:
        p = [s['idx'] for s in usable]; rng.shuffle(p); order.extend(p)
    for i in range(total):
        s = stats[order[i]]; conv = convs[order[i]]
        T = min(s['src_turns'], s['t_fit'])
        salt = '[session %s-%05d-%08x]\n' % (name, i, rng.getrandbits(32))
        turns = []
        for t in range(T):
            msgs = [dict(m) for m in conv[t]['messages']]
            if t == 0: msgs[0]['content'] = salt + msgs[0]['content']
            turns.append({'messages': msgs, 'prompt_tokens': conv[t]['prompt_tokens']})
        out_convs.append(turns)
        meta_inst.append(dict(i=i, src=order[i], turns=T, salt=salt.strip(), warmup=i >= cnt))
    meta = dict(src['metadata'])
    meta.update(dict(derived_from='lightseekorg/agentic-dataset@2779acd48a05c211f2f82eee71bdbdf688d984ca agentic_dataset.json',
                     source_sha256=hashlib.sha256(open(a.source, 'rb').read()).hexdigest(),
                     builder='build_workload.py', seed=a.seed, segment=name, measured_instances=cnt, warmup_instances=a.warmup,
                     warmup_offset=cnt, first_turn_ds_tokens=a.first_turn_ds_tokens, min_turns=min(m['turns'] for m in meta_inst), max_turns=max(m['turns'] for m in meta_inst),
                     num_conversations=total, deepseek_prompt_budget=budget,
                     note='prompt_tokens fields are the ORIGINAL Kimi-K2.5 counts (unused by evalscope)'))
    path = os.path.join(a.out_dir, f'workload-{name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'metadata': meta, 'conversations': out_convs}, f, ensure_ascii=False)
    json.dump(meta_inst, open(os.path.join(a.out_dir, f'workload-{name}.instances.json'), 'w'), indent=0)
    turns = [m['turns'] for m in meta_inst[:cnt]]
    maxreq = max(stats[m['src']]['ds_max_req_at_tfit'] for m in meta_inst)
    summary[name] = dict(path=path, instances=total, measured=cnt, warmup=a.warmup, turns_total=sum(turns),
                         turns_mean=sum(turns) / cnt, turns_min=min(turns), turns_max=max(turns),
                         max_request_tokens_upper_bound=maxreq, size_bytes=os.stat(path).st_size,
                         sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest())
    print(name, json.dumps(summary[name]), file=sys.stderr)
json.dump(summary, open(os.path.join(a.out_dir, 'workload-summary.json'), 'w'), indent=1)
