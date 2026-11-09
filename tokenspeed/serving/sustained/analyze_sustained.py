#!/usr/bin/env python3
"""Analyse a sustained run: client metrics (evalscope sqlite) + server decode-batch distribution.

Usage:
  analyze_sustained.py --run-dir OUT_DIR [--server-log LOG] [--log-interval 10] [--json out.json]
  analyze_sustained.py --server-log LOG [--t0 ISO --t1 ISO]           # server-log only
Window: --t0/--t1 if given, else the MEASURE_BEGIN/END markers written by run_sustained.sh (evalscope 1.11.1
db times are time.perf_counter(), not epoch; used only for client durations), else the whole log.
--steady A:B adds a 'server_steady' pass over [first batch line + A, first batch line + B].
Server lines are parsed with ../batch_log.py: "Prefill|Mix|Decode batch." lines and
"Finish! RequestStats(...)".
"""
import argparse, bisect, datetime as dt, glob, json, math, os, sqlite3, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_log import parse_line  # noqa: E402


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ') if t is not None else None


def parse_iso(s):
    s = s.strip().rstrip('Z')
    if '.' in s:
        a, b = s.split('.'); s = a + '.' + b[:6]
        d = dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        d = dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
    return d.replace(tzinfo=dt.timezone.utc).timestamp()


def wq(vals, ws, q):
    """weighted quantile (lower) of vals with weights ws"""
    pts = sorted(zip(vals, ws)); tot = sum(w for _, w in pts)
    if tot <= 0: return None
    acc = 0.0
    for v, w in pts:
        acc += w
        if acc >= q * tot: return v
    return pts[-1][0]


def wstats(vals, ws):
    tot = sum(ws)
    if not vals or tot <= 0: return None
    return dict(weight_total=round(tot, 3), mean=round(sum(v * w for v, w in zip(vals, ws)) / tot, 3),
                p10=wq(vals, ws, .10), median=wq(vals, ws, .50), p90=wq(vals, ws, .90),
                min=min(vals), max=max(vals))


def pct(xs):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not xs: return None
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return dict(n=len(xs), mean=round(sum(xs) / len(xs), 5), p50=q(.5), p90=q(.9), p99=q(.99), min=xs[0], max=xs[-1])


def hist(vals, ws, binw=1.0):
    h = defaultdict(float)
    for v, w in zip(vals, ws):
        h[math.floor(v / binw) * binw] += w
    tot = sum(h.values()) or 1
    return {('%g' % k): round(v / tot, 5) for k, v in sorted(h.items())}


def client_metrics(run_dir):
    dbs = sorted(glob.glob(os.path.join(run_dir, 'collected', '*.db'))) or \
        sorted(glob.glob(os.path.join(run_dir, 'evalscope', '**', '*.db'), recursive=True))
    if not dbs: return None, None
    con = sqlite3.connect(dbs[0])
    rows = con.execute('select start_time, completed_time, success, latency, first_chunk_latency, prompt_tokens, '
                       'completion_tokens from result').fetchall()
    ok = [r for r in rows if r[2]]
    t0 = min(r[0] for r in rows); t1 = max(r[1] for r in rows if r[1])
    win = t1 - t0
    ttft = [r[4] for r in ok if r[4] is not None]
    tpot = [(r[3] - r[4]) / (r[6] - 1) for r in ok if r[3] is not None and r[4] is not None and r[6] and r[6] > 1]
    out_tok = sum(r[6] or 0 for r in ok); in_tok = sum(r[5] or 0 for r in ok)
    res = dict(db=dbs[0], requests_total=len(rows), requests_ok=len(ok), requests_failed=len(rows) - len(ok),
               window_start=iso(t0), window_end=iso(t1), window_s=round(win, 3),
               output_tokens=out_tok, input_tokens_reported=in_tok,
               output_tok_per_s=round(out_tok / win, 2) if win > 0 else None,
               requests_per_s=round(len(ok) / win, 4) if win > 0 else None,
               ttft_s=pct(ttft), tpot_per_request_s=pct(tpot),
               completion_tokens=pct([r[6] for r in ok]), prompt_tokens=pct([r[5] for r in ok]))
    return res, (t0, t1)


def server_metrics(log_path, t0, t1, log_interval, grid_dt=0.5, stale_s=5.0, gap_cap=30.0):
    evs = []
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            if 'batch.' not in line and 'RequestStats' not in line: continue
            e = parse_line(line)
            if e is None or e['ts'] is None: continue
            if t0 is not None and e['ts'] < t0: continue
            if t1 is not None and e['ts'] > t1: continue
            evs.append(e)
    if not evs: return dict(error='no batch/RequestStats lines in window', log=log_path)
    evs.sort(key=lambda e: e['ts'])
    w0 = t0 if t0 is not None else evs[0]['ts']; w1 = t1 if t1 is not None else evs[-1]['ts']
    out = dict(log=log_path, window_start=iso(w0), window_end=iso(w1), window_s=round(w1 - w0, 3),
               log_interval_rounds=log_interval, line_counts=dict(Counter(e['kind'] for e in evs)))
    # --- RequestStats ---
    fin = [e for e in evs if e['kind'] == 'Finish']
    if fin:
        st = Counter(str(e['stats'].get('status')).strip("'") for e in fin)
        g = lambda k: sum((e['stats'].get(k) or 0) for e in fin if isinstance(e['stats'].get(k), (int, float)))
        pt, ct, ot = g('prompt_tokens'), g('cache_tokens'), g('output_tokens')
        out['request_stats'] = dict(n=len(fin), status=dict(st), prompt_tokens=pt, cache_tokens=ct,
                                    uncached_prompt_tokens=pt - ct, cache_hit_ratio=round(ct / pt, 4) if pt else None,
                                    output_tokens=ot, output_tok_per_s=round(ot / (w1 - w0), 2) if w1 > w0 else None,
                                    finished_per_s=round(len(fin) / (w1 - w0), 4) if w1 > w0 else None,
                                    ttft_ms=pct([e['stats'].get('ttft_ms') for e in fin if isinstance(e['stats'].get('ttft_ms'), (int, float)) and e['stats'].get('ttft_ms') > 0]),
                                    queue_ms=pct([e['stats'].get('queue_ms') for e in fin if isinstance(e['stats'].get('queue_ms'), (int, float))]),
                                    decode_tps=pct([e['stats'].get('decode_tps') for e in fin if isinstance(e['stats'].get('decode_tps'), (int, float))]),
                                    acc_len=pct([e['stats'].get('acc_len') for e in fin if isinstance(e['stats'].get('acc_len'), (int, float))]))
    # --- batch lines per rank ---
    batch = [e for e in evs if e['kind'] in ('Prefill', 'Mix', 'Decode')]
    ranks = sorted({e['dp_rank'] for e in batch})
    out['dp_ranks'] = ranks
    pf = [e for e in batch if e['kind'] in ('Prefill', 'Mix')]
    out['prefill_mix_lines'] = dict(n=len(pf), new_tokens=sum(e.get('new_token', 0) for e in pf),
                                    cached_tokens=sum(e.get('cached_token', 0) for e in pf),
                                    new_seq=sum(e.get('new_seq', 0) for e in pf))
    per_rank = {}; series = {}
    allv, allw_r, allw_t = [], [], []
    for r in ranks:
        re_ = [e for e in batch if e['dp_rank'] == r]
        dec = [e for e in re_ if e['kind'] == 'Decode' and 'running' in e]
        # time attribution: gap since previous batch event of this rank, attributed to this event's kind
        tk = defaultdict(float); vals, wr, wt = [], [], []
        prev = None
        for e in re_:
            gap = min(e['ts'] - prev, gap_cap) if prev is not None else 0.0
            prev = e['ts']; tk[e['kind']] += gap
            if e['kind'] == 'Decode' and 'running' in e:
                vals.append(e['running']); wr.append(log_interval); wt.append(gap)
        tt = sum(tk.values())
        acc = [(e['avg_accept_len'], log_interval) for e in dec if 'avg_accept_len' in e]
        qv = [e['queue'] for e in re_ if 'queue' in e]
        per_rank[r] = dict(decode_lines=len(dec), decode_rounds=len(dec) * log_interval,
                           prefill_lines=sum(1 for e in re_ if e['kind'] == 'Prefill'),
                           mix_lines=sum(1 for e in re_ if e['kind'] == 'Mix'),
                           time_share_by_kind={k: round(v / tt, 4) for k, v in tk.items()} if tt else None,
                           busy_logged_s=round(tt, 2),
                           running_by_rounds=wstats(vals, wr), running_by_walltime=wstats(vals, wt),
                           running_hist_by_rounds=hist(vals, wr), running_hist_by_walltime=hist(vals, wt),
                           queue=pct(qv),
                           avg_accept_len_by_rounds=(wstats([a for a, _ in acc], [w for _, w in acc]) if acc else None),
                           avg_accept_len_hist=(hist([a for a, _ in acc], [w for _, w in acc], 0.25) if acc else None))
        allv += vals; allw_r += wr; allw_t += wt
        series[r] = ([e['ts'] for e in dec], [e['running'] for e in dec])
    out['per_rank'] = per_rank
    out['all_rank_lines_pooled'] = dict(running_by_rounds=wstats(allv, allw_r), running_by_walltime=wstats(allv, allw_t))
    # --- total over ranks at matching times (step function on a grid) ---
    tv = []
    t = w0
    while t <= w1:
        s = 0
        for r in ranks:
            ts_, rv = series[r]
            i = bisect.bisect_right(ts_, t) - 1
            if i >= 0 and t - ts_[i] <= stale_s: s += rv[i]
        tv.append(s); t += grid_dt
    nz = [v for v in tv if v > 0]
    out['total_running_over_ranks'] = dict(grid_dt_s=grid_dt, stale_s=stale_s,
                                          by_walltime_incl_idle=wstats(tv, [1] * len(tv)),
                                          by_walltime_active_only=wstats(nz, [1] * len(nz)),
                                          hist_by_walltime=hist(tv, [1] * len(tv), 4.0 if max(tv or [0]) > 64 else 1.0),
                                          idle_fraction=round(1 - len(nz) / len(tv), 4) if tv else None)
    qall = [e['queue'] for e in batch if 'queue' in e]
    out['queue_all_lines'] = pct(qall)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir'); ap.add_argument('--server-log'); ap.add_argument('--t0'); ap.add_argument('--t1')
    ap.add_argument('--log-interval', type=int, default=10); ap.add_argument('--json')
    ap.add_argument('--grid-dt', type=float, default=0.5); ap.add_argument('--stale-s', type=float, default=5.0)
    ap.add_argument('--steady', default='120:900', help='A:B seconds after window start for an extra steady-state '
                    'server pass (excludes the cold-start ramp and the soft-exit drain); "" disables')
    a = ap.parse_args()
    res = dict(tool='analyze_sustained.py')
    win = None
    if a.run_dir:
        cm, win = client_metrics(a.run_dir); res['client'] = cm
        if win is not None and win[0] < 1e9:
            # evalscope 1.11.1 stores start/completed times from time.perf_counter() (monotonic, not epoch):
            # use them for client durations only; the server window comes from the run_sustained markers.
            cm['window_clock'] = 'perf_counter (monotonic); window_start/end are not wall-clock'
            win = None
        if win is None and os.path.exists(os.path.join(a.run_dir, 'markers.txt')):
            mk = {l.split()[1]: parse_iso(l.split()[0]) for l in open(os.path.join(a.run_dir, 'markers.txt')) if len(l.split()) > 1}
            if 'MEASURE_BEGIN' in mk and 'MEASURE_END' in mk: win = (mk['MEASURE_BEGIN'], mk['MEASURE_END'])
    if a.t0 or a.t1:
        win = (parse_iso(a.t0) if a.t0 else None, parse_iso(a.t1) if a.t1 else None)
    log = a.server_log or (os.path.join(a.run_dir, 'server-slice.log') if a.run_dir else None)
    if log and os.path.exists(log):
        res['server'] = server_metrics(log, win[0] if win else None, win[1] if win else None, a.log_interval, a.grid_dt, a.stale_s)
        if a.steady and win and win[0] is not None:
            sa, sb = (float(x) for x in a.steady.split(':'))
            base = win[0]
            try:  # steady window is relative to the first server batch line in the window (first request), not the marker
                with open(log, errors='replace') as f:
                    for line in f:
                        if 'batch.' in line:
                            e = parse_line(line)
                            if e and e['ts'] and e['ts'] >= win[0]: base = e['ts']; break
            except OSError: pass
            s0 = base + sa; s1 = base + sb if win[1] is None else min(base + sb, win[1])
            if s1 > s0:
                res['server_steady'] = server_metrics(log, s0, s1, a.log_interval, a.grid_dt, a.stale_s)
                res['server_steady']['steady_spec_s'] = a.steady
    txt = json.dumps(res, indent=1, default=str)
    if a.json: open(a.json, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
