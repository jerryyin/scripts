#!/usr/bin/env python3
"""HTTP fixed-batch decode window: window metrics from the parsed server log (../batch_log.py).

Configuration-agnostic: works for any number of DP ranks (lines without
"#dp-rank" are treated as rank 0) and reads avg_accept_len when present
(speculative configs).  Used by fbdw_client.py, also usable standalone:

  fbdw_analyze.py --point-dir raw/L8192_B4_rep1   # re-analyse a saved point
"""
import argparse, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_log import parse_lines  # noqa: E402


def analyze(events, B, L, t_all_first, t_abort, min_rounds=100, log_interval=10,
            clock_offset=0.0, client_ended_early=()):
    """events: parsed log events of this point (log clock). t_all_first,
    t_abort: client wall times (None allowed while running). clock_offset:
    add to a log timestamp to get client wall time.  Returns metrics dict."""
    to_log = lambda t: None if t is None else t - clock_offset
    ext = [e for e in events if e["kind"] in ("Prefill", "Mix")]
    dec = [e for e in events if e["kind"] == "Decode"]
    fin = [e for e in events if e["kind"] == "Finish"]
    # per-rank split from Prefill/Mix #new-token sums
    split_tokens = {}
    for e in ext:
        split_tokens[e["dp_rank"]] = split_tokens.get(e["dp_rank"], 0) + e.get("new_token", 0) - (
            # in a Mix batch the decode requests each contribute 1 input token
            (e.get("running", 0) - e.get("new_seq", 0)) if e["kind"] == "Mix" else 0)
    split_seqs = {r: v / L for r, v in split_tokens.items()}
    expected = {r: int(round(v)) for r, v in split_seqs.items() if round(v) > 0}
    active = sorted(expected)
    t_last_ext = max((e["ts"] for e in ext), default=None)
    t_open = None
    if t_all_first is not None and t_last_ext is not None:
        t_open = max(to_log(t_all_first), t_last_ext)
    t_close = to_log(t_abort)
    per_rank = {}
    for r in active:
        lines_r = [e for e in dec if e["dp_rank"] == r and t_open is not None and e["ts"] > t_open
                   and (t_close is None or e["ts"] <= t_close)]
        ext_r = [e["ts"] for e in ext if e["dp_rank"] == r]
        q = []  # qualifying intervals
        nonq = 0
        for prev, cur in zip(lines_r, lines_r[1:]):
            ok = (cur["running"] == expected[r] and prev["running"] == expected[r]
                  and cur.get("queue", 0) == 0 and prev.get("queue", 0) == 0
                  and not any(prev["ts"] < t <= cur["ts"] for t in ext_r)
                  and cur["ts"] > prev["ts"])
            if ok:
                dtv = cur["ts"] - prev["ts"]
                q.append({"t0": prev["ts"], "t1": cur["ts"], "dt": dtv,
                          "tokens": cur["gen_tput"] * dtv, "gen_tput": cur["gen_tput"],
                          "avg_seq_len": cur.get("avg_seq_len"),
                          "avg_accept_len": cur.get("avg_accept_len"),
                          "running": cur["running"]})
            else:
                nonq += 1
        subsec = all(e["subsec"] for e in lines_r) if lines_r else None
        d = {"expected_batch": expected[r], "split_from_prefill": split_seqs[r],
             "decode_lines_in_window": len(lines_r), "warmup_lines_discarded": 1 if lines_r else 0,
             "qualifying_intervals": len(q), "qualifying_rounds": len(q) * log_interval,
             "nonqualifying_intervals": nonq,
             "running_req_values": sorted({e["running"] for e in lines_r}),
             "timestamps_subsecond": subsec}
        if q:
            if subsec:
                tok = sum(x["tokens"] for x in q); tim = sum(x["dt"] for x in q)
                rate = tok / tim
                d["rate_method"] = "sum(gen_tput*dt)/sum(dt)"
            else:
                rate = statistics.mean(x["gen_tput"] for x in q)
                tok, tim = None, None
                d["rate_method"] = "mean(gen_tput) (log timestamps lack sub-second resolution)"
            d.update({"tokens": tok, "seconds": tim, "rate_tok_s": rate,
                      "tpot_ms": expected[r] / rate * 1000.0,
                      "window_t0": q[0]["t0"], "window_t1": q[-1]["t1"],
                      "avg_seq_len_min": min(x["avg_seq_len"] for x in q),
                      "avg_seq_len_max": max(x["avg_seq_len"] for x in q),
                      "tokens_per_interval_mean": (tok / len(q)) if tok else None,
                      "tokens_per_interval_ideal_nonspec": expected[r] * log_interval})
            acc = [x["avg_accept_len"] for x in q if x["avg_accept_len"] is not None]
            if acc:
                d["avg_accept_len_mean"] = statistics.mean(acc)
                d["avg_accept_len_min"] = min(acc); d["avg_accept_len_max"] = max(acc)
        per_rank[r] = d
    enough = bool(active) and all(per_rank[r]["qualifying_rounds"] >= min_rounds for r in active)
    res = {"B": B, "L": L, "active_ranks": active,
           "idle_ranks_note": None, "prefill_tokens_total": sum(split_tokens.values()),
           "prefill_tokens_expected": B * L,
           "t_open_log": t_open, "t_last_extend_log": t_last_ext, "t_close_log": t_close,
           "per_rank": {str(r): v for r, v in per_rank.items()}, "enough_rounds": enough,
           "finished_requests_in_log": len(fin)}
    if enough:
        rates = [per_rank[r]["rate_tok_s"] for r in active]
        tpots = [per_rank[r]["tpot_ms"] for r in active]
        res.update({"system_tok_s": sum(rates), "tpot_ms_mean": statistics.mean(tpots),
                    "tpot_ms_max": max(tpots),
                    "avg_seq_len_min": min(per_rank[r]["avg_seq_len_min"] for r in active),
                    "avg_seq_len_max": max(per_rank[r]["avg_seq_len_max"] for r in active),
                    "common_window_log": [max(per_rank[r]["window_t0"] for r in active),
                                          min(per_rank[r]["window_t1"] for r in active)]})
        accs = [per_rank[r].get("avg_accept_len_mean") for r in active]
        if all(a is not None for a in accs):
            res["avg_accept_len_mean"] = statistics.mean(accs)
        # fixed batch: every in-window line of each active rank at expected batch,
        # no extend after open, nobody finished before the abort
        steady = all(per_rank[r]["nonqualifying_intervals"] == 0 and
                     per_rank[r]["running_req_values"] == [per_rank[r]["expected_batch"]] for r in active)
        res["fixed_batch"] = bool(steady and sum(expected.values()) == B and not client_ended_early)
        res["fixed_batch_detail"] = {"steady_running_req": steady,
                                     "sum_expected_batch": sum(expected.values()),
                                     "client_streams_ended_before_abort": list(client_ended_early)}
    else:
        res["fixed_batch"] = False
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-dir", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.point_dir, "meta.json")))
    with open(os.path.join(a.point_dir, "server-log-slice.txt")) as f:
        ev = parse_lines(f.readlines())
    r = analyze(ev, meta["B"], meta["L"], meta["t_all_first"], meta["t_abort"],
                meta["min_rounds"], meta["log_interval"], meta["clock_offset"],
                meta.get("client_ended_early", []))
    json.dump(r, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
