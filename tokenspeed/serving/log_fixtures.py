"""Synthetic TokenSpeed server-log lines and EvalScope databases for this directory's tests."""

import datetime as dt
import json
import sqlite3
from pathlib import Path


def ts(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def batch(t, kind, rank, **f):
    fields = {"Prefill": ["new-seq", "new-token", "cached-token", "running-req", "queue-req"],
              "Decode": ["running-req", "gen", "queue-req"]}[kind]
    parts = [f"#dp-rank: {rank}"]
    for name in fields:
        if name == "gen":
            parts.append(f"gen throughput (token/s): {f.get('gen', 100.0)}")
        else:
            parts.append(f"#{name}: {f.get(name.replace('-', '_'), 0)}")
    if "acc" in f:
        parts.append(f"avg_accept_len: {f['acc']}")
    if "seq" in f:
        parts.append(f"avg_seq_len: {f['seq']}")
    return f"[ts] {ts(t)}  INFO ts_control::batch_log: {kind} batch. " + ", ".join(parts) + "\n"


def finish(t, rid, prompt, cache, out=500, commit=None, acc=None):
    return (f"[ts] {ts(t)}  INFO ts_control::output: Req: {rid} Finish! RequestStats(status='finished', "
            f"reason='length', prompt_tokens={prompt}, cache_tokens={cache}, output_tokens={out}, prefill_ms=10.0, "
            f"acc_len={acc}, commit_ts={commit if commit is not None else t}, finish_ts={t})\n")


def finish_acc(t, rid, acc):
    """The Finish line of builds that log only the request's mean accepted tokens per verify step (1be88db0)."""
    return f"[ts] {ts(t)}  INFO ts_control::output: Req: {rid} Finish! Accept_num_tokens_avg: {acc}\n"


# (prefill chunks, decode rounds, client gap after the turn's Finish s, context at the turn's start) per turn:
# long turns log Decode lines, short turns (< interval rounds) none, so their Prefill line is followed by the
# Finish and the next turn's Prefill; multi-chunk turns give the chunk time; a turn's last chunk holds its
# remaining new tokens (1000 + 97 x turn index here, a different size per turn as in a real log)
AGENTIC_C1_TURNS = ((2, 410, 3.0, 300000), (1, 20, 8.0, 310000), (1, 130, 10.0, 320000), (3, 35, 2.5, 330000),
                    (1, 810, 6.0, 600000), (2, 10, 4.0, 610000), (1, 250, 0.0, 620000))


def agentic_c1(T, turns=AGENTIC_C1_TURNS, interval=40, step=0.01066, chunk=0.125, acc=3.5, chunk_tokens=8192):
    """One rank serving an open-loop multi-turn client at concurrency 1 (InferenceX AgentX at C=1, as on TP4):
    each turn is its prefill chunks (`chunk` s each), its decode rounds (`step` s each; a Decode line per
    `interval` rounds, logged at the end of the interval) and a Finish line, then the client's gap.
    Returns (log lines, AIPerf profile_export.jsonl records, truth) where truth holds the window (first
    request start, last request end) and the prefill, decode and idle seconds of the window."""
    lines, recs, t = [], [], T
    tp = td = 0.0
    for k, (nch, rounds, gap, seq) in enumerate(turns):
        start = t
        for c in range(nch):
            last = c == nch - 1
            lines.append(batch(t, "Prefill", 0, new_seq=1 if last else 0, new_token=1000 + 97 * k if last else chunk_tokens,
                               cached_token=seq, running_req=1))
            t += chunk; tp += chunk
        for j in range(1, rounds // interval + 1):
            lines.append(batch(t + j * interval * step, "Decode", 0, running_req=1, gen=round(acc / step, 1),
                               acc=acc, seq=int(seq + j * interval * acc)))
        t += rounds * step; td += rounds * step
        lines.append(finish_acc(t, f"turn{k}", acc))
        recs.append({"metadata": {"benchmark_phase": "profiling", "request_start_ns": int(round((start - 0.002) * 1e9)),
                                  "request_end_ns": int(round((t + 0.001) * 1e9))}})
        t += gap
    window = (recs[0]["metadata"]["request_start_ns"] / 1e9, recs[-1]["metadata"]["request_end_ns"] / 1e9)
    span = window[1] - window[0]
    return lines, recs, {"window": window, "prefill_s": tp, "decode_s": td, "idle_s": span - tp - td,
                         "decode_step_s": step, "turns": len(turns)}


def write_agentic_c1(d, T=1_800_000_000.0):
    """Write agentic_c1()'s server.log, profile_export.jsonl and truth.json into directory d."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    lines, recs, truth = agentic_c1(T)
    (d / "server.log").write_text("".join(lines))
    (d / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    return truth


def evalscope_db(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE result(request TEXT, start_time REAL, completed_time REAL, success INTEGER, latency REAL, "
              "first_chunk_latency REAL, prompt_tokens INTEGER, completion_tokens INTEGER, response_messages TEXT)")
    for r in rows:
        c.execute("INSERT INTO result VALUES (?,?,?,?,?,?,?,?,?)", r)
    c.commit()
    c.close()
