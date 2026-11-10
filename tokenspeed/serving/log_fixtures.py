"""Synthetic TokenSpeed server-log lines and EvalScope databases for this directory's tests."""

import datetime as dt
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
    return f"[ts] {ts(t)}  INFO ts_control::batch_log: {kind} batch. " + ", ".join(parts) + "\n"


def finish(t, rid, prompt, cache, out=500, commit=None, acc=None):
    return (f"[ts] {ts(t)}  INFO ts_control::output: Req: {rid} Finish! RequestStats(status='finished', "
            f"reason='length', prompt_tokens={prompt}, cache_tokens={cache}, output_tokens={out}, prefill_ms=10.0, "
            f"acc_len={acc}, commit_ts={commit if commit is not None else t}, finish_ts={t})\n")


def evalscope_db(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE result(request TEXT, start_time REAL, completed_time REAL, success INTEGER, latency REAL, "
              "first_chunk_latency REAL, prompt_tokens INTEGER, completion_tokens INTEGER, response_messages TEXT)")
    for r in rows:
        c.execute("INSERT INTO result VALUES (?,?,?,?,?,?,?,?,?)", r)
    c.commit()
    c.close()
