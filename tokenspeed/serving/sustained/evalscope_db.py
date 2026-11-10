#!/usr/bin/env python3
"""Read an EvalScope perf database (benchmark_data.db) of a multi-turn agentic run, and the steady
window shared by this directory's analyses.

read_results(db) returns one dict per request (start_time, completed_time, success, latency,
first_chunk_latency, prompt_tokens, completion_tokens, request, response_id, salt, user_turns),
without unpickling the stored responses: the response id is searched in their decoded bytes.
salt is the conversation salt build_workload.py puts in the first user message ("[session S]").

steady_bounds(anchor, "A:B", end) is the window [anchor + A, min(anchor + B, end)]. Each analysis
names its anchor in its output: analyze_sustained.py uses the first request start (client) and the
first batch line (server); prefix_reuse.py uses the first commit.
"""
import base64
import json
import re
import sqlite3

SALT = re.compile(r"\[session ([^\]]+)\]")
RESPONSE_ID = re.compile(
    rb"chatcmpl-[A-Za-z0-9]+-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
COLUMNS = ("start_time", "completed_time", "success", "latency", "first_chunk_latency", "prompt_tokens",
           "completion_tokens", "request", "response_messages")


def salt_of(text):
    """The conversation salt in TEXT (a message or a whole request), or None."""
    if not isinstance(text, str):
        text = json.dumps(text) if not isinstance(text, bytes) else text.decode("utf-8", "ignore")
    m = SALT.search(text)
    return m.group(1) if m else None


def read_results(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(f"select {', '.join(COLUMNS)} from result").fetchall()
    finally:
        con.close()
    out = []
    for row in rows:
        r = dict(zip(COLUMNS, row))
        encoded = r.pop("response_messages")
        m = RESPONSE_ID.search(base64.b64decode(encoded)) if encoded else None
        r["response_id"] = m.group(0).decode() if m else None
        msgs = json.loads(r["request"]).get("messages") or []
        users = [x["content"] for x in msgs if x.get("role") == "user"]
        r["salt"] = salt_of(users[0]) if users else None
        r["user_turns"] = len(users)
        out.append(r)
    return out


def parse_steady(spec):
    """'A:B' -> (A, B) seconds, or None for '' / None."""
    if not spec:
        return None
    try:
        a, b = (float(x) for x in spec.split(":"))
    except ValueError:
        raise ValueError(f"steady window {spec!r}: expected A:B seconds") from None
    if not 0 <= a < b:
        raise ValueError(f"steady window {spec!r}: needs 0 <= A < B")
    return a, b


def steady_bounds(anchor, steady, end=None):
    s0, s1 = anchor + steady[0], anchor + steady[1]
    return s0, s1 if end is None else min(s1, end)
