#!/usr/bin/env python3
"""List Claude Code sessions on this machine, including headless (-p) ones that
the `claude --resume` picker does not show.

Usage:  python3 list_claude_sessions.py [substring-filter]
Then:   cd <the cwd column>  &&  claude --resume <session id>
"""
import glob
import json
import os
import sys
import time

FILTER = sys.argv[1].lower() if len(sys.argv) > 1 else ""


def first_prompt(path):
    """First human-authored text in the session, however it was recorded."""
    try:
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                content = record.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip().replace("\n", " ")
                message = record.get("message")
                if isinstance(message, dict):
                    inner = message.get("content")
                    if isinstance(inner, str) and inner.strip():
                        return inner.strip().replace("\n", " ")
                    if isinstance(inner, list):
                        for part in inner:
                            if isinstance(part, dict) and part.get("type") == "text" and part.get("text", "").strip():
                                return part["text"].strip().replace("\n", " ")
    except OSError as exc:
        return f"<unreadable: {exc}>"
    return "<no text found>"


def session_cwd(path):
    """Real working directory, recorded in the session; the project-dir name is
    ambiguous because dashes in a path are escaped as dashes."""
    try:
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record.get("cwd"), str):
                    return record["cwd"]
    except OSError:
        pass
    return "<unknown>"


def session_kind(path):
    """Interactive sessions open with a mode marker; -p runs enqueue a prompt."""
    try:
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("type") == "mode":
                    return "interactive"
                if record.get("type") == "queue-operation":
                    return "headless"
                return "?"
    except OSError:
        pass
    return "?"


rows = []
for path in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")):
    # Skip per-subagent transcripts; they are not separately resumable.
    if os.sep + "subagents" + os.sep in path:
        continue
    cwd = session_cwd(path)
    prompt = first_prompt(path)
    if FILTER and FILTER not in prompt.lower() and FILTER not in cwd.lower():
        continue
    rows.append((
        os.path.getmtime(path),
        os.path.basename(path)[:-len(".jsonl")],
        session_kind(path),
        cwd,
        os.path.getsize(path) // 1024,
        prompt[:62],
    ))

rows.sort(reverse=True)
print(f"{'when':<12}  {'session id':<36}  {'kind':<11}  {'cwd':<16}  {'KB':>6}  first prompt")
for mtime, sid, kind, cwd, kb, prompt in rows:
    when = time.strftime("%m-%d %H:%M", time.localtime(mtime))
    print(f"{when:<12}  {sid:<36}  {kind:<11}  {cwd:<16}  {kb:>6}  {prompt}")
if not rows:
    print("(no sessions matched)")
