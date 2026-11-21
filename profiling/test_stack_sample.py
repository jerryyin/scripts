#!/usr/bin/env python3
"""CPU tests for stack_sample.py (pytest). The live test attaches py-spy to a Python child it starts."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stack_sample  # noqa: E402

PY_SPY = stack_sample.PY_SPY


def _rec(t, pid, frames, error=None, tid=1, name="MainThread"):
    return {"t": t, "utc": stack_sample.utc(t), "pid": pid, "comm": "python3", "error": error,
            "threads": [] if error else [{"os_thread_id": tid, "name": name, "frames": frames}]}


def test_summary_groups_identical_top_frames_into_runs() -> None:
    # frames below the top --frames do not split a run
    recs = [_rec(100.0 + i, 7, ["a (x.py:1)", "b (x.py:2)", f"c{i} (x.py:3)"]) for i in range(2)]
    recs += [_rec(102.0 + i, 7, ["wait (dist.py:9)", "step (run.py:5)"]) for i in range(30)]  # a 30 s stall
    recs += [_rec(132.0, 7, [], error="py-spy could not attach")]
    runs = stack_sample.summarize(recs, frames=2, interval=1.0)
    assert [r["samples"] for r in runs] == [2, 30, 1]
    stall = runs[1]
    assert stall["frames"] == ["wait (dist.py:9)", "step (run.py:5)"] and stall["duration_s"] == 30.0
    assert stall["start_utc"] == stack_sample.utc(102.0) and stall["end_utc"] == stack_sample.utc(131.0)
    assert runs[2]["frames"] == ["<dump failed>"]
    assert stack_sample.guess_interval(recs) == 1.0


def test_find_targets_by_process_group_and_comm() -> None:
    me = os.getpid()
    pgid = os.getpgid(0)
    found = stack_sample.find_targets([pgid], [], None)
    assert me in found and found[me]["pgid"] == pgid
    assert me not in stack_sample.find_targets([pgid], [], r"^no-such-comm$")
    assert me not in stack_sample.find_targets([pgid], [], None, exclude={me})


def test_native_with_nonblocking_is_refused() -> None:
    with pytest.raises(SystemExit):
        stack_sample.main(["run", "--pid", "1", "--native", "--nonblocking", "--duration", "1", "--out", "/dev/null"])


def test_attach_failure_is_reported(tmp_path) -> None:
    """A py-spy that cannot attach: every record carries the error, the run exits 2 and names ptrace."""
    fake = tmp_path / "py-spy"
    fake.write_text("#!/bin/sh\necho 'Error: Failed to suspend process: Operation not permitted (os error 1)' >&2\nexit 1\n")
    fake.chmod(0o755)
    out = tmp_path / "s.jsonl"
    child = subprocess.Popen(["sleep", "30"])
    try:
        rc = stack_sample.main(["run", "--pid", str(child.pid), "--py-spy", str(fake), "--interval", "0.1",
                                "--duration", "0.35", "--out", str(out)])
    finally:
        child.kill()
        child.wait()
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert rc == 2 and len(recs) >= 2
    assert all(r["rc"] == 1 and "ptrace refused" in r["error"] for r in recs)


@pytest.mark.skipif(not shutil.which(PY_SPY) and not os.path.exists(PY_SPY), reason="py-spy not installed")
def test_live_dump_of_a_sleeping_child(tmp_path) -> None:
    """Start a Python process (own process group) that sleeps in inner(); sample its group; one long run."""
    code = textwrap.dedent("""
        import time
        def inner():
            time.sleep(60)
        def outer():
            inner()
        outer()
    """)
    child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        time.sleep(0.5)
        out, stop = tmp_path / "s.jsonl", tmp_path / "stop"
        # bounded by the stop file, created after ~2 s
        threading.Timer(2.0, stop.touch).start()
        rc = stack_sample.main(["run", "--pgid", str(child.pid), "--interval", "0.5", "--duration", "10",
                                "--stop-file", str(stop), "--out", str(out), "--py-spy", PY_SPY])
        recs = [json.loads(x) for x in out.read_text().splitlines()]
        if all(r["error"] for r in recs):
            pytest.skip(f"py-spy cannot attach here: {recs[0]['error']}")
        assert rc == 0 and 3 <= len(recs) <= 6 and {r["pid"] for r in recs} == {child.pid}
        runs = stack_sample.summarize(recs, frames=3, interval=0.5)
        assert len(runs) == 1 and runs[0]["frames"][0].startswith("inner (") and runs[0]["samples"] == len(recs)
        assert runs[0]["frames"][1].startswith("outer (")
    finally:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait()
