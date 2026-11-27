#!/usr/bin/env python3
"""CPU tests of the runtime module the patch adds (step_timing.py, taken from instrumentation.diff) with a
fake torch: the sampled steps' graph span, the GPU step to the next forward, its kind, and harvesting
only completed records (no host synchronization while serving)."""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = "python/tokenspeed/runtime/utils/step_timing.py"


def module_source() -> str:
    """The new file step_timing.py as instrumentation.diff adds it."""
    out, inside = [], False
    for line in (HERE / "instrumentation.diff").read_text().splitlines():
        if line.startswith("diff --git "):
            inside = line.endswith(f"b/{MODULE}")
        elif inside and line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    assert out, f"{MODULE} not found in instrumentation.diff"
    return "\n".join(out) + "\n"


class Fake:
    """GPU clock in ms; an event completes `lag` ms after it was recorded."""

    def __init__(self, lag=0.5):
        self.clock, self.capturing, self.lag = 0.0, False, lag
        fake = self

        class Stream:
            stream_id = 1

            def __eq__(self, other):
                return True

        class Event:
            def __init__(self, enable_timing=False, external=False):
                self.t = None

            def record(self, stream=None):
                self.t = fake.clock

            def query(self):
                return self.t is not None and self.t <= fake.clock - fake.lag

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return other.t - self.t

        self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            Event=Event, current_stream=Stream, is_current_stream_capturing=lambda: fake.capturing))


def load(dump: str, every: int, fake: Fake):
    env = {"TS_STEP_TIMING": "1", "TS_STEP_TIMING_DIR": dump, "TS_STEP_TIMING_EVERY": str(every)}
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    sys.modules["torch"] = fake.torch
    try:
        spec = importlib.util.spec_from_loader("step_timing_under_test", loader=None)
        mod = importlib.util.module_from_spec(spec)
        exec(compile(module_source(), MODULE, "exec"), mod.__dict__)
    finally:
        del sys.modules["torch"]
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v
    return mod


def capture(st, fake, graph):
    st.capture_begin(graph)
    fake.capturing = True
    for n in ("model.begin", "final", "model.end"):
        st.mark(n, -1)
    fake.capturing = False
    st.capture_end(graph, 1)


def steps(dump):
    recs = [json.loads(l) for f in glob.glob(os.path.join(dump, "*.jsonl")) for l in open(f) if l.strip()]
    return [r for r in recs if r["type"] == "step"]


def test_graph_span_and_gpu_step_to_next_replay() -> None:
    fake = Fake()
    with tempfile.TemporaryDirectory() as d:
        st = load(d, 2, fake)
        g = object()
        capture(st, fake, g)
        for i in range(10):
            fake.clock += 1.0  # GPU outside the graph
            if i == 6:
                st.other_forward()  # a prefill forward after the sampled replay 6
                fake.clock += 7.0
            tok = st.before_replay(g, None, 1, 1)
            fake.clock += 4.0  # graph
            st.after_replay(tok)
        st.flush()
        s = steps(d)
        assert [r["replay"] for r in s] == [2, 4, 6, 8, 10]
        assert all(r["graph_ms"] == 4.0 and len(r["t"]) == 3 for r in s)
        assert [r.get("next_kind") for r in s] == ["replay", "replay", "other", "replay", None]
        assert [r.get("outside_graph_ms") for r in s] == [1.0, 1.0, 1.0, 1.0, None]
        assert s[0]["next_begin_ms"] == 5.0 and "launch_ms" in s[0]


def test_harvest_waits_for_completed_events() -> None:
    """While the GPU lags the host, records stay pending instead of the host synchronizing."""
    fake = Fake(lag=1e9)
    with tempfile.TemporaryDirectory() as d:
        st = load(d, 1, fake)
        g = object()
        capture(st, fake, g)
        for _ in range(5):
            st.after_replay(st.before_replay(g, None, 1, 1))
            fake.clock += 5.0
        assert len(st._S.pending) == 5 and st._S.records == 0
        st.flush()  # at exit: harvest everything
        assert len(steps(d)) == 5 and st._S.pending == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
