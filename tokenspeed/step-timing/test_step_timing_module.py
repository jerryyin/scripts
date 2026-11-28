#!/usr/bin/env python3
"""CPU tests of the runtime module the patch adds (step_timing.py, taken from instrumentation.diff) with a
fake torch: the sampled steps' graph span, the GPU step to the next forward, its kind, and harvesting
(SYNC=1: a sampled step is read before the next forward re-records its markers; SYNC=0: only completed
records, markers not completed written as null), plus the analyzer's handling of such dumps."""

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


def load(dump: str, every: int, fake: Fake, sync: str = "1", eager: str = "0"):
    env = {"TS_STEP_TIMING": "1", "TS_STEP_TIMING_DIR": dump, "TS_STEP_TIMING_EVERY": str(every),
           "TS_STEP_TIMING_SYNC": sync, "TS_STEP_TIMING_EAGER": eager}
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


def records(dump, kind):
    recs = [json.loads(l) for f in glob.glob(os.path.join(dump, "*.jsonl")) for l in open(f) if l.strip()]
    return [r for r in recs if r["type"] == kind]


def steps(dump):
    return records(dump, "step")


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
    """SYNC=0: while the GPU lags the host, records stay pending instead of the host synchronizing."""
    fake = Fake(lag=1e9)
    with tempfile.TemporaryDirectory() as d:
        st = load(d, 1, fake, sync="0")
        g = object()
        capture(st, fake, g)
        for _ in range(5):
            st.after_replay(st.before_replay(g, None, 1, 1))
            fake.clock += 5.0
        assert len(st._S.pending) == 5 and st._S.records == 0
        st.flush()  # at exit: harvest everything
        assert len(steps(d)) == 5 and st._S.pending == []


def replay(st, fake, graph, ms=4.0):
    """One replay: the graph's captured marker events are re-recorded by every replay."""
    tok = st.before_replay(graph, None, 1, 1)
    seq = st._S.seqs[id(graph)]
    for i, _ in enumerate(seq.names):
        seq.pool[i].record()
        fake.clock += ms / len(seq.names)
    st.after_replay(tok)
    fake.clock += 1.0


def test_sync_reads_markers_of_the_sampled_replay() -> None:
    """SYNC=1 (default): a sampled replay is harvested before the next replay re-records its markers,
    so its marker times lie inside its own graph span (0 .. graph_ms), not in the next replay's."""
    fake = Fake(lag=0.0)
    with tempfile.TemporaryDirectory() as d:
        st = load(d, 2, fake)
        g = object()
        capture(st, fake, g)
        for _ in range(8):
            replay(st, fake, g)
        st.flush()
        s = steps(d)
        assert [q["sync"] for q in records(d, "seq")] == [1]
        assert len(s) == 4 and all("incomplete" not in r for r in s)
        assert all(0.0 <= t <= r["graph_ms"] for r in s for t in r["t"]), s


def test_nosync_marks_overwritten_markers_incomplete() -> None:
    """SYNC=0: by the time the record is harvested the next replay has re-recorded the markers; those
    it has not completed are written as None and listed in "incomplete" (no exception)."""
    fake = Fake(lag=4.5)
    with tempfile.TemporaryDirectory() as d:
        st = load(d, 2, fake, sync="0")
        g = object()
        capture(st, fake, g)
        for _ in range(8):
            replay(st, fake, g)
        st.flush()
        s = steps(d)
        assert [q["sync"] for q in records(d, "seq")] == [0]
        assert s and any(r.get("incomplete") for r in s)
        assert all(r["t"][i] is None for r in s for i in r.get("incomplete", []))


def eager_step(st, fake, ms=4.0):
    """One timed eager decode: the eager pool's events are recorded again by every eager forward."""
    tok = st.eager_begin(None, 1)
    for n in ("model.begin", "final", "model.end"):
        st.mark(n, -1)
        fake.clock += ms / 3
    st.eager_end(tok)
    fake.clock += 1.0


def test_eager_sync_harvests_before_the_next_eager_forward() -> None:
    """EAGER=1: SYNC=1 writes a step before the next eager decode re-records the pool; SYNC=0 leaves it pending."""
    for sync, written in (("1", 1), ("0", 0)):
        fake = Fake(lag=1e9)  # no event completes on its own
        with tempfile.TemporaryDirectory() as d:
            st = load(d, 1, fake, sync=sync, eager="1")
            eager_step(st, fake)
            st.eager_begin(None, 1)
            assert st._S.records == written, (sync, st._S.records)
            st.flush()  # write the rest before the directory goes (flush also runs at exit)


def test_analyzer_reports_nosync_and_incomplete_records() -> None:
    """The analyzer on dumps from a SYNC=0 rank with incomplete markers: no exception, filled markers,
    and report lines counting both (over the analysed records)."""
    sys.path.insert(0, str(HERE))
    import analyze_step_timing as ast_

    with tempfile.TemporaryDirectory() as d:
        dump = os.path.join(d, "dump")
        ast_.synth(dump, steps=40)
        f = os.path.join(dump, "step-timing-r0-p1000.jsonl")
        lines = [json.loads(l) for l in open(f)]
        lines[0]["sync"] = 0
        last = len(lines[0]["names"]) - 1
        for rec in lines[1:11]:  # 10 records whose last marker had not completed
            rec["t"][last] = None
            rec["incomplete"] = [last]
        open(f, "w").write("\n".join(json.dumps(x) for x in lines) + "\n")
        ast_.analyze(dump, os.path.join(d, "out"), skip=0)
        report = open(os.path.join(d, "out", "report.md")).read()
        assert "rank 0: 40 records dumped with SYNC=0" in report, report
        assert f"rank 0: 10 records with markers not completed at harvest (filled with the previous marker's time): " \
               f"{lines[0]['names'][last]}" in report, report
        assert "rank 1: " not in report.split("## busy vs idle")[-1]
        ast_.analyze(dump, os.path.join(d, "out-skip"), skip=15)  # the incomplete records are all skipped
        report = open(os.path.join(d, "out-skip", "report.md")).read()
        assert "rank 0: 25 records dumped with SYNC=0" in report and "not completed" not in report, report


def test_analyzer_fills_incomplete_markers() -> None:
    """analyze_step_timing.fill_incomplete: a None marker takes the previous marker's time."""
    sys.path.insert(0, str(HERE))
    import analyze_step_timing as ast_

    assert ast_.fill_incomplete([0.5, None, 2.0, None, None]) == [0.5, 0.5, 2.0, 2.0, 2.0]
    assert ast_.fill_incomplete([None, 1.0]) == [0.0, 1.0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
