#!/usr/bin/env python3
"""Non-GPU unit tests for graph_event_smoke.replay_checks (the second-replay verdict)."""

from __future__ import annotations

import graph_event_smoke as ges

T = [0.515, 0.619, 0.720, 0.820, 0.921, 1.021, 1.026, 1.032, 1.79]   # replay 1, ms after `begin`
SPAN = 2.34


def test_same_replay_passes() -> None:
    r = ges.replay_checks(T, SPAN, list(T), SPAN)
    assert r["second_replay_consistent"] and r["max_offset_shift_ms"] == 0.0


def test_uniform_launch_gap_shift_passes_and_is_reported() -> None:
    # replay 2 starts 0.8 ms later after the host idle: every offset moves together
    t2 = [x + 0.8 for x in T]
    r = ges.replay_checks(T, SPAN, t2, SPAN + 0.8)
    assert r["second_replay_consistent"] and r["second_replay_refreshed"]
    assert abs(r["max_offset_shift_ms"] - 0.8) < 1e-9 and r["max_in_graph_shift_ms"] < 1e-9
    assert r["launch_gap_ms"] == [T[0], T[0] + 0.8]


def test_stale_event_fails() -> None:
    # event 3 not refreshed by replay 2: it still holds replay 1's time, which is before replay 2's `begin`
    t2 = list(T); t2[3] = -50.0
    r = ges.replay_checks(T, SPAN, t2, SPAN)
    assert not r["second_replay_refreshed"] and not r["second_replay_consistent"]


def test_all_events_stale_fails() -> None:
    r = ges.replay_checks(T, SPAN, [x - 50.0 for x in T], SPAN)
    assert not r["second_replay_consistent"]


def test_one_event_moved_inside_graph_fails() -> None:
    t2 = list(T); t2[5] += 0.6
    r = ges.replay_checks(T, SPAN, t2, SPAN + 0.6)
    assert r["second_replay_refreshed"] and not r["second_replay_in_graph_consistent"]
    assert not r["second_replay_consistent"] and r["max_in_graph_shift_index"] == 5


def test_event_after_replay_end_fails() -> None:
    t2 = list(T); t2[-1] = SPAN + 1.0
    assert not ges.replay_checks(T, SPAN, t2, SPAN)["second_replay_refreshed"]
