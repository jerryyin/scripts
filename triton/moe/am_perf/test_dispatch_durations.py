from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("dispatch_durations.py")
SPEC = importlib.util.spec_from_file_location("dispatch_durations", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
dispatch_durations = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dispatch_durations
SPEC.loader.exec_module(dispatch_durations)


def marker(kind: str, *, grid: list[int] | None = None, symbol: str = "kernel") -> str:
    payload = {
        "queue_id": "F1-H0-8W",
        "kernel_symbol": symbol,
        "grid": grid or [20],
        "clock_period_ps": 500.0,
        "profile": "8xcc-profile",
        "counter_source": "am-dispatch-clock",
    }
    return f"AM_REPLAY_TARGET_{kind} {json.dumps(payload, sort_keys=True)}\n"


def native(dispatch_id: int, start: int, end: int, grid: int) -> str:
    return (
        f"DispatchId {dispatch_id}:: CP_clk ={start} Execute Dispatch on pipe 0 "
        f"countInCB=0, x={grid}\n"
        f"DumpDispatchEndTime Time:1 foo DispatchDone:{dispatch_id} bar clk {end}\n"
    )


def test_marker_selects_only_native_dispatch_inside_window() -> None:
    text = native(1, 1, 50, 20) + marker("PRE") + native(2, 100, 250, 20) + marker("POST")
    rows, markers = dispatch_durations.parse_log(text.splitlines(keepends=True))
    row, parsed_marker, selection = dispatch_durations.select_dispatch(
        rows, markers, expected_grid=(20,), kernel_symbol="kernel"
    )
    assert row.dispatch_id == 2
    assert parsed_marker is not None
    assert selection == "unique_grid_and_symbol"


def test_ambiguous_dispatches_between_markers_fail() -> None:
    text = marker("PRE") + native(2, 100, 250, 20) + native(3, 251, 400, 20) + marker("POST")
    rows, markers = dispatch_durations.parse_log(text.splitlines(keepends=True))
    with pytest.raises(dispatch_durations.ParseError, match="must be unique"):
        dispatch_durations.select_dispatch(
            rows, markers, expected_grid=(20,), kernel_symbol="kernel"
        )


def test_mismatched_pre_post_fail() -> None:
    text = marker("PRE") + native(2, 100, 250, 20) + marker("POST", symbol="other")
    rows, markers = dispatch_durations.parse_log(text.splitlines(keepends=True))
    with pytest.raises(dispatch_durations.ParseError, match="payloads differ"):
        dispatch_durations.select_dispatch(
            rows, markers, expected_grid=(20,), kernel_symbol="kernel"
        )


def test_single_marker_side_fails() -> None:
    text = marker("PRE") + native(2, 100, 250, 20)
    rows, markers = dispatch_durations.parse_log(text.splitlines(keepends=True))
    with pytest.raises(dispatch_durations.ParseError, match="PRE=1, POST=0"):
        dispatch_durations.select_dispatch(
            rows, markers, expected_grid=(20,), kernel_symbol="kernel"
        )


def test_explicit_dispatch_legacy_log_requires_metadata() -> None:
    rows, markers = dispatch_durations.parse_log(native(7, 10, 30, 1).splitlines(True))
    with pytest.raises(dispatch_durations.ParseError, match="kernel symbol"):
        dispatch_durations.select_dispatch(rows, markers, dispatch_id=7)
    row, parsed_marker, selection = dispatch_durations.select_dispatch(
        rows, markers, dispatch_id=7, kernel_symbol="kernel"
    )
    result = dispatch_durations.result_dict(
        row,
        parsed_marker,
        kernel_symbol="kernel",
        expected_grid=None,
        clock_period_ps=555.0,
        counter_source="am-dispatch-clock",
        log_sha256="0" * 64,
        selection=selection,
    )
    assert result["duration_cycles"] == 20
    assert result["converted_us"] == pytest.approx(0.0111)
    assert result["admissible_study_selection"] is True


def test_last_is_labeled_debug_only() -> None:
    rows, markers = dispatch_durations.parse_log(
        (native(1, 10, 30, 1) + native(2, 31, 60, 1)).splitlines(True)
    )
    row, parsed_marker, selection = dispatch_durations.select_dispatch(rows, markers, last=True)
    assert row.dispatch_id == 2
    result = dispatch_durations.result_dict(
        row,
        parsed_marker,
        kernel_symbol="debug-kernel",
        expected_grid=None,
        clock_period_ps=500.0,
        counter_source="am-dispatch-clock",
        log_sha256="0" * 64,
        selection=selection,
    )
    assert result["admissible_study_selection"] is False


def test_cli_json_is_stable_and_hashes_source_log(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    raw = (marker("PRE") + native(2, 100, 250, 20) + marker("POST")).encode()
    log.write_bytes(raw)
    command = [
        sys.executable,
        str(MODULE_PATH),
        str(log),
        "--expected-grid",
        "20",
        "--kernel-symbol",
        "kernel",
        "--json",
    ]
    first = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    second = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    assert first == second
    result = json.loads(first)
    assert result["dispatch_id"] == 2
    assert result["source_log_sha256"] == hashlib.sha256(raw).hexdigest()


def test_duplicate_native_record_fails() -> None:
    duplicate = native(1, 1, 2, 1) + native(1, 3, 4, 1)
    with pytest.raises(dispatch_durations.ParseError, match="duplicate start"):
        dispatch_durations.parse_log(duplicate.splitlines(True))
