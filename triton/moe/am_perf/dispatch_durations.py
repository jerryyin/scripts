#!/usr/bin/env python3
"""Select one AM dispatch from a model log without timing heuristics.

Study runs bracket the target launch with ``AM_REPLAY_TARGET_PRE`` and
``AM_REPLAY_TARGET_POST`` JSON records.  The native AM start/end messages still
provide the authoritative dispatch ID and cycle window.  A legacy log can be
inspected by supplying all metadata explicitly, but the parser never guesses
that the longest dispatch is the target.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence


START = re.compile(
    r"DispatchId\s+(\d+)::\s+CP_clk\s*=\s*(\d+)\s+Execute Dispatch"
    r".*?\bx\s*=\s*(\d+)"
)
END = re.compile(
    r"DumpDispatchEndTime\s+Time:\d+\s+.*?DispatchDone:(\d+)\s+.*?\bclk\s+(\d+)"
)
MARKER = re.compile(r"^AM_REPLAY_TARGET_(PRE|POST)\s+(\{.*\})\s*$")


class ParseError(ValueError):
    """The log cannot identify exactly one dispatch under the requested rules."""


@dataclasses.dataclass(frozen=True)
class NativeDispatch:
    dispatch_id: int
    grid: tuple[int, ...]
    start_cycle: int
    end_cycle: int
    start_line: int
    end_line: int

    @property
    def duration_cycles(self) -> int:
        return self.end_cycle - self.start_cycle


@dataclasses.dataclass(frozen=True)
class TargetMarker:
    kind: str
    line_number: int
    queue_id: str
    kernel_symbol: str
    grid: tuple[int, ...]
    clock_period_ps: float
    profile: str
    counter_source: str

    def payload(self) -> tuple[object, ...]:
        return (
            self.queue_id,
            self.kernel_symbol,
            self.grid,
            self.clock_period_ps,
            self.profile,
            self.counter_source,
        )


def _positive_int_tuple(value: object, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ParseError(f"{context}: grid must be a non-empty JSON array")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
        raise ParseError(f"{context}: grid entries must be positive integers")
    return tuple(value)


def _parse_marker(kind: str, payload_text: str, line_number: int) -> TargetMarker:
    context = f"line {line_number} AM_REPLAY_TARGET_{kind}"
    try:
        value = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ParseError(f"{context}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ParseError(f"{context}: payload must be a JSON object")
    required = {
        "queue_id",
        "kernel_symbol",
        "grid",
        "clock_period_ps",
        "profile",
        "counter_source",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ParseError(f"{context}: missing fields {missing}")
    unexpected = sorted(value.keys() - required)
    if unexpected:
        raise ParseError(f"{context}: unexpected fields {unexpected}")
    for name in ("queue_id", "kernel_symbol", "profile", "counter_source"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ParseError(f"{context}: {name} must be a non-empty string")
    clock = value["clock_period_ps"]
    if isinstance(clock, bool) or not isinstance(clock, (int, float)) or clock <= 0:
        raise ParseError(f"{context}: clock_period_ps must be positive")
    return TargetMarker(
        kind=kind,
        line_number=line_number,
        queue_id=value["queue_id"],
        kernel_symbol=value["kernel_symbol"],
        grid=_positive_int_tuple(value["grid"], context=context),
        clock_period_ps=float(clock),
        profile=value["profile"],
        counter_source=value["counter_source"],
    )


def parse_log(lines: Iterable[str]) -> tuple[list[NativeDispatch], list[TargetMarker]]:
    starts: dict[int, tuple[int, tuple[int, ...], int]] = {}
    ends: dict[int, tuple[int, int]] = {}
    markers: list[TargetMarker] = []
    for line_number, line in enumerate(lines, start=1):
        marker = MARKER.match(line.rstrip("\r\n"))
        if marker:
            markers.append(_parse_marker(marker.group(1), marker.group(2), line_number))
            continue
        start = START.search(line)
        if start:
            dispatch_id = int(start.group(1))
            if dispatch_id in starts:
                raise ParseError(f"dispatch {dispatch_id}: duplicate start records")
            starts[dispatch_id] = (
                int(start.group(2)),
                (int(start.group(3)),),
                line_number,
            )
            continue
        end = END.search(line)
        if end:
            dispatch_id = int(end.group(1))
            if dispatch_id in ends:
                raise ParseError(f"dispatch {dispatch_id}: duplicate end records")
            ends[dispatch_id] = (int(end.group(2)), line_number)

    rows: list[NativeDispatch] = []
    for dispatch_id in sorted(starts.keys() & ends.keys()):
        start_cycle, grid, start_line = starts[dispatch_id]
        end_cycle, end_line = ends[dispatch_id]
        if end_cycle <= start_cycle:
            raise ParseError(
                f"dispatch {dispatch_id}: end cycle {end_cycle} is not after "
                f"start cycle {start_cycle}"
            )
        if end_line <= start_line:
            raise ParseError(f"dispatch {dispatch_id}: end record precedes start record")
        rows.append(
            NativeDispatch(
                dispatch_id=dispatch_id,
                grid=grid,
                start_cycle=start_cycle,
                end_cycle=end_cycle,
                start_line=start_line,
                end_line=end_line,
            )
        )
    return rows, markers


def _marker_pair(markers: Sequence[TargetMarker]) -> tuple[TargetMarker, TargetMarker] | None:
    if not markers:
        return None
    pre = [marker for marker in markers if marker.kind == "PRE"]
    post = [marker for marker in markers if marker.kind == "POST"]
    if len(pre) != 1 or len(post) != 1:
        raise ParseError(
            "study markers must contain exactly one PRE and one POST record; "
            f"found PRE={len(pre)}, POST={len(post)}"
        )
    if pre[0].line_number >= post[0].line_number:
        raise ParseError("AM_REPLAY_TARGET_POST must follow AM_REPLAY_TARGET_PRE")
    if pre[0].payload() != post[0].payload():
        raise ParseError("AM_REPLAY_TARGET PRE/POST payloads differ")
    return pre[0], post[0]


def _grid_matches(native: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    # Current native AM start records expose x only.  A multi-dimensional study
    # grid is admissible only when its trailing dimensions are one.
    if len(native) == len(expected):
        return native == expected
    return len(native) == 1 and expected[0] == native[0] and all(v == 1 for v in expected[1:])


def select_dispatch(
    rows: Sequence[NativeDispatch],
    markers: Sequence[TargetMarker],
    *,
    dispatch_id: int | None = None,
    expected_grid: tuple[int, ...] | None = None,
    kernel_symbol: str | None = None,
    last: bool = False,
) -> tuple[NativeDispatch, TargetMarker | None, str]:
    choices = sum((dispatch_id is not None, expected_grid is not None, last))
    if choices != 1:
        raise ParseError("select exactly one of dispatch ID, expected grid, or --last")
    if not rows:
        raise ParseError("no complete native AM dispatches found")
    pair = _marker_pair(markers)
    marker = pair[0] if pair else None

    candidates = list(rows)
    if pair is not None:
        pre, post = pair
        candidates = [
            row
            for row in candidates
            if pre.line_number < row.start_line < row.end_line < post.line_number
        ]
        if not candidates:
            raise ParseError("no complete native dispatch occurs between target PRE/POST markers")

    if dispatch_id is not None:
        candidates = [row for row in candidates if row.dispatch_id == dispatch_id]
        selection = "dispatch_id"
    elif last:
        candidates = [max(candidates, key=lambda row: row.start_line)]
        selection = "last_debug_only"
    else:
        assert expected_grid is not None
        candidates = [row for row in candidates if _grid_matches(row.grid, expected_grid)]
        selection = "unique_grid_and_symbol"

    if marker is not None:
        if expected_grid is not None and marker.grid != expected_grid:
            raise ParseError(
                f"requested grid {list(expected_grid)} differs from marker grid {list(marker.grid)}"
            )
        if kernel_symbol is not None and marker.kernel_symbol != kernel_symbol:
            raise ParseError(
                f"requested symbol {kernel_symbol!r} differs from marker symbol "
                f"{marker.kernel_symbol!r}"
            )
        candidates = [row for row in candidates if _grid_matches(row.grid, marker.grid)]
    elif not last and kernel_symbol is None:
        raise ParseError("kernel symbol is required when target markers are absent")

    if len(candidates) != 1:
        ids = [row.dispatch_id for row in candidates]
        raise ParseError(
            "dispatch selection must be unique; "
            f"matched {len(candidates)} complete dispatches with IDs {ids}"
        )
    return candidates[0], marker, selection


def result_dict(
    row: NativeDispatch,
    marker: TargetMarker | None,
    *,
    kernel_symbol: str | None,
    expected_grid: tuple[int, ...] | None,
    clock_period_ps: float | None,
    counter_source: str | None,
    log_sha256: str,
    selection: str,
) -> dict[str, object]:
    symbol = marker.kernel_symbol if marker else kernel_symbol
    grid = marker.grid if marker else (expected_grid or row.grid)
    clock = marker.clock_period_ps if marker else clock_period_ps
    source = marker.counter_source if marker else counter_source
    if not symbol:
        raise ParseError("kernel symbol is unavailable")
    if clock is None or clock <= 0:
        raise ParseError("a positive clock period is required")
    if not source:
        raise ParseError("counter source is required")
    return {
        "schema_version": 1,
        "queue_id": marker.queue_id if marker else None,
        "profile": marker.profile if marker else None,
        "dispatch_id": row.dispatch_id,
        "kernel_symbol": symbol,
        "grid": list(grid),
        "start_cycle": row.start_cycle,
        "end_cycle": row.end_cycle,
        "duration_cycles": row.duration_cycles,
        "clock_period_ps": clock,
        "converted_us": row.duration_cycles * clock / 1_000_000.0,
        "counter_source": source,
        "selection": selection,
        "admissible_study_selection": selection != "last_debug_only",
        "source_log_sha256": log_sha256,
    }


def _parse_grid(text: str) -> tuple[int, ...]:
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid must be comma-separated integers") from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("grid entries must be positive")
    return tuple(values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dispatch-id", type=int)
    selection.add_argument("--expected-grid", type=_parse_grid)
    selection.add_argument("--last", action="store_true", help="debugging only; never study-admissible")
    parser.add_argument("--kernel-symbol")
    parser.add_argument("--clock-period-ps", type=float)
    parser.add_argument("--counter-source")
    parser.add_argument("--json", action="store_true", help="emit one stable JSON object")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw = args.log.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        rows, markers = parse_log(text.splitlines(keepends=True))
        row, marker, selection = select_dispatch(
            rows,
            markers,
            dispatch_id=args.dispatch_id,
            expected_grid=args.expected_grid,
            kernel_symbol=args.kernel_symbol,
            last=args.last,
        )
        result = result_dict(
            row,
            marker,
            kernel_symbol=args.kernel_symbol,
            expected_grid=args.expected_grid,
            clock_period_ps=args.clock_period_ps,
            counter_source=args.counter_source,
            log_sha256=hashlib.sha256(raw).hexdigest(),
            selection=selection,
        )
    except (OSError, ParseError) as error:
        print(f"dispatch_durations.py: error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(
            "id grid start_cycle end_cycle duration_cycles symbol counter_source\n"
            f"{result['dispatch_id']} {result['grid']} {result['start_cycle']} "
            f"{result['end_cycle']} {result['duration_cycles']} "
            f"{result['kernel_symbol']} {result['counter_source']}"
        )
        print(
            "suggested itrace_analyze stall window: "
            f"{result['start_cycle']} {result['end_cycle']}"
        )
        if not result["admissible_study_selection"]:
            print("warning: --last is debugging-only and not study-admissible", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
