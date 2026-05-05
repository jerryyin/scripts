#!/usr/bin/env python3
"""Parse and summarize stats from IREE-compiled .rocmasm assembly files.

Counts opcodes, register usage, and other quick metrics that are useful
when comparing two compilations (e.g. with vs. without a flag).

Usage:
    # Single file: print stats.
    python3 inspect_isa.py path/to/foo.rocmasm

    # Two files: print side-by-side comparison (e.g. baseline vs experiment).
    python3 inspect_isa.py baseline.rocmasm experiment.rocmasm \\
        --labels base exp

    # A directory: pick the first/largest .rocmasm and dump stats.
    python3 inspect_isa.py path/to/dump_dir/

    # Two directories: locate the .rocmasm in each and compare.
    python3 inspect_isa.py dir_a/ dir_b/ --labels A B

Library usage:
    from inspect_isa import collect_stats, find_rocmasm
"""

from __future__ import annotations

import argparse
import os
import re
import sys


# Opcodes worth counting. Order matters for printing.
OPCODES = [
    "v_mfma",
    "buffer_load",
    "buffer_store",
    "ds_read",
    "ds_write",
    "s_barrier",
    "s_waitcnt",
    "v_perm",
    "v_mad_u64",
]


def find_rocmasm(path: str) -> str | None:
    """If `path` is a directory, return the first .rocmasm inside (recursively).
    If `path` is a file, return it directly. Returns None if nothing found."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None
    matches = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".rocmasm"):
                matches.append(os.path.join(root, f))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return matches[0]


def collect_stats(asm_path: str) -> dict:
    """Read a .rocmasm file and return basic stats dict."""
    stats = {"path": asm_path, "lines": 0, "vgpr": None, "sgpr": None}
    for op in OPCODES:
        stats[op] = 0

    vgpr_re = re.compile(r"vgpr_count\s*:\s*(\d+)")
    sgpr_re = re.compile(r"sgpr_count\s*:\s*(\d+)")

    with open(asm_path) as f:
        for line in f:
            stats["lines"] += 1
            if stats["vgpr"] is None:
                m = vgpr_re.search(line)
                if m:
                    stats["vgpr"] = int(m.group(1))
            if stats["sgpr"] is None:
                m = sgpr_re.search(line)
                if m:
                    stats["sgpr"] = int(m.group(1))
            for op in OPCODES:
                if op in line:
                    stats[op] += 1
    return stats


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def print_single(stats: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}{stats['path']}")
    print(f"  Lines={stats['lines']}  VGPR={_fmt(stats['vgpr'])}  "
          f"SGPR={_fmt(stats['sgpr'])}")
    parts = [f"{op}={stats[op]}" for op in OPCODES]
    print("  " + "  ".join(parts))


def print_compare(a: dict, b: dict, label_a: str, label_b: str) -> None:
    cols = ["lines", "vgpr", "sgpr"] + OPCODES
    width_metric = max(len(c) for c in cols)
    width_val = max(len(label_a), len(label_b), 6)

    print(f"  {'metric':<{width_metric}}  "
          f"{label_a:>{width_val}}  {label_b:>{width_val}}  delta")
    print(f"  {'-' * width_metric}  "
          f"{'-' * width_val}  {'-' * width_val}  -----")
    for c in cols:
        av, bv = a.get(c), b.get(c)
        if av is None or bv is None:
            delta = ""
        else:
            d = bv - av
            delta = f"{d:+d}"
        print(f"  {c:<{width_metric}}  "
              f"{_fmt(av):>{width_val}}  {_fmt(bv):>{width_val}}  {delta}")


def _resolve_or_die(path: str) -> str:
    asm = find_rocmasm(path)
    if asm is None:
        print(f"No .rocmasm found at {path}", file=sys.stderr)
        sys.exit(1)
    return asm


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="+",
        help="One or two .rocmasm files (or directories containing them).",
    )
    parser.add_argument(
        "--labels", nargs=2, metavar=("A", "B"), default=None,
        help="Labels to use when comparing two files.",
    )
    args = parser.parse_args()

    if len(args.paths) > 2:
        print("Pass at most 2 paths.", file=sys.stderr)
        sys.exit(1)

    if len(args.paths) == 1:
        asm = _resolve_or_die(args.paths[0])
        print_single(collect_stats(asm))
        return

    asm_a = _resolve_or_die(args.paths[0])
    asm_b = _resolve_or_die(args.paths[1])
    label_a, label_b = args.labels if args.labels else ("A", "B")
    stats_a = collect_stats(asm_a)
    stats_b = collect_stats(asm_b)
    print(f"[{label_a}] {asm_a}")
    print(f"[{label_b}] {asm_b}")
    print()
    print_compare(stats_a, stats_b, label_a, label_b)


if __name__ == "__main__":
    main()
