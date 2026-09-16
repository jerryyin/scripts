#!/usr/bin/env python3
"""Summarise one replay A/B log directory into a JSON record, without going near the device.

Companion to reproduce.sh: `reproduce.sh run` launches each variant in blocks and prints,
per block, how many DISTINCT output hashes came back from identical input bytes. This
reads those logs plus the context files beside them and turns them into one record.

It parses the driver's own printed output rather than re-deriving anything, so the record
cannot claim something the run did not print.

WHAT IT REFUSES TO DO, and these are deliberate rather than unfinished. It does not pool
blocks that do not belong together, it does not convert counts into rates, and it does not
compare across boots or across a provenance boundary. A share of a sub-boundary is not a
share of the whole, and a count is not a rate.

THE UNION IS PER VARIANT AND WITHIN ONE LOCK HOLD. A variant's blocks run back to back on
the same boot, against the same code object, from the same capture, under one acquisition
of the board lock, so the set of distinct output hashes over all its executions is the
union of its blocks. The per-block counts are kept beside the union so a reader can see
whether the union hid a block that behaved differently.

USAGE
    python3 summarize_replay_ab.py <logdir> [--json <path>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# The driver prints "  <count> x  <sha256>" for each distinct hash, then a summary line.
HASH_LINE = re.compile(r"^\s+(\d+)\s+x\s+([0-9a-f]{64})\s*$")
SUMMARY_LINE = re.compile(r"^(\S+):\s+(\d+)\s+runs,\s+(\d+)\s+distinct output hash\(es\)\s*$")
EXIT_LINE = re.compile(r"^EXIT=(-?\d+)\s*$")
GRID_LINE = re.compile(r"^grid\s+(.*)$")
DEVICE_LINE = re.compile(r"^device\s+(\S+)\s*$")
ACCEPTS_LINE = re.compile(r"^object accepts (\d+) threads per workgroup, (\d+) dynamic LDS bytes\s*$")


@dataclass
class Block:
    """One driver invocation: a variant, an ordinal, and the hashes it saw."""

    path: Path
    variant: str
    block: int
    label: str = ""
    executions: int = 0
    exit_code: int | None = None
    device: str = ""
    grid: str = ""
    accepts_threads: int = 0
    accepts_dyn_lds: int = 0
    hashes: dict[str, int] = field(default_factory=dict)

    @property
    def distinct(self) -> int:
        return len(self.hashes)

    @property
    def counted_executions(self) -> int:
        return sum(self.hashes.values())


def parse_block(path: Path, variant: str, block: int) -> Block:
    b = Block(path=path, variant=variant, block=block)
    for line in path.read_text().splitlines():
        m = HASH_LINE.match(line)
        if m:
            digest = m.group(2)
            if digest in b.hashes:
                raise RuntimeError(f"{path}: hash {digest} printed twice by the driver")
            b.hashes[digest] = int(m.group(1))
            continue
        m = SUMMARY_LINE.match(line)
        if m:
            b.label, b.executions = m.group(1), int(m.group(2))
            declared = int(m.group(3))
            continue
        m = EXIT_LINE.match(line)
        if m:
            b.exit_code = int(m.group(1))
            continue
        m = DEVICE_LINE.match(line)
        if m:
            b.device = m.group(1)
            continue
        m = GRID_LINE.match(line)
        if m:
            b.grid = m.group(1)
            continue
        m = ACCEPTS_LINE.match(line)
        if m:
            b.accepts_threads, b.accepts_dyn_lds = int(m.group(1)), int(m.group(2))

    if b.exit_code is None:
        raise RuntimeError(f"{path}: no EXIT= line, so this block did not finish through the launcher")
    if not b.label:
        raise RuntimeError(f"{path}: the driver printed no summary line, so this block produced no reading")
    if b.counted_executions != b.executions:
        raise RuntimeError(
            f"{path}: the per-hash counts sum to {b.counted_executions} but the driver says "
            f"{b.executions} executions ran"
        )
    if declared != b.distinct:
        raise RuntimeError(
            f"{path}: the driver declared {declared} distinct hashes and printed {b.distinct}"
        )
    return b


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def summarize(logdir: Path) -> dict[str, object]:
    context = read_kv(logdir / "context.txt")
    if not context:
        raise RuntimeError(f"{logdir}/context.txt is absent or empty; this is not a run directory")

    blocks: list[Block] = []
    for block in (1, 2):
        for variant in ("default", "expert"):
            path = logdir / f"block{block}-{variant}.txt"
            if not path.exists():
                raise RuntimeError(f"{path} is absent; the run did not complete all four blocks")
            blocks.append(parse_block(path, variant, block))

    per_variant: dict[str, object] = {}
    for variant in ("default", "expert"):
        mine = [b for b in blocks if b.variant == variant]
        union: dict[str, int] = {}
        for b in mine:
            for digest, n in b.hashes.items():
                union[digest] = union.get(digest, 0) + n
        per_variant[variant] = {
            "executions": sum(b.executions for b in mine),
            "distinct_output_hashes": len(union),
            "distinct_output_hashes_per_block": {f"block{b.block}": b.distinct for b in mine},
            "executions_per_block": {f"block{b.block}": b.executions for b in mine},
            "position_in_each_block": {"block1": "first" if variant == "default" else "second",
                                       "block2": "second" if variant == "default" else "first"},
            "hashes": [{"sha256": d, "executions": n} for d, n in sorted(union.items(), key=lambda kv: -kv[1])],
            "deterministic": len(union) == 1,
            "exit_codes": {f"block{b.block}": b.exit_code for b in mine},
        }

    devices = {b.device for b in blocks if b.device}
    grids = {b.grid for b in blocks if b.grid}

    return {
        "record_type": "replay_ab_distinct_output_hash_count",
        "schema_version": 1,
        "arm": context.get("ARM"),
        "llvm_revision": context.get("LLVM"),
        "capture": context.get("CAPTURE"),
        "boot_id": context.get("BOOT_ID"),
        "untuned_boot": context.get("UNTUNED_BOOT") == "true",
        "device": sorted(devices),
        "grid": sorted(grids),
        "executions_per_variant": int(context.get("EXECUTIONS_PER_VARIANT", "0")),
        "order_balancing": (
            "Each variant ran two blocks of 60 within one acquisition of the board lock, and "
            "the order of the two variants was reversed between blocks, so each was first "
            "once and second once. Per-block counts are reported beside the union."
        ),
        "per_variant": per_variant,
        "the_claim_is_the_count": (
            "The finding is the number of DISTINCT output hashes over identical input bytes. "
            "It is a count, not a rate, and it is not a throughput or a magnitude of any kind. "
            "Nothing in the driver interprets the data."
        ),
        "pooling_prohibition": (
            "This row belongs to boot " + str(context.get("BOOT_ID")) + " and may not be pooled "
            "with, differenced against, or presented in the same figure as a row from another "
            "boot. Rows from before a provenance boundary are on the other side of it as "
            "well. Named beside each other with boot and side attached is fine; combined is not."
        ),
        "lock": {
            "path": os.environ.get("GPU_LOCK_FILE", "/data/lock/amd-gpu.lock"),
            "wait_seconds": context.get("LOCK_WAIT_SECONDS"),
            "acquired_at_utc": context.get("LOCK_ACQUIRED_AT_UTC"),
            "held_seconds": context.get("LOCK_HELD_SECONDS"),
            "discipline": "blocking wait, never polled; the holder and every waiter untouched",
        },
        "neighbours": {
            "before": read_kv(logdir / "bracket-before.txt").get("FOREIGN"),
            "after": read_kv(logdir / "bracket-after.txt").get("FOREIGN"),
            "why": "from inside the container the count is not observable; UNREADABLE is never coerced to zero",
        },
        "preflight": {
            "verdict": context.get("PREFLIGHT_VERDICT"),
            "checked_at_utc": context.get("PREFLIGHT_CHECKED_AT_UTC"),
            "age_is_recorded_not_enforced": True,
        },
        "driver": {
            "source": "investigations/a8w4-decode-performance/llvm-ticket-expert-scheduling/driver/replay.cpp",
            "source_sha256": context.get("REPLAY_CPP_SHA256"),
            "binary_sha256": context.get("REPLAY_BINARY_SHA256"),
        },
        "failed_blocks": int(context.get("FAILED_BLOCKS", "0")),
        "pair_identity": (logdir / "pair-identity.txt").read_text()
        if (logdir / "pair-identity.txt").exists()
        else "NOT RECORDED -- do not draw a comparison from the pair until it is re-established",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    record = summarize(args.logdir)
    out = args.json or (args.logdir / "RECORD.json")
    if out.exists():
        raise RuntimeError(f"{out} already exists; a record identity is never overwritten")
    out.write_text(json.dumps(record, indent=2) + "\n")

    arm, rev = record["arm"], record["llvm_revision"]
    print(f"{arm} on LLVM {rev}, boot {record['boot_id']}")
    for variant in ("default", "expert"):
        v = record["per_variant"][variant]
        print(
            f"  {variant:8s} {v['executions']:3d} executions, "
            f"{v['distinct_output_hashes']:3d} distinct output hash(es)  "
            f"(per block {v['distinct_output_hashes_per_block']})"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
