#!/usr/bin/env python3
"""Put the AM simulator's cycle attribution next to hardware ATT's for the same kernel.

Answers a question a total-time comparison cannot: does the simulator lose cycles for
the REASONS hardware does? A simulator can rank a set of configurations correctly and
still be right by accident -- in the run this was built for, AM ranked correctly while
overestimating total time by 1.42x. Ranking says nothing about mechanism. Both sides
carry per-instruction cycles, so the question becomes concrete and checkable: does the
same opcode category absorb the same share of the cycles on both sides?

INPUTS: a summary JSON from parse_am_itrace.py (the AM side) and a decoded ATT CSV
directory (the hardware side). Prints a per-category table and optionally writes JSON.

Two pairs of columns, and they mean different things.

`span` (AM) and `latency` (ATT) both measure occupancy of the issue slot: how
long the wave sat at an instruction before moving to the next. Neither is
execution time. A long-latency load shows a small figure in both, because the
waiting lands on whichever instruction is blocked, usually a wait.

`stall` on both sides is cycles an instruction was blocked at issue. AM's comes
from TYPE_DEP records, one per cycle, each naming the dependency that blocked it;
ATT's comes from the hardware's own stall accounting. These are the columns that
say *why* time went missing, and they are the point of the comparison.

Both sides are bucketed by the same att_analyze.py, imported rather than
restated, and its hash is recorded. A comparison run with two different decoders
would be measuring the taxonomy rather than the machine, so a mismatch against
the hash the AM summary recorded is refused unless it is acknowledged.

Scope worth remembering when reading the output: ATT sees the waves on one SIMD,
AM's summary covers every wave of the workgroup. The AM waves of these endpoints
are near enough symmetric -- each holds about a quarter of the workgroup's stall
cycles -- that the shares stay comparable, but they are not the same population.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The sibling profiling/ directory of this checkout first, so no environment is needed in
# the normal case; ATT_ANALYZE overrides it, then SCRIPTS_ROOT, then $HOME.
def _default_att_analyze() -> Path:
    if os.environ.get("ATT_ANALYZE"):
        return Path(os.environ["ATT_ANALYZE"])
    sibling = Path(__file__).resolve().parent.parent / "profiling" / "att_analyze.py"
    if sibling.is_file():
        return sibling
    if os.environ.get("SCRIPTS_ROOT"):
        return Path(os.environ["SCRIPTS_ROOT"]) / "profiling" / "att_analyze.py"
    return Path.home() / "scripts" / "profiling" / "att_analyze.py"


DEFAULT_ATT_ANALYZE = _default_att_analyze()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_categorize(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("att_analyze", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.categorize


def read_att(paths: list[Path], categorize: Any) -> dict[str, Any]:
    """Whole-trace totals per category, plus each dispatch's own shares.

    Every row's Latency and Stall are already summed over that instruction's
    hits, so summing rows gives the whole traced execution rather than one loop
    iteration. Each CSV is one dispatch, and because every dispatch here runs the
    same single workgroup, the spread across them is a reproducibility check
    rather than a set of different measurements.
    """
    total_latency: Counter[str] = Counter()
    total_stall: Counter[str] = Counter()
    uncategorized: Counter[str] = Counter()
    per_dispatch = []
    for path in paths:
        latency: Counter[str] = Counter()
        stall: Counter[str] = Counter()
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise SystemExit(f"{path}: no instruction rows; an empty capture cannot be compared")
        for row in rows:
            text = row["Instruction"].strip()
            mnemonic = text.split()[0] if text else ""
            category = categorize(mnemonic)
            if category == "other" and mnemonic:
                uncategorized[mnemonic] += 1
            latency[category] += int(row["Latency"])
            stall[category] += int(row["Stall"])
        total_latency.update(latency)
        total_stall.update(stall)
        per_dispatch.append(
            {
                "csv": path.name,
                "latency_cycles": sum(latency.values()),
                "stall_cycles": sum(stall.values()),
                "stall_share_by_category": shares(stall),
            }
        )
    return {
        "latency_cycles": dict(total_latency),
        "stall_cycles": dict(total_stall),
        "uncategorized_mnemonics": dict(uncategorized),
        "per_dispatch": per_dispatch,
        "dispatches": len(paths),
    }


def shares(counts: Counter[str] | dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {name: 100.0 * value / total for name, value in counts.items()}


def render(am: dict[str, Any], hw: dict[str, Any], label: str) -> None:
    am_categories = am["categories"]
    am_span = {name: rec["span_cycles"] for name, rec in am_categories.items()}
    am_stall = {name: rec["stall_cycles"] for name, rec in am_categories.items()}
    am_span_share, am_stall_share = shares(am_span), shares(am_stall)
    hw_latency_share = shares(hw["latency_cycles"])
    hw_stall_share = shares(hw["stall_cycles"])

    names = sorted(
        set(am_span) | set(hw["latency_cycles"]),
        key=lambda name: -max(am_stall_share.get(name, 0.0), hw_stall_share.get(name, 0.0)),
    )

    print(f"\n{label}")
    print("=" * 78)
    print(
        f"{'category':<18}{'AM span%':>10}{'HW lat%':>10}"
        f"{'  ':>4}{'AM stall%':>11}{'HW stall%':>11}{'  delta':>9}"
    )
    print("-" * 78)
    for name in names:
        am_st = am_stall_share.get(name, 0.0)
        hw_st = hw_stall_share.get(name, 0.0)
        print(
            f"{name:<18}{am_span_share.get(name, 0.0):>9.1f}%{hw_latency_share.get(name, 0.0):>9.1f}%"
            f"{'  ':>4}{am_st:>10.1f}%{hw_st:>10.1f}%{hw_st - am_st:>+8.1f}"
        )
    print("-" * 78)
    print(
        f"{'totals (cycles)':<18}{sum(am_span.values()):>10}{sum(hw['latency_cycles'].values()):>10}"
        f"{'  ':>4}{sum(am_stall.values()):>11}{sum(hw['stall_cycles'].values()):>11}"
    )

    spread = [entry["stall_share_by_category"] for entry in hw["per_dispatch"]]
    if len(spread) > 1:
        worst = max(
            (
                max(entry.get(name, 0.0) for entry in spread)
                - min(entry.get(name, 0.0) for entry in spread)
                for name in hw["stall_cycles"]
            ),
            default=0.0,
        )
        print(
            f"\nHardware reproducibility: {len(spread)} dispatches of the same workgroup, "
            f"widest stall-share spread {worst:.1f} points"
        )
    if hw["uncategorized_mnemonics"]:
        print(f"\nATT mnemonics left in other: {hw['uncategorized_mnemonics']}")
    if am.get("uncategorized_mnemonics"):
        print(f"AM mnemonics left in other: {am['uncategorized_mnemonics']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--am-summary", type=Path, required=True)
    parser.add_argument("--att-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--att-analyze", type=Path, default=DEFAULT_ATT_ANALYZE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--allow-decoder-mismatch",
        action="store_true",
        help="compare even though the AM summary was written by a different att_analyze.py",
    )
    args = parser.parse_args()

    am = json.loads(args.am_summary.read_text(encoding="utf-8"))
    decoder_sha = sha256_file(args.att_analyze)
    recorded = am.get("provenance", {}).get("att_analyze_sha256")
    if recorded != decoder_sha and not args.allow_decoder_mismatch:
        raise SystemExit(
            "the AM summary was bucketed by a different att_analyze.py "
            f"({recorded} != {decoder_sha}); re-run parse_am_itrace.py, or pass "
            "--allow-decoder-mismatch if the difference cannot touch these categories"
        )

    categorize = load_categorize(args.att_analyze)
    hw = read_att(list(args.att_csv), categorize)
    label = args.label or args.am_summary.parent.name
    render(am, hw, label)

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "label": label,
                    "am": {
                        "span_cycles": {n: r["span_cycles"] for n, r in am["categories"].items()},
                        "stall_cycles": {n: r["stall_cycles"] for n, r in am["categories"].items()},
                        "stall_reasons_by_category": am.get("stall_reasons_by_category", {}),
                        "provenance": am.get("provenance", {}),
                    },
                    "hardware": hw,
                    "att_analyze_sha256": decoder_sha,
                    "decoder_mismatch_allowed": bool(args.allow_decoder_mismatch),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
