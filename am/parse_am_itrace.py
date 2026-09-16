#!/usr/bin/env python3
"""Turn an AM instruction trace into per-opcode-class cycle attribution.

The trace is a merged timeline of every wave on one WGP. Each line carries the
wave handle and an instruction sequence number, so lines belonging to different
waves interleave freely and nothing may be parsed as a block:

    60b00031 00000094: VGLOBAL[WGP00_SIMD11_WAVE0] TS=50860
    60b00031 00000094:   global_load_b32  v4, v4, s[26:27] ... // 0005...: EE05...
    60b00031 00000095: // SQ:TYPE_DEP VEC: INST_COUNTER_VM_VSRC:1 TS=50861

Two cycle accounts come out of this, and they answer different questions.

`stall` is exact. Every TYPE_DEP line is one cycle that one instruction spent
blocked, tagged with the dependency that blocked it, and the records for an
instruction end on the cycle before it issues -- verified across 1,910 of the
1,924 dependency groups in the pilot trace. Summing them attributes waiting to
a named cause.

`span` is the gap to the next issue in the same wave, which is what ItraceViz
draws and what its own documentation warns about: it is not the instruction's
execution time. A long-latency load whose wave then stalls shows a small span,
because the wait lands on whichever instruction is blocked, usually the
s_wait_loadcnt. Read spans as occupancy of the issue slot, not as cost.

Categories come from att_analyze.py, the decoder used on the hardware ATT
captures, imported rather than restated so the two sides cannot drift into
bucketing the same instruction differently. Its hash is recorded in the output
so a later comparison can prove both sides used one taxonomy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

# att_analyze.py supplies the opcode taxonomy (see the module docstring). It normally sits
# in the sibling profiling/ directory of the same checkout, so that is tried first and works
# with no environment set at all. ATT_ANALYZE and SCRIPTS_ROOT override it when this script
# is run from somewhere else; $HOME is the last resort.
_SIBLING = Path(__file__).resolve().parent.parent / "profiling" / "att_analyze.py"
_SCRIPTS_ROOT = os.environ.get("SCRIPTS_ROOT")
ATT_ANALYZE_CANDIDATES = tuple(
    candidate
    for candidate in (
        Path(os.environ["ATT_ANALYZE"]) if os.environ.get("ATT_ANALYZE") else None,
        _SIBLING,
        Path(_SCRIPTS_ROOT) / "profiling" / "att_analyze.py" if _SCRIPTS_ROOT else None,
        Path.home() / "scripts" / "profiling" / "att_analyze.py",
    )
    if candidate is not None
)

INSTRUCTION = re.compile(
    r"^(?P<wave>\S+) (?P<seq>\S+): (?P<encoding>[A-Z_0-9]+)"
    r"\[(?P<wgp>WGP\d+)_(?P<simd>SIMD\d+)_(?P<wave_slot>WAVE\d+)\] TS=(?P<ts>\d+)"
)
DISASSEMBLY = re.compile(
    r"^(?P<wave>\S+) (?P<seq>\S+):\s+(?P<inst>\S.*?)\s+//\s+[0-9A-Fa-f]+:"
)
TYPE_DEP = re.compile(
    r"^(?P<wave>\S+) (?P<seq>\S+): // SQ:TYPE_DEP\s+(?P<unit>\S+):\s+"
    r"(?P<detail>.*?)\s*TS=(?P<ts>\d+)"
)
# A blocked cycle names one or more counters with the depth outstanding on each,
# run together: "INST_COUNTER_VA_SDST:1INST_COUNTER_VA_SSRC:2". Keying on the
# raw text would split one cause across as many buckets as it has depths, so
# only the counter names identify the reason.
DEPENDENCY_COUNTER = re.compile(r"(INST_COUNTER_[A-Z_]+?):(\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_categorizer(explicit: Path | None) -> tuple[Callable[[str], str], list[str], Path, str]:
    candidates = (explicit,) if explicit else ATT_ANALYZE_CANDIDATES
    for path in candidates:
        try:
            readable = path is not None and path.is_file()
        except PermissionError:
            # The container path is unreadable from a workstation checkout.
            continue
        if readable:
            spec = importlib.util.spec_from_file_location("att_analyze", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.categorize, list(module.CATEGORY_ORDER), path, sha256_file(path)
    raise SystemExit(
        "att_analyze.py not found; the hardware decoder defines the categories "
        "and restating them here would let the two sides drift apart"
    )


def open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def parse(path: Path, categorize: Callable[[str], str]) -> dict[str, Any]:
    issue: dict[tuple[str, str], dict[str, Any]] = {}
    order: dict[str, list[tuple[str, str]]] = defaultdict(list)
    stalls: dict[tuple[str, str], Counter] = defaultdict(Counter)
    wave_identity: dict[str, tuple[str, str, str]] = {}
    wave_start: dict[str, int] = {}
    pending: tuple[str, str] | None = None
    orphan_stalls = 0

    with open_trace(path) as stream:
        for line in stream:
            found = INSTRUCTION.match(line)
            if found:
                key = (found["wave"], found["seq"])
                if found["encoding"] == "WAVE_START":
                    wave_identity[found["wave"]] = (
                        found["wgp"], found["simd"], found["wave_slot"]
                    )
                    wave_start[found["wave"]] = int(found["ts"])
                    pending = None
                    continue
                wave_identity.setdefault(
                    found["wave"], (found["wgp"], found["simd"], found["wave_slot"])
                )
                if key not in issue:
                    issue[key] = {"ts": int(found["ts"]), "inst": None}
                    order[found["wave"]].append(key)
                pending = key
                continue

            if pending is not None:
                disassembled = DISASSEMBLY.match(line)
                if disassembled and (disassembled["wave"], disassembled["seq"]) == pending:
                    if issue[pending]["inst"] is None:
                        issue[pending]["inst"] = disassembled["inst"]
                    pending = None
                    continue

            blocked = TYPE_DEP.match(line)
            if blocked:
                key = (blocked["wave"], blocked["seq"])
                counters = DEPENDENCY_COUNTER.findall(blocked["detail"])
                names = "+".join(
                    name.removeprefix("INST_COUNTER_") for name, _ in counters
                ) or blocked["detail"].strip()
                stalls[key][f"{blocked['unit']} {names}"] += 1

    per_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "span_cycles": 0, "stall_cycles": 0}
    )
    per_wave: dict[str, dict[str, Any]] = {}
    reasons_by_category: dict[str, Counter] = defaultdict(Counter)
    reasons = Counter()
    # att_analyze.py predates gfx1250 and has no rule for several of its
    # mnemonics, so whatever lands in `other` has to be named. The hardware side
    # buckets it identically, which keeps the comparison sound, but a large
    # unnamed bucket would hide the mechanism from both.
    unclassified: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "span_cycles": 0, "stall_cycles": 0}
    )
    undecoded = 0

    for wave, keys in order.items():
        keys.sort(key=lambda k: issue[k]["ts"])
        wave_span = 0
        wave_stall = 0
        for index, key in enumerate(keys):
            record = issue[key]
            text = record["inst"]
            if text is None:
                undecoded += 1
                category = "other"
            else:
                category = categorize(text)
            # The final instruction of a wave has no following issue to measure
            # against, so it contributes its stall but no span.
            span = (
                issue[keys[index + 1]]["ts"] - record["ts"]
                if index + 1 < len(keys)
                else 0
            )
            stall = sum(stalls.get(key, Counter()).values())
            bucket = per_category[category]
            bucket["count"] += 1
            bucket["span_cycles"] += span
            bucket["stall_cycles"] += stall
            if category == "other" and text is not None:
                named = unclassified[text.split(maxsplit=1)[0].lower()]
                named["count"] += 1
                named["span_cycles"] += span
                named["stall_cycles"] += stall
            for reason, hits in stalls.get(key, Counter()).items():
                reasons[reason] += hits
                reasons_by_category[category][reason] += hits
            wave_span += span
            wave_stall += stall
        wgp, simd, slot = wave_identity.get(wave, ("?", "?", "?"))
        first = issue[keys[0]]["ts"] if keys else None
        last = issue[keys[-1]]["ts"] if keys else None
        per_wave[wave] = {
            "wgp": wgp,
            "simd": simd,
            "wave": slot,
            "instructions": len(keys),
            "first_issue_cycle": first,
            "last_issue_cycle": last,
            "span_cycles": wave_span,
            "stall_cycles": wave_stall,
            # Consecutive spans tile the wave's lifetime with no gap and no
            # overlap, so this must hold exactly. It is the check that catches a
            # dropped or double-counted instruction.
            "span_reconstructs_extent": (
                first is not None and wave_span == last - first
            ),
        }

    issued_keys = set(issue)
    orphan_stalls = sum(
        sum(counter.values()) for key, counter in stalls.items() if key not in issued_keys
    )

    return {
        "waves": per_wave,
        "categories": {name: dict(value) for name, value in per_category.items()},
        "stall_reasons": dict(reasons.most_common()),
        "stall_reasons_by_category": {
            name: dict(counter.most_common())
            for name, counter in reasons_by_category.items()
        },
        "uncategorized_mnemonics": {
            name: dict(value)
            for name, value in sorted(
                unclassified.items(), key=lambda item: -item[1]["span_cycles"]
            )
        },
        "instructions": len(issue),
        "undecoded_instructions": undecoded,
        "stall_cycles_without_matching_instruction": orphan_stalls,
    }


def render(result: dict[str, Any], category_order: list[str]) -> str:
    lines: list[str] = []
    waves = result["waves"]
    lines.append(
        f"{len(waves)} waves, {result['instructions']} instructions, "
        f"{sum(w['stall_cycles'] for w in waves.values())} stalled cycles"
    )
    lines.append("")
    lines.append("Per wave (waves run concurrently, so these do not sum to dispatch time)")
    lines.append(
        f"  {'wave':22s} {'inst':>6s} {'first':>8s} {'last':>8s} {'stall':>8s}  check"
    )
    for wave in sorted(waves.values(), key=lambda w: (w["simd"], w["wave"])):
        name = f"{wave['wgp']}_{wave['simd']}_{wave['wave']}"
        check = "ok" if wave["span_reconstructs_extent"] else "SPANS DO NOT TILE"
        lines.append(
            f"  {name:22s} {wave['instructions']:6d} {wave['first_issue_cycle']:8d} "
            f"{wave['last_issue_cycle']:8d} {wave['stall_cycles']:8d}  {check}"
        )

    categories = result["categories"]
    total_span = sum(value["span_cycles"] for value in categories.values())
    total_stall = sum(value["stall_cycles"] for value in categories.values())
    lines.append("")
    lines.append("Per category, summed over waves")
    lines.append(
        f"  {'category':18s} {'count':>7s} {'span':>9s} {'span%':>7s} "
        f"{'stall':>8s} {'stall%':>7s}"
    )
    ordered = [name for name in category_order if name in categories]
    ordered += sorted(name for name in categories if name not in category_order)
    for name in ordered:
        value = categories[name]
        span_share = 100.0 * value["span_cycles"] / total_span if total_span else 0.0
        stall_share = 100.0 * value["stall_cycles"] / total_stall if total_stall else 0.0
        lines.append(
            f"  {name:18s} {value['count']:7d} {value['span_cycles']:9d} "
            f"{span_share:6.1f}% {value['stall_cycles']:8d} {stall_share:6.1f}%"
        )

    uncategorized = result["uncategorized_mnemonics"]
    if uncategorized:
        lines.append("")
        lines.append(
            "What 'other' holds -- att_analyze.py has no rule for these gfx1250 "
            "mnemonics, on both sides"
        )
        lines.append(f"  {'mnemonic':26s} {'count':>7s} {'span':>9s} {'stall':>8s}")
        for name, value in list(uncategorized.items())[:8]:
            lines.append(
                f"  {name:26s} {value['count']:7d} {value['span_cycles']:9d} "
                f"{value['stall_cycles']:8d}"
            )

    lines.append("")
    lines.append("Why waves waited, by named dependency")
    # Sorted here rather than relying on insertion order: the JSON is written
    # with sorted keys, so a reloaded result orders these alphabetically and
    # "the top few" would silently mean "the first few by name".
    ranked = sorted(result["stall_reasons"].items(), key=lambda item: -item[1])
    for reason, hits in ranked[:10]:
        share = 100.0 * hits / total_stall if total_stall else 0.0
        lines.append(f"  {reason:40s} {hits:7d} {share:6.1f}%")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="xcc*_itrace_emu.mon or .mon.gz")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--att-analyze", type=Path)
    args = parser.parse_args()

    categorize, category_order, att_path, att_sha = load_categorizer(args.att_analyze)
    result = parse(args.trace, categorize)
    result["provenance"] = {
        "trace": str(args.trace),
        "trace_sha256": sha256_file(args.trace),
        "att_analyze": str(att_path),
        "att_analyze_sha256": att_sha,
    }

    print(render(result, category_order))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
