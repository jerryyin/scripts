#!/usr/bin/env python3
"""
Analyze ATT (Advanced Thread Trace) stats CSV from rocprofv3.

Reads the per-instruction stats CSV produced by rocprofv3 ATT profiling,
categorizes instructions, and reports latency/stall breakdowns for the
hot loop body.

Usage:
  # Analyze a single ATT stats CSV:
  att_analyze.py <stats_csv>

  # Compare two runs (e.g. 2-stage vs 3-stage):
  att_analyze.py <baseline_csv> <experiment_csv> --labels "2-stage" "3-stage"

  # Show per-instruction detail for the loop body:
  att_analyze.py <stats_csv> --detail

  # Filter to a code-object load ID and explicitly select a loop hitcount:
  att_analyze.py <stats_csv> --codeobj 2 --loop-hitcount 200
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def categorize(inst: str) -> str:
    """Classify an instruction into a performance-relevant category."""
    if "v_mfma" in inst or "v_smfma" in inst or "v_wmma" in inst:
        return "mfma"
    if (
        ("buffer_load" in inst and "lds" in inst)
        or "global_load_async_to_lds" in inst
    ):
        return "buffer_load_lds"
    if "buffer_load" in inst:
        return "buffer_load"
    if "global_load" in inst:
        return "global_load"
    if "buffer_store" in inst:
        return "buffer_store"
    if "global_store" in inst:
        return "global_store"
    if "ds_read" in inst or "ds_load" in inst:
        return "ds_read"
    if "ds_write" in inst or "ds_store" in inst:
        return "ds_write"
    if "s_barrier" in inst:
        return "s_barrier"
    if "s_clause" in inst:
        return "s_clause"
    if any(
        wait in inst
        for wait in (
            "s_waitcnt",
            "s_wait_xcnt",
            "s_wait_loadcnt",
            "s_wait_storecnt",
            "s_wait_dscnt",
            "s_wait_kmcnt",
        )
    ):
        return "s_waitcnt"
    if "v_perm" in inst:
        return "v_perm"
    if "s_endpgm" in inst:
        return "s_endpgm"
    if "s_cbranch" in inst or "s_branch" in inst:
        return "branch"
    if any(k in inst for k in [
        "v_add", "v_sub", "v_mul", "v_or", "v_and", "v_xor",
        "v_lshl", "v_lshr", "v_bfe", "v_mov", "v_readfirstlane",
        "v_cmp", "v_cndmask",
        "s_add", "s_sub", "s_mul", "s_and", "s_or", "s_lshl", "s_lshr",
        "s_mov", "s_cmp", "s_brev", "s_bfe",
    ]):
        return "alu"
    if inst.startswith(";") or "nop" in inst:
        return "nop"
    return "other"


CATEGORY_ORDER = [
    "mfma", "buffer_load_lds", "buffer_load", "global_load",
    "buffer_store", "global_store",
    "ds_read", "ds_write", "s_barrier", "s_clause", "s_waitcnt",
    "v_perm", "alu", "branch", "nop", "s_endpgm", "other",
]

CATEGORY_LABELS = {
    "mfma": "Matrix (MFMA/WMMA)",
    "buffer_load_lds": "DMA (global→LDS)",
    "buffer_load": "buffer_load",
    "global_load": "global_load",
    "buffer_store": "buffer_store",
    "global_store": "global_store",
    "ds_read": "ds_read (LDS)",
    "ds_write": "ds_write (LDS)",
    "s_barrier": "s_barrier",
    "s_clause": "s_clause",
    "s_waitcnt": "s_waitcnt",
    "v_perm": "v_perm (swizzle)",
    "alu": "ALU / addr",
    "branch": "branch",
    "nop": "nop",
    "s_endpgm": "s_endpgm",
    "other": "other",
}


REQUIRED_COLUMNS = {
    "codeobj",
    "vaddr",
    "instruction",
    "hitcount",
    "latency",
    "stall",
    "idle",
}


def _column_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def parse_att_csv(path: str):
    """Parse ATT stats CSV, return list of instruction records."""
    records = []
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing ATT CSV header")
        normalized = {_column_name(name): name for name in reader.fieldnames}
        missing = sorted(REQUIRED_COLUMNS - set(normalized))
        if missing:
            raise ValueError(f"{path}: missing ATT CSV columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                rec = {
                    "codeobj": int(row[normalized["codeobj"]]),
                    "vaddr": int(row[normalized["vaddr"]]),
                    "inst": row[normalized["instruction"]].strip(),
                    "hitcount": int(row[normalized["hitcount"]]),
                    "latency": int(row[normalized["latency"]]),
                    "stall": int(row[normalized["stall"]]),
                    "idle": int(row[normalized["idle"]]),
                    "source": row.get(normalized.get("source", ""), ""),
                }
                rec["category"] = categorize(rec["inst"])
                records.append(rec)
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed ATT instruction row"
                ) from error
    return records


def find_loop_hitcount(records):
    """Find the hitcount value corresponding to the main loop body."""
    counts = defaultdict(int)
    for r in records:
        if r["hitcount"] > 0:
            counts[r["hitcount"]] += 1
    if not counts:
        return 0
    return max(counts, key=lambda h: h * counts[h])


def select_code_object(records, requested_code_object_id=None):
    """Select exactly one code-object load ID and return it with its records."""
    if requested_code_object_id is not None:
        records = [
            record
            for record in records
            if record["codeobj"] == requested_code_object_id
        ]
    if not records:
        raise ValueError("no ATT instruction records matched the requested code object")
    code_object_ids = {record["codeobj"] for record in records}
    if len(code_object_ids) != 1:
        raise ValueError(
            "selected records contain multiple Codeobj load IDs "
            f"{sorted(code_object_ids)}; select exactly one with --codeobj"
        )
    return records, code_object_ids.pop()


def analyze(records, loop_hitcount=None):
    """Aggregate stats by category for instructions with given hitcount."""
    if loop_hitcount is None:
        loop_hitcount = find_loop_hitcount(records)

    cats = defaultdict(lambda: {"count": 0, "latency": 0, "stall": 0, "idle": 0})
    for r in records:
        if r["hitcount"] != loop_hitcount:
            continue
        cat = r["category"]
        cats[cat]["count"] += 1
        cats[cat]["latency"] += r["latency"]
        cats[cat]["stall"] += r["stall"]
        cats[cat]["idle"] += r["idle"]

    positive_latency = sum(r["latency"] for r in records if r["hitcount"] > 0)
    selected_latency = sum(c["latency"] for c in cats.values())
    coverage_pct = 100.0 * selected_latency / positive_latency if positive_latency else 0.0
    return dict(cats), loop_hitcount, coverage_pct


def print_report(cats, loop_hitcount, coverage_pct, label=""):
    """Print a formatted analysis report."""
    total_lat = sum(c["latency"] for c in cats.values())
    total_stall = sum(c["stall"] for c in cats.values())
    total_idle = sum(c["idle"] for c in cats.values())
    total_count = sum(c["count"] for c in cats.values())
    latency_denominator = total_lat or 1
    stall_denominator = total_stall or 1

    if label:
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")
    print(
        f"  Loop hitcount: {loop_hitcount}  |  Instructions per iteration: {total_count}"
        f"  |  Selected-latency coverage: {coverage_pct:.1f}%"
    )
    print()
    print(f"  {'Category':<20} {'#':>4} {'Latency':>12} {'Stall':>12} {'Idle':>10}  {'Lat%':>6} {'Stall%':>7}")
    print(f"  {'-' * 20} {'-' * 4} {'-' * 12} {'-' * 12} {'-' * 10}  {'-' * 6} {'-' * 7}")

    for cat in CATEGORY_ORDER:
        if cat not in cats:
            continue
        c = cats[cat]
        lat_pct = 100.0 * c["latency"] / latency_denominator
        stall_pct = 100.0 * c["stall"] / stall_denominator
        lbl = CATEGORY_LABELS.get(cat, cat)
        print(f"  {lbl:<20} {c['count']:>4} {c['latency']:>12,} {c['stall']:>12,} {c['idle']:>10,}  {lat_pct:>5.1f}% {stall_pct:>6.1f}%")

    print(f"  {'-' * 20} {'-' * 4} {'-' * 12} {'-' * 12} {'-' * 10}")
    print(f"  {'TOTAL':<20} {total_count:>4} {total_lat:>12,} {total_stall:>12,} {total_idle:>10,}")
    print()
    print(f"  Overall stall rate: {100.0 * total_stall / latency_denominator:.1f}%")
    print(f"  Overall idle rate:  {100.0 * total_idle / latency_denominator:.1f}%")
    print()

    print("  Top stall sources:")
    ranked = sorted(cats.items(), key=lambda kv: kv[1]["stall"], reverse=True)
    for cat, c in ranked[:5]:
        if c["stall"] == 0:
            break
        lbl = CATEGORY_LABELS.get(cat, cat)
        print(f"    {lbl:<20} {100.0 * c['stall'] / stall_denominator:>5.1f}%  ({c['stall']:>12,})")
    print()


def print_detail(records, loop_hitcount):
    """Print per-instruction detail for the loop body."""
    print(f"\n{'=' * 90}")
    print(f"  Per-instruction detail (hitcount={loop_hitcount})")
    print(f"{'=' * 90}")
    print(f"  {'Vaddr':>8} {'Category':<18} {'Latency':>10} {'Stall':>10} {'Idle':>8}  Instruction")
    print(f"  {'-' * 8} {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 8}  {'-' * 30}")

    for r in records:
        if r["hitcount"] != loop_hitcount:
            continue
        cat = CATEGORY_LABELS.get(r["category"], r["category"])
        print(f"  {r['vaddr']:>8} {cat:<18} {r['latency']:>10,} {r['stall']:>10,} {r['idle']:>8,}  {r['inst']}")
    print()


def print_comparison(cats1, hc1, label1, cats2, hc2, label2):
    """Print side-by-side comparison of two runs."""
    all_cats = set(list(cats1.keys()) + list(cats2.keys()))
    total1 = sum(c["latency"] for c in cats1.values())
    total2 = sum(c["latency"] for c in cats2.values())
    stall1 = sum(c["stall"] for c in cats1.values())
    stall2 = sum(c["stall"] for c in cats2.values())
    latency_denominator1 = total1 or 1
    latency_denominator2 = total2 or 1
    stall_denominator1 = stall1 or 1
    stall_denominator2 = stall2 or 1

    print(f"\n{'=' * 80}")
    print(f"  Comparison: {label1} vs {label2}")
    print(f"{'=' * 80}")
    print(f"  {'':>20} {'--- ' + label1 + ' ---':>24}  {'--- ' + label2 + ' ---':>24}  {'Delta':>7}")
    print(f"  {'Category':<20} {'Lat%':>7} {'Stall%':>7} {'#':>4}    {'Lat%':>7} {'Stall%':>7} {'#':>4}  {'Lat':>7}")
    print(f"  {'-' * 20} {'-' * 7} {'-' * 7} {'-' * 4}    {'-' * 7} {'-' * 7} {'-' * 4}  {'-' * 7}")

    for cat in CATEGORY_ORDER:
        if cat not in all_cats:
            continue
        c1 = cats1.get(cat, {"count": 0, "latency": 0, "stall": 0, "idle": 0})
        c2 = cats2.get(cat, {"count": 0, "latency": 0, "stall": 0, "idle": 0})
        lp1 = 100.0 * c1["latency"] / latency_denominator1
        lp2 = 100.0 * c2["latency"] / latency_denominator2
        sp1 = 100.0 * c1["stall"] / stall_denominator1
        sp2 = 100.0 * c2["stall"] / stall_denominator2
        delta = lp2 - lp1
        lbl = CATEGORY_LABELS.get(cat, cat)
        print(f"  {lbl:<20} {lp1:>6.1f}% {sp1:>6.1f}% {c1['count']:>4}    {lp2:>6.1f}% {sp2:>6.1f}% {c2['count']:>4}  {delta:>+6.1f}%")

    cnt1 = sum(c["count"] for c in cats1.values())
    cnt2 = sum(c["count"] for c in cats2.values())
    sr1 = 100.0 * stall1 / latency_denominator1
    sr2 = 100.0 * stall2 / latency_denominator2
    print(f"  {'-' * 20}")
    print(f"  Total instructions:  {cnt1:>4}  vs  {cnt2:>4}")
    norm1 = total1 / hc1
    norm2 = total2 / hc2
    print(f"  Raw total latency:   {total1:>12,}  vs  {total2:>12,}")
    delta = 100.0 * norm2 / norm1 - 100 if norm1 else float("nan")
    print(f"  Latency / hitcount:  {norm1:>12,.2f}  vs  {norm2:>12,.2f}  ({delta:>+.1f}%)")
    print(f"  Stall rate:          {sr1:>5.1f}%  vs  {sr2:>5.1f}%")
    print()


def json_summary(path, label, cats, loop_hitcount, coverage_pct, codeobj):
    total_latency = sum(category["latency"] for category in cats.values())
    total_stall = sum(category["stall"] for category in cats.values())
    total_idle = sum(category["idle"] for category in cats.values())
    total_count = sum(category["count"] for category in cats.values())
    return {
        "schema_version": 1,
        "path": str(Path(path).resolve()),
        "label": label,
        "code_object_id": codeobj,
        "loop_hitcount": loop_hitcount,
        "instructions_per_iteration": total_count,
        "selected_latency_coverage_pct": coverage_pct,
        "other_latency_pct": (
            100.0 * cats.get("other", {}).get("latency", 0) / (total_latency or 1)
        ),
        "totals": {
            "latency": total_latency,
            "stall": total_stall,
            "idle": total_idle,
            "latency_per_hitcount": total_latency / loop_hitcount,
            "stall_rate_pct": 100.0 * total_stall / (total_latency or 1),
            "idle_rate_pct": 100.0 * total_idle / (total_latency or 1),
        },
        "categories": cats,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze ATT stats CSV from rocprofv3")
    parser.add_argument("csv_files", nargs="+", help="ATT stats CSV file(s)")
    parser.add_argument("--labels", nargs="*", help="Labels for each CSV (for comparison)")
    parser.add_argument("--detail", action="store_true", help="Show per-instruction detail")
    parser.add_argument(
        "--codeobj",
        type=int,
        help="filter the CSV Codeobj column (a code-object load ID, not a dispatch ID)",
    )
    parser.add_argument("--dispatch", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--loop-hitcount",
        type=int,
        action="append",
        help="explicit hot-loop hitcount; pass once for all CSVs or once per CSV",
    )
    parser.add_argument("--json-out", type=Path, help="write the metric vector as JSON")
    args = parser.parse_args()
    if args.dispatch is not None:
        parser.error(
            "--dispatch was incorrect: the CSV column is Codeobj, not Dispatch_Id; "
            "use --codeobj only when a stats file contains multiple code objects"
        )
    if args.loop_hitcount and len(args.loop_hitcount) not in (1, len(args.csv_files)):
        parser.error("--loop-hitcount must be supplied once or once per CSV")

    all_data = []
    for i, path in enumerate(args.csv_files):
        try:
            records = parse_att_csv(path)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        try:
            records, selected_code_object_id = select_code_object(
                records, args.codeobj
            )
        except ValueError as error:
            parser.error(f"{path}: {error}")
        requested_hitcount = None
        if args.loop_hitcount:
            requested_hitcount = args.loop_hitcount[0 if len(args.loop_hitcount) == 1 else i]
        cats, hc, coverage_pct = analyze(records, requested_hitcount)
        if hc <= 0 or not cats:
            parser.error(f"{path}: no positive-hitcount loop body was found")
        label = args.labels[i] if args.labels and i < len(args.labels) else path
        all_data.append(
            (cats, hc, coverage_pct, label, records, path, selected_code_object_id)
        )

    if len(all_data) == 1:
        cats, hc, coverage_pct, label, records, _, _ = all_data[0]
        print_report(cats, hc, coverage_pct, label)
        if args.detail:
            print_detail(records, hc)
    elif len(all_data) == 2:
        cats1, hc1, coverage1, label1, records1, _, _ = all_data[0]
        cats2, hc2, coverage2, label2, records2, _, _ = all_data[1]
        print_report(cats1, hc1, coverage1, label1)
        print_report(cats2, hc2, coverage2, label2)
        print_comparison(cats1, hc1, label1, cats2, hc2, label2)
        if args.detail:
            print_detail(records1, hc1)
            print_detail(records2, hc2)
    else:
        for cats, hc, coverage_pct, label, records, _, _ in all_data:
            print_report(cats, hc, coverage_pct, label)
            if args.detail:
                print_detail(records, hc)

    if args.json_out:
        summaries = [
            json_summary(path, label, cats, hc, coverage_pct, code_object_id)
            for (
                cats,
                hc,
                coverage_pct,
                label,
                _records,
                path,
                code_object_id,
            ) in all_data
        ]
        args.json_out.write_text(
            json.dumps({"captures": summaries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
