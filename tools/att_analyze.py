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

  # Filter to specific dispatch (default: highest-hitcount dispatch):
  att_analyze.py <stats_csv> --dispatch 2
"""
import argparse
import csv
import sys
from collections import defaultdict


def categorize(inst: str) -> str:
    """Classify an instruction into a performance-relevant category."""
    if "v_mfma" in inst or "v_smfma" in inst:
        return "mfma"
    if "buffer_load" in inst and "lds" in inst:
        return "buffer_load_lds"
    if "buffer_load" in inst:
        return "buffer_load"
    if "buffer_store" in inst:
        return "buffer_store"
    if "ds_read" in inst:
        return "ds_read"
    if "ds_write" in inst:
        return "ds_write"
    if "s_barrier" in inst:
        return "s_barrier"
    if "s_waitcnt" in inst:
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
    "mfma", "buffer_load_lds", "buffer_load", "buffer_store",
    "ds_read", "ds_write", "s_barrier", "s_waitcnt",
    "v_perm", "alu", "branch", "nop", "s_endpgm", "other",
]

CATEGORY_LABELS = {
    "mfma": "MFMA (compute)",
    "buffer_load_lds": "DMA (buf→LDS)",
    "buffer_load": "buffer_load",
    "buffer_store": "buffer_store",
    "ds_read": "ds_read (LDS)",
    "ds_write": "ds_write (LDS)",
    "s_barrier": "s_barrier",
    "s_waitcnt": "s_waitcnt",
    "v_perm": "v_perm (swizzle)",
    "alu": "ALU / addr",
    "branch": "branch",
    "nop": "nop",
    "s_endpgm": "s_endpgm",
    "other": "other",
}


def parse_att_csv(path: str):
    """Parse ATT stats CSV, return list of instruction records."""
    records = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                rec = {
                    "codeobj": int(row[0]),
                    "vaddr": int(row[1]),
                    "inst": row[2].strip(),
                    "hitcount": int(row[3]),
                    "latency": int(row[4]),
                    "stall": int(row[5]),
                    "idle": int(row[6]),
                    "source": row[7] if len(row) > 7 else "",
                }
                rec["category"] = categorize(rec["inst"])
                records.append(rec)
            except (ValueError, IndexError):
                continue
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

    return dict(cats), loop_hitcount


def print_report(cats, loop_hitcount, label=""):
    """Print a formatted analysis report."""
    total_lat = sum(c["latency"] for c in cats.values()) or 1
    total_stall = sum(c["stall"] for c in cats.values()) or 1
    total_idle = sum(c["idle"] for c in cats.values())
    total_count = sum(c["count"] for c in cats.values())

    if label:
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")
    print(f"  Loop hitcount: {loop_hitcount}  |  Instructions per iteration: {total_count}")
    print()
    print(f"  {'Category':<20} {'#':>4} {'Latency':>12} {'Stall':>12} {'Idle':>10}  {'Lat%':>6} {'Stall%':>7}")
    print(f"  {'-' * 20} {'-' * 4} {'-' * 12} {'-' * 12} {'-' * 10}  {'-' * 6} {'-' * 7}")

    for cat in CATEGORY_ORDER:
        if cat not in cats:
            continue
        c = cats[cat]
        lat_pct = 100.0 * c["latency"] / total_lat
        stall_pct = 100.0 * c["stall"] / total_stall
        lbl = CATEGORY_LABELS.get(cat, cat)
        print(f"  {lbl:<20} {c['count']:>4} {c['latency']:>12,} {c['stall']:>12,} {c['idle']:>10,}  {lat_pct:>5.1f}% {stall_pct:>6.1f}%")

    print(f"  {'-' * 20} {'-' * 4} {'-' * 12} {'-' * 12} {'-' * 10}")
    print(f"  {'TOTAL':<20} {total_count:>4} {total_lat:>12,} {total_stall:>12,} {total_idle:>10,}")
    print()
    print(f"  Overall stall rate: {100.0 * total_stall / total_lat:.1f}%")
    print(f"  Overall idle rate:  {100.0 * total_idle / total_lat:.1f}%")
    print()

    print("  Top stall sources:")
    ranked = sorted(cats.items(), key=lambda kv: kv[1]["stall"], reverse=True)
    for cat, c in ranked[:5]:
        if c["stall"] == 0:
            break
        lbl = CATEGORY_LABELS.get(cat, cat)
        print(f"    {lbl:<20} {100.0 * c['stall'] / total_stall:>5.1f}%  ({c['stall']:>12,})")
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
    total1 = sum(c["latency"] for c in cats1.values()) or 1
    total2 = sum(c["latency"] for c in cats2.values()) or 1
    stall1 = sum(c["stall"] for c in cats1.values()) or 1
    stall2 = sum(c["stall"] for c in cats2.values()) or 1

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
        lp1 = 100.0 * c1["latency"] / total1
        lp2 = 100.0 * c2["latency"] / total2
        sp1 = 100.0 * c1["stall"] / stall1
        sp2 = 100.0 * c2["stall"] / stall2
        delta = lp2 - lp1
        lbl = CATEGORY_LABELS.get(cat, cat)
        print(f"  {lbl:<20} {lp1:>6.1f}% {sp1:>6.1f}% {c1['count']:>4}    {lp2:>6.1f}% {sp2:>6.1f}% {c2['count']:>4}  {delta:>+6.1f}%")

    cnt1 = sum(c["count"] for c in cats1.values())
    cnt2 = sum(c["count"] for c in cats2.values())
    sr1 = 100.0 * stall1 / total1
    sr2 = 100.0 * stall2 / total2
    print(f"  {'-' * 20}")
    print(f"  Total instructions:  {cnt1:>4}  vs  {cnt2:>4}")
    print(f"  Total latency:       {total1:>12,}  vs  {total2:>12,}  ({100.0*total2/total1 - 100:>+.1f}%)")
    print(f"  Stall rate:          {sr1:>5.1f}%  vs  {sr2:>5.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze ATT stats CSV from rocprofv3")
    parser.add_argument("csv_files", nargs="+", help="ATT stats CSV file(s)")
    parser.add_argument("--labels", nargs="*", help="Labels for each CSV (for comparison)")
    parser.add_argument("--detail", action="store_true", help="Show per-instruction detail")
    parser.add_argument("--dispatch", type=int, help="Filter to specific dispatch code object ID")
    args = parser.parse_args()

    all_data = []
    for i, path in enumerate(args.csv_files):
        records = parse_att_csv(path)
        if args.dispatch:
            records = [r for r in records if r["codeobj"] == args.dispatch]
        cats, hc = analyze(records)
        label = args.labels[i] if args.labels and i < len(args.labels) else path
        all_data.append((cats, hc, label, records))

    if len(all_data) == 1:
        cats, hc, label, records = all_data[0]
        print_report(cats, hc, label)
        if args.detail:
            print_detail(records, hc)
    elif len(all_data) == 2:
        cats1, hc1, label1, records1 = all_data[0]
        cats2, hc2, label2, records2 = all_data[1]
        print_report(cats1, hc1, label1)
        print_report(cats2, hc2, label2)
        print_comparison(cats1, hc1, label1, cats2, hc2, label2)
        if args.detail:
            print_detail(records1, hc1)
            print_detail(records2, hc2)
    else:
        for cats, hc, label, records in all_data:
            print_report(cats, hc, label)
            if args.detail:
                print_detail(records, hc)


if __name__ == "__main__":
    main()
