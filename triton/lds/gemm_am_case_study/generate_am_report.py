#!/usr/bin/env python3
"""
Generate a bank conflict report from AM performance counter data.

Parses perf_counters_*_absolute.txt files produced by the AM simulator
and writes a markdown report summarising LDS/bank-conflict counters,
execution counters, and dispatch timing.

Usage:
    python generate_am_report.py                         # uses ./results/
    python generate_am_report.py --results-dir /path/to/results
"""
import os
import argparse
from pathlib import Path


BANK_CONFLICT_KEYS = [
    "SQ_INSTS_LDS",
    "SP_LDS_CYCLES",
    "SQ_LDS_IDLE_CYCLES",
    "SQ_LDS_LAT",
    "SQ_LDS_LAT_CNT",
    "GL0_LDS_REQ",
    "GL0_LDS_READ_BANK_CONFLICT",
    "GL0_LDS_WRITE_BANK_CONFLICT",
    "GL0_TCP_READ_BANK_CONFLICT",
    "GL0_TCP_WRITE_BANK_CONFLICT",
    "GL0_PARTITION_READ_CONFLICT",
    "GL0_PARTITION_WRITE_CONFLICT",
    "GL0_LDS_GL0_CONFLICT_PORT0",
    "GL0_LDS_GL0_CONFLICT_PORT1",
    "GL0_WAVE_LATENCY0_LDS",
    "GL0_WAVE_LATENCY0_LDS_CNT",
    "GL0_WAVE_LATENCY1_LDS",
    "GL0_WAVE_LATENCY1_LDS_CNT",
    "GL0_LDS_READ_PORT0",
    "GL0_LDS_WRITE_PORT0",
    "GL0_LDS_READ_PORT1",
    "GL0_LDS_WRITE_PORT1",
    "GL0_LDS_READ_PIPE0",
    "GL0_LDS_WRITE_PIPE0",
    "GL0_LDS_READ_PIPE1",
    "GL0_LDS_WRITE_PIPE1",
]

CYCLE_KEYS = [
    "SCLK",
    "GRBM_GUI_ACTIVE",
    "SPI_WAVES_LAUNCH",
    "SPI_WAVES_DONE",
    "SPI_BUSY",
    "SQ_INSTS",
    "SQ_INSTS_VALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_SCA",
    "SQ_INSTS_SMEM",
]

MIPERF_KEYS = [
    "DS_READ_BANK_CONFLICTS_SUM",
    "CU_CACHE_LDS_PARTITION_READ_CONFLICTS",
    "VMEM_READ_BANK_CONFLICTS",
]


def parse_absolute_txt(filepath):
    """Parse perf_counters_*_absolute.txt.  Format: 'COUNTER_NAME value'."""
    result = {}
    if not os.path.exists(filepath):
        return result
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    result[key] = float(parts[-1])
                except ValueError:
                    result[key] = parts[-1]
    return result


def get_status(run_dir):
    status_file = os.path.join(run_dir, "status.txt")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            return f.read().strip()
    return "NO_STATUS"


def get_dispatch_info(run_dir):
    """Extract dispatch timing from dumpPerDrawPerf.csv."""
    dump_file = os.path.join(run_dir, "dumpPerDrawPerf.csv")
    if not os.path.exists(dump_file):
        return None
    with open(dump_file, "r") as f:
        for line in f:
            if "drawID:" in line and "Clock Duration:" in line:
                parts = line.split(",")
                for i, p in enumerate(parts):
                    if "Clock Duration:" in p and i + 1 < len(parts):
                        try:
                            return int(parts[i + 1].strip())
                        except ValueError:
                            pass
    return None


def _format_val(val):
    if isinstance(val, (int, float)):
        return f"{val:.0f}"
    return str(val or "N/A")


def _counter_table(lines, keys, data_dict, source_key):
    """Append a markdown table comparing counter values across builds."""
    builds = sorted(data_dict.keys())
    header = "| Counter | " + " | ".join(builds) + " |"
    sep    = "|---------|" + "|".join(["------"] * len(builds)) + "|"
    lines.append(header)
    lines.append(sep)
    for key in keys:
        vals = [data_dict[b].get(source_key, {}).get(key) for b in builds]
        if all(v is None for v in vals):
            continue
        row = f"| {key} | " + " | ".join(_format_val(v) for v in vals) + " |"
        lines.append(row)
    lines.append("")


def generate_report(results_base):
    results_base = Path(results_base)
    if not results_base.is_dir():
        print(f"Results directory not found: {results_base}")
        print("Run the benchmark first, or pass --results-dir.")
        return ""

    lines = []
    lines.append("# Bank Conflict Report")
    lines.append("")
    lines.append("Auto-generated from AM performance counter data.")
    lines.append("")

    # Discover builds (sub-directories that contain *_am or *_ffm children)
    builds = sorted(
        d.name for d in results_base.iterdir()
        if d.is_dir() and any((d / sub).is_dir() for sub in
                              ["fp16_am", "fp8_am", "fp16_ffm", "fp8_ffm"])
    )
    if not builds:
        # Flat layout: results_base itself contains fp16_am/, etc.
        builds = [""]

    am_data = {}
    for build in builds:
        build_dir = results_base / build if build else results_base
        for dtype in ["fp16", "fp8"]:
            am_dir = build_dir / f"{dtype}_am"
            if not am_dir.is_dir():
                continue
            label = f"{build}/{dtype}" if build else dtype
            gfxperf = parse_absolute_txt(str(am_dir / "perf_counters_gfxperf_absolute.txt"))
            miperf  = parse_absolute_txt(str(am_dir / "perf_counters_miperf_absolute.txt"))
            am_data[label] = {
                "gfxperf": gfxperf,
                "miperf": miperf,
                "status": get_status(str(am_dir)),
                "dispatch_cycles": get_dispatch_info(str(am_dir)),
            }

    if not am_data:
        lines.append("No AM results found under " + str(results_base))
        report_text = "\n".join(lines)
        print(report_text)
        return report_text

    lines.append("## Execution Summary")
    lines.append("")
    lines.append("| Run | Status | Dispatch Cycles |")
    lines.append("|-----|--------|-----------------|")
    for label, d in am_data.items():
        cyc = str(d["dispatch_cycles"]) if d["dispatch_cycles"] else "N/A"
        lines.append(f"| {label} | {d['status']} | {cyc} |")
    lines.append("")

    lines.append("## LDS and Bank Conflict Counters")
    lines.append("")
    _counter_table(lines, BANK_CONFLICT_KEYS, am_data, "gfxperf")

    lines.append("## General Execution Counters")
    lines.append("")
    _counter_table(lines, CYCLE_KEYS, am_data, "gfxperf")

    lines.append("## miperf Counters")
    lines.append("")
    _counter_table(lines, MIPERF_KEYS, am_data, "miperf")

    report_text = "\n".join(lines)
    report_path = results_base / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report written to {report_path}")
    print()
    print(report_text)
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate bank conflict report from AM counter data"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Root of results tree (default: ./results/ next to this script)"
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    generate_report(results_dir)


if __name__ == "__main__":
    main()
