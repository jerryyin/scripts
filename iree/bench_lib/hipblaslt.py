#!/usr/bin/env python3
"""
Usage:
  python run_hipblaslt_refactor_flat.py commands.txt --out parsed_results.txt --raw-dir raw_outputs --add-arg --use_gpu_timer --timeout 300 --verbose
"""

import argparse
import csv
import io
import os
import re
import shlex
import subprocess
import sys
import pandas as pd
from typing import List, Optional, Tuple

def run_single_command(cmd_line: str, add_args: List[str], timeout: int) -> Tuple[str, bool]:
    """Run a command (append add_args). Return (combined_stdout_stderr, error_flag)."""
    if add_args:
        cmd_line = cmd_line + " " + " ".join(add_args)

    args = shlex.split(cmd_line)
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return proc.stdout or "", False
    except Exception as exc:
        return f"Exception while executing command: {exc}\n", True


def write_raw_output(path: str, text: str) -> None:
    """Write raw stdout/stderr to file (best-effort)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        # We intentionally don't raise here — raw writing failure shouldn't stop processing.
        print(f"WARNING: failed to write raw output '{path}': {e}", file=sys.stderr)


def _is_csv_like(s: str) -> bool:
    return "," in s and bool(re.search(r"[A-Za-z0-9]", s))


def parse_last_section_into_dataframe(text: str) -> Optional[pd.DataFrame]:
    """
    Simplified parser: find the last CSV-like header line immediately followed by
    at least one CSV-like value line, parse them into a DataFrame and return it.
    Caller chooses which column(s) to extract.
    """
    if pd is None:
        raise RuntimeError("pandas is required. Install with `pip install pandas`.")

    lines = text.splitlines()
    norm_lines = [re.sub(r"^\[\d+\]\s*:", "", ln).strip() for ln in lines]

    header_idx = None
    for i in range(len(norm_lines) - 1, 0, -1):
        if _is_csv_like(norm_lines[i]) and _is_csv_like(norm_lines[i - 1]):
            header_idx = i - 1
            break

    if header_idx is None:
        return None

    header_line = norm_lines[header_idx]
    value_lines = []
    for j in range(header_idx + 1, len(norm_lines)):
        if norm_lines[j] == "":
            break
        if _is_csv_like(norm_lines[j]):
            value_lines.append(norm_lines[j])
        else:
            break

    if not value_lines:
        return None

    try:
        header_tokens = [h.strip() for h in next(csv.reader(io.StringIO(header_line)))]
    except Exception:
        return None

    rows = []
    for vl in value_lines:
        try:
            row_tokens = [t.strip() for t in next(csv.reader(io.StringIO(vl)))]
        except Exception:
            row_tokens = []
        if len(row_tokens) < len(header_tokens):
            row_tokens += [""] * (len(header_tokens) - len(row_tokens))
        elif len(row_tokens) > len(header_tokens):
            row_tokens = row_tokens[:len(header_tokens)]
        rows.append(row_tokens)

    try:
        df = pd.DataFrame(rows, columns=header_tokens)
    except Exception:
        return None

    return df


def find_columns_in_df(df: pd.DataFrame, targets: List[str]) -> List[str]:
    """
    Find column names in df matching any target string.
    Strategy: exact (ci) -> substring (ci) -> regex fallback.
    Returns deduplicated list preserving df column order.
    """
    found: List[str] = []
    cols = list(df.columns)

    for target in targets:
        t = target.strip().lower()
        # exact match
        for c in cols:
            if c.strip().lower() == t:
                found.append(c)
        if any(c.strip().lower() == t for c in cols):
            continue

        # substring
        for c in cols:
            if t in c.strip().lower():
                found.append(c)
        if any(t in c.strip().lower() for c in cols):
            continue

        # regex fallback
        pat = re.compile(re.escape(t), flags=re.IGNORECASE)
        for c in cols:
            if pat.search(c):
                found.append(c)

    # deduplicate preserving order
    dedup: List[str] = []
    seen = set()
    for c in found:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup


def extract_values_from_df(df: pd.DataFrame, selected_cols: List[str]) -> List[str]:
    """
    Safely extract first-row values for selected_cols from df.
    Returns list of strings (empty string if missing/error).
    """
    values: List[str] = []
    for col in selected_cols:
        try:
            v = df[col].iloc[0]
            val = "" if pd.isna(v) else str(v).strip()
        except Exception:
            val = ""
        values.append(val)
    return values


def run_commands(
    input_path: str,
    out_path: str,
    raw_dir: str,
    add_args: List[str],
    timeout: int,
    verbose: bool,
) -> None:
    os.makedirs(raw_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as infile, \
            open(out_path, "w", newline="", encoding="utf-8") as outfile:
        csv_writer = csv.writer(outfile)

        for idx, raw_line in enumerate(infile, start=1):
            raw_filename = os.path.join(raw_dir, f"line_{idx:04d}.txt")

            # guard: comment / blank -> write empty CSV row (newline) and continue
            if raw_line.strip() == "" or raw_line.lstrip().startswith("#"):
                csv_writer.writerow([])
                write_raw_output(raw_filename, "")
                if verbose:
                    print(f"[{idx:04d}] comment/blank -> wrote empty line", file=sys.stderr)
                continue

            cmd_line = raw_line.rstrip("\n")
            if verbose:
                display_cmd = cmd_line + (" " + " ".join(add_args) if add_args else "")
                print(f"[{idx:04d}] running: {display_cmd}", file=sys.stderr)

            output_text, error_flag = run_single_command(cmd_line, add_args, timeout)
            write_raw_output(raw_filename, output_text)

            if error_flag:
                csv_writer.writerow([])
                if verbose:
                    print(f"[{idx:04d}] command error -> wrote empty line", file=sys.stderr)
                continue

            df = parse_last_section_into_dataframe(output_text)
            if df is None:
                csv_writer.writerow([])
                if verbose:
                    print(f"[{idx:04d}] parse failed -> wrote empty line", file=sys.stderr)
                continue

            df.columns = [c.strip() for c in df.columns]
            target_cols = find_columns_in_df(df, ["us"])
            if not target_cols:
                csv_writer.writerow([])
                if verbose:
                    print(f"[{idx:04d}] 'us' column not found in parsed columns: {list(df.columns)}", file=sys.stderr)
                continue

            values = extract_values_from_df(df, target_cols)
            csv_writer.writerow(values)
            if verbose:
                print(f"[{idx:04d}] parsed cols = {dict(zip(target_cols, values))}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hipBLASLt commands file and extract benchmark results (flattened control flow)."
    )
    parser.add_argument("input", help="Input file with one command (or comment) per line")
    parser.add_argument("--out", default="parsed_results.txt", help="Output CSV file (one row per input line)")
    parser.add_argument("--raw-dir", default="raw_outputs", help="Directory to save raw stdout/stderr per line")
    parser.add_argument("--add-arg", action="append", default=None,
                        help="Additional argument to append to every command. Can be used multiple times.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout (seconds) for each command (default 300s)")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    add_args = args.add_arg if args.add_arg is not None else ["--use_gpu_timer"]
    run_commands(args.input, args.out, args.raw_dir, add_args, args.timeout, args.verbose)


if __name__ == "__main__":
    main()
