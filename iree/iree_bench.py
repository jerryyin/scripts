#!/usr/bin/env python3
"""Compile MLIR to VMFB and benchmark with iree-benchmark-module.

Usable as a standalone CLI or as an importable library.

CLI examples:
    # Compile
    python3 iree_bench.py compile foo.mlir -o foo.vmfb --target gfx950

    # Benchmark
    python3 iree_bench.py bench foo.vmfb --function=matmul \
        --input=1024x2048xbf16 --input=2048x512xbf16

Library usage:
    from iree_bench import compile_mlir, run_benchmark, parse_benchmark_time_us
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


DEFAULT_TARGET = "gfx950"


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_mlir(
    mlir_path: str,
    vmfb_path: str,
    target: str = DEFAULT_TARGET,
    extra_flags: list[str] | None = None,
    iree_compile: str = "iree-compile",
    timeout: int = 600,
) -> tuple[bool, str]:
    """Compile an MLIR file to a VMFB.

    Returns (success, error_message).
    """
    cmd = [
        iree_compile, mlir_path,
        "--iree-hal-target-backends=rocm",
        f"--iree-rocm-target={target}",
        "-o", vmfb_path,
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return False, result.stderr[:500]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def parse_benchmark_time_us(output: str) -> float | None:
    """Parse benchmark time in microseconds from iree-benchmark-module output.

    Looks for real_time_median first, then real_time_mean, then any BM_ line.
    """
    for pattern in ["real_time_median", "real_time_mean"]:
        for line in output.split("\n"):
            if pattern in line:
                t = _parse_time_from_line(line)
                if t is not None:
                    return t

    for line in output.split("\n"):
        if line.strip().startswith("BM_"):
            t = _parse_time_from_line(line)
            if t is not None:
                return t
    return None


def _parse_time_from_line(line: str) -> float | None:
    m = re.search(r"([\d.]+)\s+(us|ms|ns|s)\b", line)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "ms":
        val *= 1000
    elif unit == "ns":
        val /= 1000
    elif unit == "s":
        val *= 1_000_000
    return val


def run_benchmark(
    vmfb_path: str,
    func_name: str,
    lhs_spec: str,
    rhs_spec: str,
    repetitions: int = 5,
    warmup: float = 1.0,
    iree_benchmark: str = "iree-benchmark-module",
    timeout: int = 600,
) -> float | None:
    """Benchmark a VMFB and return the median time in microseconds, or None."""
    cmd = [
        iree_benchmark,
        f"--module={vmfb_path}",
        "--device=hip",
        f"--function={func_name}",
        f"--input={lhs_spec}",
        f"--input={rhs_spec}",
        f"--benchmark_repetitions={repetitions}",
        f"--benchmark_min_warmup_time={warmup}",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None
        return parse_benchmark_time_us(result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return None


# ---------------------------------------------------------------------------
# Generic command runner
# ---------------------------------------------------------------------------

def run_cmd(cmd: str, timeout: int = 600) -> tuple[bool, str]:
    """Run a shell command string. Returns (success, combined_output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_compile(args):
    ok, err = compile_mlir(
        args.mlir, args.output,
        target=args.target,
        extra_flags=args.extra_flags.split() if args.extra_flags else None,
        iree_compile=args.iree_compile,
        timeout=args.timeout,
    )
    if ok:
        print(f"Compiled to {args.output}", file=sys.stderr)
    else:
        print(f"Compilation failed: {err}", file=sys.stderr)
        sys.exit(1)


def _cmd_bench(args):
    if len(args.input) < 2:
        print("Need at least 2 --input specs (lhs, rhs)", file=sys.stderr)
        sys.exit(1)
    t = run_benchmark(
        args.vmfb, args.function,
        args.input[0], args.input[1],
        repetitions=args.reps,
        warmup=args.warmup,
        iree_benchmark=args.iree_benchmark,
        timeout=args.timeout,
    )
    if t is not None:
        print(f"{t:.1f} us")
    else:
        print("Benchmark failed", file=sys.stderr)
        sys.exit(1)


def _main():
    parser = argparse.ArgumentParser(
        description="Compile MLIR and benchmark IREE modules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compile sub-command
    p_compile = sub.add_parser("compile", help="Compile MLIR to VMFB")
    p_compile.add_argument("mlir", help="Input MLIR file")
    p_compile.add_argument("-o", "--output", required=True, help="Output VMFB")
    p_compile.add_argument("--target", default=DEFAULT_TARGET)
    p_compile.add_argument("--iree-compile", default="iree-compile")
    p_compile.add_argument("--extra-flags", default="",
                           help="Extra flags (space-separated string)")
    p_compile.add_argument("--timeout", type=int, default=600)
    p_compile.set_defaults(func=_cmd_compile)

    # bench sub-command
    p_bench = sub.add_parser("bench", help="Benchmark a VMFB")
    p_bench.add_argument("vmfb", help="VMFB file to benchmark")
    p_bench.add_argument("--function", required=True)
    p_bench.add_argument("--input", action="append", default=[],
                         help="Input spec (e.g. 1024x2048xbf16). "
                              "Provide at least twice.")
    p_bench.add_argument("--reps", type=int, default=5)
    p_bench.add_argument("--warmup", type=float, default=1.0)
    p_bench.add_argument("--iree-benchmark", default="iree-benchmark-module")
    p_bench.add_argument("--timeout", type=int, default=600)
    p_bench.set_defaults(func=_cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()
