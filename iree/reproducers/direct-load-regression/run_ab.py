#!/usr/bin/env python3
"""A/B benchmark: compile shapes with two configs and compare performance.

Supports matmul and batch matmul with any transpose/dtype combination.
Shapes are loaded from CSV files (see data/*.csv for presets).

Workflow:
  1. Generate MLIR for each shape.
  2. Compile with config A (baseline) and config B (experiment).
  3. Skip benchmarking when both VMFBs are identical (no codegen diff).
  4. Benchmark both and report A/B comparison.
  5. Optionally write a results CSV.

Usage:
    # Use a preset CSV from data/ (full_regression compares no-DL vs DL by default):
    python3 run_ab.py --preset=full_regression --b-flags="--iree-llvmgpu-use-direct-load"

    # Quick A/B on the top-20 non-TN shapes:
    python3 run_ab.py --preset=non_tn_top20 \\
        --b-flags="--iree-llvmgpu-use-direct-load"

    # Use an arbitrary CSV file:
    python3 run_ab.py --csv=shapes.csv

    # Filter out tiny / huge cases (useful for the full regression sweep):
    python3 run_ab.py --preset=full_regression --skip-n1 --max-output-bytes=4G

    # Only specific shapes (1-based indices):
    python3 run_ab.py --preset=non_tn_top20 --only=05,07,10

    # Save results to CSV:
    python3 run_ab.py --preset=full_regression --out-csv=data/results.csv
"""

import argparse
import csv
import filecmp
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.realpath(os.path.join(_SCRIPT_DIR, "..", "..")))

from gen_matmul import gen_matmul_mlir, get_input_specs
from iree_bench import compile_mlir, run_benchmark, DEFAULT_TARGET

DATA_DIR = os.path.join(_SCRIPT_DIR, "data")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path):
    shapes = []
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["name"]
            shapes.append({
                "name": name,
                "M": int(row["M"]),
                "N": int(row["N"]),
                "K": int(row["K"]),
                "transpose": row.get("transpose") or "none",
                "batch": int(row["batch"]) if row.get("batch") else None,
                "dtype_lhs": row.get("dtype_lhs") or "bf16",
                "dtype_rhs": row.get("dtype_rhs") or "bf16",
            })
    return shapes


def _discover_presets():
    """Find all CSV files in data/ that can serve as presets."""
    presets = {}
    if not os.path.isdir(DATA_DIR):
        return presets
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".csv") and fname != "results.csv":
            presets[fname.removesuffix(".csv")] = os.path.join(DATA_DIR, fname)
    return presets


def _parse_size(s: str) -> int:
    """Parse a human-friendly byte count: 4G, 500M, 1024."""
    s = s.strip().upper()
    mult = 1
    if s.endswith("K"):
        mult, s = 1024, s[:-1]
    elif s.endswith("M"):
        mult, s = 1024 ** 2, s[:-1]
    elif s.endswith("G"):
        mult, s = 1024 ** 3, s[:-1]
    return int(float(s) * mult)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    presets = _discover_presets()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=sorted(presets.keys()) if presets else None,
                     help="Use a preset CSV from data/ directory.")
    src.add_argument("--csv", help="Load shapes from a CSV file "
                     "(columns: name,M,N,K,transpose,batch,dtype_lhs,dtype_rhs).")

    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--iree-compile", default="iree-compile")
    parser.add_argument("--iree-benchmark", default="iree-benchmark-module")
    parser.add_argument("--a-flags", default="",
                        help="Extra iree-compile flags for config A (baseline).")
    parser.add_argument("--b-flags", default="",
                        help="Extra iree-compile flags for config B (experiment).")
    parser.add_argument("--a-label", default="A", help="Label for config A.")
    parser.add_argument("--b-label", default="B", help="Label for config B.")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--work-dir",
                        default=os.path.join(_SCRIPT_DIR, "_work", "ab"))
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip", default="",
                        help="Comma-separated shape indices to skip (1-based).")
    parser.add_argument("--only", default="",
                        help="Comma-separated shape indices to run (1-based).")
    parser.add_argument("--skip-n1", action="store_true",
                        help="Skip shapes where N=1 (often degenerate).")
    parser.add_argument("--max-output-bytes", default=None,
                        help="Skip shapes whose output exceeds this size "
                             "(suffixes: K/M/G allowed, e.g. 4G).")
    parser.add_argument("--out-csv", default=None,
                        help="Write per-shape results to this CSV.")
    args = parser.parse_args()

    max_out = _parse_size(args.max_output_bytes) if args.max_output_bytes else None

    if args.preset:
        shapes = _load_csv(presets[args.preset])
    else:
        shapes = _load_csv(args.csv)

    skip_set = set(args.skip.split(",")) if args.skip else set()
    only_set = set(args.only.split(",")) if args.only else set()

    a_flags = args.a_flags.split() if args.a_flags else []
    b_flags = args.b_flags.split() if args.b_flags else []
    base_flags = [
        "--iree-dispatch-creation-enable-aggressive-fusion=true",
        "--iree-codegen-llvmgpu-use-vector-distribution",
    ]

    mlir_dir = os.path.join(args.work_dir, "mlir")
    a_dir = os.path.join(args.work_dir, "vmfb_a")
    b_dir = os.path.join(args.work_dir, "vmfb_b")
    os.makedirs(mlir_dir, exist_ok=True)
    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    results = []

    for i, s in enumerate(shapes):
        num = f"{i+1:02d}"
        name = s["name"]
        M, N, K = s["M"], s["N"], s["K"]
        trans = s["transpose"]
        batch = s.get("batch")
        lt = s.get("dtype_lhs", "bf16")
        rt = s.get("dtype_rhs", "bf16")

        if num in skip_set:
            print(f"SKIP {name}")
            results.append({"name": name, "M": M, "N": N, "K": K,
                            "transpose": trans, "a_us": "", "b_us": "",
                            "ratio": "", "status": "SKIP"})
            continue
        if only_set and num not in only_set:
            continue

        if args.skip_n1 and N == 1:
            print(f"SKIP {name} (N=1)")
            results.append({"name": name, "M": M, "N": N, "K": K,
                            "transpose": trans, "a_us": "", "b_us": "",
                            "ratio": "", "status": "SKIP_N1"})
            continue

        if max_out is not None:
            out_bytes = (batch or 1) * M * N * 4
            if out_bytes > max_out:
                print(f"SKIP {name} (output {out_bytes/1e9:.1f} GB > "
                      f"{max_out/1e9:.1f} GB)")
                results.append({"name": name, "M": M, "N": N, "K": K,
                                "transpose": trans, "a_us": "", "b_us": "",
                                "ratio": "", "status": "SKIP_LARGE"})
                continue

        mlir_path = os.path.join(mlir_dir, f"{name}.mlir")
        a_vmfb = os.path.join(a_dir, f"{name}.vmfb")
        b_vmfb = os.path.join(b_dir, f"{name}.vmfb")

        with open(mlir_path, "w") as f:
            f.write(gen_matmul_mlir(name, M, N, K, transpose=trans,
                                    dtype_lhs=lt, dtype_rhs=rt, batch=batch))

        if not args.skip_compile:
            print(f"Compiling {name}...", end=" ", flush=True)
            ok_a, err_a = compile_mlir(
                mlir_path, a_vmfb, target=args.target,
                extra_flags=base_flags + a_flags,
                iree_compile=args.iree_compile,
            )
            ok_b, err_b = compile_mlir(
                mlir_path, b_vmfb, target=args.target,
                extra_flags=base_flags + b_flags,
                iree_compile=args.iree_compile,
            )
            if not ok_a or not ok_b:
                fails = []
                if not ok_a: fails.append(f"{args.a_label}: {err_a[:100]}")
                if not ok_b: fails.append(f"{args.b_label}: {err_b[:100]}")
                print(f"COMPILE_FAIL ({'; '.join(fails)})")
                results.append({"name": name, "M": M, "N": N, "K": K,
                                "transpose": trans, "a_us": "", "b_us": "",
                                "ratio": "", "status": "COMPILE_FAIL"})
                continue
            print("OK", flush=True)

        if not os.path.exists(a_vmfb) or not os.path.exists(b_vmfb):
            results.append({"name": name, "M": M, "N": N, "K": K,
                            "transpose": trans, "a_us": "", "b_us": "",
                            "ratio": "", "status": "NO_VMFB"})
            continue

        if filecmp.cmp(a_vmfb, b_vmfb, shallow=False):
            print(f"  {name}: identical VMFBs, skipping benchmark")
            results.append({"name": name, "M": M, "N": N, "K": K,
                            "transpose": trans, "a_us": "", "b_us": "",
                            "ratio": "1.00", "status": "SAME"})
            continue

        lhs_spec, rhs_spec = get_input_specs(M, N, K, transpose=trans,
                                              dtype_lhs=lt, dtype_rhs=rt,
                                              batch=batch)
        print(f"Benchmarking {name}...", end=" ", flush=True)

        a_t = run_benchmark(
            a_vmfb, name, lhs_spec, rhs_spec,
            repetitions=args.reps, warmup=args.warmup,
            iree_benchmark=args.iree_benchmark,
        )
        b_t = run_benchmark(
            b_vmfb, name, lhs_spec, rhs_spec,
            repetitions=args.reps, warmup=args.warmup,
            iree_benchmark=args.iree_benchmark,
        )

        if a_t and b_t:
            ratio = b_t / a_t
            print(f"{args.a_label}={a_t:.1f}us  {args.b_label}={b_t:.1f}us  "
                  f"{ratio:.2f}x")
            results.append({"name": name, "M": M, "N": N, "K": K,
                            "transpose": trans,
                            "a_us": f"{a_t:.1f}", "b_us": f"{b_t:.1f}",
                            "ratio": f"{ratio:.3f}", "status": "OK"})
        else:
            print(f"BENCH_FAIL ({args.a_label}={a_t}, {args.b_label}={b_t})")
            results.append({"name": name, "M": M, "N": N, "K": K,
                            "transpose": trans,
                            "a_us": f"{a_t:.1f}" if a_t else "",
                            "b_us": f"{b_t:.1f}" if b_t else "",
                            "ratio": "", "status": "BENCH_FAIL"})

    al, bl = args.a_label, args.b_label
    print(f"\n{'='*100}")
    print(f"{'#':<4} {'Shape':<44} {al:>10} {bl:>10} "
          f"{f'{bl}/{al}':>9} {'Status'}")
    print(f"{'='*100}")
    for i, r in enumerate(results):
        num = f"{i+1:02d}"
        if r["status"] == "SAME":
            print(f"{num:<4} {r['name']:<44} {'-':>10} {'-':>10} "
                  f"{'1.00x':>9} SAME")
        elif r["status"] == "OK":
            print(f"{num:<4} {r['name']:<44} "
                  f"{r['a_us']:>10} {r['b_us']:>10} "
                  f"{r['ratio']+'x':>9}")
        else:
            print(f"{num:<4} {r['name']:<44} "
                  f"{r['a_us']:>10} {r['b_us']:>10} "
                  f"{'':>9} {r['status']}")
    print(f"{'='*100}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)),
                    exist_ok=True)
        fieldnames = ["name", "M", "N", "K", "transpose",
                      "a_us", "b_us", "ratio", "status"]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\nResults written to {args.out_csv}")


if __name__ == "__main__":
    main()
