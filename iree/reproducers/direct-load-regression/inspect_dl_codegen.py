#!/usr/bin/env python3
"""Inspect direct-load codegen for a single matmul shape.

For one shape, this script:
  1. Generates MLIR.
  2. Compiles twice (with and without --iree-llvmgpu-use-direct-load),
     dumping the IR before/after the prefetch pass.
  3. Reports whether software pipelining activated, plus scf.if counts.
  4. Optionally extracts ISA and prints register/instruction stats
     (using ../../inspect_isa.py) for both compilations.

Usage:
    python3 inspect_dl_codegen.py 1134 2048 150000 --transpose LHS_T
    python3 inspect_dl_codegen.py 24576 2048 512 --transpose RHS_T --isa
    python3 inspect_dl_codegen.py 64 64 64 --batch 8 --dtype bf16
"""

import argparse
import os
import shutil
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.realpath(os.path.join(_SCRIPT_DIR, "..", "..")))

from gen_matmul import gen_matmul_mlir
from iree_bench import DEFAULT_TARGET
from inspect_isa import collect_stats, find_rocmasm, print_compare


WORK_DIR = os.path.join(_SCRIPT_DIR, "_work", "investigation")


def analyze_prefetch_ir(ir_dump: str) -> dict:
    """Parse the before/after sections of the prefetch IR dump."""
    after_idx = ir_dump.rfind("IR Dump After")
    before_idx = ir_dump.find("IR Dump Before")
    before_end = after_idx if after_idx != -1 else len(ir_dump)
    before = ir_dump[before_idx:before_end] if before_idx != -1 else ""
    after = ir_dump[after_idx:] if after_idx != -1 else ""
    return {
        "gather_before": before.count("amdgpu.gather_to_lds"),
        "gather_after": after.count("amdgpu.gather_to_lds"),
        "scf_if_before": before.count("scf.if"),
        "scf_if_after": after.count("scf.if"),
        "transfer_read_before": before.count("vector.transfer_read"),
        "transfer_read_after": after.count("vector.transfer_read"),
        "asyncmark_after": "rocdl.asyncmark" in after,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--transpose", default="none",
                        choices=["none", "LHS_T", "RHS_T"])
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--dtype", default="bf16",
                        help="Element type for both LHS and RHS.")
    parser.add_argument("--name", default=None,
                        help="Tag used for output filenames "
                             "(default: derived from shape).")
    parser.add_argument("--isa", action="store_true",
                        help="Extract ISA and print stats comparison.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--iree-compile", default="iree-compile")
    args = parser.parse_args()

    iree_compile = shutil.which(args.iree_compile) or args.iree_compile
    print(f"Using: {iree_compile}")

    name = args.name or (
        f"b{args.batch}_" if args.batch else ""
    ) + f"{args.M}x{args.N}x{args.K}_{args.transpose}".replace("none", "nn")
    print(f"\n{'='*60}")
    print(f"  {name}: M={args.M} N={args.N} K={args.K} "
          f"transpose={args.transpose} batch={args.batch} dtype={args.dtype}")
    print(f"{'='*60}")

    os.makedirs(WORK_DIR, exist_ok=True)
    mlir_path = os.path.join(WORK_DIR, f"{name}.mlir")
    with open(mlir_path, "w") as f:
        f.write(gen_matmul_mlir(
            "matmul", args.M, args.N, args.K,
            transpose=args.transpose,
            dtype_lhs=args.dtype, dtype_rhs=args.dtype,
            batch=args.batch,
        ))

    base_flags = (
        f"--iree-hal-target-backends=rocm "
        f"--iree-rocm-target={args.target} "
        f"--iree-hal-benchmark-dispatch-repeat-count=100"
    )

    isa_dirs = {}
    for mode in ["dl", "nodl"]:
        dl_flag = "--iree-llvmgpu-use-direct-load" if mode == "dl" else ""
        vmfb_path = os.path.join(WORK_DIR, f"{name}_{mode}.vmfb")
        ir_dump_path = os.path.join(WORK_DIR, f"{name}_{mode}_prefetch.log")

        cmd = (
            f"{iree_compile} {mlir_path} -o {vmfb_path} "
            f"{base_flags} {dl_flag} "
            f"--mlir-print-ir-before=iree-llvmgpu-prefetch-shared-memory "
            f"--mlir-print-ir-after=iree-llvmgpu-prefetch-shared-memory "
            f"2>{ir_dump_path}"
        )
        print(f"  [{mode.upper()}] Compiling...")
        ret = os.system(cmd)
        if ret != 0:
            print(f"  [{mode.upper()}] COMPILE FAILED (exit {ret})")
            continue

        with open(ir_dump_path) as f:
            info = analyze_prefetch_ir(f.read())

        if mode == "dl":
            gb, ga = info["gather_before"], info["gather_after"]
            pipelined = ga > gb
            status = (f"ACTIVE (gather: {gb}->{ga})" if pipelined
                      else f"NOT ACTIVE (gather: {gb}->{ga})")
        else:
            tb, ta = info["transfer_read_before"], info["transfer_read_after"]
            pipelined = ta > tb
            status = (f"ACTIVE (transfer_read: {tb}->{ta})" if pipelined
                      else f"NOT ACTIVE (transfer_read: {tb}->{ta})")

        print(
            f"  [{mode.upper()}] Pipelining {status}, "
            f"scf.if: {info['scf_if_before']}->{info['scf_if_after']}, "
            f"asyncmark={info['asyncmark_after']}"
        )

        if args.isa:
            isa_dir = os.path.join(WORK_DIR, f"{name}_{mode}_isa")
            isa_cmd = (
                f"{iree_compile} {mlir_path} -o /dev/null "
                f"{base_flags} {dl_flag} "
                f"--iree-hal-dump-executable-sources-to={isa_dir}"
            )
            print(f"  [{mode.upper()}] Extracting ISA...")
            os.system(isa_cmd)
            isa_dirs[mode] = isa_dir

    if args.isa and "dl" in isa_dirs and "nodl" in isa_dirs:
        asm_nodl = find_rocmasm(isa_dirs["nodl"])
        asm_dl = find_rocmasm(isa_dirs["dl"])
        if asm_nodl and asm_dl:
            print(f"\n{'-'*60}")
            print("  ISA stats")
            print(f"{'-'*60}")
            print(f"  [nodl] {asm_nodl}")
            print(f"  [dl]   {asm_dl}")
            print()
            print_compare(collect_stats(asm_nodl), collect_stats(asm_dl),
                          "nodl", "dl")
        else:
            print("\n  ISA: could not locate .rocmasm files")

    print(f"\nArtifacts saved to {WORK_DIR}/")


if __name__ == "__main__":
    main()
