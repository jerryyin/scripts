#!/usr/bin/env python3
"""Generate MLIR for matmul / batch-matmul with arbitrary transpose and dtype.

Usable as a standalone CLI or as an importable library.

CLI examples:
    python3 gen_matmul.py -m 1024 -n 2048 -k 512
    python3 gen_matmul.py -m 1024 -n 2048 -k 512 --transpose LHS_T --dtype bf16
    python3 gen_matmul.py -m 64 -n 64 -k 64 --batch 8 -o bmm.mlir
    python3 gen_matmul.py -m 1024 -n 2048 -k 512 --func-name my_mm --stdout

Library usage:
    from gen_matmul import gen_matmul_mlir, get_input_specs
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def gen_matmul_mlir(
    func_name: str,
    M: int, N: int, K: int,
    transpose: str = "none",
    dtype_lhs: str = "bf16",
    dtype_rhs: str = "bf16",
    dtype_acc: str = "f32",
    batch: int | None = None,
) -> str:
    """Generate MLIR for a matmul (or batch matmul) with given parameters.

    Args:
        func_name: Name for the MLIR func.func.
        M, N, K: Matmul dimensions (result is MxN, reduction is K).
        transpose: "none", "LHS_T", "RHS_T".
            LHS_T: LHS is KxM (transposed), RHS is KxN.
            RHS_T: LHS is MxK, RHS is NxK (transposed).
            none: LHS is MxK, RHS is KxN.
        dtype_lhs, dtype_rhs: Element types for inputs.
        dtype_acc: Accumulator / output type.
        batch: If set, generates a batch matmul with this batch size.

    Returns:
        MLIR source string.
    """
    if batch is not None:
        return _gen_bmm_mlir(func_name, batch, M, N, K, transpose,
                             dtype_lhs, dtype_rhs, dtype_acc)

    if transpose == "LHS_T":
        lhs_shape, rhs_shape = f"{K}x{M}", f"{K}x{N}"
        lhs_map = "affine_map<(d0, d1, d2) -> (d2, d0)>"
        rhs_map = "affine_map<(d0, d1, d2) -> (d2, d1)>"
    elif transpose == "RHS_T":
        lhs_shape, rhs_shape = f"{M}x{K}", f"{N}x{K}"
        lhs_map = "affine_map<(d0, d1, d2) -> (d0, d2)>"
        rhs_map = "affine_map<(d0, d1, d2) -> (d1, d2)>"
    else:
        lhs_shape, rhs_shape = f"{M}x{K}", f"{K}x{N}"
        if dtype_lhs == "bf16" and dtype_rhs == "bf16" and dtype_acc == "f32":
            return _gen_named_matmul(func_name, lhs_shape, rhs_shape,
                                     f"{M}x{N}", dtype_lhs, dtype_rhs,
                                     dtype_acc)
        lhs_map = "affine_map<(d0, d1, d2) -> (d0, d2)>"
        rhs_map = "affine_map<(d0, d1, d2) -> (d2, d1)>"

    out_map = "affine_map<(d0, d1, d2) -> (d0, d1)>"
    return _gen_generic_matmul(
        func_name,
        lhs_shape, rhs_shape, f"{M}x{N}",
        [lhs_map, rhs_map, out_map],
        ["parallel", "parallel", "reduction"],
        dtype_lhs, dtype_rhs, dtype_acc,
    )


def _gen_bmm_mlir(func_name, B, M, N, K, transpose, lt, rt, acc):
    if transpose == "LHS_T":
        lhs_shape, rhs_shape = f"{B}x{K}x{M}", f"{B}x{K}x{N}"
        lhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>"
        rhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>"
    elif transpose == "RHS_T":
        lhs_shape, rhs_shape = f"{B}x{M}x{K}", f"{B}x{N}x{K}"
        lhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>"
        rhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>"
    else:
        lhs_shape, rhs_shape = f"{B}x{M}x{K}", f"{B}x{K}x{N}"
        lhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>"
        rhs_map = "affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>"

    out_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>"
    return _gen_generic_matmul(
        func_name,
        lhs_shape, rhs_shape, f"{B}x{M}x{N}",
        [lhs_map, rhs_map, out_map],
        ["parallel", "parallel", "parallel", "reduction"],
        lt, rt, acc,
    )


def _gen_named_matmul(func_name, lhs_s, rhs_s, out_s, lt, rt, acc):
    """linalg.matmul for standard bf16->f32 case."""
    return (
        f"!LHS = tensor<{lhs_s}x{lt}>\n"
        f"!RHS = tensor<{rhs_s}x{rt}>\n"
        f"!RES = tensor<{out_s}x{acc}>\n"
        f"func.func @{func_name}(%lhs : !LHS, %rhs : !RHS) -> !RES {{\n"
        f"    %c0 = arith.constant 0.0 : {acc}\n"
        f"    %empty = tensor.empty() : !RES\n"
        f"    %fill = linalg.fill ins(%c0 : {acc}) outs(%empty : !RES) -> !RES\n"
        f"    %mm = linalg.matmul ins(%lhs, %rhs : !LHS, !RHS) "
        f"outs(%fill : !RES) -> !RES\n"
        f"    return %mm : !RES\n"
        f"}}\n"
    )


def _gen_generic_matmul(func_name, lhs_s, rhs_s, out_s, maps, iters,
                        lt, rt, acc):
    """linalg.generic for transposed / mixed-type matmuls."""
    ext_lines = []
    av, bv = "%a", "%b"
    if lt != acc:
        ext_lines.append(f"        %ae = arith.extf %a : {lt} to {acc}")
        av = "%ae"
    if rt != acc:
        ext_lines.append(f"        %be = arith.extf %b : {rt} to {acc}")
        bv = "%be"
    ext_block = "\n".join(ext_lines)
    if ext_block:
        ext_block += "\n"

    maps_str = ", ".join(maps)
    iters_str = ", ".join(f'"{i}"' for i in iters)
    return (
        f"!LHS = tensor<{lhs_s}x{lt}>\n"
        f"!RHS = tensor<{rhs_s}x{rt}>\n"
        f"!RES = tensor<{out_s}x{acc}>\n"
        f"func.func @{func_name}(%lhs : !LHS, %rhs : !RHS) -> !RES {{\n"
        f"    %c0 = arith.constant 0.0 : {acc}\n"
        f"    %empty = tensor.empty() : !RES\n"
        f"    %fill = linalg.fill ins(%c0 : {acc}) outs(%empty : !RES) -> !RES\n"
        f"    %mm = linalg.generic {{\n"
        f"        indexing_maps = [{maps_str}],\n"
        f"        iterator_types = [{iters_str}]\n"
        f"    }} ins(%lhs, %rhs : !LHS, !RHS) outs(%fill : !RES) {{\n"
        f"    ^bb0(%a: {lt}, %b: {rt}, %out: {acc}):\n"
        f"{ext_block}"
        f"        %mul = arith.mulf {av}, {bv} : {acc}\n"
        f"        %add = arith.addf %out, %mul : {acc}\n"
        f"        linalg.yield %add : {acc}\n"
        f"    }} -> !RES\n"
        f"    return %mm : !RES\n"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# Input shape computation
# ---------------------------------------------------------------------------

def get_input_specs(
    M: int, N: int, K: int,
    transpose: str = "none",
    dtype_lhs: str = "bf16",
    dtype_rhs: str = "bf16",
    batch: int | None = None,
) -> tuple[str, str]:
    """Return (lhs_spec, rhs_spec) for iree-benchmark-module --input flags.

    Specs include the shape and element type, e.g. "1285x2048xbf16".
    """
    if batch is not None:
        B = batch
        if transpose == "LHS_T":
            return f"{B}x{K}x{M}x{dtype_lhs}", f"{B}x{K}x{N}x{dtype_rhs}"
        elif transpose == "RHS_T":
            return f"{B}x{M}x{K}x{dtype_lhs}", f"{B}x{N}x{K}x{dtype_rhs}"
        else:
            return f"{B}x{M}x{K}x{dtype_lhs}", f"{B}x{K}x{N}x{dtype_rhs}"

    if transpose == "LHS_T":
        return f"{K}x{M}x{dtype_lhs}", f"{K}x{N}x{dtype_rhs}"
    elif transpose == "RHS_T":
        return f"{M}x{K}x{dtype_lhs}", f"{N}x{K}x{dtype_rhs}"
    else:
        return f"{M}x{K}x{dtype_lhs}", f"{K}x{N}x{dtype_rhs}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(
        description="Generate MLIR for matmul / batch-matmul.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--M", type=int, required=True)
    parser.add_argument("-n", "--N", type=int, required=True)
    parser.add_argument("-k", "--K", type=int, required=True)
    parser.add_argument("--transpose", default="none",
                        choices=["none", "LHS_T", "RHS_T"])
    parser.add_argument("--dtype", default="bf16",
                        help="Element type for both LHS and RHS (default: bf16)")
    parser.add_argument("--dtype-lhs", default=None)
    parser.add_argument("--dtype-rhs", default=None)
    parser.add_argument("--dtype-acc", default=None,
                        help="Accumulator type (default: f32 for bf16/f16, "
                             "same as dtype otherwise)")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--func-name", default="matmul")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--stdout", action="store_true",
                        help="Print to stdout even if -o is given")
    args = parser.parse_args()

    dt_lhs = args.dtype_lhs or args.dtype
    dt_rhs = args.dtype_rhs or args.dtype
    if args.dtype_acc:
        dt_acc = args.dtype_acc
    else:
        dt_acc = "f32" if dt_lhs in ("bf16", "f16") else dt_lhs

    mlir = gen_matmul_mlir(
        args.func_name, args.M, args.N, args.K,
        transpose=args.transpose,
        dtype_lhs=dt_lhs, dtype_rhs=dt_rhs, dtype_acc=dt_acc,
        batch=args.batch,
    )

    if args.output and not args.stdout:
        with open(args.output, "w") as f:
            f.write(mlir)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(mlir)


if __name__ == "__main__":
    _main()
