#!/usr/bin/env python3
"""
Usage:
  python bmm_to_hipblaslt.py aten::bmm "<shapes>" "<dtypes>" "<strides>" "<misc>"

Example:
  python bmm_to_hipblaslt.py aten::bmm "[[16, 192, 384], [16, 384, 384]]" "['c10::BFloat16', 'c10::BFloat16']" "[[73728, 1, 192], [147456, 1, 384]]" "['', '']"
"""
import sys
import ast

def dtype_map(s: str) -> str:
    s = str(s).lower()
    if "bfloat" in s or "bf16" in s:
        return "bf16_r"
    if "float" in s or "fp32" in s:
        return "f32_r"
    if "half" in s or "f16" in s:
        return "f16_r"
    return "f32_r"

def aten_args_to_hipblaslt(shapes, dtypes, strides,
                           alpha=1.0, beta=0.0,
                           compute_type="f32_r",
                           scale_type="f32_r",
                           bias_type="f32_r"):
    # shapes: [[B, N, K], [B, K, M]]
    A_shape, B_shape = shapes
    sa, sb = strides
    Bcnt, N, K = A_shape
    Bcnt2, K2, M = B_shape
    if Bcnt != Bcnt2 or K != K2:
        raise ValueError("Mismatched batch or K dims between A and B")

    batch_count = int(Bcnt)
    m, n, k = int(M), int(N), int(K)

    lda = int(sb[2])
    ldb = int(sa[2])
    ldc = int(sb[2])
    ldd = ldc

    stride_a = int(sb[0]) if sb[0] != 0 else k * m
    stride_b = int(sa[0]) if sa[0] != 0 else k * n
    stride_c = m * n
    stride_d = stride_c

    a_type = dtype_map(dtypes[1]) if len(dtypes) > 1 else dtype_map(dtypes[0])
    b_type = dtype_map(dtypes[0])
    c_type = a_type
    d_type = a_type

    parts = [
        "hipblaslt-bench --api_method c",
        f"-m {m} -n {n} -k {k}",
        f"--lda {lda} --ldb {ldb} --ldc {ldc} --ldd {ldd}",
        f"--stride_a {stride_a} --stride_b {stride_b} --stride_c {stride_c} --stride_d {stride_d}",
        f"--alpha {alpha:.6f} --beta {beta:.6f}",
        "--transA T --transB T",
        f"--batch_count {batch_count}",
        f"--a_type {a_type} --b_type {b_type} --c_type {c_type} --d_type {d_type}",
        f"--scale_type {scale_type} --bias_type {bias_type} --compute_type {compute_type}"
    ]
    return " ".join(parts)

def main(argv):
    # Expect exactly: script aten::bmm "<shapes>" "<dtypes>" "<strides>" "<misc>"
    if len(argv) < 6:
        print("Usage: python aten_to_hipblaslt.py aten::bmm \"<shapes>\" \"<dtypes>\" \"<strides>\" \"<misc>\"", file=sys.stderr)
        sys.exit(2)

    op = argv[1]
    if op != "aten::bmm":
        print("First arg must be 'aten::bmm'", file=sys.stderr)
        sys.exit(2)

    raw_shapes = argv[2]
    raw_dtypes = argv[3]
    raw_strides = argv[4]
    # raw_misc = argv[5]  # ignored for conversion

    try:
        shapes = ast.literal_eval(raw_shapes)
        dtypes = ast.literal_eval(raw_dtypes)
        strides = ast.literal_eval(raw_strides)
    except Exception as e:
        print("Failed to parse one of the provided quoted arguments:", e, file=sys.stderr)
        sys.exit(3)

    try:
        hip_cmd = aten_args_to_hipblaslt(shapes, dtypes, strides)
    except Exception as e:
        print("Conversion error:", e, file=sys.stderr)
        sys.exit(4)

    print(hip_cmd)

if __name__ == "__main__":
    main(sys.argv)

