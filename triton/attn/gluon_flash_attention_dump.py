#!/usr/bin/env python3
"""
Gluon Flash Attention IR Dump Tool (gfx1250).

Compile a single configuration and dump IR artifacts side-by-side
with flash_attention_tool.py outputs.

Usage:
  python gluon_flash_attention_dump.py --batch 8 --heads 8 --seqlen-q 512 --seqlen-k 512 \
      --head-dim 128 --block-m 128 --block-n 64 --pipeline

Environment Variables:
  TRITON_SAVETEMPS_DIR: Directory to save IR artifacts (default: ./ir_output)
"""

import argparse
import math
import os
import sys
from typing import Optional, Sequence


def resolve_output_dir(out_dir: Optional[str]) -> str:
    """
    Resolve output directory with consistent priority.

    Priority: explicit arg > TRITON_SAVETEMPS_DIR env var > ./ir_output
    """
    if out_dir:
        return os.path.abspath(out_dir)

    env_dir = os.environ.get("TRITON_SAVETEMPS_DIR", "").strip()
    if env_dir:
        return os.path.abspath(env_dir)

    return os.path.abspath(os.path.join(os.getcwd(), "ir_output"))


def contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


def config_suffix(args) -> str:
    if args.seqlen_q == args.seqlen_k:
        n_str = f"N{args.seqlen_q}"
    else:
        n_str = f"NQ{args.seqlen_q}_NK{args.seqlen_k}"
    pipe = "pipe" if args.pipeline else "nopipe"
    return (f"B{args.batch}_H{args.heads}_{n_str}_D{args.head_dim}_BM{args.block_m}_"
            f"BN{args.block_n}_{pipe}_{args.dtype}")


def compile_only_kernel(attn_fn, args):
    import torch
    from triton.runtime.jit import MockTensor

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    q_shape = (args.batch, args.heads, args.seqlen_q, args.head_dim)
    kv_shape = (args.batch, args.heads, args.seqlen_k, args.head_dim)

    q = MockTensor(dtype, q_shape)
    k = MockTensor(dtype, kv_shape)
    v = MockTensor(dtype, kv_shape)
    o = MockTensor(dtype, q_shape)

    stride_qz, stride_qh, stride_qm, stride_qk = contiguous_strides(q_shape)
    stride_kz, stride_kh, stride_kn, stride_kk = contiguous_strides(kv_shape)
    stride_vz, stride_vh, stride_vn, stride_vk = contiguous_strides(kv_shape)
    stride_oz, stride_oh, stride_om, stride_on = contiguous_strides(q_shape)

    sm_scale = 1.0 / math.sqrt(args.head_dim)
    grid = (args.batch, args.heads, (args.seqlen_q + args.block_m - 1) // args.block_m)

    return attn_fn.warmup(
        q, k, v, o,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_on,
        sm_scale, args.seqlen_q, args.seqlen_k,
        args.block_m, args.block_n,
        args.head_dim,
        num_warps=4,
        waves_per_eu=1,
        grid=grid,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump Gluon Flash Attention IR (gfx1250)")
    parser.add_argument("--batch", "-b", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--heads", "-H", type=int, default=8, help="Number of heads (default: 8)")
    parser.add_argument("--seqlen-q", type=int, default=512, help="Q sequence length (default: 512)")
    parser.add_argument("--seqlen-k", type=int, default=512, help="K/V sequence length (default: 512)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--block-m", type=int, default=128, help="BLOCK_M tile size (default: 128)")
    parser.add_argument("--block-n", type=int, default=64, help="BLOCK_N tile size (default: 64)")
    parser.add_argument("--pipeline", action="store_true", help="Use pipelined kernel")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                        help="Input/output dtype for compile-only (default: fp16)")
    parser.add_argument("--out-dir", "-o", type=str, default=None,
                        help="Output directory for IR (default: TRITON_SAVETEMPS_DIR or ./ir_output)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    sys.path.insert(0, "/root/triton-mi450/python")
    sys.path.insert(0, "/root/triton-mi450/third_party/amd/python/examples/gluon")

    import f16_fa_gfx1250 as fa

    out_dir = resolve_output_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    attn_fn = fa.attn_fwd_pipelined_kernel if args.pipeline else fa.attn_fwd_kernel
    try:
        kernel = compile_only_kernel(attn_fn, args)
    except Exception as exc:
        if args.dtype == "fp16":
            print("Compile-only failed with fp16. This kernel currently mixes bf16 and fp16")
            print("inside WMMA operands; use --dtype bf16 unless the kernel is updated.")
        raise

    prefix = f"gluon_attn_fwd_{config_suffix(args)}"
    ext_map = {"source": "mlir"}

    if not args.quiet:
        print("=" * 60)
        print("Gluon Flash Attention IR Dump")
        print(f"Config: {prefix}")
        print(f"Output: {out_dir}")
        print("=" * 60)

    for stage, content in kernel.asm.items():
        ext = ext_map.get(stage, stage)
        path = os.path.join(out_dir, f"{prefix}.{ext}")
        if isinstance(content, str):
            with open(path, "w") as f:
                f.write(content)
        elif isinstance(content, bytes):
            with open(path, "wb") as f:
                f.write(content)
        else:
            with open(path, "w") as f:
                f.write(str(content))

        if not args.quiet:
            size = os.path.getsize(path)
            print(f"  {stage}: {os.path.basename(path)} ({size:,} bytes)")

    if not args.quiet:
        print("=" * 60)
        print("Done!")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
