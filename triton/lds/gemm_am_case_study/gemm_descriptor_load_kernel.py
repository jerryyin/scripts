#!/usr/bin/env python3
"""
Descriptor-load GEMM kernel for LDS bank conflict measurement on gfx1250.

Based on third_party/amd/python/examples/f16_gemm.py from the Triton repo.
Uses device-side tl.make_tensor_descriptor for loads (descriptor loads go
through tensor_load_to_lds). Output uses standard tl.store (global_store)
to avoid tensor_store_from_lds which crashes the AM simulator.

The Triton compiler controls all LDS layouts (padding / swizzle) automatically.

Supports fp16 and fp8 (e4m3fn) inputs with fp32 accumulator.

Usage:
    python gemm_descriptor_load_kernel.py                       # fp16 default
    python gemm_descriptor_load_kernel.py --dtype fp8
    python gemm_descriptor_load_kernel.py -M 256 -N 256 -K 2048
"""
import hip

hip.hip.hipInit(0)

import torch
import argparse

import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc = tl.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * K,
        shape=(M, K), strides=(K, 1),
        block_shape=(BLOCK_M, BLOCK_K),
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N,
        shape=(K, N), strides=(N, 1),
        block_shape=(BLOCK_K, BLOCK_N),
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([0, k])
        b = b_desc.load([k, 0])
        accumulator = tl.dot(a, b, acc=accumulator)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)


def main():
    parser = argparse.ArgumentParser(
        description="Descriptor-load GEMM kernel for LDS bank conflict measurement"
    )
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-N", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("--dtype", choices=["fp16", "fp8"], default="fp16")
    parser.add_argument("--block_m", type=int, default=128)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = args.block_m, args.block_n, args.block_k

    torch.manual_seed(42)

    if args.dtype == "fp8":
        a = torch.randint(0x08, 0x77, (M, K), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        b = torch.randint(0x08, 0x77, (K, N), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    else:
        a = torch.randn((M, K), dtype=torch.float16)
        b = torch.randn((K, N), dtype=torch.float16)

    print(f"GEMM: M={M}, N={N}, K={K}, dtype={args.dtype}")
    print(f"Tiles: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print(f"Config: num_warps={args.num_warps}")

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = torch.empty((M, N), dtype=torch.float32).cuda()

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    gemm_kernel[grid](
        a_device, b_device, c_device,
        M, N, K,
        c_device.stride(0), c_device.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=args.num_warps,
    )

    c_triton = c_device.cpu()
    c_ref = a.float() @ b.float()

    if args.dtype == "fp8":
        atol, rtol = 0.125, 1e-2
    else:
        atol, rtol = 5e-2, 1e-2

    try:
        torch.testing.assert_close(c_triton, c_ref, rtol=rtol, atol=atol)
        print(f"Correctness: PASS (atol={atol})")
    except AssertionError as e:
        max_diff = (c_triton - c_ref).abs().max().item()
        print(f"Correctness: FAIL (max_diff={max_diff}, atol={atol})")

    print(f"output shape={c_triton.shape}, dtype={c_triton.dtype}")
    print(f"output[:3,:3]=\n{c_triton[:3,:3]}")


if __name__ == "__main__":
    main()
