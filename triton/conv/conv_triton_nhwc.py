from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid



import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(
    num_stages=2,
    num_warps=8,
    triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=304, cc='gfx942', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'kernel_name': 'triton_tem_fused_convolution_0', 'backend_hash': '62BEA4D65C6FD70208CBEA9DEA8405AC6F99DA36705E29D15C086C229AFFE751', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.166723584},
)
@triton.jit
def triton_tem_fused_convolution_0(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    ALLOW_TF32 : tl.constexpr = True

    # Tensor sizes
    BATCH = 16
    IN_C = 768
    IN_H = 48
    IN_W = 32
    OUT_C = 2048
    OUT_H = 48
    OUT_W = 32

    # ---- Strides for NHWC input ----
    stride_xc = 1
    stride_xw = IN_C
    stride_xh = IN_W * IN_C
    stride_xn = IN_H * IN_W * IN_C

    # ---- Strides for OHWI weights ----
    stride_wo = KERNEL_H * KERNEL_W * IN_C
    stride_wh = KERNEL_W * IN_C
    stride_ww = IN_C
    stride_wi = 1

    # ---- Strides for NHWC output ----
    stride_yc = 1
    stride_yw = OUT_C
    stride_yh = OUT_W * OUT_C
    stride_yn = OUT_H * OUT_W * OUT_C

    # gemmM: (N * H * W)
    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n   = nh // OUT_H
    # gemmN: (O)
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K = (IN_C * KH * KW)
    BLOCK_K_COUNT = (IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k  = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i  = ij // KERNEL_W
        j  = ij % KERNEL_W

        # Input pixel coords
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        # Input pointer (NHWC)
        x_ptrs = arg_X + (
            idx_n[:, None] * stride_xn +
            idx_x_h[:, None] * stride_xh +
            idx_x_w[:, None] * stride_xw +
            idx_x_c[None, :] * stride_xc
        )
        mask_x = (
            (idx_n < BATCH)[:, None] &
            (idx_x_h >= 0)[:, None] & (idx_x_h < IN_H)[:, None] &
            (idx_x_w >= 0)[:, None] & (idx_x_w < IN_W)[:, None] &
            (idx_x_c < IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight pointer (OHWI)
        w_ptrs = arg_W + (
            idx_y_c[None, :] * stride_wo +
            i * stride_wh +
            j * stride_ww +
            idx_x_c[:, None] * stride_wi
        )
        mask_w = (idx_x_c[:, None] < IN_C) & (idx_y_c[None, :] < OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Accumulate
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)

    # ---- Store output (NHWC) ----
    mask_out = (
        (idx_n < BATCH)[:, None] &
        (idx_y_h < OUT_H)[:, None] &
        (idx_y_w < OUT_W)[:, None] &
        (idx_y_c < OUT_C)[None, :]
    )
    out_ptrs = out_ptr0 + (
        idx_n[:, None] * stride_yn +
        idx_y_h[:, None] * stride_yh +
        idx_y_w[:, None] * stride_yw +
        idx_y_c[None, :] * stride_yc
    )
    tl.store(out_ptrs, acc, mask=mask_out)


def get_args():
    # Input: NHWC
    arg_0 = rand_strided(
        (16, 48, 32, 768),
        (1179648, 24576, 768, 1),
        device='cuda:0',
        dtype=torch.bfloat16,
    )
    # Weights: OHWI
    arg_1 = rand_strided(
        (2048, 3, 3, 768),
        (6912, 2304, 768, 1),
        device='cuda:0',
        dtype=torch.bfloat16,
    )
    # Output: NHWC
    arg_2 = rand_strided(
        (16, 48, 32, 2048),
        (3145728, 65536, 2048, 1),
        device='cuda:0',
        dtype=torch.bfloat16,
    )
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_0.run(*args, grid=(192, 16, 1), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_tem_fused_convolution_0.benchmark_all_configs(*args, grid=(192, 16, 1))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.166723584
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")

