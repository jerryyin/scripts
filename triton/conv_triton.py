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
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    matrix_instr_nonkdim : tl.constexpr = 0
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 16
    IN_C = 768
    IN_H = 48
    IN_W = 32
    OUT_C = 2048
    OUT_H = 48
    OUT_W = 32

    # Strides:
    stride_xn = 1179648
    stride_xc = 1536
    stride_xh = 32
    stride_xw = 1
    stride_wc_out = 6912
    stride_wc_in = 9
    stride_wh = 3
    stride_ww = 1

    # Output of block
    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    # Per block pointer base for input and filter
    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # K loop: y * x * BLOCK_K_COUNT
    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

         # input_x = kernel_x + output_x - padding
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        # Gather the input for corresponding filter (along channel dimension)
        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight tile before MFMA
        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        # IGEMM
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 32*idx_h + 1536*idx_c + 3145728*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)


def get_args():
    arg_0 = rand_strided((16, 768, 48, 32), (1179648, 1536, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((2048, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((16, 2048, 48, 32), (3145728, 1536, 32, 1), device='cuda:0', dtype=torch.bfloat16)
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
