"""TDM gather kernel with explicit layout control for perf exploration."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def tdm_gather_kernel(inp_ptr, out_ptr, idx_ptr,
                      M_inp, N_inp, stride_m, stride_n,
                      BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                      NUM_INDICES: ttgl.constexpr, SRC_COL_OFFSET: ttgl.constexpr,
                      SHARED_LAYOUT: ttgl.constexpr, IDX_LAYOUT: ttgl.constexpr,
                      REG_LAYOUT: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(inp_ptr.type.element_ty, (BLOCK_M, BLOCK_N), SHARED_LAYOUT)

    inp_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=inp_ptr, shape=(M_inp, N_inp), strides=(stride_m, 1),
        block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)

    idx_offs = ttgl.arange(0, NUM_INDICES, layout=IDX_LAYOUT)
    src_row_indices = ttgl.load(idx_ptr + idx_offs)

    ttgl.amd.gfx1250.tdm.async_gather(inp_desc, src_row_indices, SRC_COL_OFFSET, smem)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    result = smem.load(REG_LAYOUT)
    offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, REG_LAYOUT))
    offs_n = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, REG_LAYOUT))
    ttgl.store(out_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], result)


def make_layout(variant, num_indices, num_warps):
    """Create a SliceLayout for the given variant name.

    Returns:
        ttgl.SliceLayout wrapping a 2D BlockedLayout.
    """
    ni = num_indices
    nw = num_warps

    if variant == "replicated":
        # All warps hold all ni indices (replicated). Only 1 warp fires;
        # the rest are predicated off via tile_dim1=0 in their descriptors.
        # warpsPerCTA = [1, nw]: warps spread on dim1 (the non-sliced dim),
        # so after slicing dim1 every warp sees the same ni indices.
        parent = ttgl.BlockedLayout([ni, 1], [1, 32], [1, nw], [1, 0])
    elif variant == "partitioned":
        # Indices evenly partitioned across warps. Each warp owns ni/nw
        # unique indices and issues its own gather instruction(s).
        # warpsPerCTA = [nw, 1]: warps spread on dim0 (the sliced dim).
        assert ni >= nw and ni % nw == 0, f"Cannot partition {ni} across {nw} warps"
        parent = ttgl.BlockedLayout([ni // nw, 1], [1, 32], [nw, 1], [1, 0])
    elif variant == "mixed_2x":
        # Half the warps are active, half are redundant copies.
        # warpsPerCTA = [active, 2]: active warps on dim0, 2x redundancy on dim1.
        active = max(1, nw // 2)
        redundant = nw // active
        assert ni % active == 0
        parent = ttgl.BlockedLayout([ni // active, 1], [1, 32], [active, redundant], [1, 0])
    elif variant == "redundant_reg":
        # sizePerThread > 1 in sliced dim => duplicate register entries per thread.
        # regMask filtering removes duplicates before issuing gather.
        parent = ttgl.BlockedLayout([ni, 2], [1, 32], [1, nw], [1, 0])
    elif variant == "greedy":
        # sizePerThread = max_per_instr (16 for i16). Greedy allocation:
        # activates only as many warps as needed to cover ni, so small ni
        # stalls the fewest warps without sacrificing wave_active_clocks at
        # large ni. When ni > 16*nw the encoding wraps and the lowering
        # unrolls multiple gathers per warp.
        parent = ttgl.BlockedLayout([16, 1], [1, 32], [nw, 1], [1, 0])
    elif variant == "greedy_i32":
        # Same as greedy but for i32 indices (max 8 per instruction).
        parent = ttgl.BlockedLayout([8, 1], [1, 32], [nw, 1], [1, 0])
    elif variant == "default":
        # Mimics standard Triton's default for tl.arange(0, N):
        # threads distribute along the index dim first, then warps.
        # threadsPerWarp[0] = min(32, ni), remaining threads on dim1.
        # warpsPerCTA[0] = ni / (spt * tpw0), remaining warps on dim1.
        tpw0 = min(32, ni)
        tpw1 = 32 // tpw0
        spt0 = 1
        wpc0 = ni // (spt0 * tpw0)
        if wpc0 < 1:
            wpc0 = 1
            spt0 = ni // tpw0
        wpc0 = min(wpc0, nw)
        # If we still don't cover ni, increase sizePerThread
        while spt0 * tpw0 * wpc0 < ni:
            spt0 += 1
        wpc1 = nw // wpc0
        parent = ttgl.BlockedLayout([spt0, 1], [tpw0, tpw1],
                                    [wpc0, wpc1], [1, 0])
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return ttgl.SliceLayout(1, parent)


def run_gather(variant, num_indices, num_warps, block_n,
               dtype=torch.float16, index_dtype=torch.int16,
               m_inp=2048, verify=True):
    """Run a single TDM gather and optionally verify correctness."""
    block_m = num_indices
    idx_layout = make_layout(variant, num_indices, num_warps)
    shared_layout = ttgl.PaddedSharedLayout.with_identity_for(
        [[block_n, 8]], [block_m, block_n], [1, 0])
    # sizePerThread * threadsPerWarp * warpsPerCTA must cover [block_m, block_n]
    # Use a standard blocked layout: each thread handles multiple columns
    elems_per_thread_n = block_n // 32  # 32 threads per warp in N dim
    elems_per_thread_m = block_m // num_warps if block_m >= num_warps else 1
    warps_m = min(num_warps, block_m)
    warps_n = num_warps // warps_m
    threads_n = 32 // (1 if warps_n == 1 else 1)
    reg_layout = ttgl.BlockedLayout([elems_per_thread_m, elems_per_thread_n],
                                    [1, 32], [warps_m, warps_n], [1, 0])

    inp = torch.randn((m_inp, block_n), dtype=dtype, device='cuda')
    out = torch.zeros((block_m, block_n), dtype=dtype, device='cuda')
    indices = torch.arange(num_indices, dtype=index_dtype, device='cuda')

    tdm_gather_kernel[(1,)](
        inp, out, indices,
        M_inp=m_inp, N_inp=block_n,
        stride_m=inp.stride(0), stride_n=inp.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, NUM_INDICES=num_indices,
        SRC_COL_OFFSET=0, SHARED_LAYOUT=shared_layout, IDX_LAYOUT=idx_layout,
        REG_LAYOUT=reg_layout, num_warps=num_warps)

    if verify:
        ref = inp[:num_indices, :block_n]
        torch.testing.assert_close(out, ref)
        print(f"  [OK] {variant} ni={num_indices} nw={num_warps} bn={block_n}")

    return out


if __name__ == "__main__":
    import sys
    variant = sys.argv[1] if len(sys.argv) > 1 else "replicated"
    ni = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    bn = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    print(f"Running gather: variant={variant} ni={ni} nw={nw} bn={bn}")
    run_gather(variant, ni, nw, bn)
