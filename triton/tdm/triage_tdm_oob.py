"""
Triage: which dimension mismatch triggers the AM assertion?

Test 1 — Only dim2 OOB:
  Physical tensor [1, 3, 7, 128], tile [1, 2, 8, 128]
  dim2: tile=8 > tensor=7   (OOB)
  dim3: tile=128 = tensor=128 (exact match)

Test 2 — Only dim3 (innermost) OOB:
  Physical tensor [1, 3, 8, 125] (sliced from [1, 3, 8, 128]), tile [1, 2, 8, 128]
  dim2: tile=8 = tensor=8   (exact match)
  dim3: tile=128 > tensor=125 (OOB)
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def tdm_load_store_kernel(out_ptr, a_ptr, shape, strides,
                          BLOCK_SHAPE, out_shape, out_strides,
                          SHARED_LAYOUT: ttgl.constexpr):
    ndim: ttgl.constexpr = len(BLOCK_SHAPE)
    desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr, shape=shape, strides=strides,
        block_shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)
    offs = (0,) * ndim
    block_shared = ttgl.allocate_shared_memory(
        desc.dtype, shape=desc.block_shape, layout=desc.layout)
    ttgl.amd.gfx1250.tdm.async_load(desc, offs, block_shared)
    ttgl.amd.gfx1250.tdm.async_wait(0)
    out_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=out_ptr, shape=out_shape, strides=out_strides,
        block_shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)
    ttgl.amd.gfx1250.tdm.async_store(out_desc, offs, block_shared)
    ttgl.amd.gfx1250.tdm.async_wait(0)


def run_test(name, inp, BLOCK_SHAPE):
    ndim = len(BLOCK_SHAPE)
    order = [ndim - 1 - i for i in range(ndim)]
    SHARED_LAYOUT = ttgl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=order)
    constexpr_block_shape = tuple(ttgl.constexpr(v) for v in BLOCK_SHAPE)

    out = torch.empty(BLOCK_SHAPE, dtype=inp.dtype, device=inp.device)

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  inp shape:    {list(inp.shape)}")
    print(f"  inp strides:  {list(inp.stride())}")
    print(f"  BLOCK_SHAPE:  {list(BLOCK_SHAPE)}")
    print(f"  out shape:    {list(out.shape)}")
    print(f"{'='*60}")

    try:
        tdm_load_store_kernel[(1,)](
            out, inp, inp.shape, inp.stride(),
            constexpr_block_shape, out.shape, out.stride(),
            SHARED_LAYOUT)
        torch.cuda.synchronize()
        print(f"  RESULT: PASSED")
        return True
    except Exception as e:
        print(f"  RESULT: FAILED — {e}")
        return False


def main():
    BLOCK_SHAPE = (2, 4, 8, 128)

    # ── Test 1: Only dim2 OOB (tile=8, tensor=7), innermost exact ──
    # Tensor [1, 3, 7, 128] — no slicing, innermost matches tile exactly
    t1 = torch.randn(1, 3, 7, 128, dtype=torch.float16, device="cuda")
    run_test("dim2 OOB only (tile=8 vs tensor=7), innermost exact",
             t1, BLOCK_SHAPE)

    # ── Test 2: Only dim3 OOB (tile=128, tensor=125), dim2 exact ──
    # Tensor [1, 3, 8, 128] sliced to [1, 3, 8, 125] — dim2 matches tile
    t2_alloc = torch.randn(1, 3, 8, 128, dtype=torch.float16, device="cuda")
    t2 = t2_alloc[..., :125]  # shape [1, 3, 8, 125], stride [3072, 1024, 128, 1]
    run_test("dim3 OOB only (tile=128 vs tensor=125), dim2 exact",
             t2, BLOCK_SHAPE)

    # ── Test 3 (original): Both dim2 and dim3 OOB ──
    t3_alloc = torch.randn(1, 3, 7, 128, dtype=torch.float16, device="cuda")
    t3 = t3_alloc[..., :125]  # shape [1, 3, 7, 125], stride [2688, 896, 128, 1]
    run_test("both dim2 OOB + dim3 OOB (original failing case)",
             t3, BLOCK_SHAPE)

    # ── Test 4 (baseline): No OOB at all ──
    t4 = torch.randn(2, 4, 8, 128, dtype=torch.float16, device="cuda")
    run_test("no OOB (tensor matches tile exactly)",
             t4, BLOCK_SHAPE)


if __name__ == "__main__":
    main()
