"""
Run on FFM to determine what the HW does with OOB elements.
For each test, inspect the output at OOB positions to distinguish:
  - Zero-fill: OOB positions are 0
  - Clamp/skip: OOB positions are uninitialized (whatever was in out tensor)
  - Read-through: OOB positions contain data from the underlying allocation
"""
import torch
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

    # Fill output with a sentinel value (42.0) so we can detect untouched positions
    out = torch.full(BLOCK_SHAPE, 42.0, dtype=inp.dtype, device=inp.device)

    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"  inp shape:    {list(inp.shape)}")
    print(f"  inp strides:  {list(inp.stride())}")
    print(f"  BLOCK_SHAPE:  {list(BLOCK_SHAPE)}")

    tdm_load_store_kernel[(1,)](
        out, inp, inp.shape, inp.stride(),
        constexpr_block_shape, out.shape, out.stride(),
        SHARED_LAYOUT)
    torch.cuda.synchronize()

    out_cpu = out.cpu()
    inp_cpu = inp.cpu()

    # Check in-bounds region: should match input
    idx = tuple(slice(None, s) for s in inp.shape)
    inbounds = out_cpu[idx]
    expected = inp_cpu
    inbounds_match = torch.equal(inbounds, expected)
    print(f"  In-bounds match input: {inbounds_match}")

    # Check OOB region: what values are there?
    # Zero out the in-bounds part, then look at what's left
    oob_check = out_cpu.clone()
    oob_check[idx] = 0.0
    oob_all_zero = (oob_check == 0).all().item()
    oob_all_sentinel = (oob_check == 42.0).all().item()
    oob_has_sentinel = (oob_check == 42.0).any().item()
    oob_has_nonzero = (oob_check != 0).any().item()

    print(f"  OOB all zeros:       {oob_all_zero}")
    print(f"  OOB all sentinel(42):{oob_all_sentinel}")
    print(f"  OOB has sentinel(42):{oob_has_sentinel}")
    print(f"  OOB has non-zero:    {oob_has_nonzero}")

    if oob_all_zero:
        print(f"  >> VERDICT: HW zero-fills OOB elements")
    elif oob_all_sentinel:
        print(f"  >> VERDICT: HW did NOT write OOB positions (sentinel untouched)")
    elif oob_has_sentinel:
        print(f"  >> VERDICT: MIXED — some OOB untouched, some overwritten")
    else:
        print(f"  >> VERDICT: OOB contains non-zero, non-sentinel data")

    # Show a few OOB values for inspection
    # Innermost dim OOB (if tile > tensor in innermost)
    if BLOCK_SHAPE[-1] > inp.shape[-1]:
        inner_oob = out_cpu[..., inp.shape[-1]:]
        vals = inner_oob.flatten()[:10]
        print(f"  Innermost OOB sample (dim3, positions {inp.shape[-1]}+): {vals.tolist()}")

    # dim2 OOB (if tile > tensor in dim2)
    if len(BLOCK_SHAPE) >= 3 and BLOCK_SHAPE[-2] > inp.shape[-2]:
        dim2_oob = out_cpu[..., inp.shape[-2]:, :]
        vals = dim2_oob.flatten()[:10]
        print(f"  dim2 OOB sample (positions {inp.shape[-2]}+): {vals.tolist()}")

    print(f"{'='*70}")


def main():
    BLOCK_SHAPE = (2, 4, 8, 128)

    # ── Test 1: Only dim2 OOB (tile=8, tensor=7), innermost exact ──
    t1 = torch.randn(1, 3, 7, 128, dtype=torch.float16, device="cuda")
    run_test("dim2 OOB only (tile=8 vs tensor=7), innermost exact", t1, BLOCK_SHAPE)

    # ── Test 2: Only dim3 OOB (tile=128, tensor=125), dim2 exact ──
    t2_alloc = torch.randn(1, 3, 8, 128, dtype=torch.float16, device="cuda")
    t2 = t2_alloc[..., :125]
    run_test("dim3 OOB only (tile=128 vs tensor=125), dim2 exact", t2, BLOCK_SHAPE)

    # ── Test 3: Both dim2 and dim3 OOB (original failing case) ──
    t3_alloc = torch.randn(1, 3, 7, 128, dtype=torch.float16, device="cuda")
    t3 = t3_alloc[..., :125]
    run_test("both dim2 OOB + dim3 OOB (original failing case)", t3, BLOCK_SHAPE)

    # ── Test 4: No OOB at all ──
    t4 = torch.randn(2, 4, 8, 128, dtype=torch.float16, device="cuda")
    run_test("no OOB (tensor matches tile exactly)", t4, BLOCK_SHAPE)


if __name__ == "__main__":
    main()
