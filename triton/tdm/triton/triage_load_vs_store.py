"""
Triage: is TDM async_load or async_store causing the AM crash?

Run as: python3 triage_load_vs_store.py load|store|both
"""
import sys
import time
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def tdm_load_only(a_ptr, shape, strides,
                  BLOCK_SHAPE, SHARED_LAYOUT: ttgl.constexpr):
    """TDM load only — result stays in LDS, no TDM store."""
    ndim: ttgl.constexpr = len(BLOCK_SHAPE)
    desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr, shape=shape, strides=strides,
        block_shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)
    offs = (0,) * ndim
    smem = ttgl.allocate_shared_memory(
        desc.dtype, shape=desc.block_shape, layout=desc.layout)
    ttgl.amd.gfx1250.tdm.async_load(desc, offs, smem)
    ttgl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def tdm_store_only(out_ptr, out_shape, out_strides,
                   BLOCK_SHAPE, SHARED_LAYOUT: ttgl.constexpr):
    """TDM store only — allocate LDS and store zeros via TDM."""
    ndim: ttgl.constexpr = len(BLOCK_SHAPE)
    smem = ttgl.allocate_shared_memory(
        ttgl.float16, shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)
    out_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=out_ptr, shape=out_shape, strides=out_strides,
        block_shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)
    offs = (0,) * ndim
    ttgl.amd.gfx1250.tdm.async_store(out_desc, offs, smem)
    ttgl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def tdm_load_and_store(out_ptr, a_ptr, shape, strides,
                       BLOCK_SHAPE, out_shape, out_strides,
                       SHARED_LAYOUT: ttgl.constexpr):
    """TDM load then TDM store (the combination used in all crashing tests)."""
    ndim: ttgl.constexpr = len(BLOCK_SHAPE)
    desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr, shape=shape, strides=strides,
        block_shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)
    offs = (0,) * ndim
    smem = ttgl.allocate_shared_memory(
        desc.dtype, shape=desc.block_shape, layout=desc.layout)
    ttgl.amd.gfx1250.tdm.async_load(desc, offs, smem)
    ttgl.amd.gfx1250.tdm.async_wait(0)
    out_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=out_ptr, shape=out_shape, strides=out_strides,
        block_shape=BLOCK_SHAPE, layout=SHARED_LAYOUT)
    ttgl.amd.gfx1250.tdm.async_store(out_desc, offs, smem)
    ttgl.amd.gfx1250.tdm.async_wait(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    BLOCK_SHAPE = (2, 4, 8, 128)
    ndim = len(BLOCK_SHAPE)
    order = [ndim - 1 - i for i in range(ndim)]
    SHARED_LAYOUT = ttgl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=order)
    constexpr_block_shape = tuple(ttgl.constexpr(v) for v in BLOCK_SHAPE)

    t = torch.randn(*BLOCK_SHAPE, dtype=torch.float16, device="cuda")
    out = torch.empty(*BLOCK_SHAPE, dtype=torch.float16, device="cuda")

    print(f"\nTEST: {mode} (4D SwizzledSharedLayout, no OOB)")
    print(f"  tensor shape: {list(t.shape)}")
    print(f"  BLOCK_SHAPE:  {list(BLOCK_SHAPE)}")

    start = time.time()
    if mode == "load":
        tdm_load_only[(1,)](
            t, t.shape, t.stride(),
            constexpr_block_shape, SHARED_LAYOUT)
    elif mode == "store":
        tdm_store_only[(1,)](
            out, out.shape, out.stride(),
            constexpr_block_shape, SHARED_LAYOUT)
    elif mode == "both":
        tdm_load_and_store[(1,)](
            out, t, t.shape, t.stride(),
            constexpr_block_shape, out.shape, out.stride(),
            SHARED_LAYOUT)
    else:
        print(f"Usage: {sys.argv[0]} load|store|both")
        sys.exit(1)

    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  RESULT: PASSED ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
