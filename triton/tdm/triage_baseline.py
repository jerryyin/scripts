"""Baseline: no OOB at all — tensor matches tile exactly."""
import time
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


BLOCK_SHAPE = (2, 4, 8, 128)
ndim = 4
order = [ndim - 1 - i for i in range(ndim)]
SHARED_LAYOUT = ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=order)
constexpr_block_shape = tuple(ttgl.constexpr(v) for v in BLOCK_SHAPE)

t = torch.randn(2, 4, 8, 128, dtype=torch.float16, device="cuda")
out = torch.empty(BLOCK_SHAPE, dtype=t.dtype, device=t.device)

start = time.time()
tdm_load_store_kernel[(1,)](
    out, t, t.shape, t.stride(),
    constexpr_block_shape, out.shape, out.stride(),
    SHARED_LAYOUT)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"PASSED in {elapsed:.1f}s")
