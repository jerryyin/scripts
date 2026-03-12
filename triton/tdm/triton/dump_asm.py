"""Dump the AMDGCN assembly for TDM load-only and store-only kernels."""
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def tdm_store_only(out_ptr, out_shape, out_strides,
                   BLOCK_SHAPE, SHARED_LAYOUT: ttgl.constexpr):
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
def tdm_load_only(a_ptr, shape, strides,
                  BLOCK_SHAPE, SHARED_LAYOUT: ttgl.constexpr):
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
def tdm_load_and_store(out_ptr, a_ptr, shape, strides,
                       BLOCK_SHAPE, out_shape, out_strides,
                       SHARED_LAYOUT: ttgl.constexpr):
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
    BLOCK_SHAPE = (2, 4, 8, 128)
    ndim = 4
    order = [ndim - 1 - i for i in range(ndim)]
    SHARED_LAYOUT = ttgl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=order)
    constexpr_bs = tuple(ttgl.constexpr(v) for v in BLOCK_SHAPE)

    t = torch.randn(*BLOCK_SHAPE, dtype=torch.float16, device="cuda")
    out = torch.empty(*BLOCK_SHAPE, dtype=torch.float16, device="cuda")

    print("=" * 80)
    print("STORE-ONLY KERNEL AMDGCN")
    print("=" * 80)
    k_store = tdm_store_only[(1,)](
        out, out.shape, out.stride(), constexpr_bs, SHARED_LAYOUT)
    print(k_store.asm["amdgcn"])

    print()
    print("=" * 80)
    print("LOAD-ONLY KERNEL AMDGCN")
    print("=" * 80)
    k_load = tdm_load_only[(1,)](
        t, t.shape, t.stride(), constexpr_bs, SHARED_LAYOUT)
    print(k_load.asm["amdgcn"])

    print()
    print("=" * 80)
    print("LOAD+STORE KERNEL AMDGCN")
    print("=" * 80)
    k_both = tdm_load_and_store[(1,)](
        out, t, t.shape, t.stride(), constexpr_bs,
        out.shape, out.stride(), SHARED_LAYOUT)
    print(k_both.asm["amdgcn"])


if __name__ == "__main__":
    main()
