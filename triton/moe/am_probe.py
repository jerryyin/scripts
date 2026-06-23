"""Incremental AM capability probe: climb from a trivial CUDA op to the full
a8w4 gluon GEMM1 to find exactly which step deadlocks AM CModel startup.

Each rung prints START/OK markers (flushed) so a hang localizes to one step.
Run under AM:
    GPU_ARCHS=gfx1250 run_on_model.sh --backend am -- \
        python3 am_probe.py --rung 1
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# aiter is not pip-installed; make it importable (mirrors lib_moe_ffm.py).
_AITER_HOME = os.environ.get("AITER_HOME", "/root/aiter")
if _AITER_HOME not in sys.path:
    sys.path.insert(0, _AITER_HOME)


def log(msg: str):
    print(f"[probe t={time.monotonic():.1f}] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rung", type=int, required=True,
                   help="1=cuda op, 2=triton vadd, 3=gemm_a16w16, 4=routing, 5=gluon GEMM1")
    p.add_argument("--backend", choices=["gluon", "triton"], default="gluon")
    args = p.parse_args()
    os.environ["AITER_FORCE_TRITON"] = "1" if args.backend == "triton" else "0"

    log(f"rung={args.rung} backend={args.backend}: importing torch")
    import torch
    log("torch imported; touching cuda device")
    dev = "cuda:0"

    # Rung 1: simplest possible GPU dispatch (triggers HSA/DTIF CModel startup).
    log("rung1: tiny cuda elementwise op")
    a = torch.ones(8, device=dev)
    b = (a + a).sum().item()
    log(f"rung1 OK: sum={b}")
    if args.rung == 1:
        return 0

    # Rung 2: a trivial custom triton kernel (JIT + launch under AM).
    log("rung2: importing triton + compiling vadd")
    import triton
    import triton.language as tl

    @triton.jit
    def _vadd(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask) +
                 tl.load(y_ptr + offs, mask=mask), mask=mask)

    n = 256
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    o = torch.empty(n, device=dev)
    _vadd[(1,)](x, y, o, n, BLOCK=256)
    torch.cuda.synchronize()
    log(f"rung2 OK: max_abs_err={(o - (x + y)).abs().max().item():.3e}")
    if args.rung == 2:
        return 0

    # Rung 3: aiter's gate GEMM (a simple triton GEMM through aiter's stack).
    log("rung3: import aiter gemm_a16w16")
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    from aiter.ops.triton.utils._triton.arch_info import get_arch
    log(f"aiter imported; arch={get_arch()}")
    xx = torch.randn((128, 2048), dtype=torch.bfloat16, device=dev)
    ww = torch.randn((256, 2048), dtype=torch.bfloat16, device=dev)  # (N, K)
    bb = torch.randn((256,), dtype=torch.bfloat16, device=dev)
    logits = gemm_a16w16(xx, ww, bb)  # contracts over K=2048 -> (128, 256)
    torch.cuda.synchronize()
    log(f"rung3 OK: logits.shape={tuple(logits.shape)}")
    if args.rung == 3:
        return 0

    # Rung 4: aiter routing (builds gather/scatter + histogram).
    log("rung4: import + run routing")
    from aiter.ops.triton.moe.moe_routing.routing import routing
    rdata, gather_indx, scatter_indx = routing(logits, 8)
    torch.cuda.synchronize()
    log(f"rung4 OK: block_m={rdata.block_m} hist_sum={int(rdata.expt_hist.sum())}")
    if args.rung == 4:
        return 0

    # Rung 5: full a8w4 gluon GEMM1 (the actual target kernel).
    log("rung5: build inputs for a8w4 GEMM1")
    from lib_moe_ffm import Shape, build, _swizzle, _quant_act, STATIC_SCALE
    from aiter.ops.triton.moe import moe_op_gemm_a8w4 as a8w4
    shape = Shape(dim1=2048, dim2=7168, n_expts_tot=256, n_expts_act=8)
    d = build(shape, 128, seed=0)
    ss = torch.tensor(STATIC_SCALE, device=d["x"].device)
    w1s, sw1 = _swizzle("a8w4", args.backend, d["w1_scale"], shape.dim2, shape.dim1)
    x_q, x_scales, _ = _quant_act("a8w4", d["x"])
    log("rung5: launching gluon GEMM1")
    y1 = a8w4.moe_gemm_a8w4(
        x_q, d["w1q"], None, w1s, ss, ss, d["b1"], d["rdata"],
        gather_indx=d["gather_indx"], swizzle_mx_scale=sw1,
        out_dtype=torch.float8_e4m3fn, apply_swiglu=True,
    )
    torch.cuda.synchronize()
    log(f"rung5 OK: y1.shape={tuple(y1.shape)}")
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
