"""Run the a8w4 GEMM1 kernel under AM for itrace, in one of two modes.

Default (--data): load a precomputed payload (from precompute_routing.py) and
fire ONLY the GEMM1 launch. No routing kernel runs, so the AM ifrit abort on the
routing dispatch is avoided -- this is the mode used to actually capture a trace.
    LD_PRELOAD= GPU_ARCHS=gfx1250 run_on_model.sh --backend am -- \
        python3 itrace_gemm1_pre.py --backend gluon --data moe_decode.pt

Full build (--build): build EVERYTHING inline on the device -- gate GEMM,
routing, weight quant -- then run GEMM1. Under AM this REPRODUCES THE CRASH: the
aiter routing kernel aborts the model ("ifrit ... risky access of scalar_l0_inv
on SPI_SQ_cmd"). Use this to reproduce/debug that abort.
    LD_PRELOAD= GPU_ARCHS=gfx1250 run_on_model.sh --backend am -- \
        python3 itrace_gemm1_pre.py --backend gluon --build \
            --shape 2048 7168 --experts 256 8 --batch 128

With itrace enabled in am_env.sh the (default mode) run emits
xcc0se0sa0_itrace_emu.mon.
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Shared lib_moe_ffm.py lives in the parent (shared) dir after the folder reorg.
sys.path.insert(0, os.path.dirname(_HERE))
_AITER_HOME = os.environ.get("AITER_HOME", "/root/aiter")
if _AITER_HOME not in sys.path:
    sys.path.insert(0, _AITER_HOME)


def log(msg: str):
    print(f"[itrace t={time.monotonic():.1f}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["gluon", "triton"], required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="precomputed payload .pt (GEMM1-only mode)")
    src.add_argument("--build", action="store_true",
                     help="build inputs inline incl. routing on-device "
                          "(reproduces the AM routing crash)")
    # used only with --build
    p.add_argument("--shape", type=int, nargs=2, metavar=("K", "N"), default=(2048, 7168))
    p.add_argument("--experts", type=int, nargs=2, metavar=("TOT", "ACT"), default=(256, 8))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_from_payload(args, torch, a8w4, dev):
    log(f"loading {args.data} -> {dev}")
    pl = torch.load(args.data, map_location=dev, weights_only=False)
    m = pl["meta"]
    log(f"loaded backend={args.backend} block_m={m['block_m']} "
        f"K={m['dim1']} N={m['dim2']} experts={m['n_expts_tot']}/{m['n_expts_act']}")
    ss = torch.tensor(m["static_scale"], device=dev)
    bk = pl[args.backend]
    log("launching GEMM1")
    y1 = a8w4.moe_gemm_a8w4(
        pl["x_q"], pl["w1q"], None, bk["w1s"], ss, ss, pl["b1"], pl["rdata"],
        gather_indx=pl["gather_indx"], swizzle_mx_scale=bk["sw1"],
        out_dtype=torch.float8_e4m3fn, apply_swiglu=True,
    )
    return y1


def run_from_build(args, torch, a8w4, dev):
    # Builds gate GEMM + routing + weight quant ON THE DEVICE. Under AM the
    # routing kernel aborts the model -- this is the crash-repro path.
    from lib_moe_ffm import Shape, build, _swizzle, _quant_act, STATIC_SCALE
    K, N = args.shape
    E, A = args.experts
    shape = Shape(dim1=K, dim2=N, n_expts_tot=E, n_expts_act=A)
    log(f"building inline (on-device routing) backend={args.backend} "
        f"K={K} N={N} experts={E}/{A} batch={args.batch}")
    d = build(shape, args.batch, seed=args.seed)   # <-- on-device routing(): AM aborts here
    log(f"build done: block_m={d['rdata'].block_m} "
        f"hist_sum={int(d['rdata'].expt_hist.sum())}")
    ss = torch.tensor(STATIC_SCALE, device=dev)
    w1s, sw1 = _swizzle("a8w4", args.backend, d["w1_scale"], N, K)
    x_q, _, _ = _quant_act("a8w4", d["x"])
    log("launching GEMM1")
    y1 = a8w4.moe_gemm_a8w4(
        x_q, d["w1q"], None, w1s, ss, ss, d["b1"], d["rdata"],
        gather_indx=d["gather_indx"], swizzle_mx_scale=sw1,
        out_dtype=torch.float8_e4m3fn, apply_swiglu=True,
    )
    return y1


def main() -> int:
    args = parse_args()
    os.environ["AITER_FORCE_TRITON"] = "1" if args.backend == "triton" else "0"

    import torch
    from aiter.ops.triton.moe import moe_op_gemm_a8w4 as a8w4

    dev = "cuda:0"
    y1 = run_from_build(args, torch, a8w4, dev) if args.build \
        else run_from_payload(args, torch, a8w4, dev)
    torch.cuda.synchronize()
    log(f"GEMM1 done: y1.shape={tuple(y1.shape)} dtype={y1.dtype}")
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
