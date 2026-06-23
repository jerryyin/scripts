"""Run ONLY the a8w4 GEMM1 kernel under AM from precomputed inputs, for itrace.

Loads the payload written by precompute_routing.py (which ran routing+build under
FFM), moves it to the AM device, and fires a single moe_gemm_a8w4 GEMM1 launch.
No routing kernel runs here, so the AM ifrit abort on the routing dispatch is
avoided. With itrace enabled in am_env.sh this emits xcc0se0sa0_itrace_emu.mon.

Run under AM:
    LD_PRELOAD= GPU_ARCHS=gfx1250 run_on_model.sh --backend am -- \
        python3 itrace_gemm1_pre.py --backend gluon --data moe_decode.pt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_AITER_HOME = os.environ.get("AITER_HOME", "/root/aiter")
if _AITER_HOME not in sys.path:
    sys.path.insert(0, _AITER_HOME)


def log(msg: str):
    print(f"[itrace t={time.monotonic():.1f}] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["gluon", "triton"], required=True)
    p.add_argument("--data", required=True, help="payload .pt from precompute_routing.py")
    args = p.parse_args()
    os.environ["AITER_FORCE_TRITON"] = "1" if args.backend == "triton" else "0"

    import torch
    from aiter.ops.triton.moe import moe_op_gemm_a8w4 as a8w4

    dev = "cuda:0"
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
    torch.cuda.synchronize()
    log(f"GEMM1 done: y1.shape={tuple(y1.shape)} dtype={y1.dtype}")
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
