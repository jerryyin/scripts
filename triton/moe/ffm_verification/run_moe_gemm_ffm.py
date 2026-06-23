#!/usr/bin/env python3
"""Verify a single (kernel, backend, phase) cell of the aiter MoE GEMM matrix
against a dequantized torch reference, under FFM on gfx1250.

Each invocation is one matrix cell. Examples:
    python run_moe_gemm_ffm.py --kernel a4w4 --phase prefill
    python run_moe_gemm_ffm.py --kernel a8w4 --backend gluon  --phase decode
    python run_moe_gemm_ffm.py --kernel a8w4 --backend triton --phase prefill

Backends:
    a4w4 is pure-Triton (no gluon). a8w4 defaults to gluon on gfx1250; --backend
    triton forces the Triton kernel (sets AITER_FORCE_TRITON=1, which the patched
    moe_op_gemm_a8w4.py honours) and selects the CDNA4 scale layout it requires.

Exit status is non-zero if any checked GEMM fails, so this is CI-usable.
"""
import argparse
import os
import sys

# Shared lib_moe_ffm.py lives in the parent dir (../) after the folder reorg.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Phase -> batch size. With the default shape (dim1=256, dim2=512, 32/4 experts)
# routing yields block_m=16 for batch 64 and block_m=128 for batch 2048.
PHASE_BATCH = {"decode": 64, "prefill": 2048}
PHASE_BLOCK_M = {"decode": 16, "prefill": 128}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", choices=["a4w4", "a8w4"], required=True)
    p.add_argument("--backend", choices=["auto", "gluon", "triton"], default="auto")
    p.add_argument("--phase", choices=["decode", "prefill"], required=True)
    p.add_argument("--batch", type=int, default=None,
                   help="Override the phase's default batch size.")
    p.add_argument("--shape", type=int, nargs=2, metavar=("DIM1", "DIM2"), default=None,
                   help="MoE feature dims (default 256 512).")
    p.add_argument("--experts", type=int, nargs=2, metavar=("TOT", "ACT"), default=None,
                   help="Total and active experts (default 32 4). Too few experts "
                        "with many tokens makes routing degenerate; see warning.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_backend(kernel: str, backend: str) -> str:
    if kernel == "a4w4":
        if backend == "gluon":
            sys.exit("error: a4w4 has no gluon kernel; use --backend triton or auto.")
        return "triton"
    return "gluon" if backend == "auto" else backend


def main() -> int:
    args = parse_args()
    backend = resolve_backend(args.kernel, args.backend)
    # The a8w4 gluon/triton switch reads this env at call time.
    os.environ["AITER_FORCE_TRITON"] = "1" if backend == "triton" else "0"

    from lib_moe_ffm import Shape, build, run_forward, get_arch

    kw = {}
    if args.shape is not None:
        kw["dim1"], kw["dim2"] = args.shape
    if args.experts is not None:
        kw["n_expts_tot"], kw["n_expts_act"] = args.experts
    shape = Shape(**kw)
    batch = args.batch if args.batch is not None else PHASE_BATCH[args.phase]
    d = build(shape, batch, seed=args.seed)
    block_m = d["rdata"].block_m
    print(f"arch={get_arch()} kernel={args.kernel} backend={backend} "
          f"phase={args.phase} batch={batch} experts={shape.n_expts_tot}/{shape.n_expts_act} "
          f"block_m={block_m}")

    if args.batch is None and block_m != PHASE_BLOCK_M[args.phase]:
        sys.exit(f"error: phase {args.phase} expected block_m={PHASE_BLOCK_M[args.phase]} "
                 f"but routing produced block_m={block_m}; adjust --batch/shape.")

    # The torch reference desyncs from the kernel when routing is degenerate
    # (histogram does not sum to batch*n_expts_act) -- happens with too few
    # experts for the token count. Warn so a FAIL is not misread as a kernel bug.
    hist_sum = int(d["rdata"].expt_hist.sum())
    if hist_sum != batch * shape.n_expts_act:
        print(f"  WARNING: degenerate routing (hist_sum={hist_sum} != "
              f"{batch * shape.n_expts_act}); reference is unreliable for this shape.")

    r = run_forward(args.kernel, backend, shape, d)
    ok = r.passed()
    print(f"  forward: {'PASS' if ok else 'FAIL'}  finite_frac={r.finite_frac:.3f} "
          f"rel_err={r.rel_err:.4g} cosine={r.cosine:.6f}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    _rc = main()
    # FFM hangs on normal interpreter teardown; os._exit avoids the hang (and the
    # zombie processes that otherwise pile up and contend for the single device).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
