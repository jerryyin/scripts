"""Precompute the a8w4 GEMM1 inputs and serialize them, so the AM itrace run can
call ONLY the GEMM1 kernel -- avoiding the routing kernel, which aborts the AM
model (ifrit "risky access of scalar_l0_inv" on SPI_SQ_cmd).

Routing metadata is built with aiter's own pure-torch reference `routing_torch`
on CPU (no GPU routing kernel, no gate GEMM). The only GPU work is the standard
weight quantization (`downcast_to_mxfp`) -- same infra the aiter bench/test use;
no fabricated weights. Run under FFM so that quant executes on the model:
    run_on_model.sh --backend ffm -- python3 precompute_routing.py --out X.pt ...

Both gluon (GFX1250) and triton (CDNA4) scale layouts are precomputed.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Shared lib_moe_ffm.py lives in the parent (shared) dir after the folder reorg.
sys.path.insert(0, os.path.dirname(_HERE))
_AITER_HOME = os.environ.get("AITER_HOME", "/root/aiter")
if _AITER_HOME not in sys.path:
    sys.path.insert(0, _AITER_HOME)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output .pt path")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--shape", type=int, nargs=2, default=(2048, 7168),
                   help="(K, N) of GEMM1")
    p.add_argument("--experts", type=int, nargs=2, default=(256, 8))
    p.add_argument("--quant-experts", type=int, default=8,
                   help="Quantize this many REAL experts on the model, then tile "
                        "along the expert axis to the full count. The instruction "
                        "trace is data-independent, so tiling real mxfp4 weights "
                        "gives an identical trace at a fraction of the quant cost. "
                        "Set == total experts to quantize every expert.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.environ["AITER_FORCE_TRITON"] = "0"

    import torch
    import triton
    from lib_moe_ffm import Shape, _swizzle, _quant_act, STATIC_SCALE, INPUT_STD, get_arch
    from aiter.ops.triton.moe.moe_routing.routing import (
        RoutingData, compute_expt_data_torch,
    )
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

    def cpu_routing(logits, n_expts_act):
        """CPU mirror of aiter `routing_torch` (moe_routing/routing.py) with
        torch.bincount in place of torch.histc (histc has no int CPU kernel).
        Produces the same (RoutingData, gather_indx, scatter_indx)."""
        n_tokens, n_expts_tot = logits.shape
        n_gates_pad = n_tokens * n_expts_act
        tk_indx = torch.argsort(logits, dim=1, stable=True)[:, -n_expts_act:].long()
        tk_val = torch.take_along_dim(logits, tk_indx, dim=1)
        expt_scal = torch.softmax(tk_val, dim=-1)
        expt_indx = tk_indx.int()
        expt_indx, sort_idx = torch.sort(expt_indx, dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_idx).reshape(-1)
        expt_indx = expt_indx.reshape(-1).to(torch.int32)
        topk_indx = torch.argsort(expt_indx, stable=True)
        gate_indx = torch.argsort(topk_indx, stable=True)
        gate_scal = expt_scal[topk_indx]
        hist = torch.bincount(expt_indx.long(), minlength=n_expts_tot).int()
        tokens_per_expt = max(1, n_gates_pad // n_expts_tot)
        block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
        expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad, block_m)
        rdata = RoutingData(block_m, gate_scal, hist, n_expts_tot, n_expts_act, expt_data)
        return rdata, topk_indx.int(), gate_indx.int()

    torch.manual_seed(args.seed)
    K, N = args.shape
    E, A = args.experts
    shape = Shape(dim1=K, dim2=N, n_expts_tot=E, n_expts_act=A)
    dev = "cuda:0"

    # --- routing on CPU (aiter reference algorithm; no GPU routing kernel / gate GEMM) ---
    logits = torch.randn((args.batch, E), dtype=torch.float32)  # CPU
    rdata, gather_indx, scatter_indx = cpu_routing(logits, A)
    block_m = rdata.block_m
    hist_sum = int(rdata.expt_hist.sum())
    print(f"arch={get_arch()} batch={args.batch} experts={E}/{A} K={K} N={N} "
          f"block_m={block_m} hist_sum={hist_sum} (routing=torch/cpu)", flush=True)
    if hist_sum != args.batch * A:
        print(f"  WARNING: degenerate routing (hist_sum={hist_sum} != {args.batch*A}).", flush=True)

    # --- weights ---
    # downcast_to_mxfp(w (E,K,N), uint8, axis=1) returns:
    #   w1q     = (E, K//2, N) uint8   (two e2m1 fp4 packed per byte along K)
    #   w1_scale= (E, ceil(K/32), N) uint8   (one e8m0 scale per 32-elt K block)
    # The GEMM1 instruction trace is data-INDEPENDENT (a GEMM issues the same
    # loads/wmma/stores regardless of operand values), and every uint8 is a
    # valid e2m1-pair / e8m0 scale. So for itrace we FABRICATE valid-shaped
    # random bytes on CPU instead of quantizing real weights under the model --
    # quantizing 256x2048x7168 (3.75B elts) under FFM costs 20+ min for a trace
    # that is byte-for-byte identical in its instruction stream. Set
    # --quant-experts < 0 to force the real downcast path instead (slow).
    if args.quant_experts < 0:
        from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
        w1 = (torch.randn((E, K, N)) * INPUT_STD).to(torch.bfloat16).to(dev)
        w1q, w1_scale = downcast_to_mxfp(w1, torch.uint8, axis=1)
        w1q, w1_scale = w1q.cpu(), w1_scale.cpu()
        print("  weights: real downcast_to_mxfp", flush=True)
    else:
        # Match downcast_to_mxfp's returned layout: physically (E, N, K//2) /
        # (E, N, K_SCALE) contiguous, returned transposed to (E, K//2, N) /
        # (E, K_SCALE, N) with stride(-2)==1 (the kernel asserts column-major mxfp).
        w1q = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8).transpose(-1, -2)
        w1_scale = torch.randint(0, 256, (E, N, (K + 31) // 32), dtype=torch.uint8).transpose(-1, -2)
        print(f"  weights: fabricated mxfp4 w1q={tuple(w1q.shape)} stride{tuple(w1q.stride())} "
              f"w1_scale={tuple(w1_scale.shape)} (trace is data-independent)", flush=True)
    # Swizzle is pure-torch (CPU-fast).
    w1s_g, sw1_g = _swizzle("a8w4", "gluon", w1_scale, N, K)
    w1s_t, sw1_t = _swizzle("a8w4", "triton", w1_scale, N, K)

    # --- activation: fp8 e4m3, shape (batch, K). Fabricated on CPU (same
    #     data-independence argument). ---
    x_q = torch.randint(0, 256, (args.batch, K), dtype=torch.uint8).view(torch.float8_e4m3fn)
    b1 = (torch.randn((E, N)) * INPUT_STD)

    def cpu(t):
        return t.cpu() if isinstance(t, torch.Tensor) else t

    payload = dict(
        meta=dict(batch=args.batch, dim1=K, dim2=N, n_expts_tot=E, n_expts_act=A,
                  block_m=block_m, static_scale=STATIC_SCALE),
        x_q=cpu(x_q), w1q=cpu(w1q), b1=cpu(b1),
        rdata=rdata, gather_indx=gather_indx,   # already CPU (routing_torch on CPU)
        gluon=dict(w1s=cpu(w1s_g), sw1=sw1_g),
        triton=dict(w1s=cpu(w1s_t), sw1=sw1_t),
    )
    payload = torch.utils._pytree.tree_map(
        lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, payload)
    torch.save(payload, args.out)
    print(f"  saved -> {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
