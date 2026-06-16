"""Shared helpers for verifying aiter MoE GEMM kernels (a4w4 / a8w4) under FFM.

Single source of truth for input construction, scale swizzling, the dequantized
torch reference, and pass/fail comparison. Driver scripts import from here so the
per-kernel logic is never duplicated.

Design notes:
- Verification is on the FULL two-GEMM MoE forward (the same computation the aiter
  bench scripts run), not on isolated GEMMs. The final output is the scattered
  batch rows: every row is consumed, so the check is deterministic. Isolated-GEMM
  checks instead read gather-padding rows backed by torch.empty (uninitialized),
  which falsely report NaN even when the kernel is correct.
- Correctness is measured against a dequantized torch reference, never inferred
  from finiteness: a broken kernel can emit finite garbage (~1e23).
"""
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass

import torch

# aiter is not pip-installed; it is importable from its repo root. Point AITER_HOME
# at the clone if it lives somewhere other than /root/aiter.
_AITER_HOME = os.environ.get("AITER_HOME", "/root/aiter")
if _AITER_HOME not in sys.path:
    sys.path.insert(0, _AITER_HOME)

from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.moe.quant_moe import (
    downcast_to_static_fp8,
    downcast_to_mxfp,
    upcast_from_mxfp,
)
from aiter.ops.triton.moe import moe_op_gemm_a4w4 as a4w4
from aiter.ops.triton.moe import moe_op_gemm_a8w4 as a8w4

# Static activation scale for the a8w4 fp8 path. Kept small so x/scale stays in
# fp8 e4m3 range for the N(0, INPUT_STD) inputs below (avoids saturation that
# would be unrelated to the kernel under test).
STATIC_SCALE = 0.1
# Inputs scaled so two stacked GEMMs stay inside bf16 range; keeps "finite vs
# garbage" unambiguous without masking real kernel errors.
INPUT_STD = 0.1

_moe_gemm_torch = a4w4.moe_gemm_torch  # pure-torch, kernel-agnostic reference


@dataclass(frozen=True)
class Shape:
    dim1: int = 256
    dim2: int = 512
    # 32 experts / top-4: with too few experts (e.g. 8) and many tokens, aiter's
    # routing returns a histogram that does not sum to batch*n_expts_act (a
    # structural degeneracy), which desyncs the torch reference from the kernel.
    # 32/4 keeps routing well-formed (hist_sum == batch*n_expts_act) and is closer
    # to real MoE configs than 8/2.
    n_expts_tot: int = 32
    n_expts_act: int = 4

    @property
    def gemm2_k(self) -> int:
        # GEMM2 contracts over the swiglu-halved hidden dim.
        return self.dim2 // 2


@dataclass(frozen=True)
class CompareResult:
    finite_frac: float  # fraction of output rows that are all-finite
    rel_err: float      # mean |ref - y| / mean |ref|
    cosine: float       # cosine similarity vs reference

    def passed(self, cos_tol: float = 0.99, finite_tol: float = 0.999) -> bool:
        # Verdict is on the full 2-layer forward (final batch rows), with
        # well-formed routing. Correct kernels land at cosine >= 0.9999 (rel_err
        # ~1e-3 for a4w4, ~1e-2 for a8w4's fp8 intermediate). A regressed kernel
        # would drop well below; finiteness alone is never sufficient since a
        # broken kernel can still emit finite garbage.
        return self.finite_frac >= finite_tol and self.cosine > cos_tol


def compare(ref: torch.Tensor, y: torch.Tensor) -> CompareResult:
    ref, y = ref.float(), y.float()
    row_finite = torch.isfinite(y).all(dim=1)
    finite_frac = float(row_finite.float().mean().item())
    rel = (ref - y).abs().mean() / ref.abs().mean().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), y.flatten(), dim=0)
    return CompareResult(finite_frac, float(rel.item()), float(cos.item()))


def _cpu_routing(rdata):
    """A device-free shim of RoutingData carrying only the fields moe_gemm_torch
    reads, so the reference matmuls run on CPU (FFM simulates GPU matmuls)."""
    return types.SimpleNamespace(
        n_expts_act=rdata.n_expts_act,
        n_expts_tot=rdata.n_expts_tot,
        expt_hist=rdata.expt_hist.cpu(),
    )


def _swizzle(kernel: str, backend: str, scale: torch.Tensor, N: int, K: int):
    """Return (swizzled_scale, swizzle_mx_scale_str) matched to the target kernel.

    a4w4 has a single CDNA4-style layout. a8w4's gluon kernel wants the GFX1250
    layout; its triton kernel only understands CDNA4 (or None) and produces NaN
    if fed the GFX1250 layout.
    """
    if not (N % 32 == 0 and K % (32 * 8) == 0):
        return scale, None
    if kernel == "a4w4":
        return a4w4.swizzle_scales(scale), "CDNA4_SCALE"
    if backend == "gluon":
        return a8w4.swizzle_scales_gfx1250(scale), "GFX1250_SCALE"
    return a8w4.swizzle_scales_gfx950(scale), "CDNA4_SCALE"


def _quant_act(kernel: str, x: torch.Tensor):
    """Quantize an activation the way each kernel expects, and return its bf16
    dequantization (what the torch reference and the next layer's reference use).

    Returns (x_q, x_scales, x_deq).
    """
    if kernel == "a4w4":
        x_fp4, x_scale = a4w4.mxfp4_quant(x)
        x_deq = upcast_from_mxfp(x_fp4, x_scale, torch.bfloat16, axis=1)
        return x_fp4, x_scale, x_deq
    x_fp8 = downcast_to_static_fp8(x, torch.tensor(STATIC_SCALE, device=x.device))
    x_deq = (x_fp8.float() * STATIC_SCALE).to(torch.bfloat16)
    return x_fp8, None, x_deq


def build(shape: Shape, batch: int, seed: int = 0):
    """Build routing + quantized weights/biases for a 2-layer MoE MLP."""
    dev = "cuda:0"
    torch.manual_seed(seed)
    s = shape
    wg = torch.randn((s.dim1, s.n_expts_tot), device=dev)
    wg = wg.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
    bg = torch.randn((s.n_expts_tot,), device=dev)

    w1 = torch.randn((s.n_expts_tot, s.dim1, s.dim2), device=dev) * INPUT_STD
    w2 = torch.randn((s.n_expts_tot, s.gemm2_k, s.dim1), device=dev) * INPUT_STD
    b1 = torch.randn((s.n_expts_tot, s.dim2), device=dev) * INPUT_STD
    b2 = torch.randn((s.n_expts_tot, s.dim1), device=dev) * INPUT_STD
    w1q, w1_scale = downcast_to_mxfp(w1.to(torch.bfloat16), torch.uint8, axis=1)
    w2q, w2_scale = downcast_to_mxfp(w2.to(torch.bfloat16), torch.uint8, axis=1)

    x = torch.randn((batch, s.dim1), dtype=torch.bfloat16, device=dev) * INPUT_STD
    logits = gemm_a16w16(x, wg.T, bg)
    rdata, gather_indx, scatter_indx = routing(logits, s.n_expts_act)
    return dict(
        x=x, w1q=w1q, w2q=w2q, w1_scale=w1_scale, w2_scale=w2_scale,
        b1=b1, b2=b2, rdata=rdata, gather_indx=gather_indx, scatter_indx=scatter_indx,
    )


def run_forward(kernel: str, backend: str, shape: Shape, d: dict) -> CompareResult:
    """Run the full 2-layer MoE forward (GEMM1+swiglu -> requant -> GEMM2+scatter)
    and compare the final batch output against a dequantized torch reference.

    End-to-end is the robust check: the kernel and moe_gemm_torch use different
    intermediate row layouts (the kernel pads the gather to block_m; the torch
    reference packs raw expert ranges), so per-GEMM intermediates do NOT line up.
    Both pipelines are internally self-consistent, so the final per-token batch
    output agrees regardless of layout."""
    ss = torch.tensor(STATIC_SCALE, device=d["x"].device)
    w1s, sw1 = _swizzle(kernel, backend, d["w1_scale"], shape.dim2, shape.dim1)
    w2s, sw2 = _swizzle(kernel, backend, d["w2_scale"], shape.dim1, shape.gemm2_k)
    w1_deq = upcast_from_mxfp(d["w1q"], d["w1_scale"], torch.bfloat16, axis=1)
    w2_deq = upcast_from_mxfp(d["w2q"], d["w2_scale"], torch.bfloat16, axis=1)
    x_q, x_scales, x_deq = _quant_act(kernel, d["x"])

    if kernel == "a4w4":
        y1 = a4w4.moe_gemm_a4w4(
            x_q, d["w1q"], x_scales, w1s, None, None, d["b1"], d["rdata"],
            gather_indx=d["gather_indx"], swizzle_mx_scale=sw1, apply_swiglu=True,
        )
        y1_q, y1_scales, _ = _quant_act("a4w4", y1)
        y2 = a4w4.moe_gemm_a4w4(
            y1_q, d["w2q"], y1_scales, w2s, None, None, d["b2"], d["rdata"],
            scatter_indx=d["scatter_indx"], swizzle_mx_scale=sw2,
        )
    else:
        y1 = a8w4.moe_gemm_a8w4(
            x_q, d["w1q"], None, w1s, ss, ss, d["b1"], d["rdata"],
            gather_indx=d["gather_indx"], swizzle_mx_scale=sw1,
            out_dtype=torch.float8_e4m3fn, apply_swiglu=True,
        )
        y2 = a8w4.moe_gemm_a8w4(
            y1, d["w2q"], None, w2s, ss, None, d["b2"], d["rdata"],
            scatter_indx=d["scatter_indx"], swizzle_mx_scale=sw2, out_dtype=torch.bfloat16,
        )
    torch.cuda.synchronize()

    # Reference: dequantized matmuls (on CPU -- FFM simulates every GPU matmul, so
    # a GPU reference dominates runtime) with the same inter-layer requant the
    # kernel uses (the cheap requant stays on GPU so it matches exactly).
    rd_cpu = _cpu_routing(d["rdata"])
    gather_cpu, scatter_cpu = d["gather_indx"].cpu(), d["scatter_indx"].cpu()
    r1 = _moe_gemm_torch(x_deq.cpu(), w1_deq.cpu(), d["b1"].cpu(), rd_cpu,
                         gather_indx=gather_cpu, apply_swiglu=True)
    _, _, r1_deq = _quant_act(kernel, r1.to(torch.bfloat16).to(d["x"].device))
    r2 = _moe_gemm_torch(r1_deq.cpu(), w2_deq.cpu(), d["b2"].cpu(), rd_cpu,
                         scatter_indx=scatter_cpu, apply_swiglu=False)
    return compare(r2, y2.cpu())
