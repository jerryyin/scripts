#!/usr/bin/env python3
"""Compile TokenSpeed MoE kernels for gfx950 with no GPU (no device opened, nothing launched), and compare
two AMDGCN listings' instruction streams.

  petit_compile_check.py compile --ts-src TREE --kernel KERNEL [--profile dsv41_flash] [--asm-out FILE]
  petit_compile_check.py compare A.s B.s

compile: a stub Triton driver answers JITFunction.run's target queries (hip / gfx950 / wave 64) and the kernel's
warmup() compiles one specialization; prints compile time, VGPRs, LDS bytes and the instruction count, and
writes the AMDGCN to --asm-out. KERNEL is one of
  stage1, stage2, combine      Gluon Petit's launches as MegaMoESolutionAdapter.Kernels builds them (stage 1:
                               one row, external inputs)
  stage1_n128, stage2_n128     Petit's 32x128 stage-1 / stage-2 tiles (Kernels(n128=True) of
                               TOKENSPEED_PETIT_STAGE1_N128, Kernels(stage2_n128=True) of TOKENSPEED_PETIT_STAGE2_N128)
  combine_residual             Petit's combine with the post add (MegaMoECombineResidual, TOKENSPEED_PETIT_FUSE_POST)
  trace_ping                   Petit's trace clock-alignment kernel (MegaMoETracePing; needs TOKENSPEED_PETIT_TRACE=1)
  a4w4_ep_finalize             not Petit: the A4W4 EP registration's ep_route_finalize kernel
                               (TOKENSPEED_A4W4_EP_FUSED_FINALIZE) at the DeepSeek V4.1 target shape, top-6, or
                               with --profile dsv41_flash_dspark the draft's top-3
The Petit kernels need a TokenSpeed with tokenspeed_kernel.ops.moe.gluon._megamoe_config (46671ec2c and later
on users/jerryyin/ep4-beats-tp4; upstream main has none; compile says so and exits), the opt-in kernels the
commits that add them. TokenSpeed's switches in the environment apply as they do at import in a server. The platform is the owner fixture
`mi350_platform` of tokenspeed-kernel/test/conftest.py (evaluated from that file). Run it under
board/no-gpu/no-gpu-run when another process holds the GPUs.
compare (generic, any two AMDGCN listings): the instruction lines of the two files (no directives, labels,
comments or debug lines), so e.g. two trees' switch-off kernels can be shown identical although their source
paths and line numbers differ; exit 0 when identical, 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import sys
import time
from pathlib import Path

KERNELS = ("stage1", "stage1_n128", "stage2", "stage2_n128", "combine", "combine_residual", "trace_ping",
           "a4w4_ep_finalize")


def instructions(asm: str) -> list[str]:
    """The instruction lines of an AMDGCN listing: no directives, labels, comments or blank lines."""
    out = []
    for line in asm.splitlines():
        line = line.split(";")[0].rstrip()
        if re.match(r"^\t[a-z_][a-z0-9_]*", line) and not line.lstrip().startswith("."):
            out.append(line.strip())
    return out


def compare(a: str, b: str) -> dict:
    ia, ib = instructions(a), instructions(b)
    digest = lambda x: hashlib.sha256("\n".join(x).encode()).hexdigest()  # noqa: E731
    return {"identical": ia == ib, "lines": (len(ia), len(ib)), "sha256": (digest(ia), digest(ib))}


# What compile needs from TokenSpeed beyond upstream main: tokenspeed_kernel.ops.moe.gluon._megamoe_config,
# which the owner test re-exports (added by TokenSpeed 46671ec2c, "Size Gluon Petit's per-rank token
# capacity from the server", on branch users/jerryyin/ep4-beats-tp4 of jerryyin/tokenspeed).
NEEDS_TOKENSPEED = ("a TokenSpeed whose tokenspeed_kernel.ops.moe.gluon has _megamoe_config (46671ec2c and later on "
                    "users/jerryyin/ep4-beats-tp4; upstream main has none)")


def require_megamoe_config(test_module) -> None:
    """Exit naming the TokenSpeed this tool needs when the owner test has no _megamoe_config."""
    if not hasattr(test_module, "_megamoe_config"):
        raise SystemExit(f"petit_compile_check.py compile needs {NEEDS_TOKENSPEED}; "
                         f"{test_module.__file__} imports none")


def _platform(ts_src: Path):
    src = (ts_src / "tokenspeed-kernel/test/conftest.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "mi350_platform")
    ret = next(n for n in fn.body if isinstance(n, ast.Return))
    from tokenspeed_kernel import platform as P
    P.Platform.override(eval(compile(ast.Expression(ret.value), "conftest", "eval"), {k: getattr(P, k) for k in dir(P)}))


def compile_kernel(ts_src: Path, kernel: str, profile: str = "dsv41_flash"):
    for sub in ("tokenspeed-kernel/python", "tokenspeed-kernel-amd/python", "python", "tokenspeed-kernel/test",
                "tokenspeed-kernel/test/amd/ops/moe"):
        sys.path.insert(0, str(ts_src / sub))
    import torch
    from tokenspeed_triton.backends.compiler import GPUTarget
    from tokenspeed_triton.runtime import driver

    class StubDriver:  # never touches a device
        def get_current_device(self): return 0
        def get_current_stream(self, device=None): return 0
        def get_current_target(self): return GPUTarget("hip", "gfx950", 64)
        def get_active_torch_device(self): return torch.device("cpu")
        def is_active(self): return True

    driver.set_active(StubDriver())
    _platform(ts_src)
    if kernel == "a4w4_ep_finalize":
        return _compile_a4w4_ep_finalize(profile)
    import test_gluon_petit as t  # the owner test's config helpers
    pk = t._import_petit_kernel()
    import lib.moe.rocm.mega_moe as fwd
    mod = sys.modules[fwd.MegaMoESolutionAdapter.__module__]
    from lib.moe.rocm.mega_moe import mega_moe_two_stage_kernel as K
    prof = {"dsv41_flash": t._DSV41_FLASH_PROFILE, "dsv41_flash_dspark": t._DSV41_FLASH_DSPARK_PROFILE}[profile]
    require_megamoe_config(t)
    config = t._megamoe_config(pk, prof, 4, 1024)
    adapter = mod._MegaMoESolution(config._solution_id_for_tokens(1))
    hidden = adapter.Config.kHiddenSize
    u8 = lambda n: torch.empty(n, dtype=torch.uint8)  # noqa: E731
    bf = lambda *s: torch.empty(*s, dtype=torch.bfloat16)  # noqa: E731
    t0 = time.time()
    if kernel.startswith("stage1"):
        s1, _, _ = adapter.Kernels(True, False, False, True, **({"n128": True} if kernel == "stage1_n128" else {}))
        k = K.MegaMoEStage1.warmup(u8(64), u8(64), 1, None, u8(64), 0, u8(64), u8(64), u8(64), s1,
                                   num_warps=s1.kNumWarps, enable_fp_fusion=False, grid=(s1.kNumSMs,))
    elif kernel == "trace_ping":  # TOKENSPEED_PETIT_TRACE's clock-alignment kernel
        ping = K.MegaMoETracePingKernel(adapter.Config)
        k = K.MegaMoETracePing.warmup(u8(64), 0, 0, 0, ping, num_warps=1, grid=(1,))
    elif kernel.startswith("stage2"):
        _, s2, _ = adapter.Kernels(True, False, False, True, **({"stage2_n128": True} if kernel == "stage2_n128" else {}))
        k = K.MegaMoEStage2.warmup(u8(64), u8(64), None, u8(64), 0, s2, num_warps=s2.kNumWarps,
                                   enable_fp_fusion=False, grid=(s2.kStage2GridBlocks,))
    elif kernel == "combine_residual":
        _, _, comb = adapter.Kernels(True, False, False, True, add_residual=True)
        k = K.MegaMoECombineResidual.warmup(bf(4, hidden), 4, hidden, bf(4, hidden), hidden, u8(64), 0, comb,
                                            num_warps=comb.kNumWarps, enable_fp_fusion=False, grid=(comb.kNumSMs,))
    else:
        _, _, comb = adapter.Kernels(True, False, False, True)
        k = K.MegaMoECombine.warmup(bf(4, hidden), 4, hidden, u8(64), 0, comb,
                                    num_warps=comb.kNumWarps, enable_fp_fusion=False, grid=(comb.kNumSMs,))
    asm = k.asm["amdgcn"]
    vgpr = next((l.split()[-1] for l in asm.splitlines() if ".vgpr_count" in l), "?")
    return asm, {"kernel": kernel, "compile_s": round(time.time() - t0, 1), "vgprs": vgpr,
                 "lds_bytes": getattr(k.metadata, "shared", None), "instructions": len(instructions(asm))}


def _compile_a4w4_ep_finalize(profile: str):
    """ep_route_finalize's kernel as the registration launches it (FP32 weights, BF16 routes and output)."""
    import torch
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import ep_route_finalize as F
    top_k = {"dsv41_flash": 6, "dsv41_flash_dspark": 3}[profile]
    rows, hidden = 6, 5120
    t0 = time.time()
    k = F._ep_route_finalize_kernel.warmup(
        torch.empty(rows * top_k, hidden, dtype=torch.bfloat16), torch.empty(rows * top_k, dtype=torch.int32),
        torch.empty(rows, top_k, dtype=torch.float32), torch.empty(rows, hidden, dtype=torch.bfloat16),
        hidden, hidden, top_k, 1, hidden, TOPK=top_k, BLOCK_N=F.EP_ROUTE_FINALIZE_BLOCK_N, num_warps=4,
        enable_fp_fusion=False, grid=(rows, -(-hidden // F.EP_ROUTE_FINALIZE_BLOCK_N)))
    asm = k.asm["amdgcn"]
    vgpr = next((l.split()[-1] for l in asm.splitlines() if ".vgpr_count" in l), "?")
    return asm, {"kernel": "a4w4_ep_finalize", "compile_s": round(time.time() - t0, 1), "vgprs": vgpr,
                 "lds_bytes": getattr(k.metadata, "shared", None), "instructions": len(instructions(asm)),
                 "fma": sum(1 for i in instructions(asm) if "fma" in i.split()[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--ts-src", required=True, type=Path)
    c.add_argument("--kernel", required=True, choices=KERNELS)
    c.add_argument("--profile", default="dsv41_flash", choices=("dsv41_flash", "dsv41_flash_dspark"))
    c.add_argument("--asm-out", default=None)
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        r = compare(Path(a.a).read_text(), Path(a.b).read_text())
        print(f"identical={r['identical']} lines={r['lines']} sha256={r['sha256']}")
        return 0 if r["identical"] else 1
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    asm, info = compile_kernel(a.ts_src.resolve(), a.kernel, a.profile)
    if a.asm_out:
        Path(a.asm_out).write_text(asm)
    print(" ".join(f"{k}={v}" for k, v in info.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
