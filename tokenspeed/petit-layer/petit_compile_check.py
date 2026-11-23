#!/usr/bin/env python3
"""Compile TokenSpeed's Gluon Petit stage-2 / combine kernels for gfx950 with no GPU (no device opened, nothing
launched), and compare two compiles' instruction streams.

  petit_compile_check.py compile --ts-src TREE --kernel stage2|stage2_n128|combine|combine_residual \
      [--profile dsv41_flash] [--asm-out FILE]
  petit_compile_check.py compare A.s B.s

compile: a stub Triton driver answers JITFunction.run's target queries (hip / gfx950 / wave 64), the kernel's
warmup() compiles the specialization Petit's MegaMoESolutionAdapter.Kernels builds (stage2_n128: C5b's
MegaMoEStage2N128Config; combine_residual: C3's MegaMoECombineResidual), and the AMDGCN is written; prints
compile time, VGPRs and LDS bytes. The platform is the owner fixture `mi350_platform` of
tokenspeed-kernel/test/conftest.py (evaluated from that file). Run it with an LD_PRELOAD that refuses /dev/kfd
and /dev/dri opens when another process holds the GPUs.
compare: the instruction lines of two AMDGCN files (no directives, comments or debug lines), so two trees'
switch-off kernels can be shown identical although their source paths and line numbers differ.
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

KERNELS = ("stage2", "stage2_n128", "combine", "combine_residual")


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
    if kernel.startswith("stage2"):
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
