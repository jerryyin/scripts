#!/usr/bin/env python3
"""Probe whether Triton's proton profiler (hook='triton', rocprofiler-sdk) works
in the current environment.

The aiter MoE bench scripts wrap their loop in proton.start(hook='triton'). Under
FFM this fails: rocprofiler-sdk cannot enumerate the simulated agents. This script
isolates that dependency so the failure is diagnosed without a full bench run.

Exit 0 => proton works (real hardware). Exit 1 => proton unavailable (e.g. FFM):
run the kernels directly with run_moe_gemm_ffm.py instead.
"""
import os
import sys
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


def main() -> int:
    try:
        import triton.profiler as proton
    except Exception as e:  # noqa: BLE001 - report any import-time failure verbatim
        print(f"proton import FAILED: {type(e).__name__}: {e}")
        return 1

    n = 4096
    x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    o = torch.empty_like(x)
    fpath = Path(tempfile.mktemp())
    try:
        proton.start(str(fpath), hook="triton")
        for _ in range(5):
            _add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
        proton.finalize()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"proton profiling FAILED: {type(e).__name__}: {e}")
        return 1

    print("proton profiling OK")
    return 0


if __name__ == "__main__":
    _rc = main()
    # FFM hangs on normal interpreter teardown; os._exit avoids the hang.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
