"""Correctness driver for the moe_gfx1250 dispatch matmul.

Calls the in-tree test_matmul harness (builds inputs, computes a torch reference,
runs the kernel, asserts numerics). Use under FFM (faster than AM) to validate a
kernel edit. Shapes via env MM_M/MM_N/MM_K.

IMPORTANT: test_matmul has FFM-only `pytest.skip`s gated on $HSA_MODEL_TOML for a
few specific (m,n,k)+block combos that crash/hang in CI. Pick a shape NOT on that
list -- e.g. (256,256,512) -- so the assert actually runs. (300,400,416) and
(128,128,512) ARE on the list.
"""
import os
import sys

sys.path.insert(0, "/root/triton-mi450/third_party/amd/python/examples/gluon")
sys.path.insert(0, "/root/triton-mi450/python/triton_kernels")

import moe_gfx1250 as M  # noqa: E402


def _b(name, default):
    return os.environ.get(name, default) not in ("0", "false", "False", "")


m = int(os.environ.get("MM_M", 256))
n = int(os.environ.get("MM_N", 256))
k = int(os.environ.get("MM_K", 512))
swiglu_env = os.environ.get("MM_SWIGLU", "1.1,1.4")
swiglu_opts = None if swiglu_env == "none" else tuple(float(x) for x in swiglu_env.split(","))

print(f"DRIVE correctness: m={m} n={n} k={k} fp8xmxfp4 swiglu={swiglu_opts} "
      f"gather={_b('MM_GATHER','0')} bias={_b('MM_BIAS','1')}", flush=True)

rc = 0
try:
    M.test_matmul(
        m=m, n=n, k=k, block_m=256, block_n=256, block_k=256,
        dtype_a="float8_e4m3fn", dtype_b="mxfloat4_e2m1",
        do_gather=_b("MM_GATHER", "0"), do_scatter=_b("MM_SCATTER", "0"),
        do_bias=_b("MM_BIAS", "1"), SCALE_PRESHUFFLING=_b("MM_PRESHUFFLE", "1"),
        swiglu_opts=swiglu_opts, num_buffers=int(os.environ.get("MM_NUM_BUFFERS", 2)),
        schedule=os.environ.get("MM_SCHEDULE", "baseline"),
        pingpong=_b("MM_PINGPONG", "0"), num_warps=int(os.environ.get("MM_NUM_WARPS", 4)),
    )
    print("DRIVE: PASS (correctness asserted)", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"DRIVE: FAIL/SKIP: {type(e).__name__}: {e}", flush=True)
    rc = 1

sys.stdout.flush()
sys.stderr.flush()
os._exit(rc)
