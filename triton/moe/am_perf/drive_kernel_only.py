"""Kernel-only driver for the moe_gfx1250 dispatch matmul (_matmul).

Builds inputs and launches ONLY the matmul kernel once -- no routing, no torch
reference, no assert -- so the AM/itrace trace is dominated by a single grid=1
kernel and `dispatch_durations.py` gives a clean per-kernel cycle count.

This is the TIMING driver. For correctness use drive_matmul.py under FFM.

Config via env (defaults = the b0 dispatch path: fp8 x mxfp4 + swiglu + bias):
    MM_M, MM_N, MM_K            problem shape         (128, 128, 2048)
    MM_BM, MM_BN, MM_BK         block sizes           (256, 256, 256)   keep 256: smaller
                                                      block_m breaks the pow2 preshuffle assert
    MM_DTYPE_A, MM_DTYPE_B      "float8_e4m3fn", "mxfloat4_e2m1"
    MM_GATHER, MM_SCATTER, MM_BIAS   "0"/"1"          (0, 0, 1)
    MM_PRESHUFFLE               "0"/"1"               (1)
    MM_SWIGLU                   "alpha,limit" or "none"   ("1.1,1.4")
    MM_NUM_BUFFERS              2/3/4                 (2)
    MM_SCHEDULE                 baseline/sliceK/sliceNK   (baseline)
    MM_PINGPONG                 "0"/"1"               (0)
    MM_NUM_WARPS                4/8                   (4)

Grid is grid_m*grid_n; with M<=block_m and N<=block_n and no gather you get
grid=1 (one workgroup -> small, fast trace). Pick MM_K large enough to exercise
the K-loop but small enough to keep AM tractable.
"""
import os
import sys

sys.path.insert(0, "/root/triton-mi450/third_party/amd/python/examples/gluon")
sys.path.insert(0, "/root/triton-mi450/python/triton_kernels")

import torch  # noqa: E402
import moe_gfx1250 as M  # noqa: E402


def _b(name, default):
    return os.environ.get(name, default) not in ("0", "false", "False", "")


m = int(os.environ.get("MM_M", 128))
n = int(os.environ.get("MM_N", 128))
k = int(os.environ.get("MM_K", 2048))
bm = int(os.environ.get("MM_BM", 256))
bn = int(os.environ.get("MM_BN", 256))
bk = int(os.environ.get("MM_BK", 256))
dtype_a = os.environ.get("MM_DTYPE_A", "float8_e4m3fn")
dtype_b = os.environ.get("MM_DTYPE_B", "mxfloat4_e2m1")
do_gather = _b("MM_GATHER", "0")
do_scatter = _b("MM_SCATTER", "0")
do_bias = _b("MM_BIAS", "1")
preshuffle = _b("MM_PRESHUFFLE", "1")
swiglu_env = os.environ.get("MM_SWIGLU", "1.1,1.4")
swiglu_opts = None if swiglu_env == "none" else tuple(float(x) for x in swiglu_env.split(","))
num_buffers = int(os.environ.get("MM_NUM_BUFFERS", 2))
schedule = os.environ.get("MM_SCHEDULE", "baseline")
pingpong = _b("MM_PINGPONG", "0")
num_warps = int(os.environ.get("MM_NUM_WARPS", 4))

print(f"DRIVE kernel-only: m={m} n={n} k={k} block=({bm},{bn},{bk}) "
      f"{dtype_a} x {dtype_b} gather={do_gather} scatter={do_scatter} bias={do_bias} "
      f"preshuffle={preshuffle} swiglu={swiglu_opts} nbuf={num_buffers} "
      f"sched={schedule} pingpong={pingpong} nwarps={num_warps}", flush=True)

torch.manual_seed(0)
device = "cuda"
a_dtype = M.DType(dtype_a)
b_dtype = M.DType(dtype_b)
c_dtype = a_dtype
is_not_ragged = not do_gather and not do_scatter
n_slices = 1 if is_not_ragged else 10
a_scale_preshuffling = preshuffle and not do_gather

a, a_scales, a_rm = M.make_random_tensor(
    shape=(m, k), n_slices=n_slices, dtype=a_dtype, device=device,
    ragged_dim=None if is_not_ragged else 0, transpose=False,
    squeeze_batch_dim=is_not_ragged,
    mxfp_dim=-1 if a_dtype.has_mx_scale else None,
    scale_hbm_swizzling=M.layout.make_default_matmul_mxfp8_act_scale_layout
    if a_dtype.has_mx_scale and a_scale_preshuffling else None,
)
b, b_scale_tri, b_rm = M.make_random_tensor(
    shape=(k, n), n_slices=n_slices, dtype=b_dtype, device=device, ragged_dim=None,
    transpose=True, squeeze_batch_dim=is_not_ragged,
    mxfp_dim=-2 if b_dtype.has_mx_scale else None,
    scale_hbm_swizzling=M.layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2, num_warps=num_warps)
    if preshuffle else None,
)
gather_indx = None if not do_gather else torch.randint(0, max(m, 1), (m,), dtype=torch.int32, device=device)
scatter_indx = None if not do_scatter else torch.randperm(m, dtype=torch.int32, device=device)
bias = None if not do_bias else torch.randn(b.shape[:-2] + b.shape[-1:], dtype=torch.float32, device=device)

fused_activation = None
if swiglu_opts is not None:
    fused_activation = M.FusedActivation(
        M.FnSpecs("swiglu", M.swiglu_fn, ("alpha", "limit"), reduction_n=2), swiglu_opts)

precision_opt = M.PrecisionConfig(
    flex_ctx=M.FlexCtx(M.InFlexData(), M.InFlexData(), M.OutFlexData()),
    acc_scale=1.0, out_dtype=c_dtype.torch_dtype,
    a_mx_scale=a_scales, b_mx_scale=b_scale_tri,
)

print("launching matmul...", flush=True)
tri_y, kh = M.matmul(
    a, b, bias, a_rm, b_rm, gather_indx, scatter_indx, precision_opt,
    fused_activation=fused_activation, num_buffers=num_buffers,
    block_m=bm, block_n=bn, block_k=bk, schedule=schedule,
    pingpong=pingpong, num_warps=num_warps,
)
torch.cuda.synchronize()
print("matmul done; out shape:", tuple(tri_y.shape), flush=True)
try:
    M.static_profile(kh)
except Exception as e:
    print("static_profile skipped:", e, flush=True)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
