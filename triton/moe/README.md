# aiter MoE GEMM kernels under FFM (gfx1250)

Verify the aiter Mixture-of-Experts GEMM kernels behind
`bench_moe_gemm_a4w4.py` / `bench_moe_gemm_a8w4.py` run **correctly** under FFM,
for every (kernel, backend, phase) combination — without the proton profiling
layer the bench scripts use (which does not work under FFM).

## TL;DR results

All cells **PASS** under FFM on gfx1250 (full 2-layer forward vs a dequantized
torch reference, default shape `dim1=256 dim2=512`, `32/4` experts):

| kernel | backend | decode (block_m=16) | prefill (block_m=128) |
|--------|---------|---------------------|-----------------------|
| a4w4   | triton (only path) | PASS cos≈1.0000 | PASS cos≈0.99998 |
| a8w4   | gluon (default)    | PASS cos≈0.99996 | PASS cos≈0.99993 |
| a8w4   | triton (forced)    | PASS cos≈0.99996 | PASS cos≈0.99993 |

Key facts:
- **a4w4** is pure Triton already (no gluon path) — it works out of the box.
- **a8w4** dispatches to **gluon** by default on gfx1250. Forcing **triton**
  works too and produces output **byte-identical to gluon** (rel_err 0), *provided
  the scales are swizzled with the CDNA4 layout* (see below).
- The bench wrappers themselves do **not** run under FFM because of proton; drive
  the kernels directly (this is what the scripts here do).

## Files (orthogonal; shared logic lives in the lib)

| file | purpose |
|------|---------|
| `lib_moe_ffm.py` | Shared helpers: input build, scale swizzle, dequantized torch reference, comparison. Single source of truth — no duplication. |
| `run_moe_gemm_ffm.py` | Matrix driver: verify one (kernel, backend, phase) cell vs the reference. Non-zero exit on FAIL (CI-usable). |
| `check_proton_ffm.py` | Orthogonal probe: does Triton's `proton` profiler (rocprofiler-sdk) work here? Exit 0 = yes, 1 = no (FFM). |

## Setup

```bash
pip install psutil                      # required to import aiter
export AITER_HOME=/root/aiter           # only if aiter is not at /root/aiter
```

The scripts read `AITER_HOME` (default `/root/aiter`) and add it to `sys.path`
(aiter is not pip-installed). Run them from **anywhere except** the triton source
tree `/root/triton` (running there breaks `import triton.profiler`).

## The aiter change (one line, on a branch)

Forcing triton for a8w4 needs a switch in aiter. It lives on branch
`users/jerryyin/moe-a8w4-force-triton-env` (commit in `/root/aiter`), which adds
an env gate to the existing dispatch in `moe_op_gemm_a8w4.py`:

```python
use_gluon = get_arch() == "gfx1250" and os.environ.get("AITER_FORCE_TRITON", "0") != "1"
```

Default behaviour is unchanged. `run_moe_gemm_ffm.py --backend triton` sets
`AITER_FORCE_TRITON=1` and also swizzles scales with the CDNA4 layout that the
triton kernel requires (the gluon kernel uses the GFX1250 layout; feeding that to
the triton kernel yields NaN).

## Reproduction matrix — one command per cell

Decode uses batch 64 (→ block_m 16); prefill uses batch 2048 (→ block_m 128).

```bash
cd /root/scripts/triton/moe

# a4w4 (pure Triton)
python3 run_moe_gemm_ffm.py --kernel a4w4 --phase decode
python3 run_moe_gemm_ffm.py --kernel a4w4 --phase prefill

# a8w4 — gluon (the default path on gfx1250)
python3 run_moe_gemm_ffm.py --kernel a8w4 --backend gluon --phase decode
python3 run_moe_gemm_ffm.py --kernel a8w4 --backend gluon --phase prefill

# a8w4 — triton (forced via AITER_FORCE_TRITON + CDNA4 swizzle)
python3 run_moe_gemm_ffm.py --kernel a8w4 --backend triton --phase decode
python3 run_moe_gemm_ffm.py --kernel a8w4 --backend triton --phase prefill
```

Each prints a line like:

```
arch=gfx1250 kernel=a8w4 backend=triton phase=prefill batch=2048 experts=32/4 block_m=128
  forward: PASS  finite_frac=1.000 rel_err=0.01006 cosine=0.999926
RESULT: PASS
```

Custom shapes / expert counts (e.g. closer to gpt-oss):

```bash
python3 run_moe_gemm_ffm.py --kernel a8w4 --backend triton --phase prefill \
    --shape 256 512 --experts 128 4
```

## Check the proton profiler (why the bench scripts don't run here)

```bash
python3 check_proton_ffm.py    # under FFM: prints the rocprofiler-sdk failure, exit 1
```

Under FFM rocprofiler-sdk cannot enumerate the simulated agents, so
`proton.start(hook="triton")` aborts. The bench scripts depend on it for timing;
on FFM you can only do the functional verification above (FFM perf is meaningless
anyway).

## Notes / caveats

- **Verify the full forward, not isolated GEMMs.** The kernel pads the gather to
  `block_m`; `moe_gemm_torch` packs raw expert ranges. Intermediate row layouts
  differ, so per-GEMM tensors do not line up — only the final batch output is
  comparable. `lib_moe_ffm.run_forward` does this.
- **Degenerate routing.** Too few experts for the token count (e.g. `--experts 8 2`
  at batch 2048) makes aiter routing return a histogram that does not sum to
  `batch*n_expts_act`; the torch reference then desyncs and the driver prints a
  `WARNING` (the verdict is unreliable, not a kernel bug — the kernels still agree
  with each other). The default `32/4` is well-formed.
- **FFM teardown.** FFM hangs on normal interpreter exit, so the scripts call
  `os._exit()`. If you write your own driver, do the same or processes will zombie
  and contend for the single simulated device (causing apparent multi-minute hangs).
- The reference matmuls run on **CPU** on purpose: under FFM every GPU matmul is
  simulated instruction-by-instruction, so a GPU reference would dominate runtime.
```
