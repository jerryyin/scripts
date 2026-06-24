# aiter MoE a8w4 / a4w4 GEMM on gfx1250 — FFM / AM / B0 tooling

Tooling for the aiter Mixture-of-Experts GEMM kernels (ticket
AMD-Triton/triton-mi450#56) on gfx1250, across the **FFM** and **AM** simulators and
**B0** physical hardware. Shared code lives at the top level; each environment has its
own folder:

```
moe/
├── README.md                 # this file (overview)
├── lib_moe_ffm.py            # SHARED lib: input build, scale swizzle, quant,
│                             #   dequant torch reference, comparison
├── precompute_routing.py     # SHARED: CPU routing + fabricated mxfp4 weights -> .pt
│                             #   payload (used by AM itrace AND B0 ATT)
├── run_a8w4_gemm1.py         # SHARED a8w4 GEMM1 launcher (AM/FFM/B0): --data/--build,
│                             #   --iters loop; os._exit under sim, normal exit on HW
├── am_probe.py               # SHARED diagnostic: AM capability ladder
│
├── ffm_verification/         # FFM: kernel correctness vs torch ref
│   ├── run_moe_gemm_ffm.py
│   └── check_proton_ffm.py
│
├── am_itrace/                # AM: instruction trace (itrace) of GEMM1
│   ├── analyze_itrace.py     #   .mon -> per-WGP instruction-mix breakdown
│   ├── run_decode_itrace.sh  #   end-to-end orchestration (uses ../run_a8w4_gemm1.py)
│   ├── AM_ITRACE_NOTES.md    #   generic AM-itrace procedure + every gotcha & fix
│   └── MOE_DECODE_ITRACE_CHRONICLE.md
│
└── b0_bringup/               # B0 hardware: rocprofv3 ATT of the GEMM
    ├── att_collect.sh        #   collect 4 decoded traces (thin wrapper on tools/prof.sh)
    └── README.md             #   workflow, version pinning, os._exit gotcha, findings
```

## Common setup

```bash
pip install psutil                 # required to import aiter
export AITER_HOME=/root/aiter       # only if aiter is not at /root/aiter
```

- Scripts read `AITER_HOME` (default `/root/aiter`) and add it to `sys.path`
  (aiter is not pip-installed).
- **Run from anywhere except the triton source tree `/root/triton`** (running
  there breaks `import triton.profiler` / `triton.language`).
- Simulator runs go through the canonical wrapper `~/scripts/tools/run_on_model.sh`
  (`--backend ffm` or `--backend am`). For AM, prefix `LD_PRELOAD= GPU_ARCHS=gfx1250`
  (am_env.sh reads `$LD_PRELOAD` under `set -u`; AM has no working `rocminfo`).

The kernels themselves: **a4w4** is pure Triton; **a8w4** dispatches to **gluon**
by default on gfx1250 and to the **triton** kernel when `AITER_FORCE_TRITON=1`
(which needs the CDNA4 scale swizzle, not the GFX1250 one). The drivers handle
this; see `lib_moe_ffm._swizzle`.

---

## Job 1 — FFM correctness verification (`ffm_verification/`)

Confirms each a4w4 / a8w4 kernel runs **correctly** under FFM against a
dequantized torch reference — without the proton layer the aiter bench scripts
use (proton doesn't work under FFM). Verifies the **full two-GEMM forward** (the
final scattered batch rows), not isolated GEMMs.

```bash
cd /root/scripts/triton/moe

# a4w4 (pure Triton)
run_on_model.sh --backend ffm -- python3 ffm_verification/run_moe_gemm_ffm.py --kernel a4w4 --phase decode
run_on_model.sh --backend ffm -- python3 ffm_verification/run_moe_gemm_ffm.py --kernel a4w4 --phase prefill

# a8w4 — gluon (default on gfx1250)
run_on_model.sh --backend ffm -- python3 ffm_verification/run_moe_gemm_ffm.py --kernel a8w4 --backend gluon  --phase decode
# a8w4 — triton (forced via AITER_FORCE_TRITON + CDNA4 swizzle)
run_on_model.sh --backend ffm -- python3 ffm_verification/run_moe_gemm_ffm.py --kernel a8w4 --backend triton --phase decode
```

Each prints e.g. `forward: PASS  ... cosine=0.999955` and `RESULT: PASS`
(non-zero exit on FAIL → CI-usable). Decode uses batch 64 (→ block_m 16);
prefill uses batch 2048 (→ block_m 128). Custom shapes via `--shape DIM1 DIM2`
and `--experts TOT ACT`.

**Result (all PASS):** a4w4 works out of the box; a8w4 gluon and a8w4 triton both
pass and agree (triton byte-identical to gluon given the CDNA4 swizzle).

Forcing triton for a8w4 needs the one-line env gate in aiter (branch
`users/jerryyin/moe-a8w4-force-triton-env`):
`use_gluon = get_arch()=="gfx1250" and os.environ.get("AITER_FORCE_TRITON","0")!="1"`.

`check_proton_ffm.py` is an orthogonal probe — exit 1 under FFM because
rocprofiler-sdk can't enumerate the simulated agents (why the bench scripts
can't run here).

### Caveats
- **Verify the full forward**, not isolated GEMMs (gather padding differs; only
  the final batch output lines up). `lib_moe_ffm.run_forward` does this.
- **Degenerate routing**: too few experts for the token count makes aiter's
  histogram not sum to `batch*n_expts_act`; the driver warns. Default 32/4 is
  well-formed.
- **FFM teardown**: FFM hangs on normal interpreter exit → the drivers call
  `os._exit()`. Do the same in any custom driver.
- The torch reference matmuls run on **CPU** on purpose (FFM simulates every GPU
  matmul instruction-by-instruction).

---

## Job 2 — AM instruction trace of GEMM1 (`am_itrace/`)

Captures an instruction trace (itrace) of the a8w4 **layer-1 GEMM** under AM and
compares gluon vs triton per-WGP. itrace is AM-only; the aiter routing kernel
aborts AM, so routing + weights are precomputed off-model and only **GEMM1** runs
under AM.

```bash
# one command does everything (idempotent: reuses any existing .pt / .mon / .html):
bash am_itrace/run_decode_itrace.sh
# defaults to a tractable decode shape (32 experts, block_m=16, K=2048, N=7168).
# For ticket-exact (very large, slow; gluon will FATAL — see below):
EXPERTS_TOT=256 EXPERTS_ACT=8 BATCH=128 bash am_itrace/run_decode_itrace.sh
```

Artifacts land in `/root/itrace_runs/decode_<backend>/` (per-WGP HTML timeline via
ItraceViz + `run.log`). `analyze_itrace.py <mon> <wgp>` prints the instruction-mix
breakdown.

To **reproduce the routing-kernel AM crash** (instead of avoiding it), run
`run_a8w4_gemm1.py --build`, which builds inputs inline on-device so `routing()`
dispatches under AM and aborts — see `am_itrace/AM_ITRACE_NOTES.md` §4.

**Key results (decode):** decode GEMM1 is **memory/addressing-bound**, not
compute-bound (wmma ≈ 2%); time goes to weight movement (`ds_load` /
`global_load_async_to_lds` / `s_wait_dscnt`) and ragged gather/scatter index
math. **Triton traces cleanly; gluon aborts AM** on its TDM activation gather
(`gl.amd.gfx1250.tdm.async_gather`, the `x` load — pinned to the crashing
instruction in the trace), which AM services as per-row direct copies and
overflows the async-copy tracker (depth not raisable enough) — so a steady-state
gluon trace needs B0 hardware or a deeper-tracker AM build.

See **`am_itrace/AM_ITRACE_NOTES.md`** (generic procedure + every gotcha: missing
`m4`, `LD_PRELOAD`, `GPU_ARCHS`, the routing abort, itrace env flags) and
**`am_itrace/MOE_DECODE_ITRACE_CHRONICLE.md`** (the full decode run log: every
hiccup, what was edited where and why — incl. `tcp_async_copy_depth` — and the
results).

`am_probe.py` (shared, top level) is the diagnostic that localized the routing
abort: an AM capability ladder (cuda → triton vadd → gate GEMM → routing →
GEMM1). Reusable when a new kernel/model breaks under AM.
