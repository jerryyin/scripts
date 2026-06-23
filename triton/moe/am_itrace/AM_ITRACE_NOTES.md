# AM itrace for a8w4 MoE GEMM1 (gluon vs triton) — workarounds & steps

Working notes for ticket **AMD-Triton/triton-mi450#56** (MoE perf bring-up on
triton). Goal: capture an instruction trace (itrace) of the a8w4 layer-1 GEMM for
the **gluon** (default gfx1250) and **triton** (`AITER_FORCE_TRITON=1`) backends
under the AM model, and compare them in ItraceViz.

Methodology per the ticket: **AM (`rocdtif-7.13-am+ffmlite-mi400-r4.05`) + itrace**.
ItraceViz: <https://github.com/AMD-Triton/ItraceViz>, cloned at `/root/ItraceViz`.

> Status: **pipeline proven end-to-end on a tiny shape.** The real ticket shape
> (decode M=128, N×K=7168×2048, 256/8) has not been traced yet — see
> [Open items](#open-items).

---

## TL;DR — the capture pipeline

```bash
# 0. one-time: itrace needs the m4 macro processor (see Gotcha 1)
apt-get install -y m4

# 1. enable itrace in the AM env (see Gotcha 5); backup kept at am_env.sh.bak.itrace

# 2. precompute routing + quantized weights under FFM (routing is on CPU; only
#    the weight quant runs on the model). Writes a .pt payload.
~/scripts/tools/run_on_model.sh --backend ffm -- \
    python3 precompute_routing.py --out moe.pt --shape 2048 7168 --experts 256 8 --batch 128

# 3. run GEMM1-ONLY under AM from the payload, per backend -> emits *.mon
GPU_ARCHS=gfx1250 ~/scripts/tools/run_on_model.sh --backend am -- \
    python3 itrace_gemm1_pre.py --backend gluon  --data moe.pt     # then --backend triton

# 4. extract one WGP and render the timeline HTML
grep -A1 WGP00 xcc0se0sa0_itrace_emu.mon > wgp0.txt
python3 /root/ItraceViz/gen_timeline.py wgp0.txt out.html

# 5. compare instruction mix across backends
python3 analyze_itrace.py xcc0se0sa0_itrace_emu.mon 0
```

Each AM run takes minutes (cycle-accurate). `precompute_routing.py` is run once
and its `.pt` is reused by both backends' AM runs.

---

## Why we can't just run the bench under AM

- The ticket's `bench_moe_gemm_a8w4.py` depends on **proton** (`triton.profiler`),
  which does **not** work under AM/FFM (rocprofiler-sdk can't enumerate the
  simulated agents). So we drive the kernels directly.
- itrace is an **AM model feature** (enabled via `am_env.sh`), not an FFM feature.
  So the trace must come from AM, not FFM.
- The aiter **routing kernel aborts the AM model** (see Gotcha 4). So routing
  cannot run under AM at all — we precompute it elsewhere and feed the GEMM1.

## "Profile the MoE kernel only, without routing" — how the repo does it

I checked how aiter's own bench/test isolate the GEMM:

- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4.py` (`bench_mlp_single_weight_init`)
  and `op_tests/triton_tests/moe/test_moe_gemm_a8w4.py` (`init_routing_data`)
  **both call the real `routing()` GPU kernel** and real `downcast_to_mxfp` for
  weights. **Neither fabricates weights, neither skips routing.** The bench
  "isolates layer1" only by naming proton scopes and filtering with
  `--op-regex '.*_layer1.*'` post-hoc — routing still executes.
- So there is no existing "skip routing" path. But aiter ships a pure-torch
  reference, **`routing_torch()`** (+ `compute_expt_data_torch()`) in
  `aiter/ops/triton/moe/moe_routing/routing.py`, that produces the identical
  `(RoutingData, gather_indx, scatter_indx)` with plain torch ops.

**Adopted solution:** build routing metadata with aiter's reference algorithm on
**CPU** (so it never touches the simulator and can't trigger the AM abort), and
quantize weights with the standard `downcast_to_mxfp` (same infra as bench/test —
**no fabricated weights**). One caveat: `routing_torch` uses `torch.histc`, which
has no int CPU kernel, so `precompute_routing.py:cpu_routing()` mirrors the exact
algorithm with `torch.bincount` instead. The GEMM1 kernel accepts the resulting
metadata unchanged (verified under AM).

---

## Gotchas & fixes (each one blocked AM until resolved)

### 1. AM hangs at startup — missing `m4`
`/am-ffm/package/bin/m4` is a symlink to `/usr/bin/m4`, which did not exist.
AM preprocesses `model.conf` with m4; the failure throws a `[FATAL]
processModelInitCommandLine` **and the SystemC threads don't tear down**, so it
looks like a futex/GIL deadlock (all threads asleep, ~0% CPU). It is not.
**Fix:** `apt-get install -y m4`.

Evidence: `sh: 1: /am-ffm/package/bin/m4: not found` →
`AM [FATAL] Assertion failed ... processModelInitCommandLine`.

### 2. `LD_PRELOAD: unbound variable`
`run_on_model.sh` runs under `set -u`; the vendor `am_env.sh` references
`$LD_PRELOAD` before it is set, which aborts on an unset variable. **Fixed
permanently:** `run_on_model.sh` now does `export LD_PRELOAD="${LD_PRELOAD:-}"`
before sourcing the env scripts, so no `LD_PRELOAD=` prefix is needed anymore.

### 3. `Get GPU arch from rocminfo failed`
AM has no working `rocminfo`, so aiter's arch detection (`_detect_native`)
aborts. **Fix:** `GPU_ARCHS=gfx1250` (aiter's `chip_info.get_gfx_custom_op_core`
honors it and skips rocminfo).

### 4. Routing kernel aborts the AM model (the core blocker)
Running aiter `routing()` under AM aborts the model on the routing dispatch:
```
On interface SPI_SQ_cmd, risky access of field scalar_l0_inv
Aborting due to ifrit error. ... signal 6 (SIGABRT)
```
This is a **hardcoded model-level assertion** (`*_AbortRiskyAccess` symbols in
`/am-ffm/package/lib64/*.so`) about reading an undefined hardware field
(`scalar_l0_inv`, scalar-L0 cache-invalidate) on the SPI→SQ command interface.
No suppression flag exists in the package conf or library strings.

Plain CUDA ops, a custom Triton vadd, the aiter gate GEMM (`gemm_a16w16`), and the
a8w4 GEMM1 itself do **not** abort — only routing does (capability ladder below).

**Fix:** never run routing under AM; precompute it (Gotcha-4 → the whole pipeline).

### 5. Enabling itrace in `am_env.sh`
Set `DtifExtraTestArgs=""` (drop `-no_itrace`, line ~79) and uncomment
`"test.enable_itrace=true"` / `"test.itrace_perf_detail=true"` (lines ~126–127).
Backup at `/am-ffm/am_env.sh.bak.itrace`. **itrace is currently ENABLED.**

### 6. Single simulated device — serialize runs
A `kill -9` of an FFM/AM run leaves the single simulated device briefly wedged;
the next run produces an empty log / never initializes. Let it settle (a few
seconds) and retry. Don't run two sims concurrently.

### AM capability ladder (how Gotcha 4 was localized)
`am_probe.py --rung N` climbs: 1 cuda op ✓ · 2 triton vadd ✓ · 3 aiter gate
GEMM ✓ · 4 **routing → SIGABRT** · 5 gluon GEMM1 (reached only via precompute).

---

## Working example (tiny shape) — exact commands & results

Shape: `dim1(K)=256 dim2(N)=512`, experts `8/8`, batch `64` → `block_m=64`.
(Chosen tiny to validate the *pipeline* fast; **not** representative — see caveat.)

```bash
# precompute (CPU routing + FFM weight quant) -> 631 KB payload, <1 min
~/scripts/tools/run_on_model.sh --backend ffm -- \
  python3 precompute_routing.py --out moe_tiny2.pt --shape 256 512 --experts 8 8 --batch 64

# AM GEMM1, per backend (each emits xcc0se{0,1}sa{0,1}_itrace_emu.mon)
GPU_ARCHS=gfx1250 ~/scripts/tools/run_on_model.sh --backend am -- \
  python3 itrace_gemm1_pre.py --backend gluon  --data moe_tiny2.pt
GPU_ARCHS=gfx1250 ~/scripts/tools/run_on_model.sh --backend am -- \
  python3 itrace_gemm1_pre.py --backend triton --data moe_tiny2.pt

# visualize + analyze WGP00
grep -A1 WGP00 xcc0se0sa0_itrace_emu.mon > wgp0.txt
python3 /root/ItraceViz/gen_timeline.py wgp0.txt out.html
python3 analyze_itrace.py xcc0se0sa0_itrace_emu.mon 0
```

Artifacts produced (under `/root/itrace_runs/`):
`am_gluon/gluon_wgp0.html`, `am_triton/triton_wgp0.html`, and the `.mon` traces.

### WGP00 instruction mix — gluon vs triton (tiny shape)

| metric (WGP00)      | gluon          | triton        |
|---------------------|----------------|---------------|
| instructions issued | 18,842         | 10,438        |
| TS span (cycles)    | 76,170         | 67,728        |
| matrix (wmma)       | 384  (2.0%)    | 64  (0.6%)    |
| vector (valu)       | 7,957 (42%)    | 5,441 (52%)   |
| scalar (salu/smem)  | 9,057 (48%)    | 4,073 (39%)   |
| lds (ds)            | 696            | 228           |
| wait/barrier        | 520            | 384           |

Distinctive gluon overhead: `s_set_vgpr_msb` ×3,336 (uses >256 VGPRs → high
register pressure), and 2× the f32 division/dequant sequence
(`v_div_scale/v_rcp/v_div_fmas/v_div_fixup`). Gluon also lands 6× more wmma tiles
on WGP00 (different work distribution across WGPs).

> ⚠️ **Caveat — do not draw production conclusions from this.** At this tiny size
> the GEMM is **overhead-dominated** (wmma is <2% of issued instructions). The
> real bottleneck (wmma/memory bound) only shows at the ticket shape. This run
> validates the *methodology*, not the answer.

---

## Scripts

In this directory (`am_itrace/`):

| file | role |
|------|------|
| `precompute_routing.py` | CPU routing (`cpu_routing`, mirrors aiter `routing_torch`) + fabricated mxfp4 weights + scale swizzle → `.pt` payload |
| `itrace_gemm1_pre.py`   | AM: load payload, fire a single a8w4 GEMM1 launch (gluon/triton) → `.mon` |
| `analyze_itrace.py`     | categorize per-WGP instruction mix + TS span from a `.mon` |
| `run_decode_itrace.sh`  | end-to-end, idempotent orchestration of the whole flow |

Shared, in the parent dir (`../`):

| file | role |
|------|------|
| `am_probe.py`           | AM capability ladder (cuda→triton→gate GEMM→routing→GEMM1) used to localize the routing abort |
| `lib_moe_ffm.py`        | shared input/quant/swizzle helpers (pre-existing) |

---

## Open items

1. **Real ticket shape not yet traced.** Need decode M=128, N×K=7168×2048, 256/8
   (and prefill M=16384 — likely too large for a usable single-WGP trace).
2. **Precompute cost.** FFM time was dominated by a fixed cost that barely scaled
   with experts (256 ≈ 15 min vs 32 ≈ 9 min). Strong evidence that cost was the
   **routing + gate GEMM kernels** (they scale with `n_gates=batch×act`, constant
   across those runs), which the CPU-routing rewrite now removes — so the real
   shape should precompute much faster. Needs measurement.
3. **Two SAs in the trace.** Each AM run emits `xcc0se0sa0`, `xcc0se1sa0` (~150 MB)
   and `xcc0se{0,1}sa1` (~0.6 MB). We analyze `se0sa0` WGP00; confirm that's the
   intended WGP for the comparison.
4. **Fairness of the comparison.** gluon vs triton land different tile counts on
   WGP00; for a clean per-WGP comparison, consider normalizing by tiles/wmma or
   comparing aggregate over all WGPs.
```
