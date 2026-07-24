STATUS: blocked
CONCLUSION: RT1 bails clean on the R1 blocker, re-verified LIVE today: under `am_fast_env.sh` bare `torch.cuda.is_available()` still crashes rc=11 / `signal 11` on `libdtif.so: undefined symbol: hsaKmtCreateQueueV2` (AM ROCm-7.13 DTIF vs host torch-2.11/ROCm-7.14 KMT-thunk ABI mismatch) BEFORE any Triton compile or kernel dispatch — so no AM cycle number for `_moe_gemm_a8w4_prefill` (contract ON vs OFF, any shape) can be produced on this host as configured. FFM control passes (gfx1250) on the same tree, isolating the fault to AM. NO fabricated numbers.

---

## Frozen experiment

- **Cell:** RT1 (Run 2, Phase C). Task: AM PREFILL runtime impact of the noalias contract for F1 (`_moe_gemm_a8w4_prefill`, with/without `noalias_args=["GatherIndx"]`) across >=3 small representative shapes, contract ON vs OFF, per-shape ON/OFF timing + delta% with rep count and spread.
- **Blocked-by:** R1 (AM timing harness) — `STATUS: blocked` (read on disk: `ledgers/R1.md`). Per PLAN Run-2 note ("If R1 (AM) reports blocked, its dependent RT cells must bail-clean … do not flail") RT1 must bail unless the blocker is lifted. I did NOT trust the ledger blindly — I re-verified the blocker live (Facts below).
- **Triton tree/SHA:** `/root/triton` (installed `triton` 3.8.0 resolves to this tree; branch `users/jerryyin/moe-gather-sload-contract`, #120 contract landed) — matches R1/S1. Not rebuilt, not modified.
- **AITer tree:** `/root/aiter`. **RT1 edited NO aiter source** — no kernel-source toggle was reached because init crashes upstream of compile. Tree left clean.
- **AM package:** `/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06` (ROCm 7.13.0a; DTIF), same package R1 used. `/am-ffm` is EMPTY on this host (confirmed `ls -la /am-ffm` = only `.`/`..`), so `run_on_model.sh --backend am` auto-detect would mis-target it — env scripts (`am_fast_env.sh`, `ffmlite_env.sh`, …) sourced directly from `$PKG` as R1 prescribed.
- **Host runtime:** torch `2.11.0+rocm7.14.0a20260623`, HIP `7.14.60850` (confirmed today) — the newer half of the ABI mismatch. 8× MI355X (gfx950) present.
- **Target of interest:** gfx1250 TDM gluon `_moe_gemm_a8w4_prefill` (cannot run on gfx950 — TDM is gfx1250 ISA — so AM is the only timing track per PLAN two-track strategy; FFM is correctness-only, no timing).
- **Metric sought (never obtained):** per-dispatch AM cycle count (`draw.log` DrawId→DrawDone clk-delta, corroborated by `perf_counters.csv` `cp_dispatch_busy_cycles`), contract ON vs OFF, several small shapes, median+spread. R1 proved this metric exists and is readable; the gap is purely device init.
- **Isolation:** `TRITON_CACHE_DIR=/tmp/tc-RT1`-class scratch; all probes in per-run `/tmp/am-rt1-*` dirs; never touched `~/.triton/cache`. No GPU-lock taken — AM is a CPU simulator and never reached dispatch (nothing to lock).

---

## HOW-facts (real evidence, re-verified today 2026-07-17)

### Fact 1 — the R1 blocker is CURRENT, not stale (env matches R1's frozen state exactly)
- `/am-ffm` = empty (`ls -la` shows only `.`/`..`, dirs dated Jun 26). AM package = `/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06` (present; env scripts `am_env.sh am_fast_env.sh am_profile_env.sh ffmlite_env.sh …`).
- Host torch = `2.11.0+rocm7.14.0a20260623`, HIP `7.14.60850`. This is the identical ROCm-7.13(AM) vs 7.14(host) split R1 documented.

### Fact 2 — `hsaKmtCreateQueueV2` has NO provider; it is an unsatisfiable import in libdtif.so
- Exhaustive `nm -D` over every `libhsakmt*`, `libhsa-runtime64*`, `libdtif.so*` under `$PKG` and `/opt/rocm`: **zero** definitions (`T`) of `hsaKmtCreateQueueV2` anywhere. Result line: `==> NO PROVIDER (T) of hsaKmtCreateQueueV2 anywhere scanned`.
- `nm -D $PKG/package/lib64/libdtif.so | grep hsaKmtCreateQueueV2` → symbol not listed as defined (it is an undefined import satisfied at load time only if the loader's HSA thunk provides it — nothing does). Matches R1 Fact 3.

### Fact 3 — LIVE probe: bare torch.cuda init under AM segfaults (reproduces R1 Probe C)
`source $PKG/am_fast_env.sh; python3 -c "import torch; print('CUDA_AVAIL', torch.cuda.is_available())"` in a fresh `/tmp/am-rt1-probe` dir:
```
=== rc=11 ===
GetExportAddress failed: /zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/package/lib64/libdtif.so: undefined symbol: hsaKmtCreateQueueV2
AbortHandler: signal 11
... (tail) ...
*I:  HydraPhase::Wait()   Phase 'reset' DONE(0)
AbortHandler: signal 11
```
The AM model engages (affinity pin, MT-config, `AmTestChip`, `HydraPhase reset DONE`) then crashes in the reset phase on the missing symbol — i.e. **before** any Triton compilation or kernel dispatch. This is upstream of aiter arch detection and upstream of the `noalias_args` toggle, so no contract-ON-vs-OFF comparison can even begin.

### Fact 4 — FFM control PASSES on the identical host/tree/torch (fault is AM-specific)
`source $PKG/ffmlite_env.sh; python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"`:
```
=== ffm control rc=0 ===
FFM_ARCH GPUTarget(backend='hip', arch='gfx1250', warp_size=32)
```
Same host, same Triton tree, same torch — only the backend env differs. FFM initializes Triton and reports the correct gfx1250 target. So the tree/Triton are healthy; the failure is strictly the AM DTIF↔HIP ABI mismatch, not a broken repro. (FFM gives correctness but no timing, per PLAN — it cannot substitute for the AM cycle number RT1 needs.)

---

## Refutation trail (did I try to knock the blocker down before accepting it?)

- **"Maybe R1's ledger is stale / another env variant works."** REFUTED — Fact 1 (env matches R1 exactly) + Fact 3 (live re-run under `am_fast_env.sh` reproduces the exact crash). Not stale.
- **"Maybe the symbol IS provided by some lib and it's an LD ordering problem."** REFUTED — Fact 2: exhaustive `nm -D` finds zero `T` providers of `hsaKmtCreateQueueV2` on the whole host/package. No LD_LIBRARY_PATH ordering can supply a symbol nothing defines.
- **"Maybe it's only the heavy MoE runner / aiter rocminfo that crashes; a bare probe is fine."** REFUTED — Fact 3 is the *barest possible* probe (`torch.cuda.is_available()`), which is upstream of aiter and of Triton kernel compile, and it still crashes.
- **"Maybe the tree/Triton is broken, not AM."** REFUTED — Fact 4: FFM control on the identical host returns rc=0 gfx1250.
- Not attempted (out of scope, Rule 4 — report, don't invent): installing a ROCm-≤7.13 torch to match the AM package, or obtaining a ROCm-7.14 AM package. These are the sanctioned unblock paths (see counter-experiment); RT1 does not fabricate them.

---

## CLAIM STATUS

**CLAIM (grounded, blocked):** RT1's deliverable — AM per-dispatch cycle timing for `_moe_gemm_a8w4_prefill` with the noalias contract ON vs OFF across small representative shapes — **cannot be produced on this host as configured.** The AM DTIF simulator (ROCm 7.13) is ABI-incompatible with host torch-2.11/ROCm-7.14: `libdtif.so`'s `hsaKmtCreateQueueV2` import is unsatisfiable, so torch/Triton device init segfaults (signal 11) during the AM reset phase, before any compile or dispatch. The AM *timing metric itself* exists and is readable (R1 Fact 1 proved this via HelloWorld: `draw.log` clk-deltas == `cp_dispatch_busy_cycles`); only device init blocks it. No number is fabricated in lieu of a measurement.

**Honest resolution status vs the "~1% microbench" claim RT1 was meant to rigorize:** UNMEASURED on AM. RT1 can neither confirm nor refute a ~1% prefill delta on AM. The stronger *asm/mechanism* evidence already exists (S4: contract flips the in-loop `v_readfirstlane` 16→0 and selects `s_load` on gfx1250; S5 ceiling), which is the non-execution proof PLAN relies on for the gfx1250 target — but a per-kernel AM cycle % is not obtainable here.

**Counter-experiment that would REFUTE this blocker (turn-key unblock, hand-off):**
Provide a torch built against ROCm ≤ 7.13 (matching the AM DTIF HIP ABI), OR an AM package built against ROCm 7.14, OR a `libhsakmt` exporting `hsaKmtCreateQueueV2` matching DTIF's expectation. Then re-run Fact-3's probe under `am_fast_env.sh`; if `CUDA_AVAIL True` (no signal 11), the blocker is lifted and R1's turn-key harness (below) yields the RT1 number directly.

---

## Turn-key harness — READY the moment the blocker is lifted (verbatim from R1, validated env)

```bash
PKG=/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06
RUNNER=/root/scripts/triton/moe/ffm_verification/run_moe_gemm_ffm.py
mkdir -p /tmp/am-rt1 && cd /tmp/am-rt1                 # AM dumps many files into cwd
export TRITON_CACHE_DIR=/tmp/tc-RT1 TRITON_DUMP_DIR=/tmp/td-RT1 TRITON_ALWAYS_COMPILE=1
export GPU_ARCHS=gfx1250                                # bypass aiter rocminfo (exit-11 under AM)
source $PKG/am_fast_env.sh                              # engages tb_am_rs64_fw, gfx1250-class, perf on
python3 $RUNNER --kernel a8w4 --backend gluon --phase prefill --batch 16 --experts 32 4
# metric: draw.log 'DrawId:N clk X' / 'DrawDone:N clk Y' -> cycles = Y - X per dispatch;
#         corroborate perf_counters.csv col model.gpu0.cp.cp_dispatch_busy_cycles.
# contract ON vs OFF: toggle noalias_args=["GatherIndx"] in aiter moe_op_gemm_a8w4.py (Python only, no rebuild).
# shape variety (>=3): e.g. (--batch 16 --experts 32 4), (--batch 32 --experts 32 4),
#   (--batch 16 dim1/dim2 doubled) — smallest that keeps K%256==0 (scale-swizzle) AND well-formed top-4 routing.
# reps: repeat >=5 in fresh scratch dirs; report median + min/max. AM deterministic-ish -> low spread expected.
```
Smallest representative shape rationale (from R1, unvalidated by execution): `dim1=256` = min multiple of 32*8 keeping the gfx1250 scale-swizzle K-loop real; `--batch 16` gives well-formed top-4 routing so GatherIndx is populated and the gather is exercised; experts must stay 32/4 (8/2 degenerates routing).

---

## Remaining unknowns / hand-offs

- **The RT1 number (per-shape ON/OFF AM cycles + delta%):** UNOBTAINED — blocked at HIP device init, never reached dispatch. Metric + shapes are solved; only the ABI blocker remains.
- **draw.log DrawId↔kernel mapping** for the multi-dispatch MoE forward (GEMM1/GEMM2/routing): untested — RT1's follow-up must identify which DrawId is the a8w4 prefill gemm (dispatch order or `--disp <regex>`).
- **AM wall-time per real MoE dispatch:** unknown; a real K-loop is far heavier than HelloWorld's 16 s — budget accordingly, keep shapes minimal / reps low.
- This blocker equally gates **RT2** (F1 decode AM). RT2 should read this + R1 and bail-clean the same way unless the ABI blocker is lifted.
