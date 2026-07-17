# #1885 — MoE gather-index `v_readfirstlane` on gfx1250 (the noalias contract)

Eliminating the in-loop `v_readfirstlane` that lift the TDM gather-index descriptor
from VGPR→SGPR every K-loop iteration in the aiter MoE GEMM kernels.

**This supersedes `/root/GOLDEN-1885-readfirstlane.md`, which reached the wrong
conclusion** (it claimed the fix needs a `uniformizeAddr` + `!invariant.load`
backend lowering). The verified conclusion below is: the fix is the **noalias
caller contract**; the backend lowering is redundant.

---

## TL;DR (verified)

1. **The fix is the `noalias` contract, not any backend lowering.** A wave-uniform,
   read-only gather-index load whose base pointer arg carries `llvm.noalias` is
   selected by ISel as a scalar `s_load` (hoisted to the prologue), so the
   per-iteration descriptor `v_readfirstlane` vanish. `readonly` is **not**
   propagated — LLVM's own `FunctionAttrs` inference already stamps it on read-only
   pointer args.
2. **The `LoadStoreOpToLLVM.cpp` s_load lowering (`uniformizeAddr` /
   `readFirstLanePtr` / `!invariant.load` / the gate) is redundant** and was
   removed. Toggling it changes nothing in-loop (A/not-A kill-switch); toggling
   `noalias` is what flips the churn.
3. **LLVM version is not the difference.** Same IR → same in-loop count on stock
   `llc` 62b7cf96 and 56421f92 (prefill 0/0). The reason GOLDEN "needed"
   `uniformizeAddr` is that it was developed on triton `9c795a41fc`, which predates
   the `noalias_args` frontend feature — so noalias-alone was not an option then.
4. **`noalias_args` had a propagation bug** (`JITFunction.__init__` never stored
   `self.noalias_args`), so `triton_kernels.specialize` silently dropped it for
   activation-fused kernels (`_matmul` → `_matmul_swiglu_fn`). Fixed by storing it.
5. **Decode residual (8 in-loop rfl) is a separate, LLVM-side problem** — the
   descriptor lift there IS loop-invariant but MachineLICM won't hoist a convergent
   `readfirstlane`. Needs the LLVM patch, see
   `../reproducers/amdgpu_readfirstlane_licm/`.

## Prefill results (stock LLVM 56421f92), before/after noalias

| kernel | BEFORE (no noalias) | AFTER (noalias) | index load AFTER |
|---|---|---|---|
| a8w4 gluon (`_moe_gemm_a8w4_prefill`) | 16 | **0** | 32× `s_load_u16` (prologue) |
| in-tree `moe_gfx1250.py` (`_matmul_swiglu_fn`) | 264 | **8** | `s_load_b512/b256/…` (prologue) |

- a8w4 AFTER: hot loop has **zero** `v_readfirstlane`.
- moe_gfx1250 AFTER: residual **8** = the `tdm.async_gather` descriptor built per
  pipeline buffer (`load_idx % NUM_BUFFERS`) — **loop-variant**, so neither noalias
  nor the MachineLICM patch removes it. A separate follow-up if 0 is required.

Mechanism, same source line before/after (moe_gfx1250.py:264 `gl.load(GatherIndx…)`):
`global_load_b128` (per-lane VGPR) → `s_load_b512` (scalar SGPR).

## Coordinates

- Ticket: AMD-Triton/triton-tickets#1885. PR: AMD-Triton/triton-mi450#120.
- Branch `users/jerryyin/moe-gather-sload-contract`:
  - `5d8d2ec91a` — original contract commit (had the redundant lowering).
  - `6dd79710f8` — "Reduce MoE gather s_load to the noalias contract" (removes the
    lowering + readonly propagation + kill switch; adds the `self.noalias_args` fix).
  - (unstaged at time of writing) comment trims + `noalias_args=["GatherIndx"]` on
    `moe_gfx1250.py`'s `_matmul`.
- The contract source annotations live in **aiter**
  (`_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py`, `noalias_args=["GatherIndx"]`)
  and in-tree `moe_gfx1250.py`.
- Stock LLVM prebuilts: `/root/.triton/llvm/llvm-{62b7cf96,56421f92}-*`; patched
  (MachineLICM hoist) tree: `/root/llvm-project` → `install/`.

## End-to-end micro-benchmark (runtime / TFLOPS + ATT)

Measures how much the contract changes *runtime*, on the tdm-fusion MoE GEMM, plus
ATT traces. Lives in `scripts/` + `results/`.

### Result (gfx1250, sliceNK, 5-rep medians)

| BN | baseline | PR (noalias) | speedup | codegen (`_matmul.amdgcn`) |
|----|----------|--------------|---------|----------------------------|
| 256 | 4.404 ms / 1975 TFLOPS | 4.358 ms / 1996 TFLOPS | **+1.06 %** | v_readfirstlane 75→44, s_load 10→14 |
| 512 | 4.123 ms / 2109 TFLOPS | 4.050 ms / 2148 TFLOPS | **+1.78 %** | (same kernel) |

Raw reps in `results/perf_reps.md`; driver snapshots in `results/{pr,baseline}_results.csv`
(cold-start, provenance only — use `perf_reps.md` medians). The −31 `v_readfirstlane`
is the same mechanism as the asm study above, seen end-to-end in the tdm moe kernel.

### Key decisions (why this config)
- **PR grafted onto tdm-fusion, not built standalone.** The benchmark config comes from
  `amd/kylewng/moe_shared_tdm_fusion` (tip `f6077ab09a`). baseline = tip; PR = tip + the
  2 PR commits cherry-picked. Net delta = exactly the noalias contract
  (`scripts/02_pr_noalias_contract.patch`). PR's own base (`ba4fd67b`) can't build here —
  its LLVM `56421f92` prebuilt 404s; the tdm tip pins the available `850a2b1b`.
- **`sliceNK`, not `sliceMNK`.** `run_moe_microbench.sh` uses `sliceMNK` +
  `partial_tdm/tdm_split/resolve`, but those knobs are sliceMNK-only and **`sliceMNK`
  GPU-faults in the tdm branch's own `test_matmul`** (WIP tip). `sliceNK` is stable
  (`validate_moe.py` → rel_err 0.0). The gather `s_load` is schedule-independent, so the
  delta is representative; absolute TFLOPS are sliceNK.
- **Harness forward-port** (`scripts/01_harness_intermediate_out_dtype.patch`): the tdm moe
  example predates the `intermediate_out_dtype` threading its `triton_kernels` now needs.
  Applied identically to both branches (dead for split_k==1), so it doesn't affect the delta.

### Exact bench command (per BN ∈ {256, 512})
```
python3 third_party/amd/python/examples/gluon/moe_gfx1250.py \
    -b 2048 -d1 2880 -d2 5760 -et 128 -ea 4 --x_dtype fp8 --w_dtype mx4 \
    --num_buffers 3 -a dispatch --num_warps 4 -bm 128 -bn <BN> -bk 256 \
    --schedule sliceNK --benchmark-mode eager --benchmark-num-iters 200
```
`-a dispatch` (gather) is required for the PR effect. GEMM shape M=262144 N=5760 K=2880.
Env (both configs): `HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1
TRITON_HIP_USE_EXPERT_SCHEDULING=1 TRITON_HIP_USE_COEXEC_SCHEDULER=1`.

### Reproduce (turn-key; some manual steps)
Prereqs: Triton at `~/triton`, AMD remote fetched, gfx1250, `~/.triton/llvm/llvm-850a2b1b-*`,
`rocprofv3` on PATH, ATT decoder lib (default `/root/rocm-systems/projects/rocprof-trace-decoder/build/lib`;
override `ATT_LIB` in `scripts/run_moe_att_bench.sh`), and `~/scripts/tools/gpu-lock`
(lock file `/data/lock/amd-gpu.lock`).
```bash
cd ~/triton
REPRO=~/scripts/triton/reproducers/issue-1885-moe-gather-noalias
# 1) build both bench branches (cherry-pick approach; handles the moe decorator conflict)
"$REPRO"/scripts/setup_branches.sh ~/triton
# 2) stage rocprofv3 ATT config at a results root
RESULTS=~/bench_moe_pr_vs_base; mkdir -p "$RESULTS"; cp "$REPRO"/scripts/att.json "$RESULTS"/
# 3) baseline: build + bench (perf run for TFLOPS + single-launch rocprofv3 ATT)
git checkout users/jerryyin/bench-tdmfusion-baseline && pip install -e . --no-build-isolation
"$REPRO"/scripts/run_moe_att_bench.sh baseline "$RESULTS"
# 4) PR: rebuild (only noalias C++ differs → fast) + bench
git checkout users/jerryyin/bench-tdmfusion-pr && pip install -e . --no-build-isolation
"$REPRO"/scripts/run_moe_att_bench.sh pr "$RESULTS"
# 5) stable perf: 5 reps/config/BN, take medians (see results/perf_reps.md for the loop)
# 6) correctness: cd third_party/amd/python/examples/gluon &&
#    PYTHONPATH=$PWD gpu-lock python3 "$REPRO"/scripts/validate_moe.py sliceNK None   # rel_err 0.0
# 7) mechanism: TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/ir <bench cmd w/o --benchmark-mode>;
#    grep -c v_readfirstlane and -cE '\ss_load' the dumped _matmul.amdgcn per config
```
ATT traces (raw `*.att` + decoded `ui_output_*`) land under `$RESULTS/{pr,baseline}_bn*/att/`.
They are **not** committed (large, regenerable) — scp them off the host as needed.

## Files here

- `scripts/` — the runtime/ATT benchmark: `setup_branches.sh` (construct the two bench
  branches via cherry-pick), `run_moe_att_bench.sh` (driver: perf + rocprofv3 ATT, under
  `gpu-lock`), `validate_moe.py` (assert_close vs torch ref), `att.json` (ATT config),
  `01_harness_intermediate_out_dtype.patch` + `02_pr_noalias_contract.patch`.
- `results/` — `perf_reps.md` (the record), `{pr,baseline}_results.csv` (driver snapshots).
- `gen_before_after.sh` — regenerate the BEFORE/AFTER assembly for both kernels
  into `asm/` on demand (toggles the aiter/example `noalias_args`, stock `llc`,
  reports in-loop rfl + hot-loop bounds). The raw `.s`/`.llir` dumps are not
  committed — run this to reproduce them; the results summary above is the record.
- `isolate.sh` — the experiments that establish the conclusions: readonly×noalias
  2×2 (noalias is the switch), and the same-IR old-vs-new `llc` check (LLVM is
  invariant).
- `ticket_route_repro/` — independent reproduction of the ticket's `uniformizeAddr`
  + `!invariant.load` route (`ticket_route.patch` + `verify.sh`): confirms it is a
  valid alternate route to the noalias contract (16→0 in-loop), not a false claim.
- `widen_sload/` — can the (now-`s_load`) gather-index loads be made wider? Root
  cause = AMDGPU has no wide sub-dword (i16) SMEM load (`subdword.ll` + `reproduce.sh`),
  so a8w4's uint16 index stays narrow; wide SMEM needs dword granularity. Includes the
  end-to-end `measure_a8w4.sh` and the full investigation `ledger.md`.

## Related tooling (elsewhere in ~/scripts/triton)

- `moe/ffm_verification/run_moe_gemm_ffm.py` — the FFM correctness/kernel runner
  (`--kernel {a4w4,a8w4} --backend {gluon,triton} --phase {prefill,decode}`).
- `reproducers/amdgpu_readfirstlane_licm/` — standalone LLVM reproducer for the
  decode residual (MachineLICM won't hoist convergent readfirstlane).
- `../compare_uniform_sload.sh` — **STALE**: uses the removed
  `TRITON_AMD_DISABLE_UNIFORM_SLOAD` kill switch. Superseded by `gen_before_after.sh`.
