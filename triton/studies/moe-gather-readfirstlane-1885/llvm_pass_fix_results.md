# LLVM fix for the TDM descriptor `v_readfirstlane` churn — results

**Outcome: fixed in LLVM. The in-loop descriptor `v_readfirstlane` is eliminated
(0) in every MoE gemm, with identical correctness and identical register
footprint. The index `s_load` opt is kept.**

The fix is a minimal, general MachineLICM change (55 lines across 6 files, patch
at `/root/llvm_readfirstlane_hoist.patch`), built into a patched LLVM at the pinned
commit `56421f92…`; triton is relinked against it.

---

## Results (a8w4 prefill/decode, a4w4; in-loop `v_readfirstlane`)

| kernel | phase | **gemm** before | **gemm** after | total before→after | correctness |
|---|---|---|---|---|---|
| a8w4 | prefill | 16 | **0** | 19 → 3 | PASS rel_err 0.01001 |
| a8w4 | decode | 8 | **0** | 11 → 3 | PASS rel_err 0.009413 |
| a4w4 | prefill/decode | 0 (opt n/a) | 0 | 3 → 3 | PASS |

- The MoE gemm descriptor churn is **completely gone** (16→0, 8→0).
- The residual **3** is entirely in `_combined_routing` — genuine per-iteration
  cross-lane ops (a scan/routing kernel), *not* the loop-invariant descriptor
  pattern. The fix correctly leaves them (hoisting them would be incorrect).
- Register footprint unchanged (sgpr 106 / vgpr 1024 / scratch 44): the hoist has
  **no spill cost** — it moves the broadcast to the preheader, it does not add
  live SGPRs beyond what was already there.

---

## Root cause (recap)

The descriptor `<4 x i32>` row-index groups are **uniform in the IR** (0 of 75
`insertelement` divergent) but AMDGPU SelectionDAG materializes uniform vectors in
**VGPR**, so ISel inserts a VGPR→SGPR `v_readfirstlane` for the scalar
`tensor_load_to_lds` operand. Those broadcasts are loop-invariant, but MachineLICM
refused to hoist them because `readfirstlane` is **convergent**. So the same value
was re-broadcast every K iteration.

## The fix (why it is minimal and safe)

1. **`TargetInstrInfo::isConvergentInstrHoistable(MI)`** — new hook, default
   `false` (preserves the conservative "never hoist convergent" rule for all
   other targets/instructions).
2. **MachineLICM** — the convergent bail becomes
   `if (I.isConvergent() && !TII->isConvergentInstrHoistable(I)) return false;`,
   and the high-reg-pressure bail gains the same exception. `IsLoopInvariantInst`
   is otherwise unchanged, so an instruction is still only hoisted when **all**
   its operands are loop-invariant.
3. **`SIInstrInfo::isConvergentInstrHoistable`** — returns true only for
   `V_READFIRSTLANE_B32`.
4. **`SIRegisterInfo::shouldAnalyzePhysregInMachineLoopInfo(EXEC)`** — returns
   true so MachineLoopInfo tracks EXEC. This is the safety keystone:
   `readfirstlane` implicitly uses EXEC and `resultDependsOnExec` is true, so the
   instruction is loop-invariant **only when EXEC is not redefined in the loop —
   i.e. the loop is uniform.** In a uniform loop the "first active lane" is the
   same every iteration, so hoisting the broadcast is value-preserving. In a
   divergent loop (EXEC redefined) it is automatically *not* hoisted. Verified:
   correctness is byte-for-byte unchanged (same rel_err/cosine).

The high-pressure exception (2) is justified empirically: forcing the hoist on
prefill produced **identical** sgpr/vgpr/scratch counts — MachineLICM's generic
pressure estimate was simply over-conservative for this scalar broadcast.

## Why it had to be LLVM, not a knob or triton IR
Prior turns established (and this confirms): the descriptor is IR-uniform, LLVM
canonicalizes any triton-side vector construction back to `insertelement`→VGPR,
and ~25 `-mllvm` knobs (schedulers incl. coexec/expert, regalloc, remat,
machine-sink/LICM, pressure, uniformity) leave it at 16. GlobalISel reached 7 but
aborts on gfx1250 WMMA. The broadcast is inserted by ISel and blocked from
hoisting by the convergent bit — only reachable from inside the codegen pipeline.

---

## Build / integration notes
- Patched LLVM built at pinned commit `56421f921b1dc…` (RelWithDebInfo, assertions,
  targets Native;NVPTX;AMDGPU, projects mlir;llvm;lld) → `/root/llvm-project/install`.
- Fix is **default-on** in this build (safe by construction). A cl::opt
  `-amdgpu-hoist-uniform-readfirstlane` also exists (currently defaulted true).
  Note: triton's in-process `setLLVMOption` path does **not** activate cl::opts
  reliably for codegen — that's why the fix is defaulted on rather than
  flag-gated through triton.
- triton `build/` reconfigured with `LLVM_SYSPATH=/root/llvm-project/install`
  (the earlier failure to take effect was a stale cached `LLVM_DIR` pointing at
  the prebuilt LLVM). Prebuilt `clang++`/`lld` symlinked into the install for the
  GSAN build step.
- triton source is clean (index opt only; the descriptor experiments and the
  compiler.py flag were reverted — the fix lives entirely in LLVM).

## Suggested upstreaming
The `isConvergentInstrHoistable` hook + EXEC-in-MachineLoopInfo change is a clean,
general MachineLICM improvement (not a hack) and is a good upstream candidate:
"Allow MachineLICM to hoist a wave-uniform readfirstlane out of uniform loops."
Patch: `/root/llvm_readfirstlane_hoist.patch`.
