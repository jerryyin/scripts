# LLVM knob exploration for the TDM descriptor `v_readfirstlane` churn

Goal: find a minimal LLVM knob (esp. recent experimental gfx1250 work — coexec
scheduler, etc.) that removes the in-loop descriptor `v_readfirstlane`, then
re-evaluate all kernels.

**Result: no LLVM knob eliminates it.** ~25 knobs tested. GlobalISel is the only
lever (16→7 prefill, 8→7 decode) but it **crashes on WMMA** so it is unusable for
any real MoE kernel. The churn is ISel operand-legalization, not scheduling/RA.

Method: fast iteration with `llc` on the dumped `.llir` (reproduces the full-run
counts: prefill opt-on = 16, opt-off = 0; decode = 8), plus end-to-end checks
through triton's flag plumbing (`flags` → `setLLVMOption`, and the
`TRITON_HIP_USE_*` knobs).

---

## 1. Knobs tested (a8w4 prefill, opt ON; baseline = 16 in-loop rfl)

| category | knob(s) | in-loop rfl |
|---|---|---|
| **coexec / expert scheduler** (as asked) | `TRITON_HIP_USE_COEXEC_SCHEDULER`, `TRITON_HIP_USE_EXPERT_SCHEDULING`, `-amdgpu-sched-strategy=coexec`, `-mattr=+coexec-friendly-isel`, `-amdgpu-coexec-friendly-isel` | **16** |
| scheduler strategies | `max-ilp`, `max-memory-clause`, `iterative-ilp`, `iterative-minreg`, `iterative-maxocc`, `-amdgpu-schedule-metric-bias=0/100` | **16** |
| reschedule stages | `-amdgpu-disable-clustered-low-occupancy-reschedule`, `-amdgpu-disable-unclustered-high-rp-reschedule` (each + both) | **16** |
| RP trackers / thresholds | `-amdgpu-use-amdgpu-trackers`, `-amdgpu-vgpr-excess-threshold-percent=100`, `-amdgpu-lirp-vgpr-reduction=0` | **16** |
| machine sink / LICM | `-disable-machine-sink`, `-disable-postra-machine-sink`, `-disable-machine-licm`, `-sink-insts-to-avoid-spills=false` | **16** |
| register allocation | `-sgpr-regalloc=greedy/basic/fast`, `-vgpr-regalloc=greedy/basic/fast` (and combined) | **16** |
| uniformity / scalarization | `-amdgpu-enable-uniform-intrinsic-combine`, `-amdgpu-scalarize-global-loads`, `-amdgpu-scalar-ir-passes` (each + combined) | **16** |
| kernarg preload | `-amdgpu-kernarg-preload-count=16` | **16** |
| **GlobalISel** | `-global-isel` (+ any of the above) | **7** ⬅ only mover |

Decode (opt-independent, baseline 8): SelectionDAG = 8, GISel = 7. No knob → 0.

---

## 2. Why GlobalISel is not usable

GISel lowers the uniform `<4xi32>` descriptor into SGPRs more often (16→7), but
enabling it end-to-end through triton **aborts**:

```
LLVM ERROR: unable to translate instruction:
  call llvm.amdgcn.wmma.f32.16x16x32.bf16 …
```

Every MoE gemm uses WMMA; GISel has no legalization for the gfx1250 WMMA
intrinsics. So GISel cannot compile these kernels at all, and even where it does
it only reaches 7, not 0.

---

## 3. Why no SelectionDAG knob can fix it (mechanism)

The in-loop `v_readfirstlane` are **ISel operand-legalization copies**, not a
scheduling or RA artifact:
- `llvm.amdgcn.tensor.load.to.lds` requires **SGPR** operands for the descriptor
  row-index groups.
- Those groups are `insertelement <4 x i32>` built from a **global load**, which
  SelectionDAG's divergence analysis marks **divergent** → the vector lands in
  **VGPR** → ISel must insert VGPR→SGPR `v_readfirstlane` at the operand.
- The intrinsic (and thus the copies) sit inside the K-loop and `readfirstlane`
  is **convergent**, so nothing can hoist them.

No scheduler/RA/remat/sink/LICM knob changes this — they act *after* the copies
already exist. The only fix is to make the descriptor **uniform/SGPR at ISel**,
which is exactly what GISel does better (7) and what SelectionDAG has no knob for.
Confirmed: disabling machine-sink/LICM/remat/spill-sinking leaves it at 16.

---

## 4. Corroborating IR finding

The clean, knob-free way to 0 in-loop rfl on prefill is the **opt-off** path: the
index is emitted as per-lane `load <1 x i16>` and the **backend natively
scalarizes** it to `s_load_u16`, keeping the descriptor scalar (0 in-loop). Our
opt's explicit `readfirstlane(ptr)` + coalesced `s_load_b128` is what produces
the divergent-vector descriptor the backend then round-trips. On this toolchain
the backend already does the ticket's job; the opt fights it.

Decode churns (8) even opt-off — that path does not native-scalarize, and no knob
recovers it.

---

## 5. Re-evaluation of all kernels

No knob to apply (none work), so the numbers are unchanged from the prior sweep
(see `/root/moe_readfirstlane_eval.md`). Re-run confirms (`/root/reeval_all.log`):

| kernel | phase | opt OFF | opt ON |
|---|---|---|---|
| a8w4 | prefill | 3 (gemm 0) | 19 (gemm 16) |
| a8w4 | decode  | 11 (gemm 8) | 11 (gemm 8) |
| a4w4 | prefill/decode | 3 | 3 (opt doesn't fire) |

---

## 6. Bottom line / recommendation
- **There is no minimal LLVM knob workaround** — including the coexec/expert
  experimental schedulers you flagged. The churn is baked into SelectionDAG's
  divergent-vector → scalar-intrinsic legalization.
- **GlobalISel is the only path that reduces it**, and it can't compile MoE
  (WMMA). If AMD wants a knob-based fix, the actionable upstream ask is either
  (a) teach SelectionDAG to select a uniform `<4xi32>` feeding
  `tensor.load.to.lds` into SReg (no readfirstlane), or (b) add GISel WMMA
  legalization so `-global-isel` becomes usable (still only 7, not 0).
- Practical path remains: **gate the opt** to fire only where the backend does
  not already native-scalarize the index (i.e. old toolchains / kernels that
  still emit `global_load_u16`), so it never regresses the cases the backend
  already handles.
