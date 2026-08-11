# MoE in-loop `v_readfirstlane` — full evaluation + why the descriptor churn is an LLVM bug

Follow-up to #1885. You asked to keep the index `s_load` opt, add a descriptor
fix, prove `v_readfirstlane` is gone across all MoE kernels, and file the LLVM
issue. **Result: the index opt stays and is correct, but the remaining in-loop
`v_readfirstlane` is a genuine LLVM register-allocation bug that cannot be fixed
from triton IR** — so it cannot be made "gone in all" from a triton branch. Full
evidence below.

---

## 1. Evaluation across all MoE kernels (in-loop `v_readfirstlane`, FFM gfx1250)

Totals across all ~11 kernels the harness compiles (routing + helpers + gemm).
The constant **~3** is helper kernels unrelated to the gather. The a8w4 **gemm
itself** is broken out.

| kernel | phase | backend | opt OFF | opt ON | gemm-only OFF→ON | correctness |
|---|---|---|---|---|---|---|
| a8w4 | prefill | gluon | 3 | **19** | **0 → 16** | PASS |
| a8w4 | decode  | gluon | 11 | 11 | **8 → 8** | PASS |
| a8w4 | prefill | triton | 3 | 19 | 0 → 16 | FAIL* |
| a8w4 | decode  | triton | 11 | 11 | 8 → 8 | FAIL* |
| a4w4 | prefill | triton | 3 | 3 | opt doesn't fire | PASS |
| a4w4 | decode  | triton | 3 | 3 | opt doesn't fire | PASS |

\* The a8w4 **triton** backend fails correctness on FFM **independent of this
work** (gluon PASSes on the same branch); flagging separately, not caused here.

Triton in-tree MoE/gather kernels (`examples/gluon/05-moe-bmm1-fused-gather.py`,
`tutorials/gluon/09-tma-gather-scatter.py`) are **Blackwell/NVIDIA-only**
(`fp4_padded requires blackwell`), so they don't run on gfx1250; the a8w4
triton-backend is the triton-native MoE proxy and shares the identical descriptor
path.

Reading the table:
- The opt only fires for **a8w4** (it needs the `tt.readonly`/`tt.noalias`
  contract; a4w4 has no annotations → 3=3).
- The opt **regresses a8w4 prefill** (0→16 on the gemm) and is **neutral on
  decode** (8→8).
- **In-loop `v_readfirstlane` is never 0 anywhere** — decode and the helper
  kernels churn with the opt OFF entirely.

---

## 2. Why there is no triton-side descriptor fix (4 attempts)

The churning readfirstlanes lift the TDM gather **descriptor row-index groups**
(`<4xi32>`) from VGPR→SGPR for the scalar `tensor_load_to_lds`. All four fixes
below were built and measured; all failed, each for a principled reason:

| # | fix | result | why it fails |
|---|---|---|---|
| 1 | IR `readfirstlane` on descriptor dwords | 0→16 in *disabled* (regressed) | `llvm.amdgcn.readfirstlane` is **convergent**; LICM cannot hoist it, so it pins in-loop |
| 2 | `addrspace(4)` (constant) on the index load | 16 (no change) | the load became uniform (64 `addrspace(4)` in IR) but `insertelement` is **always VGPR** on AMDGPU regardless of input uniformity |
| 3 | scalar `i128` + `bitcast` (avoid insertelement) | 16 (no change) | **LLVM canonicalizes it back to `insertelement`** — 0 `i128` survive in the emitted LLIR; `%271 = insertelement <4xi32> …` |
| 4 | `addrspace(4)` + `i128` combined | 16 (no change) | same canonicalization; ISel still builds VGPR + readfirstlane |

Whatever descriptor IR triton emits, LLVM normalizes it to `insertelement`→VGPR
before ISel, and the RA then rematerializes the VGPR→SGPR copy in-loop under SGPR
pressure. Correctness stayed PASS for all attempts. All reverted; the branch
carries **only the index opt**.

---

## 3. Proof the churn is LLVM, not triton or the opt

1. **It appears with the opt entirely OFF.** a8w4 **decode baseline** = 8 in-loop
   readfirstlane, no opt involved. The a4w4/helper baseline = 3. So it is not
   caused by the opt in general.
2. **The descriptor lowering code is byte-identical** across triton versions
   (`fillGatherScatterChunk`/`packIndices`), and the kernels are structurally
   identical.
3. **The behavior flips with the LLVM pin.** On LLVM `62b7cf96` (ticket base) the
   a8w4 prefill baseline had **8 in-loop** (reproduced); on `56421f92` (current)
   it has **0**. The `RegionSuccessor` API was refactored between the pins
   (neither source cross-compiles), confirming a substantial LLVM change.
4. **The opt only tips a balanced case over the edge.** Moving the index into
   SGPRs (and coalescing to `s_load_b128`) raises SGPR pressure just enough for
   the RA to rematerialize the loop-invariant descriptor copy in-loop
   (prefill 0→16). Register totals are identical (sgpr 106 / vgpr 1023 / spill 8)
   — a scheduling flip, not spilling.

Mechanism, one line: ISel builds the wave-uniform `<4xi32>` descriptor in VGPR
and inserts a VGPR→SGPR `v_readfirstlane` for the scalar `tensor_load_to_lds`;
MachineLICM/RA **rematerializes that convergent copy inside the K-loop** instead
of keeping it live in the preheader.

---

## 4. LLVM issue (draft — ready to file; needs a target)

**Title:** [AMDGPU] Loop-invariant VGPR→SGPR `v_readfirstlane` for a scalar
`tensor_load_to_lds` operand is rematerialized in-loop under SGPR pressure

**Summary:** A wave-uniform `<4xi32>` built by `insertelement` and consumed by a
scalar-operand intrinsic (`llvm.amdgcn.tensor.load.to.lds`) is materialized in
VGPRs; the VGPR→SGPR `readfirstlane` copies ISel inserts are loop-invariant but
get rematerialized inside the loop rather than hoisted to the preheader once SGPR
pressure rises. Register totals are unchanged — purely a rematerialization/RA
scheduling choice. Regressed between LLVM `62b7cf96`→`56421f92`.

**Minimal repro assets I have:** the exact `.llir` (uniform index → `insertelement`
descriptor → `tensor.load.to.lds`) and `.amdgcn` (16× in-loop `v_readfirstlane`
feeding `tensor_load_to_lds s[..],s[..],s[40:43],s[52:55]`), plus the opt-off
version (0 in-loop). In `/root/uniform_sload_compare.prefix-backup/` and
`/root/repro_baseline/`.

**Where to file?** Upstream `llvm/llvm-project`, or the internal AMD tracker, or a
`triton-tickets` follow-up to #1885 — tell me and I'll file it (I did not
auto-post; filing is outward-facing).

---

## 5. Recommendation
- **Keep the index opt** (as you decided) — it is correct and it is the right fix
  on toolchains where the baseline still uses the wasteful `global_load_u16`
  (e.g. the ticket's `62b7cf96`).
- **But gate it to fire only where it doesn't regress.** On `56421f92` for a8w4
  prefill it converts a clean baseline (0) into 16 in-loop readfirstlane for no
  benefit (the baseline already `s_load`s the index). Suggest gating on "baseline
  does not already scalarize" or excluding indices whose sole consumer is a TDM
  gather descriptor.
- **The descriptor churn itself must be fixed in LLVM** (§4) — it is present in
  decode with no opt, so no triton branch can make readfirstlane "gone in all".

## Branch / git status
`users/jerryyin/moe-tdm-descriptor-sgpr` = contract HEAD + index opt only (all
four descriptor experiments reverted). **Nothing new to commit** beyond the
index opt that already lives on the contract branch. Say the word and I'll push
the branch (index opt) and/or file the LLVM issue where you want it.
