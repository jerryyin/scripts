# Overnight plan — descriptor churn, combined eval, ticket reproduction

Context: on the current base (`shared/gfx1250` + migrated aiter), the a8w4 prefill
K-loop has **0** in-loop `v_readfirstlane` with our index-`s_load` opt OFF, and
**16** with it ON. Investigation showed the 16 are **not** the index (which is
correctly `s_load`) — they are the **TDM gather descriptor** groups
(`<4xi32>`/`<8xi32>`, built via `insert_element` in `TDMUtility.cpp`) that the
scalar `@llvm.amdgcn.tensor.load.to.lds` needs in SGPRs. The descriptor lives in
VGPRs and is readfirstlane'd to SGPRs; its loop-invariant part gets **hoisted**
when the opt is off, **rematerialized per-iteration** when on. The regression is
therefore a *descriptor-lowering* issue that our index opt merely perturbs.

Branches will be created off `users/jerryyin/moe-gather-sload-contract` (triton)
unless a question below redirects me.

---

## Part 1 — Fix the descriptor VGPR→SGPR churn

**Goal:** the TDM gather/scatter descriptor's loop-invariant groups stay in SGPRs
(built scalar and/or hoisted), so `tensor_load_to_lds` needs no per-iteration
`v_readfirstlane`, independent of the index opt.

**Approach (in order, stop when one works & verifies):**
1. Pin *why* the backend VGPR-materializes the descriptor: dump pre-RA LLVM IR /
   `-print-after` around the AMDGPU rematerialization + uniformity passes; confirm
   the `<Nxi32>` group is treated as divergent because `insert_element` mixes
   loop-invariant (base, row indices) with per-iteration (`advanceGlobalAddr`
   column offset) dwords.
2. Fix candidates:
   - (a) Split the descriptor into a **loop-invariant** group (base + row-index
     dwords, built/hoisted once, provably uniform) and the **per-iteration**
     column update, so the backend keeps the invariant part in SGPRs.
   - (b) Build the invariant group from uniform scalars so uniformity analysis
     keeps it in SGPRs (avoid the VGPR `insert_element` round-trip), or emit an
     explicit `readfirstlane`-once + reuse.
   - (c) If ISel still won't scalarize, materialize the invariant group directly
     into SGPRs at the lowering (mirror the index `readFirstLanePtr` idea for the
     descriptor group).
**Scope note:** `TDMUtility.cpp` is shared by all TDM ops (copy/gather/scatter,
load/store). Any change is verified against the full AMD TDM lit suite before
trusting it.

**Verify:** AMD conversion lit (`tritongpu_tdm_to_llvm`, `invalid`, the sload
tests) all green; a8w4 prefill e2e PASS (rel_err ~0.010); `compare_uniform_sload.sh`
shows in-loop `v_readfirstlane` ≈ 0 with the descriptor fix.

---

## Part 2 — Evaluate descriptor-fix vs. +index-`s_load`, end to end

Extend `compare_uniform_sload.sh` into a 2x2 (or 3-way) matrix and dump IR per
cell:
| config | index opt | descriptor fix |
|---|---|---|
| baseline | off | off |
| index-only | on | off |
| desc-only | off | on |
| both | on | on |

Metrics per cell (a8w4 prefill, and decode if it triggers): **in-loop
`v_readfirstlane`**, total `v_readfirstlane`, `global_load` count, `s_load`
count/width, sgpr/vgpr/spill, correctness (rel_err/cosine). Goal: show whether
"both" is strictly better than baseline on the hot loop, and whether index-`s_load`
still adds value once the descriptor is fixed.
*Perf caveat:* FFM is a functional simulator — I can give static instruction/loop
metrics, not cycle counts, unless a perf harness exists (see Q3).

---

## Part 3 — Faithfully reproduce the ticket's original observation

**Sub-goals:**
- (a) **Observe** the recorded baseline in-loop `v_readfirstlane`. First, download
  the IR tarball kwang102 attached to the ticket (`IR_moe_gluon_4x512x1024_…`) via
  `~/scripts/tools/download_github_attachment.py` and inspect it directly — that is
  the faithful recorded artifact, no rebuild needed.
- (b) **Check the descriptor pattern** in that recorded IR: is the VGPR→SGPR
  descriptor churn present there too, or is it new to the current base?
- (c) **Rebuild** the baseline (triton + aiter) at the ticket-era state, reproduce
  the in-loop `v_readfirstlane`, then apply the index-`s_load` fix and see whether
  it removes them *there* (where the index may still be a per-lane vector load, vs.
  the current base where the backend already scalarizes it).

**Confounder (called out by you):** the aiter kernel was migrated to the pure
`async_gather` API (commit `482cd32`). Faithful reproduction needs the *pre-migration*
aiter paired with a compatible triton. I'll reconstruct the pair (see Q2) and
document exactly which commits I used.

---

## Deliverable
A single markdown (`/root/overnight_results.md`) answering all three: what the
descriptor fix does, the combined-eval matrix, and the ticket-reproduction
findings (baseline churn present? index fix effective there? descriptor problem
present there?), each backed by concrete IR/asm counts and the commits used.

## Decisions (locked by user)
- **Git:** each part on a **new** branch, **commit + push** (jerryyin). Do **not**
  commit to the existing contract branches at all.
- **Part 1 scope:** restrict the descriptor fix to the **gather/scatter** path;
  do not touch the shared copy/load descriptor code.
- **Part 2 acceptance criteria:** inspect the assembly and confirm **no in-loop
  `v_readfirstlane`** in the a8w4 hot loop. That is the pass/fail bar.
- **Part 3:** use the **exact branch + commit the ticket records** as the original.
  Reproduce it; if reproduced, determine precisely **what changed** between then
  and now that introduced the new (descriptor) readfirstlane regression. Account
  for the aiter migration as a separate variable.
- Evaluation = static asm/loop metrics + correctness (FFM is functional; no cycles).
