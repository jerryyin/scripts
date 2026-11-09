# GOLDEN — #1885 MoE gather-index `v_readfirstlane` (definitive)

This supersedes every earlier note in `/root/*.md`. Those are wrong in at least one
way each (see "Corrections"). This is the verified, reconciled picture.

Everything below is measured, not remembered. Method that settles most questions:
compile a fixed `.llir` with **both** LLVM pins' `llc` and compare — that isolates
the LLVM backend from triton's IR emission (no triton rebuild needed).

---

## TL;DR

1. **There is no LLVM backend regression.** For any *fixed* IR, `llc @ 62b7cf96`
   and `llc @ 56421f92` emit the **same** in-loop `v_readfirstlane` count. The
   differences people saw came from different *triton IR*, not the LLVM bump.
2. **The a8w4 *prefill* churn (16) is a self-inflicted triton regression.** The
   ticket's original opt used `uniformizeAddr` + `!invariant.load` and got
   **prefill = 0**. The contract-branch rewrite swapped that for a *layout-delta
   scalarization* and *dropped* `!invariant.load` → **prefill = 16**. Restoring
   `uniformizeAddr` + `!invariant.load` → **prefill = 0 on stock llc** (no LLVM
   change), opt fires, correct, identical registers.
3. **Both pieces of the ticket opt are load-bearing:** `uniformizeAddr` (a
   *vectorized* load from a readfirstlane'd address) **and** `ld.setInvariant(true)`.
   Dropping either, or scalarizing per-element, churns.
4. **The a8w4 *decode* residual (8) is a genuine, longstanding LLVM ISel
   missed-opt** — present in the decode *baseline* (no opt at all) and *not*
   fixable by `uniformizeAddr`. Only the MachineLICM hoist patch reaches it
   (8→0). So the LLVM patch is **not** useless; it's the complementary safety net.
5. **Together → 0 everywhere:** `uniformizeAddr`+invariant (triton, prefill) +
   MachineLICM hoist (LLVM, decode).

---

## Coordinates

- **Ticket:** AMD-Triton/triton-tickets#1885 (prefill-focused; `--M 1024`,
  block_m=128).
- **Ticket base:** triton `triton-lang/triton@9c795a41fc`, LLVM pin `62b7cf96`,
  aiter `moe_a8w4_multicast@3539d32` (gone from server; rebased tip `633f098`,
  same content for our purposes).
- **Current:** triton `users/jerryyin/moe-gather-sload-contract` (LLVM pin
  `56421f92`), aiter `users/jerryyin/moe-a8w4-contract`.
- Both LLVM prebuilts are local: `/root/.triton/llvm/llvm-{62b7cf96,56421f92}-*`.
- Kernel: `_moe_gemm_a8w4_{prefill,decode}`, gfx1250, FFM-lite.
- The "count" everywhere = in-loop `v_readfirstlane` in the **gemm** kernel's
  K-loop (a constant ~3 from unrelated `_combined_routing` is excluded).

---

## The mechanism (one paragraph)

The TDM gather descriptor's row-index groups are `<4 x i32>` built by
`insertelement` from the (wave-uniform) gather indices. Even when the whole thing
is **IR-uniform** (0 divergent `insertelement`), AMDGPU SelectionDAG materializes
the vector in **VGPR**, so ISel inserts VGPR→SGPR `v_readfirstlane` at the scalar
`tensor_load_to_lds` operand; being loop-invariant, they *should* be hoisted, but
`readfirstlane` is convergent so MachineLICM leaves them in the loop. Two
independent ways to avoid the in-loop copies: (A) feed ISel a load it keeps
scalar end-to-end so the descriptor stays scalar — a **vectorized load from a
uniform (readfirstlane'd) address marked `!invariant.load`** (`uniformizeAddr`);
(B) let MachineLICM **hoist** the loop-invariant convergent readfirstlane. (A) is
triton-side and works for prefill; (B) is LLVM-side and is the only thing that
reaches decode.

---

## Data 1 — no LLVM regression (same IR, both llc pins)

| IR (identical file) | `llc 62b7cf96` | `llc 56421f92` |
|---|---|---|
| 9c795a prefill baseline | 8 | 8 |
| 9c795a prefill + **ticket opt** (`uniformizeAddr`+inv) | **0** | **0** |
| 9c795a prefill + my layout-delta port | 8 | 8 |
| current prefill baseline | 0 | 0 |
| current prefill + contract opt (layout-delta) | 16 | 16 |
| current prefill + `uniformizeAddr`+inv | 0 | 0 |
| current decode + `uniformizeAddr`+inv | 8 | **0** |

Every row except the last is llc-invariant. The last row is the decode residual:
same IR, but the patched (MachineLICM) llc hoists it. That is the *only* llc-level
difference found, and it is the fix, not a regression.

## Data 2 — isolation (what makes prefill 0)

| addressing | `!invariant.load` | prefill in-loop rfl |
|---|---|---|
| `uniformizeAddr` (vectorized, uniform addr) | yes | **0** |
| `uniformizeAddr` | **no** | 8 |
| layout-delta (per-element scalarized) | yes | 16 |
| layout-delta | no | 16 |

Both `uniformizeAddr` **and** `!invariant.load` are required.

## Data 3 — the fix, per kernel (current triton, aiter, stock llc)

| kernel | baseline | contract opt (layout-delta) | **fixed opt** (`uniformizeAddr`+inv) |
|---|---|---|---|
| prefill | 0 | 16 | **0** ✓ |
| decode | 8 | 8 | 8 (→ 0 only with MachineLICM patch) |

Correctness PASS throughout (prefill rel_err 0.01001, decode 0.009413); reg
footprint unchanged (sgpr 106 / vgpr 1024 / scratch 44).

---

## The fix (ported, staged, uncommitted)

`third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp` on branch
`users/jerryyin/moe-gather-sload-contract` (+77/−55): replaced the layout-delta
`emitScalarLoad` block with `uniformizeAddr` + `readFirstLaneInt` helpers and, in
the vectorized load loop, `LLVM::LoadOp(vecTy, uniformizeAddr(ptr))` +
`ld.setInvariant(true)`, gated by the existing readonly/noalias contract
(`isWaveUniformTensorLoad && baseIsReadOnly`, kill switch
`TRITON_AMD_DISABLE_UNIFORM_SLOAD`). Verified prefill 16→0 on **stock** llc (opt
fires: `global_load_u16`=0, `s_load` for index, `!invariant` loads present).

`uniformizeAddr`: peel `ptr` to its single-dynamic-index GEP; readfirstlane the
shared base once (cached) and the per-element offset (`readFirstLaneInt`,
sext/zext-peeled); rebuild the GEP. Falls back to whole-pointer readfirstlane if
not such a GEP. (Contract branch lacks 9c795a's `lookThroughExtractValue`; the
GEP-direct + fallback form is equivalent here.)

---

## The LLVM patch (still valid, reframed)

`/root/llvm-project` branch `users/jerryyin/amdgpu-hoist-uniform-readfirstlane`
@ `db4972674` (base `56421f921`), and reproducer at
`~/scripts/triton/reproducers/amdgpu_readfirstlane_licm/`.

- **Reframe:** it is **not** a regression and **not** the fix for prefill. It is a
  **longstanding missed optimization** — MachineLICM won't hoist a loop-invariant
  convergent `v_readfirstlane` out of a uniform loop — and it is the **only** thing
  that fixes the **decode** residual (8→0), which `uniformizeAddr` can't reach and
  which exists in the decode baseline with no opt.
- Change: `TargetInstrInfo::isConvergentInstrHoistable` hook (default false),
  consulted by MachineLICM instead of bailing unconditionally on convergent;
  `SIInstrInfo` opts in `V_READFIRSTLANE_B32`; `SIRegisterInfo` tracks `EXEC` in
  MachineLoopInfo so hoisting is gated to uniform loops (divergent loops keep it).
- The reproducer's `ISSUE.md` framing ("longstanding missed-opt, not a regression")
  is correct; but its real-world hook should be the **decode** case (index already
  scalar, descriptor VGPR) — the prefill case is better fixed in triton.

---

## Recommendation

- **Land the `uniformizeAddr`+invariant fix** on the contract branch — it fixes
  the prefill regression on shipping LLVM, no LLVM change. Clear win.
- **Decode residual (8):** either (a) pursue the LLVM patch upstream, reframed as
  a longstanding missed-opt with the decode repro; or (b) accept it (prefill was
  the ticket's scope; decode's 8 predates all this and is in the baseline).
- Keep the **index `s_load` opt** itself — it hands the backend uniform IR and is
  the trigger for the descriptor staying scalar on prefill.

---

## Corrections to earlier `/root/*.md` (all wrong somewhere)

- `overnight_results.md`: called it an **LLVM regression** — wrong (confounded
  triton-version with LLVM-version; same IR is llc-invariant).
- `moe_readfirstlane_eval.md`, `llvm_knob_exploration.md`,
  `llvm_pass_fix_results.md`: called it a **longstanding LLVM missed-opt with no
  triton fix, needing the LLVM patch** — wrong for prefill (fixable in triton via
  `uniformizeAddr`+invariant; the churn I saw was from my flawed layout-delta
  re-implementation, not inherent).
- My "**LLVM patch entirely unnecessary**" claim — wrong; it's the only fix for
  the decode residual.
- Net: **prefill = triton-side regression (fix in triton); decode = genuine LLVM
  missed-opt (fix in LLVM).** Not one or the other.

## Artifacts
- Fix (staged): `/root/triton` LoadStoreOpToLLVM.cpp, contract branch.
- LLVM branch: `/root/llvm-project` `users/jerryyin/amdgpu-hoist-uniform-readfirstlane`.
- Repro: `~/scripts/triton/reproducers/amdgpu_readfirstlane_licm/`.
- IR used for the matrices: `/root/{fix_optfix,fix_decode,fix_baseline,old_baseline,old_opt,old_ticketpatch}.llir`,
  `/root/uniform_sload_compare.prefix-backup/`, `/root/repro_baseline/`.
