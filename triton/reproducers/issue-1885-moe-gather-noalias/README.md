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

## Files here

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

## Related tooling (elsewhere in ~/scripts/triton)

- `moe/ffm_verification/run_moe_gemm_ffm.py` — the FFM correctness/kernel runner
  (`--kernel {a4w4,a8w4} --backend {gluon,triton} --phase {prefill,decode}`).
- `reproducers/amdgpu_readfirstlane_licm/` — standalone LLVM reproducer for the
  decode residual (MachineLICM won't hoist convergent readfirstlane).
- `../compare_uniform_sload.sh` — **STALE**: uses the removed
  `TRITON_AMD_DISABLE_UNIFORM_SLOAD` kill switch. Superseded by `gen_before_after.sh`.
