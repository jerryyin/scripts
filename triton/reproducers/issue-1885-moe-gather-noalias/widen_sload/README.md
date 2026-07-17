# Widening the a8w4 gather-index s_load — investigation

Follow-on to the parent `../README.md`: given the noalias contract already turns the
prefill gather-index load into `s_load` (0 in-loop readfirstlane), can those s_loads
be made **wider** (32× `s_load_u16` → fewer, wider `s_load_b*`) *without* regressing
to `global_load` or reintroducing churn?

**Answer: not with the current i16 index dtype.** The narrow width is forced by an
AMDGPU sub-dword limitation, not by contiguity or uniformity. There is a concrete
widening path, but it needs a dtype + host change.

## Root cause (isolated, `subdword.ll` / `subdword2.ll`)

`s_load` vs `global_load` is about address uniformity/invariance, **not** contiguity —
*except* for sub-dword loads, where AMDGPU has **no wide/vectorized sub-dword SMEM
load**. Isolated llc (uniform, `!invariant.load`, noalias, kernel-arg ptr, gfx1250):

| load | contiguous? | result |
|---|---|---|
| i16 scattered  (a8w4 with `% M`) | no  | `s_load_u16` — **SMEM** (scalar sub-dword OK) |
| i16 contiguous (a8w4 no `% M`)   | yes | `global_load_u16` — **VMEM** (sub-dword vector → no SMEM) |
| i32 contiguous (moe_gfx1250)     | yes | `s_load_b32/b*` — **wide SMEM** |

So:
- a8w4's index is `uint16`. Scattered (the `% M` wrap) → per-element `s_load_u16`
  (SMEM, narrow) = the current state.
- Making it contiguous (drop `% M`) *does* coalesce (32×`u16` → 4× `global_load_b128`)
  — but into **VMEM**, because a contiguous sub-dword load vectorizes and sub-dword
  vectors can't be SMEM → the descriptor lift churns again (16 in-loop readfirstlane).
  See `measure_a8w4.sh`.
- `moe_gfx1250`'s index is `i32` (dword) → contiguous coalesces to wide `s_load_b512`.

This corrected an earlier wrong conclusion ("contiguity → VMEM"): contiguity and
load-kind are independent for dword; the coupling is sub-dword-specific. See
`ledger.md` (iteration 7) for the refutation trail.

## Widening path (grounded, but a real change — not a free lever)

Load the index **dword-granular and contiguous** → wide SMEM:
1. **Contiguity:** drop `% M` and **host-pad `GatherIndx`** to a `BLOCK_M` multiple so
   the over-read is in-bounds (the kernel's existing `where(mask_idx, …, oob_idx)`
   already discards the padding entries → still correct).
2. **Dword granularity:** bitcast/load two adjacent `uint16` as one `i32`, unpack the
   pair after the load.

Then contiguous `i32` → `s_load_b*` (wide SMEM), 0 churn — grounded by `@t32` and
moe's real `s_load_b512`. Cost: kernel change (bitcast+unpack) + host padding.

**LLVM alternative:** teach AMDGPU to form wide sub-dword SMEM loads (or coalesce
contiguous scalar `s_load_u16` → `s_load_b*` + unpack). Then a8w4 widens with no
kernel/host change.

## Files
- `subdword.ll` — `@t16` (contiguous i16) + `@t32` (contiguous i32), isolated.
- `subdword2.ll` — `@t16nc` (scattered i16).
- `reproduce.sh` — llc both, print the truth table (no GPU needed).
- `measure_a8w4.sh` — end-to-end on the real a8w4 kernel: toggles `% M`, shows the
  coalesce-into-VMEM + churn regression (includes the exact one-line kernel diff).
- `ledger.md` — the full investigation log (hypotheses, refutations, the correction).
