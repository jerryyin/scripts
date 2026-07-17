# Widen the a8w4 gather-index s_load — ledger

## Goal & done-claims
- CLAIM-1: the a8w4 gather-index scalar loads can be made WIDER (fewer, wider
  s_load_bN) than the current 32× s_load_u16. | status: unchecked
- CLAIM-2: while wider, they still (a) stay s_load (no global_load), (b) 0 in-loop
  v_readfirstlane, (c) correctness PASS. | status: unchecked
- DONE = a change that reduces s_load count/widens, keeps a+b+c.

## Established facts (HOW)
- a8w4 gluon prefill (noalias-only): 32× s_load_u16, 0 in-loop rfl, PASS (this session)
- moe_gfx1250 prefill: index is i32, already coalesced to s_load_b512/b256 (wide)
- Ticket #1885 §2a: the VECTOR load was vec=1 (16× global_load_u16) because
  align=1, from `% M` (runtime M) + runtime `start_m` (GatherIndx += start_m).
  Their vectorization hack (multiple_of(start_m,BLOCK_M) + drop %M) widened it but
  is unsound for real multi-expert (degenerate E=1 only).

## Hypotheses
- H1: a8w4 s_load stays narrow (u16) because LLVM's scalar-load coalescer can't
  prove the group is contiguous+aligned (align=2, or non-adjacent gather addrs).
- H2: the width lever is ALIGNMENT/contiguity of the index-load address; making it
  provable coalesces the u16 loads. Question: can it be done SOUNDLY (unlike §2a)?

## Variables
- kernel/phase (a8w4 gluon prefill), the noalias contract (keep on), llc pin (stock)
- measure: s_load width histogram, global_load count, in-loop rfl, correctness

## Next decisive experiment
- Inspect the a8w4 index loads in .llir: are addresses adjacent? what align? why no
  coalesce? Contrast moe_gfx1250 (wide).

## Iteration 1 result (ROOT CAUSE)
- a8w4 index load addr = `getelementptr [2 x i8], %base, i64 %offset_i`, and each
  %offset_i = sext(srem(block_off+arange_i, M)) — i.e. the `% M` wrap-mask.
- srem(...,M) makes the 32 offsets INDEPENDENT runtime values -> SILoadStoreOptimizer
  can't prove adjacency -> 32 narrow s_load_u16. (moe_gfx1250 has no such %M on the
  index addr -> contiguous consts -> coalesced to s_load_b512.)
- This is exactly ticket #1885 §2a: `% M` -> contiguity 1. H1/H2 confirmed: the lever
  is address contiguity, killed by `% M` (and the runtime start_m alignment).

## Next decisive experiment
- Remove `% M` (+ multiple_of start_m) in the aiter gluon kernel (the §2a change) and
  measure: do the s_load widen (b128+), stay s_load, 0 in-loop rfl, correctness?
  Then judge soundness (ticket said the §2a change is unsound for real multi-expert).

## Iteration 2-3 results (the lever fails)
- Dropping `% M` (the contiguity killer) does NOT widen: s_load_u16 32->0, in-loop
  rfl 0->16, global_load 5->9. It DISRUPTS the s_load (violates the constraint).
- Not vectorization: both with/without %M emit 32x `load <1 x i16>` (vec=1). Refuted H
  that contiguity -> vec>1 -> vector-global.
- TTGIR layout identical both ways (slice<dim=0>, wave-uniform). The flip s_load<->global
  is at ISel, driven by the ADDRESS form: srem-wrapped addr -> s_load (uniform-recognized);
  contiguous addr (block+arange) -> global_load + descriptor churn. Mechanism NOT pinned
  (counterintuitive: the contiguous form is the one that loses s_load).

## CLAIM STATUS
- CLAIM-1/2: could NOT find a widening that preserves s_load + 0 churn. The narrow
  32x s_load_u16 is entangled with the %M addressing that ISel uses to pick s_load;
  removing %M for contiguity flips ISel to global_load+churn. No safe lever found.

## Contrast that shows it's possible in principle
- moe_gfx1250 index (i32, contiguous, no %M killer) DOES coalesce to s_load_b512.
  So a wave-uniform contiguous index load CAN be a wide s_load -- a8w4 differs
  (i16 + IDX_LAYOUT + mask/where) in a way that sends its contiguous load to global.

## Blocker / what would unblock (loop stop: report, don't invent)
- Open question: why does a8w4's contiguous-address uniform i16 load lower to
  global_load (not wide s_load) while moe's i32 does wide s_load? Answering that
  (AxisInfo/layout/ISel divergence for the a8w4 index) is the prerequisite to any
  widening. Plus: even a widened contiguous load needs %M handled soundly (host-pad
  GatherIndx so the over-read is safe), since the ticket's bare %M-drop is unsound.

## Iteration 4-5: ROOT CAUSE (confirmed in asm)
- Both w/wo index loads are IR-uniform (0/32 divergent), same GEP. The flip is at ISel.
- w (with %M): non-contiguous srem addr -> address kept in SGPR (s_add_nc_u64 s[8:9])
  -> SMEM `s_load_u16` (0 churn).
- wo (no %M): CONTIGUOUS addr triggers AMDGPU scaled VMEM mode:
  `v_mov_b32 v1, s12 ; global_load_b32 v1, v1, s[2:3] scale_offset` -> scalar addr
  moved to VGPR -> VMEM global_load -> VGPR result -> descriptor churn (16).
- => The narrow s_load exists BECAUSE %M defeats the scaled-VMEM path, forcing SMEM.
  Contiguity (needed to widen) re-enables scaled-global and loses SMEM. Also needs
  noalias (invariance) on top; s_load = %M-form addr + noalias.

## Widening hypothesis (H3, testing)
- Force SMEM on a CONTIGUOUS load by making the address SGPR-only (ticket
  uniformizeAddr: readfirstlane addr) -> scaled-VMEM mode inapplicable -> SMEM, and
  contiguous -> coalesce to wide s_load. Needs: drop %M (contiguous) + uniformizeAddr
  (force SMEM) + host-pad GatherIndx (over-read soundness).

## Iteration 6: H3 REFUTED + final root cause
- no%M + uniformizeAddr (B) == no%M + no-lowering (C): both global_load, 16 churn.
  => forcing the address SGPR via readfirstlane does NOT override AMDGPU's scaled-VMEM
  preference for a contiguous address. The contiguous load goes global_load regardless.
- So: SMEM(s_load) requires the NON-contiguous %M/srem address (defeats scaled-VMEM);
  WIDE requires a CONTIGUOUS address (triggers scaled-VMEM -> global). Mutually
  exclusive for this kernel. The narrow 32x s_load_u16 is a forced local optimum.

## CONCLUSION (loop done: grounded negative)
- No widening achievable via the addressing levers tried. Root cause: AMDGPU ISel
  picks a scaled global_load (VMEM) for contiguous uniform loads, and per-element SMEM
  s_load only for the non-contiguous (%M) form. Widening needs contiguity, which loses
  SMEM.
- Open (would unblock widening): moe_gfx1250's contiguous i32 index load DOES lower to
  wide s_load_b512 (SMEM), not scaled-VMEM. Why a8w4's contiguous i16 goes scaled-VMEM
  but moe's i32 goes wide-SMEM (dtype? layout? scale_offset trigger?) is the remaining
  question and the only known path to widening. Alternatively an AMDGPU ISel change to
  prefer wide SMEM over scaled-VMEM for uniform invariant contiguous loads.

## Iteration 7: CORRECTION (user challenged "contiguity==load-kind" — correctly)
- My "contiguity -> VMEM" was WRONG. Isolated llc test (uniform, !invariant.load,
  noalias, kernel-arg ptr):
    * i16 scattered  -> s_load_u16  (SMEM)   [@t16nc: 4 s_load_u16, 0 global]
    * i16 contiguous -> global_load_u16 (VMEM) [@t16: global_load]
    * i32 contiguous -> s_load_b32/b* (wide SMEM) [@t32; moe = s_load_b512]
- Also: dropping %M DID widen (wo coalesced 32 u16 -> 4 global_load_b128) -- but into
  VMEM. So contiguity=width and SMEM/VMEM ARE separate (user right); the coupling is a
  SUB-DWORD limitation: AMDGPU has no wide/vectorized sub-dword SMEM load, so contiguous
  i16 vectorizes -> VMEM; scattered i16 stays scalar -> s_load_u16; i32 -> wide SMEM.

## REAL ROOT CAUSE
- a8w4 index is uint16 (sub-dword). Wide SMEM for sub-dword doesn't exist in AMDGPU.
  => narrow s_load_u16 is forced UNLESS the index is loaded dword-granular (i32).

## WIDENING PATH (PROPOSED — mechanism-verified only, NOT implemented on a8w4)
- Load the index DWORD-granular + contiguous -> wide SMEM. Concretely:
  (1) drop %M for contiguity                     [DONE/MEASURED -> regresses to VMEM+churn]
  (2) host-pad GatherIndx to BLOCK_M multiple    [NOT DONE - safety prereq for (1)]
  (3) bitcast/load 2 adjacent uint16 as one i32, unpack  [NOT DONE - the step that buys SMEM]
- VERIFIED: (1)'s regression (measure_a8w4.sh); the mechanism (3) relies on, in
  ISOLATION only -> @t32 (contiguous i32 -> s_load_b*) + moe's real s_load_b512.
- NOT verified: the combined a8w4 change (drop %M + pad + i32-pack). No kernel diff for
  (2)/(3) exists; end-to-end wide-SMEM on a8w4 is unproven, just mechanism-backed.

## Dead ends
- Drop `% M` alone: contiguous i16 -> VMEM global_load + 16 churn (sub-dword, no wide SMEM).
- Drop `% M` + uniformizeAddr: same (sub-dword can't wide-SMEM regardless of addr).
