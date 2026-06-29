# Root cause: gfx950 convert_layout race after swizzle PR #9662

**Verdict: a Triton swizzle-layout bug, not an LLVM backend bug.** The backend is
exonerated by direct experiment (waitcnt, instruction selection, and scheduling
all ruled out). The defect is an interaction between PR #9662's 64-bank swizzle
and a pre-existing vectorization-boost reorder in `GenericSwizzling`.

## Causal chain

1. **#9662** (`f4a3db9`, "Refine optimal swizzling for wavefront64") makes gfx950
   use `numBanks=64` (was 32). Per its own description it *drops a swizzle phase*
   (`log2Phase` lane bases) on the assumption that wavefront64 LDS accesses are
   serialized into non-conflicting thread groups. Result: a **less-swizzled
   64-bank `smem`** layout.

2. For fp32 `convert_layout` with broadcasting and low vectorization
   (`log2Vec < 2`) — the softmax LSE-normalization convert
   `tensor<128x1xf32, #mma> -> #mma1` — the **pre-existing reorder branch**
   (`lib/Tools/GenericSwizzling.cpp:630`, from #6982) reorders that 64-bank
   `smem`'s bank bases to maximize vectorization. On the 64-bank layout this
   yields a **contiguous** load: `load <2 x float>` → `ds_read_b64`.
   (Store stays `store <1 x float>`; the store/load asymmetry is the tell.)

3. That contiguous layout, on wavefront64, carries a **cross-wavefront LDS access
   hazard** → nondeterministic output (forward-output mismatch). The strided
   layout produced with 32 banks (or without the reorder) is safe.

The reorder was correct for 32 banks (NVIDIA wf32, CDNA3 gfx942). It becomes
unsafe only when applied to the new 64-bank `smem`. #9662 didn't touch the
reorder; it changed the `numBanks` fed into it.

## Evidence (direct experiments)

Standalone harness: recompile `ir/attn_fwd.full.ttgir` -> hsaco -> run N times, compare
run0 vs runs (bf16 max_abs_diff, tol 0.01).

Layers eliminated:

| hypothesis | experiment | result |
|---|---|---|
| backend missing waitcnt | max `s_waitcnt lgkmcnt/vmcnt(0)` after every LDS op (asm) | still races |
| the `ds_read_b64` instruction | replace with scalar `ds_read_b32` same addrs (asm) | still races |
| timing / serialization | `s_barrier` after every ALU op, 796 total (asm) | still races (0.015) |
| (positive control) | `s_barrier` after every LDS op, 148 total (asm) | **fixed 0.000** |
| inter-op barrier (membar) | maximally-conservative membar, all aliasing pairs | still races (38 barriers) |
| convert warp-sync skip | `MEMBAR_NO_WARPSYNC` | no change (path never taken; 0 `wave.barrier`) |
| AMD async barrier filter | `MEMBAR_NO_FILTER` | no change (no async in kernel) |

Layer confirmed (swizzle), both keep correctness:

| fix | banks | ds_read_b64 | result |
|---|---|---|---|
| baseline | 64 | 5 | races 0.035 |
| skip `log2Vec<2` reorder (`SWIZ_NO_REORDER`) | 64 | 0 | **fixed 0.000** |
| clamp banks for low-vec (`SWIZ_LOWVEC_32`) | 32 | 0 | **fixed 0.000** |

Both drive `ds_read_b64 -> 0`: the vectorized contiguous access *is* the hazard,
so any correct fix reverts this convert to the strided load.

## Why barriers "fix" it but it isn't a barrier bug

`s_barrier` after every LDS instruction fixes it (148 barriers) because brute-
force cross-wave serialization hides the hazard; ALU-point barriers (even 796)
don't, so it's a real LDS cross-wave effect, not timing. But the IR is already
fully CTA-barriered (`store->s.barrier->load->s.barrier`) and conservative
inter-op membar can't fix it — the needed ordering is *intra-op*, below membar's
granularity. The actual defect is the layout, not a missing barrier.

## Minimal fix

Condition the vectorization-boost reorder (or the bank count for low-vec
converts) on the target: do not produce the contiguous layout for wavefront64
when it is unsafe. Env-gated prototypes (default off, in tree):
`SWIZ_NO_REORDER` (surgical, keeps 64-bank base swizzle) and `SWIZ_LOWVEC_32`.
A proper patch should be target-conditioned (gfx950 / wavefront64) and ship with
a regression test in `unittest/Dialect/TritonGPU/SwizzleTest.cpp`.

## Open item

The exact hardware reason the contiguous wavefront64 layout races despite CTA
barriers (the wavefront64 LDS serialization mechanism behind #9662's dropped
phase) is not proven here — it would take an ATT/hardware trace. The trigger,
layer, and fix are pinned by the experiments above.
