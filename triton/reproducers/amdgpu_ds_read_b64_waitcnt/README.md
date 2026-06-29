# gfx950 `convert_layout` race after swizzle PR #9662

Investigation of AMD-Triton issue #1881: on **gfx950 (CDNA4)**, the AITER
`test_mha_backward_with_pe` forward output is numerically nondeterministic after
Triton swizzle PR **#9662** (`f4a3db9`, "Refine optimal swizzling for
wavefront64"). Last good commit before it: `4a1df47`.

## Conclusion (current)

**It is a Triton swizzle-layout bug, not an LLVM backend bug.**

- #9662 makes gfx950 use `numBanks=64`, which (via the pre-existing `log2Vec<2`
  vectorization-boost reorder in `GenericSwizzling`) lowers the softmax
  correction convert `tensor<128x1xf32,#mma> -> #mma1` to a **contiguous
  `ds_read_b64`** LDS read instead of a strided pair of scalar reads.
- That contiguous convert is **correct in isolation**. It only races when it
  **coexists at LDS offset 0 with the `#shared2` P-reshape** (`local_alloc` /
  `local_load` feeding the PV `tt.dot`) of the same attention step.
- The backend is exonerated by direct experiment: not a missing `s_waitcnt`
  (max waits don't fix), not the `ds_read_b64` instruction (scalar reads of the
  same addresses still race), not `-O3` vs `-O0` (both race), not timing
  (LDS-point barriers fix, ALU-point barriers don't).
- Fix lever: revert this convert to the strided layout (`SWIZ_NO_REORDER`, or
  clamp `effNumBanks` to 32 for `log2Vec<2`). Both drive `ds_read_b64 -> 0` and
  make the kernel deterministic, while keeping #9662's 64-bank modeling elsewhere.

The exact silicon reason the contiguous + coexisting pattern races despite a
correct barrier is the one open item (would need an ATT trace); the trigger,
layer, and fix are pinned by the experiments below.

## Layout

```
README.md                  ← this file
analysis/
  root-cause.md            ← the full causal chain + evidence table (start here)
  access-pattern.md        ← exact per-lane LDS addresses, wrong vs correct, from the IR
  minimal-repro.md         ← standalone-repro attempts + the trim-down to the minimal trace
  barrier-analysis.md      ← the asm barrier experiments (sync layer analysis)
ir/
  attn_fwd.full.ttgir              ← full kernel TTGIR (input to repro)
  attn_fwd.wrong.ll               ← racy 64-bank LLVM IR (contiguous ds_read_b64)
  attn_fwd.correct.ll             ← NO_REORDER LLVM IR (strided scalar reads; correct)
  minimal-trace.ttgir             ← full kernel trimmed (forced constants) to the minimal racer
  minimal-trace.canonicalized.ttgir ← above, dead branches folded (310 lines)
  minimal-convert.ttgir / .ll     ← faithful standalone convert (#5) — does NOT race alone
harness/
  driver.cpp               ← standalone HIP launcher (loads hsaco, runs N times, diffs runs)
  compile.py               ← triton.compile(ttgir) -> hsaco + amdgcn (honors SWIZ_* env)
archive/                   ← superseded "LLVM -O3 backend miscompile" framing (disproven)
```

## Reproduce

Requires a gfx950 GPU, a Triton build, and the env-gated prototype in the Triton
tree (`SWIZ_NO_REORDER` in `lib/Tools/GenericSwizzling.cpp`).

```bash
# 1. compile the full kernel TTGIR -> hsaco  (baseline = racy)
python harness/compile.py ir/attn_fwd.full.ttgir /tmp/baseline.hsaco

# 2. build + run the launcher: run N times, compare run0 vs runs
#    exits non-zero (worst max_abs_diff > 0.01) when it races
hipcc -O2 harness/driver.cpp -o /tmp/driver
/tmp/driver /tmp/baseline.hsaco 8        # baseline: ~0.035 (RACES)

# 3. A/B: recompile with the fix, confirm deterministic
SWIZ_NO_REORDER=1 TRITON_ALWAYS_COMPILE=1 python harness/compile.py ir/attn_fwd.full.ttgir /tmp/fixed.hsaco
/tmp/driver /tmp/fixed.hsaco 8           # fixed: 0.000 (deterministic)
```

The minimal trace (`ir/minimal-trace.ttgir`) reproduces the same way and is the
smallest racing configuration: one attention step (no loops, no QK dot) =
convert + `#shared2` P-reshape + PV dot. See `analysis/minimal-repro.md`.

## Status

Root-caused and a working fix identified (`SWIZ_NO_REORDER`). Not yet a
finalized upstream patch (needs target-conditioning on wavefront64 + a regression
test in `unittest/Dialect/TritonGPU/SwizzleTest.cpp`). Nothing committed.
