# gfx950 `ds_read_b64` Reproducer: Final Root Cause and GenericSwizzling Notes

This note records the investigation path from the raw repro to the final patch.
It is intentionally written as an audit trail: what was observed, what was
rejected, what was fixed, and why the fix is a root-cause compiler change rather
than a barrier or scheduling mask.

## Executive Summary

Update: the context gap has now been reduced below the attention kernel. The
minimal source-level reproducer is `ir/micro-dot.ttgir`, documented in
`analysis/micro-dot-root-cause.md`. It contains no attention loads, masks,
loops, or softmax control flow; it keeps only the generated LDS/MFMA ingredients
needed to expose the bad low-vector 64-bank f32 converts. With the pre-fix
low-vector reorder it emits three f32 `ds_read_b64` converts and races. With the
guarded 64-bank fix it emits zero f32 `ds_read_b64` converts and is stable.

The regression is caused by an interaction between PR #9662's new 64-bank LDS
model for gfx950 and an older low-vectorization reorder in
`lib/Tools/GenericSwizzling.cpp`.

PR #9662 correctly teaches the swizzler that gfx950 has 64 LDS banks. However,
for low-vectorization conversions (`log2Vec < 2`), the existing reorder tries to
boost vectorization by moving register-derived bases earlier in the shared-memory
bank basis order. On the failing conversion:

```mlir
tensor<128x1xf32, #mma> -> tensor<128x1xf32, #mma1>
```

that reorder moves the destination register basis into bank bit 0 on a 64-bank
target. The resulting shared-memory layout is still internally consistent and
bank-conflict-free according to the model, but it creates a contiguous LDS read
pattern that LLVM lowers to `ds_read_b64`. That pattern is stable in isolation
but becomes nondeterministic in the surrounding LDS/MFMA schedule with adjacent
P-reshape local traffic and scratch reuse. The reduced `micro-dot.ttgir`
reproducer demonstrates that attention-specific loads, masks, loops, and softmax
control flow are not required.

The final patch keeps the PR #9662 64-bank model intact, but disables the
low-vectorization vectorization-boost reorder for 64-bank targets:

```cpp
if (log2Vec < 2 && numBanks <= 32) {
  ...
}
```

This is not a barrier mask. It prevents the compiler from creating the bad
layout in the first place.

Important scope: this identifies the Triton regression mechanism, not the full
silicon-level failure physics. The remaining unresolved question is why gfx950
can execute the contiguous LDS pattern deterministically in a standalone convert
kernel but nondeterministically in the surrounding attention trace. The compiler
side is still actionable: PR #9662 exposed an old vectorization heuristic on a
new 64-bank layout shape, and that heuristic creates the fragile generated code.

## Final Patch

Only two files are changed:

```text
lib/Tools/GenericSwizzling.cpp
unittest/Dialect/TritonGPU/SwizzleTest.cpp
```

The implementation change is one guard:

```diff
-  if (log2Vec < 2) {
+  if (log2Vec < 2 && numBanks <= 32) {
```

The regression test constructs the exact failing f32 conversion:

```cpp
auto src = mfma(4, {4, 1}, {32, 32, 16}, true);
auto dst = mfma(4, {4, 1}, {16, 16, 32}, true);
SmallVector<int64_t> shape = {128, 1};
```

It removes broadcasted register bases like lowering does, checks that the
conversion is a 32-bit low-vector conversion, asks `optimalSwizzlingLdSt` for a
64-bank shared layout, and verifies the destination register basis remains high:

```cpp
EXPECT_EQ(dstToSmem.getBasis(S("register"), /*pos=*/0, S("bank")), 32);
EXPECT_EQ(dstToSmem.getBasis(S("register"), /*pos=*/0, S("segment")), 0);
```

That is the invariant the bad 64-bank reorder violated: it moved that register
basis to bank bit 0, producing the contiguous LDS access pattern.

## Verification

Baseline before the patch:

```text
minimal trace:
  shared bytes: 14336
  ds_read_b64 : 3
  s_barrier   : 13
  full-grid run: nondeterministic, worst max_abs_diff about 0.08691

full kernel:
  shared bytes: 14336
  ds_read_b64 : 5
  s_barrier   : 29
  run-to-run output differs, worst max_abs_diff about 0.034-0.036
```

After the final patch:

```text
minimal trace:
  shared bytes: 14336
  ds_read_b64 : 0
  s_barrier   : 13
  50 full-grid runs: worst max_abs_diff = 0.00000

full kernel:
  shared bytes: 14336
  ds_read_b64 : 0
  s_barrier   : 29
  20 runs: worst max_abs_diff = 0.00000
```

Unit tests:

```text
TestSwizzling: 10/10 passed
```

## Investigation Timeline

### 1. Re-establish the baseline

The first step was to treat every previous conclusion as provisional and rebuild
the known failing artifacts from the TTGIR.

For the full kernel:

```text
shared bytes: 14336
ds_read_b64 : 5
s_barrier   : 29
```

The HIP driver reproduced nondeterminism immediately.

For the canonical minimal trace:

```text
shared bytes: 14336
ds_read_b64 : 3
s_barrier   : 13
```

The minimal trace also reproduced nondeterminism at full grid.

That established the current local toolchain and gfx950 machine could reproduce
the issue without relying on archived conclusions.

### 2. Re-test the prior barrier story

The previous write-up claimed that inserting barriers around LDS operations
fixed the issue. That sounded important, but it needed to be independently
checked.

Fresh assembly experiments inserted `s_waitcnt lgkmcnt(0); s_barrier` after:

- every LDS operation,
- all LDS reads,
- all LDS writes,
- convert reads,
- P-reshape reads,
- P-reshape writes,
- and pre/post LDS operation variants.

These did not reliably fix the minimal trace or the full kernel in the fresh
runs. Some variants changed the number of bad elements, but they did not remove
the race.

That result was important because it stopped the investigation from converging
on a membar patch. The failure was not solved by adding more synchronization in
the generated assembly.

### 3. Test whether `ds_read_b64` itself is the bug

The next question was whether `ds_read_b64` was intrinsically broken.

A targeted assembly variant replaced a `ds_read_b64` with two scalar
`ds_read_b32` operations at the same addresses. The kernel still raced.

That showed the opcode itself is not the root cause. The important thing is the
address/layout pattern that the compiler created. `ds_read_b64` is the visible
signature of that pattern, not the whole explanation.

### 4. Isolate the exact conversion

The TTGIR showed the failing conversion is:

```mlir
%acc = ttg.convert_layout %x
  : tensor<128x1xf32, #mma> -> tensor<128x1xf32, #mma1>
```

where:

```mlir
#mma  = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1],
                        instrShape = [32, 32, 16], isTransposed = true}>

#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1],
                        instrShape = [16, 16, 32], isTransposed = true}>
```

This is a low-vector f32 conversion. After broadcasted register bases are
removed, `getVecBitwidthLdSt` reports 32 bits. That detail matters: the AMD
target hook `getSharedLdStTiles(vecBitwidth)` has special CDNA4 behavior for
128-bit vectors, but this conversion does not go through that path.

So the problem was narrowed to the generic swizzling path:

```cpp
optimalSwizzlingLdSt(srcLayout, dstLayout, bitwidth, numBanks, ...)
```

not to AMD's special 128-bit local load/store tile table.

### 5. Audit PR #9662 in context

PR #9662 changed GenericSwizzling to become target-bank-aware. In particular,
for wavefront64 it changed how many lane bases are dropped when modeling which
threads are active in the same serialized LDS phase.

The key model is:

```text
log2Phase = log2((Threads * BitsPerThread) / (Banks * 32))
```

For 64 lanes, 64-bit access, 64 banks, that means fewer phase bits are dropped
than on 32-bank hardware. This is sensible as a bank-conflict model: gfx950 has
more banks, so not every conflict that existed in a 32-bank model is real.

But the failing conversion was not a 128-bit dot operand case covered by the PR's
tests. It was a 32-bit low-vector f32 `convert_layout`. That led to the older
low-vector reorder.

### 6. Inspect the swizzle decision directly

A temporary C++ unit test printed the exact layouts for:

```cpp
mfma(4, {4, 1}, {32, 32, 16}, true)
  -> mfma(4, {4, 1}, {16, 16, 32}, true)
shape = {128, 1}
bitwidth = 32
numBanks = 64
```

After mirroring lowering's broadcast removal, the source layout is:

```text
register is size 1
lane bases: 1, 2, 4, 8, 16, 0
warp bases: 32, 64
```

The destination layout is:

```text
register basis: 64
lane bases: 1, 2, 4, 8, 0, 0
warp bases: 16, 32
```

With 32 banks, the old layout uses:

```text
bank bases:    1, 2, 4, 8, 16
segment bases: 32, 64
```

With PR #9662's 64-bank model plus the low-vector reorder, the bad layout uses:

```text
bank bases:    64, 1, 2, 4, 8, 16
segment bases: 32
```

So the destination register basis `64` was moved into bank bit 0. That is the
specific transformation that made the destination load contiguous and let LLVM
emit the `ds_read_b64` pattern.

With the reorder disabled, the 64-bank layout stays:

```text
bank bases:    1, 2, 4, 8, 16, 64
segment bases: 32
```

That still uses 64 banks and still has zero modeled read/write bank conflicts.
It simply does not force the destination register basis into the lowest bank bit.

### 7. Check whether the isolated conversion fails

A standalone `minimal-convert.ttgir` kernel was compiled and run. It contains the
same `#mma -> #mma1` conversion and lowers to the same scalar store plus
contiguous LDS load style:

```text
ds_write_b32 ...
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64 ...
```

That kernel was stable for repeated runs.

This prevented overclaiming. The conversion's contiguous access pattern is a
necessary trigger in the real attention trace, but it is not sufficient by
itself. The failure needs the surrounding LDS traffic and scratch reuse in the
attention trace.

### 8. Verify a scoped fix

Several possible fixes were considered:

- Add barriers: rejected as masking and not reliable in fresh tests.
- Disable `ds_read_b64`: too low-level and not the real cause.
- Clamp all low-vector swizzling to 32 banks: works, but throws away too much of
  PR #9662's 64-bank model.
- Disable the low-vector register-basis reorder only for 64-bank targets: scoped
  to the transformation that creates the bad pattern.

The last option was tested:

```cpp
if (log2Vec < 2 && numBanks <= 32) {
```

It preserved the 64-bank model in `optimalSwizzling` but prevented the old
low-vector vectorization boost from reordering the 64-bank basis into the bad
shape.

Both minimal trace and full kernel became deterministic.

## Why This Was Found Quickly

The speed mostly came from reducing the search space early and refusing to trust
the previous narrative.

### The repro was already high quality

The directory had:

- full TTGIR,
- minimal trace TTGIR,
- good/bad LLVM snapshots,
- a standalone HIP driver,
- and compile scripts.

That meant experiments could be run at the right abstraction level without first
building a repro harness from scratch.

### The investigation used several independent oracles

The conclusion was not based on one signal. It used:

- `ds_read_b64` counts from generated assembly,
- run-to-run nondeterminism from the HIP driver,
- `SWIZ_NO_REORDER` and `SWIZ_LOWVEC_32` controls from the previous local tree,
- direct `GenericSwizzling` unit instrumentation,
- standalone conversion kernels,
- and fresh barrier/opcode replacement experiments.

When a hypothesis survived all of these, it became much more credible.

### The key clue was the PR boundary

PR #9662 was about bank-count-aware phase modeling. So the obvious question was
not "where can we add a barrier?" but:

```text
Which swizzle decision changes when gfx950 uses 64 banks?
```

The answer was visible immediately once the exact f32 conversion was printed:
the register basis moved from a high bank bit to bank bit 0 only in the 64-bank
low-vector reorder path.

### The investigation separated visible symptom from root cause

`ds_read_b64` was the visible symptom. It would have been easy to blame:

- waitcnt,
- barriers,
- LLVM instruction selection,
- or `ds_read_b64` itself.

But scalarizing the load did not fix the issue, and the isolated convert was
stable. That pushed the cause back up into the layout decision that created the
address pattern, not the final opcode.

### The final fix had a narrow prediction

The proposed fix predicted all of the following:

1. The repro should compile with zero f32-convert `ds_read_b64` instances.
2. Shared memory size should remain the same.
3. Barrier counts should remain the same.
4. The 64-bank swizzle model should still be used.
5. `TestSwizzling` should pass.
6. The full/minimal repro should become deterministic.

All six held.

That is why the fix is more credible than a broad workaround.

## GenericSwizzling: Overall Picture

GenericSwizzling chooses a temporary shared-memory layout for transfers between
two register layouts. The common case is a `convert_layout` that cannot be done
purely in registers, so Triton stores values to LDS in one layout and loads them
back in another.

At a high level:

```text
source register layout
  -> store to chosen shared-memory layout
  -> load from chosen shared-memory layout
  -> destination register layout
```

The chosen shared layout tries to satisfy several goals:

- preserve correctness,
- minimize LDS bank conflicts,
- enable vectorized stores/loads when possible,
- account for target bank count,
- account for special local-memory instruction lane behavior,
- and avoid unnecessary scratch size growth.

### LinearLayout

Triton represents layouts using `LinearLayout`. A layout maps hardware
coordinates such as:

```text
register, lane, warp, block
```

to tensor dimensions such as:

```text
dim0, dim1
```

For shared memory, the layout typically maps:

```text
vector, bank, segment, block, reps
```

to tensor dimensions.

The names are meaningful:

- `vector`: contiguous values in one vectorized memory operation.
- `bank`: address bits that choose the LDS bank.
- `segment`: higher address grouping beyond the bank line.
- `block`: CTA-level split.
- `reps`: extra replicated storage if the transfer needs it.

### `getVecBasisLdSt`

`getVecBasisLdSt` determines which bases can form the vectorized contiguous part
of the transfer. This controls the natural vector size for the LDS store/load
pair.

For the failing conversion, after removing broadcasted registers, it reports:

```text
vecBitwidth = 32
```

So this is a low-vector path. It is not one of the 128-bit CDNA4 special tile
cases.

### `getLaneTile`

`getLaneTile` decides which lane bases participate in the same hardware LDS
phase for bank-conflict analysis.

This is where PR #9662's bank-count awareness matters. On wavefront64 hardware,
not all 64 lanes necessarily participate in the same serialized LDS phase for a
given access width and bank count. So the swizzler should not optimize against
conflicts between lanes that are not actually simultaneous.

The PR made this phase model depend on `numBanks`.

That part is not the problem.

### `optimalSwizzling`

`optimalSwizzling` chooses the shared layout. Conceptually it places bases into:

```text
vector bits, bank bits, segment bits, reps
```

trying to avoid conflicts for both the store side and the load side.

For 64-bank gfx950, `optimalSwizzling` can legally use six bank bits instead of
five. That is the point of PR #9662.

### The low-vector reorder

After `optimalSwizzling` chooses a layout, `optimalSwizzlingLdSt` had an extra
post-pass for `log2Vec < 2`.

Its purpose is to opportunistically improve vectorization when the base swizzle
does not already produce a wide vector. It searches bank bases that also appear
in source or destination register bases, then reorders bank columns to put those
bases earlier.

On 32-bank targets, this is an old heuristic that can improve generated code.

On 64-bank gfx950, that heuristic becomes dangerous for the failing f32
conversion because there is one extra bank bit available. The heuristic chooses
the destination register basis `64` and moves it to bank bit 0:

```text
good 64-bank order:
  bank = [1, 2, 4, 8, 16, 64]

bad 64-bank low-vector reorder:
  bank = [64, 1, 2, 4, 8, 16]
```

Both are bank-conflict-free in the abstract model. But the second one creates
the contiguous LDS destination read pattern that triggers the real repro.

The final fix leaves 64-bank `optimalSwizzling` intact and only prevents this
32-bank-era low-vector reorder from applying to 64-bank targets.

## Why This Is Root Cause, Not a Mask

Yes, this is a root-cause fix at the Triton-regression level. It is not yet a
complete hardware-causal explanation.

The precise statement is:

```text
The compiler regression is caused by applying the low-vector bank-column reorder
to 64-bank swizzles. That reorder changes the f32 #mma -> #mma1 conversion from
a strided 64-bank shared layout into a contiguous LDS load pattern. Removing that
specific transformation removes the regression.
```

The stronger statement:

```text
The hardware-level reason this contiguous pattern is nondeterministic only when
surrounded by the attention trace is fully explained.
```

is not proven by the current evidence.

It is not a cover-up for these reasons:

### It changes the cause, not the symptom

Masking fixes would include:

- adding extra barriers,
- forcing conservative membar everywhere,
- disabling `ds_read_b64` globally,
- clamping all low-vector swizzling to 32 banks,
- or reverting all of PR #9662.

The final patch does none of those.

It prevents the specific swizzle transformation that creates the bad address
pattern for low-vector 64-bank conversions.

### It preserves the important part of PR #9662

The final layout still uses:

```text
bank size = 64
segment size = 2
```

So this is not "pretend gfx950 has 32 banks." It is still a 64-bank swizzle.

The fix only says:

```text
Do not apply the old low-vector bank-column reorder on targets with more than
32 banks.
```

That is a scoped correction to a heuristic whose assumptions no longer hold
after PR #9662.

### It explains every key observation

The final explanation accounts for:

- why PR #9662 exposed the issue,
- why `SWIZ_NO_REORDER` fixed it,
- why `SWIZ_LOWVEC_32` fixed it,
- why conservative membar did not fix it,
- why isolated convert did not fail,
- why `ds_read_b64` was present in failing builds,
- why scalarizing `ds_read_b64` was not sufficient,
- and why keeping 64 banks but blocking the low-vector reorder fixes the repro.

### The remaining unknown is below the compiler contract

There is still a lower-level question:

```text
Why does this particular contiguous LDS pattern become nondeterministic only in
the surrounding attention trace on gfx950?
```

Answering that fully would require deeper hardware/ATT-level analysis. But that
does not make the compiler fix a mask. The compiler regression is that the
swizzler generated a fragile low-vector 64-bank layout that was not generated
before PR #9662 and is not required for bank-conflict freedom.

The root cause in Triton is the invalid application of the low-vector reorder to
64-bank targets.

## Practical Mental Model

Think of PR #9662 as adding a new, more accurate 64-bank map.

That part was good.

But after the map was created, an older heuristic came in and rearranged some
map columns to get a nicer vectorized load. That heuristic was written in a
world where 32 banks was the relevant shape. With 64 banks, it found a new
"better" arrangement that looked fine to the bank-conflict model but produced an
unsafe contiguous LDS access pattern in the real workload.

The fix does not throw away the new map. It stops applying that old rearranger
when the map has 64 banks.
