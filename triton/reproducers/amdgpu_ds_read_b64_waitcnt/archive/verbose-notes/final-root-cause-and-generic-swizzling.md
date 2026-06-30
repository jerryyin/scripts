# Final root cause and GenericSwizzling notes

This is the current investigation record. Older attention-only and
backend-exonerating notes are archived under `../archive/outdated-investigations/`.

## Executive summary

The minimal source reproducer is `ir/micro-dot.ttgir`. It is not an attention
kernel: it has no Q/K/V global-load schedule, masks, loops, or softmax control
flow. It keeps the generated LDS/MFMA ingredients that make the failure visible:

- a reduction-fed f32 `#mma -> #mma1` convert,
- P-reshape LDS traffic at the same scratch base,
- B operand LDS traffic and a dot,
- two post-dot low-vector f32 converts,
- non-uniform data so wrong LDS values are numerically observable.

With the pre-fix swizzle condition:

```cpp
if (log2Vec < 2) {
```

the micro reproducer emits three f32 convert `ds_read_b64` reads and races:

```text
RESULT worst=0.12500000 total_changed=12416   # checked-in failing assembly, 1000 runs, grid 2048
```

With the guarded condition:

```cpp
if (log2Vec < 2 && numBanks <= 32) {
```

the same source IR emits zero f32 convert `ds_read_b64` reads and is stable.

The guard is a root-cause Triton fix for the regression. It does not add
barriers, waits, NOPs, or pretend gfx950 has 32 banks. It preserves the 64-bank
swizzle model and disables only an older low-vector bank-column reorder whose
assumptions stop holding for this 64-bank case.

There is also backend-facing evidence. With the same source IR and visible
waits, extra waitcnt-only does not fix the race, and scalarizing to same-address
`ds_read_b32` pairs still races — pointing at a lower-level gfx950 LDS/MFMA
schedule interaction on the contiguous read, for backend/hardware investigation.
(An earlier `s_nop`-padding "fix" was later **falsified** — non-monotonic, i.e.
schedule-perturbation luck. The reliable contrast is the strided lowering.)

## What GenericSwizzling is doing

`convert_layout` sometimes cannot be implemented only by register moves. Triton
then round-trips through LDS:

```text
source register layout
  -> store to a chosen shared-memory layout
  -> load from that shared-memory layout
  -> destination register layout
```

`GenericSwizzling` chooses that temporary shared layout. It tries to:

- preserve the logical mapping,
- avoid LDS bank conflicts,
- expose vectorized LDS operations when profitable,
- respect the target bank count,
- avoid unnecessary scratch growth.

The key data structure is `LinearLayout`. For registers it maps bases such as
`register`, `lane`, and `warp` to tensor dimensions. For shared memory it maps
bases such as `vector`, `bank`, `segment`, `block`, and `reps`.

PR #9662 made this machinery bank-count-aware for wavefront64. On gfx950 the
model now uses 64 LDS banks instead of treating the target like a 32-bank shape.
That part is correct and should be preserved.

## The failing swizzle decision

The important conversion is:

```mlir
tensor<128x1xf32, #mma> -> tensor<128x1xf32, #mma1>
```

with:

```text
#mma  = amd_mfma version 4, warpsPerCTA=[4,1], instrShape=[32,32,16], transposed
#mma1 = amd_mfma version 4, warpsPerCTA=[4,1], instrShape=[16,16,32], transposed
```

After broadcasted register bases are removed, this is a low-vector 32-bit
transfer. It does not use the special 128-bit CDNA4 local-load/store tile path.
It goes through `optimalSwizzlingLdSt`.

For this conversion, the 64-bank base swizzle is valid without the post-pass:

```text
bank bases:    1, 2, 4, 8, 16, 64
segment bases: 32
```

The old low-vector reorder then tries to improve vectorization by moving
register-derived bases earlier in the shared-memory bank basis order. On this
64-bank shape it moves the destination register basis into bank bit 0:

```text
bank bases:    64, 1, 2, 4, 8, 16
segment bases: 32
```

Both layouts are internally consistent in the abstract swizzle model. The second
layout creates a contiguous destination LDS read pattern that LLVM lowers to
`ds_read_b64`. In the micro-dot schedule that contiguous pattern is
nondeterministic on gfx950.

The fix keeps the first 64-bank layout and skips only that low-vector reorder
when `numBanks > 32`.

## Why the guard is okay

The guard does not change logical correctness. The shared layout chosen by
`GenericSwizzling` is still used consistently by the store side and the load
side.

The guard does not throw away PR #9662. `optimalSwizzling` still sees
`numBanks=64` on gfx950. The only disabled piece is the later
vectorization-boost reorder for `log2Vec < 2`.

The guard is conservative for conflicts. The unreordered 64-bank layout remains
bank-conflict-free for this transfer. The low-vector reorder was an optimization
heuristic, not a correctness requirement.

The guard explains the observed compiler symptoms:

- pre-fix micro-dot emits three f32 convert `ds_read_b64` reads and races,
- fixed micro-dot emits zero such reads and is stable,
- the original trimmed attention trace is also stable after the same fix,
- `TestSwizzling` still passes with the regression test added.

## Why this is not masking

Masking would be adding broad waits, barriers, NOPs, disabling `ds_read_b64`
globally, or reverting PR #9662. The Triton patch does none of those.

It removes the specific compiler transformation that creates the fragile
low-vector 64-bank layout.

The current split is:

```text
Triton root cause:
  the 32-bank-era low-vector bank-column reorder is applied to a 64-bank swizzle
  and creates the fragile contiguous LDS/MFMA schedule.

Backend/hardware question:
  why the generated gfx950 schedule can fail despite visible waits, present
  equally in the racy and stable builds.
```

That split is important. The Triton fix is root-cause for the regression, while
the assembly A/B gives the backend team a minimal full-ISA case demonstrating a
lower-level correctness hazard.

## How the investigation converged quickly

The directory already contained good raw material: full TTGIR, reduced traces,
good/bad LLVM snapshots, compile scripts, and a HIP driver. That made it
possible to test at source IR, LLVM IR, and ISA levels without building a
harness from scratch.

The decisive move was to stop treating `ds_read_b64` as the root by itself.
Scalarizing the same addresses still raced, and a standalone faithful convert
did not race. That pushed the search from "one bad opcode" to "the generated
LDS/MFMA schedule produced by a swizzle choice."

PR #9662 narrowed the compiler search. Since the PR changed the bank-count model
for wavefront64, the useful question was: which low-vector swizzle decision
changes only because gfx950 now has 64 banks? Instrumenting
`GenericSwizzling` showed the destination register basis moving from a high bank
bit into bank bit 0 only in the low-vector reorder path.

The final hypothesis made specific predictions:

- disabling only that reorder on 64-bank targets should remove the f32 convert
  `ds_read_b64` reads,
- shared-memory size and barrier structure should not need a broad change,
- the micro reproducer and trimmed attention trace should become deterministic,
- the 64-bank model should remain active,
- the swizzling unit tests should pass.

Those predictions held.
