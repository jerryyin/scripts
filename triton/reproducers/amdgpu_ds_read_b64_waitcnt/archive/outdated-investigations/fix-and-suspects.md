# The fix (`effNumBanks` / `NO_REORDER`), and an honest assessment of suspects

## What the fix changes

`numBanks` is the LDS bank count handed to the swizzler. On gfx950 the hardware
has **64 banks** (`TargetInfo.cpp:162`), and that 64 flows into
`optimalSwizzlingLdSt` -> `optimalSwizzling` / `getLaneTile`.

`effNumBanks` is a local override (`GenericSwizzling.cpp:622`): for low-vector
converts (`log2Vec < 2`) it feeds the swizzler **32** instead of the true 64.

```cpp
int32_t effNumBanks = numBanks;                 // 64 on gfx950
if (log2Vec < 2 && getenv("SWIZ_LOWVEC_32"))
  effNumBanks = std::min(numBanks, 32);
... getLaneTile(..., effNumBanks); optimalSwizzling(..., effNumBanks);
```

`numBanks` only drives **bank-conflict avoidance**:
- `getLaneTile` (`:280`): `log2Phase = max(0, log2Vec + lanes - log2(numBanks))`
  -> how many swizzle phases to apply.
- `optimalSwizzling` (`:460`): `bankBits = numBanks*32` -> the bank/segment split.

For the fp32 convert (`bitwidth=32, log2Vec=1`, 64 lanes):
- 64 banks -> `log2Phase=1` -> less swizzle -> **contiguous** `ds_read_b64`.
- 32 banks -> `log2Phase=2` -> more swizzle -> **strided** scalar reads.

## Why feeding "32" to a 64-bank GPU is safe

1. **Correctness:** any `numBanks` yields a bijective layout used consistently by
   store and load; it round-trips. `numBanks` changes the physical permutation,
   not the logical mapping.
2. **No extra conflicts:** conflict-free assuming 32 banks implies conflict-free
   on 64 (addresses distinct `mod 32` are distinct `mod 64`). Clamping is strictly
   more conservative.
3. **Cost is throughput only:** the 64 value is what unlocks the wide contiguous
   load; clamping forgoes it for narrower/strided loads. No corruption, no extra
   conflicts.

So clamping = "swizzle as if fewer banks -> scramble more -> give up the wide
contiguous load." That contiguous load is exactly the race trigger, so clamping
fixes by avoiding the problematic access pattern.

## Honest assessment of the fix

- **Correctness-safe and effective** - it changes which valid layout is chosen,
  not a race papered over.
- **A "back off the optimization" fix, not a "repair the optimization" fix** -
  IF #9662's layout was supposed to be safe (see open question below).
- **Blunt:** `log2Vec < 2` catches all low-vec converts, including ones that may
  be safe -> a small perf regression beyond the one buggy case.
- `SWIZ_NO_REORDER` is the more surgical sibling: keep `numBanks=64` for the base
  swizzle, skip only the `log2Vec<2` vectorization-boost reorder that manufactures
  the contiguous load. Backs off less.

## Suspect 1 - barriers (reconciled honestly)

Measured:
- wrong (contiguous), ~29 barriers -> races
- wrong (contiguous), +~26 operand-region barriers -> fixed
- correct (strided), ~30 barriers -> safe

So **barriers ARE a real lever** - adding cross-wave barriers fixes it, and
intra-wave `s_waitcnt` does not, proving it is a genuine cross-wavefront ordering
hazard. But it is **not a clean "membar dropped one barrier"** bug:
- conservative membar (force every op-level barrier) added only ~9 and did NOT
  fix it;
- the correct build adds no extra barriers either - it's safe via layout.

The extra sync the contiguous layout needs is **large (~26) and sub-op**, so the
barrier route is a real-but-impractical fix (heavy serialization). That is why the
layout route is preferred - not because barriers are innocent.

## Suspect 2 - the swizzle logic itself (likely, per the clamp signal)

That clamping `numBanks` to 32 fixes it is **direct evidence the swizzle is where
the defect originates.** Two senses of "broken":
- **Wrong mapping** (non-bijective / wrong elements): RULED OUT - the convert
  round-trips correctly standalone.
- **Produces an unsafe layout** (correct mapping, races in practice): SUPPORTED -
  this is what "clamping helps" indicates.

Live hypothesis: **#9662's premise may be wrong.** It drops a swizzle phase on the
assumption that wavefront64 serializes threads into non-conflicting groups, which
is what lets it pack the convert contiguously. If that assumption is invalid for
this access, the 64-bank layout is genuinely wrong, and clamping to 32 is
**reverting a bad assumption** (a real fix), not a band-aid.

## The open question that decides "real fix vs workaround"

**Is #9662's wavefront64 phase-drop premise valid?**
- If NO -> the 64-bank contiguous layout is genuinely incorrect; clamp/NO_REORDER
  is the correct fix.
- If YES -> the layout is fine and something downstream (sync/hardware) is wrong;
  clamp is a workaround.

Not yet resolved (would need a wavefront64 LDS-serialization microbenchmark or an
ATT trace). The fix works either way; only the framing of the upstream report
depends on the answer.
