# Micro-dot root cause reproducer

This note supersedes the weaker conclusion that the issue only reproduces in the
attention kernel.

## Minimal reproducer

Files:

- `ir/micro-dot.ttgir`
- `harness/micro_dot_driver.cpp`

The kernel is not an attention kernel. It has no Q/K/V global-load schedule, no
attention masks, no loops, and no softmax control flow. It keeps only the
generated-code ingredients that were necessary:

1. a reduction-fed f32 `#mma -> #mma1` convert,
2. a `#shared2` bf16 P reshape local_alloc/local_load at LDS base 0,
3. a double-buffered B operand local_alloc/local_load before the P reshape,
4. a dot consuming those local_loads,
5. two post-dot low-vector f32 converts,
6. non-uniform row/column data so wrong LDS values are visible.

The uniform-data version of this kernel was stable because many wrong reads are
numerically invisible when all rows/columns carry the same values. Making the
data non-uniform exposes the race.

## Bad compiler result

With the pre-fix swizzle condition:

```c++
if (log2Vec < 2) {
```

compile and run:

```bash
ninja -C /root/triton/build/cmake.linux-x86_64-cpython-3.12 /root/triton/python/triton/_C/libtriton.so
TRITON_ALWAYS_COMPILE=1 python harness/compile.py ir/micro-dot.ttgir /tmp/micro-dot-bad.hsaco
/opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx950 -O2 -std=c++17 harness/micro_dot_driver.cpp -o /tmp/micro_dot_driver
/tmp/micro_dot_driver /tmp/micro-dot-bad.hsaco 50 2048
```

Observed compile shape:

```text
shared bytes: 10240
ds_read_b64 : 3
s_barrier   : 9
```

Observed runtime:

```text
RESULT worst=0.12500000 total_changed=1728
```

The failures were stochastic. Example failing runs changed 192 or 256 bf16
elements, often first visible around output indices 1280 or 1792.

## Fixed compiler result

With the guarded condition:

```c++
if (log2Vec < 2 && numBanks <= 32) {
```

the exact same TTGIR compiles as:

```text
shared bytes: 10240
ds_read_b64 : 0
s_barrier   : 9
```

Runtime for 50 runs at grid 2048:

```text
RESULT worst=0.00000000 total_changed=0
```

The original trimmed attention trace also remains stable under the same fixed
compiler:

```text
RESULT: worst max_abs_diff = 0.00000
```

## Backend-facing A/B

The current backend handoff is the read-lowering A/B (`ds_read_b64` races vs
`ds_read2_b32` stable); see the top-level `ISSUE.md` / `reproduce.sh`.

(An earlier `s_nop` "timing padding fixes it" claim was later **falsified** — the
dose-response is non-monotonic, i.e. schedule-perturbation luck, not a fix. It is
not reproduced here.)

## What this proves

The bad f32 convert is not wrong in isolation. A standalone semantic checker for
the same reduction-fed convert returns exact expected values and is stable.

The bare LDS address family is also stable when tested directly with raw
`ds_write_b32` / `ds_read_b64` and explicit waits.

The reproducing unit is the generated LDS/MFMA schedule: low-vector f32 converts
that the old post-pass turns into contiguous `ds_read_b64` operations, plus the
neighboring P-reshape local traffic and dot consumption, with non-uniform data.

That is why the fix is not a barrier mask. It removes the compiler
transformation that creates the required bad schedule. It does not pretend
gfx950 has 32 banks: the 64-bank swizzle model from PR #9662 is still used. The
guard only prevents the older 32-bank-era low-vector bank-column reorder from
rotating a destination register basis into the lowest bank bit on 64-bank
targets.

For the failing f32 convert, that post-pass changes a safe 64-bank layout into a
layout that vectorizes the destination register pair as adjacent LDS words. The
abstract bank-conflict model still reports no conflicts, but the real gfx950
execution of the surrounding LDS/MFMA schedule is nondeterministic. Keeping the
register basis high on 64-bank targets avoids that schedule while preserving
64-bank-aware swizzling.
