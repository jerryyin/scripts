# LLVM bisect of the ds_read_b64 / s_waitcnt miscompile

## TL;DR

**Not a recent regression.** The `-O3` miscompile is present at the *earliest
commit where the IR is even valid* — the introduction of the async-load-to-LDS /
`asyncmark` feature (**#180466 + #180467, 2026-02-11**). The faulty
`ds_read_b64` codegen is byte-identical from that commit through current `main`
(`ebe87bafc`). It is a latent bug in the async-LDS / `asyncmark` waitcnt path,
not a regression from previously-correct code.

## Method

- **Test input:** `ir/kernel.ll` with the two newest attributes stripped
  (`nocreateundeforpoison`, `captures(none)`) so it parses across the range →
  `kernel_portable.ll`. Verified this still reproduces (tip `-O3` = 55
  `s_waitcnt`, `-O0` = 128).
- **Predicate (`bisect_run.sh`):** build this commit's `llc`, run
  `llc -mcpu=gfx950 -O3`, count `s_waitcnt`. `<= 70` ⇒ **bad** (bug present),
  else **good**; exit 125 (skip) if the build or IR parse fails.
- **Build:** incremental `ninja llc` with ccache; adjacent commits rebuild in
  seconds, large jumps in ~15–20 min.

## The hard floor: the IR only parses from 2026-02-11

Before #180466/#180467 the async intrinsics either don't exist or have a
different signature, so `llc` rejects the IR (`Intrinsic has incorrect argument
type! ... llvm.amdgcn.raw.ptr.buffer.load.async.lds`). The IR uses
`llvm.amdgcn.asyncmark`, `llvm.amdgcn.wait.asyncmark`, and
`llvm.amdgcn.raw.ptr.buffer.load.async.lds` — i.e. the exact async-LDS feature
introduced on 2026-02-11. So the bisect floor is the feature's birth.

## Probes (portable IR, `llc -O3`, gfx950)

| commit | date | `s_waitcnt` | verdict |
|--------|------|-------------|---------|
| `db48679835` | 2026-01-19 | — | IR rejected (pre-feature) |
| `128437fb6a` #180467 asyncmark intro | 2026-02-11 | **55** | BAD (earliest valid) |
| `1bcdf716ae` #190872 LDSDMA sched mask | 2026-04-23 | 55 | BAD |
| `884a434491` (parent of #201942) | 2026-06-08 | 55 | BAD |
| `b01fe4e3d1` #201942 LDSDMA→S_WAIT latency | 2026-06-08 | 55 | BAD |
| `ebe87bafc` (tip) | 2026-06-26 | 55 | BAD |

The critical region (the `ds_read_b64` of the layout-conversion + its consumers)
is **byte-identical** between the 2026-02-11 commit and tip; the ~555 other
instruction differences across the 4.5-month window are unrelated codegen drift.

## Conclusion

`git bisect` cannot find a "good" commit: every commit where the IR is valid is
bad, and the IR cannot be expressed before the feature exists. The defect is
therefore inherent to the async-LDS / `asyncmark` `-O3` codegen as introduced,
most plausibly in the `SIInsertWaitcnts` `lgkmcnt` accounting around `asyncmark`
(see e.g. `mergeAsyncMarks`, #193499) interacting with the regular `ds_read_b64`
of the conversion scratch. Suspect commits to start from:

- #180466 / #180467 — async-LDS + asyncmark introduction (where it first appears)
- #190872 — sched group mask for LDSDMA
- #201942 — "Do not always add latency between LDSDMA -> S_WAIT_LDSDMA"
- #193499 — `mergeAsyncMarks` in `SIInsertWaitcnts`

## Reproduce the bisect probe

```bash
cd <llvm-checkout>
git checkout <commit>
ninja -C build llc
build/bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 kernel_portable.ll -o /tmp/c.s
grep -c s_waitcnt /tmp/c.s     # 55 => bug present
```
`bisect_run.sh` automates this as a `git bisect run` predicate.
