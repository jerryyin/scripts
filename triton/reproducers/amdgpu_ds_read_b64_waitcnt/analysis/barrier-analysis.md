# The barrier experiment — what it was, what it proved, and what it didn't

## Essence

Taking the **compiled assembly** of the racy kernel and inserting a full
cross-wavefront synchronization (`s_waitcnt lgkmcnt(0)` + `s_barrier`) after
**every** LDS read and write, then reassembling, makes the numerical race vanish
(0.000 across all runs). Forcing the *wait counters alone* does **not** fix it.
That difference is the whole story: the bug is about **cross-wavefront ordering**
(`s_barrier`), not wait counters, and not the backend.

Honest caveat: this is a **class** localization, not a single-line one. The race
only disappears when *all* LDS ops are barriered (reads **and** writes); every
narrower subset still races. See "What this does NOT prove".

## Where it sits

```
TTGIR ──(Triton AMD backend)──> LLVM IR ──(llc)──> AMDGCN assembly (.s)
   │                               │                       │
 layout + barriers          llvm.amdgcn.s.barrier      s_barrier   <- edited HERE
 (MembarAnalysis)           (IR intrinsic)             (machine instr)
```

The edit is at the **assembly** level (post-codegen): it isolates the backend
(the backend already ran; we only add sync to its output) and iterates in seconds
with `llvm-mc` instead of recompiling Triton.

## The two sync primitives — only one helps

| primitive | waits for | scope |
|---|---|---|
| `s_waitcnt lgkmcnt(0)` | *this wave's own* LDS ops to drain | **intra-wave** |
| `s_barrier` | *all 4 wavefronts* to reach this point | **inter-wave** |

256 threads/block = 4 wavefronts sharing one block's LDS. `s_waitcnt` guarantees
my own loads landed; it says nothing about the other three waves. `s_barrier` is
a rendezvous. Max `s_waitcnt` everywhere → still races. Add `s_barrier` → fixed.
⇒ the missing guarantee is cross-wave.

## The exact edit

IR level (what Triton emitted — convert already has store→barrier→load):

```llvm
store <1 x float> %476, ptr addrspace(3) %241, align 4   ; my element out
tail call void @llvm.amdgcn.s.waitcnt(i32 49279)         ; drain my counter
tail call void @llvm.amdgcn.s.barrier()                  ; rendezvous
%477 = load <2 x float>, ptr addrspace(3) %245, align 8  ; read 2 elems OTHER waves wrote (ds_read_b64)
...
tail call void @llvm.amdgcn.s.barrier()                  ; rendezvous before scratch reuse
```

Assembly level — append `s_waitcnt lgkmcnt(0)` + `s_barrier` after every
`ds_read*`/`ds_write*`. Barrier count 29 → 148 (covers 39 reads + 80 writes).

## Which insertions helped (honest table)

```
barrier after ds_read_b64 only ............ 0.0245  RACE
barrier after ds_write_b32 only ........... 0.0241  RACE
barrier after ds_write_b16 only ........... 0.0239  RACE
barrier after ds_read_b64_tr_b16 only ..... 0.0237  RACE
barrier after ALL reads ................... 0.0353  RACE
barrier after ALL writes .................. 0.0277  RACE
barrier after ALL reads AND writes ........ 0.0000  FIXED   <- only this
```

No single op-type, and no single side, fixes it. Only reads AND writes together.

## Mental model: the hazard that needs a barrier but lacks one

```
  RAW (read-after-write)            WAR (write-after-read)
  wave A: write X ──┐               wave A: read X ──┐
                 [BARRIER]                        [BARRIER]
  wave B: read  X ──┘               wave B: write X ──┘
  "don't read until producer wrote" "don't overwrite until readers done"
```

- RAW needs a barrier *after writes*; WAR needs a barrier *after reads*. The
  table shows both are required — consistent with a genuine two-directional
  cross-wave hazard on the reused offset-0 scratch (converts + P-reshape share it).
- Trap: the convert *already has* barriers. The deficiency is not "zero barriers";
  it is that the 64-bank (contiguous) layout widens the set of op-pairs that alias
  across waves beyond what was barriered.

## Where the missing barrier comes from (MembarAnalysis)

`s_barrier` originates as `ttg.barrier`/`llvm.amdgcn.s.barrier`, inserted by
Triton's **`MembarAnalysis`** (`lib/Analysis/Membar.cpp`); the backend only
transcribes it. Two code paths can legitimately drop a CTA barrier for a
`convert_layout`:

1. **`isWarpSync` skip** — `Membar.cpp:330-357`:
   ```cpp
   isWarpSync = mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
   ...
   if (insertCTABarrier || !isWarpSync)
     blockInfo->sync();      // skipped when isWarpSync && no CTA barrier
   ```
   When a convert is deemed warp-synchronous it uses a warp.sync internally, and
   membar does not clear pending cross-wave deps. `isCvtDimSync` (`Utility.cpp:1330`)
   guards against broadcasting over warp — its own comment: *"if there is
   [broadcasting], we'll deduplicate the writes and the reads will read from data
   that other warp has written."* That is the exact symptom. Introduced/changed by
   PR #7810 and #9317.

2. **`membarFilter`** (`third_party/amd/lib/.../MembarUtility.cpp`) —
   `filterAsyncLocalLoadsDependencies` removes barriers between async
   global→local copies and local loads on the same buffer, assuming the pipeliner
   uses ≥2 buffers. Its own comment warns it "can produce wrong IR/assembly...
   filters out a required `ttg.barrier`."

## What this does NOT prove (open gaps)

1. **Not line-pinned.** Brute-force (all LDS ops) is the only thing that worked.
2. **Timing not fully excluded.** 148 barriers also serialize execution. Control
   test still owed: insert the same *count* of barriers at non-LDS points; if that
   also "fixes" it, part of the effect is timing. The reads-vs-writes asymmetry
   argues for genuine ordering, but is not proof.
3. **Isolation still passes** (one workgroup, 0/30) — consistent with a cross-wave
   hazard needing ≥2 competing waves, but inferred, not demonstrated.

## Why this matters

This is the first evidence pointing at a specific, fixable layer: the missing
guarantee is a cross-wave `s_barrier` emitted by Triton's `MembarAnalysis`, not a
backend waitcnt or instruction-selection issue (both already excluded). The real
fix lives in `MembarAnalysis`/`isCvtDimSync`/`membarFilter` vs. the 64-bank
swizzle's wider cross-wave footprint — not the 32-bank clamp, which merely avoids
the layout that needs the extra barrier.
