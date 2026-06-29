# Does the convert reproduce standalone? — No. Hard evidence.

Question: can the gfx950 race be reproduced by a minimal standalone kernel that
contains only the `convert_layout #mma -> #mma1` (the two layouts), without the
full attention kernel?

**Answer: No.** Even a kernel whose convert lowers to the *byte-for-byte
identical* LDS sequence as the full kernel runs fully deterministically. The race
requires full-kernel context beyond the convert itself.

## Method

Each minimal kernel is hand-written TTGIR, compiled with `triton.compile(...,
GPUTarget("hip","gfx950",64))`, launched via ctypes (grid as noted, block 256 =
4 wavefronts), run 30x with fixed random input, comparing run-to-run for
nondeterminism. Full attention kernel = 0.035 max_abs_diff (races).

## Results

| # | minimal kernel | shared | convert lowering | reproduces? |
|---|---|---|---|---|
| 1 | single convert, plain load -> mma -> mma1 -> blocked, grid 1 | 512 | `store<4xf>` vectorized (NOT faithful) | no (0/29) |
| 2 | same, grid 2048 | 512 | (not faithful) | no |
| 3 | convert in scf.for loop (scratch reuse), grid 2048 | 512 | (not faithful) | no |
| 4 | convert + live coexisting `#shared2` 8KB buffer, grid 2048 | 24576 | 5x ds_read_b64 | no (0/29) |
| 5 | **reduction-fed convert** (`expand_dims(reduce(qk,axis=1))`), grid 2048 | 16384 | `store<1xf>`/lgkmcnt(0)/s.barrier/`load<2xf>` **FAITHFUL** | **no (0/29)** |
| - | full attention kernel | 14336 | same faithful lowering | **YES (0.035)** |

## The crucial control (#5)

Earlier minimal kernels (#1-4) were invalid: feeding the convert from a plain
`tt.load` produces a *vectorized* `store<4xfloat>`, not the full kernel's scalar
`store<1xfloat>`. The scalar store is the broadcast-dedup signature, and it only
appears when the convert input is a reduction result. The full kernel's input is
`expand_dims(reduce(qk, axis=1))` (`ir/attn_fwd.full.ttgir:332`).

Kernel #5 reproduces that: load 128x32 -> #mma, reduce axis=1 -> slice, expand_dims
-> 128x1 #mma, convert -> #mma1. Its convert lowers to (ir/minimal-convert.ll:131-141):
```llvm
store <1 x float> %118, ptr addrspace(3) %117, align 4
tail call void @llvm.amdgcn.s.waitcnt(i32 49279)   ; lgkmcnt(0)
tail call void @llvm.amdgcn.s.barrier()
%123 = load <2 x float>, ptr addrspace(3) %122, align 8
tail call void @llvm.amdgcn.s.waitcnt(i32 49279)
tail call void @llvm.amdgcn.s.barrier()
```
This is identical to the full kernel (`attn_fwd.wrong.ll:505-512`): same scalar store,
same `lgkmcnt(0)` + `s.barrier`, same contiguous `load<2xfloat>` (ds_read_b64).
Yet it runs deterministically across 30 runs at grid 2048.

## What this means

- The convert's access pattern + sync, even reproduced exactly, is **NOT
  sufficient** to cause the race. The contiguous `ds_read_b64` convert is correct
  in a standalone kernel.
- The race is an **emergent property of the full attention kernel** - it needs
  context the convert alone doesn't provide. Leading candidates (not yet
  isolated): concurrent LDS traffic from the pipelined K/V loads and the MFMA
  (`tt.dot`) operand `local_load`s interleaved with the convert's barrier window.
- This refines the earlier framing: the contiguous layout is a **necessary
  trigger** (changing it via NO_REORDER/clamp fixes the full kernel) but **not
  sufficient** on its own. The fix is still valid (it removes the trigger), but
  the bug is context-dependent, which is consistent with a hardware-level
  timing/contention hazard rather than a pure layout-correctness error.

## Reproducer artifacts

- `ir/minimal-convert.ttgir` - kernel #5 (faithful convert lowering, does NOT race)
- `ir/attn_fwd.full.ttgir` - full kernel (races)
- Minimal does not race => the standalone reproducer is, at minimum, the full
  attention kernel (or a trim of it that preserves the concurrent LDS context).

---

# Trim-down: the minimal trace that still reproduces

Method: start from the full kernel (`ir/attn_fwd.full.ttgir`, races 0.035), force
`scf.if` conditions / loop bounds to constants so canonicalization prunes whole
chunks, recompile, run driver 8x. Watch only the run-to-run nondeterminism
signal (correctness irrelevant for bisection). `worst>0.01` = races.

## Bisection results

| configuration | result |
|---|---|
| full kernel (baseline) | races 0.035 |
| prune loop2 (masked, `%3=false`) | races 0.033 -> **loop2 not needed** |
| prune loop1 block (full-blocks, `%2=false`) | **0.000 -> loop1 block needed** |
| loop1 = 0 iterations (keep peeled drain) | races 0.199 -> **loop iterations not needed** |
| loop1 + peeled removed | races 0.033 -> loop1 alone also races |
| single peeled iteration only | races 0.16 -> **one attention step is enough** |
| one step, REMOVE PV dot + `#shared2` P-reshape | **0.000 -> P-reshape coexistence REQUIRED** |
| one step, REMOVE QK dot (softmax on zeros) | races 0.087 -> **QK dot not needed** |
| minimal (no loops, no QK), baseline | races 0.087 (b64=3) |
| minimal, `SWIZ_NO_REORDER` | **0.000 (b64=0)** -> convert layout is the lever |

## The minimal trace (994 -> 310 live lines)

Three coexisting ops are the irreducible trigger:
```mlir
%177 = ttg.convert_layout %176 : tensor<128x1xf32, #mma> -> tensor<128x1xf32, #mma1>   // offset 0
%182 = ttg.local_alloc %181 : (tensor<128x32xbf16, #mma>) -> memdesc<128x32xbf16, #shared2>  // offset 0 (REUSES convert scratch)
%183 = ttg.local_load %182 ... -> dot_op<opIdx=0, parent=#mma1>
%184 = tt.dot %183, %165, %179 ...                                                     // PV dot consumes P-reshape
```

## What this proves

- The convert does NOT race in isolation (confirmed earlier, and here: removing
  the `#shared2` P-reshape makes it deterministic while the convert + its
  `ds_read_b64` remain, b64 unchanged).
- The race REQUIRES the `#shared2` P-reshape `local_alloc`/`local_load`
  coexisting with the convert **at the same LDS offset 0** (scratch reuse).
- It does NOT require: the masked loop, any loop iterations, the QK dot, or
  multiple attention steps. One step's tail suffices.
- The convert's layout is still the lever: `SWIZ_NO_REORDER` (strided convert)
  fixes the minimal trace (b64 -> 0, 0.000).

## Mechanism (now concrete)

The convert's contiguous offset-0 `ds_read_b64` access and the `#shared2`
P-reshape's reuse of the same offset-0 LDS region interact: the contiguous
convert layout is only unsafe when that offset-0 scratch is also driven by the
adjacent P-reshape store/load + dot traffic. The strided (pre-#9662 / NO_REORDER)
convert layout avoids it. This is why no standalone convert-only kernel
reproduces - the P-reshape coexistence is a necessary second ingredient.

## Artifacts

- `ir/minimal-trace.ttgir` - trimmed full kernel (forced constants; races, fixed by NO_REORDER)
- `ir/minimal-trace.canonicalized.ttgir` - dead branches folded (310 lines)
