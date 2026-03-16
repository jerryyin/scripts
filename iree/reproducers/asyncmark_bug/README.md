# LLVM asyncmark Correctness Bug Reproducer

Bug in the LLVM AMDGPU backend where `asyncmark`/`wait.asyncmark` intrinsics
cause incorrect code generation for 3-stage software-pipelined GEMM kernels
using async DMA-to-LDS (`buffer_load_to_lds`) on gfx950.

## Quick Start

```bash
./reproduce.sh
```

## Evidence

### The Experiment

Starting from the `.optimized.ll` (LLVM IR after target-independent
optimizations) of a failing 4096x4096x4096 f32 GEMM with 3-stage pipelining:

1. Replace `asyncmark`/`wait.asyncmark` intrinsics with explicit `s_waitcnt`
2. Compile both original and modified IR through `clang -O3` to assembly
3. Substitute the modified assembly into the runtime binary
4. Compare numerical results against a known-good baseline

### The Replacement

```
Original IR:                          Modified IR:
  @llvm.amdgcn.asyncmark()            ; [REMOVED]
  @llvm.amdgcn.wait.asyncmark(i16 2)  @llvm.amdgcn.s.waitcnt(i32 8184) ; vmcnt(8)
  @llvm.amdgcn.wait.asyncmark(i16 0)  @llvm.amdgcn.s.waitcnt(i32 8176) ; vmcnt(0)
```

`vmcnt(8)` = `wait.asyncmark(2)` × 4 loads/group. Bitfield `8184` encodes
`vmcnt(8) expcnt(7) lgkmcnt(63)` (only vmcnt constrained).

### Results

```
Original (with asyncmark intrinsics):
    max_abs_diff    = ~50
    wrong elements  = ~6.8M / 16.8M (40%)
    verdict:  FAIL

Modified (asyncmark → s_waitcnt):
    max_abs_diff    = 0.000000
    wrong elements  = 0 / 16,777,216 (0.0%)
    verdict:  PASS
```

Same LLVM IR, same `clang -O3`, same runtime — only the wait mechanism differs.

### Assembly Comparison

The original assembly (828 lines) and fixed assembly (766 lines) show
structurally different code: different register allocation, different
instruction ordering, different loop structure. This is visible in the
provided files:

```bash
diff original/assembly.s fixed/assembly.s
diff original/optimized.ll fixed/optimized.ll
```

Key sync instructions in the assembly:

**Original** (with asyncmark pseudos):
```asm
; asyncmark                         ← scheduling barrier (side effect)
; asyncmark
; asyncmark
; wait_asyncmark(2)
s_waitcnt vmcnt(8)
; wait_asyncmark(0)
s_waitcnt vmcnt(0)
```

**Fixed** (explicit s_waitcnt, no asyncmark pseudos):
```asm
s_waitcnt vmcnt(8)
s_waitcnt vmcnt(0)
```

## Files

```
asyncmark_bug/
├── reproduce.sh           Self-contained reproducer (runs the full experiment)
├── test_mm.mlir           4096x4096x4096 f32 GEMM input
├── README.md              This file
├── original/              BROKEN — uses asyncmark intrinsics
│   ├── optimized.ll       LLVM IR after target-independent opts (input to ISel)
│   └── assembly.s         Assembly from clang -target amdgcn-amd-amdhsa -mcpu=gfx950 -O3
└── fixed/                 CORRECT — asyncmark replaced with explicit s_waitcnt
    ├── optimized.ll       Modified LLVM IR (same as original, only waits changed)
    └── assembly.s         Assembly from clang -target amdgcn-amd-amdhsa -mcpu=gfx950 -O3
```

## Bug Analysis

The `ASYNCMARK`/`WAIT_ASYNCMARK` pseudo-instructions inherit
`IntrHasSideEffects` from their intrinsic definitions (`IntrinsicsAMDGPU.td`).
This causes `isGlobalMemoryObject()` to return true, making them scheduling
barriers in the machine scheduler DAG. The result is structurally different
code (different instruction ordering and register allocation) that interacts
badly with `mergeAsyncMarks()` in `SIInsertWaitcnts.cpp` during loop backedge
processing for 3+ stage pipelines.

## References

- [LLVM PR #180467](https://github.com/llvm/llvm-project/pull/180467) — asyncmark/wait.asyncmark intrinsics
- [LLVM PR #180466](https://github.com/llvm/llvm-project/pull/180466) — async load.to.lds
