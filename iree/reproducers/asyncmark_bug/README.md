# LLVM asyncmark Correctness Bug Reproducer

Bug in the LLVM AMDGPU backend where `asyncmark`/`wait.asyncmark` intrinsics
cause incorrect results for 3-stage software-pipelined GEMM kernels
using async DMA-to-LDS (`buffer_load_to_lds`) on gfx950.

## Quick Start

```bash
# Edit tool paths at top of script, then:
./reproduce_standalone.sh
```

Requires only `llvm-mc`, `ld.lld`, and `hipcc` — no IREE.

## What It Does

1. Assembles `original/assembly.s` and `fixed/assembly.s` to HSACOs
2. Builds a standalone HIP driver (`driver.cpp`)
3. For each HSACO: launches the kernel, compares GPU output against a host-computed reference
4. Note: the host reference matmul takes ~5 minutes per run

## Results

```
original (asyncmark scheduling):
    wrong elements = ~6.8M / 16.8M (40%)     FAIL

fixed    (s_waitcnt scheduling):
    wrong elements = 0 / 16.8M (0.0%)        PASS
```

Consistently reproducible. The exact wrong values vary between runs
(characteristic of a race condition), but the failure rate is always ~40%.

## The Assembly Difference

The two assemblies were generated from the same LLVM IR
(`original/optimized.ll`), differing only in how async waits are expressed:

- **Original**: uses `asyncmark`/`wait.asyncmark` intrinsics
- **Fixed**: `asyncmark` removed, `wait.asyncmark` replaced with explicit `s_waitcnt`

```
Original IR:                              Fixed IR:
  @llvm.amdgcn.asyncmark()                ; [REMOVED]
  @llvm.amdgcn.wait.asyncmark(i16 2)      @llvm.amdgcn.s.waitcnt(i32 8184)  ; vmcnt(8)
  @llvm.amdgcn.wait.asyncmark(i16 0)      @llvm.amdgcn.s.waitcnt(i32 8176)  ; vmcnt(0)
```

Both compiled with `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3`.
The full diff (`diff original/assembly.s fixed/assembly.s`):

- Same 922 instructions, identical register allocation
- **Identical `s_waitcnt` values**: both produce `vmcnt(8)` and `vmcnt(0)`
- Only difference: 4 ALU instructions reordered in the prolog

## O0 vs O3

Compiling the **same original IR** (with asyncmark intrinsics) at `-O0`:

```bash
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O0 original/optimized.ll -o original_O0.s
# assemble, link, run → PASS
```

```
           | asyncmark | opt level | s_waitcnt values     | verdict
-----------+-----------+-----------+----------------------+---------
llc -O3    | yes       | O3        | vmcnt(8), vmcnt(0)   | FAIL
llc -O0    | yes       | O0        | vmcnt(8), vmcnt(0)   | PASS
llc -O3    | no        | O3        | vmcnt(8), vmcnt(0)   | PASS
```

All three produce identical `s_waitcnt vmcnt(8)` and `s_waitcnt vmcnt(0)`.
`-O0` disables the machine scheduler, producing a different (correct)
instruction ordering from the same IR.

## Root Cause

The `asyncmark` intrinsics carry `IntrHasSideEffects`, which makes them
scheduling barriers in the machine scheduler DAG (`isGlobalMemoryObject()`
returns true). At `-O3`, this constrains the scheduler's freedom in the
prolog, producing a specific instruction ordering that triggers a
hardware-level race condition in the async DMA-to-LDS path on gfx950.

The `mergeAsyncMarks()` logic in `SIInsertWaitcnts.cpp` computes the correct
vmcnt values — the bug is not in the wait count computation. The bug is that
the scheduling barrier effect of the intrinsics causes codegen that is
incorrect on hardware.

## Files

```
asyncmark_bug/
├── reproduce_standalone.sh   Standalone reproducer (llvm-mc + hipcc only)
├── driver.cpp                HIP harness: loads HSACO, compares vs host reference
├── original/                 BROKEN — uses asyncmark intrinsics
│   ├── optimized.ll          LLVM IR with asyncmark/wait.asyncmark
│   └── assembly.s            llc -O3 output
├── fixed/                    CORRECT — explicit s_waitcnt
│   ├── optimized.ll          LLVM IR with asyncmark → s_waitcnt
│   └── assembly.s            llc -O3 output
├── reproduce.sh              Full reproducer (requires IREE)
├── reproduce_from_asm.sh     IREE-based reproducer from assembly
└── test_mm.mlir              MLIR input (for IREE-based reproducers)
```

## References

- [LLVM PR #180467](https://github.com/llvm/llvm-project/pull/180467) — asyncmark/wait.asyncmark intrinsics
- [LLVM PR #180466](https://github.com/llvm/llvm-project/pull/180466) — async load.to.lds
