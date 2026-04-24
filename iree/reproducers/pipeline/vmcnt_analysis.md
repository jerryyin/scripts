# Pipeline vmcnt(0) Analysis: Multi-Buffered Async Copy Mode

## Problem Statement

When using `amdgpu.gather_to_lds` (which lowers to `buffer_load_dwordx4 ... lds`)
for pipelined async DMA from global memory to LDS, the LLVM backend's
`SIInsertWaitcnts` pass conservatively inserts `s_waitcnt vmcnt(0)` between the
DMA writes (for the *new* iteration) and the `ds_read` operations (reading the
*previous* iteration's data from a different multi-buffer slot).

This `vmcnt(0)` is unnecessary because multi-buffering ensures the DMA writes
target a different LDS slot than the `ds_read` operations. Eliminating it allows
the DMA writes to overlap with the `ds_read` operations, improving performance.

## Assembly Structure (2-stage pipeline, main loop)

```
.LBB0_1:                         ; Main loop
    s_waitcnt vmcnt(0)            ; [1] Wait for PRIOR iteration's DMAs (needed)
    s_barrier                     ; Cross-wavefront sync

    buffer_load_dwordx4 ... lds   ; DMA write to NEW slot (LHS)
    buffer_load_dwordx4 ... lds   ; DMA write to NEW slot (LHS)
    buffer_load_dwordx4 ... lds   ; DMA write to NEW slot (RHS)
    buffer_load_dwordx4 ... lds   ; DMA write to NEW slot (RHS)

    ; >>> THIS IS THE CRITICAL POINT <<<
    ; WITHOUT fix: s_waitcnt vmcnt(0)  <-- Conservative, blocks overlap
    ; WITH fix: (no vmcnt here)        <-- DMAs overlap with ds_reads

    ds_read_b128 ...              ; Read from PREVIOUS slot (LHS)
    ds_read_b64_tr_b16 ...        ; Read from PREVIOUS slot (RHS)
    ...
    v_mfma_f32_16x16x32_f16 ...  ; Compute
    ...
    s_cbranch_scc1 .LBB0_1
```

Wait [1] is necessary and expected. The critical optimization is removing the
second `vmcnt(0)` between the new DMA writes and the `ds_read` operations.

## Backend Mechanism: SIInsertWaitcnts

`SIInsertWaitcnts` in `llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp` tracks
in-flight `load_to_lds` DMA stores in LDSDMA slots. When a `ds_read` is
encountered:

1. If the `ds_read` has `AAInfo` (alias metadata) and `Ptr`, it enters the
   "smart path" and checks `mayAlias(AA, *LDSDMAStores[I])` for each tracked
   DMA store.
2. If the `ds_read` has no `AAInfo`, it falls back to waiting on all DMAs
   (`vmcnt(0)`).

The key insight is that `load_to_lds` stores are tracked per-slot only if they
have `AAI.Scope` (alias_scope metadata). Without it, only the generic LDSDMA
score is updated, and the smart path for `ds_read` finds no stores to compare
against.

## Solution: Explicit Async Operations

The proper solution uses `rocdl.load.async.to.lds` + `rocdl.asyncmark` +
`rocdl.wait.asyncmark` instead of relying on alias analysis. This maps to:
- `amdgpu.gather_to_lds async` (lowers to `rocdl.load.async.to.lds`)
- `rocdl.asyncmark` (marks end of a DMA group)
- `rocdl.wait.asyncmark N` (waits until N or fewer groups are pending)

With explicit async, `SIInsertWaitcnts` does NOT insert any `vmcnt` for the
`load.async.to.lds` operations. Instead, the compiler-inserted `asyncmark`
and `wait.asyncmark` provide precise synchronization.

## Verification

Use `verify_pipeline_vmcnt.sh` to check that the second `vmcnt(0)` is absent
from compiled assembly.
