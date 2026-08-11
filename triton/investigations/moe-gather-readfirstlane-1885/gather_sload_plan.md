# Proposal: Scalar loads for wave-uniform TDM gather/scatter indices (gfx1250)

**Ticket:** AMD-Triton/triton-tickets#1885.
**Branches:** `triton: users/jerryyin/moe-gather-sload-shared` (on `amd/shared/gfx1250`),
`aiter: users/jerryyin/moe-a8w4-gather-api-migration`.

---

## 1. The problem

The MoE `a8w4` kernel gathers activation rows with a TDM gather. The row indices
come from a small `uint16` array (`GatherIndx`) whose Gluon layout **slices the
lane dimension away** — every lane of the wave holds the *same* indices:

```mlir
IDX_LAYOUT = SliceLayout(dim=0,
  BlockedLayout(sizePerThread=[1,16], threadsPerWarp=[32,1],
                warpsPerCTA=[1,num_warps], order=[0,1]))
%idx = tt.load %ptrs : tensor<128xi16, #ttg.slice<{dim=0, parent=#blocked}>>
```

The value is **wave-uniform**, but Triton lowers the load **per-lane**. On gfx1250
this produces one `global_load` per index element — 32 of them for this tile —
each fetching the *same* bytes into every lane, wasting 31/32 lanes and the
memory traffic, then packing them for the TDM descriptor:

```asm
; BASELINE — the wave-uniform index loaded as a per-lane vector load
global_load_u16 v1,  v4, s[8:9] scale_offset      ; 32x total, all lanes read
global_load_u16 v11, v6, s[8:9] scale_offset      ; the SAME address
global_load_u16 v12, v5, s[8:9] scale_offset
...                                               ; + v_lshl_or_b32 packing
```

Because the TDM `tensor_load_to_lds` descriptor operand is **scalar (SGPR)**, the
VGPR-resident indices must then be moved to SGPRs — the `v_readfirstlane` churn
that motivated the ticket.

**Goal:** load the wave-uniform index **directly into SGPRs** (a scalar load), so
there is no per-lane vector load and no VGPR→SGPR round trip.

---

## 2. The idea

A wave-uniform, read-only load can be issued as a **scalar memory load** whose
result lands in SGPRs — exactly what the TDM descriptor wants. Three questions
shape the design:

1. **Which loads?** Only those whose result feeds a TDM gather/scatter row-index
   operand — a scalar sink. (Scalarizing an arbitrary uniform load whose consumer
   is *vector* would just force a broadcast back to VGPRs.)
2. **Which instruction?** `s_load` (pointer scalar load) vs `s_buffer_load`.
3. **Is it safe?** A scalar load reads through the scalar cache, which is not
   coherent with in-kernel vector stores — so the buffer must be read-only.

---

## 3. The solution

A single, automatic transform:

> For a `tt.load` that (a) transitively feeds a TDM gather/scatter row-index
> operand, (b) is wave-uniform, and (c) is read-only, load it straight into SGPRs
> with a scalar load, addressed via layout-derived offsets.

Two pieces, both in the AMD backend:

**(a) Marking** — `collectUniformGatherIndexLoads`, run at the start of the
`ConvertToLLVM` pass. It walks each `amdgpu.async_tdm_gather/scatter`, traces the
row-index operand back through **layout-preserving** ops (bitcast + integer
elementwise) to the producing `tt.load`, and — if the load is read-only (base is
a kernel argument with no aliasing store in the function) — records it in a
`DenseSet<Operation*>` side-table that the pass hands to the lowering patterns.
Note there is **no uniformity check here**: the gather/scatter verifier already
requires the index layout to be lane-uniform (§4.2), and because the trace only
crosses layout-preserving ops the load shares that layout — so uniformity is a
guaranteed invariant, and read-only is the only condition left to prove. We use a
side-table rather than an IR attribute because the
decision is a private, pass-local handoff from analysis to lowering *within the
same pass*: an attribute would leak backend-internal state onto the op, would have
to be documented/verified/round-tripped, and — the concrete failure we hit — did
not reliably survive the intervening `make_ttgir` transforms when set earlier.
The side-table is computed and consumed in one pass, so it sidesteps all of that.

**(b) Lowering** — `LoadOpConversion`, for loads in that set: readfirstlane the
base address once, then GEP each element by its **true tensor-index offset from
the LinearLayout** and emit a plain scalar load. The readfirstlane'd *uniform
address* alone is enough for ISel to select `s_load` (we deliberately do **not**
assert `!invariant.load` — see §4.3), and it coalesces the contiguous run into
wide loads:

```asm
; +S_LOAD — the wave-uniform index loaded once, straight to SGPRs
s_load_b128 s[...], s[...]        ; index run coalesced into wide scalar loads
; (no global_load for the index; no per-lane waste; descriptor reads SGPRs)
```

Measured on the `a8w4` prefill GEMM1 (batch 2048 → block_m 128), the index's
per-lane vector load is **eliminated**:

| | `global_load_u16` (index) | index in |
|---|---:|---|
| baseline | **32** (+ packing) | VGPR |
| +s_load  | **0** | SGPR via `s_load_b128` |

Correctness unchanged: full forward PASS, rel_err `0.010`, cosine `0.99993`.

---

## 4. Design decisions

### 4.1 `s_load`, not `s_buffer_load`
`GatherIndx` is a plain global pointer, for which `s_load` is the idiomatic scalar
load — it needs no buffer resource. We spiked `s_buffer_load` too
(`@llvm.amdgcn.s.buffer.load` + a hand-built `<4xi32>` V#): it *selects* and even
coalesces better (`2× s_buffer_load_b256`), but on gfx1250 it reads **wrong data**
(rel_err `0.594`) — the classic `<4xi32>` V# does not match the gfx1250 v2i64
descriptor format, and there is no ptr-based scalar-buffer-load intrinsic to pair
with the correct `MakeBufferRsrcOp` resource. `s_load` is correct and simpler.

### 4.2 Consumer-scoped, and uniformity comes for free
The transform fires only for loads feeding a TDM gather/scatter index, not for
*every* wave-uniform load. A uniform value consumed by vector math would be
pessimized by SGPR residency (the backend would broadcast it back to VGPRs). The
TDM descriptor is a genuine scalar sink, so scalarizing there is a pure win — and
this scoping is what lets the transform run **unconditionally** (no env gate).

Scoping to the gather/scatter index also makes **uniformity a guaranteed
invariant rather than something we detect.** `AsyncTDMGatherOp`/`AsyncTDMScatterOp`
`::verify()` reject any index whose layout distributes values across lanes ("the
warp-level TDM instruction reads the descriptor from SGPRs — all lanes see the
same descriptor"); the check is `laneFreeMask == numLanes - 1` on the index's
LinearLayout — exactly the wave-uniformity predicate. (NVIDIA's `gather4` lowering
does the identical check; this is the shared "require-and-elect" philosophy —
mandate the index be uniform rather than recover it.) Since our trace to the
producing load only crosses layout-preserving ops, the load inherits that
verifier-guaranteed lane-uniformity. So gate 1 (uniformity) is *free* — no
`isWaveUniformTensorLoad` re-check — and §4.3 read-only is the only real work.
(We added the missing lane-uniform check to the scatter verifier to make this
symmetric; gather already had it.)

### 4.3 Read-only proven by the compiler
Scalar-cache loads are only sound on read-only memory, so before firing we
establish that the index buffer is never written. This comes in two levels that
are worth keeping distinct:

- **Function-local read-only — *proven*.** We trace the load's base to a
  kernel-argument `BlockArgument` and scan every memory-writing op in the function
  (`MemoryEffectOpInterface` → `Write` on `GlobalMemory`), tracing each write's
  target back to its base argument. If none is the index arg, *this function*
  never mutates that buffer under any name it can see. A write through the index
  arg itself is caught here and we bail. Unproven ⇒ fall back to the normal vector
  load (never miscompile). No source annotation required.
- **True read-only (incl. the caller) — *assumed*.** The one thing intra-procedural
  analysis cannot see is a caller that passes the *same buffer* as both `GatherIndx`
  and a written argument. Closing that needs an argument-level contract, discussed
  in the soundness boundary.

**We deliberately do not emit `!invariant.load`.** It is tempting — the memory is
read-only by construction — but `invariant` is a *promise to the optimizer* that
the bytes never change for the lifetime of the pointer, and it licenses four
things: hoisting the load, CSE/rematerializing it, **reordering it across stores**,
and **eliding a reload after a store**. The last two are the dangerous half: if the
read-only proof were ever wrong (the aliasing hole below), `invariant` lets LLVM
slide the load above an aliasing store or forward the pre-store value — a
*guaranteed* stale read, i.e. a silent miscompile. Empirically the flag buys us
**nothing** in exchange: the readfirstlane'd uniform address by itself makes ISel
select `s_load` (verified — dropping `invariant` leaves the index
`global_load_u16` count at 0 and all 289 `s_load`s intact, a8w4 still PASS). So we
take the free win and skip the promise. Dropping it converts the load from
"provably reorderable/forwardable" back to an ordinary load: because Triton emits
no `noalias`, LLVM's *default* alias analysis is conservative (may-alias) and
refuses to reorder or forward it across a store it cannot prove disjoint. This is
defense-in-depth, not a substitute for the read-only proof — it neutralizes only
the *optimizer's* half. The *hardware* half — the scalar K-cache staying stale
after an aliasing vector store — is inherent to `s_load` and untouched by any IR
flag; only the read-only proof closes it.

### 4.4 Layout-derived addressing, not naive `+i`
The register→memory order of the index tensor is a **permutation** (slice of a
blocked layout), so addressing element `i` at byte `i·size` is wrong (it silently
mis-reads). Instead we ask the LinearLayout for the tensor index `t_i` each
register holds and GEP by the constant delta `t_i − t_0` from one readfirstlane'd
base. This is correct *and* lets the run coalesce into wide `s_load`.

### 4.5 Marking at the ConvertToLLVM boundary
Read-only is proven by walking the function's memory effects against the load's
kernel-arg base — a fact of the final IR. We do it at the start of the
`ConvertToLLVM` pass so the analysis runs on the final IR and the side-table is
consumed in the same pass — computing it earlier ran into intervening
`make_ttgir` transforms that move/rewrite prologue loads and did not carry the
marking forward. (Uniformity is *not* recomputed here — the layout-preserving
trace plus the gather/scatter verifier already guarantee it; see §4.2.)

---

## 5. Where this runs: `shared/gfx1250`

The change is built on `amd/shared/gfx1250`, the AMD gfx1250 development branch,
which already carries the surrounding TDM work:

- **#10685** made `async_tdm_gather/scatter` "pure" — the column offset moved from
  a positional argument into the descriptor. The `aiter` a8w4 kernel is migrated
  to this API (companion branch): each `async_gather(desc, idx, col, dst)` becomes
  `async_gather(update_tensor_descriptor(desc, add_offsets=[0, col],
  clamp_bounds=True), idx, dst)`. `clamp_bounds=True` is required for correct
  tail-K OOB handling (without it, rel_err `0.024`; with it, `0.010`).
- **#10686** reuses one TDM descriptor across chunks, which removes the in-loop
  `v_readfirstlane` on the gather **descriptor** (the other half of the ticket).

Together with this proposal's index-load scalarization, the wave-uniform gather
index is now (a) loaded once into SGPRs via `s_load` (no per-lane vector load) and
(b) referenced by the descriptor with no in-loop churn.

---

## 6. Results & validation

`a8w4` on the FFM gfx1250 model, `run_moe_gemm_ffm.py`:

| kernel/phase | correctness | index `global_load` → | in-loop `v_readfirstlane` |
|---|---|---|---|
| a8w4 prefill (gather) | PASS rel_err 0.010 | 32 → **0** (`s_load`) | 16 (descriptor base; #10686 territory, unchanged by this change) |
| a8w4 decode (gather)  | PASS | index vector load → `s_load` | — |
| a4w4 decode (control) | PASS | n/a | — |

Unit tests:
- `test/Conversion/amd/uniform_index_sload_detect.mlir` — the real side-table
  path (no env): a read-only index feeding a gather is scalarized (readfirstlane'd
  address → `s_load`); adding a store to the index buffer forfeits read-only and
  leaves it a vector load.
- `test/TritonGPU/amd/invalid.mlir` — verifier negatives for a lane-distributed
  index on **both** gather and scatter (the uniformity guarantee this transform
  relies on).

Regression: the AMD TDM conversion lit suite passes.

Saved assemblies for inspection:
- `/root/a8w4_shared_baseline_prefill.amdgcn` — 32× `global_load_u16` for the index.
- `/root/a8w4_shared_sload_prefill.amdgcn` — 0 `global_load_u16`; index via `s_load`.
Quick check:
```bash
grep -c global_load_u16 /root/a8w4_shared_baseline_prefill.amdgcn   # 32
grep -c global_load_u16 /root/a8w4_shared_sload_prefill.amdgcn      # 0
```

---

## 7. Status & follow-ups

- **Done:** consumer-scoped, read-only, layout-derived `s_load` (un-gated, with a
  `TRITON_AMD_DISABLE_UNIFORM_SLOAD` escape hatch); aiter a8w4 migrated to the
  pure gather API; validated end-to-end on `shared/gfx1250`; both branches pushed
  to the `jerryyin` forks.
- **Scatter:** the marking already covers `async_tdm_scatter`; validate a
  scatter-heavy config end-to-end.
- **Descriptor-base in-loop `v_readfirstlane` (the residual 16):** owned by the
  TDM descriptor path (#10686 reduces it); a further reduction would come from the
  backend not rematerializing convergent `readfirstlane` under SGPR pressure,
  independent of the index load.
- **Upstreaming:** open PRs for the two pushed branches; keep read-only proof and
  the negative lit test as the correctness guards.

---

## 8. Robustness worklist (Phase-1 analysis hardening)

Framing: **uniformity is solved** (verifier-guaranteed precondition, carried to
the load by a layout-preserving trace). **Read-only is the whole struggle** — it
is where all the fragile analysis lives *and* it carries an irreducible
in-function assumption (no-alias-across-args). Lane-uniform does **not** imply
read-only; they are orthogonal, which is why read-only needs its own gate.

Two kinds of walk, opposite requirements:
- *Coverage* walks (`traceToProducingLoad`, `traceToBasePtr`, `resolveWriteBase`)
  may be incomplete — unrecognized op ⇒ fall through to a non-arg ⇒ caller bails ⇒
  lost optimization, never a miscompile. Fail-safe.
- *Soundness* default (the effect walk in `baseIsReadOnly`) must be conservative —
  a missed writer is a miscompile.

Worklist (agreed direction; details TBD in discussion):

1. **[correctness — the one real gap] `baseIsReadOnly` default.** Today it
   *skips* ops that don't implement `MemoryEffectOpInterface` (fail-unsafe: an
   opaque op or a `call` to a writer is treated as harmless). Flip it: skip only
   `isMemoryEffectFree(op)`; bail (`readOnly = false`) on anything opaque. In
   practice Triton inlines before ConvertToLLVM so there are no calls, but this is
   what makes it sound for arbitrary input.
2. **[clarity] De-enumerate `traceToProducingLoad`.** Replace the `arith::*`
   allowlist with *pure* (`isMemoryEffectFree`) **and** layout-preserving (operand
   shape+encoding == result) — covers any elementwise/bitcast op, still refuses
   `convert_layout`/`reshape`/`broadcast`/`reduce`. Strictly more robust, less
   code.
3. **[cleanup] Trust `EffectInstance::getValue()`.** The writer ops already attach
   `MemWrite<GlobalMemory>` to their pointer/desc operand (e.g.
   `AsyncTDMCopyLocalToGlobalOp`'s `$desc`), so `getValue()` is authoritative; the
   `isPtrOrDescType` operand-scan fallback is now dead. Remove it. Any op that
   reports a *bare* write is fixed in its `.td` (attach the effect to its
   operand), not worked around here; a bare write then bails (fail-safe).
4. **[deferred — fail-safe as-is] `traceToBasePtr` / `resolveWriteBase`.**
   Unwrapping a value to its underlying kernel buffer is inherently semantic
   (`addptr` base operand, `splat` src, descriptor `make/update` chain); no
   general "underlying object" interface exists in the Triton dialect. Keep the
   small walks, document the fail-safe contract (unknown ⇒ non-arg ⇒ bail), and
   optionally route the descriptor ops through a shared interface so future desc
   ops are covered automatically.

**Strategic question (discuss before committing to interim hardening):** read-only
is both the fragile part and the part with an unclosable in-function assumption.
The durable fix is to move it from *inferred* to *declared* — an arg-level
`readonly`+`noalias` contract surfaced from Gluon and lowered by `FuncOpToLLVM`
(see Soundness boundary). That deletes the effect walk **and** closes the
cross-argument aliasing gap at once. Open question: invest in the contract, or
ship the pt-5/1/2 hardening as the interim and layer the contract later?

---

## Appendix A — Worked example: the chain and the read-only walk

The analysis runs on the MLIR (TTGIR at the start of ConvertToLLVM), not the
assembly. The "load → gather" chain is a chain of IR ops; the assembly is
produced later. Real SSA names from the a8w4 prefill kernel:

**Chain A — index load → gather** (what "load to gather" means):
```mlir
%GatherIndx : !tt.ptr<i16>                              // kernel arg (BlockArgument)
%GatherIndx_12 = tt.addptr %GatherIndx, %start_m_6      // + per-expert offset
%offs_x_m_13   = tt.splat  %GatherIndx_12               // broadcast to a tensor of ptrs
%offs_x_m_14   = tt.addptr %offs_x_m_13, %offs_x_m_11   // + arange
%offs_x_m_15   = tt.load   %offs_x_m_14 : tensor<128xi16, #ttg.slice<{dim=0,...}>>  // THE LOAD (wave-uniform)
%offs_x_m_16   = arith.divui %offs_x_m_15, %cst_3       // // N_EXPTS_ACT
%3 = amdg.async_tdm_gather %1[%offs_x_m_16] to %2       // THE GATHER (uses the index)
```

**Chain B — the output write** (a separate chain, to a different buffer):
```mlir
%Y : !tt.ptr<f8E4M3FN>                                  // kernel arg (output, BlockArgument)
%Y_80      = tt.addptr %Y, %Y_79                        // + per-expert offset
%y_desc_81 = tt.make_tensor_descriptor %Y_80, [...]     // output descriptor, base = %Y_80
%36 = amdg.update_tensor_descriptor %y_desc_81 add_offsets=[...]  // positioned output tile
%37 = amdg.async_tdm_copy_local_to_global %36 from %y_buffer      // THE OUTPUT WRITE (global)
```

At the asm level these are distinct global accesses: index read
`global_load_u16 v1, v4, s[8:9]` (per-lane — what becomes `s_load`); output write
`tensor_store_from_lds s[24:27], s[16:23]` (descriptor store to `%Y`).

**The walk:**
1. *Marking* — from the gather `%3[%offs_x_m_16]`, `traceToProducingLoad` follows
   `divui → %offs_x_m_15` (the `tt.load`). Uniform (slice `lane` free). Base:
   `traceToBasePtr(%offs_x_m_14) = addptr → splat → %GatherIndx_12 → addptr →`
   **`%GatherIndx`** (arg).
2. *Read-only* — walk global writes; the one that matters is `%37`.

**Why the output write "looked like" it might hit the index.** `%37` declares a
*bare* `Write<GlobalMemory>` effect: it says *that* it writes global memory, not
*which* pointer (`getEffect().getValue()` is null). A conservative analysis then
sees "an op writes somewhere in global memory, target unknown," cannot prove that
"somewhere" isn't `%GatherIndx`, and bails. The misunderstanding is not that the
write targets the index — it's that the target is **unknown**, and
unknown-in-global ⊇ the-index.

**The fix recovers the target** (it does not relax safety). For a bare-write op,
inspect its own ptr/descriptor operands — the only things it can write through —
and resolve them:
```
%37 operand %36 (descriptor)
   → update_tensor_descriptor → getDesc() → %y_desc_81
   → make_tensor_descriptor   → getBase() → %Y_80
   → traceToBasePtr           → %Y   (arg)
```
`%Y ≠ %GatherIndx` — two distinct kernel arguments — so the write provably lands
on the output, not the index, and it stops blocking the optimization.

### Soundness boundary
- The **index need not be a kernel arg.** If the gather index is computed, or a
  splat constant, or otherwise not produced by a `tt.load`, `traceToProducingLoad`
  returns null and we simply don't fire — there is no memory load to scalarize.
  We *require the load's base to be a `BlockArgument`* only as the read-only
  anchor: it is a *sufficient* condition that lets us reason about aliasing by
  argument identity. A load off a non-argument base is conservatively left alone.
  For a **pointer-of-pointer** base (the base is itself loaded from memory), the
  mechanics of scalarizing would work fine — we readfirstlane the final address
  either way — but we cannot *prove read-only*: a runtime-loaded pointer can point
  into any buffer, so argument-identity aliasing has nothing to compare against.
  Allowing it would require a points-to analysis proving the loaded pointer targets
  read-only memory; that is a much harder, rarely-provable property, so we bail.
  This is a coverage limit, not a correctness bug.
- **Aliasing rests on one unproven assumption.** We strip the write's address
  arithmetic and compare only its base. *Same* base ⇒ bail (sound). *Different*
  argument ⇒ assumed disjoint — the no-alias-across-arguments convention. Triton
  does not emit `noalias` on pointer args, so this is an assumption, not a proof:
  a caller that passes the same buffer as both `GatherIndx` and `Y` would break
  it. We do **not** do offset-range analysis, so we cannot certify disjointness of
  overlapping-arg sub-ranges; we bail on same-base instead. The only fully sound
  closure is an **arg-level read-only/`noalias` contract**: the frontend knows the
  routing indices are an input, so Gluon would surface a `const`/read-only
  qualifier that `FuncOpToLLVM` lowers to LLVM `readonly` + `noalias` on the
  pointer argument (today it emits only `byval`/`align`/`grid_constant`, and there
  is no `tt.readonly` arg attribute upstream). That turns "different arg ⇒
  disjoint" from a convention into a checked fact — and, as with C `restrict`,
  reclassifies a caller that aliases anyway as *the caller's* undefined behavior
  rather than a latent compiler bug.
  - *No such qualifier exists in Gluon today* — the only `read_only` in the API is
    `tma.store_wait(read_only=)`, an unrelated barrier-scope flag. But the plumbing
    is a small, well-trodden extension, not new infrastructure: the frontend
    already attaches arbitrary arg attributes via `set_arg_attr` (that is how
    `tt.divisibility`/`tt.nv_tma_desc` are set), and `FuncOpToLLVM`'s
    `handleByvalTmaDescArgs` already maps one such attr (`tt.nv_tma_desc` →
    `byval`+`grid_constant`+`align`) — the direct template for a
    `tt.noalias`/`tt.readonly` → LLVM `noalias`/`readonly` mapping.
  - *For this hazard `noalias` is the load-bearing attribute, not `readonly`.*
    `readonly` on the index arg only re-states what our intra-procedural walk
    already proves (the kernel does not write *through that name*). The hazard is a
    *different, writable* arg overlapping the index buffer, and only `noalias`
    asserts no other pointer aliases it — which is exactly what makes "different arg
    ⇒ disjoint" a fact LLVM's alias analysis will honor. `readonly` is a useful
    complement (it would let us drop the walk), but `noalias` is what closes the
    hole.
- **Can a hostile caller still break it?** Yes — and this is the honest bound.
  Function-local read-only is *proven*, so a write through the index arg itself is
  always caught. But a caller that deliberately aliases `GatherIndx` with a
  written argument sits in the one blind spot intra-procedural analysis cannot
  cover by construction, and would get a silent stale read. There is no way to
  make this *physically* unbreakable while keeping the win: the coherent
  alternative is exactly the vector load we are removing, and runtime pointer-alias
  checks are not free. The realistic bar is therefore not "impossible to break"
  but "sound for every contract-abiding caller, UB for a violator" — the same
  guarantee `restrict`/`noalias` give everywhere else. The `readonly`+`noalias`
  attribute is what makes that guarantee *documented and enforced* rather than
  *implicit and assumed*. Note this optimization does introduce a new failure
  *mode*: unoptimized, an aliasing caller reads coherent (if racy) global memory
  and stays semantically defined; with `s_load` it reads stale — so the contract
  is what earns back the safety the scalar path gives up.
- **The two coverage gaps have very different severity — this is why they are
  treated asymmetrically.**
  - *Consumer-scope gap (benign, perf-only).* If we scalarized a uniform load
    whose consumer actually wanted vector data, the backend just broadcasts
    SGPR→VGPR — a wasted copy, never a wrong answer. This is why §4.2's consumer
    scoping can be a heuristic: getting it wrong costs cycles, not correctness.
  - *Read-only gap (malignant, silent miscompile).* If the read-only proof is
    wrong — a caller aliases `GatherIndx` with a written arg, and that write
    actually fires — the failure is **not** "the backend broadcasts a scalar."
    The gather reads through the scalar cache, which is incoherent with the
    aliasing vector store, so it can return **stale index bytes** → wrong rows
    gathered → wrong output, with no crash and no diagnostic (and possibly
    data-dependent/flaky). And no downstream register trick recovers it: a
    SGPR→VGPR *broadcast* happens **after** the load, so it just copies the stale
    bytes already in the SGPR — coherence is a property of the load's memory path,
    not of where the value lands. Dropping `!invariant.load` (§4.3) removes the
    optimizer-reordering half of this hazard, but the hardware-cache half is
    intrinsic to `s_load`. This asymmetry is exactly why consumer-scope may be a
    heuristic while read-only must stay conservative (bail on any doubt).
