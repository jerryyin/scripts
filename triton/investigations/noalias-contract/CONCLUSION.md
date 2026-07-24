# The `noalias` contract for Triton gfx1250 kernels — consolidated report

*This is the single canonical writeup of the investigation (it replaces the former
REPORT_A mechanism / REPORT_B dossier / REPORT_C runtime split). Deep per-cell evidence
— frozen experiments, real IR/asm, A/not-A kill switches, refutation trails — lives in
`ledgers/S*.md`, `R*.md`, `RT*.md`; `REGISTRY.md` is the cell→conclusion→ledger index.*

**Question that started this:** the MoE prefill gather-index `noalias` win (#1885 /
PR #120) looked important; exposing `tt.noalias`/`tt.readonly` as a first-class
cross-kernel contract is a big undertaking — so is it *essential*, and *how far does it
generalize*?

**Answer, one line:** `noalias` is essential in exactly one place among all AITer gfx1250
kernels — the MoE prefill gather-index load — and it does **not** generalize. Broadening
it beyond F1 yields **zero** additional codegen wins (proven by md5-identical `.text`
across every other group). The big undertaking is **not justified**; the F1 contract is
the whole prize.

> **CORRECTION (post-review, 2026-07-20; re-audit complete).** An independent user
> investigation exposed a methodology flaw: the whole investigation annotated only the
> *single index pointer* per kernel, never "all uniform read-only args." Findings after
> reproducing the report and re-auditing every family under the corrected all-args protocol:
> 1. **Decode is fixed by the frontend contract, not the LLVM patch.** Old S10 was **wrong**.
>    Decode's 8 in-loop `v_readfirstlane` come from the expert-metadata loads
>    (`ExptData/ExptHist/ExptOffs`) staying VGPR; adding those to `noalias_args` scalarizes
>    them → **8→0 on stock LLVM**, FFM PASS (reproduced here, S10b). Same `noalias→readonly→
>    s_load` chain (§1–2), one hop upstream. Recommendation #2 rewritten.
> 2. **F2–F6 re-audited under the all-args protocol → all still no-ops, decode was the SOLE
>    false negative.** Two reasonings were corrected: (a) F2's old "already s_load" was measured on
>    the *prefill* mla kernel; the *decode* kernel `_mla_decode_fwd_kernel` has a 1×in-loop-rfl
>    round-trip on the software-pipelined block-table prefetch — root cause **pinned to an AMDGPU
>    backend bug** (`isReallyAClobber` treats the in-loop LDS/TDM def as a clobber *without* an AA
>    check → no `!amdgpu.noclobber` → VMEM+`readfirstlane`), NOT aliasing or representability, and
>    **`noalias`-independent** (S10c; fix implemented + verified); (b) the F3/F4/gmm "divergent"
>    claims were re-confirmed by uniformity analysis, and F5's "no raw load" re-confirmed (0 `tt.load`).
> 3. **Soundness, concretely (RA-F2):** annotating the *data* pointers (query/kv/scales) —
>    which are NOT uniform-read-only and one aliases a writable output — **miscompiles** (80.4%
>    mismatch). The contract must be restricted to genuinely uniform read-only args (see §8).
> The mechanism model (§1–2) is unchanged and was reinforced by the re-audit.

---

## Frozen provenance (shared across all cells)

- **Runs:** `wf_ba8e0fce-520` (S1–S16, mechanism + families) and `wf_876cb89a-3d2`
  (Run 2: runtime, kernel fixes, open items). 27 fresh-context agents; ledgers-only state.
- **Triton:** `/root/triton` @ `ba4fd67b8ed2…` (branch `users/jerryyin/moe-gather-sload-contract`,
  #120 contract landed; installed `triton` 3.8.0 resolves here). The contract is a Python
  annotation (`@{triton,gluon}.jit(noalias_args=[...])`) — **no C++ rebuild** to toggle it.
  `triton-opt`: `/root/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt`.
- **AITer:** `/root/aiter`. R3 edited three kernel source files (§6); tree left clean, no commits.
- **LLVM pins (compile-only ISel/opt study, LLVM 23.0.0git, assertions):**
  `/root/.triton/llvm/llvm-{56421f92,62b7cf96}-*/bin/{opt,llc}`. Every decisive case selects
  identically on both pins → results are **not** an LLVM-version artifact. Patched tree
  `/root/llvm-project/install` = `56421f921` + `db4972674` (MachineLICM `readfirstlane` hoist),
  used only for the F1-decode leg (§6/S10).
- **Target:** gfx1250 (`-mcpu=gfx1250`). Mechanism claims are compile-only (pre-opt
  `LLVM_IR_ENABLE_DUMP`, `triton-opt` conversion, stock `opt`/`llc`) — an ISel/opt-selection
  claim needs no execution.
- **Correctness:** FFM-lite, gfx1250, rel-close at smallest exercising shape (FFM = correctness,
  **no timing**). F1 prefill is byte-identical numerically ON vs OFF (rel_err 0.01001 = the a8w4
  quant tolerance) — the contract is a pure hint.
- **The kill switch throughout:** the only input delta between BEFORE/AFTER in the F1 study is the
  single token `noalias` on arg `%19`; both post-Triton-opt `.llir` are otherwise byte-identical
  (325 loads / 32 stores / 0 calls in both). Every behavioral difference is attributable to it.

---

## 1. The rule (what `noalias` actually does)

`noalias` changes codegen **only when may-alias is *actively blocking* an optimization AND
`noalias` is the *only* thing that removes the block.** Both halves must hold; byte-identical
`.text` everywhere else is the *evidence*, this rule is the *cause*. For a win to exist, **all
four** conditions must hold:

1. a **raw-pointer load** — `tt.load` → `global_load`, not TDM DMA, not a buffer op, not `ptrtoint`;
2. a **uniform** address (per-lane divergent ⇒ VMEM forever);
3. a **scalarizable-to-SMEM shape** — dword-granular (i32/i64 coalesce to wide `s_load_dwordxK`)
   **or** a single-element sub-dword extload (`s_load_u8/u16`); a *sub-dword vector* (`<N×i16>`)
   has no scalar SMEM form and falls to VMEM (the sub-dword ceiling, §2.2);
4. an **interfering in-loop write whose may-alias LLVM cannot otherwise disprove** — so that
   `noalias` is the *only* route to invariance. If no confounding write exists, LLVM's own
   `readonly`-inference already makes the load invariant and `noalias` is inert.

**Mechanism refinement (correcting the shipped mental model):** the operative chain is
`noalias` → (alias analysis proves every store NoAlias) → **`!amdgpu.noclobber`** on the load →
scalar `s_load`. It is **not** `noalias → readonly → s_load`: arg-`readonly` ("not written
*through this pointer*") is inferred by LLVM in *both* on/off states (S1/S4) and does not feed the
cross-pointer no-clobber analysis — so `readonly` is neither necessary nor sufficient. The gap
`noalias` closes is aliasing *through a different, writable argument*, a caller-only fact
arg-`readonly` cannot express. `readonly`-inference does most of the work for free; `noalias` only
adds value where a write defeats that inference.

---

## 2. Mechanism — per-optimization verdict (necessary / sufficient / neither)

*Necessary* = removing `noalias`, all else equal, kills the win. *Sufficient* = adding `noalias`
alone (given the default-on pipeline) makes it fire. Consolidated matrix (evidence in S4–S8):

| LLVM/AMDGPU optimization | verdict | where the win lives / caveat |
|---|---|---|
| invariance carrier (backend `!amdgpu.noclobber`) → scalar `s_load` (T1/T2) | **necessary; sufficient jointly** with default-on `-amdgpu-scalarize-global-loads` | **This IS the shipped F1 win.** Generic `!invariant.load`/LICM: *neither* (absent). `readonly`: *neither* (present in both). |
| ISel `s_load` selection gate (T2) | necessary for the noclobber leg **only where a store can clobber**; never sufficient alone | align≥4 / uniformity / addrspace / representability are independent gates. Sub-dword *vector* → VMEM even when uniform+noclobber (ceiling). |
| `global_load` ⇒ `buffer_load` (T3) | **neither** | Gated on 32-bit-offset provability (`tt.pointer_range`) in Triton MLIR; aliasing never consulted. |
| `s_buffer_load` (T3) | **neither** | Not emitted by Triton at all; unreachable by any current contract. |
| LoadStoreVectorizer widening (T4, IR) | **necessary; sufficient** (noalias on either base) | Proven on minimal A/not-A; real-kernel firing site is a family question. |
| SILoadStoreOptimizer coalescing (T4, MI) | **necessary; sufficient** (indirectly, via ISel) | Unblocked because noalias makes both loads scalar-noclobber, not by alias metadata. |
| SLPVectorizer (T4) | **neither** | Wrong pass — vectorizes arithmetic, not these memory ops. |
| GVN / CSE redundant-load elim (T5) | *sufficient in general* (synthetic); **neither** in the shipped kernels | No firing site: all stores are LDS (already NoAlias to global loads). |
| DSE / MemCpyOpt / alias-gated scheduling (T5) | **neither** | No store made dead / no memcpy idiom / memory order unchanged. |

### 2.1 The shipped win — invariance → `s_load` (S4)
`noalias %19` → `FunctionAttrs` folds it into `noalias nofree readonly captures(none)` →
`AMDGPUAnnotateUniformValues::visitLoadInst` tags `!amdgpu.uniform` and, because
`isClobberedInFunction`→`isReallyAClobber`→`AA->alias(load,store)` returns **NoAlias** for every
writeonly store when the base is `noalias`, also tags **`!amdgpu.noclobber`** → the global-load
scalarizer selects `s_load_u16` → the TDM `tensor_load_to_lds` descriptor is built once in SGPRs
in the prologue instead of per-iteration in VGPRs → the 16 in-loop `v_readfirstlane` vanish
(**16 → 0**; whole-fn `s_load` 9→41, `global_load` 37→5, `v_readfirstlane` 37→3, with the 66 hot-loop
LDS ops byte-for-byte unchanged). The 2×2 (noalias × scalarize-global) shows `s_load` appears **iff**
both hold; `!invariant.load` count is 0 in both — the carrier is `!amdgpu.noclobber`, not LICM.

### 2.2 The sub-dword ceiling (S5)
`s_load` fires iff (SIISelLowering.cpp:13148-13158): uniform address; no-clobber (from `noalias` or
`readonly`-derived invariant MMO); global/const addrspace + scalarize-on; align≥4; and a
representable scalar SMEM node. AMDGPU has **no sub-dword *vector* SMEM instruction**
(`SMInstructions.td:329-332`; no `s_load_u16x4`), so a contiguous sub-dword load coalesced into
`<N×i16>` has no scalar form → VMEM even when uniform+noclobber. This is why the a8w4 uint16 gather
stays 32× narrow `s_load_u16` (scattered, per-element — fine) and cannot be *widened* without
dropping to VMEM; an i32 index coalesces to wide `s_load_dwordxK`.

### 2.3 buffer_load (S6) — Yin's question, answered "no"
`noalias` is neither necessary nor sufficient. Buffer selection is a *representation change* made
in Triton MLIR by `tritonamdgpu-convert-buffer-ops` (`ConvertToBufferOps.cpp::canUseBufferOps`
135-173), gated on 32-bit-offset provability (`tt.pointer_range=32` / 2GB range analysis);
`grep -c noalias` in that file = 0. The only LLVM buffer pass, `amdgpu-lower-buffer-fat-pointers`,
merely lowers a *pre-existing* addrspace(7) pointer — nothing synthesizes a buffer from
`addrspace(1)+noalias`. (The F1 win is `s_load` = scalar *global*, a distinct instruction from
`s_buffer_load`; do not conflate.)

### 2.4 / 2.5 widening and middle-end opts (S7, S8)
`noalias` **does** unblock LoadStoreVectorizer (IR) and SILoadStoreOptimizer (MI) widening —
necessary and sufficient in a minimal A/not-A (2×2: neither annotated → no widening; noalias on
either base → `load <2 x float>` / `s_load_b64`). But in the *shipped* kernels it unblocks **none**
of GVN/DSE/CSE/MemCpyOpt/scheduling: all 32 stores in `_moe_gemm_a8w4_prefill` are `addrspace(3)`
LDS, which AA proves NoAlias to the `addrspace(1)` gather loads *by construction* — so there is no
clobbering store for `noalias` to disambiguate (`-stats` diff empty; `aa: 853 MayAlias / 5365
NoAlias` identical on/off). GVN load-forwarding is sufficient in the abstract (synthetic
`load p; store q; load p`), just with no firing site here.

---

## 3. Why F1 prefill is the sole win, and everything else a no-op

**F1 prefill** is the one kernel hitting all four conditions: its read-only, uniform gather-index
load sits in the K-loop next to a **TDM descriptor store** that LLVM's AA cannot prove non-aliasing
→ condition #4 bites → only `noalias` recovers the `s_load`.

Every other group fails a *different* condition — and note several are gather/scatter *index* loads
that are still no-ops, so "gather/scatter" is **not** the discriminator:

| Failure mode | Condition failed | Families |
|---|---|---|
| **Already `s_load` at baseline** (readonly-inference suffices; no interfering write) | #4 | attention `block_tables` (mla), PA `kv_indptr`, kv-cache `slot_mapping`, causal_conv1d indices |
| **No raw-pointer load exists** (all TDM / `ptrtoint`) | #1 | norm (`fused_rmsnorm_add`), GEMM A/B (`gemm_a16w16/mxfp4`) |
| **Load can't be scalarized** (divergent vector) | #2/#3 | kv-cache read-gather (`gather_kv_b_proj`), gmm `group_sizes` |
| **No firing site** (single trailing store, no reload/RMW) | #4 (differently) | GEMM C buffer_store |

Corollaries: (a) the read/write nature of A/B/C is a **red herring** — GEMM primaries miss out
because they flow through TDM (no pointer load), not because C is written; (b) **plumbing is not
the blocker** — any pointer incl. primary A/B/C is annotatable today with zero backend change;
the only removed capability is *user-declared* `readonly` (deleted in #120, now inferred only),
which no scanned family needs (S2).

---

## 4. Complete per-family verdict

| Group | Kernel(s) | Verdict | Why | Ledgers |
|---|---|---|---|---|
| **F1 prefill** | `moe_op_gemm_a8w4` prefill | ✅ **REAL WIN — shipped (#120)** | Only family satisfying all 4 conditions; in-loop rfl 16→0 | S4 |
| **F1 decode** | `moe_op_gemm_a8w4` decode | ✅ **Win — expanded frontend contract (stock LLVM)** | 8 in-loop rfl come from expert-metadata loads (`ExptData/ExptHist/ExptOffs`) staying VGPR → `SIFixSGPRCopies` VGPR↔SGPR round-trip. Adding those to `noalias_args` scalarizes them → index stays SGPR → **8→0 on stock LLVM**, FFM PASS. MachineLICM `db4972674` is now only a backstop. | **S10b** (corrects S10) |
| **F2 attention/PA** | mla (prefill+**decode**), unified_attention, pa_decode_sparse | ❌ no-op *(root cause pinned → backend bug)* | mla **decode** has a 1×rfl round-trip on the pipelined block-table prefetch; cause = AMDGPU `isReallyAClobber` skips the AA check for the in-loop LDS/TDM def → no `!amdgpu.noclobber` → VMEM+rfl (backend-fixable, `noalias`-independent, S10c); pa gather truly divergent | S9, RA-F2, **S10c** |
| **F3 GEMM** | gemm_a16w16, gemm_mxfp4 (incl. primary A/B/C) | ❌ no-op *(re-audited)* | A/B+scales via TDM `ptrtoint`; the one raw load (bias) is genuinely divergent; 0 round-trip | S11, **RA-F3** |
| **F4 KV-cache** | fused_kv_cache, kv_cache, gather_kv_b_proj | ❌ no-op *(re-audited)* | metadata already `s_load` at baseline; read-gather **truly divergent** (uniformity-confirmed); only scheduler noise | S12, S12b, **RA-F4** |
| **F5 norm** | fused_rmsnorm_add | ❌ no-op *(re-audited)* | genuinely 0 `tt.load` — all `make_tensor_descriptor`; nothing for noalias to act on | S13, **RA-F5** |
| **F6 causal_conv1d** | causal_conv1d | ❌ no-op *(re-audited)* | full uniform-arg set = the 3 already tested; i32 indices already `s_load`; i8 flag = sub-dword ceiling | S17, **RA-F6conv** |
| **F6 gmm** | gmm | ❌ no-op *(re-audited)* | `group_sizes` load **truly divergent** (uniformity-confirmed, per-lane arange); byte-identical asm | S18, **RA-F6gmm** |

**Net: 1 shipped win (F1 prefill) + 1 modest LLVM-patch win (F1 decode); six other groups proven
no-ops; zero surviving candidates.**

---

## 5. Runtime status (the one open caveat)

Impact was to be measured two ways — **gfx950** (MI355X) for non-TDM kernels, **AM** for TDM
kernels — but **both tracks are blocked by host environment/ABI issues orthogonal to `noalias`**,
and the agents refused to fabricate numbers:

- **AM (F1/TDM) — blocked (R1/RT1/RT2):** AM emits per-dispatch cycle timing (proven via a
  HelloWorld run), but torch/Triton device init segfaults —
  `libdtif.so: undefined symbol hsaKmtCreateQueueV2` (AM ROCm 7.13 DTIF vs host torch 2.11 / ROCm
  7.14 KMT-thunk ABI mismatch) — before any kernel dispatches. FFM on the same tree passes,
  isolating the fault to AM.
- **gfx950 (non-TDM) — blocked (R2):** the only installed torch is compiled `gfx1250`-only
  (`get_arch_list()==['gfx1250']`) → every MI355X launch fails `hipErrorInvalidImage`.

Two honest notes: (1) the gfx950 track is **moot regardless** — every non-TDM survivor is a no-op,
so there's nothing to time even with a gfx950 torch; (2) the impact verdict does **not** weaken,
because every group that could have shown a delta was proven **byte-identical** (md5-matched
`.text`) — asm-identity is a stronger, noise-free proof of zero runtime impact than any measurement.
The only real win (F1) remains quantified by the **static ~5.7%-of-K-loop** estimate (decode, S10)
and Kyle's original **~1% microbench** (prefill) — consistent with the #1885 ATT finding that the
churn is cheap per-op (the win is real but small). A matched AM/torch stack (or a real gfx1250
device) is needed to convert the static estimate to a live cycle count.

---

## 6. Non-compiling kernels — fixed and re-measured (R3 → S9b, S12b)

R3 fixed both with minimal, behavior-preserving edits (git-confirmed root causes, controlled
reverts, FFM PASS, tree clean):
- `pa_decode_sparse.py` (2 `async_gather`) + `fused_kv_cache.py` reshape (1 `async_gather`): dropped
  the removed `src_col_offset=0` positional (triton PR #61 made `async_tdm_gather` pure); every call
  gathered at column 0, so the migration cannot change which rows/columns are read.
- `fused_kv_cache.py` MLA-cat launcher: inserted the missing `MAX_EMBD_POS` (`cos.shape[0]`) arg —
  without it every stride shifted one slot and `kv_cache_stride_d` fell off. `MAX_EMBD_POS` is unused
  in the kernel body (semantically inert).

Both re-measures confirmed the Run-1 predictions: **no-ops** (S9b: the one raw index already
`s_load_b64` at baseline, the other rides TDM; S12b: primary `slot_mapping` already `s_load_b64` at
baseline, code section byte-identical, metadata-only diff).

---

## 7. Recommendation

1. **Keep the F1 prefill contract** (`noalias_args=["GatherIndx"]`) — shipped, sound, the one real win.
2. **Fix F1 decode with the expanded frontend contract** —
   `noalias_args=["GatherIndx","ExptData","ExptHist","ExptOffs"]` takes decode's in-loop
   `v_readfirstlane` 8→0 on **stock** LLVM (FFM PASS, S10b); no patch required. Keep the
   MachineLICM patch `db4972674` only as a backstop for kernels that don't declare the full
   contract. **Lesson: annotate *every* uniform read-only arg feeding a gather index, not just
   the index pointer** — miss one and it silently regresses.
3. **Do NOT broaden `noalias`** to F2/F3/F4/F5/F6 or to primary operands — proven no-ops that add
   only silent-miscompile liability (§8) with zero codegen upside.
4. **Do NOT re-expose user-declared `readonly`** unless a future kernel surfaces a read pointer where
   LLVM fails to infer it (~3-line backend revert + a `readonly_args` kwarg if ever needed).
5. **Runtime quantification of F1 is optional** — static/microbench numbers already bound it as a
   small win; pursue the AM torch-free path (compile `.hsaco` on host, dispatch under AM with a bare
   HIP launcher — `run_on_model.sh --capture`) only if a hard number is required.

## 8. Soundness reminder (wherever the contract IS used)

The caller must guarantee the annotated pointer is never written through in-kernel (readonly axis)
AND never aliases a writable arg (noalias axis). A violated promise is a **silent** miscompile —
LLVM CSEs the stale pre-store value and the gfx1250 scalar K-cache is incoherent with in-kernel
vector stores → wrong gather index → wrong output, no crash, no diagnostic. The MoE caller honors
both (S3).

**Concrete demonstration (RA-F2):** annotating the mla *data* pointers (query / kv_buffer /
query_scales) with `noalias` — pointers that are NOT uniform-read-only, one of which aliases a
writable output — produces a **silent 80.4 % output mismatch** under FFM. This is exactly the
failure the contract must avoid: restrict `noalias_args` to genuinely uniform read-only args, and
never to a pointer that can alias a written buffer. The expanded decode set
(`GatherIndx/ExptData/ExptHist/ExptOffs`) is safe precisely because all four are read-only metadata.

---

*Every claim above is grounded in a ledger fact with the counter-experiment that failed to refute
it. No fabricated numbers: where a measurement could not be taken (AM/gfx950), it is stated as a
grounded blocker. Runs `wf_ba8e0fce-520` and `wf_876cb89a-3d2`.*
