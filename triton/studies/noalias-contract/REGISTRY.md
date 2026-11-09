# Session registry — the topic × family matrix, as an ordered queue

Status: `pending` | `in-progress` | `done` | `blocked`. Work the lowest unblocked ID.
Each session updates its own row (status + one-line conclusion + ledger link) and nothing else.
"Blocked-by" must be `done` before a row can start.

> **RUN wf_ba8e0fce-520 COMPLETE (2026-07-17).** S1–S16 all `done`, zero `blocked`. Conclusions
> merged below from each ledger's `STATUS`/`CONCLUSION`. Consolidated deliverable: `CONCLUSION.md`
> (the former REPORT_A/B/C were folded into it). Two new survivor cells (S17, S18) from S14 triage.

| ID | Phase | Cell (topic × family) | Question driven to the bottom | Blocked-by | Status | Conclusion (one line) | Ledger |
|----|-------|------------------------|-------------------------------|-----------|--------|-----------------------|--------|
| S1 | 0 | T0 (baseline) | What alias attrs do gfx1250 Triton kernel ptr args carry today + default aliasing assumption? | — | done | Ptr args carry ZERO alias attrs by default (bare may-alias); only opt-in `noalias_args` adds `llvm.noalias` (nothing else); 1 file opts in today. | ledgers/S1.md |
| S2 | 0 | T7 (plumbing) | How does noalias/readonly flow tt.func→llvm; what's missing to annotate any arg incl. primary? | — | done | Any pointer incl. primary A/B/C is annotatable TODAY (no role gate, no backend change); ONLY gap = user-declared `readonly` (deleted in #120, now LLVM-inferred only; ~3-line re-add if ever needed). | ledgers/S2.md |
| S3 | 0 | T6 (soundness) | Exact caller contract; miscompile mechanism; readonly vs noalias; cross-arg UB. | — | done | Caller must guarantee index never written-through (readonly axis) AND never aliases a writable arg (noalias axis); a lie → LLVM CSEs stale pre-store value + K$ incoherence → silent wrong output, no diagnostic. | ledgers/S3.md |
| S4 | 1 | T1 × F1 | How noalias⇒readonly⇒invariant⇒LICM/s_load; canonical prefill proof. | S1 | done | Real carrier = backend `!amdgpu.noclobber` (AA sees noalias), NOT `!invariant.load`/`readonly`; noalias is the single necessary emission, sufficient jointly with default-on `-amdgpu-scalarize-global-loads`; 16→0 rfl. | ledgers/S4.md |
| S5 | 1 | T2 × F1 | When does uniform+invariant⇒ISel s_load fire vs not (incl. sub-dword ceiling)? | S4 | done | s_load iff {uniform addr, no-clobber, global/const + scalarize-on, align≥4, dword-granular OR scalar sub-dword extload}; ceiling = no sub-dword *vector* SMEM, so coalesced `<N×i16>` falls to VMEM. | ledgers/S5.md |
| S6 | 1 | T3 × F3 | Does noalias enable global_load⇒buffer_load/s_buffer_load? Pass needed? | S1 | done | NEITHER necessary nor sufficient: buffer_load is a Triton MLIR pass (`convert-buffer-ops`) gated on 32-bit-offset provability, not aliasing; s_buffer_load never emitted by Triton. | ledgers/S6.md |
| S7 | 1 | T4 (× F1/F3) | Does noalias unblock LoadStoreVectorizer/SLP/SILoadStoreOptimizer widening? | S1 | done | YES at both IR (LoadStoreVectorizer) and MI (SILoadStoreOptimizer) levels — necessary AND sufficient in A/not-A cases; SLP not relevant (arithmetic, not memory). | ledgers/S7.md |
| S8 | 1 | T5 (× F1/F3) | Which of GVN/DSE/CSE/MemCpyOpt/scheduling does noalias unblock here? | S1 | done | NONE in the shipped kernels (all stores are LDS/addrspace(3), already NoAlias to the index loads); the whole win is the ISel s_load effect. GVN forwarding sufficient in the abstract, not observed here. | ledgers/S8.md |
| S9 | 2 | F2 (attention/PA kv-index) | noalias story for PA/attention index loads: mechanism + delta + FFM. | S4,S5,S3 | done | INERT: mla `block_tables_ptr` already s_load_b32 at baseline via readonly-invariant leg (no in-loop aliasing store); noalias = 2 metadata lines, 0 instructions. Other F2 kernels non-candidates/non-compilable. **NO-SHIP.** | ledgers/S9.md |
| S10 | 2 | F1 decode residual | Drive decode 8×rfl MachineLICM residual to ship/no-ship LLVM-patch decision. | S4,S5 | done | Residual = loop-invariant convergent descriptor lift (not load-selection); stays 8 under contract; LLVM patch db4972674 hoists 8→0 byte-identically. **SHIP the patch (modest ~5.7% of K-loop); 8 acceptable fallback.** | ledgers/S10.md |
| S11 | 2 | F3 primary A/B/C | Does noalias on primary GEMM operands unblock anything? Where blocked? | S6,S7,S8 | done | Unblocks NOTHING as written: A/B via TDM ptrtoint (provenance lost) → inert; C buffer_store propagates noalias but has no firing site (single trailing store). Not read/write nature, not plumbing. **NO-SHIP.** | ledgers/S11.md |
| S12 | 2 | F4 (KV-cache fusion) | noalias story for fused_kv_cache/kv_cache/gather_kv_b_proj. | S4,S5,S3 | done | Buys NOTHING: read-gather `kv_indices` is a divergent VECTOR buffer_load (S5 ceiling); scatter `slot_mapping` already s_load at baseline; gluon primary won't compile (tree drift). **NO-SHIP.** | ledgers/S12.md |
| S13 | 2 | F5 (norm) | noalias on fused_rmsnorm_add aux/scalar pointers: any asm delta? | S4,S7,S3 | done | ZERO asm delta (amdgcn bitwise identical, same md5): all traffic is TDM make_tensor_descriptor, no raw-pointer load exists; no dynamic scalar-ptr args. Grounded negative. **NO-SHIP.** | ledgers/S13.md |
| S14 | 2 | F6 (family scan) | Triage non-gluon ops/triton + causal_conv1d for a noalias-addressable pointer. | S1 | done | 2 survivors: `causal_conv1d` (uniform read-only i32 index loads → S5 s_load candidate, strongest) and `gmm` (`group_sizes_ptr` reloaded per work-steal iter + atomic → S8 LICM candidate). topk/softmax/activation = dead-ends. → S17/S18. | ledgers/S14.md |
| S15 | 3 | Synthesis A | Assemble the mechanism research report from T-ledgers. | S4–S8 | done | Folded into `CONCLUSION.md` §2; per-opt verdict: invariance→s_load NECESSARY+SUFFICIENT (w/ default pass); widening NECESSARY+SUFFICIENT; GVN/CSE sufficient-but-not-observed; DSE/MemCpyOpt/sched NEITHER; buffer_load NEITHER. | ledgers/S15.md |
| S16 | 3 | Synthesis B | Assemble broadened-impl plan + per-kernel dossier + asm-delta table. | S9–S14 | done | Folded into `CONCLUSION.md` §3–4; headline: the F1-prefill win does NOT generalize — 1 shipped + 1 modest LLVM-patch win, 4 proven no-ops, 2 unverified survivors; plumbing already supports any arg. | ledgers/S16.md |
| S17 | 4 | F6 causal_conv1d (NEW) | Measure noalias A/not-A on causal_conv1d uniform i32 index loads (S5 s_load candidate). | S5,S3 | pending | *(from S14 — strongest unverified survivor; UNMEASURED)* | |
| S18 | 4 | F6 gmm (NEW) | Measure noalias on gmm `group_sizes_ptr` (S8 GVN/LICM candidate, vector addr → not s_load). | S8,S3 | pending | *(from S14 — medium-confidence unverified survivor; UNMEASURED)* | |

Notes:
- Rows can be split if a cell proves larger than one session (append S9a/S9b, keep IDs stable).
- A `blocked` row must name the exact grounded blocker in its ledger (not "needs more work").

---

## Run 2 — close the caveats: runtime numbers, non-compiling kernels, open items

Goal: (1) real *measured* impact (variety of representative shapes per group), (2) fix
non-compiling kernels and re-measure, (3) close open items S17/S18. Two-track runtime: **gfx950**
(MI355X) for non-TDM kernels, **AM** for TDM gfx1250 kernels at small shapes. Same protocol,
orchestrated, cap 2, Opus/high, ledgers-only.

| ID | Phase | Cell | Question / task | Blocked-by | Status | Conclusion | Ledger |
|----|-------|------|-----------------|-----------|--------|-----------|--------|
| R1 | A | AM timing harness | Establish AM timing for a TDM kernel; gates AM runtime cells. | — | **blocked** | AM emits per-dispatch cycle timing, BUT torch/Triton device init segfaults under AM (`libdtif.so: undefined symbol hsaKmtCreateQueueV2` — AM ROCm7.13 DTIF vs host torch2.11/ROCm7.14 KMT-thunk ABI mismatch); no F1 TDM kernel dispatchable under AM here. | ledgers/R1.md |
| R2 | A | gfx950 runtime harness | Build MI355X runtime driver for non-TDM kernels; gates gfx950 runtime cells. | — | **blocked** | Harness built + validated to GPU-exec, BUT the only installed torch (2.11+rocm7.14) was compiled `gfx1250`-ONLY (`get_arch_list()==['gfx1250']`), so every MI355X launch fails `hipErrorInvalidImage`; needs a gfx950-capable torch build. | ledgers/R2.md |
| R3 | A | Fix non-compiling kernels | Edit AITer source so pa_decode_sparse + fused_kv_cache compile at frozen SHA. | — | done | Fixed with minimal faithful edits: 3 async_gather calls drop the removed `src_col_offset` positional (triton PR #61) + MLA-cat launcher gains missing `MAX_EMBD_POS`; all compile + FFM-correct. Enables S9b/S12b. | ledgers/R3.md |
| S17 | B | F6 causal_conv1d (open) | Full noalias A/not-A on gfx1250 (asm delta, s_load, FFM). | S5,S3 | done | **NO-OP:** `.text` byte-identical ON/OFF (md5 779cabac…); the i32 index loads are ALREADY scalar s_load at baseline (readonly-invariant leg, uniform program_id addr); i8 has_initial_states stays global_load (S5 sub-dword ceiling). NO-SHIP. | ledgers/S17.md |
| S18 | B | F6 gmm (open) | Full noalias A/not-A on gfx1250 (asm delta, GVN/LICM, FFM). | S8,S3 | done | **NO-OP:** amdgcn byte-identical (only the `noalias` IR token differs); reload NOT hoisted either way — AA already proves NoAlias (buffer-load addrspace(8) vs atomic addrspace(1)), and deleting the atomic still doesn't enable the hoist. S14 hypothesis refuted. NO-SHIP. | ledgers/S18.md |
| S9b | B | F2 pa_decode_sparse (re-measure) | Now compiling (R3): noalias mechanism + asm delta + FFM. | R3,S4,S5 | done | **INERT:** its one raw-pointer index (`kv_indptr`) already s_load_b64 at baseline (readonly-invariant); `kv_indices` rides TDM (no raw load). Metadata-only delta, 0 instructions. Confirms S9/S16. NO-SHIP. | ledgers/S9b.md |
| S12b | B | F4 fused_kv_cache gluon (re-measure) | Now compiling (R3): noalias mechanism + asm delta + FFM. | R3,S4,S5 | done | **INERT:** gluon primary `slot_mapping` is a uniform-scalar `s_load_b64` already at baseline (noclobber already set); `.text` byte-identical, metadata-only delta. Reshape sibling is the divergent-vector leg (S5 ceiling). NO-SHIP. | ledgers/S12b.md |
| RT1 | C | F1 prefill runtime (AM) | AM timing, variety of small shapes, ON vs OFF. | R1 | **blocked** | Bails clean on R1 (AM device init segfault re-verified live); no AM cycle number for prefill obtainable on this host. FFM control passes, isolating fault to AM. No fabricated numbers. | ledgers/RT1.md |
| RT2 | C | F1 decode runtime (AM) | AM timing, small shapes, +LLVM patch, ON vs OFF. | R1 | **blocked** | Bails clean on R1; 3-config decode delta not measurable dynamically here. Run-1 S10 static ~5.7%-of-K-loop estimate remains the only quantification. | ledgers/RT2.md |
| RT3 | C | causal_conv1d runtime (gfx950) | MI355X, ON vs OFF, only if S17 delta. | R2,S17 | done | No run — S17 asm-identical (md5 779cabac…) ⇒ runtime no-op by construction (gating rule). NO-SHIP. | ledgers/RT3.md |
| RT4 | C | gmm runtime (gfx950) | MI355X, ON vs OFF, only if S18 delta. | R2,S18 | done | No run — S18 asm-identical (md5 a45bf892…) ⇒ runtime no-op by construction (gating rule). NO-SHIP. | ledgers/RT4.md |
| S20 | D | Runtime-impact synthesis | Consolidate Run-2 numbers; write REPORT_C; fold into REPORT_B. | RT1,RT2,RT3,RT4,S9b,S12b | done | Every caveat closed EXCEPT F1 live cycle count: all 6 non-F1 groups proven byte-identical no-ops (broadening = zero extra wins); both runtime tracks blocked by host ABI issues orthogonal to noalias; F1 impact = S10 static ~5.7% only. Folded into `CONCLUSION.md` §5–6. | ledgers/S20.md |

Run-2 notes:
- No-op families (F2 mla, F3, F5, F4 no-op paths) get NO runtime — asm-identity is the proof.
- If R1 (AM) or R2 (gfx950) reports `blocked`, its dependent RT cells must bail-clean (read the
  harness ledger; if no harness, write `STATUS: blocked` naming the missing harness — do not flail).
- S9b/S12b are predicted no-ops (S9/S12); if a re-measure surprises with a delta, note it for a
  follow-up RT cell rather than expanding scope mid-run.

---

## Run 3 — post-review re-audit (corrected "all uniform read-only args" protocol)

Trigger: an independent user investigation found decode was a FALSE NEGATIVE — the whole
investigation had annotated only the *single index pointer* per kernel, never all uniform
read-only args. Reproduced (S10b) + re-audited every family; main assistant spot-checked
F2/F4/gmm independently (base-vs-after instruction-stream diff on stock llc).

| ID | Cell | Status | Conclusion | Ledger |
|----|------|--------|-----------|--------|
| S10b | F1 decode re-audit (reproduce user result) | done | **S10 was WRONG.** noalias on GatherIndx+ExptData+ExptHist+ExptOffs → in-loop rfl **8→0 on stock LLVM**, FFM PASS (rel_err 0.009413). Frontend contract fixes decode; MachineLICM patch is only a backstop. | ledgers/S10b.md |
| RA-F2 | attention/PA re-audit | done | No-op, **reasoning corrected**: mla *decode* has a real round-trip; noalias chain FIRES (noclobber 0→5) but ISel keeps pipelined `global_load_saddr` (S5 gate, not aliasing). Data-ptr annotation MISCOMPILES (80.4%). | ledgers/RA-F2.md |
| RA-F3 | GEMM re-audit | done | No-op confirmed. mxfp4 all-TDM (scales too); a16w16 bias is genuinely divergent; 0 SIFixSGPRCopies round-trip. | ledgers/RA-F3.md |
| RA-F4 | KV-cache re-audit | done | No-op confirmed. Metadata already s_load at baseline; read-gather truly divergent (uniformity); only scheduler noise. | ledgers/RA-F4.md |
| RA-F5 | norm re-audit | done | No-op confirmed. Genuinely 0 `tt.load` — all make_tensor_descriptor; nothing to act on. | ledgers/RA-F5.md |
| RA-F6conv | causal_conv1d re-audit | done | No-op confirmed. Full uniform set = the 3 S17 tested; already s_load; i8 flag = sub-dword ceiling. | ledgers/RA-F6conv.md |
| RA-F6gmm | gmm re-audit | done | No-op confirmed. group_sizes truly divergent (per-lane arange); byte-identical asm. | ledgers/RA-F6gmm.md |

**Run-3 verdict:** decode was the SOLE false negative; F2–F6 remain no-ops under the corrected
protocol (two reasonings fixed). The headline stands, strengthened: broadening `noalias` beyond
F1 yields no codegen wins — but F1 *decode* itself now wins via the expanded frontend contract
(not the LLVM patch), and the F2 mla-decode case is a "mechanism fires, blocked downstream by
pipelined-SADDR ISel" near-miss. Soundness: over-broad annotation (F2 data pointers) miscompiles.
