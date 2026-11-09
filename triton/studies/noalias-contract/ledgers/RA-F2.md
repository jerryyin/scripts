STATUS: done
CONCLUSION: The corrected all-uniform-read-only-args protocol does NOT change the F2 verdict — it stays a NO-OP (byte-identical instruction stream), but the OLD no-op REASONING for mla was WRONG. S9's claim ("mla block_tables already s_load at baseline") was measured on the PREFILL kernel; the actual DECODE kernel (`_mla_decode_fwd_kernel`, never dumped before) DOES have a decode-style round-trip: its in-loop `physical_block_idx = gl.load(block_tables_ptr_shifted + j_hbm_start + safe_j)` (line 847, inlined @1875) is a UNIFORM read-only load that at baseline lowers to `global_load_b32` (VGPR) + `v_readfirstlane` feeding the TDM `async_load` descriptor row-offset — in-loop rfl = 1, NOT 0. Adding `noalias_args=["block_tables_ptr","seq_lens_ptr","query_start_len_ptr"]` DOES recover the IR-level `!amdgpu.noclobber` on that in-loop load (annotate-uniform: 4→5 noclobber tags; the new tag is line-847@1875) — so the S4/S10b noalias→noclobber chain FIRES. But it is INERT on the final asm: ISel still selects `GLOBAL_LOAD_DWORD_SADDR` (scalar-base %512 + VGPR voffset = COPY of scalar `%2977=S_MIN_I32 safe_j`, +2048 imm) for the pipelined next-iter prefetch, so the load result stays VGPR and the `v_readfirstlane` round-trip persists. The blocked leg is the S5 scalar-SMEM-REPRESENTABILITY gate, NOT noclobber/aliasing: the software-pipelined prefetch presents a base+loop-varying-voffset address that ISel maps to the SADDR addressing mode, which has no scalar `s_load` form the global-load scalarizer will rewrite (scalarize on/off leaves this load `global_load_b32` identically). Instruction stream BASELINE vs AFTER = byte-identical (0 differing insn lines, both llc pins), in-loop rfl 1→1. pa_decode_sparse: ZERO raw global_load in the whole kernel; its async_gather index (`safe_slot_cur`) is a GENUINELY DIVERGENT per-lane LDS read (uniformity: DIVERGENT), so its 32 in-loop rfl are inherent gather/WMMA lane-shuffles, not a uniform-stranded-in-VGPR round-trip — unfixable by noalias (confirms S9b). The narrow protocol-correct set is FFM-correct (1 passed); the maximally-broad set including DATA pointers (query/kv_buffer/query_scales) MISCOMPILES (80.4% mismatch) — a soundness caution, since those are not uniform-read-only and one aliases a writable output.

> **CORRECTION (S10c supersedes the root-cause claim below).** This ledger's "blocked by the
> S5 scalar-SMEM-REPRESENTABILITY gate" is **WRONG**, and "noclobber 4→5 on the in-loop 847 load"
> was a mis-attribution. Pinned in `S10c.md` by tracing the LLVM backend: the mla-decode
> **software-pipelined block-table prefetch never receives `!amdgpu.noclobber`** (its two
> out-of-loop siblings do, and they DO become `s_load` — so the load is fully `s_load`-representable,
> refuting the S5 claim). Root cause: `AMDGPU::isReallyAClobber` (AMDGPUMemoryUtils.cpp) only
> AA-checks *atomics*; for the in-loop TDM `tensor_load_to_lds` (addrspace(3) LDS) def it hits the
> blanket `return true` **without consulting AA**, so `isClobberedInFunction` reports the addrspace(1)
> prefetch clobbered → no `noclobber` → `SIISelLowering.cpp:13153` skips the scalar path → VMEM+rfl.
> It is a **backend bug, `noalias`-independent** (addrspace(3)↛addrspace(1) is NoAlias without any
> contract). The no-op *verdict* and the divergence/soundness findings below still stand.

# RA-F2 — F2 attention/PA re-audit under the corrected all-uniform-read-only-args protocol

Re-audits ledgers/S9.md + S9b.md for F2 (mla, pa_decode_sparse, unified_attention_2d/3d) under
the S10b-corrected protocol (annotate ALL uniform read-only args, not just the single index).
Does NOT edit REGISTRY/CONCLUSION/other ledgers.

## Why the prior verdict needed re-auditing (the S9 miss)
S9 concluded "F2 block-table index already `s_load_b32` at baseline → noalias inert" but measured
ONLY `_mla_prefill_fwd_kernel_non_pipelined` (its own "the one compilable direct-index case"),
and EXTRAPOLATED unified/decode. The DECODE kernel `_mla_decode_fwd_kernel` (line 1500) is
software-pipelined and was never dumped. This cell dumps it and finds S9's extrapolation is wrong
about the asm shape (there IS an in-loop round-trip) even though the ship verdict (no-op) survives
for a DIFFERENT, deeper reason.

## Frozen experiment
- Triton `/root/triton` @ ba4fd67b (branch users/jerryyin/moe-gather-sload-contract, #120 landed).
  Contract is a Python `noalias_args=[...]` toggle — no rebuild.
- AITer `/root/aiter` @ 93d8ffb8, with the R3 async_gather fix present on pa_decode_sparse.py
  (pre-existing, required to compile; owned by R3, left as-is — this cell made ZERO net edits;
  mla.py restored, `git diff mla.py` empty).
- Kernels: `attention/mla.py::_mla_decode_fwd_kernel` (the pipelined decode kernel, the F2 decode
  analog); `attention/pa_decode_sparse.py::_pa_decode_sparse` (TDM async_gather);
  `attention/unified_attention_3d.py` (same `load_physical_block_idx_with_mod` construct as mla).
- Target/mode: gfx1250, FFM-lite for compile+correctness
  (`/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh`, `LD_LIBRARY_PATH=/opt/rocm/lib:...`);
  asm/ISel study compile-only on stock llc pins 56421f92 + 62b7cf96, `-mcpu=gfx1250`.
- Smallest exercising shapes:
  - mla decode: `test_mla_decode_fwd[True-q_dtype0-kv_dtype0-out_dtype0-64-False-True-32768-512-64-num_heads0-200-1-1]`
    (batch=1, decode_qlen=1, ctx=200, heads=(16,1), block=64, bf16, shuffled). Compiles the gluon
    `_mla_decode_fwd_kernel` (num_warps=2, num_stages=2, ALL_DECODE=1). ~4.5min/leg under FFM.
  - pa_decode_sparse: `test_pa_decode_sparse_vs_reference[True-False-False-100-512-1-1]`
    (skip_reduce, no sentinels, no var_len, kv_len=100, D=512, H=1, T=1, bf16). ~5s.
- A/not-A kill switch (mla decode):
  - BASELINE = as-shipped (no noalias). Cache `/tmp/tc-RA-F2-base` (hash SGPU3S...).
  - AFTER-broad = noalias on 9 read pointers incl DATA (block_tables, seq_lens, query_start_len,
    query_scales, q_scale, kv_scale, out_scale, kv_buffer, query). Cache `/tmp/tc-RA-F2-na` (J65V...).
  - AFTER-narrow = the PROTOCOL-CORRECT set: only the uniform-read-only metadata/index args
    (block_tables_ptr, seq_lens_ptr, query_start_len_ptr). Cache `/tmp/tc-RA-F2-na2` (OVBJXQ...).
- Metric: in-loop `v_readfirstlane` (awk loop-body counter), load histogram, whole-.text md5,
  comment/metadata-stripped instruction-stream diff; MIR trace via llc `-stop-after=amdgpu-isel`
  and `-stop-after=si-fix-sgpr-copies`; `opt -passes='print<uniformity>'` and
  `'amdgpu-annotate-uniform'`.

## Enumerated uniform read-only pointer args (step 1 of the protocol)

### `_mla_decode_fwd_kernel` (args mapped from the post-opt define + TTGIR loc names)
| arg | %N | read-only? | uniform? | how consumed | raw uniform load? |
|---|---|---|---|---|---|
| `block_tables_ptr` | %6 | YES (readonly inferred) | YES (uniformity: NOT divergent) | `gl.load(...+j_hbm_start+safe_j)` (l.847) → `physical_block_idx` → `get_kv_buffer_row_offsets` → TDM `async_load` offsets `[row_offsets,0]` | **YES — the round-trip source** |
| `seq_lens_ptr` | (%16 region) | YES | YES | `gl.load(seq_lens_ptr+seq_idx)` (l.1598) → seq_len → loop bounds | YES (already `S_LOAD_DWORD "amdgpu-noclobber"` at baseline) |
| `query_start_len_ptr` | (%16 region) | YES | YES | `gl.load(query_start_len_ptr+seq_idx)` (l.1591) → q_start_idx | YES (already `S_LOAD_DWORD "amdgpu-noclobber"` at baseline) |
| `q_scale_ptr`/`kv_scale_ptr`/`out_scale_ptr` | — | YES | YES (scalar) | scalar scale loads | None in bf16 path (None-specialized away) |
| `query_ptr` | %3 | YES | NO (per-lane data) | buffer/global data load (l.1678) | data, not index — EXCLUDE |
| `kv_buffer_ptr` | %5 | YES | NO (TDM data base) | TDM descriptor base | data, not index — EXCLUDE |
| `query_scales_ptr` | %4 | (bare/readnone) | — | conditional scales | EXCLUDE |
The protocol-correct AFTER set = {block_tables_ptr, seq_lens_ptr, query_start_len_ptr}.

### `_pa_decode_sparse` (from S9b + this cell's dump)
| arg | read-only? | uniform? | consumed | raw uniform load? |
|---|---|---|---|---|
| `kv_indptr_ptr` | YES | YES (t=pid) | `kv_start/kv_end = gl.load(kv_indptr+t/t+1)` → kv_len | YES → already `s_load_b64` at baseline |
| `kv_indices_ptr` | YES | — | TDM `tensor_load_to_lds` → LDS slot_bufs → `ds_load` → async_gather index | NO (TDM path; the read-back is per-lane DIVERGENT) |
| `kv_scales_ptr` | YES (QUANT only) | — | async_copy | NO raw index |
| `q_ptr`/`attn_sink_ptr` | YES | NO (data) | buffer_load | data |
There is NO uniform read-only raw load beyond kv_indptr, and it is already scalar. The
async_gather descriptor index is DIVERGENT (see HOW-fact PA).

## HOW-facts (real asm + MIR trace + uniformity)

### FACT M1 — mla decode BASELINE has an in-loop round-trip (S9 never saw this)
BASELINE `_mla_decode_fwd_kernel.amdgcn` (hash SGPU3S...): in-loop rfl = **1**, whole-fn rfl = 1,
and there is a `global_load_b32` at `mla.py:847:30 @[mla.py:1875:42]`:
```
.loc 1 847 30 ; mla.py:847:30 @[ mla.py:1875:42 ]   (physical_block_idx in-loop prefetch)
global_load_b32 v4, v4, s[10:11] scale_offset
...
v_readfirstlane_b32 s25, v4          ; VGPR->SGPR round-trip to feed the TDM descriptor
```
Load histogram: ds_load_b128 x324, global_load_b128 x18 (KV data), s_load_b32 x5, tensor_load_to_lds
x4, s_load_b128 x2, s_load_b512 x1, **global_load_b32 x1** (the block_tables index). FFM: 1 passed.

### FACT M2 — the load is UNIFORM, not divergent (re-audit of any "divergent" claim)
`opt -passes='print<uniformity>'` on the baseline llir: the in-loop block_tables load `%1146 =
load <1 x i32>, ptr addrspace(1) %1145` and its address `%1145`/`%1144` are NOT prefixed
`DIVERGENT:` (4239 DIVERGENT lines exist for comparison; workitem-id chains are marked). So the
value is uniform-but-forced-to-VGPR — the S10b fixable class, not truly divergent.

### FACT M3 — noalias DOES recover `!amdgpu.noclobber` (the S4/S10b chain fires at IR level)
`opt -passes='amdgpu-annotate-uniform'`:
- BASELINE: 4 noclobber tags — lines 1598 (seq_lens), 1591 (query_start_len), 852 (prologue
  load_physical_block_idx), 847@1866 (prologue-mod load). The in-loop 847@1875 load has NONE.
- AFTER (narrow noalias): **5** tags — the extra one is line 847 (`!99`, the in-loop @1875 load).
So `noalias → readonly → !amdgpu.noclobber` succeeds on exactly the in-loop load, as S10b predicts.
The IR-level mechanism is NOT the blocker.

### FACT M4 — but ISel still selects SADDR-vector; the round-trip persists (the real gate)
`llc -stop-after=amdgpu-isel` on the AFTER-narrow llir, the two block_tables loads:
```
prologue @1866: %2541:sreg_32_xm0_xexec = S_LOAD_DWORD_IMM killed %2540, 0, 0
                :: ("amdgpu-noclobber" load (s32) from %ir.681, !alias.scope, !noalias)   ; SCALAR
in-loop  @1875: %2978:vgpr_32 = GLOBAL_LOAD_DWORD_SADDR %512, killed %2979, 0, 2048
                :: (load (s32) from %ir.1640, !alias.scope, !noalias)                     ; VECTOR
```
The MMO on the in-loop load carries `!alias.scope/!noalias` (from the annotation) but the
`"amdgpu-noclobber"` string is GONE by ISel, and the addressing is base+voffset:
```
%2977:sreg_32 = S_MIN_I32 %1169, %513        ; safe_j = min(j,max-1)  -- SCALAR (SGPR!)
%2979:vgpr_32 = COPY %2977                    ; SGPR -> VGPR (VALU-ization of a uniform value)
%2978:vgpr_32 = GLOBAL_LOAD_DWORD_SADDR %512, killed %2979, 0, 2048
```
`si-fix-sgpr-copies` then inserts `%1301:sreg_32 = V_READFIRSTLANE_B32 %2978` to get the loaded
index back into SGPR for the descriptor. WHY prologue-scalar but in-loop-vector for the SAME source
line: the prologue address is a full 64-bit SGPR (`S_ADD_U64_PSEUDO %512, S_LSHL(...)`, imm off 0)
→ `S_LOAD_DWORD_IMM`; the in-loop version is the software-pipelined NEXT-iteration prefetch, whose
loop-varying scalar index `safe_j` ISel maps to the `GLOBAL_LOAD_..._SADDR` addressing mode
(scalar base + VGPR voffset + `+2048` imm). That SADDR form has no scalar `s_load` equivalent the
global-load scalarizer will rewrite.

### FACT M5 — the S5 representability leg, not noclobber, is what fails (scalarize toggle)
`llc -amdgpu-scalarize-global-loads={true(default),false}` on the AFTER-narrow llir: the in-loop
847@1875 load is `global_load_b32` in BOTH (the toggle changed the whole-kernel global_load_b32
count 1↔5 for OTHER loads but left THIS one identical). ⇒ this load is not even a scalarizer
candidate; the SADDR base+voffset addressing bypasses the noclobber→s_load path. This is S5's
"scalarizable-to-SMEM shape" condition (#3) failing via addressing-mode, distinct from decode where
the metadata loads had a directly-scalar address that DID scalarize.

### FACT M6 — A/not-A asm result: byte-identical instruction stream (no-op), both pins
Comment/metadata-stripped instruction diff BASELINE vs AFTER-narrow = **0 differing lines**;
in-loop rfl **1 → 1**; whole-fn rfl 1 → 1. Confirmed on BOTH llc pins (56421f92: rfl 1/1;
62b7cf96: rfl 1/1). The AFTER-broad variant (which also includes block_tables) likewise had a
0-line instruction diff for this specialization — noalias changed only kernarg metadata
(`.actual_access: read_only`), never an instruction. (whole-.text md5 differs only by that metadata.)

### FACT PA — pa_decode_sparse: no raw uniform load, gather index is truly DIVERGENT
BASELINE `_pa_decode_sparse.amdgcn`: in-loop rfl = 32, whole-fn rfl = 64, and **ZERO global_load**
in the entire kernel (histogram: ds_load_b128 x208, buffer_load_b128 x16, tensor_load_to_lds x11,
s_load_b64 x2, s_load_b32 x1, s_load_b256 x1, s_load_b128 x1). The kv_indptr load is `s_load_b64`
(S9b). The 32 in-loop rfl are at `pa_decode_sparse.py:304` = `async_gather(kv_desc, safe_slot_cur,
...)`. `opt print<uniformity>` on the llir: the LDS slot reads feeding safe_slot_cur are
**DIVERGENT** (`load <8 x i16>, ptr addrspace(3) ...` marked DIVERGENT; 547 divergent addrspace(3)
loads total). So the gather index is genuinely per-lane divergent (each lane gathers a different KV
row — the point of a sparse gather); the readfirstlane there is inherent to the gather/WMMA, NOT a
uniform-value-stranded-in-VGPR. noalias cannot help (fails condition #2 truly-divergent). Confirms
and mechanistically grounds S9b.

### FACT U — unified_attention_3d shares the identical mla-decode construct
`unified_attention_3d.py:1044-1055` `load_physical_block_idx_with_mod` is the SAME
`physical_block_idx = gl.load(block_tables_ptr_shifted + j_hbm_start + safe_j)` clamp feeding a TDM
`async_load` descriptor as mla decode. Same software-pipelined prefetch → same SADDR addressing-mode
gate → same inert-noalias outcome. Grounded extrapolation from the mla decode ISel evidence (not
independently dumped: the smallest unified test uses 32768 blocks, impractical in this cell's
budget; the source construct is byte-identical so the ISel choice is the same). unified_2d uses the
`self.block_tables_ptr_shifted + i` helper (l.460), same load class.

## FFM correctness
- AFTER-narrow {block_tables_ptr, seq_lens_ptr, query_start_len_ptr}: `1 passed` (byte-identical
  .text ⇒ identical numerics; FFM corroborates). SOUND no-op.
- AFTER-broad {+query_ptr, kv_buffer_ptr, query_scales_ptr, scale ptrs}: **1 FAILED**, 80.4%
  elements mismatch. Since THIS specialization's .text is byte-identical to baseline, the
  miscompile is a soundness violation from one of the DATA pointers not honoring the noalias promise
  (it aliases a writable arg in some path/variant — e.g. the reduce kernel or another
  specialization). This is a CAUTION, not a codegen result: it confirms the protocol's restriction
  to *uniform read-only* args matters — do NOT annotate data pointers.
- pa_decode_sparse BASELINE: `1 passed`.

## A/not-A result (the point of the cell)
| kernel | set added | in-loop rfl BASE→AFTER | instr-stream diff | verdict |
|---|---|---|---|---|
| mla decode | block_tables_ptr, seq_lens_ptr, query_start_len_ptr | 1 → 1 | 0 lines (both pins) | NO-OP (noclobber recovered but SADDR gate blocks) |
| pa_decode_sparse | (S9b: kv_indptr, kv_indices) + audited metadata | 32 → 32 | (S9b: metadata-only) | NO-OP (gather index truly divergent; 0 raw global_load) |
| unified_2d/3d | block_tables_ptr (+metadata) | 1 → 1 (grounded extrapolation) | 0 lines (same construct) | NO-OP (same SADDR gate as mla decode) |

## Counter-experiment (what would refute the no-op, and its result)
- "Add the uniform read-only set and observe the in-loop 847 load scalarize to s_load / rfl drop
  to 0" → AFTER-narrow: in-loop rfl 1→1, load still `global_load_b32`, 0-line instr diff, both pins.
  Not refuted.
- "Show noalias failed to reach the noclobber leg (so the miss is aliasing, fixable)" → REFUTED as
  a cause: annotate-uniform DOES add noclobber to the in-loop load (4→5, FACT M3). The miss is
  downstream (SADDR addressing-mode representability, FACT M4/M5), not aliasing.
- "Show the in-loop load is truly divergent (so noalias could never help)" → REFUTED for mla:
  uniformity says it is uniform (FACT M2). It is uniform-but-VGPR — yet still not scalarizable
  because of the pipelined SADDR form. (For pa_decode_sparse the gather index IS truly divergent —
  FACT PA — a different no-op cause.)
- "Show a raw uniform read-only load hiding in pa_decode_sparse" → 0 global_load in the kernel;
  only kv_indptr (already s_load_b64) and TDM. Not refuted.

## CLAIM STATUS
The corrected all-uniform-read-only-args protocol does NOT flip F2 to a win — every F2 kernel stays
a byte-identical no-op — BUT it corrects the S9 REASONING: mla/unified decode is NOT "already
s_load"; it has a real uniform-metadata-load VGPR round-trip that noalias's noclobber chain reaches
yet cannot cash because the software-pipelined prefetch's SADDR addressing mode has no scalar SMEM
form (S5 condition #3, representability, fails — not condition #4 aliasing). pa_decode_sparse fails
condition #2 (the gather index is genuinely divergent) and #1 (no raw global load). This is the
specific, evidence-backed condition that fails, distinct from decode's fixable case.

## Remaining unknowns / hand-offs (not blockers)
- unified_2d/3d asm not independently dumped (32768-block test heavy); verdict is a construct-level
  extrapolation grounded in the mla-decode ISel trace (identical `safe_j`+TDM source). A direct dump
  would confirm the byte count; the mechanism (SADDR gate) is arch-general.
- The mla-decode round-trip could in principle be closed by an ISel/addressing change that keeps a
  loop-varying uniform index in SGPR and emits `s_load` with an SGPR offset (or by hoisting the
  readfirstlane, cf. the MachineLICM backstop db4972674) — a BACKEND lever, not a noalias-contract
  lever. Out of scope for this cell (noalias is proven inert on the asm).
- No runtime numbers (asm-identity ⇒ zero runtime delta by construction, per the no-op rule).

## Harness friction
- `run_on_model.sh` broken (hardcodes /am-ffm); sourced env inline:
  `source /zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh; export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH`.
- `aiter` not pip-installed → `PYTHONPATH=/root/aiter`; pytest needs `-p no:pylama`.
- The mla decode FFM leg takes ~4.5 min and exceeds the 2-min bash cap → ran each leg with
  `run_in_background`. pa_decode_sparse is ~5s.
- pa_decode_sparse required the R3 async_gather fix (already present in the tree; owned by R3, left
  as-is). This cell made ZERO net source edits (mla.py restored, `git diff mla.py` empty).
