STATUS: done
CONCLUSION: On the now-compiling (R3) pa_decode_sparse, noalias is INERT: the kernel has exactly ONE raw-pointer index load — the coalesced 2-element prologue read `kv_start/kv_end = gl.load(kv_indptr_ptr + t)` (i32x2, uniform, `readonly`-inferred arg %4) — and it ALREADY selects a scalar `s_load_b64 s[56:57], s[4:5], 0x0` at baseline via the readonly->invariant leg of the S5 gate (noclobber count = 0 in BOTH variants); the other index `kv_indices_ptr` (%3) rides the TDM path (getelementptr -> `tensor.load.to.lds` -> LDS -> `ds_load`/async_gather), never a raw global load, so noalias cannot scalarize it. Annotating BOTH pointers changes exactly ONE amdgcn hunk (metadata-only `.actual_access: read_only` on the kernarg descriptor), ZERO machine instructions, rfl 64->64, both FFM-correct. Confirms S9/S16 prediction. **NO-SHIP.**

# S9b — F2 pa_decode_sparse (re-measure now that R3 made it compile)

**Cell:** S9b (Run 2, Phase B). **Blocked-by:** R3 (done), S4 (done), S5 (done).
**Question:** pa_decode_sparse routes kv_indices via TDM async_load->LDS->async_gather (Run-1 S9
said not a scalar-load candidate). Now that it compiles, confirm on the real kernel: is there ANY
raw-pointer uniform index load that noalias could scalarize, or is it all TDM (=> no-op like the
rest of F2)? Toggle noalias, dump asm, FFM-correct. Grounded verdict.

---

## Frozen experiment

- **Triton tree/SHA:** `/root/triton` @ `ba4fd67b8ed28d37935f95ddbe8717853de7212d`
  (branch `users/jerryyin/moe-gather-sload-contract`; #120 contract landed). NOT rebuilt — the
  contract is a Python annotation toggle (`noalias_args=[...]`), no C++ change.
- **AITer tree/SHA:** `/root/aiter` @ `93d8ffb8e2a101f7451653b8e3b9e12b61334e46`, with the R3-owned
  async_gather fix present on `.../gfx1250/attention/pa_decode_sparse.py` (drop the removed
  `src_col_offset` positional, triton PR #61). No other edits; tree restored to R3 state after
  the cell (noalias annotation removed).
- **Kernel under test:** `_pa_decode_sparse` (gluon, gfx1250). Dispatched from
  `aiter/ops/triton/attention/pa_decode_sparse.py::pa_decode_sparse` when `arch == gfx1250`
  (`use_gluon = True`). Compiled variant this shape: `KV_SPLITS=4`, BLOCK_H=16, BLOCK_D=512,
  BLOCK_K=32, H=1, D=512.
- **Target/mode:** gfx1250 via FFM-lite
  (`/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh`, mi450 topology). FFM = correctness
  only (no timing), per PLAN. This is a compile + asm-delta + FFM-correctness cell; a proven
  metadata-only asm delta means no runtime is owed (PLAN Run-2 note: no-op families get NO runtime,
  asm-identity is the proof — here asm is instruction-identical).
- **Smallest exercising shape:** `test_pa_decode_sparse_vs_reference[True-False-False-100-512-1-1]`
  = skip_reduce=True, sentinels=False, var_len=False, kv_len=100, D=512, H=1, T=1 (bf16). Exercises
  the tile loop + prologue + epilogue (kv_len 100 / BLOCK_K 32 -> 4 tiles) and the kv_indptr load.
- **Two variants (A/not-A kill switch):** BASELINE = as-shipped (R3 fix, no noalias).
  NOALIAS = add `noalias_args=["kv_indptr_ptr", "kv_indices_ptr"]` to the `@gluon.jit` decorator
  only (both the raw-load index-pointer AND the TDM index-pointer, maximally favorable), nothing
  else changed.
- **Isolation:** BASELINE `TRITON_CACHE_DIR=/tmp/tc-S9b TRITON_DUMP_DIR=/tmp/td-S9b`; NOALIAS
  `TRITON_CACHE_DIR=/tmp/tc-S9b-na TRITON_DUMP_DIR=/tmp/td-S9b-na`; both `TRITON_ALWAYS_COMPILE=1
  TRITON_KERNEL_DUMP=1`. Never shared `~/.triton/cache`. No GPU-lock/gfx950/AM used (compile +
  FFM-correctness only; the asm claim needs no execution).
- **Artifacts:**
  - BASELINE `/tmp/td-S9b/CF2GUNM6ECBKME45FZBXAHKBDOQNJIBY4UVYGLA2SJRCC7YWU7YQ/` (amdgcn md5
    `dae3c78cc32594a81dbb0538172f3ddc`).
  - NOALIAS `/tmp/td-S9b-na/YNRLPULFQPDA2OUQGGU4QOS4H6TUS27XSWXBMNGUHITLWGE6OPXA/` (amdgcn md5
    `14d9ec3532d0cfe6755d986c19c60c1a` — differs ONLY by the one metadata hunk below).
    `{.amdgcn,.llir,.ttgir}` each.
- **Metric:** raw-pointer index-load ISel instruction, full load-kind histogram, in-loop/total
  `v_readfirstlane`, whole-amdgcn diff, FFM PASS.
- **Sanity:** BOTH variants `1 passed` under FFM (`torch.testing.assert_close`, atol/rtol 5e-3).

---

## The load inventory (every raw-pointer load in pa_decode_sparse, traced)

| load site (source) | pointer | kind | consumed by | raw scalar gl.load? | noalias-relevant? |
|--------------------|---------|------|-------------|---------------------|-------------------|
| `kv_start = gl.load(kv_indptr_ptr + t)` / `kv_end = gl.load(+t+1)` (L190/191) | `kv_indptr_ptr` (arg %4) | **raw global** i32x2, uniform (`t=program_id(0)`) | scalar prologue math (kv_len) | **YES** — the ONLY one | already `s_load_b64` at baseline (readonly-invariant) |
| `slot_desc` gather of `kv_indices` (L269, L283-290, L322-334) | `kv_indices_ptr` (arg %3) | **TDM** `tensor.load.to.lds` -> LDS `slot_bufs` -> `ds_load` | async_gather of KV | **NO** (TDM path) | inert (no raw load to scalarize) |
| q load (L175), attn_sink (L207), out/partials store (L481+) | q/out/etc | `buffer_load`/`buffer_store` (addrspace 8) | data | NO | n/a (not index, not global_load) |
| KV cache gather (L304/344), slots (L290/331), kv math (L361/432) | via kv_desc/LDS | TDM `async_gather` + `ds_load` | WMMA | NO | n/a (TDM/LDS) |

So there is **exactly one** raw-pointer index load (`kv_indptr_ptr`), and it is the trivially-scalar
2-element prologue read S9 Fact 4 already named. `kv_indices` is 100% TDM, as Run-1 S9 said.

---

## HOW-facts — each on real IR/asm

### FACT 1 — EMISSION: the toggle lands `noalias` on BOTH requested args (kill switch)
NOALIAS TTGIR carries `tt.noalias = 1` on `%kv_indices_ptr` and `%kv_indptr_ptr` (2 args). NOALIAS
post-opt `.llir` define, args %3/%4:
```
BASELINE: ptr addrspace(1)         %3           (kv_indices_ptr)
          ptr addrspace(1) nofree readonly captures(none) %4   (kv_indptr_ptr)
NOALIAS : ptr addrspace(1) noalias %3
          ptr addrspace(1) noalias nofree readonly captures(none) %4
```
Two `addrspace(1) noalias` in the NOALIAS module, zero in baseline. Note the ASYMMETRY:
- `%4` (kv_indptr_ptr, the raw load) is `readonly` in BOTH variants (FunctionAttrs infers it; the
  kernel never writes through it) — noalias is added on top.
- `%3` (kv_indices_ptr, the TDM index) is BARE in baseline (no readonly/readnone/captures) because
  it feeds a `getelementptr`->TDM-descriptor->`tensor.load.to.lds` path whose access mode LLVM
  cannot classify; noalias lands but nothing downstream consumes it. This is the S11/F3 TDM-provenance
  pattern (the arg is not a classifiable memory access), not a scalar-load site.

### FACT 2 — the one raw index load ALREADY selects scalar `s_load_b64` at BASELINE
BASELINE amdgcn, `pa_decode_sparse.py:190:16` (line 333):
```
	s_load_b64 s[56:57], s[4:5], 0x0        ; kv_start(i32) + kv_end(i32) coalesced, one dwordx2
```
Both `kv_start` and `kv_end` (adjacent i32 in `kv_indptr`) coalesce to a single scalar SMEM
`s_load_b64`, address in SGPRs (`s[4:5] = s[10:11] + (t<<2)`), no `global_load`/`flat_load` of the
index. This is the S5 gate satisfied via the readonly->invariant leg (i32 dword-granular, uniform,
align, `readonly` makes the MMO invariant) WITHOUT noalias — exactly S9 Fact 2/3b (mla) and S5
Fact E corollary. **`amdgpu.noclobber` count = 0 in the module** => scalar selection is NOT via the
noalias/noclobber leg.

### FACT 3 — adding noalias produces ZERO instruction change (inert)
Whole-amdgcn diff BASELINE vs NOALIAS = **exactly one hunk, metadata-only**:
```
4095c4095,4096
<       - .address_space:  global
---
>       - .actual_access:  read_only
>         .address_space:  global
```
This is the kernel-argument descriptor metadata for the annotated arg gaining `.actual_access:
read_only` — NOT a single machine instruction. Concretely identical across variants:
- The `s_load_b64 s[56:57], s[4:5], 0x0` at line 333 is byte-identical in both.
- Load-kind histogram identical: `buffer_load_b128 x16, s_load_b256 x1, s_load_b128 x1,
  s_load_b64 x2, s_load_b32 x1` (+ TDM `tensor.load.to.lds x12`, `ds.load.tr16 x65`,
  `ds_load_b128`, `raw.ptr.buffer.load x17` — all identical).
- **Total `v_readfirstlane` = 64 -> 64** (these are WMMA-layout data lane-shuffles, NOT index-load
  churn; noalias correctly leaves them untouched).
- **Zero `global_load`/`flat_load` in the entire kernel** — the index never becomes a raw global
  load in either variant.

### FACT 3b — WHY noalias is inert here (mechanism, S4/S5 applied)
- The only raw-pointer index load (`kv_indptr`) is (a) i32 **dword-granular** (not sub-dword like
  F1's i16), (b) `readonly`-inferred, (c) uniform, (d) has **no in-loop aliasing store** that AA
  cannot disprove. So the "invariant OR noclobber" leg of the S5 gate (SIISelLowering 13148-13158)
  is satisfied by `readonly`-invariance alone -> `s_load_b64` at baseline; noalias->noclobber is a
  redundant second route to a leg already satisfied (noclobber count 0). This is the F2/mla mechanism
  (S9): fails mechanism-condition #4 (no interfering in-loop store).
- `kv_indices` fails mechanism-condition #1 (raw-pointer load): it is consumed by
  `getelementptr`->TDM `tensor.load.to.lds` (12x), never a `tt.load`/global_load. noalias on a
  TDM-fed pointer has no load-selection site to act on (S13/F5 and S11/F3 TDM pattern). The slots
  are read back from LDS via `ds_load`/`ds.load.tr16` — LDS (addrspace 3), which is already NoAlias
  to global by construction (S8), so no GVN/LICM win either.

### FACT 4 — FFM-correctness (both variants)
`test_pa_decode_sparse_vs_reference[True-False-False-100-512-1-1]` -> `1 passed` for BOTH baseline
and noalias (identical numeric result at reduced size). Strongest correctness statement: the two
binaries are instruction-identical (only kernarg metadata differs), so identical numerics is by
construction; FFM PASS corroborates.

---

## S3 soundness applied (shippability)

Per S3, `noalias_args=[...]` is a two-part caller promise: (readonly) never write through the arg,
and (noalias) the arg does not overlap any writable arg.
- `kv_indptr_ptr`, `kv_indices_ptr` are read-only index arrays; the writable args are the outputs
  (`out_ptr`/`m_partial`/`l_partial`/`acc_partial`, %5-%7,%9), distinct tensors in every caller.
  So the annotation WOULD be sound. BUT Fact 3 shows it yields **zero instruction change** — no
  codegen benefit — while adding S3 silent-miscompile liability (a future caller that aliases an
  output into the index arrays would be miscompiled with no diagnostic). **Ship recommendation:
  DO NOT annotate pa_decode_sparse index pointers.**

---

## Hypotheses + refutation trail

- **H1 "now that it compiles, pa_decode_sparse has a MoE-style raw index gather noalias will
  scalarize."** REFUTED — the only raw-pointer index load is `kv_indptr` (2-elem prologue), already
  `s_load_b64` at baseline (Fact 2); `kv_indices` is 100% TDM (Fact 1/3b, `tensor.load.to.lds x12`,
  0 global_load). noalias -> 0 instructions (Fact 3).
- **H2 "readonly is not enough; noalias is needed for the scalar s_load."** REFUTED — `readonly`
  present on %4 in both variants; baseline is already `s_load_b64`; noclobber count 0 (Fact 2).
  Matches S4 Hop1 / S5 Fact E / S9.
- **H3 "kv_indices noalias will unblock a load in the loop body."** REFUTED — the loop-body slot
  read is `ds_load` from LDS (addrspace 3, already NoAlias), fed by TDM `tensor.load.to.lds`; there
  is no global/raw load of the index to unblock, and %3 is bare-noalias with no downstream consumer
  (Fact 1/3b).
- **H4 "the 64 v_readfirstlane are un-scalarized index churn noalias would kill."** REFUTED —
  rfl 64->64 unchanged; they are WMMA-layout lane shuffles, not index loads (index is already
  scalar/LDS). noalias leaves them untouched, correctly.
- **H5 "it's an LLVM-version effect."** N/A — same triton, same backend, one attr delta; delta is
  metadata-only.

---

## CLAIM STATUS

**CLAIM (answered, irrefutable at IR/asm level):** On the compiling (R3) pa_decode_sparse, noalias
is INERT. The kernel has exactly one raw-pointer index load (`kv_indptr_ptr`, i32x2 prologue read),
already selected as scalar `s_load_b64` at baseline via the readonly->invariant leg (noclobber
count 0); `kv_indices_ptr` rides the TDM `tensor.load.to.lds`->LDS->async_gather path and is never a
raw load. Annotating BOTH pointers changes exactly one metadata hunk (`.actual_access: read_only`),
ZERO instructions, rfl 64->64, load histogram identical, both FFM `1 passed`. This confirms the
S9/S16 prediction on the actual now-compiling kernel (S9 could only inspect it statically because it
did not compile). **NO-SHIP** (no codegen gain; only added S3 liability).

**Counter-experiments that would refute (and their results):**
1. "Add noalias and observe a load scalarize / churn drop absent at baseline." -> 1 metadata line,
   0 instructions; kv_indptr already `s_load_b64`, rfl 64->64 (Fact 2/3). Not refuted.
2. "Show the kv_indptr load is a global/flat load at baseline (not yet scalar)." -> it is
   `s_load_b64 s[56:57], s[4:5], 0x0` at `pa_decode_sparse.py:190:16` (Fact 2); zero global/flat
   loads in the whole kernel. Not refuted.
3. "Show kv_indices becomes a raw scalar gl.load noalias could scalarize." -> it feeds
   `getelementptr`->`tensor.load.to.lds` (TDM, 12x), read back via `ds_load`; %3 is bare-noalias,
   no load-selection site (Fact 1/3b). Not refuted.
4. "Show noalias breaks FFM correctness." -> both variants `1 passed` (Fact 4). Not refuted.
5. "Show readonly is not what carries invariance (needs noalias)." -> readonly on %4 in both,
   noclobber 0, baseline scalar (Fact 1/2). Not refuted.

---

## Remaining unknowns (hand-offs, not blockers)

- Only the smallest bf16 T=1 / KV_SPLITS=4 shape was dumped. Larger shapes / `sentinels=True` /
  `QUANT_KV` change the loop structure and the scale-load path but add NO new raw-pointer index
  load (the index handling — kv_indptr raw scalar read + kv_indices TDM gather — is shape/dtype
  independent). The scales path (`kv_scales_ptr`, %2, `readnone` at baseline in this non-QUANT
  build) uses `async_copy.global_load_to_shared` under QUANT_KV, not a raw index load; not a noalias
  scalar-load candidate either. High-confidence the verdict holds across shapes; not exhaustively
  dumped.
- No runtime numbers produced — correctly: the asm is instruction-identical (metadata-only delta),
  so runtime is identical by construction (PLAN Run-2 no-op rule). No AM/gfx950 cycles owed for S9b.
  Flag for S20: S9b is a confirmed no-op, no RT cell needed.

## Harness friction (report)
- FFM env is at `/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh` (the `/am-ffm` path in
  `.zshrc`/`run_on_model.sh` does not exist on this host); sourced inline (same as S9/R3).
- `aiter` not pip-installed -> `PYTHONPATH=/root/aiter`; pytest needs `-p no:pylama`.
- `ls`/`find` colorized the cache subdir path (ANSI codes leaked into a var) — set `LS_COLORS=`
  and hard-code the hash dir to get clean `diff`/`md5sum`. Minor.

## Next unblocked cell
S12b (F4 fused_kv_cache gluon primary re-measure; blocked-by R3 done). Also S17/S18 remain
independently unblocked.
