STATUS: done
CONCLUSION: VERDICT CONFIRMED (still a no-op). The corrected all-uniform-read-only-args protocol does NOT change the F4 verdict. F4 is NOT the decode case: the decode false-negative arose because upstream uniform metadata loads were VGPR-resident (`global_load`) and SIFixSGPRCopies VALU-ized the descriptor into an in-loop VGPR->SGPR readfirstlane round-trip. In F4 there is exactly ONE loop-bearing kernel (`gather_kv_b_proj._triton_gather_kv_b_proj_impl`); its upstream uniform read-only metadata loads (`kv_indptr`, `kv_prefix_sum_context_lens`) are ALREADY selected as `s_load_b64` at BASELINE (uniformity analysis says UNIFORM; `amdgpu-annotate-uniform` already tags them `!amdgpu.noclobber` WITHOUT noalias because they are entry-block loads with no store that MemorySSA can reach — S4 condition #4 fails). Consequently baseline in-loop v_readfirstlane = 0 and SIFixSGPRCopies inserts ZERO V_READFIRSTLANE (0 before-pass, 0 after-pass) — there is NO descriptor round-trip to fix. Adding the full uniform arg set {kv_indptr, kv_prefix_sum_context_lens, k_scale, kv_indices} leaves the load histogram byte-identical and in-loop rfl 0->0 (only scheduler/ALU-form noise: s_wait_loadcnt 42->41, v_cndmask form flips, v_nop 30->34), FFM PASS. RE-AUDIT of the prior "divergent vector" claim: the read gather (`kv_indices` -> `raw.ptr.buffer.load.v4i32`, 111 occurrences) is TRULY DIVERGENT per uniformity analysis (every buffer.load tagged DIVERGENT), NOT uniform-forced-VGPR — S5 ceiling holds, noclobber count 4->4 and buffer.load count 111->111 unchanged by noalias. The two no-loop kernels (`kv_cache._cat_and_cache_mla_kernel`, gluon `_fused_qk_rope_cat_and_cache_mla_kernel`) cannot host an in-loop round-trip structurally; the gluon primary with the FULL uniform set {slot_mapping, pos, k_scale, cos, sin} is BYTE-IDENTICAL `.text` (md5 beb49740, code diff 0 lines), FFM PASS. The specific condition that fails: ALREADY s_load AND no round-trip (gather metadata) + genuinely divergent vector (the read gather) + no loop (the two scatter kernels).

# RA-F4 — F4 KV-cache re-audit under the corrected "all uniform read-only args" protocol

Supersedes the METHOD (not the verdict) of ledgers/S12.md and S12b.md. Re-tests
`gather_kv_b_proj`, `kv_cache._cat_and_cache_mla_kernel`, and gluon `fused_kv_cache`
under the S10b-corrected protocol (annotate EVERY uniform read-only arg, incl. the
upstream metadata pointers the original cells skipped, not just the single index).

## Why this re-audit exists
S12/S12b annotated only the single index/scatter pointer (`kv_indices` / `slot_mapping`)
and never traced the upstream uniform metadata loads through SIFixSGPRCopies — the exact
blind spot that made the decode verdict a false negative (S10b). This cell enumerates ALL
uniform read-only args, adds them, and traces the mechanism to prove whether F4 hides a
decode-style round-trip. It does not.

## Enumerated uniform read-only args (step 1 of the protocol)

### gather_kv_b_proj `_triton_gather_kv_b_proj_impl` (the ONLY loop-bearing F4 kernel; `for chunk_id`, line 611)
Pointer args (Python order after scalar `batch_size`): k_buffer(%0), k_scale(%1),
kv_indptr(%2), kv_indices(%3), kv_prefix_sum_context_lens(%4), kv_proj_weight(%5),
kv_proj_scale(%6), k_prefix(%7 writeonly), v_prefix(%8 writeonly).
- **UNIFORM + read-only (the metadata S12 MISSED):**
  - `kv_indptr` — `kv_block_start=load(kv_indptr+pid_batch)`, `kv_block_end=load(+1)` (lines 421/422); `pid_batch` uniform. `kv_block_start` is used IN-LOOP at line 618 (`kv_indices + kv_block_start + ...`); `total_kv_block` (derived) is the loop trip count. **This is the decode-analog metadata load.**
  - `kv_prefix_sum_context_lens` — `context_start/context_end` (lines 424/425), uniform; `context_start` used in-loop for the store masks (line 730).
  - `k_scale` — `k_scalar_scale=load(k_scale)` (line 436), single-element uniform scalar (only when k_buffer is not bf16). At the smallest bf16 shape this arg is `readnone` (unread).
- **read-only but DIVERGENT-addressed (cannot be s_load — S5 ceiling):**
  - `kv_indices` — gather index `load(kv_indices + ... + arange(ChunkK)//KBlockSize)` (line 616), per-lane vector.
  - `k_buffer` — gathered read via `kv_block_idx[:,None]*stride` (divergent).
  - `kv_proj_weight`, `kv_proj_scale` — addressed by `offs_n_k/offs_k` (arange), divergent; loaded once outside the loop (hoisted).
- **written (not read-only):** k_prefix, v_prefix.
- **AFTER set tested:** `noalias_args=["kv_indptr","kv_prefix_sum_context_lens","k_scale","kv_indices"]`.

### kv_cache `_cat_and_cache_mla_kernel` (NO loop — single scatter store)
- UNIFORM + read-only: `slot_mapping_ptr` (line 178, `pid_b` uniform; the S12 index),
  `k_scale_ptr` (line 187, single-element, the arg S12 MISSED).
- read-only DIVERGENT: `k_nope_ptr`, `k_pe_ptr` (addressed by `d_nope_offs=arange`).
- written: `kv_cache_ptr`.
- No `for`/`while` -> no in-loop rfl is structurally possible; not separately recompiled
  (over-determined by the gather + gluon results below; S12 Fact 3/4 already showed
  slot_mapping already-`s_load_b32`, 3-line-metadata-only noalias diff).

### gluon `_fused_qk_rope_cat_and_cache_mla_kernel` (NO loop — per-pid single pass, `if pid < B*QH:`)
- UNIFORM + read-only: `slot_mapping_ptr`(483), `pos_ptr`(482), `k_scale_ptr`(493) — all
  `pid_b`-derived uniform scalar entry loads. `cos_ptr`/`sin_ptr`(504/507) are TDM-loaded
  (streamed through LDS), not raw uniform-scalar loads feeding a descriptor round-trip.
- **AFTER set tested:** `noalias_args=["slot_mapping_ptr","pos_ptr","k_scale_ptr","cos_ptr","sin_ptr"]`
  (superset of S12b's slot_mapping-only test).

## Frozen experiment
- aiter `/root/aiter` @ 93d8ffb8e (R3 compile fixes present in fused_kv_cache.py both files +
  pa_decode_sparse.py; a concurrent cell's mla.py noalias edit also present — not mine, left
  untouched). My noalias toggles added then fully reverted; owned files (gather_kv_b_proj.py,
  kv_cache.py) end git-clean; gluon fused_kv_cache.py retains only the R3 async_gather fix.
- triton `/root/triton` @ ba4fd67 (#120 contract; Python annotation toggle, no rebuild).
- Vehicle: gfx1250 via FFM-lite `/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh`
  (canonical /am-ffm env empty; run_on_model.sh broken here). asm/IR = FFM-compiled gfx1250.
- Isolation: per-cell TRITON_CACHE_DIR=/tmp/tc-RA-F4-{base,after,gluon-base,gluon-after},
  TRITON_ALWAYS_COMPILE=1, TRITON_KERNEL_DUMP=1. pytest: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  PYTEST_PLUGINS=ffm_teardown PYTHONPATH=/root/scripts/tools.
- Shapes (smallest exercising): gather_kv_b_proj `test_gather_kv_b_proj[1-1-4-k_buffer_type4-512]`
  (batch=1, block=1, bf16 k_buffer, weight_preshuffle=True, per-block scale -> the loop impl).
  gluon MLA-cat `test_fused_qk_rope_cat_and_cache_mla[False-cache_dtype0-True-64-False-1-16384-512-64-1-16-1]`.
- Stock llc/opt pin: `/root/.triton/llvm/llvm-56421f92-ubuntu-x64-1/bin/{opt,llc}`, `-mcpu=gfx1250`.
- Metric: in-loop `v_readfirstlane` (awk loop-body counter); load histogram; `.text` md5 +
  diff; SIFixSGPRCopies V_READFIRSTLANE count (before/after pass); uniformity print;
  amdgpu-annotate-uniform noclobber count; FFM PASS.
- Kill switch: the ONLY delta is the noalias_args set.

## HOW-facts (real IR/asm + traces)

### FACT 0 — the loop-bearing gather compiles + FFM PASS in BASE and AFTER
Both `test_gather_kv_b_proj[1-1-4-k_buffer_type4-512]` runs: `1 passed`.

### FACT 1 — gather metadata loads are UNIFORM and ALREADY `s_load_b64` at baseline
BASE llir (kv_indptr=%2, kv_prefix_sum=%4), 4 uniform `<1 x i32>` entry loads:
```
%18 = load <1 x i32>, ptr addrspace(1) %17(=gep %2, sext pid_batch), align 4   ; kv_block_start
%23 = load <1 x i32>, ptr addrspace(1) %22(=gep %2, pid_batch+1), align 4      ; kv_block_end
%26 = load <1 x i32>, ptr addrspace(1) %25(=gep %4, pid_batch), align 4        ; context_start
%29 = load <1 x i32>, ptr addrspace(1) %28(=gep %4, pid_batch+1), align 4      ; context_end
```
`opt -passes='print<uniformity>'`: NONE of %18/%23/%26/%29 appear in the DIVERGENT list ->
UNIFORM. BASE amdgcn prologue (lines 421/424 loc): they coalesce to
`s_load_b64 s[4:5], s[10:11], 0x0` and `s_load_b64 s[12:13], s[16:17], 0x0` — scalar SMEM
already, WITHOUT noalias.

### FACT 2 — baseline gets `!amdgpu.noclobber` on the metadata loads WITHOUT noalias
`opt -mcpu=gfx1250 -passes='amdgpu-annotate-uniform'` on the BASE llir tags exactly the 4
metadata loads `!amdgpu.noclobber` (count = 4) and 22 `!amdgpu.uniform`, with NO noalias arg.
Mechanism: these are entry-block loads before any store; `isClobberedInFunction`/MemorySSA
finds no reaching store (the k_prefix/v_prefix stores are later + in the loop), so
readonly-inference alone yields noclobber. (The shipped TRITON dump shows 0 noclobber only
because that carrier is added inside the backend pipeline; the annotate pass materializes it.)
This is precisely S4 condition #4 FAILING: no interfering write that AA cannot otherwise
disprove -> noalias has nothing to add.

### FACT 3 — NO descriptor round-trip: SIFixSGPRCopies inserts ZERO V_READFIRSTLANE
Stock `llc -mcpu=gfx1250` on BASE llir:
- `-print-before=si-fix-sgpr-copies`: V_READFIRSTLANE count = **0**.
- `-stop-after=si-fix-sgpr-copies`: V_READFIRSTLANE count = **0**.
There is NO scalar-descriptor-VALU-ized-into-VGPR to legalize (contrast decode/S10b: the
metadata was VGPR-resident `global_load`, SIFixSGPRCopies VALU-ized the TDM descriptor and
inserted 8 in-loop readfirstlane). Here `kv_block_start`/`context_start` stay in SGPR across
the loop; in-loop rfl = 0, whole-fn rfl = 0 in BASE. Nothing for MachineLICM to hoist either.

### FACT 4 — AFTER (all uniform args) is a codegen no-op on the gather
IR delta = noalias added to %1,%2,%3,%4 only. amdgpu-annotate-uniform noclobber count
**4 -> 4** (unchanged); `raw.ptr.buffer.load` count **111 -> 111**; metadata loads still
`s_load_b64` (asm lines 424/426). Load histogram BYTE-IDENTICAL BASE vs AFTER:
`93 buffer_load_b128 | 9 buffer_load_b32 | 2 s_load_b256 | 3 s_load_b64` (both). in-loop rfl
0 -> 0, whole-fn rfl 0 -> 0. `.text` md5 differs (3d2e4e1e vs 227e6833) but the diff is
scheduler/ALU noise only: `s_delay_alu 82->85`, `s_wait_loadcnt 42->41` (one fewer wait — AA
gives the scheduler marginally more freedom, same signature as S12 Fact 2), `v_cndmask_b32`
e32/e64 form flips (15/28 -> 12/33), `v_nop 30->34`, `v_or_b32 82->81`. The memory-op lines
in the diff differ ONLY in register allocation (e.g. `ds_load_b128 v[152:155]` vs
`v[120:123]`, same address operand, same opcode) — ZERO change to any load selection. FFM PASS.

### FACT 5 — RE-AUDIT of the "divergent vector" claim: the read gather is TRULY divergent
`opt print<uniformity>` marks every `kv_indices`/`k_buffer`/weight read as
`DIVERGENT: %N = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(...)` (111 of them).
This is genuine per-lane divergence (each lane loads a different index via
`arange(ChunkK)//KBlockSize`), NOT uniform-forced-VGPR. S5 ceiling is over-determined:
divergent address -> VMEM regardless of noalias; and a multi-lane vector has no scalar SMEM
form. noalias on `kv_indices` cannot ever make it `s_load` (buffer.load 111->111).

### FACT 6 — RE-AUDIT "all-TDM / no raw uniform load" for the gluon primary: raw uniform loads DO exist, but already s_load
The gluon MLA-cat kernel has raw uniform scalar loads (`gl.load(slot_mapping_ptr+pid_b)`,
`gl.load(pos_ptr+...)`, `gl.load(k_scale_ptr)`) alongside its TDM input path — i.e. it is NOT
"all-TDM." But it has no loop. BASE amdgcn: whole-fn rfl = 0, load histogram
`1 s_load_b256 | 6 s_load_b32 | 1 s_load_b512 | 4 s_load_b64 | 1 s_load_b96` (no VMEM). With
the FULL uniform set {slot_mapping,pos,k_scale,cos,sin}, AFTER `.text` md5 = BASE md5
(**beb49740b3992b75798cfcd304394e06**, code diff = 0 lines, load histogram identical, rfl
0->0). Even stronger than S12b's slot_mapping-only result: byte-identical code. FFM PASS in both.

## A/not-A result
| kernel | metric | BASE | AFTER (all uniform args) |
|---|---|---|---|
| gather_kv_b_proj (loop) | in-loop rfl | 0 | 0 |
| | load histogram | 93 bl_b128 / 9 bl_b32 / 2 sl_b256 / 3 sl_b64 | identical |
| | noclobber (annotate) | 4 | 4 |
| | buffer.load | 111 | 111 |
| | .text | 3d2e4e1e | 227e6833 (scheduler/regalloc noise only) |
| gluon MLA-cat (no loop) | in-loop rfl | 0 | 0 |
| | .text md5 | beb49740 | beb49740 (IDENTICAL) |
| | code diff lines | — | 0 |

## FFM correctness
- gather_kv_b_proj `[1-1-4-k_buffer_type4-512]`: BASE `1 passed`, AFTER `1 passed`.
- gluon MLA-cat `[False-cache_dtype0-True-64-False-1-16384-512-64-1-16-1]`: BASE `1 passed`,
  AFTER `1 passed`. (Contract is pure codegen; numerics identical by construction.)

## Counter-experiment (the kill-check that noalias is inert)
Claim to refute: "adding all uniform read-only args changes load selection / removes an
in-loop round-trip." Results: (1) gather noclobber 4->4, buffer.load 111->111, metadata still
s_load_b64, in-loop rfl 0->0; (2) gluon primary `.text` byte-identical; (3) SIFixSGPRCopies
inserts 0 V_READFIRSTLANE in BASE (no round-trip existed to remove). NOT REFUTED — the
corrected protocol is a no-op for F4.

## Why F4 differs from decode (the load-selection-vs-round-trip distinction)
Decode (S10b): uniform metadata loads (ExptData/ExptHist/ExptOffs) were reused to build a TDM
descriptor across an in-loop store AA could not disprove -> stayed VGPR `global_load` ->
SIFixSGPRCopies VALU-ized the descriptor + 8 in-loop readfirstlane; noalias -> noclobber ->
s_load fixed it 8->0. F4-gather: the uniform metadata (`kv_indptr`/`kv_prefix_sum`) is consumed
at ENTRY to compute loop-invariant `kv_block_start`/`context_start`, with NO preceding store ->
readonly-inference already yields noclobber -> already s_load -> no VGPR excursion -> no
round-trip. F4's ONLY divergent load (`kv_indices`) is a genuine per-lane gather (S5 ceiling),
which noalias can never scalarize. The two scatter kernels have no loop at all.

## Remaining unknowns / scope
- kv_cache `_cat_and_cache_mla_kernel` (non-gluon, no loop) not separately recompiled with the
  +k_scale_ptr arg; over-determined no-op (no loop -> no in-loop rfl; S12 Fact 3/4 already
  showed slot_mapping already-s_load, metadata-only noalias diff; k_scale_ptr is another
  uniform entry load with no interfering preceding store).
- Larger shapes / FP4 (`_triton_gather_kv_b_proj_fp4_impl`) / shuffled paths not run. The
  metadata loads' uniform-entry shape and the gather's per-lane-divergent shape are
  size/dtype-independent, so the verdict is not a small-size artifact.
- No runtime numbers (mechanism/asm/FFM-correctness cell). Byte-identical (gluon) / load-
  histogram-identical (gather) `.text` => identical runtime by construction; no AM/gfx950 leg
  warranted.

## Harness friction (for the orchestrator)
- The full parametrized `test_fused_qk_rope_cat_and_cache_mla` (all shapes) exceeds the 2-min
  bash timeout; must pin the single smallest node id
  `[False-cache_dtype0-True-64-False-1-16384-512-64-1-16-1]`.
- FFM-lite env `/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh` + `LD_LIBRARY_PATH`
  prepend of /opt/rocm/lib; pytest needs PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (pylama plugin) +
  PYTEST_PLUGINS=ffm_teardown + PYTHONPATH=/root/scripts/tools. `--collect-only` without these
  flags aborts (pylama PluginValidationError).
- The aiter tree already carried R3 compile fixes + a concurrent cell's mla.py (F2) noalias
  edit on arrival; left both untouched (R3 is required to compile the gluon primary; mla.py is
  not this cell's file). My owned files reverted clean.
