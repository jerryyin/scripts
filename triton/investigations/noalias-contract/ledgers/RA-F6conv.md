STATUS: done
CONCLUSION: CONFIRMED no-op — the corrected "all uniform read-only args" protocol does NOT change the F6 causal_conv1d (non-TDM) verdict. Unlike decode, there is NO upstream metadata pointer the original cell missed: the complete set of uniform read-only args feeding the index path is exactly {query_start_loc_ptr, cache_indices_ptr, has_initial_states_ptr}, the same three S17 already toggled. Annotating ALL of them changes NOTHING: the executable instruction stream is byte-identical baseline vs after (md5 90a38e05… after stripping the debug-string-table artifact; §HOW-2), in-loop v_readfirstlane = 0 in both, whole-fn rfl = 1 in both, on BOTH stock llc pins. The specific failing condition is "already s_load AND no round-trip": the two int32 index loads are already S_LOAD_DWORDX2/DWORD scalar at baseline (they even carry `"amdgpu-noclobber"` in their MMO WITHOUT noalias, via the readonly-derived invariant-load route, S5 Fact E), so SIFixSGPRCopies has no VGPR→SGPR copy to legalize on the index path (§HOW-4); and the one surviving v_readfirstlane is the i8 has_initial_states load hitting the S5 sub-dword ceiling (global_load_u8 → V_READFIRSTLANE, in block .LBB0_5, NOT the compute loop), which noalias cannot fix. Nothing is "uniform-but-forced-to-VGPR" on a scalarizable-shape load. FFM PASS on the AFTER variant. NO-SHIP.

# RA-F6conv — F6 causal_conv1d (non-TDM) re-audit under the corrected all-args protocol

Supersedes nothing (S17's verdict stands, now proven under the stronger protocol). Owns ONLY
`aiter/ops/triton/_triton_kernels/conv/causal_conv1d.py::_causal_conv1d_fwd_kernel`.
This cell re-tests whether S17's index-only annotation was a decode-style FALSE NEGATIVE. It was NOT.

## Why decode's failure mode does NOT apply here (the point of the re-audit)
Decode (S10b) round-tripped because a SEPARATE set of upstream uniform read-only *metadata*
pointers (ExptData/ExptHist/ExptOffs) stayed VGPR and VALU-ized the TDM descriptor. For
causal_conv1d fwd I enumerated every pointer arg and found there is NO such hidden metadata
pointer: the only uniform read-only loads are the three index/flag arrays S17 already covered, and
there is NO TDM descriptor for them to feed (non-TDM kernel). So the decode mechanism has no analog
here — the re-audit closes the methodological gap and confirms the null result.

## Step 1 — enumerated uniform read-only pointer args (address provenance traced)
Kernel `_causal_conv1d_fwd_kernel` (arg indices = post-opt LLVM `define` order):
- `query_start_loc_ptr` (%6, i32): `tl.load(query_start_loc_ptr + idx_seq)` / `+ idx_seq + 1`
  (src :70/:71). Address derives from `tl.program_id(0)` → UNIFORM. Never stored through → read-only.
  → **INCLUDE.**
- `cache_indices_ptr` (=conv_state_indices_ptr, %4, i32): `tl.load(conv_state_indices_ptr + idx_seq)`
  (src :88). program_id-derived → UNIFORM, read-only. → **INCLUDE.**
- `has_initial_states_ptr` (%5, i8/i1): `tl.load(has_initial_states_ptr + idx_seq)` (src :111).
  program_id-derived → UNIFORM, read-only. → **INCLUDE.**
- `x_ptr` (%0), `w_ptr` (%1), `bias_ptr` (%2): all addressed with `idx_feats =
  program_id(2)*BLOCK_N + tl.arange(0,BLOCK_N)` → per-lane → **DIVERGENT** address. Not uniform.
  → EXCLUDE (condition #2 fails; divergent ⇒ VMEM forever regardless of noalias).
- `initial_states_ptr`/`conv_states` (%3): both LOADED and STORED through (`tl.store` :183/:228/…)
  → not read-only. → EXCLUDE.
- `o_ptr` (%7): stored only → not read-only. → EXCLUDE.

**Full uniform-read-only set = {cache_indices_ptr, has_initial_states_ptr, query_start_loc_ptr}.**
No metadata pointer was missed (contrast decode). This is the exact set S17 tested — but S17's
verdict is now re-proven under the corrected protocol AND with the si-fix-sgpr-copies trace S17
never ran.

## Frozen experiment
- Triton: `/root/triton` @ ba4fd67b8ed2… (branch users/jerryyin/moe-gather-sload-contract, #120 landed),
  installed `triton` 3.8.0 resolves here. No rebuild (contract is a Python annotation).
- AITer: `/root/aiter`. Kernel file edited ONLY for the FFM AFTER leg, then reverted (tree CLEAN,
  `diff` against backup empty; no commits).
- BASELINE = `@triton.jit()` (0 noalias). AFTER = `@triton.jit(noalias_args=[the 3 args above])`
  (3 noalias on define). The ONLY delta is that set (verified at TTIR: 0 vs 1 `tt.noalias`; at LLVM
  define: 0 vs 3 `noalias`).
- Target: gfx1250. Compile-only asm study via JIT `kernel.warmup(warmup=True)` with forced
  `GPUTarget("hip","gfx1250",32)` (S17 pattern; this host is gfx950 so runtime dumps need FFM).
  Constexprs KERNEL_WIDTH=4, BLOCK_M=8, BLOCK_N=256, dim=256, NP2_STATELEN=4, all feature flags on.
- Stock llc pins (both): `/root/.triton/llvm/llvm-56421f92-…/bin/llc` and
  `…/llvm-62b7cf96-…/bin/llc`, `-mcpu=gfx1250 -filetype=asm`. Stock `opt` 56421f92 for uniformity.
  (No `-amdgpu-hoist-uniform-readfirstlane`; absent on stock, not used.)
- Isolation: per-mode `TRITON_CACHE_DIR=/tmp/RA-F6conv/tc-{baseline,after}`, `TRITON_ALWAYS_COMPILE=1`.
- Metric: s_load/global_load/buffer_load/v_readfirstlane histogram; in-loop rfl (awk loop-body
  counter); normalized `.text` md5; post-opt `!amdgpu.noclobber`; `opt print<uniformity>`; llc
  `-print-before/-print-after=si-fix-sgpr-copies` V_READFIRSTLANE count.

## HOW-facts (real IR/asm/MIR)

### HOW-1 — contract plumbed (kill switch verified)
LLVM `define` diff, the ONLY difference (args %4/%5/%6):
```
BASELINE: ptr addrspace(1)         nofree readonly captures(none) %4 / %5 / %6
AFTER   : ptr addrspace(1) noalias nofree readonly captures(none) %4 / %5 / %6
```
Both are `readonly` (LLVM-inferred in BOTH) — matches S4 Hop1 / S17 Fact1.

### HOW-2 — instruction stream BYTE-IDENTICAL baseline vs after (the headline)
Naive `.text` md5 DIFFERED (b28bb01e… vs 491419cc… on pin 56421f92) — but this is a PURE ARTIFACT:
the two variants were compiled from two temp files with different names (`_kern_baseline.py` vs
`_kern_after.py`), which appear in every `.loc`/`.file` debug directive and in the `.asciz`
debug-string-table (whose byte length — and therefore string-table offsets — differ). After
stripping `.loc`/`.file`/`.asciz` and any line naming the source file, the instruction stream is
**md5-identical: 90a38e05778ca511e58587c519eda04b for BOTH** (706 lines each, `diff` empty). Same
on pin 62b7cf96. (S17 reported byte-identical `.text` because it toggled the decorator on ONE file
→ same filename → no string-table delta; this cell's temp-file harness surfaced the artifact, which
I then eliminated.) **Net: zero executable-code delta.**

### HOW-3 — memory-op histogram identical on BOTH stock pins
```
                    BASELINE   AFTER
s_load_b64               1         1   <- query_start_loc[idx_seq] & [idx_seq+1] coalesced (:70/:71)
s_load_b32               4         4   <- incl. cache_indices[idx_seq] (:88)
s_load_b512              1         1   <- kernarg block
global_load_u16         10        10
global_load_u8           1         1   <- has_initial_states (:111), i8
buffer_load_b128/b32/u16 3/5/8  3/5/8  <- x/w feature tiles (divergent, buffer path)
v_readfirstlane_b32      1         1   <- NOT in loop (see HOW-4)
```
in-loop rfl = **0 / 0**; whole-fn rfl = **1 / 1**. No s_load gained, no global_load lost.

### HOW-4 — the si-fix-sgpr-copies TRACE (S17 never did this; the decisive new evidence)
`opt print<uniformity>` (BASELINE): the four candidate loads
`%29,%34,%44 = load <1 x i32>` (query_start_loc×2, cache_indices) and `%54 = load <1 x i8>`
(has_initial_states) are NONE flagged DIVERGENT ⇒ all UNIFORM. The only DIVERGENT memory ops are
the `<1 x bfloat>` load/stores on `%3` (conv_states/x, addressed by the `idx_feats` arange).

llc `-print-before=si-fix-sgpr-copies` (BASELINE) — the int32 index loads are ALREADY scalar SMEM:
```
%114 = S_LOAD_DWORDX2_IMM_ec …  ("amdgpu-noclobber" load (s64) from %ir.17 … addrspace 1); :70:28
%124 = S_LOAD_DWORD_IMM …        ("amdgpu-noclobber" load (s32) from %ir.29 … addrspace 1); :88:34
```
They carry `"amdgpu-noclobber"` in the MMO **without noalias** — the noclobber came from the
readonly-derived invariant-load route (S5 Fact E: uniform program_id address, prologue, no
clobbering store), NOT from the contract. Post-opt `.llir` confirms: these loads carry NO
`!amdgpu.noclobber` IR metadata in EITHER variant (the 10 IR-level noclobbers are all on the `%3`
bfloat loads); the noclobber that matters here is materialized at the MI level, identically in both.

`-print-after=si-fix-sgpr-copies`: **V_READFIRSTLANE count = 1 in BOTH** variants, and it is on the
i8 path only:
```
%179 = GLOBAL_LOAD_UBYTE_SADDR … ("amdgpu-noclobber" load (s8) from %ir.54 … addrspace 1); :111:31
%656 = REG_SEQUENCE %179.lo16 …
%180 = V_READFIRSTLANE_B32 %656 …; :111:31
```
So: NO scalar descriptor is VALU-ized into VGPR + readfirstlane on the index path (there is no TDM
descriptor at all, and the int32 loads never left SGPR). The lone readfirstlane is the uniform i8
`has_initial_states` value read back from a VGPR `global_load_u8` — the S5 sub-dword ceiling (no
wide scalar SMEM form for a single i8), fixable by neither noalias nor annotation. Final asm places
it in `.LBB0_5` (chunk_offset==0 init-state prep), NOT the inner compute loop `.LBB0_16`. The MIR
S_LOAD/V_READFIRSTLANE fragments are identical baseline vs after modulo a debug-location index.

### RE-AUDIT of the two prior claims
- "divergent vector" claim (x/w/bias): RE-CONFIRMED TRULY divergent — uniformity analysis marks the
  `%3` bfloat memory ops DIVERGENT; addressed by `idx_feats` arange (per-lane). Not noalias-fixable.
- "all-TDM / no raw uniform load" concern: N/A — this kernel is non-TDM and DOES have raw
  `gl.load` of uniform read-only args (the three index/flag loads). I audited them directly: two
  are already scalar S_LOAD; the third is the sub-dword-ceiling i8. No uniform load on a
  scalarizable shape is forced to VGPR/global_load. So no decode-style hidden win.

### HOW-5 — FFM correctness (AFTER variant, real gfx1250)
Applied the 3-arg `noalias_args` to the real source; ran the smallest full-path shape
`test_causal_conv1d_varlen[1-False-2048-64-4-True-True-itype0]` (batch1, dim2048, seqlen64,
width4, bias, silu, bf16) under FFM (`/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh`):
**1 passed**. The FFM-run compiled asm (real driver, width4) shows the identical index structure:
`s_load_b64` + `s_load_b32` scalar, `global_load_u8` + 1 non-loop `v_readfirstlane`, in-loop rfl 0,
3 noalias on define. Counter-experiment BASELINE (0 noalias, same shape): **1 passed** too — the
contract is numerically inert (pure hint), as expected. Tree reverted, `diff` clean.

## Counter-experiments that would refute (results)
1. "Some uniform read-only arg is global_load in BASELINE and s_load in AFTER." → both int32 are
   S_LOAD in both; the i8 is global_load in both. Not refuted.
2. "A scalar index descriptor is VALU-ized → V_READFIRSTLANE that noalias removes." → si-fix trace
   shows the sole V_READFIRSTLANE is the i8 flag, count 1 in BOTH; index path never leaves SGPR.
   Not refuted.
3. "The `.text` sections genuinely differ (codegen delta)." → identical after removing the
   debug-string-table artifact (md5 90a38e05…, empty diff), on both pins. Not refuted.
4. "There's a hidden uniform read-only metadata pointer (decode-style) I missed." → full arg
   enumeration (Step 1) shows the only uniform read-only ptrs are the 3 index/flag arrays; no
   metadata pointer exists. Not refuted.

## Verdict
CONFIRMED (verdict UNCHANGED from S17). The corrected all-args protocol is a NO-OP for F6
causal_conv1d (non-TDM). Failing condition: "already s_load (via readonly-derived MMO noclobber,
no noalias needed) AND no VGPR→SGPR round-trip on the index path" for the int32 loads; "sub-dword
ceiling" for the i8 flag. This is NOT a decode-style false negative because there is no upstream
uniform-VGPR metadata load to scalarize. NO-SHIP.

## Harness friction
- `run_on_model.sh` broken here (hardcodes /am-ffm); sourced the FFM env directly as instructed.
- pytest plugin `pylama` is incompatible with the installed pluggy (`PluginValidationError` on
  `pytest_collect_file`); ran FFM tests with `-p no:pylama`.
- The `.text` md5 "difference" was a debug-string-table artifact from compiling two differently
  named temp files — NOT a codegen delta. Normalize the source filename (or strip
  `.loc`/`.file`/`.asciz`) before trusting a whole-.text md5 across a two-file A/not-A. S17's
  same-file toggle avoided this; document it for future S-RA cells.
