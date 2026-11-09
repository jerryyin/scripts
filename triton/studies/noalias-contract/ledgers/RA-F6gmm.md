STATUS: done
CONCLUSION: CONFIRMED (verdict UNCHANGED). The corrected all-uniform-read-only-args protocol does NOT change the F6 gmm (non-TDM) verdict — it remains a no-op. Unlike decode (S10b), gmm has only ONE read-only metadata pointer (`group_sizes_ptr`, arg idx 2), and its two loads are GENUINELY DIVERGENT, not uniform-forced-to-VGPR: `opt -passes=print<uniformity>` marks the group_sizes buffer loads `%25`/`%118` and their entire offset chain (`%16 workitem.id.x` → `%20 = %16 & 3` → `%23` → `%24`) as DIVERGENT — the offset is a per-lane `tl.arange(0,BLOCK_SIZE_G)`. Because condition #2 (uniform address) fails at the source, the load lowers to a per-lane `BUFFER_LOAD_DWORD_VBUFFER_OFFEN` (vgpr, OFFEN) that has no scalar form regardless of noalias. AFTER (noalias on group_sizes_ptr) vs BASELINE differ by exactly the single `noalias` LLIR define-token; the amdgcn INSTRUCTION STREAM is byte-identical (md5 3e821e8b915d2975d1ed0d3fc2bac592). The one whole-fn `v_readfirstlane` (in-loop rfl = 0 in both) is the `tl.atomic_add(tile_counter_ptr)` return broadcast at gmm.py:390 (PHI at debug-loc !11), NOT a group_sizes-derived descriptor round-trip — so decode's SIFixSGPRCopies V2S failure mode is absent here. S18 was RIGHT; this re-audit adds the uniformity + SIFixSGPRCopies evidence S18 lacked. FFM PASS at the smallest work-stealing shape (rel_err 0.000000, cosine 1.000000).

# RA-F6gmm — F6 gmm (non-TDM) re-audit under the corrected "all uniform read-only args" protocol

Supersedes the *method* behind S18 (which toggled only `group_sizes_ptr` — but that happens to
be the ONLY uniform-read-only candidate in gmm, so S18's conclusion stands). This cell adds the
two pieces the corrected protocol demands and S18 omitted: (1) uniformity-analysis proof that the
group_sizes load is TRULY divergent (not uniform-forced-VGPR, the decode false-negative pattern),
and (2) the SIFixSGPRCopies trace proving no scalar-descriptor→VGPR round-trip exists.

## Why decode's blind spot does NOT apply to gmm
Decode (S10b) missed the UPSTREAM uniform metadata loads (ExptData/ExptHist/ExptOffs), which were
separate read-only pointers stranded in VGPR. gmm has only ONE metadata pointer — `group_sizes_ptr`
— and it is exactly the arg S18 tested. There is no second uniform read-only pointer to have missed.
Moreover gmm's single metadata load is *divergent by construction* (per-lane `tl.arange`), the
opposite of decode's *uniform-forced-VGPR*. Different condition fails (#2 uniform address vs decode's
#1/round-trip), so the fix that worked for decode cannot apply.

## Step 1 — enumerated uniform read-only pointer args (gmm_kernel, WORK_STEALING=True → _work_stealing_gmm)
Arg-attr inference in the post-opt LLIR define line confirms the read/write classification:
| idx | arg | LLIR attr | read-only? | uniform addr? | verdict |
|-----|-----|-----------|-----------|---------------|---------|
| 0 | lhs_ptr | readonly | yes | NO — per-lane matmul operand (`offs_lhs_m` divergent) | not a candidate (#2) |
| 1 | rhs_ptr | readonly | yes | NO — per-lane matmul operand | not a candidate (#2) |
| 2 | **group_sizes_ptr** | readonly | **yes** | **NO — `group_sizes_ptr + tl.arange(0,BLOCK_SIZE_G)` is per-lane (proven divergent below)** | **the only metadata ptr; TESTED** |
| 3 | out_ptr | writeonly | no (tl.store) | — | excluded (written) |
| 4 | bias_ptr | readnone | (USE_BIAS=False → unused; when used, `bias_ptr+g*N+offs_bias_n` is per-lane over N) | NO | excluded / not a candidate |
| 5 | tile_counter_ptr | (neither ro/wo) | no (atomic_add RMW) | — | excluded (written) |
| 10,11 | (trailing None) | readnone | — | — | excluded (unused) |

The ONLY read-only pointer feeding an addressing/derived computation is `group_sizes_ptr` — the
exact set S18 tested. There is no missed metadata pointer (contrast decode's 3). AFTER = add
`group_sizes_ptr` (idx 2) to noalias_args; BASELINE = none. This is the S18 set, but the re-audit
below re-checks the DIVERGENCE claim that S18 asserted without uniformity evidence.

## Frozen experiment
- **AITer:** `/root/aiter`, kernel `aiter/ops/triton/_triton_kernels/gmm.py::gmm_kernel` →
  `_work_stealing_gmm`. (The task pointed at `ops/triton/gmm.py`, the wrapper; the @triton.jit
  bodies live in `_triton_kernels/gmm.py`, per S18's file-location note.) A/not-A toggle =
  `@triton.jit(noalias_args=["group_sizes_ptr"])` on the source (FFM leg) and, for the offline
  asm study, `KernelParam.noalias` on `gmm_kernel.fn.params` with `device_caches.clear()`.
- **Triton:** installed tree resolves to `/root/triton` (#120 contract landed; noalias is a pure
  Python annotation — no rebuild).
- **Target:** gfx1250. Offline compile via `warmup` with MockTensors under a monkeypatched
  `driver.active.get_current_target → GPUTarget("hip","gfx1250",32)` and
  `arch_info.get_arch → "gfx1250"` (host is gfx950; compile-only asm/ISel study, matching S18).
- **Shape (offline asm):** M=512,K=64,N=64,G=4, BLOCK_SIZE_M=64,BLOCK_SIZE_K=32,BLOCK_SIZE_N=64,
  GROUP_SIZE=1,GRID_DIM=8,int32 group_sizes,WORK_STEALING=True — smallest that emits the
  work-stealing loop + in-loop atomic + in-loop group_sizes reload (matches S18).
- **Shape (FFM correctness):** M=512,K=64,N=128,G=4 even split [128,128,128,128], work_stealing=True
  (N=128≠K disambiguates the rhs-layout check; smallest exercising work-stealing shape).
- **LLVM pins (STOCK):** `opt`/`llc` `/root/.triton/llvm/llvm-56421f92-ubuntu-x64-1/bin`,
  `-mcpu=gfx1250 -filetype=asm`. No patched flags used.
- **FFM env:** `source /zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh; export
  LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- **Isolation:** per-cell TRITON_CACHE_DIR (`/tmp/tc-RA-F6gmm-{base,after,ffm3}`),
  TRITON_ALWAYS_COMPILE=1. Artifacts in `/tmp/ra-gmm/{base,after}/gmm.{ttir,ttgir,llir,amdgcn}`,
  `sfsc_after.mir`, `uniformity.txt`.
- **Kill switch:** the ONLY delta is `noalias` on group_sizes_ptr. Tree reverted after (gmm.py
  clean; `git diff` shows only unrelated pre-existing fused_kv_cache edits, not mine).

## HOW-facts (real IR/asm + traces)

### Fact A — post-opt LLIR delta = exactly one `noalias` token (reproduces S18 Fact A)
```
< ... ptr addrspace(1)         nofree readonly captures(none) %2, ...   (BASELINE)
> ... ptr addrspace(1) noalias nofree readonly captures(none) %2, ...   (AFTER)
```
TTGIR `tt.noalias` count: base 0 / after 1. Everything else byte-identical.

### Fact B — amdgcn INSTRUCTION STREAM byte-identical (reproduces S18 Fact B)
md5 of the stripped instruction stream (s_/v_/ds_/buffer_/global_/tensor_ …):
`3e821e8b915d2975d1ed0d3fc2bac592` for BOTH base and after. The only amdgcn diff is a
metadata-YAML line (`+ .actual_access: read_only` on the noalias arg) — changes no instruction.
Histogram identical: `buffer_load_b128 4, buffer_load_b32 2, s_load_b256 1, s_load_b64 1,
s_load_b96 1, v_readfirstlane_b32 1`. In-loop rfl = 0 in BOTH.

### Fact C — the group_sizes load is TRULY DIVERGENT (the decisive re-audit S18 lacked)
The read is `llvm.amdgcn.raw.ptr.buffer.load.i32(%22, %24, ...)` where
`%22 = make.buffer.rsrc(group_sizes_ptr %2)` and the offset `%24` traces to:
```
%16 = llvm.amdgcn.workitem.id.x()          ; per-lane thread id
%20 = and i32 %16, 3                        ; = tl.arange(0,BLOCK_SIZE_G) (BLOCK_SIZE_G=4)
%23 = shl nuw nsw i32 %20, 2                ; *4 bytes
%24 = select i1 %21, i32 %23, i32 -2147483648
%25  = raw.ptr.buffer.load.i32(%22, %24)   ; entry  (from _total_gmm_tiles)
%118 = raw.ptr.buffer.load.i32(%22, %24)   ; in loop (from _resolve_gmm_tile)
```
`opt -passes='print<uniformity>'` (stock 56421f92) verdict — ALL DIVERGENT:
```
DIVERGENT:  %16 = ... workitem.id.x()
DIVERGENT:  %20 = and i32 %16, 3
DIVERGENT:  %23 = shl nuw nsw i32 %20, 2
DIVERGENT:  %24 = select i1 %21, i32 %23, i32 -2147483648
DIVERGENT:  %25  = ... raw.ptr.buffer.load.i32(%22, %24, ...)
DIVERGENT:  %118 = ... raw.ptr.buffer.load.i32(%22, %24, ...)
```
=> Each lane reads a DIFFERENT element `group_sizes[lane & 3]`. This is a per-lane gather, NOT a
uniform load. It fails condition #2 (uniform address) at the source. Contrast decode's metadata
loads, which uniformity marks UNIFORM (block-uniform index) but were merely stranded in VGPR — a
fixable state. gmm's is the genuine-divergence case that noalias can never help.

### Fact D — SIFixSGPRCopies: no scalar-descriptor→VGPR round-trip (decode's failure mode absent)
`llc -mcpu=gfx1250 -stop-after=si-fix-sgpr-copies` on the BASELINE LLIR:
- group_sizes loads: `%272:vgpr_32 = BUFFER_LOAD_DWORD_VBUFFER_OFFEN %11, killed %268, ...`
  and `%397:vgpr_32 = BUFFER_LOAD_DWORD_VBUFFER_OFFEN %11, killed %396, ...` — VGPR-dest vector
  buffer loads with a per-lane (OFFEN) offset. No scalar descriptor is built from group_sizes, so
  there is nothing for SIFixSGPRCopies to VALU-ize / no VGPR→SGPR readfirstlane to insert.
- The single `V_READFIRSTLANE_B32` in the whole MIR: `%1527 = V_READFIRSTLANE_B32 %242`, where
  `%242 = PHI %1520,%bb.25, %241,%bb.26` at debug-location `!11` = gmm.py:390 =
  `tl.atomic_add(tile_counter_ptr, 1)`. It broadcasts the (divergent) atomic return value across
  lanes to advance the shared tile counter. `tile_counter_ptr` is a WRITTEN pointer (atomic RMW),
  excluded from the read-only set; this rfl is inherent to the work-stealing atomic and unrelated
  to group_sizes or to noalias. It is present identically in BOTH variants and is NOT in the inner
  K-loop (in-loop rfl = 0).

This is the opposite of decode: decode had a scalar TDM descriptor being VALU-ized because its
SGPR operand was fed by VGPR-resident *uniform* metadata loads (the SIFixSGPRCopies V2S anchor);
adding noalias scalarized those loads and removed the round-trip. gmm has no such anchor — the
group_sizes load is divergent and buffer-lowered, and the only rfl is an atomic broadcast.

### Fact E — re-audit of the "raw uniform load hiding in TDM" concern: N/A
gmm is non-TDM (a plain buffer-ops matmul, S6 path). There is no TDM descriptor and no hidden raw
`gl.load` of a uniform read-only arg. All reads are `raw.ptr.buffer.load` (lhs/rhs/group_sizes/bias)
or LDS `load addrspace(3)`. The lhs/rhs/bias loads are all per-lane divergent (matmul operands),
group_sizes is per-lane divergent (Fact C). No uniform read-only raw load exists to scalarize.

### Fact F — FFM correctness of the AFTER variant (S18's blocked leg, now unblocked)
Source-annotated `@triton.jit(noalias_args=["group_sizes_ptr"])`, work_stealing=True, under FFM
(`/zyin/...ffmlite_env.sh`), smallest exercising shape M=512,K=64,N=128,G=4 even split:
```
RESULT: PASS rel_err 0.000000 cosine 1.000000
```
FFM-compiled llir carries the `noalias` token and 3 `raw.ptr.buffer.load.i32` (matches offline).
rel_err is exactly 0 because the amdgcn is byte-identical to baseline — AFTER and BASELINE are the
same program, so correctness is preserved by construction. (S3-style soundness: group_sizes_ptr is
never stored/atomic'd anywhere in the file, so the annotation is safe as well as inert.)

## A/not-A result
| metric | BASELINE (noalias_args=[]) | AFTER (noalias_args=[group_sizes_ptr]) | delta |
|--------|---------------------------|----------------------------------------|-------|
| LLIR define | no noalias on %2 | `noalias` on %2 | +1 token |
| amdgcn instr-stream md5 | 3e821e8b… | 3e821e8b… | IDENTICAL |
| in-loop v_readfirstlane | 0 | 0 | 0 |
| whole-fn v_readfirstlane | 1 (atomic broadcast) | 1 (atomic broadcast) | 0 |
| s_load / buffer_load histogram | s_load 3, buffer_load 6 | s_load 3, buffer_load 6 | 0 |

No codegen change. Verdict UNCHANGED: no-op.

## Counter-experiments (what would refute, and the result)
1. "group_sizes is uniform-forced-VGPR (like decode), so noalias should scalarize it to s_load."
   → REFUTED: uniformity analysis marks the load AND its whole offset chain DIVERGENT (Fact C);
   the offset is `workitem.id.x() & 3` = per-lane `tl.arange`. Condition #2 fails at the source.
2. "A scalar descriptor is VALU-ized into VGPR+readfirstlane (decode round-trip); noalias fixes it."
   → REFUTED: SIFixSGPRCopies MIR shows the group_sizes loads go straight to `vgpr_32` via
   `BUFFER_LOAD_DWORD_VBUFFER_OFFEN` (no scalar descriptor, no round-trip); the sole rfl is the
   atomic_add broadcast at gmm.py:390 on a WRITTEN pointer (Fact D).
3. "A second uniform read-only metadata pointer was missed (the decode blind spot)."
   → REFUTED: gmm has exactly one metadata pointer, group_sizes_ptr; all other read-only ptrs
   (lhs/rhs/bias) are per-lane matmul operands (Fact C table). Nothing to have missed.
4. "The amdgcn differs somewhere."
   → REFUTED: instruction-stream md5 identical (Fact B). Only a metadata-YAML descriptor line moves.

## CLAIM STATUS
CONFIRMED / UNCHANGED. The corrected all-uniform-read-only-args protocol reduces, for gmm, to the
single arg S18 tested (there is no second metadata pointer), and the divergence claim S18 asserted
is now PROVEN by uniformity analysis (not merely stated). Specific failing condition: **#2 (uniform
address) — the group_sizes load is genuinely per-lane divergent (`tl.arange(0,BLOCK_SIZE_G)` over
`workitem.id.x`), lowered to a VGPR OFFEN buffer load with no scalar form**; additionally there is
no SIFixSGPRCopies round-trip to remove (Fact D). Not the decode pattern. NO-SHIP for gmm.

## Harness friction
- The task pointed at `ops/triton/gmm.py` (wrapper); the @triton.jit bodies are in
  `_triton_kernels/gmm.py` (S18 file-location note holds).
- `driver.active` (not `driver.driver.active`) is the HIPDriver; override its `get_current_target`
  for offline gfx1250 compile. Host is gfx950/warp64.
- The op_test `test_gmm_alt_trans_rhs...work_stealing` uses a FIXED huge shape
  (M=267424,K=1280,N=2560) that times out in FFM functional sim (>550s). Used a small standalone
  harness (`/tmp/ra_gmm_ffm.py`) calling the real `gmm(..., work_stealing=True)` wrapper instead.
- pytest fails to start under the repo's autoloaded `pylama` plugin (hookspec mismatch); set
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- FFM prints `_amdgpu_device_initialize: amdgpu_get_auth failed (-1)` but proceeds and runs
  correctly — benign.
- K==N in the test setup trips an rhs-layout ambiguity assert in `get_gmm_transposition`; use N≠K.
