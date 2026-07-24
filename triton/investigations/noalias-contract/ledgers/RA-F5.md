STATUS: done
CONCLUSION: VERDICT CONFIRMED (no-op). The corrected "all uniform read-only args" protocol does NOT change F5's verdict. Unlike decode, F5 has NO raw uniform `gl.load` hiding anywhere: every one of the 5 pointer args is consumed ONLY by `tt.make_tensor_descriptor`, TTGIR has 0 `tt.load`/0 `tt.store`, post-opt LLIR has 0 `load addrspace(1)`, and the AMDGPU MIR after `si-fix-sgpr-copies` has 0 GLOBAL_LOAD and 0 READFIRSTLANE. Adding noalias to the 3 uniform read-only args (x1_ptr,w1_ptr,res1_ptr) — or the maximal all-5 set — leaves the gfx1250 `.text` BITWISE IDENTICAL (md5 8e683109… on pin 56421f92, 7d2548b1… on pin 62b7cf96) with in-loop rfl 0→0. The specific failing condition is condition #1 (no raw-pointer load exists): the decode false-negative was driven by VGPR-resident `global_load` of ExptData/ExptHist/ExptOffs; F5 has zero such loads, so there is nothing for `noalias→!amdgpu.noclobber→s_load` to act on and nothing for SIFixSGPRCopies to VALU-ize. S13's grounded negative stands, now re-proven with the upstream-metadata-load audit S13 never ran.

# RA-F5 — re-audit of F5 norm (fused_rmsnorm_add) under the corrected all-uniform-read-only-args protocol

Supersedes nothing (confirms S13); re-audits it against the S10b blind spot (index-only annotation
→ missed upstream uniform read-only metadata loads). The S10b lesson: decode was "mostly TDM" yet
had raw uniform `gl.load` of metadata args that stayed VGPR and forced a VGPR→SGPR round-trip. This
cell asks the same question of F5: is there ANY raw uniform read-only load hiding behind the TDM?
Answer, proven at IR + MIR + asm: NO.

## Enumerated uniform read-only pointer args (protocol step 1)
Kernel `_gluon_fused_rms_kernel` (`@gluon.jit`, no noalias_args as-shipped). Signature → LLVM args:
```
%0 x1_ptr        read-only, uniform (kernarg base)  — consumed by make_tensor_descriptor → TDM
%1 w1_ptr        read-only, uniform (kernarg base)  — consumed by make_tensor_descriptor → TDM
%2 res1_ptr      read-only, uniform (kernarg base)  — consumed by make_tensor_descriptor → TDM
%3 out1_ptr      WRITTEN  (async_store)             — not read-only
%4 out_res1_ptr  WRITTEN  (async_store)             — not read-only
%5 eps1          f32   scalar-by-value (not a pointer)
%6..%11 M,N,4 strides  i32 scalar-by-value (not pointers)
%12,%13          compiler scratch/profile ptrs (nofree readnone captures(none))
```
Uniform-read-only pointer args (the protocol's AFTER set) = **{x1_ptr, w1_ptr, res1_ptr}**.
There are NO index pointers and NO metadata pointers here (contrast decode's GatherIndx +
ExptData/ExptHist/ExptOffs). There are NO dynamic scalar-param pointers: eps1 and all strides are
passed by value. So the only thing to annotate is those 3 read pointers (+ optionally the 2 write
outputs, tested as a maximal counter-experiment).

## Frozen experiment
- Kernel: `~/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/norm/fused_rmsnorm_add.py`
  `_gluon_fused_rms_kernel`, gfx1250. Launcher `.../normalization/fused_rmsnorm_add.py`
  `_fused_rmsnorm_add_core` (gluon path taken iff `get_arch()=="gfx1250"`, satisfied under FFM).
- Triton: installed tree (branch with #120 contract; noalias is a pure Python annotation toggle,
  no rebuild). AITer tree left CLEAN (kernel reverted; no git commits). F5 needed NO R3 fix.
- Compile+correctness: FFM-lite `/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh`,
  `LD_LIBRARY_PATH=/opt/rocm/lib:…`, `TRITON_ALWAYS_COMPILE=1`, per-variant `TRITON_CACHE_DIR`
  (/tmp/RA-F5/tc-{BASE,AFTER,ALL5}). Shape M=2,N=32,bf16,has_res=True (smallest exercising all 5
  pointers incl. the residual add) — driver /tmp/RA-F5/run.py.
- Measure: stock llc pins `56421f92` and `62b7cf96`, `-mcpu=gfx1250 -filetype=asm`; in-loop rfl via
  the awk loop-body counter; whole-.text md5; global_load histogram; plus `opt -passes=print<uniformity>`
  and `llc -stop-after=si-fix-sgpr-copies` on BASE.llir. (Patched-only flag
  `-amdgpu-hoist-uniform-readfirstlane` NOT used; absent on stock llc.)
- A/not-A kill switch — the ONLY delta is the noalias_args set:
  - BASELINE (as-shipped): `@gluon.jit` — no noalias on any arg.
  - AFTER (protocol): `@gluon.jit(noalias_args=["x1_ptr","w1_ptr","res1_ptr"])` — the 3 uniform RO args.
  - ALL5 (counter-experiment, maximal): + `"out1_ptr","out_res1_ptr"`.
- Artifacts: /tmp/RA-F5/out/{BASE,AFTER}.{llir,amdgcn,ttgir}, *.pin2.s, BASE.sfsc.mir.

## HOW-facts (real IR / MIR / asm — the S13 blind-spot audit S13 never ran)

### 1. TTGIR: every pointer flows to make_tensor_descriptor; ZERO raw loads/stores
BASE.ttgir histogram: `tt.load`=0, `tt.store`=0, `make_tensor_descriptor`=5, `async_tdm_copy`=5,
`local_load`=3. All data movement is TDM async DMA to/from LDS. No raw-pointer gather/scatter.

### 2. Post-opt LLIR: ZERO addrspace(1) loads; the only loads are LDS (addrspace 3)
`grep -cE 'load .*addrspace\(1\)'` = **0**; `addrspace(3)` = 3 (the three `smem*.load(...)` tile
reads at lines 148/162/232, `load <8 x i16> ptr addrspace(3)`). No raw uniform read-only global
load of ANY arg exists — this is the exact construct that was decode's VGPR source, and it is
absent here.

### 3. amdgcn: no global_load, no v_readfirstlane; s_load are kernarg-only
BASE.amdgcn memory histogram: 3 `tensor_load_to_lds` (TDM x1/res1/w1), 2 `tensor_store_from_lds`
(TDM out_res1/out1), 3 `ds_load_b128`, 2 `ds_store_b128`, 1 each `s_load_b256/b128/b96/b64`.
`global_load`=0, `v_readfirstlane`=0. All 4 `s_load` read `s[0:1]` (the kernarg segment pointer,
addrspace 4) at fixed offsets 0x0/0x20/0x28/0x38 — they load the *arguments*, uniform by
construction and independent of any pointer-value noalias.

### 4. Uniformity analysis (opt -passes=print<uniformity>, BASE.llir)
The ONLY loads it classifies are the 3 addrspace(3) LDS reads, and they are **DIVERGENT**
(per-lane tile reads — correctly divergent; each lane reads its own element, NOT a uniform value
forced to VGPR). There is NO uniform addrspace(1) load to classify. RE-AUDIT of any "divergent
vector" framing: the divergent loads here are LDS, genuinely per-lane — they are not a
uniform-but-forced-to-VGPR case that noalias could fix (that case requires a uniform address; these
addresses are per-lane). So neither the "truly divergent" nor the "uniform-forced-to-VGPR" branch
offers a win: there simply is no uniform global load at all.

### 5. SIFixSGPRCopies trace (llc -stop-after=si-fix-sgpr-copies, BASE.llir) — the decisive round-trip audit
Post-pass MIR: **READFIRSTLANE = 0**, **GLOBAL_LOAD = 0**, TENSOR_LOAD_TO_LDS = 3, S_LOAD = 4.
The 4 S_LOAD all read `%1(p4)` (kernarg base, addrspace 4), each `dereferenceable invariant load …
from %ir..kernarg.offset*` — i.e. argument loads, NOT a data descriptor. There is NO scalar
descriptor being VALU-ized into VGPR and NO VGPR-resident uniform load to serve as a V2S anchor.
The decode failure mode (uniform metadata `global_load` → VGPR descriptor operand → SIFixSGPRCopies
inserts VGPR→SGPR readfirstlane round-trip) structurally cannot occur: its precondition (a VGPR
uniform load) is absent. The TDM `tensor_load_to_lds` descriptor operands are built from the
kernarg S_LOADs directly in SGPRs — no VGPR excursion.

## A/not-A result
| variant | define carries | md5(.text) pin1 | md5(.text) pin2 | in-loop rfl | global_load |
|---|---|---|---|---|---|
| BASELINE | (none) | 8e683109c26c2192430ba5e0e449514c | 7d2548b167cbd7bf78c7e1d25d4fb348 | 0 | 0 |
| AFTER (3 RO args) | noalias %0,%1,%2 | 8e683109…(identical) | 7d2548b1…(identical) | 0 | 0 |
| ALL5 (counter-exp) | noalias %0..%4 | 8e683109…(identical) | 0 | 0 | 0 |

`diff BASE.amdgcn AFTER.amdgcn` = empty (IDENTICAL). Post-opt LLIR bodies identical; the ONLY
difference in the whole module is the `noalias` token on the define. Both stock pins agree — not an
LLVM-version artifact. The corrected all-args set (and even the maximal all-5 superset) is a pure
no-op: 0 instructions change.

## FFM correctness (AFTER + ALL5 variants, M=2 N=32 has_res, gfx1250)
BASELINE / AFTER / ALL5 all `RESULT: PASS`: OUT max abs diff 0.0, OUTRES max abs diff 0.0,
rel_err 0.0, cosine 0.9999998. The contract is a pure hint; here it changes neither codegen nor
numerics.

## Counter-experiment (would refute "no delta")
Exhibit a raw uniform read-only `gl.load` of an arg (a decode-style hidden metadata/index load)
whose lowering flips under noalias. Result: NONE exists — TTGIR 0 `tt.load`, LLIR 0 `load
addrspace(1)`, MIR 0 GLOBAL_LOAD, asm 0 `global_load`. The maximal all-5-pointer annotation still
produces the identical md5. The positive control (F1/S4, S10b) confirms the toggle is live on a
kernel that DOES have a raw uniform load; the null here is specific to F5's all-TDM data path.

## Why the S10b blind spot does NOT apply to F5 (the crux of this re-audit)
Decode's false negative = the investigation annotated only the index pointer and missed the
UPSTREAM uniform read-only METADATA loads (ExptData/ExptHist/ExptOffs) that were `global_load` into
VGPR. RA-F5 explicitly enumerated ALL uniform read-only pointer args and traced every one through
uniformity + SIFixSGPRCopies. F5's difference from decode is categorical, not one-of-degree: decode
consumes its metadata pointers via raw `gl.load`; F5 consumes ALL its pointers via
`make_tensor_descriptor` only. No F5 pointer is ever dereferenced by a raw load — so there is no
uniform load to leave in VGPR, no descriptor operand to VALU-ize, and thus no round-trip for either
the frontend contract or the MachineLICM backstop to remove. Condition #1 of the §1 rule (a
raw-pointer load must exist) fails; conditions #2–#4 are never reached.

## Remaining unknowns / hand-off
- Tested M=2,N=32 (BLOCK_SIZE_M=1). The null is structural (single TDM copy per tensor; no raw
  load), so larger N cannot introduce a raw-pointer load noalias could affect — same as S13.
- The non-gluon `_triton_fused_rms_kernel` fallback (non-gfx1250 arches, raw tl.load/tl.store) is
  out of scope for this gfx1250 cell; it is the only place an F5-family noalias delta could exist
  on other targets. Flagged, not tested.
