STATUS: done
CONCLUSION: VERDICT CONFIRMED (no-op) for F3 GEMM under the corrected all-uniform-read-only-args protocol. Annotating EVERY read-only pointer arg (a16w16: a_ptr,b_ptr,bias_ptr; mxfp4: a_fp4_ptr,b_preshuf_ptr,a_scale_ptr,b_scale_ptr) changes ZERO codegen: in-loop v_readfirstlane 0->0 in all three kernels, s_load/global_load histograms identical, and the code section (pre-.amdgpu_metadata) is md5-identical BASE vs AFTER (a16w16-bw d0a148dc, a16w16-cb 505f972d, mxfp4 99adf82d — each unchanged). The decode false-negative mechanism CANNOT reproduce here: (1) mxfp4 has ZERO raw addrspace(1) loads — every global access incl. BOTH scale tensors (the decode metadata-analog) is TDM tensor.load.to.lds, so the "hidden raw uniform load" condition genuinely does not exist; (2) a16w16's ONLY raw gl.load is bias_ptr, and uniformity analysis proves its address AND the load are DIVERGENT (per-lane `<4 x float>` gather via `pid_n*BLOCK_N + arange(0,BLOCK_N)`) — noalias cannot scalarize a divergent load (AMDGPUAnnotateUniformValues bails at the uniformity leg before any noclobber check); (3) SIFixSGPRCopies inserts 0 V_READFIRSTLANE in all F3 kernels at BOTH base and after — there is NO VGPR->SGPR descriptor round-trip because the TDM descriptors are built from the SGPR-resident kernel-arg base (ptrtoint) + uniform scalar strides/pids, never routed through a raw VGPR-resident load like decode's ExptData/ExptHist/ExptOffs. S11's verdict (A/B via TDM ptrtoint inert; C no firing site) survives the re-audit, and is extended: the scales are also TDM (same ptrtoint inertness), and the one raw load (bias) fails condition #2 (divergent), not merely condition #1/#4.

# RA-F3 — F3 GEMM re-audit under the corrected "all uniform read-only args" protocol

Supersedes/confirms the F3 verdict of ledgers/S11.md (§4 CONCLUSION.md row for F3). Does NOT
edit REGISTRY/CONCLUSION or other ledgers.

## Why this re-audit exists
S10b showed the investigation-wide blind spot: only the single index pointer per kernel was
annotated, so decode's 8 in-loop v_readfirstlane (driven by UPSTREAM raw uniform read-only
metadata loads ExptData/ExptHist/ExptOffs staying VGPR) were a false negative. This cell
re-tests F3 GEMM under the corrected protocol: enumerate and annotate EVERY uniform read-only
pointer arg (index AND metadata AND primaries AND scales), then A/not-A diff the gfx1250 asm.

## Step 1 — enumeration of uniform read-only POINTER args (the corrected scope)
Scalars (M,N,K,strides,pids) are already SGPR scalar kernel args, not pointers — irrelevant.
Pointer args and their fate:

### gemm_a16w16 (both bandwidth_bound and compute_bound): args a_ptr,b_ptr,c_ptr,bias_ptr
- **a_ptr** — read-only, uniform base. Consumed by TDM make_tensor_descriptor -> async_load
  (tensor.load.to.lds). ptrtoint into i32 descriptor -> noalias inert (S11 Fact 2).
- **b_ptr** — same as a_ptr (TDM).
- **bias_ptr** — read-only, uniform base, but the ONLY raw `gl.load` in the kernel
  (line 360 bandwidth_bound / 799 compute_bound), gated on ADD_BIAS. Address is
  `pid_n*BLOCK_N + gl.arange(0,BLOCK_N)` -> PER-LANE DIVERGENT (see HOW Fact A).
- **c_ptr** — WRITTEN. bandwidth_bound: buffer_store (noalias propagates through
  make.buffer.rsrc but NO firing site, S11 Fact 1/3). compute_bound: TDM async_store (ptrtoint
  inert). Not read-only; excluded from the read-only set but its write-path was re-audited too.
- **ANNOTATED SET (a16w16):** ["a_ptr","b_ptr","bias_ptr"] (all read-only pointers).

### gemm_mxfp4 (gemm_mxfp4_preshuffle_gfx1250): args a_fp4_ptr,b_preshuf_ptr,c_ptr,a_scale_ptr,b_scale_ptr
- **a_fp4_ptr, b_preshuf_ptr** — read-only, uniform. TDM async_load. ptrtoint inert.
- **a_scale_ptr, b_scale_ptr** — read-only, uniform. THE decode-metadata analog (scales are the
  "metadata"). BUT they go through TDM make_tensor_descriptor -> async_load, NOT raw gl.load.
  ptrtoint inert. (This is the key finding: the closest thing to decode's metadata is TDM here.)
- **c_ptr** — WRITTEN via TDM async_store (ptrtoint inert).
- **ANNOTATED SET (mxfp4):** ["a_fp4_ptr","b_preshuf_ptr","a_scale_ptr","b_scale_ptr"] (all RO ptrs).

NO other raw global memory ops exist: `grep` of both kernels shows every global access is
TDM async_load/async_store or (a16w16 bw) buffer_store, plus the single a16w16 bias gl.load.
All `.load(layout=...)` calls are `smem_*.index().load()` = addrspace(3) LDS reads, not global.

## Frozen experiment
- **AITer tree:** /root/aiter, F3 kernels
  aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/{gemm_a16w16.py,gemm_mxfp4.py}. The ONLY
  delta per A/not-A pair is the noalias_args set on the @gluon.jit decorator. Tree reverted +
  verified clean after (git diff empty, no residual noalias_args). No commits. No R3 fix needed
  (both F3 kernels compiled as-is).
- **Triton:** installed triton resolving to /root/triton @ the #120-contract branch. No rebuild
  (noalias_args is a Python annotation toggle).
- **Target/mode:** gfx1250 via FFM-lite (/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06/ffmlite_env.sh;
  arch_info.get_arch()->gfx1250). FFM = correctness only. Compilation driven end-to-end by FFM
  (torch device tensors), so this is real gfx1250 codegen, not hand-written IR (unlike S11 which
  had to model the IR because FFM was unavailable then — this cell UPGRADES S11 to real asm).
- **Isolation:** per-variant TRITON_CACHE_DIR=/tmp/tc-RA-F3-{base,after,cb-base,cb-after,mxfp4-base,
  mxfp4-after}, TRITON_ALWAYS_COMPILE=1. Artifacts .llir/.amdgcn under the hash subdir.
- **Stock llc pin (mechanism trace):** /root/.triton/llvm/llvm-62b7cf96-ubuntu-x64-2/bin/{opt,llc},
  -mcpu=gfx1250. STOCK (no -amdgpu-hoist-uniform-readfirstlane).
- **Shapes (smallest exercising):** a16w16 bandwidth_bound M=N=256,K=512 (8 k-tiles), bf16, +bias;
  a16w16 compute_bound M=N=512,K=4096,BLOCK=128,NB=3 (forces the compute_bound variant, +bias);
  mxfp4 M=N=32,K=256 (N%32==0,K%256==0,M>=32), shuffle_scales+shuffle_weight, via the test's
  generate_gemm_afp4wfp4_inputs. Drivers: /tmp/ra_f3/drive_{a16w16,cb2,mxfp4}.py.
- **Metric:** in-loop v_readfirstlane (awk loop-body counter), s_load/global_load/rfl histogram,
  and code-section md5 (everything before .amdgpu_metadata — excludes the descriptor YAML which
  legitimately gains `.actual_access: read_only` under noalias but is not code).

## HOW-facts (real asm + uniformity + si-fix-sgpr-copies trace)

### Fact A — a16w16 bias is a genuinely DIVERGENT load (condition #2 fails, not #1/#4)
`opt -passes='print<uniformity>'` on the real llir (base):
```
DIVERGENT:   %924 = or disjoint i32 %23, %923            ; pid_n*BLOCK_N + arange lane id
DIVERGENT:   %928 = sext i32 %924 to i64
DIVERGENT:   %929 = getelementptr [4 x i8], ptr addrspace(1) %3, i64 %928   ; %3 = bias_ptr
DIVERGENT:   %935 = load <4 x float>, ptr addrspace(1) %929, align 16, !dbg !87  ; line 360
```
It lowers to 4x `global_load_b128 v[..], v[50:51], off` (VGPR address). AFTER (noalias on
bias_ptr): the load `%935..%947` is STILL marked DIVERGENT and STILL `global_load_b128`.
noalias cannot help — AMDGPUAnnotateUniformValues.visitLoadInst returns at line 69
`if (UA->isDivergentAtDef(Ptr)) return;` before the noclobber leg. This is the fundamentally
different condition from decode (whose metadata loads were UNIFORM but stranded in VGPR).

### Fact B — mxfp4 has ZERO raw uniform loads; scales are TDM (condition #1 fails)
Base llir: `grep -c 'load .*addrspace(1)'` = **0**; `tensor.load.to.lds/store.from.lds` = 11.
Both a_scale_ptr and b_scale_ptr flow through make_tensor_descriptor->async_load->
tensor.load.to.lds (ptrtoint base). There is no raw uniform load for noalias to scalarize.
The decode false-negative required a raw VGPR-resident uniform metadata load; mxfp4 has none.

### Fact C — SIFixSGPRCopies inserts 0 round-trips (no decode-style V2S anchor)
`llc -stop-after=si-fix-sgpr-copies` on the real llir:
```
mxfp4   base: V_READFIRSTLANE = 0   after: 0
a16w16  base: V_READFIRSTLANE = 0 (TENSOR_LOAD_TO_LDS = 12, all scalar)  after: 0
```
No scalar TDM descriptor is being VALU-ized into VGPR + readfirstlane. The descriptors are
scalar from the start: built from the SGPR kernel-arg base (ptrtoint of an SGPR) + uniform
scalar strides/pids, never through a VGPR-resident load. This is the exact contrast with decode,
where the descriptor's row-index operand was fed by VGPR-resident ExptData/etc., forcing the
8 round-trips.

### Fact D — descriptor build has no VGPR-fed round-trip (the S11 TDM re-audit)
Confirmed via Fact C (0 V_READFIRSTLANE post-SIFixSGPRCopies) that the TDM descriptor's SGPR
operands are never sourced from a VGPR uniform load. S11's ptrtoint-inertness claim holds AND
there is no upstream VGPR excursion feeding the descriptor.

## A/not-A result (the kill switch — only delta is the noalias_args set)

| kernel | BASE noalias on define | AFTER noalias | in-loop rfl B->A | s_load B->A | global_load B->A | code-section md5 B / A |
|---|---|---|---|---|---|---|
| a16w16 bandwidth_bound | 0 | 3 (a,b,bias) | 0 -> 0 | 99 -> 99 | 4 -> 4 | d0a148dc / d0a148dc (IDENTICAL) |
| a16w16 compute_bound   | 0 | 3 (a,b,bias) | 0 -> 0 | 387 -> 387 | 8 -> 8 | 505f972d / 505f972d (IDENTICAL) |
| mxfp4 preshuffle       | 0 | 4 (a,b,as,bs)| 0 -> 0 | 44 -> 44  | 0 -> 0 | 99adf82d / 99adf82d (IDENTICAL) |

The ONLY whole-file diff (a16w16) is the kernel-arg metadata YAML gaining
`.actual_access: read_only` — a descriptor annotation, zero code impact. Code section identical.

## FFM correctness (AFTER variant, smallest exercising shape)
- a16w16 bandwidth_bound AFTER: rel_err 0.002336 PASS.
- a16w16 compute_bound AFTER: rel_err 0.003472 PASS (BASE 0.001445 PASS; both fine, RNG jitter).
- mxfp4 AFTER: rel_err 0.000000 PASS (BASE 0.000000). Since codegen is byte-identical this is a
  null/no-op correctness check, as expected.

## Counter-experiment (necessary refuters, all failed to refute)
1. "The corrected all-args set changes F3 codegen" -> code-section md5 identical for all 3
   kernels. Not refuted.
2. "mxfp4 scales are the hidden raw uniform load decode missed" -> 0 raw addrspace(1) loads;
   scales are TDM. Not refuted (they are structurally TDM, not raw).
3. "a16w16 bias is uniform-but-VGPR-stranded (fixable like decode metadata)" -> uniformity
   analysis says DIVERGENT at def, load stays DIVERGENT/global_load with noalias on. Not refuted.
4. "A scalar TDM descriptor is VALU-ized + readfirstlane'd (decode round-trip)" ->
   SIFixSGPRCopies inserts 0 V_READFIRSTLANE, base and after, both kernels. Not refuted.

## CLAIM STATUS
CONFIRMED no-op for F3 GEMM under the corrected protocol. The specific conditions that fail:
- **mxfp4:** genuinely NO raw uniform load (condition #1) — all globals incl. both scales are TDM
  (ptrtoint inert); no VGPR->SGPR descriptor round-trip (SIFixSGPRCopies = 0).
- **a16w16 bandwidth_bound:** A/B TDM (ptrtoint inert); C buffer_store already s_load-clean with
  no firing site; the one raw load (bias) is TRULY DIVERGENT (condition #2), not VGPR-stranded.
- **a16w16 compute_bound:** A/B/C all TDM (ptrtoint inert); bias divergent as above.
This is the same mechanism as S11, now proven on REAL gfx1250 asm + uniformity + SIFixSGPRCopies
rather than modeled IR. Decode's false-negative pattern does not reproduce because F3 has no raw
VGPR-resident uniform load feeding a descriptor.

## Remaining unknowns / hand-offs
- The hypothetical Fact-1 firing site for C (a fused read-modify-write / C-reload epilogue) is
  still absent in the current F3 kernels; gemm_a16w16_atomic (a separate kernel, not this cell's
  pair) would be the candidate if a written-operand win is ever wanted. Not in scope.
- Recovering an A/B/scale benefit would require carrying the alias promise onto
  tensor.load.to.lds (keep a pointer operand or attach !alias.scope) — a backend design change,
  not an arg-attr toggle. Design note, not a current-tree fact.
