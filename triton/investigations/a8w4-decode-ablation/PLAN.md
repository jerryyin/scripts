# gfx1250 A8W4 MoE decode kernel ideas — correctness-first ablation plan

## Thesis

Basically, the current Triton PR kernel and AITER's A8W4 kernels are expert-grouped scaled-WMMA designs, while TokenSpeed's tiny-decode kernel and Cursor's Warp Decode reorganize work around routes or final outputs and stream weights through wave-local reductions. This study turns those implementation ideas into separate source kernels, one idea per parent→child edge, and proves only correctness under FFM in round 1. It does not measure or rank performance yet.

The purpose of round 1 is to leave a directory of independently runnable, FFM-correct kernels. Later rounds can profile them without reconstructing which source changes were combined.

## Frozen source state

| Source | State | Role |
|---|---|---|
| Triton target | `/root/triton-mi450` at PR #154 head `80e223f93d59359161f5482fbb69bbfab29c0a0b` | Baseline and destination |
| AITER | `/root/aiter` at `4a1cc773f34cbfc74387259e51262556ee38edd0` | Small-M grouped WMMA, persistent-N, packed routing, preshuffled weights, and static/per-32 dynamic FP8 epilogues |
| TokenSpeed | `/root/tokenspeed` at `3e725ac2b785b71f27ff9e9ac3796349c495d225` | Concrete route-direct wave-GEMV and direct-MFMA decode families, medium grouped decode, fused routing, and atomic combine ideas |
| Conceptual writeup | [Cursor Warp Decode](https://cursor.com/blog/warp-decode) | Architectural checklist for output-owned decode: no expert grouping, no staging buffers, one warp per output, top-k folded into W2 |

PR #154's head was resolved from `refs/pull/154/head` on 2026-07-29. The former `912658ba227ff1b9017d176804841bebe5b2e747` snapshot is superseded by a non-fast-forward branch rewrite; do not treat equal commit subjects as equal source states.

The target body is now [`_matmul_decode`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py:17), while selection, argument construction, reference code, and tests remain in [`moe_gfx1250.py`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:1594). The body fixes M-ragged routing, split-K=1, and the baseline schedule ([invariants](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py:39)); it still instantiates the shared [`MoEPipelinedProgram`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_common_gfx1250.py:509), whose hot loop stages X/W/scales through LDS and issues scaled WMMA ([pipeline](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_common_gfx1250.py:620), [scaled WMMA](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_common_gfx1250.py:381)). Therefore this revision changes the baseline source boundary and accepted test surface, not the study's grouped-versus-route-direct execution-family split.

Before `H0`, resolve `refs/pull/154/head` again and require it to equal the pinned SHA above and `git rev-parse HEAD`. If the PR moves, stop and re-audit its patch before creating any kernel; do not silently execute this plan against a different head.

## Scope

- Target gfx1250 A8W4 MoE decode correctness.
- Put all temporary kernel modules under `/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/`.
- Preserve `moe_decode_gfx1250.py`, `moe_common_gfx1250.py`, and `moe_gfx1250.py` as read-only source-of-truth files.
- Give every implementation cell one source file. Never overwrite or repurpose a completed variant.
- Treat [`KERNEL_COVERAGE.md`](/root/scripts/triton/investigations/a8w4-decode-ablation/KERNEL_COVERAGE.md) as the completeness manifest. A source kernel may not influence the study without a linked active row, performance-only disposition, explicit rejection, or evidence-only classification.
- Use FFM for compilation and numerical correctness. FFM provides no timing, so round 1 must not make performance claims.
- Keep the first round source-level. Dispatch-policy tuning, AM timing, B0 profiling, cleanup, and upstream-quality refactoring are later work.
- Do not commit or push.

## Two semantic families

The study must not hide an apples-to-oranges comparison.

### Grouped-WMMA family

`H0` establishes a configuration-covered baseline, and `G2–G6` preserve its contract: expert-grouped ragged rows enter one A8W4 GEMM and one routed row leaves it. `G7` retains grouped compute ownership but changes the W2 epilogue to atomic token accumulation; `G8` retains grouped ownership but changes the W1 numerical boundary to E4M3 plus per-32 UE8M0 scales. Both ledgers must state their changed output contracts.

```text
expert block × N tile -> TDM X/W/scales -> LDS -> scaled WMMA -> routed row
```

### Route-direct family

These variants intentionally change ownership and, for some cells, the input/output boundary. They are implementation prototypes, not direct drop-in performance comparisons with the grouped GEMM.

```text
route × output tile -> stream selected W -> wave reduction or duplicated-row scaled MFMA -> route result
token × output tile -> loop top-k -> wave reduction or scaled MFMA -> weighted FP32 sum -> final token result
```

Every route-direct ledger must state its different contract before discussing results.

### Routing and integration rows

`Q1–Q6` make the decode routing kernels first-class correctness experiments: standalone one-workgroup top-k, general precomputed top-k grouping, the structurally simpler M=1 flat route, and fused router-plus-grouping variants. `I1–I2` then prove that the independently tested W1 boundary and output-owned W2 ideas compose into complete two-stage decode paths. These rows do not claim another independent optimization; they close correctness boundaries that must not be postponed to performance work.

## One-change discipline

1. `H0` creates an execution-equivalent self-contained baseline and the shared correctness runner. `k00_pr154_baseline.py` mechanically snapshots the PR's decode body plus the decode-reachable layout/configuration, descriptor, TDM/LDS pipeline, and scaled-WMMA helpers from `moe_common_gfx1250.py`; `run_ffm.py` reproduces the current launch and reference contract from `moe_gfx1250.py`. The test suite—not a command-line tile override—selects a sparse set of joint configurations that gives every decode-supported tuning axis multiple correctness values.
2. Every later source file begins from the parent named in `REGISTRY.md`.
3. One parent→child edge represents one implementation idea. A child may require many mechanical edits to realize that idea, but it may not smuggle in a second optimization.
4. Retain every parent file. Never create a cumulative “latest” kernel.
5. Put this header near the top of every kernel module:

```python
ABLATION_ID = "<cell-id>"
ABLATION_PARENT = "<parent file>"
ABLATION_IDEA = "<one sentence>"
ABLATION_PROVENANCE = "<exact source file:line or writeup>"
```

6. Before FFM, save `diff -u <parent> <child>` in the ledger and classify every hunk as essential to the one idea or mechanical support for it.
7. Do not modify the PR's shared helpers. Later variants change only their self-contained child copy; if a variant needs contract-specific reference code, keep it inside that variant's owned module.
8. If correctness requires another independent idea, stop and propose a new cell/edge. Do not fold it into the current file.
9. Tuning parameters are not source ideas. Every cell's tests must exercise at least two values of each tuning axis that the cell exposes unless the implementation intrinsically restricts that axis; record and test any such restriction explicitly. Use sparse joint configurations, not a Cartesian tuning sweep.

Dependency chains are allowed when an idea cannot exist alone. For example, route-direct gate/up fusion builds on a correct route-direct single projection, but both source files remain available and the `R1→R2` diff contains only gate/up fusion.

## Source idea catalog

| Idea | Observed source | Mechanism being isolated | Cell |
|---|---|---|---|
| Decode-only baseline specialization and representative configuration coverage | Triton PR #154 places the body in `moe_decode_gfx1250.py`, fixes M-ragged, split-K=1, a single `(1,1,1)` subtile, and the ordinary pipeline ([body and invariants](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py:17), [configuration](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py:65), [pipeline selection](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py:129)). Its general test bundles multiple block, warp, and buffer choices in `Case` objects and separately crosses scale preshuffling, while its tiny-M test fixes one default configuration ([case builder](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:1751), [scale-layout parameter](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:1808), [tiny-M test](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:2012)). | Establish one parameterized source control and a sparse correctness suite in which every supported tuning axis has multiple values; no tuning or separate source change | H0 |
| Joint configuration policy | AITER derives `BLOCK_M` from average routed rows per expert and dispatches its dedicated decode body only at 16 ([heuristic](/root/aiter/aiter/ops/triton/moe/moe_routing/routing.py:303), [dispatch](/root/aiter/aiter/ops/triton/moe/moe_op_gemm_a8w4.py:526)); the target test instead enumerates selected joint configurations ([cases](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:1777)). | Performance search and runtime policy across tile sizes, warps, buffers, and layouts—not a new kernel implementation | P5 |
| Decode-specific WMMA warp layout | AITER decode uses `warp_bases=[[0,1],[0,2]]` ([source](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:1007)) | Change warp ownership inside a 16-row tile without changing routing or pipeline | G2 |
| Persist across three N tiles | AITER decodes routing once and walks `N_ITERS` consecutive N tiles ([ownership](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:152), [loop](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:454)) | Amortize expert/block decode and activation setup across N | G3 |
| Packed expert/block schedule | AITER loads one packed `ExptData` entry and unpacks expert and block IDs ([source](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:865)) | Replace the baseline's general ragged offset path with one decode-specific schedule load | G4 |
| Preshuffled packed weights | AITER supports a preshuffled packed-N weight descriptor and unshuffles in LDS ([descriptor](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:190), [local transform](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:582)) | Change global weight layout/coalescing while preserving scaled WMMA | G5 |
| Single-buffer direct-load grouped decode | TokenSpeed's gfx950 medium-decode specialization bypasses the generic TDM pipeline, loads X/W/scales explicitly, and builds one LDS tile for M=8/16 ([body](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:4671), [selection](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:5079)) | Test a decode-specific transport body without importing the prefill pipeline | G6 |
| Grouped-WMMA atomic combine | TokenSpeed's BF16 decode W2 keeps expert-grouped MFMA but atomically adds route-weighted FP32 results directly to the token output ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/stage2_decode_kernel.py:48), [atomic epilogue](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/stage2_decode_kernel.py:170)) | Remove per-route W2 scratch and the combine launch without changing grouped compute ownership | G7 |
| Dynamic per-32 MXFP8 intermediate emit | AITER's Triton A8W4 epilogue derives one UE8M0 scale per 32 output columns and writes E4M3 values plus scales in the GEMM1 store path ([kernel contract](/root/aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py:166), [epilogue](/root/aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py:432)) | Add dynamic group quantization to a grouped output tile that already owns complete 32-column groups, avoiding a hidden ownership change | G8 |
| One-workgroup softmax top-k | TokenSpeed computes rowwise softmax, stable top-k IDs, selected softmax gates, optional renormalization, and routing scale in one Gluon workgroup ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:129)) | Replace a generic multi-launch router with one bounded tiny-M routing kernel while preserving the direct-route IDs/weights contract | Q1 |
| One-workgroup sigmoid+bias top-k | TokenSpeed computes sigmoid scores, rounds at the router dtype boundary, adds correction bias for selection, and emits IDs plus un-biased routing gates ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:219)) | Isolate the biased-routing numerical contract from the softmax router rather than hiding it in one polymorphic cell | Q2 |
| Precomputed top-k grouped materialization | TokenSpeed's live fixed-one-wave kernel consumes existing top-k IDs/weights and produces expert histograms, prefix offsets, block schedules, stable per-expert ranks, gather/scatter indices, and gates ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10915), [live wrapper](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:11003), [one-wave launch](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:11062)) | Collapse the post-top-k sorting/metadata chain without changing how top-k is computed | Q3 |
| Flat M=1 precomputed route | TokenSpeed's standalone launchable kernel observes that one token's top-k expert IDs are unique, keeps route rows in top-k slot order, writes slice sizes/offsets and the compact expert schedule directly, and omits the general histogram/prefix/rank path ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:9722), [currently unreachable caller branch](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:8037)). At the pinned revision, the containing helper requires `M>=4` while this branch requires `M==1`, so it is an explicit structural source rather than a live-selected path. | Isolate the structural removal of general grouping work for exactly one token; this is not merely a one-wave tuning choice | Q6 |
| Fused softmax top-k plus grouped materialization | TokenSpeed keeps top-k results in registers and immediately constructs histograms, stable per-expert ranks, packed schedules, gather/scatter indices, and gates ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:9385)) | Remove the materialized IDs/weights boundary between `Q1` and `Q3` | Q4 |
| Fused biased grouped top-k plus grouped materialization | TokenSpeed applies grouped biased selection and produces the same grouped metadata in one workgroup ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:9508)) | Remove the materialized IDs/weights boundary between `Q2` and `Q3` under the biased grouped-routing contract | Q5 |
| Route-direct wave GEMV | TokenSpeed explicitly avoids sorting/padding and grouped GEMM, maps `(route, output tile)` directly, streams the selected expert's packed weights, uses `scaled_upcast`, performs elementwise multiply rather than MFMA/WMMA, and reduces K with `gl.sum` ([design](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:21), [mapping](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:98), [stream/reduce](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:150)); Cursor independently describes one warp owning an output and streaming weights | Atomic algorithmic fork for tiny decode: no expert grouping, no padded M tile, and no matrix instruction | R1 |
| Gate/up from one activation stream | TokenSpeed loads each activation fragment once and updates both gate and up accumulators ([source](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:147)); Cursor describes each warp reading the gate/up rows for one intermediate neuron without shared staging | Reuse X immediately across gate and up without introducing LDS handoffs | R2 |
| BF16 intermediate, no FP8 requantization | TokenSpeed applies activation and writes a BF16 intermediate ([source](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:223)); Cursor identifies the BF16 intermediate as the way to remove MXFP8 requantization between W1 and W2 | Isolate the simpler decode-stage boundary, intentionally making W2 A16W4 rather than strict A8W4 | R3 |
| Static FP8 intermediate emit | AITER's A8W4 Gluon epilogue quantizes before writeback when `Quant_static_scale` is present ([source](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:1318)) | Preserve a strict A8W4 W2 contract as the sibling alternative to `R3` | R4 |
| Output-owned top-k down projection | TokenSpeed maps `(token, output tile)`, loops top-k, accumulates routing weights in FP32, and stores once ([mapping](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:273), [loop](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:297), [store](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:356)); Cursor describes the same ownership and the resulting removal of per-route outputs and combine buffers | Fold top-k weighting into W2 so each output has one owner and one final store | R5 |
| Route-direct scaled MFMA | TokenSpeed's gfx950 direct decode reads original top-k IDs but duplicates one real route across an MFMA M layout and stores only the real row ([A4W4 stage 1](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:915), [single-row store](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:1041)); its A8W4 cooperative decode uses the same direct-route matrix family ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10121)) | Retain direct routing but substitute scaled MFMA for the wave reduction | R6 |
| Output-owned top-k scaled MFMA | TokenSpeed's gfx950 A8W4 W2 kernel maps `(token, output tile)`, loops top-k, and performs direct scaled MFMA before one output store ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10403), [top-k loop](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10489)) | Test the same `R5` ownership with a matrix primitive instead of wave GEMV | R7 |
| Register-lookahead K pipeline | The same TokenSpeed W2 kernel loads the next pair of K tiles before MFMA-ing the current pair, without an LDS ping-pong pipeline ([source](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10512)) | Isolate decode-specific direct-MFMA load/compute overlap after `R7` is correct | R8 |
| Recompute top-k inside direct-MFMA W1 | TokenSpeed's cooperative A8W4 W1 maps work by `(token, N tile, slot)`, computes top-k from logits inside every program, writes IDs/weights once, and immediately feeds the selected expert into the stage-1 compute body ([kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10121), [routing block](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10173)) | Remove the precomputed-route input from `R6` and expose the cost/benefit of replicated in-program routing as a correctness-valid variant | R9 |
| Complete BF16-boundary wave-decode path | `R3` supplies the W1 stage boundary and `R5` supplies output-owned W2 | Add the already-isolated W2 ownership change to the BF16 W1 parent and prove the full two-stage contract | I1 |
| Complete strict-A8 wave-decode path | `R4` supplies the W1 stage boundary and `R5` supplies output-owned W2 | Add the already-isolated W2 ownership change to the static-FP8 W1 parent and prove the full two-stage A8W4 contract | I2 |

TokenSpeed's idea corpus spans gfx950 A4W4, A8W4, A16W4, and BF16. Its A8W4 direct-MFMA kernels are precision-matched but not architecture-matched; its A16W4/BF16 wave and atomic kernels are ownership/epilogue sources rather than drop-in counterparts. Cursor's writeup targets NVIDIA B200 MXFP8. AITER's gfx1250 attention source proves that gfx1250 exposes packed-FP4 `scaled_upcast` ([source](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py:1489)); it does not prove that the exact A8 activation/scale form needed by `R1` is accepted.

### Structural coverage gate

Before admitting cells, scan every source along six axes: program ownership, routing/grouping, math primitive, weight/activation movement, intermediate/final buffers, and numerical boundary. Every discovered decode mechanism must be admitted or merged with an active row. Deferral is allowed only for a performance schedule or crossover whose functional dataflow is already represented by a correct active row; an out-of-scope mechanism must be explicitly rejected instead.

Path labels below are relative to the named checkout, while the links open the pinned local file. A file appears in more than one row when distinct kernel families in that file contribute different ideas. [`KERNEL_COVERAGE.md`](/root/scripts/triton/investigations/a8w4-decode-ablation/KERNEL_COVERAGE.md) is the function-level inventory inside these files.

| Source | Repository-relative path(s) | Decode-relevant mechanisms found | Round-1 disposition |
|---|---|---|---|
| Triton PR #154 | [`third_party/amd/python/examples/gluon/moe_decode_gfx1250.py`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py)<br>[`third_party/amd/python/examples/gluon/moe_common_gfx1250.py`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_common_gfx1250.py)<br>[`third_party/amd/python/examples/gluon/moe_gfx1250.py`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py) | Dedicated decode body and frozen invariants; shared TDM/LDS scaled-WMMA hot loop; launch/reference surface and explicit `M=1,2,4,8,16` tests | `H0`; generic-parent schedule alternatives are performance-only `P1` |
| AITER gfx1250 Gluon A8W4 compute and selection | [`aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py`](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py)<br>[`aiter/ops/triton/moe/moe_op_gemm_a8w4.py`](/root/aiter/aiter/ops/triton/moe/moe_op_gemm_a8w4.py)<br>[`aiter/ops/triton/moe/moe_routing/routing.py`](/root/aiter/aiter/ops/triton/moe/moe_routing/routing.py) | BM16 fallback, decode WMMA layout, persistent N walk, packed expert schedule, preshuffled weights, static FP8 epilogue, and route-density-derived M tile | `H0`, `G2–G5`, `R4`; the M-tile formula is performance-policy evidence `P5` |
| AITER combine, Triton sibling, and gfx1250 primitive evidence | [`aiter/ops/triton/_gluon_kernels/gfx1250/moe/reduce.py`](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/reduce.py)<br>[`aiter/ops/triton/_triton_kernels/moe/reduce.py`](/root/aiter/aiter/ops/triton/_triton_kernels/moe/reduce.py)<br>[`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`](/root/aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py)<br>[`aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py`](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py) | Separate FP32 route combine, per-32 dynamic MXFP8 emit, and packed-FP4 `scaled_upcast` availability | `G7–G8`, `R5`; attention kernel is evidence-only `S1` |
| TokenSpeed gfx1250 grouped A8W4 | [`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx1250.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx1250.py) | Same grouped ownership; baseline/slice-K/slice-NK schedule family | Baseline reinforces `H0`; schedule variants are performance-only `P1` |
| TokenSpeed gfx950 grouped and direct A8W4 | [`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py) | M=8/16 single-buffer grouped body, direct-route scaled-MFMA W1, output-owned W2, register-lookahead K overlap, routing, quantization, and partial-reduce helpers | `G6`, `Q3–Q6`, `R4`, `R6–R9`; split-K/reduction is `P2`, the shadowed variable-wave route is `P3`, and A4-only preprocessing is rejected as `X2` |
| TokenSpeed gfx950 A16W4 route-direct decode | [`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py)<br>[`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py`](/root/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py) | Direct top-k IDs, no expert grouping/padding, wave GEMV instead of MFMA/WMMA, streamed selected-expert weights, gate/up reuse, BF16 intermediate, output-owned top-k combine, and the `M<=4` dispatch guard | `R1–R3`, `R5`; exact grouped-versus-route-direct selection is performance-only `P4`, and the wrapper is evidence-only `S2` |
| TokenSpeed gfx950 A4W4 direct-MFMA and historical wave decode | [`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py)<br>[`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_stage1.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_stage1.py)<br>[`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_stage2.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_stage2.py)<br>[`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/moe.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/moe.py) | Direct top-k ownership with scaled MFMA, output-owned W2, register-lookahead K overlap, historical scalar-FP4 wave GEMV, top-k routing, and package dispatch/re-exports | `Q1–Q2`, `R6–R8`; reinforces `R1–R5`; dispatch/re-exports are evidence-only `S2` |
| TokenSpeed gfx950 BF16 decode siblings | [`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/warp_decode_gluon_kernel.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/warp_decode_gluon_kernel.py)<br>[`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/stage2_decode_kernel.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/stage2_decode_kernel.py) | Independent wave-decode confirmation and grouped route-weighted FP32 atomic combine | Reinforces `R1–R3`, `R5`; atomic combine supplies `G7` |
| Cursor Warp Decode | [`cursor.com/blog/warp-decode`](https://cursor.com/blog/warp-decode) (external writeup; no repository-relative path) | Output ownership, no expert sorting/padding, one warp per output, no shared staging, BF16 stage boundary, top-k folded into W2 | Merged into the concrete TokenSpeed-backed `R1–R3`, `R5` cells |

Round 1 admits every attributable mechanism that changes `M<=4` decode dataflow, ownership, fusion, or numerical boundaries and can be proven correct under FFM. Alternative schedules that preserve those semantics remain performance-only work; prefill-only or target-precision-mismatched mechanisms are rejected rather than deferred.

### Cursor checklist → concrete cells

The Cursor writeup and TokenSpeed source are not separate competing kernels here: Cursor supplies the architectural explanation, while TokenSpeed supplies a close Gluon realization. The plan must make every decode idea visible instead of hiding that overlap inside “route-direct.”

| Cursor idea | Concrete source ablation | What is intentionally absent |
|---|---|---|
| Organize work by outputs rather than grouped experts; one warp computes one output by streaming weights | `R1` | Expert sorting, M padding, LDS-staged matrix tiles, MFMA/WMMA |
| W1 warp consumes one activation stream for both gate and up without shared staging | `R2` | A second activation traversal or a producer/consumer handoff |
| Keep the activated intermediate in BF16 and remove intermediate MXFP8 quantization | `R3` | FP8 quantize/dequantize work; `R4` is the strict-A8 sibling control |
| W2 output owner loops over top-k routes, applies routing weights in FP32, and stores once | `R5` | Per-route W2 output, scatter/combine buffer, separate top-k reduction |

## Performance-only deferred work

- **P1 — Matrix compute schedules:** The latest PR body statically rejects slice-K, slice-NK, slice-MNK, ping-pong, L2 prefetch, partial TDM, TDM split, and partition-conflict resolution ([guards](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py:43)). Those generic-parent choices preserve the grouped-WMMA functional dataflow and are not round-1 decode ideas; reconsider any of them only after the unsliced decode parent is correct and AM/B0 evidence identifies a concrete stall or crossover.
- **P2 — Output-owned split-K and partial reduction:** The non-split `R7` dataflow is active now. TokenSpeed's split-K form plus `_moe_partial_reduce` is an alternative performance schedule for long K that restores a partial buffer and extra launch; correctness of the underlying ownership does not depend on it.
- **P3 — Variable-wave scheduling for the general precomputed route:** Active `Q3` is TokenSpeed's live fixed-one-wave implementation, and active `Q6` captures the structurally different M=1 flat route. The earlier `_precomputed_topk_route_small_m` parameterizes its layouts by wave count, but its wrapper at line 10792 is overwritten by a later function with the same Python name at line 11003; repository search found no other live caller ([variable-wave kernel](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:9621), [shadowed wrapper](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10792), [overriding wrapper](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:11003)). Reviving one-versus-two-wave layouts would change only scheduling of `Q3` and therefore waits for AM/B0 timing.
- **P4 — Runtime family dispatch only:** TokenSpeed selects route-direct A16W4 for `0<M<=4` plus layout/alignment guards and otherwise falls back to grouped MFMA ([guard](/root/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py:144)). Correctness can validate both families but cannot choose the crossover; `M`, route density, shape, and alignment thresholds require timing.
- **P5 — Exhaustive joint-configuration search and selection:** `H0` proves representative values of every decode-supported tuning axis but deliberately does not take their Cartesian product or choose winners. AITER's average-route-density formula is one candidate for the `BLOCK_M` component and rebuilds its tile schedule from the selected value ([heuristic](/root/aiter/aiter/ops/triton/moe/moe_routing/routing.py:303), [routing metadata](/root/aiter/aiter/ops/triton/moe/moe_routing/routing.py:495)). AM/B0 work must search and select `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, warp count, buffering, and scale layout jointly.
- **AM/B0 tuning and ranking:** FFM supplies correctness only. Tile tuning, schedule selection, crossover measurement, and a “best kernel” conclusion begin after the correctness audit.

## Rejected rather than deferred

- **X1 — Prefill-only activation-scale transport:** AITER's direct `async_copy` path appears in `_moe_gemm_a8w4_prefill` and is coupled to that multi-buffer prefill pipeline. It is not a decode row and is excluded rather than held for a vague future port.
- **X2 — A4 activation preprocessing:** TokenSpeed's MXFP4 quantizers are required by A4W4 direct-MFMA variants, while this study's destination contract is A8W4. Their matrix ownership ideas are represented by `R6–R8`; changing the destination activation precision is out of scope.
- **Evidence-only sources:** Dispatch wrappers, re-export modules, and AITER's attention use of `scaled_upcast` establish selection or primitive availability but contain no additional MoE decode mechanism to ablate.

## Expected temporary source layout

`H0` creates the directory and baseline/harness. Later agents add only their owned file.

```text
/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/
├── __init__.py
├── run_ffm.py
├── k00_pr154_baseline.py
├── k02_aiter_decode_wmma_layout.py
├── k03_aiter_persistent_n3.py
├── k04_aiter_packed_schedule.py
├── k05_aiter_preshuffled_weight.py
├── k06_route_direct_wave_gemv.py
├── k07_route_direct_gate_up.py
├── k08_route_direct_gate_up_bf16.py
├── k09_route_direct_gate_up_fp8.py
├── k10_output_owned_topk.py
├── k11_grouped_direct_load.py
├── k12_grouped_atomic_combine.py
├── k13_route_direct_scaled_mfma.py
├── k14_output_owned_topk_scaled_mfma.py
├── k15_output_owned_topk_scaled_mfma_lookahead.py
├── k16_grouped_mxfp8_emit.py
├── k17_route_direct_scaled_mfma_fused_topk.py
├── k18_softmax_topk_router.py
├── k19_sigmoid_bias_topk_router.py
├── k20_precomputed_topk_group_route.py
├── k21_fused_softmax_group_route.py
├── k22_fused_biased_group_route.py
├── k23_integrated_wave_bf16.py
├── k24_integrated_wave_fp8.py
└── k25_flat_m1_group_route.py
```

Temporary duplication is intentional. Cleanup and helper extraction would destroy the source-level independence this study needs.

## Correctness contract

### Common FFM rules

- Run every variant under gfx1250 FFM through `/root/scripts/tools/run_on_model.sh --backend ffm`.
- Use a unique `TRITON_CACHE_DIR=/tmp/a8w4-decode-ablation/<cell-id>` and `TRITON_ALWAYS_COMPILE=1` so one variant cannot reuse another variant's binary.
- Serialize the simulator with `flock /data/lock/amd-gpu.lock`.
- Set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; use the existing `ffm_teardown` path or explicit `os._exit` so a passed process does not hang at interpreter teardown.
- FFM output is correctness evidence only. Ignore durations and never report a speedup.
- Reuse the baseline's dequantized torch reference and tolerance rather than inventing a looser threshold. The current MXFP4 test constructs the reference at [`test_matmul`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:1980) and sets the A8W4 tolerance immediately before [`assert_close`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:2001).
- Record output shape, dtype, finite fraction, maximum error, and the reference assertion result.
- For every tuning parameter exposed by a cell, the full suite must exercise at least two values across sparse joint configurations. A single legal value is acceptable only when the source has an intrinsic constraint; the ledger must cite and test that rejection boundary.
- Reconstruct all dependent state for each configuration. In particular, changing `BLOCK_M` requires matching ragged routing metadata and grid construction; no case may reuse metadata from another tile size.

### Grouped-family matrix

The common runner starts with the current PR's complete 20-case tiny-M default family:

| Dimension | Cases |
|---|---|
| Useful decode rows | `M = 1, 2, 4, 8, 16` |
| Route movement | gather and scatter |
| Activation format | E4M3 and E5M2 |
| Weight format | packed E2M1 + UE8M0 |
| Default shape | `N=256`, `K=512`; explicit sparse cases may override N or K |
| Expert slices | 10 |
| Default configuration | `BLOCK_M=128`, `BLOCK_N=128`, `BLOCK_K=256`, 4 warps, 2 buffers, strided scales |
| Other controls | bias enabled |

That family reproduces the source test exactly ([source test](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:2012)). H0 then adds five test-selected joint configurations rather than crossing every value with every M/routing/dtype case:

| Case | M / N / K / movement / activation | `BLOCK_M` | `BLOCK_N` | `BLOCK_K` | Warps | Buffers | Scale layout |
|---|---|---:|---:|---:|---:|---:|---|
| C1 | `1 / 256 / 512 / gather / E4M3` | 16 | 64 | 128 | 4 | 2 | strided |
| C2 | `4 / 256 / 512 / scatter / E5M2` | 64 | 128 | 256 | 8 | 2 | strided |
| C3 | `8 / 400 / 416 / gather / E5M2` | 128 | 128 | 128 | 4 | 3 | strided |
| C4 | `16 / 256 / 512 / scatter / E4M3` | 256 | 256 | 256 | 8 | 2 | preshuffled |
| C5 | `2 / 256 / 1024 / scatter / E4M3` | 128 | 128 | 256 | 4 | 4 | strided |

Together, the 25 cases cover `BLOCK_M={16,64,128,256}`, `BLOCK_N={64,128,256}`, `BLOCK_K={128,256}`, warps `{4,8}`, buffers `{2,3,4}`, and both strided and preshuffled scales. C3 also exercises non-divisible N/K tails, while C5 supplies enough K iterations to make four buffers meaningful. The values are correctness representatives, not a claim that omitted powers or combinations are invalid or slow. If one proposed joint case is structurally illegal, H0 must record the exact constraint and replace it with a case that preserves the missing axis value; it may not silently collapse the axis to one value.

The test suite owns these configurations. `run_ffm.py` must not make `BLOCK_M` or another tile a required command-line choice for the coordinator. Later grouped variants inherit the parameterized H0 parent and must retain multi-value coverage for every tuning axis they still expose. A donor-specific layout may restrict one axis—for example AITER's decode WMMA layout starts at `BLOCK_M=16`—but the child must test multiple values of its remaining axes and reject unsupported values explicitly rather than hiding them as hardcoded implementation details.

The following are not sweep axes for the current decode body:

- `SCHEDULE='baseline'`, `PINGPONG=False`, `L2_PREFETCH_DISTANCE=-1`, `PARTIAL_TDM=False`, `RESOLVE_PARTITION_CONFLICTS=False`, and `TDM_SPLIT=False` are kernel invariants enforced by static assertions ([guards](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py:43)).
- `SCALE_BLOCK=32` is the MX scale-group semantic, not a performance knob for these inputs ([configuration field](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_common_gfx1250.py:113)).
- `GROUP_M` and XCD swizzle are derived by the existing AMD option policy rather than exposed by `matmul`; H0 records their resolved values but does not alter the baseline API merely to sweep them ([policy](/root/triton-mi450/python/triton_kernels/triton_kernels/matmul_details/opt_flags.py:101)).

- For `G6`, include `M=8,16` because those are the source kernel's explicit medium-decode regimes.
- For `G7`, route at least two slots of one token through distinct experts and prove their FP32 atomic contributions equal the reference sum before final conversion.
- For `G8`, compare every emitted 32-column E4M3 group and UE8M0 scale against a host implementation of the same group quantization, including an all-zero group and groups with unequal dynamic ranges.

### Route-direct matrix

Route-direct cells use their own torch reference because their contract differs. Exercise:

- `M = 1, 4`.
- `top_k = 1, 4`.
- At least two values of every route-direct tuning axis that the cell exposes, arranged as sparse joint cases rather than a Cartesian product.
- At least two local expert IDs and one rejected/non-local ID case.
- Exact-tile K plus one unsupported K case that must reject cleanly rather than silently mask the wrong contract.
- E4M3 activations, packed E2M1 weights, UE8M0 scales.
- For `R1`, apply the A8 activation scale explicitly around the vector multiply unless gfx1250 compilation proves an equivalent native scaled-upcast form; do not silently turn the destination into A16W4.
- For `R2`, compare gate, up, activation, and final route intermediate separately.
- For `R3`, round/store the activated intermediate as BF16 and compare against the `R2` reference at that explicit boundary.
- For `R4`, dequantize the statically scaled E4M3 output and compare to the unquantized `R2` result under AITER's static-quant contract.
- For `R5`, compare every per-route contribution and the final weighted top-k sum; preserve the chosen BF16/FP32 rounding boundary explicitly.
- For `R6`, prove the duplicated MFMA rows do not create duplicate stores and compare its one real row directly with `R1`.
- For `R7`, compare each top-k route partial and the final output against `R5` with the same rounding order.
- For `R8`, require bitwise equality with `R7`; the only allowed change is load/compute ordering.
- For `R9`, compare top-k IDs against `Q1`, compare weights against the source kernel's top-k-only softmax reference, and compare W1 output against `R6` fed those exact weights; repeated routing work across N tiles must produce identical route decisions.

### Routing matrix

- For `Q1` and `Q2`, exercise `M=1,4`, `top_k=1,4`, tied logits, extreme logits, and optional normalization/scaling. Check IDs exactly and weights at the source kernel's explicit dtype boundaries.
- For `Q2`, include correction bias that changes the selected expert without changing the emitted un-biased gate, plus a case that exposes router-dtype rounding.
- For `Q3`, compare slice sizes/offsets, block schedules, gather/scatter indices, and gates against a stable expert sort for valid and rejected expert IDs.
- For `Q6`, require exactly `M=1` with unique valid top-k IDs, compare every metadata field against `Q3`, and prove that `M!=1` rejects cleanly rather than applying the uniqueness shortcut.
- For `Q4` and `Q5`, compare every routing and metadata output against the corresponding unfused composition (`Q1 + Q3` or `Q2 + Q3`), not merely the final GEMM result.

### Integration matrix

- `I1` must run W1→BF16 intermediate→output-owned W2 for `M=1,4` and `top_k=1,4`, comparing the final token output and every explicit stage boundary.
- `I2` must run W1→static-E4M3 intermediate→output-owned W2 under the same shapes, validate quantize/dequantize boundaries separately, and compare the final token output with the strict-A8 reference.

### Command shape

`H0` writes the exact runner interface, but all cells use this environment shape:

```bash
cd /root/triton-mi450
PYTHONPATH=/root/triton-mi450/python:/root/triton-mi450/python/triton_kernels \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TRITON_ALWAYS_COMPILE=1 \
TRITON_CACHE_DIR=/tmp/a8w4-decode-ablation/<CELL-ID> \
flock /data/lock/amd-gpu.lock \
  /root/scripts/tools/run_on_model.sh --backend ffm -- \
  python3 third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/run_ffm.py \
    --variant <variant> --suite <smoke|full>
```

The runner must call `os._exit(status)` after flushing output. A cell may not replace FFM with a CPU-only test; CPU reference checks are necessary but not sufficient.

## Per-cell protocol

1. Read this plan, the cell row, the parent source, and the cited provenance source.
2. Freeze the cell experiment in its ledger before editing: SHAs, parent file, child file, allowed idea, test cases, FFM command, and correctness metric.
3. Copy the parent to the child; add the ablation header; implement only the named idea.
4. Audit `diff -u parent child`. Remove unrelated cleanup, renaming, formatting, and tuning.
5. Run import/compile smoke under FFM, then the required correctness matrix.
6. If a failure occurs, determine whether it is the idea, a mechanical port error, an unsupported contract, or a harness problem. Do not add an unrelated optimization as a fix.
7. Write `ledgers/<cell-id>.md` with `STATUS`, `CONCLUSION`, frozen experiment, diff classification, FFM evidence, observed/inferred/unknown, accepted/rejected examples, and remaining unknowns.
8. Stop. Do not start another cell and do not make performance claims.

## Definition of done for round 1

- `H0` reproduces the latest PR's 20-case tiny-M default matrix and passes the five additional sparse joint-configuration cases under FFM.
- Every tuning axis exposed by a retained cell has multi-value correctness coverage or a cited, tested intrinsic restriction; exhaustive combination search and winner selection remain performance work.
- Every retained implementation cell has a distinct source file and a passing required FFM matrix.
- Every kernel in the selected corpus has a linked disposition in `KERNEL_COVERAGE.md`, and every active idea row appears in `REGISTRY.md`.
- Every decode dataflow, fusion, ownership, and numerical-boundary mechanism is active; only performance-equivalent schedules/crossovers may remain deferred.
- Every parent→child diff contains one named idea with exact provenance.
- Any idea that cannot be made correct without another independent change is split into a new cell or removed from the round after review; it is not called complete while blocked.
- The original `moe_decode_gfx1250.py`, `moe_common_gfx1250.py`, and `moe_gfx1250.py` remain unchanged.
- No AM/B0 timing, ranking, or “best kernel” conclusion exists yet.

## Alternatives Considered

### Edit one kernel repeatedly

Rejected because later fixes would erase the exact source state associated with earlier ideas and make ablation results unreproducible.

### Build one cumulative “all ideas” kernel

Rejected for round 1 because a correct or incorrect result would not identify which ownership, schedule, layout, or quantization change caused it.

### Encode `BLOCK_M=16` as a dedicated source child

Rejected because the target already accepts `BLOCK_M` as a compile-time configuration and uses it to construct routing metadata and the launch grid. Hardcoding one value would duplicate an existing tuning axis, falsely classify configuration as a source idea, and make the downstream cells depend on an arbitrary tile rather than the parameterized baseline.

### Take the Cartesian product of all tuning values in round 1

Rejected because round 1 needs to catch hidden configuration coupling, not measure or rank every combination. A sparse joint suite gives each exposed axis multiple correctness values at manageable FFM cost; comprehensive search belongs to `P5` under AM/B0.

### Treat route-direct and grouped-WMMA kernels as direct peers

Rejected. They have different routing ownership and output boundaries. They may eventually compete at the full-operation level, but correctness must first be established under their own contracts.

### Reuse the AITER FFM driver unchanged

Rejected as the primary verifier because it dispatches AITER's operator and full two-GEMM forward, not the temporary Triton source modules. Its input/reference utilities may be reused only if the target variant is demonstrably exercised.

### Collect timing under FFM

Rejected because FFM is functional and provides no meaningful timing.

### Defer routing and final composition because they are outside the GEMM

Rejected because both choices change end-to-end decode data movement and launch boundaries. Standalone/fused routing is represented by `Q1–Q6`, in-program routing by `R9`, and the two complete stage compositions by `I1–I2`; none waits for the performance round.

### Copy the former single-file PR baseline

Rejected because the current PR body lives in `moe_decode_gfx1250.py`, calls its hot loop from `moe_common_gfx1250.py`, and is selected and tested through `moe_gfx1250.py`. Using the former `moe_gfx1250.py`-only snapshot would both miss the current source boundary and reintroduce code that the latest decode specialization removed.
