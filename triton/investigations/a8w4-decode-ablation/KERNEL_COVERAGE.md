# Decode-kernel coverage and idea provenance

## Coverage claim

This is the closed manifest for the A8W4 decode ablation plan. It covers every launchable kernel entry point that was read or cited during the Triton/AITER/TokenSpeed comparisons, plus every decode-labelled Gluon entry point found in the selected gfx1250 and gfx950 MoE source files at the pinned revisions. A kernel is not silently omitted: each maps to an active registry cell, reinforces an active cell, is a performance-only schedule whose functional dataflow already has an active row, is explicitly rejected as out of scope, or is evidence rather than a decode candidate.

This is not a claim that the three repositories contain no other MoE kernels for other precisions or architectures. The corpus boundary is: gfx1250 A8W4 counterparts, TokenSpeed's gfx950 A4W4/A8W4/A16W4 and BF16 decode idea sources, their decode routing/combine kernels, and the non-decode helpers explicitly cited by the prior reports.

Pinned state:

- Triton PR #154 checkout: `/root/triton-mi450` at `80e223f93d59359161f5482fbb69bbfab29c0a0b`.
- AITER: `/root/aiter` at `4a1cc773f34cbfc74387259e51262556ee38edd0`.
- TokenSpeed: `/root/tokenspeed` at `3e725ac2b785b71f27ff9e9ac3796349c495d225`.

## Idea-row vocabulary

Active rows are defined in `REGISTRY.md`: `H0`, `G2–G8`, `Q1–Q6`, `R1–R9`, `I1–I2`, and `A1`.

| Performance/rejected/support row | Disposition |
|---|---|
| P1 | Generic-parent compute-hiding schedules and long-K transport knobs that the latest decode body rejects; performance only if later timing identifies a concrete need |
| P2 | Output-owned split-K plus partial reduction; performance schedule for active non-split `R7` |
| P3 | Variable-wave layouts in the shadowed general precomputed-route implementation; performance scheduling for active fixed-one-wave `Q3` |
| P4 | Runtime selection between already-correct grouped and route-direct kernel families |
| P5 | Exhaustive search and runtime selection across H0-covered tile, warp, buffer, and layout axes; AITER's route-density heuristic is one candidate policy component |
| X1 | Rejected: prefill-only direct async-copy of narrow activation scales |
| X2 | Rejected: A4 activation preprocessing outside the A8W4 destination contract |
| S1 | Architecture/primitive proof only; not a MoE decode kernel |
| S2 | Wrapper or dispatch evidence only; no separately launchable kernel |

## Triton and AITER compute kernels

| Kernel | Execution role | Idea rows | Disposition |
|---|---|---|---|
| [Triton `_matmul`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:1329) | Generic expert-grouped scaled-WMMA parent; can express alternate schedules and long-K transport choices | H0, P1 | Baseline ancestry only; the latest decode body rejects those variants |
| [Triton `_matmul_decode`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py:17) | Dedicated M-ragged, split-K=1 decode body parameterized by block sizes, warps, buffers, and scale layout | H0 | Exact destination body; H0 uses sparse multi-value configuration coverage without source forks |
| [Triton `MoEPipelinedProgram`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_common_gfx1250.py:509) | Shared TDM/LDS pipeline and scaled-WMMA hot loop invoked by `_matmul_decode` | H0 | Decode-reachable helper closure that `H0` snapshots with the body |
| [AITER `_moe_gemm_a8w4_decode_persistent`](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:101) | BM16 grouped scaled-WMMA; one workgroup walks several N tiles | H0, G2, G3, G4, G5, R4, P5 | BM16 is an H0 configuration/provenance point; layout, persistence, schedule, weight layout, and epilogue are active source ideas |
| [AITER `_moe_gemm_a8w4_decode`](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:769) | BM16 grouped scaled-WMMA decode | H0, G2, G4, G5, R4, P5 | BM16 is an H0 configuration/provenance point; remaining mechanisms map to active source ideas |
| [AITER `_moe_gemm_a8w4_prefill`](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:1346) | Grouped prefill kernel with a narrow-scale transport path | X1 | Rejected because the mechanism is coupled to the prefill pipeline, not deferred |
| [AITER Triton `_moe_gemm_a8w4`](/root/aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py:109) | Generic grouped A8W4 kernel with optional per-32 MXFP8 output | G8, R4 | Per-32 dynamic and static intermediate quantization are both active |

## TokenSpeed grouped and route-direct compute kernels

| Kernel | Execution role | Idea rows | Disposition |
|---|---|---|---|
| [gfx1250 `_matmul`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx1250.py:1652) | Expert-grouped A8W4 scaled-WMMA with baseline/slice schedule family | H0, P1 | Reinforces grouped baseline; alternate schedules are performance-only |
| [gfx950 `_pipelined_moe_kernel_scaled`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:4990) with [medium-decode body](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:4671) | M=8/16 grouped A8W4 decode using a single-buffer direct-load body | G4, G6, R4 | Active source for grouped direct-load transport |
| [A16W4 stage 1 `_stage1_a16w4_situ_warp_gemv`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:65) | Route-owned gate/up wave GEMV; no grouping or matrix instruction | R1, R2, R3 | Active source for wave-GEMV W1 |
| [A16W4 stage 2 `_stage2_a16w4_warp_gemv_combine`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a16w4_situ_decode.py:241) | Token/output-owned wave GEMV; loops top-k and stores once | R1, R3, R5 | Active source for output-owned W2 |
| [Historical A16W4 stage 1 `_stage1_mxfp4_warp_gemv_gluon`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:425) | Route-owned wave GEMV with scalar FP4 decoding | R1, R2, R3 | Reinforces wave ownership; scalar nibble decoding is not ported |
| [Historical A16W4 stage 2 `_stage2_mxfp4_warp_gemv_gluon`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:599) | Output-owned wave GEMV with fused top-k | R1, R5 | Reinforces output ownership; scalar nibble decoding is not ported |
| [BF16 stage 1 `_stage1_warp_gemv_gluon`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/warp_decode_gluon_kernel.py:41) | Route-owned wave GEMV for gate/up | R1, R2, R3 | Independent dtype confirmation of the wave-decode structure |
| [BF16 stage 2 `_stage2_warp_gemv_gluon`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/warp_decode_gluon_kernel.py:156) | Output-owned wave GEMV with fused top-k | R1, R5 | Independent dtype confirmation of output ownership |
| [A4W4 direct-MFMA stage 1 `_stage1_mxfp4_direct_mfma_gluon`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:915) | Route-owned direct scaled MFMA with a duplicated M row | R2, R3, R6 | Active source for the matrix-primitive sibling of R1 |
| [A4W4 direct-MFMA stage 2 `_stage2_mxfp4_direct_mfma_gluon`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:1123) | Output-owned direct scaled MFMA with fused top-k and optional register lookahead | R5, R7, R8 | Active source for direct-MFMA W2 |
| [A8W4 cooperative W1 `_warp_decode_topk_stage1_coop_kernel`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10121) | Direct-route scaled-MFMA W1; also recomputes top-k inside each program | R2, R4, R6, R9 | Matrix/dataflow and in-program routing fusion are active |
| [A8W4 output-owned W2 `_warp_decode_stage2_fp8_mxfp4_kernel`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10403) | Direct scaled-MFMA W2, loops top-k, software-pipelines K, optionally split-K | R5, R7, R8, P2 | Non-split ownership/pipeline is active; split-K is performance-only |
| [BF16 grouped W2 `gluon_bf16_moe_stage2_atomic_kernel`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/stage2_decode_kernel.py:48) | Expert-grouped MFMA whose route-weighted results atomically accumulate into FP32 token output | G7 | Active alternative to output-owned top-k |

## Decode routing and combine kernels

These kernels matter to end-to-end decode, but most are not source changes to the destination A8W4 GEMM. Listing them prevents “not in round 1” from becoming “forgotten.”

| Kernel | Execution role | Idea rows | Disposition |
|---|---|---|---|
| [TokenSpeed `_softmax_topk_route_gluon_kernel`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:129) | Single-workgroup softmax/top-k producing direct-route IDs and weights | Q1 | Active standalone routing row; also supplies the routing primitive for `Q4` and `R9` |
| [TokenSpeed `_sigmoid_bias_topk_route_gluon_kernel`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py:219) | Single-workgroup biased sigmoid/top-k routing | Q2 | Active standalone biased-routing row; also supplies the routing primitive for `Q5` |
| [TokenSpeed `_fused_route_small_m`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:9385) | Fuses top-k, histogram, packed schedule, gather/scatter, and gate materialization | G4, Q4 | Packed schedule reinforces `G4`; full routing fusion is active as `Q4` |
| [TokenSpeed `_fused_biased_grouped_route_small_m`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:9508) | Biased/grouped-top-k version of fused small-M route materialization | G4, Q5 | Packed schedule reinforces `G4`; biased grouped-routing fusion is active as `Q5` |
| [TokenSpeed `_precomputed_topk_route_small_m`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:9621) | Builds grouped metadata from precomputed top-k with a variable-wave layout | G4, P3 | Same general metadata algorithm as `Q3`, but its only wrapper is shadowed; revive only as a performance schedule comparison |
| [TokenSpeed `_precomputed_topk_route_m1_flat`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:9722) | For M=1, directly maps unique top-k slots to expert slices and compact schedules without histogram/prefix/rank construction | G4, Q6 | Active structural decode row, not a scheduling variant |
| [TokenSpeed `_fused_precomputed_topk_route_small_m`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10915) | Live fixed-one-wave precomputed-top-k grouped metadata construction | G4, Q3 | Active general precomputed-route row |
| [AITER Gluon `reduce_grouped_gluon`](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/moe/reduce.py:16) | Loads per-route rows and performs a separate FP32 token combine | G7, R5 | Negative-space source: G7 and R5 remove this launch in different ways |
| [AITER Triton `_reduce_grouped`](/root/aiter/aiter/ops/triton/_triton_kernels/moe/reduce.py:8) | Fallback grouped row combine | G7, R5 | Same negative-space role |
| [TokenSpeed `_moe_partial_reduce`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10591) | Reduces output-owned split-K or grouped-combine partials | G7, R5, P2 | Reinforces the separate-combine boundary removed by `G7`/`R5`; only its use for split-K is performance-deferred |
| [TokenSpeed `_mxfp4_quantize_cdna4_scale_kernel`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:6970) | Quantizes gathered activations to packed MXFP4 plus CDNA4-swizzled scales | R6, R7, X2 | Ownership evidence reinforces active direct-MFMA rows; A4 preprocessing itself is rejected from the A8W4 target |
| [TokenSpeed `_mxfp4_quantize_cdna4_scale_tiled_kernel`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:7061) | Tiled form of the same MXFP4 activation quantization | R6, R7, X2 | Same explicit precision rejection |
| [TokenSpeed `_fp8_quantize_kernel`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:11387) | Standalone scalar-scale FP8 conversion used around A8W4 paths | G8, R4 | Negative-space source for fusing static or per-group FP8 emission into active epilogue rows |

## Supporting sources that are not omitted decode kernels

| Source | Why it was read | Idea rows | Classification |
|---|---|---|---|
| [AITER gfx1250 unified attention kernel](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py:1924) and its [packed-FP4 `scaled_upcast`](/root/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py:1489) | Proves gfx1250 exposes the primitive needed to adapt TokenSpeed's streamed FP4 weights | R1, S1 | Supporting architecture evidence, not MoE |
| [AITER A8W4 dispatch wrapper](/root/aiter/aiter/ops/triton/moe/moe_op_gemm_a8w4.py:250) | Establishes BM16 and persistent-decode selection | H0, G3, P5, S2 | Wrapper evidence; BM16 is a tested configuration and selection remains performance policy |
| [AITER routing heuristic](/root/aiter/aiter/ops/triton/moe/moe_routing/routing.py:303) | Selects an M tile from average routed rows per expert and clamps it to `[16,128]` | H0, P5, S2 | H0 includes representative M-tile values with matching metadata/grid; exhaustive joint search and the selection formula are performance policy, not a separate kernel |
| [TokenSpeed A16W4 dispatch guard](/root/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py:144) | Establishes the conservative `M<=4` route-direct regime | R1, P4, S2 | Active regime evidence for `R1`; exact crossover is performance-only |
| [Triton decode selection and launch](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:1713), [general configuration cases](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:1751), and [tiny-M test](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_gfx1250.py:2012) | Select the split-out body, rebuild metadata/grid for the chosen blocks, enumerate joint configurations, and define the current `M=1,2,4,8,16` gather/scatter E4M3/E5M2 acceptance matrix | H0, P5, S2 | Launcher/reference/test evidence; representative correctness configurations feed H0, while exhaustive selection remains performance policy |
| [TokenSpeed shadowed and live precomputed-route wrappers](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:10792) and [later overriding definition](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py:11003) | Establish that the variable-wave wrapper is replaced at module load and the live wrapper launches fixed-one-wave `Q3` | P3, S2 | Source-selection evidence; repository search found no other live caller of the variable-wave kernel |
| [TokenSpeed A4W4 `decode_stage1.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_stage1.py:22) and [`decode_stage2.py`](/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_stage2.py:22) | Re-export the direct-MFMA launchers defined in `decode_kernels.py` | R6, R7, R8, S2 | No additional kernel body |
| [Cursor Warp Decode](https://cursor.com/blog/warp-decode) | Conceptual source for output ownership, no grouping/staging, BF16 boundary, and top-k-in-W2 | R1, R2, R3, R5 | First-class writeup, not repository kernel |

## Completeness procedure

Before round 1 starts, `A1`, or any plan revision:

1. Reconfirm the three pinned SHAs and resolve `refs/pull/154/head`; no row may start if the live PR head differs from the pinned Triton SHA.
2. Search the selected files for launchable `@gluon.jit`, `@triton.jit`, and registered kernel entry points containing or selected by decode paths.
3. Add every newly read kernel to this manifest before extracting an idea.
4. Give every decode mechanism an active row or existing-row reinforcement. Use `P*` only when an active row already represents the same functional dataflow and the remaining difference is purely a performance schedule/crossover; otherwise record an explicit `X*` rejection or an evidence-only `S*` classification.
5. Reject completion if a linked kernel has no disposition or an active idea row has no registry cell.

Observed: the latest PR rewrites the target into a split decode body, shared pipeline helper, and launch/test surface. It adds explicit tiny-M tests but no second compute ownership model, so it changes `H0` and the baseline matrix rather than adding a new optimization row.

Inferred: `Q1–Q6`, `R6–R9`, and `G6–G8` are transferable to gfx1250, but correctness and performance on gfx1250 remain untested.

Unknown: whether TokenSpeed's gfx950 direct-MFMA ownership or the wave-GEMV ownership wins at the target shapes, and whether FP32 atomic combine is acceptable under the target determinism contract.
