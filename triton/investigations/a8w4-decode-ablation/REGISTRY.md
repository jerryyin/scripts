# A8W4 decode ablation registry

Status values: `pending` | `in-progress` | `done` | `blocked`. Work only an unblocked row. Agents write their own ledger and do not edit this registry; the coordinator merges status after reviewing the ledger.

All source paths below are relative to `/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/`.

| ID | Phase | Parent | Owned source | One idea | Blocked-by | Status | Conclusion | Ledger |
|---|---:|---|---|---|---|---|---|---|
| H0 | 0 | Triton PR #154 `moe_decode_gfx1250.py`; decode-reachable helpers from `moe_common_gfx1250.py`; launch/reference contract from `moe_gfx1250.py` | `k00_pr154_baseline.py`, `run_ffm.py`, `__init__.py` | Freeze one execution-equivalent parameterized baseline; reproduce the 20 source tiny-M cases and add five sparse joint cases covering multiple values of every decode-supported tuning axis | — | pending | — | |
| G2 | 1 | `k00_pr154_baseline.py` | `k02_aiter_decode_wmma_layout.py` | At the donor-compatible `BLOCK_M=16`, substitute AITER's decode-specific WMMA warp layout only | H0 | pending | — | |
| G3 | 1 | `k00_pr154_baseline.py` | `k03_aiter_persistent_n3.py` | At the donor-compatible `BLOCK_M=16`, one workgroup walks three consecutive N tiles after decoding routing once | H0 | pending | — | |
| G4 | 1 | `k00_pr154_baseline.py` | `k04_aiter_packed_schedule.py` | Replace general ragged offset decode with one packed expert/block schedule entry while retaining the selected `BLOCK_M` as configuration | H0 | pending | — | |
| G5 | 1 | `k00_pr154_baseline.py` | `k05_aiter_preshuffled_weight.py` | Consume AITER-style preshuffled packed weights and unshuffle locally | H0 | pending | — | |
| G6 | 1 | `k00_pr154_baseline.py` | `k11_grouped_direct_load.py` | Replace the generic TDM pipeline with TokenSpeed's decode-specific single-buffer direct-load X/W/scale transport | H0 | pending | — | |
| G7 | 1 | `k00_pr154_baseline.py` | `k12_grouped_atomic_combine.py` | Keep grouped scaled-WMMA but atomically add route-weighted FP32 W2 results directly to token output | H0 | pending | — | |
| G8 | 1 | `k00_pr154_baseline.py` | `k16_grouped_mxfp8_emit.py` | Quantize each complete 32-column output group with a dynamic UE8M0 scale and emit E4M3 values plus scales | H0 | pending | — | |
| Q1 | 1 | Semantic port from TokenSpeed softmax router | `k18_softmax_topk_router.py` | Compute softmax, stable top-k IDs, routing gates, normalization, and routing scale in one bounded decode workgroup | H0 | pending | — | |
| Q2 | 1 | Semantic port from TokenSpeed biased router | `k19_sigmoid_bias_topk_router.py` | Compute sigmoid+bias top-k with the source dtype/weight boundary in one bounded decode workgroup | H0 | pending | — | |
| Q3 | 1 | Semantic port from TokenSpeed precomputed-top-k route | `k20_precomputed_topk_group_route.py` | Convert existing top-k IDs/weights into grouped histograms, offsets, schedules, gather/scatter indices, and gates in one workgroup | H0 | pending | — | |
| Q4 | 2 | `k20_precomputed_topk_group_route.py`; router donor `k18_softmax_topk_router.py` | `k21_fused_softmax_group_route.py` | Keep softmax top-k in registers and feed it directly into grouped metadata construction | Q1,Q3 | pending | — | |
| Q5 | 2 | `k20_precomputed_topk_group_route.py`; router donor `k19_sigmoid_bias_topk_router.py` | `k22_fused_biased_group_route.py` | Keep biased grouped top-k in registers and feed it directly into grouped metadata construction | Q2,Q3 | pending | — | |
| Q6 | 2 | `k20_precomputed_topk_group_route.py` | `k25_flat_m1_group_route.py` | For exactly one token, keep rows in top-k slot order and directly emit slice/schedule metadata without histogram, prefix-scan, or stable-rank construction | Q3 | pending | — | |
| R1 | 2 | Semantic fork from `k00_pr154_baseline.py` | `k06_route_direct_wave_gemv.py` | Cursor/TokenSpeed tiny-decode algorithm: address routes directly, stream selected expert weights, reduce K within a wave, and use no expert grouping, M padding, LDS matrix tile, or MFMA/WMMA | H0 | pending | — | |
| R2 | 2 | `k06_route_direct_wave_gemv.py` | `k07_route_direct_gate_up.py` | Apply the Cursor/TokenSpeed W1 pattern: consume one activation stream into dual gate/up accumulators and apply the fused activation | R1 | pending | — | |
| R3 | 2 | `k07_route_direct_gate_up.py` | `k08_route_direct_gate_up_bf16.py` | Add the Cursor/TokenSpeed BF16 intermediate boundary with no FP8 requantization | R2 | pending | — | |
| R4 | 2 | `k07_route_direct_gate_up.py` | `k09_route_direct_gate_up_fp8.py` | Add AITER Gluon's static FP8 quantization as the strict-A8 sibling stage boundary | R2 | pending | — | |
| R5 | 2 | `k06_route_direct_wave_gemv.py` | `k10_output_owned_topk.py` | Apply the Cursor/TokenSpeed W2 pattern: own `(token, output tile)`, loop top-k with FP32 routing-weight accumulation, and store once | R1 | pending | — | |
| R6 | 2 | `k06_route_direct_wave_gemv.py` | `k13_route_direct_scaled_mfma.py` | Retain route-direct IDs but replace wave GEMV with TokenSpeed's duplicated-row direct scaled MFMA | R1 | pending | — | |
| R7 | 2 | `k10_output_owned_topk.py` | `k14_output_owned_topk_scaled_mfma.py` | Retain output-owned top-k but replace each wave reduction with direct scaled MFMA | R5 | pending | — | |
| R8 | 2 | `k14_output_owned_topk_scaled_mfma.py` | `k15_output_owned_topk_scaled_mfma_lookahead.py` | Prefetch the next K-tile pair into registers before MFMA-ing the current pair | R7 | pending | — | |
| R9 | 2 | `k13_route_direct_scaled_mfma.py`; router donor `k18_softmax_topk_router.py` | `k17_route_direct_scaled_mfma_fused_topk.py` | Recompute top-k from logits inside each direct-MFMA W1 program and immediately consume the selected expert | R6,Q1 | pending | — | |
| I1 | 3 | `k08_route_direct_gate_up_bf16.py`; W2 donor `k10_output_owned_topk.py` | `k23_integrated_wave_bf16.py` | Add the already-isolated output-owned W2 stage to the BF16-boundary W1 path and verify the complete decode operation | R3,R5 | pending | — | |
| I2 | 3 | `k09_route_direct_gate_up_fp8.py`; W2 donor `k10_output_owned_topk.py` | `k24_integrated_wave_fp8.py` | Add the already-isolated output-owned W2 stage to the static-FP8 W1 path and verify the complete strict-A8 decode operation | R4,R5 | pending | — | |
| A1 | 4 | All completed implementation cells | No source; audit only | Verify kernel-corpus coverage, one-change diffs, provenance, FFM coverage, and unchanged original target | G2–G8,Q1–Q6,R1–R9,I1,I2 | pending | — | |

## Dependency notes

- Phase 1 grouped cells `G2–G8` may run in parallel after `H0`. They all branch directly from the same parameterized baseline; tuning values are selected by tests and never introduced through separate source parents. Each child retains multi-value correctness coverage for every tuning axis it exposes.
- `Q1–Q3` are independent routing foundations. `Q4` adds only the materialization-boundary fusion between `Q1` and `Q3`; `Q5` does the same for `Q2` and `Q3`; `Q6` removes the general grouping machinery under the separately proven `M=1` uniqueness invariant.
- `R1` is one atomic algorithmic fork, not four artificial micro-ablation edges: direct route ownership, streaming weight rows, wave reduction, and abandoning MFMA/WMMA are the mutually dependent structure of the TokenSpeed algorithm. It must state its changed ABI/reference. `R2–R9` isolate follow-on or sibling ideas within that family.
- `I1–I2` are correctness integration rows, not new optimization ideas. Each edge adds the already-tested `R5` W2 ownership to one W1 boundary so the complete pipeline cannot be postponed to performance work.
- `A1` does not choose a winner. It certifies that round 1 produced correct, independently attributable source variants and complete integration paths.
