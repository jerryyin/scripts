# A8W4 MoE decode ablation — conclusions

## Bottom line

Basically, the study found one major algorithmic fork and several smaller choices around it. The grouped path assigns work to padded expert tiles and uses coordinated scaled-WMMA pipelines. The route-direct path does not group tokens by expert: W1 assigns programs directly to routed rows, streams the selected expert's weights, and either reduces K within a wave or duplicates the real row into gfx1250 scaled WMMA; W2 assigns one program to a token/output tile, loops over top-k experts, and stores the final weighted result once. These are different execution models, not tuning variants of one kernel.

The most transferable decode idea is ownership, not a particular instruction. For tiny decode, [`R1`](ledgers/R1.md) and [`R2`](ledgers/R2.md) remove expert grouping, padded M work, and matrix-tile cooperation from W1, while [`R5`](ledgers/R5.md) removes per-route W2 output and a separate combine by giving each token/output tile one owner. [`R6`](ledgers/R6.md) and [`R7`](ledgers/R7.md) show that the same direct ownership can use gfx1250 scaled WMMA instead of wave reductions, but they are compute-primitive siblings rather than proof that matrix instructions are always preferable.

The investigation does not identify a fastest kernel. It establishes that the isolated mechanisms are correctness-valid under gfx1250 FFM, eliminates several false candidates, and reduces the next hardware phase to a small set of meaningful comparisons.

## Spot-check result

All 26 ledgers were re-read at their headline, evidence, observed-result, unknown, and final-dependency sections. All 25 implementation-source hashes still match the exact snapshot recorded by [`A1`](ledgers/A1.md), every registry row remains `done`, and the per-cell full-case counts reproduce the audited total of 233. Adding one smoke and one independent repeat per implementation cell gives at least 283 accepted invocations before rejection and control runs are counted.

The evidence is internally consistent:

- H0 and G2–G6 contribute 150 full cases.
- G7–G8 contribute 10.
- Q1–Q6 contribute 29.
- R1–R9 contribute 36.
- I1–I2 contribute 8.

No substantive ledger contradiction was found. The only spot-check nit is a duplicated final section heading in `I1.md`; it has no effect on source, evidence, or conclusion and is left untouched so the audited ledger manifest remains stable.

The protected Triton sources remain unchanged. Local Triton and the live PR #154 head both resolve to `80e223f93d59359161f5482fbb69bbfab29c0a0b`; AITER remains `4a1cc773f34cbfc74387259e51262556ee38edd0`; TokenSpeed remains `3e725ac2b785b71f27ff9e9ac3796349c495d225`. FFM establishes functional and generated-code correctness only; it provides no timing evidence.

## The execution models that survived

| Family | Program ownership | Work and traffic retained or removed | Compute | Correctness result | Main unresolved tradeoff |
|---|---|---|---|---|---|
| Grouped decode | `(expert block, N tile)` over padded expert-major rows | Retains grouping, padded M tiles, coordinated LDS/TDM or explicit LDS transport, and grouped reuse | gfx1250 scaled WMMA | H0 and G2–G8 pass their frozen matrices | Whether reuse across grouped rows repays sorting, padding, staging, and combine overhead at each decode shape |
| Route-direct wave W1 | `(route, N tile)` | Removes grouping and padded M; streams only the selected expert's weights; reuses one X fragment across gate/up | Packed-FP4 upcast, elementwise FMA, wave reduction | R1–R4 pass; I1/I2 validate complete two-stage compositions | Weight streaming and repeated X reads may lose to matrix reuse as useful rows per expert grow |
| Output-owned wave W2 | `(token, N tile)` | Loops top-k locally, applies FP32 routing weights, eliminates per-route W2 rows and combine storage | Packed-FP4 upcast, elementwise FMA, wave reduction | R5 passes and is integrated by I1/I2 | Register lifetime and repeated selected-expert loads versus grouped reuse |
| Route-direct matrix W1 | `(route, N tile)` with one real row duplicated into a 16-row matrix tile | Keeps direct routing and zero-LDS streaming but performs fictitious duplicate-row matrix work | gfx1250 scaled WMMA | R6 passes with 16 bitwise-identical rows and one real-row store | Whether higher instruction efficiency compensates for duplicated rows and narrower legal configurations |
| Output-owned matrix W2 | `(token, N tile)` with top-k loop and duplicated matrix rows | Keeps the one-owner/one-store W2 contract while replacing wave reduction | gfx1250 scaled WMMA | R7 passes and matches R5 at every captured boundary | No complete matrix W1→W2 integration cell exists yet; performance and end-to-end resource use are unknown |
| In-program routing W1 | `(token, N tile, slot)`; every program recomputes the same route | Removes a separate route input/launch for the narrow specialization but replicates router work | One-wave top-k plus scaled WMMA | R9 passes with bitwise-identical decisions across programs | Restricted to `E<=32`, normalized gates, and one-wave `BLOCK_N=16`; replicated work may exceed the saved boundary |

## Conclusions by kernel idea

### 1. Choose ownership before choosing the instruction

**Observed:** [`R1` maps `route = pid // NUM_PID_N`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k06_route_direct_wave_gemv.py:121), loads only that route's expert, and reduces K with `gl.sum` at [line 211](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k06_route_direct_wave_gemv.py:211). [`R6`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k13_route_direct_scaled_mfma.py:144) keeps the same route ownership but substitutes `wmma_scaled` at [line 273](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k13_route_direct_scaled_mfma.py:273). Both are correct.

**Inference:** the meaningful first dispatch question is whether the workload has enough useful rows per expert to amortize grouping and padded matrix tiles. Wave versus WMMA is a second-level choice inside the direct family.

**Unknown:** the actual crossover. Token count alone is an incomplete proxy because top-k, expert collision rate, N/K shape, cache locality, and the number of useful rows per expert all change the economics.

### 2. Direct W1 should reuse X across gate and up

**Observed:** [`R2`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k07_route_direct_gate_up.py:84) adds the dual gate/up path on top of R1 while retaining direct ownership and one activation stream. Gate, up, and `SiLU(gate) * up` passed independently checked boundaries. This is the gfx1250 realization of the mechanism described by [Cursor Warp Decode](https://cursor.com/blog/warp-decode): one output owner streams the two expert rows and immediately reuses the activation.

**Inference:** this should remain part of any route-direct W1 candidate. Splitting gate and up into separate activation passes would discard reuse without restoring grouped-row reuse.

**Unknown:** the corresponding dual-projection scaled-WMMA W1 has not been integrated as a complete ablation. R6 proves direct scaled-WMMA projection, not the full R2 gate/up composition.

### 3. Output-owned W2 is the cleanest buffer-elimination idea

**Observed:** [`R5` maps one program to a token and output tile](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k10_output_owned_topk.py:148), loops top-k at [line 180](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k10_output_owned_topk.py:180), and retains one semantic final store with no per-route output or combine buffer. [`R7`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k14_output_owned_topk_scaled_mfma.py:162) preserves that ownership with scaled WMMA.

**Inference:** W2 output ownership is more generally useful than any one W1 compute primitive because both wave and matrix implementations can preserve it. It is the first idea to keep when building another tiny-decode path.

**Unknown:** grouped atomic combine in G7 may win when grouped reuse is already profitable, while R5/R7 may win when top-k is small and output tiles expose enough independent programs. Only hardware timing can decide.

### 4. Scaled WMMA is a sibling path, not a universal upgrade

**Observed:** R6 and R7 compile to the legal gfx1250 `v_wmma_scale_f32_16x16x128_f8f6f4`, not the CDNA4 MFMA used by the TokenSpeed donor. They duplicate one semantic row across 16 matrix rows, prove all duplicates are bitwise equal, and store only the real row.

**Inference:** direct scaled WMMA is a credible candidate when K and N fit the native tile and the duplicated-row cost is smaller than the savings from replacing elementwise reduction. It should be measured against R1/R5 as a separate kernel, not mixed into their first performance baseline.

**Unknown:** matrix-versus-wave crossover, register pressure, cache traffic, and whether a complete gate/up plus W2 matrix path fits without spills or harmful live ranges.

### 5. Routing has three distinct boundaries

**Observed:** [`Q1`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k18_softmax_topk_router.py:65) and [`Q2`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k19_sigmoid_bias_topk_router.py:74) prove standalone one-workgroup softmax and sigmoid+bias top-k contracts. [`Q3`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k20_precomputed_topk_group_route.py:92) proves general precomputed-top-k grouping. [`Q4`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k21_fused_softmax_group_route.py:99) and [`Q5`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k22_fused_biased_group_route.py:109) remove the global top-k boundary while preserving executable Q1→Q3 and Q2→Q3 results. [`R9`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k17_route_direct_scaled_mfma_fused_topk.py:166) moves routing inside every W1 program instead.

**Inference:** Q4/Q5 are the natural routing fusions for a grouped fallback; Q1/Q2 are the clean first controls for direct decode; R9 is a narrow optional specialization rather than the default fusion strategy.

**Unknown:** whether eliminating the route tensors and launch offsets Q4/Q5 dynamic-LDS/live-state cost, and whether R9's replicated routing is worthwhile inside its one-wave domain.

### 6. Q6 is a valid idea source but not a live optimization

**Observed:** [`Q6`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k25_flat_m1_group_route.py:91) correctly replaces the general histogram/prefix/rank path with direct expert-by-slot reductions for unique valid `M=1` routes and matches Q3 on all seven fields. The pinned TokenSpeed caller is unreachable: its containing helper requires `M>=4`, while the flat branch requires `M==1`.

**Conclusion:** retain Q6 as a standalone structural specialization, but do not describe it as an active TokenSpeed dispatch or include it in timing until a real caller and its validation cost are defined.

### 7. The stage boundary is an explicit design choice

**Observed:** [`R3`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k08_route_direct_gate_up_bf16.py:91) writes a bitwise-checked BF16 W1 intermediate with no FP8 requantization. [`R4`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k09_route_direct_gate_up_fp8.py:90) instead applies a supplied static scale and emits exact E4M3 bytes. [`G8`](/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/k16_grouped_mxfp8_emit.py:718) derives one dynamic UE8M0 scale per complete 32-column group and matches the donor's integer oracle byte-for-byte.

**Inference:** BF16, static E4M3, and dynamic per-32 MXFP8 are alternatives, not additive optimizations. The boundary should be selected with the downstream consumer and model-accuracy contract, not attached to every kernel variant.

**Unknown:** end-to-end model quality, conversion/store cost, and whether a narrow route-direct tile can own a complete 32-column dynamic-scale group without widening the program or adding cooperation.

### 8. The grouped path still has useful decode-specific variants

**Observed:** H0 is the expert-grouped scaled-WMMA control. G2 changes the BM16 WMMA warp layout; G3 walks three N tiles after one route decode; G5 consumes a preshuffled weight layout and locally inverts it; G6 replaces generic input TDM with explicit X/W/scale loads through one LDS stage; G7 folds route-weighted results into token output with FP32 atomics; G8 changes the output quantization boundary.

**Inference:** G6 is the cleanest grouped transport candidate to measure first because it isolates direct loading without changing ownership or arithmetic. G2, G3, and G5 should remain independent toggles until hardware data shows which source of reuse or traffic matters.

**Unknown:** whether any grouped variant beats route-direct execution at the intended decode shapes and where expert collision begins to make grouped reuse dominant.

### 9. Two source-level ideas should not be promoted as performance candidates yet

**Observed:** G4's packed-schedule source simplification lowers to the same inspected TTGIR load/mask/shift/load/multiply sequence as H0. R8's paired source lookahead is numerically bitwise-equal to R7, but K=512 TTGIR hoists all 12 operand loads before the first WMMA rather than preserving the intended pairwise overlap.

**Conclusion:** G4 is useful source clarification, not a distinct codegen candidate. R8 should not enter a timing comparison until generated code or a compiler change demonstrates the intended overlap.

## Every cell's disposition

| Cell | What it established | Disposition after spot-check |
|---|---|---|
| [H0](ledgers/H0.md) | Exact PR #154 grouped-decode control over representative tuning values | Keep as the correctness and grouped-performance control |
| [G2](ledgers/G2.md) | BM16 all-N four-warp WMMA layout | Conditional grouped candidate; measure only at its intrinsic BM16/NW4 boundary |
| [G3](ledgers/G3.md) | One route decode feeds three consecutive N tiles | Conditional persistence candidate; measure route/setup amortization and N tails |
| [G4](ledgers/G4.md) | Direct packed schedule source formulation | Do not treat as a separate performance candidate under the current lowering |
| [G5](ledgers/G5.md) | Preshuffled packed weights plus LDS-local inverse and K-tail padding | Conditional layout candidate; measure traffic versus inverse/padding cost |
| [G6](ledgers/G6.md) | Single-buffer explicit grouped X/W/scale transport | Primary grouped transport candidate |
| [G7](ledgers/G7.md) | Grouped scaled-WMMA plus FP32 atomic token combine | Alternative W2 ownership/combine candidate; compare with R5/R7 |
| [G8](ledgers/G8.md) | Dynamic per-32 E4M3/UE8M0 output boundary | Retain as a quantization-boundary candidate, not a universal epilogue |
| [Q1](ledgers/Q1.md) | Stable softmax top-k router | Standalone direct-decode routing control |
| [Q2](ledgers/Q2.md) | Sigmoid+bias top-k with router-dtype boundary | Standalone biased-routing control |
| [Q3](ledgers/Q3.md) | General precomputed-top-k grouped metadata construction | Grouped-routing foundation and Q4/Q5 reference |
| [Q4](ledgers/Q4.md) | Softmax top-k fused with grouped materialization | Grouped routing-fusion candidate |
| [Q5](ledgers/Q5.md) | Biased top-k fused with grouped materialization | Biased grouped routing-fusion candidate |
| [Q6](ledgers/Q6.md) | Unique-valid-M1 direct metadata construction | Standalone structural candidate; no live caller at the pinned revision |
| [R1](ledgers/R1.md) | Route-direct wave GEMV without grouping, padding, LDS, or matrix instructions | Core tiny-decode ownership candidate |
| [R2](ledgers/R2.md) | One activation stream feeds gate and up with fused SwiGLU | Core route-direct W1 candidate |
| [R3](ledgers/R3.md) | BF16 intermediate without FP8 requantization | BF16 boundary option |
| [R4](ledgers/R4.md) | Supplied-scale static E4M3 intermediate | Strict-A8 boundary option |
| [R5](ledgers/R5.md) | Output-owned top-k W2 with one final store | Core route-direct W2 candidate |
| [R6](ledgers/R6.md) | Route-direct duplicated-row scaled-WMMA projection | Matrix W1 component candidate; full gate/up integration remains open |
| [R7](ledgers/R7.md) | Output-owned top-k scaled-WMMA W2 | Matrix W2 component candidate; full two-stage integration remains open |
| [R8](ledgers/R8.md) | Correct paired source lookahead with compiler-hoisted loads | Hold until intended native overlap is observable |
| [R9](ledgers/R9.md) | Replicated in-program top-k feeding direct scaled-WMMA W1 | Narrow one-wave fusion candidate only |
| [I1](ledgers/I1.md) | Exact two-launch R3 BF16 W1 → BF16-input output-owned W2 | Complete direct-wave BF16 correctness candidate for hardware timing |
| [I2](ledgers/I2.md) | Exact two-launch R4 static-E4M3 W1 → R5 output-owned W2 | Complete strict-A8 direct-wave correctness candidate for hardware timing |
| [A1](ledgers/A1.md) | Corpus, source, ledger, provenance, evidence, and performance-claim closure | Audit control; no kernel or winner |

## Recommended next hardware phase

Do not start with one kitchen-sink kernel. Measure a small set of exact, already-correct candidates:

1. Use I1 and I2 as the two complete route-direct wave baselines: BF16 intermediate versus strict-A8 static E4M3.
2. Compare R1 versus R6 at the same route-owned projection boundary and R5 versus R7 at the same output-owned W2 boundary. This isolates wave reduction versus duplicated-row scaled WMMA before attempting a full matrix two-stage integration.
3. Compare H0 versus G6 as the grouped transport control. Add G2, G3, and G5 one at a time only after a trace identifies layout, repeated route/setup work, or weight traffic as the limiting factor.
4. Compare Q1/Q2 standalone routing with Q4/Q5 grouped fusion only inside the grouped path. Test R9 separately within its exact one-wave domain.
5. Compare G7 atomics with R5/R7 output ownership using identical routing distributions and output shapes.

The performance matrix should vary `M`, top-k, N, K, expert count, and actual route collision/useful rows per expert. Record generated-code identity, register pressure/spills, LDS, wait placement, cache traffic, launch boundaries, and end-to-end latency. Dispatch policy should be derived from those measurements; it should not be assumed from M alone.

Before timing a full matrix route-direct pipeline, add one correctness integration that combines dual gate/up scaled-WMMA W1 with output-owned scaled-WMMA W2. R6 and R7 prove the individual projection mechanisms, but the current study does not prove their combined gate/up, intermediate-format, and resource contract.

## Alternatives Considered

- **Merge every passing idea into one kernel.** Rejected because several choices conflict: grouped versus direct ownership, wave reduction versus duplicated-row WMMA, atomic versus single-owner W2 combine, and BF16 versus static or dynamic FP8 boundaries.
- **Declare route-direct execution the winner from the Cursor result.** Rejected because Cursor measured NVIDIA B200, while this study produced only gfx1250 FFM correctness evidence. The ownership principle transfers; the speedup does not.
- **Choose the grouped path solely for matrix reuse.** Rejected because no measurement yet shows that reuse amortizes routing, padding, staging, and combine work at the target decode shapes.
- **Promote G4 or R8 directly to hardware timing.** Rejected because G4 has no observed codegen difference and R8 does not retain its intended overlap in TTGIR.
- **Use Q6 as an existing live fast path.** Rejected because its pinned caller branch is unreachable.
- **Use one intermediate format for all candidates.** Rejected because I1 and I2 deliberately establish different precision/traffic contracts, and G8 requires different ownership granularity.
- **Dispatch only on token count.** Rejected because useful rows per expert, top-k, expert collision, N/K, and cache locality determine whether grouping has reusable work.

## Observed, inferred, and unknown

**Observed:** all 25 owned implementation variants match their audited source hashes and pass their frozen FFM correctness matrices; generated artifacts establish the claimed ownership, instructions, stores, LDS/scratch state, or explicit absence thereof. The protected sources are unchanged.

**Inferred:** the best near-term design space is a route-direct W1 plus output-owned W2 family for sparse tiny decode, alongside a decode-specialized grouped fallback. Wave and scaled-WMMA implementations should remain separate candidates until hardware establishes their crossover.

**Unknown:** all physical MI450 performance, occupancy, register pressure, cache behavior, routing-family crossovers, model-level quality effects, expert-parallel remote-route behavior, production dispatcher integration, and the complete scaled-WMMA two-stage resource contract.

## Final conclusion

The study succeeded as a correctness-first idea collection. It proved that the Cursor/TokenSpeed route-direct model can be expressed on gfx1250 without grouping or matrix instructions, that its output-owned W2 idea removes the per-route combine boundary, and that the same direct ownership can also be implemented with legal gfx1250 scaled WMMA. It also preserved a credible grouped family with independently testable layout, persistence, transport, combine, and quantization variants.

The next step is not another broad source-level ablation grid. It is one targeted correctness integration for the full scaled-WMMA route-direct pipeline, followed by controlled AM/B0 measurement of the exact surviving candidates. Until then, the correct conclusion is a shortlist and a dispatch hypothesis—not a winner.
