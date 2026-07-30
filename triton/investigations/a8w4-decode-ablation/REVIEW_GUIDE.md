# A8W4 decode ablation review guide

## Bottom line

`k00_pr154_baseline.py` is the self-contained H0 baseline. Its `_matmul_decode` definition is textually identical to the pinned `moe_decode_gfx1250.py::_matmul_decode`, while the nine decode-reachable definitions that the original imports from `moe_common_gfx1250.py` are copied into the same module. The local launch, reference, source-fidelity, and FFM machinery lives in `run_ffm.py`, not in `k00`.

The variants form a dependency graph, not a collection of independent diffs against `k00`. Review each declared parent→child edge. Diffing a descendant against `k00` repeats every inherited idea and hides the one change that the cell owns.

The frozen variant snapshot is `/root/triton-mi450` branch `users/jerryyin/a8w4-decode-ablation-variants` at `e201aa8ae21ec5be3bd15f45d440bc98d0171057`, based on pinned PR #154 head `80e223f93d59359161f5482fbb69bbfab29c0a0b`.

## What `k00` means

The original decode surface is split:

- `moe_decode_gfx1250.py` owns `_matmul_decode`.
- `moe_common_gfx1250.py` owns the layouts, `MoEConfig`, descriptor construction, TDM/LDS transport, pipeline, and scaled-WMMA implementation called by `_matmul_decode`.
- `moe_gfx1250.py` owns the host launch, input construction, and reference contract used by the source tests.

H0 freezes those mutable dependencies into an experiment-owned boundary:

- `k00_pr154_baseline.py` mechanically combines the decode body with its decode-reachable compute/helper closure.
- `run_ffm.py` provides the local launch and reference path, pins all three parent hashes, rejects imports of the three original target modules, and compares location-insensitive AST digests for the nine helpers plus `_matmul_decode`.

Therefore `k00` is execution-equivalent to the pinned decode path, but the whole file is not byte-for-byte equal to `moe_decode_gfx1250.py`. A raw two-file diff reports hundreds of added lines because the original helper imports became local definitions; that is a source-boundary change, not a kernel algorithm change. Review H0 through [`ledgers/H0.md`](ledgers/H0.md) and the source-fidelity implementation in `/root/triton-mi450/third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation/run_ffm.py`, rather than treating that noisy raw diff as an optimization.

## Recommended review workflow

### 1. Enter the frozen source snapshot

```bash
cd /root/triton-mi450
git switch users/jerryyin/a8w4-decode-ablation-variants
ABLATION_DIR=third_party/amd/python/examples/gluon/moe_a8w4_decode_ablation
```

### 2. Read the cell contract

Find the cell in [`REGISTRY.md`](REGISTRY.md). Read only its `Parent`, `One idea`, and `Conclusion` first. That tells you which edge to compare and the single mechanism the changed hunks must implement.

### 3. Inspect a compact textual diff

For a normal one-parent edge:

```bash
git diff --no-index --histogram --function-context -- \
  "$ABLATION_DIR/k00_pr154_baseline.py" \
  "$ABLATION_DIR/k02_aiter_decode_wmma_layout.py"
```

`git diff --no-index` exits with status 1 when it finds differences; that status is expected. `--histogram` handles moved or copied Python blocks better than the default algorithm, and `--function-context` keeps changed device functions intelligible.

### 4. Use read-only `vimdiff` for the detailed review

Yes: `vimdiff` each declared pair is the best interactive review path.

```bash
vimdiff -R \
  -c 'set diffopt+=algorithm:histogram,indent-heuristic,context:3' \
  "$ABLATION_DIR/k00_pr154_baseline.py" \
  "$ABLATION_DIR/k02_aiter_decode_wmma_layout.py"
```

Useful review keys are `]c` for the next changed hunk, `[c` for the previous hunk, `zo` to open folded context, `zc` to close it, and `:diffupdate` after changing diff options. `-R` keeps the review read-only.

Do not spend equal attention on every changed line. Classify each hunk as:

1. kernel mechanism;
2. necessary ABI, launch, oracle, or rejection-boundary support for that mechanism;
3. ablation metadata; or
4. unrelated change.

Category 4 is a defect in a one-change edge. Categories 2 and 3 should be checked for semantic necessity and then skipped when reasoning about the optimization.

### 5. Reconstruct the kernel idea

For the changed device path, answer these questions:

1. What tuple does one program own: expert tile, route and N tile, or token and output tile?
2. Where are routes grouped, padded, or consumed directly?
3. What computes the dot product: wave reduction or scaled WMMA?
4. During one K iteration, what is loaded from global memory, what is staged through LDS, and what stays in registers?
5. Which activation or weight values are reused, and across what axis?
6. What global intermediate or final output does the program store?
7. Which source restriction defines the legal regime?

If the diff cannot be summarized as one ownership, dataflow, compute, or boundary idea after ignoring mechanical support, it is not a clean ablation edge.

### 6. Use the ledger as evidence, not as a substitute for source review

After reviewing the source, read the cell ledger. Confirm that its changed-hunk classification matches yours, that the owned child actually dispatched, and that the FFM and generated-code evidence tests the claimed boundary. FFM proves correctness, not performance.

## Exact review graph

All filenames in the table are relative to `moe_a8w4_decode_ablation/`.

| Cell | Review edge | Kernel idea to isolate |
|---|---|---|
| H0 | virtual parent: pinned `moe_common_gfx1250.py` helper closure plus pinned `moe_decode_gfx1250.py::_matmul_decode` → `k00_pr154_baseline.py`; use the H0 fidelity audit instead of raw `vimdiff` | Self-contained execution-equivalent baseline |
| G2 | `k00_pr154_baseline.py` → `k02_aiter_decode_wmma_layout.py` | AITER decode-specific WMMA warp ownership |
| G3 | `k00_pr154_baseline.py` → `k03_aiter_persistent_n3.py` | One workgroup walks three N tiles after decoding routing once |
| G4 | `k00_pr154_baseline.py` → `k04_aiter_packed_schedule.py` | Direct packed schedule decode |
| G5 | `k00_pr154_baseline.py` → `k05_aiter_preshuffled_weight.py` | Consume preshuffled packed weights and invert the permutation locally |
| G6 | `k00_pr154_baseline.py` → `k11_grouped_direct_load.py` | Replace generic input TDM with direct one-buffer X/W/scale transport |
| G7 | `k00_pr154_baseline.py` → `k12_grouped_atomic_combine.py` | Route-weight and atomically combine grouped W2 output |
| G8 | `k00_pr154_baseline.py` → `k16_grouped_mxfp8_emit.py` | Dynamic per-32-column MXFP8 output boundary |
| Q1 | TokenSpeed `decode_kernels.py` softmax router → `k18_softmax_topk_router.py` | One-workgroup softmax and stable top-k port; this is not an H0 child |
| Q2 | TokenSpeed `decode_kernels.py` biased router → `k19_sigmoid_bias_topk_router.py` | One-workgroup sigmoid+bias top-k port; this is not an H0 child |
| Q3 | TokenSpeed `fused_mxfp_gfx950.py` precomputed-top-k route → `k20_precomputed_topk_group_route.py` | One-workgroup construction of grouped routing metadata; this is not an H0 child |
| Q4 | primary `k20_precomputed_topk_group_route.py`, donor `k18_softmax_topk_router.py` → `k21_fused_softmax_group_route.py` | Remove the global Q1→Q3 top-k boundary |
| Q5 | primary `k20_precomputed_topk_group_route.py`, donor `k19_sigmoid_bias_topk_router.py` → `k22_fused_biased_group_route.py` | Remove the global Q2→Q3 top-k boundary |
| Q6 | `k20_precomputed_topk_group_route.py` → `k25_flat_m1_group_route.py` | Replace general grouping with the exact unique-route M=1 construction |
| R1 | `k00_pr154_baseline.py` → `k06_route_direct_wave_gemv.py` | Atomic algorithmic fork to route-direct ownership, streaming weights, and wave reduction |
| R2 | `k06_route_direct_wave_gemv.py` → `k07_route_direct_gate_up.py` | Reuse one activation stream for gate and up projections |
| R3 | `k07_route_direct_gate_up.py` → `k08_route_direct_gate_up_bf16.py` | BF16 intermediate boundary |
| R4 | `k07_route_direct_gate_up.py` → `k09_route_direct_gate_up_fp8.py` | Static E4M3 intermediate boundary |
| R5 | `k06_route_direct_wave_gemv.py` → `k10_output_owned_topk.py` | Token/output ownership with an in-kernel top-k loop and one final store |
| R6 | `k06_route_direct_wave_gemv.py` → `k13_route_direct_scaled_mfma.py` | Replace wave reduction with duplicated-row gfx1250 scaled WMMA |
| R7 | `k10_output_owned_topk.py` → `k14_output_owned_topk_scaled_mfma.py` | Preserve output ownership while replacing each wave reduction with scaled WMMA |
| R8 | `k14_output_owned_topk_scaled_mfma.py` → `k15_output_owned_topk_scaled_mfma_lookahead.py` | Source-level paired K-tile lookahead |
| R9 | primary `k13_route_direct_scaled_mfma.py`, donor `k18_softmax_topk_router.py` → `k17_route_direct_scaled_mfma_fused_topk.py` | Recompute and consume top-k inside each direct W1 program |
| I1 | primary `k08_route_direct_gate_up_bf16.py`, W2 donor `k10_output_owned_topk.py` → `k23_integrated_wave_bf16.py` | Complete two-stage wave pipeline through a BF16 boundary |
| I2 | primary `k09_route_direct_gate_up_fp8.py`, W2 donor `k10_output_owned_topk.py` → `k24_integrated_wave_fp8.py` | Complete two-stage wave pipeline through a static-FP8 boundary |

For a multi-parent row, first diff the child against the listed primary parent. Then compare only the added router or W2 definition against its donor. A whole-file three-way comparison is usually noisier than these two focused checks.

The external donor locations are:

- Q1 and Q2: `/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/gluon_a4w4_gfx950/decode_kernels.py`.
- Q3: `/root/tokenspeed/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py`.

## Suggested review order

1. H0 source fidelity and baseline execution story.
2. G2–G8, because each is a direct and independent grouped-path child of H0.
3. R1, then R2/R5/R6, then their descendants R3/R4/R7/R8/R9.
4. Q1–Q3 donor ports, then the Q4–Q6 routing compositions.
5. I1 and I2 only after their W1 and W2 parents are understood.

This order ensures that each inherited mechanism is learned once. Reviewing in filename order would jump between unrelated families and repeatedly rediscover prerequisite changes.

## Alternatives Considered

### Diff every file against `k00`

Rejected because descendant diffs accumulate all prerequisite ideas. For example, `k00` → `k08` contains the route-direct R1 fork, the dual gate/up R2 change, and the BF16 R3 boundary. The isolating review edge is `k07` → `k08`.

### Raw `vimdiff` between `moe_decode_gfx1250.py` and `k00`

Rejected as the primary H0 review because the original file imports its pipeline while `k00` embeds it. The resulting diff is dominated by copied helper definitions. The H0 AST/source-fidelity audit directly tests the intended equivalence.

### Read only the ledgers

Rejected because a ledger is an evidence index and conclusion, not a replacement for checking the changed device path. Use it after the source diff to validate dispatch, correctness boundaries, generated code, and unknowns.

### Review by filename order

Rejected because numeric filenames reflect creation order, not the conceptual parent graph. Use the registry edge and suggested review order instead.
